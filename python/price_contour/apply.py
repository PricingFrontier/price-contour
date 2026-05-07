"""ApplyOptimiser — apply stored lambdas to new data (single-pass, no iteration)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from price_contour._price_contour import (
    ApplyResult,
    apply_from_grid_py,
    apply_lambdas_py,
)
from price_contour.builder import QuoteGrid
from price_contour.solver import (
    _RatioApplyResultWrapper,
    _linearise_ratio_constraints,
    _none_threshold_constraints,
    _ratio_constraint_names,
    _reject_ratio_for_grid,
    _spec_direction,
    _validate_constraint_dict,
    _validate_dataframe,
)


# Internal sentinel column names used by ``with_explainer_columns`` for
# join scaffolding. The double-underscore prefix + ``__pc_explainer__``
# stamp makes accidental collision with user data implausible; we drop
# them before returning. Centralised here so the implementation reads
# cleanly without inline magic strings.
_EXPLAINER_OPT_STEP = "__pc_explainer_opt_step"
_EXPLAINER_BASELINE_SI = "__pc_explainer_baseline_scenario_index"
_EXPLAINER_BASELINE_DIST = "__pc_explainer_baseline_distance"


class ApplyOptimiser:
    """Apply pre-computed Lagrange multipliers to a scored DataFrame.

    This performs a single forward pass — no iteration, no lambda updates.
    Each quote picks the step that maximises the Lagrangian with fixed lambdas.

    Parameters
    ----------
    lambdas : dict[str, float]
        Lagrange multipliers keyed by constraint name.
    objective : str
        Objective column name.
    constraints : dict[str, dict[str, float]]
        Constraint specifications (same format as OnlineOptimiser).
    quote_id, scenario_index, scenario_value : str
        Column name overrides.
    """

    def __init__(
        self,
        lambdas: dict[str, float],
        objective: str = "expected_income",
        constraints: dict[str, dict[str, float]] | None = None,
        *,
        quote_id: str = "quote_id",
        scenario_index: str = "scenario_index",
        scenario_value: str = "scenario_value",
    ) -> None:
        self.lambdas = lambdas
        self.objective = objective
        self.constraints = {} if constraints is None else constraints
        self.quote_id = quote_id
        self.scenario_index = scenario_index
        self.scenario_value = scenario_value
        _validate_constraint_dict(self.constraints)
        # Apply mode runs a fixed forward pass with a known threshold per
        # constraint, so a ``None`` threshold has no meaning here. (Online
        # and Ratebook accept None as a frontier-only marker; Apply doesn't
        # have a frontier, so reject at construction with an apply-specific
        # message that points at the only valid use.)
        none_names = _none_threshold_constraints(self.constraints)
        if none_names:
            raise ValueError(
                f"ApplyOptimiser does not support None thresholds; apply "
                f"mode requires a fixed threshold per constraint. "
                f"Offending constraint: '{none_names[0]}'."
            )
        if self.lambdas:
            lambda_keys = set(self.lambdas.keys())
            constraint_keys = set(self.constraints.keys())
            extra = lambda_keys - constraint_keys
            if extra:
                raise ValueError(
                    f"Lambda keys {sorted(extra)} do not match any "
                    f"constraint. Valid constraint keys are "
                    f"{sorted(constraint_keys)}."
                )

    def apply(self, df: pl.DataFrame) -> ApplyResult:
        """Apply lambdas to the scored DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Long-format scored DataFrame (same schema as solve input).

        Returns
        -------
        ApplyResult
            Result with .total_objective, .total_constraints,
            .baseline_objective, .baseline_constraints, .lambdas, .dataframe.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")
        # DataFrame schema validation runs BEFORE the linearisation so
        # that missing/null numerator/denominator columns surface the
        # precise schema error naming the column and the constraint
        # label, rather than crashing inside the linearisation expression.
        _validate_dataframe(
            df,
            quote_id=self.quote_id,
            scenario_index=self.scenario_index,
            scenario_value=self.scenario_value,
            objective=self.objective,
            constraint_cols=list(self.constraints.keys()),
            constraints=self.constraints,
        )
        # Ratio-constraint path (C6). Apply runs a fixed forward pass with
        # stored lambdas; we linearise each ratio spec at apply time using
        # the apply-time DataFrame's baseline LR (``min_pct`` / ``max_pct``
        # are anchored on the scoring-time data, not on a frozen solve-time
        # baseline). The synthetic linearised column is materialised on a
        # working copy of the apply-time DataFrame; the existing Rust
        # ``apply_lambdas_py`` runs with the linearised sum-shape
        # constraints; the wrapper stitches ``optimal_<num>`` /
        # ``optimal_<denom>`` onto the result and re-reports actual
        # ratios in ``total_constraints`` / ``baseline_constraints``.
        # Sum-only constraint dicts skip the linearisation entirely so
        # the existing fast path is preserved bit-for-bit.
        ratio_names = _ratio_constraint_names(self.constraints)
        if ratio_names:
            (
                modified_df,
                sum_constraints,
                _grid_cols,
                ratio_columns,
                _threshold_shift,
            ) = _linearise_ratio_constraints(
                df,
                self.constraints,
                scenario_value_col=self.scenario_value,
                quote_id_col=self.quote_id,
            )
            inner_result = apply_lambdas_py(
                modified_df,
                lambdas=self.lambdas,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
                constraints=sum_constraints,
            )
            return _RatioApplyResultWrapper(
                inner_result,
                original_df=df,
                ratio_columns=ratio_columns,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
            )
        return apply_lambdas_py(
            df,
            lambdas=self.lambdas,
            quote_id=self.quote_id,
            scenario_index=self.scenario_index,
            scenario_value=self.scenario_value,
            objective=self.objective,
            constraints=self.constraints,
        )

    def with_explainer_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Append optimiser-consistent explainer columns to ``df``.

        Returns a new :class:`pl.DataFrame` with all input rows and
        columns preserved, plus deterministic explainer columns that
        reconcile with :meth:`apply` exactly:

        * ``decision_score`` — fixed-lambda Lagrangian score per row,
          satisfying ``decision_score == objective + sum(lambda_term_*)``
          to floating-point precision.
        * ``selected`` — ``True`` for the row that :meth:`apply` chose
          for that quote (same row, same tie-break).
        * ``is_baseline`` — ``True`` for the per-quote baseline row
          (lowest ``scenario_index`` among rows minimising
          ``|scenario_value - 1.0|``).
        * ``linearised_<name>`` — value used in the score for each
          constraint. For sum constraints this equals the original
          constraint column, cast to apply's ``Float32`` precision. For
          ratio constraints this is the internal ``num - L * denom``
          value used by the fixed-lambda apply.
        * ``lambda_term_<name>`` — signed contribution of the
          constraint to ``decision_score``: ``+lambda`` for ``min``
          direction, ``-lambda`` for ``max`` direction, multiplied by
          ``linearised_<name>``.

        With ``constraints == {}`` only ``decision_score``,
        ``selected``, and ``is_baseline`` are appended; ``decision_score``
        equals the objective column.

        Validation matches :meth:`apply` (missing/null/NaN columns,
        unknown lambda keys at construction); a column collision
        between the input DataFrame and the appended explainer columns
        raises :class:`ValueError` rather than silently overwriting.

        Parameters
        ----------
        df : pl.DataFrame
            Long-format scored DataFrame (same schema accepted by
            :meth:`apply`).

        Returns
        -------
        pl.DataFrame
            Original ``df`` with explainer columns appended.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")

        # Refuse to overwrite caller-supplied columns. Polars
        # ``with_columns`` would silently shadow; we'd rather raise
        # naming the offender. Runs before ``apply`` so a name
        # collision surfaces with a method-specific message instead of
        # being masked by the (also-valid) generic apply errors.
        reserved: set[str] = {"decision_score", "selected", "is_baseline"}
        for name in self.constraints:
            reserved.add(f"linearised_{name}")
            reserved.add(f"lambda_term_{name}")
        collisions = sorted(c for c in df.columns if c in reserved)
        if collisions:
            raise ValueError(
                f"Input DataFrame already contains column(s) {collisions} "
                f"that with_explainer_columns would append. Rename or "
                f"drop them before calling this method."
            )

        # Reuse the canonical apply pass: argmax (so ``selected`` is
        # bit-identical to ``apply(df)`` including tie-breaking) AND
        # schema validation (missing columns, null/NaN, ratio
        # numerator/denominator existence) live there already. No
        # standalone ``_validate_dataframe`` call here — that would
        # duplicate apply's check and create a code path to keep in
        # sync.
        apply_result = self.apply(df)

        # Linearise ratio constraints on a working copy. For sum-only
        # configurations we keep ``df`` as-is so the no-ratio fast
        # path skips the linearisation expression. For ratio configs,
        # ``modified_df`` carries a synthetic column under the ratio
        # label; we read it for ``linearised_<name>`` and project it
        # away from the final output below so callers don't see a
        # phantom column they didn't supply.
        ratio_names: set[str] = set(_ratio_constraint_names(self.constraints))
        if ratio_names:
            modified_df, _sum_specs, _grid_cols, _ratio_columns, _shift = (
                _linearise_ratio_constraints(
                    df,
                    self.constraints,
                    scenario_value_col=self.scenario_value,
                    quote_id_col=self.quote_id,
                )
            )
        else:
            modified_df = df

        # ``linearised_<name>`` reads from ``modified_df[name]``: sum
        # constraints pass through unchanged; ratio constraints expose
        # the ``num - L * denom`` synthetic value computed by the
        # apply-path linearisation. Cast to Float32 to mirror the Rust
        # apply argmax exactly: QuoteGrid stores row values as f32 and
        # ``compute_lambda_signs_f32`` casts lambdas to f32 before the
        # per-row score is accumulated.
        linearised_exprs: list[pl.Expr] = [
            pl.col(name).cast(pl.Float32).alias(f"linearised_{name}")
            for name in self.constraints
        ]

        # ``lambda_term_<name>`` = signed_lambda * linearised_<name>.
        # Sign convention pinned by spec: +lambda for min, -lambda for
        # max. ``self.lambdas.get(name, 0.0)`` mirrors apply's
        # ``order_lambdas`` default for missing keys (apply-time
        # __init__ already rejected unknown lambda keys at construction,
        # so there is no ambiguity here).
        lambda_term_exprs: list[pl.Expr] = []
        for name, spec in self.constraints.items():
            direction = _spec_direction(spec)
            lam = float(self.lambdas.get(name, 0.0))
            signed = lam if direction == "min" else -lam
            lambda_term_exprs.append(
                (pl.lit(signed, dtype=pl.Float32) * pl.col(f"linearised_{name}"))
                .cast(pl.Float32)
                .alias(f"lambda_term_{name}")
            )

        # ``decision_score`` = objective + sum(lambda_term_*), accumulated
        # in the same precision and order as apply mode. The Rust binding
        # sorts constraint columns before grid ingestion, so f32 addition
        # must follow sorted constraint-name order here as well.
        if self.constraints:
            score_expr: pl.Expr = pl.col(self.objective).cast(pl.Float32)
            for name in sorted(self.constraints):
                score_expr = (
                    score_expr + pl.col(f"lambda_term_{name}").cast(pl.Float32)
                ).cast(pl.Float32)
            decision_expr = score_expr.cast(pl.Float64).alias("decision_score")
        else:
            decision_expr = (
                pl.col(self.objective)
                .cast(pl.Float32)
                .cast(pl.Float64)
                .alias("decision_score")
            )

        # ``selected`` lookup: apply's result DataFrame uses literal
        # ``"quote_id"`` regardless of ``self.quote_id`` (see
        # ``crates/price-contour/src/solver_py.rs::build_result_dataframe``).
        # Rename to the user's column so the join key matches; rename
        # ``optimal_step`` to a sentinel so a user-supplied column of
        # that name isn't shadowed during the join.
        selected_lookup = (
            apply_result.dataframe.select(["quote_id", "optimal_step"])
            .rename(
                {
                    "quote_id": self.quote_id,
                    "optimal_step": _EXPLAINER_OPT_STEP,
                }
            )
            .with_columns(pl.col(self.quote_id).cast(df.schema[self.quote_id]))
        )

        # Per-quote baseline lookup: lowest ``scenario_index`` among
        # rows minimising ``|scenario_value - 1.0|``. Matches
        # ``grid.baseline_totals`` semantics in ``data.rs`` (``min_by``
        # on ``|sv - 1.0|`` keeps the first occurrence) at portfolio
        # level. In the current online-optimiser shape ``scenario_values``
        # are grid-wide so this resolves to the same scenario_index for
        # every quote; the per-quote computation is what the spec asks
        # for and stays correct should that ever loosen.
        baseline_lookup = (
            modified_df.lazy()
            .select(
                [
                    pl.col(self.quote_id),
                    pl.col(self.scenario_index),
                    (pl.col(self.scenario_value).cast(pl.Float64) - 1.0)
                    .abs()
                    .alias(_EXPLAINER_BASELINE_DIST),
                ]
            )
            .group_by(self.quote_id, maintain_order=True)
            .agg(
                pl.col(self.scenario_index)
                .sort_by([_EXPLAINER_BASELINE_DIST, self.scenario_index])
                .first()
                .alias(_EXPLAINER_BASELINE_SI)
            )
            .collect()
        )

        # Compose the final DataFrame. Order:
        #   1. linearised_* columns (so lambda_term_* can reference them)
        #   2. lambda_term_* columns
        #   3. decision_score (sums the lambda_terms)
        #   4. selected (join + compare against optimal_step)
        #   5. is_baseline (join + compare against baseline_scenario_index)
        # Final ``.select`` projects to (original columns + appended
        # explainer columns), dropping the synthetic ratio-label columns
        # that ``modified_df`` carried as scaffolding.
        out = modified_df
        if linearised_exprs:
            out = out.with_columns(linearised_exprs)
        if lambda_term_exprs:
            out = out.with_columns(lambda_term_exprs)
        out = out.with_columns(decision_expr)

        out = (
            out.join(selected_lookup, on=self.quote_id, how="left")
            .with_columns(
                (pl.col(self.scenario_index) == pl.col(_EXPLAINER_OPT_STEP))
                .alias("selected")
            )
            .drop(_EXPLAINER_OPT_STEP)
        )

        out = (
            out.join(baseline_lookup, on=self.quote_id, how="left")
            .with_columns(
                (pl.col(self.scenario_index) == pl.col(_EXPLAINER_BASELINE_SI))
                .alias("is_baseline")
            )
            .drop(_EXPLAINER_BASELINE_SI)
        )

        appended = (
            [f"linearised_{name}" for name in self.constraints]
            + [f"lambda_term_{name}" for name in self.constraints]
            + ["decision_score", "selected", "is_baseline"]
        )
        return out.select([*df.columns, *appended])

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file.

        Parameters
        ----------
        path : str | Path
            File path to write config.json.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "version": 1,
            "lambdas": self.lambdas,
            "objective": self.objective,
            "constraints": self.constraints,
            "quote_id": self.quote_id,
            "scenario_index": self.scenario_index,
            "scenario_value": self.scenario_value,
        }
        path.write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ApplyOptimiser:
        """Load configuration from a JSON file.

        Parameters
        ----------
        path : str | Path
            File path to read config.json.

        Returns
        -------
        ApplyOptimiser
            Configured ApplyOptimiser instance.
        """
        path = Path(path)
        config = json.loads(path.read_text())
        if "lambdas" not in config:
            raise ValueError(
                f"Invalid config file {path}: missing 'lambdas' key. "
                f"Available keys: {list(config.keys())}"
            )
        version = config.get("version", 0)
        if version > 1:
            raise ValueError(
                f"Config file version {version} is not supported by this "
                f"version of price-contour (max supported: 1)"
            )
        # Allowlist for known keys; unknown keys indicate either a
        # corrupted file, a future-version config, or a hand-edited
        # mistake — surface those rather than silently ignoring.
        known_keys = {
            "version",
            "lambdas",
            "objective",
            "constraints",
            "quote_id",
            "scenario_index",
            "scenario_value",
        }
        extras = set(config.keys()) - known_keys
        if extras:
            raise ValueError(f"unknown keys in saved config: {sorted(extras)}")
        return cls(
            lambdas=config["lambdas"],
            objective=config.get("objective", "expected_income"),
            constraints=config.get("constraints"),
            quote_id=config.get("quote_id", "quote_id"),
            scenario_index=config.get("scenario_index", "scenario_index"),
            scenario_value=config.get("scenario_value", "scenario_value"),
        )


def apply_from_grid(
    grid: QuoteGrid,
    lambdas: dict[str, float],
    constraints: dict[str, dict[str, float]],
) -> ApplyResult:
    """Single-pass Lagrangian apply on an existing QuoteGrid.

    This performs the same argmax as ``ApplyOptimiser.apply()``, but
    operates directly on an in-memory QuoteGrid — no DataFrame
    re-ingestion. Useful when the grid is already available (e.g. after
    a ``solve()`` or ``frontier()`` call) and you want O(N) evaluation
    at specific lambdas with no iteration overhead.

    Parameters
    ----------
    grid : QuoteGrid
        Pre-built QuoteGrid (e.g. from ``SolveResult.grid`` or a builder).
    lambdas : dict[str, float]
        Fixed Lagrange multipliers keyed by constraint name.
    constraints : dict[str, dict[str, float]]
        Constraint specifications (same format as ``OnlineOptimiser``).

    Returns
    -------
    ApplyResult
        Result with ``.total_objective``, ``.total_constraints``,
        ``.baseline_objective``, ``.baseline_constraints``, ``.lambdas``,
        ``.dataframe``.
    """
    # Mirror ``ApplyOptimiser.__init__``'s rejection: apply mode runs a
    # fixed forward pass with a known threshold per constraint, so a
    # ``None`` threshold has no meaning. The Rust backstop in
    # ``apply_from_grid_py`` would also reject, but with the generic
    # ``solve()``/``frontier()`` wording — surface the apply-specific
    # message here so the user-facing error names the actual mode.
    none_names = _none_threshold_constraints(constraints)
    if none_names:
        raise ValueError(
            f"ApplyOptimiser does not support None thresholds; apply "
            f"mode requires a fixed threshold per constraint. "
            f"Offending constraint: '{none_names[0]}'."
        )
    # Pre-built grid path: the linearisation needs raw numerator /
    # denominator columns at apply time, but a frozen QuoteGrid does
    # NOT carry them — they'd have had to be added at grid build time
    # and even then the apply-time baseline LR for ``min_pct`` /
    # ``max_pct`` cannot be recovered from the opaque grid. Mirror the
    # solve_from_grid / frontier wording so the entry-point story is
    # consistent across modes. We raise ``ValueError`` (NOT
    # ``NotImplementedError`` — the feature is available, just not via
    # this entry point) and point the user at the DataFrame-shape apply.
    _reject_ratio_for_grid(constraints, mode="apply")
    return apply_from_grid_py(grid, lambdas, constraints)
