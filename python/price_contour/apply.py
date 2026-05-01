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
    _validate_constraint_dict,
    _validate_dataframe,
)


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
