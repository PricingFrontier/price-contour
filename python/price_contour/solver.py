"""OnlineOptimiser — Python orchestration for the online Lagrangian solver."""

from __future__ import annotations

import json
import math
from typing import Any

import polars as pl

from price_contour._frontier_helpers import (
    _compute_sv_stats_from_dataframe,
    _PythonFrontierResult,
    _python_frontier_orchestrator,
)
from price_contour._grid_utils import build_grid
from price_contour._price_contour import (
    ApplyResult,
    FrontierResult,
    QuoteGrid,
    SolveResult,
    solve_from_grid_py,
    solve_online_py,
    sweep_frontier_py,
)
from price_contour._ratio_results import (
    _safe_ratio_from_columns,
    _stitch_optimal_ratio_columns,
)


class OnlineOptimiser:
    """Portfolio-level price optimisation via Lagrangian dual decomposition.

    Parameters
    ----------
    quote_id : str
        Column name for quote identifiers.
    scenario_index : str
        Column name for step indices.
    scenario_value : str
        Column name for scenario values.
    objective : str
        Column name for the objective function (e.g. expected income).
    constraints : dict[str, dict[str, float]]
        Constraint specifications. Keys are column names, values are dicts
        with one of: ``min`` (absolute), ``max`` (absolute),
        ``min_pct`` (fraction of baseline), ``max_pct`` (fraction of
        baseline).
    max_iter : int
        Maximum solver iterations.
    tolerance : float
        Convergence tolerance.
    record_history : bool
        If True, record per-iteration convergence history on the result.
    """

    def __init__(
        self,
        objective: str = "expected_income",
        constraints: dict[str, dict[str, float]] | None = None,
        *,
        quote_id: str = "quote_id",
        scenario_index: str = "scenario_index",
        scenario_value: str = "scenario_value",
        max_iter: int = 50,
        tolerance: float = 1e-5,
        record_history: bool = False,
    ) -> None:
        self.quote_id = quote_id
        self.scenario_index = scenario_index
        self.scenario_value = scenario_value
        self.objective = objective
        self.constraints = {} if constraints is None else constraints
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.record_history = record_history
        _validate_constraint_dict(self.constraints)

    def solve(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        *,
        lambdas: dict[str, float] | None = None,
    ) -> SolveResult:
        """Run the optimisation.

        Parameters
        ----------
        df_or_grid : pl.DataFrame | QuoteGrid
            Long-format scored DataFrame, or a pre-built QuoteGrid.
        lambdas : dict[str, float], optional
            Initial lambda values for warm-start.

        Returns
        -------
        SolveResult
            Result object with .lambdas, .converged, .iterations,
            .total_objective, .total_constraints, .dataframe properties.
        """
        if isinstance(df_or_grid, QuoteGrid):
            # Pre-built grid path: ratio constraints would need both the
            # numerator and denominator columns embedded at grid build
            # time AND the synthetic linearised column materialised in
            # advance. C2 does the linearisation in Python on the
            # DataFrame branch only, so a ratio + grid call is
            # effectively a setup-time error. We surface ``ValueError``
            # (NOT ``NotImplementedError``) because the feature is
            # available — just not via this entry point — and the
            # message names the offending constraint label and points
            # the user at the DataFrame-shape solve.
            _reject_ratio_for_grid(self.constraints)
            _reject_none_for_solve(self.constraints)
            return solve_from_grid_py(
                df_or_grid,
                constraints=self.constraints,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                lambdas=lambdas,
                record_history=self.record_history,
            )
        if not isinstance(df_or_grid, pl.DataFrame):
            raise TypeError(
                f"Expected pl.DataFrame or QuoteGrid, got {type(df_or_grid).__name__}"
            )
        # DataFrame schema validation runs BEFORE the linearisation so
        # missing / null / NaN numerator-denominator columns raise a
        # specific ValueError naming the column and the constraint label
        # rather than crashing inside the Polars expression below.
        _validate_dataframe(
            df_or_grid,
            quote_id=self.quote_id,
            scenario_index=self.scenario_index,
            scenario_value=self.scenario_value,
            objective=self.objective,
            constraint_cols=list(self.constraints.keys()),
            constraints=self.constraints,
        )
        # ``None`` thresholds are frontier-only markers (B1); reject
        # before any linearisation work because ``L`` is undefined.
        _reject_none_for_solve(self.constraints)

        # If any ratio constraints are present, linearise them into
        # synthetic sum constraints and switch to ``solve_online_py``
        # with the modified DataFrame (linearised columns appended).
        # We deliberately do NOT push numerator / denominator columns
        # into the grid: doing so would force the Rust solver to carry
        # them as slack constraints (specs.len() == grid.constraints.len()
        # invariant) and inflate the per-iteration constraint loop by
        # O(N_extra). Instead, the result is wrapped so the
        # ``optimal_<numerator>`` / ``optimal_<denominator>`` columns
        # are stitched in lazily on ``result.dataframe`` access. Sum-only
        # constraint dicts skip the linearisation pass entirely so the
        # existing fast path is preserved bit-for-bit.
        ratio_names = _ratio_constraint_names(self.constraints)
        if ratio_names:
            (
                modified_df,
                sum_constraints,
                _grid_cols,
                ratio_columns,
                threshold_shift,
            ) = _linearise_ratio_constraints(
                df_or_grid,
                self.constraints,
                scenario_value_col=self.scenario_value,
                quote_id_col=self.quote_id,
            )
            inner_result = solve_online_py(
                modified_df,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
                constraints=sum_constraints,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                lambdas=lambdas,
                record_history=self.record_history,
            )
            return _RatioSolveResultWrapper(
                inner_result,
                original_df=df_or_grid,
                modified_df=modified_df,
                ratio_columns=ratio_columns,
                sum_constraints=sum_constraints,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
                threshold_shift=threshold_shift,
            )

        return solve_online_py(
            df_or_grid,
            quote_id=self.quote_id,
            scenario_index=self.scenario_index,
            scenario_value=self.scenario_value,
            objective=self.objective,
            constraints=self.constraints,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            lambdas=lambdas,
            record_history=self.record_history,
        )

    def frontier(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        *,
        threshold_ranges: dict[str, tuple[float, float]],
        n_points_per_dim: int = 10,
        initial_lambdas: dict[str, float] | None = None,
        max_total_points: int = 10_000,
        parallel: bool = False,
    ) -> FrontierResult:
        """Sweep the efficient frontier over constraint threshold ranges.

        Parameters
        ----------
        df_or_grid : pl.DataFrame | QuoteGrid
            Scored data or pre-built QuoteGrid.
        threshold_ranges : dict[str, tuple[float, float]]
            Per-constraint (lo, hi) range for the threshold sweep.
            Units follow the constraint key: absolute for ``min`` /
            ``max``; fractions of baseline for ``min_pct`` / ``max_pct``.
        n_points_per_dim : int
            Number of points per constraint dimension.
        initial_lambdas : dict[str, float], optional
            Lambdas to warm-start the first frontier point. Typically from
            a prior ``solve()`` call on the same grid. Subsequent points
            warm-start from their nearest neighbour as usual.

        Returns
        -------
        FrontierResult
            Result with .points (DataFrame) and .n_points.
        """
        # Detect ratio constraints up front. When any are present we
        # dispatch to the Python-side sweep so that each point can run
        # the C2 linearisation in ``solve()``. We also delegate to
        # Python when any constraint is left out of ``threshold_ranges``
        # (D1: numeric thresholds without a range stay fixed at their
        # constructor value; the Rust fast path's cartesian-over-all-axes
        # design assumes one range per constraint, so the unswept-axis
        # case is handled by the Python orchestrator).
        ratio_names = _ratio_constraint_names(self.constraints)
        has_unswept = any(
            name not in threshold_ranges for name in self.constraints
        )
        if ratio_names or has_unswept:
            if ratio_names and not isinstance(df_or_grid, pl.DataFrame):
                # The ratio linearisation needs raw numerator / denominator
                # columns at solve time; a pre-built grid has already
                # frozen its constraint columns. Mirror the solve() error
                # wording so the entry-point story is consistent.
                _reject_ratio_for_grid(self.constraints, mode="frontier")
            if isinstance(df_or_grid, pl.DataFrame):
                _validate_dataframe(
                    df_or_grid,
                    quote_id=self.quote_id,
                    scenario_index=self.scenario_index,
                    scenario_value=self.scenario_value,
                    objective=self.objective,
                    constraint_cols=list(self.constraints.keys()),
                    constraints=self.constraints,
                )
            return self._python_frontier_sweep(
                df_or_grid,
                threshold_ranges=threshold_ranges,
                n_points_per_dim=n_points_per_dim,
                initial_lambdas=initial_lambdas,
                max_total_points=max_total_points,
            )

        if isinstance(df_or_grid, pl.DataFrame):
            # Validate DataFrame BEFORE any ratio-specific dispatch so
            # ratio-shape callers with missing numerator/denominator
            # columns surface the precise schema error first.
            _validate_dataframe(
                df_or_grid,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
                constraint_cols=list(self.constraints.keys()),
                constraints=self.constraints,
            )
            # Sum-constraint columns only — the ratio constraint key is a
            # display label, not a column, so we must NOT pass it to the
            # grid builder. (Ratio path returned above.)
            sum_constraint_cols = [
                c
                for c in self.constraints.keys()
                if not _is_ratio_spec(self.constraints[c])
            ]
            grid = build_grid(
                df_or_grid,
                constraint_columns=sum_constraint_cols,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
            )
        else:
            # Pre-built grid: skip DataFrame validation (already encoded
            # in the grid).
            grid = df_or_grid

        return sweep_frontier_py(
            grid,
            constraints=self.constraints,
            threshold_ranges=threshold_ranges,
            n_points_per_dim=n_points_per_dim,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            initial_lambdas=initial_lambdas,
            max_total_points=max_total_points,
            parallel=parallel,
        )

    def _python_frontier_sweep(
        self,
        df: pl.DataFrame | QuoteGrid,
        *,
        threshold_ranges: dict[str, tuple[float, float]],
        n_points_per_dim: int,
        initial_lambdas: dict[str, float] | None,
        max_total_points: int,
    ) -> "_PythonFrontierResult":
        """Sweep the frontier in Python by calling ``solve()`` per point.

        Used when at least one ratio constraint is present, or when one
        or more numeric-threshold constraints are omitted from
        ``threshold_ranges`` (D1 — held fixed at the constructor value).
        Each threshold combination becomes a fresh constraint dict with
        the swept axes' per-point threshold values; unswept axes keep
        their constructor spec verbatim. Lambdas warm-start from the
        previous point's result (sequential traversal — simpler than
        nearest-neighbour ordering and adequate for correctness; the
        Rust fast-path keeps NN ordering for the pure all-swept
        sum-constraint case).
        """
        # D1 contract: a ``None`` threshold MUST have a
        # ``threshold_ranges`` entry (B1 marker rule preserved); a
        # numeric threshold may omit its range (held fixed at the
        # constructor value). Validate None-without-range first so the
        # error message matches the B1 wording the test agent pinned.
        for name, spec in self.constraints.items():
            if name in threshold_ranges:
                continue
            if _spec_threshold_is_none(spec):
                raise ValueError(
                    f"No threshold_range for constraint '{name}'"
                )

        constraint_names = list(self.constraints.keys())
        # ``self.constraints`` cannot be empty here: ``frontier()`` only
        # dispatches to this Python sweep when ``ratio_names`` or
        # ``has_unswept`` is truthy (both presuppose constraints exist),
        # so an empty dict would have routed to the Rust fast path which
        # raises a clear "no constraints" error itself.
        swept_names = [n for n in constraint_names if n in threshold_ranges]
        if not swept_names:
            raise ValueError(
                "No threshold_range entries supplied — frontier "
                "requires at least one threshold_ranges entry"
            )

        # Per-axis reporting value for unswept constraints — the
        # constructor threshold echoed verbatim into ``threshold_<name>``
        # (user units: absolute for ``min`` / ``max``, fractional for
        # ``min_pct`` / ``max_pct``).
        unswept_thresholds = {
            name: _spec_numeric_threshold(self.constraints[name])
            for name in constraint_names
            if name not in threshold_ranges
        }

        # Pre-flight zero-baseline-denominator check for ratio _pct
        # constraints. The contract is to raise at setup time before
        # any solve runs, naming the constraint and the zero condition.
        # ``_linearise_ratio_constraints`` enforces this, but we want
        # the error to fire BEFORE we burn any compute on the first
        # point so that callers see the failure mode immediately on
        # frontier dispatch rather than on the first internal solve.
        # Skip cleanly if a QuoteGrid was passed (no ratio possible —
        # ratios always come in via the DataFrame branch).
        if isinstance(df, pl.DataFrame):
            _check_zero_baseline_denominator_for_pct_ratios(
                df,
                self.constraints,
                scenario_value_col=self.scenario_value,
            )

        def solve_one(
            combo: list[float], prev_lambdas: dict[str, float] | None
        ) -> tuple[Any, dict[str, float]]:
            # Build a per-point constraint dict by overriding only the
            # swept axes' threshold values. Unswept axes copy their
            # constructor spec verbatim (direction key + numeric
            # threshold), so the inner solve enforces them at the
            # constructor value across every point. Direction key is
            # preserved on swept axes too (e.g. ``max_pct`` stays
            # ``max_pct`` so the linearisation scales by baseline LR
            # internally).
            per_point_constraints = _override_thresholds(
                self.constraints, combo, swept_names
            )
            point_solver = OnlineOptimiser(
                objective=self.objective,
                constraints=per_point_constraints,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                record_history=False,
            )
            result = point_solver.solve(df, lambdas=prev_lambdas)
            return result, dict(result.lambdas)

        def compose_row(
            result: Any, combo: list[float], swept: list[str]
        ) -> dict[str, Any]:
            row: dict[str, Any] = {
                "total_objective": float(result.total_objective),
                "iterations": int(result.iterations),
                "converged": bool(result.converged),
            }
            totals = result.total_constraints
            lambdas = result.lambdas
            swept_idx_map = {name: i for i, name in enumerate(swept)}
            for name in constraint_names:
                if name in swept_idx_map:
                    threshold_val = float(combo[swept_idx_map[name]])
                    row[f"threshold_{name}"] = threshold_val
                row[f"total_{name}"] = float(totals.get(name, float("nan")))
                row[f"lambda_{name}"] = float(lambdas.get(name, 0.0))

            # sv_* distribution stats from the per-quote optimal
            # scenario values. Mirrors the Rust frontier emitter so
            # existing downstream summary code (selected_sv_*) continues
            # to work for ratio frontiers without special-casing.
            sv_stats = _compute_sv_stats_from_dataframe(result.dataframe)
            row.update(sv_stats)
            return row

        return _python_frontier_orchestrator(
            constraint_names=constraint_names,
            swept_names=swept_names,
            threshold_ranges=threshold_ranges,
            unswept_thresholds=unswept_thresholds,
            n_points_per_dim=n_points_per_dim,
            max_total_points=max_total_points,
            initial_lambdas=initial_lambdas,
            solve_one=solve_one,
            compose_row=compose_row,
        )

    def config_dict(self) -> dict[str, Any]:
        """Return a serialisable dict of the solver configuration."""
        return {
            "objective": self.objective,
            "constraints": self.constraints,
            "quote_id": self.quote_id,
            "scenario_index": self.scenario_index,
            "scenario_value": self.scenario_value,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "record_history": self.record_history,
        }

    def summary(self, result: SolveResult) -> dict[str, Any]:
        """Package a solve result into MLflow-ready dicts.

        Returns a dict with three keys:

        ``params``
            Flat ``dict[str, str | int | float]`` — pass straight to
            ``mlflow.log_params()``.

        ``metrics``
            Flat ``dict[str, float]`` — pass straight to
            ``mlflow.log_metrics()``.

        ``artifacts``
            ``dict[str, Any]`` of serialisable objects:

            - ``"lambdas"`` — ``dict[str, float]``, write as JSON.
            - ``"config"`` — ``dict``, write as JSON.
            - ``"summary"`` — ``dict``, human-readable overview, write as JSON.
            - ``"convergence"`` — ``pl.DataFrame | None``, write as Parquet
              (present only when ``record_history=True``).
        """
        config = self.config_dict()
        baseline_obj = result.baseline_objective
        out_df = result.dataframe
        opt_mults = out_df["optimal_scenario_value"]

        # --- params (flat, all scalars) ---
        params: dict[str, Any] = {
            "objective": config["objective"],
            "max_iter": config["max_iter"],
            "tolerance": config["tolerance"],
            "n_quotes": result.n_quotes,
            "n_steps": result.n_steps,
        }
        if config["constraints"]:
            params["constraints"] = json.dumps(config["constraints"])
        mults = result.scenario_values
        if mults:
            params["scenario_value_min"] = round(float(min(mults)), 4)
            params["scenario_value_max"] = round(float(max(mults)), 4)

        # --- metrics (flat, all floats) ---
        metrics: dict[str, float] = {
            "total_objective": result.total_objective,
            "baseline_objective": baseline_obj,
            "iterations": float(result.iterations),
            "converged": float(result.converged),
        }
        if baseline_obj != 0:
            metrics["uplift_pct"] = (
                (result.total_objective - baseline_obj) / abs(baseline_obj)
            ) * 100

        for name, total in result.total_constraints.items():
            metrics[f"constraint_{name}_total"] = total
        for name, baseline in result.baseline_constraints.items():
            metrics[f"constraint_{name}_baseline"] = baseline
            if baseline != 0:
                metrics[f"constraint_{name}_ratio"] = (
                    result.total_constraints.get(name, 0.0) / baseline
                )
        for name, lam in result.lambdas.items():
            metrics[f"lambda_{name}"] = lam

        metrics["scenario_value_mean"] = float(opt_mults.mean())
        sv_std = float(opt_mults.std()) if len(opt_mults) > 1 else 0.0
        metrics["scenario_value_std"] = sv_std
        for pct in (5, 25, 50, 75, 95):
            metrics[f"scenario_value_p{pct}"] = float(opt_mults.quantile(pct / 100))

        # --- artifacts ---
        # Human-readable summary
        summary_dict: dict[str, Any] = {
            "solver_type": "online",
            "objective": config["objective"],
            "n_quotes": result.n_quotes,
            "n_steps": result.n_steps,
            "iterations": result.iterations,
            "converged": result.converged,
            "total_objective": result.total_objective,
            "baseline_objective": baseline_obj,
            "lambdas": result.lambdas,
            "constraints": {},
            "scenario_value_distribution": {
                "mean": metrics["scenario_value_mean"],
                "std": metrics["scenario_value_std"],
                "p5": metrics["scenario_value_p5"],
                "p25": metrics["scenario_value_p25"],
                "p50": metrics["scenario_value_p50"],
                "p75": metrics["scenario_value_p75"],
                "p95": metrics["scenario_value_p95"],
            },
            "config": config,
        }
        if baseline_obj != 0:
            summary_dict["uplift_pct"] = metrics["uplift_pct"]
        for name, total in result.total_constraints.items():
            baseline = result.baseline_constraints.get(name, 0.0)
            entry: dict[str, Any] = {
                "total": total,
                "baseline": baseline,
                "lambda": result.lambdas.get(name, 0.0),
                "spec": config["constraints"].get(name, {}),
            }
            if baseline != 0:
                entry["ratio_to_baseline"] = total / baseline
            summary_dict["constraints"][name] = entry

        convergence_df: pl.DataFrame | None = None
        if result.history is not None:
            rows = []
            for rec in result.history:
                row: dict[str, Any] = {
                    "iteration": rec["iteration"],
                    "total_objective": rec["total_objective"],
                    "max_lambda_change": rec["max_lambda_change"],
                    "all_constraints_satisfied": rec["all_constraints_satisfied"],
                }
                for cname, val in rec["lambdas"].items():
                    row[f"lambda_{cname}"] = val
                for cname, val in rec["total_constraints"].items():
                    row[f"constraint_{cname}"] = val
                rows.append(row)
            convergence_df = pl.DataFrame(rows)

        artifacts: dict[str, Any] = {
            "lambdas": result.lambdas,
            "config": config,
            "summary": summary_dict,
            "convergence": convergence_df,
        }

        return {
            "params": params,
            "metrics": metrics,
            "artifacts": artifacts,
        }


# Tuple (not set) so error messages render with deterministic order
# matching the Rust-side `VALID_KEYS` array. Module-level so both sum and
# ratio validation reuse the same canonical list.
_DIRECTION_KEYS = ("min", "max", "min_pct", "max_pct")


def _is_ratio_spec(spec: dict) -> bool:
    """Return True iff a spec is a ratio constraint.

    Detection is the **conjunction** of ``numerator`` and ``denominator``
    keys. A loose disjunction would silently absorb sum specs that
    accidentally include only one of the two pair keys (e.g. a typo of
    ``"numerator"`` on a sum spec); a single-key sum spec is more useful
    if it falls through to the existing sum-shape error than if it gets
    re-routed into the ratio branch and produces a misleading message.

    The single-side case (one of ``numerator`` / ``denominator`` present
    without the other) is still detected and surfaced separately —
    :func:`_validate_constraint_dict` checks the OR-condition and routes
    such specs into :func:`_validate_ratio_spec` for a precise error.
    """
    return "numerator" in spec and "denominator" in spec


def _validate_ratio_spec(name: str, spec: dict) -> None:
    """Validate a single ratio-constraint spec body.

    Called once we know the spec belongs in the ratio branch. The caller
    (:func:`_validate_constraint_dict`) is responsible for the dispatch
    decision; this function only enforces the ratio-specific rules.

    Rules enforced (in this order — first failure wins):

    1. Both ``numerator`` and ``denominator`` keys present (single-side
       reach-here paths from a partial spec are rejected with a message
       naming the missing key AND the constraint label).
    2. Both pair values are ``str``.
    3. Numerator and denominator names differ (degenerate ratio == 1.0).
    4. Exactly one direction key from ``_DIRECTION_KEYS`` alongside.
    5. Direction value is numeric (int/float) or ``None`` (B1).
    6. If numeric, value is finite (no NaN / inf — mirrors the sum rule).
    """
    # Rule 1: both pair keys must be present. Single-side calls produce
    # a precise error naming the missing key plus the constraint label.
    if "numerator" not in spec:
        raise ValueError(
            f"Ratio constraint '{name}' is missing 'numerator'; "
            f"both 'numerator' and 'denominator' must be supplied as "
            f"column names."
        )
    if "denominator" not in spec:
        raise ValueError(
            f"Ratio constraint '{name}' is missing 'denominator'; "
            f"both 'numerator' and 'denominator' must be supplied as "
            f"column names."
        )

    numerator = spec["numerator"]
    denominator = spec["denominator"]

    # Rule 2: numerator / denominator must be column-name strings.
    if not isinstance(numerator, str):
        raise ValueError(
            f"Ratio constraint '{name}' has non-string 'numerator' "
            f"(got {type(numerator).__name__}); must be a column name."
        )
    if not isinstance(denominator, str):
        raise ValueError(
            f"Ratio constraint '{name}' has non-string 'denominator' "
            f"(got {type(denominator).__name__}); must be a column name."
        )

    # Rule 3: degenerate same-column ratio is almost always a typo. Name
    # both the duplicated column AND the constraint label so the user can
    # locate it without grepping.
    if numerator == denominator:
        raise ValueError(
            f"Ratio constraint '{name}' has identical 'numerator' and "
            f"'denominator' ('{numerator}'); this would be a degenerate "
            f"ratio of 1.0 — pick distinct columns."
        )

    # Rule 4: exactly one direction key alongside the pair, matching the
    # sum-constraint rule. Zero or multiple → error naming the label and
    # listing valid direction keys.
    direction_keys_present = [k for k in _DIRECTION_KEYS if k in spec]
    if len(direction_keys_present) == 0:
        raise ValueError(
            f"Ratio constraint '{name}' has no direction key; must "
            f"include exactly one of: {list(_DIRECTION_KEYS)}."
        )
    if len(direction_keys_present) > 1:
        raise ValueError(
            f"Ratio constraint '{name}' has multiple direction keys "
            f"{direction_keys_present}; must include exactly one of: "
            f"{list(_DIRECTION_KEYS)}."
        )

    direction_key = direction_keys_present[0]
    value = spec[direction_key]

    # Rule 5 + 6: same numeric / None contract as sum constraints (B1).
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(
            f"Ratio constraint '{name}' value for '{direction_key}' must "
            f"be numeric or None, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ValueError(
            f"Ratio constraint '{name}' value for '{direction_key}' must "
            f"be a finite number, got {value}"
        )


def _validate_constraint_dict(
    constraints: dict[str, dict[str, float]],
) -> None:
    """Validate constraint specification dict structure and values.

    Post-A1 valid keys for **sum** constraints:
    * ``min`` / ``max``         → absolute thresholds.
    * ``min_pct`` / ``max_pct`` → fraction-of-baseline thresholds.

    Ratio constraints (C1) are detected by the presence of BOTH a
    ``numerator`` and a ``denominator`` key. The dict key on the outer
    constraints map is a *display label* for ratio constraints (not a
    column name), in contrast to sum constraints where the key is the
    column name.

    The old ``min_abs`` / ``max_abs`` keys have been removed and raise a
    migration ``ValueError`` naming both the removed key and its
    replacement. The migration error fires BEFORE the ratio-detection
    branch so that a user porting old code who happens to add ratio
    fields still sees the rename hint first.
    """
    for name, spec in constraints.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"Constraint '{name}' value must be a dict, got {type(spec).__name__}"
            )
        # Surface the rename hint BEFORE the generic invalid-key error so
        # users see the migration path instead of "invalid key". This
        # also takes precedence over the ratio-detection branch — a user
        # porting code who accidentally added ratio fields still sees the
        # rename hint first.
        if "min_abs" in spec:
            raise ValueError(
                "'min_abs' has been renamed to 'min'; "
                "the previous fraction-of-baseline 'min' is now 'min_pct'"
            )
        if "max_abs" in spec:
            raise ValueError(
                "'max_abs' has been renamed to 'max'; "
                "the previous fraction-of-baseline 'max' is now 'max_pct'"
            )
        # Ratio detection: route into the ratio validator if either pair
        # key is present (the ratio validator itself enforces "both must
        # be present"). This is the OR-form so a partial spec gets a
        # precise "missing numerator" / "missing denominator" message
        # rather than falling through into the sum-shape error.
        if "numerator" in spec or "denominator" in spec:
            _validate_ratio_spec(name, spec)
            continue
        if len(spec) != 1:
            raise ValueError(
                f"Constraint '{name}' must have exactly one key from "
                f"{list(_DIRECTION_KEYS)}, got {list(spec.keys())}"
            )
        key = next(iter(spec))
        if key not in _DIRECTION_KEYS:
            raise ValueError(
                f"Constraint '{name}' has invalid key '{key}'. "
                f"Must be one of: {list(_DIRECTION_KEYS)}"
            )
        value = spec[key]
        # ``None`` is a valid frontier-only marker (B1): the constraint will
        # be supplied by the frontier sweep, not by the caller. Numeric
        # values still go through type + finite checks.
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Constraint '{name}' value for '{key}' must be numeric or None, "
                f"got {type(value).__name__}"
            )
        # NaN and inf pass ``isinstance(_, float)`` but are not meaningful
        # thresholds; reject them explicitly so the solver does not
        # silently propagate non-finite values into the dual.
        if not math.isfinite(value):
            raise ValueError(
                f"Constraint '{name}' value for '{key}' must be a finite "
                f"number, got {value}"
            )


def _none_threshold_constraints(
    constraints: dict[str, dict[str, float | None]],
) -> list[str]:
    """Return the names of constraints whose threshold value is ``None``.

    Used by ``solve()`` paths to identify frontier-only constraints that
    must be rejected (``None`` has no meaning when no sweep is running),
    and by ``frontier()`` paths to confirm a matching ``threshold_ranges``
    entry exists for each.

    Post-A1 each sum spec has exactly one direction key (``min`` / ``max`` /
    ``min_pct`` / ``max_pct``). For ratio specs (C1), the direction key
    sits alongside ``numerator`` / ``denominator`` so we look it up
    explicitly rather than reading the first dict value.
    """
    out: list[str] = []
    for name, spec in constraints.items():
        if not (isinstance(spec, dict) and spec):
            continue
        if _is_ratio_spec(spec):
            for k in _DIRECTION_KEYS:
                if k in spec and spec[k] is None:
                    out.append(name)
                    break
        else:
            if next(iter(spec.values())) is None:
                out.append(name)
    return out


def _ratio_constraint_names(
    constraints: dict[str, dict[str, float | None]],
) -> list[str]:
    """Return the names of ratio constraints (in iteration order)."""
    return [
        name
        for name, spec in constraints.items()
        if isinstance(spec, dict) and _is_ratio_spec(spec)
    ]


def _reject_ratio_for_grid(
    constraints: dict[str, dict[str, float | None]],
    *,
    mode: str = "solve",
) -> None:
    """Raise ``ValueError`` if any ratio constraint is paired with a
    pre-built grid input.

    The synthetic-column materialisation happens in Python on the
    DataFrame branch, so a ratio constraint passed alongside a
    pre-built ``QuoteGrid`` cannot be linearised retroactively (the
    grid is opaque and frozen). This is a setup-time error, not an
    unimplemented feature, so we raise ``ValueError`` (NOT
    ``NotImplementedError`` — the feature exists, just not via this
    entry point) and point the user at the DataFrame-shape entry.

    ``mode`` selects the message suffix:

    * ``"solve"``    — generic message (default).
    * ``"apply"``    — apply-mode wording with apply-time baseline note.
    * ``"frontier"`` — frontier-mode wording mentioning each point's
      linearisation.
    """
    ratio_names = _ratio_constraint_names(constraints)
    if not ratio_names:
        return
    name = ratio_names[0]
    if mode == "apply":
        raise ValueError(
            f"Ratio constraint '{name}' requires a DataFrame input "
            f"(not a pre-built QuoteGrid) so the linearisation can "
            f"materialise the synthetic numerator-minus-L*denominator "
            f"column at apply time; the QuoteGrid path is not "
            f"supported. Use ApplyOptimiser.apply(df) on the raw "
            f"DataFrame instead."
        )
    if mode == "frontier":
        raise ValueError(
            f"Ratio constraint '{name}' requires a DataFrame input "
            f"(not a pre-built QuoteGrid) so each frontier point's "
            f"linearisation can materialise the synthetic "
            f"numerator-minus-L*denominator column; the QuoteGrid "
            f"path is not supported."
        )
    raise ValueError(
        f"Ratio constraint '{name}' requires a DataFrame input "
        f"(not a pre-built QuoteGrid) so the linearisation can "
        f"materialise the synthetic numerator-minus-L*denominator "
        f"column; the QuoteGrid path is not supported via this entry "
        f"point."
    )


def _reject_none_for_solve(
    constraints: dict[str, dict[str, float | None]],
) -> None:
    """Raise ``ValueError`` if any constraint has a ``None`` threshold.

    ``None`` marks a frontier-only constraint; ``solve()`` cannot pick a
    threshold from thin air, so the caller must either supply a numeric
    value or use ``frontier()`` with a matching ``threshold_ranges`` entry.

    The error names the offending constraint AND mentions ``frontier()``
    so the user immediately sees the migration path.
    """
    none_names = _none_threshold_constraints(constraints)
    if none_names:
        name = none_names[0]
        raise ValueError(
            f"Constraint '{name}' has no threshold (value is None); "
            f"solve() requires a numeric threshold per constraint. "
            f"Use frontier() with a matching threshold_ranges entry to sweep "
            f"this constraint, or supply a numeric threshold."
        )


def _linearise_ratio_constraints(
    df: pl.DataFrame,
    constraints: dict[str, dict[str, float | None]],
    *,
    scenario_value_col: str,
    quote_id_col: str,
) -> tuple[
    pl.DataFrame,
    dict[str, dict[str, float | None]],
    list[str],
    list[tuple[str, str, str]],
    dict[str, float],
]:
    """Linearise ratio constraints into synthetic sum constraints.

    Each ratio constraint ``Sigma num_i / Sigma denom_i ⟂ L`` is rewritten
    as ``Sigma (num_i - L * denom_i) ⟂ 0``. This function:

    * computes the per-ratio threshold ``L`` (verbatim for ``min`` / ``max``,
      ``pct * baseline_LR`` for ``min_pct`` / ``max_pct``);
    * materialises the synthetic per-quote-step column
      ``c_i = num_i - L * denom_i`` on a working copy of the input
      DataFrame, named for the ratio's display label;
    * builds a sum-shape constraints dict where each ratio spec is
      replaced by ``{<direction>: 0.0}`` while sum specs pass through
      untouched (direction preserved: max-ratio → max-sum, min-ratio →
      min-sum);
    * collects the deduplicated list of grid columns the downstream
      Rust solver should see — the original sum-constraint columns plus
      the synthetic linearised column. Numerator / denominator columns
      are NOT pushed into the grid here: they are recorded separately
      so the per-ratio downstream wrapper can stitch in
      ``optimal_<numerator>`` / ``optimal_<denominator>`` columns
      without bloating the dual update with extra slack constraints.

    Setup-time rejection:

    * ``min_pct`` / ``max_pct`` modes need ``baseline_LR``; if
      ``Sigma_baseline denom == 0`` the ratio is undefined and we
      raise ``ValueError`` naming the constraint and the zero condition.
    * Absolute ``min`` / ``max`` modes do NOT depend on ``baseline_LR``
      and are exempt from the zero-denom check.

    Per-row denominator == 0 is *not* rejected: the linearisation
    ``c_i = num_i - L * denom_i`` reduces to ``c_i = num_i`` for those
    rows, which is the correct contribution to the linearised sum.
    Down-stream consumers should be aware that an aggregate ratio
    computed over a subset where ``Sigma denom == 0`` will surface as
    ``nan`` via :func:`_safe_ratio_from_columns` (the ratio sentinel),
    not silently as zero.

    Returns
    -------
    (modified_df, sum_constraints, grid_constraint_cols, ratio_columns,
     threshold_shift)
        ``modified_df`` is the working DataFrame with synthetic
        linearised columns appended; the original ``df`` is not mutated.
        ``sum_constraints`` mirrors ``constraints`` with ratio specs
        rewritten to sum specs (direction preserved, threshold = 0).
        ``grid_constraint_cols`` is the deduplicated list of column
        names the grid should carry (sum cols + linearised cols, NOT
        numerator / denominator). ``ratio_columns`` is a list of
        ``(label, numerator, denominator)`` triples used by the result
        wrapper to emit the ``optimal_<numerator>`` /
        ``optimal_<denominator>`` columns. ``threshold_shift`` is a
        forward-compatibility hook returning per-ratio offsets the
        wrapper would subtract from reported ``total_constraints`` /
        history; currently always 0.0 for the natural threshold=0
        linearisation, but kept in the signature so a future numerical
        adjustment can plumb through without an API break.
    """
    sum_constraints: dict[str, dict[str, float | None]] = {}
    grid_cols: list[str] = []
    seen: set[str] = set()
    ratio_columns: list[tuple[str, str, str]] = []
    threshold_shift: dict[str, float] = {}

    def _track(col: str) -> None:
        if col not in seen:
            seen.add(col)
            grid_cols.append(col)

    # Ratio label collision: the synthetic linearised column is
    # materialised under the ratio's display label, so a label that
    # already names a real column in the DataFrame would clobber it on
    # the working copy. Reject up-front so the user gets a clear, named
    # error rather than a silent data-corruption bug.
    existing_cols = set(df.columns)
    for name, spec in constraints.items():
        if _is_ratio_spec(spec) and name in existing_cols:
            raise ValueError(
                f"Ratio constraint '{name}' label collides with an "
                f"existing DataFrame column of the same name. The "
                f"label is used as the synthetic linearised column "
                f"name; pick a different label or rename the column."
            )

    # First pass: aggregate per-ratio numerator / denominator baseline
    # totals via a single ``filter + sum`` over the baseline rows. We
    # batch all numerator / denominator columns into one ``select`` so
    # Polars can vectorise the aggregation; the alternative
    # ``df.filter(...)[col].sum()`` per side iterates the predicate
    # expression once per column and is measurably slower on the perf
    # fixture.
    ratio_items: list[tuple[str, dict[str, float | None]]] = [
        (name, spec) for name, spec in constraints.items() if _is_ratio_spec(spec)
    ]

    baseline_totals: dict[str, float] = {}
    if ratio_items:
        baseline_cols: list[str] = []
        baseline_seen: set[str] = set()
        for _, spec in ratio_items:
            for col in (spec["numerator"], spec["denominator"]):
                if col not in baseline_seen:
                    baseline_seen.add(col)
                    baseline_cols.append(col)
        baseline_df = df.filter(pl.col(scenario_value_col) == 1.0)
        if baseline_cols:
            agg = baseline_df.select(
                [pl.col(c).cast(pl.Float64).sum().alias(c) for c in baseline_cols]
            )
            if agg.height > 0:
                row = agg.row(0)
                for col, val in zip(baseline_cols, row):
                    baseline_totals[col] = float(val) if val is not None else 0.0
            else:
                # Empty baseline: every aggregated total is treated as
                # zero so the ``_pct`` zero-denom check fires uniformly.
                for col in baseline_cols:
                    baseline_totals[col] = 0.0

    # Second pass: walk constraints in iteration order and build the
    # synthetic column expressions.
    new_columns: list[pl.Expr] = []
    for name, spec in constraints.items():
        if not _is_ratio_spec(spec):
            sum_constraints[name] = spec
            _track(name)
            continue

        numerator_col = spec["numerator"]
        denominator_col = spec["denominator"]

        # Resolve direction key (validation already enforced exactly one).
        direction_key: str | None = None
        for k in _DIRECTION_KEYS:
            if k in spec:
                direction_key = k
                break
        if direction_key is None:
            # Defensive: validator enforces this.
            raise ValueError(
                f"Ratio constraint '{name}' missing direction key."
            )
        threshold_value = spec[direction_key]
        if threshold_value is None:
            raise ValueError(
                f"Constraint '{name}' has no threshold (value is None); "
                f"solve() requires a numeric threshold per constraint. "
                f"Use frontier() with a matching threshold_ranges entry "
                f"to sweep this constraint, or supply a numeric threshold."
            )

        # Compute L. Absolute modes use the value verbatim; pct modes
        # need ``baseline_LR = Sigma_baseline num / Sigma_baseline denom``,
        # which is undefined when ``Sigma_baseline denom == 0``.
        is_pct = direction_key in ("min_pct", "max_pct")
        denom_baseline_total = baseline_totals.get(denominator_col, 0.0)
        if is_pct:
            if denom_baseline_total == 0.0:
                raise ValueError(_zero_denom_message(name, denominator_col))
            num_total = baseline_totals.get(numerator_col, 0.0)
            baseline_lr = num_total / denom_baseline_total
            L = float(threshold_value) * baseline_lr
        else:
            L = float(threshold_value)

        # Map ratio direction key onto the corresponding sum-direction.
        # ``min`` / ``min_pct`` → sum ``min``; ``max`` / ``max_pct`` →
        # sum ``max``. Threshold collapses to 0 because the linearisation
        # rewrote the constraint as ``Sigma c_i ⟂ 0``.
        sum_direction = "min" if direction_key in ("min", "min_pct") else "max"
        sum_constraints[name] = {sum_direction: 0.0}
        threshold_shift[name] = 0.0

        # Materialise the synthetic column ``c_i = num_i - L * denom_i``.
        # Cast to Float32 to match the existing constraint-column dtype
        # contract; otherwise the QuoteGridBuilder rejects with
        # "<col> must be Float32".
        new_columns.append(
            (pl.col(numerator_col) - L * pl.col(denominator_col))
            .cast(pl.Float32)
            .alias(name)
        )

        _track(name)
        ratio_columns.append((name, numerator_col, denominator_col))

    if new_columns:
        df = df.with_columns(new_columns)

    return df, sum_constraints, grid_cols, ratio_columns, threshold_shift


class _RatioSolveResultWrapper:
    """Decorate a :class:`SolveResult` with ratio-aware reporting.

    The Rust solver materialises ``optimal_<colname>`` for every
    constraint column in the grid. We pass only the linearised column
    (label) into the grid to keep the dual update lean — extra slack
    constraints would multiply the per-iteration constraint-loop cost
    by O(N_extra) and bust the C2 perf budget. So the result DataFrame
    out of Rust is missing ``optimal_<numerator>`` / ``optimal_<denominator>``
    columns that the test contract (and downstream ratio reporting)
    expects, and the inner ``total_constraints`` / ``baseline_constraints``
    map each ratio label to its linearised total ``Sigma (num - L*denom)``
    rather than the actual ratio.

    The wrapper papers over these gaps:

    * ``dataframe``: stitches in ``optimal_<numerator>`` /
      ``optimal_<denominator>`` columns via a join against the original
      input DataFrame on ``(quote_id, scenario_index)``. Cached on first
      access so repeat reads are O(1).
    * ``total_constraints[<ratio_label>]`` (C3): replaces the inner's
      linearised total with the actual ratio
      ``Sigma_optimal num / Sigma_optimal denom``, computed from the
      surfaced ``optimal_*`` columns. Sum constraints pass through.
    * ``baseline_constraints[<ratio_label>]`` (C3): replaces the inner's
      linearised baseline with the actual baseline ratio
      ``Sigma_baseline num / Sigma_baseline denom``, computed from the
      ``scenario_value == 1.0`` slice of the original DataFrame. Sum
      constraints pass through.
    * ``history`` (C3): for each iteration record, replays the
      Lagrangian argmax in pure Python using the record's lambdas to
      recover that iteration's optimal_steps, then computes the actual
      ratio from the original DataFrame at those steps. Sum entries
      pass through unchanged (the inner's recorded value is already a
      sum).

    Why the Python replay for history.  The Rust ``IterationRecord``
    carries only the aggregated ``total_constraints: Vec<f64>`` (the
    linearised totals for synthetic ratio columns); per-iteration
    ``Sigma num`` and ``Sigma denom`` are not currently surfaced. Two
    routes were considered:

    (a) Extend Rust ``IterationRecord`` to carry per-iteration ratio
        sums alongside the linearised total. Cleaner long-term but
        invasive (Rust struct change, pyo3 binding, ratebook /
        grouped solver follow-on).
    (b) Replay the Lagrangian argmax in Python on the linearised
        DataFrame for each iteration's lambdas. ``record_history`` is
        an opt-in diagnostic feature (off by default), so the cost of
        an O(N * max_iter) Python pass is acceptable.

    We picked (b) for C3 to keep the diff small. The replay reproduces
    the Rust solver's per-iteration optimal_steps because the recorded
    lambdas are the lambdas USED for that iteration's argmax (not the
    post-update lambdas), and the argmax is deterministic given lambdas
    + grid. A future feature can switch to (a) without breaking the
    Python contract.

    Reporting adjustment for ``threshold_shift``. The linearisation
    maps each ratio label to a Rust threshold via ``threshold_shift``
    (currently always 0.0 — see :func:`_linearise_ratio_constraints`
    for the forward-compat rationale). The shift is preserved here as a
    no-op because the C3 reporting recomputes from per-row sums that
    don't go through the shifted Rust values; the field is kept for
    API stability so a future linearisation refinement can plumb a
    non-zero shift without further changes here.

    All other attributes / properties delegate via ``__getattr__`` so
    the wrapper is API-compatible with the underlying ``SolveResult``.
    """

    __slots__ = (
        "_inner",
        "_original_df",
        "_modified_df",
        "_ratio_columns",
        "_sum_constraints",
        "_quote_id",
        "_scenario_index",
        "_scenario_value",
        "_objective",
        "_threshold_shift",
        "_dataframe_cache",
        "_total_constraints_cache",
        "_baseline_constraints_cache",
        "_history_cache",
    )

    def __init__(
        self,
        inner: SolveResult,
        *,
        original_df: pl.DataFrame,
        modified_df: pl.DataFrame,
        ratio_columns: list[tuple[str, str, str]],
        sum_constraints: dict[str, dict[str, float | None]],
        quote_id: str,
        scenario_index: str,
        scenario_value: str,
        objective: str,
        threshold_shift: dict[str, float],
    ) -> None:
        self._inner = inner
        self._original_df = original_df
        self._modified_df = modified_df
        self._ratio_columns = ratio_columns
        self._sum_constraints = sum_constraints
        self._quote_id = quote_id
        self._scenario_index = scenario_index
        self._scenario_value = scenario_value
        self._objective = objective
        self._threshold_shift = threshold_shift
        self._dataframe_cache: pl.DataFrame | None = None
        self._total_constraints_cache: dict[str, float] | None = None
        self._baseline_constraints_cache: dict[str, float] | None = None
        self._history_cache: list[dict[str, Any]] | None = None

    def __getattr__(self, name: str) -> Any:
        # Delegate any attribute we don't override to the wrapped
        # SolveResult. ``__getattr__`` only fires for attributes not
        # found via the normal MRO, so the explicit overrides below
        # take precedence. Dunder attributes are bounced to AttributeError
        # so pickle / copy / repr machinery doesn't recurse through the
        # delegate looking for protocol hooks the inner class doesn't
        # define.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    @property
    def total_constraints(self) -> dict[str, float]:
        if self._total_constraints_cache is not None:
            return self._total_constraints_cache
        # C3 contract: each ratio label reports the actual ratio at the
        # optimum, ``Sigma_optimal num / Sigma_optimal denom`` — recompute
        # from the surfaced ``optimal_<col>`` columns rather than reusing
        # the inner's linearised total. Sum constraints pass through.
        out = dict(self._inner.total_constraints)
        if self._ratio_columns:
            df = self.dataframe
            for label, num_col, denom_col in self._ratio_columns:
                out[label] = _safe_ratio_from_columns(
                    df, f"optimal_{num_col}", f"optimal_{denom_col}"
                )
        self._total_constraints_cache = out
        return out

    @property
    def baseline_constraints(self) -> dict[str, float]:
        if self._baseline_constraints_cache is not None:
            return self._baseline_constraints_cache
        # C3 contract: each ratio label reports the actual baseline ratio
        # ``Sigma_baseline num / Sigma_baseline denom`` from rows where
        # ``scenario_value == 1.0``. Sum constraints pass through.
        out = dict(self._inner.baseline_constraints)
        if self._ratio_columns:
            baseline_slice = self._original_df.filter(
                pl.col(self._scenario_value) == 1.0
            )
            for label, num_col, denom_col in self._ratio_columns:
                out[label] = _safe_ratio_from_columns(
                    baseline_slice, num_col, denom_col
                )
        self._baseline_constraints_cache = out
        return out

    @property
    def history(self) -> list[dict[str, Any]] | None:
        if self._history_cache is not None:
            return self._history_cache
        inner_history = self._inner.history
        if inner_history is None:
            return None
        # C3 contract: each iteration's ``total_constraints`` entry for a
        # ratio label is the actual ratio at that iteration's optimal
        # steps, NOT the linearised value. The Rust IterationRecord only
        # carries the aggregated linearised total, so we replay the
        # Lagrangian argmax in pure Python (using the record's lambdas)
        # to recover each iteration's optimal_steps, then sum num / denom
        # from the original DataFrame to produce the actual ratio. Sum
        # entries are passed through unchanged. See the class docstring
        # for the route-(b) rationale.
        #
        # Final-iteration alignment.  When the solver does not converge,
        # the Rust path runs an extra averaged-lambdas pass AFTER the
        # main loop and reports the result via ``inner.total_constraints``
        # — the last history record's lambdas (the pre-update lambdas
        # of the final loop iteration) differ from the post-loop
        # ``inner.lambdas``. Re-using the final-pass result for the
        # LAST history entry keeps the trajectory's terminus consistent
        # with the surfaced ``total_constraints``: history[-1] is the
        # convergence point reached, not a transient one-step earlier.
        adjusted: list[dict[str, Any]] = []
        last_idx = len(inner_history) - 1
        # Cache the final actual-ratio map (computed from the wrapper's
        # ``dataframe``) once so the last entry's substitution is O(1).
        final_ratios: dict[str, float] | None = None
        if self._ratio_columns:
            final_ratios = {
                label: self.total_constraints[label]
                for label, _num, _den in self._ratio_columns
            }
        for i, rec in enumerate(inner_history):
            new_rec = dict(rec)
            tc = dict(rec["total_constraints"])
            if self._ratio_columns:
                if i == last_idx and final_ratios is not None:
                    # Use the final-pass ratios (same code path as the
                    # surfaced ``total_constraints``) so the trajectory's
                    # terminus matches the result. This collapses to the
                    # replay value when the solver converged (lambdas
                    # identical across the post-loop pass), and tracks
                    # the post-loop averaged-lambda pass otherwise.
                    tc.update(final_ratios)
                else:
                    ratios = self._actual_ratios_for_lambdas(rec["lambdas"])
                    tc.update(ratios)
            new_rec["total_constraints"] = tc
            adjusted.append(new_rec)
        self._history_cache = adjusted
        return adjusted

    def _actual_ratios_for_lambdas(
        self, lambdas: dict[str, float]
    ) -> dict[str, float]:
        """Replay the Lagrangian argmax in Python and compute actual
        ratios per ratio label.

        The replay mirrors the Rust solver's per-quote argmax: build the
        per-quote-step Lagrangian ``L_qj = obj_qj + Sigma_k sign_k *
        lambda_k * c_k_qj`` on the linearised DataFrame, pick the step
        with maximum ``L`` per quote, then sum ``num`` / ``denom`` from
        the original DataFrame at the chosen rows.

        Sign convention matches :func:`compute_lambda_signs_f32` in the
        Rust solver: ``+lambda`` for Min direction, ``-lambda`` for Max.
        """
        # Build the Lagrangian column on the linearised DataFrame.
        # Sum constraints (post-linearisation) include both original
        # sum specs and the synthetic ratio columns rewritten as sum
        # specs; their direction key drives the sign. Cast to Float32
        # to match the Rust solver's f32 argmax precision so the
        # Python replay reproduces the same per-quote optimal step
        # under tie-breaks.
        lagr_expr = pl.col(self._objective).cast(pl.Float32)
        for name, spec in self._sum_constraints.items():
            direction = next(iter(spec))  # "min" or "max"
            sign = 1.0 if direction == "min" else -1.0
            lam = float(lambdas.get(name, 0.0))
            if lam == 0.0:
                continue
            lagr_expr = lagr_expr + sign * lam * pl.col(name).cast(pl.Float32)
        work = self._modified_df.with_columns(_lagrangian=lagr_expr)
        # Per-quote argmax → optimal_step per quote. ``maintain_order``
        # keeps quotes in insertion order for deterministic downstream
        # joins; ``arg_max`` is stable on ties (returns the first index)
        # which mirrors the Rust solver's tie-break.
        opt_steps = (
            work.group_by(self._quote_id, maintain_order=True).agg(
                pl.col(self._scenario_index)
                .get(pl.col("_lagrangian").arg_max())
                .alias("_optimal_step")
            )
        )
        # Inner-join back onto the original DataFrame at (quote_id,
        # scenario_index == optimal_step) to pull num / denom values at
        # the chosen step for each quote.
        chosen = self._original_df.join(
            opt_steps,
            left_on=[self._quote_id, self._scenario_index],
            right_on=[self._quote_id, "_optimal_step"],
            how="inner",
        )
        out: dict[str, float] = {}
        for label, num_col, denom_col in self._ratio_columns:
            out[label] = _safe_ratio_from_columns(chosen, num_col, denom_col)
        return out

    @property
    def dataframe(self) -> pl.DataFrame:
        if self._dataframe_cache is not None:
            return self._dataframe_cache
        base = self._inner.dataframe
        # Stitch in numerator / denominator columns via the shared helper
        # so the wrapper's join + dedup logic stays identical to the apply
        # wrapper and the ratebook stitcher.
        joined = _stitch_optimal_ratio_columns(
            base_df=base,
            original_df=self._original_df,
            ratio_columns=self._ratio_columns,
            quote_id_col=self._quote_id,
            scenario_index_col=self._scenario_index,
        )
        # The base DataFrame already contains ``optimal_<label>`` for
        # the linearised synthetic column. That column carries the
        # SHIFTED value ``c_i' = c_i + beta`` per row; subtract the
        # per-row beta on the ratio's optimal column so downstream
        # consumers see the natural linearised value. We use the
        # threshold_shift / n_quotes ratio to recover beta.
        n_rows_out = joined.height
        unshift_exprs: list[pl.Expr] = []
        for label, _num, _den in self._ratio_columns:
            shift = self._threshold_shift.get(label, 0.0)
            opt_col = f"optimal_{label}"
            if opt_col in joined.columns and shift != 0.0 and n_rows_out > 0:
                beta = shift / n_rows_out
                unshift_exprs.append(
                    (pl.col(opt_col) - beta).cast(pl.Float32).alias(opt_col)
                )
        if unshift_exprs:
            joined = joined.with_columns(unshift_exprs)
        self._dataframe_cache = joined
        return joined


class _RatioApplyResultWrapper:
    """Decorate an :class:`ApplyResult` with ratio-aware reporting (C6).

    Apply mode runs a single fixed-lambda forward pass on the linearised
    DataFrame. Like the solve wrapper, the inner ``ApplyResult`` is
    missing ``optimal_<numerator>`` / ``optimal_<denominator>`` columns
    and reports the linearised total / baseline rather than the actual
    ratio. This wrapper papers over those gaps using the same recipe as
    :class:`_RatioSolveResultWrapper`:

    * ``dataframe``: stitches in ``optimal_<numerator>`` /
      ``optimal_<denominator>`` columns via a join against the original
      input DataFrame on ``(quote_id, scenario_index)``.
    * ``total_constraints[<ratio_label>]``: the actual ratio
      ``Sigma_optimal num / Sigma_optimal denom`` from the surfaced
      ``optimal_*`` columns. Sum constraints pass through.
    * ``baseline_constraints[<ratio_label>]``: the actual baseline ratio
      ``Sigma_baseline num / Sigma_baseline denom`` over rows where
      ``scenario_value == 1.0`` on the apply-time DataFrame. Sum
      constraints pass through.

    No ``history`` attribute — :class:`ApplyResult` does not carry one
    (apply runs a single forward pass, no iteration trajectory).

    All other attributes / properties delegate via ``__getattr__`` so
    the wrapper is API-compatible with the underlying ``ApplyResult``.
    """

    __slots__ = (
        "_inner",
        "_original_df",
        "_ratio_columns",
        "_quote_id",
        "_scenario_index",
        "_scenario_value",
        "_dataframe_cache",
        "_total_constraints_cache",
        "_baseline_constraints_cache",
    )

    def __init__(
        self,
        inner: ApplyResult,
        *,
        original_df: pl.DataFrame,
        ratio_columns: list[tuple[str, str, str]],
        quote_id: str,
        scenario_index: str,
        scenario_value: str,
    ) -> None:
        self._inner = inner
        self._original_df = original_df
        self._ratio_columns = ratio_columns
        self._quote_id = quote_id
        self._scenario_index = scenario_index
        self._scenario_value = scenario_value
        self._dataframe_cache: pl.DataFrame | None = None
        self._total_constraints_cache: dict[str, float] | None = None
        self._baseline_constraints_cache: dict[str, float] | None = None

    def __getattr__(self, name: str) -> Any:
        # Delegate any attribute we don't override to the wrapped
        # ApplyResult. ``__getattr__`` only fires for attributes not
        # found via the normal MRO, so the explicit overrides below
        # take precedence. Dunder attributes are bounced to AttributeError
        # so pickle / copy / repr machinery doesn't recurse through the
        # delegate looking for protocol hooks the inner class doesn't
        # define.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    @property
    def total_constraints(self) -> dict[str, float]:
        if self._total_constraints_cache is not None:
            return self._total_constraints_cache
        out = dict(self._inner.total_constraints)
        if self._ratio_columns:
            df = self.dataframe
            for label, num_col, denom_col in self._ratio_columns:
                out[label] = _safe_ratio_from_columns(
                    df, f"optimal_{num_col}", f"optimal_{denom_col}"
                )
        self._total_constraints_cache = out
        return out

    @property
    def baseline_constraints(self) -> dict[str, float]:
        if self._baseline_constraints_cache is not None:
            return self._baseline_constraints_cache
        out = dict(self._inner.baseline_constraints)
        if self._ratio_columns:
            baseline_slice = self._original_df.filter(
                pl.col(self._scenario_value) == 1.0
            )
            for label, num_col, denom_col in self._ratio_columns:
                out[label] = _safe_ratio_from_columns(
                    baseline_slice, num_col, denom_col
                )
        self._baseline_constraints_cache = out
        return out

    @property
    def dataframe(self) -> pl.DataFrame:
        if self._dataframe_cache is not None:
            return self._dataframe_cache
        base = self._inner.dataframe
        # Stitch in numerator / denominator columns via the shared helper
        # so the wrapper's join + dedup logic stays identical to the solve
        # wrapper and the ratebook stitcher.
        joined = _stitch_optimal_ratio_columns(
            base_df=base,
            original_df=self._original_df,
            ratio_columns=self._ratio_columns,
            quote_id_col=self._quote_id,
            scenario_index_col=self._scenario_index,
        )
        self._dataframe_cache = joined
        return joined


def _validate_dataframe(
    df: pl.DataFrame,
    *,
    quote_id: str,
    scenario_index: str,
    scenario_value: str,
    objective: str,
    constraint_cols: list[str],
    constraints: dict[str, dict[str, float | None]] | None = None,
) -> None:
    """Validate DataFrame schema before passing to Rust layer.

    Sum-constraint columns come from ``constraint_cols`` (where the
    constraint key IS the column name). Ratio constraints are described
    by ``constraints`` — for each ratio spec the validator checks the
    ``numerator`` and ``denominator`` columns (the dict key is a label,
    not a column).

    Errors name the offending column AND, for ratio constraints, the
    constraint label so users can locate the problem without spelunking
    the DataFrame schema.

    NaN handling: NaN cells in any constraint column (sum or ratio
    numerator/denominator) and the objective column are rejected with
    a ValueError. Nulls (None) are also rejected. The two checks are
    distinct because Polars stores NaN floats inside the value array
    (``null_count`` returns 0 for them) while nulls are tracked by a
    separate validity bitmap.

    Ratio label-vs-column collision is *not* checked here — see
    :func:`_linearise_ratio_constraints` for the linearisation callsite.
    The collision is only meaningful when the linearisation actually
    runs (DataFrame branch of ``OnlineOptimiser.solve()`` and analogous
    entry points), so it lives next to that code.
    """
    sum_cols = list(constraint_cols)
    ratio_specs: list[tuple[str, str, str]] = []
    if constraints:
        # Filter out ratio constraints from the sum-cols list (their dict
        # key is a display label, not a column) and build a list of
        # (label, numerator, denominator) for the ratio-specific check.
        sum_cols = [
            c for c in constraint_cols if not _is_ratio_spec(constraints.get(c, {}))
        ]
        for name, spec in constraints.items():
            if _is_ratio_spec(spec):
                ratio_specs.append((name, spec["numerator"], spec["denominator"]))

    required = [quote_id, scenario_index, scenario_value, objective] + sum_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Check for nulls AND NaN in objective and sum-constraint columns.
    # NaN is checked explicitly because Polars stores floating-point NaN
    # inside the value array (``null_count`` returns 0). Both conditions
    # mean the value is non-finite and would silently propagate into the
    # dual update; reject up-front with a column-named message.
    for col_name in [objective] + sum_cols:
        _reject_nulls_and_nans(df, col_name, label_context=None)

    # Ratio-constraint column checks: existence + non-null + non-NaN.
    # Errors name both the offending column and the constraint label.
    # Note: label-vs-column collision is checked at the linearisation
    # call site (:func:`_linearise_ratio_constraints`), NOT here.
    for label, numerator_col, denominator_col in ratio_specs:
        for role, col in (("numerator", numerator_col), ("denominator", denominator_col)):
            if col not in df.columns:
                raise ValueError(
                    f"Ratio constraint '{label}' references {role} column "
                    f"'{col}' which is not in the DataFrame."
                )
            _reject_nulls_and_nans(df, col, label_context=(label, role))


def _reject_nulls_and_nans(
    df: pl.DataFrame,
    col_name: str,
    *,
    label_context: tuple[str, str] | None,
) -> None:
    """Raise ``ValueError`` if ``df[col_name]`` contains nulls or NaN.

    ``label_context`` tags the error with a ratio constraint's display
    label and role (``"numerator"`` / ``"denominator"``) when the column
    is a ratio side; ``None`` produces the bare-column message used by
    the objective and sum-constraint check.
    """
    series = df[col_name]
    if series.null_count() > 0:
        if label_context is not None:
            label, role = label_context
            raise ValueError(
                f"Ratio constraint '{label}' {role} column '{col_name}' "
                f"contains null values."
            )
        raise ValueError(f"Column '{col_name}' contains null values")
    # ``is_nan`` is only defined for float columns; non-float columns
    # cannot hold NaN by construction so the check is safely skipped.
    # Polars' ``is_nan`` raises on integer / Utf8 dtypes — guard with the
    # dtype check instead of a try / except so the validator stays cheap.
    if series.dtype.is_float():
        nan_count = int(series.is_nan().sum())
        if nan_count > 0:
            if label_context is not None:
                label, role = label_context
                raise ValueError(
                    f"Ratio constraint '{label}' {role} column '{col_name}' "
                    f"contains NaN values."
                )
            raise ValueError(f"Column '{col_name}' contains NaN values")


# ---------------------------------------------------------------------------
# Python-side frontier sweep (C4)
# ---------------------------------------------------------------------------
#
# When ratio constraints are present, ``OnlineOptimiser.frontier()``
# delegates to a Python sweep so that each frontier point can run the
# C2 linearisation in ``solve()``. The Rust ``sweep_frontier_py`` fast
# path stays in place for sum-only constraint dicts (no regression).


def _spec_direction_key(name: str, spec: dict) -> str:
    """Return the direction key (``min`` / ``max`` / ``min_pct`` /
    ``max_pct``) on a constraint spec. Raises ``ValueError`` naming the
    constraint if no direction key is present (defensive — the validator
    pins exactly one direction key per spec, so this should be
    unreachable for well-formed inputs).
    """
    for k in _DIRECTION_KEYS:
        if k in spec:
            return k
    raise ValueError(
        f"Constraint '{name}' has no direction key; cannot "
        f"override threshold for frontier sweep."
    )


def _spec_threshold_is_none(spec: dict) -> bool:
    """Return True iff the spec's direction key carries a ``None``
    threshold (B1 marker — frontier-only)."""
    for k in _DIRECTION_KEYS:
        if k in spec:
            return spec[k] is None
    return False


def _spec_numeric_threshold(spec: dict) -> float:
    """Return the numeric threshold on a spec's direction key. Caller
    must have already established the threshold is numeric (D1 holds
    unswept numeric thresholds at this value across every frontier
    point)."""
    for k in _DIRECTION_KEYS:
        if k in spec:
            val = spec[k]
            if val is None:
                raise ValueError(
                    "Constraint spec carries a None threshold; cannot "
                    "hold it fixed across the frontier sweep."
                )
            return float(val)
    raise ValueError("Constraint spec has no direction key.")


def _override_thresholds(
    constraints: dict[str, dict[str, float | None]],
    combo: list[float],
    swept_names: list[str],
) -> dict[str, dict[str, float | None]]:
    """Build a per-frontier-point constraint dict with the supplied
    thresholds substituted into each swept axis's direction key.

    Swept constraints (``swept_names``) take per-point thresholds from
    ``combo``; unswept constraints copy the constructor spec verbatim
    so the inner solve enforces them at their constructor threshold for
    every frontier point (D1 — numeric thresholds without a range stay
    fixed). The direction key (``min`` / ``max`` / ``min_pct`` /
    ``max_pct``) and ratio fields (``numerator`` / ``denominator``) are
    preserved on every spec.
    """
    out: dict[str, dict[str, float | None]] = {}
    swept_index = {name: idx for idx, name in enumerate(swept_names)}
    for name, spec in constraints.items():
        if name in swept_index:
            new_spec: dict[str, float | None] = {}
            # Copy ratio fields (numerator / denominator) verbatim.
            for k in ("numerator", "denominator"):
                if k in spec:
                    new_spec[k] = spec[k]
            direction_key = _spec_direction_key(name, spec)
            new_spec[direction_key] = float(combo[swept_index[name]])
            out[name] = new_spec
        else:
            # Unswept: copy the original spec verbatim so the inner
            # solve sees the constructor threshold (D1 — numeric
            # thresholds held fixed). Shallow copy is sufficient because
            # spec values are immutable (str / float / None).
            out[name] = dict(spec)
    return out


def _zero_denom_message(name: str, denom_col: str) -> str:
    """Canonical error wording for the ``Sigma_baseline denom == 0`` case
    on a ``_pct`` ratio constraint. Shared by the linearisation
    helper (per-ratio loop, raises after computing baseline_totals) and
    the frontier pre-flight check (raises before any solve runs)."""
    return (
        f"Ratio constraint '{name}': baseline denominator "
        f"sum (Sigma '{denom_col}' at scenario_value=1.0) "
        f"is zero, so baseline ratio is undefined. Absolute "
        f"'min' / 'max' modes do not need a baseline ratio; "
        f"use those if you don't want to anchor on baseline."
    )


def _check_zero_baseline_denominator_for_pct_ratios(
    df: pl.DataFrame,
    constraints: dict[str, dict[str, float | None]],
    *,
    scenario_value_col: str,
) -> None:
    """Pre-flight check for ``min_pct`` / ``max_pct`` ratio constraints.

    Mirrors the linearisation's setup-time error so the failure surfaces
    before any per-point solve runs. ``min`` / ``max`` ratio constraints
    don't depend on baseline LR and are exempt. Uses the same wording
    as :func:`_linearise_ratio_constraints` via ``_zero_denom_message``
    so the two paths stay in lockstep.
    """
    pct_ratios: list[tuple[str, str]] = []
    for name, spec in constraints.items():
        if not _is_ratio_spec(spec):
            continue
        for k in ("min_pct", "max_pct"):
            if k in spec:
                pct_ratios.append((name, spec["denominator"]))
                break
    if not pct_ratios:
        return
    baseline = df.filter(pl.col(scenario_value_col) == 1.0)
    for name, denom_col in pct_ratios:
        # Defensive: if the baseline slice is empty, treat as zero
        # (the linearisation helper does the same).
        if baseline.height == 0:
            denom_total = 0.0
        else:
            denom_total = float(baseline[denom_col].cast(pl.Float64).sum())
        if denom_total == 0.0:
            raise ValueError(_zero_denom_message(name, denom_col))


