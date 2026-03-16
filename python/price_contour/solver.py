"""OnlineOptimiser — Python orchestration for the online Lagrangian solver."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from price_contour._grid_utils import build_grid
from price_contour._price_contour import (
    FrontierResult,
    QuoteGrid,
    SolveResult,
    solve_from_grid_py,
    solve_online_py,
    sweep_frontier_py,
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
        with one of: ``min`` (relative), ``max`` (relative),
        ``min_abs`` (absolute), ``max_abs`` (absolute).
    max_iter : int
        Maximum solver iterations.
    chunk_size : int
        Deprecated: no longer used by the solver. Retained for API
        compatibility. Internal parallelism is managed by rayon grain sizes.
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
        chunk_size: int = 500_000,
        tolerance: float = 1e-5,
        record_history: bool = False,
    ) -> None:
        self.quote_id = quote_id
        self.scenario_index = scenario_index
        self.scenario_value = scenario_value
        self.objective = objective
        self.constraints = constraints or {}
        self.max_iter = max_iter
        self.chunk_size = chunk_size
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
            return solve_from_grid_py(
                df_or_grid,
                constraints=self.constraints,
                max_iter=self.max_iter,
                chunk_size=self.chunk_size,
                tolerance=self.tolerance,
                lambdas=lambdas,
                record_history=self.record_history,
            )
        if not isinstance(df_or_grid, pl.DataFrame):
            raise TypeError(
                f"Expected pl.DataFrame or QuoteGrid, got {type(df_or_grid).__name__}"
            )
        _validate_dataframe(
            df_or_grid,
            quote_id=self.quote_id,
            scenario_index=self.scenario_index,
            scenario_value=self.scenario_value,
            objective=self.objective,
            constraint_cols=list(self.constraints.keys()),
        )
        return solve_online_py(
            df_or_grid,
            quote_id=self.quote_id,
            scenario_index=self.scenario_index,
            scenario_value=self.scenario_value,
            objective=self.objective,
            constraints=self.constraints,
            max_iter=self.max_iter,
            chunk_size=self.chunk_size,
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
            For relative constraints (min/max), these are fractions of baseline.
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
        if isinstance(df_or_grid, pl.DataFrame):
            grid = build_grid(
                df_or_grid,
                constraint_columns=list(self.constraints.keys()),
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
            )
        else:
            grid = df_or_grid

        return sweep_frontier_py(
            grid,
            constraints=self.constraints,
            threshold_ranges=threshold_ranges,
            n_points_per_dim=n_points_per_dim,
            max_iter=self.max_iter,
            chunk_size=self.chunk_size,
            tolerance=self.tolerance,
            initial_lambdas=initial_lambdas,
            max_total_points=max_total_points,
            parallel=parallel,
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
            "chunk_size": self.chunk_size,
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
            "chunk_size": config["chunk_size"],
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


def _validate_constraint_dict(
    constraints: dict[str, dict[str, float]],
) -> None:
    """Validate constraint specification dict structure and values."""
    valid_keys = {"min", "max", "min_abs", "max_abs"}
    for name, spec in constraints.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"Constraint '{name}' value must be a dict, got {type(spec).__name__}"
            )
        if len(spec) != 1:
            raise ValueError(
                f"Constraint '{name}' must have exactly one key from "
                f"{valid_keys}, got {set(spec.keys())}"
            )
        key = next(iter(spec))
        if key not in valid_keys:
            raise ValueError(
                f"Constraint '{name}' has invalid key '{key}'. "
                f"Must be one of: {valid_keys}"
            )
        value = spec[key]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Constraint '{name}' value for '{key}' must be numeric, "
                f"got {type(value).__name__}"
            )


def _validate_dataframe(
    df: pl.DataFrame,
    *,
    quote_id: str,
    scenario_index: str,
    scenario_value: str,
    objective: str,
    constraint_cols: list[str],
) -> None:
    """Validate DataFrame schema before passing to Rust layer."""
    required = [quote_id, scenario_index, scenario_value, objective] + constraint_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Check for nulls in objective and constraint columns
    for col_name in [objective] + constraint_cols:
        if df[col_name].null_count() > 0:
            raise ValueError(f"Column '{col_name}' contains null values")
