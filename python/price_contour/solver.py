"""OnlineOptimiser — Python orchestration for the online Lagrangian solver."""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from price_contour._price_contour import SolveResult, solve_online_py


class OnlineOptimiser:
    """Portfolio-level price optimisation via Lagrangian dual decomposition.

    Parameters
    ----------
    quote_id : str
        Column name for quote identifiers.
    scenario_step : str
        Column name for step indices.
    multiplier : str
        Column name for price multipliers.
    objective : str
        Column name for the objective function (e.g. expected income).
    constraints : dict[str, dict[str, float]]
        Constraint specifications. Keys are column names, values are dicts
        with one of: ``min`` (relative), ``max`` (relative),
        ``min_abs`` (absolute), ``max_abs`` (absolute).
    max_iter : int
        Maximum solver iterations.
    chunk_size : int
        Quotes per memory chunk for parallel processing.
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
        scenario_step: str = "scenario_step",
        multiplier: str = "multiplier",
        max_iter: int = 50,
        chunk_size: int = 500_000,
        tolerance: float = 1e-6,
        record_history: bool = False,
    ) -> None:
        self.quote_id = quote_id
        self.scenario_step = scenario_step
        self.multiplier = multiplier
        self.objective = objective
        self.constraints = constraints or {}
        self.max_iter = max_iter
        self.chunk_size = chunk_size
        self.tolerance = tolerance
        self.record_history = record_history

    def solve(
        self,
        df: pl.DataFrame,
        *,
        lambdas: dict[str, float] | None = None,
    ) -> SolveResult:
        """Run the optimisation.

        Parameters
        ----------
        df : pl.DataFrame
            Long-format scored DataFrame.
        lambdas : dict[str, float], optional
            Initial lambda values for warm-start.

        Returns
        -------
        SolveResult
            Result object with .lambdas, .converged, .iterations,
            .total_objective, .total_constraints, .dataframe properties.
        """
        return solve_online_py(
            df,
            quote_id=self.quote_id,
            scenario_step=self.scenario_step,
            multiplier=self.multiplier,
            objective=self.objective,
            constraints=self.constraints,
            max_iter=self.max_iter,
            chunk_size=self.chunk_size,
            tolerance=self.tolerance,
            lambdas=lambdas,
            record_history=self.record_history,
        )

    def config_dict(self) -> dict[str, Any]:
        """Return a serialisable dict of the solver configuration."""
        return {
            "objective": self.objective,
            "constraints": self.constraints,
            "quote_id": self.quote_id,
            "scenario_step": self.scenario_step,
            "multiplier": self.multiplier,
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
        opt_mults = out_df["optimal_multiplier"]

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
        mults = result.multipliers
        if mults:
            params["multiplier_min"] = round(float(min(mults)), 4)
            params["multiplier_max"] = round(float(max(mults)), 4)

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

        metrics["multiplier_mean"] = float(opt_mults.mean())
        metrics["multiplier_std"] = float(opt_mults.std())
        for pct in (5, 25, 50, 75, 95):
            metrics[f"multiplier_p{pct}"] = float(
                opt_mults.quantile(pct / 100)
            )

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
            "multiplier_distribution": {
                "mean": metrics["multiplier_mean"],
                "std": metrics["multiplier_std"],
                "p5": metrics["multiplier_p5"],
                "p25": metrics["multiplier_p25"],
                "p50": metrics["multiplier_p50"],
                "p75": metrics["multiplier_p75"],
                "p95": metrics["multiplier_p95"],
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
                    "all_constraints_satisfied": rec[
                        "all_constraints_satisfied"
                    ],
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
