"""RatebookOptimiser — coordinate descent for ratebook factor optimisation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from price_contour._price_contour import (
    GroupedSolveResult,
    QuoteGrid,
    QuoteGridBuilder,
    solve_grouped_py,
)


@dataclass
class RatebookResult:
    """Result of ratebook coordinate descent optimisation."""

    factor_tables: dict[str, dict[str, float]]
    lambdas: dict[str, float]
    total_objective: float
    total_constraints: dict[str, float]
    baseline_objective: float
    baseline_constraints: dict[str, float]
    cd_iterations: int
    converged: bool
    clamp_rate: float
    per_factor_results: list[GroupedSolveResult] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Save factor tables to a parameters folder.

        Creates one JSON per factor plus a config.json:

            path/
              config.json
              region.json
              age_band.json
              ...

        Parameters
        ----------
        path : str | Path
            Directory to write the parameters folder into.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        factor_order = list(self.factor_tables.keys())

        config = {
            "lambdas": self.lambdas,
            "constraints": {
                name: val
                for name, val in zip(
                    self.total_constraints.keys(), self.total_constraints.values()
                )
            } if isinstance(self.total_constraints, dict) else {},
            "factors": factor_order,
            "factor_order": factor_order,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

        for factor_name, table in self.factor_tables.items():
            cols = factor_name.split(":")
            factor_data = {
                "columns": cols,
                "table": table,
            }
            filename = factor_name.replace(":", "_") + ".json"
            (path / filename).write_text(json.dumps(factor_data, indent=2))

    def to_rating_entries(self) -> dict[str, pl.DataFrame]:
        """Convert factor tables to rating-step DataFrames.

        Returns a dict mapping factor name to a DataFrame with columns:
        [level_col_1, ..., level_col_n, factor].
        """
        result = {}
        for factor_name, table in self.factor_tables.items():
            cols = factor_name.split(":")
            if len(cols) == 1:
                # Single column factor
                levels = list(table.keys())
                values = [table[k] for k in levels]
                result[factor_name] = pl.DataFrame(
                    {cols[0]: levels, "factor": values}
                )
            else:
                # Interaction factor: keys are "val1:val2:..."
                rows = []
                for key, val in table.items():
                    parts = key.split(":")
                    row = {c: p for c, p in zip(cols, parts)}
                    row["factor"] = val
                    rows.append(row)
                result[factor_name] = pl.DataFrame(rows)
        return result


class RatebookOptimiser:
    """Ratebook factor optimisation via coordinate descent.

    Each CD iteration loops over factor specs, calling `solve_grouped`
    to find the best per-group factor value, then updates the residual
    multiplier for the next factor.

    Parameters
    ----------
    objective : str
        Objective column name.
    constraints : dict
        Constraint specifications (same format as OnlineOptimiser).
    factor_columns : list[list[str]]
        List of factor specs. Each spec is a list of column names whose
        interaction defines a rating factor. If None, must be provided
        at solve time or auto-discovered.
    candidate_min, candidate_max : float
        Range for candidate factor values.
    candidate_steps : int
        Number of candidate factor values.
    max_cd_iterations : int
        Maximum coordinate descent iterations.
    cd_tolerance : float
        CD convergence tolerance (max factor value change).
    max_iter : int
        Maximum Lagrangian iterations per inner solve.
    chunk_size : int
        Quotes per chunk for inner solver.
    tolerance : float
        Lagrangian convergence tolerance.
    """

    def __init__(
        self,
        objective: str = "expected_income",
        constraints: dict[str, dict[str, float]] | None = None,
        *,
        factor_columns: list[list[str]] | None = None,
        candidate_min: float = 0.70,
        candidate_max: float = 1.40,
        candidate_steps: int = 50,
        max_cd_iterations: int = 3,
        cd_tolerance: float = 1e-3,
        max_iter: int = 50,
        chunk_size: int = 500_000,
        tolerance: float = 1e-6,
    ) -> None:
        self.objective = objective
        self.constraints = constraints or {}
        self.factor_columns = factor_columns
        self.candidate_min = candidate_min
        self.candidate_max = candidate_max
        self.candidate_steps = candidate_steps
        self.max_cd_iterations = max_cd_iterations
        self.cd_tolerance = cd_tolerance
        self.max_iter = max_iter
        self.chunk_size = chunk_size
        self.tolerance = tolerance

    def solve(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        factors: pl.DataFrame,
        *,
        factor_columns: list[list[str]] | None = None,
    ) -> RatebookResult:
        """Run ratebook optimisation.

        Parameters
        ----------
        df_or_grid : pl.DataFrame | QuoteGrid
            Scored DataFrame or pre-built QuoteGrid.
        factors : pl.DataFrame
            Per-quote factors DataFrame with N rows (one per quote).
            Must contain the columns referenced by factor_columns.
        factor_columns : list[list[str]], optional
            Override factor_columns from init.

        Returns
        -------
        RatebookResult
        """
        factor_specs = factor_columns or self.factor_columns
        if factor_specs is None:
            factor_specs = self._discover_structure(df_or_grid, factors)

        # Build grid if needed
        if isinstance(df_or_grid, pl.DataFrame):
            builder = QuoteGridBuilder(
                list(self.constraints.keys()),
                objective=self.objective,
            )
            builder.append(df_or_grid)
            grid = builder.build()
        else:
            grid = df_or_grid

        n_quotes = grid.n_quotes

        # Build candidates
        candidates = [
            self.candidate_min
            + (self.candidate_max - self.candidate_min) * i / max(self.candidate_steps - 1, 1)
            for i in range(self.candidate_steps)
        ]

        # Build per-factor group labels from the factors DataFrame
        factor_group_labels: list[list[str]] = []
        for spec in factor_specs:
            if len(spec) == 1:
                labels = factors[spec[0]].cast(pl.Utf8).to_list()
            else:
                # Interaction: concatenate column values
                label_cols = [factors[c].cast(pl.Utf8) for c in spec]
                labels = [
                    ":".join(str(col[i]) for col in label_cols)
                    for i in range(n_quotes)
                ]
            factor_group_labels.append(labels)

        # Initialise factor tables: each group level → 1.0
        factor_tables: list[dict[str, float]] = []
        for labels in factor_group_labels:
            unique_labels = sorted(set(labels))
            factor_tables.append({label: 1.0 for label in unique_labels})

        # Overall multiplier per quote = product of all factor values
        overall_mult = [1.0] * n_quotes

        per_factor_results: list[GroupedSolveResult] = []
        cd_converged = False
        cd_iter = 0
        last_lambdas: dict[str, float] | None = None

        for cd_iter in range(1, self.max_cd_iterations + 1):
            max_change = 0.0

            for f_idx, (spec, labels) in enumerate(
                zip(factor_specs, factor_group_labels)
            ):
                factor_name = ":".join(spec)

                # Compute residuals: overall_mult / current_factor_value_for_this_quote
                old_table = factor_tables[f_idx]
                residuals = []
                for i in range(n_quotes):
                    fv = old_table[labels[i]]
                    residuals.append(overall_mult[i] / fv if fv != 0.0 else 1.0)

                result = solve_grouped_py(
                    grid,
                    group_labels=labels,
                    residuals=residuals,
                    candidates=candidates,
                    constraints=self.constraints if self.constraints else None,
                    max_iter=self.max_iter,
                    chunk_size=self.chunk_size,
                    tolerance=self.tolerance,
                    lambdas=last_lambdas,
                )

                per_factor_results.append(result)
                last_lambdas = result.lambdas

                # Update factor table
                new_table = dict(result.optimal_factor_values)
                for label in old_table:
                    if label not in new_table:
                        new_table[label] = old_table[label]

                # Track max change
                for label in old_table:
                    change = abs(new_table.get(label, 1.0) - old_table[label])
                    max_change = max(max_change, change)

                factor_tables[f_idx] = new_table

                # Update overall_mult
                for i in range(n_quotes):
                    fv_old = old_table[labels[i]]
                    fv_new = new_table[labels[i]]
                    if fv_old != 0.0:
                        overall_mult[i] = overall_mult[i] / fv_old * fv_new
                    else:
                        overall_mult[i] = fv_new

            if max_change < self.cd_tolerance:
                cd_converged = True
                break

        # Get final metrics from last result
        last = per_factor_results[-1] if per_factor_results else None
        named_tables = {
            ":".join(spec): table
            for spec, table in zip(factor_specs, factor_tables)
        }

        # Compute final clamp rate as average across per-factor results
        avg_clamp = (
            sum(r.clamp_rate for r in per_factor_results) / len(per_factor_results)
            if per_factor_results
            else 0.0
        )

        return RatebookResult(
            factor_tables=named_tables,
            lambdas=last.lambdas if last else {},
            total_objective=last.total_objective if last else 0.0,
            total_constraints=last.total_constraints if last else {},
            baseline_objective=last.baseline_objective if last else 0.0,
            baseline_constraints=last.baseline_constraints if last else {},
            cd_iterations=cd_iter,
            converged=cd_converged,
            clamp_rate=avg_clamp,
            per_factor_results=per_factor_results,
        )

    def _discover_structure(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        factors: pl.DataFrame,
    ) -> list[list[str]]:
        """Auto-discover factor structure by screening main effects.

        Screens each column in the factors DataFrame by running a quick
        grouped solve and ranking by objective lift.
        """
        # Build grid if needed
        if isinstance(df_or_grid, pl.DataFrame):
            builder = QuoteGridBuilder(
                list(self.constraints.keys()),
                objective=self.objective,
            )
            builder.append(df_or_grid)
            grid = builder.build()
        else:
            grid = df_or_grid

        n_quotes = grid.n_quotes
        candidates = [
            self.candidate_min
            + (self.candidate_max - self.candidate_min) * i / max(self.candidate_steps - 1, 1)
            for i in range(self.candidate_steps)
        ]

        lifts: list[tuple[str, float]] = []
        for col in factors.columns:
            labels = factors[col].cast(pl.Utf8).to_list()
            residuals = [1.0] * n_quotes

            result = solve_grouped_py(
                grid,
                group_labels=labels,
                residuals=residuals,
                candidates=candidates,
                constraints=self.constraints if self.constraints else None,
                max_iter=10,  # quick screen
            )

            baseline = result.baseline_objective
            lift = (
                (result.total_objective - baseline) / abs(baseline)
                if baseline != 0
                else 0.0
            )
            lifts.append((col, lift))

        # Select factors with positive lift, sorted descending
        lifts.sort(key=lambda x: x[1], reverse=True)
        selected = [col for col, lift in lifts if lift > 0.0]

        if not selected:
            # Fallback: use all columns
            selected = list(factors.columns)

        return [[col] for col in selected]

    def summary(self, result: RatebookResult) -> dict[str, Any]:
        """Package a ratebook result into MLflow-ready dicts."""
        params: dict[str, Any] = {
            "objective": self.objective,
            "candidate_min": self.candidate_min,
            "candidate_max": self.candidate_max,
            "candidate_steps": self.candidate_steps,
            "max_cd_iterations": self.max_cd_iterations,
            "cd_tolerance": self.cd_tolerance,
            "max_iter": self.max_iter,
            "n_factors": len(result.factor_tables),
        }
        if self.constraints:
            import json
            params["constraints"] = json.dumps(self.constraints)

        metrics: dict[str, float] = {
            "total_objective": result.total_objective,
            "baseline_objective": result.baseline_objective,
            "cd_iterations": float(result.cd_iterations),
            "converged": float(result.converged),
            "clamp_rate": float(result.clamp_rate),
        }
        if result.baseline_objective != 0:
            metrics["uplift_pct"] = (
                (result.total_objective - result.baseline_objective)
                / abs(result.baseline_objective)
            ) * 100

        for name, val in result.total_constraints.items():
            metrics[f"constraint_{name}_total"] = val
        for name, val in result.baseline_constraints.items():
            metrics[f"constraint_{name}_baseline"] = val
        for name, val in result.lambdas.items():
            metrics[f"lambda_{name}"] = val

        artifacts: dict[str, Any] = {
            "factor_tables": result.factor_tables,
            "rating_entries": {
                name: df for name, df in result.to_rating_entries().items()
            },
        }

        return {
            "params": params,
            "metrics": metrics,
            "artifacts": artifacts,
        }
