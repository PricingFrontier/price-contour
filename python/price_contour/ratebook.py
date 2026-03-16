"""RatebookOptimiser — coordinate descent for ratebook factor optimisation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from price_contour._grid_utils import build_grid
from price_contour._price_contour import (
    FrontierResult,
    GroupedSolveResult,
    QuoteGrid,
    build_interaction_labels_py,
    compute_residuals_py,
    solve_grouped_py,
    update_multipliers_py,
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
            "constraints": dict(self.total_constraints.items())
            if isinstance(self.total_constraints, dict)
            else {},
            "factor_order": factor_order,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2))

        for factor_name, table in self.factor_tables.items():
            cols = factor_name.split(":")
            # Convert unit separator in keys to colon for JSON serialisation
            serialised_table = {k.replace("\x1f", ":"): v for k, v in table.items()}
            factor_data = {
                "columns": cols,
                "table": serialised_table,
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
                result[factor_name] = pl.DataFrame({cols[0]: levels, "factor": values})
            else:
                # Interaction factor: keys use unit separator (\x1f) between
                # values, falling back to colon for backward compatibility
                rows = []
                for key, val in table.items():
                    if "\x1f" in key:
                        parts = key.split("\x1f")
                    else:
                        parts = key.split(":")
                    row = {c: p for c, p in zip(cols, parts)}
                    row["factor"] = val
                    rows.append(row)
                result[factor_name] = pl.DataFrame(rows)
        return result

    @classmethod
    def load(cls, path: str | Path) -> "RatebookResult":
        """Load a RatebookResult from a saved parameters folder.

        Parameters
        ----------
        path : str | Path
            Directory containing config.json and per-factor JSON files.

        Returns
        -------
        RatebookResult
        """
        path = Path(path)
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = json.loads(config_path.read_text())

        factor_tables = {}
        factor_order = config.get("factor_order", config.get("factors", []))
        for factor_name in factor_order:
            filename = factor_name.replace(":", "_") + ".json"
            factor_path = path / filename
            if factor_path.exists():
                factor_data = json.loads(factor_path.read_text())
                factor_tables[factor_name] = factor_data.get("table", {})

        return cls(
            factor_tables=factor_tables,
            lambdas=config.get("lambdas", {}),
            total_objective=config.get("total_objective", 0.0),
            total_constraints=config.get("constraints", {}),
            baseline_objective=config.get("baseline_objective", 0.0),
            baseline_constraints=config.get("baseline_constraints", {}),
            converged=config.get("converged", False),
            cd_iterations=config.get("cd_iterations", 0),
            clamp_rate=config.get("clamp_rate", 0.0),
            per_factor_results=[],
        )


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
        quote_id: str = "quote_id",
        scenario_index: str = "scenario_index",
        scenario_value: str = "scenario_value",
        factor_columns: list[list[str]] | None = None,
        candidate_min: float = 0.70,
        candidate_max: float = 1.40,
        candidate_steps: int = 50,
        max_cd_iterations: int = 3,
        cd_tolerance: float = 1e-3,
        max_iter: int = 50,
        chunk_size: int = 500_000,
        tolerance: float = 1e-5,
    ) -> None:
        self.objective = objective
        self.constraints = constraints or {}
        self.quote_id = quote_id
        self.scenario_index = scenario_index
        self.scenario_value = scenario_value
        self.factor_columns = factor_columns
        self.candidate_min = candidate_min
        self.candidate_max = candidate_max
        self.candidate_steps = candidate_steps
        self.max_cd_iterations = max_cd_iterations
        self.cd_tolerance = cd_tolerance
        self.max_iter = max_iter
        self.chunk_size = chunk_size
        self.tolerance = tolerance

    def _build_candidates(self) -> list[float]:
        """Build the evenly-spaced candidate factor values."""
        return [
            self.candidate_min
            + (self.candidate_max - self.candidate_min)
            * i
            / max(self.candidate_steps - 1, 1)
            for i in range(self.candidate_steps)
        ]

    def solve(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        factors: pl.DataFrame,
        *,
        factor_columns: list[list[str]] | None = None,
        lambdas: dict[str, float] | None = None,
        _constraints_override: dict[str, dict[str, float]] | None = None,
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
        lambdas : dict[str, float], optional
            Initial lambda values for warm-start. Typically from a prior
            solve or adjacent frontier point.

        Returns
        -------
        RatebookResult
        """
        constraints = (
            _constraints_override
            if _constraints_override is not None
            else self.constraints
        )

        factor_specs = factor_columns or self.factor_columns
        if factor_specs is None:
            factor_specs = self._discover_structure(df_or_grid, factors)

        # Validate factors DataFrame
        if isinstance(df_or_grid, pl.DataFrame):
            n_steps = _count_steps(df_or_grid, self.quote_id)
            if n_steps > 0:
                expected_quotes = df_or_grid.shape[0] // n_steps
                if factors.shape[0] != expected_quotes:
                    raise ValueError(
                        f"factors row count {factors.shape[0]} != "
                        f"DataFrame quote count {expected_quotes} "
                        f"(rows={df_or_grid.shape[0]} / n_steps={n_steps})"
                    )

        # Validate factor_columns reference columns that exist in factors
        for spec in factor_specs:
            for col in spec:
                if col not in factors.columns:
                    raise ValueError(
                        f"Factor column '{col}' not found in factors DataFrame. "
                        f"Available: {list(factors.columns)}"
                    )

        # Build grid if needed
        if isinstance(df_or_grid, pl.DataFrame):
            grid = build_grid(
                df_or_grid,
                constraint_columns=list(constraints.keys()),
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
            )
        else:
            grid = df_or_grid

        n_quotes = grid.n_quotes

        # Build candidates
        candidates = self._build_candidates()

        # Build per-factor group labels from the factors DataFrame
        factor_group_labels: list[list[str]] = []
        for spec in factor_specs:
            if len(spec) == 1:
                labels = factors[spec[0]].cast(pl.Utf8).to_list()
            else:
                # Interaction: concatenate column values using unit separator
                # (ASCII 31) to avoid collisions with colons in data values
                label_cols = [factors[c].cast(pl.Utf8).to_list() for c in spec]
                labels = build_interaction_labels_py(label_cols, "\x1f")
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
        last_lambdas: dict[str, float] | None = lambdas

        for cd_iter in range(1, self.max_cd_iterations + 1):
            max_change = 0.0

            for f_idx, (spec, labels) in enumerate(
                zip(factor_specs, factor_group_labels)
            ):
                # Compute residuals: overall_mult / current_factor_value_for_this_quote
                old_table = factor_tables[f_idx]
                residuals = compute_residuals_py(overall_mult, labels, old_table)

                result = solve_grouped_py(
                    grid,
                    group_labels=labels,
                    residuals=residuals,
                    candidates=candidates,
                    constraints=constraints if constraints else None,
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

                # Update overall_mult via Rust helper
                overall_mult = update_multipliers_py(
                    overall_mult, labels, old_table, new_table
                )

            if max_change < self.cd_tolerance:
                cd_converged = True
                break

        # Get final metrics from last result
        last = per_factor_results[-1] if per_factor_results else None
        named_tables = {
            ":".join(spec): table for spec, table in zip(factor_specs, factor_tables)
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

        n_quotes = grid.n_quotes
        candidates = self._build_candidates()

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
            import warnings

            warnings.warn(
                "No factor column showed positive objective lift. "
                "Using all columns as factors."
            )
            selected = list(factors.columns)

        return [[col] for col in selected]

    def frontier(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        factors: pl.DataFrame,
        *,
        threshold_ranges: dict[str, tuple[float, float]],
        n_points_per_dim: int = 5,
        factor_columns: list[list[str]] | None = None,
        initial_lambdas: dict[str, float] | None = None,
        max_total_points: int = 10_000,
        parallel: bool = False,
    ) -> FrontierResult:
        """Sweep the efficient frontier by running coordinate descent at each threshold.

        Each frontier point is a full CD solve with modified constraint
        bounds. Results are warm-started from adjacent points using
        nearest-neighbour ordering.

        Parameters
        ----------
        df_or_grid : pl.DataFrame | QuoteGrid
            Scored DataFrame or pre-built QuoteGrid.
        factors : pl.DataFrame
            Per-quote factors DataFrame (same as ``solve``).
        threshold_ranges : dict[str, tuple[float, float]]
            Per-constraint (lo, hi) range. For relative constraints
            (min/max), these are fractions of baseline.
        n_points_per_dim : int
            Number of points per constraint dimension. Default 5
            (lower than online frontier because each point is a full CD).
        factor_columns : list[list[str]], optional
            Override factor_columns from init.
        initial_lambdas : dict[str, float], optional
            Lambdas to warm-start the first frontier point.
        parallel : bool
            Accepted for API consistency with ``OnlineOptimiser.frontier()``,
            but has no effect. Ratebook frontier is Python-orchestrated and
            always runs sequentially with warm-starting.

        Returns
        -------
        FrontierResult
            Result with ``.points`` (DataFrame) and ``.n_points``.
        """
        # Build grid once for all frontier points
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

        constraint_names = list(self.constraints.keys())
        if not constraint_names:
            raise ValueError("frontier requires at least one constraint")

        # Validate threshold_ranges keys match constraints
        for name in constraint_names:
            if name not in threshold_ranges:
                raise ValueError(
                    f"No threshold_range for constraint '{name}'. "
                    f"Available: {list(threshold_ranges.keys())}"
                )

        # Generate threshold grid
        dim_grids = []
        for name in constraint_names:
            lo, hi = threshold_ranges[name]
            n = n_points_per_dim
            if n <= 1:
                dim_grids.append([lo])
            else:
                dim_grids.append([lo + (hi - lo) * i / (n - 1) for i in range(n)])

        # Cartesian product
        combos: list[list[float]] = [[]]
        for dim in dim_grids:
            combos = [existing + [val] for existing in combos for val in dim]

        if not combos:
            raise ValueError("Empty threshold grid")

        if len(combos) > max_total_points:
            raise ValueError(
                f"Frontier would generate {len(combos)} points "
                f"(exceeds max_total_points={max_total_points}). "
                f"Reduce n_points_per_dim or increase max_total_points."
            )

        # Nearest-neighbour ordering for warm-start efficiency
        order = _nn_order(combos, [threshold_ranges[n] for n in constraint_names])

        # Sweep
        prev_lambdas = initial_lambdas
        points: list[tuple[int, dict[str, Any]]] = []

        for idx in order:
            thresholds = combos[idx]

            # Build modified constraints with this point's thresholds
            modified_constraints = {}
            for k, name in enumerate(constraint_names):
                spec = self.constraints[name]
                # Replace the threshold value, keeping direction
                if "min" in spec:
                    modified_constraints[name] = {"min": thresholds[k]}
                elif "max" in spec:
                    modified_constraints[name] = {"max": thresholds[k]}
                elif "min_abs" in spec:
                    modified_constraints[name] = {"min_abs": thresholds[k]}
                elif "max_abs" in spec:
                    modified_constraints[name] = {"max_abs": thresholds[k]}

            result = self.solve(
                grid,
                factors,
                factor_columns=factor_columns,
                lambdas=prev_lambdas,
                _constraints_override=modified_constraints,
            )

            prev_lambdas = result.lambdas

            points.append(
                (
                    idx,
                    {
                        "thresholds": thresholds,
                        "total_objective": result.total_objective,
                        "total_constraints": result.total_constraints,
                        "lambdas": result.lambdas,
                        "cd_iterations": result.cd_iterations,
                        "converged": result.converged,
                        "clamp_rate": result.clamp_rate,
                    },
                )
            )

        # Sort back to original (cartesian product) order
        points.sort(key=lambda x: x[0])

        # Build a Polars DataFrame matching the FrontierResult.points format
        columns: dict[str, list[Any]] = {}
        for k, name in enumerate(constraint_names):
            columns[f"threshold_{name}"] = [p[1]["thresholds"][k] for p in points]

        columns["total_objective"] = [p[1]["total_objective"] for p in points]

        for name in constraint_names:
            columns[f"total_{name}"] = [
                p[1]["total_constraints"].get(name, 0.0) for p in points
            ]

        for name in constraint_names:
            columns[f"lambda_{name}"] = [p[1]["lambdas"].get(name, 0.0) for p in points]

        columns["iterations"] = [p[1]["cd_iterations"] for p in points]
        columns["converged"] = [p[1]["converged"] for p in points]
        columns["clamp_rate"] = [p[1]["clamp_rate"] for p in points]

        return _RatebookFrontierResult(
            df=pl.DataFrame(columns),
            _constraint_names=constraint_names,
        )

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


def _nn_order(
    points: list[list[float]], ranges: list[tuple[float, float]]
) -> list[int]:
    """Greedy nearest-neighbour ordering through normalised threshold space."""
    n = len(points)
    if n == 0:
        return []

    # Normalise to [0, 1]
    normalised = []
    for p in points:
        norm = []
        for val, (lo, hi) in zip(p, ranges):
            span = hi - lo
            norm.append((val - lo) / span if abs(span) > 1e-15 else 0.5)
        normalised.append(norm)

    # Start from point nearest origin
    def sq_dist_origin(idx: int) -> float:
        return sum(v * v for v in normalised[idx])

    current = min(range(n), key=sq_dist_origin)
    visited = [False] * n
    order: list[int] = []

    for _ in range(n):
        visited[current] = True
        order.append(current)

        best_dist = float("inf")
        best_next = 0
        for j in range(n):
            if visited[j]:
                continue
            dist = sum((a - b) ** 2 for a, b in zip(normalised[current], normalised[j]))
            if dist < best_dist:
                best_dist = dist
                best_next = j
        current = best_next

    return order


class _RatebookFrontierResult:
    """Lightweight frontier result for ratebook mode.

    Mirrors the interface of ``FrontierResult`` (from the Rust frontier)
    so Haute can handle both modes uniformly.
    """

    def __init__(self, df: pl.DataFrame, _constraint_names: list[str]) -> None:
        self._df = df
        self._constraint_names = _constraint_names

    @property
    def points(self) -> pl.DataFrame:
        return self._df

    @property
    def n_points(self) -> int:
        return self._df.shape[0]

    @property
    def constraint_names(self) -> list[str]:
        return self._constraint_names


def _count_steps(df: pl.DataFrame, quote_id_col: str) -> int:
    """Count the number of steps for the first quote in a sorted DataFrame."""
    if df.shape[0] == 0 or quote_id_col not in df.columns:
        return 0
    first_qid = df[quote_id_col][0]
    return int((df[quote_id_col] == first_qid).sum())
