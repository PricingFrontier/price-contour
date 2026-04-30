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
    compute_residuals_py,
    extract_factor_labels_py,
    solve_grouped_py,
    update_multipliers_py,
)
from price_contour._frontier_helpers import (
    _cartesian_product,
    _linspace,
)
from price_contour._ratio_results import (
    _safe_ratio_from_columns,
    _stitch_optimal_ratio_columns,
)
from price_contour.solver import (
    _is_ratio_spec,
    _linearise_ratio_constraints,
    _override_thresholds,
    _ratio_constraint_names,
    _reject_none_for_solve,
    _reject_ratio_for_grid,
    _spec_numeric_threshold,
    _spec_threshold_is_none,
    _validate_constraint_dict,
    _validate_dataframe,
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
            "baseline_constraints": dict(self.baseline_constraints.items())
            if isinstance(self.baseline_constraints, dict)
            else {},
            "total_objective": self.total_objective,
            "baseline_objective": self.baseline_objective,
            "cd_iterations": self.cd_iterations,
            "converged": self.converged,
            "clamp_rate": self.clamp_rate,
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

        def _required(key: str) -> Any:
            if key not in config:
                raise ValueError(
                    f"missing required field '{key}' in saved RatebookResult"
                )
            return config[key]

        factor_tables: dict[str, dict[str, float]] = {}
        # ``factor_order`` is required (the on-disk shape always writes it
        # via :meth:`save`); the ``"factors"`` fallback was a transitional
        # alias and is no longer supported.
        factor_order = _required("factor_order")
        for factor_name in factor_order:
            filename = factor_name.replace(":", "_") + ".json"
            factor_path = path / filename
            if factor_path.exists():
                factor_data = json.loads(factor_path.read_text())
                factor_tables[factor_name] = factor_data.get("table", {})

        return cls(
            factor_tables=factor_tables,
            lambdas=_required("lambdas"),
            total_objective=_required("total_objective"),
            total_constraints=_required("constraints"),
            baseline_objective=_required("baseline_objective"),
            baseline_constraints=_required("baseline_constraints"),
            converged=_required("converged"),
            cd_iterations=_required("cd_iterations"),
            clamp_rate=_required("clamp_rate"),
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
        tolerance: float = 1e-5,
    ) -> None:
        self.objective = objective
        self.constraints = {} if constraints is None else constraints
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
        self.tolerance = tolerance
        _validate_constraint_dict(self.constraints)

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

        # ``None`` thresholds are frontier-only markers (B1). Reject before
        # any work so the user sees a clear, named-constraint message that
        # mentions ``frontier()``. We check ``constraints`` (not
        # ``self.constraints``) so the frontier path that passes a fully
        # numeric ``_constraints_override`` is not blocked by a None left
        # on the parent optimiser.
        _reject_none_for_solve(constraints)

        # Detect ratio constraints. The ratebook supports ratio constraints
        # by linearising each ratio per (quote x scenario) into a synthetic
        # sum constraint (same recipe as the online solver). The
        # linearisation requires the raw numerator / denominator columns
        # at solve time, so a ratio constraint paired with a pre-built
        # QuoteGrid is a setup-time error (mirrors the online solver's
        # rejection wording via the shared helper).
        ratio_names = _ratio_constraint_names(constraints)
        if ratio_names and not isinstance(df_or_grid, pl.DataFrame):
            _reject_ratio_for_grid(constraints, mode="solve")

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

        # When ratio constraints are present, validate the DataFrame
        # schema (existence + non-null + non-NaN of numerator /
        # denominator columns) BEFORE linearisation so missing columns
        # surface a precise schema error rather than failing inside the
        # Polars expression. Mirrors OnlineOptimiser.solve()'s ordering.
        if ratio_names and isinstance(df_or_grid, pl.DataFrame):
            _validate_dataframe(
                df_or_grid,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
                constraint_cols=list(constraints.keys()),
                constraints=constraints,
            )

        # Linearise ratio specs. ``original_df`` is preserved so the
        # post-CD ratio reporting can recover ``Sigma_baseline num`` /
        # ``Sigma_baseline denom`` and stitch ``optimal_<num>`` /
        # ``optimal_<denom>`` columns onto the last grouped result.
        # Sum-only constraint dicts skip the linearisation pass entirely
        # so the existing fast path is preserved bit-for-bit.
        original_df = df_or_grid if isinstance(df_or_grid, pl.DataFrame) else None
        ratio_columns: list[tuple[str, str, str]] = []
        if ratio_names and original_df is not None:
            (
                modified_df,
                sum_constraints,
                _grid_cols,
                ratio_columns,
                _threshold_shift,
            ) = _linearise_ratio_constraints(
                original_df,
                constraints,
                scenario_value_col=self.scenario_value,
                quote_id_col=self.quote_id,
            )
            grid_input: pl.DataFrame | QuoteGrid = modified_df
            cd_constraints: dict[str, dict[str, float]] = sum_constraints
        else:
            grid_input = df_or_grid
            cd_constraints = constraints

        # Build grid if needed. ``cd_constraints`` carries only sum-shape
        # specs at this point (any ratio specs have been rewritten into
        # synthetic sum specs whose key is the linearised column name).
        if isinstance(grid_input, pl.DataFrame):
            grid = build_grid(
                grid_input,
                constraint_columns=list(cd_constraints.keys()),
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
            )
        else:
            grid = grid_input

        n_quotes = grid.n_quotes

        # Build candidates
        candidates = self._build_candidates()

        # Build per-factor group labels from the factors DataFrame.
        #
        # `extract_factor_labels_py` casts each spec's columns to Utf8 inside
        # Rust and emits the per-quote labels directly — skipping the
        # `factors[col].to_list()` round-trip that previously materialised
        # gigabytes of transient Python `list[str]` objects on large
        # portfolios. ASCII 31 (unit separator) is the canonical interaction
        # join character; it avoids collisions with colons that legitimately
        # appear in factor values.
        factor_group_labels: list[list[str]] = extract_factor_labels_py(
            factors, factor_specs, "\x1f"
        )

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
                    constraints=cd_constraints if cd_constraints else None,
                    max_iter=self.max_iter,
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

        # C5 (carries C3 reporting through to ratebook): each ratio
        # label's ``total_constraints`` / ``baseline_constraints`` entry
        # reports the **actual** ratio at the optimum / baseline rather
        # than the linearised total. We recompute these from the original
        # DataFrame at the last grouped result's optimal steps; sum
        # entries pass through unchanged. ``RatebookResult`` is a
        # Python-side dataclass so we populate the dicts directly here
        # instead of wrapping a Rust object (no ``_RatioSolveResultWrapper``
        # equivalent needed — it would only delegate ``__getattr__`` to
        # fields the dataclass already owns).
        total_constraints = dict(last.total_constraints) if last else {}
        baseline_constraints = dict(last.baseline_constraints) if last else {}
        if ratio_columns and original_df is not None and last is not None:
            # Actual ratio at optimum: stitch ``optimal_<num>`` /
            # ``optimal_<denom>`` columns onto the last grouped result's
            # dataframe via a join on ``(quote_id, optimal_step)``, then
            # sum and divide.
            optimum_df = _stitch_optimal_ratio_columns(
                base_df=last.dataframe,
                original_df=original_df,
                ratio_columns=ratio_columns,
                quote_id_col=self.quote_id,
                scenario_index_col=self.scenario_index,
            )
            for label, num_col, denom_col in ratio_columns:
                total_constraints[label] = _safe_ratio_from_columns(
                    optimum_df, f"optimal_{num_col}", f"optimal_{denom_col}"
                )
            # Actual baseline ratio: ``Sigma_baseline num / Sigma_baseline
            # denom`` from rows where ``scenario_value == 1.0``.
            baseline_slice = original_df.filter(
                pl.col(self.scenario_value) == 1.0
            )
            for label, num_col, denom_col in ratio_columns:
                baseline_constraints[label] = _safe_ratio_from_columns(
                    baseline_slice, num_col, denom_col
                )

        return RatebookResult(
            factor_tables=named_tables,
            lambdas=last.lambdas if last else {},
            total_objective=last.total_objective if last else 0.0,
            total_constraints=total_constraints,
            baseline_objective=last.baseline_objective if last else 0.0,
            baseline_constraints=baseline_constraints,
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

        # Extract every column's labels in one Rust-side pass — same memory
        # win as the main solve loop. Column order in `all_labels` matches
        # `factors.columns` order.
        all_labels = extract_factor_labels_py(
            factors, [[col] for col in factors.columns], "\x1f"
        )

        lifts: list[tuple[str, float]] = []
        for col, labels in zip(factors.columns, all_labels):
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
            screened = [col for col, _ in lifts]
            raise ValueError(
                f"No factor column showed positive objective lift. "
                f"Screened columns: {screened}. Supply "
                f"`factor_columns` explicitly or revisit the factors "
                f"DataFrame — auto-discovery cannot pick a default."
            )

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
            Per-constraint (lo, hi) range. Units follow the constraint key:
            absolute for ``min`` / ``max``; fractions of baseline for
            ``min_pct`` / ``max_pct``.
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
        constraint_names = list(self.constraints.keys())
        if not constraint_names:
            raise ValueError("frontier requires at least one constraint")

        # D1 contract: a ``None`` threshold MUST have a
        # ``threshold_ranges`` entry (B1 marker rule preserved); a
        # numeric threshold may omit its range (held fixed at the
        # constructor value across every frontier point). Validate
        # None-without-range first so the error message matches the B1
        # wording. Fires BEFORE any grid build / linearisation so the
        # user sees the failure mode immediately on dispatch.
        for name in constraint_names:
            if name in threshold_ranges:
                continue
            if _spec_threshold_is_none(self.constraints[name]):
                raise ValueError(
                    f"No threshold_range for constraint '{name}'. "
                    f"Available: {list(threshold_ranges.keys())}"
                )

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

        # Detect ratio constraints up front. When any are present we
        # CANNOT pre-build the grid: each point's linearisation needs a
        # per-point threshold ``L`` and materialises a new synthetic
        # column. Pass the raw DataFrame through to ``solve()`` per
        # point and let the linearisation run there. Sum-only constraint
        # dicts keep the existing pre-built-grid fast path.
        ratio_names = _ratio_constraint_names(self.constraints)
        if ratio_names and not isinstance(df_or_grid, pl.DataFrame):
            _reject_ratio_for_grid(self.constraints, mode="frontier")

        if ratio_names:
            # Validate the input DataFrame schema once. Per-point
            # linearisation re-runs cheap pre-flight checks but a
            # missing numerator / denominator column should surface here
            # before any sweep work.
            assert isinstance(df_or_grid, pl.DataFrame)
            _validate_dataframe(
                df_or_grid,
                quote_id=self.quote_id,
                scenario_index=self.scenario_index,
                scenario_value=self.scenario_value,
                objective=self.objective,
                constraint_cols=list(self.constraints.keys()),
                constraints=self.constraints,
            )
            grid: pl.DataFrame | QuoteGrid = df_or_grid
        elif isinstance(df_or_grid, pl.DataFrame):
            # The grid build needs every sum-constraint column
            # (including unswept axes — they're enforced at the
            # constructor value at every point). Ratio "constraints"
            # use display labels rather than columns and are excluded.
            sum_constraint_cols = [
                c
                for c in constraint_names
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
            grid = df_or_grid

        # Generate threshold grid for the swept axes only via the shared
        # ``_linspace`` / ``_cartesian_product`` helpers; unswept axes
        # contribute one fixed value (the constructor threshold) to
        # every output row but do not multiply the combo count.
        dim_grids = [
            _linspace(float(threshold_ranges[name][0]), float(threshold_ranges[name][1]), n_points_per_dim)
            for name in swept_names
        ]
        combos = _cartesian_product(dim_grids)

        if not combos:
            raise ValueError("Empty threshold grid")

        if len(combos) > max_total_points:
            raise ValueError(
                f"Frontier would generate {len(combos)} points "
                f"(exceeds max_total_points={max_total_points}). "
                f"Reduce n_points_per_dim or increase max_total_points."
            )

        # Nearest-neighbour ordering for warm-start efficiency. The
        # ranges fed to ``_nn_order`` mirror the swept axes only — the
        # unswept axes are constant and would contribute zero distance,
        # so omitting them is equivalent to including them.
        order = _nn_order(combos, [threshold_ranges[n] for n in swept_names])

        # Sweep
        prev_lambdas = initial_lambdas
        points: list[tuple[int, dict[str, Any]]] = []

        for idx in order:
            thresholds = combos[idx]

            # Build per-point constraint dict by overriding only the
            # swept axes' threshold values. Unswept axes copy their
            # constructor spec verbatim so the inner solve enforces
            # them at the constructor value at every point. Reuses the
            # C4 helper so ratio specs preserve ``numerator`` /
            # ``denominator`` and the direction key (``max_pct`` stays
            # ``max_pct`` so the C2 linearisation scales by baseline_LR
            # internally per point).
            modified_constraints = _override_thresholds(
                self.constraints, list(thresholds), swept_names
            )

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

        # Build a Polars DataFrame matching the FrontierResult.points
        # format. Threshold columns: swept axes echo per-point combo
        # values; unswept axes echo the constructor threshold verbatim
        # at every row.
        columns: dict[str, list[Any]] = {}
        swept_index = {name: idx for idx, name in enumerate(swept_names)}
        for name in constraint_names:
            if name in swept_index:
                k = swept_index[name]
                columns[f"threshold_{name}"] = [
                    p[1]["thresholds"][k] for p in points
                ]
            else:
                fixed_val = unswept_thresholds[name]
                columns[f"threshold_{name}"] = [fixed_val for _ in points]

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
