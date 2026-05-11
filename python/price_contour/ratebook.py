"""RatebookOptimiser — coordinate descent for ratebook factor optimisation."""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from price_contour._frontier_helpers import (
    _cartesian_product,
    _linspace,
)
from price_contour._grid_utils import build_grid
from price_contour._price_contour import (
    FactorContext,
    FrontierResult,
    QuoteGrid,
    RatebookFactorContexts,
    build_ratebook_factor_contexts_from_parquet_chunked_py,
    run_cd_pass_py,
    solve_grouped_py,
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


def build_ratebook_factor_contexts_from_parquet_chunked(
    path: str,
    factor_specs: list[list[str]],
    chunk_size: int,
    *,
    quote_id: str | None = "quote_id",
    separator: str = "\x1f",
    expected_quote_ids: list[str] | None = None,
    expected_n_quotes: int | None = None,
) -> RatebookFactorContexts:
    """Build factor contexts from a Parquet file by streaming row slices.

    Re-exported as ``price_contour.build_ratebook_factor_contexts_from_parquet_chunked``
    by ``__init__.py``. The underlying Rust function has a ``_py``
    suffix (PyO3 convention); the public name drops it.

    Reads only ``quote_id`` plus the columns referenced by
    ``factor_specs`` — the rest of the parquet schema is never decoded.
    Memory for the IO buffer scales with ``chunk_size``, not the full
    file size.

    Parameters
    ----------
    path
        Path to the factor parquet file. Each row is one quote.
    factor_specs
        List of factor specs; each spec is a list of column names whose
        interaction defines a rating factor.
    chunk_size
        Rows per IO slice. Must be > 0.
    quote_id
        Column name whose values are the alignment fingerprint source.
        Pass ``None`` when the parquet is already positionally aligned
        to your quote grid; combine with ``expected_quote_ids`` to give
        the contexts a verifiable fingerprint.
    separator
        Interaction separator for multi-column factor specs. ASCII 31
        (unit separator) by default.
    expected_quote_ids
        When supplied, the builder reorders the contexts to this exact
        quote order and stores ``hash(expected_quote_ids)`` as the
        fingerprint. Use ``quote_grid.quote_ids`` here to align with a
        previously built grid.
    expected_n_quotes
        Cross-check: the total number of rows read must equal this.

    Returns
    -------
    RatebookFactorContexts
        Opaque handle suitable for ``RatebookOptimiser.solve()`` /
        ``RatebookOptimiser.frontier()``.
    """
    return build_ratebook_factor_contexts_from_parquet_chunked_py(
        path,
        factor_specs,
        chunk_size,
        quote_id=quote_id,
        separator=separator,
        expected_quote_ids=expected_quote_ids,
        expected_n_quotes=expected_n_quotes,
    )


class PerFactorRecord(Protocol):
    """Lightweight per-(cd_iter × factor) record exposed on
    :attr:`RatebookResult.per_factor_results`. Carries only the fields
    consumers need (CD-monotonicity tests, debugging) — the full
    `GroupedSolveResult` shape is not preserved because materialising one
    PyClass per inner solve would defeat the FFI saving in
    ``run_cd_pass_py``.
    """

    total_objective: float
    lambdas: dict[str, float]


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
    per_factor_results: list[PerFactorRecord] = field(default_factory=list)

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
                # Interaction factor: keys use the unit separator (\x1f)
                # in-memory but the colon when round-tripped through
                # save/load (which substitutes \x1f → ':' for JSON
                # readability). Accept either at this seam.
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
        factors: pl.DataFrame | RatebookFactorContexts,
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
        factors : pl.DataFrame | RatebookFactorContexts
            Either a per-quote factors DataFrame (one row per quote,
            columns referenced by ``factor_columns``) or a prebuilt
            :class:`RatebookFactorContexts` opaque handle. The contexts
            handle is what production pipelines should use: it
            short-circuits per-call label extraction and validates
            quote-axis alignment against the grid in O(1) via a 64-bit
            fingerprint.
        factor_columns : list[list[str]], optional
            Override factor_columns from init. Rejected when ``factors``
            is a :class:`RatebookFactorContexts` whose own
            ``factor_specs`` disagrees — the contexts' specs are the
            source of truth in that mode.
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

        # Resolve `factor_specs` based on the `factors` argument shape.
        # When `factors` is a `RatebookFactorContexts`, its own specs
        # are authoritative and auto-discovery is unavailable
        # (contexts already encode the chosen factors). When `factors`
        # is a DataFrame, fall through to the legacy resolution path.
        if isinstance(factors, RatebookFactorContexts):
            factor_specs = _resolve_factor_specs_from_contexts(
                factors, factor_columns, self.factor_columns
            )
        else:
            factor_specs = factor_columns or self.factor_columns
            if factor_specs is None:
                factor_specs = self._discover_structure(df_or_grid, factors)

            # Pre-grid-build validation: rows align by count, columns
            # referenced by specs exist. The fingerprint check below
            # gives us the stricter alignment proof, but these checks
            # surface human-friendly errors before any grid-build cost.
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

        # Build candidates
        candidates = self._build_candidates()

        # Resolve factors to a `RatebookFactorContexts` whose fingerprint
        # matches the grid's quote axis. The wrapper owns one
        # `Arc<GroupMapping>` per factor; we hand the underlying
        # `list[FactorContext]` to the solver. The fingerprint check is
        # O(1) — a single u64 compare on every solve, including the
        # ones a frontier sweep makes — so we can safely repeat it
        # at every entry instead of relying on a private override.
        factor_contexts_obj = _resolve_factor_contexts(
            factors,
            factor_specs,
            grid=grid,
            quote_id_col=self.quote_id,
        )
        factor_contexts: list[FactorContext] = (
            factor_contexts_obj._factor_contexts_for_solver()
        )

        # Per-factor group label lists for stitching the final `dict[str,
        # float]` factor tables back together. These are the unique
        # labels per factor in group-index order, owned by the context.
        factor_group_labels: list[list[str]] = [
            ctx.group_labels for ctx in factor_contexts
        ]

        # Run the entire CD pass in Rust. Replaces the previous
        # Python-side `for cd_iter: for f_idx: compute_residuals_py +
        # solve_grouped_py + update_multipliers_py + bookkeeping` loop;
        # `overall_mult`, residuals, and per-factor `factor_values` all
        # stay Rust-side, eliminating ~`max_cd × n_factors × 2` PyO3
        # round-trips of 100k-element f32 buffers per `solve()` call.
        cd_result = run_cd_pass_py(
            grid,
            factor_contexts,
            candidates,
            constraints=cd_constraints if cd_constraints else None,
            max_iter=self.max_iter,
            tolerance=self.tolerance,
            max_cd_iterations=self.max_cd_iterations,
            cd_tolerance=self.cd_tolerance,
            lambdas=lambdas,
        )

        # Lightweight per-call records for the CD-monotonicity tests
        # that consume `RatebookResult.per_factor_results`. Each element
        # exposes ``.total_objective`` and ``.lambdas`` — the only
        # attributes those tests read. Building real `GroupedSolveResult`
        # objects per call would defeat the FFI saving the Rust CD
        # pass landed.
        per_call_objectives = cd_result.per_call_total_objectives
        per_call_lambdas = cd_result.per_call_lambdas
        per_factor_results: list[PerFactorRecord] = [
            types.SimpleNamespace(total_objective=obj, lambdas=lams)
            for obj, lams in zip(per_call_objectives, per_call_lambdas)
        ]
        cd_converged = cd_result.converged
        cd_iter = cd_result.cd_iterations
        factor_values_rust = cd_result.factor_values

        # Convert per-group `Vec<f32>` factor values back to the public
        # `dict[str, float]` shape that `RatebookResult.factor_tables`
        # carries (and that callers persist via `save()`).
        factor_tables: list[dict[str, float]] = [
            {label: float(value) for label, value in zip(labels, values)}
            for labels, values in zip(factor_group_labels, factor_values_rust)
        ]
        named_tables = {
            ":".join(spec): table for spec, table in zip(factor_specs, factor_tables)
        }

        avg_clamp = cd_result.clamp_rate

        # C5 (carries C3 reporting through to ratebook): each ratio
        # label's ``total_constraints`` / ``baseline_constraints`` entry
        # reports the **actual** ratio at the optimum / baseline rather
        # than the linearised total. We recompute these from the original
        # DataFrame at the last grouped result's optimal steps; sum
        # entries pass through unchanged. ``cd_result.dataframe`` is the
        # last grouped solve's per-quote results (built lazily Rust-side
        # from `optimal_steps_per_quote` + grid).
        total_constraints = dict(cd_result.total_constraints)
        baseline_constraints = dict(cd_result.baseline_constraints)
        if ratio_columns and original_df is not None:
            optimum_df = _stitch_optimal_ratio_columns(
                base_df=cd_result.dataframe,
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
            baseline_slice = original_df.filter(pl.col(self.scenario_value) == 1.0)
            for label, num_col, denom_col in ratio_columns:
                baseline_constraints[label] = _safe_ratio_from_columns(
                    baseline_slice, num_col, denom_col
                )

        return RatebookResult(
            factor_tables=named_tables,
            lambdas=cd_result.lambdas,
            total_objective=cd_result.total_objective,
            total_constraints=total_constraints,
            baseline_objective=cd_result.baseline_objective,
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

        # Build a `FactorContext` per candidate column via the same
        # opaque wrapper the public path uses. The wrapper does
        # alignment validation against `grid.quote_ids`; even though
        # the screening loop only uses each per-factor `FactorContext`
        # to drive `solve_grouped_py`, routing through the public
        # builder keeps the dataframe-extraction path in one place.
        contexts_wrapper = RatebookFactorContexts.from_dataframe(
            factors,
            [[col] for col in factors.columns],
            quote_id=self.quote_id if self.quote_id in factors.columns else None,
            expected_quote_ids=grid.quote_ids,
        )
        all_contexts = contexts_wrapper._factor_contexts_for_solver()

        lifts: list[tuple[str, float]] = []
        for col, ctx in zip(factors.columns, all_contexts):
            residuals = [1.0] * n_quotes

            result = solve_grouped_py(
                grid,
                context=ctx,
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
        factors: pl.DataFrame | RatebookFactorContexts,
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
                c for c in constraint_names if not _is_ratio_spec(self.constraints[c])
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
            _linspace(
                float(threshold_ranges[name][0]),
                float(threshold_ranges[name][1]),
                n_points_per_dim,
            )
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

        # Pre-build a `RatebookFactorContexts` once for the whole
        # sweep. The same opaque handle is threaded through the public
        # `factors` argument on every per-point `self.solve()` call,
        # so each point pays only an O(1) fingerprint check rather
        # than re-extracting labels and re-hashing the factor source.
        if isinstance(factors, RatebookFactorContexts):
            # Contexts mode: take the specs from the wrapper. Reject
            # an explicit `factor_columns` that disagrees.
            resolved_factor_specs = _resolve_factor_specs_from_contexts(
                factors, factor_columns, self.factor_columns
            )
            frontier_factor_contexts = factors
        else:
            # DataFrame mode: resolve specs (auto-discovery permitted),
            # then build contexts once with `expected_quote_ids` set so
            # every per-point solve's fingerprint check passes
            # trivially. For sum-only sweeps we already have a grid
            # built; ratio sweeps build per-point grids inside
            # solve(), so we derive expected quote IDs from the
            # DataFrame directly.
            resolved_factor_specs = factor_columns or self.factor_columns
            if resolved_factor_specs is None:
                resolved_factor_specs = self._discover_structure(df_or_grid, factors)
            expected_quote_ids = _expected_quote_ids_for_frontier(
                df_or_grid, grid, self.quote_id
            )
            frontier_factor_contexts = RatebookFactorContexts.from_dataframe(
                factors,
                resolved_factor_specs,
                quote_id=(self.quote_id if self.quote_id in factors.columns else None),
                separator="\x1f",
                expected_quote_ids=expected_quote_ids,
            )

        # Sweep — predictor-corrector warm starting. The frontier visits
        # points in nearest-neighbour order, so the optimal λ vector
        # changes smoothly along the visit path (away from active-set
        # kinks). Instead of zero-order warm starting (copy the previous
        # point's λ), we linearly extrapolate from the two most recent
        # visits into the next threshold combo and use that as
        # ``initial_lambdas``. The extrapolated λ is fed to the solver as
        # a starting hint; the inner subgradient corrector still runs to
        # convergence, so a bad predictor degrades gracefully to one or
        # two extra iterations rather than producing a wrong answer.
        prev_thresholds: list[float] | None = None
        prev_lambdas: dict[str, float] | None = (
            dict(initial_lambdas) if initial_lambdas else None
        )
        prev2_thresholds: list[float] | None = None
        prev2_lambdas: dict[str, float] | None = None
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

            # Predictor: linearly extrapolate λ along the prev2→prev path
            # to the new point's thresholds. Falls back to zero-order
            # (the previous point's λ) for the first two points and for
            # paths where the predictor would be degenerate (zero-length
            # base segment). All-None initial state keeps the very first
            # point on whatever ``initial_lambdas`` the user supplied.
            init_lambdas = prev_lambdas
            if (
                prev_thresholds is not None
                and prev2_thresholds is not None
                and prev_lambdas is not None
                and prev2_lambdas is not None
            ):
                predicted = _extrapolate_lambdas(
                    prev2_thresholds,
                    prev2_lambdas,
                    prev_thresholds,
                    prev_lambdas,
                    list(thresholds),
                )
                if predicted is not None:
                    init_lambdas = predicted

            result = self.solve(
                grid,
                frontier_factor_contexts,
                factor_columns=resolved_factor_specs,
                lambdas=init_lambdas,
                _constraints_override=modified_constraints,
            )

            prev2_thresholds = prev_thresholds
            prev2_lambdas = prev_lambdas
            prev_thresholds = list(thresholds)
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
                columns[f"threshold_{name}"] = [p[1]["thresholds"][k] for p in points]
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


def _extrapolate_lambdas(
    t_prev2: list[float],
    lam_prev2: dict[str, float],
    t_prev: list[float],
    lam_prev: dict[str, float],
    t_next: list[float],
) -> dict[str, float] | None:
    """Linearly extrapolate λ for the next frontier point along the
    direction the path just travelled.

    Models λ as locally affine in threshold along the visit path:

        λ(t) ≈ λ_prev + ((t - t_prev) · (t_prev - t_prev2) / |t_prev - t_prev2|²)
                       · (λ_prev - λ_prev2)

    The scalar projection of ``t_next - t_prev`` onto ``t_prev - t_prev2``
    gives "how far along the prev2→prev direction" the next point sits,
    so a step in the same direction extrapolates by one full slope, a
    perpendicular step extrapolates by zero (recovers zero-order), and a
    backward step extrapolates by a negative slope (cancels overshoot).

    Returns None if the prev2→prev segment has zero length — the slope
    is undefined and the caller should fall back to copying ``lam_prev``
    verbatim. λ values are clamped to be non-negative (Lagrange
    multipliers for one-sided sum/ratio constraints can never be < 0).
    """
    dt_prev_squared = sum((a - b) ** 2 for a, b in zip(t_prev, t_prev2))
    if dt_prev_squared == 0.0:
        return None

    dot = sum((tn - tp) * (tp - tp2) for tn, tp, tp2 in zip(t_next, t_prev, t_prev2))
    fraction = dot / dt_prev_squared

    predicted: dict[str, float] = {}
    for name, lam in lam_prev.items():
        prev_lam = lam_prev2.get(name, lam)
        slope = lam - prev_lam
        predicted[name] = max(0.0, lam + fraction * slope)
    return predicted


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


def _resolve_factor_specs_from_contexts(
    factors: RatebookFactorContexts,
    factor_columns_arg: list[list[str]] | None,
    optimiser_factor_columns: list[list[str]] | None,
) -> list[list[str]]:
    """Pick the authoritative factor specs when the caller supplied a
    :class:`RatebookFactorContexts`. The contexts' own
    ``factor_specs`` are the source of truth; any other source must
    either match or be absent.

    Auto-discovery is unavailable in contexts mode because the contexts
    already encode the chosen factors — re-discovering would either
    contradict them or be a wasted scan.
    """
    contexts_specs = factors.factor_specs
    if factor_columns_arg is not None and factor_columns_arg != contexts_specs:
        raise ValueError(
            f"factor_columns argument {factor_columns_arg} conflicts with "
            f"RatebookFactorContexts.factor_specs {contexts_specs}. The "
            f"contexts' specs are authoritative; either omit factor_columns "
            f"or rebuild the contexts with the desired specs."
        )
    if (
        optimiser_factor_columns is not None
        and optimiser_factor_columns != contexts_specs
    ):
        raise ValueError(
            f"RatebookOptimiser.factor_columns={optimiser_factor_columns} "
            f"conflicts with RatebookFactorContexts.factor_specs={contexts_specs}. "
            f"The contexts' specs are authoritative; either reconstruct the "
            f"optimiser with matching factor_columns=None (or matching specs) "
            f"or rebuild the contexts."
        )
    return contexts_specs


def _resolve_factor_contexts(
    factors: pl.DataFrame | RatebookFactorContexts,
    factor_specs: list[list[str]],
    *,
    grid: QuoteGrid,
    quote_id_col: str,
) -> RatebookFactorContexts:
    """Coerce ``factors`` into a :class:`RatebookFactorContexts` whose
    fingerprint matches ``grid.quote_id_fingerprint``.

    For DataFrame inputs we build the contexts here with
    ``expected_quote_ids=grid.quote_ids`` so the resulting fingerprint
    is, by construction, identical to the grid's. For pre-built
    contexts we verify the fingerprint instead and reject mismatches.

    Validation rules:

    * contexts with ``quote_id_fingerprint is None`` are rejected —
      they were built without a quote-id source AND without
      ``expected_quote_ids``, so no alignment can be proven.
    * contexts with a fingerprint that disagrees with the grid are
      rejected — they were built against a different quote axis.
    * contexts whose ``n_quotes`` disagrees with the grid are rejected
      with both counts in the message.
    """
    if isinstance(factors, RatebookFactorContexts):
        if factors.n_quotes != grid.n_quotes:
            raise ValueError(
                f"RatebookFactorContexts has n_quotes={factors.n_quotes} "
                f"but QuoteGrid has n_quotes={grid.n_quotes}"
            )
        if factors.quote_id_fingerprint is None:
            raise ValueError(
                "RatebookFactorContexts has no quote_id_fingerprint; build "
                "the contexts with expected_quote_ids=quote_grid.quote_ids "
                "(or include a quote_id column) so solve-time alignment can "
                "be proven."
            )
        if factors.quote_id_fingerprint != grid.quote_id_fingerprint:
            raise ValueError(
                f"RatebookFactorContexts quote_id_fingerprint "
                f"0x{factors.quote_id_fingerprint:016x} does not match "
                f"QuoteGrid fingerprint 0x{grid.quote_id_fingerprint:016x}. "
                f"The contexts were built against a different quote axis; "
                f"rebuild them against this grid's quote_ids."
            )
        return factors

    # DataFrame mode: build contexts aligned to the grid quote axis.
    # If the DataFrame happens to include a quote_id column, pass it
    # through so the chunked-builder path validates IDs explicitly;
    # otherwise fall back to positional-trust against
    # expected_quote_ids=grid.quote_ids.
    return RatebookFactorContexts.from_dataframe(
        factors,
        factor_specs,
        quote_id=quote_id_col if quote_id_col in factors.columns else None,
        separator="\x1f",
        expected_quote_ids=grid.quote_ids,
    )


def _expected_quote_ids_for_frontier(
    df_or_grid: pl.DataFrame | QuoteGrid,
    grid: pl.DataFrame | QuoteGrid,
    quote_id_col: str,
) -> list[str]:
    """Derive the canonical (lex-sorted) quote-id sequence the frontier
    will solve against.

    In sum-constraint frontier mode ``grid`` is already a built
    :class:`QuoteGrid` — we just return its ``quote_ids``. In
    ratio-constraint mode the frontier defers grid-build to per-point
    solves (each point linearises with a different ``L``), so ``grid``
    here is still the raw DataFrame. In that case we compute the
    sorted unique ``quote_id`` values from the DataFrame, matching the
    order ``QuoteGridBuilder`` will lex-sort them into on every
    per-point build.
    """
    if isinstance(grid, QuoteGrid):
        return grid.quote_ids
    # Ratio-frontier path: grid is still a DataFrame. Derive the
    # canonical quote_ids the same way QuoteGridBuilder.build() will,
    # via cast-to-str then lex-sort over unique values.
    df = grid if isinstance(grid, pl.DataFrame) else df_or_grid
    assert isinstance(df, pl.DataFrame), (
        "frontier(): expected DataFrame for quote-id derivation in ratio mode"
    )
    return (
        df.select(pl.col(quote_id_col).cast(pl.Utf8).unique().sort())
        .to_series()
        .to_list()
    )
