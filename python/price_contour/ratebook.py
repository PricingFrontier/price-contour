"""RatebookOptimiser — coordinate descent for ratebook factor optimisation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from price_contour._frontier_helpers import (
    _cartesian_product,
    _linspace,
    frontier_points_schema,
)
from price_contour._grid_utils import build_grid
from price_contour._price_contour import (
    FactorContext,
    QuoteGrid,
    RatebookEvaluation,
    RatebookFactorContexts,
    build_ratebook_factor_contexts_from_parquet_chunked_py,
    evaluate_ratebook_py,
    run_cd_pass_py,
    solve_grouped_py,
)
from price_contour._ratio_results import (
    _safe_ratio_from_columns,
    _stitch_optimal_ratio_columns,
)
from price_contour.solver import (
    _baseline_rows,
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


FACTOR_SEPARATOR = "\x1f"
"""Separator between the constituent levels of a composite factor label."""

_FORMAT_VERSION = 2

# ``config.json`` keys by save format. ``load`` accepts exactly these.
_CONFIG_KEYS_V1 = frozenset(
    {
        "lambdas",
        "constraints",
        "baseline_constraints",
        "total_objective",
        "baseline_objective",
        "cd_iterations",
        "converged",
        "clamp_rate",
        "factor_order",
    }
)
_CONFIG_KEYS_V2 = _CONFIG_KEYS_V1 | {
    "format_version",
    "factor_separator",
    "per_factor_results",
    "constraint_bounds",
    "scenario_values",
    "baseline_scenario_value",
    "n_quotes",
    "n_quotes_clamped_low",
    "n_quotes_clamped_high",
}


class ResultUnavailableError(RuntimeError):
    """A result field that was not computed or not persisted was accessed.

    Subclasses ``RuntimeError`` rather than ``AttributeError`` so ``hasattr``
    cannot silently hide it.
    """


def quote_results_schema(constraint_names: list[str]) -> dict[str, pl.DataType]:
    """Exact ``{column: dtype}`` of ``quote_results`` for sum constraints
    ``constraint_names`` (DESIGN_DECISIONS §13.2), in column order."""
    schema: dict[str, pl.DataType] = {
        "quote_id": pl.String(),
        "optimal_step": pl.Int32(),
        "optimal_scenario_value": pl.Float32(),
        "optimal_objective": pl.Float32(),
    }
    for name in constraint_names:
        schema[f"optimal_{name}"] = pl.Float32()
    schema["factor_product"] = pl.Float32()
    schema["clamped_low"] = pl.Boolean()
    schema["clamped_high"] = pl.Boolean()
    return schema


@dataclass(frozen=True)
class PerFactorRecord:
    """One inner grouped solve of the coordinate-descent loop."""

    cd_iteration: int
    """1-based coordinate-descent pass."""
    factor: str
    """Factor spec name (``":".join(columns)``)."""
    factor_index: int
    """Position of the factor in the spec order."""
    total_objective: float
    total_constraints: dict[str, float]
    lambdas: dict[str, float]
    clamp_rate: float
    """Fraction of (quote, candidate) targets outside the grid in this solve."""
    inner_iterations: int
    inner_converged: bool


def _unavailable(field_name: str, why: str) -> ResultUnavailableError:
    return ResultUnavailableError(f"RatebookResult.{field_name} is unavailable: {why}")


_LOADED_V1 = "this result was loaded from a format-1 save, which did not persist it"


@dataclass
class RatebookResult:
    """Result of ratebook coordinate descent optimisation.

    Every total is the canonical evaluation of ``factor_tables``
    (DESIGN_DECISIONS §13.1): each quote at the grid step nearest the f32
    product of its factor values. ``RatebookOptimiser.evaluate(grid,
    factors, result.factor_tables)`` reproduces them exactly.

    Fields that a loaded result does not carry raise
    :class:`ResultUnavailableError` on access instead of returning a default.
    """

    factor_tables: dict[str, dict[str, float]]
    lambdas: dict[str, float]
    total_objective: float
    total_constraints: dict[str, float]
    baseline_objective: float
    baseline_constraints: dict[str, float]
    cd_iterations: int
    converged: bool
    """Coordinate-descent convergence only: the largest factor-value change
    over the last pass fell below ``cd_tolerance``. It does not check that
    the constraints are met."""
    clamp_rate: float
    """Search-space diagnostic (§13.11): the mean, over every grouped solve,
    of the fraction of (quote, candidate) targets outside the scenario range.
    It does not count quotes at a grid edge; see ``n_quotes_clamped_low`` /
    ``n_quotes_clamped_high``."""
    _per_factor_results: tuple[PerFactorRecord, ...] | None = field(
        default=None, repr=False
    )
    _constraint_bounds: dict[str, float] | None = field(default=None, repr=False)
    _scenario_values: tuple[float, ...] | None = field(default=None, repr=False)
    _baseline_scenario_value: float | None = field(default=None, repr=False)
    _n_quotes: int | None = field(default=None, repr=False)
    _n_quotes_clamped_low: int | None = field(default=None, repr=False)
    _n_quotes_clamped_high: int | None = field(default=None, repr=False)
    _evaluation: RatebookEvaluation | None = field(
        default=None, repr=False, compare=False
    )
    _quote_results_frame: pl.DataFrame | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def per_factor_results(self) -> tuple[PerFactorRecord, ...]:
        """One record per inner grouped solve, in (CD pass, factor) order."""
        if self._per_factor_results is None:
            raise _unavailable("per_factor_results", _LOADED_V1)
        return self._per_factor_results

    @property
    def constraint_bounds(self) -> dict[str, float]:
        """Absolute bound of each constraint (§13.5), in constraint order."""
        if self._constraint_bounds is None:
            raise _unavailable("constraint_bounds", _LOADED_V1)
        return self._constraint_bounds

    @property
    def scenario_values(self) -> tuple[float, ...]:
        """The grid's scenario values, ascending."""
        if self._scenario_values is None:
            raise _unavailable("scenario_values", _LOADED_V1)
        return self._scenario_values

    @property
    def baseline_scenario_value(self) -> float:
        """Scenario value of the baseline step (nearest 1.0, §13.6)."""
        if self._baseline_scenario_value is None:
            raise _unavailable("baseline_scenario_value", _LOADED_V1)
        return self._baseline_scenario_value

    @property
    def n_quotes(self) -> int:
        if self._n_quotes is None:
            raise _unavailable("n_quotes", _LOADED_V1)
        return self._n_quotes

    @property
    def n_quotes_clamped_low(self) -> int:
        """Quotes whose factor product lies strictly below the scenario range."""
        if self._n_quotes_clamped_low is None:
            raise _unavailable("n_quotes_clamped_low", _LOADED_V1)
        return self._n_quotes_clamped_low

    @property
    def n_quotes_clamped_high(self) -> int:
        """Quotes whose factor product lies strictly above the scenario range."""
        if self._n_quotes_clamped_high is None:
            raise _unavailable("n_quotes_clamped_high", _LOADED_V1)
        return self._n_quotes_clamped_high

    @property
    def quote_results(self) -> pl.DataFrame:
        """Per-quote evaluation (§13.2); see :func:`quote_results_schema`.

        Not persisted by :meth:`save`: on a loaded result, reproduce it with
        ``RatebookOptimiser.evaluate(grid, factors, result.factor_tables)``.
        """
        if self._quote_results_frame is not None:
            return self._quote_results_frame
        if self._evaluation is None:
            raise _unavailable(
                "quote_results",
                "per-quote results are not persisted; call "
                "RatebookOptimiser.evaluate(grid, factors, result.factor_tables)",
            )
        return self._evaluation.quote_results

    def save(self, path: str | Path) -> None:
        """Save the result to a parameters folder (format 2, §13.10).

        Creates one JSON per factor plus a config.json:

            path/
              config.json
              region.json
              age_band.json
              ...

        Factor-table keys are written verbatim. The per-quote frame is not
        saved; :meth:`RatebookOptimiser.evaluate` reproduces it exactly from
        the saved tables.
        """
        path = Path(path)
        factor_order = list(self.factor_tables.keys())
        filenames = {name: _factor_filename(name) for name in factor_order}
        clashes = _duplicates(filenames.values())
        if clashes:
            raise ValueError(
                f"factor names {sorted(n for n, f in filenames.items() if f in clashes)} "
                f"map to the same file name; rename a factor column"
            )
        config = {
            "format_version": _FORMAT_VERSION,
            "factor_separator": FACTOR_SEPARATOR,
            "lambdas": self.lambdas,
            "constraints": self.total_constraints,
            "baseline_constraints": self.baseline_constraints,
            "total_objective": self.total_objective,
            "baseline_objective": self.baseline_objective,
            "cd_iterations": self.cd_iterations,
            "converged": self.converged,
            "clamp_rate": self.clamp_rate,
            "factor_order": factor_order,
            "per_factor_results": [asdict(r) for r in self.per_factor_results],
            "constraint_bounds": self.constraint_bounds,
            "scenario_values": list(self.scenario_values),
            "baseline_scenario_value": self.baseline_scenario_value,
            "n_quotes": self.n_quotes,
            "n_quotes_clamped_low": self.n_quotes_clamped_low,
            "n_quotes_clamped_high": self.n_quotes_clamped_high,
        }
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(config, indent=2))
        for factor_name, table in self.factor_tables.items():
            factor_data = {"columns": factor_name.split(":"), "table": table}
            (path / filenames[factor_name]).write_text(
                json.dumps(factor_data, indent=2)
            )

    def to_rating_entries(self) -> dict[str, pl.DataFrame]:
        """Convert factor tables to rating-step DataFrames.

        Returns a dict mapping factor name to a DataFrame with columns:
        [level_col_1, ..., level_col_n, factor].
        """
        result = {}
        for factor_name, table in self.factor_tables.items():
            cols = factor_name.split(":")
            rows = []
            for key, value in table.items():
                parts = _split_label(key, len(cols), factor_name)
                row: dict[str, Any] = dict(zip(cols, parts))
                row["factor"] = value
                rows.append(row)
            result[factor_name] = pl.DataFrame(
                rows, schema={**{c: pl.String for c in cols}, "factor": pl.Float64}
            )
        return result

    @classmethod
    def load(cls, path: str | Path) -> RatebookResult:
        """Load a result saved by :meth:`save` (format 1 or 2).

        The loaded result has no per-quote frame. A format-1 save lacks the
        0.5.0 fields; they raise :class:`ResultUnavailableError` on access.
        """
        path = Path(path)
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = json.loads(config_path.read_text())
        version = config.get("format_version", 1)
        allowed = {1: _CONFIG_KEYS_V1, 2: _CONFIG_KEYS_V2}.get(version)
        if allowed is None:
            raise ValueError(f"unsupported RatebookResult format_version {version!r}")
        unknown = set(config) - allowed
        if unknown:
            raise ValueError(
                f"unknown field(s) {sorted(unknown)} in saved RatebookResult "
                f"(format {version})"
            )
        missing = (allowed - {"format_version"}) - set(config)
        if missing:
            raise ValueError(
                f"missing field(s) {sorted(missing)} in saved RatebookResult "
                f"(format {version})"
            )
        if version == 2 and config["factor_separator"] != FACTOR_SEPARATOR:
            raise ValueError(
                f"saved factor_separator {config['factor_separator']!r} != "
                f"{FACTOR_SEPARATOR!r}"
            )

        factor_tables: dict[str, dict[str, float]] = {}
        for factor_name in config["factor_order"]:
            factor_path = path / _factor_filename(factor_name)
            if not factor_path.exists():
                raise FileNotFoundError(
                    f"factor table for '{factor_name}' not found: {factor_path}"
                )
            factor_data = json.loads(factor_path.read_text())
            if "table" not in factor_data:
                raise ValueError(f"factor file {factor_path} has no 'table'")
            table = factor_data["table"]
            if version == 1:
                table = _v1_table_labels(table, factor_name)
            factor_tables[factor_name] = table

        common: dict[str, Any] = {
            "factor_tables": factor_tables,
            "lambdas": config["lambdas"],
            "total_objective": config["total_objective"],
            "total_constraints": config["constraints"],
            "baseline_objective": config["baseline_objective"],
            "baseline_constraints": config["baseline_constraints"],
            "converged": config["converged"],
            "cd_iterations": config["cd_iterations"],
            "clamp_rate": config["clamp_rate"],
        }
        if version == 1:
            return cls(**common)
        return cls(
            **common,
            _per_factor_results=tuple(
                PerFactorRecord(**record) for record in config["per_factor_results"]
            ),
            _constraint_bounds=config["constraint_bounds"],
            _scenario_values=tuple(config["scenario_values"]),
            _baseline_scenario_value=config["baseline_scenario_value"],
            _n_quotes=config["n_quotes"],
            _n_quotes_clamped_low=config["n_quotes_clamped_low"],
            _n_quotes_clamped_high=config["n_quotes_clamped_high"],
        )


def _factor_filename(factor_name: str) -> str:
    return factor_name.replace(":", "_") + ".json"


def _duplicates(values: Any) -> set[Any]:
    seen: set[Any] = set()
    dupes: set[Any] = set()
    for v in values:
        (dupes if v in seen else seen).add(v)
    return dupes


def _split_label(label: str, n_columns: int, factor_name: str) -> list[str]:
    """Split a (possibly composite) factor level label into its parts."""
    if n_columns == 1:
        return [label]
    parts = label.split(FACTOR_SEPARATOR)
    if len(parts) != n_columns:
        raise ValueError(
            f"level {label!r} of composite factor '{factor_name}' does not split "
            f"into {n_columns} parts on the factor separator"
        )
    return parts


def _v1_table_labels(table: dict[str, float], factor_name: str) -> dict[str, float]:
    """Format 1 wrote composite labels with ':' in place of the separator.
    Convert back, refusing any label that is ambiguous."""
    n_columns = len(factor_name.split(":"))
    if n_columns == 1:
        return table
    converted: dict[str, float] = {}
    for key, value in table.items():
        parts = key.split(":")
        if len(parts) != n_columns:
            raise ValueError(
                f"format-1 level {key!r} of composite factor '{factor_name}' is "
                f"ambiguous: it does not split into {n_columns} parts on ':'"
            )
        converted[FACTOR_SEPARATOR.join(parts)] = value
    return converted


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
        _validate_candidate_settings(
            candidate_min, candidate_max, candidate_steps, max_cd_iterations
        )

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
        ratio_bounds: dict[str, float] = {}
        if ratio_names and original_df is not None:
            linearised = _linearise_ratio_constraints(
                original_df,
                constraints,
                scenario_value_col=self.scenario_value,
                scenario_index_col=self.scenario_index,
                quote_id_col=self.quote_id,
            )
            ratio_columns = linearised.ratio_columns
            ratio_bounds = linearised.ratio_bounds
            grid_input: pl.DataFrame | QuoteGrid = linearised.df
            cd_constraints: dict[str, dict[str, float]] = linearised.sum_constraints
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

        evaluation = cd_result.evaluation
        spec_names = [":".join(spec) for spec in factor_specs]
        per_factor_results = tuple(
            PerFactorRecord(factor=spec_names[record["factor_index"]], **record)
            for record in cd_result.per_call_records
        )

        # Convert per-group `Vec<f32>` factor values back to the public
        # `dict[str, float]` shape that `RatebookResult.factor_tables`
        # carries (and that callers persist via `save()`).
        named_tables = {
            name: {label: float(value) for label, value in zip(labels, values)}
            for name, labels, values in zip(
                spec_names, factor_group_labels, cd_result.factor_values
            )
        }

        # Every total is the canonical evaluation of the final factor
        # tables (§13.1). Ratio labels are then reported as the actual ratio
        # at the optimum / baseline rather than the linearised total, with
        # the numerator / denominator columns stitched onto the per-quote
        # frame, and the ratio bound in ratio units (§13.5).
        total_constraints = dict(evaluation.total_constraints)
        baseline_constraints = dict(evaluation.baseline_constraints)
        constraint_bounds = dict(cd_result.constraint_bounds)
        quote_results_frame: pl.DataFrame | None = None
        if ratio_columns and original_df is not None:
            quote_results_frame = _stitch_optimal_ratio_columns(
                base_df=evaluation.quote_results,
                original_df=original_df,
                ratio_columns=ratio_columns,
                quote_id_col=self.quote_id,
                scenario_index_col=self.scenario_index,
            )
            baseline_slice = _baseline_rows(
                original_df,
                quote_id_col=self.quote_id,
                scenario_index_col=self.scenario_index,
                scenario_value_col=self.scenario_value,
            )
            for label, num_col, denom_col in ratio_columns:
                total_constraints[label] = _safe_ratio_from_columns(
                    quote_results_frame, f"optimal_{num_col}", f"optimal_{denom_col}"
                )
                baseline_constraints[label] = _safe_ratio_from_columns(
                    baseline_slice, num_col, denom_col
                )
                constraint_bounds[label] = ratio_bounds[label]

        return RatebookResult(
            factor_tables=named_tables,
            lambdas=cd_result.lambdas,
            total_objective=evaluation.total_objective,
            total_constraints=total_constraints,
            baseline_objective=evaluation.baseline_objective,
            baseline_constraints=baseline_constraints,
            cd_iterations=cd_result.cd_iterations,
            converged=cd_result.converged,
            clamp_rate=cd_result.clamp_rate,
            _per_factor_results=per_factor_results,
            _constraint_bounds=constraint_bounds,
            _scenario_values=tuple(evaluation.scenario_values),
            _baseline_scenario_value=evaluation.baseline_scenario_value,
            _n_quotes=evaluation.n_quotes,
            _n_quotes_clamped_low=evaluation.n_quotes_clamped_low,
            _n_quotes_clamped_high=evaluation.n_quotes_clamped_high,
            _evaluation=evaluation,
            _quote_results_frame=quote_results_frame,
        )

    def evaluate(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        factors: pl.DataFrame | RatebookFactorContexts,
        factor_tables: Mapping[str, Mapping[str, float]],
    ) -> RatebookEvaluation:
        """Evaluate ratebook factor tables per quote (DESIGN_DECISIONS §13.3).

        Runs the same kernel ``solve()`` reports from: each quote is priced at
        the grid step nearest the f32 product of its factor values. Given the
        tables, the result does not depend on λ.
        ``evaluate(grid, factors, result.factor_tables)`` reproduces
        ``result``'s totals and ``quote_results`` exactly, which makes it the
        supported way to materialise a frontier point or a loaded result.

        Parameters
        ----------
        df_or_grid : pl.DataFrame | QuoteGrid
            Scored DataFrame or pre-built QuoteGrid.
        factors : pl.DataFrame | RatebookFactorContexts
            Per-quote factors, as for ``solve()``.
        factor_tables : Mapping[str, Mapping[str, float]]
            One table per factor spec (``":".join(columns)``), covering
            exactly the levels present in ``factors``. Rates must be finite
            and > 0.

        Returns
        -------
        RatebookEvaluation
            ``quote_results``, ``total_objective``, ``total_constraints``,
            ``baseline_objective``, ``baseline_constraints``, ``n_quotes``,
            ``n_quotes_clamped_low``, ``n_quotes_clamped_high``,
            ``scenario_values`` and ``baseline_scenario_value``.
        """
        ratio_names = _ratio_constraint_names(self.constraints)
        if ratio_names:
            raise ValueError(
                f"RatebookOptimiser.evaluate() does not support ratio "
                f"constraints ({ratio_names}): a ratio has no per-step sum to "
                f"read from the grid. Use solve() for ratio reporting."
            )
        if isinstance(factors, RatebookFactorContexts):
            factor_specs = _resolve_factor_specs_from_contexts(
                factors, None, self.factor_columns
            )
        elif self.factor_columns is None:
            raise ValueError(
                "evaluate() with a factors DataFrame needs factor_columns on "
                "the optimiser (or pass a RatebookFactorContexts)"
            )
        else:
            factor_specs = self.factor_columns
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
        contexts = _resolve_factor_contexts(
            factors, factor_specs, grid=grid, quote_id_col=self.quote_id
        )
        solver_contexts = contexts._factor_contexts_for_solver()
        return evaluate_ratebook_py(
            grid,
            solver_contexts,
            _factor_values_for_contexts(factor_specs, solver_contexts, factor_tables),
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
    ) -> RatebookFrontierResult:
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
            Must be False: the ratebook frontier runs sequentially, each
            point warm-started from its neighbours. True raises rather than
            being silently ignored.

        Returns
        -------
        RatebookFrontierResult
            ``.points`` (one row per point; see ``frontier_points_schema``),
            ``.n_points``, ``.constraint_names`` and ``.factor_tables`` (one
            table set per row). A point's totals are the canonical
            evaluation of its tables, so
            ``evaluate(grid, factors, frontier.point_factor_tables(i))``
            reproduces row ``i`` exactly; re-solving with the row's λ is not
            guaranteed to.
        """
        if parallel:
            raise ValueError(
                "RatebookOptimiser.frontier() runs sequentially (each point is "
                "warm-started from its neighbours); parallel=True is not supported"
            )
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
        points: list[tuple[int, list[float], _FrontierRow]] = []

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

            # Record only what the row and the point's tables need, so the
            # point's per-quote evaluation is released before the next solve.
            points.append((idx, thresholds, _FrontierRow.from_result(result)))
            del result

        # Sort back to original (cartesian product) order
        points.sort(key=lambda x: x[0])
        results = [row for _, _, row in points]

        # Threshold columns: swept axes echo per-point combo values; unswept
        # axes echo the constructor threshold verbatim at every row (user
        # units). ``bound_*`` is the absolute bound each point was solved
        # against (§13.5).
        columns: dict[str, list[Any]] = {}
        swept_index = {name: idx for idx, name in enumerate(swept_names)}
        for name in constraint_names:
            if name in swept_index:
                k = swept_index[name]
                columns[f"threshold_{name}"] = [t[k] for _, t, _ in points]
            else:
                columns[f"threshold_{name}"] = [unswept_thresholds[name]] * len(points)
        for name in constraint_names:
            columns[f"bound_{name}"] = [r.constraint_bounds[name] for r in results]
        columns["total_objective"] = [r.total_objective for r in results]
        for name in constraint_names:
            columns[f"total_{name}"] = [r.total_constraints[name] for r in results]
        for name in constraint_names:
            columns[f"lambda_{name}"] = [r.lambdas[name] for r in results]
        columns["iterations"] = [r.cd_iterations for r in results]
        columns["converged"] = [r.converged for r in results]
        columns["clamp_rate"] = [r.clamp_rate for r in results]
        columns["n_quotes_clamped_low"] = [r.n_quotes_clamped_low for r in results]
        columns["n_quotes_clamped_high"] = [r.n_quotes_clamped_high for r in results]

        return RatebookFrontierResult(
            points=pl.DataFrame(
                columns, schema=frontier_points_schema("ratebook", constraint_names)
            ),
            constraint_names=constraint_names,
            factor_tables=[r.factor_tables for r in results],
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

        for name in self.constraints:
            metrics[f"constraint_{name}_total"] = result.total_constraints[name]
            metrics[f"constraint_{name}_baseline"] = result.baseline_constraints[name]
            metrics[f"lambda_{name}"] = result.lambdas[name]

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


def _validate_candidate_settings(
    candidate_min: float,
    candidate_max: float,
    candidate_steps: int,
    max_cd_iterations: int,
) -> None:
    """Candidate factor values must be a finite, positive, ordered range: a
    zero or negative rate would zero or flip a quote's price."""
    if not (math.isfinite(candidate_min) and candidate_min > 0):
        raise ValueError(f"candidate_min must be finite and > 0, got {candidate_min}")
    if not (math.isfinite(candidate_max) and candidate_max >= candidate_min):
        raise ValueError(
            f"candidate_max must be finite and >= candidate_min "
            f"({candidate_min}), got {candidate_max}"
        )
    if candidate_steps < 1:
        raise ValueError(f"candidate_steps must be >= 1, got {candidate_steps}")
    if max_cd_iterations < 1:
        raise ValueError(f"max_cd_iterations must be >= 1, got {max_cd_iterations}")


def _factor_values_for_contexts(
    factor_specs: list[list[str]],
    contexts: list[FactorContext],
    factor_tables: Mapping[str, Mapping[str, float]],
) -> list[list[float]]:
    """Order ``factor_tables`` into per-context value lists (context spec
    order, each context's ``group_labels`` order). Tables must cover exactly
    the factors and the levels present in the data."""
    names = [":".join(spec) for spec in factor_specs]
    unknown = sorted(set(factor_tables) - set(names))
    if unknown:
        raise ValueError(
            f"factor tables for unknown factor(s) {unknown}; factors: {names}"
        )
    missing = [name for name in names if name not in factor_tables]
    if missing:
        raise ValueError(f"no factor table for factor(s) {missing}")
    values: list[list[float]] = []
    for name, context in zip(names, contexts):
        table = factor_tables[name]
        labels = context.group_labels
        absent = [label for label in labels if label not in table]
        if absent:
            raise ValueError(
                f"factor table '{name}' has no rate for level(s) {absent[:10]!r}"
                f"{' …' if len(absent) > 10 else ''}"
            )
        extra = sorted(set(table) - set(labels))
        if extra:
            raise ValueError(
                f"factor table '{name}' has level(s) {extra[:10]!r} that no quote "
                f"has{' …' if len(extra) > 10 else ''}"
            )
        values.append([table[label] for label in labels])
    return values


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


@dataclass(frozen=True)
class _FrontierRow:
    """The scalars and factor tables a ratebook frontier row needs from one
    point's solve (not its per-quote evaluation)."""

    factor_tables: dict[str, dict[str, float]]
    total_objective: float
    total_constraints: dict[str, float]
    constraint_bounds: dict[str, float]
    lambdas: dict[str, float]
    cd_iterations: int
    converged: bool
    clamp_rate: float
    n_quotes_clamped_low: int
    n_quotes_clamped_high: int

    @classmethod
    def from_result(cls, result: RatebookResult) -> _FrontierRow:
        return cls(
            factor_tables=result.factor_tables,
            total_objective=result.total_objective,
            total_constraints=result.total_constraints,
            constraint_bounds=result.constraint_bounds,
            lambdas=result.lambdas,
            cd_iterations=result.cd_iterations,
            converged=result.converged,
            clamp_rate=result.clamp_rate,
            n_quotes_clamped_low=result.n_quotes_clamped_low,
            n_quotes_clamped_high=result.n_quotes_clamped_high,
        )


class RatebookFrontierResult:
    """Frontier result for ratebook mode.

    Mirrors the interface of ``FrontierResult`` (``points``, ``n_points``,
    ``constraint_names``) and adds each point's factor tables.
    """

    def __init__(
        self,
        points: pl.DataFrame,
        constraint_names: list[str],
        factor_tables: list[dict[str, dict[str, float]]],
    ) -> None:
        if len(factor_tables) != points.height:
            raise ValueError(
                f"{len(factor_tables)} factor-table sets for {points.height} points"
            )
        self._points = points
        self._constraint_names = constraint_names
        self._factor_tables = factor_tables

    @property
    def points(self) -> pl.DataFrame:
        return self._points

    @property
    def n_points(self) -> int:
        return self._points.height

    @property
    def constraint_names(self) -> list[str]:
        return self._constraint_names

    @property
    def n_converged(self) -> int:
        return int(self._points["converged"].sum())

    @property
    def factor_tables(self) -> list[dict[str, dict[str, float]]]:
        """One ``factor_tables`` dict per row of ``points``."""
        return self._factor_tables

    def point_factor_tables(self, index: int) -> dict[str, dict[str, float]]:
        """The factor tables of point ``index`` (a row of ``points``)."""
        if not 0 <= index < self.n_points:
            raise IndexError(f"point index {index} out of range [0, {self.n_points})")
        return self._factor_tables[index]


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


def _validate_factor_specs(factor_specs: list[list[str]]) -> None:
    """Factor names are ``":".join(columns)`` and composite levels join their
    parts with ``FACTOR_SEPARATOR``, so a column containing either character,
    or two specs with the same name, would make results ambiguous."""
    for spec in factor_specs:
        for column in spec:
            if ":" in column or FACTOR_SEPARATOR in column:
                raise ValueError(
                    f"factor column {column!r} contains ':' or the factor "
                    f"separator; rename it (factor names are ':'-joined columns)"
                )
    names = [":".join(spec) for spec in factor_specs]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"factor(s) {repeated} are specified more than once")


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
    _validate_factor_specs(factor_specs)
    if isinstance(factors, RatebookFactorContexts):
        if factors.separator != FACTOR_SEPARATOR:
            raise ValueError(
                f"RatebookFactorContexts were built with separator "
                f"{factors.separator!r}; ratebook results label composite levels "
                f"with {FACTOR_SEPARATOR!r} (FACTOR_SEPARATOR). Rebuild the "
                f"contexts with the default separator."
            )
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
