"""
price_contour — High-performance insurance price optimisation.

Lagrangian dual decomposition for portfolio-level price optimisation,
with Rust core and Polars DataFrame interop.
"""

from enum import Enum
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("price-contour")
except PackageNotFoundError:
    # Editable installs without metadata fall back to a sentinel; this
    # should never happen in a built wheel but keeps importable from a
    # source tree that hasn't been `maturin develop`'d yet.
    __version__ = "0.0.0+local"

from price_contour.apply import ApplyOptimiser, apply_from_grid
from price_contour.builder import QuoteGrid, QuoteGridBuilder
from price_contour.frontier import FrontierResult, FrontierResultLike, frontier_summary
from price_contour.ratebook import RatebookOptimiser, RatebookResult
from price_contour.solver import OnlineOptimiser
from price_contour._price_contour import (
    ApplyResult,
    ChunkedApplyResult,
    GroupedSolveResult,
    SolveResult,
    apply_lambdas_to_parquet_chunked_py as _apply_lambdas_to_parquet_chunked_inner,
    build_grid_from_parquet_py as _build_grid_from_parquet_inner,
    build_grid_from_parquet_chunked_py as _build_grid_from_parquet_chunked_inner,
)


class SolverPath(str, Enum):
    """Mirror of the Rust ``SolverPath`` enum surfaced as strings on the
    frontier result's ``solver_path`` column.

    Use these constants instead of bare string literals when filtering
    or comparing — a typo against ``SolverPath.BISECTION`` is a
    ``NameError`` at import; a typo against ``"bisect"`` silently
    returns no rows. The ``str`` mixin makes ``SolverPath.BISECTION ==
    "bisection"`` evaluate ``True`` so legacy comparisons keep working.
    """

    BISECTION = "bisection"
    SUBGRADIENT = "subgradient"


class NonConvergenceReason(str, Enum):
    """Mirror of the Rust ``NonConvergenceReason`` enum surfaced as
    strings on the frontier result's ``non_convergence_reason`` column.

    See :class:`SolverPath` for the typo-safety rationale.
    """

    ABOVE_ENVELOPE = "above_envelope"
    BRACKET_EXHAUSTED = "bracket_exhausted"
    ITERATION_BUDGET_EXHAUSTED = "iteration_budget_exhausted"


def build_grid_from_parquet(
    path: str,
    constraint_columns: list[str],
    *,
    quote_id: str = "quote_id",
    scenario_index: str = "scenario_index",
    scenario_value: str = "scenario_value",
    objective: str = "expected_income",
) -> QuoteGrid:
    """Build a QuoteGrid directly from a Parquet file.

    Loads the entire Parquet file into memory at once. For very large files
    that may exceed available memory, use :func:`build_grid_from_parquet_chunked`
    instead, which streams the file in fixed-size row slices.

    Parameters
    ----------
    path : str
        Path to the Parquet file.
    constraint_columns : list[str]
        Column names to use as constraints.
    quote_id : str
        Column name for quote identifiers.
    scenario_index : str
        Column name for scenario step indices.
    scenario_value : str
        Column name for scenario values (e.g. price multipliers).
    objective : str
        Column name for the objective values.

    Returns
    -------
    QuoteGrid
    """
    return _build_grid_from_parquet_inner(
        path,
        constraint_columns,
        quote_id=quote_id,
        scenario_index=scenario_index,
        scenario_value=scenario_value,
        objective=objective,
    )


def build_grid_from_parquet_chunked(
    path: str,
    constraint_columns: list[str],
    chunk_size: int,
    *,
    quote_id: str = "quote_id",
    scenario_index: str = "scenario_index",
    scenario_value: str = "scenario_value",
    objective: str = "expected_income",
    n_steps: int | None = None,
) -> QuoteGrid:
    """Build a QuoteGrid by streaming a Parquet file in fixed-size row slices.

    The parquet IO buffer scales with ``chunk_size`` (only one slice resident
    at a time) — a measurable win over the one-shot path's whole-file read
    and Polars sort buffer. The final ``QuoteGrid`` itself is still
    O(total_rows × n_columns × 4 bytes), because every quote ends up
    resident in flat Rust vectors; that's inherent to the solver's data
    layout. Use this when your parquet exceeds available memory in its raw
    form, not as a way to avoid loading the grid.

    Each slice is read via Polars' ``ParquetReader::with_slice`` so only
    the row groups overlapping the slice range are deserialised; column
    projection means only the four schema columns plus ``constraint_columns``
    are decoded.

    ``chunk_size`` is rounded down to a multiple of ``n_steps`` so every
    slice ends on a quote boundary; no carry buffer is needed for a
    well-formed parquet. ``chunk_size`` must be at least ``n_steps``.

    The resulting grid is sorted by ``quote_id`` (the underlying builder
    performs an in-place sort at build time), so the order of quotes in the
    parquet does not matter — only the per-quote layout (``n_steps`` rows in
    ``scenario_index`` order) matters.

    Parameters
    ----------
    path : str
        Path to the Parquet file.
    constraint_columns : list[str]
        Column names to use as constraints.
    chunk_size : int
        Target number of rows per IO slice. Rounded down to the nearest
        multiple of ``n_steps``. Must be > 0 and >= ``n_steps``.
    quote_id : str
        Column name for quote identifiers.
    scenario_index : str
        Column name for scenario step indices.
    scenario_value : str
        Column name for scenario values.
    objective : str
        Column name for the objective values.
    n_steps : int, optional
        If known upfront, locks the per-quote step count. When ``None``
        (default), the value is auto-detected from the first chunk by
        finding the first ``scenario_index`` reset to 0. Pass explicitly if
        you need to support a parquet whose first chunk would not contain
        at least two complete quotes.

    Returns
    -------
    QuoteGrid
    """
    return _build_grid_from_parquet_chunked_inner(
        path,
        constraint_columns,
        chunk_size,
        quote_id=quote_id,
        scenario_index=scenario_index,
        scenario_value=scenario_value,
        objective=objective,
        n_steps=n_steps,
    )


def apply_lambdas_to_parquet_chunked(
    parquet_in: str,
    parquet_out: str,
    lambdas: dict[str, float],
    constraints: dict[str, dict[str, float]],
    chunk_size: int,
    *,
    quote_id: str = "quote_id",
    scenario_index: str = "scenario_index",
    scenario_value: str = "scenario_value",
    objective: str = "expected_income",
    n_steps: int | None = None,
) -> ChunkedApplyResult:
    """Apply fixed lambdas to a Parquet input, streaming results to a Parquet output.

    Reads ``parquet_in`` in fixed-size row slices and writes per-quote
    apply results to ``parquet_out`` one row group at a time. Returns a
    :class:`ChunkedApplyResult` with aggregate totals — the
    **whole-portfolio** per-quote ``optimal_steps`` array is never
    materialised; only one chunk's ``optimal_steps`` (``chunk_size / n_steps``
    entries) is alive at a time and is dropped after the chunk's row group
    is written. Callers who want the per-row results read them back via
    ``pl.read_parquet(parquet_out)``.

    Use this when the input parquet exceeds available memory or you want
    bounded peak memory regardless of file size. For small inputs, the
    in-memory :class:`ApplyOptimiser` is simpler.

    **Ratio constraints are not supported on this path** — the per-chunk
    mini-grid drops the raw numerator/denominator columns that the
    apply-time linearisation needs. Use :class:`ApplyOptimiser` on a
    DataFrame for ratio constraints. The chunked path will reject
    ratio-shaped specs upfront with a clear error.

    Parameters
    ----------
    parquet_in : str
        Path to the input Parquet file. Rows must be grouped by
        ``quote_id`` (each quote occupies ``n_steps`` contiguous rows in
        ``scenario_index`` order). Within a chunk the layout is validated
        per-row; across chunks the order does not matter.
    parquet_out : str
        Path to the output Parquet file. Overwritten if it exists. Schema
        is ``quote_id`` (Utf8), ``optimal_step`` (Int32),
        ``optimal_scenario_value`` (Float32), ``optimal_objective``
        (Float32), and one ``optimal_<name>`` (Float32) per constraint.
    lambdas : dict[str, float]
        Lagrange multipliers keyed by constraint name. Missing keys
        default to ``0.0``; extra keys are rejected (matches
        :class:`ApplyOptimiser`).
    constraints : dict[str, dict[str, float]]
        Constraint specifications (same format as :class:`ApplyOptimiser`).
        ``min_pct`` / ``max_pct`` thresholds are accepted but the
        threshold value is metadata only — the math depends on
        ``lambdas`` alone.
    chunk_size : int
        Target rows per IO slice. Rounded down to a multiple of
        ``n_steps``. Must be > 0 and >= ``n_steps``.
    quote_id, scenario_index, scenario_value, objective : str
        Column-name overrides (defaults match the rest of the API).
    n_steps : int, optional
        If known upfront, skips the auto-detection probe. When ``None``,
        ``n_steps`` is inferred from the first chunk by finding the first
        ``scenario_index`` reset to 0.

    Returns
    -------
    ChunkedApplyResult
        Aggregate result with ``.total_objective``,
        ``.total_constraints``, ``.baseline_objective``,
        ``.baseline_constraints``, ``.lambdas``, and ``.output_path``.

    Notes
    -----
    Output rows for each chunk are sorted by ``quote_id`` (the underlying
    builder sorts at ``build()`` time); across chunks, output appears in
    the order chunks were read from the input parquet. Sort the output
    parquet downstream if you need a globally-sorted result.

    On error, the partially-written ``parquet_out`` is best-effort
    deleted so callers never observe a half-written file. Cleanup only
    applies after this call has opened ``parquet_out`` for writing, so a
    failed read does not delete a pre-existing output file. This implies
    ``parquet_out`` must not be shared across concurrent calls — a
    failed call's cleanup would clobber a sibling's in-progress output.
    ``parquet_out`` must also be distinct from ``parquet_in``.
    """
    return _apply_lambdas_to_parquet_chunked_inner(
        parquet_in,
        parquet_out,
        lambdas,
        constraints,
        chunk_size,
        quote_id=quote_id,
        scenario_index=scenario_index,
        scenario_value=scenario_value,
        objective=objective,
        n_steps=n_steps,
    )


__all__ = [
    "__version__",
    "ApplyOptimiser",
    "apply_from_grid",
    "apply_lambdas_to_parquet_chunked",
    "ApplyResult",
    "ChunkedApplyResult",
    "FrontierResult",
    "FrontierResultLike",
    "GroupedSolveResult",
    "NonConvergenceReason",
    "OnlineOptimiser",
    "QuoteGrid",
    "QuoteGridBuilder",
    "RatebookOptimiser",
    "RatebookResult",
    "SolverPath",
    "SolveResult",
    "build_grid_from_parquet",
    "build_grid_from_parquet_chunked",
    "frontier_summary",
]
