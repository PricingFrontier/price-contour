"""
price_contour — High-performance insurance price optimisation.

Lagrangian dual decomposition for portfolio-level price optimisation,
with Rust core and Polars DataFrame interop.
"""

__version__ = "0.1.0"

from price_contour.apply import ApplyOptimiser, apply_from_grid
from price_contour.builder import QuoteGrid, QuoteGridBuilder
from price_contour.frontier import FrontierResult, FrontierResultLike, frontier_summary
from price_contour.ratebook import RatebookOptimiser, RatebookResult
from price_contour.solver import OnlineOptimiser
from price_contour._price_contour import (
    ApplyResult,
    GroupedSolveResult,
    SolveResult,
    build_grid_from_parquet_py as _build_grid_from_parquet_inner,
)


def build_grid_from_parquet(
    path: str,
    constraint_columns: list[str],
    *,
    quote_id: str = "quote_id",
    scenario_index: str = "scenario_index",
    scenario_value_col: str = "scenario_value",
    objective: str = "expected_income",
) -> QuoteGrid:
    """Build a QuoteGrid directly from a Parquet file.

    Note: This loads the entire Parquet file into memory at once. For very large
    files that may exceed available memory, use ``QuoteGridBuilder`` with chunked
    reading instead::

        builder = QuoteGridBuilder(...)
        for chunk in pl.read_parquet(path).iter_slices(chunk_size):
            builder.append(chunk)
        grid = builder.build()

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
    scenario_value_col : str
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
        scenario_value_col=scenario_value_col,
        objective=objective,
    )


__all__ = [
    "__version__",
    "ApplyOptimiser",
    "apply_from_grid",
    "ApplyResult",
    "FrontierResult",
    "FrontierResultLike",
    "GroupedSolveResult",
    "OnlineOptimiser",
    "QuoteGrid",
    "QuoteGridBuilder",
    "RatebookOptimiser",
    "RatebookResult",
    "SolveResult",
    "build_grid_from_parquet",
    "frontier_summary",
]
