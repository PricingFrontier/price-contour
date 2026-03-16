"""
price_contour — High-performance insurance price optimisation.

Lagrangian dual decomposition for portfolio-level price optimisation,
with Rust core and Polars DataFrame interop.
"""

__version__ = "0.1.0"

from price_contour.apply import ApplyOptimiser, apply_from_grid
from price_contour.builder import QuoteGrid, QuoteGridBuilder
from price_contour.frontier import FrontierResult, frontier_summary
from price_contour.ratebook import RatebookOptimiser, RatebookResult
from price_contour.solver import OnlineOptimiser
from price_contour._price_contour import (
    ApplyResult,
    GroupedSolveResult,
    SolveResult,
    build_grid_from_parquet_py as build_grid_from_parquet,
)

__all__ = [
    "__version__",
    "ApplyOptimiser",
    "apply_from_grid",
    "ApplyResult",
    "FrontierResult",
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
