"""
price_contour — High-performance insurance price optimisation.

Lagrangian dual decomposition for portfolio-level price optimisation,
with Rust core and Polars DataFrame interop.
"""

__version__ = "0.1.0"

from price_contour.apply import ApplyOptimiser
from price_contour.solver import OnlineOptimiser
from price_contour._price_contour import ApplyResult, SolveResult

__all__ = [
    "__version__",
    "ApplyOptimiser",
    "ApplyResult",
    "OnlineOptimiser",
    "SolveResult",
]
