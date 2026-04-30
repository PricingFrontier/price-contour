"""Type stubs for the _price_contour Rust extension module."""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# QuoteGrid / QuoteGridBuilder
# ---------------------------------------------------------------------------

class QuoteGrid:
    """Opaque handle to a validated quote grid stored in Rust memory."""

    @property
    def n_quotes(self) -> int: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def scenario_values(self) -> list[float]: ...
    @property
    def constraint_names(self) -> list[str]: ...
    @property
    def quote_ids(self) -> list[str]: ...

class QuoteGridBuilder:
    """Incrementally build a QuoteGrid from DataFrame chunks."""

    def __init__(
        self,
        constraint_columns: list[str],
        *,
        quote_id: str = "quote_id",
        scenario_index: str = "scenario_index",
        scenario_value_col: str = "scenario_value",
        objective: str = "expected_income",
    ) -> None: ...
    def append(self, df: pl.DataFrame) -> None: ...
    def build(self) -> QuoteGrid: ...
    @property
    def n_quotes(self) -> int: ...

# ---------------------------------------------------------------------------
# SolveResult
# ---------------------------------------------------------------------------

class SolveResult:
    """Result of the online Lagrangian solver."""

    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def lambdas(self) -> dict[str, float]: ...
    @property
    def total_objective(self) -> float: ...
    @property
    def total_constraints(self) -> dict[str, float]: ...
    @property
    def baseline_objective(self) -> float: ...
    @property
    def baseline_constraints(self) -> dict[str, float]: ...
    @property
    def dataframe(self) -> pl.DataFrame: ...
    @property
    def history(self) -> list[dict[str, object]] | None: ...
    @property
    def n_quotes(self) -> int: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def scenario_values(self) -> list[float]: ...
    @property
    def grid(self) -> QuoteGrid: ...

# ---------------------------------------------------------------------------
# ApplyResult
# ---------------------------------------------------------------------------

class ApplyResult:
    """Result of applying fixed lambdas (single forward pass)."""

    @property
    def lambdas(self) -> dict[str, float]: ...
    @property
    def total_objective(self) -> float: ...
    @property
    def total_constraints(self) -> dict[str, float]: ...
    @property
    def baseline_objective(self) -> float: ...
    @property
    def baseline_constraints(self) -> dict[str, float]: ...
    @property
    def dataframe(self) -> pl.DataFrame: ...

# ---------------------------------------------------------------------------
# GroupedSolveResult
# ---------------------------------------------------------------------------

class GroupedSolveResult:
    """Result of the grouped Lagrangian solver."""

    @property
    def optimal_factor_values(self) -> dict[str, float]: ...
    @property
    def optimal_steps_per_quote(self) -> list[int]: ...
    @property
    def lambdas(self) -> dict[str, float]: ...
    @property
    def iterations(self) -> int: ...
    @property
    def converged(self) -> bool: ...
    @property
    def total_objective(self) -> float: ...
    @property
    def total_constraints(self) -> dict[str, float]: ...
    @property
    def baseline_objective(self) -> float: ...
    @property
    def baseline_constraints(self) -> dict[str, float]: ...
    @property
    def clamp_rate(self) -> float: ...
    @property
    def group_labels(self) -> list[str]: ...
    @property
    def history(self) -> list[dict[str, object]] | None: ...
    @property
    def dataframe(self) -> pl.DataFrame: ...

# ---------------------------------------------------------------------------
# FrontierResult
# ---------------------------------------------------------------------------

class FrontierResult:
    """Result of a frontier sweep across constraint thresholds."""

    @property
    def points(self) -> pl.DataFrame: ...
    @property
    def n_points(self) -> int: ...
    @property
    def n_converged(self) -> int: ...
    @property
    def constraint_names(self) -> list[str]: ...

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def solve_online_py(
    df: pl.DataFrame,
    quote_id: str = "quote_id",
    scenario_index: str = "scenario_index",
    scenario_value: str = "scenario_value",
    objective: str = "expected_income",
    constraints: dict[str, dict[str, float]] | None = None,
    max_iter: int = 50,
    tolerance: float = 1e-5,
    lambdas: dict[str, float] | None = None,
    record_history: bool = False,
) -> SolveResult: ...
def solve_from_grid_py(
    grid: QuoteGrid,
    constraints: dict[str, dict[str, float]] | None = None,
    max_iter: int = 50,
    tolerance: float = 1e-5,
    lambdas: dict[str, float] | None = None,
    record_history: bool = False,
) -> SolveResult: ...
def apply_lambdas_py(
    df: pl.DataFrame,
    lambdas: dict[str, float],
    quote_id: str = "quote_id",
    scenario_index: str = "scenario_index",
    scenario_value: str = "scenario_value",
    objective: str = "expected_income",
    constraints: dict[str, dict[str, float]] | None = None,
) -> ApplyResult: ...
def apply_from_grid_py(
    grid: QuoteGrid,
    lambdas: dict[str, float],
    constraints: dict[str, dict[str, float]],
) -> ApplyResult: ...
def solve_grouped_py(
    grid: QuoteGrid,
    group_labels: list[str],
    residuals: list[float],
    candidates: list[float],
    constraints: dict[str, dict[str, float]] | None = None,
    max_iter: int = 50,
    tolerance: float = 1e-5,
    lambdas: dict[str, float] | None = None,
    record_history: bool = False,
) -> GroupedSolveResult: ...
def sweep_frontier_py(
    grid: QuoteGrid,
    constraints: dict[str, dict[str, float]],
    threshold_ranges: dict[str, tuple[float, float]],
    n_points_per_dim: int = 10,
    max_iter: int = 50,
    tolerance: float = 1e-5,
    initial_lambdas: dict[str, float] | None = None,
    max_total_points: int = 10_000,
    parallel: bool = False,
) -> FrontierResult: ...
def build_grid_from_parquet_py(
    path: str,
    constraint_columns: list[str],
    *,
    quote_id: str = "quote_id",
    scenario_index: str = "scenario_index",
    scenario_value_col: str = "scenario_value",
    objective: str = "expected_income",
) -> QuoteGrid: ...
