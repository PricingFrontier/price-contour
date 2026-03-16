"""ApplyOptimiser — apply stored lambdas to new data (single-pass, no iteration)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from price_contour._price_contour import (
    ApplyResult,
    apply_from_grid_py,
    apply_lambdas_py,
)
from price_contour.builder import QuoteGrid


class ApplyOptimiser:
    """Apply pre-computed Lagrange multipliers to a scored DataFrame.

    This performs a single forward pass — no iteration, no lambda updates.
    Each quote picks the step that maximises the Lagrangian with fixed lambdas.

    Parameters
    ----------
    lambdas : dict[str, float]
        Lagrange multipliers keyed by constraint name.
    objective : str
        Objective column name.
    constraints : dict[str, dict[str, float]]
        Constraint specifications (same format as OnlineOptimiser).
    quote_id, scenario_index, scenario_value : str
        Column name overrides.
    chunk_size : int
        Quotes per parallel chunk.
    """

    def __init__(
        self,
        lambdas: dict[str, float],
        objective: str = "expected_income",
        constraints: dict[str, dict[str, float]] | None = None,
        *,
        quote_id: str = "quote_id",
        scenario_index: str = "scenario_index",
        scenario_value: str = "scenario_value",
        chunk_size: int = 500_000,
    ) -> None:
        self.lambdas = lambdas
        self.objective = objective
        self.constraints = constraints or {}
        self.quote_id = quote_id
        self.scenario_index = scenario_index
        self.scenario_value = scenario_value
        self.chunk_size = chunk_size

    def apply(self, df: pl.DataFrame) -> ApplyResult:
        """Apply lambdas to the scored DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Long-format scored DataFrame (same schema as solve input).

        Returns
        -------
        ApplyResult
            Result with .total_objective, .total_constraints,
            .baseline_objective, .baseline_constraints, .lambdas, .dataframe.
        """
        return apply_lambdas_py(
            df,
            lambdas=self.lambdas,
            quote_id=self.quote_id,
            scenario_index=self.scenario_index,
            scenario_value=self.scenario_value,
            objective=self.objective,
            constraints=self.constraints,
            chunk_size=self.chunk_size,
        )

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file.

        Parameters
        ----------
        path : str | Path
            File path to write config.json.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "lambdas": self.lambdas,
            "objective": self.objective,
            "constraints": self.constraints,
            "quote_id": self.quote_id,
            "scenario_index": self.scenario_index,
            "scenario_value": self.scenario_value,
            "chunk_size": self.chunk_size,
        }
        path.write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ApplyOptimiser:
        """Load configuration from a JSON file.

        Parameters
        ----------
        path : str | Path
            File path to read config.json.

        Returns
        -------
        ApplyOptimiser
            Configured ApplyOptimiser instance.
        """
        path = Path(path)
        config = json.loads(path.read_text())
        return cls(
            lambdas=config["lambdas"],
            objective=config.get("objective", "expected_income"),
            constraints=config.get("constraints"),
            quote_id=config.get("quote_id", "quote_id"),
            scenario_index=config.get("scenario_index", "scenario_index"),
            scenario_value=config.get("scenario_value", "scenario_value"),
            chunk_size=config.get("chunk_size", 500_000),
        )


def apply_from_grid(
    grid: QuoteGrid,
    lambdas: dict[str, float],
    constraints: dict[str, dict[str, float]],
    chunk_size: int = 500_000,
) -> ApplyResult:
    """Single-pass Lagrangian apply on an existing QuoteGrid.

    This performs the same argmax as ``ApplyOptimiser.apply()``, but
    operates directly on an in-memory QuoteGrid — no DataFrame
    re-ingestion. Useful when the grid is already available (e.g. after
    a ``solve()`` or ``frontier()`` call) and you want O(N) evaluation
    at specific lambdas with no iteration overhead.

    Parameters
    ----------
    grid : QuoteGrid
        Pre-built QuoteGrid (e.g. from ``SolveResult.grid`` or a builder).
    lambdas : dict[str, float]
        Fixed Lagrange multipliers keyed by constraint name.
    constraints : dict[str, dict[str, float]]
        Constraint specifications (same format as ``OnlineOptimiser``).
    chunk_size : int
        Quotes per parallel chunk.

    Returns
    -------
    ApplyResult
        Result with ``.total_objective``, ``.total_constraints``,
        ``.baseline_objective``, ``.baseline_constraints``, ``.lambdas``,
        ``.dataframe``.
    """
    return apply_from_grid_py(grid, lambdas, constraints, chunk_size)
