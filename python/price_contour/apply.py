"""ApplyOptimiser — apply stored lambdas to new data (single-pass, no iteration)."""

from __future__ import annotations

import polars as pl

from price_contour._price_contour import ApplyResult, apply_lambdas_py


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
    quote_id, scenario_step, multiplier : str
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
        scenario_step: str = "scenario_step",
        multiplier: str = "multiplier",
        chunk_size: int = 500_000,
    ) -> None:
        self.lambdas = lambdas
        self.objective = objective
        self.constraints = constraints or {}
        self.quote_id = quote_id
        self.scenario_step = scenario_step
        self.multiplier = multiplier
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
            scenario_step=self.scenario_step,
            multiplier=self.multiplier,
            objective=self.objective,
            constraints=self.constraints,
            chunk_size=self.chunk_size,
        )
