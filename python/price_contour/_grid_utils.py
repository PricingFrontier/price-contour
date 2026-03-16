"""Shared grid-building helper to avoid duplicating the QuoteGridBuilder pattern."""

from __future__ import annotations

import polars as pl

from price_contour._price_contour import QuoteGrid, QuoteGridBuilder


def build_grid(
    df: pl.DataFrame,
    *,
    constraint_columns: list[str],
    quote_id: str,
    scenario_index: str,
    scenario_value: str,
    objective: str,
) -> QuoteGrid:
    """Build a QuoteGrid from a scored DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format scored DataFrame.
    constraint_columns : list[str]
        Column names for constraints.
    quote_id, scenario_index, scenario_value, objective : str
        Column name mappings.

    Returns
    -------
    QuoteGrid
    """
    builder = QuoteGridBuilder(
        constraint_columns,
        quote_id=quote_id,
        scenario_index=scenario_index,
        scenario_value_col=scenario_value,
        objective=objective,
    )
    builder.append(df)
    return builder.build()
