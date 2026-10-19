"""QuoteGridBuilder — chunked data ingestion for building QuoteGrid objects."""

from __future__ import annotations

import polars as pl

from price_contour._native_columns import select_native_columns
from price_contour._price_contour import QuoteGrid
from price_contour._price_contour import _QuoteGridBuilder

__all__ = ["QuoteGrid", "QuoteGridBuilder"]


class QuoteGridBuilder:
    """Incrementally build a QuoteGrid from DataFrame chunks.

    Holds the native builder and hands it, on each :meth:`append`, only the
    columns it reads: ``quote_id``, ``scenario_index``, ``scenario_value``,
    the objective and the constraint columns (DESIGN_DECISIONS §15.2).
    Every other column of a chunk is ignored, whatever its dtype.

    Each chunk's rows must already be grouped by ``quote_id`` (each quote
    occupies ``n_steps`` contiguous rows) and ordered by ``scenario_index``
    within each group; across chunks the order is free. ``n_steps``, if
    given, locks the step count; otherwise it is detected from the first
    chunk.
    """

    def __init__(
        self,
        constraint_columns: list[str],
        *,
        quote_id: str = "quote_id",
        scenario_index: str = "scenario_index",
        scenario_value: str = "scenario_value",
        objective: str = "expected_income",
        n_steps: int | None = None,
    ) -> None:
        self._inner = _QuoteGridBuilder(
            constraint_columns,
            quote_id=quote_id,
            scenario_index=scenario_index,
            scenario_value=scenario_value,
            objective=objective,
            n_steps=n_steps,
        )
        self._read_columns = [
            quote_id,
            scenario_index,
            scenario_value,
            objective,
            *constraint_columns,
        ]

    def append(self, df: pl.DataFrame) -> None:
        """Append a chunk of data (see the class docstring for its layout)."""
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")
        self._inner.append(select_native_columns(df, self._read_columns))

    def build(self) -> QuoteGrid:
        """Consume the builder and return a QuoteGrid sorted by ``quote_id``."""
        return self._inner.build()

    @property
    def n_quotes(self) -> int:
        """Quotes appended so far."""
        return self._inner.n_quotes
