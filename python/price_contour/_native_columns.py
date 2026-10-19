"""Project a caller's frame onto the columns a native call reads.

The bridge into the native module converts every column of a frame it is
handed, so a column the call never reads still costs a conversion, and one
the extension's Polars cannot import (Polars 2's ``Map``, or a ``Struct``,
``Decimal`` or ``Array`` column) fails the call. Every Python entry point
that hands a caller's frame to the native module projects it here first
(DESIGN_DECISIONS §15.2).
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl


def select_native_columns(df: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    """Return ``df`` restricted to ``columns``, in the order named.

    Duplicate names are kept once. A named column that ``df`` lacks is
    left out rather than raising here: the native call then reports it
    missing with the message it has always given, so the projection never
    changes or hides an error. Selecting existing columns copies no data.
    """
    present = set(df.columns)
    wanted = [c for c in dict.fromkeys(columns) if c in present]
    return df.select(wanted)
