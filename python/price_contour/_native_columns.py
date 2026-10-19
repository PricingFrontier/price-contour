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
    missing with the message it has always given. The exception is a
    non-empty ``df`` with none of the named columns, which raises here:
    projecting it would leave a frame with no rows, which the native call
    would take for an empty input. Selecting existing columns copies no
    data.
    """
    named = list(dict.fromkeys(columns))
    present = set(df.columns)
    wanted = [c for c in named if c in present]
    if not wanted and df.height > 0:
        raise ValueError(
            f"Missing column(s): {', '.join(named)}. The DataFrame has none "
            f"of the columns this call reads (it has {df.columns})."
        )
    return df.select(wanted)
