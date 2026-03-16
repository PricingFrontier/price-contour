"""Shared pytest fixtures for price-contour tests."""

from __future__ import annotations

import polars as pl
import pytest

from helpers import make_small_df


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_df() -> pl.DataFrame:
    """50-quote, 5-step test DataFrame."""
    return make_small_df(n_quotes=50, n_steps=5)


@pytest.fixture(scope="session")
def small_df_100() -> pl.DataFrame:
    """100-quote, 5-step test DataFrame (more quotes for stable constraint tests)."""
    return make_small_df(n_quotes=100, n_steps=5)
