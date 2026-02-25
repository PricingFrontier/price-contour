"""Shared pytest fixtures for price-contour tests."""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
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


@pytest.fixture(scope="session")
def solved_result(small_df_100):
    """Constrained solve result for reuse across tests.

    Returns (solver, result, df) tuple.
    """
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": 0.90}},
        max_iter=200,
    )
    result = solver.solve(small_df_100)
    return solver, result, small_df_100
