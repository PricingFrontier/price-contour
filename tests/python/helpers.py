"""Shared test helpers and constants for price-contour tests."""

from __future__ import annotations

import math

import polars as pl

# ---------------------------------------------------------------------------
# Tolerance constant — used by all constraint-satisfaction assertions
# ---------------------------------------------------------------------------

CONSTRAINT_RTOL = 0.02  # 2% relative tolerance for constraint checks


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def make_small_df(n_quotes: int = 50, n_steps: int = 5) -> pl.DataFrame:
    """Build a small test DataFrame with known properties.

    Each quote has a logistic conversion curve parameterised by elasticity,
    producing a realistic objective/volume/loss_ratio tradeoff across
    scenario values (price multipliers).
    """
    rows = []
    mults = [0.8 + 0.1 * j for j in range(n_steps)]
    for q in range(n_quotes):
        elasticity = 1.5 + 3.5 * q / n_quotes
        base = 80.0 + 40.0 * q / n_quotes
        for j, mult in enumerate(mults):
            conversion = 1.0 / (1.0 + math.exp(elasticity * (mult - 1.0)))
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": mult,
                    "expected_income": base * mult * conversion,
                    "volume": conversion,
                    "loss_ratio": 0.6 / mult * (1.0 + 0.1 * (mult - 1.0)),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "quote_id": pl.Utf8,
            "scenario_index": pl.Int32,
            "scenario_value": pl.Float32,
            "expected_income": pl.Float32,
            "volume": pl.Float32,
            "loss_ratio": pl.Float32,
        },
    )


def make_factors(n_quotes: int = 50) -> pl.DataFrame:
    """Build a factors DataFrame with two factors: region and age_band."""
    regions = ["North", "South", "East", "West"]
    age_bands = ["18-25", "26-35", "36-50", "51+"]
    return pl.DataFrame(
        {
            "region": [regions[i % len(regions)] for i in range(n_quotes)],
            "age_band": [age_bands[i % len(age_bands)] for i in range(n_quotes)],
        }
    )
