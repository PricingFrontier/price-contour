"""Tests for the RatebookOptimiser (coordinate descent loop)."""

from __future__ import annotations

import math

import polars as pl
import pytest

from price_contour.ratebook import RatebookOptimiser, RatebookResult


def _make_small_df(n_quotes: int = 50, n_steps: int = 5) -> pl.DataFrame:
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


def _make_factors(n_quotes: int = 50) -> pl.DataFrame:
    """Build a factors DataFrame with two factors: region and age_band."""
    regions = ["North", "South", "East", "West"]
    age_bands = ["18-25", "26-35", "36-50", "51+"]
    return pl.DataFrame(
        {
            "region": [regions[i % len(regions)] for i in range(n_quotes)],
            "age_band": [age_bands[i % len(age_bands)] for i in range(n_quotes)],
        }
    )


class TestRatebook:
    def test_single_factor_cd(self):
        """Single factor CD = single grouped solve result."""
        n = 50
        df = _make_small_df(n_quotes=n)
        factors = _make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=100,
        )
        result = opt.solve(df, factors)

        assert isinstance(result, RatebookResult)
        assert "region" in result.factor_tables
        assert len(result.factor_tables["region"]) == 4  # 4 regions
        assert result.total_objective > 0
        assert len(result.per_factor_results) == 1

    def test_two_factor_cd_converges(self):
        """Two-factor CD converges on synthetic problem."""
        n = 50
        df = _make_small_df(n_quotes=n)
        factors = _make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=100,
        )
        result = opt.solve(df, factors)

        assert "region" in result.factor_tables
        assert "age_band" in result.factor_tables
        assert result.cd_iterations >= 1
        assert result.total_objective > 0

    def test_to_rating_entries_valid(self):
        """to_rating_entries() produces valid DataFrames."""
        n = 50
        df = _make_small_df(n_quotes=n)
        factors = _make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)
        entries = result.to_rating_entries()

        assert "region" in entries
        assert "age_band" in entries
        region_df = entries["region"]
        assert "region" in region_df.columns
        assert "factor" in region_df.columns
        assert region_df.shape[0] == 4  # 4 unique regions

    def test_clamp_rate_below_threshold(self):
        """Clamp rate should be low with a sufficiently wide grid."""
        n = 50
        df = _make_small_df(n_quotes=n)
        factors = _make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            factor_columns=[["region"]],
            candidate_min=0.80,
            candidate_max=1.20,
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)
        assert result.clamp_rate < 0.5, f"clamp rate too high: {result.clamp_rate}"

    def test_auto_discover_selects_factors(self):
        """Auto-discover selects factors from the factors DataFrame."""
        n = 50
        df = _make_small_df(n_quotes=n)
        factors = _make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)

        # Should have discovered at least one factor
        assert len(result.factor_tables) >= 1

    def test_summary_structure(self):
        """summary() produces valid params/metrics/artifacts."""
        n = 50
        df = _make_small_df(n_quotes=n)
        factors = _make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)
        s = opt.summary(result)

        assert set(s.keys()) == {"params", "metrics", "artifacts"}
        assert isinstance(s["params"]["n_factors"], int)
        assert isinstance(s["metrics"]["total_objective"], float)
        assert isinstance(s["artifacts"]["factor_tables"], dict)
        assert isinstance(s["artifacts"]["rating_entries"], dict)
