"""Tests for QuoteGridBuilder and QuoteGrid."""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from price_contour._price_contour import QuoteGrid, QuoteGridBuilder
from helpers import make_small_df


class TestQuoteGridBuilder:
    """Tests for the QuoteGridBuilder."""

    def test_one_chunk_matches_oneshot(self):
        """Build grid from one chunk = same result as one-shot DataFrame path."""
        df = make_small_df(n_quotes=50, n_steps=5)

        # One-shot via DataFrame
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result_df = solver.solve(df)

        # Via builder (one chunk)
        builder = QuoteGridBuilder(["volume"])
        builder.append(df)
        grid = builder.build()

        result_grid = solver.solve(grid)

        assert result_grid.n_quotes == result_df.n_quotes
        assert result_grid.n_steps == result_df.n_steps
        assert abs(result_grid.total_objective - result_df.total_objective) < 1e-3

    def test_five_chunks_matches_oneshot(self):
        """Build grid from 5 chunks = same result as one-shot."""
        n_quotes = 50
        n_steps = 5
        df = make_small_df(n_quotes=n_quotes, n_steps=n_steps)

        # Split into 5 chunks of 10 quotes each
        chunk_size = 10
        builder = QuoteGridBuilder(["volume"])
        for start in range(0, n_quotes, chunk_size):
            end = start + chunk_size
            chunk_ids = [f"Q{q:04d}" for q in range(start, end)]
            chunk = df.filter(pl.col("quote_id").is_in(chunk_ids))
            builder.append(chunk)

        assert builder.n_quotes == n_quotes
        grid = builder.build()
        assert grid.n_quotes == n_quotes
        assert grid.n_steps == n_steps

        # Solve from grid
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result_grid = solver.solve(grid)

        # Solve from DataFrame
        result_df = solver.solve(df)

        # Totals should match (small float tolerance)
        assert abs(result_grid.total_objective - result_df.total_objective) < 1.0

    def test_append_after_build_raises(self):
        """Append after build() raises error."""
        df = make_small_df(n_quotes=10, n_steps=5)
        builder = QuoteGridBuilder(["volume"])
        builder.append(df)
        _grid = builder.build()

        # Builder is consumed; further append should fail
        with pytest.raises(ValueError):
            builder.append(df)

    def test_mismatched_constraint_count_raises(self):
        """Appending data with wrong constraint count raises."""
        # Builder expects ["volume", "loss_ratio"] but we only supply volume data
        # This is handled at the Rust level since we pass constraint_cols at init
        # The DataFrame must contain all constraint columns
        df = make_small_df(n_quotes=10, n_steps=5)
        builder = QuoteGridBuilder(["volume", "nonexistent_col"])
        with pytest.raises(ValueError):
            builder.append(df)

    def test_empty_builder_raises(self):
        """Build on empty builder raises."""
        builder = QuoteGridBuilder(["volume"])
        with pytest.raises(ValueError):
            builder.build()

    def test_grid_properties(self):
        """QuoteGrid exposes expected getters."""
        df = make_small_df(n_quotes=20, n_steps=5)
        builder = QuoteGridBuilder(["volume"])
        builder.append(df)
        grid = builder.build()

        assert grid.n_quotes == 20
        assert grid.n_steps == 5
        assert len(grid.scenario_values) == 5
        assert abs(grid.scenario_values[0] - 0.8) < 0.01
        assert grid.constraint_names == ["volume"]

    def test_grid_repr(self):
        """QuoteGrid __repr__ works."""
        df = make_small_df(n_quotes=10, n_steps=5)
        builder = QuoteGridBuilder(["volume"])
        builder.append(df)
        grid = builder.build()
        r = repr(grid)
        assert "QuoteGrid" in r
        assert "10" in r
        assert "5" in r, "n_steps should appear in repr"
        assert "volume" in r, "constraint name should appear in repr"
