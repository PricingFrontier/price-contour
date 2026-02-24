"""Tests for build_grid_from_parquet — Rust-native parquet → QuoteGrid."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from price_contour import build_grid_from_parquet


def _make_small_df(n_quotes: int = 50, n_steps: int = 5) -> pl.DataFrame:
    """Build a small test DataFrame with known properties."""
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


class TestBuildGridFromParquet:
    """build_grid_from_parquet produces the same QuoteGrid as the builder path."""

    def test_matches_builder(self, tmp_path: Path):
        """Parquet path produces identical grid to QuoteGridBuilder path."""
        df = _make_small_df(n_quotes=50, n_steps=5)

        # Builder path
        builder = pc.QuoteGridBuilder(["volume"])
        builder.append(df)
        grid_builder = builder.build()

        # Parquet path
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)
        grid_parquet = build_grid_from_parquet(
            pq_path,
            ["volume"],
        )

        assert grid_parquet.n_quotes == grid_builder.n_quotes
        assert grid_parquet.n_steps == grid_builder.n_steps
        assert grid_parquet.scenario_values == grid_builder.scenario_values
        assert grid_parquet.constraint_names == grid_builder.constraint_names
        assert grid_parquet.quote_ids == grid_builder.quote_ids

    def test_solve_matches_dataframe_path(self, tmp_path: Path):
        """Solving from a parquet-built grid matches the DataFrame path."""
        df = _make_small_df(n_quotes=50, n_steps=5)

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )

        # DataFrame path
        result_df = solver.solve(df)

        # Parquet path
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)
        grid = build_grid_from_parquet(pq_path, ["volume"])
        result_pq = solver.solve(grid)

        assert result_pq.n_quotes == result_df.n_quotes
        # Both paths produce identical QuoteGrids; solver is deterministic
        assert result_pq.total_objective == result_df.total_objective

    def test_multiple_constraints(self, tmp_path: Path):
        """Works with multiple constraint columns."""
        df = _make_small_df(n_quotes=20, n_steps=5)
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)

        grid = build_grid_from_parquet(
            pq_path,
            ["volume", "loss_ratio"],
        )
        assert grid.n_quotes == 20
        assert grid.n_steps == 5
        assert grid.constraint_names == ["volume", "loss_ratio"]

    def test_custom_column_names(self, tmp_path: Path):
        """Works with custom column names."""
        df = _make_small_df(n_quotes=10, n_steps=5)
        df = df.rename({
            "quote_id": "policy_id",
            "scenario_index": "step",
            "scenario_value": "price_factor",
            "expected_income": "revenue",
        })
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)

        grid = build_grid_from_parquet(
            pq_path,
            ["volume"],
            quote_id="policy_id",
            scenario_index="step",
            scenario_value_col="price_factor",
            objective="revenue",
        )
        assert grid.n_quotes == 10
        assert grid.n_steps == 5

    def test_missing_file_raises(self):
        """Non-existent file raises ValueError."""
        with pytest.raises(ValueError, match="Failed to open"):
            build_grid_from_parquet("/no/such/file.parquet", ["volume"])

    def test_corrupted_parquet_raises(self, tmp_path: Path):
        """Corrupted/invalid parquet file raises ValueError."""
        bad_path = str(tmp_path / "corrupted.parquet")
        Path(bad_path).write_bytes(b"this is not a parquet file")
        with pytest.raises(ValueError, match="Failed to read parquet"):
            build_grid_from_parquet(bad_path, ["volume"])

    def test_empty_zero_row_parquet_raises(self, tmp_path: Path):
        """Parquet file with schema but zero rows raises ValueError."""
        df = pl.DataFrame(
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        pq_path = str(tmp_path / "empty.parquet")
        df.write_parquet(pq_path)
        with pytest.raises(ValueError, match="Empty DataFrame"):
            build_grid_from_parquet(pq_path, ["volume"])

    def test_wrong_column_types_raises(self, tmp_path: Path):
        """Parquet with wrong column types raises ValueError."""
        df = _make_small_df(n_quotes=5, n_steps=3)
        df = df.with_columns(pl.col("scenario_value").cast(pl.Utf8))
        pq_path = str(tmp_path / "wrong_types.parquet")
        df.write_parquet(pq_path)
        with pytest.raises(ValueError, match="must be Float32"):
            build_grid_from_parquet(pq_path, ["volume"])

    def test_missing_columns_raises(self, tmp_path: Path):
        """Parquet with missing required columns raises ValueError."""
        df = _make_small_df(n_quotes=5, n_steps=3).drop("expected_income")
        pq_path = str(tmp_path / "missing_cols.parquet")
        df.write_parquet(pq_path)
        with pytest.raises(ValueError, match="Missing column"):
            build_grid_from_parquet(pq_path, ["volume"])

    def test_sink_parquet_then_read(self, tmp_path: Path):
        """Simulates the haute pipeline: lazy sink → Rust read."""
        df = _make_small_df(n_quotes=30, n_steps=5)
        pq_path = str(tmp_path / "sunk.parquet")

        # Sink from lazy (like haute does)
        lf = df.lazy()
        lf.sink_parquet(pq_path)

        grid = build_grid_from_parquet(pq_path, ["volume"])
        assert grid.n_quotes == 30
        assert grid.n_steps == 5
