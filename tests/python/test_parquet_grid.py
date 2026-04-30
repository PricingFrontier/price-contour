"""Tests for build_grid_from_parquet — Rust-native parquet → QuoteGrid."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from price_contour import build_grid_from_parquet
from helpers import make_small_df


class TestBuildGridFromParquet:
    """build_grid_from_parquet produces the same QuoteGrid as the builder path."""

    def test_matches_builder(self, tmp_path: Path):
        """Parquet path produces identical grid to QuoteGridBuilder path."""
        df = make_small_df(n_quotes=50, n_steps=5)

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
        df = make_small_df(n_quotes=50, n_steps=5)

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
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
        df = make_small_df(n_quotes=20, n_steps=5)
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
        df = make_small_df(n_quotes=10, n_steps=5)
        df = df.rename(
            {
                "quote_id": "policy_id",
                "scenario_index": "step",
                "scenario_value": "price_factor",
                "expected_income": "revenue",
            }
        )
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
        df = make_small_df(n_quotes=5, n_steps=3)
        df = df.with_columns(pl.col("scenario_value").cast(pl.Utf8))
        pq_path = str(tmp_path / "wrong_types.parquet")
        df.write_parquet(pq_path)
        with pytest.raises(ValueError, match="must be Float32"):
            build_grid_from_parquet(pq_path, ["volume"])

    def test_missing_columns_raises(self, tmp_path: Path):
        """Parquet with missing required columns raises ValueError."""
        df = make_small_df(n_quotes=5, n_steps=3).drop("expected_income")
        pq_path = str(tmp_path / "missing_cols.parquet")
        df.write_parquet(pq_path)
        # Column projection is now done up front by Polars, so the error names
        # the missing column and lists valid ones — strictly more informative
        # than the old "Missing column: ..." string from ingest_dataframe.
        with pytest.raises(
            ValueError, match=r"(?i)expected_income|missing|not found"
        ):
            build_grid_from_parquet(pq_path, ["volume"])

    def test_sink_parquet_then_read(self, tmp_path: Path):
        """Simulates the haute pipeline: lazy sink → Rust read."""
        df = make_small_df(n_quotes=30, n_steps=5)
        pq_path = str(tmp_path / "sunk.parquet")

        # Sink from lazy (like haute does)
        lf = df.lazy()
        lf.sink_parquet(pq_path)

        grid = build_grid_from_parquet(pq_path, ["volume"])
        assert grid.n_quotes == 30
        assert grid.n_steps == 5


class TestBuildGridFromParquetChunked:
    """Issue 2: chunked parquet → QuoteGrid path.

    Reads the parquet in `chunk_size`-row slices via Polars' ``with_slice``
    API rather than materialising the whole file. Memory peak should scale
    with `chunk_size`, not with the parquet's total row count.
    """

    def test_matches_oneshot(self, tmp_path: Path):
        """Chunked path produces an identical grid to the all-at-once path."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=50, n_steps=5)
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)

        oneshot = build_grid_from_parquet(pq_path, ["volume"])
        chunked = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=37
        )

        assert chunked.n_quotes == oneshot.n_quotes
        assert chunked.n_steps == oneshot.n_steps
        assert chunked.scenario_values == oneshot.scenario_values
        assert chunked.constraint_names == oneshot.constraint_names
        assert chunked.quote_ids == oneshot.quote_ids

    def test_chunk_size_smaller_than_one_quote_works(self, tmp_path: Path):
        """`chunk_size < n_steps` is unrepresentable; require >= n_steps."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=10, n_steps=5)
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)

        with pytest.raises(ValueError, match=r"(?i)chunk_size|n_steps"):
            build_grid_from_parquet_chunked(
                pq_path, ["volume"], chunk_size=3
            )

    def test_chunk_size_exactly_one_quote(self, tmp_path: Path):
        """`chunk_size == n_steps` reads one quote per IO."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=20, n_steps=5)
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)

        oneshot = build_grid_from_parquet(pq_path, ["volume"])
        chunked = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=5
        )
        assert chunked.quote_ids == oneshot.quote_ids
        assert chunked.scenario_values == oneshot.scenario_values

    def test_chunk_size_larger_than_file(self, tmp_path: Path):
        """`chunk_size > total_rows` reads everything in one go."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=4, n_steps=5)
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)

        oneshot = build_grid_from_parquet(pq_path, ["volume"])
        chunked = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=10_000
        )
        assert chunked.quote_ids == oneshot.quote_ids

    def test_solver_result_matches_oneshot(self, tmp_path: Path):
        """End-to-end: solving from chunked-built grid matches one-shot."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=40, n_steps=5)
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result_oneshot = solver.solve(build_grid_from_parquet(pq_path, ["volume"]))
        result_chunked = solver.solve(
            build_grid_from_parquet_chunked(pq_path, ["volume"], chunk_size=29)
        )
        # Same data → solver determinism → identical totals.
        assert result_chunked.total_objective == result_oneshot.total_objective

    def test_chunk_size_zero_rejected(self, tmp_path: Path):
        """`chunk_size = 0` rejected at API."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=2, n_steps=5)
        pq_path = str(tmp_path / "test.parquet")
        df.write_parquet(pq_path)
        with pytest.raises(ValueError, match=r"(?i)chunk_size"):
            build_grid_from_parquet_chunked(
                pq_path, ["volume"], chunk_size=0
            )

    def test_missing_file_raises(self):
        """Non-existent file raises ValueError."""
        from price_contour import build_grid_from_parquet_chunked

        with pytest.raises(ValueError, match="Failed to open"):
            build_grid_from_parquet_chunked(
                "/no/such/file.parquet", ["volume"], chunk_size=100
            )

    def test_corrupted_parquet_raises(self, tmp_path: Path):
        """Corrupted parquet raises ValueError."""
        from price_contour import build_grid_from_parquet_chunked

        bad_path = str(tmp_path / "corrupt.parquet")
        Path(bad_path).write_bytes(b"not parquet at all")
        with pytest.raises(ValueError, match=r"(?i)parquet|read"):
            build_grid_from_parquet_chunked(
                bad_path, ["volume"], chunk_size=100
            )

    def test_empty_parquet_raises(self, tmp_path: Path):
        """Zero-row parquet raises a clear error."""
        from price_contour import build_grid_from_parquet_chunked

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
        with pytest.raises(ValueError, match=r"(?i)empty|no rows"):
            build_grid_from_parquet_chunked(
                pq_path, ["volume"], chunk_size=100
            )

    def test_missing_columns_raises(self, tmp_path: Path):
        """Parquet missing a required column raises a clear error."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=5, n_steps=3).drop("expected_income")
        pq_path = str(tmp_path / "missing.parquet")
        df.write_parquet(pq_path)
        with pytest.raises(ValueError, match=r"(?i)missing|column"):
            build_grid_from_parquet_chunked(
                pq_path, ["volume"], chunk_size=100
            )

    def test_total_rows_not_divisible_by_n_steps_raises(self, tmp_path: Path):
        """If the parquet's total rows don't divide n_steps, error early."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=4, n_steps=5)
        # Drop one row to make total = 19, which is not a multiple of n_steps=5.
        df = df.head(19)
        pq_path = str(tmp_path / "ragged.parquet")
        df.write_parquet(pq_path)
        with pytest.raises(ValueError, match=r"(?i)divisible|n_steps"):
            build_grid_from_parquet_chunked(
                pq_path, ["volume"], chunk_size=10
            )

    def test_multi_row_group_parquet(self, tmp_path: Path):
        """A parquet with multiple row groups still produces the canonical grid."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=60, n_steps=5)
        pq_path = str(tmp_path / "multi_rg.parquet")
        # Force multiple row groups by writing with a small row group size.
        df.write_parquet(pq_path, row_group_size=50)

        oneshot = build_grid_from_parquet(pq_path, ["volume"])
        chunked = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=43
        )
        assert chunked.quote_ids == oneshot.quote_ids
        assert chunked.scenario_values == oneshot.scenario_values

    def test_unsorted_parquet_still_sorted_by_build(self, tmp_path: Path):
        """Builder-time sort handles parquet whose quote_ids aren't globally sorted."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=30, n_steps=5)
        # Reverse the quote order — within each quote, scenario_index still runs
        # 0..n_steps so the per-chunk contract holds.
        reversed_quotes = pl.concat(
            [df.filter(pl.col("quote_id") == f"Q{q:04d}") for q in reversed(range(30))]
        )
        pq_path = str(tmp_path / "reversed.parquet")
        reversed_quotes.write_parquet(pq_path)

        chunked = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=37
        )
        # build()-time sort restores canonical order.
        assert chunked.quote_ids == [f"Q{q:04d}" for q in range(30)]

    def test_large_chunk_count(self, tmp_path: Path):
        """Many small chunks (~50 chunks) produce the same grid as one chunk."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=100, n_steps=5)
        pq_path = str(tmp_path / "many.parquet")
        df.write_parquet(pq_path)

        oneshot = build_grid_from_parquet(pq_path, ["volume"])
        # chunk_size=10 → 50 chunks total
        chunked = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=10
        )
        assert chunked.quote_ids == oneshot.quote_ids
        assert chunked.scenario_values == oneshot.scenario_values

    def test_explicit_n_steps_skips_autodetect(self, tmp_path: Path):
        """Passing `n_steps` explicitly bypasses auto-detection entirely."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=10, n_steps=5)
        pq_path = str(tmp_path / "explicit_ns.parquet")
        df.write_parquet(pq_path)

        # chunk_size=5 (= n_steps) would normally rely on the peek-row
        # confirmation; passing n_steps=5 explicitly skips the peek.
        grid = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=5, n_steps=5
        )
        assert grid.n_steps == 5
        assert grid.n_quotes == 10

    def test_explicit_n_steps_inconsistent_with_data_raises(self, tmp_path: Path):
        """An explicit n_steps that doesn't match the parquet's layout must error."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=4, n_steps=5)
        pq_path = str(tmp_path / "wrong_ns.parquet")
        df.write_parquet(pq_path)

        # Real n_steps is 5; user claims 3.
        with pytest.raises(
            ValueError,
            match=r"(?i)n_steps|divisible|scenario_index|contiguous",
        ):
            build_grid_from_parquet_chunked(
                pq_path, ["volume"], chunk_size=15, n_steps=3
            )

    def test_extra_columns_in_parquet_ignored(self, tmp_path: Path):
        """Columns outside the projection are pruned (not loaded into memory)."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=10, n_steps=5).with_columns(
            pl.lit(0.0, dtype=pl.Float32).alias("unused_a"),
            pl.lit("noise").alias("unused_b"),
        )
        pq_path = str(tmp_path / "extras.parquet")
        df.write_parquet(pq_path)

        grid = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=20
        )
        assert grid.n_quotes == 10
        # Only the requested constraint should be loaded.
        assert grid.constraint_names == ["volume"]

    def test_grid_invariant_under_chunk_size(self, tmp_path: Path):
        """Same parquet, different chunk_sizes → identical grids."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=40, n_steps=5)
        pq_path = str(tmp_path / "invariant.parquet")
        df.write_parquet(pq_path)

        grids = [
            build_grid_from_parquet_chunked(
                pq_path, ["volume"], chunk_size=cs
            )
            for cs in (5, 13, 37, 100, 10_000)
        ]
        ref = grids[0]
        for g in grids[1:]:
            assert g.quote_ids == ref.quote_ids
            assert g.scenario_values == ref.scenario_values
            assert g.n_quotes == ref.n_quotes

    def test_row_groups_cutting_mid_quote(self, tmp_path: Path):
        """Row group boundaries that cut mid-quote still produce the canonical grid."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=20, n_steps=5)  # 100 rows total
        pq_path = str(tmp_path / "mid_quote_rg.parquet")
        # 37 doesn't divide n_steps=5 → most row groups cut mid-quote.
        df.write_parquet(pq_path, row_group_size=37)

        oneshot = build_grid_from_parquet(pq_path, ["volume"])
        chunked = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=29
        )
        assert chunked.quote_ids == oneshot.quote_ids
        assert chunked.scenario_values == oneshot.scenario_values

    def test_single_quote_file_autodetects(self, tmp_path: Path):
        """A parquet containing one complete quote auto-detects n_steps via the whole-file branch."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=1, n_steps=5)
        pq_path = str(tmp_path / "single.parquet")
        df.write_parquet(pq_path)

        grid = build_grid_from_parquet_chunked(
            pq_path, ["volume"], chunk_size=100
        )
        assert grid.n_quotes == 1
        assert grid.n_steps == 5

    def test_duplicate_constraint_columns_raises(self, tmp_path: Path):
        """Listing the same constraint twice must error before reaching Polars."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=5, n_steps=5)
        pq_path = str(tmp_path / "dup.parquet")
        df.write_parquet(pq_path)

        with pytest.raises(ValueError, match=r"(?i)constraint|duplicate|listed more"):
            build_grid_from_parquet_chunked(
                pq_path, ["volume", "volume"], chunk_size=20
            )

    def test_constraint_column_collides_with_schema_column_raises(
        self, tmp_path: Path
    ):
        """A constraint named like a schema column must error early."""
        from price_contour import build_grid_from_parquet_chunked

        df = make_small_df(n_quotes=5, n_steps=5)
        pq_path = str(tmp_path / "collision.parquet")
        df.write_parquet(pq_path)

        with pytest.raises(ValueError, match=r"(?i)collide|schema"):
            # "quote_id" overlaps with schema's quote_id column name.
            build_grid_from_parquet_chunked(
                pq_path, ["quote_id"], chunk_size=20
            )
