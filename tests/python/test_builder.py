"""Tests for QuoteGridBuilder and QuoteGrid."""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from price_contour._price_contour import QuoteGridBuilder
from helpers import make_small_df


class TestQuoteGridBuilder:
    """Tests for the QuoteGridBuilder."""

    def test_one_chunk_matches_oneshot(self):
        """Build grid from one chunk = same result as one-shot DataFrame path."""
        df = make_small_df(n_quotes=50, n_steps=5)

        # One-shot via DataFrame
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
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
            constraints={"volume": {"min_pct": 0.90}},
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


class TestQuoteGridBuilderUnsortedChunks:
    """Issue 1: build()-time sort.

    Upstream callers can't always materialise the whole DataFrame to sort it
    globally, so they feed chunks that are individually well-formed but
    cross-chunk-unordered. The builder must accept those chunks and produce
    a globally sorted QuoteGrid at build() time.
    """

    def test_reverse_order_chunks_produce_sorted_grid(self):
        """Quotes appended back-to-front still yield a grid sorted by quote_id."""
        df = make_small_df(n_quotes=20, n_steps=5)
        # Split the DataFrame into per-quote chunks and append in reverse.
        chunks = [
            df.filter(pl.col("quote_id") == f"Q{q:04d}") for q in range(20)
        ]
        builder = QuoteGridBuilder(["volume"])
        for chunk in reversed(chunks):
            builder.append(chunk)
        grid = builder.build()

        # Solve, compare to canonical (ordered) build.
        oneshot_builder = QuoteGridBuilder(["volume"])
        oneshot_builder.append(df)
        canonical_grid = oneshot_builder.build()

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result_unsorted = solver.solve(grid)
        result_canonical = solver.solve(canonical_grid)
        # Same data, same solve — totals must match exactly.
        assert (
            abs(result_unsorted.total_objective - result_canonical.total_objective)
            < 1e-3
        )

    def test_interleaved_single_quote_chunks_produce_sorted_grid(self):
        """Many tiny chunks in a scrambled order still yield the canonical grid."""
        df = make_small_df(n_quotes=15, n_steps=5)
        order = [7, 2, 11, 0, 14, 4, 9, 1, 13, 5, 8, 3, 12, 6, 10]
        builder = QuoteGridBuilder(["volume"])
        for q in order:
            builder.append(df.filter(pl.col("quote_id") == f"Q{q:04d}"))
        grid = builder.build()

        # Compare to a canonical one-shot build solve.
        oneshot_builder = QuoteGridBuilder(["volume"])
        oneshot_builder.append(df)
        canonical_grid = oneshot_builder.build()

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        r_a = solver.solve(grid)
        r_b = solver.solve(canonical_grid)
        assert abs(r_a.total_objective - r_b.total_objective) < 1e-3

    def test_duplicate_quote_id_across_chunks_raises(self):
        """Same quote_id in two chunks must surface a clear error at build()."""
        df = make_small_df(n_quotes=10, n_steps=5)
        builder = QuoteGridBuilder(["volume"])
        builder.append(df.filter(pl.col("quote_id") == "Q0003"))
        builder.append(df.filter(pl.col("quote_id") == "Q0003"))
        with pytest.raises(ValueError, match=r"(?i)duplicate|Q0003"):
            builder.build()

    def test_within_chunk_scenario_index_must_be_ordered(self):
        """Each quote's rows in a chunk must appear in scenario_index order."""
        df = make_small_df(n_quotes=2, n_steps=5)
        # Reverse the rows of Q0000 so its scenario_index goes 4..0.
        q0 = df.filter(pl.col("quote_id") == "Q0000").reverse()
        q1 = df.filter(pl.col("quote_id") == "Q0001")
        bad_chunk = pl.concat([q0, q1])

        builder = QuoteGridBuilder(["volume"])
        with pytest.raises(ValueError, match=r"(?i)scenario_index"):
            builder.append(bad_chunk)

    def test_within_chunk_quote_rows_must_be_contiguous(self):
        """A chunk that interleaves rows of different quotes must be rejected."""
        df = make_small_df(n_quotes=3, n_steps=5)
        # Sort by scenario_index then quote_id — this interleaves quotes.
        scrambled = df.sort(["scenario_index", "quote_id"])

        builder = QuoteGridBuilder(["volume"])
        with pytest.raises(
            ValueError, match=r"(?i)contiguous|n_steps|scenario_index"
        ):
            builder.append(scrambled)

    def test_empty_chunk_is_noop(self):
        """Appending an empty chunk leaves the builder state unchanged."""
        df = make_small_df(n_quotes=10, n_steps=5)
        builder = QuoteGridBuilder(["volume"])
        empty = df.head(0)
        # Empty chunk before any data: must not crash, must not "initialise" the
        # builder with garbage scenario_values.
        builder.append(empty)
        assert builder.n_quotes == 0
        # Real chunk in the middle.
        builder.append(df.filter(pl.col("quote_id") == "Q0000"))
        assert builder.n_quotes == 1
        # Empty chunk after data: also a no-op.
        builder.append(empty)
        assert builder.n_quotes == 1
        # Continuing to append real data must still work.
        builder.append(df.filter(pl.col("quote_id") == "Q0001"))
        assert builder.n_quotes == 2

    def test_scenario_value_mismatch_across_chunks_raises(self):
        """Chunks with divergent scenario grids must surface a clear error."""
        df = make_small_df(n_quotes=4, n_steps=5)
        good = df.filter(pl.col("quote_id") == "Q0000")
        # Build a "bad" chunk where Q0001 has a corrupted scenario_value at step 2.
        bad = df.filter(pl.col("quote_id") == "Q0001").with_columns(
            pl.when(pl.col("scenario_index") == 2)
            .then(pl.lit(99.9, dtype=pl.Float32))
            .otherwise(pl.col("scenario_value"))
            .alias("scenario_value")
        )
        builder = QuoteGridBuilder(["volume"])
        builder.append(good)
        with pytest.raises(ValueError, match=r"(?i)scenario_value|scenario grid"):
            builder.append(bad)

    def test_nan_scenario_value_in_subsequent_chunk_raises(self):
        """A NaN scenario_value at row > 0 must be rejected.

        ``(NaN - x).abs() > tol`` is false for any tolerance, so without an
        explicit finiteness check this would silently slip past the chunk
        consistency comparison.
        """
        df = make_small_df(n_quotes=4, n_steps=5)
        good = df.filter(pl.col("quote_id") == "Q0000")
        bad = df.filter(pl.col("quote_id") == "Q0001").with_columns(
            pl.when(pl.col("scenario_index") == 3)
            .then(pl.lit(float("nan"), dtype=pl.Float32))
            .otherwise(pl.col("scenario_value"))
            .alias("scenario_value")
        )
        builder = QuoteGridBuilder(["volume"])
        builder.append(good)
        with pytest.raises(ValueError, match=r"(?i)scenario_value|nan|grid"):
            builder.append(bad)

    def test_n_steps_kwarg_locks_contract(self):
        """Passing n_steps to __init__ skips auto-detection — useful for streaming."""
        df = make_small_df(n_quotes=10, n_steps=5)
        builder = QuoteGridBuilder(["volume"], n_steps=5)
        builder.append(df)
        grid = builder.build()
        assert grid.n_steps == 5
        assert grid.n_quotes == 10

    def test_n_steps_kwarg_zero_rejected(self):
        """n_steps=0 must be rejected at construction."""
        with pytest.raises(ValueError, match=r"(?i)n_steps"):
            QuoteGridBuilder(["volume"], n_steps=0)

    def test_n_steps_kwarg_mismatch_with_chunk_layout_raises(self):
        """If n_steps is locked, a chunk that doesn't conform must error."""
        df = make_small_df(n_quotes=2, n_steps=5)
        # Tell the builder n_steps=3 but feed it a 5-step DataFrame.
        builder = QuoteGridBuilder(["volume"], n_steps=3)
        with pytest.raises(
            ValueError, match=r"(?i)n_steps|scenario_index|contiguous"
        ):
            builder.append(df)

    def test_scenario_values_not_permuted_by_sort(self):
        """The single-grid scenario_values vector survives the build()-time sort."""
        df = make_small_df(n_quotes=5, n_steps=5)
        builder = QuoteGridBuilder(["volume"])
        # Append in reverse to force a non-trivial sort.
        for q in reversed(range(5)):
            builder.append(df.filter(pl.col("quote_id") == f"Q{q:04d}"))
        grid = builder.build()
        # scenario_values must remain in step-index order (0.8, 0.9, 1.0, 1.1, 1.2).
        sv = list(grid.scenario_values)
        assert sv == sorted(sv), f"scenario_values reordered by sort: {sv}"
        assert abs(sv[0] - 0.8) < 1e-5
        assert abs(sv[-1] - 1.2) < 1e-5

    def test_solver_result_unaffected_by_chunk_order(self):
        """End-to-end: the final solve must be invariant under chunk permutation."""
        df = make_small_df(n_quotes=30, n_steps=5)

        def solve_with_order(order: list[int]) -> float:
            b = QuoteGridBuilder(["volume"])
            for q in order:
                b.append(df.filter(pl.col("quote_id") == f"Q{q:04d}"))
            grid = b.build()
            solver = pc.OnlineOptimiser(
                objective="expected_income",
                constraints={"volume": {"min_pct": 0.90}},
                max_iter=200,
            )
            return solver.solve(grid).total_objective

        forward = solve_with_order(list(range(30)))
        reverse = solve_with_order(list(reversed(range(30))))
        scrambled = solve_with_order([13, 0, 27, 5, 19, 8, 22, 1, 11, 28, 3, 17, 9,
                                       15, 25, 6, 21, 12, 4, 23, 16, 2, 29, 18, 7,
                                       20, 10, 26, 14, 24])
        assert abs(forward - reverse) < 1e-3
        assert abs(forward - scrambled) < 1e-3
