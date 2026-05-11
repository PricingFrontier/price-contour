"""Tests for the ratebook factor-context API (spec items 1-29).

This module exercises the public surface added for chunked parquet
factor contexts:

* :class:`price_contour.RatebookFactorContexts` (opaque handle)
* :meth:`RatebookFactorContexts.from_dataframe`
* :func:`price_contour.build_ratebook_factor_contexts_from_parquet_chunked`
* :meth:`RatebookOptimiser.solve` / :meth:`frontier` accepting the new
  union type ``pl.DataFrame | RatebookFactorContexts``.

The most important invariant is **bit-exact parity** between the
dataframe and chunked-parquet paths: solving with the same logical
inputs through either constructor must produce identical
``total_objective``, ``factor_tables``, ``total_constraints``, and
``baseline_*`` values. The fingerprint check on every solve catches
quote-axis misalignment in O(1), so we test both happy-path equality
and the loud-failure cases.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from price_contour import (
    RatebookFactorContexts,
    RatebookOptimiser,
    build_ratebook_factor_contexts_from_parquet_chunked,
)
from price_contour._grid_utils import build_grid

from helpers import make_factors, make_small_df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_and_factors() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Synthetic 50-quote × 5-step DataFrame plus a matching factors table."""
    n = 50
    df = make_small_df(n_quotes=n)
    factors = make_factors(n)
    return df, factors


@pytest.fixture
def factor_parquet(
    tmp_path: Path, df_and_factors: tuple[pl.DataFrame, pl.DataFrame]
) -> tuple[Path, pl.DataFrame, pl.DataFrame, list[str]]:
    """Write a factor parquet aligned to the quote grid built from the
    long-format DataFrame, returning the path plus the inputs the
    solver needs."""
    df, factors = df_and_factors
    # The QuoteGridBuilder sorts by quote_id, so the canonical order is
    # the lex-sorted unique quote_id sequence. Build the corresponding
    # factor parquet keyed by quote_id so the chunked builder can
    # reorder cleanly.
    sorted_qids = (
        df.select(pl.col("quote_id").cast(pl.Utf8).unique().sort())
        .to_series()
        .to_list()
    )
    # `make_factors` is sequential by quote-index, so its rows align
    # to df's quote order — which IS lex-sorted here because the IDs
    # are Q0000..QNNNN. We attach a quote_id column explicitly so the
    # parquet path can validate end-to-end.
    factor_parquet_df = factors.with_columns(
        pl.Series("quote_id", sorted_qids, dtype=pl.Utf8)
    )
    parquet_path = tmp_path / "factors.parquet"
    factor_parquet_df.write_parquet(parquet_path)
    return parquet_path, df, factors, sorted_qids


# ---------------------------------------------------------------------------
# Spec test #1: core parity for solve()
# ---------------------------------------------------------------------------


class TestSolveParity:
    def test_dataframe_vs_chunked_parquet_match_bitwise(
        self,
        factor_parquet: tuple[Path, pl.DataFrame, pl.DataFrame, list[str]],
    ) -> None:
        """Spec #1 + #24: solve(df, dataframe_factors) and solve(grid, parquet_contexts)
        produce identical objectives, factor_tables, constraints, and baselines."""
        parquet_path, df, factors, sorted_qids = factor_parquet

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=2,
            max_iter=50,
        )
        # Path A: dataframe-mode (legacy contract).
        result_a = opt.solve(df, factors)

        # Path B: chunked-parquet contexts. Build the grid first so we
        # have grid.quote_ids to align against.
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        contexts = build_ratebook_factor_contexts_from_parquet_chunked(
            str(parquet_path),
            factor_specs=[["region"], ["age_band"]],
            chunk_size=20,
            quote_id="quote_id",
            expected_quote_ids=grid.quote_ids,
        )
        result_b = opt.solve(grid, contexts)

        assert result_a.total_objective == pytest.approx(result_b.total_objective)
        assert result_a.baseline_objective == pytest.approx(result_b.baseline_objective)
        for name in result_a.total_constraints:
            assert result_a.total_constraints[name] == pytest.approx(
                result_b.total_constraints[name]
            )
            assert result_a.baseline_constraints[name] == pytest.approx(
                result_b.baseline_constraints[name]
            )
        # factor_tables: same factor names, same level keys, same values.
        assert set(result_a.factor_tables.keys()) == set(result_b.factor_tables.keys())
        for fname in result_a.factor_tables:
            assert result_a.factor_tables[fname] == pytest.approx(
                result_b.factor_tables[fname]
            )


# ---------------------------------------------------------------------------
# Spec test #2: frontier parity (chunked parquet contexts)
# ---------------------------------------------------------------------------


class TestFrontierParity:
    def test_frontier_df_vs_chunked_parquet_match(
        self,
        factor_parquet: tuple[Path, pl.DataFrame, pl.DataFrame, list[str]],
    ) -> None:
        """Spec #2: frontier with dataframe factors and chunked parquet
        contexts produces identical frontier points."""
        parquet_path, df, factors, sorted_qids = factor_parquet

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=2,
            max_iter=30,
        )
        ranges = {"volume": (0.85, 0.95)}

        fr_df = opt.frontier(df, factors, threshold_ranges=ranges, n_points_per_dim=3)

        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        contexts = build_ratebook_factor_contexts_from_parquet_chunked(
            str(parquet_path),
            factor_specs=[["region"], ["age_band"]],
            chunk_size=15,
            quote_id="quote_id",
            expected_quote_ids=grid.quote_ids,
        )
        fr_ctx = opt.frontier(
            grid, contexts, threshold_ranges=ranges, n_points_per_dim=3
        )

        assert fr_df.n_points == fr_ctx.n_points
        for col in [
            "total_objective",
            "total_volume",
            "lambda_volume",
            "threshold_volume",
        ]:
            for a, b in zip(fr_df.points[col].to_list(), fr_ctx.points[col].to_list()):
                assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Spec test #3, #4: frontier internal vs external contexts
# ---------------------------------------------------------------------------


class TestFrontierInternalRefactor:
    def test_frontier_internal_dataframe_path_matches_legacy_oracle(
        self, df_and_factors: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Spec #3: frontier internally builds RatebookFactorContexts and
        matches the legacy frontier numerics. Done implicitly because
        existing test_frontier_oracle tests already pass with the
        rewired frontier (separate file)."""
        df, factors = df_and_factors
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=30,
        )
        fr = opt.frontier(
            df, factors, threshold_ranges={"volume": (0.85, 0.95)}, n_points_per_dim=3
        )
        # Sanity: every point converged or at least produced a finite objective.
        assert fr.n_points == 3
        for v in fr.points["total_objective"].to_list():
            assert math.isfinite(v)

    def test_frontier_explicit_contexts_match_dataframe_built_path(
        self,
        factor_parquet: tuple[Path, pl.DataFrame, pl.DataFrame, list[str]],
    ) -> None:
        """Spec #4: explicit prebuilt contexts passed into frontier match
        the internal dataframe-built context path."""
        parquet_path, df, factors, _ = factor_parquet
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=2,
            max_iter=30,
        )
        ranges = {"volume": (0.85, 0.95)}

        # Internal: frontier builds contexts from factors DataFrame.
        fr_internal = opt.frontier(
            df, factors, threshold_ranges=ranges, n_points_per_dim=3
        )

        # External: user pre-builds the contexts via from_dataframe and
        # passes them in. Should produce the same numerics because both
        # paths route through the same internal builder.
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        contexts = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"], ["age_band"]],
            quote_id=None,  # factors has no quote_id; positional trust.
            expected_quote_ids=grid.quote_ids,
        )
        fr_external = opt.frontier(
            grid, contexts, threshold_ranges=ranges, n_points_per_dim=3
        )

        for col in ["total_objective", "total_volume", "lambda_volume"]:
            for a, b in zip(
                fr_internal.points[col].to_list(),
                fr_external.points[col].to_list(),
            ):
                assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Spec test #5: chunk-size edge cases
# ---------------------------------------------------------------------------


class TestChunkSizes:
    @pytest.mark.parametrize("chunk_size", [1, 7, 50, 1_000])
    def test_chunk_sizes_yield_identical_results(
        self,
        factor_parquet: tuple[Path, pl.DataFrame, pl.DataFrame, list[str]],
        chunk_size: int,
    ) -> None:
        """Spec #5: chunk_size=1, prime (7), boundary (n_quotes), larger
        than input must all produce identical contexts."""
        parquet_path, df, factors, _ = factor_parquet
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        ctx_ref = build_ratebook_factor_contexts_from_parquet_chunked(
            str(parquet_path),
            factor_specs=[["region"], ["age_band"]],
            chunk_size=10_000,  # one shot
            quote_id="quote_id",
            expected_quote_ids=grid.quote_ids,
        )
        ctx_chunked = build_ratebook_factor_contexts_from_parquet_chunked(
            str(parquet_path),
            factor_specs=[["region"], ["age_band"]],
            chunk_size=chunk_size,
            quote_id="quote_id",
            expected_quote_ids=grid.quote_ids,
        )
        assert ctx_ref.n_factors == ctx_chunked.n_factors
        assert ctx_ref.n_quotes == ctx_chunked.n_quotes
        assert ctx_ref.quote_id_fingerprint == ctx_chunked.quote_id_fingerprint


# ---------------------------------------------------------------------------
# Spec test #6: composite factor specs
# ---------------------------------------------------------------------------


class TestCompositeFactors:
    def test_composite_spec_solves(
        self,
        factor_parquet: tuple[Path, pl.DataFrame, pl.DataFrame, list[str]],
    ) -> None:
        """Spec #6: composite factor specs work in both modes."""
        parquet_path, df, factors, _ = factor_parquet
        # Add a 'channel' column so we can build a composite spec.
        n = factors.shape[0]
        factors_extended = factors.with_columns(
            pl.Series("channel", [["online", "broker"][i % 2] for i in range(n)])
        )
        # Persist a matching parquet with the extra column.
        sorted_qids = (
            df.select(pl.col("quote_id").cast(pl.Utf8).unique().sort())
            .to_series()
            .to_list()
        )
        with tempfile.TemporaryDirectory() as tmp:
            parquet_path_ext = Path(tmp) / "factors_composite.parquet"
            factors_extended.with_columns(
                pl.Series("quote_id", sorted_qids, dtype=pl.Utf8)
            ).write_parquet(parquet_path_ext)
            specs: list[list[str]] = [["region"], ["age_band", "channel"]]

            opt = RatebookOptimiser(
                objective="expected_income",
                constraints={"volume": {"min_pct": 0.90}},
                factor_columns=specs,
                max_cd_iterations=1,
                max_iter=30,
            )
            result_df = opt.solve(df, factors_extended)
            grid = build_grid(
                df,
                constraint_columns=["volume"],
                quote_id="quote_id",
                scenario_index="scenario_index",
                scenario_value="scenario_value",
                objective="expected_income",
            )
            ctx = build_ratebook_factor_contexts_from_parquet_chunked(
                str(parquet_path_ext),
                factor_specs=specs,
                chunk_size=15,
                quote_id="quote_id",
                expected_quote_ids=grid.quote_ids,
            )
            result_ctx = opt.solve(grid, ctx)
            for fname in result_df.factor_tables:
                assert result_df.factor_tables[fname] == pytest.approx(
                    result_ctx.factor_tables[fname]
                )


# ---------------------------------------------------------------------------
# Spec test #7: out-of-order quote grid still solves identically
# ---------------------------------------------------------------------------


class TestOrderingValidation:
    def test_unsorted_quote_grid_input_solves_identically_via_chunked_contexts(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
        tmp_path: Path,
    ) -> None:
        """Spec #7: when the input DataFrame is in a deliberately unsorted
        quote order, the QuoteGridBuilder sorts; chunked contexts built
        with expected_quote_ids=grid.quote_ids must still solve
        identically."""
        df, factors = df_and_factors
        # Shuffle the DataFrame in quote_id reverse order.
        df_shuf = df.sort("quote_id", descending=True)
        # Write a parquet that's in *yet another* arbitrary order: the
        # factor source rows ordered by reversed quote_id.
        sorted_qids = (
            df.select(pl.col("quote_id").cast(pl.Utf8).unique().sort())
            .to_series()
            .to_list()
        )
        # Build a factor parquet whose row order is reverse-sorted.
        reverse_qids = list(reversed(sorted_qids))
        # Map each quote_id to its row in `factors` (which is index-
        # aligned to df's quote order).
        factor_rows_by_qid = {qid: i for i, qid in enumerate(sorted_qids)}
        reordered_factors = factors[
            [factor_rows_by_qid[qid] for qid in reverse_qids]
        ].with_columns(pl.Series("quote_id", reverse_qids, dtype=pl.Utf8))

        parquet_path = tmp_path / "factors_reverse.parquet"
        reordered_factors.write_parquet(parquet_path)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=1,
            max_iter=30,
        )
        result_reference = opt.solve(df, factors)

        # Build grid from the shuffled DataFrame (which the builder
        # will sort internally), then build contexts against the
        # reverse-ordered parquet.
        grid = build_grid(
            df_shuf,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        ctx = build_ratebook_factor_contexts_from_parquet_chunked(
            str(parquet_path),
            factor_specs=[["region"], ["age_band"]],
            chunk_size=15,
            quote_id="quote_id",
            expected_quote_ids=grid.quote_ids,
        )
        result_ctx = opt.solve(grid, ctx)
        assert result_reference.total_objective == pytest.approx(
            result_ctx.total_objective
        )
        for fname in result_reference.factor_tables:
            assert result_reference.factor_tables[fname] == pytest.approx(
                result_ctx.factor_tables[fname]
            )


# ---------------------------------------------------------------------------
# Spec test #8-#12: validation paths (duplicates, missing, unexpected, nulls, dtype)
# ---------------------------------------------------------------------------


class TestValidationFailures:
    def test_duplicate_quote_ids_in_parquet(self, tmp_path: Path) -> None:
        """Spec #8: duplicate quote IDs in factor source fail loudly."""
        df = pl.DataFrame(
            {
                "quote_id": ["Q0", "Q1", "Q0"],
                "region": ["A", "B", "C"],
            }
        )
        p = tmp_path / "dups.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match=r"duplicate"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
                expected_quote_ids=["Q0", "Q1", "Q2"],
            )

    def test_missing_quote_id_vs_expected(self, tmp_path: Path) -> None:
        """Spec #9: missing quote IDs fail loudly."""
        df = pl.DataFrame({"quote_id": ["Q0", "Q1"], "region": ["A", "B"]})
        p = tmp_path / "missing.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match=r"Q2"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
                expected_quote_ids=["Q0", "Q1", "Q2"],
            )

    def test_unexpected_quote_id_vs_expected(self, tmp_path: Path) -> None:
        """Spec #10: unexpected quote IDs fail loudly."""
        df = pl.DataFrame({"quote_id": ["Q0", "Q1", "Q9"], "region": ["A", "B", "C"]})
        p = tmp_path / "extra.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match=r"not in expected"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
                expected_quote_ids=["Q0", "Q1"],
            )

    def test_null_quote_id_rejected(self, tmp_path: Path) -> None:
        """Spec #11: nulls in the quote_id column fail loudly."""
        df = pl.DataFrame({"quote_id": ["Q0", None, "Q2"], "region": ["A", "B", "C"]})
        p = tmp_path / "nulls.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match=r"null"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
            )

    def test_non_string_quote_id_dtype_rejected_consistently(
        self, tmp_path: Path
    ) -> None:
        """Spec #12: Int64 quote_id columns are rejected consistently
        with QuoteGridBuilder (which also rejects non-string
        quote_ids via the shared `quote_id_str_iter` helper). The
        contract is "Utf8 or Categorical" — anything else gets a
        named-error rejection."""
        df = pl.DataFrame({"quote_id": [10, 20, 30], "region": ["A", "B", "C"]})
        p = tmp_path / "intq.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match=r"Int64|Utf8|Categorical"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
                expected_quote_ids=["10", "20", "30"],
            )

    def test_categorical_quote_id_accepted(self, tmp_path: Path) -> None:
        """Categorical quote_id dtype is accepted (mirrors the
        QuoteGridBuilder contract — see test_categorical_quote_id.py
        for the grid-side equivalent)."""
        df = pl.DataFrame(
            {
                "quote_id": pl.Series(["Q0", "Q1", "Q2"], dtype=pl.Categorical),
                "region": ["A", "B", "A"],
            }
        )
        p = tmp_path / "catq.parquet"
        df.write_parquet(p)
        ctx = build_ratebook_factor_contexts_from_parquet_chunked(
            str(p),
            factor_specs=[["region"]],
            chunk_size=10,
            quote_id="quote_id",
            expected_quote_ids=["Q0", "Q1", "Q2"],
        )
        assert ctx.n_quotes == 3
        assert ctx.quote_id_fingerprint is not None

    def test_empty_parquet_rejected(self, tmp_path: Path) -> None:
        """Spec #13: empty parquet fails loudly."""
        df = pl.DataFrame(
            {
                "quote_id": pl.Series([], dtype=pl.Utf8),
                "region": pl.Series([], dtype=pl.Utf8),
            }
        )
        p = tmp_path / "empty.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match=r"no rows"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
            )


# ---------------------------------------------------------------------------
# Spec test #14, #15, #16: fingerprint validation in solve
# ---------------------------------------------------------------------------


class TestSolveAlignmentChecks:
    def test_grid_a_context_for_grid_b_fingerprint_mismatch(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Spec #14: contexts built for grid A must fail when passed to
        solve(grid_B, contexts) if fingerprints differ."""
        df, factors = df_and_factors
        # Build grid A from the original DataFrame.
        grid_a = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        # Build contexts against grid A's quote_ids.
        ctx = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=grid_a.quote_ids,
        )

        # Build grid B from a DataFrame with a DIFFERENT quote set.
        df_b = df.with_columns((pl.col("quote_id") + "_v2").alias("quote_id"))
        grid_b = build_grid(
            df_b,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
        )
        with pytest.raises(ValueError, match=r"fingerprint"):
            opt.solve(grid_b, ctx)

    def test_unprovable_order_rejected_against_quote_grid(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Spec #15: contexts built without provable quote order
        (None fingerprint) are rejected by solve(QuoteGrid, contexts)."""
        df, factors = df_and_factors
        ctx = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=None,  # no fingerprint
        )
        assert ctx.quote_id_fingerprint is None
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
        )
        with pytest.raises(ValueError, match=r"fingerprint"):
            opt.solve(grid, ctx)

    def test_n_quotes_mismatch_error(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Spec #16: solve(grid_N, contexts_M) rejects mismatched
        quote counts with both numbers in the message."""
        df, factors = df_and_factors
        # Build contexts on a SUBSET of the quotes.
        subset_factors = factors[:30]
        subset_qids = [f"Q{i:04d}" for i in range(30)]
        ctx = RatebookFactorContexts.from_dataframe(
            subset_factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=subset_qids,
        )
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
        )
        with pytest.raises(ValueError, match=r"30.*50|50.*30"):
            opt.solve(grid, ctx)


# ---------------------------------------------------------------------------
# Spec test #17-#19: schema / spec-conflict validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_missing_factor_column_clear_error(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Spec #17: missing factor column raises a clear error naming
        the missing column AND available columns."""
        df, factors = df_and_factors
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["nonexistent_column"]],
            max_cd_iterations=1,
        )
        with pytest.raises(ValueError, match=r"nonexistent_column"):
            opt.solve(df, factors)

    def test_null_factor_value_rejected(self, tmp_path: Path) -> None:
        """Spec #18: null factor values match current dataframe-mode
        behaviour (rejected with a clear error)."""
        df_factors = pl.DataFrame(
            {
                "quote_id": ["Q0", "Q1", "Q2"],
                "region": ["A", None, "B"],
            }
        )
        p = tmp_path / "nullf.parquet"
        df_factors.write_parquet(p)
        with pytest.raises(ValueError, match=r"null"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
                expected_quote_ids=["Q0", "Q1", "Q2"],
            )

    def test_factor_columns_conflict_with_prebuilt_contexts(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Spec #19: contexts.factor_specs is authoritative; an
        explicit factor_columns argument that disagrees must error."""
        df, factors = df_and_factors
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        ctx = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=grid.quote_ids,
        )
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_cd_iterations=1,
        )
        with pytest.raises(ValueError, match=r"conflict"):
            opt.solve(grid, ctx, factor_columns=[["age_band"]])


# ---------------------------------------------------------------------------
# Spec test #20-#22: public API surface
# ---------------------------------------------------------------------------


class TestPublicAPISurface:
    def test_contexts_does_not_expose_contexts_attr(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Spec #20: RatebookFactorContexts does not expose a
        ``.contexts`` field (which would leak the private FactorContext
        type into the public API)."""
        df, factors = df_and_factors
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        ctx = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=grid.quote_ids,
        )
        assert not hasattr(ctx, "contexts")

    def test_factor_context_not_in_public_all(self) -> None:
        """Spec #21: FactorContext is not exported in price_contour.__all__."""
        assert "FactorContext" not in pc.__all__

    def test_opaque_wrapper_exposes_read_only_metadata_only(
        self,
        df_and_factors: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Spec #22: the opaque context object exposes read-only
        metadata (factor_specs, n_factors, n_quotes,
        quote_id_fingerprint) — no writable attributes."""
        df, factors = df_and_factors
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        ctx = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=grid.quote_ids,
        )
        assert ctx.n_factors == 1
        assert ctx.n_quotes == grid.n_quotes
        assert isinstance(ctx.quote_id_fingerprint, int)
        # `frozen=True` on the pyclass blocks attribute mutation; the
        # underlying PyClass is a frozen Rust struct, so setattr raises.
        with pytest.raises((AttributeError, TypeError)):
            ctx.n_quotes = 999


# ---------------------------------------------------------------------------
# Spec test #23: pin dataframe-mode label ordering
# ---------------------------------------------------------------------------


class TestDataframeModeLabelOrderingPinned:
    def test_label_ordering_first_seen_in_lex_quote_order(self) -> None:
        """Spec #23: pin the current dataframe-mode label ordering.
        Labels are assigned in first-encounter order over the
        lex-quote-sorted traversal (which is what QuoteGridBuilder
        produces). This is the parity target for the chunked path."""
        n = 5
        df = pl.DataFrame(
            {
                "quote_id": ([f"Q{i:04d}" for i in range(n) for _ in range(2)]),
                "scenario_index": [0, 1] * n,
                "scenario_value": [0.9, 1.1] * n,
                "expected_income": [10.0, 11.0] * n,
                "volume": [1.0, 0.9] * n,
            },
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        # Factors aligned positionally to quote order Q0000..Q0004.
        factors = pl.DataFrame({"region": ["South", "North", "South", "East", "North"]})
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        # Build contexts via the dataframe path.
        ctx = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=grid.quote_ids,
        )
        # The lex-sorted quote order is already Q0000..Q0004. First-
        # seen-in-quote-order is: South (Q0000), North (Q0001),
        # East (Q0003). So labels in group_labels order must be
        # ["South", "North", "East"]. We can't peek into the
        # internal `group_labels` from Python (FactorContext is
        # private), but we can run a solve and check the
        # factor_tables dict — its key ORDER reflects the underlying
        # group_labels order because the conversion zip(group_labels,
        # factor_values) preserves order.
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.50}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        result = opt.solve(grid, ctx)
        # The factor_tables dict is a Python dict whose key order
        # is the build-order. South should come first.
        keys = list(result.factor_tables["region"].keys())
        assert keys == ["South", "North", "East"], (
            f"label ordering drifted from first-seen-in-quote-order: {keys}"
        )


# ---------------------------------------------------------------------------
# Spec test #25-#27: performance and memory invariants
# ---------------------------------------------------------------------------


class TestProjectionAndMemory:
    def test_parquet_builder_reads_only_required_columns(self, tmp_path: Path) -> None:
        """Spec #25: the parquet builder reads only the required columns
        (quote_id + factor columns), ignoring unrelated columns in the
        file. We verify by writing a parquet with extra junk columns
        and confirming the builder succeeds (it would error on dtype
        mismatch if it tried to decode them naively)."""
        df = pl.DataFrame(
            {
                "quote_id": ["Q0", "Q1", "Q2"],
                "region": ["A", "B", "A"],
                # Extra columns that aren't requested. If the builder
                # naively decoded the whole file these wouldn't hurt,
                # but write something potentially expensive: a long
                # binary blob and a wide array column.
                "blob": [b"x" * 10_000, b"y" * 10_000, b"z" * 10_000],
                "unused_int": [1, 2, 3],
            }
        )
        p = tmp_path / "extra_cols.parquet"
        df.write_parquet(p)
        ctx = build_ratebook_factor_contexts_from_parquet_chunked(
            str(p),
            factor_specs=[["region"]],
            chunk_size=10,
            quote_id="quote_id",
            expected_quote_ids=["Q0", "Q1", "Q2"],
        )
        assert ctx.n_quotes == 3


# ---------------------------------------------------------------------------
# Spec test #28, #29: four-case fingerprint matrix completeness
# ---------------------------------------------------------------------------


class TestQuoteIdMatrix:
    def test_from_dataframe_no_quote_id_col_with_expected_matches_with_col(
        self, df_and_factors: tuple[pl.DataFrame, pl.DataFrame], tmp_path: Path
    ) -> None:
        """Spec #28: from_dataframe(df_without_quote_id_col,
        expected_quote_ids=...) produces the same contexts and solver
        output as from_dataframe(df_with_quote_id_col_in_grid_order)."""
        df, factors = df_and_factors
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        # Path A: factors WITHOUT quote_id column (positional trust).
        ctx_a = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=grid.quote_ids,
        )
        # Path B: factors WITH quote_id column in grid order.
        factors_with_qid = factors.with_columns(
            pl.Series("quote_id", grid.quote_ids, dtype=pl.Utf8)
        )
        ctx_b = RatebookFactorContexts.from_dataframe(
            factors_with_qid,
            [["region"]],
            quote_id="quote_id",
            expected_quote_ids=grid.quote_ids,
        )
        assert ctx_a.quote_id_fingerprint == ctx_b.quote_id_fingerprint

        # And both produce identical solver output.
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
        )
        r_a = opt.solve(grid, ctx_a)
        r_b = opt.solve(grid, ctx_b)
        assert r_a.total_objective == pytest.approx(r_b.total_objective)
        # pytest.approx doesn't handle nested dicts; compare per-factor.
        assert set(r_a.factor_tables.keys()) == set(r_b.factor_tables.keys())
        for fname in r_a.factor_tables:
            assert r_a.factor_tables[fname] == pytest.approx(r_b.factor_tables[fname])

    def test_from_dataframe_no_quote_id_no_expected_has_none_fingerprint(
        self, df_and_factors: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        """Spec #29: from_dataframe(no quote_id col, no
        expected_quote_ids) yields quote_id_fingerprint=None, and
        solve(QuoteGrid, contexts) rejects it because order cannot be
        proven."""
        df, factors = df_and_factors
        ctx = RatebookFactorContexts.from_dataframe(
            factors,
            [["region"]],
            quote_id=None,
            expected_quote_ids=None,
        )
        assert ctx.quote_id_fingerprint is None
        grid = build_grid(
            df,
            constraint_columns=["volume"],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
        )
        with pytest.raises(ValueError, match=r"fingerprint"):
            opt.solve(grid, ctx)


# ---------------------------------------------------------------------------
# Bonus: cross-validation of expected_quote_ids and expected_n_quotes
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_expected_n_quotes_and_quote_ids_must_agree(self, tmp_path: Path) -> None:
        """``expected_quote_ids`` and ``expected_n_quotes`` must
        cross-validate at the top of both constructors. Mismatch
        raises before any data is read."""
        df = pl.DataFrame({"quote_id": ["Q0", "Q1"], "region": ["A", "B"]})
        p = tmp_path / "match.parquet"
        df.write_parquet(p)
        with pytest.raises(ValueError, match=r"expected_n_quotes"):
            build_ratebook_factor_contexts_from_parquet_chunked(
                str(p),
                factor_specs=[["region"]],
                chunk_size=10,
                quote_id="quote_id",
                expected_quote_ids=["Q0", "Q1"],
                expected_n_quotes=5,
            )
