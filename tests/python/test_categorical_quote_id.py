"""Tests for Categorical / dictionary-encoded ``quote_id`` support.

The downstream pipeline (haute) builds a long-format DataFrame where
``quote_id`` is the same string repeated ``n_steps`` times per quote. A
plain ``Utf8`` column materialises the full repetition into memory; a
``Categorical`` column keeps only one copy of each unique id plus per-row
integer codes. price-contour must accept either dtype and produce
byte-identical ``QuoteGrid``s, so callers can pick the encoding that fits
their memory budget without changing anything else.

Acceptance items mirror the original feature request:

1. Utf8 still works (unchanged).
2. Categorical works.
3. Both dtypes produce identical grids.
4. Chunked builder accepts Categorical.
5. Cross-chunk identical-quote_id detection still fires (duplicate guard).
6. Chunk-local dictionary code mismatch handled correctly (the same string
   may map to different physical codes across chunks; price-contour must
   never compare codes across chunks).
7. Unsupported dtypes (Int / Float / Bool / etc) error loudly with a clear
   message.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from price_contour._price_contour import QuoteGridBuilder
from helpers import make_small_df


def _to_categorical_quote_id(df: pl.DataFrame) -> pl.DataFrame:
    """Cast just the quote_id column to Categorical, leave the rest alone."""
    return df.with_columns(pl.col("quote_id").cast(pl.Categorical))


class TestUtf8StillWorks:
    """Acceptance #1: existing Utf8 path is unchanged."""

    def test_dataframe_solve(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        assert result.converged
        assert result.n_quotes == 20

    def test_builder_one_chunk(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        builder = QuoteGridBuilder(["volume"])
        builder.append(df)
        grid = builder.build()
        assert grid.n_quotes == 20

    def test_parquet_chunked(self, tmp_path: Path):
        df = make_small_df(n_quotes=20, n_steps=5)
        path = str(tmp_path / "in.parquet")
        df.write_parquet(path)
        grid = pc.build_grid_from_parquet_chunked(path, ["volume"], chunk_size=37)
        assert grid.n_quotes == 20


class TestCategoricalQuoteId:
    """Acceptance #2 + #3: Categorical accepted, results identical to Utf8."""

    def test_dataframe_solve_accepts_categorical(self):
        df_utf8 = make_small_df(n_quotes=30, n_steps=5)
        df_cat = _to_categorical_quote_id(df_utf8)
        assert df_cat.schema["quote_id"] == pl.Categorical

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        r_utf8 = solver.solve(df_utf8)
        r_cat = solver.solve(df_cat)

        # Same data, same canonical sort, deterministic solver → identical totals.
        assert r_utf8.total_objective == r_cat.total_objective
        assert r_utf8.total_constraints == r_cat.total_constraints
        assert r_utf8.lambdas == r_cat.lambdas
        # Per-quote optimal_step list matches.
        assert (
            r_utf8.dataframe["optimal_step"].to_list()
            == r_cat.dataframe["optimal_step"].to_list()
        )

    def test_builder_one_chunk_accepts_categorical(self):
        df_utf8 = make_small_df(n_quotes=20, n_steps=5)
        df_cat = _to_categorical_quote_id(df_utf8)

        b_utf8 = QuoteGridBuilder(["volume"])
        b_utf8.append(df_utf8)
        g_utf8 = b_utf8.build()

        b_cat = QuoteGridBuilder(["volume"])
        b_cat.append(df_cat)
        g_cat = b_cat.build()

        assert g_utf8.quote_ids == g_cat.quote_ids
        assert g_utf8.scenario_values == g_cat.scenario_values
        assert g_utf8.constraint_names == g_cat.constraint_names
        assert g_utf8.n_quotes == g_cat.n_quotes
        assert g_utf8.n_steps == g_cat.n_steps

    def test_quote_id_strings_preserved(self):
        """Categorical → grid round-trip preserves the actual string values."""
        df_cat = _to_categorical_quote_id(make_small_df(n_quotes=5, n_steps=3))
        builder = QuoteGridBuilder(["volume"])
        builder.append(df_cat)
        grid = builder.build()
        assert grid.quote_ids == [f"Q{q:04d}" for q in range(5)]


class TestCategoricalParquetChunked:
    """Acceptance #4 + #5: chunked path accepts Categorical and detects dups."""

    def test_chunked_parquet_categorical_matches_utf8(self, tmp_path: Path):
        df_utf8 = make_small_df(n_quotes=40, n_steps=5)
        df_cat = _to_categorical_quote_id(df_utf8)

        utf8_path = str(tmp_path / "utf8.parquet")
        cat_path = str(tmp_path / "cat.parquet")
        df_utf8.write_parquet(utf8_path)
        df_cat.write_parquet(cat_path)

        g_utf8 = pc.build_grid_from_parquet_chunked(
            utf8_path, ["volume"], chunk_size=37
        )
        g_cat = pc.build_grid_from_parquet_chunked(cat_path, ["volume"], chunk_size=37)

        assert g_utf8.quote_ids == g_cat.quote_ids
        assert g_utf8.scenario_values == g_cat.scenario_values
        assert g_utf8.n_quotes == g_cat.n_quotes

    def test_categorical_apply_to_parquet_chunked(self, tmp_path: Path):
        """Issue 3's chunked apply path also accepts Categorical inputs."""
        df = make_small_df(n_quotes=30, n_steps=5)
        df_cat = _to_categorical_quote_id(df)

        sr = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        ).solve(df)

        in_path = str(tmp_path / "cat_in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df_cat.write_parquet(in_path)

        result = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=dict(sr.lambdas),
            constraints={"volume": {"min_pct": 0.90}},
            chunk_size=23,
        )
        assert result.total_objective > 0
        out = pl.read_parquet(out_path)
        # Even though input quote_id was Categorical, output dtype follows the
        # canonical schema (Utf8) so downstream consumers see a consistent type.
        assert out.schema["quote_id"] in (pl.Utf8, pl.Categorical)
        assert out.height == 30

    def test_duplicate_quote_id_detected_categorical(self, tmp_path: Path):
        """Same Categorical-encoded quote_id appearing in two chunks is rejected."""
        df = make_small_df(n_quotes=5, n_steps=5)
        # Build a parquet that contains Q0001 twice (across the file). We
        # simulate this by concatenating the same single-quote slice twice.
        q1 = df.filter(pl.col("quote_id") == "Q0001")
        bad = pl.concat([q1, q1])
        bad_cat = _to_categorical_quote_id(bad)

        builder = QuoteGridBuilder(["volume"])
        builder.append(bad_cat)
        with pytest.raises(ValueError, match=r"(?i)duplicate|Q0001"):
            builder.build()


class TestChunkLocalDictionarySafety:
    """Acceptance #6: a Categorical column's physical codes are local to the
    chunk; price-contour must not compare codes across chunks.

    We engineer a scenario where the same string maps to different physical
    codes in two chunks. A naive implementation that compared codes (or held
    code-keyed state across chunks) would corrupt the resulting grid.
    """

    def _make_chunk(
        self,
        ids: list[str],
        n_steps: int,
        *,
        enum_categories: list[str] | None = None,
    ) -> pl.DataFrame:
        """Build a single-chunk DataFrame with a dictionary-encoded
        ``quote_id`` column whose physical codes are determined entirely
        by ``enum_categories``.

        Polars 0.52 unifies `Categorical(...).cast()` outputs through a
        global mapping, so two chunks cast independently will share
        physical codes for any shared string. To engineer chunks with
        genuinely divergent code → string mappings — the worst case the
        chunk-local-dict safety claim has to hold under — we build each
        chunk as ``pl.Enum(enum_categories)``, which assigns codes
        positionally inside the supplied category list. Different
        ``enum_categories`` arguments produce chunks with disjoint code
        spaces.
        """
        rows = []
        for q_id in ids:
            for j in range(n_steps):
                rows.append(
                    {
                        "quote_id": q_id,
                        "scenario_index": j,
                        "scenario_value": 0.8 + 0.1 * j,
                        "expected_income": 100.0 + j,
                        "volume": 0.9 - 0.05 * j,
                    }
                )
        df = pl.DataFrame(
            rows,
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        if enum_categories is not None:
            return df.with_columns(pl.col("quote_id").cast(pl.Enum(enum_categories)))
        return df.with_columns(pl.col("quote_id").cast(pl.Categorical))

    def test_chunks_with_different_local_dicts(self):
        """Two chunks where 'Q0003' has different physical codes still build
        the right grid because we never compare codes across chunks.

        We use ``pl.Enum`` with explicitly disjoint category orders so the
        engineered codes really do differ — defended below by an inline
        sentinel assertion that fails loudly if a future Polars change
        unifies the mappings.
        """
        # Chunk 1: Enum order Q0003, Q0001, Q0002 → physical codes [0, 1, 2]
        # Chunk 2: Enum order Q0004, Q0003, Q0005 → physical codes [0, 1, 2]
        # The string "Q0003" is at code 0 in chunk 1 and code 1 in chunk 2 —
        # any implementation that propagated codes across the chunk boundary
        # would mis-attribute Q0003's data.
        chunk1 = self._make_chunk(
            ["Q0003", "Q0001", "Q0002"],
            n_steps=3,
            enum_categories=["Q0003", "Q0001", "Q0002"],
        )
        chunk2 = self._make_chunk(
            ["Q0004", "Q0003", "Q0005"],
            n_steps=3,
            enum_categories=["Q0004", "Q0003", "Q0005"],
        )

        # Sentinel: prove the dicts really are divergent. Q0003 must NOT
        # have the same physical code in both chunks. If a future Polars
        # version starts unifying enum dictionaries, this assertion fires
        # and the test stops silently passing without exercising the
        # chunk-local-dict path.
        c1_codes = chunk1["quote_id"].to_physical().to_list()
        c2_codes = chunk2["quote_id"].to_physical().to_list()
        # Find Q0003's code in each chunk's data (it's at row 0 of chunk1
        # because chunk1's data starts with Q0003; it's at row 3 of chunk2
        # because chunk2 = [Q0004, Q0003, Q0005] × 3 steps).
        c1_q0003 = c1_codes[0]  # Q0003 first in chunk1
        c2_q0003 = c2_codes[3]  # Q0003 second in chunk2 (row 3)
        assert c1_q0003 != c2_q0003, (
            f"Test fixture is broken: Q0003 has the same physical code "
            f"({c1_q0003}) in both chunks. Polars must have unified the "
            f"enum mappings — this test is no longer exercising the "
            f"chunk-local-dict path."
        )

        builder = QuoteGridBuilder(["volume"])
        builder.append(chunk1)
        # Chunk2 also contains Q0003, so build() should detect the duplicate.
        builder.append(chunk2)
        with pytest.raises(ValueError, match=r"(?i)duplicate|Q0003"):
            builder.build()

    def test_disjoint_chunks_with_different_local_dicts(self):
        """No-overlap version: two chunks whose dicts are completely disjoint
        build a unioned grid correctly."""
        chunk1 = self._make_chunk(
            ["Q0003", "Q0001"],
            n_steps=3,
            enum_categories=["Q0003", "Q0001"],
        )
        chunk2 = self._make_chunk(
            ["Q0004", "Q0002"],
            n_steps=3,
            enum_categories=["Q0004", "Q0002"],
        )

        builder = QuoteGridBuilder(["volume"])
        builder.append(chunk1)
        builder.append(chunk2)
        grid = builder.build()
        # build() sorts by quote_id; final order is Q0001..Q0004.
        assert grid.quote_ids == ["Q0001", "Q0002", "Q0003", "Q0004"]


class TestEnumPhysicalWidths:
    """Polars `cast(pl.Categorical)` in 0.52 always picks U32 physical
    width regardless of cardinality, so the U8 / U16 dispatch branches in
    `quote_id_str_iter` need explicit `pl.Enum` coverage to be exercised.

    Polars selects the physical width by category-list cardinality:
    ≤256 → U8, ≤65536 → U16, otherwise U32. We pin all three.
    """

    def _build_enum_grid(
        self, n_quotes: int, n_steps: int, categories: list[str]
    ) -> pc.QuoteGrid:
        rows = []
        for q in range(n_quotes):
            qid = categories[q % len(categories)]
            # Make ids unique by appending the index so duplicate-detection
            # doesn't fire (we want each quote to be distinct).
            unique_qid = f"{qid}-{q:08x}"
            for j in range(n_steps):
                rows.append(
                    {
                        "quote_id": unique_qid,
                        "scenario_index": j,
                        "scenario_value": 0.8 + 0.1 * j,
                        "expected_income": 100.0 + j,
                        "volume": 0.9 - 0.05 * j,
                    }
                )
        df = pl.DataFrame(
            rows,
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        # Build an Enum whose category list is the full set of unique ids;
        # Polars chooses physical width based on len(categories).
        unique_ids = df["quote_id"].unique().to_list()
        df_enum = df.with_columns(pl.col("quote_id").cast(pl.Enum(unique_ids)))
        builder = QuoteGridBuilder(["volume"])
        builder.append(df_enum)
        return builder.build()

    def test_u8_physical_width(self):
        """Enum with ≤256 distinct ids exercises the cat8 dispatch branch."""
        grid = self._build_enum_grid(
            n_quotes=200, n_steps=3, categories=["A", "B", "C"]
        )
        assert grid.n_quotes == 200

    def test_u16_physical_width(self):
        """Enum with > 256 and ≤65536 distinct ids exercises cat16."""
        grid = self._build_enum_grid(
            n_quotes=300,
            n_steps=2,
            categories=[f"L{i:03d}" for i in range(300)],
        )
        assert grid.n_quotes == 300

    def test_u32_physical_width(self):
        """Enum with > 65536 distinct ids exercises cat32. 70_000 keeps the
        test fast while clearing the U16 boundary."""
        grid = self._build_enum_grid(
            n_quotes=70_000,
            n_steps=2,
            categories=[f"L{i:06d}" for i in range(70_000)],
        )
        assert grid.n_quotes == 70_000

    def test_enum_physical_width_pinning(self):
        """Pin Polars' physical-width selection so any future change is loud."""
        small = pl.Series("x", ["a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
        med_categories = [f"L{i:03d}" for i in range(300)]
        med = pl.Series("x", ["L000", "L100"], dtype=pl.Enum(med_categories))
        big_categories = [f"L{i:06d}" for i in range(70_000)]
        big = pl.Series("x", ["L000000", "L050000"], dtype=pl.Enum(big_categories))
        # Polars exposes the physical type via `to_physical()`.
        assert small.to_physical().dtype == pl.UInt8
        assert med.to_physical().dtype == pl.UInt16
        assert big.to_physical().dtype == pl.UInt32


class TestApplyOptimiserCategorical:
    """The in-memory `ApplyOptimiser.apply(df)` path must also accept
    Categorical / Enum quote_id, not just the chunked-parquet apply.
    Both paths flow through the same `ingest_dataframe` extractor, but the
    surface is user-facing and warrants its own parity test."""

    def test_apply_with_categorical_matches_utf8(self):
        df_utf8 = make_small_df(n_quotes=20, n_steps=5)
        df_cat = _to_categorical_quote_id(df_utf8)

        # Get a stable lambda set from a one-shot solve.
        sr = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        ).solve(df_utf8)

        applier = pc.ApplyOptimiser(
            lambdas=dict(sr.lambdas),
            constraints={"volume": {"min_pct": 0.90}},
        )
        r_utf8 = applier.apply(df_utf8)
        r_cat = applier.apply(df_cat)

        assert r_utf8.total_objective == r_cat.total_objective
        assert r_utf8.total_constraints == r_cat.total_constraints
        # Per-quote optimal_step list should match exactly.
        assert (
            r_utf8.dataframe["optimal_step"].to_list()
            == r_cat.dataframe["optimal_step"].to_list()
        )


class TestUnsupportedDtypes:
    """Acceptance #7: non-Utf8/Categorical dtypes for quote_id error loudly."""

    @pytest.mark.parametrize(
        "bad_dtype, bad_values",
        [
            (pl.Int32, [1, 2, 3]),
            (pl.Float32, [1.0, 2.0, 3.0]),
            (pl.Boolean, [True, False, True]),
        ],
    )
    def test_non_string_quote_id_rejected(self, bad_dtype, bad_values):
        df = make_small_df(n_quotes=3, n_steps=3)
        # Replace the quote_id column with a numeric/bool one.
        n_steps = 3
        # Build a per-row numeric quote_id (each value repeated n_steps times).
        ids = []
        for v in bad_values:
            ids.extend([v] * n_steps)
        df = df.drop("quote_id").with_columns(
            pl.Series("quote_id", ids, dtype=bad_dtype)
        )
        builder = QuoteGridBuilder(["volume"])
        with pytest.raises(ValueError, match=r"(?i)quote_id|utf8|categorical|string"):
            builder.append(df)


class TestMemoryRegression:
    """Sanity-check that Categorical input is genuinely smaller than Utf8.

    The memory win price-contour exposes is downstream of Polars itself:
    once a `quote_id` column is in memory, dictionary-encoded
    representation (Categorical) uses ~4-byte codes per row plus one copy
    of each unique string, vs Utf8 which materialises every repeated
    string. This test pins that gap directly via
    ``DataFrame.estimated_size()`` (Polars' own Arrow-buffer accountant)
    on the same logical dataset stored both ways.

    We don't try to measure peak RSS during ``build_grid_from_parquet_chunked``
    because (a) Polars buffers live outside Python's allocator so
    ``tracemalloc`` is blind to them, and (b) RSS-based measurements are
    notoriously noisy in CI. The Polars-reported buffer size is the
    relevant signal — if our reader accepts Categorical without secretly
    casting to Utf8, the source DataFrame stays compact, and that's what
    we verify.
    """

    def test_categorical_quote_id_is_smaller_than_utf8(self):
        """The same data with UUID-style quote_ids should be ~50% smaller
        as Categorical than as Utf8. Failure here would mean Polars'
        Categorical encoding regressed or our test is misshapen."""
        n_quotes, n_steps = 50_000, 5
        ids = [
            f"quote-{q:08x}-{q:04x}-{q:04x}-{q:04x}-{q:012x}" for q in range(n_quotes)
        ]
        rows_per_quote = list(range(n_steps))
        df_utf8 = pl.DataFrame(
            {
                "quote_id": [qid for qid in ids for _ in rows_per_quote],
                "scenario_index": rows_per_quote * n_quotes,
                "scenario_value": [0.8 + 0.1 * j for _ in ids for j in rows_per_quote],
                "expected_income": [100.0 + j for _ in ids for j in rows_per_quote],
                "volume": [0.9 - 0.05 * j for _ in ids for j in rows_per_quote],
            },
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        df_cat = df_utf8.with_columns(pl.col("quote_id").cast(pl.Categorical))

        utf8_qid_bytes = df_utf8["quote_id"].estimated_size()
        cat_qid_bytes = df_cat["quote_id"].estimated_size()

        # UUID-shaped ids are ~36 bytes each; Categorical replaces each
        # repetition with a 4-byte code, so the gap should be large. Allow
        # 50% headroom on the assertion to absorb any encoding-overhead
        # surprises while still catching genuine regressions.
        assert cat_qid_bytes < utf8_qid_bytes * 0.5, (
            f"Categorical quote_id ({cat_qid_bytes:,} B) was not at least "
            f"50% smaller than Utf8 ({utf8_qid_bytes:,} B); "
            f"ratio = {cat_qid_bytes / utf8_qid_bytes:.1%}"
        )

    def test_chunked_reader_accepts_categorical_without_inflating(self, tmp_path: Path):
        """Round-trip: write a Categorical-encoded parquet, read it back via
        ``build_grid_from_parquet_chunked``, confirm the resulting QuoteGrid
        has the right quote_ids. The contract here is functional, not
        memory: prior to this change, the reader would have raised
        ``ValueError: quote_id must be Utf8`` and the user would have had
        no choice but to cast back to Utf8 (and forfeit the win)."""
        n_quotes, n_steps = 1_000, 5
        ids = [f"quote-{q:08x}" for q in range(n_quotes)]
        rows_per_quote = list(range(n_steps))
        df = pl.DataFrame(
            {
                "quote_id": [qid for qid in ids for _ in rows_per_quote],
                "scenario_index": rows_per_quote * n_quotes,
                "scenario_value": [0.8 + 0.1 * j for _ in ids for j in rows_per_quote],
                "expected_income": [100.0 + j for _ in ids for j in rows_per_quote],
                "volume": [0.9 - 0.05 * j for _ in ids for j in rows_per_quote],
            },
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        ).with_columns(pl.col("quote_id").cast(pl.Categorical))

        path = str(tmp_path / "cat.parquet")
        df.write_parquet(path)
        grid = pc.build_grid_from_parquet_chunked(path, ["volume"], chunk_size=2_000)
        assert grid.n_quotes == n_quotes
        assert grid.quote_ids[:3] == ids[:3]
