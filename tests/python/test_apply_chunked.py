"""Tests for the chunked, streaming-output apply path.

The function reads an input parquet in fixed-size row slices, runs
`apply_lambdas` per chunk, writes per-quote results to an output parquet
incrementally, and returns aggregate totals — without ever holding the full
optimal_steps array in memory.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from helpers import make_small_df


@pytest.fixture
def df_and_lambdas() -> tuple[
    pl.DataFrame, dict[str, float], dict[str, dict[str, float]]
]:
    """Run a one-shot solve to get a stable lambda set + baseline DataFrame."""
    df = make_small_df(n_quotes=40, n_steps=5)
    constraints = {"volume": {"min_pct": 0.92}}
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints=constraints,
        max_iter=200,
    )
    sr = solver.solve(df)
    return df, dict(sr.lambdas), constraints


class TestApplyChunked:
    def test_totals_match_oneshot(self, tmp_path: Path, df_and_lambdas):
        """Chunked apply totals match the one-shot apply totals."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        oneshot = pc.ApplyOptimiser(
            lambdas=lambdas,
            constraints=constraints,
        ).apply(df)

        chunked = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=37,
        )

        assert abs(chunked.total_objective - oneshot.total_objective) < 1e-3
        assert abs(chunked.baseline_objective - oneshot.baseline_objective) < 1e-3
        for k in oneshot.total_constraints:
            assert (
                abs(chunked.total_constraints[k] - oneshot.total_constraints[k]) < 1e-3
            )
            assert (
                abs(chunked.baseline_constraints[k] - oneshot.baseline_constraints[k])
                < 1e-3
            )

    def test_output_parquet_schema(self, tmp_path: Path, df_and_lambdas):
        """Output parquet has the expected per-quote columns."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        result = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=20,
        )
        assert result.output_path == out_path

        out = pl.read_parquet(out_path)
        expected_cols = {
            "quote_id",
            "optimal_step",
            "optimal_scenario_value",
            "optimal_objective",
            "optimal_volume",
        }
        assert set(out.columns) == expected_cols
        assert out.height == df.select("quote_id").n_unique()

    def test_output_matches_oneshot_per_quote(self, tmp_path: Path, df_and_lambdas):
        """Per-quote output (joined on quote_id) matches the one-shot DataFrame."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        oneshot_df = (
            pc.ApplyOptimiser(
                lambdas=lambdas,
                constraints=constraints,
            )
            .apply(df)
            .dataframe
        )

        pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=23,
        )

        chunked_df = pl.read_parquet(out_path).sort("quote_id")
        oneshot_df = oneshot_df.sort("quote_id")

        # Same quotes, same optimal step per quote.
        assert chunked_df["quote_id"].to_list() == oneshot_df["quote_id"].to_list()
        assert (
            chunked_df["optimal_step"].to_list() == oneshot_df["optimal_step"].to_list()
        )

    def test_invariant_under_chunk_size(self, tmp_path: Path, df_and_lambdas):
        """Same input + lambdas → same totals across multiple chunk sizes."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)

        results = []
        for cs in (5, 13, 37, 100, 10_000):
            out_path = str(tmp_path / f"out_{cs}.parquet")
            results.append(
                pc.apply_lambdas_to_parquet_chunked(
                    parquet_in=in_path,
                    parquet_out=out_path,
                    lambdas=lambdas,
                    constraints=constraints,
                    chunk_size=cs,
                )
            )
        ref = results[0]
        # Different chunk_sizes cut the file into different chunk
        # partitions, so the per-chunk sums are added in different orders.
        # f64 addition is non-associative, so a tiny drift (a few ULPs) is
        # expected. The drift is bounded by O(N × ε), well below 1e-6 for
        # n_quotes ≤ 100.
        for r in results[1:]:
            assert abs(r.total_objective - ref.total_objective) < 1e-6
            assert abs(r.baseline_objective - ref.baseline_objective) < 1e-6
            for k in ref.total_constraints:
                assert abs(r.total_constraints[k] - ref.total_constraints[k]) < 1e-6

    def test_multiple_constraints(self, tmp_path: Path):
        """Two constraints in the input parquet — both apply correctly."""
        df = make_small_df(n_quotes=30, n_steps=5)
        constraints = {
            "volume": {"min_pct": 0.90},
            "loss_ratio": {"max_pct": 1.05},
        }
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints=constraints,
            max_iter=200,
        )
        sr = solver.solve(df)
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        oneshot = pc.ApplyOptimiser(
            lambdas=dict(sr.lambdas),
            constraints=constraints,
        ).apply(df)
        chunked = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=dict(sr.lambdas),
            constraints=constraints,
            chunk_size=29,
        )

        assert abs(chunked.total_objective - oneshot.total_objective) < 1e-3
        assert set(chunked.total_constraints.keys()) == {"volume", "loss_ratio"}

        out = pl.read_parquet(out_path)
        assert "optimal_volume" in out.columns
        assert "optimal_loss_ratio" in out.columns

    def test_zero_lambdas_unconstrained(self, tmp_path: Path):
        """Zero lambdas → each quote picks max-objective step (unconstrained)."""
        df = make_small_df(n_quotes=20, n_steps=5)
        constraints = {"volume": {"min_pct": 0.90}}
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        result = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas={"volume": 0.0},
            constraints=constraints,
            chunk_size=15,
        )
        # Unconstrained → each quote picks its max-objective step.
        out = pl.read_parquet(out_path).sort("quote_id")
        per_quote_max = (
            df.group_by("quote_id")
            .agg(pl.col("expected_income").max().alias("max_obj"))
            .sort("quote_id")
        )
        for opt, ref in zip(
            out["optimal_objective"].to_list(),
            per_quote_max["max_obj"].to_list(),
        ):
            assert abs(opt - ref) < 1e-3
        assert result.total_objective > 0.0

    def test_explicit_n_steps_kwarg(self, tmp_path: Path, df_and_lambdas):
        """Passing `n_steps` skips auto-detection."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        chunked = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=5,
            n_steps=5,
        )
        assert chunked.total_objective > 0.0
        out = pl.read_parquet(out_path)
        assert out.height == 40

    def test_chunk_size_zero_rejected(self, tmp_path: Path, df_and_lambdas):
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)
        with pytest.raises(ValueError, match=r"(?i)chunk_size"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=str(tmp_path / "out.parquet"),
                lambdas=lambdas,
                constraints=constraints,
                chunk_size=0,
            )

    def test_missing_input_file_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"(?i)open|parquet"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in="/no/such/file.parquet",
                parquet_out=str(tmp_path / "out.parquet"),
                lambdas={},
                constraints={},
                chunk_size=100,
            )

    def test_missing_input_does_not_delete_existing_output(self, tmp_path: Path):
        """A pre-read failure must not clean up a file this call never opened."""
        out_path = tmp_path / "existing.parquet"
        out_path.write_bytes(b"existing output")

        with pytest.raises(ValueError, match=r"(?i)open|parquet"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in="/no/such/file.parquet",
                parquet_out=str(out_path),
                lambdas={},
                constraints={},
                chunk_size=100,
            )

        assert out_path.read_bytes() == b"existing output"

    def test_same_input_output_path_rejected_and_input_preserved(
        self, tmp_path: Path, df_and_lambdas
    ):
        """Never truncate the input parquet by using it as the output path."""
        df, lambdas, constraints = df_and_lambdas
        in_path = tmp_path / "in.parquet"
        df.write_parquet(in_path)

        with pytest.raises(ValueError, match=r"(?i)different|overwrite|input"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=str(in_path),
                parquet_out=str(in_path),
                lambdas=lambdas,
                constraints=constraints,
                chunk_size=20,
            )

        assert pl.read_parquet(in_path).height == df.height

    def test_extra_lambda_key_rejected(self, tmp_path: Path, df_and_lambdas):
        """Mirrors ApplyOptimiser: lambdas may be missing, but not extra."""
        df, _lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)

        with pytest.raises(ValueError, match=r"(?i)lambda|valid constraint"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=str(tmp_path / "out.parquet"),
                lambdas={"volume": 0.0, "not_a_constraint": 1.0},
                constraints=constraints,
                chunk_size=20,
            )

    def test_constraint_collision_with_schema_raises(
        self, tmp_path: Path, df_and_lambdas
    ):
        df, _lambdas, _constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)
        with pytest.raises(ValueError, match=r"(?i)collide|schema"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=str(tmp_path / "out.parquet"),
                lambdas={"quote_id": 0.0},
                constraints={"quote_id": {"min": 0.0}},
                chunk_size=100,
            )

    def test_missing_lambda_defaults_to_zero(self, tmp_path: Path, df_and_lambdas):
        """Mirrors one-shot apply: a constraint with no lambda key is treated as 0."""
        df, _lambdas, _ = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)

        # `lambdas={}` for a constraint matches the one-shot apply behaviour
        # (`order_lambdas` defaults missing keys to 0.0). We assert chunked
        # apply behaves the same so callers see one contract.
        chunked = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=str(tmp_path / "chunked.parquet"),
            lambdas={},
            constraints={"volume": {"min_pct": 0.90}},
            chunk_size=100,
        )
        oneshot = pc.ApplyOptimiser(
            lambdas={},
            constraints={"volume": {"min_pct": 0.90}},
        ).apply(df)
        assert abs(chunked.total_objective - oneshot.total_objective) < 1e-3

    def test_no_constraints_works(self, tmp_path: Path):
        """Empty constraints dict → grid is built without constraint columns."""
        df = make_small_df(n_quotes=10, n_steps=5)
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        result = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas={},
            constraints={},
            chunk_size=20,
        )
        out = pl.read_parquet(out_path)
        # Only the four schema columns + optimal_*; no per-constraint columns.
        assert set(out.columns) == {
            "quote_id",
            "optimal_step",
            "optimal_scenario_value",
            "optimal_objective",
        }
        assert result.total_objective > 0.0

    def test_apply_then_read_back_solver_compatible(
        self, tmp_path: Path, df_and_lambdas
    ):
        """Output parquet schema is documented; this test pins it."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=37,
        )

        out = pl.read_parquet(out_path)
        # Schema invariant: dtypes pinned for downstream consumers.
        assert out.schema["quote_id"] == pl.Utf8
        assert out.schema["optimal_step"] in (pl.Int32, pl.Int64)
        assert out.schema["optimal_scenario_value"] == pl.Float32
        assert out.schema["optimal_objective"] == pl.Float32

    def test_overwrite_replaces_previous_output(self, tmp_path: Path, df_and_lambdas):
        """Calling twice with the same parquet_out fully replaces the file.

        Previous chunked-write residue must not leak through; the second run's
        row count must equal the second input's quote count, not the
        sum/intersection of the two runs.
        """
        df_a, lambdas, constraints = df_and_lambdas  # 40 quotes
        df_b = make_small_df(n_quotes=15, n_steps=5)
        in_a = str(tmp_path / "in_a.parquet")
        in_b = str(tmp_path / "in_b.parquet")
        out_path = str(tmp_path / "out.parquet")
        df_a.write_parquet(in_a)
        df_b.write_parquet(in_b)

        # First run with the larger input.
        pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_a,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=37,
        )
        # Second run with a smaller input — must overwrite, not append.
        pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_b,
            parquet_out=out_path,
            lambdas={"volume": 0.0},
            constraints={"volume": {"min_pct": 0.90}},
            chunk_size=20,
        )
        out = pl.read_parquet(out_path)
        assert out.height == 15
        # Quote_ids should match the second input only.
        assert set(out["quote_id"].to_list()) == {f"Q{q:04d}" for q in range(15)}

    def test_failure_mid_stream_removes_output(self, tmp_path: Path):
        """A mid-stream error (corrupt scenario_value) must not leave a partial output."""
        df = make_small_df(n_quotes=20, n_steps=5)
        # Corrupt scenario_value for Q0010 step 2 — the per-row builder
        # validation will catch it on the chunk that contains those rows.
        df = df.with_columns(
            pl.when((pl.col("quote_id") == "Q0010") & (pl.col("scenario_index") == 2))
            .then(pl.lit(99.0, dtype=pl.Float32))
            .otherwise(pl.col("scenario_value"))
            .alias("scenario_value")
        )
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        with pytest.raises(ValueError):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=out_path,
                lambdas={"volume": 0.0},
                constraints={"volume": {"min_pct": 0.90}},
                chunk_size=10,  # forces multiple chunks; corrupt row is in chunk 2
            )
        # Output must NOT exist (best-effort cleanup on error).
        assert not Path(out_path).exists(), (
            "Failed run left a partial parquet on disk; cleanup-on-error broken"
        )

    def test_many_chunks(self, tmp_path: Path, df_and_lambdas):
        """50+ chunks — accumulator stability + writer state across many row groups."""
        df_orig, _, _ = df_and_lambdas
        # Bigger dataset to support many small chunks.
        df = make_small_df(n_quotes=500, n_steps=5)
        constraints = {"volume": {"min_pct": 0.92}}
        sr = pc.OnlineOptimiser(
            objective="expected_income",
            constraints=constraints,
            max_iter=200,
        ).solve(df)
        in_path = str(tmp_path / "many.parquet")
        out_path = str(tmp_path / "many_out.parquet")
        df.write_parquet(in_path)

        oneshot = pc.ApplyOptimiser(
            lambdas=dict(sr.lambdas),
            constraints=constraints,
        ).apply(df)

        # chunk_size=50 → 50 chunks (500 quotes × 5 steps / 50 rows = 50 row groups).
        chunked = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=dict(sr.lambdas),
            constraints=constraints,
            chunk_size=50,
        )
        assert abs(chunked.total_objective - oneshot.total_objective) < 1e-2
        out = pl.read_parquet(out_path)
        assert out.height == 500

    def test_round_trip_via_scan_parquet(self, tmp_path: Path, df_and_lambdas):
        """Output parquet is consumable via `pl.scan_parquet` for downstream lazy work."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)

        result = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=37,
        )

        # Lazy join: aggregate optimal_objective and check it equals the
        # in-memory total reported on the result.
        sum_obj = (
            pl.scan_parquet(result.output_path)
            .select(pl.col("optimal_objective").sum())
            .collect()
            .item()
        )
        assert abs(sum_obj - result.total_objective) < 1e-2

    def test_output_directory_missing_raises(self, tmp_path: Path, df_and_lambdas):
        """An output path under a non-existent directory errors clearly."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)
        bogus_out = str(tmp_path / "no_such_dir" / "out.parquet")
        with pytest.raises(ValueError, match=r"(?i)open|output|parquet"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=bogus_out,
                lambdas=lambdas,
                constraints=constraints,
                chunk_size=20,
            )
        # No leftover file in the bogus path.
        assert not Path(bogus_out).exists()

    def test_mid_quote_row_group_input(self, tmp_path: Path, df_and_lambdas):
        """Input parquet whose row groups cut mid-quote still produces correct output."""
        df, lambdas, constraints = df_and_lambdas  # 40 quotes × 5 steps = 200 rows
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        # gcd(37, 5) = 1, so row groups land mid-quote.
        df.write_parquet(in_path, row_group_size=37)

        oneshot = pc.ApplyOptimiser(
            lambdas=lambdas,
            constraints=constraints,
        ).apply(df)
        chunked = pc.apply_lambdas_to_parquet_chunked(
            parquet_in=in_path,
            parquet_out=out_path,
            lambdas=lambdas,
            constraints=constraints,
            chunk_size=23,
        )
        assert abs(chunked.total_objective - oneshot.total_objective) < 1e-3

    def test_chunk_size_below_n_steps_raises(self, tmp_path: Path, df_and_lambdas):
        """chunk_size < n_steps must error early via the shared chunked-reader."""
        df, lambdas, constraints = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)
        with pytest.raises(ValueError, match=r"(?i)chunk_size|n_steps"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=str(tmp_path / "out.parquet"),
                lambdas=lambdas,
                constraints=constraints,
                chunk_size=3,  # < n_steps=5
            )

    def test_ratio_constraints_rejected(self, tmp_path: Path):
        """Ratio constraints can't be linearised in a per-chunk mini-grid.
        The PyO3 signature rejects ratio specs at the type-system boundary
        (the `numerator` / `denominator` string values can't deserialise
        into the constraints' `Option<f64>` value type), so the user
        always sees a clear error rather than silent behaviour. The
        Python wrapper's docstring directs ratio callers to
        ``ApplyOptimiser.apply(df)`` — this test pins the rejection."""
        df = make_small_df(n_quotes=10, n_steps=5)
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)
        # PyO3 raises TypeError (a subclass of Exception) at the boundary;
        # the user-visible behaviour is "this fails loudly, not silently".
        with pytest.raises(
            (TypeError, ValueError), match=r"(?i)real number|str|not supported"
        ):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=str(tmp_path / "out.parquet"),
                lambdas={"loss_ratio": 0.0},
                constraints={
                    "loss_ratio": {
                        "numerator": "expected_income",
                        "denominator": "volume",
                        "max": 1.0,
                    }
                },
                chunk_size=20,
            )

    def test_extra_lambda_key_error_lists_all_extras_sorted(
        self, tmp_path: Path, df_and_lambdas
    ):
        """Lambda keys not matching any constraint are rejected; the error
        names ALL extras (not just the first) and lists them sorted, so
        users debugging large lambda dicts see deterministic output that
        matches `ApplyOptimiser`'s wording."""
        df, _, _ = df_and_lambdas
        in_path = str(tmp_path / "in.parquet")
        df.write_parquet(in_path)
        with pytest.raises(ValueError) as excinfo:
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=str(tmp_path / "out.parquet"),
                # Two extras in non-alphabetical order; the error must
                # report them sorted.
                lambdas={"zzz": 1.0, "aaa": 0.5, "volume": 0.1},
                constraints={"volume": {"min_pct": 0.90}},
                chunk_size=20,
            )
        msg = str(excinfo.value)
        assert "aaa" in msg and "zzz" in msg
        # Sorted order: 'aaa' must appear before 'zzz' in the message.
        assert msg.index("aaa") < msg.index("zzz")
        # And the wording must mirror ApplyOptimiser's plural form.
        assert "Lambda keys" in msg

    def test_total_rows_not_divisible_raises(self, tmp_path: Path, df_and_lambdas):
        """Truncated parquet (rows % n_steps != 0) errors before any output is written."""
        df, lambdas, constraints = df_and_lambdas  # 40×5 = 200 rows
        df = df.head(199)  # 199 % 5 != 0
        in_path = str(tmp_path / "in.parquet")
        out_path = str(tmp_path / "out.parquet")
        df.write_parquet(in_path)
        with pytest.raises(ValueError, match=r"(?i)divisible|n_steps"):
            pc.apply_lambdas_to_parquet_chunked(
                parquet_in=in_path,
                parquet_out=out_path,
                lambdas=lambdas,
                constraints=constraints,
                chunk_size=20,
            )
        # No output should have been written.
        assert not Path(out_path).exists()
