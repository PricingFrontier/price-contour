"""The frame boundary on Polars 2 (DESIGN_DECISIONS §15).

* §15.1: what the bridge delivers to the native module. Columns keep their
  chunks (no rechunk on the way in), and String, Categorical and Enum
  quote ids keep their dtypes.
* §15.1: every quote-id dtype gives the same results through apply, solve
  and the chunked Parquet reader, including Parquet written the way haute
  writes it.
* §15.2: only the columns a native call reads cross the boundary, so a
  column the extension's Polars cannot import (Map, Struct, Decimal,
  Array, Int128) no longer fails a call.
* §15.3: row orders that are part of a contract hold on shuffled input.
"""

from __future__ import annotations

import decimal
from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from price_contour._price_contour import _received_columns
from helpers import make_factors, make_small_df

SUM_CONSTRAINTS = {"volume": {"min_pct": 0.95}, "loss_ratio": {"max_pct": 1.02}}
RATIO_CONSTRAINTS = {
    "lr": {"numerator": "loss_ratio", "denominator": "volume", "max_pct": 0.98},
}
LAMBDAS = {"volume": 3.0, "loss_ratio": 1.5}


def _with_quote_id_dtype(df: pl.DataFrame, dtype: str) -> pl.DataFrame:
    """``df`` with ``quote_id`` as String, Categorical or Enum. The Enum's
    categories are in reverse lexical order, so a sort by category order
    differs from the lexical order grids use (§15.4)."""
    if dtype == "String":
        return df
    if dtype == "Categorical":
        return df.with_columns(pl.col("quote_id").cast(pl.Categorical))
    categories = sorted(df["quote_id"].unique().to_list(), reverse=True)
    return df.with_columns(pl.col("quote_id").cast(pl.Enum(categories)))


def _exotic_columns(height: int) -> list[pl.Series]:
    """One column of each dtype the extension's Polars cannot import."""
    return [
        pl.Series(
            "x_map",
            [{"k": i} for i in range(height)],
            dtype=pl.Map(pl.String, pl.Int64),
        ),
        pl.Series("x_struct", [{"a": i, "b": float(i)} for i in range(height)]),
        pl.Series(
            "x_decimal",
            [decimal.Decimal(i) / 4 for i in range(height)],
            dtype=pl.Decimal(12, 2),
        ),
        pl.Series(
            "x_array", [[i, i + 1] for i in range(height)], dtype=pl.Array(pl.Int64, 2)
        ),
        pl.Series("x_int128", list(range(height)), dtype=pl.Int128),
    ]


def _with_exotic_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(_exotic_columns(df.height))


def _shuffled(df: pl.DataFrame, seed: int = 7) -> pl.DataFrame:
    """Rows in a fixed random order (quotes no longer grouped or sorted)."""
    return df.sample(fraction=1.0, shuffle=True, seed=seed)


# ---------------------------------------------------------------------------
# §15.1 What the bridge delivers
# ---------------------------------------------------------------------------


class TestBridgeDelivery:
    @pytest.mark.parametrize("dtype", ["String", "Categorical", "Enum"])
    def test_quote_id_dtype_arrives_unchanged(self, dtype: str) -> None:
        # An Enum used to arrive as Categorical (polars #25993), so the
        # Enum branch of quote_id_str_iter was never reached.
        df = _with_quote_id_dtype(make_small_df(n_quotes=3, n_steps=2), dtype)
        received = dict((name, dt) for name, dt, _ in _received_columns(df))
        assert received["quote_id"] == dtype
        assert received["scenario_index"] == "i32"
        assert received["scenario_value"] == "f32"

    def test_columns_are_not_rechunked(self) -> None:
        df = make_small_df(n_quotes=4, n_steps=3)
        multi = pl.concat([df.head(6), df.tail(6)], rechunk=False)
        assert all(multi[c].n_chunks() == 2 for c in multi.columns)
        received = _received_columns(multi)
        assert [(name, n) for name, _, n in received] == [(c, 2) for c in multi.columns]

    @pytest.mark.parametrize("column", _exotic_columns(2), ids=lambda s: s.name)
    def test_exotic_dtypes_cannot_cross_the_bridge(self, column: pl.Series) -> None:
        # Why §15.2 exists: these fail inside the bridge (some as a Rust
        # panic, hence BaseException), so they must never be sent.
        with pytest.raises(BaseException):  # noqa: B017
            _received_columns(pl.DataFrame([column]))


# ---------------------------------------------------------------------------
# §15.1 Every quote-id dtype gives the same results
# ---------------------------------------------------------------------------

RESULT_SCHEMA = {
    "quote_id": pl.String,
    "optimal_step": pl.Int32,
    "optimal_scenario_value": pl.Float32,
    "optimal_objective": pl.Float32,
    "optimal_loss_ratio": pl.Float32,
    "optimal_volume": pl.Float32,
}


@pytest.mark.parametrize("dtype", ["String", "Categorical", "Enum"])
class TestQuoteIdDtypeRoundTrip:
    """Every dtype, sorted or shuffled, gives the String result: rows and
    the grid's quote order are lexical by id (§15.4)."""

    @pytest.mark.parametrize("shuffle", [False, True], ids=["sorted", "shuffled"])
    def test_apply(self, dtype: str, shuffle: bool) -> None:
        df = make_small_df(n_quotes=30, n_steps=5)
        applier = pc.ApplyOptimiser(lambdas=LAMBDAS, constraints=SUM_CONSTRAINTS)
        expected = applier.apply(df)
        given = _with_quote_id_dtype(_shuffled(df) if shuffle else df, dtype)
        result = applier.apply(given)
        assert dict(result.dataframe.schema) == RESULT_SCHEMA
        assert result.dataframe.equals(expected.dataframe)
        assert result.total_objective == expected.total_objective
        assert result.total_constraints == expected.total_constraints

    @pytest.mark.parametrize("shuffle", [False, True], ids=["sorted", "shuffled"])
    def test_solve(self, dtype: str, shuffle: bool) -> None:
        df = make_small_df(n_quotes=30, n_steps=5)
        solver = pc.OnlineOptimiser(constraints=SUM_CONSTRAINTS, max_iter=200)
        expected = solver.solve(df)
        given = _with_quote_id_dtype(_shuffled(df) if shuffle else df, dtype)
        result = solver.solve(given)
        assert dict(result.dataframe.schema) == RESULT_SCHEMA
        assert result.dataframe.equals(expected.dataframe)
        assert result.lambdas == expected.lambdas
        builder = pc.QuoteGridBuilder(["loss_ratio", "volume"])
        builder.append(_with_quote_id_dtype(df, dtype))
        built = builder.build()
        assert result.grid.quote_ids == built.quote_ids
        assert result.grid.quote_id_fingerprint == built.quote_id_fingerprint

    def test_parquet_chunked(self, dtype: str, tmp_path: Path) -> None:
        df = make_small_df(n_quotes=30, n_steps=5)
        path = str(tmp_path / "grid.parquet")
        _with_quote_id_dtype(df, dtype).write_parquet(path)
        grid = pc.build_grid_from_parquet_chunked(
            path, ["loss_ratio", "volume"], chunk_size=20
        )
        expected = pc.QuoteGridBuilder(["loss_ratio", "volume"])
        expected.append(df)
        expected_grid = expected.build()
        assert grid.quote_ids == expected_grid.quote_ids
        assert grid.quote_id_fingerprint == expected_grid.quote_id_fingerprint
        solver = pc.OnlineOptimiser(constraints=SUM_CONSTRAINTS, max_iter=200)
        assert solver.solve(grid).dataframe.equals(
            solver.solve(expected_grid).dataframe
        )


# ---------------------------------------------------------------------------
# §15.1 Parquet written by Polars 2 the way haute writes it
# ---------------------------------------------------------------------------


class TestHauteShapedParquet:
    """haute streams the solver input through a sink: quote_id cast to
    Categorical, Int32 scenario_index, Float32 values, plus private analysis
    columns the reader does not select."""

    def test_grid_and_factor_contexts(self, tmp_path: Path) -> None:
        df = make_small_df(n_quotes=40, n_steps=5)
        grid_path = str(tmp_path / "solver_input.parquet")
        (
            df.lazy()
            .with_columns(
                pl.col("quote_id").cast(pl.Categorical),
                pl.lit("analysis").alias("__haute_note"),
                (pl.col("expected_income") * 2).cast(pl.Float64).alias("__haute_x"),
            )
            .sink_parquet(grid_path)
        )
        factors = make_factors(n_quotes=40).with_columns(
            pl.Series("quote_id", [f"Q{q:04d}" for q in range(40)]),
            pl.lit(1.5).alias("__haute_weight"),
        )
        factors_path = str(tmp_path / "factors.parquet")
        factors.lazy().sink_parquet(factors_path)

        grid = pc.build_grid_from_parquet_chunked(
            grid_path,
            ["volume"],
            chunk_size=25,
            n_steps=5,
        )
        expected = pc.QuoteGridBuilder(["volume"])
        expected.append(df)
        expected_grid = expected.build()
        assert grid.quote_ids == expected_grid.quote_ids

        contexts = pc.build_ratebook_factor_contexts_from_parquet_chunked(
            factors_path,
            [["region"], ["age_band"]],
            chunk_size=16,
            quote_id="quote_id",
            expected_quote_ids=grid.quote_ids,
            expected_n_quotes=grid.n_quotes,
        )
        assert contexts.n_quotes == 40
        assert contexts.quote_id_fingerprint == grid.quote_id_fingerprint

        opt = pc.RatebookOptimiser(
            constraints={"volume": {"min_pct": 0.95}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=2,
            max_iter=100,
        )
        from_parquet = opt.solve(grid, contexts)
        from_frames = opt.solve(df, factors)
        assert from_parquet.factor_tables == from_frames.factor_tables
        assert from_parquet.total_objective == from_frames.total_objective


# ---------------------------------------------------------------------------
# §15.2 Only the columns a native call reads cross the boundary
# ---------------------------------------------------------------------------


class TestUnreadColumnsAreNotSent:
    def test_apply(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(lambdas=LAMBDAS, constraints=SUM_CONSTRAINTS)
        result = applier.apply(_with_exotic_columns(df))
        assert result.dataframe.equals(applier.apply(df).dataframe)

    def test_apply_ratio(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(lambdas={"lr": 2.0}, constraints=RATIO_CONSTRAINTS)
        result = applier.apply(_with_exotic_columns(df))
        assert result.dataframe.equals(applier.apply(df).dataframe)
        assert result.total_constraints == applier.apply(df).total_constraints

    def test_with_explainer_columns_keeps_every_input_column(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        exotic = _with_exotic_columns(df)
        applier = pc.ApplyOptimiser(lambdas=LAMBDAS, constraints=SUM_CONSTRAINTS)
        out = applier.with_explainer_columns(exotic)
        assert out.select(exotic.columns).equals(exotic)
        appended = [c for c in out.columns if c not in exotic.columns]
        assert out.select(appended).equals(
            applier.with_explainer_columns(df).select(appended)
        )

    @pytest.mark.parametrize("constraints", [SUM_CONSTRAINTS, RATIO_CONSTRAINTS])
    def test_solve(self, constraints: dict) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(constraints=constraints, max_iter=200)
        result = solver.solve(_with_exotic_columns(df))
        expected = solver.solve(df)
        assert result.dataframe.equals(expected.dataframe)
        assert result.lambdas == expected.lambdas

    def test_frontier(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(constraints={"volume": {"min_pct": 0.95}})
        ranges = {"volume": (0.9, 0.99)}
        result = solver.frontier(
            _with_exotic_columns(df), threshold_ranges=ranges, n_points_per_dim=3
        )
        expected = solver.frontier(df, threshold_ranges=ranges, n_points_per_dim=3)
        assert result.points.equals(expected.points)

    def test_quote_grid_builder_append(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        builder = pc.QuoteGridBuilder(["volume"])
        builder.append(_with_exotic_columns(df.head(50)))
        builder.append(_with_exotic_columns(df.tail(50)))
        assert builder.n_quotes == 20
        grid = builder.build()
        expected = pc.QuoteGridBuilder(["volume"])
        expected.append(df)
        assert grid.quote_ids == expected.build().quote_ids

    def test_ratebook_factors_frame(self) -> None:
        df = make_small_df(n_quotes=40, n_steps=5)
        factors = make_factors(n_quotes=40)
        opt = pc.RatebookOptimiser(
            constraints={"volume": {"min_pct": 0.95}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=2,
            max_iter=100,
        )
        result = opt.solve(df, _with_exotic_columns(factors))
        expected = opt.solve(df, factors)
        assert result.factor_tables == expected.factor_tables

    @pytest.mark.parametrize("earlier_chunks", [0, 1])
    def test_chunk_with_no_read_column_is_rejected(self, earlier_chunks: int) -> None:
        # Projecting it would leave a frame with no rows, which the native
        # builder would skip as an empty chunk.
        builder = pc.QuoteGridBuilder(["volume"])
        if earlier_chunks:
            builder.append(make_small_df(n_quotes=4, n_steps=3))
        unrelated = pl.DataFrame({"unrelated": [1, 2, 3]}).with_columns(
            _exotic_columns(3)
        )
        with pytest.raises(ValueError, match=r"Missing column\(s\): quote_id"):
            builder.append(unrelated)

    def test_empty_chunk_with_no_read_column_is_a_no_op(self) -> None:
        builder = pc.QuoteGridBuilder(["volume"])
        builder.append(pl.DataFrame({"unrelated": pl.Series([], dtype=pl.Int64)}))
        builder.append(make_small_df(n_quotes=4, n_steps=3))
        assert builder.build().n_quotes == 4

    def test_missing_column_error_is_unchanged(self) -> None:
        # The projection leaves a missing column for the native validation,
        # which names it as before.
        df = make_small_df(n_quotes=4, n_steps=3).drop("volume")
        builder = pc.QuoteGridBuilder(["volume"])
        with pytest.raises(ValueError, match="volume"):
            builder.append(_with_exotic_columns(df))

    def test_builder_rejects_non_dataframe(self) -> None:
        builder = pc.QuoteGridBuilder(["volume"])
        with pytest.raises(TypeError, match="Expected pl.DataFrame, got dict"):
            builder.append({"quote_id": ["Q1"]})


# ---------------------------------------------------------------------------
# §15.3 Row order is stated, never assumed
# ---------------------------------------------------------------------------


class TestRowOrder:
    def test_with_explainer_columns_returns_input_order(self) -> None:
        df = _shuffled(make_small_df(n_quotes=30, n_steps=5))
        for constraints, lambdas in (
            (SUM_CONSTRAINTS, LAMBDAS),
            (RATIO_CONSTRAINTS, {"lr": 2.0}),
        ):
            applier = pc.ApplyOptimiser(lambdas=lambdas, constraints=constraints)
            out = applier.with_explainer_columns(df)
            assert out.select(df.columns).equals(df)

    @pytest.mark.parametrize("mode", ["solve", "apply"])
    def test_ratio_results_follow_grid_quote_order(self, mode: str) -> None:
        df = _shuffled(make_small_df(n_quotes=30, n_steps=5))
        if mode == "solve":
            result = pc.OnlineOptimiser(
                constraints=RATIO_CONSTRAINTS, max_iter=200
            ).solve(df)
        else:
            result = pc.ApplyOptimiser(
                lambdas={"lr": 2.0}, constraints=RATIO_CONSTRAINTS
            ).apply(df)
        frame = result.dataframe
        assert frame["quote_id"].to_list() == sorted(df["quote_id"].unique())
        assert "optimal_loss_ratio" in frame.columns

    def test_ratio_history_totals_are_sums_in_input_order(self) -> None:
        # Numerators spanning 16 orders of magnitude make the Float64 total
        # depend on summation order, so a replay that summed the chosen rows
        # in any other order would report different last bits.
        exponent = (pl.col("quote_id").str.slice(1).cast(pl.Int32) % 3 - 1) * 8
        df = _shuffled(
            make_small_df(n_quotes=60, n_steps=5).with_columns(
                (pl.col("loss_ratio") * pl.lit(10.0).pow(exponent)).cast(pl.Float32)
            )
        )
        constraints = {
            "lr": {"numerator": "loss_ratio", "denominator": "volume", "min_pct": 1.05}
        }
        result = pc.OnlineOptimiser(
            constraints=constraints, max_iter=30, record_history=True
        ).solve(df)
        history = result.history
        assert history is not None and len(history) >= 3
        # The last record carries the final pass's ratio; check the replayed
        # ones against an oracle that picks each quote's step with the native
        # apply argmax and sums the chosen rows in input order via a filter.
        order_sensitive = 0
        for record in history[:-1]:
            steps = pc.ApplyOptimiser(
                lambdas=record["lambdas"], constraints=constraints
            ).apply(df)
            chosen_step = dict(
                zip(
                    steps.dataframe["quote_id"].to_list(),
                    steps.dataframe["optimal_step"].to_list(),
                )
            )
            chosen = df.filter(
                pl.col("scenario_index")
                == pl.col("quote_id").replace_strict(chosen_step, return_dtype=pl.Int32)
            )

            def ratio(rows: pl.DataFrame) -> float:
                num = rows["loss_ratio"].cast(pl.Float64).sum()
                return num / rows["volume"].cast(pl.Float64).sum()

            assert record["total_constraints"]["lr"] == ratio(chosen)
            order_sensitive += ratio(chosen) != ratio(chosen.reverse())
        # Precondition: the fixture really is order-sensitive.
        assert order_sensitive >= 1
