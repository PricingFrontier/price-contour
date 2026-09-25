"""0.5.0 consumer contract beyond the ratebook result itself
(DESIGN_DECISIONS §13.4–13.6, §13.10, §13.12): frontier bounds and schemas,
ratebook frontier points, persistence format 2, and the single baseline rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import price_contour as pc
import pytest
from helpers import make_factors, make_small_df
from price_contour import (
    OnlineOptimiser,
    RatebookFactorContexts,
    RatebookOptimiser,
    RatebookResult,
    ResultUnavailableError,
    frontier_points_schema,
)
from price_contour._grid_utils import build_grid


def _grid(df: pl.DataFrame, constraint_columns: list[str]) -> pc.QuoteGrid:
    return build_grid(
        df,
        constraint_columns=constraint_columns,
        quote_id="quote_id",
        scenario_index="scenario_index",
        scenario_value="scenario_value",
        objective="expected_income",
    )


def _even_grid_df(n_quotes: int = 12) -> pl.DataFrame:
    """Six float32 linspace steps over [0.9, 1.1]: no exact 1.0 row."""
    sv = np.linspace(0.9, 1.1, 6, dtype=np.float32)
    rows = []
    for q in range(n_quotes):
        for k, v in enumerate(sv):
            conversion = 1.0 / (1.0 + np.exp((1.5 + q * 0.2) * (float(v) - 1.0)))
            premium = (100.0 + q) * float(v) * conversion
            rows.append(
                {
                    "quote_id": f"Q{q:03d}",
                    "scenario_index": k,
                    "scenario_value": float(v),
                    "expected_income": premium,
                    "volume": conversion,
                    "incurred": premium * (0.5 + 0.02 * q) / float(v),
                    "premium": premium,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "quote_id": pl.String,
            "scenario_index": pl.Int32,
            "scenario_value": pl.Float32,
            "expected_income": pl.Float32,
            "volume": pl.Float32,
            "incurred": pl.Float32,
            "premium": pl.Float32,
        },
    )


# ---------------------------------------------------------------------------
# §13.5 absolute bounds and §13.12 schemas on every frontier path
# ---------------------------------------------------------------------------


class TestFrontierBounds:
    def test_online_rust_frontier_bounds_and_schema(self):
        df = make_small_df(n_quotes=40)
        opt = OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}},
            max_iter=100,
        )
        frontier = opt.frontier(
            df, threshold_ranges={"volume": (0.85, 0.95)}, n_points_per_dim=3
        )
        points = frontier.points
        assert dict(points.schema) == frontier_points_schema("online", ["volume"])
        grid = _grid(df, ["volume"])
        baseline_volume = (
            pc.OnlineOptimiser(
                objective="expected_income", constraints={"volume": {"min": 0.0}}
            )
            .solve(grid)
            .baseline_constraints["volume"]
        )
        assert points["threshold_volume"].to_list() == pytest.approx([0.85, 0.9, 0.95])
        assert points["bound_volume"].to_list() == pytest.approx(
            [baseline_volume * f for f in (0.85, 0.9, 0.95)], rel=1e-12
        )

    def test_online_orchestrated_frontier_bounds_and_schema(self):
        # An unswept axis routes the sweep through the Python orchestrator.
        df = make_small_df(n_quotes=40)
        opt = OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}, "loss_ratio": {"max": 1e9}},
            max_iter=100,
        )
        frontier = opt.frontier(
            df, threshold_ranges={"volume": (0.85, 0.95)}, n_points_per_dim=2
        )
        points = frontier.points
        names = list(opt.constraints)
        assert dict(points.schema) == frontier_points_schema("online", names)
        assert points["bound_loss_ratio"].to_list() == [1e9, 1e9]
        assert points["threshold_volume"].to_list() == pytest.approx([0.85, 0.95])
        ratio = points["bound_volume"] / points["threshold_volume"]
        assert ratio[0] == pytest.approx(ratio[1], rel=1e-12)
        assert set(points["solver_path"]) == {"subgradient"}

    def test_orchestrated_frontier_rejects_parallel(self):
        df = make_small_df(n_quotes=20)
        opt = OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}, "loss_ratio": {"max": 1e9}},
        )
        with pytest.raises(ValueError, match="parallel=True is not supported"):
            opt.frontier(df, threshold_ranges={"volume": (0.85, 0.95)}, parallel=True)

    def test_zero_baseline_pct_frontier_raises(self):
        df = make_small_df(n_quotes=20).with_columns(
            pl.lit(0.0).cast(pl.Float32).alias("volume")
        )
        opt = OnlineOptimiser(
            objective="expected_income", constraints={"volume": {"min_pct": None}}
        )
        with pytest.raises(ValueError, match="baseline total is 0"):
            opt.frontier(df, threshold_ranges={"volume": (0.8, 0.9)})

    def test_frontier_summary_reports_bounds_by_name(self):
        df = make_small_df(n_quotes=30)
        opt = OnlineOptimiser(
            objective="expected_income", constraints={"volume": {"min_pct": None}}
        )
        frontier = opt.frontier(
            df, threshold_ranges={"volume": (0.85, 0.95)}, n_points_per_dim=2
        )
        metrics = pc.frontier_summary(frontier, 1)["metrics"]
        assert metrics["selected_bound_volume"] == frontier.points["bound_volume"][1]
        assert metrics["selected_threshold_volume"] == pytest.approx(0.95)

    def test_online_solve_constraint_bounds(self):
        df = make_small_df(n_quotes=30)
        result = OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}, "loss_ratio": {"max": 5.0}},
        ).solve(df)
        assert result.constraint_bounds == {
            "loss_ratio": 5.0,
            "volume": result.baseline_constraints["volume"] * 0.9,
        }
        assert list(result.constraint_bounds) == list(result.total_constraints)

    def test_unknown_frontier_warm_start_lambda_raises(self):
        df = make_small_df(n_quotes=20)
        opt = OnlineOptimiser(
            objective="expected_income", constraints={"volume": {"min_pct": None}}
        )
        with pytest.raises(ValueError, match="unknown constraint"):
            opt.frontier(
                df,
                threshold_ranges={"volume": (0.85, 0.95)},
                initial_lambdas={"volum": 0.1},
            )


# ---------------------------------------------------------------------------
# §13.4 ratebook frontier
# ---------------------------------------------------------------------------


def _ratebook_frontier(threshold_ranges, constraints):
    df = make_small_df(n_quotes=48)
    factors = make_factors(48)
    opt = RatebookOptimiser(
        objective="expected_income",
        constraints=constraints,
        factor_columns=[["region"], ["age_band"]],
        max_cd_iterations=2,
        max_iter=60,
    )
    grid = _grid(df, [n for n in constraints])
    contexts = RatebookFactorContexts.from_dataframe(
        factors,
        [["region"], ["age_band"]],
        quote_id=None,
        expected_quote_ids=grid.quote_ids,
    )
    frontier = opt.frontier(
        grid, contexts, threshold_ranges=threshold_ranges, n_points_per_dim=3
    )
    return opt, grid, contexts, frontier


class TestRatebookFrontier:
    @pytest.mark.parametrize(
        ("constraints", "ranges"),
        [
            ({"volume": {"min_pct": None}}, {"volume": (0.85, 0.95)}),
            ({"volume": {"min": None}}, {"volume": (20.0, 24.0)}),
            (
                {"volume": {"min_pct": None}, "loss_ratio": {"max": 1e9}},
                {"volume": (0.85, 0.95)},
            ),
        ],
    )
    def test_evaluate_reproduces_every_row_exactly(self, constraints, ranges):
        opt, grid, contexts, frontier = _ratebook_frontier(ranges, constraints)
        points = frontier.points
        assert len(frontier.factor_tables) == frontier.n_points == points.height
        names = list(constraints)
        for i in range(frontier.n_points):
            evaluation = opt.evaluate(grid, contexts, frontier.point_factor_tables(i))
            assert evaluation.total_objective == points["total_objective"][i]
            for name in names:
                assert evaluation.total_constraints[name] == points[f"total_{name}"][i]
            assert evaluation.n_quotes_clamped_low == points["n_quotes_clamped_low"][i]
            assert (
                evaluation.n_quotes_clamped_high == points["n_quotes_clamped_high"][i]
            )

    def test_schema_and_bounds(self):
        constraints = {"volume": {"min_pct": None}, "loss_ratio": {"max": 1e9}}
        opt, grid, contexts, frontier = _ratebook_frontier(
            {"volume": (0.85, 0.95)}, constraints
        )
        points = frontier.points
        assert dict(points.schema) == frontier_points_schema(
            "ratebook", list(constraints)
        )
        baseline = opt.evaluate(
            grid, contexts, frontier.point_factor_tables(0)
        ).baseline_constraints["volume"]
        assert points["bound_volume"].to_list() == [
            baseline * t for t in points["threshold_volume"]
        ]
        assert points["bound_loss_ratio"].to_list() == [1e9] * 3
        assert frontier.n_converged == int(points["converged"].sum())

    def test_sweep_does_not_retain_per_point_results(self, monkeypatch):
        """Each point's result (and with it the per-quote evaluation, which
        only the result references) is released once its row is recorded,
        before the next point is solved, so a sweep's memory does not grow
        with its point count. The result is tracked because PyO3 classes do
        not support weak references."""
        import gc
        import weakref

        refs: list[weakref.ref] = []
        alive_before_each_solve: list[int] = []
        real_solve = RatebookOptimiser.solve

        def tracking_solve(self, *args, **kwargs):
            gc.collect()
            alive_before_each_solve.append(sum(ref() is not None for ref in refs))
            result = real_solve(self, *args, **kwargs)
            refs.append(weakref.ref(result))
            return result

        monkeypatch.setattr(RatebookOptimiser, "solve", tracking_solve)
        _, _, _, frontier = _ratebook_frontier(
            {"volume": (0.85, 0.95)}, {"volume": {"min_pct": None}}
        )
        gc.collect()
        assert frontier.n_points == 3
        assert alive_before_each_solve == [0, 0, 0]
        assert all(ref() is None for ref in refs)

    def test_point_factor_tables_out_of_range(self):
        _, _, _, frontier = _ratebook_frontier(
            {"volume": (0.85, 0.95)}, {"volume": {"min_pct": None}}
        )
        with pytest.raises(IndexError):
            frontier.point_factor_tables(frontier.n_points)

    def test_parallel_raises(self):
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}},
            factor_columns=[["region"]],
        )
        with pytest.raises(ValueError, match="parallel=True is not supported"):
            opt.frontier(
                make_small_df(20),
                make_factors(20),
                threshold_ranges={"volume": (0.85, 0.95)},
                parallel=True,
            )


# ---------------------------------------------------------------------------
# §13.6 one baseline rule
# ---------------------------------------------------------------------------


class TestBaselineRule:
    def test_grid_baseline_on_even_float32_grid(self):
        grid = _grid(_even_grid_df(), ["volume"])
        sv = np.asarray(grid.scenario_values, dtype=np.float32)
        expected = int(np.argmin(np.abs(sv - np.float32(1.0))))
        assert grid.baseline_step == expected
        assert grid.baseline_scenario_value == sv[expected]

    def test_ratio_baseline_uses_the_nearest_step(self):
        df = _even_grid_df()
        grid = _grid(df, ["volume"])
        at_baseline = df.filter(pl.col("scenario_index") == grid.baseline_step)
        expected_lr = float(
            at_baseline["incurred"].cast(pl.Float64).sum()
            / at_baseline["premium"].cast(pl.Float64).sum()
        )
        result = OnlineOptimiser(
            objective="expected_income",
            constraints={
                "lr": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": 1.1,
                }
            },
            max_iter=80,
        ).solve(df)
        assert result.baseline_constraints["lr"] == pytest.approx(expected_lr, rel=1e-9)
        assert result.constraint_bounds["lr"] == pytest.approx(
            expected_lr * 1.1, rel=1e-9
        )

    def test_ratio_absolute_baseline_is_finite_without_exact_one(self):
        df = _even_grid_df()
        result = OnlineOptimiser(
            objective="expected_income",
            constraints={
                "lr": {"numerator": "incurred", "denominator": "premium", "max": 0.9}
            },
            max_iter=80,
        ).solve(df)
        assert np.isfinite(result.baseline_constraints["lr"])
        assert result.constraint_bounds["lr"] == 0.9

    @pytest.mark.parametrize("key", ["max_pct", "min_pct"])
    def test_zero_baseline_ratio_pct_raises(self, key):
        # A zero baseline numerator makes the baseline ratio 0, so a fraction
        # of it is undefined, like a zero sum baseline.
        df = _even_grid_df().with_columns(
            pl.lit(0.0).cast(pl.Float32).alias("incurred")
        )
        opt = OnlineOptimiser(
            objective="expected_income",
            constraints={
                "lr": {"numerator": "incurred", "denominator": "premium", key: 0.9}
            },
        )
        with pytest.raises(ValueError, match="baseline ratio is 0"):
            opt.solve(df)

    def test_quote_missing_its_baseline_row_raises(self):
        df = _even_grid_df()
        baseline_index = _grid(df, []).baseline_step
        broken = df.filter(
            ~(
                (pl.col("quote_id") == "Q000")
                & (pl.col("scenario_index") == baseline_index)
            )
        )
        with pytest.raises(ValueError, match="no row at the baseline step"):
            pc.solver._baseline_rows(
                broken,
                quote_id_col="quote_id",
                scenario_index_col="scenario_index",
                scenario_value_col="scenario_value",
            )


# ---------------------------------------------------------------------------
# §13.10 persistence
# ---------------------------------------------------------------------------


def _composite_result():
    df = make_small_df(n_quotes=40)
    factors = pl.DataFrame(
        {
            "region": [["N:E", "S"][i % 2] for i in range(40)],
            "age": [["young", "old"][i % 3 % 2] for i in range(40)],
        }
    )
    opt = RatebookOptimiser(
        objective="expected_income",
        constraints={"volume": {"min_pct": 0.9}},
        factor_columns=[["region", "age"]],
        max_cd_iterations=2,
    )
    return opt, df, factors, opt.solve(df, factors)


class TestPersistence:
    def test_round_trip_keeps_every_field(self, tmp_path: Path):
        _, _, _, result = _composite_result()
        result.save(tmp_path)
        loaded = RatebookResult.load(tmp_path)
        assert loaded == result
        assert loaded.per_factor_results == result.per_factor_results
        assert loaded.constraint_bounds == result.constraint_bounds
        assert loaded.scenario_values == result.scenario_values
        assert loaded.baseline_scenario_value == result.baseline_scenario_value
        assert (
            loaded.n_quotes,
            loaded.n_quotes_clamped_low,
            loaded.n_quotes_clamped_high,
        ) == (
            result.n_quotes,
            result.n_quotes_clamped_low,
            result.n_quotes_clamped_high,
        )

    def test_loaded_result_has_no_quote_results_but_evaluate_reproduces_them(
        self, tmp_path: Path
    ):
        # A level containing ':' must survive the round trip unchanged.
        opt, df, factors, result = _composite_result()
        result.save(tmp_path)
        loaded = RatebookResult.load(tmp_path)
        with pytest.raises(ResultUnavailableError, match="evaluate"):
            _ = loaded.quote_results
        assert not isinstance(ResultUnavailableError("x"), AttributeError)
        evaluation = opt.evaluate(df, factors, loaded.factor_tables)
        assert evaluation.quote_results.equals(result.quote_results)
        assert evaluation.total_objective == result.total_objective

    def test_missing_factor_file_raises(self, tmp_path: Path):
        _, _, _, result = _composite_result()
        result.save(tmp_path)
        (tmp_path / "region_age.json").unlink()
        with pytest.raises(FileNotFoundError, match="region:age"):
            RatebookResult.load(tmp_path)

    def test_unknown_config_key_raises(self, tmp_path: Path):
        _, _, _, result = _composite_result()
        result.save(tmp_path)
        config = json.loads((tmp_path / "config.json").read_text())
        config["surprise"] = 1
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="surprise"):
            RatebookResult.load(tmp_path)

    def test_format_1_loads_and_new_fields_raise(self, tmp_path: Path):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "lambdas": {"volume": 0.1},
                    "constraints": {"volume": 20.0},
                    "baseline_constraints": {"volume": 22.0},
                    "total_objective": 100.0,
                    "baseline_objective": 90.0,
                    "cd_iterations": 2,
                    "converged": True,
                    "clamp_rate": 0.1,
                    "factor_order": ["region:age"],
                }
            )
        )
        (tmp_path / "region_age.json").write_text(
            json.dumps({"columns": ["region", "age"], "table": {"N:young": 1.05}})
        )
        loaded = RatebookResult.load(tmp_path)
        assert loaded.factor_tables == {"region:age": {"N\x1fyoung": 1.05}}
        for name in ("per_factor_results", "constraint_bounds", "n_quotes_clamped_low"):
            with pytest.raises(ResultUnavailableError, match="format-1"):
                getattr(loaded, name)

    def test_format_1_ambiguous_composite_label_raises(self, tmp_path: Path):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "lambdas": {},
                    "constraints": {},
                    "baseline_constraints": {},
                    "total_objective": 1.0,
                    "baseline_objective": 1.0,
                    "cd_iterations": 1,
                    "converged": True,
                    "clamp_rate": 0.0,
                    "factor_order": ["region:age"],
                }
            )
        )
        (tmp_path / "region_age.json").write_text(
            json.dumps({"columns": ["region", "age"], "table": {"N:E:young": 1.0}})
        )
        with pytest.raises(ValueError, match="ambiguous"):
            RatebookResult.load(tmp_path)

    def test_factor_filename_collision_raises(self, tmp_path: Path):
        _, _, _, result = _composite_result()
        result.factor_tables = {"a:b": {"x\x1fy": 1.0}, "a_b": {"z": 1.0}}
        with pytest.raises(ValueError, match="same file name"):
            result.save(tmp_path)
