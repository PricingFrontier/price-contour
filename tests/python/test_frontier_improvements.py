"""Tests for the four frontier improvements from FRONTIER_IMPROVEMENTS.md.

1. Frontier warm-start from prior solve
2. Single-pass apply on a QuoteGrid (apply_from_grid)
3. Scenario value stats per frontier point
4. Ratebook frontier
"""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from price_contour.apply import apply_from_grid
from price_contour.ratebook import RatebookOptimiser
from helpers import make_small_df, make_factors


# ---------------------------------------------------------------------------
# 1. Frontier warm-start from prior solve
# ---------------------------------------------------------------------------


class TestFrontierWarmStart:
    """Tests for initial_lambdas parameter on frontier()."""

    def test_frontier_accepts_initial_lambdas(self):
        """frontier() accepts initial_lambdas without error."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        solve_result = solver.solve(df)

        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
            initial_lambdas=solve_result.lambdas,
        )

        assert result.n_points == 3
        assert all(v > 0 for v in result.points["total_objective"].to_list())

    def test_warm_start_reduces_first_point_iterations(self):
        """Warm-starting the frontier from solve lambdas should reduce
        total iterations compared to cold-start."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        solve_result = solver.solve(df)

        # Build grid once so both calls use identical data
        grid = solve_result.grid

        cold = solver.frontier(
            grid,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
        )
        warm = solver.frontier(
            grid,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
            initial_lambdas=solve_result.lambdas,
        )

        cold_total = cold.points["iterations"].sum()
        warm_total = warm.points["iterations"].sum()

        # Warm-start should use no more total iterations
        assert warm_total <= cold_total, (
            f"warm-start ({warm_total}) should use <= iterations than cold ({cold_total})"
        )

    def test_warm_start_none_is_default(self):
        """Passing initial_lambdas=None behaves the same as omitting it."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )

        result_default = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
        )
        result_none = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
            initial_lambdas=None,
        )

        # Both should have same shape and similar results
        assert result_default.n_points == result_none.n_points

    def test_warm_start_results_match_objectives(self):
        """Warm-started frontier produces the same total_objective values
        (within tolerance) as cold-started frontier."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        solve_result = solver.solve(df)
        grid = solve_result.grid

        cold = solver.frontier(
            grid,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
        )
        warm = solver.frontier(
            grid,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
            initial_lambdas=solve_result.lambdas,
        )

        cold_objs = cold.points.sort("threshold_volume")["total_objective"].to_list()
        warm_objs = warm.points.sort("threshold_volume")["total_objective"].to_list()

        for i, (c, w) in enumerate(zip(cold_objs, warm_objs)):
            scale = abs(c) or 1.0
            assert abs(c - w) / scale < 0.05, (
                f"Point {i}: cold={c:.2f} vs warm={w:.2f} differ by >{5}%"
            )


# ---------------------------------------------------------------------------
# 2. Single-pass apply on a QuoteGrid (apply_from_grid)
# ---------------------------------------------------------------------------


class TestApplyFromGrid:
    """Tests for apply_from_grid() — single-pass apply on existing QuoteGrid."""

    def test_apply_from_grid_matches_full_apply(self):
        """apply_from_grid with solve lambdas produces same result as ApplyOptimiser."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        solve_result = solver.solve(df)
        grid = solve_result.grid

        # apply_from_grid
        grid_result = apply_from_grid(
            grid,
            lambdas=solve_result.lambdas,
            constraints={"volume": {"min_pct": 0.90}},
        )

        # ApplyOptimiser (re-ingests DataFrame)
        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        df_result = applier.apply(df)

        assert abs(grid_result.total_objective - df_result.total_objective) < 1e-3
        for name in ["volume"]:
            assert (
                abs(
                    grid_result.total_constraints[name]
                    - df_result.total_constraints[name]
                )
                < 1e-3
            )

    def test_apply_from_grid_matches_solve(self):
        """apply_from_grid with solve lambdas reproduces the solve's total_objective."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        solve_result = solver.solve(df)

        grid_result = apply_from_grid(
            solve_result.grid,
            lambdas=solve_result.lambdas,
            constraints={"volume": {"min_pct": 0.90}},
        )

        assert abs(grid_result.total_objective - solve_result.total_objective) < 1e-3

    def test_apply_from_grid_has_baselines(self):
        """Result includes baseline metrics."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        solve_result = solver.solve(df)

        result = apply_from_grid(
            solve_result.grid,
            lambdas=solve_result.lambdas,
            constraints={"volume": {"min_pct": 0.90}},
        )

        assert result.baseline_objective > 0
        assert "volume" in result.baseline_constraints
        assert result.baseline_constraints["volume"] > 0

    def test_apply_from_grid_has_dataframe(self):
        """Result includes per-quote dataframe."""
        n_quotes = 50
        df = make_small_df(n_quotes=n_quotes, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        solve_result = solver.solve(df)

        result = apply_from_grid(
            solve_result.grid,
            lambdas=solve_result.lambdas,
            constraints={"volume": {"min_pct": 0.90}},
        )

        out = result.dataframe
        assert out.shape[0] == n_quotes
        assert "quote_id" in out.columns
        assert "optimal_step" in out.columns
        assert "optimal_scenario_value" in out.columns

    def test_apply_from_grid_zero_lambdas(self):
        """Zero lambdas picks max-objective step (unconstrained)."""
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        solve_result = solver.solve(df)

        result = apply_from_grid(
            solve_result.grid,
            lambdas={"volume": 0.0},
            constraints={"volume": {"min_pct": 0.90}},
        )

        out = result.dataframe
        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx, f"quote {qid}"

    def test_apply_from_grid_two_constraints(self):
        """Works with multiple constraints."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.92},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=200,
        )
        solve_result = solver.solve(df)

        result = apply_from_grid(
            solve_result.grid,
            lambdas=solve_result.lambdas,
            constraints={
                "volume": {"min_pct": 0.92},
                "loss_ratio": {"max_pct": 1.05},
            },
        )

        assert "volume" in result.total_constraints
        assert "loss_ratio" in result.total_constraints
        assert "volume" in result.lambdas
        assert "loss_ratio" in result.lambdas

    def test_apply_from_grid_in_public_api(self):
        """apply_from_grid is importable from the top-level package."""
        assert hasattr(pc, "apply_from_grid")


# ---------------------------------------------------------------------------
# 3. Scenario value stats per frontier point
# ---------------------------------------------------------------------------


class TestFrontierScenarioValueStats:
    """Tests for sv_* columns in frontier points DataFrame."""

    def test_sv_columns_present(self):
        """Frontier points DataFrame has sv_* stat columns."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
        )
        pts = result.points
        expected_cols = {
            "sv_mean",
            "sv_std",
            "sv_min",
            "sv_p5",
            "sv_p25",
            "sv_median",
            "sv_p75",
            "sv_p95",
            "sv_max",
            "sv_pct_increase",
            "sv_pct_decrease",
        }
        assert expected_cols.issubset(set(pts.columns)), (
            f"Missing columns: {expected_cols - set(pts.columns)}"
        )

    def test_sv_percentiles_ordered(self):
        """Scenario value percentiles are monotonically ordered per point."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
        )
        pts = result.points

        for row in pts.iter_rows(named=True):
            assert row["sv_min"] <= row["sv_p5"], "min > p5"
            assert row["sv_p5"] <= row["sv_p25"], "p5 > p25"
            assert row["sv_p25"] <= row["sv_median"], "p25 > median"
            assert row["sv_median"] <= row["sv_p75"], "median > p75"
            assert row["sv_p75"] <= row["sv_p95"], "p75 > p95"
            assert row["sv_p95"] <= row["sv_max"], "p95 > max"

    def test_sv_mean_within_range(self):
        """sv_mean is between sv_min and sv_max."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
        )
        pts = result.points

        for row in pts.iter_rows(named=True):
            assert row["sv_min"] <= row["sv_mean"] <= row["sv_max"], (
                f"mean={row['sv_mean']:.4f} outside [{row['sv_min']:.4f}, {row['sv_max']:.4f}]"
            )

    def test_sv_pct_sum_at_most_one(self):
        """pct_increase + pct_decrease <= 1.0 (some quotes may have sv == 1.0)."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
        )
        pts = result.points

        for row in pts.iter_rows(named=True):
            total = row["sv_pct_increase"] + row["sv_pct_decrease"]
            assert total <= 1.0 + 1e-9, f"pct_increase + pct_decrease = {total} > 1.0"

    def test_sv_std_non_negative(self):
        """sv_std is non-negative."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
        )
        pts = result.points
        assert all(v >= 0.0 for v in pts["sv_std"].to_list())

    def test_2d_frontier_has_sv_stats(self):
        """SV stats work with multi-dimensional frontier."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (0.85, 1.0),
                "loss_ratio": (1.0, 1.10),
            },
            n_points_per_dim=3,
        )
        pts = result.points
        assert "sv_mean" in pts.columns
        assert pts.shape[0] == 9  # 3x3

    def test_frontier_summary_still_works(self):
        """frontier_summary() still works with new sv columns."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
        )

        summary = pc.frontier_summary(result, selected_index=2)
        assert set(summary.keys()) == {"params", "metrics", "artifacts"}


# ---------------------------------------------------------------------------
# 4. Ratebook frontier
# ---------------------------------------------------------------------------


class TestRatebookFrontier:
    """Tests for RatebookOptimiser.frontier()."""

    def test_ratebook_frontier_basic(self):
        """Ratebook frontier produces valid points."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )

        result = opt.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
        )

        assert result.n_points == 3
        pts = result.points
        assert "threshold_volume" in pts.columns
        assert "total_objective" in pts.columns
        assert "total_volume" in pts.columns
        assert "lambda_volume" in pts.columns
        assert "iterations" in pts.columns
        assert "converged" in pts.columns
        assert all(v > 0 for v in pts["total_objective"].to_list())

    def test_ratebook_frontier_is_frontier_result_like(self):
        """Ratebook frontier result has the same interface as FrontierResult."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )

        result = opt.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
        )

        # Duck-type check: has .points, .n_points, .constraint_names
        assert isinstance(result.points, pl.DataFrame)
        assert isinstance(result.n_points, int)
        assert isinstance(result.constraint_names, list)
        assert result.constraint_names == ["volume"]

    def test_ratebook_frontier_monotonicity(self):
        """Tighter volume constraint should reduce objective."""
        n = 100
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=100,
        )

        result = opt.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.80, 1.0)},
            n_points_per_dim=5,
        )

        pts = result.points.sort("threshold_volume")
        objectives = pts["total_objective"].to_list()

        # As threshold increases (tighter), objective should generally decrease
        # Allow 2% tolerance for solver noise
        for i in range(len(objectives) - 1):
            assert objectives[i] >= objectives[i + 1] - abs(objectives[i + 1]) * 0.02, (
                f"Ratebook frontier non-monotonic at point {i}: "
                f"{objectives[i]:.2f} vs {objectives[i + 1]:.2f}"
            )

    def test_ratebook_frontier_warm_start(self):
        """Ratebook frontier accepts initial_lambdas."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )

        solve_result = opt.solve(df, factors)

        result = opt.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
            initial_lambdas=solve_result.lambdas,
        )

        assert result.n_points == 3

    def test_ratebook_frontier_has_clamp_rate(self):
        """Ratebook frontier points include clamp_rate column."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )

        result = opt.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
        )

        assert "clamp_rate" in result.points.columns

    def test_ratebook_frontier_missing_range_raises(self):
        """Missing threshold_range for a constraint raises ValueError."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )

        with pytest.raises(ValueError, match="No threshold_range"):
            opt.frontier(
                df,
                factors,
                threshold_ranges={},  # missing "volume"
                n_points_per_dim=3,
            )

    def test_ratebook_frontier_no_constraints_raises(self):
        """Frontier with no constraints raises ValueError."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )

        with pytest.raises(ValueError, match="at least one constraint"):
            opt.frontier(
                df,
                factors,
                threshold_ranges={"volume": (0.85, 1.0)},
                n_points_per_dim=3,
            )


class TestRatebookSolveWarmStart:
    """Tests for the lambdas parameter on RatebookOptimiser.solve()."""

    def test_solve_accepts_lambdas(self):
        """solve() accepts lambdas for warm-starting."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )

        result1 = opt.solve(df, factors)
        result2 = opt.solve(df, factors, lambdas=result1.lambdas)

        assert result2.total_objective > 0


class TestNNOrder:
    """Tests for the Python nearest-neighbour ordering."""

    def test_visits_all_points(self):
        """NN ordering visits every point exactly once."""
        from price_contour.ratebook import _nn_order

        points = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.0, 1.0]]
        ranges = [(0.0, 1.0), (0.0, 1.0)]
        order = _nn_order(points, ranges)

        assert len(order) == 4
        assert sorted(order) == [0, 1, 2, 3]

    def test_empty_points(self):
        """NN ordering on empty list returns empty."""
        from price_contour.ratebook import _nn_order

        assert _nn_order([], []) == []

    def test_single_point(self):
        """NN ordering on single point returns [0]."""
        from price_contour.ratebook import _nn_order

        assert _nn_order([[1.0]], [(0.0, 2.0)]) == [0]
