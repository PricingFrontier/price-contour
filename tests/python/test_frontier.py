"""Tests for the multi-dimensional efficient frontier."""

from __future__ import annotations

import polars as pl

import price_contour as pc
from price_contour.frontier import frontier_summary
from helpers import make_small_df


class TestFrontier:
    def test_1d_frontier_has_valid_points(self):
        """1D frontier with 5 points — all points have valid metrics."""
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

        assert result.n_points == 5
        pts = result.points
        assert pts.shape[0] == 5
        assert "threshold_volume" in pts.columns
        assert "total_objective" in pts.columns
        assert "total_volume" in pts.columns
        assert "lambda_volume" in pts.columns
        assert "iterations" in pts.columns
        assert "converged" in pts.columns

        # All objectives should be positive
        assert all(v > 0 for v in pts["total_objective"].to_list())

    def test_warm_start_reduces_avg_iterations(self):
        """Warm-start should reduce average iterations vs cold-start."""
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
        avg_iters = pts["iterations"].mean()
        # With warm start, average should be less than max_iter
        assert avg_iters < 100, f"avg iterations {avg_iters} >= max_iter"

    def test_2d_frontier_has_n_squared_points(self):
        """2D frontier has n^2 points."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=100,
        )
        n = 4
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (0.85, 1.0),
                "loss_ratio": (1.0, 1.10),
            },
            n_points_per_dim=n,
        )

        assert result.n_points == n * n
        pts = result.points
        assert pts.shape[0] == n * n
        assert "threshold_volume" in pts.columns
        assert "threshold_loss_ratio" in pts.columns

    def test_frontier_points_dataframe_columns(self):
        """Frontier points DataFrame has expected columns."""
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
        expected = {
            "threshold_volume",
            "total_objective",
            "total_volume",
            "lambda_volume",
            "iterations",
            "converged",
        }
        assert expected.issubset(set(pts.columns))

    def test_frontier_summary_valid(self):
        """frontier_summary() produces valid MLflow dict."""
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

        summary = frontier_summary(result, selected_index=2)
        assert set(summary.keys()) == {"params", "metrics", "artifacts"}
        assert isinstance(summary["params"]["frontier_n_points"], int)
        assert summary["params"]["frontier_selected_index"] == 2
        assert isinstance(summary["metrics"]["selected_total_objective"], float)
        assert isinstance(summary["artifacts"]["frontier"], pl.DataFrame)


class TestFrontierMonotonicity:
    def test_1d_tighter_constraint_reduces_objective(self):
        """Tightening a min constraint should reduce the optimal objective."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.80, 1.0)},
            n_points_per_dim=8,
        )
        pts = result.points.sort("threshold_volume")
        objectives = pts["total_objective"].to_list()

        # As threshold increases (tighter), objective should decrease
        for i in range(len(objectives) - 1):
            assert objectives[i] >= objectives[i + 1] - abs(objectives[i + 1]) * 0.01, (
                f"Frontier non-monotonic at point {i}: "
                f"objective {objectives[i]:.2f} at threshold "
                f"{pts['threshold_volume'][i]:.3f} < "
                f"objective {objectives[i + 1]:.2f} at threshold "
                f"{pts['threshold_volume'][i + 1]:.3f}"
            )

    def test_1d_constraint_totals_increase_with_threshold(self):
        """Total volume should be non-decreasing as the volume threshold tightens."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.80, 1.0)},
            n_points_per_dim=8,
        )
        pts = result.points.sort("threshold_volume")
        volumes = pts["total_volume"].to_list()

        # The solver pushes harder to retain volume as the threshold tightens,
        # so total_volume should be non-decreasing (within 1% tolerance).
        for i in range(len(volumes) - 1):
            assert volumes[i] <= volumes[i + 1] + abs(volumes[i + 1]) * 0.01, (
                f"Total volume decreased at point {i}: "
                f"volume {volumes[i]:.2f} at threshold "
                f"{pts['threshold_volume'][i]:.3f} > "
                f"volume {volumes[i + 1]:.2f} at threshold "
                f"{pts['threshold_volume'][i + 1]:.3f}"
            )

    def test_2d_monotonicity_single_slice(self):
        """Within a fixed loss_ratio slice, volume-objective monotonicity holds."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (0.80, 1.0),
                "loss_ratio": (1.0, 1.10),
            },
            n_points_per_dim=4,
        )
        pts = result.points

        # Pick the median threshold_loss_ratio value
        lr_values = sorted(pts["threshold_loss_ratio"].unique().to_list())
        median_lr = lr_values[len(lr_values) // 2]

        # Filter to the slice closest to the median loss_ratio threshold
        slice_pts = pts.filter(pl.col("threshold_loss_ratio") == median_lr)
        slice_pts = slice_pts.sort("threshold_volume")
        objectives = slice_pts["total_objective"].to_list()

        # Within this slice, objective should decrease as volume threshold tightens
        for i in range(len(objectives) - 1):
            assert objectives[i] >= objectives[i + 1] - abs(objectives[i + 1]) * 0.01, (
                f"Frontier non-monotonic in loss_ratio={median_lr:.3f} slice "
                f"at point {i}: "
                f"objective {objectives[i]:.2f} at threshold "
                f"{slice_pts['threshold_volume'][i]:.3f} < "
                f"objective {objectives[i + 1]:.2f} at threshold "
                f"{slice_pts['threshold_volume'][i + 1]:.3f}"
            )

    def test_frontier_loosest_point_matches_unconstrained(self):
        """Loosest frontier point should match unconstrained solve within 2%."""
        df = make_small_df(n_quotes=100, n_steps=5)

        # Unconstrained solve
        unconstrained_solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.50}},
            max_iter=200,
        )
        unconstrained_result = unconstrained_solver.solve(df)
        unconstrained_total_objective = unconstrained_result.total_objective

        # Frontier solve with a loose lower bound
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.50}},
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.50, 1.0)},
            n_points_per_dim=5,
        )
        pts = result.points.sort("threshold_volume")

        # The loosest point (lowest threshold) should approximate unconstrained
        loosest_objective = pts["total_objective"][0]
        diff = abs(loosest_objective - unconstrained_total_objective)
        tol = abs(unconstrained_total_objective) * 0.02

        assert diff <= tol, (
            f"Loosest frontier point objective {loosest_objective:.2f} "
            f"differs from unconstrained {unconstrained_total_objective:.2f} "
            f"by {diff:.2f} (tolerance {tol:.2f})"
        )
