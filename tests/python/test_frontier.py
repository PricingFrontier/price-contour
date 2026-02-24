"""Tests for the multi-dimensional efficient frontier."""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc
from price_contour.frontier import frontier_summary


def _make_small_df(n_quotes: int = 50, n_steps: int = 5) -> pl.DataFrame:
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


class TestFrontier:
    def test_1d_frontier_has_valid_points(self):
        """1D frontier with 5 points — all points have valid metrics."""
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
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
        df = _make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
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
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 0.90},
                "loss_ratio": {"max": 1.05},
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
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
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
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
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
