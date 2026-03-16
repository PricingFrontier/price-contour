"""Tests for min_abs and max_abs constraint modes."""

from __future__ import annotations


import price_contour as pc
from helpers import make_small_df, CONSTRAINT_RTOL


class TestAbsoluteConstraints:
    def test_min_abs_constraint(self):
        """min_abs sets an absolute minimum threshold."""
        df = make_small_df(n_quotes=50)
        # Get baseline volume via a relative-constraint solver (includes volume in grid)
        baseline_solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 1.0}},
            max_iter=1,
        )
        baseline_result = baseline_solver.solve(df)
        baseline_vol = baseline_result.baseline_constraints["volume"]

        # Set min_abs to 80% of baseline volume (absolute value)
        target = baseline_vol * 0.8
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_abs": target}},
            max_iter=200,
        )
        result = solver.solve(df)
        assert result.total_constraints["volume"] >= target * (1 - CONSTRAINT_RTOL), (
            f"volume {result.total_constraints['volume']} < {target * (1 - CONSTRAINT_RTOL)} "
            f"(target {target} with {CONSTRAINT_RTOL:.0%} tolerance)"
        )

    def test_max_abs_constraint(self):
        """max_abs sets an absolute maximum threshold."""
        df = make_small_df(n_quotes=50)
        # Get baseline loss_ratio via a relative-constraint solver
        baseline_solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": 1.0}},
            max_iter=1,
        )
        baseline_result = baseline_solver.solve(df)
        baseline_lr = baseline_result.baseline_constraints["loss_ratio"]

        # Set max_abs to 120% of baseline loss_ratio
        target = baseline_lr * 1.2
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max_abs": target}},
            max_iter=200,
        )
        result = solver.solve(df)
        # Max constraints are harder to converge on small data; use wider tolerance
        assert result.total_constraints["loss_ratio"] <= target * (
            1 + CONSTRAINT_RTOL * 3
        ), (
            f"loss_ratio {result.total_constraints['loss_ratio']} > "
            f"{target * (1 + CONSTRAINT_RTOL * 3)} "
            f"(target {target} with {CONSTRAINT_RTOL * 3:.0%} tolerance)"
        )
