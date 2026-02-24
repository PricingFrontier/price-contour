"""Tests for the grouped Lagrangian solver."""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc
from price_contour._price_contour import (
    QuoteGrid,
    QuoteGridBuilder,
    GroupedSolveResult,
    solve_grouped_py,
)


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


def _build_grid(df: pl.DataFrame) -> QuoteGrid:
    builder = QuoteGridBuilder(["volume"])
    builder.append(df)
    return builder.build()


class TestGroupedSolver:
    def test_all_distinct_groups_matches_online(self):
        """N groups with residuals=1.0 and candidates=scenario_values ~ online result."""
        n = 50
        df = _make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        # Each quote is its own group
        group_labels = [f"G{i}" for i in range(n)]
        residuals = [1.0] * n
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            group_labels=group_labels,
            residuals=residuals,
            candidates=candidates,
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )

        # Compare with online solver
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        online_result = solver.solve(df)

        # Should be in the same ballpark
        scale = abs(online_result.total_objective) or 1.0
        diff = abs(result.total_objective - online_result.total_objective) / scale
        assert diff < 0.05, (
            f"grouped vs online objective diff too large: "
            f"{result.total_objective} vs {online_result.total_objective}"
        )

    def test_single_group_picks_one_factor(self):
        """All quotes in one group: a single factor for the whole portfolio."""
        n = 20
        df = _make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        group_labels = ["ALL"] * n
        residuals = [1.0] * n
        candidates = [0.8 + 0.02 * i for i in range(21)]

        result = solve_grouped_py(
            grid,
            group_labels=group_labels,
            residuals=residuals,
            candidates=candidates,
            max_iter=1,
        )

        assert len(result.optimal_factor_values) == 1
        assert "ALL" in result.optimal_factor_values
        fv = result.optimal_factor_values["ALL"]
        assert 0.8 <= fv <= 1.2

    def test_known_two_group_problem(self):
        """3-quote, 2-group problem with known structure."""
        n = 3
        df = _make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        # Group 0: quotes 0,1; Group 1: quote 2
        group_labels = ["A", "A", "B"]
        residuals = [1.0, 1.0, 1.0]
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            group_labels=group_labels,
            residuals=residuals,
            candidates=candidates,
            max_iter=1,
        )

        assert len(result.optimal_factor_values) == 2
        assert "A" in result.optimal_factor_values
        assert "B" in result.optimal_factor_values
        assert len(result.optimal_steps_per_quote) == n

    def test_clamp_rate_positive_with_extreme_residuals(self):
        """Extreme residuals push targets outside grid."""
        n = 20
        df = _make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        group_labels = [f"G{i}" for i in range(n)]
        residuals = [3.0] * n  # will push targets above grid max
        candidates = [1.0]

        result = solve_grouped_py(
            grid,
            group_labels=group_labels,
            residuals=residuals,
            candidates=candidates,
            max_iter=1,
        )

        assert result.clamp_rate > 0.0

    def test_clamp_rate_zero_with_normal_residuals(self):
        """Normal residuals within grid range."""
        n = 20
        df = _make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        group_labels = [f"G{i}" for i in range(n)]
        residuals = [1.0] * n
        candidates = [1.0]

        result = solve_grouped_py(
            grid,
            group_labels=group_labels,
            residuals=residuals,
            candidates=candidates,
            max_iter=1,
        )

        assert result.clamp_rate == 0.0

    def test_result_properties(self):
        """GroupedSolveResult exposes expected getters."""
        n = 20
        df = _make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        group_labels = ["A"] * 10 + ["B"] * 10
        residuals = [1.0] * n
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            group_labels=group_labels,
            residuals=residuals,
            candidates=candidates,
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )

        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert isinstance(result.total_objective, float)
        assert isinstance(result.lambdas, dict)
        assert isinstance(result.total_constraints, dict)
        assert isinstance(result.baseline_objective, float)
        assert isinstance(result.baseline_constraints, dict)
        assert isinstance(result.clamp_rate, float)
        assert isinstance(result.group_labels, list)
        assert set(result.group_labels) == {"A", "B"}
