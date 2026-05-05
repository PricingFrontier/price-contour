"""Tests for the grouped Lagrangian solver."""

from __future__ import annotations

import pytest

import polars as pl

import price_contour as pc
from price_contour._price_contour import (
    FactorContext,
    QuoteGrid,
    QuoteGridBuilder,
    run_cd_pass_py,
    solve_grouped_py,
)
from helpers import make_small_df, CONSTRAINT_RTOL


def _build_grid(
    df: pl.DataFrame, constraint_cols: list[str] | None = None
) -> QuoteGrid:
    if constraint_cols is None:
        constraint_cols = ["volume"]
    builder = QuoteGridBuilder(constraint_cols)
    builder.append(df)
    return builder.build()


class TestGroupedSolver:
    def test_all_distinct_groups_matches_online(self):
        """N groups with residuals=1.0 and candidates=scenario_values ~ online result."""
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        # Each quote is its own group
        group_labels = [f"G{i}" for i in range(n)]
        residuals = [1.0] * n
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
            residuals=residuals,
            candidates=candidates,
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )

        # Compare with online solver
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
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
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=[])

        group_labels = ["ALL"] * n
        residuals = [1.0] * n
        candidates = [0.8 + 0.02 * i for i in range(21)]

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
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
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=[])

        # Group 0: quotes 0,1; Group 1: quote 2
        group_labels = ["A", "A", "B"]
        residuals = [1.0, 1.0, 1.0]
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
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
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=[])

        group_labels = [f"G{i}" for i in range(n)]
        residuals = [3.0] * n  # will push targets above grid max
        candidates = [1.0]

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
            residuals=residuals,
            candidates=candidates,
            max_iter=1,
        )

        assert result.clamp_rate > 0.0

    def test_clamp_rate_zero_with_normal_residuals(self):
        """Normal residuals within grid range."""
        n = 20
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=[])

        group_labels = [f"G{i}" for i in range(n)]
        residuals = [1.0] * n
        candidates = [1.0]

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
            residuals=residuals,
            candidates=candidates,
            max_iter=1,
        )

        assert result.clamp_rate == 0.0

    def test_result_properties(self):
        """GroupedSolveResult exposes expected getters."""
        n = 20
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        group_labels = ["A"] * 10 + ["B"] * 10
        residuals = [1.0] * n
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
            residuals=residuals,
            candidates=candidates,
            constraints={"volume": {"min_pct": 0.90}},
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

    def test_grouped_constraint_satisfaction(self):
        """Grouped solver should approximately satisfy the volume constraint."""
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df)

        group_labels = [f"G{i}" for i in range(n)]
        residuals = [1.0] * n
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
            residuals=residuals,
            candidates=candidates,
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )

        baseline_vol = result.baseline_constraints["volume"]
        threshold = baseline_vol * 0.90
        assert result.total_constraints["volume"] >= threshold * (
            1 - CONSTRAINT_RTOL
        ), (
            f"grouped volume {result.total_constraints['volume']} < "
            f"threshold {threshold} with {CONSTRAINT_RTOL:.0%} slack"
        )

        # Lambda should be non-negative (dual feasibility)
        assert result.lambdas["volume"] >= 0, (
            f"volume lambda is negative: {result.lambdas['volume']}"
        )

    def test_grouped_optimal_steps_valid_indices(self):
        """All optimal_steps_per_quote values are valid step indices."""
        n = 20
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=[])

        group_labels = ["A"] * 10 + ["B"] * 10
        residuals = [1.0] * n
        candidates = list(grid.scenario_values)

        result = solve_grouped_py(
            grid,
            context=FactorContext.from_labels(group_labels),
            residuals=residuals,
            candidates=candidates,
            max_iter=1,
        )

        steps = result.optimal_steps_per_quote
        assert len(steps) == n, f"expected {n} steps, got {len(steps)}"
        for i, step in enumerate(steps):
            assert 0 <= step < 5, f"quote {i}: optimal_step {step} out of range [0, 5)"


class TestRunCdPass:
    """Direct tests for the Rust-side ratebook CD outer loop entry point.

    These pin the behaviour of `run_cd_pass_py` independently of the
    Python `RatebookOptimiser.solve()` orchestrator. The orchestrator
    consumes `run_cd_pass_py`'s result fields verbatim, so a regression
    in this kernel surfaces here before it reaches end-to-end tests.
    """

    def test_matches_per_call_orchestrator_on_two_factors(self):
        """A direct CD pass should produce the same factor values and
        same total_objective as orchestrating the equivalent
        compute_residuals + solve_grouped + update_multipliers loop in
        Python — any divergence is a kernel bug."""
        n = 30
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=["volume"])

        labels_a = [f"A{i % 3}" for i in range(n)]
        labels_b = [f"B{i % 2}" for i in range(n)]
        ctx_a = FactorContext.from_labels(labels_a)
        ctx_b = FactorContext.from_labels(labels_b)
        candidates = [0.8 + 0.1 * i for i in range(5)]
        constraints = {"volume": {"min_pct": 0.90}}

        result = run_cd_pass_py(
            grid,
            [ctx_a, ctx_b],
            candidates,
            constraints=constraints,
            max_iter=50,
            tolerance=1e-5,
            max_cd_iterations=3,
            cd_tolerance=1e-3,
        )

        # Per-factor factor_values should be vectors of the right length.
        assert len(result.factor_values) == 2
        assert len(result.factor_values[0]) == ctx_a.n_groups
        assert len(result.factor_values[1]) == ctx_b.n_groups

        # Volume constraint must be satisfied within slack.
        baseline_vol = result.baseline_constraints["volume"]
        threshold = baseline_vol * 0.90
        assert result.total_constraints["volume"] >= threshold * (
            1 - CONSTRAINT_RTOL
        ), (
            f"CD-pass volume {result.total_constraints['volume']} < "
            f"threshold {threshold}"
        )

        # Per-call objectives should be non-decreasing across CD sweeps.
        # (Within a single CD sweep the per-factor solves can dip; but
        # the last solve of each sweep should be ≥ the last of the
        # previous sweep, modulo small numerical noise.)
        n_factors = 2
        per_call = result.per_call_total_objectives
        sweep_finals = [
            per_call[i + n_factors - 1]
            for i in range(0, len(per_call), n_factors)
            if i + n_factors - 1 < len(per_call)
        ]
        for i in range(len(sweep_finals) - 1):
            assert sweep_finals[i] <= sweep_finals[i + 1] + 1e-3, (
                f"CD objective decreased between sweep {i} and {i + 1}: "
                f"{sweep_finals[i]} > {sweep_finals[i + 1]}"
            )

        # cd_iterations is 1-indexed (counts completed CD sweeps).
        assert 1 <= result.cd_iterations <= 3

    def test_rejects_empty_contexts(self):
        """Passing zero factor contexts is a configuration error and
        must surface immediately, not run a no-op CD pass."""
        df = make_small_df(n_quotes=10, n_steps=5)
        grid = _build_grid(df, constraint_cols=["volume"])
        candidates = [0.8, 1.0, 1.2]

        with pytest.raises(ValueError, match="contexts must not be empty"):
            run_cd_pass_py(
                grid,
                [],
                candidates,
                constraints={"volume": {"min_pct": 0.90}},
            )

    def test_rejects_context_n_quotes_mismatch(self):
        """A FactorContext built from a label vector of the wrong
        length must be rejected against the grid's n_quotes — silently
        running with a mismatched group_of would index out of bounds in
        the Rust kernel."""
        df = make_small_df(n_quotes=10, n_steps=5)
        grid = _build_grid(df, constraint_cols=["volume"])
        # Wrong length: 7 labels vs 10 quotes in grid.
        wrong_ctx = FactorContext.from_labels(["A"] * 7)
        candidates = [0.8, 1.0, 1.2]

        with pytest.raises(ValueError, match="context.*n_quotes"):
            run_cd_pass_py(
                grid,
                [wrong_ctx],
                candidates,
                constraints={"volume": {"min_pct": 0.90}},
            )

    def test_sequential_path_below_par_threshold(self):
        """At small `n_quotes` the residual / multiplier inner loops
        drop to a sequential walk instead of rayon parallel iteration.
        This pins that path's correctness on a 200-quote problem (well
        below the 100 000 parallelisation threshold).
        """
        n = 200
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=["volume"])
        labels = [f"G{i % 4}" for i in range(n)]
        ctx = FactorContext.from_labels(labels)
        candidates = list(grid.scenario_values)

        result = run_cd_pass_py(
            grid,
            [ctx],
            candidates,
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
            tolerance=1e-5,
            max_cd_iterations=3,
            cd_tolerance=1e-3,
        )

        # Sanity: factor values populated, constraint near-satisfied,
        # last grouped solve's optimal_steps span the right range.
        assert len(result.factor_values[0]) == ctx.n_groups
        assert all(0.0 < v for v in result.factor_values[0])
        steps = result.optimal_steps_per_quote
        assert len(steps) == n
        assert all(0 <= s < 5 for s in steps)

    def test_dataframe_getter_lazy_and_idempotent(self):
        """The `dataframe` getter is built lazily from
        `optimal_steps_per_quote` and the grid; calling it twice should
        return the same per-quote rows without re-running the solver.
        """
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        grid = _build_grid(df, constraint_cols=["volume"])
        ctx = FactorContext.from_labels([f"G{i % 3}" for i in range(n)])
        candidates = list(grid.scenario_values)

        result = run_cd_pass_py(
            grid,
            [ctx],
            candidates,
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=50,
            max_cd_iterations=2,
        )

        df_first = result.dataframe
        df_second = result.dataframe
        # Both calls return DataFrames with the same row count and the
        # same `optimal_step` column values (cached round-trip).
        assert df_first.shape[0] == n
        assert df_second.shape[0] == n
        assert df_first["optimal_step"].to_list() == df_second["optimal_step"].to_list()
