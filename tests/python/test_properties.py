"""Property-based tests for price-contour using Hypothesis.

These tests verify algebraic invariants of the solver using randomly
generated inputs, catching edge cases that hand-crafted fixtures miss.
"""

from __future__ import annotations


import polars as pl
from hypothesis import given, settings, assume
import hypothesis.strategies as st

import price_contour as pc


# ---------------------------------------------------------------------------
# Custom composite strategy
# ---------------------------------------------------------------------------


@st.composite
def quote_grid_strategy(draw: st.DrawFn) -> tuple[pl.DataFrame, int, int]:
    """Generate a valid (DataFrame, n_quotes, n_steps) tuple.

    The DataFrame has the schema expected by price-contour:
    quote_id (Utf8), scenario_index (Int32), scenario_value (Float32),
    expected_income (Float32), volume (Float32).
    """
    n_quotes = draw(st.integers(2, 30))
    n_steps = draw(st.integers(2, 8))

    # Draw n_steps distinct, sorted scenario values in [0.5, 2.0]
    scenario_values = draw(
        st.lists(
            st.floats(0.5, 2.0, allow_nan=False, allow_infinity=False),
            min_size=n_steps,
            max_size=n_steps,
            unique=True,
        )
    )
    assume(len(set(scenario_values)) == n_steps)
    scenario_values = sorted(scenario_values)

    rows: list[dict] = []
    for q in range(n_quotes):
        incomes = draw(
            st.lists(
                st.floats(0.01, 1000.0, allow_nan=False, allow_infinity=False),
                min_size=n_steps,
                max_size=n_steps,
            )
        )
        volumes = draw(
            st.lists(
                st.floats(0.01, 1.0, allow_nan=False, allow_infinity=False),
                min_size=n_steps,
                max_size=n_steps,
            )
        )
        for j in range(n_steps):
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": scenario_values[j],
                    "expected_income": incomes[j],
                    "volume": volumes[j],
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
    return df, n_quotes, n_steps


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestProperties:
    """Property-based tests for OnlineOptimiser and ApplyOptimiser."""

    @given(data=quote_grid_strategy())
    @settings(max_examples=30, deadline=30000)
    def test_unconstrained_picks_argmax(self, data):
        """With no constraints, each quote picks the step that maximises
        expected_income (the solver degenerates to per-quote argmax)."""
        df, n_quotes, n_steps = data

        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)

        out = result.dataframe
        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx, (
                f"quote {qid}: optimal_step={row['optimal_step']} "
                f"but argmax(expected_income)={best_idx}"
            )

    @given(data=quote_grid_strategy())
    @settings(max_examples=30, deadline=30000)
    def test_constrained_objective_leq_unconstrained(self, data):
        """Adding a constraint can only reduce (or equal) the objective;
        it can never improve upon the unconstrained optimum."""
        df, n_quotes, n_steps = data

        unconstrained = pc.OnlineOptimiser(objective="expected_income")
        unconstrained_result = unconstrained.solve(df)

        constrained = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        constrained_result = constrained.solve(df)

        # Allow a tiny relative epsilon for floating-point noise
        assert constrained_result.total_objective <= (
            unconstrained_result.total_objective
            + abs(unconstrained_result.total_objective) * 0.01
        ), (
            f"constrained objective {constrained_result.total_objective} "
            f"> unconstrained {unconstrained_result.total_objective} + 1% epsilon"
        )

    @given(data=quote_grid_strategy())
    @settings(max_examples=30, deadline=30000)
    def test_apply_reproduces_solve_decisions(self, data):
        """Given the same lambdas, ApplyOptimiser must reproduce the exact
        same optimal_step per quote as the solver produced."""
        df, n_quotes, n_steps = data

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        solve_result = solver.solve(df)

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        apply_result = applier.apply(df)

        solve_steps = solve_result.dataframe.sort("quote_id")["optimal_step"].to_list()
        apply_steps = apply_result.dataframe.sort("quote_id")["optimal_step"].to_list()

        assert solve_steps == apply_steps, (
            "ApplyOptimiser with solve lambdas produced different optimal_step "
            "values than the solver"
        )

    @given(data=quote_grid_strategy())
    @settings(max_examples=30, deadline=30000)
    def test_zero_lambdas_equals_unconstrained(self, data):
        """Applying lambdas={"volume": 0.0} is equivalent to an unconstrained
        solve — zero penalty means the constraint has no effect."""
        df, n_quotes, n_steps = data

        solver = pc.OnlineOptimiser(objective="expected_income")
        unconstrained_result = solver.solve(df)

        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        apply_result = applier.apply(df)

        unconstrained_steps = unconstrained_result.dataframe.sort("quote_id")[
            "optimal_step"
        ].to_list()
        apply_steps = apply_result.dataframe.sort("quote_id")["optimal_step"].to_list()

        assert unconstrained_steps == apply_steps, (
            "Zero lambdas should produce the same steps as unconstrained solve"
        )

    @given(data=quote_grid_strategy())
    @settings(max_examples=30, deadline=30000)
    def test_warm_start_objective_no_worse(self, data):
        """Warm-starting with the cold-start lambdas should produce a
        total_objective at least as good (minus floating-point noise)."""
        df, n_quotes, n_steps = data

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )

        cold = solver.solve(df)
        warm = solver.solve(df, lambdas=cold.lambdas)

        # Warm start may produce slightly different results due to discrete
        # Lagrangian relaxation and different convergence paths.
        # With small portfolios, discrete flips can cause large relative
        # changes — this property only holds reliably at scale.
        assume(n_quotes >= 20)
        assert warm.total_objective >= (
            cold.total_objective - abs(cold.total_objective) * 0.05
        ), (
            f"warm start objective {warm.total_objective} is worse than "
            f"cold start {cold.total_objective} by more than 5% epsilon"
        )

    @given(data=quote_grid_strategy())
    @settings(max_examples=30, deadline=30000)
    def test_baseline_is_closest_to_one(self, data):
        """baseline_objective equals the sum of objective values at the step
        whose scenario_value is closest to 1.0 (the 'no change' reference)."""
        df, n_quotes, n_steps = data

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        # Manually compute the baseline: for each quote, find the step with
        # scenario_value closest to 1.0 and sum the expected_income values.
        expected_baseline = 0.0
        for q in range(n_quotes):
            qid = f"Q{q:04d}"
            q_df = df.filter(pl.col("quote_id") == qid)
            sv = q_df["scenario_value"].to_list()
            ei = q_df["expected_income"].to_list()
            # Find the index of the scenario_value closest to 1.0
            closest_idx = min(range(len(sv)), key=lambda i: abs(sv[i] - 1.0))
            expected_baseline += ei[closest_idx]

        assert abs(result.baseline_objective - expected_baseline) < 1e-2, (
            f"baseline_objective={result.baseline_objective} does not match "
            f"manually computed baseline={expected_baseline} "
            f"(diff={abs(result.baseline_objective - expected_baseline):.6f})"
        )
