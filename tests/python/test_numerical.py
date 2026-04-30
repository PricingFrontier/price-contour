"""Numerical stability tests for Float32 Lagrangian dual solver."""

from __future__ import annotations

import math

import polars as pl

import price_contour as pc
from helpers import make_small_df, CONSTRAINT_RTOL


class TestNumericalStability:
    """Verify the solver handles extreme and degenerate numerical regimes.

    The Rust core uses Float32 throughout, so these tests probe the edges
    of that representation: large magnitudes, tiny magnitudes, mixed
    scales, near-zero constraints, all-zero constraints, and precision
    boundaries near 2^23.
    """

    def test_large_objective_values(self):
        """Objectives scaled to ~1e6 should not overflow or produce NaN."""
        df = make_small_df(n_quotes=50, n_steps=5)
        df = df.with_columns(pl.col("expected_income") * 1e6)

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        # Result must be finite
        assert math.isfinite(result.total_objective), (
            f"total_objective is not finite: {result.total_objective}"
        )

        # All optimal_step values must be valid indices
        out = result.dataframe
        for s in out["optimal_step"].to_list():
            assert 0 <= s < 5, f"optimal_step {s} out of range [0, 5)"

        # Constraint approximately satisfied
        baseline_vol = result.baseline_constraints["volume"]
        threshold = baseline_vol * 0.90
        assert result.total_constraints["volume"] >= threshold * (
            1 - CONSTRAINT_RTOL
        ), (
            f"volume {result.total_constraints['volume']} < "
            f"{threshold * (1 - CONSTRAINT_RTOL)} "
            f"(threshold {threshold} with {CONSTRAINT_RTOL:.0%} slack)"
        )

        # Solver should converge
        assert result.converged is True, (
            f"solver did not converge after {result.iterations} iterations"
        )

    def test_small_objective_values(self):
        """Objectives scaled to ~1e-6 should not underflow or crash."""
        df = make_small_df(n_quotes=50, n_steps=5)
        df = df.with_columns(pl.col("expected_income") * 1e-6)

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        # Result must be finite (not NaN or Inf)
        assert math.isfinite(result.total_objective), (
            f"total_objective is not finite: {result.total_objective}"
        )

        # All optimal_step values must be valid indices
        out = result.dataframe
        for s in out["optimal_step"].to_list():
            assert 0 <= s < 5, f"optimal_step {s} out of range [0, 5)"

    def test_mixed_scale_objective_and_constraint(self):
        """Objectives in millions, volume in [0, 1] — mirrors real insurance data.

        The Lagrangian multiplier must bridge a ~1e6 scale gap between the
        objective and the constraint column.  This is the most common
        numerical regime in production.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        df = df.with_columns(pl.col("expected_income") * 1e6)

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        # Finite result
        assert math.isfinite(result.total_objective), (
            f"total_objective is not finite: {result.total_objective}"
        )

        # Constraint satisfaction within tolerance
        baseline_vol = result.baseline_constraints["volume"]
        threshold = baseline_vol * 0.90
        assert result.total_constraints["volume"] >= threshold * (
            1 - CONSTRAINT_RTOL
        ), (
            f"volume {result.total_constraints['volume']} < "
            f"{threshold * (1 - CONSTRAINT_RTOL)} "
            f"(threshold {threshold} with {CONSTRAINT_RTOL:.0%} slack)"
        )

    def test_near_zero_constraint_values(self):
        """Constraint column scaled to ~1e-8 should not produce NaN lambdas."""
        df = make_small_df(n_quotes=50, n_steps=5)
        df = df.with_columns(pl.col("volume") * 1e-8)

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        # Must not crash and must return finite objective
        assert math.isfinite(result.total_objective), (
            f"total_objective is not finite: {result.total_objective}"
        )

        # Lambdas must be finite (no NaN or Inf from division by tiny values)
        assert all(math.isfinite(v) for v in result.lambdas.values()), (
            f"lambdas contain non-finite values: {result.lambdas}"
        )

    def test_all_zero_constraint_values(self):
        """All-zero constraint column — the constraint is meaningless.

        The solver may or may not converge (there is no gradient signal in
        the constraint), but it must not crash or produce NaN/Inf.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias("volume"))

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=50,
        )
        result = solver.solve(df)

        # Must not crash; objective must be finite
        assert math.isfinite(result.total_objective), (
            f"total_objective is not finite: {result.total_objective}"
        )

    def test_f32_precision_boundary(self):
        """Objectives near 2^23 where Float32 loses integer precision.

        Float32 can represent integers exactly up to 2^23 = 8_388_608.
        Beyond that, consecutive representable values differ by >1,
        so close objective values may compare incorrectly.

        We add 8_388_608.0 to every expected_income value, pushing them
        into the precision-loss zone, and verify the solver still picks
        the correct argmax per quote.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        df = df.with_columns(pl.col("expected_income") + 8_388_608.0)

        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)

        out = result.dataframe

        # Verify each quote picks the argmax of expected_income
        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx, (
                f"quote {qid}: expected step {best_idx}, got {row['optimal_step']}; "
                f"f32 precision loss may have caused incorrect argmax"
            )
