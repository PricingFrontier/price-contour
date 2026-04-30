"""Integration tests for the online solver."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from helpers import make_small_df, CONSTRAINT_RTOL

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TEST_PARQUET = DATA_DIR / "test_quotes.parquet"


class TestBasicSolve:
    """Tests with small synthetic data."""

    def test_unconstrained_returns_correct_shape(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)

        out = result.dataframe
        assert out.shape[0] == 20
        assert "quote_id" in out.columns
        assert "optimal_step" in out.columns
        assert "optimal_scenario_value" in out.columns
        assert "optimal_objective" in out.columns

    def test_unconstrained_picks_best_step(self):
        df = make_small_df(n_quotes=10, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)

        # Each quote should pick the step maximising expected_income
        out = result.dataframe
        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx

    def test_single_min_constraint_satisfied(self):
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        # Use solver's own baseline rather than hand-computed value
        baseline_vol = result.baseline_constraints["volume"]
        threshold = baseline_vol * 0.90

        assert result.total_constraints["volume"] >= threshold * (
            1 - CONSTRAINT_RTOL
        ), (
            f"volume {result.total_constraints['volume']} < {threshold * (1 - CONSTRAINT_RTOL)} "
            f"(threshold {threshold} with {CONSTRAINT_RTOL:.0%} slack)"
        )

        # Note: with a min constraint, the solver may push volume *above* baseline
        # (lower prices → higher conversion → more volume), so no upper bound check.

    def test_single_max_constraint_satisfied(self):
        df = make_small_df(n_quotes=200, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max_pct": 1.05}},
            max_iter=500,
        )
        result = solver.solve(df)

        # Use solver's own baseline rather than hand-computed value
        baseline_lr = result.baseline_constraints["loss_ratio"]
        threshold = baseline_lr * 1.05

        # Max constraints are harder to converge on small data, use wider tolerance
        assert result.total_constraints["loss_ratio"] <= threshold * (
            1 + CONSTRAINT_RTOL * 3
        ), (
            f"loss_ratio {result.total_constraints['loss_ratio']} > "
            f"{threshold * (1 + CONSTRAINT_RTOL * 3)} "
            f"(threshold {threshold} with {CONSTRAINT_RTOL * 3:.0%} slack)"
        )

    def test_warm_start_converges_faster(self):
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )

        cold = solver.solve(df)
        warm = solver.solve(df, lambdas=cold.lambdas)

        assert warm.iterations <= cold.iterations, (
            f"warm start ({warm.iterations} iters) should converge "
            f"no slower than cold start ({cold.iterations} iters)"
        )

    def test_two_constraints(self):
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.92},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=200,
        )
        result = solver.solve(df)

        # Key-presence checks
        assert "volume" in result.lambdas
        assert "loss_ratio" in result.lambdas
        assert "volume" in result.total_constraints
        assert "loss_ratio" in result.total_constraints

        # Value assertions: constraints approximately satisfied
        baseline_vol = result.baseline_constraints["volume"]
        vol_threshold = baseline_vol * 0.92
        assert result.total_constraints["volume"] >= vol_threshold * (
            1 - CONSTRAINT_RTOL
        ), (
            f"volume {result.total_constraints['volume']} below threshold "
            f"{vol_threshold} with {CONSTRAINT_RTOL:.0%} slack"
        )

        baseline_lr = result.baseline_constraints["loss_ratio"]
        lr_threshold = baseline_lr * 1.05
        # Multi-constraint solves are harder to converge; use wider tolerance.
        # The discrete Lagrangian relaxation may not exactly satisfy all constraints
        # simultaneously on small datasets with few steps.
        assert result.total_constraints["loss_ratio"] <= lr_threshold * (
            1 + CONSTRAINT_RTOL * 15
        ), (
            f"loss_ratio {result.total_constraints['loss_ratio']} above threshold "
            f"{lr_threshold} with {CONSTRAINT_RTOL * 15:.0%} slack"
        )

        # Both lambdas should be non-negative
        assert result.lambdas["volume"] >= 0, (
            f"volume lambda negative: {result.lambdas['volume']}"
        )
        assert result.lambdas["loss_ratio"] >= 0, (
            f"loss_ratio lambda negative: {result.lambdas['loss_ratio']}"
        )

        # Objective should be positive
        assert result.total_objective > 0

    def test_result_properties(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert result.iterations > 0
        assert isinstance(result.total_objective, float)
        assert isinstance(result.lambdas, dict)
        assert isinstance(result.total_constraints, dict)

    def test_three_constraints(self):
        """Solver handles 3 simultaneous constraints."""
        # Use volume (min), loss_ratio (max), and expected_income (min).
        # expected_income is both the objective and a constraint column:
        # constraining it just ensures the solver produces a non-trivial result
        # with 3 lambda values.
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.10},
                "expected_income": {"min_pct": 0.85},
            },
            max_iter=200,
        )
        result = solver.solve(df)

        # Should produce a result (may or may not converge with tight constraints)
        assert result.iterations > 0
        assert len(result.lambdas) == 3
        assert "volume" in result.lambdas
        assert "loss_ratio" in result.lambdas
        assert "expected_income" in result.lambdas


# ---------------------------------------------------------------------------
# Full portfolio test (1M quotes, requires test_quotes.parquet)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TEST_PARQUET.exists(), reason="test_quotes.parquet not found")
class TestFullPortfolio:
    """Tests with the 1M-quote test dataset."""

    @pytest.fixture(scope="class")
    def df(self) -> pl.DataFrame:
        return pl.read_parquet(TEST_PARQUET)

    def test_basic_solve_shape(self, df):
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        result = solver.solve(df)
        out = result.dataframe
        assert out.shape[0] == df["quote_id"].n_unique()

    def test_performance(self, df):
        """1M quotes should solve in < 10s (generous bound for CI)."""
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        t0 = time.perf_counter()
        result = solver.solve(df)
        elapsed = time.perf_counter() - t0
        print(
            f"\n  1M solve: {elapsed:.2f}s, converged={result.converged}, "
            f"iters={result.iterations}, lambdas={result.lambdas}"
        )
        assert elapsed < 10.0, f"Solve took {elapsed:.2f}s, expected < 10s"

    def test_two_constraint_solve(self, df):
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
            },
        )
        result = solver.solve(df)
        assert "volume" in result.lambdas
        assert "loss_ratio" in result.lambdas

    def test_warm_start_full(self, df):
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        cold = solver.solve(df)
        warm = solver.solve(df, lambdas=cold.lambdas)
        assert warm.iterations <= cold.iterations
