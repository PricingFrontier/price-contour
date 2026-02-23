"""Integration tests for the online solver."""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import pytest

import price_contour as pc

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
TEST_PARQUET = DATA_DIR / "test_quotes.parquet"


# ---------------------------------------------------------------------------
# Small synthetic data (fast, deterministic)
# ---------------------------------------------------------------------------


def _make_small_df(n_quotes: int = 50, n_steps: int = 5) -> pl.DataFrame:
    """Build a small test DataFrame with known properties."""
    rows = []
    mults = [0.8 + 0.1 * j for j in range(n_steps)]
    for q in range(n_quotes):
        elasticity = 1.5 + 3.5 * q / n_quotes
        base = 80.0 + 40.0 * q / n_quotes
        for j, mult in enumerate(mults):
            conversion = 1.0 / (1.0 + (elasticity * (mult - 1.0)) ** 1)
            import math

            conversion = 1.0 / (1.0 + math.exp(elasticity * (mult - 1.0)))
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_step": j,
                    "multiplier": mult,
                    "expected_income": base * mult * conversion,
                    "volume": conversion,
                    "loss_ratio": 0.6 / mult * (1.0 + 0.1 * (mult - 1.0)),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "quote_id": pl.Utf8,
            "scenario_step": pl.Int32,
            "multiplier": pl.Float32,
            "expected_income": pl.Float32,
            "volume": pl.Float32,
            "loss_ratio": pl.Float32,
        },
    )


class TestBasicSolve:
    """Tests with small synthetic data."""

    def test_unconstrained_returns_correct_shape(self):
        df = _make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)

        out = result.dataframe
        assert out.shape[0] == 20
        assert "quote_id" in out.columns
        assert "optimal_step" in out.columns
        assert "optimal_multiplier" in out.columns
        assert "optimal_objective" in out.columns

    def test_unconstrained_picks_best_step(self):
        df = _make_small_df(n_quotes=10, n_steps=5)
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
        df = _make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        # Compute baseline volume
        baseline = df.filter(pl.col("scenario_step") == 2)  # mult=1.0
        baseline_vol = baseline["volume"].sum()
        threshold = baseline_vol * 0.90

        assert (
            result.total_constraints["volume"] >= threshold * 0.95
        ), f"volume {result.total_constraints['volume']} < 95% of threshold {threshold}"

    def test_single_max_constraint_satisfied(self):
        df = _make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": 1.05}},
            max_iter=200,
        )
        result = solver.solve(df)

        baseline = df.filter(pl.col("scenario_step") == 2)
        baseline_lr = baseline["loss_ratio"].sum()
        threshold = baseline_lr * 1.05

        assert (
            result.total_constraints["loss_ratio"] <= threshold * 1.10
        ), f"loss_ratio {result.total_constraints['loss_ratio']} > 110% of threshold {threshold}"

    def test_warm_start_converges_faster(self):
        df = _make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )

        cold = solver.solve(df)
        warm = solver.solve(df, lambdas=cold.lambdas)

        assert warm.iterations <= cold.iterations, (
            f"warm start ({warm.iterations} iters) should converge "
            f"no slower than cold start ({cold.iterations} iters)"
        )

    def test_two_constraints(self):
        df = _make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 0.92},
                "loss_ratio": {"max": 1.05},
            },
            max_iter=200,
        )
        result = solver.solve(df)

        assert "volume" in result.lambdas
        assert "loss_ratio" in result.lambdas
        assert "volume" in result.total_constraints
        assert "loss_ratio" in result.total_constraints

    def test_result_properties(self):
        df = _make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)

        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)
        assert result.iterations > 0
        assert isinstance(result.total_objective, float)
        assert isinstance(result.lambdas, dict)
        assert isinstance(result.total_constraints, dict)


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
            constraints={"volume": {"min": 0.90}},
        )
        result = solver.solve(df)
        out = result.dataframe
        assert out.shape[0] == df["quote_id"].n_unique()

    def test_performance(self, df):
        """1M quotes should solve in < 10s (generous bound for CI)."""
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
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
                "volume": {"min": 0.90},
                "loss_ratio": {"max": 1.05},
            },
        )
        result = solver.solve(df)
        assert "volume" in result.lambdas
        assert "loss_ratio" in result.lambdas

    def test_warm_start_full(self, df):
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        cold = solver.solve(df)
        warm = solver.solve(df, lambdas=cold.lambdas)
        assert warm.iterations <= cold.iterations
