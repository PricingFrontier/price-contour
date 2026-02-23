"""Integration tests for the apply module."""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc


def _make_small_df(n_quotes: int = 50, n_steps: int = 5) -> pl.DataFrame:
    """Build a small test DataFrame with known properties."""
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


class TestApply:
    """Tests for ApplyOptimiser."""

    def test_apply_with_solve_lambdas_matches_objective(self):
        df = _make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        solve_result = solver.solve(df)

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        apply_result = applier.apply(df)

        assert abs(apply_result.total_objective - solve_result.total_objective) < 1e-3

    def test_apply_result_has_baselines(self):
        df = _make_small_df(n_quotes=50, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        apply_result = applier.apply(df)

        assert apply_result.baseline_objective > 0
        assert "volume" in apply_result.baseline_constraints
        assert apply_result.baseline_constraints["volume"] > 0

    def test_apply_result_dataframe_shape(self):
        n_quotes = 50
        df = _make_small_df(n_quotes=n_quotes, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        apply_result = applier.apply(df)

        out = apply_result.dataframe
        assert out.shape[0] == n_quotes
        assert "quote_id" in out.columns
        assert "optimal_step" in out.columns
        assert "optimal_multiplier" in out.columns
        assert "optimal_objective" in out.columns
        assert "optimal_volume" in out.columns

    def test_apply_zero_lambdas_unconstrained(self):
        """With zero lambdas, apply should pick max-objective step per quote."""
        df = _make_small_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        apply_result = applier.apply(df)
        out = apply_result.dataframe

        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx, f"quote {qid}"

    def test_apply_two_constraints(self):
        df = _make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 0.92},
                "loss_ratio": {"max": 1.05},
            },
            max_iter=200,
        )
        solve_result = solver.solve(df)

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="expected_income",
            constraints={
                "volume": {"min": 0.92},
                "loss_ratio": {"max": 1.05},
            },
        )
        apply_result = applier.apply(df)

        assert "volume" in apply_result.total_constraints
        assert "loss_ratio" in apply_result.total_constraints
        assert "volume" in apply_result.lambdas
        assert "loss_ratio" in apply_result.lambdas


class TestSolveResultNewGetters:
    """Tests for new getters on SolveResult."""

    def test_baseline_objective(self):
        df = _make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        result = solver.solve(df)
        assert isinstance(result.baseline_objective, float)
        assert result.baseline_objective > 0

    def test_baseline_constraints(self):
        df = _make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        result = solver.solve(df)
        assert isinstance(result.baseline_constraints, dict)
        assert "volume" in result.baseline_constraints

    def test_multipliers(self):
        df = _make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        assert len(result.multipliers) == 5
        assert abs(result.multipliers[0] - 0.8) < 0.01

    def test_n_quotes_n_steps(self):
        df = _make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        assert result.n_quotes == 20
        assert result.n_steps == 5

    def test_history_none_by_default(self):
        df = _make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        result = solver.solve(df)
        assert result.history is None

    def test_history_recorded(self):
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
            record_history=True,
        )
        result = solver.solve(df)
        assert result.history is not None
        assert len(result.history) == result.iterations
        rec = result.history[0]
        assert "iteration" in rec
        assert "total_objective" in rec
        assert "lambdas" in rec
        assert "total_constraints" in rec
        assert "max_lambda_change" in rec
        assert "all_constraints_satisfied" in rec


class TestSummary:
    """Tests for OnlineOptimiser.summary()."""

    def test_summary_keys(self):
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        s = solver.summary(result)

        assert set(s.keys()) == {"params", "metrics", "artifacts"}

    def test_params_are_flat_scalars(self):
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        params = solver.summary(result)["params"]

        for k, v in params.items():
            assert isinstance(v, (str, int, float)), f"param {k!r} is {type(v)}"
        assert params["objective"] == "expected_income"
        assert params["n_quotes"] == 50
        assert params["n_steps"] == 5

    def test_metrics_are_flat_floats(self):
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        metrics = solver.summary(result)["metrics"]

        for k, v in metrics.items():
            assert isinstance(v, float), f"metric {k!r} is {type(v)}"
        assert "total_objective" in metrics
        assert "baseline_objective" in metrics
        assert "uplift_pct" in metrics
        assert "constraint_volume_total" in metrics
        assert "constraint_volume_baseline" in metrics
        assert "constraint_volume_ratio" in metrics
        assert "lambda_volume" in metrics
        assert "multiplier_mean" in metrics
        assert "multiplier_p50" in metrics

    def test_artifacts_structure(self):
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        artifacts = solver.summary(result)["artifacts"]

        assert isinstance(artifacts["lambdas"], dict)
        assert isinstance(artifacts["config"], dict)
        assert isinstance(artifacts["summary"], dict)
        assert artifacts["convergence"] is None  # record_history=False

    def test_convergence_dataframe_when_history(self):
        df = _make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
            record_history=True,
        )
        result = solver.solve(df)
        conv = solver.summary(result)["artifacts"]["convergence"]

        assert isinstance(conv, pl.DataFrame)
        assert conv.shape[0] == result.iterations
        assert "iteration" in conv.columns
        assert "total_objective" in conv.columns
        assert "lambda_volume" in conv.columns
        assert "constraint_volume" in conv.columns
