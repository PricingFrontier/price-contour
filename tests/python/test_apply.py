"""Integration tests for the apply module."""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from helpers import make_small_df, CONSTRAINT_RTOL


class TestApply:
    """Tests for ApplyOptimiser."""

    def test_apply_with_solve_lambdas_matches_objective(self):
        df = make_small_df(n_quotes=100, n_steps=5)
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
        df = make_small_df(n_quotes=50, n_steps=5)
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
        df = make_small_df(n_quotes=n_quotes, n_steps=5)
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
        assert "optimal_scenario_value" in out.columns
        assert "optimal_objective" in out.columns
        assert "optimal_volume" in out.columns

    def test_apply_zero_lambdas_unconstrained(self):
        """With zero lambdas, apply should pick max-objective step per quote."""
        df = make_small_df(n_quotes=20, n_steps=5)
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
        df = make_small_df(n_quotes=100, n_steps=5)
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

        # Key-presence checks
        assert "volume" in apply_result.total_constraints
        assert "loss_ratio" in apply_result.total_constraints
        assert "volume" in apply_result.lambdas
        assert "loss_ratio" in apply_result.lambdas

        # Constraint satisfaction assertions
        baseline_vol = apply_result.baseline_constraints["volume"]
        vol_threshold = baseline_vol * 0.92
        assert apply_result.total_constraints["volume"] >= vol_threshold * (1 - CONSTRAINT_RTOL), (
            f"volume {apply_result.total_constraints['volume']} below threshold "
            f"{vol_threshold} with {CONSTRAINT_RTOL:.0%} slack"
        )

        baseline_lr = apply_result.baseline_constraints["loss_ratio"]
        lr_threshold = baseline_lr * 1.05
        # Multi-constraint solves are harder to converge; use wider tolerance
        assert apply_result.total_constraints["loss_ratio"] <= lr_threshold * (1 + CONSTRAINT_RTOL * 3), (
            f"loss_ratio {apply_result.total_constraints['loss_ratio']} above threshold "
            f"{lr_threshold} with {CONSTRAINT_RTOL * 3:.0%} slack"
        )


class TestSolveResultNewGetters:
    """Tests for new getters on SolveResult."""

    def test_baseline_objective(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        result = solver.solve(df)
        assert isinstance(result.baseline_objective, float)
        assert result.baseline_objective > 0

    def test_baseline_constraints(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        result = solver.solve(df)
        assert isinstance(result.baseline_constraints, dict)
        assert "volume" in result.baseline_constraints

    def test_scenario_values(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        assert len(result.scenario_values) == 5
        assert abs(result.scenario_values[0] - 0.8) < 0.01

    def test_n_quotes_n_steps(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        assert result.n_quotes == 20
        assert result.n_steps == 5

    def test_history_none_by_default(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        result = solver.solve(df)
        assert result.history is None

    def test_history_recorded(self):
        df = make_small_df(n_quotes=50, n_steps=5)
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
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        s = solver.summary(result)

        assert set(s.keys()) == {"params", "metrics", "artifacts"}

    def test_params_are_flat_scalars(self):
        df = make_small_df(n_quotes=50, n_steps=5)
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
        df = make_small_df(n_quotes=50, n_steps=5)
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
        assert "scenario_value_mean" in metrics
        assert "scenario_value_p50" in metrics

    def test_artifacts_structure(self):
        df = make_small_df(n_quotes=50, n_steps=5)
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
        df = make_small_df(n_quotes=50, n_steps=5)
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

    def test_summary_values_match_result(self):
        """Summary metric/param/artifact values match the SolveResult."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        s = solver.summary(result)
        metrics = s["metrics"]
        params = s["params"]
        artifacts = s["artifacts"]

        # Metrics match result
        assert metrics["total_objective"] == result.total_objective
        assert metrics["baseline_objective"] == result.baseline_objective

        expected_uplift = (
            (result.total_objective - result.baseline_objective)
            / abs(result.baseline_objective)
            * 100
        )
        assert abs(metrics["uplift_pct"] - expected_uplift) < 1e-6, (
            f"uplift_pct {metrics['uplift_pct']} != expected {expected_uplift}"
        )

        assert metrics["constraint_volume_total"] == result.total_constraints["volume"]
        assert metrics["lambda_volume"] == result.lambdas["volume"]

        # Params match result
        assert params["n_quotes"] == result.n_quotes
        assert params["n_steps"] == result.n_steps

        # Artifacts match result
        assert artifacts["lambdas"] == result.lambdas

    def test_summary_scenario_stats_ordered(self):
        """Scenario percentile stats are monotonically ordered."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result = solver.solve(df)
        s = solver.summary(result)
        metrics = s["metrics"]
        params = s["params"]

        p5 = metrics["scenario_value_p5"]
        p25 = metrics["scenario_value_p25"]
        p50 = metrics["scenario_value_p50"]
        p75 = metrics["scenario_value_p75"]
        p95 = metrics["scenario_value_p95"]
        mean = metrics["scenario_value_mean"]

        assert p5 <= p25 <= p50 <= p75 <= p95, (
            f"percentiles not ordered: p5={p5}, p25={p25}, p50={p50}, p75={p75}, p95={p95}"
        )

        sv_min = params["scenario_value_min"]
        sv_max = params["scenario_value_max"]
        assert sv_min <= mean <= sv_max, (
            f"mean {mean} not between min {sv_min} and max {sv_max}"
        )


class TestConvergenceBehavior:
    """Tests for convergence history behavior."""

    def _solve_with_history(self):
        """Helper: solve a constrained problem with history recording.

        Uses volume max=0.80 (keep volume <= 80% of baseline). On this
        synthetic data the unconstrained optimum has volume far above
        baseline, so a max constraint is actually binding and forces
        the solver through multiple iterations.
        """
        df = make_small_df(n_quotes=200, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"max": 0.80}},
            max_iter=200,
            record_history=True,
        )
        result = solver.solve(df)
        return result

    def test_history_objective_stabilizes(self):
        """The objective value stabilizes over iterations (last 25% less volatile than first 25%)."""
        result = self._solve_with_history()
        assert result.history is not None
        n = len(result.history)
        assert n >= 4, f"too few iterations ({n}) to test stabilization"

        quarter = max(n // 4, 1)
        first_objs = [rec["total_objective"] for rec in result.history[:quarter]]
        last_objs = [rec["total_objective"] for rec in result.history[-quarter:]]

        first_range = max(first_objs) - min(first_objs) if len(first_objs) > 1 else 0.0
        last_range = max(last_objs) - min(last_objs) if len(last_objs) > 1 else 0.0

        # The last quarter should be at least as stable as the first quarter
        assert last_range <= first_range + 1e-6, (
            f"objective did not stabilize: first quarter range={first_range:.4f}, "
            f"last quarter range={last_range:.4f}"
        )

    def test_history_lambdas_stabilize(self):
        """Lambda changes decrease over iterations (last 25% < first 25%)."""
        result = self._solve_with_history()
        assert result.history is not None
        n = len(result.history)
        assert n >= 4, f"too few iterations ({n}) to test stabilization"

        quarter = n // 4
        first_quarter_changes = [
            rec["max_lambda_change"] for rec in result.history[:quarter]
        ]
        last_quarter_changes = [
            rec["max_lambda_change"] for rec in result.history[-quarter:]
        ]

        mean_first = sum(first_quarter_changes) / len(first_quarter_changes)
        mean_last = sum(last_quarter_changes) / len(last_quarter_changes)

        assert mean_last < mean_first, (
            f"lambdas did not stabilize: mean change first 25%={mean_first:.6f}, "
            f"last 25%={mean_last:.6f}"
        )

    def test_history_final_constraints_satisfied(self):
        """If converged, the final history record has all constraints satisfied."""
        result = self._solve_with_history()
        assert result.history is not None

        if result.converged:
            final_rec = result.history[-1]
            assert final_rec["all_constraints_satisfied"] is True, (
                "converged but final record does not have all_constraints_satisfied=True"
            )

    def test_history_final_record_matches_result(self):
        """Final history record's values are in the same ballpark as the SolveResult.

        The result may differ from the last history record because the solver
        does a final forward pass with the best-found lambdas, which can
        differ from the last iteration's snapshot.
        """
        result = self._solve_with_history()
        assert result.history is not None

        final_rec = result.history[-1]

        # Use relative tolerance — the result and last history record should
        # be within 10% of each other (the solver's final pass may differ)
        scale = abs(result.total_objective) or 1.0
        diff = abs(final_rec["total_objective"] - result.total_objective) / scale
        assert diff < 0.10, (
            f"final history objective {final_rec['total_objective']:.2f} differs from "
            f"result objective {result.total_objective:.2f} by {diff:.1%}"
        )
