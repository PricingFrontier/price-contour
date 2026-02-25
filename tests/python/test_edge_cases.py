"""Edge case and boundary condition tests."""

from __future__ import annotations

from helpers import make_small_df
import price_contour as pc
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Class 1: Degenerate shapes
# ---------------------------------------------------------------------------


class TestDegenerateShapes:
    def test_single_quote_single_step(self):
        """1 quote, 1 step -- must pick the only option."""
        df = make_small_df(n_quotes=1, n_steps=1)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        out = result.dataframe

        assert out.shape[0] == 1
        assert out["optimal_step"][0] == 0
        # total_objective should equal the single expected_income value
        single_value = float(df["expected_income"][0])
        assert result.total_objective == pytest.approx(single_value, rel=1e-5)

    def test_single_quote_multiple_steps(self):
        """1 quote, 5 steps -- must pick the step with max expected_income."""
        df = make_small_df(n_quotes=1, n_steps=5)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        out = result.dataframe

        assert out.shape[0] == 1
        # Compute argmax manually from the input df
        best_idx = df["expected_income"].arg_max()
        assert out["optimal_step"][0] == best_idx

    def test_many_quotes_single_step(self):
        """100 quotes, 1 step each -- every quote must pick step 0."""
        df = make_small_df(n_quotes=100, n_steps=1)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        out = result.dataframe

        assert out.shape[0] == 100
        assert all(s == 0 for s in out["optimal_step"].to_list())

    def test_all_identical_objectives(self):
        """All steps have same objective -- any step is valid.

        Build a custom DataFrame (not make_small_df) where all
        expected_income values are 100.0 for every quote across all steps.
        """
        n_quotes = 10
        n_steps = 3
        rows = []
        for q in range(n_quotes):
            for j in range(n_steps):
                rows.append({
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": 0.8 + 0.1 * j,
                    "expected_income": 100.0,
                    "volume": 0.5,
                })
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

        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        out = result.dataframe

        assert out.shape[0] == n_quotes
        # All steps are valid -- just check indices are in range
        for s in out["optimal_step"].to_list():
            assert 0 <= s < n_steps

    def test_large_step_count(self):
        """10 quotes, 50 steps -- verify correct argmax per quote."""
        df = make_small_df(n_quotes=10, n_steps=50)
        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        out = result.dataframe

        assert out.shape[0] == 10
        # Check each quote against the input data
        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx, f"quote {qid}"


# ---------------------------------------------------------------------------
# Class 2: Impossible constraints
# ---------------------------------------------------------------------------


class TestImpossibleConstraints:
    def test_impossible_min_constraint_does_not_crash(self):
        """Volume min=2.0 (200% of baseline) is impossible.

        The solver should return a result without crashing and
        converged should be False.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 2.0}},
            max_iter=50,
        )
        result = solver.solve(df)
        assert result.converged is False

    def test_impossible_max_constraint_does_not_crash(self):
        """loss_ratio max=0.01 is impossibly tight.

        The solver should return a result without crashing and
        converged should be False.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": 0.01}},
            max_iter=50,
        )
        result = solver.solve(df)
        assert result.converged is False

    def test_conflicting_constraints_returns_result(self):
        """Conflicting constraints: volume min=0.99 AND loss_ratio max=0.90.

        The solver should not crash and the result should contain
        both constraint keys.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 0.99},
                "loss_ratio": {"max": 0.90},
            },
            max_iter=50,
        )
        result = solver.solve(df)
        assert "volume" in result.total_constraints
        assert "loss_ratio" in result.total_constraints

    def test_constraint_at_exactly_baseline(self):
        """Volume max=1.0 means 'do not exceed baseline volume'.

        On this synthetic data the unconstrained optimum has volume far
        above baseline (lower prices → higher conversion). A max=1.0
        constraint forces volume back down to the baseline level.
        """
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"max": 1.0}},
            max_iter=200,
        )
        result = solver.solve(df)

        baseline_vol = result.baseline_constraints["volume"]
        actual_vol = result.total_constraints["volume"]
        # Volume should be closer to baseline than the unconstrained optimum.
        # On small discrete data, the solver can't hit the target exactly,
        # but it should push volume down significantly.
        assert actual_vol <= baseline_vol * 1.10, (
            f"volume {actual_vol} exceeds baseline {baseline_vol} by more than 10%"
        )


# ---------------------------------------------------------------------------
# Class 3: Negative objectives
# ---------------------------------------------------------------------------


class TestNegativeObjectives:
    def test_negative_objective_values(self):
        """All expected_income values multiplied by -1.

        Unconstrained solve should pick the argmax (least negative) per quote.
        """
        df = make_small_df(n_quotes=20, n_steps=5)
        df = df.with_columns(
            (pl.col("expected_income") * -1).alias("expected_income")
        )

        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        out = result.dataframe

        assert out.shape[0] == 20
        # Each quote should pick the argmax of the (negative) expected_income
        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx, (
                f"quote {qid}: expected step {best_idx}, got {row['optimal_step']}"
            )

    def test_mixed_positive_negative_objectives(self):
        """Half the quotes have positive expected_income, half negative.

        Unconstrained solve should pick argmax per quote independently.
        """
        n_quotes = 20
        n_steps = 5
        df = make_small_df(n_quotes=n_quotes, n_steps=n_steps)

        # Build a mask: first half of quotes get multiplied by -1
        quote_ids = df["quote_id"].unique(maintain_order=True).to_list()
        negative_quotes = set(quote_ids[: n_quotes // 2])
        df = df.with_columns(
            pl.when(pl.col("quote_id").is_in(negative_quotes))
            .then(pl.col("expected_income") * -1)
            .otherwise(pl.col("expected_income"))
            .alias("expected_income")
        )

        solver = pc.OnlineOptimiser(objective="expected_income")
        result = solver.solve(df)
        out = result.dataframe

        assert out.shape[0] == n_quotes
        # Each quote should pick the argmax of its own expected_income
        for row in out.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["expected_income"].arg_max()
            assert row["optimal_step"] == best_idx, (
                f"quote {qid}: expected step {best_idx}, got {row['optimal_step']}"
            )
