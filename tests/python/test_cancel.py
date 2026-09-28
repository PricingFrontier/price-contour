"""Cooperative cancellation of apply_from_grid and RatebookOptimiser.evaluate
(DESIGN_DECISIONS §14)."""

from __future__ import annotations

import threading

import numpy as np
import polars as pl
import pytest
from price_contour import (
    Cancelled,
    CancelToken,
    RatebookFactorContexts,
    RatebookOptimiser,
    apply_from_grid,
)
from price_contour._grid_utils import build_grid

# More quotes than one cancellation poll block (CANCEL_POLL_QUOTES = 65_536)
# and many argmax grains, so every polling loop crosses a block boundary.
N_QUOTES = 70_001
SCENARIO_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
CONSTRAINTS = {"volume": {"min_pct": 0.9}}
LAMBDAS = {"volume": 3.5}
REGIONS = ["North", "South", "East", "West", "Mid"]


def _book(n_quotes: int = N_QUOTES) -> pl.DataFrame:
    n_steps = len(SCENARIO_VALUES)
    q = np.repeat(np.arange(n_quotes), n_steps)
    sv = np.tile(np.array(SCENARIO_VALUES, dtype=np.float32), n_quotes)
    elasticity = 1.5 + 3.5 * (q % 97) / 97
    conversion = 1.0 / (1.0 + np.exp(elasticity * (sv - 1.0)))
    return pl.DataFrame(
        {
            "quote_id": [f"Q{i:06d}" for i in q],
            "scenario_index": np.tile(np.arange(n_steps, dtype=np.int32), n_quotes),
            "scenario_value": sv,
            "expected_income": ((80.0 + (q % 41)) * sv * conversion).astype(np.float32),
            "volume": conversion.astype(np.float32),
        }
    )


@pytest.fixture(scope="module")
def grid():
    return build_grid(
        _book(),
        constraint_columns=["volume"],
        quote_id="quote_id",
        scenario_index="scenario_index",
        scenario_value="scenario_value",
        objective="expected_income",
    )


@pytest.fixture(scope="module")
def ratebook(grid):
    factors = pl.DataFrame({"region": [REGIONS[i % 5] for i in range(N_QUOTES)]})
    contexts = RatebookFactorContexts.from_dataframe(
        factors, [["region"]], quote_id=None, expected_quote_ids=grid.quote_ids
    )
    optimiser = RatebookOptimiser(
        objective="expected_income",
        constraints=CONSTRAINTS,
        factor_columns=[["region"]],
    )
    tables = {"region": {r: 0.85 + 0.07 * i for i, r in enumerate(REGIONS)}}
    return optimiser, contexts, tables


def _cancelled() -> CancelToken:
    token = CancelToken()
    token.cancel()
    return token


class TestCancelToken:
    def test_is_a_one_way_flag(self):
        token = CancelToken()
        assert not token.cancelled
        token.cancel()
        token.cancel()
        assert token.cancelled
        assert repr(token) == "CancelToken(cancelled=True)"

    def test_cancelled_is_a_runtime_error_not_a_value_error(self):
        assert issubclass(Cancelled, RuntimeError)
        assert not issubclass(Cancelled, ValueError)

    def test_cancel_is_keyword_only(self, grid):
        with pytest.raises(TypeError):
            apply_from_grid(grid, LAMBDAS, CONSTRAINTS, CancelToken())  # type: ignore[misc]


class TestApplyFromGrid:
    def test_uncancelled_token_matches_no_token(self, grid):
        plain = apply_from_grid(grid, LAMBDAS, CONSTRAINTS)
        tokened = apply_from_grid(grid, LAMBDAS, CONSTRAINTS, cancel=CancelToken())
        assert tokened.dataframe.equals(plain.dataframe)
        assert tokened.lambdas == plain.lambdas
        assert tokened.baseline_objective == plain.baseline_objective
        assert tokened.baseline_constraints == plain.baseline_constraints
        assert tokened.total_objective == pytest.approx(
            plain.total_objective, rel=1e-12
        )
        assert tokened.total_constraints["volume"] == pytest.approx(
            plain.total_constraints["volume"], rel=1e-12
        )

    def test_pre_cancelled_token_raises(self, grid):
        with pytest.raises(Cancelled):
            apply_from_grid(grid, LAMBDAS, CONSTRAINTS, cancel=_cancelled())

    def test_pre_cancelled_token_raises_with_absolute_bounds(self, grid):
        # No pct bound, so no baseline scan while parsing: the argmax's own
        # entry check raises.
        with pytest.raises(Cancelled):
            apply_from_grid(
                grid, LAMBDAS, {"volume": {"min": 1.0}}, cancel=_cancelled()
            )

    @pytest.mark.parametrize(
        ("lambdas", "constraints"),
        [
            ({"nope": 1.0}, CONSTRAINTS),
            (LAMBDAS, {"nope": {"min": 1.0}}),
            (LAMBDAS, {"volume": {"min": 1.0, "max": 2.0}}),
        ],
    )
    def test_input_errors_win_over_cancellation(self, grid, lambdas, constraints):
        with pytest.raises(ValueError):
            apply_from_grid(grid, lambdas, constraints, cancel=_cancelled())

    def test_cancel_after_return_stops_the_cold_frame_on_another_thread(self, grid):
        token = CancelToken()
        results: dict[str, object] = {}

        def produce():
            results["result"] = apply_from_grid(
                grid, LAMBDAS, CONSTRAINTS, cancel=token
            )

        def read():
            try:
                results["frame"] = results["result"].dataframe  # type: ignore[attr-defined]
            except Cancelled as exc:
                results["error"] = exc

        for target in (produce, token.cancel, read):
            thread = threading.Thread(target=target)
            thread.start()
            thread.join()

        assert isinstance(results.get("error"), Cancelled)
        assert "frame" not in results
        # Nothing was cached: a second access raises again.
        with pytest.raises(Cancelled):
            results["result"].dataframe  # type: ignore[attr-defined]

    def test_a_built_frame_survives_a_later_cancel(self, grid):
        token = CancelToken()
        result = apply_from_grid(grid, LAMBDAS, CONSTRAINTS, cancel=token)
        frame = result.dataframe
        token.cancel()
        assert result.dataframe.equals(frame)

    def test_racing_cancels_never_yield_a_partial_result(self, grid):
        expected = apply_from_grid(grid, LAMBDAS, CONSTRAINTS).dataframe
        outcomes = set()
        rng = np.random.default_rng(7)
        for delay in rng.uniform(0.0, 0.004, size=40):
            token = CancelToken()
            timer = threading.Timer(float(delay), token.cancel)
            timer.start()
            try:
                frame = apply_from_grid(
                    grid, LAMBDAS, CONSTRAINTS, cancel=token
                ).dataframe
            except Cancelled:
                outcomes.add("cancelled")
            else:
                assert frame.equals(expected)
                outcomes.add("completed")
            finally:
                timer.join()
        assert "cancelled" in outcomes


class TestRatebookEvaluate:
    def test_uncancelled_token_matches_no_token(self, grid, ratebook):
        optimiser, contexts, tables = ratebook
        plain = optimiser.evaluate(grid, contexts, tables)
        tokened = optimiser.evaluate(grid, contexts, tables, cancel=CancelToken())
        assert tokened.quote_results.equals(plain.quote_results)
        assert tokened.total_objective == plain.total_objective
        assert tokened.total_constraints == plain.total_constraints
        assert tokened.baseline_objective == plain.baseline_objective
        assert tokened.baseline_constraints == plain.baseline_constraints

    def test_pre_cancelled_token_raises(self, grid, ratebook):
        optimiser, contexts, tables = ratebook
        with pytest.raises(Cancelled):
            optimiser.evaluate(grid, contexts, tables, cancel=_cancelled())

    def test_input_errors_win_over_cancellation(self, grid, ratebook):
        optimiser, contexts, tables = ratebook
        bad = {"region": {**tables["region"], "North": -1.0}}
        with pytest.raises(ValueError):
            optimiser.evaluate(grid, contexts, bad, cancel=_cancelled())

    def test_cancel_after_return_stops_the_cold_frame(self, grid, ratebook):
        optimiser, contexts, tables = ratebook
        token = CancelToken()
        evaluation = optimiser.evaluate(grid, contexts, tables, cancel=token)
        cancel_thread = threading.Thread(target=token.cancel)
        cancel_thread.start()
        cancel_thread.join()
        with pytest.raises(Cancelled):
            evaluation.quote_results
        with pytest.raises(Cancelled):
            evaluation.quote_results

    def test_racing_cancels_never_yield_a_partial_result(self, grid, ratebook):
        optimiser, contexts, tables = ratebook
        expected = optimiser.evaluate(grid, contexts, tables).quote_results
        outcomes = set()
        rng = np.random.default_rng(11)
        for delay in rng.uniform(0.0, 0.004, size=40):
            token = CancelToken()
            timer = threading.Timer(float(delay), token.cancel)
            timer.start()
            try:
                frame = optimiser.evaluate(
                    grid, contexts, tables, cancel=token
                ).quote_results
            except Cancelled:
                outcomes.add("cancelled")
            else:
                assert frame.equals(expected)
                outcomes.add("completed")
            finally:
                timer.join()
        assert "cancelled" in outcomes
