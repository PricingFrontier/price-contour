"""0.5.0 consumer contract for ratebook results (DESIGN_DECISIONS §13.1–13.3,
§13.5, §13.7–13.9, §13.11).

Every reported ratebook number comes from one canonical evaluation of the
final factor tables; these tests pin that contract from the public API.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from helpers import make_factors, make_small_df
from price_contour import (
    PerFactorRecord,
    RatebookFactorContexts,
    RatebookOptimiser,
    quote_results_schema,
)
from price_contour._grid_utils import build_grid


def _grid_df(
    scenario_values: list[float],
    objective: list[list[float]],
    volume: list[list[float]] | None = None,
) -> pl.DataFrame:
    """Long-format frame: ``objective[q][k]`` for quote ``q`` at step ``k``."""
    rows = []
    for q, obj_row in enumerate(objective):
        for k, sv in enumerate(scenario_values):
            row = {
                "quote_id": f"Q{q}",
                "scenario_index": k,
                "scenario_value": sv,
                "expected_income": obj_row[k],
            }
            if volume is not None:
                row["volume"] = volume[q][k]
            rows.append(row)
    schema = {
        "quote_id": pl.Utf8,
        "scenario_index": pl.Int32,
        "scenario_value": pl.Float32,
        "expected_income": pl.Float32,
    }
    if volume is not None:
        schema["volume"] = pl.Float32
    return pl.DataFrame(rows, schema=schema)


def _solve_two_factor(**kwargs):
    df = make_small_df(n_quotes=60)
    factors = make_factors(60)
    opt = RatebookOptimiser(
        objective="expected_income",
        constraints={"volume": {"min_pct": 0.9}},
        factor_columns=[["region"], ["age_band"]],
        max_cd_iterations=3,
        max_iter=100,
        **kwargs,
    )
    result = opt.solve(df, factors)
    grid = build_grid(
        df,
        constraint_columns=["volume"],
        quote_id="quote_id",
        scenario_index="scenario_index",
        scenario_value="scenario_value",
        objective="expected_income",
    )
    contexts = RatebookFactorContexts.from_dataframe(
        factors,
        [["region"], ["age_band"]],
        quote_id=None,
        expected_quote_ids=grid.quote_ids,
    )
    return opt, result, grid, contexts


def _step_rule(scenario_values: np.ndarray, product: np.ndarray) -> np.ndarray:
    """Independent statement of the step rule: nearest f32 scenario value,
    ends clamp, an exact midpoint goes to the lower step."""
    steps = np.empty(product.shape[0], dtype=np.int64)
    for i, p in enumerate(product):
        if p <= scenario_values[0]:
            steps[i] = 0
        elif p >= scenario_values[-1]:
            steps[i] = len(scenario_values) - 1
        else:
            above = int(np.searchsorted(scenario_values, p, side="right"))
            below = above - 1
            d_below = np.float32(p - scenario_values[below])
            d_above = np.float32(scenario_values[above] - p)
            steps[i] = below if d_below <= d_above else above
    return steps


class TestQuoteResultsFrame:
    def test_schema_and_quote_order(self):
        _, result, grid, _ = _solve_two_factor()
        frame = result.quote_results
        assert dict(frame.schema) == quote_results_schema(["volume"])
        assert list(frame.columns) == [
            "quote_id",
            "optimal_step",
            "optimal_scenario_value",
            "optimal_objective",
            "optimal_volume",
            "factor_product",
            "clamped_low",
            "clamped_high",
        ]
        assert frame["quote_id"].to_list() == grid.quote_ids
        assert result.n_quotes == 60

    def test_totals_reconcile_with_frame(self):
        _, result, _, _ = _solve_two_factor()
        frame = result.quote_results
        assert math.isclose(
            frame["optimal_objective"].cast(pl.Float64).sum(),
            result.total_objective,
            rel_tol=1e-12,
        )
        assert math.isclose(
            frame["optimal_volume"].cast(pl.Float64).sum(),
            result.total_constraints["volume"],
            rel_tol=1e-12,
        )

    def test_factor_product_is_f32_product_of_tables(self):
        _, result, _, _ = _solve_two_factor()
        factors = make_factors(60)
        region = np.array(
            [result.factor_tables["region"][r] for r in factors["region"]],
            dtype=np.float32,
        )
        age = np.array(
            [result.factor_tables["age_band"][a] for a in factors["age_band"]],
            dtype=np.float32,
        )
        expected = (np.float32(1.0) * region) * age
        # make_factors rows are positional against the lex-sorted quote ids
        # Q0000..Q0059, which is also the grid order.
        assert np.array_equal(
            result.quote_results["factor_product"].to_numpy(), expected
        )

    def test_steps_follow_the_step_rule(self):
        _, result, grid, _ = _solve_two_factor()
        frame = result.quote_results
        sv = np.array(grid.scenario_values, dtype=np.float32)
        expected = _step_rule(sv, frame["factor_product"].to_numpy())
        assert np.array_equal(frame["optimal_step"].to_numpy(), expected)
        assert np.array_equal(frame["optimal_scenario_value"].to_numpy(), sv[expected])

    def test_evaluate_reproduces_result_exactly(self):
        opt, result, grid, contexts = _solve_two_factor()
        evaluation = opt.evaluate(grid, contexts, result.factor_tables)
        assert evaluation.total_objective == result.total_objective
        assert evaluation.total_constraints == result.total_constraints
        assert evaluation.quote_results.equals(result.quote_results)
        assert evaluation.n_quotes_clamped_low == result.n_quotes_clamped_low
        assert evaluation.n_quotes_clamped_high == result.n_quotes_clamped_high

    def test_evaluate_accepts_dataframes(self):
        opt, result, _, _ = _solve_two_factor()
        evaluation = opt.evaluate(
            make_small_df(60), make_factors(60), result.factor_tables
        )
        assert evaluation.total_objective == result.total_objective

    def test_scenario_values_and_baseline(self):
        _, result, grid, _ = _solve_two_factor()
        assert result.scenario_values == tuple(grid.scenario_values)
        assert result.baseline_scenario_value == grid.baseline_scenario_value
        assert result.baseline_scenario_value == pytest.approx(1.0)


class TestClampEdges:
    def test_clamp_counts_and_steps_at_both_edges(self):
        # One factor with three single-quote levels whose rates straddle the
        # grid [0.9, 1.0, 1.1]: 0.8 (below), 1.0 (inside), 1.2 (above).
        df = _grid_df([0.9, 1.0, 1.1], [[1.0, 2.0, 3.0]] * 3)
        factors = pl.DataFrame({"band": ["lo", "mid", "hi"]})
        opt = RatebookOptimiser(objective="expected_income", factor_columns=[["band"]])
        evaluation = opt.evaluate(
            df, factors, {"band": {"lo": 0.8, "mid": 1.0, "hi": 1.2}}
        )
        frame = evaluation.quote_results
        assert frame["optimal_step"].to_list() == [0, 1, 2]
        assert frame["clamped_low"].to_list() == [True, False, False]
        assert frame["clamped_high"].to_list() == [False, False, True]
        assert (evaluation.n_quotes_clamped_low, evaluation.n_quotes_clamped_high) == (
            1,
            1,
        )

    def test_product_exactly_on_an_edge_is_not_clamped(self):
        df = _grid_df([0.75, 1.0, 1.25], [[1.0, 2.0, 3.0]] * 2)
        factors = pl.DataFrame({"band": ["lo", "hi"]})
        opt = RatebookOptimiser(objective="expected_income", factor_columns=[["band"]])
        evaluation = opt.evaluate(df, factors, {"band": {"lo": 0.75, "hi": 1.25}})
        assert evaluation.quote_results["optimal_step"].to_list() == [0, 2]
        assert (evaluation.n_quotes_clamped_low, evaluation.n_quotes_clamped_high) == (
            0,
            0,
        )

    def test_all_quotes_clamped_low(self):
        # Every candidate (0.70..1.40) lies below a [2.0, 2.5] grid.
        df = _grid_df([2.0, 2.5], [[1.0, 2.0], [3.0, 1.0]])
        factors = pl.DataFrame({"band": ["a", "b"]})
        opt = RatebookOptimiser(
            objective="expected_income", factor_columns=[["band"]], max_cd_iterations=2
        )
        result = opt.solve(df, factors)
        assert result.n_quotes_clamped_low == 2
        assert result.clamp_rate == pytest.approx(1.0)
        assert result.quote_results["optimal_step"].to_list() == [0, 0]

    def test_exact_midpoint_goes_to_lower_step(self):
        df = _grid_df([0.75, 1.25], [[1.0, 2.0]])
        factors = pl.DataFrame({"band": ["a"]})
        opt = RatebookOptimiser(objective="expected_income", factor_columns=[["band"]])
        evaluation = opt.evaluate(df, factors, {"band": {"a": 1.0}})
        assert evaluation.quote_results["optimal_step"].to_list() == [0]


class TestClampRateDefinition:
    def test_hand_computed_clamp_rate(self):
        """§13.11 worked example.

        Grid [0.75, 1.0, 1.25]; 2 quotes; objective per step [0, 0, 1]; no
        constraints; factors A and B with one level each; candidates
        [0.5, 0.75, 1.0, 1.25, 1.5]; one CD pass.

        * Solve A (residual 1.0): targets 0.5 and 1.5 fall outside the grid
          → clamp fraction 2/5. Lagrangians per candidate are [0,0,0,2,2];
          the tie goes to the lowest → A = 1.25.
        * Solve B (residual 1.25): targets [0.625, 0.9375, 1.25, 1.5625,
          1.875]; three fall outside → 3/5. Lagrangians [0,0,2,2,2] → B = 1.0.
        * clamp_rate = mean(0.4, 0.6) = 0.5, yet the final product 1.25 sits
          exactly on the grid edge, so no quote is clamped.
        """
        df = _grid_df([0.75, 1.0, 1.25], [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        factors = pl.DataFrame({"a": ["x", "x"], "b": ["y", "y"]})
        opt = RatebookOptimiser(
            objective="expected_income",
            factor_columns=[["a"], ["b"]],
            candidate_min=0.5,
            candidate_max=1.5,
            candidate_steps=5,
            max_cd_iterations=1,
        )
        result = opt.solve(df, factors)
        assert result.factor_tables == {"a": {"x": 1.25}, "b": {"y": 1.0}}
        assert [r.clamp_rate for r in result.per_factor_results] == pytest.approx(
            [0.4, 0.6]
        )
        assert result.clamp_rate == pytest.approx(0.5, abs=1e-7)
        assert (result.n_quotes_clamped_low, result.n_quotes_clamped_high) == (0, 0)
        assert result.quote_results["optimal_step"].to_list() == [2, 2]


class TestPerFactorResults:
    def test_records_name_their_factor_and_pass(self):
        _, result, _, _ = _solve_two_factor()
        records = result.per_factor_results
        assert isinstance(records, tuple)
        assert all(isinstance(r, PerFactorRecord) for r in records)
        assert len(records) == result.cd_iterations * 2
        assert [r.factor for r in records] == [
            "region",
            "age_band",
        ] * result.cd_iterations
        assert [r.factor_index for r in records] == [0, 1] * result.cd_iterations
        assert [r.cd_iteration for r in records] == [
            i for i in range(1, result.cd_iterations + 1) for _ in range(2)
        ]
        assert set(records[0].total_constraints) == {"volume"}
        assert set(records[0].lambdas) == {"volume"}
        assert np.mean([r.clamp_rate for r in records]) == pytest.approx(
            result.clamp_rate, rel=1e-6
        )

    def test_composite_factor_name(self):
        df = make_small_df(n_quotes=40)
        factors = make_factors(40)
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
            factor_columns=[["region", "age_band"]],
            max_cd_iterations=1,
        )
        result = opt.solve(df, factors)
        assert [r.factor for r in result.per_factor_results] == ["region:age_band"]
        assert all("\x1f" in label for label in result.factor_tables["region:age_band"])


class TestBoundsAndOrdering:
    def test_min_pct_bound_is_fraction_of_baseline(self):
        _, result, _, _ = _solve_two_factor()
        assert result.constraint_bounds == {
            "volume": result.baseline_constraints["volume"] * 0.9
        }

    def test_absolute_bound_is_the_threshold(self):
        df = make_small_df(n_quotes=40)
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 17.5}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
        )
        result = opt.solve(df, make_factors(40))
        assert result.constraint_bounds == {"volume": 17.5}

    def test_zero_baseline_pct_raises(self):
        df = _grid_df([0.9, 1.0, 1.1], [[1.0, 2.0, 3.0]], volume=[[0.0, 0.0, 0.0]])
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
            factor_columns=[["band"]],
        )
        with pytest.raises(
            ValueError, match="min_pct/max_pct on 'volume' is undefined"
        ):
            opt.solve(df, pl.DataFrame({"band": ["a"]}))

    def test_dict_outputs_follow_constraint_order(self):
        df = make_small_df(n_quotes=40)
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.8}, "loss_ratio": {"max": 1e9}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
        )
        result = opt.solve(df, make_factors(40))
        order = list(result.total_constraints)
        assert list(result.lambdas) == order
        assert list(result.baseline_constraints) == order
        assert list(result.constraint_bounds) == order

    def test_summary_requires_every_constraint(self):
        opt, result, _, _ = _solve_two_factor()
        del result.total_constraints["volume"]
        with pytest.raises(KeyError, match="volume"):
            opt.summary(result)

    def test_unknown_warm_start_lambda_raises(self):
        df = make_small_df(n_quotes=20)
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
            factor_columns=[["region"]],
        )
        with pytest.raises(ValueError, match="unknown constraint"):
            opt.solve(df, make_factors(20), lambdas={"volum": 0.1})


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"candidate_min": 0.0}, "candidate_min"),
            ({"candidate_min": -0.5}, "candidate_min"),
            ({"candidate_min": float("nan")}, "candidate_min"),
            ({"candidate_max": float("inf")}, "candidate_max"),
            ({"candidate_min": 1.2, "candidate_max": 1.1}, "candidate_max"),
            ({"candidate_steps": 0}, "candidate_steps"),
            ({"max_cd_iterations": 0}, "max_cd_iterations"),
        ],
    )
    def test_invalid_candidate_settings_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RatebookOptimiser(objective="expected_income", **kwargs)

    @pytest.mark.parametrize("name", ["objective", "step", "scenario_value"])
    def test_reserved_constraint_names_raise(self, name):
        with pytest.raises(ValueError, match="reserved"):
            RatebookOptimiser(
                objective="expected_income", constraints={name: {"min": 1.0}}
            )

    @pytest.mark.parametrize("column", ["a:b", "a\x1fb"])
    def test_factor_column_with_a_separator_raises(self, column):
        df = make_small_df(n_quotes=12)
        factors = pl.DataFrame({column: ["x", "y"] * 6})
        opt = RatebookOptimiser(objective="expected_income", factor_columns=[[column]])
        with pytest.raises(ValueError, match="factor column"):
            opt.solve(df, factors)

    def test_duplicate_factor_names_raise(self):
        df = make_small_df(n_quotes=12)
        factors = pl.DataFrame({"region": ["x", "y"] * 6})
        opt = RatebookOptimiser(
            objective="expected_income", factor_columns=[["region"], ["region"]]
        )
        with pytest.raises(ValueError, match="more than once"):
            opt.solve(df, factors)

    def test_contexts_with_a_custom_separator_raise(self):
        df = make_small_df(n_quotes=12)
        grid = build_grid(
            df,
            constraint_columns=[],
            quote_id="quote_id",
            scenario_index="scenario_index",
            scenario_value="scenario_value",
            objective="expected_income",
        )
        factors = make_factors(12)
        contexts = RatebookFactorContexts.from_dataframe(
            factors,
            [["region", "age_band"]],
            quote_id=None,
            separator="|",
            expected_quote_ids=grid.quote_ids,
        )
        opt = RatebookOptimiser(objective="expected_income")
        with pytest.raises(ValueError, match="separator"):
            opt.solve(grid, contexts)

    def test_evaluate_rejects_unknown_factor(self):
        opt, result, grid, contexts = _solve_two_factor()
        tables = dict(result.factor_tables) | {"channel": {"web": 1.0}}
        with pytest.raises(ValueError, match="'channel'"):
            opt.evaluate(grid, contexts, tables)

    def test_evaluate_rejects_missing_factor(self):
        opt, result, grid, contexts = _solve_two_factor()
        with pytest.raises(ValueError, match="'age_band'"):
            opt.evaluate(grid, contexts, {"region": result.factor_tables["region"]})

    def test_evaluate_rejects_missing_level(self):
        opt, result, grid, contexts = _solve_two_factor()
        tables = {k: dict(v) for k, v in result.factor_tables.items()}
        del tables["region"]["North"]
        with pytest.raises(ValueError, match="'North'"):
            opt.evaluate(grid, contexts, tables)

    def test_evaluate_rejects_extra_level(self):
        opt, result, grid, contexts = _solve_two_factor()
        tables = {k: dict(v) for k, v in result.factor_tables.items()}
        tables["region"]["Atlantis"] = 1.0
        with pytest.raises(ValueError, match="'Atlantis'"):
            opt.evaluate(grid, contexts, tables)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_evaluate_rejects_non_positive_rates(self, bad):
        opt, result, grid, contexts = _solve_two_factor()
        tables = {k: dict(v) for k, v in result.factor_tables.items()}
        tables["region"]["North"] = bad
        with pytest.raises(ValueError, match="finite and > 0"):
            opt.evaluate(grid, contexts, tables)

    def test_evaluate_rejects_ratio_constraints(self):
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={
                "lr": {
                    "numerator": "volume",
                    "denominator": "expected_income",
                    "max": 1.0,
                }
            },
            factor_columns=[["region"]],
        )
        with pytest.raises(ValueError, match="ratio"):
            opt.evaluate(make_small_df(20), make_factors(20), {"region": {}})
