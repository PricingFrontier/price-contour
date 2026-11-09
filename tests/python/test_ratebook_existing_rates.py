"""Tests for today's rate tables as a starting point (DESIGN_DECISIONS §18)."""

from __future__ import annotations

import json
import math
import struct

import numpy as np
import polars as pl
import pytest

import price_contour as pc
from price_contour.ratebook import (
    _CONFIG_KEYS_V2,
    RatebookOptimiser,
    RatebookResult,
    ResultUnavailableError,
)
from test_ratebook_base_multiplier import (
    FINE_SV,
    N_STRICT,
    SPECS,
    factor_frame,
    scored_frame,
    step_rule,
    strict_matrices,
    strict_optimiser,
)


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def existing_for(
    factors: pl.DataFrame, *, seed: int = 8
) -> dict[str, dict[str, float]]:
    """Random rates for every level of the SPECS factors, in first-appearance order."""
    rng = np.random.default_rng(seed)
    return {
        spec[0]: {
            level: float(rng.uniform(0.8, 1.25))
            for level in factors[spec[0]].unique(maintain_order=True).to_list()
        }
        for spec in SPECS
    }


def book(n_quotes: int = N_STRICT):
    objective, volume = strict_matrices(FINE_SV, n_quotes)
    df = scored_frame(FINE_SV, objective, volume=volume)
    return df, factor_frame(n_quotes)


def product_of(tables, factors: pl.DataFrame) -> np.ndarray:
    """The f32 product of the tables, left to right in spec order, by quote_id."""
    frame = factors.sort("quote_id")
    product = np.ones(frame.height, dtype=np.float32)
    for spec in SPECS:
        table = tables[spec[0]]
        product = product * np.array(
            [table[level] for level in frame[spec[0]].to_list()], dtype=np.float32
        )
    return product


class TestFreshCard:
    def test_output_is_existing_times_change(self):
        df, factors = book()
        existing = existing_for(factors)
        # The same starting prices as a per-quote column give the changes alone.
        start = product_of(existing, factors)
        by_id = factors.sort("quote_id").with_columns(
            pl.Series("start", start, dtype=pl.Float32)
        )
        relative = strict_optimiser(base_multiplier="start").solve(df, by_id)

        fresh = strict_optimiser(existing_rates=existing).solve(df, factors)

        assert fresh.lambdas == relative.lambdas
        assert fresh.baseline_constraints == relative.baseline_constraints
        changed = 0
        for spec in SPECS:
            name = spec[0]
            for level, rate in fresh.factor_tables[name].items():
                delta = relative.factor_tables[name][level]
                assert rate == f32(f32(existing[name][level]) * delta)
                changed += delta != 1.0
        assert changed > 0

    def test_an_unchanged_level_is_exactly_its_existing_rate(self):
        df = scored_frame([0.8, 1.0, 1.2], [[5.0, 5.0, 5.0]])
        factors = pl.DataFrame({"quote_id": ["q0000"], "f": ["l"]})
        rate = 1.0123456789
        result = RatebookOptimiser(
            objective="objective",
            constraints={},
            factor_columns=[["f"]],
            existing_rates={"f": {"l": rate}},
        ).solve(df, factors)
        assert result.factor_tables["f"]["l"] == f32(rate)

    def test_levels_only_in_existing_rates_are_carried_after_the_datas(self):
        df, factors = book()
        existing = existing_for(factors)
        existing["region"]["unseen"] = 1.37
        result = strict_optimiser(existing_rates=existing).solve(df, factors)
        assert result.factor_tables["region"]["unseen"] == f32(1.37)
        assert list(result.factor_tables["region"])[-1] == "unseen"

    def test_evaluate_accepts_carried_levels_and_no_others(self):
        df, factors = book()
        existing = existing_for(factors)
        existing["region"]["unseen"] = 1.37
        optimiser = strict_optimiser(existing_rates=existing)
        result = optimiser.solve(df, factors)
        evaluation = optimiser.evaluate(df, factors, result.factor_tables)
        assert evaluation.total_objective == result.total_objective
        assert evaluation.total_constraints == result.total_constraints

        tables = {k: dict(v) for k, v in result.factor_tables.items()}
        tables["region"]["other"] = 1.0
        with pytest.raises(ValueError, match="that no quote has"):
            optimiser.evaluate(df, factors, tables)

    @pytest.mark.parametrize(
        "bad",
        [float("nan"), float("inf"), 0.0, -1.0, None, "x"],
        ids=["nan", "inf", "zero", "negative", "none", "string"],
    )
    def test_carried_levels_in_supplied_tables_are_validated(self, bad):
        df, factors = book()
        existing = existing_for(factors)
        existing["region"]["unseen"] = 1.37
        optimiser = strict_optimiser(existing_rates=existing)
        result = optimiser.solve(df, factors)
        tables = {k: dict(v) for k, v in result.factor_tables.items()}
        tables["region"]["unseen"] = bad
        with pytest.raises(
            ValueError, match=r"invalid rate\(s\) for level\(s\) \['unseen'\]"
        ):
            optimiser.evaluate(df, factors, tables)

    def test_a_result_keeps_its_own_copy_of_the_rates(self):
        df, factors = book()
        existing = existing_for(factors)
        optimiser = strict_optimiser(existing_rates=existing)
        result = optimiser.solve(df, factors)
        before = result.existing_rates["region"]["N"]
        optimiser.existing_rates["region"]["N"] = 9.0
        assert result.existing_rates["region"]["N"] == before

    def test_frontier_points_are_cards_and_reproduce(self):
        df, factors = book()
        existing = existing_for(factors)
        existing["age"]["unseen"] = 0.9
        optimiser = strict_optimiser(
            constraints={"volume": {"min_pct": None}}, existing_rates=existing
        )
        frontier = optimiser.frontier(
            df, factors, threshold_ranges={"volume": (0.97, 1.0)}, n_points_per_dim=3
        )
        for i in range(frontier.n_points):
            tables = frontier.point_factor_tables(i)
            assert tables["age"]["unseen"] == f32(0.9)
            evaluation = optimiser.evaluate(df, factors, tables)
            assert evaluation.total_objective == frontier.points["total_objective"][i]
            assert (
                evaluation.total_constraints["volume"]
                == frontier.points["total_volume"][i]
            )

    def test_contexts_give_the_same_results(self):
        df, factors = book()
        existing = existing_for(factors)
        optimiser = strict_optimiser(existing_rates=existing)
        builder = pc.QuoteGridBuilder(["volume"], objective="objective")
        builder.append(df)
        grid = builder.build()
        contexts = pc.RatebookFactorContexts.from_dataframe(
            factors, SPECS, expected_quote_ids=grid.quote_ids
        )
        direct = optimiser.solve(df, factors)
        via = optimiser.solve(grid, contexts)
        assert direct.factor_tables == via.factor_tables
        assert direct.total_objective == via.total_objective


class TestDeployedByItself:
    def test_the_output_tables_alone_reprice_every_quote(self):
        n_quotes = 4000
        rng = np.random.default_rng(21)
        commissions = np.round(np.arange(0.0, 0.40, 0.01), 2)
        odds_grid = commissions / (0.4 - commissions)
        sv = odds_grid.astype(np.float32)
        levels = {
            "region": [f"r{i}" for i in range(6)],
            "age": [f"a{i}" for i in range(8)],
            "scheme": [f"s{i}" for i in range(4)],
        }
        w0 = {
            f: {level: float(rng.normal(-0.4, 0.3)) for level in ls}
            for f, ls in levels.items()
        }
        membership = {f: rng.choice(ls, n_quotes).tolist() for f, ls in levels.items()}
        odds0 = np.exp(
            sum(
                np.array([w0[f][level] for level in membership[f]], dtype=np.float64)
                for f in levels
            )
        )
        current = 0.4 * odds0 / (1.0 + odds0)
        peak = np.clip(current + rng.normal(0.02, 0.03, n_quotes), 0.0, 0.39)[:, None]
        conversion = 1.0 - 8.0 * (commissions[None, :] - peak) ** 2
        profit = (0.5 - commissions[None, :]) * conversion
        ids = [f"q{i:05d}" for i in range(n_quotes)]
        df = scored_frame(sv, conversion, quote_ids=ids, profit=profit)
        factors = pl.DataFrame({"quote_id": ids, **membership})

        result = RatebookOptimiser(
            objective="objective",
            constraints={"profit": {"min_pct": 0.98}},
            factor_columns=[[f] for f in levels],
            candidate_min=math.exp(-0.2),
            candidate_max=math.exp(0.2),
            candidate_steps=41,
            existing_rates={
                f: {level: math.exp(w) for level, w in table.items()}
                for f, table in w0.items()
            },
        ).solve(df, factors)
        frame = result.quote_results

        # Exact: the f32 product of the output tables, nothing from today's card.
        product = np.ones(n_quotes, dtype=np.float32)
        for f in levels:
            table = result.factor_tables[f]
            product = product * np.array(
                [table[level] for level in membership[f]], dtype=np.float32
            )
        np.testing.assert_array_equal(frame["factor_product"].to_numpy(), product)
        np.testing.assert_array_equal(
            frame["optimal_step"].to_numpy(), step_rule(sv, product)
        )

        # In logs, f64: agrees to rounding, and on the step away from boundaries.
        odds = np.exp(
            sum(
                np.array(
                    [
                        math.log(result.factor_tables[f][level])
                        for level in membership[f]
                    ],
                    dtype=np.float64,
                )
                for f in levels
            )
        )
        np.testing.assert_allclose(odds, product.astype(np.float64), rtol=1e-5)
        boundaries = (odds_grid[:-1] + odds_grid[1:]) / 2.0
        gap = np.min(
            np.abs(odds[:, None] - boundaries[None, :]) / boundaries[None, :], axis=1
        )
        clear = gap > 1e-5
        assert clear.mean() > 0.99
        below = np.clip(np.searchsorted(odds_grid, odds), 1, len(odds_grid) - 1)
        lower = np.abs(odds - odds_grid[below - 1]) <= np.abs(odds_grid[below] - odds)
        expected = np.where(lower, below - 1, below)
        np.testing.assert_array_equal(
            frame["optimal_step"].to_numpy()[clear], expected[clear]
        )


class TestHeterogeneousStartByHand:
    """Two quotes share a level but start apart through another factor's rate."""

    SV = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

    def _run(self, objective, specs, existing):
        df = scored_frame(self.SV, objective, quote_ids=["q0", "q1"])
        columns = {
            spec[0]: ["z0", "z1"] if spec[0] == "z" else ["l", "l"] for spec in specs
        }
        factors = pl.DataFrame({"quote_id": ["q0", "q1"], **columns})
        return RatebookOptimiser(
            objective="objective",
            constraints={},
            factor_columns=specs,
            candidate_min=0.5,
            candidate_max=1.5,
            candidate_steps=3,
            max_cd_iterations=1,
            existing_rates=existing,
        ).solve(df, factors)

    def test_a_shared_level_sees_each_quotes_start(self):
        # a: residuals [1, 2] give totals [10, 1, 0], so 0.5. A search starting
        # both quotes from q0's price would pick 1.0 (totals [0, 11, 8]).
        # z: objective flat for q0 and best unchanged for q1, so 1.0 for both.
        result = self._run(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 8.0, 0.0, 0.0, 0.0]],
            [["a"], ["z"]],
            {"a": {"l": 1.0}, "z": {"z0": 1.0, "z1": 2.0}},
        )
        assert result.factor_tables == {"a": {"l": 0.5}, "z": {"z0": 1.0, "z1": 2.0}}
        assert result.total_objective == 10.0

    def test_a_later_factor_keeps_the_start_in_its_residuals(self):
        # a: 0.5 as above. b: residuals [0.5, 1] give totals [0, 10, 12], so 1.5.
        result = self._run(
            [[0.0] * 6, [0.0, 10.0, 12.0, 0.0, 0.0, 0.0]],
            [["a"], ["b"], ["z"]],
            {"a": {"l": 1.0}, "b": {"l": 1.0}, "z": {"z0": 1.0, "z1": 2.0}},
        )
        assert result.factor_tables == {
            "a": {"l": 0.5},
            "b": {"l": 1.5},
            "z": {"z0": 1.0, "z1": 2.0},
        }
        assert result.total_objective == 12.0


class TestBaseline:
    SV = (0.5, 1.0, 2.0, 4.0)

    def _frames(self):
        objective = [[100 * q + k for k in range(4)] for q in range(2)]
        vol = [[10 * q + k for k in range(4)] for q in range(2)]
        df = scored_frame(self.SV, objective, vol=vol, quote_ids=["q0", "q1"])
        factors = pl.DataFrame(
            {
                "quote_id": ["q0", "q1"],
                "f": ["x", "y"],
                "g": ["u", "v"],
                "col": [2.0, 1.0],
            }
        )
        return df, factors

    def _optimiser(self, **kwargs) -> RatebookOptimiser:
        settings = {
            "objective": "objective",
            "constraints": {"vol": {"min": 0.0}},
            "factor_columns": [["f"]],
            "candidate_min": 0.9,
            "candidate_max": 1.1,
            "candidate_steps": 3,
        }
        return RatebookOptimiser(**{**settings, **kwargs})

    def test_existing_rates_alone_by_hand(self):
        df, factors = self._frames()
        tables = {"f": {"x": 1.0, "y": 2.0}}
        evaluation = self._optimiser(existing_rates=tables).evaluate(
            df, factors, tables
        )
        frame = evaluation.quote_results
        assert frame["baseline_product"].to_list() == [1.0, 2.0]
        assert frame["baseline_step"].to_list() == [1, 2]
        assert "base_multiplier" not in frame.columns
        assert evaluation.baseline_objective == 1.0 + 102.0
        assert evaluation.baseline_constraints == {"vol": 1.0 + 12.0}

    def test_both_inputs_multiply_by_hand(self):
        df, factors = self._frames()
        tables = {"f": {"x": 0.5, "y": 2.0}}
        evaluation = self._optimiser(
            base_multiplier="col", existing_rates=tables
        ).evaluate(df, factors, tables)
        frame = evaluation.quote_results
        assert frame["base_multiplier"].to_list() == [2.0, 1.0]
        assert frame["baseline_product"].to_list() == [1.0, 2.0]
        assert frame["factor_product"].to_list() == [1.0, 2.0]
        assert frame["baseline_step"].to_list() == [1, 2]
        assert frame["optimal_step"].to_list() == [1, 2]
        assert dict(frame.schema) == pc.quote_results_schema(
            ["vol"], base_multiplier=True, existing_rates=True
        )

    def test_a_factor_can_move_between_the_two_inputs(self):
        df, factors = self._frames()
        f_rates = {"x": 0.5, "y": 2.0}
        g_rates = {"u": 2.0, "v": 1.0}
        whole = self._optimiser(
            factor_columns=[["f"], ["g"]], existing_rates={"f": f_rates, "g": g_rates}
        ).evaluate(df, factors, {"f": f_rates, "g": g_rates})
        # `col` is g's rate per quote, so g can sit in the per-quote column.
        split = self._optimiser(
            base_multiplier="col", existing_rates={"f": f_rates}
        ).evaluate(df, factors, {"f": f_rates})
        assert whole.baseline_objective == split.baseline_objective
        assert whole.baseline_constraints == split.baseline_constraints
        assert (
            whole.quote_results["baseline_step"].to_list()
            == split.quote_results["baseline_step"].to_list()
        )

    def test_pct_bounds_use_the_card(self):
        df, factors = self._frames()
        result = self._optimiser(
            constraints={"vol": {"min_pct": 0.5}},
            existing_rates={"f": {"x": 1.0, "y": 2.0}},
        ).solve(df, factors)
        assert result.constraint_bounds == {"vol": 13.0 * 0.5}

    def test_baseline_scenario_value_is_unavailable(self):
        df, factors = self._frames()
        tables = {"f": {"x": 1.0, "y": 2.0}}
        optimiser = self._optimiser(existing_rates=tables)
        with pytest.raises(RuntimeError, match="baseline_step"):
            _ = optimiser.evaluate(df, factors, tables).baseline_scenario_value
        with pytest.raises(ResultUnavailableError, match="baseline_step"):
            _ = optimiser.solve(df, factors).baseline_scenario_value

    def test_a_token_covers_both_evaluations(self):
        df, factors = self._frames()
        tables = {"f": {"x": 1.0, "y": 2.0}}
        optimiser = self._optimiser(existing_rates=tables)
        plain = optimiser.evaluate(df, factors, tables)
        tokened = optimiser.evaluate(df, factors, tables, cancel=pc.CancelToken())
        assert plain.total_objective == tokened.total_objective
        assert plain.baseline_constraints == tokened.baseline_constraints
        cancelled = pc.CancelToken()
        cancelled.cancel()
        with pytest.raises(pc.Cancelled):
            optimiser.evaluate(df, factors, tables, cancel=cancelled)


class TestAgreementWithAColumn:
    def test_rates_of_one_change_nothing_beside_a_column(self):
        df, factors = book()
        rng = np.random.default_rng(5)
        by_id = factors.with_columns(
            pl.Series("base", np.exp(rng.normal(0.0, 0.15, factors.height)))
        )
        ones = {
            spec[0]: dict.fromkeys(
                factors[spec[0]].unique(maintain_order=True).to_list(), 1.0
            )
            for spec in SPECS
        }
        column = strict_optimiser(base_multiplier="base").solve(df, by_id)
        both = strict_optimiser(base_multiplier="base", existing_rates=ones).solve(
            df, by_id
        )
        assert both.factor_tables == column.factor_tables
        assert both.lambdas == column.lambdas
        assert both.total_objective == column.total_objective
        assert both.baseline_constraints == column.baseline_constraints


class TestValidation:
    def _small(self):
        df = scored_frame(
            [0.5, 1.0, 1.5], [[1.0, 2.0, 3.0]] * 2, volume=[[1.0, 1.0, 1.0]] * 2
        )
        factors = pl.DataFrame({"quote_id": ["q0000", "q0001"], "f": ["a", "b"]})
        return df, factors

    def _optimiser(self, existing, **kwargs) -> RatebookOptimiser:
        settings = {
            "objective": "objective",
            "constraints": {"volume": {"min": 0.0}},
            "factor_columns": [["f"]],
            "existing_rates": existing,
        }
        return RatebookOptimiser(**{**settings, **kwargs})

    @pytest.mark.parametrize(
        "bad",
        [None, float("nan"), float("inf"), 0.0, -1.0, 1e300, 1e-300],
        ids=[
            "null",
            "nan",
            "inf",
            "zero",
            "negative",
            "overflows_f32",
            "underflows_f32",
        ],
    )
    def test_invalid_rates_report_the_count_and_an_example(self, bad):
        with pytest.raises(ValueError, match=r"2 invalid.*'f'.*'b'"):
            self._optimiser({"f": {"a": 1.0, "b": bad, "c": bad}})

    @pytest.mark.parametrize(
        "bad", [1.0, {"f": 1.0}, {"f": {"a": "x"}}, {"f": {1: 1.0}}], ids=str
    )
    def test_a_wrong_shape_is_a_type_error(self, bad):
        with pytest.raises(TypeError, match="existing_rates"):
            self._optimiser(bad)

    def test_a_level_in_the_data_without_a_rate_raises_naming_it(self):
        df, factors = self._small()
        with pytest.raises(ValueError, match=r"no rate for level\(s\) \['b'\]"):
            self._optimiser({"f": {"a": 1.0}}).solve(df, factors)

    def test_unknown_and_missing_factors_raise(self):
        df, factors = self._small()
        with pytest.raises(ValueError, match=r"unknown factor\(s\) \['g'\]"):
            self._optimiser({"f": {"a": 1.0, "b": 1.0}, "g": {"x": 1.0}}).solve(
                df, factors
            )
        with pytest.raises(ValueError, match=r"for factor\(s\) \['f'\]"):
            self._optimiser({}).solve(df, factors)

    @pytest.mark.parametrize("call", ["solve", "evaluate", "frontier"])
    def test_coverage_is_checked_before_any_grid_build(self, call):
        df, factors = self._small()
        broken = df.drop("objective")  # could never build a grid
        optimiser = self._optimiser({"f": {"a": 1.0}})
        with pytest.raises(ValueError, match="no rate for level"):
            if call == "solve":
                optimiser.solve(broken, factors)
            elif call == "evaluate":
                optimiser.evaluate(broken, factors, {"f": {"a": 1.0, "b": 1.0}})
            else:
                optimiser.frontier(
                    broken,
                    factors,
                    threshold_ranges={"volume": (0.0, 1.0)},
                    n_points_per_dim=2,
                )

    def test_ratio_constraints_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="ratio"):
            self._optimiser(
                {"f": {"a": 1.0}},
                constraints={"lr": {"numerator": "a", "denominator": "b", "max": 1.0}},
            )

    def test_auto_discovery_is_rejected(self):
        df, factors = self._small()
        with pytest.raises(ValueError, match="factor_columns"):
            self._optimiser({"f": {"a": 1.0, "b": 1.0}}, factor_columns=None).solve(
                df, factors
            )


class TestResults:
    def test_schema_variants(self):
        base = list(pc.quote_results_schema(["c"]))
        assert list(pc.quote_results_schema(["c"], existing_rates=True)) == [
            *base,
            "baseline_product",
            "baseline_step",
        ]
        assert list(pc.quote_results_schema(["c"], base_multiplier=True)) == [
            *base,
            "base_multiplier",
            "baseline_step",
        ]
        assert list(
            pc.quote_results_schema(["c"], base_multiplier=True, existing_rates=True)
        ) == [
            *base,
            "base_multiplier",
            "baseline_product",
            "baseline_step",
        ]

    def test_result_records_the_rates_and_summary_reports_them(self):
        df, factors = book()
        existing = existing_for(factors)
        optimiser = strict_optimiser(existing_rates=existing)
        result = optimiser.solve(df, factors)
        assert result.existing_rates == {
            f: {k: f32(v) for k, v in t.items()} for f, t in existing.items()
        }
        assert optimiser.summary(result)["params"]["existing_rates"] is True
        plain = strict_optimiser()
        assert "existing_rates" not in plain.summary(plain.solve(df, factors))["params"]

    @pytest.mark.parametrize("with_column", [False, True], ids=["alone", "with_column"])
    def test_format_four_round_trips(self, tmp_path, with_column):
        df, factors = book()
        existing = existing_for(factors)
        existing["region"]["unseen"] = 1.5
        kwargs = {"existing_rates": existing}
        if with_column:
            factors = factors.with_columns(pl.lit(1.1).alias("base"))
            kwargs["base_multiplier"] = "base"
        result = strict_optimiser(**kwargs).solve(df, factors)
        result.save(tmp_path)
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["format_version"] == 4
        assert config["base_multiplier"] == ("base" if with_column else None)
        assert "baseline_scenario_value" not in config

        loaded = RatebookResult.load(tmp_path)
        assert loaded.existing_rates == result.existing_rates
        assert loaded.base_multiplier == result.base_multiplier
        assert loaded.factor_tables == result.factor_tables
        with pytest.raises(ResultUnavailableError, match="baseline_step"):
            _ = loaded.baseline_scenario_value

    def test_formats_two_and_three_are_unchanged(self, tmp_path):
        df, factors = book()
        plain = strict_optimiser().solve(df, factors)
        plain.save(tmp_path / "two")
        two = json.loads((tmp_path / "two" / "config.json").read_text())
        assert two["format_version"] == 2
        assert set(two) == set(_CONFIG_KEYS_V2)
        assert RatebookResult.load(tmp_path / "two").existing_rates is None

        based = strict_optimiser(base_multiplier="base").solve(
            df, factors.with_columns(pl.lit(1.1).alias("base"))
        )
        based.save(tmp_path / "three")
        three = json.loads((tmp_path / "three" / "config.json").read_text())
        assert three["format_version"] == 3
        assert "existing_rates" not in three
        assert RatebookResult.load(tmp_path / "three").base_multiplier == "base"
