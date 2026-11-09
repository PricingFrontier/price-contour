"""Tests for the ratebook base multiplier (DESIGN_DECISIONS §17)."""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import price_contour as pc
import pytest
from price_contour.ratebook import (
    _CONFIG_KEYS_V2,
    RatebookOptimiser,
    RatebookResult,
    ResultUnavailableError,
)

SPECS = [["region"], ["age"], ["scheme"]]


def scored_frame(sv, objective, *, quote_ids=None, **columns) -> pl.DataFrame:
    """Long-format grid from per-(quote, step) matrices."""
    objective = np.asarray(objective, dtype=np.float32)
    n_quotes, n_steps = objective.shape
    ids = quote_ids or [f"q{i:04d}" for i in range(n_quotes)]
    data: dict[str, list] = {
        "quote_id": [q for q in ids for _ in range(n_steps)],
        "scenario_index": list(range(n_steps)) * n_quotes,
        "scenario_value": [float(s) for s in sv] * n_quotes,
        "objective": objective.ravel().tolist(),
    }
    schema = {
        "quote_id": pl.Utf8,
        "scenario_index": pl.Int32,
        "scenario_value": pl.Float32,
        "objective": pl.Float32,
    }
    for name, values in columns.items():
        data[name] = np.asarray(values, dtype=np.float32).ravel().tolist()
        schema[name] = pl.Float32
    return pl.DataFrame(data, schema=schema)


def step_rule(sv, values) -> np.ndarray:
    """An independent f32 implementation of the nearest-step rule."""
    sv = np.asarray(sv, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    above = np.clip(np.searchsorted(sv, values), 1, len(sv) - 1)
    lower = np.abs(values - sv[above - 1]) <= np.abs(sv[above] - values)
    steps = np.where(lower, above - 1, above)
    steps = np.where(values <= sv[0], 0, steps)
    return np.where(values >= sv[-1], len(sv) - 1, steps)


def factor_frame(n_quotes: int, *, seed: int = 11, **extra) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "quote_id": [f"q{i:04d}" for i in range(n_quotes)],
            "region": rng.choice(["N", "S", "E", "W"], n_quotes).tolist(),
            "age": rng.choice(["a", "b", "c", "d", "e"], n_quotes).tolist(),
            "scheme": rng.choice(["x", "y", "z"], n_quotes).tolist(),
            **extra,
        }
    )


def strict_matrices(sv, n_quotes: int, *, seed: int = 3):
    """A random, strictly concave objective, and a volume falling in the price.

    On a grid much finer than the candidate spacing, no two candidates tie.
    """
    rng = np.random.default_rng(seed)
    sv = np.asarray(sv, dtype=np.float64)[None, :]
    peak = rng.uniform(1.05, 1.5, n_quotes)[:, None]
    weight = rng.uniform(5.0, 15.0, n_quotes)[:, None]
    elasticity = rng.uniform(2.0, 6.0, n_quotes)[:, None]
    objective = 100.0 - weight * (sv - peak) ** 2
    volume = 1.0 / (1.0 + np.exp(elasticity * (sv - 1.0)))
    return objective, volume


FINE_SV = np.linspace(0.4, 1.8, 141, dtype=np.float32)
N_STRICT = 240


def strict_optimiser(**kwargs) -> RatebookOptimiser:
    """Candidates that already contain 1.0, so §17.4 changes nothing here."""
    settings = {
        "objective": "objective",
        "constraints": {"volume": {"min_pct": 0.995}},
        "factor_columns": SPECS,
        "candidate_min": 0.8,
        "candidate_max": 1.2,
        "candidate_steps": 5,
        "max_cd_iterations": 2,
        "max_iter": 40,
    }
    return RatebookOptimiser(**{**settings, **kwargs})


def varying_base_book(n_quotes: int = N_STRICT, seed: int = 5):
    rng = np.random.default_rng(seed)
    objective, volume = strict_matrices(FINE_SV, n_quotes)
    df = scored_frame(FINE_SV, objective, volume=volume)
    base = np.exp(rng.normal(0.0, 0.15, n_quotes))
    return df, factor_frame(n_quotes, base=base)


# Candidates 0.7, 0.8 and 0.9 all price the quote at step 1, the best.
TIED_BELOW_ONE = {
    "sv": [0.5, 0.85, 1.0, 1.1],
    "objective": [[0.0, 10.0, 5.0, 1.0]],
    "candidate_min": 0.7,
    "candidate_max": 1.1,
    "candidate_steps": 5,
}


class TestAgainstNoBase:
    def test_base_of_one_evaluates_bit_for_bit(self):
        objective, volume = strict_matrices(FINE_SV, N_STRICT)
        df = scored_frame(FINE_SV, objective, volume=volume)
        factors = factor_frame(N_STRICT, base=np.ones(N_STRICT))
        plain = strict_optimiser()
        with_base = strict_optimiser(base_multiplier="base")
        tables = {
            "region": {"N": 0.91, "S": 1.03, "E": 1.17, "W": 0.77},
            "age": {"a": 1.0, "b": 1.1, "c": 0.9, "d": 1.3, "e": 0.8},
            "scheme": {"x": 1.07, "y": 0.93, "z": 1.0},
        }
        a = plain.evaluate(df, factors, tables)
        b = with_base.evaluate(df, factors, tables)
        assert a.total_objective == b.total_objective
        assert a.total_constraints == b.total_constraints
        assert a.baseline_objective == b.baseline_objective
        assert a.baseline_constraints == b.baseline_constraints
        assert b.quote_results.select(a.quote_results.columns).equals(a.quote_results)

    def test_base_of_one_solves_bit_for_bit(self):
        objective, volume = strict_matrices(FINE_SV, N_STRICT)
        df = scored_frame(FINE_SV, objective, volume=volume)
        factors = factor_frame(N_STRICT, base=np.ones(N_STRICT))
        a = strict_optimiser().solve(df, factors)
        b = strict_optimiser(base_multiplier="base").solve(df, factors)
        assert a.factor_tables == b.factor_tables
        assert a.total_objective == b.total_objective
        assert a.total_constraints == b.total_constraints
        assert a.baseline_constraints == b.baseline_constraints
        assert a.lambdas == b.lambdas
        assert a.constraint_bounds == b.constraint_bounds

    def test_a_constant_base_equals_a_shifted_grid(self):
        # Scaling by a power of two is exact in f32, so `==` is the right check.
        shifted = np.linspace(0.4, 1.8, 141, dtype=np.float32)
        doubled = shifted * 2.0
        assert np.array_equal(doubled / 2.0, shifted)
        objective, volume = strict_matrices(shifted, N_STRICT)
        plain_df = scored_frame(shifted, objective, volume=volume)
        based_df = scored_frame(doubled, objective, volume=volume)
        factors = factor_frame(N_STRICT, base=np.full(N_STRICT, 2.0))

        plain = strict_optimiser().solve(plain_df, factors)
        based = strict_optimiser(base_multiplier="base").solve(based_df, factors)

        # A binding constraint makes the λ path depend on the scale factors.
        assert based.lambdas["volume"] > 0.0
        assert based.factor_tables == plain.factor_tables
        assert based.lambdas == plain.lambdas
        assert based.total_objective == plain.total_objective
        assert based.total_constraints == plain.total_constraints
        assert based.baseline_constraints == plain.baseline_constraints
        assert based.constraint_bounds == plain.constraint_bounds
        assert (
            based.quote_results["optimal_step"].to_list()
            == plain.quote_results["optimal_step"].to_list()
        )
        np.testing.assert_array_equal(
            based.quote_results["factor_product"].to_numpy(),
            plain.quote_results["factor_product"].to_numpy() * 2.0,
        )


class TestBaseline:
    """Grid [0.5, 1, 2, 4]; cell (q, k): objective 100q + k, vol 10q + k."""

    SV = (0.5, 1.0, 2.0, 4.0)
    BASES = (1.0, 0.75, 2.9, 0.1, 9.0)
    # 1.0 exact; 0.75 midpoint goes low; 2.9 nearer 2; 0.1 and 9.0 past the ends.
    EXPECTED_STEPS = (1, 0, 2, 0, 3)
    BASELINE_OBJECTIVE = 1.0 + 100.0 + 202.0 + 300.0 + 403.0
    BASELINE_VOL = 1.0 + 10.0 + 22.0 + 30.0 + 43.0
    GRID_LEVEL_VOL = 1.0 + 11.0 + 21.0 + 31.0 + 41.0

    def _frames(self):
        n, s = len(self.BASES), len(self.SV)
        objective = [[100 * q + k for k in range(s)] for q in range(n)]
        vol = [[10 * q + k for k in range(s)] for q in range(n)]
        df = scored_frame(self.SV, objective, vol=vol)
        factors = pl.DataFrame(
            {
                "quote_id": [f"q{i:04d}" for i in range(n)],
                "f": ["l"] * n,
                "base": list(self.BASES),
            }
        )
        return df, factors

    def _optimiser(self, constraints) -> RatebookOptimiser:
        return RatebookOptimiser(
            objective="objective",
            constraints=constraints,
            factor_columns=[["f"]],
            candidate_min=0.9,
            candidate_max=1.1,
            candidate_steps=3,
            base_multiplier="base",
        )

    def test_identity_tables_price_every_quote_at_its_baseline(self):
        df, factors = self._frames()
        result = self._optimiser({"vol": {"min": 0.0}}).evaluate(
            df, factors, {"f": {"l": 1.0}}
        )
        frame = result.quote_results
        assert frame["baseline_step"].to_list() == list(self.EXPECTED_STEPS)
        assert frame["optimal_step"].to_list() == list(self.EXPECTED_STEPS)
        assert result.baseline_objective == self.BASELINE_OBJECTIVE
        assert result.baseline_constraints == {"vol": self.BASELINE_VOL}
        assert result.total_objective == self.BASELINE_OBJECTIVE
        assert result.total_constraints == {"vol": self.BASELINE_VOL}
        assert frame["clamped_low"].to_list() == [False, False, False, True, False]
        assert frame["clamped_high"].to_list() == [False, False, False, False, True]

    def test_solve_resolves_pct_bounds_against_the_per_quote_baseline(self):
        df, factors = self._frames()
        result = self._optimiser({"vol": {"min_pct": 0.9}}).solve(df, factors)
        assert result.baseline_constraints == {"vol": self.BASELINE_VOL}
        assert result.constraint_bounds == {"vol": self.BASELINE_VOL * 0.9}
        assert result.constraint_bounds != {"vol": self.GRID_LEVEL_VOL * 0.9}

    def test_frontier_resolves_pct_ranges_against_the_per_quote_baseline(self):
        df, factors = self._frames()
        frontier = self._optimiser({"vol": {"min_pct": None}}).frontier(
            df, factors, threshold_ranges={"vol": (0.8, 0.9)}, n_points_per_dim=2
        )
        assert frontier.points["bound_vol"].to_list() == pytest.approx(
            [self.BASELINE_VOL * 0.8, self.BASELINE_VOL * 0.9]
        )

    def test_baseline_scenario_value_is_unavailable(self):
        df, factors = self._frames()
        optimiser = self._optimiser({"vol": {"min": 0.0}})
        evaluation = optimiser.evaluate(df, factors, {"f": {"l": 1.0}})
        with pytest.raises(RuntimeError, match="baseline_step"):
            _ = evaluation.baseline_scenario_value
        result = optimiser.solve(df, factors)
        with pytest.raises(ResultUnavailableError, match="baseline_step"):
            _ = result.baseline_scenario_value

    def test_a_token_changes_nothing_and_cancels(self):
        df, factors = self._frames()
        optimiser = self._optimiser({"vol": {"min": 0.0}})
        tables = {"f": {"l": 1.0}}
        plain = optimiser.evaluate(df, factors, tables)
        tokened = optimiser.evaluate(df, factors, tables, cancel=pc.CancelToken())
        assert plain.total_objective == tokened.total_objective
        assert plain.baseline_constraints == tokened.baseline_constraints
        cancelled = pc.CancelToken()
        cancelled.cancel()
        with pytest.raises(pc.Cancelled):
            optimiser.evaluate(df, factors, tables, cancel=cancelled)


class TestNoChangePreference:
    def _solve(self, sv, objective, **kwargs):
        df = scored_frame(sv, objective)
        n = len(objective)
        factors = pl.DataFrame(
            {
                "quote_id": [f"q{i:04d}" for i in range(n)],
                "f": ["l"] * n,
                "base": [1.0] * n,
            }
        )
        settings = {
            "objective": "objective",
            "constraints": {},
            "factor_columns": [["f"]],
        }
        return RatebookOptimiser(**{**settings, **kwargs}).solve(df, factors)

    def test_a_flat_level_stays_at_exactly_one(self):
        low, high = math.exp(-0.1), math.exp(0.1)
        linear = [low + (high - low) * i / 20 for i in range(21)]
        assert 1.0 not in np.asarray(linear, dtype=np.float32)
        result = self._solve(
            [0.8, 1.0, 1.2],
            [[5.0, 5.0, 5.0]],
            candidate_min=low,
            candidate_max=high,
            candidate_steps=21,
            base_multiplier="base",
        )
        assert result.factor_tables["f"]["l"] == 1.0

    def test_a_tie_below_one_takes_the_candidate_closest_to_one(self):
        result = self._solve(**TIED_BELOW_ONE, base_multiplier="base")
        assert result.factor_tables["f"]["l"] == pytest.approx(0.9, abs=1e-6)

    def test_without_a_base_a_tie_still_takes_the_lowest_index(self):
        result = self._solve(**TIED_BELOW_ONE)
        assert result.factor_tables["f"]["l"] == pytest.approx(0.7, abs=1e-6)

    def test_a_range_excluding_one_takes_the_closest_end(self):
        result = self._solve(
            [0.8, 1.0, 1.2],
            [[5.0, 5.0, 5.0]],
            candidate_min=1.05,
            candidate_max=1.25,
            candidate_steps=5,
            base_multiplier="base",
        )
        assert result.factor_tables["f"]["l"] == pytest.approx(1.05, abs=1e-6)

    def test_the_band_is_enforced(self):
        df, factors = varying_base_book()
        band = 0.15
        optimiser = strict_optimiser(
            candidate_min=math.exp(-band),
            candidate_max=math.exp(band),
            candidate_steps=31,
            base_multiplier="base",
        )
        result = optimiser.solve(df, factors)
        low = float(np.float32(math.exp(-band)))
        high = float(np.float32(math.exp(band)))
        rates = [r for table in result.factor_tables.values() for r in table.values()]
        assert all(low <= rate <= high for rate in rates)
        assert any(rate != 1.0 for rate in rates)


class TestHeterogeneousBasesByHand:
    """A search that starts every quote from one base, or drops the base after
    the first factor, gets these wrong even though evaluation stays correct."""

    SV = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

    def _run(self, objective, specs):
        df = scored_frame(self.SV, objective, quote_ids=["q0", "q1"])
        columns = {name[0]: ["l", "l"] for name in specs}
        factors = pl.DataFrame(
            {"quote_id": ["q0", "q1"], "base": [1.0, 2.0], **columns}
        )
        return RatebookOptimiser(
            objective="objective",
            constraints={},
            factor_columns=specs,
            candidate_min=0.5,
            candidate_max=1.5,
            candidate_steps=3,
            max_cd_iterations=1,
            base_multiplier="base",
        ).solve(df, factors)

    def test_one_factor_uses_each_quotes_own_base(self):
        # Candidate totals: 0.5 → 0 + 10, 1.0 → 1 + 0, 1.5 → 0 + 0.
        # Starting q1 from q0's base would make 1.0 the best (1 + 10).
        result = self._run(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 8.0, 0.0, 0.0, 0.0]], [["a"]]
        )
        assert result.factor_tables == {"a": {"l": 0.5}}
        assert result.total_objective == 10.0

    def test_a_later_factor_keeps_the_base_in_its_residuals(self):
        # First factor: residuals [1, 2] give totals [10, 0, 0], so 0.5.
        # Second: residuals [0.5, 1.0] give totals [0, 10, 12], so 1.5.
        result = self._run(
            [[0.0] * 6, [0.0, 10.0, 12.0, 0.0, 0.0, 0.0]], [["a"], ["b"]]
        )
        assert result.factor_tables == {"a": {"l": 0.5}, "b": {"l": 1.5}}
        assert result.total_objective == 12.0


class TestFrontier:
    def test_every_point_reproduces_through_evaluate(self):
        df, factors = varying_base_book()
        optimiser = strict_optimiser(
            constraints={"volume": {"min_pct": None}}, base_multiplier="base"
        )
        frontier = optimiser.frontier(
            df, factors, threshold_ranges={"volume": (0.97, 1.0)}, n_points_per_dim=4
        )
        for i in range(frontier.n_points):
            evaluation = optimiser.evaluate(
                df, factors, frontier.point_factor_tables(i)
            )
            assert evaluation.total_objective == frontier.points["total_objective"][i]
            assert (
                evaluation.total_constraints["volume"]
                == frontier.points["total_volume"][i]
            )


class TestEngineEquivalence:
    def test_new_relativities_reprice_each_quote_at_its_step(self):
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
            f: {l: float(rng.normal(-0.4, 0.3)) for l in ls} for f, ls in levels.items()
        }
        membership = {f: rng.choice(ls, n_quotes).tolist() for f, ls in levels.items()}
        sum_w0 = sum(
            np.array([w0[f][l] for l in membership[f]], dtype=np.float64)
            for f in levels
        )
        base_odds = np.exp(sum_w0)

        # Conversion falls with commission's distance from a per-quote peak.
        current = 0.4 * base_odds / (1.0 + base_odds)
        peak = np.clip(current + rng.normal(0.02, 0.03, n_quotes), 0.0, 0.39)[:, None]
        conversion = 1.0 - 8.0 * (commissions[None, :] - peak) ** 2
        profit = (0.5 - commissions[None, :]) * conversion

        df = scored_frame(
            sv,
            conversion,
            quote_ids=[f"q{i:05d}" for i in range(n_quotes)],
            profit=profit,
        )
        factors = pl.DataFrame(
            {
                "quote_id": [f"q{i:05d}" for i in range(n_quotes)],
                "base_odds": base_odds,
                **membership,
            }
        )
        result = RatebookOptimiser(
            objective="objective",
            constraints={"profit": {"min_pct": 0.98}},
            factor_columns=[[f] for f in levels],
            candidate_min=math.exp(-0.2),
            candidate_max=math.exp(0.2),
            candidate_steps=41,
            base_multiplier="base_odds",
        ).solve(df, factors)

        w_new_sum = sum(
            np.array(
                [
                    w0[f][l] + math.log(result.factor_tables[f][l])
                    for l in membership[f]
                ],
                dtype=np.float64,
            )
            for f in levels
        )
        odds = np.exp(w_new_sum)
        frame = result.quote_results
        assert any(
            rate != 1.0
            for table in result.factor_tables.values()
            for rate in table.values()
        )

        product = frame["factor_product"].to_numpy().astype(np.float64)
        np.testing.assert_allclose(odds, product, rtol=1e-5)

        boundaries = (odds_grid[:-1] + odds_grid[1:]) / 2.0
        relative_gap = np.min(
            np.abs(odds[:, None] - boundaries[None, :]) / boundaries[None, :], axis=1
        )
        clear = relative_gap > 1e-5
        assert clear.mean() > 0.99

        # The kernel rounds in odds, so compare against the odds grid, not
        # against the commission nearest the engine's continuous commission.
        below = np.clip(np.searchsorted(odds_grid, odds), 1, len(odds_grid) - 1)
        lower = np.abs(odds - odds_grid[below - 1]) <= np.abs(odds_grid[below] - odds)
        expected = np.where(lower, below - 1, below)
        steps = frame["optimal_step"].to_numpy()
        np.testing.assert_array_equal(steps[clear], expected[clear])


class TestQuoteResults:
    def test_schema_with_a_base(self):
        df, factors = varying_base_book()
        result = strict_optimiser(base_multiplier="base").solve(df, factors)
        expected = pc.quote_results_schema(["volume"], base_multiplier=True)
        assert dict(result.quote_results.schema) == expected
        assert list(expected)[-2:] == ["base_multiplier", "baseline_step"]

    def test_schema_without_a_base_is_unchanged(self):
        assert list(pc.quote_results_schema(["volume"])) == [
            "quote_id",
            "optimal_step",
            "optimal_scenario_value",
            "optimal_objective",
            "optimal_volume",
            "factor_product",
            "clamped_low",
            "clamped_high",
        ]

    def test_factor_product_is_the_full_product(self):
        df, factors = varying_base_book()
        result = strict_optimiser(base_multiplier="base").solve(df, factors)
        frame = result.quote_results
        by_id = factors.sort("quote_id")
        product = by_id["base"].to_numpy().astype(np.float32)
        for spec in SPECS:
            name = spec[0]
            table = result.factor_tables[name]
            product = product * np.array(
                [table[level] for level in by_id[name].to_list()], dtype=np.float32
            )
        np.testing.assert_array_equal(frame["factor_product"].to_numpy(), product)
        np.testing.assert_array_equal(
            frame["base_multiplier"].to_numpy(),
            by_id["base"].to_numpy().astype(np.float32),
        )
        np.testing.assert_array_equal(
            frame["optimal_step"].to_numpy(), step_rule(FINE_SV, product)
        )
        np.testing.assert_array_equal(
            frame["baseline_step"].to_numpy(),
            step_rule(FINE_SV, frame["base_multiplier"]),
        )

    def test_summary_reports_the_column(self):
        df, factors = varying_base_book()
        optimiser = strict_optimiser(base_multiplier="base")
        result = optimiser.solve(df, factors)
        assert optimiser.summary(result)["params"]["base_multiplier"] == "base"
        plain = strict_optimiser()
        assert (
            "base_multiplier" not in plain.summary(plain.solve(df, factors))["params"]
        )


class TestAlignmentAndContexts:
    def _assert_same(self, a: RatebookResult, b: RatebookResult) -> None:
        assert a.factor_tables == b.factor_tables
        assert a.total_objective == b.total_objective
        assert a.total_constraints == b.total_constraints
        assert a.baseline_constraints == b.baseline_constraints
        assert a.quote_results.equals(b.quote_results)

    def test_factor_rows_may_be_in_any_order(self):
        df, factors = varying_base_book()
        optimiser = strict_optimiser(base_multiplier="base")
        shuffled = factors.sample(fraction=1.0, shuffle=True, seed=1)
        assert shuffled["quote_id"].to_list() != factors["quote_id"].to_list()
        self._assert_same(optimiser.solve(df, factors), optimiser.solve(df, shuffled))

    def test_prebuilt_contexts_give_the_same_results(self, tmp_path):
        df, factors = varying_base_book()
        optimiser = strict_optimiser(base_multiplier="base")
        builder = pc.QuoteGridBuilder(["volume"], objective="objective")
        builder.append(df)
        grid = builder.build()
        direct = optimiser.solve(df, factors)

        contexts = pc.RatebookFactorContexts.from_dataframe(
            factors, SPECS, expected_quote_ids=grid.quote_ids, base_multiplier="base"
        )
        assert contexts.base_multiplier == "base"
        self._assert_same(direct, optimiser.solve(grid, contexts))

        path = tmp_path / "factors.parquet"
        factors.sample(fraction=1.0, shuffle=True, seed=2).write_parquet(path)
        streamed = pc.build_ratebook_factor_contexts_from_parquet_chunked(
            str(path),
            SPECS,
            37,
            expected_quote_ids=grid.quote_ids,
            base_multiplier="base",
        )
        self._assert_same(direct, optimiser.solve(grid, streamed))

    @pytest.mark.parametrize(
        ("optimiser_base", "contexts_base"),
        [("base", None), (None, "base"), ("base", "other")],
    )
    def test_a_contexts_mismatch_raises(self, optimiser_base, contexts_base):
        df, factors = varying_base_book()
        factors = factors.with_columns(pl.col("base").alias("other"))
        builder = pc.QuoteGridBuilder(["volume"], objective="objective")
        builder.append(df)
        grid = builder.build()
        contexts = pc.RatebookFactorContexts.from_dataframe(
            factors,
            SPECS,
            expected_quote_ids=grid.quote_ids,
            base_multiplier=contexts_base,
        )
        optimiser = strict_optimiser(base_multiplier=optimiser_base)
        tables = {
            spec[0]: {level: 1.0 for level in factors[spec[0]].unique().to_list()}
            for spec in SPECS
        }
        with pytest.raises(ValueError, match="base_multiplier"):
            optimiser.solve(grid, contexts)
        with pytest.raises(ValueError, match="base_multiplier"):
            optimiser.evaluate(grid, contexts, tables)
        with pytest.raises(ValueError, match="base_multiplier"):
            optimiser.frontier(
                grid,
                contexts,
                threshold_ranges={"volume": (0.9, 1.0)},
                n_points_per_dim=2,
            )


class TestPersistence:
    def test_a_base_saves_as_format_three(self, tmp_path):
        df, factors = varying_base_book()
        result = strict_optimiser(base_multiplier="base").solve(df, factors)
        assert result.base_multiplier == "base"
        with pytest.raises(ResultUnavailableError, match="baseline_step"):
            _ = result.baseline_scenario_value

        result.save(tmp_path)
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["format_version"] == 3
        assert config["base_multiplier"] == "base"
        assert "baseline_scenario_value" not in config

        loaded = RatebookResult.load(tmp_path)
        assert loaded.base_multiplier == "base"
        assert loaded.factor_tables == result.factor_tables
        assert loaded.baseline_constraints == result.baseline_constraints
        with pytest.raises(ResultUnavailableError, match="baseline_step"):
            _ = loaded.baseline_scenario_value

    def test_no_base_saves_as_format_two(self, tmp_path):
        df, factors = varying_base_book()
        result = strict_optimiser().solve(df, factors)
        assert result.base_multiplier is None
        result.save(tmp_path)
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["format_version"] == 2
        assert set(config) == set(_CONFIG_KEYS_V2)

        loaded = RatebookResult.load(tmp_path)
        assert loaded.base_multiplier is None
        assert loaded.baseline_scenario_value == result.baseline_scenario_value


class TestValidation:
    def _small(self, base):
        n = len(base)
        df = scored_frame(
            [0.5, 1.0, 1.5], [[1.0, 2.0, 3.0]] * n, volume=[[1.0, 1.0, 1.0]] * n
        )
        factors = pl.DataFrame(
            {"quote_id": [f"q{i:04d}" for i in range(n)], "f": ["l"] * n, "base": base},
            schema_overrides={"base": pl.Float64},
        )
        return df, factors

    def _optimiser(self, **kwargs) -> RatebookOptimiser:
        settings = {
            "objective": "objective",
            "constraints": {"volume": {"min": 0.0}},
            "factor_columns": [["f"]],
            "base_multiplier": "base",
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
    def test_invalid_values_report_the_count_and_a_quote(self, bad):
        df, factors = self._small([1.0, bad, 1.0, bad])
        with pytest.raises(ValueError, match=r"'base'.*2 invalid.*q0001"):
            self._optimiser().solve(df, factors)

    def test_without_a_quote_id_column_the_error_gives_a_row(self):
        df, factors = self._small([1.0, 0.0])
        with pytest.raises(ValueError, match=r"1 invalid.*row 1"):
            self._optimiser().solve(df, factors.drop("quote_id"))

    @pytest.mark.parametrize("call", ["solve", "evaluate", "frontier"])
    def test_validation_runs_before_any_grid_build(self, call):
        df, factors = self._small([1.0, -1.0])
        # Without its objective column this frame could never build a grid.
        broken = df.drop("objective")
        optimiser = self._optimiser()
        with pytest.raises(ValueError, match="base_multiplier column 'base'"):
            if call == "solve":
                optimiser.solve(broken, factors)
            elif call == "evaluate":
                optimiser.evaluate(broken, factors, {"f": {"l": 1.0}})
            else:
                optimiser.frontier(
                    broken,
                    factors,
                    threshold_ranges={"volume": (0.0, 1.0)},
                    n_points_per_dim=2,
                )

    def test_a_missing_column_raises(self):
        df, factors = self._small([1.0, 1.0])
        with pytest.raises(ValueError, match="'nope' not found"):
            self._optimiser(base_multiplier="nope").solve(df, factors)

    @pytest.mark.parametrize("cast", [pl.Int64, pl.String], ids=["integer", "string"])
    def test_a_non_float_column_raises_naming_the_dtype(self, cast):
        df, factors = self._small([1.0, 2.0])
        factors = factors.with_columns(pl.col("base").cast(cast))
        with pytest.raises(ValueError, match="float dtype"):
            self._optimiser().solve(df, factors)

    def test_float32_is_accepted(self):
        df, factors = self._small([1.0, 2.0])
        factors = factors.with_columns(pl.col("base").cast(pl.Float32))
        result = self._optimiser().solve(df, factors)
        assert result.quote_results["base_multiplier"].to_list() == [1.0, 2.0]

    def test_the_constructor_rejects_a_non_string(self):
        with pytest.raises(TypeError, match="base_multiplier must be"):
            self._optimiser(base_multiplier=1)

    def test_the_quote_id_column_cannot_be_the_base(self):
        with pytest.raises(ValueError, match="quote_id"):
            self._optimiser(base_multiplier="quote_id")

    def test_a_factor_column_cannot_be_the_base(self):
        with pytest.raises(ValueError, match="factor"):
            self._optimiser(base_multiplier="f")

    def test_a_factor_given_at_call_time_cannot_be_the_base(self):
        df, factors = self._small([1.0, 2.0])
        optimiser = self._optimiser(factor_columns=None)
        with pytest.raises(ValueError, match="factor"):
            optimiser.solve(df, factors, factor_columns=[["base"]])

    def test_ratio_constraints_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="ratio"):
            self._optimiser(
                constraints={"lr": {"numerator": "a", "denominator": "b", "max": 1.0}}
            )

    def test_auto_discovery_is_rejected(self):
        df, factors = self._small([1.0, 2.0])
        with pytest.raises(ValueError, match="factor_columns"):
            self._optimiser(factor_columns=None).solve(df, factors)

    def test_contexts_builders_validate_the_values(self, tmp_path):
        _, factors = self._small([1.0, -1.0, 1.0, 0.0])
        with pytest.raises(ValueError, match=r"'base'.*2 invalid.*q0001"):
            pc.RatebookFactorContexts.from_dataframe(
                factors, [["f"]], base_multiplier="base"
            )
        # The count covers the whole file, across chunks.
        path = tmp_path / "factors.parquet"
        factors.write_parquet(path)
        with pytest.raises(ValueError, match=r"'base'.*2 invalid.*q0001"):
            pc.build_ratebook_factor_contexts_from_parquet_chunked(
                str(path), [["f"]], 1, base_multiplier="base"
            )

    def test_contexts_builders_reject_a_base_that_is_a_factor_or_the_id(self):
        _, factors = self._small([1.0, 1.0])
        with pytest.raises(ValueError, match="factor"):
            pc.RatebookFactorContexts.from_dataframe(
                factors, [["base"]], base_multiplier="base"
            )
        with pytest.raises(ValueError, match="quote_id"):
            pc.RatebookFactorContexts.from_dataframe(
                factors, [["f"]], base_multiplier="quote_id"
            )
