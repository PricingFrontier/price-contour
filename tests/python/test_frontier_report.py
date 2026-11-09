"""Tests for the frontier impact report (DESIGN_DECISIONS §19)."""

from __future__ import annotations

import json
import math
import re
import time
from html.parser import HTMLParser

import numpy as np
import polars as pl
import pytest

import price_contour as pc
from price_contour._frontier_report import (
    _default_point,
    _evaluation_grid,
    _mark_dominance,
    _ratios,
    _review_flags,
    _slices,
)
from price_contour._frontier_report_html import (
    _axis_labels,
    _beeswarm,
    _beeswarm_caption,
    _beeswarm_domain,
    _beeswarm_rows,
    _colour_positions,
    _disp_signed,
    _distribution_caption,
    _distribution_chart,
    _ecdf_chart,
    _rate_row_limit,
)
from price_contour.ratebook import RatebookOptimiser
from test_ratebook_base_multiplier import scored_frame, step_rule

SV = np.linspace(0.6, 1.6, 21, dtype=np.float32)
N = 400
SPECS = [["region"], ["age"]]


def book(n: int = N, seed: int = 2):
    rng = np.random.default_rng(seed)
    sv = SV.astype(np.float64)[None, :]
    peak = rng.uniform(1.0, 1.4, n)[:, None]
    weight = rng.uniform(5.0, 15.0, n)[:, None]
    elasticity = rng.uniform(2.0, 6.0, n)[:, None]
    objective = 100.0 - weight * (sv - peak) ** 2
    volume = 1.0 / (1.0 + np.exp(elasticity * (sv - 1.0)))
    margin = objective * 0.2 + rng.uniform(0.0, 1.0, n)[:, None]
    df = scored_frame(SV, objective, volume=volume, margin=margin)
    factors = pl.DataFrame(
        {
            "quote_id": [f"q{i:04d}" for i in range(n)],
            "region": rng.choice(["N", "S", "E", "W"], n).tolist(),
            "age": rng.choice(["a", "b", "c", "d", "e"], n).tolist(),
            "channel": rng.choice(["web", "phone"], n).tolist(),
        }
    )
    return df, factors


def rates_for(factors: pl.DataFrame, seed: int = 8) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    return {
        spec[0]: {
            level: float(rng.uniform(0.85, 1.2))
            for level in factors[spec[0]].unique(maintain_order=True).to_list()
        }
        for spec in SPECS
    }


def make_optimiser(**kwargs) -> RatebookOptimiser:
    settings = {
        "objective": "objective",
        "constraints": {"volume": {"min_pct": None}},
        "factor_columns": SPECS,
        "candidate_min": 0.8,
        "candidate_max": 1.25,
        "candidate_steps": 19,
        "max_cd_iterations": 2,
        "max_iter": 30,
    }
    return RatebookOptimiser(**{**settings, **kwargs})


RANGES = {"volume": (0.97, 1.0)}


def sweep(optimiser, df, factors, **kwargs):
    return optimiser.frontier(
        df, factors, threshold_ranges=RANGES, n_points_per_dim=4, **kwargs
    )


def make_report(df, factors, *, existing=True, **kwargs):
    options = {"existing_rates": rates_for(factors)} if existing else {}
    optimiser = make_optimiser(**options)
    frontier = sweep(optimiser, df, factors)
    report = pc.FrontierReport(
        optimiser,
        frontier,
        df,
        factors,
        generated_at="2026-11-09T12:00:00+00:00",
        **kwargs,
    )
    return optimiser, frontier, report


@pytest.fixture(scope="module")
def built():
    df, factors = book()
    optimiser, frontier, report = make_report(
        df, factors, report_columns=["margin"], segment_columns=["channel"]
    )
    return df, factors, optimiser, frontier, report


def steps_of(optimiser, df, factors, tables) -> tuple[np.ndarray, np.ndarray]:
    frame = optimiser.evaluate(df, factors, tables).quote_results
    if "baseline_step" in frame.columns:
        return frame["optimal_step"].to_numpy(), frame["baseline_step"].to_numpy()
    return frame["optimal_step"].to_numpy(), np.full(frame.height, step_rule(SV, 1.0))


def cells(df: pl.DataFrame, column: str, n_steps: int = len(SV)) -> np.ndarray:
    return df[column].to_numpy().astype(np.float64).reshape(-1, n_steps)


def nearest_rank(values: np.ndarray, p: int) -> float:
    ordered = np.sort(values)
    return float(ordered[math.ceil(p * len(ordered) / 100) - 1])


class TestTotals:
    def test_totals_equal_the_frontier_exactly(self, built):
        *_, frontier, report = built
        for i, point in enumerate(report.data["points"]):
            assert (
                point["totals"]["objective"]["point"]
                == frontier.points["total_objective"][i]
            )
            assert (
                point["totals"]["volume"]["point"] == frontier.points["total_volume"][i]
            )

    def test_report_columns_match_a_hand_join(self, built):
        df, factors, optimiser, frontier, report = built
        margin = cells(df, "margin")
        for i, point in enumerate(report.data["points"]):
            steps, start = steps_of(
                optimiser, df, factors, frontier.point_factor_tables(i)
            )
            row = np.arange(len(steps))
            assert point["totals"]["margin"]["point"] == pytest.approx(
                math.fsum(margin[row, steps]), rel=1e-12
            )
            assert point["totals"]["margin"]["baseline"] == pytest.approx(
                math.fsum(margin[row, start]), rel=1e-12
            )

    def test_baseline_is_todays_card_by_hand(self, built):
        df, factors, optimiser, _, report = built
        by_id = factors.sort("quote_id")
        product = np.ones(N, dtype=np.float32)
        for spec in SPECS:
            table = optimiser.existing_rates[spec[0]]
            product = product * np.array(
                [table[v] for v in by_id[spec[0]].to_list()], dtype=np.float32
            )
        start = step_rule(SV, product)
        objective = cells(df, "objective")
        assert report.data["baseline"]["totals"]["objective"] == pytest.approx(
            math.fsum(objective[np.arange(N), start]), rel=1e-12
        )

    def test_baseline_without_a_starting_point_is_no_adjustment(self):
        df, factors = book()
        _, _, report = make_report(df, factors, existing=False)
        objective = cells(df, "objective")
        step = int(step_rule(SV, 1.0))
        assert report.data["baseline"]["totals"]["objective"] == pytest.approx(
            math.fsum(objective[:, step]), rel=1e-12
        )
        assert report.data["run"]["has_existing_rates"] is False
        assert "Rate changes" not in report.to_html()
        assert "The full set of new rates" in report.to_html()

    def test_a_per_quote_start_shows_changes_only(self):
        df, factors = book()
        rng = np.random.default_rng(3)
        based = factors.with_columns(pl.Series("base", np.exp(rng.normal(0.0, 0.1, N))))
        optimiser = make_optimiser(base_multiplier="base")
        frontier = sweep(optimiser, df, based)
        report = pc.FrontierReport(optimiser, frontier, df, based)
        level = report.data["points"][0]["rate_card"][0]["levels"][0]
        assert level["existing"] is None
        assert "per quote" in level["existing_reason"]
        html = report.to_html()
        assert "Rate changes" in html
        assert "The full set of new rates" not in html

    def test_data_that_isnt_the_sweeps_raises_naming_the_point(self):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        other = df.with_columns(pl.col("objective") + 1.0)
        with pytest.raises(ValueError, match="point 0.*not the data"):
            pc.FrontierReport(optimiser, frontier, other, factors)


class TestMovementAndImpacts:
    def test_movement_matches_an_independent_computation(self, built):
        df, factors, optimiser, frontier, report = built
        for i, point in enumerate(report.data["points"]):
            steps, start = steps_of(
                optimiser, df, factors, frontier.point_factor_tables(i)
            )
            move = steps.astype(int) - start.astype(int)
            movement = point["movement"]
            assert movement["up"] == int((move > 0).sum())
            assert movement["down"] == int((move < 0).sum())
            assert movement["unchanged"] == int((move == 0).sum())
            changes, counts = np.unique(move, return_counts=True)
            assert movement["histogram"]["changes"] == changes.tolist()
            assert movement["histogram"]["counts"] == counts.tolist()
            for p in (1, 5, 25, 50, 75, 95, 99):
                assert movement["percentiles_steps"][f"p{p}"] == nearest_rank(move, p)

    def test_display_percentiles_are_taken_per_quote(self):
        df, factors = book()
        display = pc.ScenarioDisplay("Price", [float(v) ** 2 for v in SV], "{:.2f}")
        optimiser, frontier, report = make_report(df, factors, scenario_display=display)
        values = np.array(display.values)
        for i, point in enumerate(report.data["points"]):
            steps, start = steps_of(
                optimiser, df, factors, frontier.point_factor_tables(i)
            )
            change = values[steps] - values[start]
            assert point["movement"]["percentiles_display"]["p50"] == nearest_rank(
                change, 50
            )

    def test_delta_summaries_match_an_independent_computation(self, built):
        df, factors, optimiser, frontier, report = built
        objective = cells(df, "objective").astype(np.float32).astype(np.float64)
        point = report.data["points"][1]
        steps, start = steps_of(optimiser, df, factors, frontier.point_factor_tables(1))
        row = np.arange(len(steps))
        change = objective[row, steps] - objective[row, start]
        delta = point["deltas"]["objective"]
        for p in (1, 5, 25, 50, 75, 95, 99):
            assert delta["percentiles"][f"p{p}"] == nearest_rank(change, p)
        assert delta["moved_up"] == pytest.approx(
            float(change[steps > start].sum()), rel=1e-12
        )
        assert delta["moved_down"] == pytest.approx(
            float(change[steps < start].sum()), abs=1e-6
        )
        histogram = delta["histogram"]
        total = sum(histogram["counts"]) + histogram["below"] + histogram["above"]
        assert total == N
        assert histogram["below"] == int((change < delta["percentiles"]["p1"]).sum())

    def test_group_impacts_match_an_independent_computation(self, built):
        df, factors, optimiser, frontier, report = built
        objective = cells(df, "objective").astype(np.float32).astype(np.float64)
        steps, start = steps_of(optimiser, df, factors, frontier.point_factor_tables(2))
        row = np.arange(len(steps))
        change = objective[row, steps] - objective[row, start]
        by_id = factors.sort("quote_id")
        card = report.data["points"][2]["rate_card"][0]
        assert card["factor"] == "region"
        for level in card["levels"]:
            mask = (by_id["region"] == level["label"]).to_numpy()
            assert level["quotes"] == int(mask.sum())
            assert level["impacts"]["objective"] == pytest.approx(
                float(change[mask].sum()), abs=1e-6
            )
            assert level["mean_step_change"] == pytest.approx(
                float((steps[mask].astype(int) - start[mask].astype(int)).mean())
            )

    def test_segments_are_ordered_and_add_up(self, built):
        *_, report = built
        point = report.data["points"][0]
        levels = point["segments"][0]["levels"]
        assert [lv["quotes"] for lv in levels] == sorted(
            (lv["quotes"] for lv in levels), reverse=True
        )
        assert sum(lv["quotes"] for lv in levels) == N
        assert sum(lv["point"]["objective"] for lv in levels) == pytest.approx(
            point["totals"]["objective"]["point"]
        )


BAND_SEED = 31
DISPLAY = [float(v) * 0.2 for v in SV]


@pytest.fixture(scope="module")
def rich():
    df, factors = book()
    premium = np.random.default_rng(BAND_SEED).uniform(100.0, 900.0, N)
    factors = factors.with_columns(pl.Series("premium", premium))
    low = df.with_columns(
        pl.col("objective") * 0.9, pl.col("volume") * 0.95, pl.col("margin") * 0.8
    )
    optimiser, frontier, report = make_report(
        df,
        factors,
        report_columns=["margin"],
        segment_columns=["channel", ["channel", "region"]],
        band_columns=["premium"],
        sensitivity_grids={"Lower": low},
        sensitivity_note="Scored with the lower model.",
        ratios={"Margin per objective": ("margin", "objective")},
        movement_band=0.02,
        projection=pc.Projection(window_days=29, horizons=[30, 365]),
        scenario_display=pc.ScenarioDisplay("Rate", DISPLAY, "{:.1%}", target=0.2),
    )
    return df, factors, optimiser, frontier, report, low


def quote_arrays(rich_parts, index: int = 0):
    df, factors, optimiser, frontier, report, _ = rich_parts
    tables = frontier.point_factor_tables(index)
    steps, base = steps_of(optimiser, df, factors, tables)
    return steps, base, report.data["points"][index], factors


class TestExampleContent:
    def test_directions_match_an_independent_computation(self, rich):
        steps, base, point, _ = quote_arrays(rich)
        df = rich[0]
        display = np.array(DISPLAY)
        objective = cells(df, "objective")
        volume = cells(df, "volume")
        rows = np.arange(N)
        change = objective[rows, steps] - objective[rows, base]
        volume_change = volume[rows, steps] - volume[rows, base]
        for key, mask in (
            ("up", steps > base),
            ("down", steps < base),
            ("unchanged", steps == base),
        ):
            entry = point["movement"]["directions"][key]
            assert entry["quotes"] == int(mask.sum())
            if mask.any():
                assert entry["mean_display_change"] == pytest.approx(
                    float((display[steps[mask]] - display[base[mask]]).mean())
                )
                assert entry["mean_changes"]["objective"] == pytest.approx(
                    float(change[mask].mean())
                )
                assert entry["mean_changes"]["volume"] == pytest.approx(
                    float(volume_change[mask].mean())
                )

    def test_scenario_summary_matches_an_independent_computation(self, rich):
        steps, base, point, _ = quote_arrays(rich)
        display = np.array(DISPLAY)
        scenario = point["scenario"]
        assert scenario["baseline_mean"] == pytest.approx(float(display[base].mean()))
        assert scenario["point_mean"] == pytest.approx(float(display[steps].mean()))
        assert scenario["baseline_max"] == pytest.approx(float(display[base].max()))
        assert scenario["point_max"] == pytest.approx(float(display[steps].max()))
        assert scenario["baseline_counts"] == np.bincount(base, minlength=21).tolist()
        assert scenario["point_counts"] == np.bincount(steps, minlength=21).tolist()

    def test_transitions_match_an_independent_computation(self, rich):
        steps, base, point, _ = quote_arrays(rich)
        df = rich[0]
        volume = cells(df, "volume")
        rows = np.arange(N)
        got = point["scenario"]["transitions"]
        assert sum(t["quotes"] for t in got) == N
        assert [(t["from"], t["to"]) for t in got] == sorted(
            {(int(a), int(b)) for a, b in zip(base, steps)}
        )
        for transition in got:
            mask = (base == transition["from"]) & (steps == transition["to"])
            assert transition["quotes"] == int(mask.sum())
            assert transition["changes"]["volume"] == pytest.approx(
                float(
                    (
                        volume[rows[mask], steps[mask]] - volume[rows[mask], base[mask]]
                    ).sum()
                )
            )

    def test_segment_display_statistics_match(self, rich):
        steps, base, point, factors = quote_arrays(rich)
        display = np.array(DISPLAY)
        channel = factors["channel"].to_numpy()
        [by_channel, _] = point["segments"]
        assert by_channel["column"] == "channel"
        for level in by_channel["levels"]:
            mask = channel == level["label"]
            moved = np.abs(display[steps[mask]] - display[base[mask]])
            assert level["baseline_display_mean"] == pytest.approx(
                float(display[base[mask]].mean())
            )
            assert level["point_display_mean"] == pytest.approx(
                float(display[steps[mask]].mean())
            )
            assert level["mean_abs_display_change"] == pytest.approx(
                float(moved.mean())
            )
            assert level["max_abs_display_change"] == pytest.approx(float(moved.max()))
            assert level["share_beyond_band"] == pytest.approx(
                float((moved > 0.02).mean())
            )
            assert level["share_at_first"] == pytest.approx(
                float((steps[mask] == 0).mean())
            )
            assert level["share_at_last"] == pytest.approx(
                float((steps[mask] == 20).mean())
            )

    def test_an_interaction_segment_reads_as_both_columns(self, rich):
        _, _, point, factors = quote_arrays(rich)
        [_, interaction] = point["segments"]
        assert interaction["column"] == "channel:region"
        pairs = factors.group_by(["channel", "region"]).len()
        expected = {f"{a} × {b}": n for a, b, n in pairs.iter_rows()}
        assert {lv["label"]: lv["quotes"] for lv in interaction["levels"]} == expected

    def test_bands_are_ten_equal_groups_in_order(self, rich):
        steps, base, point, factors = quote_arrays(rich)
        df = rich[0]
        premium = factors["premium"].to_numpy()
        ids = factors["quote_id"].to_numpy()
        order = np.lexsort((ids, premium))
        band_of = np.empty(N, dtype=int)
        band_of[order] = np.arange(N) * 10 // N
        objective = cells(df, "objective")
        rows = np.arange(N)
        [band] = point["bands"]
        assert band["column"] == "premium"
        assert [lv["label"] for lv in band["levels"]] == [f"D{k}" for k in range(1, 11)]
        assert [lv["quotes"] for lv in band["levels"]] == [40] * 10
        for k, level in enumerate(band["levels"]):
            mask = band_of == k
            assert level["mean_value"] == pytest.approx(float(premium[mask].mean()))
            assert level["point"]["objective"] == pytest.approx(
                float(objective[rows[mask], steps[mask]].sum())
            )

    def test_band_totals_add_up(self, rich):
        _, _, point, _ = quote_arrays(rich)
        [band] = point["bands"]
        assert sum(lv["point"]["objective"] for lv in band["levels"]) == pytest.approx(
            point["totals"]["objective"]["point"]
        )

    def test_ties_in_a_band_column_break_by_quote_id(self):
        df, factors = book()
        factors = factors.with_columns(pl.lit(1.0).alias("flat"))
        optimiser, frontier, report = make_report(df, factors, band_columns=["flat"])
        [band] = report.data["points"][0]["bands"]
        assert [lv["quotes"] for lv in band["levels"]] == [40] * 10
        assert all(lv["mean_value"] == 1.0 for lv in band["levels"])
        again = pc.FrontierReport(
            optimiser,
            frontier,
            df,
            factors.sample(fraction=1.0, shuffle=True, seed=3),
            band_columns=["flat"],
        )
        assert again.data["points"][0]["bands"] == report.data["points"][0]["bands"]

    def test_bands_work_with_fewer_quotes_than_bands(self):
        df, factors = book(n=7)
        factors = factors.with_columns(
            pl.Series("premium", np.arange(7, dtype=np.float64))
        )
        _, _, report = make_report(df, factors, band_columns=["premium"])
        [band] = report.data["points"][0]["bands"]
        assert sum(lv["quotes"] for lv in band["levels"]) == 7
        assert [int(lv["label"][1:]) for lv in band["levels"]] == sorted(
            int(lv["label"][1:]) for lv in band["levels"]
        )
        assert "premium" in report.to_html()

    def test_sensitivity_holds_the_points_steps_fixed(self, rich):
        steps, _, point, _ = quote_arrays(rich)
        low = rich[5]
        rows = np.arange(N)
        for column in ("objective", "volume", "margin"):
            expected = float(cells(low, column)[rows, steps].sum())
            assert point["sensitivity"]["Lower"][column] == pytest.approx(expected)

    def test_ratios_by_hand(self, rich):
        _, _, point, _ = quote_arrays(rich)
        [ratio] = point["ratios"]
        totals = point["totals"]
        assert (
            ratio["point"] == totals["margin"]["point"] / totals["objective"]["point"]
        )
        assert ratio["baseline"] == (
            totals["margin"]["baseline"] / totals["objective"]["baseline"]
        )
        assert ratio["change"] == ratio["point"] - ratio["baseline"]
        assert ratio["sensitivity"]["Lower"] == (
            point["sensitivity"]["Lower"]["margin"]
            / point["sensitivity"]["Lower"]["objective"]
        )

    def test_a_zero_denominator_has_no_ratio_and_says_why(self):
        totals = {
            "a": {"baseline": 1.0, "point": 2.0},
            "b": {"baseline": 0.0, "point": 4.0},
        }
        [ratio] = _ratios({"r": ("a", "b")}, {}, totals, {"S": {"a": 1.0, "b": 0.0}})
        assert (ratio["baseline"], ratio["baseline_reason"]) == (None, "b is zero")
        assert ratio["point"] == 0.5
        assert ratio["change"] is None
        assert ratio["sensitivity"] == {"S": None}

    def test_projection_scales_each_total_by_the_window(self, rich):
        _, _, point, _ = quote_arrays(rich)
        entry = point["totals"]["objective"]
        [month, year] = point["projection"]
        assert (month["horizon"], year["horizon"]) == (30.0, 365.0)
        assert year["totals"]["objective"]["point"] == entry["point"] * 365.0 / 29.0
        assert year["totals"]["objective"]["baseline"] == (
            entry["baseline"] * 365.0 / 29.0
        )

    def test_a_step_reads_as_its_scenario_value_without_a_display(self, built):
        df, factors, optimiser, frontier, report = built
        tables = frontier.point_factor_tables(0)
        steps, base = steps_of(optimiser, df, factors, tables)
        scenario = report.data["points"][0]["scenario"]
        assert scenario["point_mean"] == pytest.approx(
            float(SV[steps].astype(float).mean())
        )
        assert scenario["baseline_mean"] == pytest.approx(
            float(SV[base].astype(float).mean())
        )
        assert report.data["points"][0]["movement"]["percentiles_display"] is None

    def test_no_band_means_no_share(self, built):
        *_, report = built
        level = report.data["points"][0]["segments"][0]["levels"][0]
        assert level["share_beyond_band"] is None

    def test_the_tabs_show_it_all(self, rich):
        *_, report, _ = rich
        html = report.to_html()
        impact = section(html, "impact")
        for text in (
            "Scored with the lower model.",
            "Movement by direction",
            "Distribution of Rate",
            "Cumulative distribution of Rate",
            "Rate: baseline against point",
            "Forward projection",
            "Margin per objective",
            ">Lower<",
            "Mean Rate",
            "Largest Rate",
            "365 days",
        ):
            assert text in impact, text
        segments = section(html, "segments")
        for text in ("By channel:region", "By premium band", "Beyond 2.0%", "D10"):
            assert text in segments, text
        assert "Coordinate-descent passes, at most" in section(html, "checks")
        assert "×" in segments

    def test_transition_dots_are_capped_and_it_says_so(self):
        record = synthetic_report(1, 2, 3)
        ids = [c["id"] for c in record["columns"]]
        record["run"]["n_steps"] = 40
        record["points"][0]["scenario"]["transitions"] = [
            {
                "from": a,
                "to": b,
                "quotes": a + b + 1,
                "changes": dict.fromkeys(ids, 1.0),
            }
            for a in range(40)
            for b in range(40)
        ]
        impact = section(pc.frontier_report_html(record), "impact")
        assert impact.count('class="tr ') == 300
        assert "smaller pairs aren" in impact

    def test_axis_labels_stay_distinct_and_keep_their_units(self):
        ticks = [0.2950, 0.2951, 0.2952]
        assert _axis_labels(ticks, "{:.1%}") == ["29.50%", "29.51%", "29.52%"]
        assert _axis_labels([1.0, 2.0], "£{:,.0f}") == ["£1", "£2"]
        assert _axis_labels([0.1, 0.2, 0.3], "£{:,.0f} per quote") == [
            "£0.1 per quote",
            "£0.2 per quote",
            "£0.3 per quote",
        ]
        assert _axis_labels([-0.1, -0.2], "{:.0f}") == ["−0.1", "−0.2"]

    @staticmethod
    def display_record(values: list[float], before: list[int], after: list[int]):
        record = synthetic_report(1, 2, 3)
        record["run"]["scenario_display"] = {
            "label": "x",
            "values": values,
            "format": "{:.1f}",
            "target": None,
        }
        point = record["points"][0]
        point["scenario"]["baseline_counts"] = before
        point["scenario"]["point_counts"] = after
        return record, point

    def test_the_cumulative_chart_follows_the_display_value_not_the_step(self):
        record, point = self.display_record([20.0, 15.0, 10.0], [1, 1, 2], [2, 2, 0])
        svg = _ecdf_chart(record, point, 500.0, 200.0)
        assert "At or below 10.0: 50.0% at baseline, 0.0% at the point" in svg
        assert "At or below 20.0: 100.0% at baseline, 100.0% at the point" in svg

    def test_repeated_display_values_are_one_slot(self):
        record, point = self.display_record([1.0, 1.0, 2.0], [1, 3, 4], [2, 2, 4])
        cumulative = _ecdf_chart(record, point, 500.0, 200.0)
        assert "At or below 1.0: 50.0% at baseline, 50.0% at the point" in cumulative
        assert cumulative.count('class="hit"') == 2
        distribution = _distribution_chart(record, point, 500.0, 200.0)
        assert "1.0: 4 quotes at baseline, 4 at the point" in distribution
        assert distribution.count('class="hit"') == 2

    def test_a_decreasing_display_still_marks_its_means(self):
        record, point = self.display_record([3.0, 2.0, 1.0], [1, 1, 1], [1, 1, 1])
        point["scenario"]["baseline_mean"] = 2.0
        point["scenario"]["point_mean"] = 2.0
        svg = _distribution_chart(record, point, 500.0, 200.0)
        assert svg.count('class="mk-d"') == 1
        assert svg.count('class="mk-m"') == 1

    def test_a_target_off_the_scale_is_named_and_not_drawn(self):
        record, point = self.display_record([0.1, 0.2, 0.3], [1, 1, 1], [1, 1, 1])
        record["run"]["scenario_display"]["target"] = 0.8
        assert 'class="mk-t"' not in _distribution_chart(record, point, 500.0, 200.0)
        assert "the target, 0.8, is off this scale" in _distribution_caption(
            record, point
        )
        record["run"]["scenario_display"]["target"] = 0.2
        assert 'class="mk-t"' in _distribution_chart(record, point, 500.0, 200.0)
        assert "the solid line is the target, 0.2" in _distribution_caption(
            record, point
        )

    def test_a_change_that_rounds_to_nothing_has_no_sign(self):
        display = {"format": "{:.1%}"}
        assert _disp_signed(display, -0.00001) == "0.0%"
        assert _disp_signed(display, 0.01) == "+1.0%"
        assert _disp_signed(display, -0.01) == "−1.0%"
        assert _disp_signed({"format": "{:+.1%}"}, 0.01) == "+1.0%"
        assert _disp_signed(display, None) == "—"

    def test_the_convergence_charts_use_the_trace(self):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors, record_history=True)
        html = pc.FrontierReport(optimiser, frontier, df, factors).to_html()
        checks = section(html, "checks")
        assert "over the solve" in checks
        assert "Pass 1" in checks

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"movement_band": 0.0}, "movement_band"),
            ({"movement_band": float("nan")}, "movement_band"),
            ({"projection": pc.Projection(window_days=0)}, "projection"),
            ({"projection": pc.Projection(29, horizons=[])}, "projection"),
            ({"projection": pc.Projection(29, horizons=[-1])}, "projection"),
            ({"ratios": {"r": ("objective", "nope")}}, "unknown column"),
            ({"ratios": {"r": "objective"}}, "numerator"),
            ({"band_columns": ["region"]}, "must be numeric"),
            ({"band_columns": ["nope"]}, "band_columns not on factors"),
            ({"segment_columns": [[]]}, "segment_columns"),
            ({"rate_label": ""}, "rate_label"),
        ],
    )
    def test_invalid_options_raise(self, kwargs, match):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        with pytest.raises(ValueError, match=match):
            pc.FrontierReport(optimiser, frontier, df, factors, **kwargs)

    def test_a_band_column_with_nulls_raises(self):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        factors = factors.with_columns(
            pl.when(pl.col("quote_id") == "q0000")
            .then(None)
            .otherwise(1.0)
            .alias("premium")
        )
        with pytest.raises(ValueError, match="no nulls"):
            pc.FrontierReport(
                optimiser, frontier, df, factors, band_columns=["premium"]
            )

    def test_a_non_finite_target_raises(self):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        display = pc.ScenarioDisplay("x", DISPLAY, target=float("inf"))
        with pytest.raises(ValueError, match="target"):
            pc.FrontierReport(
                optimiser, frontier, df, factors, scenario_display=display
            )

    @pytest.mark.parametrize(
        "change, match",
        [
            (lambda d: d.drop("volume"), "sensitivity grid 'Lower' has no column"),
            (
                lambda d: d.with_columns(pl.col("scenario_value") * 1.5),
                "different scenario values",
            ),
            (
                lambda d: d.with_columns(pl.col("quote_id") + "x"),
                "same quotes",
            ),
        ],
    )
    def test_a_mismatched_sensitivity_grid_raises(self, change, match):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        with pytest.raises(ValueError, match=match):
            pc.FrontierReport(
                optimiser,
                frontier,
                df,
                factors,
                sensitivity_grids={"Lower": change(df)},
            )

    def test_a_sensitivity_grid_must_be_a_frame(self):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        grid = _evaluation_grid(optimiser, df, [])
        with pytest.raises(TypeError, match="must be a polars DataFrame"):
            pc.FrontierReport(
                optimiser, frontier, df, factors, sensitivity_grids={"Lower": grid}
            )

    def test_a_contexts_object_cant_hold_bands(self):
        df, factors = book()
        factors = factors.with_columns(pl.lit(1.0).alias("premium"))
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        contexts = pc.RatebookFactorContexts.from_dataframe(
            factors, SPECS, quote_id="quote_id", separator=pc.FACTOR_SEPARATOR
        )
        with pytest.raises(ValueError, match="band_columns need the factors frame"):
            pc.FrontierReport(
                optimiser, frontier, df, contexts, band_columns=["premium"]
            )

    def test_identical_inputs_give_identical_rich_pages(self, rich):
        df, factors, optimiser, frontier, report, low = rich
        again = pc.FrontierReport(
            optimiser,
            frontier,
            df,
            factors,
            report_columns=["margin"],
            segment_columns=["channel", ["channel", "region"]],
            band_columns=["premium"],
            sensitivity_grids={"Lower": low},
            sensitivity_note="Scored with the lower model.",
            ratios={"Margin per objective": ("margin", "objective")},
            movement_band=0.02,
            projection=pc.Projection(window_days=29, horizons=[30, 365]),
            scenario_display=pc.ScenarioDisplay("Rate", DISPLAY, "{:.1%}", target=0.2),
            generated_at="2026-11-09T12:00:00+00:00",
        )
        assert again.to_html() == report.to_html()


class TestRateCard:
    def test_rows_by_hand(self):
        df, factors = book()
        existing = rates_for(factors)
        existing["region"]["unseen"] = 1.5
        optimiser = make_optimiser(existing_rates=existing)
        frontier = sweep(optimiser, df, factors)
        report = pc.FrontierReport(optimiser, frontier, df, factors)
        card = report.data["points"][0]["rate_card"][0]
        table = frontier.point_factor_tables(0)["region"]
        for level in card["levels"]:
            assert level["new"] == table[level["label"]]
            assert (
                level["existing"] == optimiser.existing_rates["region"][level["label"]]
            )
            assert level["log_change"] == pytest.approx(
                math.log(table[level["label"]] / level["existing"])
            )
        carried = card["levels"][-1]
        assert carried["label"] == "unseen"
        assert carried["quotes"] == 0
        assert carried["mean_step_change"] is None
        assert carried["mean_step_change_reason"] == "no quotes"
        assert carried["impacts"] == {"objective": 0.0, "volume": 0.0}

    def test_band_edge_is_flagged_only_at_the_edges(self):
        # Every quote wants the highest price, so the search runs to the band's top.
        sv = np.linspace(0.5, 2.0, 61, dtype=np.float32)
        objective = np.tile(sv.astype(np.float64) * 10.0, (6, 1))
        df = scored_frame(sv, objective, volume=np.ones_like(objective))
        factors = pl.DataFrame(
            {"quote_id": [f"q{i:04d}" for i in range(6)], "f": list("aabbcc")}
        )
        existing = {"f": {"a": 1.0, "b": 1.1, "c": 0.9}}
        optimiser = RatebookOptimiser(
            objective="objective",
            constraints={"volume": {"min": None}},
            factor_columns=[["f"]],
            candidate_min=0.8,
            candidate_max=1.2,
            candidate_steps=5,
            existing_rates=existing,
        )
        frontier = optimiser.frontier(
            df, factors, threshold_ranges={"volume": (0.0, 1.0)}, n_points_per_dim=2
        )
        report = pc.FrontierReport(optimiser, frontier, df, factors)
        for level in report.data["points"][0]["rate_card"][0]["levels"]:
            assert level["band_edge"] == "max"

        interior = make_report(*book())[2]
        edges = [
            lv["band_edge"]
            for card in interior.data["points"][0]["rate_card"]
            for lv in card["levels"]
        ]
        assert None in edges

    def test_rate_changes_are_capped_and_by_range(self, built):
        *_, report = built
        changes = report.data["comparisons"]["rate_changes"][0]
        ranges = [
            math.log(max(lv["rates"])) - math.log(min(lv["rates"]))
            for lv in changes["levels"]
        ]
        assert ranges == sorted(ranges, reverse=True)
        assert all(
            len(lv["rates"]) == len(report.data["points"]) for lv in changes["levels"]
        )


def fake_point(
    index: int, objective: float, cost: float, thresholds: dict | None = None
) -> dict:
    return {
        "index": index,
        "name": f"Point {index}",
        "dominated": False,
        "thresholds": thresholds or {"cost": float(index)},
        "totals": {
            "objective": {
                "point": objective,
                "baseline": 10.0,
                "change": objective - 10.0,
            },
            "cost": {"point": cost, "baseline": 5.0, "change": cost - 5.0},
        },
        "audit": [
            {
                "constraint": "cost",
                "bound": 5.0,
                "slack": 0.0,
                "met": True,
                "binding": False,
            }
        ],
        "health": {"converged": True, "clamped_low": 0, "clamped_high": 0},
        "rate_card": [],
    }


class TestAcrossPoints:
    def test_dominance_respects_direction(self):
        points = [
            fake_point(0, 10.0, 5.0),
            fake_point(1, 9.0, 6.0),
            fake_point(2, 11.0, 7.0),
        ]
        _mark_dominance(points, "objective", {"cost": {"max": 1.0}})
        assert [p["dominated"] for p in points] == [False, True, False]

    def test_identical_points_dont_dominate_each_other(self):
        points = [fake_point(0, 10.0, 5.0), fake_point(1, 10.0, 5.0)]
        _mark_dominance(points, "objective", {"cost": {"max": 1.0}})
        assert [p["dominated"] for p in points] == [False, False]

    def test_marginal_rates_within_a_slice_and_duplicates(self):
        points = [
            fake_point(0, 10.0, 5.0),
            fake_point(1, 12.0, 6.0),
            fake_point(2, 13.0, 6.0),
        ]
        [piece] = _slices(points, ["cost"], "objective")
        assert piece["points"] == [0, 1, 2]
        assert piece["marginal"][0]["value"] == 2.0
        assert piece["marginal"][1]["value"] is None
        assert piece["marginal"][1]["value_reason"] == "no change in the constraint"

    def test_slices_hold_the_other_dimensions_fixed(self):
        points = []
        for i, (a, b) in enumerate(
            [(1.0, 10.0), (2.0, 10.0), (1.0, 20.0), (2.0, 20.0)]
        ):
            point = fake_point(i, 10.0 + i, 5.0 + i, {"cost": a, "other": b})
            point["totals"]["other"] = {
                "point": b + i,
                "baseline": 1.0,
                "change": b + i - 1.0,
            }
            points.append(point)
        pieces = _slices(points, ["cost", "other"], "objective")
        along_cost = [p for p in pieces if p["dimension"] == "cost"]
        assert [(p["fixed"], p["points"]) for p in along_cost] == [
            ({"other": 10.0}, [0, 1]),
            ({"other": 20.0}, [2, 3]),
        ]
        along_other = [p for p in pieces if p["dimension"] == "other"]
        assert [(p["fixed"], p["points"]) for p in along_other] == [
            ({"cost": 1.0}, [0, 2]),
            ({"cost": 2.0}, [1, 3]),
        ]

    def test_default_point_rule(self):
        points = [
            fake_point(0, 12.0, 9.0),
            fake_point(1, 11.0, 6.0),
            fake_point(2, 10.5, 5.0),
        ]
        assert _default_point(points, None, ["cost"]) == 2
        assert _default_point(points, 0, ["cost"]) == 0
        points[2]["dominated"] = True
        assert _default_point(points, None, ["cost"]) == 1
        for point in points:
            point["audit"][0]["met"] = False
        assert _default_point(points, None, ["cost"]) == 0

    def test_two_swept_dimensions_render_one_panel_each(self):
        df, factors = book()
        df = df.with_columns((pl.col("volume") * 2.0).alias("volume2"))
        optimiser = make_optimiser(
            constraints={"volume": {"min_pct": None}, "volume2": {"min_pct": None}}
        )
        frontier = optimiser.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.98, 1.0), "volume2": (0.98, 1.0)},
            n_points_per_dim=2,
        )
        report = pc.FrontierReport(optimiser, frontier, df, factors)
        slices = report.data["comparisons"]["slices"]
        assert {s["dimension"] for s in slices} == {"volume", "volume2"}
        assert report.to_html().count(">objective against volume<") == 0
        assert report.to_html().count("objective against volume2") >= 1


def beeswarm_point(
    levels_by_factor: dict[str, list[tuple[str, int, float, float]]],
) -> dict:
    """``{factor: [(label, quotes, existing, new), ...]}``."""
    card = []
    for factor, levels in levels_by_factor.items():
        card.append(
            {
                "factor": factor,
                "levels": [
                    {
                        "label": label,
                        "quotes": quotes,
                        "existing": existing,
                        "new": new,
                        "log_change": math.log(new / existing),
                    }
                    for label, quotes, existing, new in levels
                ],
            }
        )
    return {"index": 0, "rate_card": card}


def beeswarm_report(point: dict, top_n_factors: int = 25) -> dict:
    return {
        "display": {"top_n_factors": top_n_factors},
        "points": [point],
        "run": {"rate_label": "commission"},
    }


def dots_of(svg: str) -> list[dict]:
    """Every level's dot, in drawing order."""
    return [
        {
            "title": title,
            "shade": int(shade),
            "x": float(x),
            "y": float(y),
            "r": float(r),
        }
        for shade, x, y, r, title in re.findall(
            r'<circle class="sw(\d+)" cx="([0-9.]+)" cy="([0-9.]+)" r="([0-9.]+)">'
            r"<title>([^<]*)</title></circle>",
            svg,
        )
    ]


def by_level(svg: str) -> dict[str, dict]:
    return {d["title"].split(" = ")[1].split("\n")[0]: d for d in dots_of(svg)}


def draw(point: dict, kind: str = "changes", top_n_factors: int = 25) -> str:
    report = beeswarm_report(point, top_n_factors)
    return _beeswarm(report, point, kind, _beeswarm_domain(report, kind), 1166.0)


class TestBeeswarm:
    def test_one_dot_per_level_with_the_diameter_by_hand(self):
        point = beeswarm_point(
            {"f": [("big", 100, 1.0, 1.2), ("quarter", 25, 1.0, 0.85)]}
        )
        dots = by_level(draw(point))
        assert set(dots) == {"big", "quarter"}
        # Diameter is 6 + 28 x sqrt(quotes / largest).
        assert dots["big"]["r"] == pytest.approx(17.0)
        assert dots["quarter"]["r"] == pytest.approx((6 + 28 * 0.5) / 2)

    def test_the_largest_is_across_the_plot_not_the_row(self):
        point = beeswarm_point(
            {"f": [("big", 100, 1.0, 1.2)], "g": [("small", 25, 1.0, 1.2)]}
        )
        assert by_level(draw(point))["small"]["r"] == pytest.approx(10.0)

    def test_big_dots_are_drawn_first_so_small_ones_stay_visible(self):
        point = beeswarm_point(
            {
                "f": [
                    ("a", 5, 1.0, 1.1),
                    ("b", 500, 1.0, 1.1),
                    ("c", 50, 1.0, 1.1),
                ]
            }
        )
        radii = [d["r"] for d in dots_of(draw(point))]
        assert radii == sorted(radii, reverse=True)

    def test_numeric_levels_colour_by_value_not_text_order(self):
        point = beeswarm_point(
            {
                "f": [
                    ("10", 1, 1.0, 1.1),
                    ("2", 500, 1.0, 1.1),
                    ("1", 50, 1.0, 1.1),
                ]
            }
        )
        dots = by_level(draw(point))
        assert (dots["1"]["shade"], dots["2"]["shade"], dots["10"]["shade"]) == (
            0,
            5,
            10,
        )
        assert "Colour: level value" in dots["1"]["title"]

    def test_categorical_levels_colour_by_quotes(self):
        point = beeswarm_point(
            {
                "f": [
                    ("b", 10, 1.0, 1.1),
                    ("a", 30, 1.0, 1.1),
                    ("c", 20, 1.0, 1.1),
                ]
            }
        )
        dots = by_level(draw(point))
        assert (dots["b"]["shade"], dots["c"]["shade"], dots["a"]["shade"]) == (
            0,
            5,
            10,
        )
        assert "Colour: sample size" in dots["a"]["title"]

    def test_tied_quotes_rank_by_label_and_one_non_number_makes_it_categorical(self):
        point = beeswarm_point(
            {"f": [("2", 10, 1.0, 1.1), ("1", 10, 1.0, 1.1), ("x", 10, 1.0, 1.1)]}
        )
        dots = by_level(draw(point))
        assert (dots["1"]["shade"], dots["2"]["shade"], dots["x"]["shade"]) == (
            0,
            5,
            10,
        )

    def test_a_non_finite_label_is_not_a_number(self):
        positions, basis = _colour_positions(
            [{"label": "1", "quotes": 2}, {"label": "nan", "quotes": 1}]
        )
        assert basis == "sample size"
        assert positions == [1.0, 0.0]

    def test_a_single_level_is_at_the_middle_of_the_scale(self):
        point = beeswarm_point({"f": [("a", 10, 1.0, 1.1)]})
        assert dots_of(draw(point))[0]["shade"] == 5

    def test_vertical_jitter_is_the_stated_function_of_position(self):
        point = beeswarm_point(
            {"f": [("a", 10, 1.0, 1.1), ("b", 10, 1.0, 1.1), ("c", 10, 1.0, 1.1)]}
        )
        dots = by_level(draw(point))
        centre = 14.0 + 28.0 / 2
        for k, label in enumerate("abc"):
            expected = centre + ((k * 0.618034) % 1 - 0.5) * 0.36 * 28.0
            assert dots[label]["y"] == pytest.approx(expected, abs=0.051)

    def test_x_is_the_uncentred_log_of_the_rate_in_the_full_set(self):
        point = beeswarm_point({"f": [("one", 10, 1.0, 1.0), ("up", 10, 1.0, 1.25)]})
        report = beeswarm_report(point)
        domain = _beeswarm_domain(report, "full")
        dots = by_level(_beeswarm(report, point, "full", domain, 1166.0))
        zero = re.search(
            r'<line class="ref" x1="([0-9.]+)"',
            _beeswarm(report, point, "full", domain, 1166.0),
        )
        assert dots["one"]["x"] == pytest.approx(float(zero.group(1)), abs=0.11)
        assert dots["up"]["x"] > dots["one"]["x"]

    def test_the_full_set_ignores_the_existing_rate(self):
        point = beeswarm_point({"f": [("one", 10, 1.3, 1.0), ("same", 10, 1.25, 1.25)]})
        report = beeswarm_report(point)
        svg = _beeswarm(report, point, "full", _beeswarm_domain(report, "full"), 1166.0)
        zero = float(re.search(r'<line class="ref" x1="([0-9.]+)"', svg).group(1))
        dots = by_level(svg)
        assert dots["one"]["x"] == pytest.approx(zero, abs=0.11)
        assert dots["same"]["x"] > zero

    def test_x_is_the_change_in_the_changes_plot(self):
        point = beeswarm_point({"f": [("same", 10, 1.3, 1.3), ("up", 10, 1.0, 1.25)]})
        svg = draw(point)
        zero = float(re.search(r'<line class="ref" x1="([0-9.]+)"', svg).group(1))
        dots = by_level(svg)
        assert dots["same"]["x"] == pytest.approx(zero, abs=0.11)
        assert dots["up"]["x"] > zero

    def test_zero_quote_levels_draw_nothing(self):
        point = beeswarm_point({"f": [("a", 10, 1.0, 1.1), ("carried", 0, 1.0, 1.0)]})
        assert "carried" not in draw(point)

    def test_rows_order_by_size_of_change_and_ties_by_spec_order(self):
        point = beeswarm_point(
            {
                "small": [("a", 10, 1.0, 1.01)],
                "big": [("a", 10, 1.0, 1.2)],
                "tie1": [("a", 10, 1.0, 1.05)],
                "tie2": [("a", 10, 1.0, 1.05)],
            }
        )
        rows = _beeswarm_rows(point, "changes")
        assert [row["factor"] for row in rows] == ["big", "tie1", "tie2", "small"]

    def test_the_score_is_weighted_by_quotes(self):
        point = beeswarm_point(
            {
                "wide_but_small": [
                    ("a", 1, 1.0, 1.25),
                    ("b", 99, 1.0, 1.0),
                ],
                "narrow_but_big": [("a", 100, 1.0, 1.05)],
            }
        )
        rows = _beeswarm_rows(point, "changes")
        assert rows[0]["factor"] == "narrow_but_big"

    def test_the_domain_is_shared_across_points_and_widened(self):
        a = beeswarm_point({"f": [("l", 10, 1.0, 1.1)]})
        b = beeswarm_point({"f": [("l", 10, 1.0, 1.25)]})
        report = {"display": {"top_n_factors": 25}, "points": [a, b]}
        assert _beeswarm_domain(report, "changes") == pytest.approx(
            math.log(1.25) * 1.05
        )

    def test_a_zero_domain_is_widened(self):
        point = beeswarm_point({"f": [("l", 10, 1.0, 1.0)]})
        assert _beeswarm_domain(beeswarm_report(point), "changes") == 0.01

    def test_the_top_n_factors_is_respected(self):
        point = beeswarm_point(
            {f"f{i}": [("l", 10, 1.0, 1.0 + i / 100)] for i in range(5)}
        )
        assert len(dots_of(draw(point, top_n_factors=2))) == 2

    def test_axis_names_the_direction_with_plain_numbers(self):
        point = beeswarm_point({"f": [("a", 10, 1.0, 1.2), ("b", 10, 1.0, 0.8)]})
        svg = draw(point)
        assert "← decreases commission | increases commission →" in svg
        assert "log(new / existing)" in svg
        assert "%" not in re.sub(r"<title>.*?</title>", "", svg, flags=re.S)
        report = beeswarm_report(point)
        full = _beeswarm(
            report, point, "full", _beeswarm_domain(report, "full"), 1166.0
        )
        assert "Relativity, log(commission)" in full

    def test_the_colour_bar_runs_low_to_high(self):
        factors = {f"f{i}": [("a", 10, 1.0, 1.1)] for i in range(20)}
        svg = draw(beeswarm_point(factors))
        assert "Level value (numeric) or sample size (categorical)" in svg
        assert ">High<" in svg and ">Low<" in svg
        # The bar's top rectangle is the high end of the scale.
        shades = re.findall(r'<rect class="sw(\d+)"', svg)
        assert shades == [str(10 - k) for k in range(11)]

    def test_high_cardinality_stays_within_the_budget_and_inside_the_plot(self):
        rng = np.random.default_rng(1)
        factors = {
            f"f{i}": [
                (f"l{j}", int(rng.integers(1, 50)), 1.0, float(rng.uniform(0.8, 1.25)))
                for j in range(2000)
            ]
            for i in range(25)
        }
        point = beeswarm_point(factors)
        svg = draw(point)
        dots = dots_of(svg)
        assert 0 < len(dots) <= 500
        assert min(d["x"] - d["r"] for d in dots) >= 170.0
        assert max(d["x"] + d["r"] for d in dots) <= 1166.0 - 84.0
        report = beeswarm_report(point)
        caption = _beeswarm_caption(report, "changes", point)
        assert "49,500 smallest levels" in caption

    def test_a_page_with_many_points_draws_fewer_dots_each(self):
        levels = [(f"l{j}", 10, 1.0, 1.1) for j in range(600)]
        point = beeswarm_point({"f": levels})
        report = {**beeswarm_report(point), "points": [point] * 25}
        svg = _beeswarm(report, point, "changes", 0.3, 1166.0)
        assert len(dots_of(svg)) == 360
        assert "240 smallest levels" in _beeswarm_caption(report, "changes", point)

    def test_labels_keep_their_hyphens_in_the_hover(self):
        point = beeswarm_point({"a-b": [("c-d", 10, 1.0, 0.9)]})
        title = dots_of(draw(point))[0]["title"]
        assert title.startswith("a-b = c-d\n")
        assert "−10.0%" in title

    def test_a_dropped_level_is_the_smallest(self):
        levels = [(f"l{j}", 1000 + j, 1.0, 1.1) for j in range(500)]
        levels.append(("tiny", 1, 1.0, 1.2))
        svg = draw(beeswarm_point({"f": levels}))
        assert "tiny" not in svg
        assert len(dots_of(svg)) == 500


class Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)


class TestPage:
    def test_the_page_is_self_contained(self, built):
        *_, report = built
        html = report.to_html()
        parser = Tags()
        parser.feed(html)
        assert "script" not in parser.tags
        assert "link" not in parser.tags
        urls = set(re.findall(r"https?://[^\s\"'<>]+", html))
        assert urls <= {"http://www.w3.org/2000/svg"}

    def test_small_per_quote_amounts_stay_readable(self):
        from price_contour._frontier_report_html import _fine

        money = {"format": "£{:,.0f}"}
        assert _fine(money, 0.25) == "+£0.25"
        assert _fine(money, -0.25) == "−£0.25"
        assert _fine(money, 0.0) == "£0"
        assert _fine({"format": "{:.1%}"}, 0.0123) == "+1.23%"
        # A literal percent sign is text, not a request to multiply by 100.
        assert _fine({"format": "{:.2f}%"}, 1.0) == "+1%"
        assert _fine({"format": "£{:,.0f} pa"}, -2.5) == "−£2.5 pa"
        assert _fine(money, None) == "—"

    def test_choosing_a_point_or_factor_cant_move_the_page(self, built):
        *_, report = built
        html = report.to_html()
        # A clicked label focuses its input, and the browser scrolls to a focused
        # element, so these inputs must be fixed in the viewport.
        assert ".sr.stay{position:fixed" in html
        for group, cls in (("pt", "sr stay"), ("fx", "sr stay"), ("tb", "sr")):
            assert re.search(rf'<input class="{cls}"[^>]*id="{group}-0"', html), group

    def test_a_points_panel_can_be_hidden_by_the_switch(self, built):
        # An inline display would beat the stylesheet's rule that hides it.
        *_, report = built
        html = report.to_html()
        assert not re.search(
            r'class="[^"]*\bpt pt-\d+[^"]*"[^>]*style="[^"]*display', html
        )
        assert 'class="dc pt pt-0"' in html

    def test_points_with_the_same_totals_share_a_label(self):
        df, factors = book()
        optimiser = make_optimiser(
            candidate_min=1.0, candidate_max=1.0, candidate_steps=1
        )
        frontier = sweep(optimiser, df, factors)
        html = pc.FrontierReport(optimiser, frontier, df, factors).to_html()
        assert ">P0/P1/P2/P3</text>" in html

    def test_hostile_strings_are_escaped(self):
        df, factors = book()
        hostile = "<script>alert(1)</script>"
        factors = factors.with_columns(pl.lit(hostile).alias("channel"))
        _, _, report = make_report(
            df,
            factors,
            title=hostile,
            subtitle=hostile,
            labels={"objective": hostile},
            segment_columns=["channel"],
            metadata={hostile: hostile},
        )
        html = report.to_html()
        assert hostile not in html
        assert "&lt;script&gt;" in html

    def test_identical_inputs_give_identical_pages(self):
        df, factors = book()
        a = make_report(df, factors)[2].to_html()
        b = make_report(df, factors)[2].to_html()
        assert a == b

    def test_the_page_rebuilds_from_json(self, built):
        *_, report = built
        rebuilt = json.loads(json.dumps(report.data, allow_nan=False))
        assert pc.frontier_report_html(rebuilt) == report.to_html()

    def test_an_unknown_version_raises(self):
        with pytest.raises(ValueError, match="report_version"):
            pc.frontier_report_html({"report_version": 99})

    def test_non_default_limits_survive_json(self):
        df, factors = book()
        _, _, report = make_report(df, factors, top_n_levels=2, top_n_factors=1)
        assert report.data["display"] == {"top_n_levels": 2, "top_n_factors": 1}
        rebuilt = json.loads(json.dumps(report.data))
        assert pc.frontier_report_html(rebuilt) == report.to_html()
        assert (
            "+ 3 more levels" in report.to_html()
            or "+ 2 more levels" in report.to_html()
        )

    def test_save_writes_the_page(self, built, tmp_path):
        *_, report = built
        path = report.save(tmp_path / "report.html")
        assert path.read_text(encoding="utf-8") == report.to_html()

    def test_the_selected_point_defaults_to_the_highlight(self):
        df, factors = book()
        _, _, report = make_report(df, factors, highlight_point=2)
        assert report.data["default_point"] == 2
        assert re.search(r'id="pt-2" checked', report.to_html())

    def test_rows_reordered_give_the_same_report(self):
        df, factors = book()
        _, _, plain = make_report(df, factors)
        shuffled = factors.sample(fraction=1.0, shuffle=True, seed=4)
        optimiser = make_optimiser(existing_rates=rates_for(factors))
        frontier = sweep(optimiser, df, factors)
        report = pc.FrontierReport(
            optimiser, frontier, df, shuffled, generated_at="2026-11-09T12:00:00+00:00"
        )
        assert report.to_html() == plain.to_html()


def synthetic_scenario(ids: list[str], n_steps: int = 40) -> dict:
    return {
        "baseline_mean": 0.5,
        "point_mean": 0.6,
        "baseline_max": 1.0,
        "point_max": 1.0,
        "baseline_counts": [5] * n_steps,
        "point_counts": [5] * n_steps,
        "transitions": [
            {"from": k, "to": k, "quotes": 5, "changes": dict.fromkeys(ids, 0.0)}
            for k in range(n_steps)
        ],
    }


def synthetic_directions(ids: list[str]) -> dict:
    return dict.fromkeys(
        ("up", "down", "unchanged"),
        {
            "quotes": 10,
            "mean_display_change": 0.1,
            "mean_changes": dict.fromkeys(ids, 0.1),
        },
    )


def synthetic_report(
    n_points: int, n_factors: int, levels: int, n_columns: int = 2
) -> dict:
    """A record at a target size, without running a solve."""
    label = "a-rather-long-level-label-for-size-"
    columns = [
        {
            "id": f"c{k}",
            "role": "objective" if k == 0 else "constraint",
            "label": f"Column {k}",
            "format": "{:,.2f}",
        }
        for k in range(n_columns)
    ]
    ids = [c["id"] for c in columns]
    rng = np.random.default_rng(7)
    points = []
    for i in range(n_points):
        card = []
        for f in range(n_factors):
            count = 600 if f == 0 else levels
            rows = []
            for j in range(count):
                new = float(rng.uniform(0.8, 1.25))
                rows.append(
                    {
                        "label": f"{label}{j}",
                        "existing": 1.0,
                        "existing_reason": None,
                        "new": new,
                        "log_change": math.log(new),
                        "quotes": int(rng.integers(1, 500)),
                        "mean_step_change": float(rng.normal()),
                        "mean_step_change_reason": None,
                        "impacts": {c: float(rng.normal()) for c in ids},
                        "band_edge": None,
                    }
                )
            card.append({"factor": f"a-long-factor-name-{f}", "levels": rows})
        delta = {
            "percentiles": dict.fromkeys(
                ("p1", "p5", "p25", "p50", "p75", "p95", "p99"), 0.5
            ),
            "histogram": {
                "edges": [float(e) for e in range(41)],
                "counts": [10] * 40,
                "below": 1,
                "above": 1,
            },
            "moved_up": 1.0,
            "moved_down": -1.0,
        }
        points.append(
            {
                "index": i,
                "name": f"Point {i}",
                "artifact": None,
                "highlighted": False,
                "dominated": False,
                "thresholds": {"c1": 0.9},
                "bounds": {"c1": 1.0},
                "totals": {
                    c: {
                        "baseline": 1.0,
                        "point": 1.5 + i,
                        "change": 0.5 + i,
                        "pct": 0.5,
                        "pct_reason": None,
                    }
                    for c in ids
                },
                "audit": [
                    {
                        "constraint": "c1",
                        "bound": 1.0,
                        "achieved": 1.5,
                        "slack": 0.5,
                        "met": True,
                        "binding": False,
                    }
                ],
                "health": {
                    "converged": True,
                    "iterations": 3,
                    "lambdas": {"c1": 0.1},
                    "clamp_rate": 0.0,
                    "clamped_low": 0,
                    "clamped_high": 0,
                },
                "movement": {
                    "up": 10,
                    "down": 10,
                    "unchanged": 10,
                    "directions": synthetic_directions(ids),
                    "histogram": {"changes": [-1, 0, 1], "counts": [10, 10, 10]},
                    "percentiles_steps": dict.fromkeys(
                        ("p1", "p5", "p25", "p50", "p75", "p95", "p99"), 0
                    ),
                    "percentiles_display": None,
                },
                "sensitivity": {},
                "ratios": [],
                "projection": None,
                "scenario": synthetic_scenario(ids),
                "deltas": dict.fromkeys(ids, delta),
                "rate_card": card,
                "segments": [],
                "bands": [],
                "history": None,
            }
        )
    return {
        "report_version": 1,
        "generated_at": "2026-11-09T12:00:00+00:00",
        "title": "Size",
        "subtitle": None,
        "default_point": 0,
        "library": {"price_contour_version": "x"},
        "run": {
            "objective": "c0",
            "constraints": [{"name": "c1", "kind": "min_pct"}],
            "swept": ["c1"],
            "n_quotes": 200_000,
            "n_steps": 40,
            "scenario_values": [0.5 + k * 0.025 for k in range(40)],
            "scenario_display": None,
            "candidates": {"min": 0.8, "max": 1.25, "steps": 41, "values": [0.8, 1.25]},
            "has_existing_rates": True,
            "base_multiplier": None,
            "solver": {
                "max_cd_iterations": 3,
                "cd_tolerance": 1e-3,
                "max_iter": 50,
                "tolerance": 1e-6,
            },
            "sensitivity_names": [],
            "sensitivity_note": None,
            "movement_band": None,
            "rate_label": "rate",
            "projection": None,
            "metadata": [],
        },
        "columns": columns,
        "thresholds": {},
        "display": {"top_n_levels": 25, "top_n_factors": 25},
        "baseline": {
            "totals": dict.fromkeys(ids, 1.0),
            "clamped_low": 0,
            "clamped_high": 0,
        },
        "points": points,
        "comparisons": {
            "slices": [
                {
                    "dimension": "c1",
                    "fixed": {},
                    "points": list(range(n_points)),
                    "marginal": [
                        {"from": i, "to": i + 1, "value": 1.0, "value_reason": None}
                        for i in range(n_points - 1)
                    ],
                }
            ],
            "rate_changes": [],
        },
        "flags": [],
    }


def segment_level(label: str, quotes: int, ids: list[str], value: float = 2.0) -> dict:
    return {
        "label": label,
        "quotes": quotes,
        "baseline": dict.fromkeys(ids, 1.0),
        "point": dict.fromkeys(ids, value),
        "mean_step_change": 0.1,
        "mean_step_change_reason": None,
        "baseline_display_mean": 0.5,
        "point_display_mean": 0.6,
        "mean_abs_display_change": 0.1,
        "max_abs_display_change": 0.2,
        "share_beyond_band": None,
        "share_at_first": 0.0,
        "share_at_last": 0.0,
    }


def add_segments(record: dict, n_segments: int, n_levels: int) -> None:
    ids = [c["id"] for c in record["columns"]]
    for point in record["points"]:
        point["segments"] = [
            {
                "column": f"s{s}",
                "levels": [
                    segment_level(f"level{j}", 100 - j, ids) for j in range(n_levels)
                ],
            }
            for s in range(n_segments)
        ]


def section(html: str, slug: str) -> str:
    start = html.index(f'id="{slug}"')
    return html[start : html.index("</section>", start)]


class TestSize:
    def test_segment_tables_stay_within_their_own_budget(self):
        record = synthetic_report(25, 10, 5, n_columns=7)
        add_segments(record, 10, 25)
        html = pc.frontier_report_html(record)
        assert section(html, "segments").count("<td") <= 60_000
        total = section(html, "segments").count("<td") + section(
            html, "rate-card"
        ).count("<td")
        assert total <= 120_000
        assert 'with the rest in "Other"' in html
        assert "Other (" in section(html, "segments")

    def test_segments_that_cant_fit_are_omitted_and_it_says_so(self):
        record = synthetic_report(25, 5, 5)
        add_segments(record, 500, 3)
        html = pc.frontier_report_html(record)
        assert "The segment tables were omitted" in html

    def test_the_target_scale_stays_within_budget(self):
        record = synthetic_report(25, 55, 25)
        start = time.perf_counter()
        html = pc.frontier_report_html(record)
        elapsed = time.perf_counter() - start
        assert len(html.encode()) < 10_000_000
        assert elapsed < 5.0

    def test_the_cell_budget_lowers_the_row_limit_and_says_so(self):
        record = synthetic_report(25, 55, 25, n_columns=8)
        limit = _rate_row_limit(record)
        assert 0 < limit < 25
        assert f"Showing up to {limit} levels per factor" in pc.frontier_report_html(
            record
        )

    def test_a_limit_of_zero_omits_the_tables_and_says_so(self):
        record = synthetic_report(25, 55, 5, n_columns=2)
        record["columns"] += [
            {"id": f"x{k}", "role": "report", "label": f"X{k}", "format": "{:,.2f}"}
            for k in range(130)
        ]
        for point in record["points"]:
            point["segments"] = [
                {
                    "column": "s",
                    "levels": [
                        segment_level(
                            str(j), 1, [c["id"] for c in record["columns"]], 1.0
                        )
                        for j in range(25)
                    ],
                }
                for _ in range(2)
            ]
            point["totals"].update(
                {c["id"]: point["totals"]["c0"] for c in record["columns"][2:]}
            )
            point["deltas"].update(
                {c["id"]: point["deltas"]["c0"] for c in record["columns"][2:]}
            )
            ids = [c["id"] for c in record["columns"]]
            point["movement"]["directions"] = synthetic_directions(ids)
            point["scenario"] = synthetic_scenario(ids)
            for card in point["rate_card"]:
                for level in card["levels"]:
                    level["impacts"].update(
                        {c["id"]: 0.0 for c in record["columns"][2:]}
                    )
        assert _rate_row_limit(record) == 0
        assert "were omitted to keep the page" in pc.frontier_report_html(record)


class TestNumericalContracts:
    def test_no_change_anywhere(self):
        df, factors = book()
        optimiser = make_optimiser(
            candidate_min=1.0, candidate_max=1.0, candidate_steps=1
        )
        frontier = sweep(optimiser, df, factors)
        report = pc.FrontierReport(optimiser, frontier, df, factors)
        point = report.data["points"][0]
        assert point["totals"]["objective"]["change"] == 0.0
        assert point["totals"]["objective"]["pct"] == 0.0
        assert point["movement"]["unchanged"] == N
        delta = point["deltas"]["objective"]
        assert delta["histogram"]["edges"] == [0.0, 0.0]
        assert delta["histogram"]["counts"] == [N]
        marginal = report.data["comparisons"]["slices"][0]["marginal"][0]
        assert marginal["value"] is None
        assert marginal["value_reason"] == "no change in the constraint"
        assert "<svg" in report.to_html()

    def test_a_zero_baseline_has_no_percentage(self):
        df, factors = book(n=20)
        df = df.with_columns(
            (pl.col("scenario_value") - 1.0).abs().cast(pl.Float32).alias("zero_at_one")
        )
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        report = pc.FrontierReport(
            optimiser, frontier, df, factors, report_columns=["zero_at_one"]
        )
        entry = report.data["points"][-1]["totals"]["zero_at_one"]
        assert entry["baseline"] == pytest.approx(0.0, abs=1e-5)
        if entry["baseline"] == 0.0:
            assert entry["pct"] is None
            assert entry["pct_reason"] == "baseline is zero"

    def test_a_negative_baseline_divides_by_its_size(self):
        from price_contour._frontier_report import _total_entry

        entry = _total_entry(-10.0, -8.0)
        assert entry["change"] == 2.0
        assert entry["pct"] == 0.2
        zero = _total_entry(0.0, 3.0)
        assert zero["pct"] is None
        assert zero["pct_reason"] == "baseline is zero"

    def test_one_quote(self):
        df, factors = book(n=1)
        _, _, report = make_report(df, factors)
        movement = report.data["points"][0]["movement"]
        assert movement["up"] + movement["down"] + movement["unchanged"] == 1
        assert "<svg" in report.to_html()


def flag_point(**overrides) -> dict:
    point = fake_point(0, 10.0, 5.0)
    point["rate_card"] = [
        {
            "factor": "f",
            "levels": [
                {"quotes": 1, "band_edge": "max"},
                {"quotes": 1, "band_edge": None},
                {"quotes": 0, "band_edge": None},
            ],
        }
    ]
    point["health"]["clamped_low"] = overrides.get("clamped", 0)
    point["audit"][0].update(overrides.get("audit", {}))
    return point


THRESHOLDS = {
    "clamped_share": 0.1,
    "baseline_clamped_share": 0.1,
    "band_edge_share": 0.5,
    "binding_tolerance": 0.005,
    "miss_tolerance": 0.001,
}


class TestFlags:
    @pytest.mark.parametrize(
        ("clamped", "fires"), [(9, False), (10, False), (11, True)]
    )
    def test_clamped_share(self, clamped, fires):
        flags = _review_flags([flag_point(clamped=clamped)], THRESHOLDS, 100, 0)
        assert any(f["check"] == "clamped_share" for f in flags) is fires

    @pytest.mark.parametrize(
        ("clamped", "fires"), [(9, False), (10, False), (11, True)]
    )
    def test_baseline_clamped_share(self, clamped, fires):
        flags = _review_flags([flag_point()], THRESHOLDS, 100, clamped)
        assert any(f["check"] == "baseline_clamped_share" for f in flags) is fires

    @pytest.mark.parametrize(
        ("limit", "fires"), [(0.6, False), (0.5, False), (0.4, True)]
    )
    def test_band_edge_counts_only_levels_with_quotes(self, limit, fires):
        thresholds = {**THRESHOLDS, "band_edge_share": limit}
        flags = _review_flags([flag_point()], thresholds, 100, 0)
        assert any(f["check"] == "band_edge_share" for f in flags) is fires

    def test_a_disabled_check_stays_off(self):
        thresholds = {**THRESHOLDS, "clamped_share": None}
        flags = _review_flags([flag_point(clamped=99)], thresholds, 100, 0)
        assert not any(f["check"] == "clamped_share" for f in flags)

    def test_a_missed_constraint_is_critical(self):
        point = flag_point(audit={"met": False, "slack": -1.0})
        [flag] = [
            f
            for f in _review_flags([point], THRESHOLDS, 100, 0)
            if f["check"] == "constraint_missed"
        ]
        assert flag["status"] == "critical"

    def test_not_converged_and_dominated(self):
        point = flag_point()
        point["health"]["converged"] = False
        point["dominated"] = True
        checks = {
            f["check"]: f["status"] for f in _review_flags([point], THRESHOLDS, 100, 0)
        }
        assert checks["not_converged"] == "info"
        assert checks["dominated"] == "info"

    def test_audit_met_and_binding_at_the_tolerance(self):
        from price_contour._frontier_report import _audit

        constraints = {"c": {"min": 1.0}}
        at = _audit(constraints, {"c": 100.0}, {"c": 99.9}, THRESHOLDS)[0]
        assert at["met"] is True
        past = _audit(constraints, {"c": 100.0}, {"c": 99.8}, THRESHOLDS)[0]
        assert past["met"] is False
        assert (
            _audit(constraints, {"c": 100.0}, {"c": 100.5}, THRESHOLDS)[0]["binding"]
            is True
        )
        assert (
            _audit(constraints, {"c": 100.0}, {"c": 100.6}, THRESHOLDS)[0]["binding"]
            is False
        )
        maximum = _audit({"c": {"max": 1.0}}, {"c": 100.0}, {"c": 90.0}, THRESHOLDS)[0]
        assert maximum["slack"] == 10.0


class TestValidation:
    @pytest.fixture(scope="class")
    def parts(self):
        df, factors = book()
        optimiser = make_optimiser()
        return df, factors, optimiser, sweep(optimiser, df, factors)

    def test_a_different_objective_raises(self, parts):
        df, factors, _, frontier = parts
        other = make_optimiser(objective="margin")
        with pytest.raises(ValueError, match="objective"):
            pc.FrontierReport(other, frontier, df, factors)

    def test_a_flipped_constraint_direction_raises(self, parts):
        df, factors, _, frontier = parts
        other = make_optimiser(constraints={"volume": {"max_pct": None}})
        with pytest.raises(ValueError, match="constraints"):
            pc.FrontierReport(other, frontier, df, factors)

    def test_different_existing_rates_raise(self, parts):
        df, factors, _, frontier = parts
        other = make_optimiser(existing_rates=rates_for(factors))
        with pytest.raises(ValueError, match="existing_rates"):
            pc.FrontierReport(other, frontier, df, factors)

    def test_a_frontier_without_provenance_raises(self, parts):
        df, factors, optimiser, frontier = parts
        bare = type(frontier)(
            frontier.points, frontier.constraint_names, frontier.factor_tables
        )
        with pytest.raises(ValueError, match="provenance"):
            pc.FrontierReport(optimiser, bare, df, factors)

    def test_ratio_constraints_raise(self, parts):
        df, factors, _, frontier = parts
        ratio = make_optimiser(
            constraints={"lr": {"numerator": "a", "denominator": "b", "max": 1.0}}
        )
        with pytest.raises(ValueError, match="ratio"):
            pc.FrontierReport(ratio, frontier, df, factors)

    def test_missing_columns_raise(self, parts):
        df, factors, optimiser, frontier = parts
        with pytest.raises(ValueError, match="no column"):
            pc.FrontierReport(optimiser, frontier, df, factors, report_columns=["nope"])
        with pytest.raises(ValueError, match="segment_columns"):
            pc.FrontierReport(
                optimiser, frontier, df, factors, segment_columns=["nope"]
            )

    def test_a_report_column_cant_repeat_the_objective(self, parts):
        df, factors, optimiser, frontier = parts
        with pytest.raises(ValueError, match="repeat"):
            pc.FrontierReport(
                optimiser, frontier, df, factors, report_columns=["objective"]
            )

    def test_a_grid_or_contexts_cant_hold_the_extras(self, parts):
        df, factors, optimiser, frontier = parts
        builder = pc.QuoteGridBuilder(["volume"], objective="objective")
        builder.append(df)
        grid = builder.build()
        with pytest.raises(ValueError, match="long frame"):
            pc.FrontierReport(
                optimiser, frontier, grid, factors, report_columns=["margin"]
            )
        contexts = pc.RatebookFactorContexts.from_dataframe(
            factors, SPECS, expected_quote_ids=grid.quote_ids
        )
        with pytest.raises(ValueError, match="factors frame"):
            pc.FrontierReport(
                optimiser, frontier, grid, contexts, segment_columns=["channel"]
            )

    def test_a_grid_and_contexts_work_without_extras(self, parts):
        df, factors, optimiser, frontier = parts
        builder = pc.QuoteGridBuilder(["volume"], objective="objective")
        builder.append(df)
        grid = builder.build()
        contexts = pc.RatebookFactorContexts.from_dataframe(
            factors, SPECS, expected_quote_ids=grid.quote_ids
        )
        report = pc.FrontierReport(
            optimiser,
            frontier,
            grid,
            contexts,
            generated_at="2026-11-09T12:00:00+00:00",
        )
        direct = pc.FrontierReport(
            optimiser, frontier, df, factors, generated_at="2026-11-09T12:00:00+00:00"
        )
        assert report.data == direct.data

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"point_names": ["only one"]}, "point_names"),
            ({"point_artifacts": {99: "x"}}, "point_artifacts"),
            ({"highlight_point": 99}, "highlight_point"),
            ({"top_n_levels": 0}, "top_n"),
            ({"top_n_factors": 0}, "top_n"),
            ({"formats": {"objective": "{:q}"}}, "format"),
            ({"thresholds": {"nope": 1.0}}, "unknown threshold"),
            ({"thresholds": {"clamped_share": -1.0}}, ">= 0"),
            ({"thresholds": {"clamped_share": float("nan")}}, ">= 0"),
            ({"thresholds": {"miss_tolerance": None}}, "defines the audit"),
            ({"thresholds": {"binding_tolerance": None}}, "defines the audit"),
        ],
    )
    def test_invalid_options_raise(self, parts, kwargs, match):
        df, factors, optimiser, frontier = parts
        with pytest.raises(ValueError, match=match):
            pc.FrontierReport(optimiser, frontier, df, factors, **kwargs)

    def test_scenario_display_length_and_values_raise(self, parts):
        df, factors, optimiser, frontier = parts
        with pytest.raises(ValueError, match="scenario_display"):
            pc.FrontierReport(
                optimiser,
                frontier,
                df,
                factors,
                scenario_display=pc.ScenarioDisplay("x", [1.0]),
            )
        values = [float("nan")] * len(SV)
        with pytest.raises(ValueError, match="scenario_display"):
            pc.FrontierReport(
                optimiser,
                frontier,
                df,
                factors,
                scenario_display=pc.ScenarioDisplay("x", values),
            )


def key_shape(value):
    if isinstance(value, dict):
        return {k: key_shape(v) for k, v in value.items()}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return [key_shape(value[0])]
    return None


class TestProvenanceIsASnapshot:
    def test_changing_the_optimiser_after_the_sweep_is_caught(self):
        df, factors = book()
        optimiser = make_optimiser(existing_rates=rates_for(factors))
        frontier = sweep(optimiser, df, factors)
        optimiser.existing_rates["region"]["N"] = 9.0
        with pytest.raises(ValueError, match="existing_rates"):
            pc.FrontierReport(optimiser, frontier, df, factors)

    def test_changing_a_constraint_in_place_is_caught(self):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        spec = optimiser.constraints["volume"]
        spec["max_pct"] = spec.pop("min_pct")
        with pytest.raises(ValueError, match="constraints"):
            pc.FrontierReport(optimiser, frontier, df, factors)

    @pytest.mark.parametrize(
        "setting, value",
        [
            ("max_cd_iterations", 7),
            ("cd_tolerance", 0.5),
            ("max_iter", 1),
            ("tolerance", 0.5),
        ],
    )
    def test_changing_a_solver_setting_after_the_sweep_is_caught(self, setting, value):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors)
        setattr(optimiser, setting, value)
        with pytest.raises(ValueError, match=setting):
            pc.FrontierReport(optimiser, frontier, df, factors)

    def test_the_record_cites_the_sweeps_solver_settings(self, built):
        *_, optimiser, _, report = built
        assert report.data["run"]["solver"] == {
            "max_cd_iterations": optimiser.max_cd_iterations,
            "cd_tolerance": optimiser.cd_tolerance,
            "max_iter": optimiser.max_iter,
            "tolerance": optimiser.tolerance,
        }

    def test_the_record_cant_be_changed_through_the_getter(self):
        df, factors = book()
        optimiser = make_optimiser(existing_rates=rates_for(factors))
        frontier = sweep(optimiser, df, factors)
        frontier.provenance["existing_rates"]["region"]["N"] = 9.0
        pc.FrontierReport(optimiser, frontier, df, factors)


class TestRecordStructure:
    def test_the_record_matches_its_pinned_keys(self, built):
        *_, report = built
        shape = key_shape(report.data)
        assert set(shape) == {
            "report_version",
            "generated_at",
            "title",
            "subtitle",
            "default_point",
            "library",
            "run",
            "columns",
            "thresholds",
            "display",
            "baseline",
            "points",
            "comparisons",
            "flags",
        }
        assert set(shape["run"]) == {
            "objective",
            "constraints",
            "swept",
            "n_quotes",
            "n_steps",
            "scenario_values",
            "scenario_display",
            "candidates",
            "has_existing_rates",
            "base_multiplier",
            "solver",
            "sensitivity_names",
            "sensitivity_note",
            "movement_band",
            "rate_label",
            "projection",
            "metadata",
        }
        assert set(shape["run"]["solver"]) == {
            "max_cd_iterations",
            "cd_tolerance",
            "max_iter",
            "tolerance",
        }
        assert set(shape["baseline"]) == {"totals", "clamped_low", "clamped_high"}
        assert set(shape["display"]) == {"top_n_levels", "top_n_factors"}
        assert set(shape["thresholds"]) == {
            "clamped_share",
            "baseline_clamped_share",
            "band_edge_share",
            "binding_tolerance",
            "miss_tolerance",
        }
        point = shape["points"][0]
        assert set(point) == {
            "index",
            "name",
            "artifact",
            "highlighted",
            "dominated",
            "thresholds",
            "bounds",
            "totals",
            "audit",
            "health",
            "sensitivity",
            "ratios",
            "projection",
            "scenario",
            "movement",
            "deltas",
            "rate_card",
            "segments",
            "bands",
            "history",
        }
        assert set(point["scenario"]) == {
            "baseline_mean",
            "point_mean",
            "baseline_max",
            "point_max",
            "baseline_counts",
            "point_counts",
            "transitions",
        }
        assert set(point["scenario"]["transitions"][0]) == {
            "from",
            "to",
            "quotes",
            "changes",
        }
        assert set(point["totals"]["objective"]) == {
            "baseline",
            "point",
            "change",
            "pct",
            "pct_reason",
        }
        assert set(point["audit"][0]) == {
            "constraint",
            "bound",
            "achieved",
            "slack",
            "met",
            "binding",
        }
        assert set(point["movement"]) == {
            "up",
            "down",
            "unchanged",
            "directions",
            "histogram",
            "percentiles_steps",
            "percentiles_display",
        }
        assert set(point["movement"]["directions"]) == {"up", "down", "unchanged"}
        assert set(point["movement"]["directions"]["up"]) == {
            "quotes",
            "mean_display_change",
            "mean_changes",
        }
        assert set(point["deltas"]["objective"]) == {
            "percentiles",
            "histogram",
            "moved_up",
            "moved_down",
        }
        assert set(point["deltas"]["objective"]["histogram"]) == {
            "edges",
            "counts",
            "below",
            "above",
        }
        assert set(point["rate_card"][0]["levels"][0]) == {
            "label",
            "existing",
            "existing_reason",
            "new",
            "log_change",
            "quotes",
            "mean_step_change",
            "mean_step_change_reason",
            "impacts",
            "band_edge",
        }
        assert set(point["segments"][0]["levels"][0]) == {
            "label",
            "quotes",
            "baseline",
            "point",
            "mean_step_change",
            "mean_step_change_reason",
            "baseline_display_mean",
            "point_display_mean",
            "mean_abs_display_change",
            "max_abs_display_change",
            "share_beyond_band",
            "share_at_first",
            "share_at_last",
        }
        assert set(shape["comparisons"]) == {"slices", "rate_changes"}
        assert set(shape["comparisons"]["slices"][0]) == {
            "dimension",
            "fixed",
            "points",
            "marginal",
        }

    def test_columns_are_in_canonical_order(self, built):
        *_, report = built
        assert [(c["id"], c["role"]) for c in report.data["columns"]] == [
            ("objective", "objective"),
            ("volume", "constraint"),
            ("margin", "report"),
        ]

    def test_history_is_kept_when_recorded(self):
        df, factors = book()
        optimiser = make_optimiser()
        frontier = sweep(optimiser, df, factors, record_history=True)
        report = pc.FrontierReport(optimiser, frontier, df, factors)
        history = report.data["points"][0]["history"]
        assert history[0]["cd_iteration"] == 1
        assert set(history[0]) == {
            "cd_iteration",
            "factor",
            "factor_index",
            "total_objective",
            "totals",
            "inner_iterations",
            "inner_converged",
        }
        assert make_report(df, factors)[2].data["points"][0]["history"] is None
