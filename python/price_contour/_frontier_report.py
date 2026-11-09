"""Assemble a frontier impact report (DESIGN_DECISIONS §19).

``FrontierReport.data`` is a JSON-safe record with ``report_version`` 1, and
:func:`frontier_report_html` is a pure view of it, so a page can be rebuilt from
a record logged next to a run. Per-quote work is in Rust (``report_points_py``);
this module validates the inputs and shapes the aggregates into the record.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from price_contour._frontier_report_html import REPORT_VERSION, frontier_report_html
from price_contour._grid_utils import build_grid
from price_contour._native_columns import select_native_columns
from price_contour._price_contour import (
    FactorContext,
    QuoteGrid,
    RatebookFactorContexts,
    report_points_py,
)
from price_contour.ratebook import (
    FACTOR_SEPARATOR,
    RatebookFrontierResult,
    RatebookOptimiser,
    _existing_values,
    _f32,
    _factor_values_for_contexts,
    _resolve_factor_contexts,
    _validate_base_multiplier_input,
    _validate_existing_rates_input,
)
from price_contour.solver import _ratio_constraint_names

_PERCENTILES = ("p1", "p5", "p25", "p50", "p75", "p95", "p99")
_HISTOGRAM_BINS = 40
_DEFAULT_FORMAT = "{:,.4g}"
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "clamped_share": 0.01,
    "baseline_clamped_share": 0.01,
    "band_edge_share": 0.25,
    "binding_tolerance": 0.005,
    "miss_tolerance": 0.001,
}
# These define the audit's columns, so they can't be switched off.
_REQUIRED_THRESHOLDS = frozenset({"binding_tolerance", "miss_tolerance"})
_DIRECTIONS = ("min", "max", "min_pct", "max_pct")
_N_BANDS = 10
_DEFAULT_RATIO_FORMAT = "{:.2%}"


@dataclass(frozen=True)
class ScenarioDisplay:
    """How to show a scenario step to a reader, when the scenario value is a
    transform of something more natural."""

    label: str
    values: Sequence[float]
    format: str = "{:.2f}"
    target: float | None = None


@dataclass(frozen=True)
class Projection:
    """Scale each total from the days the data covers to other horizons."""

    window_days: float
    horizons: Sequence[float] = (30, 90, 365)


class FrontierReport:
    """An impact report for a ratebook frontier (DESIGN_DECISIONS §19).

    Parameters
    ----------
    optimiser : RatebookOptimiser
        The optimiser that produced ``frontier``.
    frontier : RatebookFrontierResult
    grid_df : pl.DataFrame | QuoteGrid
        The long scored frame used for the sweep, or a ``QuoteGrid`` when there
        are no ``report_columns``.
    factors : pl.DataFrame | RatebookFactorContexts
        The same per-quote factors, or contexts when there are no
        ``segment_columns``.
    title, subtitle : str
    labels : dict[str, str], optional
        Display name per column.
    formats : dict[str, str], optional
        ``str.format`` pattern per column, for example ``"£{:,.0f}"``.
    report_columns : list[str], optional
        Extra ``grid_df`` columns to total at every point.
    segment_columns : list[str | list[str]], optional
        Per-quote columns on ``factors`` to break the impact down by. A list of
        columns is an interaction.
    band_columns : list[str], optional
        Numeric per-quote columns on ``factors``, each cut into ten equal-count
        bands and broken down like a segment.
    scenario_display : ScenarioDisplay, optional
        Show steps in natural units, with an optional target for their mean.
    sensitivity_grids : dict[str, pl.DataFrame], optional
        The same quotes and steps scored under other assumptions, for example
        ``{"Lower": low, "Upper": high}``. Each column is totalled with every
        point's steps held fixed.
    sensitivity_note : str, optional
        How those grids were built; shown beneath the summary.
    ratios : dict[str, tuple[str, str]], optional
        Label to (numerator, denominator) columns. ``formats`` may give a pattern
        for the label; the default is ``"{:.2%}"``.
    movement_band : float, optional
        In display units: the report shows the share of quotes moved further.
    projection : Projection, optional
        Scale totals from the data's window to other horizons.
    rate_label : str
        Names the direction on the beeswarm axes.
    point_names : list[str], optional
        Display name per point.
    point_artifacts : dict[int, str], optional
        File name of each point's rate card.
    highlight_point : int, optional
        A point to mark as recommended, selected when the page opens.
    metadata : dict[str, str], optional
        Free-form run facts.
    thresholds : dict[str, float | None], optional
        Overrides the review thresholds.
    top_n_levels, top_n_factors : int
        Display caps: level rows per factor, and factors per beeswarm.
    generated_at : str, optional
        A fixed timestamp, for reproducible output.
    """

    def __init__(
        self,
        optimiser: RatebookOptimiser,
        frontier: RatebookFrontierResult,
        grid_df: pl.DataFrame | QuoteGrid,
        factors: pl.DataFrame | RatebookFactorContexts,
        *,
        title: str = "Frontier impact report",
        subtitle: str | None = None,
        labels: Mapping[str, str] | None = None,
        formats: Mapping[str, str] | None = None,
        report_columns: Sequence[str] | None = None,
        segment_columns: Sequence[str | Sequence[str]] | None = None,
        band_columns: Sequence[str] | None = None,
        scenario_display: ScenarioDisplay | None = None,
        sensitivity_grids: Mapping[str, pl.DataFrame] | None = None,
        sensitivity_note: str | None = None,
        ratios: Mapping[str, tuple[str, str]] | None = None,
        movement_band: float | None = None,
        projection: Projection | None = None,
        rate_label: str = "rate",
        point_names: Sequence[str] | None = None,
        point_artifacts: Mapping[int, str] | None = None,
        highlight_point: int | None = None,
        metadata: Mapping[str, str] | None = None,
        thresholds: Mapping[str, float | None] | None = None,
        top_n_levels: int = 25,
        top_n_factors: int = 25,
        generated_at: str | None = None,
    ) -> None:
        self.data: dict[str, Any] = _build_report(
            optimiser,
            frontier,
            grid_df,
            factors,
            title=title,
            subtitle=subtitle,
            labels=dict(labels or {}),
            formats=dict(formats or {}),
            report_columns=list(report_columns or []),
            segment_columns=list(segment_columns or []),
            band_columns=list(band_columns or []),
            scenario_display=scenario_display,
            sensitivity_grids=dict(sensitivity_grids or {}),
            sensitivity_note=sensitivity_note,
            ratios=dict(ratios or {}),
            movement_band=movement_band,
            projection=projection,
            rate_label=rate_label,
            point_names=None if point_names is None else list(point_names),
            point_artifacts=dict(point_artifacts or {}),
            highlight_point=highlight_point,
            metadata=dict(metadata or {}),
            thresholds=dict(thresholds or {}),
            top_n_levels=top_n_levels,
            top_n_factors=top_n_factors,
            generated_at=generated_at,
        )

    def to_html(self) -> str:
        """The report as one self-contained HTML page."""
        return frontier_report_html(self.data)

    def save(self, path: str | Path) -> Path:
        """Write :meth:`to_html` to ``path`` and return it."""
        target = Path(path)
        target.write_text(self.to_html(), encoding="utf-8")
        return target


def _resolve_thresholds(overrides: dict[str, float | None]) -> dict[str, float | None]:
    unknown = sorted(set(overrides) - set(_DEFAULT_THRESHOLDS))
    if unknown:
        raise ValueError(
            f"unknown threshold(s) {unknown}; known: {sorted(_DEFAULT_THRESHOLDS)}"
        )
    resolved: dict[str, float | None] = dict(_DEFAULT_THRESHOLDS)
    for key, value in overrides.items():
        if value is None:
            if key in _REQUIRED_THRESHOLDS:
                raise ValueError(
                    f"threshold {key!r} defines the audit and can't be None"
                )
        elif not math.isfinite(value) or value < 0:
            raise ValueError(f"threshold {key!r} must be finite and >= 0, got {value}")
        resolved[key] = value
    return resolved


def _check_format(owner: str, pattern: str) -> None:
    try:
        pattern.format(1.5)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"format {pattern!r} for {owner} can't format a number: {exc}"
        ) from exc


def _check_provenance(
    optimiser: RatebookOptimiser, frontier: RatebookFrontierResult
) -> None:
    provenance = frontier.provenance
    if provenance is None:
        raise ValueError(
            "frontier has no provenance: build it with RatebookOptimiser.frontier() "
            "from this version of price-contour"
        )
    attributes = (
        "objective",
        "constraints",
        "candidate_min",
        "candidate_max",
        "candidate_steps",
        "base_multiplier",
        "existing_rates",
        "max_cd_iterations",
        "cd_tolerance",
        "max_iter",
        "tolerance",
    )
    differs = [a for a in attributes if getattr(optimiser, a) != provenance[a]]
    if (
        optimiser.factor_columns is not None
        and optimiser.factor_columns != provenance["factor_specs"]
    ):
        differs.append("factor_columns")
    if differs:
        raise ValueError(f"frontier did not come from this optimiser: {differs} differ")


def _validate_options(
    optimiser: RatebookOptimiser,
    frontier: RatebookFrontierResult,
    *,
    scenario_display: ScenarioDisplay | None,
    point_names: list[str] | None,
    point_artifacts: dict[int, str],
    highlight_point: int | None,
    top_n_levels: int,
    top_n_factors: int,
    movement_band: float | None,
    projection: Projection | None,
    rate_label: str,
) -> None:
    if not isinstance(optimiser, RatebookOptimiser):
        raise TypeError("optimiser must be a RatebookOptimiser")
    if not isinstance(frontier, RatebookFrontierResult):
        raise TypeError("frontier must be a RatebookFrontierResult")
    ratio_names = _ratio_constraint_names(optimiser.constraints)
    if ratio_names:
        raise ValueError(
            f"FrontierReport does not support ratio constraints ({ratio_names}): a "
            f"ratio has no per-step sum to read from the grid"
        )
    _check_provenance(optimiser, frontier)
    n_points = frontier.n_points
    if point_names is not None and len(point_names) != n_points:
        raise ValueError(
            f"point_names has {len(point_names)} names for {n_points} points"
        )
    invalid = sorted(i for i in point_artifacts if not 0 <= i < n_points)
    if invalid:
        raise ValueError(f"point_artifacts refers to invalid point(s) {invalid}")
    if highlight_point is not None and not 0 <= highlight_point < n_points:
        raise ValueError(f"highlight_point {highlight_point} is not a point")
    if top_n_levels < 1 or top_n_factors < 1:
        raise ValueError("top_n_levels and top_n_factors must be at least 1")
    if scenario_display is not None:
        _check_format("scenario_display", scenario_display.format)
        if not all(math.isfinite(float(v)) for v in scenario_display.values):
            raise ValueError("scenario_display.values must be finite")
        if scenario_display.target is not None and not math.isfinite(
            float(scenario_display.target)
        ):
            raise ValueError("scenario_display.target must be finite")
    if movement_band is not None and (
        not math.isfinite(movement_band) or movement_band <= 0
    ):
        raise ValueError(f"movement_band must be finite and > 0, got {movement_band}")
    if projection is not None:
        spans = [projection.window_days, *projection.horizons]
        if not projection.horizons or not all(
            math.isfinite(float(v)) and float(v) > 0 for v in spans
        ):
            raise ValueError(
                "projection needs a positive window_days and at least one positive "
                "horizon"
            )
    if not rate_label:
        raise ValueError("rate_label must not be empty")


def _columns(
    optimiser: RatebookOptimiser,
    report_columns: list[str],
    labels: dict[str, str],
    formats: dict[str, str],
) -> list[dict[str, str]]:
    """The report's columns in canonical order."""
    roles = [(optimiser.objective, "objective")]
    roles += [(name, "constraint") for name in optimiser.constraints]
    taken = {name for name, _ in roles}
    repeated = sorted(set(report_columns) & taken)
    if repeated:
        raise ValueError(
            f"report_columns repeat the objective or a constraint: {repeated}"
        )
    if len(set(report_columns)) != len(report_columns):
        raise ValueError(f"report_columns has duplicates: {report_columns}")
    roles += [(name, "report") for name in report_columns]
    columns = []
    for name, role in roles:
        pattern = formats.get(name, _DEFAULT_FORMAT)
        _check_format(f"column {name!r}", pattern)
        columns.append(
            {
                "id": name,
                "role": role,
                "label": labels.get(name, name),
                "format": pattern,
            }
        )
    return columns


def _evaluation_grid(
    optimiser: RatebookOptimiser,
    grid_df: pl.DataFrame | QuoteGrid,
    report_columns: list[str],
) -> QuoteGrid:
    """One grid holding every column the report totals."""
    constraint_names = list(optimiser.constraints)
    if isinstance(grid_df, QuoteGrid):
        if report_columns:
            raise ValueError(
                "report_columns need the long frame: a QuoteGrid holds only the "
                "objective and constraints it was built with"
            )
        missing = [c for c in constraint_names if c not in grid_df.constraint_names]
        if missing:
            raise ValueError(f"QuoteGrid has no constraint column(s) {missing}")
        return grid_df
    if not isinstance(grid_df, pl.DataFrame):
        raise TypeError("grid_df must be a polars DataFrame or a QuoteGrid")
    needed = [optimiser.objective, *constraint_names, *report_columns]
    missing = [c for c in needed if c not in grid_df.columns]
    if missing:
        raise ValueError(f"grid_df has no column(s) {missing}")
    frame = grid_df
    if report_columns:
        frame = grid_df.with_columns(pl.col(report_columns).cast(pl.Float32))
    return build_grid(
        frame,
        constraint_columns=[*constraint_names, *report_columns],
        quote_id=optimiser.quote_id,
        scenario_index=optimiser.scenario_index,
        scenario_value=optimiser.scenario_value,
        objective=optimiser.objective,
    )


def _segment_specs(segment_columns: list[Any]) -> list[list[str]]:
    specs = []
    for entry in segment_columns:
        spec = [entry] if isinstance(entry, str) else list(entry)
        if not spec or not all(isinstance(c, str) for c in spec):
            raise ValueError(
                f"each of segment_columns must be a column or a list of columns, "
                f"got {entry!r}"
            )
        specs.append(spec)
    return specs


def _band_series(
    position: int,
    column: str,
    factors: pl.DataFrame,
    quote_id: str | None,
    grid: QuoteGrid,
) -> tuple[pl.Series, dict[str, float]]:
    """Ten equal-count bands of a column, ties broken by quote ID: each row's band
    label and each band's mean value."""
    series = factors[column]
    if not series.dtype.is_numeric():
        raise ValueError(f"band column {column!r} must be numeric, got {series.dtype}")
    values = series.cast(pl.Float64)
    if values.null_count() > 0 or not values.is_finite().all():
        raise ValueError(f"band column {column!r} must be finite with no nulls")
    if quote_id is not None:
        ids = factors[quote_id].cast(pl.String)
    elif factors.height == grid.n_quotes:
        ids = pl.Series("id", grid.quote_ids, dtype=pl.String)
    else:
        raise ValueError(
            f"factors has {factors.height} rows for {grid.n_quotes} quotes and no "
            f"quote ID column to align them by"
        )
    ranked = (
        pl.DataFrame({"value": values, "id": ids})
        .with_row_index("row")
        .sort(["value", "id"])
        .with_columns(
            pl.format("D{}", pl.int_range(pl.len()) * _N_BANDS // pl.len() + 1).alias(
                "band"
            )
        )
    )
    means = {
        str(band): math.fsum(group["value"].to_list()) / group.height
        for (band,), group in ranked.group_by("band", maintain_order=True)
    }
    return ranked.sort("row")["band"].alias(f"__band_{position}"), means


def _segment_contexts(
    optimiser: RatebookOptimiser,
    factors: pl.DataFrame | RatebookFactorContexts,
    segment_specs: list[list[str]],
    band_columns: list[str],
    grid: QuoteGrid,
) -> tuple[list[FactorContext], list[dict[str, float]]]:
    """Contexts for each segment and then each band, and each band's mean value."""
    if not segment_specs and not band_columns:
        return [], []
    if isinstance(factors, RatebookFactorContexts):
        name = "segment_columns" if segment_specs else "band_columns"
        raise ValueError(
            f"{name} need the factors frame: a RatebookFactorContexts holds only its "
            f"factors"
        )
    segment_read = [c for spec in segment_specs for c in spec]
    missing = [c for c in segment_read if c not in factors.columns]
    if missing:
        raise ValueError(f"segment_columns not on factors: {missing}")
    missing = [c for c in band_columns if c not in factors.columns]
    if missing:
        raise ValueError(f"band_columns not on factors: {missing}")
    quote_id = optimiser.quote_id if optimiser.quote_id in factors.columns else None
    bands = [
        _band_series(k, column, factors, quote_id, grid)
        for k, column in enumerate(band_columns)
    ]
    labels = [series for series, _ in bands]
    read = [*([] if quote_id is None else [quote_id]), *segment_read]
    frame = (
        select_native_columns(factors, read).with_columns(labels)
        if read
        else pl.DataFrame(labels)
    )
    contexts = RatebookFactorContexts.from_dataframe(
        frame,
        [*segment_specs, *[[series.name] for series in labels]],
        quote_id=quote_id,
        separator=FACTOR_SEPARATOR,
        expected_quote_ids=grid.quote_ids,
    )
    return contexts._factor_contexts_for_solver(), [means for _, means in bands]


def _check_sensitivity_grid(name: str, grid: QuoteGrid, other: QuoteGrid) -> None:
    if other.constraint_names != grid.constraint_names:
        raise ValueError(
            f"sensitivity grid {name!r} has columns {other.constraint_names}, not "
            f"{grid.constraint_names}"
        )
    if list(other.scenario_values) != list(grid.scenario_values):
        raise ValueError(
            f"sensitivity grid {name!r} has different scenario values from grid_df"
        )
    if other.quote_ids != grid.quote_ids:
        raise ValueError(
            f"sensitivity grid {name!r} must have the same quotes as grid_df, in the "
            f"same order"
        )


def _check_ratios(
    ratios: dict[str, tuple[str, str]],
    column_ids: list[str],
    formats: dict[str, str],
) -> None:
    for label, pair in ratios.items():
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"ratio {label!r} must be (numerator, denominator), got {pair!r}"
            )
        missing = [c for c in pair if c not in column_ids]
        if missing:
            raise ValueError(f"ratio {label!r} names unknown column(s) {missing}")
        _check_format(f"ratio {label!r}", formats.get(label, _DEFAULT_RATIO_FORMAT))


def _total_entry(baseline: float, point: float) -> dict[str, Any]:
    change = point - baseline
    undefined = baseline == 0
    return {
        "baseline": baseline,
        "point": point,
        "change": change,
        "pct": None if undefined else change / abs(baseline),
        "pct_reason": "baseline is zero" if undefined else None,
    }


def _display_label(label: str) -> str:
    """An interaction's level, readable: its parts are held with a control character."""
    return label.replace(FACTOR_SEPARATOR, " × ")


def _kind(spec: Mapping[str, Any]) -> str:
    return next(k for k in _DIRECTIONS if k in spec)


def _audit(
    constraints: Mapping[str, Mapping[str, Any]],
    bounds: Mapping[str, float],
    totals: Mapping[str, float],
    thresholds: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    rows = []
    for name, spec in constraints.items():
        bound, total = bounds[name], totals[name]
        slack = total - bound if _kind(spec).startswith("min") else bound - total
        scale = max(abs(bound), 1.0)
        rows.append(
            {
                "constraint": name,
                "bound": bound,
                "achieved": total,
                "slack": slack,
                "met": slack >= -float(thresholds["miss_tolerance"]) * scale,  # type: ignore[arg-type]
                "binding": abs(slack) <= float(thresholds["binding_tolerance"]) * scale,  # type: ignore[arg-type]
            }
        )
    return rows


def _named_percentiles(values: Sequence[float] | None) -> dict[str, float] | None:
    return None if values is None else dict(zip(_PERCENTILES, values))


def _band_edge(
    new: float, existing: float | None, candidate_min: float, candidate_max: float
) -> str | None:
    """Compared in f32, with the candidates actually searched (§19)."""
    start = 1.0 if existing is None else existing
    if new == _f32(start * candidate_min):
        return "min"
    if new == _f32(start * candidate_max):
        return "max"
    return None


def _mean_step(total: int, quotes: int) -> tuple[float | None, str | None]:
    return (None, "no quotes") if quotes == 0 else (total / quotes, None)


def _existing_of(
    existing: Mapping[str, float] | None, level: str, why_missing: str
) -> tuple[float | None, str | None]:
    return (None, why_missing) if existing is None else (existing[level], None)


def _rate_card(
    specs: list[list[str]],
    labels_by_factor: list[list[str]],
    tables: Mapping[str, Mapping[str, float]],
    groups: list[dict[str, Any]],
    column_ids: list[str],
    column_index: dict[str, int],
    existing_rates: Mapping[str, Mapping[str, float]] | None,
    why_no_existing: str,
    candidates: list[float],
) -> list[dict[str, Any]]:
    candidate_min, candidate_max = _f32(candidates[0]), _f32(candidates[-1])
    card = []
    for spec, labels, group in zip(specs, labels_by_factor, groups):
        name = ":".join(spec)
        table = tables[name]
        today = None if existing_rates is None else existing_rates[name]
        levels = []
        seen = set()
        for g, label in enumerate(labels):
            seen.add(label)
            quotes = group["n_quotes"][g]
            existing, reason = _existing_of(today, label, why_no_existing)
            new = table[label]
            mean, mean_reason = _mean_step(group["step_change_sum"][g], quotes)
            levels.append(
                {
                    "label": _display_label(label),
                    "existing": existing,
                    "existing_reason": reason,
                    "new": new,
                    "log_change": math.log(new)
                    if existing is None
                    else math.log(new / existing),
                    "quotes": quotes,
                    "mean_step_change": mean,
                    "mean_step_change_reason": mean_reason,
                    "impacts": {
                        c: group["point"][g][column_index[c]]
                        - group["baseline"][g][column_index[c]]
                        for c in column_ids
                    },
                    "band_edge": _band_edge(
                        new, existing, candidate_min, candidate_max
                    ),
                }
            )
        for label, new in table.items():
            if label in seen:
                continue
            # Carried forward from today's card: no quote has this level.
            existing, reason = _existing_of(today, label, why_no_existing)
            levels.append(
                {
                    "label": _display_label(label),
                    "existing": existing,
                    "existing_reason": reason,
                    "new": new,
                    "log_change": 0.0,
                    "quotes": 0,
                    "mean_step_change": None,
                    "mean_step_change_reason": "no quotes",
                    "impacts": dict.fromkeys(column_ids, 0.0),
                    "band_edge": None,
                }
            )
        card.append({"factor": name, "levels": levels})
    return card


def _segments(
    names: list[str],
    labels_by_column: list[list[str]],
    groups: list[dict[str, Any]],
    column_ids: list[str],
    column_index: dict[str, int],
    has_band: bool,
) -> list[dict[str, Any]]:
    out = []
    for name, labels, group in zip(names, labels_by_column, groups):
        levels = []
        for g, label in enumerate(labels):
            quotes = group["n_quotes"][g]
            mean, mean_reason = _mean_step(group["step_change_sum"][g], quotes)

            def per_quote(key: str) -> float | None:
                return None if quotes == 0 else group[key][g] / quotes

            levels.append(
                {
                    "label": _display_label(label),
                    "quotes": quotes,
                    "baseline": {
                        c: group["baseline"][g][column_index[c]] for c in column_ids
                    },
                    "point": {
                        c: group["point"][g][column_index[c]] for c in column_ids
                    },
                    "mean_step_change": mean,
                    "mean_step_change_reason": mean_reason,
                    "baseline_display_mean": per_quote("display_baseline_sum"),
                    "point_display_mean": per_quote("display_point_sum"),
                    "mean_abs_display_change": per_quote("abs_change_sum"),
                    "max_abs_display_change": None
                    if quotes == 0
                    else group["abs_change_max"][g],
                    "share_beyond_band": per_quote("n_beyond_band")
                    if has_band
                    else None,
                    "share_at_first": per_quote("n_at_first"),
                    "share_at_last": per_quote("n_at_last"),
                }
            )
        levels.sort(key=lambda row: (-row["quotes"], row["label"]))
        out.append({"column": name, "levels": levels})
    return out


def _bands(
    segments: list[dict[str, Any]], means: list[dict[str, float]]
) -> list[dict[str, Any]]:
    """Segments of band labels, in band order and each with the mean of its column."""
    for segment, band_means in zip(segments, means):
        segment["levels"].sort(key=lambda row: int(row["label"][1:]))
        for level in segment["levels"]:
            level["mean_value"] = band_means[level["label"]]
    return segments


def _directions(
    movement: Mapping[str, Any],
    deltas: list[dict[str, Any]],
    column_ids: list[str],
    column_index: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Per-quote means for the quotes moved up, down and not at all."""

    def entry(quotes: int, display_sum: float, key: str | None) -> dict[str, Any]:
        if quotes == 0:
            return {
                "quotes": 0,
                "mean_display_change": None,
                "mean_changes": dict.fromkeys(column_ids),
            }
        return {
            "quotes": quotes,
            "mean_display_change": display_sum / quotes,
            "mean_changes": {
                c: 0.0 if key is None else deltas[column_index[c]][key] / quotes
                for c in column_ids
            },
        }

    return {
        "up": entry(movement["up"], movement["display_moved_up"], "moved_up"),
        "down": entry(movement["down"], movement["display_moved_down"], "moved_down"),
        "unchanged": entry(movement["unchanged"], 0.0, None),
    }


def _ratios(
    ratios: dict[str, tuple[str, str]],
    formats: dict[str, str],
    totals: Mapping[str, Mapping[str, Any]],
    sensitivity: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    out = []
    for label, (numerator, denominator) in ratios.items():

        def value(top: float, bottom: float) -> tuple[float | None, str | None]:
            if bottom == 0:
                return None, f"{denominator} is zero"
            return top / bottom, None

        baseline, baseline_reason = value(
            totals[numerator]["baseline"], totals[denominator]["baseline"]
        )
        point, point_reason = value(
            totals[numerator]["point"], totals[denominator]["point"]
        )
        out.append(
            {
                "label": label,
                "format": formats.get(label, _DEFAULT_RATIO_FORMAT),
                "baseline": baseline,
                "baseline_reason": baseline_reason,
                "point": point,
                "point_reason": point_reason,
                "change": None
                if baseline is None or point is None
                else point - baseline,
                "sensitivity": {
                    name: value(scored[numerator], scored[denominator])[0]
                    for name, scored in sensitivity.items()
                },
            }
        )
    return out


def _scenario(
    aggregates: Mapping[str, Any],
    column_ids: list[str],
    column_index: dict[str, int],
) -> dict[str, Any]:
    scenario = aggregates["scenario"]
    return {
        "baseline_mean": scenario["baseline_mean"],
        "point_mean": scenario["point_mean"],
        "baseline_max": scenario["baseline_max"],
        "point_max": scenario["point_max"],
        "baseline_counts": scenario["baseline_counts"],
        "point_counts": scenario["point_counts"],
        "transitions": [
            {
                "from": start,
                "to": end,
                "quotes": quotes,
                "changes": {c: changes[column_index[c]] for c in column_ids},
            }
            for start, end, quotes, changes in scenario["transitions"]
        ],
    }


def _projected(
    projection: Projection | None,
    totals: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    if projection is None:
        return None
    out = []
    for horizon in projection.horizons:
        scale = float(horizon) / float(projection.window_days)
        out.append(
            {
                "horizon": float(horizon),
                "totals": {
                    c: {
                        "baseline": entry["baseline"] * scale,
                        "point": entry["point"] * scale,
                        "change": entry["change"] * scale,
                    }
                    for c, entry in totals.items()
                },
            }
        )
    return out


def _history_by_point(
    frontier: RatebookFrontierResult, constraint_names: list[str]
) -> dict[int, list[dict[str, Any]]]:
    history = frontier.point_history
    if history is None:
        return {}
    by_point: dict[int, list[dict[str, Any]]] = {}
    for row in history.iter_rows(named=True):
        by_point.setdefault(row["point_index"], []).append(
            {
                "cd_iteration": row["cd_iteration"],
                "factor": row["factor"],
                "factor_index": row["factor_index"],
                "total_objective": row["total_objective"],
                "totals": {c: row[f"total_{c}"] for c in constraint_names},
                "inner_iterations": row["inner_iterations"],
                "inner_converged": row["inner_converged"],
            }
        )
    return by_point


def _mark_dominance(
    points: list[dict[str, Any]],
    objective: str,
    constraints: Mapping[str, Mapping[str, Any]],
) -> None:
    def better_or_equal(
        a: dict[str, Any], b: dict[str, Any], column: str, higher: bool
    ) -> bool:
        x, y = a["totals"][column]["point"], b["totals"][column]["point"]
        return x >= y if higher else x <= y

    axes = [(objective, True)]
    axes += [(c, _kind(spec).startswith("min")) for c, spec in constraints.items()]
    for point in points:
        point["dominated"] = False
        for other in points:
            if other is point:
                continue
            if all(better_or_equal(other, point, c, up) for c, up in axes) and not all(
                better_or_equal(point, other, c, up) for c, up in axes
            ):
                point["dominated"] = True
                break


def _slices(
    points: list[dict[str, Any]], swept: list[str], objective: str
) -> list[dict[str, Any]]:
    """Points sharing every other swept threshold, along each swept dimension."""
    slices = []
    for dimension in swept:
        others = [c for c in swept if c != dimension]
        groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for point in points:
            key = tuple(point["thresholds"][c] for c in others)
            groups.setdefault(key, []).append(point)
        for key, members in groups.items():
            members.sort(key=lambda p: (p["thresholds"][dimension], p["index"]))
            marginal = []
            for a, b in zip(members, members[1:]):
                delta_c = (
                    b["totals"][dimension]["point"] - a["totals"][dimension]["point"]
                )
                delta_o = (
                    b["totals"][objective]["point"] - a["totals"][objective]["point"]
                )
                undefined = delta_c == 0
                marginal.append(
                    {
                        "from": a["index"],
                        "to": b["index"],
                        "value": None if undefined else delta_o / delta_c,
                        "value_reason": "no change in the constraint"
                        if undefined
                        else None,
                    }
                )
            slices.append(
                {
                    "dimension": dimension,
                    "fixed": dict(zip(others, key)),
                    "points": [p["index"] for p in members],
                    "marginal": marginal,
                }
            )
    return slices


def _rate_changes(
    points: list[dict[str, Any]], top_n_levels: int
) -> list[dict[str, Any]]:
    """Per factor, the levels whose new rate varies most across points."""
    out = []
    for f, first in enumerate(points[0]["rate_card"]):
        rows = []
        for g, level in enumerate(first["levels"]):
            rates = [p["rate_card"][f]["levels"][g]["new"] for p in points]
            rows.append(
                (
                    -(math.log(max(rates)) - math.log(min(rates))),
                    g,
                    level["label"],
                    rates,
                )
            )
        rows.sort(key=lambda row: (row[0], row[1]))
        out.append(
            {
                "factor": first["factor"],
                "levels": [{"label": r[2], "rates": r[3]} for r in rows[:top_n_levels]],
            }
        )
    return out


def _flag(
    check: str,
    status: str,
    point: int | None,
    value: float | None,
    threshold: float | None,
    message: str,
) -> dict[str, Any]:
    return {
        "check": check,
        "status": status,
        "point": point,
        "value": value,
        "threshold": threshold,
        "message": message,
    }


def _review_flags(
    points: list[dict[str, Any]],
    thresholds: Mapping[str, float | None],
    n_quotes: int,
    baseline_clamped: int,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    miss_tolerance = float(thresholds["miss_tolerance"])  # type: ignore[arg-type]
    limit = thresholds["baseline_clamped_share"]
    share = baseline_clamped / n_quotes
    if limit is not None and share > limit:
        flags.append(
            _flag(
                "baseline_clamped_share",
                "warning",
                None,
                share,
                limit,
                f"{share:.1%} of quotes start past a grid end, so their starting "
                f"price is off the grid",
            )
        )
    for point in points:
        i, name = point["index"], point["name"]
        for row in point["audit"]:
            if not row["met"]:
                shortfall = -row["slack"] / max(abs(row["bound"]), 1.0)
                flags.append(
                    _flag(
                        "constraint_missed",
                        "critical",
                        i,
                        shortfall,
                        miss_tolerance,
                        f"{name} misses {row['constraint']} by {shortfall:.2%} of its bound",
                    )
                )
        if not point["health"]["converged"]:
            flags.append(
                _flag(
                    "not_converged",
                    "info",
                    i,
                    None,
                    None,
                    f"{name} stopped at the coordinate-descent iteration limit",
                )
            )
        limit = thresholds["clamped_share"]
        clamped = (
            point["health"]["clamped_low"] + point["health"]["clamped_high"]
        ) / n_quotes
        if limit is not None and clamped > limit:
            flags.append(
                _flag(
                    "clamped_share",
                    "warning",
                    i,
                    clamped,
                    limit,
                    f"{clamped:.1%} of {name}'s quotes are clamped at a grid end",
                )
            )
        limit = thresholds["band_edge_share"]
        levels = [
            lv
            for card in point["rate_card"]
            for lv in card["levels"]
            if lv["quotes"] > 0
        ]
        if limit is not None and levels:
            at_edge = sum(lv["band_edge"] is not None for lv in levels) / len(levels)
            if at_edge > limit:
                flags.append(
                    _flag(
                        "band_edge_share",
                        "warning",
                        i,
                        at_edge,
                        limit,
                        f"{at_edge:.1%} of {name}'s levels are at the edge of the band",
                    )
                )
        if point["dominated"]:
            flags.append(
                _flag(
                    "dominated",
                    "info",
                    i,
                    None,
                    None,
                    f"{name} is dominated by another point",
                )
            )
    order = {"critical": 0, "warning": 1, "info": 2}
    flags.sort(
        key=lambda f: (order[f["status"]], -1 if f["point"] is None else f["point"])
    )
    return flags


def _default_point(
    points: list[dict[str, Any]],
    highlight_point: int | None,
    constraint_names: list[str],
) -> int:
    if highlight_point is not None:
        return highlight_point

    def distance(point: dict[str, Any]) -> float:
        total = 0.0
        for c in constraint_names:
            entry = point["totals"][c]
            total += (entry["change"] / max(abs(entry["baseline"]), 1.0)) ** 2
        return total

    eligible = [
        p
        for p in points
        if not p["dominated"] and all(row["met"] for row in p["audit"])
    ]
    if not eligible:
        return 0
    return min(eligible, key=lambda p: (distance(p), p["index"]))["index"]


def _check_totals(
    index: int,
    aggregates: Mapping[str, Any],
    row: Mapping[str, Any],
    objective: str,
    constraint_positions: Mapping[str, int],
) -> None:
    """The contract: every point's totals equal the frontier's, exactly."""
    expected = [(objective, row["total_objective"], aggregates["total_objective"])]
    expected += [
        (c, row[f"total_{c}"], aggregates["total_constraints"][k])
        for c, k in constraint_positions.items()
    ]
    for column, frontier_total, report_total in expected:
        if frontier_total != report_total:
            raise ValueError(
                f"point {index}: the report totals {column!r} as {report_total!r} but "
                f"the frontier recorded {frontier_total!r}. grid_df or factors is not "
                f"the data this frontier was swept on."
            )


def _build_report(
    optimiser: RatebookOptimiser,
    frontier: RatebookFrontierResult,
    grid_df: pl.DataFrame | QuoteGrid,
    factors: pl.DataFrame | RatebookFactorContexts,
    *,
    title: str,
    subtitle: str | None,
    labels: dict[str, str],
    formats: dict[str, str],
    report_columns: list[str],
    segment_columns: list[Any],
    band_columns: list[str],
    scenario_display: ScenarioDisplay | None,
    sensitivity_grids: dict[str, pl.DataFrame],
    sensitivity_note: str | None,
    ratios: dict[str, tuple[str, str]],
    movement_band: float | None,
    projection: Projection | None,
    rate_label: str,
    point_names: list[str] | None,
    point_artifacts: dict[int, str],
    highlight_point: int | None,
    metadata: dict[str, str],
    thresholds: dict[str, float | None],
    top_n_levels: int,
    top_n_factors: int,
    generated_at: str | None,
) -> dict[str, Any]:
    _validate_options(
        optimiser,
        frontier,
        scenario_display=scenario_display,
        point_names=point_names,
        point_artifacts=point_artifacts,
        highlight_point=highlight_point,
        top_n_levels=top_n_levels,
        top_n_factors=top_n_factors,
        movement_band=movement_band,
        projection=projection,
        rate_label=rate_label,
    )
    resolved_thresholds = _resolve_thresholds(thresholds)
    columns = _columns(optimiser, report_columns, labels, formats)
    column_ids = [c["id"] for c in columns]
    _check_ratios(ratios, column_ids, formats)
    segment_specs = _segment_specs(segment_columns)
    constraint_names = list(optimiser.constraints)
    provenance = frontier.provenance
    assert provenance is not None  # checked by _validate_options
    specs: list[list[str]] = provenance["factor_specs"]
    swept: list[str] = provenance["swept"]
    existing_rates = optimiser.existing_rates

    # Every input check runs before the (expensive) grid build.
    _validate_base_multiplier_input(
        optimiser.base_multiplier, factors, specs, optimiser.quote_id
    )
    _validate_existing_rates_input(existing_rates, factors, specs)
    if isinstance(factors, RatebookFactorContexts) and factors.factor_specs != specs:
        raise ValueError(
            f"RatebookFactorContexts.factor_specs {factors.factor_specs} differ from "
            f"the frontier's {specs}"
        )

    for name, frame in sensitivity_grids.items():
        # A QuoteGrid can't say which column its objective came from.
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(
                f"sensitivity grid {name!r} must be a polars DataFrame, so each column "
                f"is read by name"
            )
        needed = [optimiser.objective, *constraint_names, *report_columns]
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            raise ValueError(f"sensitivity grid {name!r} has no column(s) {missing}")

    grid = _evaluation_grid(optimiser, grid_df, report_columns)
    if scenario_display is not None and len(scenario_display.values) != grid.n_steps:
        raise ValueError(
            f"scenario_display.values has {len(scenario_display.values)} values for "
            f"{grid.n_steps} steps"
        )
    sensitivity_scored = []
    for name, frame in sensitivity_grids.items():
        other = _evaluation_grid(optimiser, frame, report_columns)
        _check_sensitivity_grid(name, grid, other)
        sensitivity_scored.append(other)

    contexts_obj = _resolve_factor_contexts(
        factors,
        specs,
        grid=grid,
        quote_id_col=optimiser.quote_id,
        base_multiplier=optimiser.base_multiplier,
    )
    contexts = contexts_obj._factor_contexts_for_solver()
    segments, band_means = _segment_contexts(
        optimiser, factors, segment_specs, band_columns, grid
    )
    n_segments = len(segment_specs)
    segment_names = [":".join(spec) for spec in segment_specs]

    tables = [frontier.point_factor_tables(i) for i in range(frontier.n_points)]
    aggregates_by_point = report_points_py(
        grid,
        contexts,
        [*contexts, *segments],
        [
            _factor_values_for_contexts(specs, contexts, t, carried=existing_rates)
            for t in tables
        ],
        base_contexts=contexts_obj,
        existing_values=_existing_values(specs, contexts, existing_rates),
        display_values=(
            None
            if scenario_display is None
            else [float(v) for v in scenario_display.values]
        ),
        movement_band=movement_band,
        sensitivity_grids=sensitivity_scored,
        n_bins=_HISTOGRAM_BINS,
    )

    constraint_positions = {
        name: position
        for position, name in enumerate(grid.constraint_names)
        if name in optimiser.constraints
    }
    column_index = {optimiser.objective: 0}
    column_index.update({name: k + 1 for k, name in enumerate(grid.constraint_names)})

    labels_by_factor = [ctx.group_labels for ctx in contexts]
    labels_by_segment = [ctx.group_labels for ctx in segments]
    candidates = optimiser._build_candidates()
    why_no_existing = (
        "the starting point is per quote, not per level"
        if optimiser.base_multiplier is not None
        else "no existing rates were supplied"
    )
    history = _history_by_point(frontier, constraint_names)

    points: list[dict[str, Any]] = []
    for index, aggregates in enumerate(aggregates_by_point):
        row = frontier.points.row(index, named=True)
        _check_totals(index, aggregates, row, optimiser.objective, constraint_positions)
        name = f"Point {index}" if point_names is None else point_names[index]

        def total_of(column: str, key: str, objective_key: str) -> float:
            if column == optimiser.objective:
                return aggregates[objective_key]
            return aggregates[key][column_index[column] - 1]

        totals = {
            c: _total_entry(
                total_of(c, "baseline_constraints", "baseline_objective"),
                total_of(c, "total_constraints", "total_objective"),
            )
            for c in column_ids
        }
        bounds = {c: row[f"bound_{c}"] for c in constraint_names}
        groups = aggregates["groups"]
        movement = aggregates["movement"]
        sensitivity = {
            name: {c: aggregates["sensitivity"][k][column_index[c]] for c in column_ids}
            for k, name in enumerate(sensitivity_grids)
        }
        segment_groups = groups[len(contexts) :]
        points.append(
            {
                "index": index,
                "name": name,
                "artifact": point_artifacts.get(index),
                "highlighted": index == highlight_point,
                "dominated": False,
                "thresholds": {c: row[f"threshold_{c}"] for c in constraint_names},
                "bounds": bounds,
                "totals": totals,
                "audit": _audit(
                    optimiser.constraints,
                    bounds,
                    {c: totals[c]["point"] for c in constraint_names},
                    resolved_thresholds,
                ),
                "health": {
                    "converged": row["converged"],
                    "iterations": row["iterations"],
                    "lambdas": {c: row[f"lambda_{c}"] for c in constraint_names},
                    "clamp_rate": row["clamp_rate"],
                    "clamped_low": row["n_quotes_clamped_low"],
                    "clamped_high": row["n_quotes_clamped_high"],
                },
                "sensitivity": sensitivity,
                "ratios": _ratios(ratios, formats, totals, sensitivity),
                "projection": _projected(projection, totals),
                "scenario": _scenario(aggregates, column_ids, column_index),
                "movement": {
                    "up": movement["up"],
                    "down": movement["down"],
                    "unchanged": movement["unchanged"],
                    "directions": _directions(
                        movement, aggregates["deltas"], column_ids, column_index
                    ),
                    "histogram": {
                        "changes": movement["changes"],
                        "counts": movement["counts"],
                    },
                    "percentiles_steps": _named_percentiles(
                        movement["percentiles_steps"]
                    ),
                    "percentiles_display": _named_percentiles(
                        movement["percentiles_display"]
                    ),
                },
                "deltas": {
                    c: {
                        "percentiles": _named_percentiles(
                            aggregates["deltas"][column_index[c]]["percentiles"]
                        ),
                        "histogram": {
                            "edges": aggregates["deltas"][column_index[c]]["edges"],
                            "counts": aggregates["deltas"][column_index[c]]["counts"],
                            "below": aggregates["deltas"][column_index[c]]["below"],
                            "above": aggregates["deltas"][column_index[c]]["above"],
                        },
                        "moved_up": aggregates["deltas"][column_index[c]]["moved_up"],
                        "moved_down": aggregates["deltas"][column_index[c]][
                            "moved_down"
                        ],
                    }
                    for c in column_ids
                },
                "rate_card": _rate_card(
                    specs,
                    labels_by_factor,
                    tables[index],
                    groups[: len(contexts)],
                    column_ids,
                    column_index,
                    existing_rates,
                    why_no_existing,
                    candidates,
                ),
                "segments": _segments(
                    segment_names,
                    labels_by_segment[:n_segments],
                    segment_groups[:n_segments],
                    column_ids,
                    column_index,
                    movement_band is not None,
                ),
                "bands": _bands(
                    _segments(
                        band_columns,
                        labels_by_segment[n_segments:],
                        segment_groups[n_segments:],
                        column_ids,
                        column_index,
                        movement_band is not None,
                    ),
                    band_means,
                ),
                "history": history.get(index)
                if frontier.point_history is not None
                else None,
            }
        )

    _mark_dominance(points, optimiser.objective, optimiser.constraints)
    first = aggregates_by_point[0]
    baseline_clamped = first["baseline_clamped_low"] + first["baseline_clamped_high"]

    import price_contour  # the package imports this module, so not at the top

    record: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": title,
        "subtitle": subtitle,
        "default_point": _default_point(points, highlight_point, constraint_names),
        "library": {"price_contour_version": price_contour.__version__},
        "run": {
            "objective": optimiser.objective,
            "constraints": [
                {"name": c, "kind": _kind(spec)}
                for c, spec in optimiser.constraints.items()
            ],
            "swept": swept,
            "n_quotes": grid.n_quotes,
            "n_steps": grid.n_steps,
            "scenario_values": [float(v) for v in grid.scenario_values],
            "scenario_display": (
                None
                if scenario_display is None
                else {
                    "label": scenario_display.label,
                    "values": [float(v) for v in scenario_display.values],
                    "format": scenario_display.format,
                    "target": None
                    if scenario_display.target is None
                    else float(scenario_display.target),
                }
            ),
            "candidates": {
                "min": optimiser.candidate_min,
                "max": optimiser.candidate_max,
                "steps": optimiser.candidate_steps,
                "values": candidates,
            },
            "has_existing_rates": existing_rates is not None,
            "base_multiplier": optimiser.base_multiplier,
            "solver": {
                "max_cd_iterations": provenance["max_cd_iterations"],
                "cd_tolerance": provenance["cd_tolerance"],
                "max_iter": provenance["max_iter"],
                "tolerance": provenance["tolerance"],
            },
            "sensitivity_names": list(sensitivity_grids),
            "sensitivity_note": sensitivity_note,
            "movement_band": movement_band,
            "rate_label": rate_label,
            "projection": None
            if projection is None
            else {
                "window_days": float(projection.window_days),
                "horizons": [float(h) for h in projection.horizons],
            },
            "metadata": [[str(k), str(v)] for k, v in metadata.items()],
        },
        "columns": columns,
        "thresholds": resolved_thresholds,
        "display": {"top_n_levels": top_n_levels, "top_n_factors": top_n_factors},
        "baseline": {
            "totals": {c: points[0]["totals"][c]["baseline"] for c in column_ids},
            "clamped_low": first["baseline_clamped_low"],
            "clamped_high": first["baseline_clamped_high"],
        },
        "points": points,
        "comparisons": {
            "slices": _slices(points, swept, optimiser.objective),
            "rate_changes": _rate_changes(points, top_n_levels),
        },
        "flags": _review_flags(
            points, resolved_thresholds, grid.n_quotes, baseline_clamped
        ),
    }
    try:
        json.dumps(record, allow_nan=False)
    except ValueError as exc:
        raise ValueError(
            f"the report holds a non-finite number, so the data has NaN or infinite "
            f"values: {exc}"
        ) from exc
    return record
