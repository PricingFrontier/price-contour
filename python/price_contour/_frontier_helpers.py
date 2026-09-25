"""Shared helpers for Python-orchestrated frontier sweeps.

Both :class:`OnlineOptimiser` and :class:`RatebookOptimiser` orchestrate
frontier sweeps in Python when the Rust fast-path can't apply (ratio
constraints, mixed swept/unswept axes, etc.). The two were near-duplicates
of each other:

* identical linspace + cartesian product over swept axes;
* identical ``threshold_<name>`` / ``total_<name>`` / ``lambda_<name>``
  emission;
* identical sv_* distribution stats from the per-quote optimal scenario
  values;
* identical ordering / warm-start logic.

This module hosts the orchestrator and the SV-stats helper so both
call sites parameterise via callbacks rather than copy-pasting the
loop body. The optimiser-specific bits (per-point solve, row composition
for ratebook's CD-iteration count vs online's iteration count) are
hoisted into ``solve_one`` / ``compose_row`` callables.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import polars as pl


# Scenario-value distribution columns of the online frontier.
_SV_COLUMNS: tuple[str, ...] = (
    "sv_mean",
    "sv_std",
    "sv_min",
    "sv_p5",
    "sv_p25",
    "sv_median",
    "sv_p75",
    "sv_p95",
    "sv_max",
    "sv_pct_increase",
    "sv_pct_decrease",
)


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    """Equally-spaced ``n`` points in ``[lo, hi]`` (mirrors the Rust helper).

    Single-point degenerate case (``n <= 1``) returns ``[lo]``; matching
    the Rust ``linspace`` so ``n_points_per_dim=1`` produces exactly one
    point per axis. ``lo == hi`` returns ``n`` copies of the value.
    """
    if n <= 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def _cartesian_product(axes: list[list[float]]) -> list[list[float]]:
    """Cartesian product of per-axis grids (mirrors the Rust helper)."""
    if not axes:
        return [[]]
    out: list[list[float]] = [[]]
    for axis in axes:
        new_out: list[list[float]] = []
        for existing in out:
            for v in axis:
                new_out.append([*existing, v])
        out = new_out
    return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile (mirrors the Rust helper)."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("percentile of an empty list is undefined")
    if n == 1:
        return sorted_vals[0]
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = min(int(math.ceil(pos)), n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _compute_sv_stats_from_dataframe(out_df: pl.DataFrame) -> dict[str, float]:
    """Compute per-quote scenario-value distribution stats.

    Reads the ``optimal_scenario_value`` column from a solve result's
    output DataFrame and returns the ``sv_*`` summary metrics emitted
    by the Rust frontier (``sv_mean``, ``sv_std``, ``sv_min``,
    ``sv_p5`` / ``p25`` / ``median`` / ``p75`` / ``p95``, ``sv_max``,
    ``sv_pct_increase``, ``sv_pct_decrease``). A result with no quotes
    has no distribution and raises.
    """
    sv_series = out_df["optimal_scenario_value"]
    n = sv_series.len()
    if n == 0:
        raise ValueError("scenario-value statistics need at least one quote")
    vals: list[float] = [float(v) for v in sv_series.to_list()]
    sum_v = sum(vals)
    mean = sum_v / n
    variance = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(variance)
    n_inc = sum(1 for v in vals if v > 1.0)
    n_dec = sum(1 for v in vals if v < 1.0)
    sorted_vals = sorted(vals)
    return {
        "sv_mean": mean,
        "sv_std": std,
        "sv_min": sorted_vals[0],
        "sv_p5": _percentile(sorted_vals, 0.05),
        "sv_p25": _percentile(sorted_vals, 0.25),
        "sv_median": _percentile(sorted_vals, 0.50),
        "sv_p75": _percentile(sorted_vals, 0.75),
        "sv_p95": _percentile(sorted_vals, 0.95),
        "sv_max": sorted_vals[-1],
        "sv_pct_increase": n_inc / n,
        "sv_pct_decrease": n_dec / n,
    }


class _PythonFrontierResult:
    """Duck-typed FrontierResult for the Python-side sweep.

    Satisfies the minimal ``FrontierResultLike`` contract (``points`` +
    ``n_points``) so it slots into ``frontier_summary`` and any other
    consumer that talks to the protocol. Mirrors the Rust
    ``FrontierResult`` interface (``n_converged``, ``constraint_names``)
    for parity even where the test contract doesn't pin them.
    """

    __slots__ = ("_points", "_n_points", "_constraint_names")

    def __init__(
        self, *, points_df: pl.DataFrame, n_points: int, constraint_names: list[str]
    ) -> None:
        self._points = points_df
        self._n_points = int(n_points)
        self._constraint_names = list(constraint_names)

    @property
    def points(self) -> pl.DataFrame:
        return self._points

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def n_converged(self) -> int:
        return int(self._points["converged"].sum())

    @property
    def constraint_names(self) -> list[str]:
        return list(self._constraint_names)


def frontier_points_schema(
    mode: str, constraint_names: list[str]
) -> dict[str, pl.DataType]:
    """Exact ``{column: dtype}`` of a frontier's ``points`` (DESIGN_DECISIONS
    §13.12), in column order.

    ``mode`` is ``"online"`` (the Rust sweep and the Python-orchestrated
    sweep emit the same schema) or ``"ratebook"``.
    """
    if mode not in ("online", "ratebook"):
        raise ValueError(f"mode must be 'online' or 'ratebook', got {mode!r}")
    schema: dict[str, pl.DataType] = {}
    for prefix in ("threshold", "bound"):
        for name in constraint_names:
            schema[f"{prefix}_{name}"] = pl.Float64()
    schema["total_objective"] = pl.Float64()
    for prefix in ("total", "lambda"):
        for name in constraint_names:
            schema[f"{prefix}_{name}"] = pl.Float64()
    schema["iterations"] = pl.Int64()
    schema["converged"] = pl.Boolean()
    if mode == "online":
        schema["solver_path"] = pl.String()
        schema["non_convergence_reason"] = pl.String()
        for col in _SV_COLUMNS:
            schema[col] = pl.Float64()
    else:
        schema["clamp_rate"] = pl.Float64()
        schema["n_quotes_clamped_low"] = pl.Int64()
        schema["n_quotes_clamped_high"] = pl.Int64()
    return schema


def _build_points_dataframe(
    rows: list[dict[str, Any]], constraint_names: list[str]
) -> pl.DataFrame:
    """Assemble an online frontier's ``points`` in the canonical schema
    (:func:`frontier_points_schema`). Every row must carry exactly the
    schema's columns."""
    schema = frontier_points_schema("online", constraint_names)
    for row in rows:
        if set(row) != set(schema):
            missing = sorted(set(schema) - set(row))
            extra = sorted(set(row) - set(schema))
            raise ValueError(
                f"frontier row does not match the points schema: "
                f"missing {missing}, unexpected {extra}"
            )
    return pl.DataFrame(rows, schema=schema)


def _python_frontier_orchestrator(
    *,
    constraint_names: list[str],
    swept_names: list[str],
    threshold_ranges: dict[str, tuple[float, float]],
    unswept_thresholds: dict[str, float],
    n_points_per_dim: int,
    max_total_points: int,
    initial_lambdas: dict[str, float] | None,
    solve_one: Callable[
        [list[float], dict[str, float] | None], tuple[Any, dict[str, float]]
    ],
    compose_row: Callable[[Any, list[float], list[str]], dict[str, Any]],
) -> _PythonFrontierResult:
    """Drive a Python-side frontier sweep over the swept axes.

    ``solve_one(combo, prev_lambdas)`` runs the inner solve at a single
    threshold combination and returns ``(result, lambdas_for_warm_start)``.
    ``compose_row(result, combo, swept_names)`` produces the per-row dict
    (with ``threshold_<name>``, ``total_*``, ``lambda_*``, iterations,
    converged) for that point. Both callbacks own the optimiser-specific
    composition; this helper owns the cartesian product, max-points
    rejection, sequential warm-start traversal, and ``points`` DataFrame
    assembly.
    """
    n = max(int(n_points_per_dim), 1)
    per_axis: list[list[float]] = []
    for name in swept_names:
        lo, hi = threshold_ranges[name]
        per_axis.append(_linspace(float(lo), float(hi), n))

    combos = _cartesian_product(per_axis)
    if len(combos) > max_total_points:
        raise ValueError(
            f"Frontier would generate {len(combos)} points (exceeds "
            f"max_total_points={max_total_points}). Reduce "
            f"n_points_per_dim or increase max_total_points."
        )

    rows: list[dict[str, Any]] = []
    prev_lambdas: dict[str, float] | None = (
        dict(initial_lambdas) if initial_lambdas else None
    )

    for combo in combos:
        result, next_lambdas = solve_one(combo, prev_lambdas)
        prev_lambdas = next_lambdas
        row = compose_row(result, combo, swept_names)
        rows.append(row)

    # Inject the unswept axes' constant threshold values into each row
    # if compose_row hasn't already (the orchestrator owns the canonical
    # column order so unswept-axis emission is centralised here).
    for row in rows:
        for name in constraint_names:
            key = f"threshold_{name}"
            if key not in row and name in unswept_thresholds:
                row[key] = float(unswept_thresholds[name])

    points_df = _build_points_dataframe(rows, constraint_names)
    return _PythonFrontierResult(
        points_df=points_df, n_points=len(rows), constraint_names=constraint_names
    )
