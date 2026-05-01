"""Standalone price-contour frontier repro extracted from Haute.

Run from this directory or pass the parquet path as the first argument:

    python reproduce_frontier_issue.py haute_price_contour_frontier_repro.parquet

The script intentionally imports only polars and price_contour. It does not
import Haute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from price_contour import OnlineOptimiser, apply_from_grid, build_grid_from_parquet

QUOTE_ID = "quote_id"
SCENARIO_INDEX = "scenario_index"
SCENARIO_VALUE = "premium_multiplier"
OBJECTIVE = "expected_margin"
CONSTRAINT = "conversion_prediction"
CONSTRAINTS = {CONSTRAINT: {"min": 0.0}}
FRONTIER_RANGE = (189.03746032714844, 99695.75)
N_POINTS = 15
TOLERANCE = 1e-6


def emit(label: str, payload: object) -> None:
    print(f"{label} " + json.dumps(payload, sort_keys=True))


def scenario_envelope(path: Path) -> dict[str, object]:
    lf = pl.scan_parquet(path)
    stats = lf.select(
        pl.len().alias("rows"),
        pl.col(QUOTE_ID).n_unique().alias("n_quotes"),
        pl.col(SCENARIO_INDEX).n_unique().alias("n_scenarios"),
        pl.col(SCENARIO_VALUE).min().alias("scenario_value_min"),
        pl.col(SCENARIO_VALUE).max().alias("scenario_value_max"),
        pl.col(OBJECTIVE).null_count().alias("objective_nulls"),
        pl.col(CONSTRAINT).null_count().alias("constraint_nulls"),
    ).collect().to_dicts()[0]
    envelope = (
        lf.group_by(QUOTE_ID)
        .agg(
            pl.col(CONSTRAINT).min().alias("min_constraint"),
            pl.col(CONSTRAINT).max().alias("max_constraint"),
            pl.len().alias("steps_per_quote"),
        )
        .select(
            pl.col("min_constraint").sum().alias("sum_quote_min_constraint"),
            pl.col("max_constraint").sum().alias("sum_quote_max_constraint"),
            pl.col("steps_per_quote").min().alias("min_steps_per_quote"),
            pl.col("steps_per_quote").max().alias("max_steps_per_quote"),
        )
        .collect()
        .to_dicts()[0]
    )
    return {"stats": stats, "envelope": envelope}


def frontier_summary(grid, *, max_iter: int) -> dict[str, object]:
    solver = OnlineOptimiser(
        objective=OBJECTIVE,
        constraints=CONSTRAINTS,
        max_iter=max_iter,
        tolerance=TOLERANCE,
    )
    base = solver.solve(grid)
    frontier = solver.frontier(
        grid,
        threshold_ranges={CONSTRAINT: FRONTIER_RANGE},
        n_points_per_dim=N_POINTS,
        initial_lambdas=base.lambdas,
    )
    points = frontier.points.to_dicts()
    selected = []
    for idx in (0, 3, 7, 10, 12, 13, 14):
        point = points[idx]
        target = point[f"threshold_{CONSTRAINT}"]
        actual = point[f"total_{CONSTRAINT}"]
        selected.append(
            {
                "idx": idx,
                "target": target,
                "actual": actual,
                "gap_actual_minus_target": actual - target,
                "lambda": point[f"lambda_{CONSTRAINT}"],
                "iterations": point["iterations"],
                "converged": point["converged"],
            }
        )
    return {
        "max_iter": max_iter,
        "base": {
            "converged": base.converged,
            "iterations": base.iterations,
            "lambda": base.lambdas[CONSTRAINT],
            "actual": base.total_constraints[CONSTRAINT],
            "objective": base.total_objective,
        },
        "n_points": len(points),
        "n_converged": sum(1 for point in points if point["converged"]),
        "selected_points": selected,
        "last_point": selected[-1],
    }


def total_at_lambda(grid, lam: float) -> float:
    result = apply_from_grid(grid, {CONSTRAINT: float(lam)}, CONSTRAINTS)
    return result.total_constraints[CONSTRAINT]


def bisection_lambda_for_target(grid, target: float) -> dict[str, object]:
    base_total = total_at_lambda(grid, 0.0)
    if base_total >= target:
        return {
            "target": target,
            "lambda_by_bisection": 0.0,
            "actual_at_lambda": base_total,
            "gap_actual_minus_target": base_total - target,
            "reachable": True,
        }

    high = 1.0
    high_total = total_at_lambda(grid, high)
    while high_total < target and high < 262_144.0:
        high *= 2.0
        high_total = total_at_lambda(grid, high)

    if high_total < target:
        return {
            "target": target,
            "lambda_by_bisection": high,
            "actual_at_lambda": high_total,
            "gap_actual_minus_target": high_total - target,
            "reachable": False,
        }

    low = 0.0
    for _ in range(50):
        mid = (low + high) / 2.0
        mid_total = total_at_lambda(grid, mid)
        if mid_total >= target:
            high = mid
            high_total = mid_total
        else:
            low = mid

    return {
        "target": target,
        "lambda_by_bisection": high,
        "actual_at_lambda": high_total,
        "gap_actual_minus_target": high_total - target,
        "reachable": True,
    }


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "haute_price_contour_frontier_repro.parquet"
    )
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    emit("SCENARIO_ENVELOPE_JSON", scenario_envelope(path))

    grid = build_grid_from_parquet(
        str(path),
        [CONSTRAINT],
        quote_id=QUOTE_ID,
        scenario_index=SCENARIO_INDEX,
        scenario_value=SCENARIO_VALUE,
        objective=OBJECTIVE,
    )

    for max_iter in (50, 500, 2_000, 10_000):
        emit("FRONTIER_JSON", frontier_summary(grid, max_iter=max_iter))

    targets = [
        FRONTIER_RANGE[0] + (FRONTIER_RANGE[1] - FRONTIER_RANGE[0]) * i / (N_POINTS - 1)
        for i in range(N_POINTS)
    ]
    bisection = [bisection_lambda_for_target(grid, target) for target in targets]
    emit("BISECTION_JSON", bisection)

    high_lambdas = [0, 172.53666242656487, 294.65460671201083, 1_000, 5_000, 10_000, 20_000, 110_000]
    emit(
        "FIXED_LAMBDA_JSON",
        [
            {
                "lambda": lam,
                "total_conversion_prediction": total_at_lambda(grid, lam),
            }
            for lam in high_lambdas
        ],
    )

    # These assertions are intentionally coarse: they make the script fail
    # loudly if a future price-contour version fixes the frontier issue.
    current = frontier_summary(grid, max_iter=50)
    assert current["n_converged"] == 3, current
    assert current["last_point"]["gap_actual_minus_target"] < -20_000, current["last_point"]
    assert total_at_lambda(grid, 10_000) > 99_000


if __name__ == "__main__":
    main()
