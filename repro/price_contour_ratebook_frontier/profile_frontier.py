"""Profile the ratebook frontier sweep: per-point timings + cProfile breakdown.

Run from this folder:

    uv run python profile_frontier.py --frontier-steps 15
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path
from typing import Any

import polars as pl
from price_contour import RatebookOptimiser, build_grid_from_parquet_chunked

HERE = Path(__file__).resolve().parent


def load_metadata() -> dict[str, Any]:
    return json.loads((HERE / "metadata.json").read_text())


def build_grid(meta: dict[str, Any]):
    cfg = meta["config"]
    return build_grid_from_parquet_chunked(
        str(HERE / meta["scored_parquet"]),
        cfg["constraint_cols"],
        int(cfg["chunk_size"]),
        quote_id=cfg["quote_id"],
        scenario_index=cfg["scenario_index"],
        scenario_value=cfg["scenario_value"],
        objective=cfg["objective"],
    )


def align_factors(meta: dict[str, Any], quote_ids: list[str]) -> pl.DataFrame:
    cfg = meta["config"]
    cols = list(dict.fromkeys(c for g in cfg["factor_columns"] for c in g))
    df = (
        pl.read_parquet(HERE / meta["factors_parquet"])
        .select([cfg["quote_id"], *cols])
        .with_columns(pl.col(cfg["quote_id"]).cast(pl.Utf8).alias("quote_id"))
        .unique(subset=["quote_id"])
    )
    order = pl.DataFrame({"quote_id": quote_ids}).unique(maintain_order=True)
    return order.join(df, on="quote_id", how="left").drop("quote_id")


def make_solver(meta: dict[str, Any]) -> RatebookOptimiser:
    cfg = meta["config"]
    return RatebookOptimiser(
        objective=cfg["objective"],
        constraints=cfg["constraints"],
        factor_columns=cfg["factor_columns"],
        max_iter=int(cfg["max_iter"]),
        max_cd_iterations=int(cfg["max_cd_iterations"]),
        cd_tolerance=float(cfg["cd_tolerance"]),
        tolerance=float(cfg["tolerance"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-steps", type=int, default=15)
    args = parser.parse_args()

    meta = load_metadata()
    grid = build_grid(meta)
    factors = align_factors(meta, grid.quote_ids)

    solver = make_solver(meta)
    base = solver.solve(grid, factors)
    print(json.dumps({"base": {"converged": base.converged, "lambdas": base.lambdas}}))

    cfg = meta["config"]
    ranges = {
        n: (float(b["min"]), float(b["max"]))
        for n, b in cfg["frontier_ranges"].items()
    }

    # Monkey-patch solve() to time each frontier-point invocation.
    per_point: list[float] = []
    original_solve = solver.solve

    def timed_solve(*a, **kw):
        t0 = time.perf_counter()
        try:
            return original_solve(*a, **kw)
        finally:
            per_point.append(time.perf_counter() - t0)

    solver.solve = timed_solve  # type: ignore[method-assign]

    # cProfile the whole frontier sweep.
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    frontier = solver.frontier(
        grid,
        factors,
        threshold_ranges=ranges,
        n_points_per_dim=args.frontier_steps,
        factor_columns=cfg["factor_columns"],
        initial_lambdas=base.lambdas,
    )
    pr.disable()
    total = time.perf_counter() - t0

    print(json.dumps({"frontier_total_seconds": round(total, 3),
                      "n_points": frontier.n_points}))
    print(json.dumps({"per_point_seconds": [round(t, 3) for t in per_point]}))
    print(json.dumps({"per_point_sum": round(sum(per_point), 3)}))

    # Top cumulative time by function.
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(30)
    print("\n=== cProfile (cumulative top 30) ===")
    print(s.getvalue())

    # Top by total internal time (excluding sub-calls).
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).strip_dirs().sort_stats("tottime").print_stats(30)
    print("\n=== cProfile (tottime top 30) ===")
    print(s2.getvalue())


if __name__ == "__main__":
    main()
