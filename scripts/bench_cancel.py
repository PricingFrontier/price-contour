"""Cancellation latency benchmark (DESIGN_DECISIONS §14.5).

Builds a synthetic grid (default 1,000,000 quotes x 41 steps x 3 constraints),
times each cancellable phase uncancelled, then cancels at uniformly random
points inside it and measures the time from ``cancel()`` to the ``Cancelled``
raise. Phases:

- ``online call``: ``apply_from_grid`` (pct baseline scan, argmax, baseline totals);
- ``online frame``: a first access to ``ApplyResult.dataframe``;
- ``ratebook call``: ``RatebookOptimiser.evaluate`` (kernel, baseline totals);
- ``ratebook frame``: a first access to ``RatebookEvaluation.quote_results``.

Exits 1 when any phase's p99 exceeds the threshold.

Usage:
    python scripts/bench_cancel.py
    python scripts/bench_cancel.py --n-quotes 200000 --trials 20
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable

import numpy as np
import polars as pl

from price_contour import (
    Cancelled,
    CancelToken,
    RatebookFactorContexts,
    RatebookOptimiser,
    apply_from_grid,
)
from price_contour._grid_utils import build_grid

parser = argparse.ArgumentParser(description="Cancellation latency benchmark")
parser.add_argument("--n-quotes", type=int, default=1_000_000)
parser.add_argument("--n-steps", type=int, default=41)
parser.add_argument("--trials", type=int, default=40, help="Cancelled runs per phase")
parser.add_argument("--threshold-ms", type=float, default=50.0, help="p99 limit")
parser.add_argument("--seed", type=int, default=20260928)
args = parser.parse_args()

CONSTRAINTS = {
    "volume": {"min_pct": 0.95},
    "claims": {"max_pct": 1.02},
    "retention": {"min_pct": 0.9},
}
LAMBDAS = {"volume": 2.0, "claims": 0.5, "retention": 1.0}
N_LEVELS = 12


def build_book(n_quotes: int, n_steps: int, rng: np.random.Generator):
    sv = np.linspace(0.8, 1.2, n_steps, dtype=np.float32)
    q = np.repeat(np.arange(n_quotes, dtype=np.int64), n_steps)
    sv_long = np.tile(sv, n_quotes)
    elasticity = np.repeat(rng.uniform(1.5, 5.0, n_quotes).astype(np.float32), n_steps)
    base = np.repeat(rng.uniform(80.0, 400.0, n_quotes).astype(np.float32), n_steps)
    conversion = 1.0 / (1.0 + np.exp(elasticity * (sv_long - 1.0)))
    ids = pl.Series("quote_id", [f"Q{i:07d}" for i in range(n_quotes)]).cast(
        pl.Categorical
    )
    df = pl.DataFrame(
        {
            "quote_id": ids.gather(q),
            "scenario_index": np.tile(np.arange(n_steps, dtype=np.int32), n_quotes),
            "scenario_value": sv_long,
            "expected_income": (base * sv_long * conversion).astype(np.float32),
            "volume": conversion.astype(np.float32),
            "claims": (conversion * base * 0.6).astype(np.float32),
            "retention": (conversion * 0.9).astype(np.float32),
        }
    )
    grid = build_grid(
        df,
        constraint_columns=list(CONSTRAINTS),
        quote_id="quote_id",
        scenario_index="scenario_index",
        scenario_value="scenario_value",
        objective="expected_income",
    )
    levels = [f"L{i}" for i in range(N_LEVELS)]
    factors = pl.DataFrame({"band": rng.choice(levels, n_quotes)})
    contexts = RatebookFactorContexts.from_dataframe(
        factors, [["band"]], quote_id=None, expected_quote_ids=grid.quote_ids
    )
    tables = {"band": {lvl: 0.85 + 0.03 * i for i, lvl in enumerate(levels)}}
    return grid, contexts, tables


def trials(
    duration: float,
    prepare: Callable[[CancelToken], object],
    act: Callable[[object], object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Cancel ``act`` at a uniformly random point inside ``duration``.

    ``prepare`` runs uncancelled first (for a frame phase it produces the
    result holding the token); only ``act`` is timed. Returns the latencies
    from ``cancel()`` to the ``Cancelled`` raise (ms) and the number of runs
    that finished before the cancel landed.
    """
    latencies: list[float] = []
    missed = 0
    attempts = 0
    while len(latencies) < args.trials and attempts < args.trials * 5:
        attempts += 1
        token = CancelToken()
        prepared = prepare(token)
        cancelled_at: list[float] = []

        def fire(
            delay: float, token: CancelToken = token, out: list[float] = cancelled_at
        ) -> None:
            time.sleep(delay)
            out.append(time.perf_counter())
            token.cancel()

        timer = threading.Thread(target=fire, args=(float(rng.uniform(0.0, duration)),))
        timer.start()
        try:
            act(prepared)
        except Cancelled:
            raised_at = time.perf_counter()
            timer.join()
            latencies.append(raised_at - cancelled_at[0])
        else:
            timer.join()
            missed += 1
        del prepared
    return np.array(latencies) * 1_000, missed


def timed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def call_phase(call: Callable[[CancelToken], object], rng: np.random.Generator):
    """The call itself: cancels land anywhere inside it."""
    duration = timed(lambda: call(CancelToken()))
    return duration, *trials(duration, lambda token: token, call, rng)


def frame_phase(
    call: Callable[[CancelToken], object], attr: str, rng: np.random.Generator
):
    """A first access to the lazily built frame of a completed call."""
    duration = timed(lambda: getattr(call(CancelToken()), attr))
    duration -= timed(lambda: call(CancelToken()))
    return duration, *trials(duration, call, lambda result: getattr(result, attr), rng)


def main() -> int:
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    grid, contexts, tables = build_book(args.n_quotes, args.n_steps, rng)
    print(
        f"grid: {args.n_quotes:,} quotes x {args.n_steps} steps x {len(CONSTRAINTS)} "
        f"constraints, built in {time.perf_counter() - t0:.1f}s"
    )
    optimiser = RatebookOptimiser(
        objective="expected_income", constraints=CONSTRAINTS, factor_columns=[["band"]]
    )

    def online_call(token: CancelToken) -> object:
        return apply_from_grid(grid, LAMBDAS, CONSTRAINTS, cancel=token)

    def ratebook_call(token: CancelToken) -> object:
        return optimiser.evaluate(grid, contexts, tables, cancel=token)

    results = [
        ("online call", *call_phase(online_call, rng)),
        ("online frame", *frame_phase(online_call, "dataframe", rng)),
        ("ratebook call", *call_phase(ratebook_call, rng)),
        ("ratebook frame", *frame_phase(ratebook_call, "quote_results", rng)),
    ]

    failed = False
    print(
        f"{'phase':<16}{'uncancelled':>13}{'n':>5}{'missed':>8}{'p50':>9}{'p99':>9}{'max':>9}  (ms)"
    )
    for name, duration, lat, missed in results:
        duration *= 1_000
        if lat.size == 0:
            print(
                f"{name:<16}{duration:>13.1f}{0:>5}{missed:>8}   no cancel landed inside the phase"
            )
            failed = True
            continue
        p50, p99 = np.percentile(lat, [50, 99])
        over = p99 > args.threshold_ms
        failed |= over
        print(
            f"{name:<16}{duration:>13.1f}{lat.size:>5}{missed:>8}"
            f"{p50:>9.2f}{p99:>9.2f}{lat.max():>9.2f}" + ("  OVER" if over else "")
        )
    print(
        f"threshold: p99 < {args.threshold_ms:g} ms -> {'FAIL' if failed else 'PASS'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
