"""Frontier oracle tests — semantic correctness across the full bug class.

This file complements the existing shape/golden tests with **oracle**
assertions: every frontier point reporting ``converged=True`` must
genuinely satisfy its target constraint within a small residual, and
every ``converged=False`` point must fail loudly (the actual must sit
at the achievable envelope, not at a low-lambda undershoot).

The bug we shipped against (the user's haute repro: target 92,588 hit at
lambda 151 instead of 702, undershooting by 24,000 with `converged=False`
and no signal that the target was reachable) belongs to a class:

  - **High-target undershoot.** Required lambda ≫ baseline magnitude.
  - **Mis-flagged convergence.** ``converged=False`` returned even when
    a feasible lambda is reachable.
  - **Silent infeasibility.** Targets outside the envelope returning a
    moderate lambda with the corresponding (sub-)total, not a clear
    infeasible signal.

These oracle tests sweep magnitude regimes that the previous fixture
range (~0.85-1.10× baseline) never reached, and assert *target
satisfaction* rather than just shape/positivity.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc


# ---------------------------------------------------------------------------
# Wide-envelope fixture
#
# Key properties:
#   - per-quote volume envelope spans 1.0 → 10.0 (10× spread).
#   - per-quote objective peaks at a low step index (1-4 of 11), so the
#     unconstrained argmax picks small-volume steps. zero-lambda total
#     therefore sits well below env_hi — analogous to the user's haute
#     repro (env_hi / zero_total ≈ 4.6×).
#   - peak-index varies per quote (1..4) so the 2-D subgradient solver
#     has non-degenerate per-quote choices.
# ---------------------------------------------------------------------------


def make_wide_envelope_df(
    n_quotes: int = 100,
    n_steps: int = 11,
) -> pl.DataFrame:
    rows = []
    for q in range(n_quotes):
        base = 100.0 + 0.5 * q
        peak_idx = 1 + int(3.0 * q / max(n_quotes - 1, 1))
        for j in range(n_steps):
            frac = j / max(n_steps - 1, 1)
            scenario_value = 0.5 + 0.1 * j
            volume = 1.0 + 9.0 * frac  # 1.0 → 10.0
            distance = abs(j - peak_idx)
            objective = base * (1.0 - 0.15 * distance**1.2)
            objective = max(objective, 0.05 * base)
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": scenario_value,
                    "expected_income": objective,
                    "volume": volume,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "quote_id": pl.Utf8,
            "scenario_index": pl.Int32,
            "scenario_value": pl.Float32,
            "expected_income": pl.Float32,
            "volume": pl.Float32,
        },
    )


def make_ratio_envelope_df(n_quotes: int = 80, n_steps: int = 9) -> pl.DataFrame:
    """Add `incurred` and `premium` for ratio constraints. Ratio at step 0
    is ~0.40, at step n_steps-1 is ~1.29 — a Max ratio constraint with a
    low target needs lambda to push toward low-ratio steps."""
    base = make_wide_envelope_df(n_quotes, n_steps)
    incurred = []
    premium = []
    for q in range(n_quotes):
        for j in range(n_steps):
            frac = j / max(n_steps - 1, 1)
            prem = 100.0 - 30.0 * frac
            inc = 40.0 + 50.0 * frac
            incurred.append(inc)
            premium.append(prem)
    return base.with_columns(
        pl.Series("incurred", incurred, dtype=pl.Float32),
        pl.Series("premium", premium, dtype=pl.Float32),
    )


# ---------------------------------------------------------------------------
# Helpers — envelope and lambda-zero baseline
# ---------------------------------------------------------------------------


def _lambda_zero_total(df: pl.DataFrame, constraint: str) -> float:
    """Total of `constraint` when each quote picks its max-objective step."""
    per_quote = df.group_by("quote_id", maintain_order=True).agg(
        pl.col("expected_income"), pl.col(constraint)
    )
    total = 0.0
    for row in per_quote.iter_rows(named=True):
        objs = row["expected_income"]
        cons = row[constraint]
        best_j = max(range(len(objs)), key=lambda i: objs[i])
        total += float(cons[best_j])
    return total


def _envelope_bounds(df: pl.DataFrame, constraint: str) -> tuple[float, float]:
    per_quote = df.group_by("quote_id").agg(
        pl.col(constraint).min().alias("min"),
        pl.col(constraint).max().alias("max"),
    )
    return (
        float(per_quote.select(pl.col("min").sum()).item()),
        float(per_quote.select(pl.col("max").sum()).item()),
    )


def _assert_oracle(
    points_df: pl.DataFrame,
    constraint: str,
    direction: str,
    envelope: tuple[float, float],
    *,
    residual_tolerance_rel: float = 0.005,
    overshoot_tolerance_rel: float = 0.5,
):
    """Two-sided oracle: every converged point satisfies its target
    one-sidedly within ``residual_tolerance_rel × envelope``, and does
    NOT over-satisfy by more than ``overshoot_tolerance_rel × envelope``.
    The latter catches lambda runaway (a bug that pushes lambda to the
    cap, satisfying the target by an order of magnitude).

    Non-converged points must report ``actual`` near the corresponding
    envelope edge (Min → env_hi, Max → env_lo) — never a silent
    low-lambda undershoot."""
    env_lo, env_hi = envelope
    env_span = max(env_hi - env_lo, 1.0)
    abs_tol = env_span * residual_tolerance_rel
    overshoot_abs_tol = env_span * overshoot_tolerance_rel

    threshold_col = f"threshold_{constraint}"
    total_col = f"total_{constraint}"
    lambda_col = f"lambda_{constraint}"

    for idx, row in enumerate(points_df.to_dicts()):
        target = float(row[threshold_col])
        actual = float(row[total_col])
        lam = float(row[lambda_col])
        converged = bool(row["converged"])

        assert lam >= 0.0, f"row {idx}: lambda must be ≥ 0, got {lam}"
        assert math.isfinite(actual), f"row {idx}: actual non-finite ({actual})"
        assert math.isfinite(lam), f"row {idx}: lambda non-finite ({lam})"

        if direction == "min":
            undershoot = target - actual
            overshoot = actual - target
            if converged:
                assert undershoot <= abs_tol, (
                    f"row {idx}: Min target={target:.3f} converged=True "
                    f"but actual={actual:.3f} undershoots by "
                    f"{undershoot:.3f} (tol={abs_tol:.3f}); lambda={lam}"
                )
                # Lambda-runaway guard: a converged Min should land *near*
                # the target (one-sided), not at the env_hi ceiling.
                # `target + overshoot_tol` allows for the discrete-step
                # gap on small portfolios but flags catastrophic overshoot.
                assert overshoot <= overshoot_abs_tol, (
                    f"row {idx}: Min target={target:.3f} converged=True "
                    f"but actual={actual:.3f} overshoots by "
                    f"{overshoot:.3f} (max {overshoot_abs_tol:.3f}); "
                    f"lambda runaway? lambda={lam}"
                )
            else:
                assert actual >= env_hi - abs_tol or target > env_hi + abs_tol, (
                    f"row {idx}: Min target={target:.3f} converged=False "
                    f"but actual={actual:.3f} is far from env_hi "
                    f"({env_hi:.3f}) AND target is within envelope; "
                    f"silent undershoot. lambda={lam}"
                )
        elif direction == "max":
            overshoot = actual - target
            undershoot = target - actual
            if converged:
                assert overshoot <= abs_tol, (
                    f"row {idx}: Max target={target:.3f} converged=True "
                    f"but actual={actual:.3f} overshoots by "
                    f"{overshoot:.3f} (tol={abs_tol:.3f}); lambda={lam}"
                )
                # Lambda-runaway guard for Max: actual should land near
                # the target, not at env_lo.
                assert undershoot <= overshoot_abs_tol, (
                    f"row {idx}: Max target={target:.3f} converged=True "
                    f"but actual={actual:.3f} undershoots by "
                    f"{undershoot:.3f} (max {overshoot_abs_tol:.3f}); "
                    f"lambda runaway? lambda={lam}"
                )
            else:
                assert actual <= env_lo + abs_tol or target < env_lo - abs_tol, (
                    f"row {idx}: Max target={target:.3f} converged=False "
                    f"but actual={actual:.3f} is far from env_lo "
                    f"({env_lo:.3f}); silent overshoot. lambda={lam}"
                )
        else:
            raise ValueError(f"unknown direction: {direction}")


# ---------------------------------------------------------------------------
# 1-D Min-sum oracle across magnitude regimes (the bug class the user hit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "regime", ["below_baseline", "near_baseline", "high", "extreme"]
)
def test_oracle_1d_min_sum_across_regimes(regime: str, parallel: bool):
    """Sweep four target magnitude regimes for a 1-D Min sum constraint.

    The user's bug surfaced ONLY in the `extreme` regime
    (target ≫ baseline). Pre-existing tests covered ``below_baseline``
    and ``near_baseline`` only. This parametrisation pins all four —
    any future regression in the 1-D path fires here regardless of
    target magnitude.
    """
    df = make_wide_envelope_df(n_quotes=100, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")
    span_above = env_hi - zero_total

    regimes = {
        "below_baseline": (env_lo + 0.1 * (zero_total - env_lo), zero_total * 0.95),
        "near_baseline": (zero_total * 0.95, zero_total + 0.2 * span_above),
        "high": (zero_total + 0.3 * span_above, zero_total + 0.7 * span_above),
        "extreme": (zero_total + 0.5 * span_above, zero_total + 0.95 * span_above),
    }
    lo, hi = regimes[regime]

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
        tolerance=1e-6,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (lo, hi)},
        n_points_per_dim=8,
        parallel=parallel,
    )
    _assert_oracle(
        result.points,
        constraint="volume",
        direction="min",
        envelope=(env_lo, env_hi),
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("regime", ["near_baseline", "moderate", "high", "extreme"])
def test_oracle_1d_max_sum_across_regimes(regime: str, parallel: bool):
    """Symmetric Max-direction sweep — lower targets need higher lambda."""
    df = make_wide_envelope_df(n_quotes=100, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")
    span_below = zero_total - env_lo
    if span_below <= 0:
        pytest.skip("fixture has zero_total ≤ env_lo; Max not testable")

    regimes = {
        "near_baseline": (
            zero_total - 0.2 * span_below,
            zero_total - 0.05 * span_below,
        ),
        "moderate": (zero_total - 0.5 * span_below, zero_total - 0.3 * span_below),
        "high": (zero_total - 0.7 * span_below, zero_total - 0.5 * span_below),
        "extreme": (zero_total - 0.95 * span_below, zero_total - 0.7 * span_below),
    }
    lo, hi = regimes[regime]

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"max": None}},
        max_iter=50,
        tolerance=1e-6,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (lo, hi)},
        n_points_per_dim=8,
        parallel=parallel,
    )
    _assert_oracle(
        result.points,
        constraint="volume",
        direction="max",
        envelope=(env_lo, env_hi),
    )


# ---------------------------------------------------------------------------
# Infeasibility signalling
# ---------------------------------------------------------------------------


def test_oracle_1d_min_above_envelope_marked_infeasible():
    """A Min target above achievable max must report converged=False AND
    actual at the envelope (not a silent low-lambda undershoot)."""
    df = make_wide_envelope_df(n_quotes=100, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    target = env_hi * 1.05

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
    )
    p = result.points.to_dicts()[0]
    assert not p["converged"], (
        f"infeasible target {target} must report converged=False; "
        f"got actual={p['total_volume']}, lambda={p['lambda_volume']}"
    )
    span = env_hi - env_lo
    assert p["total_volume"] >= env_hi - span * 0.01, (
        f"infeasible-target point should report actual at env_hi "
        f"(~{env_hi}); got {p['total_volume']} — silent undershoot"
    )


def test_oracle_1d_max_below_envelope_marked_infeasible():
    df = make_wide_envelope_df(n_quotes=100, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    target = env_lo * 0.95

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"max": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
    )
    p = result.points.to_dicts()[0]
    assert not p["converged"]
    span = env_hi - env_lo
    assert p["total_volume"] <= env_lo + span * 0.01, (
        f"Max infeasible-target point should report actual at env_lo "
        f"(~{env_lo}); got {p['total_volume']}"
    )


# ---------------------------------------------------------------------------
# Threshold-mode coverage: pct thresholds
# ---------------------------------------------------------------------------


def test_oracle_1d_min_pct_threshold_satisfies_at_high_target():
    """`min_pct` (fraction-of-baseline) must satisfy after pct→absolute
    resolution. Sweeps high pct values requiring real lambda."""
    df = make_wide_envelope_df(n_quotes=100, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")

    baseline_at_1 = float(
        df.filter(pl.col("scenario_value").cast(pl.Float64) == 1.0)
        .select(pl.col("volume").sum())
        .item()
    )
    assert baseline_at_1 > 0, "fixture must include scenario_value == 1.0"

    pct_lo = 1.5
    pct_hi = min(4.0, env_hi / baseline_at_1 * 0.95)
    if pct_hi <= pct_lo:
        pytest.skip("fixture envelope can't reach 1.5× baseline")

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min_pct": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (pct_lo, pct_hi)},
        n_points_per_dim=6,
    )
    abs_tol = (env_hi - env_lo) * 0.01
    for idx, row in enumerate(result.points.to_dicts()):
        target_pct = float(row["threshold_volume"])
        target_abs = target_pct * baseline_at_1
        actual = float(row["total_volume"])
        if row["converged"]:
            residual = target_abs - actual
            assert residual <= abs_tol, (
                f"row {idx}: min_pct={target_pct} (abs={target_abs:.2f}) "
                f"converged=True but actual={actual:.2f} undershoots by "
                f"{residual:.2f}; lambda={row['lambda_volume']}"
            )


# ---------------------------------------------------------------------------
# Multi-constraint subgradient path
# ---------------------------------------------------------------------------


def test_oracle_2d_subgradient_satisfies_targets_when_converged():
    """Two constraints — exercises the subgradient solver. Every point
    reporting converged=True must satisfy ALL its targets."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    df = df.with_columns(
        (pl.col("volume") * 0.5 + pl.col("scenario_value")).alias("retention")
    )

    env_vol = _envelope_bounds(df, "volume")
    env_ret = _envelope_bounds(df, "retention")
    zero_vol = _lambda_zero_total(df, "volume")
    zero_ret = _lambda_zero_total(df, "retention")

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}, "retention": {"min": None}},
        max_iter=300,
        tolerance=1e-6,
    )
    result = solver.frontier(
        df,
        threshold_ranges={
            "volume": (
                zero_vol + 0.1 * (env_vol[1] - zero_vol),
                zero_vol + 0.5 * (env_vol[1] - zero_vol),
            ),
            "retention": (
                zero_ret + 0.1 * (env_ret[1] - zero_ret),
                zero_ret + 0.5 * (env_ret[1] - zero_ret),
            ),
        },
        n_points_per_dim=4,
    )
    abs_tol_vol = (env_vol[1] - env_vol[0]) * 0.02
    abs_tol_ret = (env_ret[1] - env_ret[0]) * 0.02
    for idx, row in enumerate(result.points.to_dicts()):
        if row["converged"]:
            v_resid = row["threshold_volume"] - row["total_volume"]
            r_resid = row["threshold_retention"] - row["total_retention"]
            assert v_resid <= abs_tol_vol, (
                f"row {idx}: 2D Min volume target={row['threshold_volume']:.2f} "
                f"converged=True but actual={row['total_volume']:.2f} "
                f"undershoots by {v_resid:.2f}; lambda={row['lambda_volume']}"
            )
            assert r_resid <= abs_tol_ret, (
                f"row {idx}: 2D Min retention target={row['threshold_retention']:.2f} "
                f"converged=True but actual={row['total_retention']:.2f} "
                f"undershoots by {r_resid:.2f}; lambda={row['lambda_retention']}"
            )


# ---------------------------------------------------------------------------
# Ratio constraints — separate code path (Python linearisation)
# ---------------------------------------------------------------------------


def test_oracle_1d_ratio_max_satisfies():
    """Ratio Max constraint: every converged point must satisfy
    actual_ratio ≤ target."""
    df = make_ratio_envelope_df(n_quotes=80, n_steps=9)

    per_quote_ratios = df.group_by("quote_id").agg(
        (pl.col("incurred") / pl.col("premium")).min().alias("min_ratio"),
        (pl.col("incurred") / pl.col("premium")).max().alias("max_ratio"),
    )
    min_ind = float(per_quote_ratios.select(pl.col("min_ratio").min()).item())
    max_ind = float(per_quote_ratios.select(pl.col("max_ratio").max()).item())

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": None,
            },
        },
        max_iter=100,
        tolerance=1e-6,
    )
    lo = min_ind + 0.2 * (max_ind - min_ind)
    hi = min_ind + 0.7 * (max_ind - min_ind)
    result = solver.frontier(
        df,
        threshold_ranges={"loss_ratio": (lo, hi)},
        n_points_per_dim=5,
    )
    for idx, row in enumerate(result.points.to_dicts()):
        if row["converged"]:
            target = float(row["threshold_loss_ratio"])
            actual = float(row["total_loss_ratio"])
            assert actual <= target + abs(target) * 0.05, (
                f"row {idx}: ratio Max target={target:.4f} converged=True "
                f"but actual={actual:.4f} overshoots; "
                f"lambda={row['lambda_loss_ratio']}"
            )


# ---------------------------------------------------------------------------
# Lambda monotonicity in target — bisection sanity check
# ---------------------------------------------------------------------------


def test_oracle_1d_min_lambda_monotone_in_target():
    """For Min: a tighter (higher) target needs lambda ≥ the lambda for a
    looser target. Pins the bisection's monotone contract."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    lo = zero_total + 0.1 * (env_hi - zero_total)
    hi = zero_total + 0.9 * (env_hi - zero_total)
    result = solver.frontier(
        df, threshold_ranges={"volume": (lo, hi)}, n_points_per_dim=10
    )
    points = result.points.sort("threshold_volume").to_dicts()
    slack = (env_hi - env_lo) * 0.001
    for i in range(1, len(points)):
        prev_lam = float(points[i - 1]["lambda_volume"])
        curr_lam = float(points[i]["lambda_volume"])
        assert curr_lam >= prev_lam - slack, (
            f"lambda non-monotone in target: "
            f"target {points[i - 1]['threshold_volume']:.2f} → λ={prev_lam:.4f}, "
            f"target {points[i]['threshold_volume']:.2f} → λ={curr_lam:.4f} "
            f"(decreased)"
        )


def test_oracle_1d_max_lambda_monotone_in_target():
    """Max: tighter (lower) target needs lambda ≥ the lambda for a
    looser (higher) target."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")

    span = zero_total - env_lo
    if span <= 0:
        pytest.skip("fixture has zero_total ≤ env_lo; Max not testable")
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"max": None}},
        max_iter=50,
    )
    lo = zero_total - 0.9 * span
    hi = zero_total - 0.1 * span
    result = solver.frontier(
        df, threshold_ranges={"volume": (lo, hi)}, n_points_per_dim=10
    )
    points = result.points.sort("threshold_volume", descending=True).to_dicts()
    slack = (env_hi - env_lo) * 0.001
    for i in range(1, len(points)):
        prev_lam = float(points[i - 1]["lambda_volume"])
        curr_lam = float(points[i]["lambda_volume"])
        assert curr_lam >= prev_lam - slack, (
            f"Max lambda non-monotone: "
            f"target {points[i - 1]['threshold_volume']:.2f} → λ={prev_lam:.4f}, "
            f"target {points[i]['threshold_volume']:.2f} → λ={curr_lam:.4f} "
            f"(decreased while target tightened)"
        )


# ---------------------------------------------------------------------------
# Density invariance — same range, different point counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_points", [3, 7, 15, 30])
def test_oracle_1d_min_satisfies_at_any_density(n_points: int):
    """Same target range with different densities — every point must
    satisfy. Catches warm-start interactions between adjacent points."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    lo = zero_total + 0.1 * (env_hi - zero_total)
    hi = zero_total + 0.9 * (env_hi - zero_total)
    result = solver.frontier(
        df, threshold_ranges={"volume": (lo, hi)}, n_points_per_dim=n_points
    )
    _assert_oracle(
        result.points,
        constraint="volume",
        direction="min",
        envelope=(env_lo, env_hi),
    )


# ---------------------------------------------------------------------------
# Regression test echoing the user's exact repro shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [False, True])
def test_oracle_haute_repro_shape_high_target_to_baseline_ratio(parallel: bool):
    """The user's haute repro had target/zero_total ≈ 4.6× — the regime
    that broke. Synthetic equivalent here pins a similar ratio so any
    future regression in this exact bug class fires immediately."""
    df = make_wide_envelope_df(n_quotes=200, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")
    ratio = env_hi / max(zero_total, 1e-9)
    assert ratio >= 2.5, (
        f"fixture should have env_hi/zero_total ≥ 2.5; got {ratio:.2f}"
        f" (zero_total={zero_total}, env_hi={env_hi}) — fixture broken"
    )

    target = min(zero_total * 3.5, env_hi * 0.95)
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
        parallel=parallel,
    )
    p = result.points.to_dicts()[0]
    assert p["converged"], (
        f"haute-shaped target={target:.2f} (env_hi={env_hi:.2f}, "
        f"zero_total={zero_total:.2f}, ratio={ratio:.2f}×) must converge — "
        f"this is the exact bug class the user hit. Got "
        f"actual={p['total_volume']:.2f}, lambda={p['lambda_volume']:.2f}"
    )
    abs_tol = (env_hi - env_lo) * 0.01
    assert p["total_volume"] >= target - abs_tol, (
        f"haute-shaped point at target={target:.2f} undershoots: "
        f"actual={p['total_volume']:.2f} (tol={abs_tol:.2f})"
    )


# ---------------------------------------------------------------------------
# Sweep-order invariance
# ---------------------------------------------------------------------------


def test_oracle_sweep_order_invariant_under_reversal():
    """The frontier visits points in cartesian order. If bisection or
    warm-start has an order-dependent bug, sweeping (lo, hi) and (hi, lo)
    would diverge. Pin that the per-point converged-actual pair matches
    under reversal (within tolerance)."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")
    lo = zero_total + 0.2 * (env_hi - zero_total)
    hi = zero_total + 0.8 * (env_hi - zero_total)

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    fwd = solver.frontier(df, threshold_ranges={"volume": (lo, hi)}, n_points_per_dim=6)
    rev = solver.frontier(df, threshold_ranges={"volume": (hi, lo)}, n_points_per_dim=6)

    fwd_by_target = {
        round(float(r["threshold_volume"]), 4): r for r in fwd.points.to_dicts()
    }
    rev_by_target = {
        round(float(r["threshold_volume"]), 4): r for r in rev.points.to_dicts()
    }
    assert set(fwd_by_target.keys()) == set(rev_by_target.keys())

    abs_tol = (env_hi - env_lo) * 0.005
    for target_key, f_row in fwd_by_target.items():
        r_row = rev_by_target[target_key]
        if f_row["converged"] and r_row["converged"]:
            f_actual = float(f_row["total_volume"])
            r_actual = float(r_row["total_volume"])
            assert abs(f_actual - r_actual) <= abs_tol, (
                f"target={target_key}: forward actual={f_actual:.4f} vs "
                f"reverse actual={r_actual:.4f}; difference > "
                f"{abs_tol:.4f} — order-dependent bug"
            )


# ---------------------------------------------------------------------------
# Boundary cases — target sits exactly at zero_total / env_hi / 1-ULP-off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset_sign", [-1, 0, 1])
def test_oracle_1d_min_target_at_zero_total_boundary(offset_sign: int):
    """Target == lambda=0 baseline ± a meaningful f32-scale offset.

    The grid builder enforces `Float32` storage on all value columns,
    so summing per-quote f32 volumes through the apply path quantises
    to f32 precision (~7 decimal digits). A naive `math.nextafter`
    in f64 would step a single f64 ULP that's invisible to the
    bisection's f32-precision arithmetic. We therefore probe with a
    multiplicative delta of `100 × f32_eps` ≈ 1.2e-5 — clearly above
    the apply-path noise floor while still small enough to test the
    `target == zero_total` short-circuit boundary."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    zero_total = _lambda_zero_total(df, "volume")

    # f32 epsilon ≈ 1.19e-7; 100× gives a delta well above f32-summation
    # noise but far below any discrete step gap.
    rel_delta = 100.0 * 1.1920929e-7
    if offset_sign == 0:
        target = zero_total
    elif offset_sign > 0:
        target = zero_total * (1.0 + rel_delta)
    else:
        target = zero_total * (1.0 - rel_delta)

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
    )
    p = result.points.to_dicts()[0]
    actual = float(p["total_volume"])
    if offset_sign <= 0:
        # At or below baseline → lambda=0 satisfies → converged.
        assert p["converged"], (
            f"target {target} ≤ zero_total {zero_total} should converge at "
            f"lambda=0; got converged=False, actual={actual}, "
            f"lambda={p['lambda_volume']}"
        )
        assert actual >= target, (
            f"converged target {target}: actual={actual} fails Min satisfaction"
        )
    else:
        # Just above baseline → either lambda climbs to a higher
        # discrete step (converged with `actual >= target`), or no step
        # gives a higher total (infeasible).
        if p["converged"]:
            assert actual >= target, (
                f"row reports converged but actual={actual} < target={target}"
            )
        else:
            # Loud-failure contract: the Rust frontier emits a non-null
            # non_convergence_reason on failures.
            reason = p.get("non_convergence_reason")
            assert reason in (
                "above_envelope",
                "bracket_exhausted",
            ), f"infeasible boundary case missing reason: {reason}"


def test_oracle_1d_min_target_at_envelope_max_boundary():
    """Target == achievable envelope max (sum of per-quote max volumes).
    Should converge at the cap or report converged=False with actual at
    the envelope. NEVER a silent low-lambda undershoot.

    Probes three positions around `env_hi` using a multiplicative delta
    of `100 × f32_eps` — see the `_at_zero_total` boundary test for the
    rationale (Float32 storage makes a single f64 ULP invisible to the
    apply path)."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    rel_delta = 100.0 * 1.1920929e-7

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    # Probe three boundary positions, with f32-scale offsets.
    for label, target in [
        ("below_env_hi", env_hi * (1.0 - rel_delta)),
        ("at_env_hi", env_hi),
        ("above_env_hi", env_hi * (1.0 + rel_delta)),
    ]:
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (target, target)},
            n_points_per_dim=1,
        )
        p = result.points.to_dicts()[0]
        actual = float(p["total_volume"])
        # Whether converged or not, actual must sit at-or-just-under the
        # envelope ceiling (within discrete-step granularity).
        assert actual >= env_hi - (env_hi - env_lo) * 0.01, (
            f"{label}: target={target} actual={actual} is far below env_hi "
            f"({env_hi}); silent undershoot. converged={p['converged']}, "
            f"lambda={p['lambda_volume']}, "
            f"reason={p.get('non_convergence_reason')}"
        )


# ---------------------------------------------------------------------------
# Multi-constraint subgradient: extreme-target regime (Test #4)
# ---------------------------------------------------------------------------


def test_oracle_2d_subgradient_extreme_targets_loud_failure():
    """Pin the contract for the multi-constraint subgradient path at
    extreme targets: every converged=True point must satisfy its target,
    AND every converged=False point must report a non_convergence_reason
    so callers can act on it.

    The 1-D bisection fast path is dispatched only when n_constraints == 1;
    multi-constraint sweeps still go through the iterative subgradient
    solver, so this is where the original 0.3.2 bug class could re-surface
    if anything regresses on lambda updates."""
    df = make_wide_envelope_df(n_quotes=80, n_steps=11)
    df = df.with_columns(
        (pl.col("volume") * 0.5 + pl.col("scenario_value")).alias("retention")
    )
    env_vol = _envelope_bounds(df, "volume")
    env_ret = _envelope_bounds(df, "retention")
    zero_vol = _lambda_zero_total(df, "volume")
    zero_ret = _lambda_zero_total(df, "retention")

    # Extreme: 0.7×–0.95× of the span above zero_total on both axes —
    # the regime the original subgradient struggled with.
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}, "retention": {"min": None}},
        max_iter=300,
        tolerance=1e-5,
    )
    result = solver.frontier(
        df,
        threshold_ranges={
            "volume": (
                zero_vol + 0.7 * (env_vol[1] - zero_vol),
                zero_vol + 0.95 * (env_vol[1] - zero_vol),
            ),
            "retention": (
                zero_ret + 0.7 * (env_ret[1] - zero_ret),
                zero_ret + 0.95 * (env_ret[1] - zero_ret),
            ),
        },
        n_points_per_dim=3,
    )
    abs_tol_vol = (env_vol[1] - env_vol[0]) * 0.02
    abs_tol_ret = (env_ret[1] - env_ret[0]) * 0.02
    for idx, row in enumerate(result.points.to_dicts()):
        if row["converged"]:
            v_resid = row["threshold_volume"] - row["total_volume"]
            r_resid = row["threshold_retention"] - row["total_retention"]
            assert v_resid <= abs_tol_vol, (
                f"row {idx}: 2D extreme volume target={row['threshold_volume']:.2f} "
                f"converged=True but actual={row['total_volume']:.2f} "
                f"undershoots by {v_resid:.2f}; lambda={row['lambda_volume']}"
            )
            assert r_resid <= abs_tol_ret, (
                f"row {idx}: 2D extreme retention target={row['threshold_retention']:.2f} "
                f"converged=True but actual={row['total_retention']:.2f} "
                f"undershoots by {r_resid:.2f}; lambda={row['lambda_retention']}"
            )
        else:
            # Every non-converged point MUST carry a reason — the 0.3.2
            # bug was precisely a converged=False point with no signal
            # to the caller. Even if the subgradient still undershoots
            # at extreme targets, the contract is that callers know it.
            reason = row.get("non_convergence_reason")
            assert reason is not None, (
                f"row {idx}: 2D extreme target reports converged=False "
                f"but no non_convergence_reason — silent failure. "
                f"target=({row['threshold_volume']:.2f}, "
                f"{row['threshold_retention']:.2f}), "
                f"actual=({row['total_volume']:.2f}, "
                f"{row['total_retention']:.2f})"
            )


# ---------------------------------------------------------------------------
# Warm-start invariance and tied-objectives determinism (Test #5)
# ---------------------------------------------------------------------------


def test_oracle_bisection_warm_start_invariant_under_initial_lambdas():
    """The 1-D bisection path uses `initial_lambdas` as a warm bracket
    floor. A correct warm-start can only TIGHTEN the bracket (skipping
    bracket-expand probes); it must NEVER change the converged lambda or
    actual to within tolerance.

    A future regression that erroneously fed `initial_lambdas[0]` as the
    starting `hi` (instead of the warm `lo`) would shift convergence,
    fired by this test."""
    df = make_wide_envelope_df(n_quotes=100, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    target = zero_total + 0.5 * (env_hi - zero_total)

    # Cold-start.
    cold = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
        initial_lambdas=None,
    )
    # Warm-start with a small lambda — should give same answer.
    warm_small = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
        initial_lambdas={"volume": 0.5},
    )
    # Warm-start with a much-too-large lambda — bisection should still
    # converge to the same correct answer (lo cap pulls it back).
    warm_big = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
        initial_lambdas={"volume": 1.0e6},
    )

    abs_tol = (env_hi - env_lo) * 0.005
    cold_row = cold.points.to_dicts()[0]
    cold_actual = float(cold_row["total_volume"])
    cold_lambda = float(cold_row["lambda_volume"])
    # Lambda tolerance: bisection brackets at adjacent warm starts can
    # converge to lambdas a few ulps apart even with the same plateau.
    # 1% of the cold lambda is generous but tight enough that a
    # plateau-wrong-end regression (e.g. feeding initial_lambdas as the
    # upper bracket) shifts lambda by ≫ 1%.
    lambda_tol = max(abs(cold_lambda) * 0.01, 1e-6)
    for label, frontier_result in [("warm_small", warm_small), ("warm_big", warm_big)]:
        row = frontier_result.points.to_dicts()[0]
        actual = float(row["total_volume"])
        lam = float(row["lambda_volume"])
        assert abs(actual - cold_actual) <= abs_tol, (
            f"{label}: warm-started actual={actual:.4f} differs from "
            f"cold-started actual={cold_actual:.4f} by more than "
            f"{abs_tol:.4f} — warm-start changed convergence"
        )
        assert abs(lam - cold_lambda) <= lambda_tol, (
            f"{label}: warm-started lambda={lam:.6f} differs from "
            f"cold-started lambda={cold_lambda:.6f} by more than "
            f"{lambda_tol:.6f} — warm-start landed on a different plateau "
            f"endpoint"
        )


def test_oracle_tied_objectives_deterministic_across_runs():
    """If multiple steps tie on BOTH objective and Lagrangian-relevant
    constraint value for a quote, the argmax tie-break must be
    deterministic. Running the same fixture twice must produce
    bit-identical optimal totals and lambdas — any non-determinism
    (parallel reduction reorder, ABA tie-break) fires here.

    The fixture deliberately ties step pairs (0,1) and (2,3) on both
    objective AND volume per quote, so a Lagrangian probe at any lambda
    keeps those pairs tied and the tie-break is forced to make a real
    choice (not just "pick the unique max")."""
    n_quotes = 50
    n_steps = 5
    rows = []
    for q in range(n_quotes):
        # Per-quote variation in objective level so quotes don't all
        # collapse to the same step.
        base = 80.0 + q * 0.5
        for j in range(n_steps):
            # Tie pairs: (0,1) share objective and volume; (2,3) share
            # objective and volume; (4) is unique. Tie pairs sit at
            # different magnitudes so different lambdas pick different
            # pairs — exercises the tie-break path under varying lambda.
            if j in (0, 1):
                obj = base
                vol = 1.0
            elif j in (2, 3):
                obj = base * 0.9
                vol = 5.0
            else:  # j == 4
                obj = base * 0.5
                vol = 10.0
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": 0.8 + 0.1 * j,
                    "expected_income": obj,
                    "volume": vol,
                }
            )
    df = pl.DataFrame(
        rows,
        schema={
            "quote_id": pl.Utf8,
            "scenario_index": pl.Int32,
            "scenario_value": pl.Float32,
            "expected_income": pl.Float32,
            "volume": pl.Float32,
        },
    )
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    a = solver.frontier(
        df, threshold_ranges={"volume": (100.0, 400.0)}, n_points_per_dim=5
    )
    b = solver.frontier(
        df, threshold_ranges={"volume": (100.0, 400.0)}, n_points_per_dim=5
    )
    a_rows = a.points.to_dicts()
    b_rows = b.points.to_dicts()
    assert len(a_rows) == len(b_rows)
    for ar, br in zip(a_rows, b_rows):
        assert ar["total_volume"] == br["total_volume"], (
            f"tied-objective fixture not deterministic across runs: "
            f"first={ar['total_volume']}, second={br['total_volume']}"
        )
        assert ar["lambda_volume"] == br["lambda_volume"]


# ---------------------------------------------------------------------------
# SolverPath / NonConvergenceReason emission contract
# ---------------------------------------------------------------------------


def test_oracle_solver_path_column_emitted_for_1d_and_2d_sweeps():
    """The `solver_path` column should report `bisection` for 1-D sweeps
    and `subgradient` for 2-D, so downstream callers can treat the two
    work-units appropriately."""
    df = make_wide_envelope_df(n_quotes=50, n_steps=11)
    df = df.with_columns(
        (pl.col("volume") * 0.5 + pl.col("scenario_value")).alias("retention")
    )

    solver_1d = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    r1 = solver_1d.frontier(
        df, threshold_ranges={"volume": (10.0, 200.0)}, n_points_per_dim=3
    )
    paths = set(r1.points["solver_path"].to_list())
    # Compare against the typed StrEnum mirror as well as the raw
    # string — proves the StrEnum round-trips against the Rust-emitted
    # column values.
    assert paths == {pc.SolverPath.BISECTION}
    assert paths == {"bisection"}, (
        f"1-D sweep should emit solver_path=bisection for every point; got {paths}"
    )

    solver_2d = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}, "retention": {"min": None}},
        max_iter=100,
    )
    env_vol = _envelope_bounds(df, "volume")
    env_ret = _envelope_bounds(df, "retention")
    r2 = solver_2d.frontier(
        df,
        threshold_ranges={
            "volume": (env_vol[0] * 1.05, env_vol[0] + 0.3 * (env_vol[1] - env_vol[0])),
            "retention": (
                env_ret[0] * 1.05,
                env_ret[0] + 0.3 * (env_ret[1] - env_ret[0]),
            ),
        },
        n_points_per_dim=3,
    )
    paths_2d = set(r2.points["solver_path"].to_list())
    assert paths_2d == {pc.SolverPath.SUBGRADIENT}
    assert paths_2d == {"subgradient"}, (
        f"2-D sweep should emit solver_path=subgradient for every point; got {paths_2d}"
    )


def test_oracle_non_convergence_reason_for_infeasible_target():
    """An infeasible Min target must come back with
    `non_convergence_reason=above_envelope`."""
    df = make_wide_envelope_df(n_quotes=50, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    target = env_hi * 1.10

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
    )
    p = result.points.to_dicts()[0]
    assert not p["converged"]
    assert p["non_convergence_reason"] == pc.NonConvergenceReason.ABOVE_ENVELOPE
    assert p["non_convergence_reason"] == "above_envelope", (
        f"infeasible target {target} should report "
        f"non_convergence_reason=above_envelope; got "
        f"{p['non_convergence_reason']}"
    )


def test_oracle_non_convergence_reason_column_is_nullable_string():
    """Pin the column-type contract for `non_convergence_reason`:
    the column must be a Polars Utf8 (nullable string), and converged
    rows must have a real null — not an empty string. This catches a
    future regression that uses `unwrap_or("")` on the optional, which
    would silently break `is_null()` filters on the consumer side."""
    df = make_wide_envelope_df(n_quotes=50, n_steps=11)
    env_lo, env_hi = _envelope_bounds(df, "volume")
    zero_total = _lambda_zero_total(df, "volume")

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    # Mix converged (mid-range) and infeasible (above envelope) targets
    # so we exercise both null and non-null column entries.
    converged_target = zero_total + 0.5 * (env_hi - zero_total)
    infeasible_target = env_hi * 1.10
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (converged_target, infeasible_target)},
        n_points_per_dim=4,
    )
    pts = result.points

    # Column dtype must be Utf8 (Polars' nullable string). A regression
    # that emits the column as e.g. an Object or a non-nullable string
    # fails this dtype check loudly.
    assert pts.schema["non_convergence_reason"] == pl.Utf8, (
        f"non_convergence_reason column dtype should be Utf8; got "
        f"{pts.schema['non_convergence_reason']}"
    )
    # solver_path is non-nullable Utf8 — every point has a path.
    assert pts.schema["solver_path"] == pl.Utf8

    # Converged rows must have null reason (not "" or any sentinel).
    converged_reasons = pts.filter(pl.col("converged")).select(
        pl.col("non_convergence_reason")
    )
    n_converged_with_null = converged_reasons.select(
        pl.col("non_convergence_reason").is_null().sum()
    ).item()
    assert n_converged_with_null == converged_reasons.height, (
        f"converged rows should have non_convergence_reason=null; "
        f"{converged_reasons.height - n_converged_with_null} rows have a "
        f"non-null reason. Sample: "
        f"{converged_reasons.head(3).to_dicts()}"
    )
    # Non-converged rows must have a non-null reason.
    nonconv = pts.filter(~pl.col("converged"))
    if nonconv.height > 0:
        n_nonconv_with_reason = nonconv.select(
            pl.col("non_convergence_reason").is_not_null().sum()
        ).item()
        assert n_nonconv_with_reason == nonconv.height, (
            f"non-converged rows should have a non_convergence_reason "
            f"string; {nonconv.height - n_nonconv_with_reason} rows are null"
        )
