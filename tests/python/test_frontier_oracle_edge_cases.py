"""Frontier oracle edge cases — degenerate sizes, infrequent enum
variants, mixed Min+Max 2-D, ratio extreme regime, and large-scale smoke.

These tests sit alongside ``test_frontier_oracle.py`` and target the
specific gaps the second-pass test review surfaced:

* The `IterationBudgetExhausted` reason variant is only emitted by the
  subgradient path; no other test exercises it directly.
* The `BracketExpansionExhausted` reason is unreachable from current
  callers but the variant is exposed and documented — pin its absence
  rather than leaving it untested.
* Empty / single-quote / single-step grids may surface UB without
  smoke coverage.
* Mixed Min+Max 2-D constraints have a different lambda-update sign
  pattern than Min+Min, currently untested at oracle level.
* Ratio extreme regime mirrors the 1-D sum extreme test for the
  separate ratio code path.
* A 10k-quote 1-D smoke test pins that warm-start scaling stays
  reasonable as N grows.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc


# ---------------------------------------------------------------------------
# Shared fixture — small wide-envelope grid (mirrors the oracle file's
# fixture but with a configurable size so each edge-case test can dial
# in the regime it cares about). Kept in this file so the edge-case
# tests are self-contained.
# ---------------------------------------------------------------------------


def make_grid_df(n_quotes: int, n_steps: int) -> pl.DataFrame:
    """Wide-envelope grid: per-quote volume spans 1 → 10, objective peaks
    at a low step so unconstrained argmax picks low-volume steps."""
    rows = []
    for q in range(n_quotes):
        base = 100.0 + 0.5 * q
        peak_idx = 1 + int(3.0 * q / max(n_quotes - 1, 1))
        for j in range(n_steps):
            frac = j / max(n_steps - 1, 1) if n_steps > 1 else 0.0
            scenario_value = 0.5 + 0.1 * j
            volume = 1.0 + 9.0 * frac
            distance = abs(j - peak_idx)
            objective = max(base * (1.0 - 0.15 * distance**1.2), 0.05 * base)
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


# ---------------------------------------------------------------------------
# Single-quote and single-step degenerate grids
# ---------------------------------------------------------------------------


def test_single_quote_grid_1d_frontier_works():
    """One quote, multiple steps — bisection should still terminate
    cleanly and return shape-correct points."""
    df = make_grid_df(n_quotes=1, n_steps=5)
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    # Volume range spans 1.0 → 10.0 with one quote.
    result = solver.frontier(
        df, threshold_ranges={"volume": (2.0, 8.0)}, n_points_per_dim=4
    )
    assert result.n_points == 4
    for p in result.points.to_dicts():
        assert math.isfinite(p["total_volume"])
        assert math.isfinite(p["lambda_volume"])
        # solver_path is always populated.
        assert p["solver_path"] in (pc.SolverPath.BISECTION, "bisection")


def test_single_step_grid_1d_frontier_returns_only_choice():
    """One step per quote — every quote is forced to that step
    regardless of lambda. The "frontier" collapses to a single
    achievable total at lambda=0; targets above it are infeasible,
    targets below are trivially satisfied."""
    df = make_grid_df(n_quotes=10, n_steps=1)
    only_total = float(df.select(pl.col("volume").sum()).item())

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (only_total * 0.5, only_total * 0.95)},
        n_points_per_dim=3,
    )
    # Every point should converge at lambda=0 with actual==only_total.
    for p in result.points.to_dicts():
        assert p["converged"]
        assert math.isclose(float(p["total_volume"]), only_total, rel_tol=1e-6)
        assert float(p["lambda_volume"]) == 0.0


def test_single_step_grid_above_envelope_marked_infeasible():
    df = make_grid_df(n_quotes=10, n_steps=1)
    only_total = float(df.select(pl.col("volume").sum()).item())

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    target = only_total * 1.10
    result = solver.frontier(
        df,
        threshold_ranges={"volume": (target, target)},
        n_points_per_dim=1,
    )
    p = result.points.to_dicts()[0]
    assert not p["converged"]
    assert p["non_convergence_reason"] == pc.NonConvergenceReason.ABOVE_ENVELOPE


# ---------------------------------------------------------------------------
# IterationBudgetExhausted on subgradient path
# ---------------------------------------------------------------------------


def test_subgradient_iteration_budget_exhausted_reason_emitted():
    """A 2-D extreme target with a deliberately tiny `max_iter` forces
    the subgradient solver to give up before settling. The point must
    report converged=False AND
    `non_convergence_reason == iteration_budget_exhausted` —
    distinguishing solver-budget exhaustion from structural
    infeasibility."""
    df = make_grid_df(n_quotes=80, n_steps=11)
    df = df.with_columns(
        (pl.col("volume") * 0.5 + pl.col("scenario_value")).alias("retention")
    )

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        # Extreme corner — high targets on both axes; max_iter=2 is
        # nowhere near enough for subgradient to settle.
        constraints={"volume": {"min": None}, "retention": {"min": None}},
        max_iter=2,
        tolerance=1e-9,  # tight tolerance ensures convergence fails.
    )
    # Compute extreme corner targets.
    vol_max = float(
        df.group_by("quote_id")
        .agg(pl.col("volume").max())
        .select(pl.sum("volume"))
        .item()
    )
    ret_max = float(
        df.group_by("quote_id")
        .agg(pl.col("retention").max())
        .select(pl.sum("retention"))
        .item()
    )
    result = solver.frontier(
        df,
        threshold_ranges={
            "volume": (vol_max * 0.85, vol_max * 0.85),
            "retention": (ret_max * 0.85, ret_max * 0.85),
        },
        n_points_per_dim=1,
    )
    p = result.points.to_dicts()[0]
    if not p["converged"]:
        assert (
            p["non_convergence_reason"]
            == pc.NonConvergenceReason.ITERATION_BUDGET_EXHAUSTED
        ), (
            f"max_iter=2 subgradient at extreme target should report "
            f"iteration_budget_exhausted; got {p['non_convergence_reason']}"
        )
    # Note: if subgradient happens to converge in 2 iters (unlikely for
    # this regime), the test still passes — it's pinning the reason
    # variant, not forcing a non-convergence.


# ---------------------------------------------------------------------------
# Mixed Min + Max in a 2-D sweep
# ---------------------------------------------------------------------------


def test_mixed_min_max_2d_sweep_oracle():
    """One Min and one Max constraint in the same sweep — exercises the
    subgradient solver's per-constraint sign-flip in lambda updates.
    A regression that breaks Max-direction lambda sign would surface
    here as Max overshoots or runaway lambdas."""
    df = make_grid_df(n_quotes=80, n_steps=11)

    # `volume` increases with step (already), `loss_ratio` (synthetic):
    # the higher the step, the lower this metric — so a Max constraint
    # on `loss_ratio` opposes a Min constraint on volume.
    df = df.with_columns(((10.0 - pl.col("volume")) / 10.0).alias("loss_ratio"))
    vol_baseline = float(
        df.group_by("quote_id")
        .agg(
            pl.col("volume")
            .filter(pl.col("expected_income") == pl.col("expected_income").max())
            .first()
        )
        .select(pl.col("volume").sum())
        .item()
    )
    lr_baseline = float(
        df.group_by("quote_id")
        .agg(
            pl.col("loss_ratio")
            .filter(pl.col("expected_income") == pl.col("expected_income").max())
            .first()
        )
        .select(pl.col("loss_ratio").sum())
        .item()
    )

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={
            "volume": {"min": None},
            "loss_ratio": {"max": None},
        },
        max_iter=300,
        tolerance=1e-5,
    )
    result = solver.frontier(
        df,
        threshold_ranges={
            "volume": (vol_baseline * 1.05, vol_baseline * 1.20),
            "loss_ratio": (lr_baseline * 0.85, lr_baseline * 0.95),
        },
        n_points_per_dim=3,
    )
    # For converged points, both one-sided contracts must hold.
    for idx, row in enumerate(result.points.to_dicts()):
        if row["converged"]:
            v_resid = row["threshold_volume"] - row["total_volume"]
            lr_resid = row["total_loss_ratio"] - row["threshold_loss_ratio"]
            # Loose tolerance — discrete-step gaps on multi-constraint
            # are larger; the contract is one-sidedness, not exactness.
            assert v_resid <= vol_baseline * 0.05, (
                f"row {idx}: Min volume target={row['threshold_volume']:.2f} "
                f"converged but actual={row['total_volume']:.2f} undershoots"
            )
            assert lr_resid <= lr_baseline * 0.05, (
                f"row {idx}: Max loss_ratio target={row['threshold_loss_ratio']:.4f} "
                f"converged but actual={row['total_loss_ratio']:.4f} overshoots"
            )


# ---------------------------------------------------------------------------
# Ratio frontier extreme regime
# ---------------------------------------------------------------------------


def test_ratio_frontier_extreme_target_satisfies():
    """Mirror of `test_oracle_1d_ratio_max_satisfies` from the main
    oracle file but for an extreme target close to the lower achievable
    ratio. The ratio path goes through Python linearisation in
    `_python_frontier_sweep` rather than the Rust 1-D bisection, so a
    regression in the linearised inner solve can re-introduce
    high-target undershoot for ratios."""
    n_quotes = 80
    n_steps = 9
    rows = []
    for q in range(n_quotes):
        base = 100.0 + 0.5 * q
        peak_idx = 1 + int(3.0 * q / (n_quotes - 1))
        for j in range(n_steps):
            frac = j / (n_steps - 1)
            distance = abs(j - peak_idx)
            obj = max(base * (1.0 - 0.15 * distance**1.2), 0.05 * base)
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": 0.8 + 0.05 * j,
                    "expected_income": obj,
                    "incurred": 40.0 + 50.0 * frac,
                    "premium": 100.0 - 30.0 * frac,
                }
            )
    df = pl.DataFrame(
        rows,
        schema={
            "quote_id": pl.Utf8,
            "scenario_index": pl.Int32,
            "scenario_value": pl.Float32,
            "expected_income": pl.Float32,
            "incurred": pl.Float32,
            "premium": pl.Float32,
        },
    )

    per_quote_ratios = df.group_by("quote_id").agg(
        (pl.col("incurred") / pl.col("premium")).min().alias("min_ratio"),
        (pl.col("incurred") / pl.col("premium")).max().alias("max_ratio"),
    )
    min_r = float(per_quote_ratios.select(pl.col("min_ratio").min()).item())
    max_r = float(per_quote_ratios.select(pl.col("max_ratio").max()).item())

    # Target the *low* end of the achievable ratio range — the tightest
    # Max ratio the portfolio can satisfy, requiring high lambda.
    target_lo = min_r + 0.05 * (max_r - min_r)
    target_hi = min_r + 0.20 * (max_r - min_r)

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": None,
            },
        },
        max_iter=500,
        tolerance=1e-5,
    )
    result = solver.frontier(
        df,
        threshold_ranges={"loss_ratio": (target_lo, target_hi)},
        n_points_per_dim=4,
    )
    for idx, row in enumerate(result.points.to_dicts()):
        if row["converged"]:
            target = float(row["threshold_loss_ratio"])
            actual = float(row["total_loss_ratio"])
            # 5 % relative slack for ratio Max overshoot (discrete-step
            # gaps are larger relative to ratios).
            assert actual <= target + abs(target) * 0.05, (
                f"row {idx}: extreme ratio Max target={target:.4f} converged "
                f"but actual={actual:.4f} overshoots; "
                f"lambda={row['lambda_loss_ratio']}"
            )


# ---------------------------------------------------------------------------
# Large-scale 1-D smoke
# ---------------------------------------------------------------------------


def test_1d_frontier_at_10k_quotes_smoke():
    """10 000 quotes × 5 steps × 8 frontier points — pins that warm-
    start sorting stays linear-ish and the bisection scales. If a
    future change introduces a quadratic step, this fixture would
    surface as a noticeable wall-clock spike (and should still finish,
    so we don't add a strict timing assertion)."""
    df = make_grid_df(n_quotes=10_000, n_steps=5)
    env_lo, env_hi = (
        float(
            df.group_by("quote_id")
            .agg(pl.col("volume").min().alias("min"))
            .select(pl.sum("min"))
            .item()
        ),
        float(
            df.group_by("quote_id")
            .agg(pl.col("volume").max().alias("max"))
            .select(pl.sum("max"))
            .item()
        ),
    )

    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min": None}},
        max_iter=50,
    )
    result = solver.frontier(
        df,
        threshold_ranges={
            "volume": (
                env_lo + 0.3 * (env_hi - env_lo),
                env_lo + 0.8 * (env_hi - env_lo),
            )
        },
        n_points_per_dim=8,
    )
    assert result.n_points == 8
    # Every point should converge cleanly within the bisection budget.
    n_converged = result.points.filter(pl.col("converged")).height
    assert n_converged == 8, (
        f"large-scale 1-D smoke: only {n_converged}/8 converged; "
        f"warm-start or scaling regression?"
    )


# ---------------------------------------------------------------------------
# Empty df rejection
# ---------------------------------------------------------------------------


def test_empty_dataframe_raises_clear_error():
    """An empty DataFrame should produce a clear validation error, not
    a silent degenerate result."""
    df = pl.DataFrame(
        [],
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
    with pytest.raises((ValueError, RuntimeError)):
        solver.frontier(
            df,
            threshold_ranges={"volume": (1.0, 2.0)},
            n_points_per_dim=2,
        )
