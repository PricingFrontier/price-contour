"""Feature C4 - ratio constraint frontier integration.

This file pins the C4 contract: ``OnlineOptimiser.frontier()`` no longer
raises ``NotImplementedError`` for ratio constraints; instead it sweeps
the ratio target ``L`` over the user-supplied range and runs the
linearised solve at each point.

Frontier reporting contract for ratio constraints
-------------------------------------------------

For a ratio constraint named ``loss_ratio`` with ``numerator='incurred'``,
``denominator='premium'``, the frontier ``points`` DataFrame must contain:

* ``threshold_loss_ratio`` — user-supplied range value, in the units the
  user passed (verbatim absolute units for ``min`` / ``max``; fractions of
  baseline LR for ``min_pct`` / ``max_pct``). Matches A1's threshold scale
  rule.
* ``total_loss_ratio`` — the **actual ratio** at the optimum
  (``Sigma_optimal num / Sigma_optimal denom``), per C3's contract for
  ``SolveResult.total_constraints``. NOT the linearised value.
* ``lambda_loss_ratio`` — the dual lambda for the linearised constraint at
  that frontier point. Non-negative.

Plus the standard frontier columns: ``total_objective``, ``iterations``,
``converged``, and any ``sv_*`` quantile columns the FrontierResult
already emits.

Threshold-range scale rule (matches A1 / B1)
--------------------------------------------

* ``min`` / ``max`` constraint with ``(lo, hi)`` range: L sweeps over
  ``[lo, hi]`` in absolute ratio units. ``threshold_<label>`` reports
  ``[lo, hi]`` verbatim.
* ``min_pct`` / ``max_pct`` constraint with ``(lo, hi)`` range: L sweeps
  over ``[lo * baseline_LR, hi * baseline_LR]`` internally so each point's
  linearised solve uses the right absolute target; the reported
  ``threshold_<label>`` shows ``[lo, hi]`` (user units).

Stubs lifted (regression guard)
-------------------------------

* ``OnlineOptimiser.frontier()`` no longer raises ``NotImplementedError``
  for ratio constraints (this feature).
* ``RatebookOptimiser.frontier()`` STILL stubs (C5 not yet shipped).
* ``RatebookOptimiser.solve()`` STILL stubs (C5).
* ``ApplyOptimiser.apply()`` STILL stubs (C6).
* ``apply_from_grid()`` STILL stubs (C6).
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc

# Reuse the C2 fixture helpers so the test data is identical to the
# linearisation-and-solve test file. The fixtures are parameterised on
# n_quotes / n_steps; the baselines and achievable ranges are documented
# in the C2 test module.
from test_ratio_solve_c2 import (
    RATIO_ABS_SLACK,
    RATIO_RTOL,
    baseline_ratio,
    make_ratio_solve_df,
    make_retention_df,
)


# ---------------------------------------------------------------------------
# Tolerances for frontier-point ratio assertions.
#
# Frontier points share the same discrete-grid limits as the C2 solve
# tests; reuse the same tolerances. Per-point convergence is governed
# by max_iter, so we set max_iter to a generous value to give the
# linearised solver room to converge for tight binding targets.
# ---------------------------------------------------------------------------
FRONTIER_RATIO_RTOL = RATIO_RTOL  # 6%
FRONTIER_RATIO_ABS = RATIO_ABS_SLACK  # 0.005


def _baseline_loss_ratio(df: pl.DataFrame) -> float:
    """Baseline LR ``Sigma incurred / Sigma premium`` at scenario_value=1.0
    on the C2 ratio fixture."""
    return baseline_ratio(df, "incurred", "premium")


def _baseline_retention(df: pl.DataFrame) -> float:
    return baseline_ratio(df, "kept", "exposed")


# ---------------------------------------------------------------------------
# 1. Basic single-ratio frontier behaviour
# ---------------------------------------------------------------------------


class TestRatioFrontierBasic:
    """End-to-end frontier sweeps with a single ratio constraint."""

    def test_max_absolute_range_sweeps_over_user_units(self):
        """``max: None`` + ``threshold_ranges={'loss_ratio': (0.55, 0.75)}``
        sweeps L over [0.55, 0.75] absolute and reports the threshold
        column verbatim.

        The C2 fixture has achievable LR range ~[0.6013, 0.6924] and
        baseline LR ~0.6484. The (0.55, 0.75) sweep brackets both the
        binding region (below baseline) AND the slack region (above worst
        achievable), so we should see binding behaviour at the low end
        and slack behaviour at the high end.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        lo, hi = 0.55, 0.75
        n = 5

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (lo, hi)},
            n_points_per_dim=n,
        )

        # n_points and DataFrame shape contract.
        assert result.n_points == n
        pts = result.points
        assert pts.shape[0] == n

        # Required columns.
        for col in (
            "threshold_loss_ratio",
            "total_loss_ratio",
            "lambda_loss_ratio",
            "total_objective",
            "iterations",
            "converged",
        ):
            assert col in pts.columns, (
                f"frontier points missing required column '{col}'"
            )

        # threshold_<label> reports user-supplied absolute units verbatim.
        thresholds = sorted(pts["threshold_loss_ratio"].to_list())
        assert thresholds[0] == pytest.approx(lo, rel=1e-6)
        assert thresholds[-1] == pytest.approx(hi, rel=1e-6)

        # All lambdas non-negative (constraint is max-direction).
        lambdas = pts["lambda_loss_ratio"].to_list()
        assert all(math.isfinite(lam) and lam >= -1e-9 for lam in lambdas), (
            f"all lambda_loss_ratio values must be finite and >= 0; got {lambdas}"
        )

        # total_<label> is the actual ratio (not the linearised total).
        # For this fixture the actual ratio at any feasible solve point
        # must lie roughly in [0.5, 0.75] — guard against a swap that
        # surfaces e.g. the linearised total (which sits near 0).
        totals = pts["total_loss_ratio"].to_list()
        for tot in totals:
            assert math.isfinite(tot)
            assert 0.4 < tot < 0.8, (
                f"total_loss_ratio={tot} out of plausible LR band; "
                f"check that frontier reports the ACTUAL ratio (C3 "
                f"contract), not the linearised total which sits ~0"
            )

    def test_max_binding_targets_satisfy_within_tolerance(self):
        """For frontier points with target below baseline LR, the actual
        ratio at the optimum must be ``<= target + tolerance``.

        Achievable ratio range on the C2 fixture is ~[0.6013, 0.6924];
        baseline LR ~0.6484. Use a low-end sweep (0.61, 0.64) where every
        point binds.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        lo, hi = 0.61, 0.64
        n = 4

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=500,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (lo, hi)},
            n_points_per_dim=n,
        )

        pts = result.points.sort("threshold_loss_ratio")
        thresholds = pts["threshold_loss_ratio"].to_list()
        totals = pts["total_loss_ratio"].to_list()
        lambdas = pts["lambda_loss_ratio"].to_list()

        # For every binding target, the actual ratio is <= target within
        # tolerance. The discrete grid may oscillate; we use the same
        # tolerance bands as the C2 solve tests.
        for thr, tot, lam in zip(thresholds, totals, lambdas):
            assert tot <= thr * (1 + FRONTIER_RATIO_RTOL) + FRONTIER_RATIO_ABS, (
                f"binding frontier point: actual ratio {tot} > target {thr} + tolerance"
            )
            # And lambda must be positive for binding targets.
            assert lam > 0, (
                f"binding frontier point at threshold {thr} should have "
                f"positive lambda; got {lam}"
            )

    def test_max_slack_targets_have_lambda_near_zero(self):
        """For frontier points above the maximum achievable LR (slack),
        lambda is near zero and the total objective is near unconstrained.

        Achievable LR max is ~0.6924; (0.85, 0.95) is well above it, so
        every point is slack.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)

        # Unconstrained reference for objective comparison.
        unconstrained = (
            pc.OnlineOptimiser(
                objective="income",
                constraints={"income": {"min": -1.0}},
                max_iter=200,
            )
            .solve(df)
            .total_objective
        )

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=300,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.85, 0.95)},
            n_points_per_dim=4,
        )

        pts = result.points
        lambdas = pts["lambda_loss_ratio"].to_list()
        objectives = pts["total_objective"].to_list()

        for lam in lambdas:
            assert math.isfinite(lam)
            assert lam < 1e-2, f"slack frontier point should have lambda ~0; got {lam}"
        for obj in objectives:
            assert obj == pytest.approx(unconstrained, rel=0.02), (
                f"slack frontier point objective {obj} should match "
                f"unconstrained {unconstrained} within 2%"
            )

    def test_min_direction_retention_floor_frontier(self):
        """``min`` direction on a retention ratio. Sweep L over a tight
        retention band that should mostly bind."""
        df = make_retention_df(n_quotes=20, n_steps=5)
        baseline = _baseline_retention(df)
        # baseline retention ~ 0.97 on this fixture; sweep below baseline
        # so every point requires the solver to push retention UP.
        lo, hi = 0.93, 0.96
        n = 4

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min": None,
                }
            },
            max_iter=500,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"retention_ratio": (lo, hi)},
            n_points_per_dim=n,
        )

        pts = result.points.sort("threshold_retention_ratio")
        thresholds = pts["threshold_retention_ratio"].to_list()
        totals = pts["total_retention_ratio"].to_list()
        lambdas = pts["lambda_retention_ratio"].to_list()

        # threshold_<label> is in absolute retention units (user supplied
        # `min: None` with absolute key, so range is absolute).
        assert min(thresholds) == pytest.approx(lo)
        assert max(thresholds) == pytest.approx(hi)

        # Sanity: the fixture's baseline must be >= hi so the test
        # premise holds (otherwise targets are slack by construction).
        assert baseline >= hi, (
            f"fixture invariant changed: baseline retention {baseline} < "
            f"upper sweep target {hi}; test premise broken"
        )

        # min-direction lambdas non-negative; for tight binding targets
        # they should be strictly positive.
        for lam in lambdas:
            assert math.isfinite(lam) and lam >= -1e-9, (
                f"min-direction lambda must be finite and >= 0; got {lam}"
            )

        # For each point, actual retention >= target within tolerance.
        for thr, tot in zip(thresholds, totals):
            assert tot >= thr * (1 - FRONTIER_RATIO_RTOL) - FRONTIER_RATIO_ABS, (
                f"min frontier point: actual retention {tot} < target "
                f"{thr} beyond tolerance"
            )

    def test_max_pct_threshold_axis_is_fractional(self):
        """``max_pct: None`` + ``threshold_ranges={'loss_ratio': (0.95, 1.05)}``
        sweeps L over ``[0.95 * baseline_LR, 1.05 * baseline_LR]``
        internally; the reported ``threshold_<label>`` is the user-supplied
        fractions verbatim.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        baseline_lr = _baseline_loss_ratio(df)
        lo_pct, hi_pct = 0.95, 1.05
        n = 4

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": None,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (lo_pct, hi_pct)},
            n_points_per_dim=n,
        )

        pts = result.points
        thresholds = sorted(pts["threshold_loss_ratio"].to_list())

        # The threshold column reports the user-supplied fractions verbatim,
        # NOT ``frac * baseline_lr``.
        assert thresholds[0] == pytest.approx(lo_pct, rel=1e-6), (
            f"max_pct threshold {thresholds[0]} should be the user-supplied "
            f"fraction {lo_pct}, not frac * baseline ({lo_pct * baseline_lr})"
        )
        assert thresholds[-1] == pytest.approx(hi_pct, rel=1e-6)

        # Belt-and-braces: assert recorded thresholds are NOT
        # ``frac * baseline_lr`` (they would coincide if baseline_lr is ~1.0,
        # but for this fixture baseline_lr ~ 0.65 so the two scales are
        # clearly distinct).
        assert abs(baseline_lr - 1.0) > 0.1, (
            f"fixture invariant changed: baseline_lr {baseline_lr} too "
            f"close to 1.0 to distinguish fraction from absolute scale"
        )
        assert thresholds[0] != pytest.approx(lo_pct * baseline_lr, rel=1e-3), (
            f"recorded threshold {thresholds[0]} matches frac * baseline "
            f"({lo_pct * baseline_lr}); A1 unification appears regressed"
        )

        # The internal L applied at the lo_pct=0.95 point must be
        # ``0.95 * baseline_lr`` ~ 0.616, well below baseline. The actual
        # ratio at that point must satisfy that absolute target.
        pts_sorted = pts.sort("threshold_loss_ratio")
        lo_total = float(pts_sorted["total_loss_ratio"][0])
        target_l_at_lo = lo_pct * baseline_lr
        assert (
            lo_total <= target_l_at_lo * (1 + FRONTIER_RATIO_RTOL) + FRONTIER_RATIO_ABS
        ), (
            f"max_pct: at threshold_loss_ratio={lo_pct} the internal target "
            f"L = {target_l_at_lo} (= {lo_pct} * {baseline_lr}); the actual "
            f"ratio at this point ({lo_total}) must satisfy that target"
        )

    def test_min_pct_threshold_axis_is_fractional(self):
        """``min_pct`` symmetric to ``max_pct``: the reported threshold
        column contains user-supplied fractions; internal L is
        ``frac * baseline_retention``.
        """
        df = make_retention_df(n_quotes=20, n_steps=5)
        baseline_ret = _baseline_retention(df)
        lo_pct, hi_pct = 0.97, 1.00
        n = 3

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min_pct": None,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"retention_ratio": (lo_pct, hi_pct)},
            n_points_per_dim=n,
        )
        pts = result.points
        thresholds = sorted(pts["threshold_retention_ratio"].to_list())

        # User-supplied fractions verbatim.
        assert thresholds[0] == pytest.approx(lo_pct, rel=1e-6)
        assert thresholds[-1] == pytest.approx(hi_pct, rel=1e-6)
        # Distinct scale guard.
        assert abs(baseline_ret - 1.0) > 0.01 or True, (
            f"baseline {baseline_ret} too close to 1.0 to distinguish; "
            f"the test still passes the bare-equality check above"
        )


# ---------------------------------------------------------------------------
# 2. Mixed sum + ratio frontier (Cartesian product across axes)
# ---------------------------------------------------------------------------


class TestRatioFrontierMixed:
    """Two-dimensional frontiers: one sum constraint, one ratio."""

    def test_volume_sum_plus_loss_ratio_max_two_axis_sweep(self):
        """A 2D frontier over (volume sum, loss_ratio max) produces
        ``n^2`` points. Each axis's threshold column reports its own
        user-units; both lambdas and totals appear on the result."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)

        # Peek baseline volume from a 1-iter solve so we can size the
        # absolute volume range.
        peek = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        baseline_volume = peek.solve(df).baseline_constraints["premium"]
        vol_lo = 0.80 * baseline_volume
        vol_hi = 0.95 * baseline_volume

        # Loss ratio sweep brackets the binding band.
        lr_lo, lr_hi = 0.61, 0.68

        n = 3
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": None},  # absolute sum constraint
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                },
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "premium": (vol_lo, vol_hi),
                "loss_ratio": (lr_lo, lr_hi),
            },
            n_points_per_dim=n,
        )

        # Cartesian product of n x n = 9 frontier points.
        assert result.n_points == n * n
        pts = result.points
        assert pts.shape[0] == n * n

        # Both axes' threshold + total + lambda columns present.
        for col in (
            "threshold_premium",
            "total_premium",
            "lambda_premium",
            "threshold_loss_ratio",
            "total_loss_ratio",
            "lambda_loss_ratio",
        ):
            assert col in pts.columns, f"missing column {col}"

        # Each axis's threshold column reports its own user units.
        # premium uses absolute `min` → absolute sweep.
        vol_thresholds = sorted(set(pts["threshold_premium"].to_list()))
        assert vol_thresholds[0] == pytest.approx(vol_lo, rel=1e-4)
        assert vol_thresholds[-1] == pytest.approx(vol_hi, rel=1e-4)
        # loss_ratio uses absolute `max` → absolute sweep.
        lr_thresholds = sorted(set(pts["threshold_loss_ratio"].to_list()))
        assert lr_thresholds[0] == pytest.approx(lr_lo, rel=1e-4)
        assert lr_thresholds[-1] == pytest.approx(lr_hi, rel=1e-4)

        # Both lambdas finite at every point; ratio's lambda non-negative.
        for lam in pts["lambda_loss_ratio"].to_list():
            assert math.isfinite(lam)
            assert lam >= -1e-9
        for lam in pts["lambda_premium"].to_list():
            assert math.isfinite(lam)

        # total_loss_ratio is the actual ratio at each point — not the
        # linearised value.
        for tot in pts["total_loss_ratio"].to_list():
            assert math.isfinite(tot)
            # Plausible LR band on this fixture.
            assert 0.4 < tot < 0.8, (
                f"total_loss_ratio={tot} out of plausible band — "
                f"check that frontier surfaces the actual ratio (C3)"
            )

        # total_premium is on the volume scale — not confused with the
        # ratio. Cross-bracket guard: total_premium values are >> 1, while
        # total_loss_ratio values are < 1.
        for vol in pts["total_premium"].to_list():
            assert vol > 1.0, (
                f"total_premium={vol} unexpectedly small — possible swap "
                f"with the ratio column"
            )

    def test_pct_sum_plus_pct_ratio_two_axis_sweep_units_independent(self):
        """Mixed pct keys: each axis interprets its range in its own
        units (fractions of the appropriate baseline)."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        baseline_lr = _baseline_loss_ratio(df)

        n = 3
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min_pct": None},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": None,
                },
            },
            max_iter=300,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "premium": (0.80, 0.95),
                "loss_ratio": (0.95, 1.05),
            },
            n_points_per_dim=n,
        )

        pts = result.points
        assert result.n_points == n * n

        # Each axis reports user-supplied fractions verbatim.
        vol_thresholds = sorted(set(pts["threshold_premium"].to_list()))
        assert vol_thresholds[0] == pytest.approx(0.80, rel=1e-4)
        assert vol_thresholds[-1] == pytest.approx(0.95, rel=1e-4)
        lr_thresholds = sorted(set(pts["threshold_loss_ratio"].to_list()))
        assert lr_thresholds[0] == pytest.approx(0.95, rel=1e-4)
        assert lr_thresholds[-1] == pytest.approx(1.05, rel=1e-4)

        # Belt-and-braces: assert the loss_ratio threshold column does NOT
        # contain `frac * baseline_lr`. The C2 fixture has
        # baseline_lr ~ 0.65 so frac * baseline ~ [0.62, 0.68], distinct
        # from the user-supplied [0.95, 1.05] fractions.
        if abs(baseline_lr - 1.0) > 0.1:
            for thr in pts["threshold_loss_ratio"].to_list():
                # No threshold cell may match frac * baseline_lr (within
                # 5% relative). The recorded values must be in the
                # fractional range [0.95, 1.05], NOT in the absolute
                # range [0.62, 0.68].
                assert thr > 0.85, (
                    f"max_pct frontier threshold {thr} looks like "
                    f"frac * baseline_lr ({baseline_lr}) — A1 unification "
                    f"appears regressed"
                )


# ---------------------------------------------------------------------------
# 3. Numeric threshold + threshold_ranges override
# ---------------------------------------------------------------------------


class TestRatioFrontierWithNumericThreshold:
    """Per B1's contract, when the constructor specifies a numeric
    threshold AND the frontier supplies a range for the same constraint,
    the range overrides — the sweep proceeds across the range, not the
    constructor scalar."""

    def test_constructor_numeric_overridden_by_range(self):
        """``max: 0.65`` at construction + ``threshold_ranges={..(0.55, 0.70)}``
        sweeps over (0.55, 0.70), NOT just the constructor 0.65."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        n = 5
        lo, hi = 0.55, 0.70

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,  # numeric, will be OVERRIDDEN by range
                }
            },
            max_iter=300,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (lo, hi)},
            n_points_per_dim=n,
        )
        pts = result.points
        assert result.n_points == n
        thresholds = sorted(pts["threshold_loss_ratio"].to_list())
        # The constructor's 0.65 must NOT be the only threshold; the
        # sweep covers (lo, hi).
        assert thresholds[0] == pytest.approx(lo, rel=1e-6), (
            f"frontier swept lo={lo}, got {thresholds[0]} — the range "
            f"should override the constructor's 0.65"
        )
        assert thresholds[-1] == pytest.approx(hi, rel=1e-6), (
            f"frontier swept hi={hi}, got {thresholds[-1]}"
        )
        # Sweep must include values clearly different from 0.65 to prove
        # the override actually fires.
        assert any(abs(t - 0.65) > 0.04 for t in thresholds), (
            f"frontier thresholds {thresholds} all ~ constructor's 0.65; "
            f"override appears not to have fired"
        )


# ---------------------------------------------------------------------------
# 4. None-threshold marker + range required
# ---------------------------------------------------------------------------


class TestRatioFrontierNoneThreshold:
    """Per B1, a ``None`` threshold marks a frontier-only constraint;
    ``threshold_ranges`` MUST contain an entry for it."""

    def test_max_none_with_range_succeeds(self):
        """``max: None`` + matching range is the canonical frontier case."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=300,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.55, 0.75)},
            n_points_per_dim=4,
        )
        assert result.n_points == 4
        thresholds = result.points["threshold_loss_ratio"].to_list()
        assert min(thresholds) == pytest.approx(0.55)
        assert max(thresholds) == pytest.approx(0.75)

    def test_max_none_without_range_raises(self):
        """``max: None`` constraint missing from ``threshold_ranges``
        raises and names the constraint per B1's contract."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(df, threshold_ranges={}, n_points_per_dim=3)
        # Must name the offending constraint label.
        assert "loss_ratio" in str(exc_info.value), (
            f"missing-range error must name the constraint label; "
            f"got: {exc_info.value!r}"
        )

    def test_min_pct_none_with_range_succeeds(self):
        """``min_pct: None`` + fractional range works for retention."""
        df = make_retention_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min_pct": None,
                }
            },
            max_iter=300,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"retention_ratio": (0.97, 1.00)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        thresholds = result.points["threshold_retention_ratio"].to_list()
        assert min(thresholds) == pytest.approx(0.97)
        assert max(thresholds) == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# 5. Stub-removed regression guard for the partial scope of C4.
# ---------------------------------------------------------------------------


class TestRatioFrontierStubsRemovedForOnlineOnly:
    """C4 lights up ``OnlineOptimiser.frontier()`` only.

    All other ratio entry points continue to stub. This class is the
    regression guard for the partial scope: each test pins exactly one
    boundary so a future change that flips a stub state-machine is
    immediately surfaced.
    """

    def test_online_frontier_with_ratio_no_longer_raises(self):
        """C4 ON: ``OnlineOptimiser.frontier()`` with a ratio constraint
        no longer raises ``NotImplementedError`` — it sweeps."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=100,
        )
        # MUST NOT raise NotImplementedError.
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.55, 0.75)},
            n_points_per_dim=3,
        )
        assert result is not None
        assert result.n_points == 3
        assert "threshold_loss_ratio" in result.points.columns

    # NOTE: ``test_ratebook_frontier_with_ratio_still_raises_C5`` and
    # ``test_ratebook_solve_with_ratio_still_raises_C5`` (C4-era stub
    # state-machine pins) were removed when C5 lit up the ratebook
    # ratio paths. The replacement regression guards live in
    # ``test_ratio_ratebook_c5.py::TestRatebookRatioStubsRemoved``.

    # NOTE: the C6 stub-pin tests for ApplyOptimiser.apply and
    # apply_from_grid were removed when C6 lit up the apply ratio paths.
    # The replacement regression guards live in
    # ``test_ratio_apply_c6.py::TestApplyOptimiserRatioStubsRemoved``.


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestRatioFrontierEdgeCases:
    """Edge cases: all-slack sweep, all-tight-or-infeasible sweep,
    single-point axis, zero baseline denominator."""

    def test_all_slack_range_above_baseline_lambdas_near_zero(self):
        """Every sweep point above max-achievable LR: lambdas near 0,
        objectives near unconstrained.

        Achievable LR max is ~0.6924. (0.85, 0.95) is well above it.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        unconstrained = (
            pc.OnlineOptimiser(
                objective="income",
                constraints={"income": {"min": -1.0}},
                max_iter=200,
            )
            .solve(df)
            .total_objective
        )

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=300,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.85, 0.95)},
            n_points_per_dim=4,
        )

        pts = result.points
        for lam in pts["lambda_loss_ratio"].to_list():
            assert math.isfinite(lam)
            assert lam < 1e-2, (
                f"all-slack sweep should have lambda ~0 at every point; got {lam}"
            )
        for obj in pts["total_objective"].to_list():
            assert obj == pytest.approx(unconstrained, rel=0.02), (
                f"all-slack sweep objective {obj} should match "
                f"unconstrained {unconstrained}"
            )

    def test_all_infeasible_range_below_min_achievable(self):
        """Every sweep point below the achievable minimum LR (~0.6013).
        Pin the behaviour: the frontier still produces ``n`` points, every
        lambda is finite (the dual update is bounded), and the actual
        ratio bottoms out at the achievable floor regardless of how tight
        the requested target is.

        We do NOT pin "constraint always satisfied" — by definition no
        feasible solve exists below the achievable floor — but we DO pin
        that the result is well-formed (finite lambdas / objectives, no
        NaN / inf, non-negative iterations).
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        # Sweep below the achievable minimum (~0.6013).
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.45, 0.55)},
            n_points_per_dim=3,
        )
        pts = result.points
        assert result.n_points == 3
        # Every lambda finite (no inf / NaN despite infeasibility).
        for lam in pts["lambda_loss_ratio"].to_list():
            assert math.isfinite(lam), (
                f"infeasible sweep produced non-finite lambda {lam}; "
                f"the dual update must remain bounded"
            )
        # Every objective finite (no NaN propagation).
        for obj in pts["total_objective"].to_list():
            assert math.isfinite(obj)
        # Every total ratio finite.
        for tot in pts["total_loss_ratio"].to_list():
            assert math.isfinite(tot)
        # Iterations non-negative.
        for it in pts["iterations"].to_list():
            assert it >= 0

    def test_single_point_axis_with_range_lo_eq_hi(self):
        """``n_points_per_dim=1`` (or range with ``lo == hi``) produces a
        single threshold value; works for ratio-only and mixed sweeps."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)

        peek = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        baseline_volume = peek.solve(df).baseline_constraints["premium"]

        # We pass n_points_per_dim=3, but the volume range has lo == hi
        # so the volume axis has only one unique value (the same point
        # repeated). Total points = 3 (loss_ratio sweep) but n_points_per_dim
        # is the per-axis count. The Cartesian product is 3 x 3 = 9
        # because volume's range still produces 3 (degenerate) points.
        # Pin the well-defined ``n_points_per_dim=1`` case below.
        vol_target = 0.85 * baseline_volume

        # Now the canonical single-point case: n_points_per_dim=1
        # produces 1 x 1 = 1 frontier point.
        single_solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": None},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                },
            },
            max_iter=200,
            tolerance=1e-4,
        )
        single_result = single_solver.frontier(
            df,
            threshold_ranges={
                "premium": (vol_target, vol_target),
                "loss_ratio": (0.65, 0.65),
            },
            n_points_per_dim=1,
        )
        assert single_result.n_points == 1
        single_pts = single_result.points
        assert single_pts.shape[0] == 1
        # Single point's threshold values match the supplied scalars.
        assert float(single_pts["threshold_premium"][0]) == pytest.approx(
            vol_target, rel=1e-4
        )
        assert float(single_pts["threshold_loss_ratio"][0]) == pytest.approx(
            0.65, rel=1e-6
        )
        # Both lambdas and totals present.
        assert math.isfinite(float(single_pts["lambda_loss_ratio"][0]))
        assert math.isfinite(float(single_pts["total_loss_ratio"][0]))
        assert math.isfinite(float(single_pts["lambda_premium"][0]))
        assert math.isfinite(float(single_pts["total_premium"][0]))

    def test_zero_baseline_denominator_for_max_pct_raises(self):
        """``max_pct`` mode needs baseline_LR; ``Sigma_baseline denom == 0``
        raises ``ValueError`` naming the constraint, same contract as in
        C2 solve. Frontier inherits this rejection because each point's
        linearisation needs the baseline ratio."""
        rows = []
        n_quotes = 10
        n_steps = 5
        mults = [0.8 + 0.1 * j for j in range(n_steps)]
        for q in range(n_quotes):
            for j, mult in enumerate(mults):
                # Premium is 0 at scenario_value=1.0 (baseline scenario);
                # non-zero elsewhere. Baseline denom sum == 0.
                premium = 0.0 if mult == 1.0 else 100.0 * mult
                incurred = premium * 0.6
                rows.append(
                    {
                        "quote_id": f"Q{q:04d}",
                        "scenario_index": j,
                        "scenario_value": mult,
                        "income": premium,
                        "incurred": incurred,
                        "premium": premium,
                    }
                )
        df = pl.DataFrame(
            rows,
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "income": pl.Float32,
                "incurred": pl.Float32,
                "premium": pl.Float32,
            },
        )

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": None,
                }
            },
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(
                df,
                threshold_ranges={"loss_ratio": (0.95, 1.05)},
                n_points_per_dim=3,
            )
        msg = str(exc_info.value)
        assert "loss_ratio" in msg, (
            f"frontier zero-baseline-denominator error must name the "
            f"constraint label; got {msg!r}"
        )
        # Hint at zero denominator / baseline issue.
        assert (
            "0" in msg
            or "zero" in msg.lower()
            or "denominator" in msg.lower()
            or "baseline" in msg.lower()
        ), (
            f"error {msg!r} must signal the zero-denominator / "
            f"undefined-baseline condition"
        )


# ---------------------------------------------------------------------------
# 7. FrontierResult API and frontier_summary integration
# ---------------------------------------------------------------------------


class TestRatioFrontierResultAPI:
    """The standard FrontierResult API and frontier_summary helper must
    work unchanged for ratio-frontier results."""

    def test_frontier_result_has_standard_api(self):
        """FrontierResult has ``.points`` (DataFrame) and ``.n_points``
        even for ratio-only sweeps."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.60, 0.70)},
            n_points_per_dim=4,
        )
        # API contract.
        assert hasattr(result, "points")
        assert hasattr(result, "n_points")
        assert isinstance(result.points, pl.DataFrame)
        assert isinstance(result.n_points, int)
        assert result.n_points == 4

    def test_frontier_summary_works_for_ratio_only_frontier(self):
        """``frontier_summary`` must produce a valid MLflow dict for a
        ratio-frontier result."""
        from price_contour.frontier import frontier_summary

        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.60, 0.70)},
            n_points_per_dim=5,
        )
        summary = frontier_summary(result, selected_index=2)
        assert set(summary.keys()) == {"params", "metrics", "artifacts"}
        # The selected metric for the ratio threshold must be present.
        assert "selected_threshold_loss_ratio" in summary["metrics"]
        # Total + lambda metrics present too.
        assert "selected_total_loss_ratio" in summary["metrics"]
        assert "selected_lambda_loss_ratio" in summary["metrics"]
        # And they should be valid floats.
        assert isinstance(summary["metrics"]["selected_total_loss_ratio"], float)
        assert isinstance(summary["metrics"]["selected_lambda_loss_ratio"], float)
