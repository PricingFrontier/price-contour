"""Feature C5 - ratio constraints in ``RatebookOptimiser``.

This file pins the C5 contract: ``RatebookOptimiser.solve()`` and
``RatebookOptimiser.frontier()`` no longer raise ``NotImplementedError``
for ratio constraints; instead they linearise the ratio per
(quote x scenario) using the same ``c_i = num_i - L * denom_i`` recipe
that C2 introduced for the online solver, then hand the synthetic sum
constraint to the existing ratebook coordinate-descent loop.

Linearisation contract recap (carries through from C2)
------------------------------------------------------

For ``Sigma num_i / Sigma denom_i ⟂ L`` the rewrite is
``Sigma (num_i - L * denom_i) ⟂ 0``. Threshold ``L`` derivation:

* ``min`` / ``max`` direction keys: ``L`` is the user-supplied threshold
  verbatim (an absolute ratio target).
* ``min_pct`` / ``max_pct`` direction keys: ``L = pct * baseline_LR``.

Why C5 is "the same trick again". The linearisation is per
(quote x scenario) and doesn't depend on the rating-factor structure —
each quote-step contributes the same synthetic ``c_i`` to the sum
regardless of which factor the coordinate descent is currently sweeping.
The ratebook solver then sees a sum constraint and runs unchanged.

C5 reporting contract
---------------------

* ``result.total_constraints[<ratio_label>]``: the **actual** ratio
  ``Sigma_optimal num / Sigma_optimal denom`` at the optimum (C3
  contract carries through to ratebook).
* ``result.baseline_constraints[<ratio_label>]``: the **actual** baseline
  ratio at scenario_value=1.0.
* For frontier: ``threshold_<label>`` echoes user-units verbatim (per A1
  / B1), and ``total_<label>`` is the actual ratio at each point (per
  C3 + C4). ``lambda_<label>`` is the dual on the linearised
  threshold==0 sum constraint.
* ``factor_tables`` is unchanged: it continues to surface the optimal
  factor values per rating factor, regardless of whether the constraints
  are sum or ratio.

Stubs lifted (regression guard)
-------------------------------

* ``RatebookOptimiser.solve()`` no longer raises ``NotImplementedError``
  for ratio constraints.
* ``RatebookOptimiser.frontier()`` no longer raises.
* ``ApplyOptimiser.apply()`` STILL stubs (C6 territory).
* ``apply_from_grid()`` STILL stubs (C6).
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc
from price_contour.apply import apply_from_grid

# Reuse the C2 fixture helpers so the linearisation path under test is
# the same in shape as the online-solve tests: identical scenario grid,
# identical baseline LR (~0.6484 for incurred/premium, ~0.97 for
# kept/exposed). The ratebook just adds a per-quote rating-factor
# DataFrame on top.
from test_ratio_solve_c2 import (
    RATIO_ABS_SLACK,
    RATIO_RTOL,
    actual_ratio_at_optimum,
    baseline_ratio,
    make_ratio_solve_df,
    make_retention_df,
)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
#
# Ratebook coordinate descent is at least as constrained as the online
# solver — it commits each factor's optimal value before moving to the
# next, so the achievable ratio band per CD sweep can be tighter (less
# flexibility per axis). We reuse the C2 tolerance bands; if a future
# CD impl needs slack we can widen here without touching the C2 file.
RATEBOOK_RATIO_RTOL = RATIO_RTOL  # 6%
RATEBOOK_RATIO_ABS = RATIO_ABS_SLACK  # 0.005

# Tight tolerance for direct-equality reporting checks (C3 contract:
# the reported value must equal the actual ratio computed independently).
REPORT_RTOL = 1e-4
REPORT_ABS = 1e-3


# ---------------------------------------------------------------------------
# Ratebook-specific factor builders (deterministic per n_quotes for stable
# fixture totals)
# ---------------------------------------------------------------------------


def _make_ratebook_factors(n_quotes: int) -> pl.DataFrame:
    """Per-quote rating factors with two columns: region (4 levels) and
    age_band (4 levels).

    Distinct from ``helpers.make_factors`` only in being a free-standing
    helper (we don't need to import the helpers shape here, but it's
    harmless to keep the same convention so any cross-fixture
    interaction is consistent).
    """
    regions = ["North", "South", "East", "West"]
    age_bands = ["18-25", "26-35", "36-50", "51+"]
    return pl.DataFrame(
        {
            "region": [regions[i % len(regions)] for i in range(n_quotes)],
            "age_band": [age_bands[i % len(age_bands)] for i in range(n_quotes)],
        }
    )


def _make_single_level_factors(n_quotes: int) -> pl.DataFrame:
    """Degenerate single-level factors DataFrame used by the edge-case
    test for "single rating factor with one level" — the CD has nothing
    to optimise per group but should still solve and surface a 1-entry
    factor table."""
    return pl.DataFrame({"only_region": ["A"] * n_quotes})


# ---------------------------------------------------------------------------
# 1. Basic solve-with-ratio behaviour
# ---------------------------------------------------------------------------


class TestRatebookRatioBasicSolve:
    """End-to-end solves that exercise the C5 linearisation through the
    ratebook coordinate-descent loop."""

    def test_single_factor_max_absolute_target_binds(self):
        """Single rating factor + ``max`` ratio constraint below baseline.

        The C2 fixture has baseline LR ~0.6484; setting ``max: 0.62``
        forces the constraint to bind. The CD loop has only one factor
        (``region``) so the inner grouped solve is the only optimisation
        — but it still needs to respect the ratio constraint.

        Pin: solver converges, lambda finite + non-negative, actual
        ratio at optimum is ``<= target`` within tolerance.
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        target = 0.62

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": target,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        # Lambda for the ratio is finite and non-negative (max-direction).
        lam = result.lambdas["loss_ratio"]
        assert math.isfinite(lam), f"lambda must be finite, got {lam}"
        assert lam >= -1e-9, (
            f"lambda for max-ratio must be >= 0, got {lam}"
        )

        # C3-through-C5 reporting: total_constraints['loss_ratio'] is
        # the actual ratio at the optimum.
        actual = result.total_constraints["loss_ratio"]
        assert math.isfinite(actual), (
            f"total_constraints['loss_ratio'] must be finite, got {actual}"
        )
        assert (
            actual
            <= target * (1 + RATEBOOK_RATIO_RTOL) + RATEBOOK_RATIO_ABS
        ), (
            f"actual loss ratio {actual} > target {target} + tolerance"
        )

        # Factor table emitted under the rating-factor name (NOT the
        # ratio label — ratio constraints don't appear in factor_tables).
        assert "region" in result.factor_tables
        assert "loss_ratio" not in result.factor_tables, (
            "ratio constraint label must not surface as a factor; "
            "factor_tables track rating factors only"
        )

    def test_single_factor_min_pct_retention_floor(self):
        """``min_pct`` direction with the retention fixture.

        baseline retention ~0.97; ``min_pct: 0.98`` resolves to
        ``L = 0.98 * baseline_retention ~ 0.951``. The constraint binds
        because the income-greedy solver pushes some quotes to high
        scenarios with retention below the floor.
        """
        n = 20
        df = make_retention_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min_pct": 0.98,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        lam = result.lambdas["retention_ratio"]
        assert math.isfinite(lam)
        # min-direction lambda non-negative; C3 contract carries through.
        assert lam >= -1e-9, (
            f"lambda for min-ratio must be >= 0, got {lam}"
        )

        baseline_ret = baseline_ratio(df, "kept", "exposed")
        target_l = 0.98 * baseline_ret
        actual = result.total_constraints["retention_ratio"]
        assert math.isfinite(actual)
        assert (
            actual >= target_l * (1 - RATEBOOK_RATIO_RTOL) - RATEBOOK_RATIO_ABS
        ), (
            f"actual retention {actual} < floor {target_l} (= 0.98 * "
            f"{baseline_ret}) beyond tolerance"
        )

    def test_multiple_factors_cd_runs_with_ratio(self):
        """Multiple rating factors + ratio constraint.

        The CD loop iterates factor-by-factor; each per-factor inner
        solve must honour the linearised ratio. Pin: factor tables
        produced for both factors, ratio constraint satisfied at the
        end of CD, ``cd_iterations >= 1`` (CD ran).
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        target = 0.63

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": target,
                }
            },
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        # Factor tables produced for both factors.
        assert "region" in result.factor_tables
        assert "age_band" in result.factor_tables
        assert len(result.factor_tables["region"]) > 0
        assert len(result.factor_tables["age_band"]) > 0
        # CD ran at least one full sweep.
        assert result.cd_iterations >= 1, (
            f"CD must run at least 1 iteration, got {result.cd_iterations}"
        )

        # Ratio constraint satisfied at the end of CD.
        actual = result.total_constraints["loss_ratio"]
        assert math.isfinite(actual)
        assert (
            actual
            <= target * (1 + RATEBOOK_RATIO_RTOL) + RATEBOOK_RATIO_ABS
        ), (
            f"actual loss ratio {actual} > target {target} + tolerance "
            f"after {result.cd_iterations} CD iterations"
        )


# ---------------------------------------------------------------------------
# 2. C3 reporting contract carries through to ratebook
# ---------------------------------------------------------------------------


class TestRatebookRatioReporting:
    """``result.total_constraints[<ratio_label>]`` and
    ``result.baseline_constraints[<ratio_label>]`` report the actual
    ratio (C3 contract carries through).

    The ratebook does not expose ``result.dataframe`` (it produces
    ``factor_tables`` instead), so we cross-check the reported ratio
    against:

    * the **constraint target** (within tolerance) for binding cases;
    * the **baseline ratio recomputed from the input DataFrame** for
      the baseline check (independent ground truth — does not rely on
      the optimiser at all).
    """

    def test_total_constraints_is_actual_ratio_max(self):
        """``total_constraints['loss_ratio']`` is the actual ratio,
        which sits in the achievable LR band [0.55, 0.75] for this
        fixture and respects the binding ``max`` target.

        The C2 linearised value would sit near zero, so the band check
        unambiguously distinguishes C3 (actual ratio) from a C2-style
        regression that surfaces the linearised total.
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        target = 0.62

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": target,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        reported = result.total_constraints["loss_ratio"]
        # Plausible LR band — guards against the linearised-value
        # regression where the surfaced value sits near zero.
        assert math.isfinite(reported)
        assert 0.55 <= reported <= 0.75, (
            f"reported total_constraints['loss_ratio']={reported} out "
            f"of plausible actual-ratio band [0.55, 0.75]; under the "
            f"C2 linearised contract this would sit near 0, so a value "
            f"outside [0.55, 0.75] is a C5/C3 regression"
        )
        # And the binding constraint must be respected.
        assert reported <= target * (1 + RATEBOOK_RATIO_RTOL) + RATEBOOK_RATIO_ABS

    def test_baseline_constraints_is_actual_baseline_ratio(self):
        """``baseline_constraints['loss_ratio']`` equals the actual
        baseline ratio recomputed independently from the input.

        This is direct-equality (within float-precision tolerance) —
        the wrapper computes ``Sigma_baseline num / Sigma_baseline denom``
        from rows where ``scenario_value == 1.0``, and we recompute the
        same quantity here from the raw DataFrame.
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        reported_baseline = result.baseline_constraints["loss_ratio"]
        recomputed_baseline = baseline_ratio(df, "incurred", "premium")
        assert reported_baseline == pytest.approx(
            recomputed_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C5 (C3 carries through): "
            f"baseline_constraints['loss_ratio']={reported_baseline} "
            f"must equal Sigma_baseline incurred / Sigma_baseline "
            f"premium = {recomputed_baseline}."
        )
        # Sanity: this fixture's baseline LR is ~0.6484.
        assert 0.6 < reported_baseline < 0.7, (
            f"fixture baseline LR {reported_baseline} outside [0.6, 0.7]"
        )

    def test_min_direction_baseline_is_actual_baseline_ratio(self):
        """Same direct-equality check on the retention fixture."""
        n = 20
        df = make_retention_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min": 0.95,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        reported_baseline = result.baseline_constraints["retention_ratio"]
        recomputed_baseline = baseline_ratio(df, "kept", "exposed")
        assert reported_baseline == pytest.approx(
            recomputed_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C5: baseline_constraints['retention_ratio']="
            f"{reported_baseline} must equal {recomputed_baseline}."
        )

    def test_mixed_sum_and_ratio_each_reports_correctly(self):
        """Sum + ratio: the ratio key reports the actual ratio, the sum
        key reports the sum (existing contract — must not regress under
        C5).
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        # Peek baseline volume via a small sum-only ratebook solve.
        peek = pc.RatebookOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        baseline_volume = peek.solve(df, factors).baseline_constraints["premium"]
        volume_floor = 0.85 * baseline_volume
        target = 0.62

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "premium": {"min": volume_floor},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": target,
                },
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        # Ratio key: actual ratio in plausible band.
        ratio_reported = result.total_constraints["loss_ratio"]
        assert math.isfinite(ratio_reported)
        assert 0.55 <= ratio_reported <= 0.75, (
            f"ratio key 'loss_ratio'={ratio_reported} out of plausible "
            f"actual-ratio band; check that ratebook propagates the C3 "
            f"actual-ratio reporting (not the linearised total)"
        )

        # Sum key: sum (>> 1 for premium totals on this fixture; the
        # ratio sits in [0.55, 0.75]). This guards against a key swap
        # where the sum and ratio reportings land on the wrong labels.
        sum_reported = result.total_constraints["premium"]
        assert math.isfinite(sum_reported)
        assert sum_reported > 1.0, (
            f"sum 'premium'={sum_reported} unexpectedly small — "
            f"possible swap with the ratio reporting"
        )

        # Baseline checks: ratio baseline matches the recomputed
        # baseline ratio; sum baseline matches the peek volume.
        assert result.baseline_constraints["loss_ratio"] == pytest.approx(
            baseline_ratio(df, "incurred", "premium"),
            rel=REPORT_RTOL,
            abs=REPORT_ABS,
        )
        assert result.baseline_constraints["premium"] == pytest.approx(
            baseline_volume, rel=REPORT_RTOL, abs=REPORT_ABS
        )


# ---------------------------------------------------------------------------
# 3. Mixed sum + ratio together (both lambdas; both satisfied; factor
#    tables respect both)
# ---------------------------------------------------------------------------


class TestRatebookRatioMixed:
    """Sum + ratio constraints together. Both lambdas present, both
    satisfied at the optimum, factor tables produced and respect both
    constraints jointly."""

    def test_volume_floor_plus_loss_ratio_ceiling(self):
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        peek = pc.RatebookOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        baseline_volume = peek.solve(df, factors).baseline_constraints["premium"]
        volume_floor = 0.85 * baseline_volume
        lr_target = 0.62

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "premium": {"min": volume_floor},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": lr_target,
                },
            },
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)

        # Both lambdas present and finite.
        assert "premium" in result.lambdas
        assert "loss_ratio" in result.lambdas
        for name, lam in result.lambdas.items():
            assert math.isfinite(lam), (
                f"lambda for '{name}' must be finite, got {lam}"
            )

        # Volume floor satisfied (within tolerance).
        actual_volume = result.total_constraints["premium"]
        assert actual_volume >= volume_floor * (1 - RATEBOOK_RATIO_RTOL), (
            f"premium {actual_volume} < floor "
            f"{volume_floor} beyond tolerance"
        )

        # Ratio ceiling satisfied (within tolerance).
        actual_lr = result.total_constraints["loss_ratio"]
        assert (
            actual_lr
            <= lr_target * (1 + RATEBOOK_RATIO_RTOL) + RATEBOOK_RATIO_ABS
        ), (
            f"actual loss_ratio {actual_lr} > target {lr_target} "
            f"beyond tolerance"
        )

        # Factor tables for both rating factors are produced.
        assert "region" in result.factor_tables
        assert "age_band" in result.factor_tables
        # The constraint labels must NOT appear as factor names
        # (factor_tables tracks rating factors only).
        assert "loss_ratio" not in result.factor_tables
        assert "premium" not in result.factor_tables


# ---------------------------------------------------------------------------
# 4. Frontier sweeps with ratio constraints
# ---------------------------------------------------------------------------


class TestRatebookRatioFrontier:
    """Frontier sweeps where at least one constraint is a ratio. The
    contract mirrors C4: ``threshold_<label>`` reports user-units
    verbatim; ``total_<label>`` reports the actual ratio at each point;
    ``lambda_<label>`` is the dual on the linearised threshold==0
    sum constraint."""

    def test_frontier_max_absolute_range_sweeps_user_units(self):
        """``max: None`` ratio + ``threshold_ranges={'loss_ratio': (lo, hi)}``
        sweeps L over [lo, hi] absolute; threshold column echoes verbatim."""
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        lo, hi = 0.55, 0.75
        n_pts = 4

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"loss_ratio": (lo, hi)},
            n_points_per_dim=n_pts,
        )

        assert result.n_points == n_pts
        pts = result.points

        # Required columns.
        for col in (
            "threshold_loss_ratio",
            "total_loss_ratio",
            "lambda_loss_ratio",
            "total_objective",
        ):
            assert col in pts.columns, (
                f"frontier points missing required column '{col}'"
            )

        # Threshold column reports user-supplied absolute units verbatim.
        thresholds = sorted(pts["threshold_loss_ratio"].to_list())
        assert thresholds[0] == pytest.approx(lo, rel=1e-6)
        assert thresholds[-1] == pytest.approx(hi, rel=1e-6)

        # total_<label> is the actual ratio (not the linearised total).
        # Plausible LR band guards against a swap that surfaces the
        # linearised total (which sits near 0).
        totals = pts["total_loss_ratio"].to_list()
        for tot in totals:
            assert math.isfinite(tot)
            assert 0.4 < tot < 0.8, (
                f"total_loss_ratio={tot} out of plausible LR band; "
                f"under C2 linearised contract this would sit near 0"
            )

        # All lambdas finite and non-negative (max-direction).
        lambdas = pts["lambda_loss_ratio"].to_list()
        for lam in lambdas:
            assert math.isfinite(lam) and lam >= -1e-9, (
                f"lambda_loss_ratio must be finite and >= 0; got {lam}"
            )

    def test_frontier_min_absolute_range_retention(self):
        """``min: None`` direction on a retention ratio. Sweep below
        baseline retention so every point binds."""
        n = 20
        df = make_retention_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        baseline_ret = baseline_ratio(df, "kept", "exposed")
        lo, hi = 0.93, 0.96
        n_pts = 3

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"retention_ratio": (lo, hi)},
            n_points_per_dim=n_pts,
        )

        assert result.n_points == n_pts
        pts = result.points.sort("threshold_retention_ratio")

        # Threshold column in absolute retention units.
        thresholds = pts["threshold_retention_ratio"].to_list()
        assert min(thresholds) == pytest.approx(lo)
        assert max(thresholds) == pytest.approx(hi)

        # Sanity: baseline >= hi so the test premise holds.
        assert baseline_ret >= hi, (
            f"fixture invariant changed: baseline retention "
            f"{baseline_ret} < upper sweep target {hi}"
        )

        # Every point's actual retention >= target within tolerance.
        totals = pts["total_retention_ratio"].to_list()
        for thr, tot in zip(thresholds, totals):
            assert math.isfinite(tot)
            assert tot >= thr * (1 - RATEBOOK_RATIO_RTOL) - RATEBOOK_RATIO_ABS, (
                f"min frontier point: actual retention {tot} < target "
                f"{thr} beyond tolerance"
            )

    def test_frontier_max_pct_threshold_axis_is_fractional(self):
        """``max_pct: None`` + ``threshold_ranges={'loss_ratio': (0.95, 1.05)}``
        sweeps L over ``[0.95 * baseline_LR, 1.05 * baseline_LR]``
        internally; the reported ``threshold_<label>`` is the user-supplied
        fraction verbatim (per A1 unification).
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        baseline_lr = baseline_ratio(df, "incurred", "premium")
        lo_pct, hi_pct = 0.95, 1.05
        n_pts = 3

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"loss_ratio": (lo_pct, hi_pct)},
            n_points_per_dim=n_pts,
        )

        pts = result.points
        thresholds = sorted(pts["threshold_loss_ratio"].to_list())

        # Threshold column reports user-supplied fractions verbatim,
        # NOT ``frac * baseline_lr``.
        assert thresholds[0] == pytest.approx(lo_pct, rel=1e-6), (
            f"max_pct threshold {thresholds[0]} should be the "
            f"user-supplied fraction {lo_pct}, not "
            f"frac * baseline ({lo_pct * baseline_lr})"
        )
        assert thresholds[-1] == pytest.approx(hi_pct, rel=1e-6)

        # Belt-and-braces: assert recorded thresholds are NOT
        # ``frac * baseline_lr``. C2 fixture has baseline_lr ~ 0.65 so
        # frac * baseline ~ [0.62, 0.68], distinct from [0.95, 1.05].
        assert abs(baseline_lr - 1.0) > 0.1, (
            f"fixture invariant changed: baseline_lr {baseline_lr} too "
            f"close to 1.0 to distinguish fraction from absolute scale"
        )
        for thr in pts["threshold_loss_ratio"].to_list():
            assert thr > 0.85, (
                f"max_pct frontier threshold {thr} looks like "
                f"frac * baseline_lr ({baseline_lr}); A1 unification "
                f"appears regressed"
            )

    def test_frontier_min_pct_threshold_axis_is_fractional(self):
        """``min_pct: None`` symmetric to ``max_pct``."""
        n = 20
        df = make_retention_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        lo_pct, hi_pct = 0.97, 1.00
        n_pts = 3

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min_pct": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"retention_ratio": (lo_pct, hi_pct)},
            n_points_per_dim=n_pts,
        )

        thresholds = sorted(result.points["threshold_retention_ratio"].to_list())
        assert thresholds[0] == pytest.approx(lo_pct, rel=1e-6)
        assert thresholds[-1] == pytest.approx(hi_pct, rel=1e-6)

    def test_frontier_mixed_sum_plus_ratio_cartesian(self):
        """Mixed 2D frontier over (volume sum, loss_ratio max). The
        Cartesian product produces ``n^2`` points; each axis's
        threshold column reports its own user-units; both lambdas and
        totals appear on the result.

        Mirrors ``test_ratio_frontier_c4`` semantics for the online
        solver but routed through the ratebook CD loop.
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        # Anchor the volume range from a peek solve.
        peek = pc.RatebookOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        baseline_volume = peek.solve(df, factors).baseline_constraints["premium"]
        vol_lo = 0.80 * baseline_volume
        vol_hi = 0.95 * baseline_volume

        # Loss ratio sweep brackets the binding band.
        lr_lo, lr_hi = 0.61, 0.68

        n_pts = 3
        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "premium": {"min": None},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                },
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={
                "premium": (vol_lo, vol_hi),
                "loss_ratio": (lr_lo, lr_hi),
            },
            n_points_per_dim=n_pts,
        )

        # Cartesian product of n_pts x n_pts.
        assert result.n_points == n_pts * n_pts
        pts = result.points

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
        vol_thresholds = sorted(set(pts["threshold_premium"].to_list()))
        assert vol_thresholds[0] == pytest.approx(vol_lo, rel=1e-4)
        assert vol_thresholds[-1] == pytest.approx(vol_hi, rel=1e-4)
        lr_thresholds = sorted(set(pts["threshold_loss_ratio"].to_list()))
        assert lr_thresholds[0] == pytest.approx(lr_lo, rel=1e-4)
        assert lr_thresholds[-1] == pytest.approx(lr_hi, rel=1e-4)

        # Both lambdas finite at every point; ratio's lambda non-negative.
        for lam in pts["lambda_loss_ratio"].to_list():
            assert math.isfinite(lam)
            assert lam >= -1e-9
        for lam in pts["lambda_premium"].to_list():
            assert math.isfinite(lam)

        # total_loss_ratio is the actual ratio at each point.
        for tot in pts["total_loss_ratio"].to_list():
            assert math.isfinite(tot)
            assert 0.4 < tot < 0.8, (
                f"total_loss_ratio={tot} out of plausible band"
            )

        # total_premium is on the volume scale (>> 1).
        for vol in pts["total_premium"].to_list():
            assert vol > 1.0, (
                f"total_premium={vol} unexpectedly small — possible "
                f"swap with the ratio column"
            )


# ---------------------------------------------------------------------------
# 5. Stub-removed regression guard for the partial scope of C5
# ---------------------------------------------------------------------------


class TestRatebookRatioStubsRemoved:
    """C5 lights up ratebook solve + frontier for ratio constraints.

    All other ratio entry points (apply, apply_from_grid) continue to
    stub. This class is the regression guard for the partial scope: each
    test pins exactly one boundary so a future change that flips a stub
    state-machine is immediately surfaced.
    """

    def test_ratebook_solve_with_ratio_no_longer_raises(self):
        """C5 ON: ``RatebookOptimiser.solve()`` with a ratio constraint
        no longer raises ``NotImplementedError`` — it solves."""
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        # MUST NOT raise NotImplementedError.
        try:
            result = solver.solve(df, factors)
        except NotImplementedError as e:
            pytest.fail(
                f"RatebookOptimiser.solve() with a ratio constraint "
                f"must not raise NotImplementedError under C5; got: {e}"
            )
        assert result is not None
        assert "loss_ratio" in result.total_constraints

    def test_ratebook_frontier_with_ratio_no_longer_raises(self):
        """C5 ON: ``RatebookOptimiser.frontier()`` with a ratio constraint
        no longer raises ``NotImplementedError`` — it sweeps."""
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        try:
            result = solver.frontier(
                df,
                factors,
                threshold_ranges={"loss_ratio": (0.55, 0.75)},
                n_points_per_dim=3,
            )
        except NotImplementedError as e:
            pytest.fail(
                f"RatebookOptimiser.frontier() with a ratio constraint "
                f"must not raise NotImplementedError under C5; got: {e}"
            )
        assert result is not None
        assert result.n_points == 3
        assert "threshold_loss_ratio" in result.points.columns

    # NOTE: the C6 stub-pin tests for ApplyOptimiser.apply and
    # apply_from_grid were removed when C6 lit up the apply ratio paths.
    # The replacement regression guards live in
    # ``test_ratio_apply_c6.py::TestApplyOptimiserRatioStubsRemoved``.


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestRatebookRatioEdgeCases:
    """Edge cases: zero baseline denominator, single-level factor,
    None-threshold + frontier range."""

    def test_zero_baseline_denominator_for_max_pct_raises(self):
        """``Sigma_baseline denom == 0`` makes baseline LR undefined.

        For ``max_pct`` mode this must raise ``ValueError`` at solve
        setup with a message naming the constraint label and signalling
        the zero-denominator condition. Same contract as C2 solve, C4
        frontier — pin that ratebook propagates the rejection (it will
        if the impl reuses the linearisation helper).
        """
        n_quotes = 10
        n_steps = 5
        rows = []
        mults = [0.8 + 0.1 * j for j in range(n_steps)]
        for q in range(n_quotes):
            for j, mult in enumerate(mults):
                # premium == 0 at baseline scenario; non-zero elsewhere
                # so the column has variation but baseline totals are 0.
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
        factors = _make_ratebook_factors(n_quotes)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": 1.0,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df, factors)
        msg = str(exc_info.value)
        assert "loss_ratio" in msg, (
            f"error {msg!r} must name the constraint label"
        )
        assert (
            "0" in msg
            or "zero" in msg.lower()
            or "denominator" in msg.lower()
            or "baseline" in msg.lower()
        ), (
            f"error {msg!r} must signal zero-denominator / "
            f"undefined-baseline condition"
        )

    def test_zero_baseline_denominator_ok_for_absolute_max(self):
        """Absolute ``max`` mode does NOT depend on baseline_LR. Even
        if Sigma_baseline denom == 0, setup must NOT raise (mirrors C2
        contract for the online solver).
        """
        n_quotes = 10
        n_steps = 5
        rows = []
        mults = [0.8 + 0.1 * j for j in range(n_steps)]
        for q in range(n_quotes):
            for j, mult in enumerate(mults):
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
        factors = _make_ratebook_factors(n_quotes)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.7,  # absolute, no baseline_LR needed
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        # MUST NOT raise — absolute mode has no baseline_LR dependency.
        result = solver.solve(df, factors)
        assert "loss_ratio" in result.total_constraints

    def test_single_level_factor_solves_degenerately(self):
        """A single rating factor with one level has nothing to
        optimise per group — every quote is in the same group, so the
        CD's grouped solve degenerates to a global solve. The result
        must still be well-formed: ratio constraint satisfied, factor
        table has one entry, lambdas finite.
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_single_level_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                }
            },
            factor_columns=[["only_region"]],
            max_cd_iterations=2,
            max_iter=100,
        )
        result = solver.solve(df, factors)

        # Factor table has exactly one entry (the single level).
        assert "only_region" in result.factor_tables
        assert len(result.factor_tables["only_region"]) == 1, (
            f"single-level factor should produce a 1-entry table; got "
            f"{result.factor_tables['only_region']}"
        )

        # Ratio constraint reported and finite.
        actual = result.total_constraints["loss_ratio"]
        assert math.isfinite(actual)
        # Lambda finite.
        lam = result.lambdas["loss_ratio"]
        assert math.isfinite(lam)

    def test_none_threshold_solve_raises(self):
        """A ``None`` threshold on a ratio constraint with ``solve()``
        must raise ``ValueError`` (B1 contract) naming the constraint and
        pointing at ``frontier()``. Pin the ratebook propagates this.
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df, factors)
        msg = str(exc_info.value)
        assert "loss_ratio" in msg, (
            f"None-threshold solve error must name the constraint "
            f"label; got: {msg!r}"
        )

    def test_none_threshold_frontier_with_range_succeeds(self):
        """``max: None`` + matching range is the canonical
        frontier-only ratebook case (mirrors B1 / C4)."""
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"loss_ratio": (0.55, 0.75)},
            n_points_per_dim=4,
        )
        assert result.n_points == 4
        thresholds = result.points["threshold_loss_ratio"].to_list()
        assert min(thresholds) == pytest.approx(0.55)
        assert max(thresholds) == pytest.approx(0.75)

    def test_none_threshold_frontier_without_range_raises(self):
        """``max: None`` without a matching range must raise
        ``ValueError`` (B1 contract)."""
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(
                df,
                factors,
                threshold_ranges={},
                n_points_per_dim=3,
            )
        # Must name the offending constraint label.
        assert "loss_ratio" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Summary integration
# ---------------------------------------------------------------------------


class TestRatebookRatioSummary:
    """``summary()`` round-trips ratio specs and surfaces the C3
    actual-ratio reporting in metrics. Per the C3 propagation pattern
    this is automatic: ``summary()`` reads ``result.total_constraints``
    which already reports the actual ratio under C5.
    """

    def test_summary_round_trips_ratio_spec(self):
        """``summary()['params']['constraints']`` JSON round-trips the
        ratio dict."""
        import json

        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.65,
            }
        }

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints=constraints,
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = solver.solve(df, factors)
        summary = solver.summary(result)

        blob = summary["params"]["constraints"]
        decoded = json.loads(blob)
        assert decoded == constraints, (
            f"round-tripped constraints {decoded} != original "
            f"{constraints}"
        )

    def test_summary_metrics_constraint_total_is_actual_ratio(self):
        """``summary()['metrics']['constraint_loss_ratio_total']`` is
        the actual ratio at the optimum (C3 propagated through ratebook
        summary path).
        """
        n = 20
        df = make_ratio_solve_df(n_quotes=n, n_steps=5)
        factors = _make_ratebook_factors(n)

        solver = pc.RatebookOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            factor_columns=[["region"]],
            max_cd_iterations=2,
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df, factors)
        metrics = solver.summary(result)["metrics"]

        # The metrics value matches the result's total_constraints value
        # exactly — they're both reading the same C3-shaped data.
        assert metrics["constraint_loss_ratio_total"] == pytest.approx(
            result.total_constraints["loss_ratio"],
            rel=REPORT_RTOL,
            abs=REPORT_ABS,
        ), (
            f"summary metrics['constraint_loss_ratio_total']="
            f"{metrics['constraint_loss_ratio_total']} must equal "
            f"result.total_constraints['loss_ratio']="
            f"{result.total_constraints['loss_ratio']}"
        )
        # And the actual ratio sits in the plausible LR band — guards
        # against a regression where summary surfaces the linearised
        # value while result.total_constraints is correctly C3-shaped.
        assert 0.55 <= metrics["constraint_loss_ratio_total"] <= 0.75
