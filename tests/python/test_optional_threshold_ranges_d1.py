"""Feature D1 - ``threshold_ranges`` becomes optional per constraint.

Today, ``frontier()`` requires a ``threshold_ranges`` entry for **every**
constraint. After D1:

* A constraint with a **numeric** threshold and **NO** ``threshold_ranges``
  entry: hold the threshold at its constructor value across all frontier
  points (effectively a 0-width sweep on that axis).
* A constraint with a ``None`` threshold MUST still have a
  ``threshold_ranges`` entry; otherwise raises (B1 contract preserved).
* All ranges supplied: existing behaviour (no change / no regression).
* No constraints have ranges (i.e. cartesian product would have zero
  axes): raise ``ValueError`` for explicitness.

Behaviour for the omitted-range constraint:

* The threshold is fixed at its constructor value for every frontier point.
* The frontier ``points`` DataFrame still emits ``threshold_<name>``,
  ``total_<name>``, and ``lambda_<name>`` columns for it. The
  ``threshold_<name>`` column is constant (same value at every point).
* If the user provides a swept range for SOME constraints, the cartesian
  product is over only those swept axes — other constraints contribute
  a fixed threshold per point.

These tests pin the post-D1 contract. They will mostly **fail today**
because the current frontier (Python and Rust paths) raise
``"No threshold_range for constraint '<name>'"`` for any constraint
without a matching ``threshold_ranges`` entry. The expected failure mode
is exactly that ValueError — the impl agent's job is to relax the
requirement for numeric thresholds while keeping it for ``None`` markers.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc

from helpers import make_small_df, make_factors

# Reuse the C2 ratio fixture so the mixed sum + ratio tests have a
# realistic ratio constraint with ``incurred`` / ``premium`` columns.
from test_ratio_solve_c2 import make_ratio_solve_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _baseline_volume(df: pl.DataFrame) -> float:
    """Baseline ``volume`` total at scenario_value=1.0 on the standard
    ``make_small_df`` fixture."""
    baseline = df.filter(pl.col("scenario_value") == 1.0)
    return float(baseline["volume"].sum())


def _baseline_loss_ratio(df: pl.DataFrame) -> float:
    """Baseline ``loss_ratio`` mean at scenario_value=1.0 on the standard
    ``make_small_df`` fixture (loss_ratio is a per-quote rate, not a sum,
    so we use the simple mean as the baseline reference)."""
    baseline = df.filter(pl.col("scenario_value") == 1.0)
    return float(baseline["loss_ratio"].mean())


# ---------------------------------------------------------------------------
# 1. OnlineOptimiser.frontier — numeric-threshold constraint with NO range
# ---------------------------------------------------------------------------


class TestOptionalRangeOnlineFrontier:
    """``OnlineOptimiser.frontier()`` accepts constraint dicts where some
    numeric-threshold constraints lack a ``threshold_ranges`` entry. The
    omitted-range constraint contributes a fixed threshold (its
    constructor value) to every frontier point; the cartesian product is
    over only the swept axes."""

    def test_two_sum_constraints_only_one_swept(self):
        """Two constraints; only one in ``threshold_ranges``. The swept
        axis varies, the omitted axis is constant at the constructor
        value, and ``n_points`` equals the swept axis's point count."""
        df = make_small_df(n_quotes=50, n_steps=5)
        fixed_volume = 8000.0
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": fixed_volume},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=100,
        )
        n_points_per_dim = 4
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (1.0, 1.10)},
            n_points_per_dim=n_points_per_dim,
        )
        # Cartesian product is over the single swept axis only.
        assert result.n_points == n_points_per_dim
        pts = result.points
        # points DataFrame height matches the swept-axis cartesian
        # product (1 swept axis ** n_points_per_dim entries).
        assert pts.height == n_points_per_dim**1
        # Both threshold columns must be present, even though only one
        # was swept.
        assert "threshold_volume" in pts.columns
        assert "threshold_loss_ratio" in pts.columns
        assert "total_volume" in pts.columns
        assert "total_loss_ratio" in pts.columns
        assert "lambda_volume" in pts.columns
        assert "lambda_loss_ratio" in pts.columns
        # The omitted-range axis is held constant at the constructor value.
        vol_thresholds = pts["threshold_volume"].to_list()
        assert all(v == pytest.approx(fixed_volume) for v in vol_thresholds), (
            f"threshold_volume must be constant at the constructor value "
            f"{fixed_volume}, got {vol_thresholds}"
        )
        # The swept axis must cover the requested range.
        lr_thresholds = sorted(pts["threshold_loss_ratio"].to_list())
        assert lr_thresholds[0] == pytest.approx(1.0, rel=1e-3)
        assert lr_thresholds[-1] == pytest.approx(1.10, rel=1e-3)

    def test_two_sum_constraints_only_other_one_swept(self):
        """Same as above but the swept axis is the other constraint."""
        df = make_small_df(n_quotes=50, n_steps=5)
        fixed_lr_pct = 1.05
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max_pct": fixed_lr_pct},
            },
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (8000.0, 12000.0)},
            n_points_per_dim=4,
        )
        assert result.n_points == 4
        pts = result.points
        # The omitted-range axis is constant at the constructor value.
        # Note: ``max_pct`` reports the user-supplied fraction, so the
        # column should hold ``fixed_lr_pct`` verbatim, not
        # ``fixed_lr_pct * baseline``.
        lr_thresholds = pts["threshold_loss_ratio"].to_list()
        assert all(v == pytest.approx(fixed_lr_pct) for v in lr_thresholds), (
            f"threshold_loss_ratio must be constant at the constructor "
            f"value {fixed_lr_pct} (user-fraction units, not absolute), "
            f"got {lr_thresholds}"
        )
        # The swept axis covers the requested range.
        vol_thresholds = sorted(pts["threshold_volume"].to_list())
        assert vol_thresholds[0] == pytest.approx(8000.0, rel=1e-3)
        assert vol_thresholds[-1] == pytest.approx(12000.0, rel=1e-3)

    def test_three_constraints_two_swept_one_fixed(self):
        """Three constraints; two in ``threshold_ranges``, one omitted.
        Cartesian product is over the two swept axes; the third is
        constant at the constructor value."""
        df = make_small_df(n_quotes=200, n_steps=5)
        fixed_income_pct = 1.0
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
                "expected_income": {"max_pct": fixed_income_pct},
            },
            max_iter=100,
        )
        n = 3
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (0.85, 0.95),
                "loss_ratio": (1.0, 1.10),
            },
            n_points_per_dim=n,
        )
        # Cartesian product over the two swept axes.
        assert result.n_points == n**2
        pts = result.points
        # points DataFrame height matches the swept-axes cartesian
        # product (2 swept axes ** n_points_per_dim entries).
        assert pts.height == n**2
        # Three threshold columns (one per constraint).
        for col in (
            "threshold_volume",
            "threshold_loss_ratio",
            "threshold_expected_income",
            "total_volume",
            "total_loss_ratio",
            "total_expected_income",
            "lambda_volume",
            "lambda_loss_ratio",
            "lambda_expected_income",
        ):
            assert col in pts.columns, f"missing column '{col}'"
        # The omitted axis is constant.
        income_thresholds = pts["threshold_expected_income"].to_list()
        assert all(v == pytest.approx(fixed_income_pct) for v in income_thresholds), (
            f"threshold_expected_income must be constant at "
            f"{fixed_income_pct} (user-fraction units), got "
            f"{income_thresholds}"
        )
        # The two swept axes cover their ranges.
        vols = sorted(set(pts["threshold_volume"].to_list()))
        lrs = sorted(set(pts["threshold_loss_ratio"].to_list()))
        assert vols[0] == pytest.approx(0.85, rel=1e-3)
        assert vols[-1] == pytest.approx(0.95, rel=1e-3)
        assert lrs[0] == pytest.approx(1.0, rel=1e-3)
        assert lrs[-1] == pytest.approx(1.10, rel=1e-3)

    def test_one_swept_one_fixed_with_absolute_units(self):
        """Mix absolute (``min``) swept axis + relative (``max_pct``)
        fixed axis. Each axis's units must follow its own constraint
        key — even when the axis is held fixed."""
        df = make_small_df(n_quotes=200, n_steps=5)
        baseline_vol = _baseline_volume(df)
        fixed_lr_pct = 1.05
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": baseline_vol * 0.85},
                "loss_ratio": {"max_pct": fixed_lr_pct},
            },
            max_iter=100,
        )
        lo_abs = baseline_vol * 0.85
        hi_abs = baseline_vol * 0.95
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (lo_abs, hi_abs)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        pts = result.points
        # volume is swept on absolute units (matches ``min`` key).
        vols = sorted(pts["threshold_volume"].to_list())
        assert vols[0] == pytest.approx(lo_abs, rel=1e-4)
        assert vols[-1] == pytest.approx(hi_abs, rel=1e-4)
        # loss_ratio is fixed on user-fraction units (matches ``max_pct``).
        lr_thresholds = pts["threshold_loss_ratio"].to_list()
        assert all(v == pytest.approx(fixed_lr_pct) for v in lr_thresholds)

    def test_omitted_axis_total_column_consistent_across_points(self):
        """When the omitted axis is held fixed AND the swept axis varies,
        the ``total_<name>`` column for the fixed axis tracks the
        per-point optimum (still computed) — but the threshold column is
        constant. We don't pin the exact total values (the swept axis
        moves the optimum), only that:

        * ``threshold_<fixed>`` column is constant;
        * ``total_<fixed>`` column is finite and sensible;
        * ``lambda_<fixed>`` column is finite.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (1.0, 1.10)},
            n_points_per_dim=3,
        )
        pts = result.points
        # threshold_volume is constant.
        vol_thresh = pts["threshold_volume"].to_list()
        assert len(set(vol_thresh)) == 1, (
            f"threshold_volume must be constant across all points, "
            f"got distinct values {set(vol_thresh)}"
        )
        # total_volume is finite at every point.
        for v in pts["total_volume"].to_list():
            assert math.isfinite(v), f"total_volume must be finite, got {v}"
        # lambda_volume is finite at every point.
        for v in pts["lambda_volume"].to_list():
            assert math.isfinite(v), f"lambda_volume must be finite, got {v}"


# ---------------------------------------------------------------------------
# 2. OnlineOptimiser.frontier — mixed sum + ratio with optional ranges
# ---------------------------------------------------------------------------


class TestOptionalRangeOnlineRatioFrontier:
    """Mixed sum + ratio constraints; D1 must work for both kinds. The
    Python ratio sweep path mirrors the Rust sum sweep path on the
    optional-range contract."""

    def test_mixed_sum_swept_ratio_fixed(self):
        """Sum constraint swept; ratio constraint fixed (numeric
        threshold, no range)."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        # The fixture's premium baseline is ~the sum of premium at
        # scenario_value=1.0 — sweep premium across a feasible region.
        baseline_premium = float(
            df.filter(pl.col("scenario_value") == 1.0)["premium"].sum()
        )
        fixed_lr_target = 0.70
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": baseline_premium * 0.85},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": fixed_lr_target,
                },
            },
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "premium": (baseline_premium * 0.85, baseline_premium * 0.95),
            },
            n_points_per_dim=3,
        )
        # Cartesian product over the single swept axis only.
        assert result.n_points == 3
        pts = result.points
        # Both constraints emit threshold / total / lambda columns.
        for col in (
            "threshold_premium",
            "threshold_loss_ratio",
            "total_premium",
            "total_loss_ratio",
            "lambda_premium",
            "lambda_loss_ratio",
        ):
            assert col in pts.columns, f"missing column '{col}'"
        # The ratio's threshold is constant at the constructor value.
        lr_thresh = pts["threshold_loss_ratio"].to_list()
        assert all(v == pytest.approx(fixed_lr_target) for v in lr_thresh), (
            f"threshold_loss_ratio must be constant at {fixed_lr_target}, "
            f"got {lr_thresh}"
        )

    def test_mixed_sum_fixed_ratio_swept(self):
        """Sum constraint fixed (no range); ratio constraint swept."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        baseline_premium = float(
            df.filter(pl.col("scenario_value") == 1.0)["premium"].sum()
        )
        fixed_premium = baseline_premium * 0.85
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": fixed_premium},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.70,
                },
            },
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.60, 0.75)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        pts = result.points
        # premium threshold is constant at the constructor value.
        prem_thresh = pts["threshold_premium"].to_list()
        assert all(v == pytest.approx(fixed_premium) for v in prem_thresh), (
            f"threshold_premium must be constant at {fixed_premium}, got {prem_thresh}"
        )
        # loss_ratio sweeps across the requested range.
        lr_thresh = sorted(pts["threshold_loss_ratio"].to_list())
        assert lr_thresh[0] == pytest.approx(0.60, rel=1e-3)
        assert lr_thresh[-1] == pytest.approx(0.75, rel=1e-3)


# ---------------------------------------------------------------------------
# 3. RatebookOptimiser.frontier — same patterns
# ---------------------------------------------------------------------------


class TestOptionalRangeRatebookFrontier:
    """``RatebookOptimiser.frontier()`` mirrors the OnlineOptimiser
    contract for optional ``threshold_ranges`` entries."""

    def test_two_sum_constraints_only_one_swept(self):
        """Ratebook: two constraints; only one in ``threshold_ranges``."""
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        fixed_volume_pct = 0.90
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": fixed_volume_pct},
                "loss_ratio": {"max_pct": 1.05},
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"loss_ratio": (1.0, 1.10)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        pts = result.points
        # Both threshold columns present.
        assert "threshold_volume" in pts.columns
        assert "threshold_loss_ratio" in pts.columns
        assert "total_volume" in pts.columns
        assert "total_loss_ratio" in pts.columns
        # volume axis constant; loss_ratio axis swept.
        vol_thresholds = pts["threshold_volume"].to_list()
        assert all(v == pytest.approx(fixed_volume_pct) for v in vol_thresholds)
        lr_thresholds = sorted(pts["threshold_loss_ratio"].to_list())
        assert lr_thresholds[0] == pytest.approx(1.0, rel=1e-3)
        assert lr_thresholds[-1] == pytest.approx(1.10, rel=1e-3)

    def test_two_sum_constraints_only_other_one_swept(self):
        """Ratebook: swap which axis is swept."""
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        fixed_lr_pct = 1.05
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": fixed_lr_pct},
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.85, 0.95)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        pts = result.points
        lr_thresholds = pts["threshold_loss_ratio"].to_list()
        assert all(v == pytest.approx(fixed_lr_pct) for v in lr_thresholds)
        vol_thresholds = sorted(pts["threshold_volume"].to_list())
        assert vol_thresholds[0] == pytest.approx(0.85, rel=1e-3)
        assert vol_thresholds[-1] == pytest.approx(0.95, rel=1e-3)

    def test_three_constraints_two_swept_one_fixed(self):
        """Ratebook: three constraints, two swept, one fixed."""
        n = 100
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        fixed_income_pct = 1.0
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
                "expected_income": {"max_pct": fixed_income_pct},
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={
                "volume": (0.85, 0.95),
                "loss_ratio": (1.0, 1.10),
            },
            n_points_per_dim=3,
        )
        assert result.n_points == 9  # 3 x 3
        pts = result.points
        income_thresholds = pts["threshold_expected_income"].to_list()
        assert all(v == pytest.approx(fixed_income_pct) for v in income_thresholds)

    def test_ratebook_omitted_axis_emits_total_and_lambda(self):
        """The omitted axis still produces ``total_<name>`` and
        ``lambda_<name>`` columns (per-point — they reflect the optimum
        at each swept-point combination)."""
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"loss_ratio": (1.0, 1.10)},
            n_points_per_dim=3,
        )
        pts = result.points
        assert "total_volume" in pts.columns
        assert "lambda_volume" in pts.columns
        for v in pts["total_volume"].to_list():
            assert math.isfinite(v)
        for v in pts["lambda_volume"].to_list():
            assert math.isfinite(v)


# ---------------------------------------------------------------------------
# 4. None-threshold + range interaction
# ---------------------------------------------------------------------------


class TestOptionalRangeWithNoneThresholds:
    """``None`` thresholds remain frontier-only markers (B1) and STILL
    require a ``threshold_ranges`` entry under D1. Numeric thresholds
    no longer require a range (D1). Mixed combinations must each behave
    according to their own threshold's contract."""

    def test_none_threshold_without_range_raises(self):
        """``None`` threshold with NO range entry → ValueError naming
        the constraint (B1 contract preserved)."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(df, threshold_ranges={}, n_points_per_dim=3)
        # Message must name the offending None constraint.
        assert "volume" in str(exc_info.value)

    def test_none_threshold_with_range_works(self):
        """``None`` threshold WITH a matching range still works (B1)."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (8000.0, 12000.0)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        ts = sorted(result.points["threshold_volume"].to_list())
        assert ts[0] == pytest.approx(8000.0)
        assert ts[-1] == pytest.approx(12000.0)

    def test_mixed_none_with_range_and_numeric_without_range(self):
        """One None-with-range + one numeric-without-range → both behave
        correctly. The None one is swept; the numeric one is held at its
        constructor value."""
        df = make_small_df(n_quotes=50, n_steps=5)
        fixed_volume = 9000.0
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": fixed_volume},  # numeric, no range
                "loss_ratio": {"max": None},  # None, has range
            },
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.5, 0.7)},
            n_points_per_dim=4,
        )
        # n_points equals the swept axis's point count.
        assert result.n_points == 4
        pts = result.points
        # volume is held constant at the constructor value.
        vol_thresholds = pts["threshold_volume"].to_list()
        assert all(v == pytest.approx(fixed_volume) for v in vol_thresholds), (
            f"threshold_volume must be constant at {fixed_volume}, got {vol_thresholds}"
        )
        # loss_ratio sweeps across the requested range.
        lr_thresholds = sorted(pts["threshold_loss_ratio"].to_list())
        assert lr_thresholds[0] == pytest.approx(0.5)
        assert lr_thresholds[-1] == pytest.approx(0.7)

    def test_mixed_none_without_range_and_numeric_without_range_raises(self):
        """A numeric-without-range constraint is fine (D1) BUT a
        None-without-range constraint still raises (B1). The error must
        name the offending None constraint."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},  # numeric, no range — fine
                "loss_ratio": {"max": None},  # None, no range — error
            },
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            # No ranges supplied at all — and there are no swept axes,
            # but the FIRST failure must be the None-without-range one
            # (because that violates B1 unconditionally).
            solver.frontier(df, threshold_ranges={}, n_points_per_dim=3)
        # Must name loss_ratio (the None one); whether volume is named
        # too is implementation-defined, but loss_ratio MUST appear.
        assert "loss_ratio" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Edge cases — empty / no-axes / all-None
# ---------------------------------------------------------------------------


class TestOptionalRangeEdgeCases:
    """Edge cases on the cartesian-product / no-axes path."""

    def test_empty_threshold_ranges_with_only_numeric_constraints_raises(self):
        """Empty ``threshold_ranges={}`` (no axes swept) on a
        constraint dict where every constraint is numeric → ValueError.
        Frontier with no axes is meaningless; the spec says raise for
        explicitness rather than silently degrade to a 1-point all-fixed
        solve.

        The error wording is implementation-defined but should be
        descriptive — at minimum it should mention that no axes are
        being swept (e.g. ``"no swept axes"`` / ``"requires at least
        one threshold_ranges entry"`` / similar)."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=50,
        )
        with pytest.raises(ValueError):
            solver.frontier(df, threshold_ranges={}, n_points_per_dim=3)

    def test_empty_threshold_ranges_single_numeric_constraint_raises(self):
        """Same as above but with a single numeric constraint."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 8000.0}},
            max_iter=50,
        )
        with pytest.raises(ValueError):
            solver.frontier(df, threshold_ranges={}, n_points_per_dim=3)

    def test_ratebook_empty_threshold_ranges_with_numeric_constraints_raises(self):
        """Ratebook mirror of the no-axes-swept rejection."""
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        with pytest.raises(ValueError):
            solver.frontier(
                df,
                factors,
                threshold_ranges={},
                n_points_per_dim=3,
            )

    def test_all_none_thresholds_with_all_ranges_works(self):
        """All constraints have ``None`` thresholds + every constraint
        has a range → existing B1 behaviour (no regression)."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": None},
                "loss_ratio": {"max": None},
            },
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (8000.0, 12000.0),
                "loss_ratio": (0.5, 0.7),
            },
            n_points_per_dim=3,
        )
        # 2D Cartesian product → 9 points.
        assert result.n_points == 9
        pts = result.points
        # Both axes swept across their ranges.
        vol_ts = sorted(set(pts["threshold_volume"].to_list()))
        lr_ts = sorted(set(pts["threshold_loss_ratio"].to_list()))
        assert vol_ts[0] == pytest.approx(8000.0)
        assert vol_ts[-1] == pytest.approx(12000.0)
        assert lr_ts[0] == pytest.approx(0.5)
        assert lr_ts[-1] == pytest.approx(0.7)

    def test_max_total_points_rejects_oversized_grid(self):
        """``max_total_points`` smaller than the cartesian product size
        must raise ``ValueError`` naming both the actual point count
        and the configured cap, so the user can pick a remedy."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(
                df,
                threshold_ranges={"loss_ratio": (1.0, 1.10)},
                n_points_per_dim=10,
                max_total_points=5,
            )
        msg = str(exc_info.value)
        assert "10" in msg and "5" in msg, (
            f"error must mention both n_points_per_dim={{10}} and "
            f"max_total_points={{5}}; got {msg!r}"
        )


# ---------------------------------------------------------------------------
# 6. Existing all-ranges-supplied frontier contract: no regression
# ---------------------------------------------------------------------------


class TestExistingFrontierContractsUnaffected:
    """An all-ranges-supplied frontier must produce the same shape and
    column set as before. This guards against accidental regressions
    when the impl agent relaxes the requirement."""

    def test_online_all_ranges_supplied_2d_shape(self):
        """All-ranges 2D frontier has n^2 points and the canonical
        column set."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.90},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=100,
        )
        n = 4
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (0.85, 1.0),
                "loss_ratio": (1.0, 1.10),
            },
            n_points_per_dim=n,
        )
        assert result.n_points == n * n
        pts = result.points
        for col in (
            "threshold_volume",
            "threshold_loss_ratio",
            "total_objective",
            "total_volume",
            "total_loss_ratio",
            "lambda_volume",
            "lambda_loss_ratio",
            "iterations",
            "converged",
        ):
            assert col in pts.columns, f"missing column '{col}'"

    def test_online_all_ranges_supplied_1d_shape(self):
        """All-ranges 1D frontier has n points with the canonical
        column set."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=5,
        )
        assert result.n_points == 5
        pts = result.points
        assert "threshold_volume" in pts.columns
        assert "total_volume" in pts.columns
        assert "lambda_volume" in pts.columns

    def test_ratebook_all_ranges_supplied_1d_shape(self):
        """Ratebook all-ranges 1D frontier shape unchanged."""
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=4,
        )
        assert result.n_points == 4
        pts = result.points
        assert "threshold_volume" in pts.columns
        assert "total_volume" in pts.columns
        assert "lambda_volume" in pts.columns
