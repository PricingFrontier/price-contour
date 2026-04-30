"""Feature A1 — frontier threshold scale follows constraint key.

After A1 (and the B1 fix-pass that unified reporting units), the
``points["threshold_<name>"]`` column reports values in the **same units
the user supplied** in ``threshold_ranges``:

* Constraint uses ``min`` / ``max`` (absolute) → ``threshold_ranges``
  values are absolute. The frontier ``points["threshold_<name>"]`` column
  contains absolute units (the same numbers the user passed).
* Constraint uses ``min_pct`` / ``max_pct`` (relative) →
  ``threshold_ranges`` values are fractions of baseline. The
  ``threshold_<name>`` column contains those fractions verbatim, NOT
  ``frac × baseline``. (The internal frontier solve still operates on
  absolute thresholds — only the reported column changes.)

This rule applies to both numeric AND ``None`` threshold constraints,
in Online and Ratebook frontier paths.

Old ``min_abs`` / ``max_abs`` keys must error in the frontier path too.
"""

from __future__ import annotations

import pytest

import price_contour as pc
from helpers import make_small_df


# Match against str(ValueError) — the message must mention both the
# removed key and its replacement.
RE_MIN_ABS_REMOVED = r"(?s)min_abs.*\bmin\b|\bmin\b.*min_abs"
RE_MAX_ABS_REMOVED = r"(?s)max_abs.*\bmax\b|\bmax\b.*max_abs"


def _baseline_volume(df) -> float:
    """Helper: read baseline volume off a fast 1-iter solve."""
    peek = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"volume": {"min_pct": 1.0}},
        max_iter=1,
    )
    return peek.solve(df).baseline_constraints["volume"]


def _baseline_loss_ratio(df) -> float:
    peek = pc.OnlineOptimiser(
        objective="expected_income",
        constraints={"loss_ratio": {"max_pct": 1.0}},
        max_iter=1,
    )
    return peek.solve(df).baseline_constraints["loss_ratio"]


# ---------------------------------------------------------------------------
# 1. Frontier with absolute (`min` / `max`) constraint
# ---------------------------------------------------------------------------


class TestFrontierAbsoluteThresholds:
    def test_frontier_absolute_min_threshold_axis_is_absolute(self):
        """`min` constraint + `threshold_ranges=(8000, 12000)` → frontier
        ``threshold_volume`` column ranges from 8000 to 12000."""
        df = make_small_df(n_quotes=200, n_steps=5)
        baseline_vol = _baseline_volume(df)

        # Pick lo, hi as absolute volumes that bracket a feasible region.
        lo_abs = baseline_vol * 0.80
        hi_abs = baseline_vol * 0.95
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": lo_abs}},
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (lo_abs, hi_abs)},
            n_points_per_dim=5,
        )

        thresholds = sorted(result.points["threshold_volume"].to_list())
        # The recorded threshold axis is in absolute units. The min and
        # max of the column must equal the configured (lo_abs, hi_abs).
        assert thresholds[0] == pytest.approx(lo_abs, rel=1e-4)
        assert thresholds[-1] == pytest.approx(hi_abs, rel=1e-4)

    def test_frontier_absolute_max_threshold_axis_is_absolute(self):
        df = make_small_df(n_quotes=200, n_steps=5)
        baseline_lr = _baseline_loss_ratio(df)

        lo_abs = baseline_lr * 1.0
        hi_abs = baseline_lr * 1.20
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": hi_abs}},
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (lo_abs, hi_abs)},
            n_points_per_dim=5,
        )
        thresholds = sorted(result.points["threshold_loss_ratio"].to_list())
        assert thresholds[0] == pytest.approx(lo_abs, rel=1e-4)
        assert thresholds[-1] == pytest.approx(hi_abs, rel=1e-4)


# ---------------------------------------------------------------------------
# 2. Frontier with relative (`min_pct` / `max_pct`) constraint
# ---------------------------------------------------------------------------


class TestFrontierPctThresholds:
    def test_frontier_pct_min_threshold_axis_is_fractional(self):
        """`min_pct` + `threshold_ranges=(0.85, 0.95)` → recorded
        ``threshold_volume`` ranges over those user-supplied fractions
        verbatim (NOT ``frac × baseline``).

        The internal solve still operates on absolute thresholds
        (``frac × baseline``), but the reported column matches the units
        the user passed in ``threshold_ranges``."""
        df = make_small_df(n_quotes=200, n_steps=5)
        # Read baseline only so we can sanity-check that the recorded
        # values are NOT frac × baseline (they should equal frac alone).
        baseline_vol = _baseline_volume(df)

        lo, hi = 0.85, 0.95
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": lo}},
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (lo, hi)},
            n_points_per_dim=5,
        )
        thresholds = sorted(result.points["threshold_volume"].to_list())
        # The threshold column reports the user-supplied fractions
        # verbatim — the same units passed in `threshold_ranges`.
        assert thresholds[0] == pytest.approx(lo, rel=1e-3)
        assert thresholds[-1] == pytest.approx(hi, rel=1e-3)
        # Belt-and-braces: assert the recorded value is NOT
        # `frac × baseline` (which would only match if `baseline_vol`
        # happened to be ~1.0).
        assert thresholds[0] < baseline_vol / 2 or baseline_vol < 2.0, (
            f"reported threshold {thresholds[0]} looks like frac × baseline "
            f"(baseline {baseline_vol})"
        )

    def test_frontier_pct_max_threshold_axis_is_fractional(self):
        df = make_small_df(n_quotes=200, n_steps=5)
        baseline_lr = _baseline_loss_ratio(df)

        lo, hi = 1.00, 1.10
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max_pct": hi}},
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (lo, hi)},
            n_points_per_dim=5,
        )
        thresholds = sorted(result.points["threshold_loss_ratio"].to_list())
        # User-supplied fractions verbatim, not `frac × baseline`.
        assert thresholds[0] == pytest.approx(lo, rel=1e-3)
        assert thresholds[-1] == pytest.approx(hi, rel=1e-3)
        # Sanity guard: the recorded values are clearly not on the
        # absolute scale of `baseline_lr`.
        # Skip this guard if the baseline happens to be near 1.0 — the
        # two reportings would coincide in that degenerate case.
        if abs(baseline_lr - 1.0) > 0.1:
            assert thresholds[-1] != pytest.approx(hi * baseline_lr, rel=1e-3), (
                f"reported threshold {thresholds[-1]} matches the "
                f"frac × baseline {hi * baseline_lr}; the unit unification "
                f"appears to have regressed"
            )


# ---------------------------------------------------------------------------
# 3. Mixed: one absolute, one relative in the same frontier sweep
# ---------------------------------------------------------------------------


class TestFrontierMixed:
    def test_mixed_frontier_each_axis_uses_its_own_constraint_units(self):
        """Two constraints, one absolute (`min`) and one relative
        (`max_pct`). The frontier must honour each axis's units
        independently — the reported column matches the user's input
        units per axis."""
        df = make_small_df(n_quotes=200, n_steps=5)
        baseline_vol = _baseline_volume(df)

        vol_lo_abs = baseline_vol * 0.85
        vol_hi_abs = baseline_vol * 0.95
        lr_lo_pct, lr_hi_pct = 1.00, 1.10

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": vol_lo_abs},  # absolute
                "loss_ratio": {"max_pct": lr_hi_pct},  # relative
            },
            max_iter=200,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (vol_lo_abs, vol_hi_abs),
                "loss_ratio": (lr_lo_pct, lr_hi_pct),
            },
            n_points_per_dim=3,
        )
        vols = sorted(set(result.points["threshold_volume"].to_list()))
        lrs = sorted(set(result.points["threshold_loss_ratio"].to_list()))

        # volume axis is absolute units (the user passed absolute).
        assert vols[0] == pytest.approx(vol_lo_abs, rel=1e-4)
        assert vols[-1] == pytest.approx(vol_hi_abs, rel=1e-4)
        # loss_ratio axis was specified as fractions; recorded values
        # are those fractions verbatim (same units the user supplied).
        assert lrs[0] == pytest.approx(lr_lo_pct, rel=1e-3)
        assert lrs[-1] == pytest.approx(lr_hi_pct, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. Frontier with old `min_abs` / `max_abs` keys must error
# ---------------------------------------------------------------------------


class TestFrontierThreeMixedConstraints:
    """Issue 4 / item 4: three-axis frontier with one absolute ``min``,
    one relative ``max_pct``, and one absolute ``max``. Each
    ``threshold_<name>`` axis must follow its own constraint's units,
    and the ``total_<name>`` columns must align with the same names
    (cross-checks the Issue 2 ordering fix in a frontier setting).
    """

    def test_three_mixed_constraints_frontier_has_correct_axis_units(self):
        df = make_small_df(n_quotes=200, n_steps=5)

        # Use loss_ratio twice? No — we need three distinct columns.
        # The standard helper exposes 'volume', 'expected_income', and
        # 'loss_ratio'. Use volume as the absolute min, loss_ratio as
        # the relative max_pct, and expected_income as the absolute max.
        baseline_vol = _baseline_volume(df)

        # Read baseline expected_income off a 1-iter solve.
        peek = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"expected_income": {"max_pct": 1.0}},
            max_iter=1,
        )
        baseline_inc = peek.solve(df).baseline_constraints["expected_income"]

        # Pick ranges sized to each axis and well within the feasible
        # region.
        vol_lo_abs = baseline_vol * 0.85  # absolute
        vol_hi_abs = baseline_vol * 0.95
        lr_lo_pct, lr_hi_pct = 1.00, 1.10  # fractional
        # expected_income is also the objective; an absolute max
        # ceiling well above its baseline keeps the constraint
        # technically valid without binding.
        inc_lo_abs = baseline_inc * 1.10
        inc_hi_abs = baseline_inc * 1.25

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": vol_lo_abs},  # absolute
                "loss_ratio": {"max_pct": lr_hi_pct},  # relative
                "expected_income": {"max": inc_hi_abs},  # absolute
            },
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (vol_lo_abs, vol_hi_abs),
                "loss_ratio": (lr_lo_pct, lr_hi_pct),
                "expected_income": (inc_lo_abs, inc_hi_abs),
            },
            n_points_per_dim=3,  # 3^3 = 27 frontier points
        )
        points = result.points

        # All required columns are present.
        for col in (
            "threshold_volume",
            "threshold_loss_ratio",
            "threshold_expected_income",
            "total_volume",
            "total_loss_ratio",
            "total_expected_income",
        ):
            assert col in points.columns

        # Volume axis is absolute units (user supplied absolute).
        vols = sorted(set(points["threshold_volume"].to_list()))
        assert vols[0] == pytest.approx(vol_lo_abs, rel=1e-4)
        assert vols[-1] == pytest.approx(vol_hi_abs, rel=1e-4)

        # loss_ratio axis was specified as fractions; recorded values
        # are those fractions verbatim (user-supplied units).
        lrs = sorted(set(points["threshold_loss_ratio"].to_list()))
        assert lrs[0] == pytest.approx(lr_lo_pct, rel=1e-3)
        assert lrs[-1] == pytest.approx(lr_hi_pct, rel=1e-3)

        # expected_income axis is absolute units.
        incs = sorted(set(points["threshold_expected_income"].to_list()))
        assert incs[0] == pytest.approx(inc_lo_abs, rel=1e-4)
        assert incs[-1] == pytest.approx(inc_hi_abs, rel=1e-4)

        # Cross-bracket guard for total_<name> alignment (Issue 2):
        # ``volume`` baseline ~ O(100); ``expected_income`` baseline
        # ~ O(10000). They differ by ~100x, so a swap between
        # ``total_volume`` and ``total_expected_income`` would be
        # immediately visible.
        # (We skip the volume vs. loss_ratio cross-bracket here
        # because those baselines happen to land at similar scales
        # in the synthetic helper data; the dedicated three-distinct-
        # scale regression test in test_constraint_ordering_a1.py
        # covers that pair more strongly.)
        assert baseline_inc > baseline_vol * 50, (
            f"helper data invariant changed: expected_income baseline "
            f"{baseline_inc} not >> volume baseline {baseline_vol}"
        )
        total_vol = points["total_volume"]
        total_inc = points["total_expected_income"]

        # total_volume must stay in the volume scale — well below
        # the expected_income baseline.
        assert total_vol.max() < baseline_inc / 10, (
            f"total_volume max {total_vol.max()} unexpectedly close to "
            f"expected_income baseline {baseline_inc} — possible "
            f"column transposition"
        )
        # total_expected_income must stay in the income scale —
        # well above the volume baseline.
        assert total_inc.min() > baseline_vol * 5, (
            f"total_expected_income min {total_inc.min()} unexpectedly "
            f"close to volume baseline {baseline_vol} — possible "
            f"column transposition"
        )


class TestFrontierRejectsOldAbsKeys:
    def test_frontier_with_min_abs_raises(self):
        """Frontier called with old ``min_abs`` constraint must raise
        ``ValueError`` mentioning both old and new key names."""
        df = make_small_df(n_quotes=50, n_steps=5)
        # Constructor must reject before frontier even runs.
        with pytest.raises(ValueError, match=RE_MIN_ABS_REMOVED):
            solver = pc.OnlineOptimiser(
                objective="expected_income",
                constraints={"volume": {"min_abs": 50.0}},
                max_iter=50,
            )
            # Guard: if construction did not raise, frontier must.
            solver.frontier(
                df,
                threshold_ranges={"volume": (40.0, 60.0)},
                n_points_per_dim=3,
            )

    def test_frontier_with_max_abs_raises(self):
        df = make_small_df(n_quotes=50, n_steps=5)
        with pytest.raises(ValueError, match=RE_MAX_ABS_REMOVED):
            solver = pc.OnlineOptimiser(
                objective="expected_income",
                constraints={"loss_ratio": {"max_abs": 1.0}},
                max_iter=50,
            )
            solver.frontier(
                df,
                threshold_ranges={"loss_ratio": (0.8, 1.0)},
                n_points_per_dim=3,
            )
