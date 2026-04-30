"""Feature A1 — regression tests for constraint-name HashMap ordering bug.

Pre-A1, ``parse_constraints`` in ``crates/price-contour/src/solver_py.rs``
and the frontier setup in ``crates/price-contour/src/frontier_py.rs``
iterated the user's constraint ``HashMap`` in arbitrary order, while
downstream code (``argmax.rs``) assumes ``specs[k]`` aligns with
``grid.constraints[k]``. The fix walks ``grid.constraint_names`` in
order so the spec vector matches the grid column ordering.

This test would fail if ``parse_constraints`` iterated ``&constraints``
(HashMap iteration order) instead of ``&grid.constraint_names``: under
the buggy version, the lambda dict and total_constraints dict returned
to Python would silently report values for the wrong constraint name,
because ``constraint_names`` (built from ``specs.iter().map(s.name)``)
would be in HashMap iteration order while
``inner.lambdas`` / ``inner.total_constraints`` would be in grid order.

The constraints are crafted so each column has very different totals.
A swap between any two would push the asserted floor/ceiling
relationship to fail, and the matching ``total_*`` and ``threshold_*``
columns in the frontier ``points`` DataFrame would land in the wrong
bracket.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc


# ---------------------------------------------------------------------------
# Data factory: three-column long-format frame with strongly-different scales.
# ---------------------------------------------------------------------------


def _make_three_column_df(
    n_quotes: int = 80,
    n_steps: int = 5,
) -> pl.DataFrame:
    """Build a long-format DataFrame with three constraint columns whose
    totals are at very different scales:

    * ``volume``        ~ O(1)        — fraction-of-portfolio, sums to ~tens.
    * ``apple_count``   ~ O(1000)     — large absolute totals.
    * ``zebra_metric``  ~ O(0.01)     — small absolute totals.

    Note ``apple_count`` is alphabetically first; ``volume`` is in the
    middle alphabetically; ``zebra_metric`` is alphabetically last.

    Each quote has a logistic conversion curve so the columns trade off
    against the objective in a non-degenerate way.
    """
    rows = []
    mults = [0.8 + 0.1 * j for j in range(n_steps)]
    for q in range(n_quotes):
        elasticity = 1.5 + 3.5 * q / n_quotes
        base = 80.0 + 40.0 * q / n_quotes
        for j, mult in enumerate(mults):
            conversion = 1.0 / (1.0 + math.exp(elasticity * (mult - 1.0)))
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": mult,
                    "expected_income": base * mult * conversion,
                    # Distinct scales chosen so a swap between any two
                    # columns visibly violates the per-column assertions.
                    "volume": conversion,
                    "apple_count": 1000.0 * conversion,
                    "zebra_metric": 0.01 * conversion,
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
            "apple_count": pl.Float32,
            "zebra_metric": pl.Float32,
        },
    )


# ---------------------------------------------------------------------------
# Issue 2 regression tests
# ---------------------------------------------------------------------------


class TestConstraintOrderingSolve:
    """Solve-path regression: lambdas and totals map to the right name
    even when user-dict order, alphabetical order, and grid order all
    differ. With three constraints any two-way swap would fail at least
    one of the per-name range assertions."""

    def test_solve_lambdas_and_totals_map_to_correct_constraint(self):
        df = _make_three_column_df(n_quotes=80, n_steps=5)

        # Peek baselines so we can pick targets per column.
        peek = pc.OnlineOptimiser(
            objective="expected_income",
            # User passes constraints in NON-alphabetical order to make
            # sure dict iteration is not what saves us.
            constraints={
                "volume": {"min_pct": 1.0},
                "apple_count": {"min_pct": 1.0},
                "zebra_metric": {"min_pct": 1.0},
            },
            max_iter=1,
        )
        baseline = peek.solve(df).baseline_constraints

        baseline_vol = baseline["volume"]
        baseline_apple = baseline["apple_count"]
        baseline_zebra = baseline["zebra_metric"]

        # Confirm the three baselines are at very different scales —
        # this is the property that makes a column swap detectable.
        assert baseline_apple > 100 * baseline_vol > 100 * baseline_zebra

        # Now do a meaningful solve: tight floor on volume, looser
        # floors on the other two. The user dict order is deliberately
        # NOT alphabetical to exercise the ordering fix.
        vol_target = baseline_vol * 0.85
        apple_target = baseline_apple * 0.80
        zebra_target = baseline_zebra * 0.80

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": vol_target},
                "apple_count": {"min": apple_target},
                "zebra_metric": {"min": zebra_target},
            },
            max_iter=200,
        )
        result = solver.solve(df)

        # Each total_constraints[name] must equal the column total
        # for that exact name. If the impl swapped any two specs,
        # one of these would land hundreds-of-times outside its
        # expected bracket.
        assert "volume" in result.total_constraints
        assert "apple_count" in result.total_constraints
        assert "zebra_metric" in result.total_constraints

        # Volume scales O(1): the result must sit in the volume bracket
        # not the apple_count bracket (~1000x larger) or the zebra
        # bracket (~100x smaller).
        vol_total = result.total_constraints["volume"]
        assert 0 < vol_total < 10 * baseline_vol, (
            f"volume total {vol_total} is not in the volume scale; "
            f"baseline volume = {baseline_vol}"
        )
        assert vol_total >= vol_target * 0.97  # constraint floor

        # apple_count scales O(1000).
        apple_total = result.total_constraints["apple_count"]
        assert 100 * baseline_vol < apple_total < 10 * baseline_apple, (
            f"apple_count total {apple_total} is not in the apple scale; "
            f"baseline apple_count = {baseline_apple}"
        )
        assert apple_total >= apple_target * 0.97

        # zebra_metric scales O(0.01).
        zebra_total = result.total_constraints["zebra_metric"]
        assert 0 < zebra_total < baseline_vol, (
            f"zebra_metric total {zebra_total} is not in the zebra scale; "
            f"baseline zebra_metric = {baseline_zebra}"
        )
        assert zebra_total >= zebra_target * 0.97

        # Lambdas must also be keyed correctly. Names should match.
        assert set(result.lambdas.keys()) == {
            "volume",
            "apple_count",
            "zebra_metric",
        }
        # All lambdas should be non-negative (min direction).
        for name, lam in result.lambdas.items():
            assert lam >= 0.0, f"lambda for {name} must be non-negative, got {lam}"

        # Sanity cross-check: totals also match what the column sum
        # is for the optimal step picked per quote — but we can't
        # reach into the result df here easily without coupling to
        # internals. The bracket assertions above are sufficient
        # because the scales are 100x apart.

    def test_baseline_constraints_map_to_correct_name(self):
        """Even pre-solve baselines must be keyed correctly. With three
        constraints at very different scales, a misalignment in
        ``baseline_constraints`` would be unmistakable."""
        df = _make_three_column_df(n_quotes=80, n_steps=5)

        peek = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 1.0},
                "apple_count": {"min_pct": 1.0},
                "zebra_metric": {"min_pct": 1.0},
            },
            max_iter=1,
        )
        baseline = peek.solve(df).baseline_constraints

        # Hand-compute baselines from the DataFrame at the baseline
        # scenario_value (mult=1.0). Baseline = sum at the closest
        # step to scenario_value=1.0 per quote. With mults
        # [0.8, 0.9, 1.0, 1.1, 1.2], step index 2 corresponds to
        # mult=1.0, so we can sum the mult==1.0 rows.
        at_one = df.filter(pl.col("scenario_value") == 1.0)
        # Each quote should have exactly one row at mult=1.0.
        assert at_one.height == 80
        expected_vol = at_one["volume"].sum()
        expected_apple = at_one["apple_count"].sum()
        expected_zebra = at_one["zebra_metric"].sum()

        # Confirm the labelled baselines match the named columns. A
        # swap would fail at least two of these (since the three
        # values differ by ~100x each).
        assert baseline["volume"] == pytest.approx(expected_vol, rel=1e-4)
        assert baseline["apple_count"] == pytest.approx(expected_apple, rel=1e-4)
        assert baseline["zebra_metric"] == pytest.approx(expected_zebra, rel=1e-4)


class TestConstraintOrderingFrontier:
    """Frontier-path regression: ``threshold_<name>`` and ``total_<name>``
    columns in the frontier ``points`` DataFrame must reflect the named
    constraint, not a transposed/swapped pair.
    """

    def test_frontier_columns_not_transposed(self):
        df = _make_three_column_df(n_quotes=80, n_steps=5)

        peek = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 1.0},
                "apple_count": {"min_pct": 1.0},
                "zebra_metric": {"min_pct": 1.0},
            },
            max_iter=1,
        )
        baseline = peek.solve(df).baseline_constraints
        baseline_vol = baseline["volume"]
        baseline_apple = baseline["apple_count"]
        baseline_zebra = baseline["zebra_metric"]

        # Pick lo/hi ranges in absolute units, sized to the right
        # column scale. Swapped, the values would obviously clash.
        vol_lo, vol_hi = baseline_vol * 0.80, baseline_vol * 0.95
        apple_lo, apple_hi = baseline_apple * 0.80, baseline_apple * 0.95
        zebra_lo, zebra_hi = baseline_zebra * 0.80, baseline_zebra * 0.95

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                # NON-alphabetical user-dict order on purpose.
                "volume": {"min": vol_lo},
                "apple_count": {"min": apple_lo},
                "zebra_metric": {"min": zebra_lo},
            },
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={
                "volume": (vol_lo, vol_hi),
                "apple_count": (apple_lo, apple_hi),
                "zebra_metric": (zebra_lo, zebra_hi),
            },
            n_points_per_dim=3,  # 3^3 = 27 points
        )

        points = result.points

        # Required columns present.
        for col in (
            "threshold_volume",
            "threshold_apple_count",
            "threshold_zebra_metric",
            "total_volume",
            "total_apple_count",
            "total_zebra_metric",
            "lambda_volume",
            "lambda_apple_count",
            "lambda_zebra_metric",
        ):
            assert col in points.columns, f"missing column {col} in frontier points"

        # Each threshold column must be in the bracket of its name.
        # If a swap had happened, threshold_volume would contain
        # apple_count-scale numbers (~1000) or zebra-scale (~0.01).
        vol_thresh = points["threshold_volume"]
        assert vol_thresh.min() == pytest.approx(vol_lo, rel=1e-3)
        assert vol_thresh.max() == pytest.approx(vol_hi, rel=1e-3)

        apple_thresh = points["threshold_apple_count"]
        assert apple_thresh.min() == pytest.approx(apple_lo, rel=1e-3)
        assert apple_thresh.max() == pytest.approx(apple_hi, rel=1e-3)

        zebra_thresh = points["threshold_zebra_metric"]
        assert zebra_thresh.min() == pytest.approx(zebra_lo, rel=1e-3)
        assert zebra_thresh.max() == pytest.approx(zebra_hi, rel=1e-3)

        # Each total_<name> column must also live in its expected
        # bracket. If specs[k] were misaligned with grid.constraints[k],
        # the totals returned by the inner solver would be in some
        # arbitrary order, and one of these assertions would fail.
        # Allow a generous margin around baseline (totals are
        # bounded above by the all-mult-0.8 case, below by the
        # all-mult-1.2 case for these monotonic columns).
        for col, lo_bound, hi_bound in [
            ("total_volume", 0.0, 5 * baseline_vol),
            ("total_apple_count", 0.0, 5 * baseline_apple),
            ("total_zebra_metric", 0.0, 5 * baseline_zebra),
        ]:
            vals = points[col]
            assert vals.min() > 0, f"{col} should be positive"
            assert vals.max() <= hi_bound, (
                f"{col} max {vals.max()} exceeds {hi_bound}; "
                f"likely a column-swap regression"
            )

        # Cross-bracket guard: total_volume is at volume scale, not
        # apple scale. apple is ~1000x volume; so total_volume must
        # be way below baseline_apple.
        assert points["total_volume"].max() < baseline_apple / 10
        # total_apple_count must be way above baseline_volume.
        assert points["total_apple_count"].min() > 10 * baseline_vol
        # total_zebra_metric must be way below baseline_volume.
        assert points["total_zebra_metric"].max() < baseline_vol / 10
