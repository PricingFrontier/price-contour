"""Feature C3 - actual-ratio reporting on ``SolveResult``.

This file pins the C3 contract: the per-ratio entries in
``SolveResult.total_constraints`` and ``SolveResult.baseline_constraints``
are the **actual** ratios — ``Sigma num / Sigma denom`` at the optimal
solve and at scenario_value=1.0 respectively — NOT the C2 interim
linearised value ``Sigma (num_i - L * denom_i)``.

Reporting deltas vs C2
----------------------

C2 contract:
  * ``total_constraints[<ratio_label>]`` returned the linearised total
    at the optimum.
  * ``baseline_constraints[<ratio_label>]`` returned the linearised
    total at scenario_value=1.0.
  * ``record_history`` recorded the linearised value per iteration.
  * ``summary(...)["metrics"]["constraint_<label>_total"]`` therefore
    surfaced a near-zero number (linearised total ~= 0 at the optimum)
    and ``constraint_<label>_baseline`` surfaced the linearised baseline
    (a non-meaningful value for human inspection).

C3 contract:
  * ``total_constraints[<ratio_label>]`` is ``Sigma_optimal num /
    Sigma_optimal denom`` (the actual portfolio ratio at the optimum).
  * ``baseline_constraints[<ratio_label>]`` is ``Sigma_baseline num /
    Sigma_baseline denom`` (the actual baseline portfolio ratio).
  * ``record_history`` records the actual ratio per iteration (computed
    from that iteration's optimal steps).
  * ``summary(...)["metrics"]["constraint_<label>_total"]`` is now the
    actual ratio at the optimum and ``"constraint_<label>_baseline"`` is
    the actual baseline ratio. ``"constraint_<label>_ratio"`` (the
    existing total/baseline ratio) is now meaningful — it's the
    ratio-of-ratios = (actual / baseline).

Sum constraints are unchanged across C2 → C3: ``total_constraints["volume"]``
remains the sum and ``baseline_constraints["volume"]`` remains the
baseline sum. The ``TestSumConstraintsUnchanged`` regression class pins
this so a C3 implementation that accidentally type-flips sum reporting
is immediately surfaced.

Decision on ``test_ratio_solve_c2.py::TestRatioSolveLinearisedReporting``
------------------------------------------------------------------------

Under the C3 swap that class's three tests pin a contract that is now
obsolete (linearised reporting). The C3 impl agent is responsible for
either renaming or deleting that class. ``TestC2InterimContractSwapped``
in this file mirrors the same three scenarios but inverted, asserting
that the C3 reporting is the actual ratio (not the linearised value). I
have intentionally NOT modified ``test_ratio_solve_c2.py`` from the
test-writing pass — the C2 file is left intact as the historical
contract pin for the linearisation interim, and ``TestC2InterimContractSwapped``
is the canonical post-C3 contract. The C3 impl agent should then
delete the obsolete ``TestRatioSolveLinearisedReporting`` class once the
new contract is shipped (it will fail under C3 by construction). I'm
choosing **delete** rather than rename because the new contract is
already pinned in this file under a clearly-labelled class — a renamed
duplicate would just be DRY-violating noise.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import price_contour as pc

# Reuse the C2 fixture helpers — they parameterise long-format
# DataFrames with realistic logistic-conversion / loss-ratio dynamics
# and provide the actual_ratio_at_optimum / baseline_ratio convenience
# helpers.
from test_ratio_solve_c2 import (
    RATIO_ABS_SLACK,
    RATIO_RTOL,
    actual_ratio_at_optimum,
    baseline_ratio,
    make_ratio_solve_df,
    make_retention_df,
)

# Tight tolerance for direct-equality reporting checks. The reported
# value must equal the recomputed-from-DataFrame value to within float32
# round-trip precision; we allow a small absolute slack (~1e-3) for the
# sums being on the order of a few hundred.
REPORT_RTOL = 1e-5
REPORT_ABS = 1e-4


# ---------------------------------------------------------------------------
# 1. total_constraints reports the actual ratio at the optimum
# ---------------------------------------------------------------------------


class TestRatioTotalConstraintsActualRatio:
    """``result.total_constraints[<ratio_label>]`` equals the actual
    ``Sigma_optimal num / Sigma_optimal denom`` to within float-precision
    tolerance.

    Recompute the ground-truth from the surfaced ``optimal_<num>`` /
    ``optimal_<denom>`` columns (the C2 wrapper guarantees these are
    available on ``result.dataframe``) and demand exact equality with
    the reported value.
    """

    def test_max_direction_reports_actual_ratio(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported = result.total_constraints["loss_ratio"]
        recomputed = actual_ratio_at_optimum(result.dataframe, "incurred", "premium")
        assert reported == pytest.approx(recomputed, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3: total_constraints['loss_ratio']={reported} must equal "
            f"the actual ratio Sigma incurred / Sigma premium = "
            f"{recomputed} at the optimum (recomputed from optimal_* "
            f"columns). C2 reported the linearised value instead; C3 "
            f"swaps it."
        )
        # And the actual ratio must be a sensible LR figure (0 < r < 1
        # for this fixture); guard against a swap-direction bug that
        # surfaces e.g. denom/num.
        assert 0.5 < reported < 0.8, (
            f"reported actual ratio {reported} out of plausible LR band "
            f"[0.5, 0.8]; check num/denom orientation"
        )

    def test_min_direction_reports_actual_ratio(self):
        df = make_retention_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min": 0.95,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported = result.total_constraints["retention_ratio"]
        recomputed = actual_ratio_at_optimum(result.dataframe, "kept", "exposed")
        assert reported == pytest.approx(recomputed, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3: total_constraints['retention_ratio']={reported} must "
            f"equal Sigma kept / Sigma exposed = {recomputed}."
        )

    def test_max_pct_reports_actual_ratio(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": 0.95,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported = result.total_constraints["loss_ratio"]
        recomputed = actual_ratio_at_optimum(result.dataframe, "incurred", "premium")
        assert reported == pytest.approx(recomputed, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3 (max_pct): total_constraints['loss_ratio']={reported} "
            f"must equal actual ratio {recomputed}."
        )

    def test_min_pct_reports_actual_ratio(self):
        df = make_retention_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min_pct": 0.98,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported = result.total_constraints["retention_ratio"]
        recomputed = actual_ratio_at_optimum(result.dataframe, "kept", "exposed")
        assert reported == pytest.approx(reported, rel=REPORT_RTOL, abs=REPORT_ABS)
        assert reported == pytest.approx(recomputed, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3 (min_pct): total_constraints['retention_ratio']="
            f"{reported} must equal actual ratio {recomputed}."
        )

    def test_mixed_sum_and_ratio_each_reports_correctly(self):
        """One sum + one ratio: the ratio key reports the actual ratio,
        the sum key reports the sum (existing contract).
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)

        # Pull baseline volume to anchor the sum constraint.
        peek = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        baseline_volume = peek.solve(df).baseline_constraints["premium"]
        volume_floor = 0.85 * baseline_volume

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": volume_floor},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                },
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        # Ratio key: actual ratio.
        ratio_reported = result.total_constraints["loss_ratio"]
        ratio_recomputed = actual_ratio_at_optimum(
            result.dataframe, "incurred", "premium"
        )
        assert ratio_reported == pytest.approx(
            ratio_recomputed, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"mixed: ratio key 'loss_ratio'={ratio_reported} must equal "
            f"actual ratio {ratio_recomputed}"
        )

        # Sum key: sum.
        sum_reported = result.total_constraints["premium"]
        sum_recomputed = float(result.dataframe["optimal_premium"].sum())
        assert sum_reported == pytest.approx(
            sum_recomputed, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"mixed: sum key 'premium'={sum_reported} must equal "
            f"Sigma optimal_premium = {sum_recomputed} (sum-constraint "
            f"contract is unchanged across C2 → C3)"
        )


# ---------------------------------------------------------------------------
# 2. baseline_constraints reports the actual baseline ratio
# ---------------------------------------------------------------------------


class TestRatioBaselineConstraintsActualRatio:
    """``result.baseline_constraints[<ratio_label>]`` equals the actual
    ``Sigma_baseline num / Sigma_baseline denom`` from the input DataFrame
    at scenario_value=1.0.
    """

    def test_max_direction_baseline_is_actual_ratio(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported_baseline = result.baseline_constraints["loss_ratio"]
        recomputed_baseline = baseline_ratio(df, "incurred", "premium")
        assert reported_baseline == pytest.approx(
            recomputed_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C3: baseline_constraints['loss_ratio']={reported_baseline} "
            f"must equal Sigma_baseline incurred / Sigma_baseline premium "
            f"= {recomputed_baseline} computed at scenario_value=1.0."
        )
        # Sanity: this fixture's baseline LR is ~0.6484.
        assert 0.6 < reported_baseline < 0.7, (
            f"fixture baseline LR {reported_baseline} outside expected [0.6, 0.7]"
        )

    def test_min_direction_baseline_is_actual_ratio(self):
        df = make_retention_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min": 0.95,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported_baseline = result.baseline_constraints["retention_ratio"]
        recomputed_baseline = baseline_ratio(df, "kept", "exposed")
        assert reported_baseline == pytest.approx(
            recomputed_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C3: baseline_constraints['retention_ratio']="
            f"{reported_baseline} must equal Sigma_baseline kept / "
            f"Sigma_baseline exposed = {recomputed_baseline}."
        )
        # Sanity: ~0.97 by construction.
        assert 0.95 < reported_baseline < 0.99

    def test_max_pct_baseline_is_actual_ratio(self):
        """``max_pct`` mode also reports actual baseline ratio (NOT the
        derived ``L = pct * baseline_LR`` value)."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": 0.95,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported_baseline = result.baseline_constraints["loss_ratio"]
        recomputed_baseline = baseline_ratio(df, "incurred", "premium")
        assert reported_baseline == pytest.approx(
            recomputed_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C3: max_pct baseline_constraints['loss_ratio']="
            f"{reported_baseline} must equal actual baseline ratio "
            f"{recomputed_baseline} (NOT the derived L = pct * baseline_LR)."
        )

    def test_mixed_sum_and_ratio_baseline_each_reports_correctly(self):
        """Sum key reports baseline sum, ratio key reports actual baseline
        ratio.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)

        peek = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        baseline_volume = peek.solve(df).baseline_constraints["premium"]

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": 0.85 * baseline_volume},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                },
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        # Ratio key: actual baseline ratio.
        ratio_baseline = result.baseline_constraints["loss_ratio"]
        ratio_baseline_expected = baseline_ratio(df, "incurred", "premium")
        assert ratio_baseline == pytest.approx(
            ratio_baseline_expected, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"mixed: ratio baseline {ratio_baseline} must equal "
            f"{ratio_baseline_expected}"
        )

        # Sum key: baseline sum (== peek-solve baseline_volume).
        sum_baseline = result.baseline_constraints["premium"]
        assert sum_baseline == pytest.approx(
            baseline_volume, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"mixed: sum baseline {sum_baseline} must equal "
            f"{baseline_volume} (unchanged across C2 → C3)"
        )


# ---------------------------------------------------------------------------
# 3. summary(...) reflects the new reporting
# ---------------------------------------------------------------------------


class TestRatioInSummary:
    """``summary(...)`` flat metrics dict and human-readable artifacts
    reflect the C3 actual-ratio reporting.
    """

    def test_metrics_constraint_total_is_actual_ratio(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)
        metrics = solver.summary(result)["metrics"]

        recomputed_actual = actual_ratio_at_optimum(
            result.dataframe, "incurred", "premium"
        )
        assert metrics["constraint_loss_ratio_total"] == pytest.approx(
            recomputed_actual, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C3: metrics['constraint_loss_ratio_total']="
            f"{metrics['constraint_loss_ratio_total']} must equal actual "
            f"ratio {recomputed_actual}."
        )

    def test_metrics_constraint_baseline_is_actual_baseline_ratio(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)
        metrics = solver.summary(result)["metrics"]

        recomputed_baseline = baseline_ratio(df, "incurred", "premium")
        assert metrics["constraint_loss_ratio_baseline"] == pytest.approx(
            recomputed_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C3: metrics['constraint_loss_ratio_baseline']="
            f"{metrics['constraint_loss_ratio_baseline']} must equal "
            f"actual baseline ratio {recomputed_baseline}."
        )

    def test_metrics_constraint_ratio_is_total_over_baseline(self):
        """``constraint_<label>_ratio`` is ``total / baseline``. Because
        both are now actual ratios, this is the ratio-of-ratios — a
        meaningful "achieved vs starting" multiplier.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)
        metrics = solver.summary(result)["metrics"]

        total = metrics["constraint_loss_ratio_total"]
        baseline = metrics["constraint_loss_ratio_baseline"]
        assert metrics["constraint_loss_ratio_ratio"] == pytest.approx(
            total / baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C3: metrics['constraint_loss_ratio_ratio']="
            f"{metrics['constraint_loss_ratio_ratio']} must equal "
            f"total/baseline = {total / baseline}."
        )
        # And it should be near 0.62/0.6484 ~= 0.956 — meaningful as a
        # "ratio achieved vs baseline ratio" figure.
        assert 0.85 < metrics["constraint_loss_ratio_ratio"] < 1.05, (
            f"constraint_loss_ratio_ratio={metrics['constraint_loss_ratio_ratio']} "
            f"out of plausible band [0.85, 1.05] for this fixture; check "
            f"that total/baseline is ratio-of-ratios"
        )

    def test_artifacts_summary_constraints_total_and_baseline_are_actual(self):
        """The human-readable ``artifacts.summary.constraints[<label>]``
        block has ``"total"`` and ``"baseline"`` keys that mirror the
        flat metrics — both must be actual ratios.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)
        constraints_block = solver.summary(result)["artifacts"]["summary"][
            "constraints"
        ]

        actual_total = actual_ratio_at_optimum(result.dataframe, "incurred", "premium")
        actual_baseline = baseline_ratio(df, "incurred", "premium")

        assert constraints_block["loss_ratio"]["total"] == pytest.approx(
            actual_total, rel=REPORT_RTOL, abs=REPORT_ABS
        )
        assert constraints_block["loss_ratio"]["baseline"] == pytest.approx(
            actual_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        )
        # And ratio_to_baseline (if surfaced) must also be total/baseline.
        if "ratio_to_baseline" in constraints_block["loss_ratio"]:
            assert constraints_block["loss_ratio"][
                "ratio_to_baseline"
            ] == pytest.approx(
                actual_total / actual_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
            )


# ---------------------------------------------------------------------------
# 4. record_history records actual ratio per iteration
# ---------------------------------------------------------------------------


class TestRatioInRecordHistory:
    """With ``record_history=True``, each iteration's ``total_constraints``
    entry for a ratio label is the actual ratio at that iteration's
    optimal_steps — NOT the linearised value.

    The history can't easily be cross-checked without the per-iteration
    optimal_steps (Rust currently aggregates only into ``total_constraints``).
    These tests therefore assert plausibility constraints rather than
    exact equality:

    * Every recorded value sits in the achievable LR band for the
      fixture — [0.55, 0.75] is a generous superset of [0.6013, 0.6924].
      The linearised values oscillate near zero (-0.5 .. +0.5), which is
      well outside this band, so the C3 swap is unambiguously visible.
    * Convergence trajectory: by the final iterations the recorded ratio
      sits within tolerance of the final ``total_constraints[label]``
      (the wrapper's reporting path is the same code path; consistency).
    """

    def test_history_ratio_in_plausible_band(self):
        """Every recorded ratio sits in the achievable LR band.

        The C2 contract recorded values close to zero (the linearised
        total oscillates around zero at the optimum). C3 swaps to the
        actual ratio, which sits in [0.6, 0.7] for this fixture. The
        band [0.55, 0.75] is a superset of achievable + small slack;
        a recorded value below 0.55 (e.g. near 0) is unambiguously the
        linearised value, not the actual ratio.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=20,
            record_history=True,
        )
        result = solver.solve(df)
        assert result.history is not None
        assert len(result.history) > 0

        for i, rec in enumerate(result.history):
            v = rec["total_constraints"]["loss_ratio"]
            assert math.isfinite(v), (
                f"history[{i}].total_constraints['loss_ratio']={v} must be finite"
            )
            assert 0.55 <= v <= 0.75, (
                f"history[{i}].total_constraints['loss_ratio']={v} "
                f"outside actual-LR band [0.55, 0.75]; the C2 linearised "
                f"value would have been near 0, so this assertion fails "
                f"under the C2 contract — C3 must record the actual "
                f"ratio."
            )

    def test_history_final_iteration_matches_total_constraints(self):
        """Final-iteration history record matches the surfaced
        ``total_constraints[label]``: same value, same code path."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
            max_iter=400,
            tolerance=1e-4,
            record_history=True,
        )
        result = solver.solve(df)
        last = result.history[-1]
        assert last["total_constraints"]["loss_ratio"] == pytest.approx(
            result.total_constraints["loss_ratio"],
            rel=REPORT_RTOL,
            abs=REPORT_ABS,
        ), (
            f"final history iter's loss_ratio "
            f"{last['total_constraints']['loss_ratio']} must match "
            f"final result.total_constraints['loss_ratio'] "
            f"{result.total_constraints['loss_ratio']}"
        )

    def test_history_converges_toward_target(self):
        """Across a binding-constraint solve the actual ratio at iter 0
        differs from the actual ratio at the final iter, and the final
        ratio sits within tolerance of the target.

        The first iteration uses zero lambda, so the optimal step picks
        income-greedy (without considering LR penalty); the last
        iteration must respect the binding LR ceiling.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        target = 0.62
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": target,
                }
            },
            max_iter=400,
            tolerance=1e-4,
            record_history=True,
        )
        result = solver.solve(df)
        first = result.history[0]["total_constraints"]["loss_ratio"]
        # Last iter: pick the median over the trailing 5 records to
        # smooth lambda-oscillation between feasible iterates.
        tail = sorted(
            rec["total_constraints"]["loss_ratio"] for rec in result.history[-5:]
        )
        median_tail = tail[len(tail) // 2]

        # First-iter recorded value must be a plausible LR (sits above
        # baseline 0.6484 because lambda=0 lets the solver pick the
        # income-max scenario, which has higher LR than baseline). The
        # C2 linearised recording would surface a value in tens
        # (Sigma incurred at iter 0 is dominated by num_total - 0*denom
        # since lambda starts at zero) — this assertion fails under C2
        # because the linearised value isn't a ratio.
        assert 0.55 <= first <= 0.75, (
            f"first iter recorded value {first} must be a plausible "
            f"actual ratio (in [0.55, 0.75]) under C3; the C2 linearised "
            f"value would be in tens."
        )
        # The trajectory drives LR DOWN as lambda rises: final ratio
        # sits below the unconstrained iter-0 ratio.
        assert first > median_tail, (
            f"binding max LR: first iter LR {first} should exceed "
            f"final-tail median {median_tail} as lambda rises and the "
            f"solver pulls LR down toward the {target} ceiling."
        )
        # And the final-tail median must respect the binding ceiling.
        assert median_tail <= target * (1 + RATIO_RTOL) + RATIO_ABS_SLACK, (
            f"final-tail median LR {median_tail} > target {target} + tolerance"
        )


# ---------------------------------------------------------------------------
# 5. C2 interim contract swapped (mirrors test_ratio_solve_c2.py
#    TestRatioSolveLinearisedReporting but inverted).
# ---------------------------------------------------------------------------


class TestC2InterimContractSwapped:
    """C2 interim reported the linearised value ``Sigma (num - L*denom)``
    in ``total_constraints[<ratio_label>]``. C3 swaps this to the actual
    ratio. The three tests below mirror C2's
    ``TestRatioSolveLinearisedReporting`` class but inverted — same
    fixtures and same constraint shapes, different expected reporting.
    """

    def test_total_constraints_returns_actual_ratio_max(self):
        """C2: linearised; C3: actual ratio."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        target = 0.62
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": target,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)
        reported = result.total_constraints["loss_ratio"]
        out = result.dataframe
        num_total = float(out["optimal_incurred"].sum())
        denom_total = float(out["optimal_premium"].sum())
        actual = num_total / denom_total

        assert reported == pytest.approx(actual, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3 (swap of C2 linearised contract): "
            f"total_constraints['loss_ratio']={reported} must equal the "
            f"actual ratio Sigma incurred / Sigma premium = {actual}. "
            f"The C2 interim value (linearised "
            f"Sigma incurred - {target} * Sigma premium = "
            f"{num_total - target * denom_total}) is no longer surfaced."
        )

    def test_total_constraints_returns_actual_ratio_min(self):
        df = make_retention_df(n_quotes=20, n_steps=5)
        target = 0.95
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min": target,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)
        reported = result.total_constraints["retention_ratio"]
        out = result.dataframe
        num_total = float(out["optimal_kept"].sum())
        denom_total = float(out["optimal_exposed"].sum())
        actual = num_total / denom_total
        assert reported == pytest.approx(actual, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3: total_constraints['retention_ratio']={reported} must "
            f"equal actual ratio Sigma kept / Sigma exposed = {actual} "
            f"(C2 reported the linearised value)."
        )

    def test_total_constraints_returns_actual_ratio_for_max_pct(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        pct = 0.95
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": pct,
                }
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)
        reported = result.total_constraints["loss_ratio"]
        out = result.dataframe
        num_total = float(out["optimal_incurred"].sum())
        denom_total = float(out["optimal_premium"].sum())
        actual = num_total / denom_total
        assert reported == pytest.approx(actual, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3: max_pct total_constraints['loss_ratio']={reported} "
            f"must equal actual ratio {actual} (C2 reported the linearised "
            f"value at L = {pct} * baseline_LR)."
        )


# ---------------------------------------------------------------------------
# 6. Sum constraints unchanged regression guard
# ---------------------------------------------------------------------------


class TestSumConstraintsUnchanged:
    """C3 swaps ratio reporting only — sum constraints continue to report
    the sum (unchanged across all features). Pin this so a C3
    implementation that accidentally type-flips the sum reporting is
    immediately surfaced.
    """

    def test_pure_sum_constraint_total_is_sum(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 0.95}},
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported = result.total_constraints["premium"]
        recomputed = float(result.dataframe["optimal_premium"].sum())
        assert reported == pytest.approx(recomputed, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"sum-constraint total_constraints['premium']={reported} "
            f"must equal Sigma optimal_premium = {recomputed}; ratios "
            f"must not change sum reporting"
        )

    def test_pure_sum_constraint_baseline_is_baseline_sum(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 0.95}},
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        reported_baseline = result.baseline_constraints["premium"]
        baseline_sum = float(
            df.filter(pl.col("scenario_value") == 1.0)["premium"].sum()
        )
        assert reported_baseline == pytest.approx(
            baseline_sum, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"sum-constraint baseline_constraints['premium']="
            f"{reported_baseline} must equal Sigma_baseline premium = "
            f"{baseline_sum}; ratios must not change sum reporting"
        )

    def test_two_sums_no_ratio_unchanged(self):
        """Pure two-sum solve: both totals are sums, both baselines are
        baseline sums. No ratio key, no behavioural change.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min_pct": 0.95},
                "incurred": {"max_pct": 1.10},
            },
            max_iter=200,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        out = result.dataframe
        for col in ("premium", "incurred"):
            assert result.total_constraints[col] == pytest.approx(
                float(out[f"optimal_{col}"].sum()), rel=REPORT_RTOL, abs=REPORT_ABS
            ), f"sum-only total for '{col}' must be Sigma optimal_{col}"
            baseline_sum = float(df.filter(pl.col("scenario_value") == 1.0)[col].sum())
            assert result.baseline_constraints[col] == pytest.approx(
                baseline_sum, rel=REPORT_RTOL, abs=REPORT_ABS
            ), f"sum-only baseline for '{col}' must be Sigma_baseline {col}"


# ---------------------------------------------------------------------------
# 7. Edge cases: degenerate denominators and column reuse
# ---------------------------------------------------------------------------


class TestRatioReportingEdgeCases:
    """Edge cases for the actual-ratio reporting code path."""

    def test_near_zero_optimal_denominator_handled_gracefully(self):
        """If ``Sigma_optimal denom`` is essentially zero, the actual
        ratio is undefined. The reporting must either (a) return a
        non-finite sentinel (``inf`` or ``nan``) cleanly OR (b) raise.

        Pin: prefer non-finite sentinel over raise — the solve completed,
        only the reporting is degenerate, so a graceful sentinel is
        more user-friendly than blowing up at attribute access. But
        accept either as the contract.

        Construction: every quote's denominator is essentially zero at
        ALL scenarios, so any optimal step gives a ~zero denom sum.
        Use absolute ``max`` direction to avoid the baseline-denominator
        guard that triggers on ``max_pct``.
        """
        n_quotes = 10
        n_steps = 5
        rows = []
        mults = [0.8 + 0.1 * j for j in range(n_steps)]
        for q in range(n_quotes):
            for j, mult in enumerate(mults):
                # Denominator is all-zero everywhere; numerator is
                # also zero (so the ratio is genuinely 0/0).
                rows.append(
                    {
                        "quote_id": f"Q{q:04d}",
                        "scenario_index": j,
                        "scenario_value": mult,
                        "income": 100.0 * mult,  # objective is non-zero
                        "incurred": 0.0,
                        "premium": 0.0,
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
                    "max": 0.7,  # absolute mode dodges baseline guard
                }
            },
            max_iter=20,
        )
        # Either the solve raises at reporting-access time, OR it
        # returns and the value is a non-finite sentinel.
        try:
            result = solver.solve(df)
            v = result.total_constraints["loss_ratio"]
        except (ValueError, ZeroDivisionError, RuntimeError):
            return  # raise path is acceptable
        # Non-finite sentinel path:
        assert not math.isfinite(v) or v == 0.0, (
            f"With Sigma_optimal denom == 0 the reported ratio must be "
            f"a non-finite sentinel (inf/nan) or zero (0/0 ambiguity), "
            f"got finite {v}; otherwise it's silently mis-reporting a "
            f"degenerate division."
        )

    def test_ratio_denominator_shared_with_sum_constraint(self):
        """Mixed sum + ratio where the ratio's denominator IS the sum
        constraint column (both reference 'premium'). The C2 wrapper had
        a duplicate-column issue here that was fixed; verify the C3
        reporting code path also doesn't trip on the shared column.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)

        peek = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        baseline_volume = peek.solve(df).baseline_constraints["premium"]

        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": 0.85 * baseline_volume},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",  # shared with sum key
                    "max": 0.62,
                },
            },
            max_iter=400,
            tolerance=1e-4,
        )
        result = solver.solve(df)

        # Both keys present, both report the right type of value.
        assert "premium" in result.total_constraints
        assert "loss_ratio" in result.total_constraints
        assert "premium" in result.baseline_constraints
        assert "loss_ratio" in result.baseline_constraints

        # Sum key reports the sum.
        assert result.total_constraints["premium"] == pytest.approx(
            float(result.dataframe["optimal_premium"].sum()),
            rel=REPORT_RTOL,
            abs=REPORT_ABS,
        )
        # Ratio key reports the actual ratio.
        actual_ratio = actual_ratio_at_optimum(result.dataframe, "incurred", "premium")
        assert result.total_constraints["loss_ratio"] == pytest.approx(
            actual_ratio, rel=REPORT_RTOL, abs=REPORT_ABS
        )
        # Sum baseline is a sum.
        assert result.baseline_constraints["premium"] == pytest.approx(
            baseline_volume, rel=REPORT_RTOL, abs=REPORT_ABS
        )
        # Ratio baseline is the actual baseline ratio.
        assert result.baseline_constraints["loss_ratio"] == pytest.approx(
            baseline_ratio(df, "incurred", "premium"),
            rel=REPORT_RTOL,
            abs=REPORT_ABS,
        )
