"""Feature C6 - ratio constraints in ``ApplyOptimiser``.

This file pins the C6 contract: ``ApplyOptimiser.apply()`` and
``apply_from_grid(...)`` no longer raise ``NotImplementedError`` for
ratio constraints; instead they perform a single fixed-lambda forward
pass that linearises each ratio spec into a per-quote-step column
``c_i = num_i - L * denom_i`` and picks ``s_i* = argmax_m [obj_i(m) -
lambda * c_i(m)]`` per quote. No iteration, no lambda updates.

Contract recap
--------------

Apply mode runs a fixed forward pass — the user has already stored
lambdas from a prior solve. For each ratio constraint:

* ``L`` is computed at apply time using the same rules as solve:
  ``min`` / ``max`` keys take the threshold verbatim; ``min_pct`` /
  ``max_pct`` keys multiply the threshold by the apply-time
  ``baseline_LR = Sigma_baseline num / Sigma_baseline denom`` from the
  apply-time DataFrame's ``scenario_value == 1.0`` slice. (This is the
  contract test :class:`TestApplyOptimiserRatioPctSemantics` pins:
  apply-time baseline, not a saved L.)
* The synthetic linearised column is materialised on a working copy of
  the apply-time DataFrame; the inner Rust ``apply_lambdas_py`` runs
  with the linearised sum-shape constraints; a Python wrapper stitches
  ``optimal_<numerator>`` / ``optimal_<denominator>`` columns onto the
  result DataFrame and reports the actual ratio in
  ``total_constraints[<ratio_label>]`` / ``baseline_constraints[<ratio_label>]``
  (C3 contract carried over from solve).

Save/load round-trip
--------------------

``ApplyOptimiser.save(path)`` serialises the constraint dict verbatim
(including the ratio shape ``{"numerator": ..., "denominator": ...,
"max": ...}``); ``ApplyOptimiser.load(path)`` reconstructs the
optimiser exactly. The on-disk JSON must match what was passed in.

Construction-time rejection
---------------------------

``ApplyOptimiser`` construction with ``None`` ratio thresholds still
raises ``ValueError`` (C1 + B1 contract: apply needs a fixed L per
constraint, and ``None`` is the frontier-only marker).

``apply_from_grid()`` with a ratio constraint
---------------------------------------------

The pre-built ``QuoteGrid`` path mirrors the solve-from-grid /
frontier-from-grid story: a frozen grid does NOT carry the raw
numerator / denominator columns, so the linearisation cannot run
retroactively. We pin that ``apply_from_grid()`` raises ``ValueError``
naming the constraint label, NOT ``NotImplementedError`` (the feature
is available — just not via this entry point). The impl agent may
choose to support the grid path by linearising in Python before the
Rust call IF the grid was built carrying numerator/denominator columns,
but the default contract here is reject-with-actionable-error.

Stubs lifted (C6 scope)
-----------------------

``ApplyOptimiser.apply()`` and ``apply_from_grid()`` no longer raise
``NotImplementedError`` for ratio constraints. The
:class:`TestApplyOptimiserRatioStubsRemoved` regression class pins
this so the impl agent's stub-state-machine flips are immediately
surfaced.

Note for the impl agent
-----------------------

The natural shape mirrors C2 / C3 / C4:

(a) Linearisation happens in :mod:`price_contour.apply` on the
    DataFrame branch BEFORE the Rust ``apply_lambdas_py`` call. The
    helper :func:`price_contour.solver._linearise_ratio_constraints`
    is reusable verbatim — it doesn't depend on the iteration loop.
(b) The result wrapper (analogous to
    :class:`price_contour.solver._RatioSolveResultWrapper`) decorates
    ``ApplyResult`` with the same ``optimal_<num>`` / ``optimal_<denom>``
    column stitching and actual-ratio reporting in
    ``total_constraints`` / ``baseline_constraints``.
(c) ``apply_from_grid()`` keeps the existing ratio-reject path — the
    grid is opaque so the linearisation cannot run retroactively.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from price_contour.apply import apply_from_grid

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

# Tight tolerance for direct-equality reporting checks on apply mode:
# apply runs a single forward pass with FIXED lambdas, so the actual
# ratio reported by ``total_constraints`` must equal the recomputed
# ratio from ``optimal_*`` columns to float-precision (no iteration
# noise).
REPORT_RTOL = 1e-5
REPORT_ABS = 1e-4


# ---------------------------------------------------------------------------
# 1. Basic apply-with-ratio behaviour
# ---------------------------------------------------------------------------


class TestApplyOptimiserRatioBasic:
    """End-to-end apply that exercises the C6 linearisation.

    Pattern: solve to derive lambdas, then construct an ApplyOptimiser
    with the same constraints + saved lambdas, then run ``apply()`` on
    the same (or a different) DataFrame.
    """

    def test_apply_with_solve_lambdas_matches_actual_ratio(self):
        """Apply with the lambdas from a prior solve produces a result
        with ``total_constraints[<ratio_label>]`` equal to the actual
        ratio at the apply-time optimum.

        The actual ratio at apply-mode argmax may differ slightly from
        the solve-time optimum because solve runs an iterative dual
        update and apply runs a single forward pass with fixed lambdas
        — but for converged solve lambdas on the same DataFrame the
        two should agree to within tolerance.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.62,
            }
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
            tolerance=1e-4,
        )
        solve_result = solver.solve(df)

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )
        apply_result = applier.apply(df)

        # 1. total_constraints reports the actual ratio (C3 contract
        #    carried over).
        reported = apply_result.total_constraints["loss_ratio"]
        recomputed = actual_ratio_at_optimum(
            apply_result.dataframe, "incurred", "premium"
        )
        assert reported == pytest.approx(recomputed, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"apply_result.total_constraints['loss_ratio']={reported} must "
            f"equal Sigma optimal_incurred / Sigma optimal_premium = "
            f"{recomputed}"
        )

        # 2. ``optimal_<num>`` / ``optimal_<denom>`` columns surfaced.
        out_df = apply_result.dataframe
        assert "optimal_incurred" in out_df.columns, (
            f"apply result DataFrame must surface 'optimal_incurred' for "
            f"the ratio constraint; got columns: {out_df.columns}"
        )
        assert "optimal_premium" in out_df.columns, (
            f"apply result DataFrame must surface 'optimal_premium' for "
            f"the ratio constraint; got columns: {out_df.columns}"
        )

        # 3. Apply on the same DF with converged solve lambdas should
        #    produce a similar actual ratio. We allow generous tolerance:
        #    apply runs a single forward pass, so the argmax ties may
        #    break differently than the dual-iteration solver's, but the
        #    actual ratio at the optimum should not drift far.
        solve_actual = actual_ratio_at_optimum(
            solve_result.dataframe, "incurred", "premium"
        )
        assert reported == pytest.approx(solve_actual, rel=RATIO_RTOL, abs=RATIO_ABS_SLACK), (
            f"apply actual ratio {reported} differs from solve actual "
            f"ratio {solve_actual} beyond tolerance"
        )

    def test_apply_to_different_dataframe_runs_without_resolving(self):
        """Live-scoring use case: stored lambdas from a prior solve are
        reused on a fresh DataFrame without any iteration.

        We construct a second DataFrame with the same schema but
        different (perturbed) numbers and verify that ``apply()`` returns
        a result that picks per-quote argmax steps without re-solving.
        """
        df_train = make_ratio_solve_df(n_quotes=20, n_steps=5)
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.62,
            }
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
            tolerance=1e-4,
        )
        solve_result = solver.solve(df_train)

        # Different DataFrame: same shape, slightly perturbed values.
        # Multiplicative noise keeps the schema identical and the
        # per-quote conversion roughly the same so the constraint can
        # still bind.
        df_live = df_train.with_columns(
            (pl.col("income") * 1.02).cast(pl.Float32).alias("income"),
            (pl.col("incurred") * 0.98).cast(pl.Float32).alias("incurred"),
            (pl.col("premium") * 1.02).cast(pl.Float32).alias("premium"),
        )

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )
        apply_result = applier.apply(df_live)

        # Result has an actual ratio reported and surfaces the
        # ``optimal_*`` columns. The actual ratio is computed against
        # the live DataFrame, not the training DataFrame.
        out_df = apply_result.dataframe
        assert "optimal_incurred" in out_df.columns
        assert "optimal_premium" in out_df.columns
        actual = actual_ratio_at_optimum(out_df, "incurred", "premium")
        reported = apply_result.total_constraints["loss_ratio"]
        assert reported == pytest.approx(actual, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"apply on live DF must report actual live-data ratio: "
            f"reported={reported}, recomputed={actual}"
        )

        # Output row count matches the apply-time DF's quote count.
        n_quotes_live = df_live.select("quote_id").n_unique()
        assert out_df.shape[0] == n_quotes_live, (
            f"apply output rows ({out_df.shape[0]}) must equal apply-time "
            f"unique quotes ({n_quotes_live})"
        )

    def test_apply_with_zero_lambda_picks_unconstrained_argmax(self):
        """With ``lambdas[<ratio_label>] = 0`` the linearisation drops
        out (lambda * c_i = 0) and apply picks the unconstrained per-quote
        objective argmax — irrespective of what the ratio threshold says.

        This is the apply-mode analogue of the existing
        ``test_apply_zero_lambdas_unconstrained`` for sum constraints.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.0},
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.50,  # tight, would bind under non-zero lambda
                }
            },
        )
        apply_result = applier.apply(df)

        # Per-quote argmax of the objective; ratio doesn't constrain
        # because lambda is zero.
        out_df = apply_result.dataframe
        for row in out_df.iter_rows(named=True):
            qid = row["quote_id"]
            q_df = df.filter(pl.col("quote_id") == qid)
            best_idx = q_df["income"].arg_max()
            assert row["optimal_step"] == best_idx, (
                f"quote {qid}: with lambda=0 apply must pick unconstrained "
                f"income argmax; got optimal_step={row['optimal_step']}, "
                f"expected {best_idx}"
            )


class TestApplyOptimiserRatioPctSemantics:
    """Pin the contract for ``min_pct`` / ``max_pct`` direction keys in
    apply mode.

    Decision: ``L`` is computed from the **apply-time DataFrame**'s
    baseline LR, not from the original solve-time baseline. Rationale:

    * The constraint dict stores ``{"max_pct": 0.95}``, NOT a frozen L.
      The pct is a fraction *of baseline*, where "baseline" means
      "scenario_value == 1.0" — and the user can run apply on data that
      has a different baseline than the solve data did (the live-scoring
      use case: a new portfolio with the same constraint shape).
    * If the user wants the same absolute L across solve and apply, they
      should use ``{"max": L}`` instead of ``{"max_pct": pct}``.
      Apply with ``max_pct`` means "track the apply-time baseline".
    * This matches the natural reading: pct is a ratio-of-ratios target,
      and the denominator (baseline) is whatever data we're scoring NOW.

    If the impl agent chooses the alternative (frozen L from solve time)
    they should rename or invert this test — but they must pin the
    chosen behaviour, not leave it unspecified.
    """

    def test_max_pct_resolves_via_apply_time_baseline(self):
        """``max_pct: 0.95`` on a DataFrame with a different baseline LR
        than the original solve must use the apply-time baseline.

        We construct two DataFrames with different baseline ratios and
        verify that apply on each produces an actual ratio anchored on
        that DF's baseline.
        """
        # Train DF: baseline LR ~ 0.6484 (from C2 fixture comment).
        df_train = make_ratio_solve_df(n_quotes=20, n_steps=5)
        baseline_train = baseline_ratio(df_train, "incurred", "premium")
        # Sanity: baseline must match the C2 fixture's documented value.
        assert 0.62 < baseline_train < 0.68, (
            f"fixture baseline LR {baseline_train} outside expected band"
        )

        # Live DF: deliberately scale incurred down so the live baseline
        # LR is materially different from the train baseline.
        df_live = df_train.with_columns(
            (pl.col("incurred") * 0.85).cast(pl.Float32).alias("incurred"),
        )
        baseline_live = baseline_ratio(df_live, "incurred", "premium")
        # Live baseline materially lower.
        assert baseline_live < baseline_train * 0.95, (
            f"live baseline {baseline_live} should be < 95% of train "
            f"baseline {baseline_train} for this test to discriminate"
        )

        # Solve on TRAIN to get lambdas with max_pct=0.95.
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max_pct": 0.95,
            }
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
            tolerance=1e-4,
        )
        solve_result = solver.solve(df_train)

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )

        # Apply on TRAIN: baseline_constraints reports the train baseline
        # ratio (the C3 contract carried over).
        apply_train = applier.apply(df_train)
        reported_train_baseline = apply_train.baseline_constraints["loss_ratio"]
        assert reported_train_baseline == pytest.approx(
            baseline_train, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"apply on train DF must report train baseline "
            f"{baseline_train}; got {reported_train_baseline}"
        )

        # Apply on LIVE: baseline_constraints reports the LIVE baseline
        # ratio, not the train baseline. This is the discriminating
        # check — the apply path must compute baseline at apply time,
        # not freeze it at construction time.
        apply_live = applier.apply(df_live)
        reported_live_baseline = apply_live.baseline_constraints["loss_ratio"]
        assert reported_live_baseline == pytest.approx(
            baseline_live, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"apply on live DF must report live-data baseline "
            f"{baseline_live} (not train baseline {baseline_train}); "
            f"got {reported_live_baseline}"
        )

    def test_min_pct_resolves_via_apply_time_baseline(self):
        """Same contract for ``min_pct`` direction (retention floor)."""
        df_train = make_retention_df(n_quotes=20, n_steps=5)
        baseline_train = baseline_ratio(df_train, "kept", "exposed")

        # Live DF: scale exposed up to lower the live retention baseline.
        df_live = df_train.with_columns(
            (pl.col("exposed") * 1.10).cast(pl.Float32).alias("exposed"),
        )
        baseline_live = baseline_ratio(df_live, "kept", "exposed")
        assert baseline_live < baseline_train, (
            f"live baseline retention {baseline_live} should be lower "
            f"than train {baseline_train} for this test to discriminate"
        )

        constraints = {
            "retention_ratio": {
                "numerator": "kept",
                "denominator": "exposed",
                "min_pct": 0.98,
            }
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
            tolerance=1e-4,
        )
        solve_result = solver.solve(df_train)
        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )

        apply_live = applier.apply(df_live)
        reported_live_baseline = apply_live.baseline_constraints[
            "retention_ratio"
        ]
        assert reported_live_baseline == pytest.approx(
            baseline_live, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"apply on live DF must report live baseline retention "
            f"{baseline_live}; got {reported_live_baseline}"
        )

    def test_apply_baseline_constraints_uses_apply_df_not_solve_df(self):
        """Pin-point the apply-time baseline invariant in isolation.

        Solve on ``df_A``, store the lambdas, apply on ``df_B`` where
        ``df_B`` has a materially different baseline LR. The
        ``apply_result.baseline_constraints[label]`` must report
        ``df_B``'s baseline ratio, not ``df_A``'s — apply-time
        semantics, not solve-time semantics. This test exists alongside
        the more elaborate ``test_max_pct_resolves_via_apply_time_baseline``
        as a focused regression guard.
        """
        df_a = make_ratio_solve_df(n_quotes=20, n_steps=5)
        baseline_a = baseline_ratio(df_a, "incurred", "premium")

        # df_b: scale incurred down so the baseline LR shifts materially.
        df_b = df_a.with_columns(
            (pl.col("incurred") * 0.70).cast(pl.Float32).alias("incurred"),
        )
        baseline_b = baseline_ratio(df_b, "incurred", "premium")
        assert baseline_b < baseline_a * 0.85, (
            f"df_b baseline {baseline_b} should be < 85% of df_a baseline "
            f"{baseline_a} for the test to discriminate"
        )

        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max_pct": 0.95,
            }
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
            tolerance=1e-4,
        )
        solve_result = solver.solve(df_a)

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )
        apply_b = applier.apply(df_b)
        reported = apply_b.baseline_constraints["loss_ratio"]
        # The reported baseline must track df_b, not df_a.
        assert reported == pytest.approx(
            baseline_b, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"apply on df_b must report df_b's baseline LR ({baseline_b}); "
            f"got {reported}, df_a baseline was {baseline_a}"
        )
        # Tighten the discriminator: df_a's baseline must NOT match the
        # reported value within tolerance (otherwise the test passes
        # vacuously).
        assert abs(reported - baseline_a) > REPORT_ABS, (
            f"reported baseline {reported} is indistinguishable from "
            f"df_a's baseline {baseline_a} — pick a more dramatic "
            f"df_b transform"
        )


# ---------------------------------------------------------------------------
# 2. Reporting contract (C3 carried over to apply mode)
# ---------------------------------------------------------------------------


class TestApplyOptimiserRatioReporting:
    """``ApplyResult.total_constraints[<ratio_label>]`` is the actual
    ratio; ``baseline_constraints[<ratio_label>]`` is the actual baseline
    ratio. Sum constraints in a mixed dict pass through unchanged.
    """

    def test_total_constraints_reports_actual_ratio(self):
        """Direct-equality check: the reported value equals the
        recomputed actual ratio from ``optimal_*`` columns.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.62,
            }
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
        )
        solve_result = solver.solve(df)
        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )
        apply_result = applier.apply(df)

        reported = apply_result.total_constraints["loss_ratio"]
        recomputed = actual_ratio_at_optimum(
            apply_result.dataframe, "incurred", "premium"
        )
        assert reported == pytest.approx(recomputed, rel=REPORT_RTOL, abs=REPORT_ABS), (
            f"C3-on-apply: total_constraints['loss_ratio']={reported} "
            f"must equal actual ratio {recomputed}"
        )
        # Sanity: must be a plausible LR figure (not the linearised
        # near-zero number C2 used to report).
        assert 0.5 < reported < 0.8, (
            f"reported actual ratio {reported} out of plausible LR band"
        )

    def test_baseline_constraints_reports_actual_baseline_ratio(self):
        """``baseline_constraints[<ratio_label>]`` is the actual ratio
        at scenario_value=1.0 from the apply-time DataFrame.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        expected_baseline = baseline_ratio(df, "incurred", "premium")

        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.62,
            }
        }
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.5},
            objective="income",
            constraints=constraints,
        )
        apply_result = applier.apply(df)

        reported_baseline = apply_result.baseline_constraints["loss_ratio"]
        assert reported_baseline == pytest.approx(
            expected_baseline, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"C3-on-apply: baseline_constraints['loss_ratio']="
            f"{reported_baseline} must equal Sigma_baseline incurred / "
            f"Sigma_baseline premium = {expected_baseline}"
        )

    def test_mixed_sum_and_ratio_reporting(self):
        """Mixed sum + ratio constraints: sum entries pass through as
        sums, ratio entries pass through as actual ratios.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        constraints = {
            "premium": {"min_pct": 0.85},
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.62,
            },
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
        )
        solve_result = solver.solve(df)

        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )
        apply_result = applier.apply(df)

        # Sum entry: report is a sum (recomputable from optimal_premium).
        reported_premium = apply_result.total_constraints["premium"]
        recomputed_premium = float(
            apply_result.dataframe["optimal_premium"].sum()
        )
        assert reported_premium == pytest.approx(
            recomputed_premium, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"sum constraint 'premium' must report the sum: reported="
            f"{reported_premium}, recomputed={recomputed_premium}"
        )
        # Ratio entry: report is an actual ratio.
        reported_lr = apply_result.total_constraints["loss_ratio"]
        recomputed_lr = actual_ratio_at_optimum(
            apply_result.dataframe, "incurred", "premium"
        )
        assert reported_lr == pytest.approx(
            recomputed_lr, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"ratio constraint 'loss_ratio' must report the actual ratio: "
            f"reported={reported_lr}, recomputed={recomputed_lr}"
        )

        # Both keys present; sanity-check both lambdas survive too.
        assert "premium" in apply_result.lambdas
        assert "loss_ratio" in apply_result.lambdas


# ---------------------------------------------------------------------------
# 3. Save/load round-trip with ratio constraints
# ---------------------------------------------------------------------------


class TestApplyOptimiserRatioRoundTrip:
    """``save()`` / ``load()`` must round-trip ratio constraint specs
    verbatim, and a loaded optimiser must produce the same apply()
    output as the saved one.
    """

    def test_save_load_preserves_ratio_constraint_dict(self, tmp_path):
        """The on-disk JSON includes the ratio spec verbatim and load
        reconstructs the optimiser exactly."""
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.65,
            },
        }
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.42},
            objective="income",
            constraints=constraints,
        )

        path = tmp_path / "config.json"
        applier.save(path)

        # Direct on-disk inspection: the JSON blob carries the ratio spec
        # verbatim, with the numerator / denominator keys preserved and
        # the max threshold a plain float.
        on_disk = json.loads(path.read_text())
        assert "constraints" in on_disk
        assert on_disk["constraints"] == constraints, (
            f"on-disk constraints {on_disk['constraints']} must match "
            f"the original {constraints}"
        )

        # Round-trip: load reconstructs the optimiser exactly.
        loaded = pc.ApplyOptimiser.load(path)
        assert loaded.lambdas == {"loss_ratio": 0.42}
        assert loaded.objective == "income"
        assert loaded.constraints == constraints

    def test_save_load_preserves_mixed_sum_and_ratio(self, tmp_path):
        """A mixed sum + ratio constraint dict round-trips with both
        spec shapes preserved."""
        constraints = {
            "premium": {"min_pct": 0.85},
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max_pct": 0.95,
            },
        }
        applier = pc.ApplyOptimiser(
            lambdas={"premium": 0.1, "loss_ratio": 0.2},
            objective="income",
            constraints=constraints,
        )
        path = tmp_path / "config.json"
        applier.save(path)
        loaded = pc.ApplyOptimiser.load(path)

        # Sum spec preserved.
        assert loaded.constraints["premium"] == {"min_pct": 0.85}
        # Ratio spec preserved with all three keys.
        assert loaded.constraints["loss_ratio"] == {
            "numerator": "incurred",
            "denominator": "premium",
            "max_pct": 0.95,
        }
        # Lambdas preserved.
        assert loaded.lambdas == {"premium": 0.1, "loss_ratio": 0.2}

    def test_loaded_optimiser_produces_same_apply_output(self, tmp_path):
        """Apply on the saved and loaded optimisers produces the same
        result on the same DataFrame."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.62,
            }
        }
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints=constraints,
            max_iter=400,
        )
        solve_result = solver.solve(df)

        applier_before = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="income",
            constraints=constraints,
        )
        result_before = applier_before.apply(df)

        path = tmp_path / "config.json"
        applier_before.save(path)
        applier_after = pc.ApplyOptimiser.load(path)
        result_after = applier_after.apply(df)

        # Total objective matches to float precision.
        assert result_after.total_objective == pytest.approx(
            result_before.total_objective, rel=REPORT_RTOL, abs=REPORT_ABS
        ), (
            f"loaded optimiser produced different total_objective "
            f"({result_after.total_objective}) than the saved one "
            f"({result_before.total_objective})"
        )
        # Actual ratio matches.
        assert result_after.total_constraints["loss_ratio"] == pytest.approx(
            result_before.total_constraints["loss_ratio"],
            rel=REPORT_RTOL,
            abs=REPORT_ABS,
        )
        # Lambdas survive verbatim through save/load.
        assert result_after.lambdas == result_before.lambdas


# ---------------------------------------------------------------------------
# 4. apply_from_grid with a ratio constraint
# ---------------------------------------------------------------------------


class TestApplyFromGridRatio:
    """``apply_from_grid()`` with a ratio constraint.

    Pinned behaviour: raises ``ValueError`` naming the constraint and
    pointing the user at the DataFrame-shape apply. Rationale parallels
    ``solve_from_grid`` (C2) and ``frontier`` on a grid (C4): the
    pre-built grid does NOT carry the raw numerator / denominator
    columns, so the linearisation cannot run retroactively.

    The error class is pinned at ``ValueError`` (NOT
    ``NotImplementedError``); the feature is available — just not via
    this entry point. See :class:`TestApplyOptimiserRatioStubsRemoved`
    for the regression guard that the ``NotImplementedError`` stub no
    longer fires.
    """

    def test_apply_from_grid_with_ratio_raises_value_error(self):
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        warmup = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        grid = warmup.solve(df).grid

        with pytest.raises(ValueError) as exc_info:
            apply_from_grid(
                grid,
                lambdas={"loss_ratio": 0.5},
                constraints={
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": 0.62,
                    }
                },
            )
        msg = str(exc_info.value)
        assert "loss_ratio" in msg, (
            f"error {msg!r} must name the offending ratio constraint"
        )
        # The error must NOT be a NotImplementedError mascarading as
        # ValueError; it must signal the grid/DataFrame mismatch (or
        # similar setup-time error) rather than "not implemented".
        assert (
            "DataFrame" in msg
            or "grid" in msg.lower()
            or "numerator" in msg.lower()
            or "denominator" in msg.lower()
        ), (
            f"error {msg!r} must point the user at the DataFrame-shape "
            f"apply or signal the grid limitation"
        )

    def test_apply_from_grid_with_mixed_constraints_raises(self):
        """Mixed sum + ratio: the ratio is the offender; the error
        names it and the message mirrors the solve_from_grid story.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        warmup = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        grid = warmup.solve(df).grid

        with pytest.raises(ValueError) as exc_info:
            apply_from_grid(
                grid,
                lambdas={"premium": 0.1, "loss_ratio": 0.5},
                constraints={
                    "premium": {"min": 100.0},
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": 0.62,
                    },
                },
            )
        msg = str(exc_info.value)
        assert "loss_ratio" in msg

    def test_apply_from_grid_pure_sum_still_works(self):
        """Regression guard: the ratio-reject path must NOT regress the
        pure-sum-constraint behaviour. ``apply_from_grid()`` with sum
        constraints still completes normally.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        warmup = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        grid = warmup.solve(df).grid

        result = apply_from_grid(
            grid,
            lambdas={"premium": 0.0},
            constraints={"premium": {"min": 100.0}},
        )
        # Smoke check: result has the basic shape.
        assert "premium" in result.total_constraints
        assert "premium" in result.lambdas


# ---------------------------------------------------------------------------
# 5. Stub-removed regression guard
# ---------------------------------------------------------------------------


class TestApplyOptimiserRatioStubsRemoved:
    """C6 lifts the apply path stubs.

    These tests pin exactly the boundary so a future change that flips a
    stub state-machine is immediately surfaced.
    """

    def test_apply_with_ratio_no_longer_raises_not_implemented(self):
        """``ApplyOptimiser.apply()`` with a ratio constraint NO LONGER
        raises ``NotImplementedError`` -- it linearises and runs the
        forward pass."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.5},
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
        )
        # MUST NOT raise NotImplementedError. We allow other exceptions
        # to propagate (so the schema validator / setup-time checks
        # still fire correctly), but we explicitly catch
        # ``NotImplementedError`` to fail loudly if the C6 stub is left
        # in place.
        try:
            result = applier.apply(df)
        except NotImplementedError as e:
            pytest.fail(
                f"ApplyOptimiser.apply() with a ratio constraint must "
                f"not raise NotImplementedError under C6; got: {e}"
            )
        assert result is not None
        assert "loss_ratio" in result.total_constraints

    def test_apply_from_grid_with_ratio_does_not_raise_not_implemented(self):
        """``apply_from_grid()`` with a ratio constraint must NOT raise
        ``NotImplementedError``. It MAY raise ``ValueError`` (the chosen
        contract — see :class:`TestApplyFromGridRatio`); it MUST NOT
        leave the C1-era stub in place.
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        warmup = pc.OnlineOptimiser(
            objective="income",
            constraints={"premium": {"min_pct": 1.0}},
            max_iter=1,
        )
        grid = warmup.solve(df).grid

        try:
            apply_from_grid(
                grid,
                lambdas={"loss_ratio": 0.5},
                constraints={
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": 0.62,
                    }
                },
            )
        except NotImplementedError as e:
            pytest.fail(
                f"apply_from_grid() with a ratio constraint must not "
                f"raise NotImplementedError under C6; got: {e}"
            )
        except ValueError:
            # Acceptable per the chosen contract.
            return


# ---------------------------------------------------------------------------
# 6. Construction-time None-rejection regression guard (B1 + C1)
# ---------------------------------------------------------------------------


class TestApplyOptimiserRatioRejectsNone:
    """B1 + C1 contract carried over to C6: ApplyOptimiser construction
    with a ``None`` threshold on a ratio constraint still raises
    ``ValueError`` with apply-specific wording.

    ``None`` is the frontier-only marker; apply mode runs a fixed
    forward pass and needs a numeric L per constraint.
    """

    def test_construction_rejects_none_max_threshold(self):
        """Apply with ``max: None`` on a ratio constraint must raise at
        construction with apply-specific wording."""
        with pytest.raises(ValueError) as exc_info:
            pc.ApplyOptimiser(
                lambdas={"loss_ratio": 0.0},
                objective="income",
                constraints={
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": None,
                    }
                },
            )
        msg = str(exc_info.value)
        assert "loss_ratio" in msg, (
            f"error {msg!r} must name the constraint label"
        )
        # Apply-specific wording: the message should signal that apply
        # mode needs a fixed threshold (or use frontier instead).
        assert (
            "Apply" in msg
            or "apply" in msg
            or "None" in msg
            or "threshold" in msg.lower()
        ), (
            f"error {msg!r} must signal the apply-mode None-rejection"
        )

    def test_construction_rejects_none_min_threshold(self):
        """Symmetric to the max case but for ``min`` direction."""
        with pytest.raises(ValueError) as exc_info:
            pc.ApplyOptimiser(
                lambdas={"retention_ratio": 0.0},
                objective="income",
                constraints={
                    "retention_ratio": {
                        "numerator": "kept",
                        "denominator": "exposed",
                        "min": None,
                    }
                },
            )
        assert "retention_ratio" in str(exc_info.value)

    def test_construction_rejects_none_max_pct_threshold(self):
        """`min_pct` / `max_pct` with None also rejected."""
        with pytest.raises(ValueError) as exc_info:
            pc.ApplyOptimiser(
                lambdas={"loss_ratio": 0.0},
                objective="income",
                constraints={
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max_pct": None,
                    }
                },
            )
        assert "loss_ratio" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestApplyOptimiserRatioEdgeCases:
    """Edge cases: zero baseline denom at apply time, missing schema
    columns, etc."""

    def test_zero_baseline_denominator_raises_for_max_pct(self):
        """Apply-time baseline denominator sum == 0 with ``max_pct`` mode
        raises at apply time with a message naming the constraint and
        signalling the zero-denom condition. Matches the solve-mode
        contract from ``test_ratio_solve_c2.py``.
        """
        rows = []
        n_quotes = 10
        n_steps = 5
        mults = [0.8 + 0.1 * j for j in range(n_steps)]
        for q in range(n_quotes):
            for j, mult in enumerate(mults):
                # Premium == 0 at scenario_value == 1.0 (baseline);
                # non-zero elsewhere so the column has variation.
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

        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.5},
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": 1.0,
                }
            },
        )
        with pytest.raises(ValueError) as exc_info:
            applier.apply(df)
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
            f"error {msg!r} must signal the zero-denominator / "
            f"undefined-baseline condition"
        )

    def test_zero_baseline_denominator_ok_for_absolute_max(self):
        """Absolute ``max`` mode does NOT depend on baseline LR.
        Even if Sigma_baseline premium == 0, apply must still run.
        """
        rows = []
        for q in range(10):
            for j, mult in enumerate([0.8, 0.9, 1.0, 1.1, 1.2]):
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
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.0},
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.7,  # absolute, no baseline_LR needed
                }
            },
        )
        # MUST NOT raise on baseline-denom check.
        result = applier.apply(df)
        assert "loss_ratio" in result.total_constraints

    def test_numerator_column_missing_raises_schema_error(self):
        """Existing C1 contract: a missing numerator column produces
        a ``ValueError`` naming the column AND the constraint label,
        NOT a generic linearisation error.

        (Pin under C6 because the new working code path is the apply
        linearisation, not the C1-era stub. The schema check must still
        fire BEFORE the linearisation.)
        """
        df = make_ratio_solve_df(n_quotes=20, n_steps=5).drop("incurred")
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.5},
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
        )
        with pytest.raises(ValueError) as exc_info:
            applier.apply(df)
        msg = str(exc_info.value)
        assert "incurred" in msg, (
            f"error {msg!r} must name the missing numerator column"
        )
        assert "loss_ratio" in msg, (
            f"error {msg!r} must name the constraint label"
        )

    def test_denominator_column_missing_raises_schema_error(self):
        """Symmetric to the numerator-missing test."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5).drop("premium")
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.5},
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                }
            },
        )
        with pytest.raises(ValueError) as exc_info:
            applier.apply(df)
        msg = str(exc_info.value)
        assert "premium" in msg
        assert "loss_ratio" in msg
