"""Feature B1 — ``None`` is a valid threshold value (frontier-only marker).

After B1, ``None`` is permitted as the value for any of the four direction
keys (``min`` / ``max`` / ``min_pct`` / ``max_pct``). The semantics:

* ``None`` signals "this constraint will be supplied by the frontier sweep,
  not the constructor". Type validation accepts it; NaN/inf rejection still
  fires for numeric values; strings/lists/dicts are still rejected.
* ``OnlineOptimiser.solve(...)`` raises ``ValueError`` if any constraint has
  a ``None`` threshold. The error names the offending constraint and
  mentions ``frontier()`` so the user sees the way out.
* ``OnlineOptimiser.frontier(...)`` *requires* a ``threshold_ranges`` entry
  for any ``None``-threshold constraint and sweeps that range as it would
  for a numeric constraint with the same range.
* ``RatebookOptimiser`` mirrors that behaviour.
* ``ApplyOptimiser`` rejects ``None`` at construction — apply mode runs a
  fixed-lambda forward pass against a known threshold; ``None`` has no
  meaning.

These tests should fail today because the current Python validator's
``isinstance(value, (int, float))`` check rejects ``None`` outright, so
construction itself raises before any of B1's downstream branches can run.
The failures pin B1's contract for the impl agent.
"""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from price_contour.solver import _validate_constraint_dict
from helpers import make_small_df, make_factors


# ---------------------------------------------------------------------------
# Regex helpers used by ValueError tests.
# Each regex must match against ``str(ValueError(...))`` so the message
# stays useful to users / log readers.
# ---------------------------------------------------------------------------

# `solve()` rejection: must name the offending constraint AND mention
# `frontier()` so the user sees the migration path. We don't pin the
# exact wording — just the two things that matter.
RE_SOLVE_REJECTION_VOLUME = r"(?s)volume.*frontier\(\)|frontier\(\).*volume"
RE_SOLVE_REJECTION_LOSS_RATIO = r"(?s)loss_ratio.*frontier\(\)|frontier\(\).*loss_ratio"
RE_SOLVE_REJECTION_LOSS_METRIC = (
    r"(?s)loss_metric.*frontier\(\)|frontier\(\).*loss_metric"
)

# `frontier()` requires a `threshold_ranges` entry for every None-threshold
# constraint. The error must name the missing constraint.
RE_FRONTIER_MISSING_RANGE_VOLUME = r"(?s)volume"
RE_FRONTIER_MISSING_RANGE_LOSS_RATIO = r"(?s)loss_ratio"

# Apply rejects None at construction. Spec error message:
# "ApplyOptimiser does not support None thresholds; apply mode requires
# a fixed threshold per constraint". We pin "apply" + (None|fixed) so the
# regex stays distinct from the pre-B1 "got NoneType" generic message
# that fires today purely because the validator's type check excludes
# None. Otherwise these tests would pass for the wrong reason.
RE_APPLY_REJECTS_NONE = r"(?si)apply.*(None|fixed)|(None|fixed).*apply"


# ---------------------------------------------------------------------------
# 1. Construction — _validate_constraint_dict accepts None for any
#    direction key. NaN/inf rejection and type rejection (for non-numeric,
#    non-None) remain intact.
# ---------------------------------------------------------------------------


class TestValidationAcceptsNone:
    """`None` is now a valid direction-key value at the validator level."""

    def test_validator_accepts_min_none(self):
        """`{"volume": {"min": None}}` is accepted (returns None / no raise)."""
        # Must not raise.
        _validate_constraint_dict({"volume": {"min": None}})

    def test_validator_accepts_max_none(self):
        _validate_constraint_dict({"loss_ratio": {"max": None}})

    def test_validator_accepts_min_pct_none(self):
        _validate_constraint_dict({"volume": {"min_pct": None}})

    def test_validator_accepts_max_pct_none(self):
        _validate_constraint_dict({"loss_ratio": {"max_pct": None}})

    def test_validator_accepts_none_in_mixed_dict(self):
        """Mixing numeric and `None` values across constraints is fine."""
        _validate_constraint_dict(
            {
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            }
        )

    def test_validator_still_rejects_string_value(self):
        """Type check still fires for non-numeric, non-None values."""
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": "not a number"}})

    def test_validator_still_rejects_list_value(self):
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": [1.0, 2.0]}})

    def test_validator_still_rejects_dict_value(self):
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": {"nested": 1.0}}})

    def test_validator_still_rejects_nan(self):
        """NaN rejection survives the None addition — None is allowed,
        but `float('nan')` is still not finite and must error."""
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": float("nan")}})

    def test_validator_still_rejects_inf(self):
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": float("inf")}})

    def test_validator_still_rejects_negative_inf(self):
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": float("-inf")}})

    def test_validator_still_rejects_nan_for_pct_keys(self):
        """NaN rejection applies to `min_pct` / `max_pct` too."""
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min_pct": float("nan")}})
        with pytest.raises(ValueError):
            _validate_constraint_dict({"loss_ratio": {"max_pct": float("nan")}})


# ---------------------------------------------------------------------------
# 2. Construction — OnlineOptimiser / RatebookOptimiser accept None;
#    ApplyOptimiser rejects None.
# ---------------------------------------------------------------------------


class TestOptimiserConstructionAcceptsNone:
    """The two iterative optimisers (Online + Ratebook) construct OK with
    a None threshold; the apply optimiser rejects."""

    def test_online_constructs_with_min_none(self):
        """No raise — frontier-only constraint at construction."""
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
        )
        # Property is preserved so the impl agent can introspect later.
        assert opt.constraints == {"volume": {"min": None}}

    def test_online_constructs_with_max_none(self):
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": None}},
        )
        assert opt.constraints == {"loss_ratio": {"max": None}}

    def test_online_constructs_with_min_pct_none(self):
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}},
        )
        assert opt.constraints == {"volume": {"min_pct": None}}

    def test_online_constructs_with_max_pct_none(self):
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max_pct": None}},
        )
        assert opt.constraints == {"loss_ratio": {"max_pct": None}}

    def test_online_constructs_with_mixed_numeric_and_none(self):
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            },
        )
        assert opt.constraints["volume"] == {"min": 8000.0}
        assert opt.constraints["loss_ratio"] == {"max": None}

    def test_ratebook_constructs_with_min_none(self):
        opt = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            factor_columns=[["region"]],
        )
        assert opt.constraints == {"volume": {"min": None}}

    def test_ratebook_constructs_with_max_none(self):
        opt = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": None}},
            factor_columns=[["region"]],
        )
        assert opt.constraints == {"loss_ratio": {"max": None}}

    def test_ratebook_constructs_with_mixed_numeric_and_none(self):
        opt = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            },
            factor_columns=[["region"]],
        )
        assert opt.constraints["volume"] == {"min": 8000.0}
        assert opt.constraints["loss_ratio"] == {"max": None}


class TestApplyRejectsNoneAtConstruction:
    """ApplyOptimiser is special: apply runs a fixed forward pass with a
    known threshold, so None has no meaning. Construction must reject it."""

    def test_apply_rejects_min_none(self):
        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            pc.ApplyOptimiser(
                lambdas={"volume": 0.1},
                objective="expected_income",
                constraints={"volume": {"min": None}},
            )

    def test_apply_rejects_max_none(self):
        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            pc.ApplyOptimiser(
                lambdas={"loss_ratio": 0.1},
                objective="expected_income",
                constraints={"loss_ratio": {"max": None}},
            )

    def test_apply_rejects_min_pct_none(self):
        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            pc.ApplyOptimiser(
                lambdas={"volume": 0.1},
                objective="expected_income",
                constraints={"volume": {"min_pct": None}},
            )

    def test_apply_rejects_max_pct_none(self):
        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            pc.ApplyOptimiser(
                lambdas={"loss_ratio": 0.1},
                objective="expected_income",
                constraints={"loss_ratio": {"max_pct": None}},
            )

    def test_apply_rejects_none_in_mixed_dict(self):
        """Even one None constraint in a mixed dict is enough to reject."""
        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            pc.ApplyOptimiser(
                lambdas={"volume": 0.1, "loss_ratio": 0.05},
                objective="expected_income",
                constraints={
                    "volume": {"min": 8000.0},
                    "loss_ratio": {"max": None},
                },
            )


# ---------------------------------------------------------------------------
# 3. solve() — None thresholds raise on iterative optimisers.
# ---------------------------------------------------------------------------


class TestOnlineSolveRejectsNone:
    """`OnlineOptimiser.solve()` must raise when any constraint has a
    None threshold. The error must name the constraint AND mention
    `frontier()` so the user immediately sees the way out."""

    def test_solve_with_min_none_raises(self):
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_VOLUME):
            solver.solve(df)

    def test_solve_with_max_none_raises(self):
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_LOSS_RATIO):
            solver.solve(df)

    def test_solve_with_min_pct_none_raises(self):
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_VOLUME):
            solver.solve(df)

    def test_solve_with_max_pct_none_raises(self):
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max_pct": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_LOSS_RATIO):
            solver.solve(df)

    def test_solve_with_mixed_numeric_and_none_raises_naming_none(self):
        """A mixed dict (one numeric, one None) still fails — and the
        message names the offending None constraint, not the numeric one."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            },
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df)
        # Must mention loss_ratio (the None one) and frontier().
        msg = str(exc_info.value)
        assert "loss_ratio" in msg, (
            f"error message {msg!r} must name the offending None constraint "
            f"'loss_ratio'"
        )
        assert "frontier()" in msg, (
            f"error message {msg!r} must mention frontier() so the user "
            f"sees the way out"
        )

    def test_solve_error_message_mentions_frontier(self):
        """Pinning the contract: the error message contains the literal
        substring 'frontier()' so users immediately see the migration."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df)
        assert "frontier()" in str(exc_info.value)
        assert "volume" in str(exc_info.value)

    def test_solve_from_grid_with_none_raises(self):
        """The pre-built QuoteGrid path must catch this too — Rust-only
        callers bypass the Python `_validate_dataframe` shim, so the
        rejection has to live somewhere both paths share."""
        df = make_small_df(n_quotes=50, n_steps=5)
        # Build a grid via a numeric-constraint solver so we have a
        # well-formed grid to feed back through the None-constraint path.
        warmup = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.0}},
            max_iter=1,
        )
        warmup_result = warmup.solve(df)
        grid = warmup_result.grid

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_VOLUME):
            solver.solve(grid)


class TestRatebookSolveRejectsNone:
    """RatebookOptimiser.solve() must reject None thresholds the same way
    as OnlineOptimiser.solve()."""

    def test_solve_with_min_none_raises(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        factors = pl.DataFrame({"region": ["A"] * 20})
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_VOLUME):
            solver.solve(df, factors)

    def test_solve_with_max_none_raises(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        factors = pl.DataFrame({"region": ["A"] * 20})
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": None}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_LOSS_RATIO):
            solver.solve(df, factors)

    def test_solve_with_mixed_numeric_and_none_raises(self):
        df = make_small_df(n_quotes=20, n_steps=5)
        factors = pl.DataFrame({"region": ["A"] * 20})
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            },
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df, factors)
        msg = str(exc_info.value)
        assert "loss_ratio" in msg
        assert "frontier()" in msg


# ---------------------------------------------------------------------------
# 4. frontier() — None thresholds REQUIRE a matching threshold_ranges
#    entry. With the entry, the sweep proceeds. Without, error.
# ---------------------------------------------------------------------------


class TestOnlineFrontierWithNone:
    """A None threshold marks a constraint for frontier-only sweeping.
    Frontier requires a `threshold_ranges` entry for it; without one, error.
    With one, the sweep proceeds as if the constraint had been numeric."""

    def test_frontier_with_min_none_and_range_succeeds(self):
        """`{"volume": {"min": None}}` + `threshold_ranges={"volume": (lo, hi)}`
        sweeps `threshold_volume` from lo to hi (absolute units)."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (8000.0, 12000.0)},
            n_points_per_dim=4,
        )
        # Must produce 4 points and the sweep must cover the requested range.
        assert result.n_points == 4
        thresholds = result.points["threshold_volume"].to_list()
        assert min(thresholds) == pytest.approx(8000.0)
        assert max(thresholds) == pytest.approx(12000.0)

    def test_frontier_with_max_none_and_range_succeeds(self):
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": None}},
            max_iter=50,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.5, 0.7)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        thresholds = result.points["threshold_loss_ratio"].to_list()
        assert min(thresholds) == pytest.approx(0.5)
        assert max(thresholds) == pytest.approx(0.7)

    def test_frontier_with_none_no_range_raises(self):
        """`threshold_ranges={}` (empty) raises and names the constraint."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError, match=RE_FRONTIER_MISSING_RANGE_VOLUME):
            solver.frontier(df, threshold_ranges={}, n_points_per_dim=3)

    def test_frontier_with_none_wrong_range_key_raises(self):
        """`threshold_ranges` for the wrong constraint name raises and
        names the missing constraint."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
            max_iter=50,
        )
        with pytest.raises(ValueError, match=RE_FRONTIER_MISSING_RANGE_VOLUME):
            solver.frontier(
                df,
                threshold_ranges={"some_other_key": (0.0, 1.0)},
                n_points_per_dim=3,
            )

    def test_frontier_with_min_pct_none_and_range_succeeds(self):
        """A `min_pct: None` constraint with a fractional range (0.85–0.95)
        is interpreted as fractions of baseline, consistent with A1's
        rule for `min_pct` / `max_pct` keys."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}},
            max_iter=50,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 0.95)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        thresholds = result.points["threshold_volume"].to_list()
        # The threshold column contains the FRACTIONS the user supplied
        # (range is interpreted in fraction-of-baseline units).
        assert min(thresholds) == pytest.approx(0.85)
        assert max(thresholds) == pytest.approx(0.95)


class TestOnlineFrontierMixedNumericAndNone:
    """Mixed setup: one numeric threshold, one None threshold. In B1, the
    contract still requires every constraint to appear in
    `threshold_ranges` (D1 will relax this for numeric constraints).
    These tests pin B1's contract; D1 will modify them."""

    def test_mixed_with_both_in_threshold_ranges_succeeds(self):
        """Both constraints in `threshold_ranges` → both axes swept.

        Note: today, frontier requires every constraint in threshold_ranges
        regardless of whether its threshold is numeric or None. D1 will
        relax that for the numeric one.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            },
            max_iter=50,
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
        assert "threshold_volume" in pts.columns
        assert "threshold_loss_ratio" in pts.columns
        # Both axes swept across their ranges.
        vol_ts = pts["threshold_volume"].to_list()
        lr_ts = pts["threshold_loss_ratio"].to_list()
        assert min(vol_ts) == pytest.approx(8000.0)
        assert max(vol_ts) == pytest.approx(12000.0)
        assert min(lr_ts) == pytest.approx(0.5)
        assert max(lr_ts) == pytest.approx(0.7)

    def test_mixed_with_numeric_missing_range_holds_numeric_fixed_post_d1(self):
        """D1 contract (lifts B1's strict-every-constraint requirement):
        numeric-threshold constraints with no range entry are held fixed
        at the constructor threshold; None-threshold constraints still
        require a range. This test pins the post-D1 behaviour for the
        mixed case where the numeric constraint has no range and the
        None constraint does.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            },
            max_iter=50,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"loss_ratio": (0.5, 0.7)},
            n_points_per_dim=3,
        )
        # Only the loss_ratio axis is swept → 3 points.
        assert result.n_points == 3
        pts = result.points
        # Volume is fixed at the constructor threshold across all points.
        vol_ts = pts["threshold_volume"].to_list()
        assert all(t == pytest.approx(8000.0) for t in vol_ts)
        # Loss ratio sweeps across the supplied range.
        lr_ts = pts["threshold_loss_ratio"].to_list()
        assert min(lr_ts) == pytest.approx(0.5)
        assert max(lr_ts) == pytest.approx(0.7)

    def test_mixed_with_none_missing_range_errors_naming_none(self):
        """The None-threshold constraint without a range entry errors
        and names the offending constraint."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {"max": None},
            },
            max_iter=50,
        )
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(
                df,
                threshold_ranges={"volume": (8000.0, 12000.0)},
                n_points_per_dim=3,
            )
        # Must name loss_ratio (the None one missing a range).
        assert "loss_ratio" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Ratebook frontier mirrors OnlineOptimiser.frontier behaviour.
# ---------------------------------------------------------------------------


class TestRatebookFrontierWithNone:
    """RatebookOptimiser.frontier() must mirror OnlineOptimiser.frontier()
    None-threshold behaviour: range required, sweep proceeds, naming on error."""

    def test_ratebook_frontier_with_none_and_range_succeeds(self):
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": None}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        result = solver.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.85, 0.95)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3
        ts = result.points["threshold_volume"].to_list()
        assert min(ts) == pytest.approx(0.85)
        assert max(ts) == pytest.approx(0.95)

    def test_ratebook_frontier_with_none_no_range_raises(self):
        n = 50
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": None}},
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
        assert "volume" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Apply save/load — None thresholds can never reach a saved config
#    because save is rejected at construction. Test that an existing
#    save/load round-trip with numeric thresholds remains unaffected.
# ---------------------------------------------------------------------------


class TestApplySaveLoadWithB1:
    """B1 doesn't change save/load semantics for numeric configs. The
    None-threshold configs simply can never exist on disk because
    construction always rejects them. Sanity check that numeric configs
    still round-trip cleanly."""

    def test_save_load_numeric_constraints_unaffected(self, tmp_path):
        constraints = {"volume": {"min": 8000.0}}
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.1},
            objective="expected_income",
            constraints=constraints,
        )
        path = tmp_path / "config.json"
        applier.save(path)
        loaded = pc.ApplyOptimiser.load(path)
        assert loaded.constraints == constraints

    def test_load_legacy_config_with_none_threshold_raises(self, tmp_path):
        """A hand-written config containing a None threshold must fail
        at load — load constructs an ApplyOptimiser and that construction
        re-validates constraints, rejecting None."""
        import json

        legacy = {
            "version": 1,
            "lambdas": {"volume": 0.1},
            "objective": "expected_income",
            "constraints": {"volume": {"min": None}},
            "quote_id": "quote_id",
            "scenario_index": "scenario_index",
            "scenario_value": "scenario_value",
        }
        path = tmp_path / "with_none.json"
        path.write_text(json.dumps(legacy))

        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            pc.ApplyOptimiser.load(path)


# ---------------------------------------------------------------------------
# 7. Issue 3 — None rejection through Rust-direct paths.
#    The Python ``_reject_none_for_solve`` shim catches None thresholds for
#    typical callers, but Rust-only paths (the raw ``solve_from_grid_py``
#    PyO3 function and ``apply_from_grid()``) bypass that shim. The Rust
#    backstop in ``solver_py.rs::parse_constraints`` must catch them too.
#    These tests pin that backstop so future refactors of the Python shim
#    cannot silently remove the safety net.
# ---------------------------------------------------------------------------


class TestNoneRejectionViaRustDirect:
    """Rust-direct paths must raise ``ValueError`` on None thresholds.

    Online's ``solve()`` flows through the Python ``_reject_none_for_solve``
    shim. Two paths skip the shim:

    1. ``_price_contour.solve_from_grid_py`` called directly (no Python
       wrapper). The backstop in ``parse_constraints`` must fire and the
       message must mention both the constraint name and ``frontier()``.
    2. ``apply_from_grid()`` is the Python helper for the apply mode's
       grid path. Apply has no frontier, so the message must be apply-
       specific (``apply`` + ``None``/``fixed`` per the regex). The
       current implementation falls through to the same Rust backstop
       and emits the misleading ``frontier()`` text — the impl agent must
       add an apply-specific check.
    """

    def _make_grid(self, n_quotes: int = 20, n_steps: int = 5):
        """Build a minimal QuoteGrid for the rejection tests.

        We need an actual valid grid to feed in — None-threshold rejection
        is about the constraints dict, not the grid itself. Build via a
        warmup numeric solve so the grid has the same shape callers see
        in normal usage. Include both ``volume`` and ``loss_ratio`` so
        the grid has constraint columns matching every test case below
        (otherwise ``validate_constraints_dict`` rejects on
        "Constraint not found in DataFrame columns" before reaching the
        None-rejection branch).
        """
        df = make_small_df(n_quotes=n_quotes, n_steps=n_steps)
        warmup = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 1.0},
                "loss_ratio": {"max_pct": 1.0},
            },
            max_iter=1,
        )
        return warmup.solve(df).grid

    def test_solve_from_grid_py_rust_direct_with_none_raises(self):
        """``_price_contour.solve_from_grid_py`` called directly (bypassing
        the Python ``_reject_none_for_solve`` shim) must raise ``ValueError``
        from the Rust backstop in ``parse_constraints``.

        This pins the backstop so a future refactor of the Python shim
        cannot leave None thresholds reaching the inner solver via a
        Rust-direct caller (e.g. an external Python user importing the
        compiled module's symbol directly).
        """
        from price_contour._price_contour import solve_from_grid_py

        grid = self._make_grid()

        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_VOLUME):
            solve_from_grid_py(
                grid,
                constraints={"volume": {"min": None}},
            )

    def test_solve_from_grid_py_rust_direct_with_min_pct_none_raises(self):
        """Same backstop must fire for ``min_pct: None`` (not just ``min``)."""
        from price_contour._price_contour import solve_from_grid_py

        grid = self._make_grid()

        with pytest.raises(ValueError, match=RE_SOLVE_REJECTION_VOLUME):
            solve_from_grid_py(
                grid,
                constraints={"volume": {"min_pct": None}},
            )

    def test_apply_from_grid_helper_with_none_raises(self):
        """``apply_from_grid()`` with a None-threshold constraint must
        raise ``ValueError`` with an apply-specific message.

        Apply mode has no frontier, so the user-facing error must NOT
        suggest ``frontier()`` as a way out — that would be misleading
        for an apply caller. The impl agent must add a Python-side check
        in ``apply_from_grid()`` mirroring ``ApplyOptimiser.__init__``'s
        ``_none_threshold_constraints`` rejection.

        Currently this test FAILS because the helper falls through to
        the Rust backstop in ``parse_constraints``, which raises the
        generic "Use frontier()..." message — that text does not match
        ``RE_APPLY_REJECTS_NONE`` (which requires "apply" near
        "None"/"fixed").
        """
        from price_contour.apply import apply_from_grid

        grid = self._make_grid()

        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            apply_from_grid(
                grid,
                lambdas={"volume": 0.0},
                constraints={"volume": {"min": None}},
            )

    def test_apply_from_grid_helper_with_max_pct_none_raises(self):
        """Symmetric coverage for ``max_pct: None`` on apply_from_grid."""
        from price_contour.apply import apply_from_grid

        grid = self._make_grid()

        with pytest.raises(ValueError, match=RE_APPLY_REJECTS_NONE):
            apply_from_grid(
                grid,
                lambdas={"loss_ratio": 0.0},
                constraints={"loss_ratio": {"max_pct": None}},
            )
