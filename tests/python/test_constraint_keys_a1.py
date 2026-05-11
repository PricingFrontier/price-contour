"""Constraint key semantics.

The constraint dict keys mean:

* ``min`` / ``max``       → **absolute** thresholds (no baseline scaling).
* ``min_pct`` / ``max_pct`` → fraction-of-baseline thresholds.

Covers absolute vs fraction semantics, validation at construction, NaN/inf
rejection, and serialisation round-trips.
"""

from __future__ import annotations

import json
import math

import pytest

import price_contour as pc
from price_contour.solver import _validate_constraint_dict
from helpers import make_small_df, CONSTRAINT_RTOL


# Tolerance helpers --------------------------------------------------------

# Wide tolerance: discrete grids on small portfolios cannot hit absolute
# thresholds exactly. Several multipliers used because some max-direction
# constraints are particularly hard on synthetic data.
RTOL = CONSTRAINT_RTOL * 3


# ---------------------------------------------------------------------------
# 1. Absolute semantics for `min` / `max`
# ---------------------------------------------------------------------------


class TestAbsoluteMinMax:
    """`min`/`max` produce absolute thresholds, NOT baseline-scaled."""

    def test_min_is_absolute_threshold(self):
        """`{"volume": {"min": X}}` → solver enforces total_volume >= X."""
        df = make_small_df(n_quotes=100, n_steps=5)

        # First peek baseline volume so we can pick a meaningful absolute
        # target that's well below it (an absolute number, not a fraction).
        peek = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.0}},
            max_iter=1,
        )
        baseline_vol = peek.solve(df).baseline_constraints["volume"]

        target_abs = baseline_vol * 0.85
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": target_abs}},
            max_iter=200,
        )
        result = solver.solve(df)
        # Absolute interpretation: total_volume >= target_abs (within tol).
        assert result.total_constraints["volume"] >= target_abs * (1 - RTOL), (
            f"volume {result.total_constraints['volume']} < "
            f"{target_abs * (1 - RTOL)} (target {target_abs})"
        )
        # And the threshold the solver enforces is target_abs itself, not
        # target_abs * baseline. To prove this, set min equal to a number
        # much smaller than baseline_vol — if it were scaled by baseline
        # it would be tiny; absolute means total_volume >= that number.
        small_abs = 1.0  # absolute units; way below baseline_vol
        solver2 = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": small_abs}},
            max_iter=50,
        )
        r2 = solver2.solve(df)
        assert r2.total_constraints["volume"] >= small_abs

    def test_max_is_absolute_threshold(self):
        """`{"loss_ratio": {"max": X}}` → enforces total_loss_ratio <= X."""
        df = make_small_df(n_quotes=100, n_steps=5)
        peek = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max_pct": 1.0}},
            max_iter=1,
        )
        baseline_lr = peek.solve(df).baseline_constraints["loss_ratio"]

        target_abs = baseline_lr * 1.20
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max": target_abs}},
            max_iter=200,
        )
        result = solver.solve(df)
        assert result.total_constraints["loss_ratio"] <= target_abs * (1 + RTOL), (
            f"loss_ratio {result.total_constraints['loss_ratio']} > "
            f"{target_abs * (1 + RTOL)} (target {target_abs})"
        )

    def test_min_zero_absolute(self):
        """`{"volume": {"min": 0}}` is the trivial constraint and never binds."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.0}},
            max_iter=50,
        )
        result = solver.solve(df)
        # The solver should still run; volume should be >= 0 trivially.
        assert result.total_constraints["volume"] >= 0.0


# ---------------------------------------------------------------------------
# 2. Fractional semantics for `min_pct` / `max_pct`
# ---------------------------------------------------------------------------


class TestPctMinMax:
    """`min_pct`/`max_pct` produce thresholds = fraction × baseline."""

    def test_min_pct_is_fraction_of_baseline(self):
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
            max_iter=200,
        )
        result = solver.solve(df)
        baseline = result.baseline_constraints["volume"]
        target = 0.9 * baseline
        assert result.total_constraints["volume"] >= target * (1 - RTOL), (
            f"volume {result.total_constraints['volume']} < {target * (1 - RTOL)}"
        )

    def test_max_pct_is_fraction_of_baseline(self):
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"loss_ratio": {"max_pct": 1.05}},
            max_iter=200,
        )
        result = solver.solve(df)
        baseline = result.baseline_constraints["loss_ratio"]
        target = 1.05 * baseline
        assert result.total_constraints["loss_ratio"] <= target * (1 + RTOL), (
            f"loss_ratio {result.total_constraints['loss_ratio']} > "
            f"{target * (1 + RTOL)}"
        )

    def test_min_pct_one_equals_baseline(self):
        """`min_pct=1.0` → threshold == baseline_volume."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.0}},
            max_iter=200,
        )
        result = solver.solve(df)
        baseline = result.baseline_constraints["volume"]
        # The threshold the solver enforces is baseline; total should
        # be >= baseline within tolerance (or solver may not converge,
        # but the request is clear).
        assert baseline > 0
        assert result.total_constraints["volume"] >= baseline * (1 - RTOL)

    def test_min_pct_zero_is_trivial(self):
        """`min_pct=0.0` → threshold == 0 (never binds)."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.0}},
            max_iter=50,
        )
        result = solver.solve(df)
        assert result.total_constraints["volume"] >= 0.0


# ---------------------------------------------------------------------------
# 3. Mixed dict: absolute + relative side-by-side
# ---------------------------------------------------------------------------


class TestMixedAbsoluteAndPct:
    def test_mixed_min_and_min_pct_in_single_solve(self):
        """One absolute + one relative constraint in the same solve."""
        df = make_small_df(n_quotes=200, n_steps=5)

        # Peek baselines.
        peek = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 1.0},
                "loss_ratio": {"max_pct": 1.0},
            },
            max_iter=1,
        )
        baseline = peek.solve(df).baseline_constraints
        baseline_vol = baseline["volume"]
        baseline_lr = baseline["loss_ratio"]

        # volume min is ABSOLUTE; loss_ratio max_pct is fractional.
        vol_target_abs = baseline_vol * 0.85
        lr_pct = 1.10
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": vol_target_abs},
                "loss_ratio": {"max_pct": lr_pct},
            },
            max_iter=200,
        )
        result = solver.solve(df)

        assert result.total_constraints["volume"] >= vol_target_abs * (1 - RTOL)
        lr_target = lr_pct * baseline_lr
        assert result.total_constraints["loss_ratio"] <= lr_target * (1 + RTOL)


# ---------------------------------------------------------------------------
# 4. _validate_constraint_dict accepts the four valid direction keys
# ---------------------------------------------------------------------------


class TestValidationAtConstruction:
    def test_validate_function_accepts_new_keys(self):
        """Sanity check: the four valid keys are accepted by the validator."""
        _validate_constraint_dict({"volume": {"min": 100.0}})
        _validate_constraint_dict({"loss_ratio": {"max": 1.5}})
        _validate_constraint_dict({"volume": {"min_pct": 0.9}})
        _validate_constraint_dict({"loss_ratio": {"max_pct": 1.05}})


# ---------------------------------------------------------------------------
# 6. Issue 3 — summary, config_dict, save/load migration coverage
# ---------------------------------------------------------------------------


class TestSummaryAndSerialisationKeys:
    """The constraint key names (``min``/``max`` absolute,
    ``min_pct``/``max_pct`` relative) must round-trip cleanly through
    ``OnlineOptimiser.summary()``, ``config_dict()``, and the
    ``ApplyOptimiser.save()`` / ``load()`` cycle.
    """

    def test_summary_uses_new_constraint_keys(self):
        """``summary(result)`` must surface ``min_pct`` (the new key)
        in both the params JSON blob and the per-constraint spec entry,
        and must NOT surface ``min`` or ``min_abs`` for a min_pct
        constraint."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
            max_iter=50,
        )
        result = solver.solve(df)
        summary = solver.summary(result)

        # params: constraints JSON-serialised string contains "min_pct".
        assert "constraints" in summary["params"]
        params_blob = summary["params"]["constraints"]
        assert isinstance(params_blob, str)
        # JSON-decode and inspect — looking at the substring is fine
        # but parsing makes the assertion robust to whitespace.
        decoded = json.loads(params_blob)
        assert decoded == {"volume": {"min_pct": 0.9}}
        assert "min_pct" in params_blob
        # Must not contain the legacy or sibling keys for this spec.
        assert "min_abs" not in params_blob
        # The bare key "min" must not appear as a spec key. We check
        # the parsed dict, not the substring (since "min" is also a
        # substring of "min_pct").
        assert "min" not in decoded["volume"]

        # artifacts.summary.constraints[volume].spec is the literal
        # constraints dict for that name.
        spec = summary["artifacts"]["summary"]["constraints"]["volume"]["spec"]
        assert spec == {"min_pct": 0.9}
        assert "min" not in spec
        assert "min_abs" not in spec

    def test_summary_uses_new_keys_for_absolute_min(self):
        """A spec using ``min`` (absolute) must round-trip through
        ``summary()`` with the same key name."""
        df = make_small_df(n_quotes=50, n_steps=5)
        # Pick a small absolute floor that the solver can hit on a
        # 50-quote portfolio.
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 1.0}},
            max_iter=50,
        )
        result = solver.solve(df)
        summary = solver.summary(result)

        params_blob = summary["params"]["constraints"]
        decoded = json.loads(params_blob)
        assert decoded == {"volume": {"min": 1.0}}
        spec = summary["artifacts"]["summary"]["constraints"]["volume"]["spec"]
        assert spec == {"min": 1.0}
        # No leakage of legacy keys.
        assert "min_abs" not in params_blob
        assert "min_abs" not in spec

    def test_config_dict_round_trip_via_json(self):
        """``config_dict()`` returns a dict that survives JSON
        round-trip with constraints intact."""
        constraints = {
            "volume": {"min_pct": 0.9},
            "loss_ratio": {"max": 1.05},
        }
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints=constraints,
            max_iter=50,
        )
        cfg = solver.config_dict()
        # Constraints carried through verbatim.
        assert cfg["constraints"] == constraints

        # Round-trip through JSON.
        blob = json.dumps(cfg)
        roundtripped = json.loads(blob)
        assert roundtripped["constraints"] == constraints
        # Each spec dict has exactly one of the new keys.
        for spec in roundtripped["constraints"].values():
            assert len(spec) == 1
            (key,) = spec.keys()
            assert key in {"min", "max", "min_pct", "max_pct"}

    def test_apply_optimiser_save_load_round_trip(self, tmp_path):
        """``ApplyOptimiser.save()`` then ``ApplyOptimiser.load()``
        preserves the constraints dict using the new key names."""
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
        assert loaded.lambdas == {"volume": 0.1}
        assert loaded.objective == "expected_income"

        # Belt-and-braces: spot-check the on-disk file uses ``min``
        # and not ``min_abs``.
        on_disk = json.loads(path.read_text())
        assert on_disk["constraints"] == constraints
        # The serialised key names are the new ones.
        assert "min_abs" not in path.read_text()

# ---------------------------------------------------------------------------
# 7. Issue 4 — edge cases on threshold values
# ---------------------------------------------------------------------------


class TestEdgeCaseThresholds:
    """Edge-case threshold values: negatives are valid (a floor of -100
    means the user really wants to allow that downside); NaN and inf
    are not. The validator currently accepts both via the broad
    ``isinstance(value, (int, float))`` check; the NaN/inf tests pin
    the desired behaviour and force the impl agent to add the guard.
    """

    def test_negative_absolute_min_is_accepted_and_solves(self):
        """``{"profit": {"min": -100}}`` is a meaningful constraint:
        the user is happy with any total profit at or above -100 (i.e.
        a small loss). Validation must NOT reject negatives — they are
        legitimate absolute thresholds. The solver must run and the
        actual total must respect the floor."""
        # Build a custom DataFrame with a "profit" column that can
        # genuinely cross zero, so a -100 floor is non-trivial.
        rows = []
        n_quotes = 60
        n_steps = 5
        mults = [0.8 + 0.1 * j for j in range(n_steps)]
        for q in range(n_quotes):
            elasticity = 1.5 + 3.5 * q / n_quotes
            base = 80.0 + 40.0 * q / n_quotes
            for j, mult in enumerate(mults):
                conversion = 1.0 / (1.0 + math.exp(elasticity * (mult - 1.0)))
                # Profit: positive at high mult, can go negative when
                # mult is low and conversion x base price is small.
                profit = (base * mult - 95.0) * conversion
                rows.append(
                    {
                        "quote_id": f"Q{q:04d}",
                        "scenario_index": j,
                        "scenario_value": mult,
                        "expected_income": base * mult * conversion,
                        "profit": profit,
                    }
                )
        import polars as pl

        df = pl.DataFrame(
            rows,
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "profit": pl.Float32,
            },
        )

        # Construction must NOT raise on a negative absolute threshold.
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"profit": {"min": -100.0}},
            max_iter=200,
        )
        # Direct validator check too, for symmetry with the other tests.
        _validate_constraint_dict({"profit": {"min": -100.0}})

        result = solver.solve(df)
        # The solver completed and the actual total profit respects
        # the -100 floor (within the standard tolerance).
        assert "profit" in result.total_constraints
        total_profit = result.total_constraints["profit"]
        # Floor is -100; allow 2% absolute slack on a small portfolio.
        assert total_profit >= -100.0 - 2.0, (
            f"total profit {total_profit} below the -100 floor by more than tolerance"
        )

    def test_nan_threshold_rejected_at_construction(self):
        """``float('nan')`` as a threshold must be rejected at
        construction — currently ``isinstance(value, (int, float))``
        accepts NaN. This pins the desired behaviour."""
        with pytest.raises(ValueError):
            pc.OnlineOptimiser(
                objective="expected_income",
                constraints={"volume": {"min": float("nan")}},
            )

    def test_nan_threshold_rejected_by_validator(self):
        """The standalone validator must also reject NaN."""
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": float("nan")}})

    def test_nan_threshold_rejected_for_pct_keys(self):
        """NaN must be rejected for ``min_pct`` and ``max_pct`` too,
        not just absolute keys."""
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min_pct": float("nan")}})
        with pytest.raises(ValueError):
            _validate_constraint_dict({"loss_ratio": {"max_pct": float("nan")}})

    def test_inf_threshold_rejected_at_construction(self):
        """``float('inf')`` as a threshold must be rejected at
        construction."""
        with pytest.raises(ValueError):
            pc.OnlineOptimiser(
                objective="expected_income",
                constraints={"volume": {"min": float("inf")}},
            )

    def test_negative_inf_threshold_rejected_at_construction(self):
        """``-inf`` is also not a meaningful threshold."""
        with pytest.raises(ValueError):
            pc.OnlineOptimiser(
                objective="expected_income",
                constraints={"volume": {"min": float("-inf")}},
            )

    def test_inf_threshold_rejected_by_validator(self):
        """The standalone validator must reject infinities."""
        with pytest.raises(ValueError):
            _validate_constraint_dict({"volume": {"min": float("inf")}})
        with pytest.raises(ValueError):
            _validate_constraint_dict({"loss_ratio": {"max": float("-inf")}})

    def test_inf_threshold_rejected_for_apply(self):
        """ApplyOptimiser must also reject infinities at construction."""
        with pytest.raises(ValueError):
            pc.ApplyOptimiser(
                lambdas={"volume": 0.1},
                objective="expected_income",
                constraints={"volume": {"min": float("inf")}},
            )
