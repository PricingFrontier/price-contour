"""Feature C1 — ratio constraint parsing & validation.

This file covers the C1 slice of the ratio-constraint roadmap: dict-shape
parsing, validation rules, and stub errors at every solve / frontier / apply
entry point. The actual ratio linearisation lands in C2; this file pins the
contract that everything UP TO solving is in place.

C1 contract recap
-----------------
A ratio constraint spec contains BOTH ``numerator`` and ``denominator`` keys
(strings naming columns in the scored DataFrame) plus exactly one direction
key (``min`` / ``max`` / ``min_pct`` / ``max_pct``). The dict key is a
*display label*, not a column name (in contrast to sum constraints, where the
dict key IS the column name).

Validation rules (all enforced at construction):
* Both ``numerator`` and ``denominator`` must be strings.
* Missing one of the two pair keys → error naming the missing key.
* ``numerator == denominator`` → degenerate (always 1.0); error.
* Exactly one direction key alongside the pair (matching the sum-constraint
  rule); zero or multiple → error.
* Direction values follow B1: numeric or ``None``; NaN/inf rejected.
* The ``min_abs`` / ``max_abs`` migration error fires before the ratio
  detection branch.

Schema validation against the DataFrame:
* Both numerator and denominator columns must exist.
* Both columns must be non-null.

Solve / frontier / apply entry points all emit a ``NotImplementedError`` for
ratio constraints, with a message naming the constraint label and mentioning
"C2" or "not yet supported".

These tests should fail today: the current validator hits the "exactly one
key" or "must be one of: min, max, min_pct, max_pct" branch before it ever
sees the ratio shape.
"""

from __future__ import annotations


import polars as pl
import pytest

import price_contour as pc
from price_contour.solver import _validate_constraint_dict, _validate_dataframe
from helpers import make_small_df, make_factors


# ---------------------------------------------------------------------------
# Helpers — DataFrames with the four ratio columns
# ---------------------------------------------------------------------------


def make_ratio_df(n_quotes: int = 50, n_steps: int = 5) -> pl.DataFrame:
    """Build a small DataFrame with the standard ratio columns.

    Adds ``incurred``, ``premium``, ``claims_plus_expenses``, and ``expenses``
    on top of the base ``volume`` / ``loss_ratio`` columns from
    :func:`helpers.make_small_df`. The values are synthetic but non-degenerate
    so a future C2 solver could use the same fixture.
    """
    base = make_small_df(n_quotes=n_quotes, n_steps=n_steps)
    n = base.shape[0]
    # Synthetic financial columns. Premium scales with the base scenario_value
    # so the ratio incurred / premium has a non-trivial variation.
    premiums = [80.0 + 50.0 * float(base["scenario_value"][i]) for i in range(n)]
    incurred = [premiums[i] * 0.55 for i in range(n)]
    expenses = [premiums[i] * 0.18 for i in range(n)]
    claims_plus_expenses = [incurred[i] + expenses[i] for i in range(n)]
    return base.with_columns(
        [
            pl.Series("incurred", incurred, dtype=pl.Float32),
            pl.Series("premium", premiums, dtype=pl.Float32),
            pl.Series("expenses", expenses, dtype=pl.Float32),
            pl.Series(
                "claims_plus_expenses", claims_plus_expenses, dtype=pl.Float32
            ),
        ]
    )


def make_ratio_df_with_nulls(
    null_col: str, n_quotes: int = 30, n_steps: int = 5
) -> pl.DataFrame:
    """Same as :func:`make_ratio_df` but injects nulls into ``null_col``."""
    df = make_ratio_df(n_quotes=n_quotes, n_steps=n_steps)
    series = df[null_col].to_list()
    series[0] = None  # one null is enough for the validator to fire
    return df.with_columns(pl.Series(null_col, series, dtype=pl.Float32))


# Regex helpers used by the stub-error tests. The contract is:
# the message must mention either "C2" or "not yet supported"/"not yet
# implemented", AND it must name the offending constraint label so the user
# can act on it.
RE_C2_STUB = (
    r"(?si)C2|not yet supported|not yet implemented|ratio constraints"
)


# Migration regexes shared with A1.
RE_MIN_ABS_REMOVED = r"(?s)min_abs.*\bmin\b|\bmin\b.*min_abs"


# ---------------------------------------------------------------------------
# 1. Pure validator: ratio dict shape acceptance / rejection.
# ---------------------------------------------------------------------------


class TestRatioConstraintValidation:
    """Direct calls to ``_validate_constraint_dict`` with ratio shapes.

    These do not touch the DataFrame; they only pin the dict-shape contract.
    """

    # --- Acceptance ------------------------------------------------------

    def test_validator_accepts_ratio_with_max(self):
        """Numeric absolute target via ``max`` is accepted."""
        _validate_constraint_dict(
            {
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            }
        )

    def test_validator_accepts_ratio_with_min(self):
        """``min`` direction also valid for a ratio (e.g. retention rate floor)."""
        _validate_constraint_dict(
            {
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min": 0.95,
                },
            }
        )

    def test_validator_accepts_ratio_with_max_pct(self):
        _validate_constraint_dict(
            {
                "combined_ratio": {
                    "numerator": "claims_plus_expenses",
                    "denominator": "premium",
                    "max_pct": 1.10,
                },
            }
        )

    def test_validator_accepts_ratio_with_min_pct(self):
        """``min_pct`` ratios validate cleanly at the validator layer
        even though the C2 stub fires later at solve time. Pinned as a
        pure-validator counterpart to
        :meth:`TestRatioStubErrorsAtSolve.test_online_solve_with_min_pct_ratio_raises`
        so that the validator/solve-stub split is preserved."""
        # Must NOT raise — the spec is a well-formed ratio with the
        # `min_pct` direction key. The `NotImplementedError` only fires
        # later, at solve / apply / frontier.
        _validate_constraint_dict(
            {
                "retention_ratio": {
                    "numerator": "kept",
                    "denominator": "exposed",
                    "min_pct": 0.95,   # 95% of baseline retention ratio
                },
            }
        )

    def test_validator_accepts_ratio_with_none_threshold(self):
        """B1's None contract carries through to ratio specs."""
        _validate_constraint_dict(
            {
                "expense_ratio": {
                    "numerator": "expenses",
                    "denominator": "premium",
                    "max": None,
                },
            }
        )
        _validate_constraint_dict(
            {
                "expense_ratio": {
                    "numerator": "expenses",
                    "denominator": "premium",
                    "max_pct": None,
                },
            }
        )

    def test_validator_accepts_mixed_sum_and_ratio(self):
        """Sum and ratio constraints in the same dict validate cleanly."""
        _validate_constraint_dict(
            {
                "volume": {"min": 8000.0},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            }
        )

    def test_validator_accepts_label_matching_a_column_name(self):
        """The display label is just a label — overlapping with a real
        column name is fine. Pinned because users reach for natural names."""
        _validate_constraint_dict(
            {
                "premium": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            }
        )

    def test_validator_accepts_numerator_matching_other_sum_constraint_column(
        self,
    ):
        """Ratio numerator can coincide with a sum constraint's column;
        both can read from the same DataFrame column."""
        _validate_constraint_dict(
            {
                "incurred": {"min": 1000.0},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            }
        )

    # --- Rejection: shape errors ----------------------------------------

    def test_missing_numerator_raises_naming_label(self):
        """``denominator`` present, ``numerator`` absent → error names both
        ``numerator`` and the constraint label so the user can locate it."""
        with pytest.raises(ValueError) as exc_info:
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "denominator": "premium",
                        "max": 0.65,
                    },
                }
            )
        msg = str(exc_info.value)
        assert "numerator" in msg, (
            f"error message {msg!r} must name the missing 'numerator' key"
        )
        assert "loss_ratio" in msg, (
            f"error message {msg!r} must name the constraint label"
        )

    def test_missing_denominator_raises_naming_label(self):
        with pytest.raises(ValueError) as exc_info:
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "max": 0.65,
                    },
                }
            )
        msg = str(exc_info.value)
        assert "denominator" in msg, (
            f"error message {msg!r} must name the missing 'denominator' key"
        )
        assert "loss_ratio" in msg, (
            f"error message {msg!r} must name the constraint label"
        )

    def test_neither_pair_key_falls_through_to_sum_rules(self):
        """Without numerator OR denominator, the spec is a sum constraint
        and follows existing sum-constraint rules. A well-formed sum spec
        in the same dict must validate. Pinned so the impl agent's
        ratio-detection branch must be a strict ``and`` (BOTH keys), not
        a loose ``or`` (EITHER), or this test fails as a false-positive
        ratio detection."""
        # Pure sum dict — no numerator or denominator anywhere.
        _validate_constraint_dict({"volume": {"min": 8000.0}})

    def test_numerator_not_a_string_raises(self):
        """Non-string numerator (e.g. int, list, dict) → ValueError.

        The error message must name the constraint label AND the
        offending pair-key (``numerator``) so the user can locate the
        typo without inspecting the validator source. The regex pins
        both substrings; either ordering is allowed so wording can
        evolve."""
        # int — message must mention the label and 'numerator'.
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*numerator|numerator.*loss_ratio",
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": 42,
                        "denominator": "premium",
                        "max": 0.65,
                    },
                }
            )
        # list
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*numerator|numerator.*loss_ratio",
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": ["incurred"],
                        "denominator": "premium",
                        "max": 0.65,
                    },
                }
            )
        # dict
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*numerator|numerator.*loss_ratio",
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": {"col": "incurred"},
                        "denominator": "premium",
                        "max": 0.65,
                    },
                }
            )

    def test_denominator_not_a_string_raises(self):
        """Mirror of the numerator test: the regex pins the label and
        ``denominator``."""
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*denominator|denominator.*loss_ratio",
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": 3.14,
                        "max": 0.65,
                    },
                }
            )
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*denominator|denominator.*loss_ratio",
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": ["premium"],
                        "max": 0.65,
                    },
                }
            )

    def test_pair_value_both_non_string_raises_on_numerator_first(self):
        """When BOTH ``numerator`` and ``denominator`` are non-strings,
        the validator must report ``numerator`` first — not surface the
        second pair-key's failure or batch them together. Pinned because
        first-failure ordering is part of the C1 contract: users iterate
        over errors one at a time, and a flapping order makes for a poor
        debugging loop. The regex requires ``numerator`` AND forbids
        ``denominator`` so the test fails if the order flips."""
        with pytest.raises(ValueError, match=r"numerator") as exc_info:
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": 42,        # int, not a string
                        "denominator": [1, 2],  # list, also not a string
                        "max": 0.65,
                    },
                }
            )
        msg = str(exc_info.value)
        # The numerator failure wins; denominator must NOT appear yet.
        assert "denominator" not in msg, (
            f"numerator failure must surface BEFORE denominator; got {msg!r}"
        )

    def test_numerator_equals_denominator_raises(self):
        """Same column on both sides yields a degenerate ratio (always 1.0)
        and is almost always a typo. Error message must name BOTH the
        column AND the constraint label so the user can fix it."""
        with pytest.raises(ValueError) as exc_info:
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "premium",
                        "denominator": "premium",
                        "max": 0.65,
                    },
                }
            )
        msg = str(exc_info.value)
        assert "premium" in msg, (
            f"error message {msg!r} must name the duplicated column"
        )
        assert "loss_ratio" in msg, (
            f"error message {msg!r} must name the constraint label"
        )

    def test_zero_direction_keys_raises(self):
        """numerator + denominator alone (no min/max/etc.) → error.
        The user must commit to a direction; the message must name the
        constraint label and indicate which direction keys are valid."""
        with pytest.raises(ValueError) as exc_info:
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                    },
                }
            )
        msg = str(exc_info.value)
        assert "loss_ratio" in msg, (
            f"error message {msg!r} must name the constraint label"
        )

    def test_multiple_direction_keys_raises_min_and_max(self):
        """``min`` and ``max`` together → error matching the sum-constraint
        rule. Most users won't hit this, but it's the same shape as
        sum constraints (which already reject this) so we match."""
        with pytest.raises(ValueError) as exc_info:
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "min": 0.50,
                        "max": 0.65,
                    },
                }
            )
        assert "loss_ratio" in str(exc_info.value)

    def test_multiple_direction_keys_raises_max_and_max_pct(self):
        """Two direction keys (max + max_pct) → error names the label.
        Pin both the label AND a recognisable direction-key token so the
        message can't drift to a generic ``invalid spec`` while still
        passing this test."""
        with pytest.raises(
            ValueError, match=r"loss_ratio.*(max_pct|max)"
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": 0.65,
                        "max_pct": 1.05,
                    },
                }
            )

    def test_min_abs_alongside_ratio_still_raises_migration_error(self):
        """``min_abs`` is removed; the migration error must fire even
        when the spec is otherwise a ratio shape. Pinned so the ratio
        detection branch does NOT short-circuit the rename hint."""
        with pytest.raises(ValueError, match=RE_MIN_ABS_REMOVED):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "min_abs": 0.65,
                    },
                }
            )

    def test_nan_threshold_rejected_for_ratio(self):
        """B1 NaN/inf rejection still applies to ratio direction values.
        Pin the label, the offending direction key, and the keyword
        ``finite`` so a future wording change still must mention the
        finiteness rule."""
        with pytest.raises(
            ValueError, match=r"loss_ratio.*max.*finite|finite.*loss_ratio"
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": float("nan"),
                    },
                }
            )

    def test_inf_threshold_rejected_for_ratio(self):
        """Same as the NaN test for ``inf``; the offending key here is
        ``max_pct`` so pin that too."""
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*max_pct.*finite|finite.*loss_ratio",
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max_pct": float("inf"),
                    },
                }
            )

    def test_string_threshold_rejected_for_ratio(self):
        """Non-numeric, non-None direction value still rejected. Pin the
        label, the offending direction key, and ``numeric`` so a wording
        drift still can't pass without mentioning the type contract."""
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*max.*numeric|numeric.*loss_ratio",
        ):
            _validate_constraint_dict(
                {
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": "0.65",
                    },
                }
            )


# ---------------------------------------------------------------------------
# 2. Optimiser construction with ratio constraints.
# ---------------------------------------------------------------------------


class TestRatioConstraintConstructionPasses:
    """Construction must succeed for well-formed ratio specs across all
    optimisers. Solve/apply still errors at the C1 stub — that's covered
    in :class:`TestRatioStubErrorsAtSolve`."""

    def test_online_optimiser_constructs_with_ratio(self):
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        # Constraint dict is preserved verbatim so the impl agent /
        # downstream code can introspect it.
        assert opt.constraints == {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.65,
            },
        }

    def test_online_optimiser_constructs_with_ratio_max_pct(self):
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "combined_ratio": {
                    "numerator": "claims_plus_expenses",
                    "denominator": "premium",
                    "max_pct": 1.10,
                },
            },
        )
        assert opt.constraints["combined_ratio"]["max_pct"] == 1.10

    def test_online_optimiser_constructs_with_ratio_none_threshold(self):
        """A frontier-only ratio constraint (None threshold) constructs OK."""
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "expense_ratio": {
                    "numerator": "expenses",
                    "denominator": "premium",
                    "max": None,
                },
            },
        )
        assert opt.constraints["expense_ratio"]["max"] is None

    def test_ratebook_optimiser_constructs_with_ratio(self):
        opt = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
            factor_columns=[["region"]],
        )
        assert opt.constraints["loss_ratio"]["numerator"] == "incurred"

    def test_apply_optimiser_constructs_with_ratio_numeric_target(self):
        """Apply construction must accept a numeric ratio target. The
        actual apply call still errors at the C1 stub — that's tested
        below. ``None`` thresholds are still rejected by Apply (B1 rule)."""
        opt = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.0},
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        assert opt.constraints["loss_ratio"]["denominator"] == "premium"

    def test_apply_optimiser_rejects_ratio_none_threshold(self):
        """B1 still applies: Apply rejects None thresholds, ratio or not.
        Pin both the constraint label and the keyword ``None`` so a
        wording change still must reference the actual cause."""
        with pytest.raises(
            ValueError, match=r"loss_ratio.*None|None.*loss_ratio"
        ):
            pc.ApplyOptimiser(
                lambdas={"loss_ratio": 0.0},
                objective="expected_income",
                constraints={
                    "loss_ratio": {
                        "numerator": "incurred",
                        "denominator": "premium",
                        "max": None,
                    },
                },
            )

    def test_online_optimiser_constructs_with_mixed_sum_and_ratio(self):
        opt = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min": 8000.0},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        assert opt.constraints["volume"] == {"min": 8000.0}
        assert opt.constraints["loss_ratio"]["max"] == 0.65


# ---------------------------------------------------------------------------
# 3. DataFrame schema validation for ratio numerator/denominator columns.
# ---------------------------------------------------------------------------


class TestRatioColumnValidation:
    """Validation against the input DataFrame: numerator and denominator
    columns must exist and be non-null. Errors must name the offending
    column AND the constraint label so the user can fix it without
    spelunking the DataFrame schema."""

    def test_numerator_column_missing_raises(self):
        """``numerator`` names a column not in the DataFrame → ValueError
        naming the missing column AND the constraint label."""
        df = make_ratio_df(n_quotes=30)
        # Drop the numerator column so we can pin the missing-column error.
        df = df.drop("incurred")
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df)
        msg = str(exc_info.value)
        assert "incurred" in msg, (
            f"error {msg!r} must name the missing numerator column"
        )
        assert "loss_ratio" in msg, (
            f"error {msg!r} must name the constraint label"
        )

    def test_denominator_column_missing_raises(self):
        df = make_ratio_df(n_quotes=30).drop("premium")
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df)
        msg = str(exc_info.value)
        assert "premium" in msg, (
            f"error {msg!r} must name the missing denominator column"
        )
        assert "loss_ratio" in msg, (
            f"error {msg!r} must name the constraint label"
        )

    def test_numerator_column_has_nulls_raises(self):
        df = make_ratio_df_with_nulls("incurred", n_quotes=30)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df)
        msg = str(exc_info.value)
        assert "incurred" in msg, (
            f"error {msg!r} must name the column containing nulls"
        )

    def test_denominator_column_has_nulls_raises(self):
        df = make_ratio_df_with_nulls("premium", n_quotes=30)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        with pytest.raises(ValueError) as exc_info:
            solver.solve(df)
        msg = str(exc_info.value)
        assert "premium" in msg, (
            f"error {msg!r} must name the column containing nulls"
        )

    def test_frontier_path_validates_numerator_column(self):
        """Same schema check applies to ``frontier()``."""
        df = make_ratio_df(n_quotes=30).drop("incurred")
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                },
            },
        )
        # Frontier validation must fire on the same missing-column rule.
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(
                df,
                threshold_ranges={"loss_ratio": (0.5, 0.7)},
                n_points_per_dim=3,
            )
        msg = str(exc_info.value)
        assert "incurred" in msg, (
            f"error {msg!r} must name the missing numerator column"
        )

    def test_frontier_path_validates_denominator_column(self):
        df = make_ratio_df(n_quotes=30).drop("premium")
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": None,
                },
            },
        )
        with pytest.raises(ValueError) as exc_info:
            solver.frontier(
                df,
                threshold_ranges={"loss_ratio": (0.5, 0.7)},
                n_points_per_dim=3,
            )
        msg = str(exc_info.value)
        assert "premium" in msg, (
            f"error {msg!r} must name the missing denominator column"
        )

    def test_label_does_not_need_to_be_a_column(self):
        """The ratio constraint's *label* (dict key) is purely a display
        name. It does NOT need to exist as a column in the DataFrame —
        in contrast to sum constraints, where the dict key IS the column
        name. Pinned because this is the C1 design's central distinction.
        """
        df = make_ratio_df(n_quotes=30)
        # ``my_made_up_label`` is not a column. The ratio columns ARE.
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "my_made_up_label": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        # Post-C2: solve completes (the label is fine; ratio solving is
        # implemented). Pre-C2 it stubbed at NotImplementedError. The point
        # is that DataFrame validation does NOT reject the label —
        # whatever the outcome, it must NOT be a
        # "Missing required columns: ['my_made_up_label']" error.
        try:
            result = solver.solve(df)
            # Solve succeeded — the label was accepted as a label.
            assert "my_made_up_label" in result.lambdas
        except ValueError as e:
            # If solve raises a ValueError, it must NOT name the label as
            # a missing column.
            msg = str(e)
            assert "Missing required columns" not in msg or (
                "my_made_up_label" not in msg
            ), f"label was incorrectly treated as a column name: {msg}"

    def test_validate_dataframe_directly_rejects_missing_denominator(self):
        """Exercise the column-existence check via ``_validate_dataframe``
        directly, without going through ``OnlineOptimiser.solve()``.

        The ``_validate_dataframe`` helper is module-private but is the
        single source of truth for schema validation. Pinning it directly
        makes the column-existence check independent of the
        ``OnlineOptimiser`` /  ``ApplyOptimiser`` call sites — if either
        caller-level wiring breaks, the validator's own contract is still
        covered."""
        df = make_ratio_df(n_quotes=20).drop("premium")
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.65,
            },
        }
        with pytest.raises(ValueError) as exc_info:
            _validate_dataframe(
                df,
                quote_id="quote_id",
                scenario_index="scenario_index",
                scenario_value="scenario_value",
                objective="expected_income",
                constraint_cols=list(constraints.keys()),
                constraints=constraints,
            )
        msg = str(exc_info.value)
        assert "premium" in msg, (
            f"validator error {msg!r} must name the missing denominator column"
        )
        assert "loss_ratio" in msg, (
            f"validator error {msg!r} must name the constraint label"
        )

    def test_validate_dataframe_directly_rejects_missing_numerator(self):
        """Mirror of the denominator test for the numerator column."""
        df = make_ratio_df(n_quotes=20).drop("incurred")
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.65,
            },
        }
        with pytest.raises(ValueError) as exc_info:
            _validate_dataframe(
                df,
                quote_id="quote_id",
                scenario_index="scenario_index",
                scenario_value="scenario_value",
                objective="expected_income",
                constraint_cols=list(constraints.keys()),
                constraints=constraints,
            )
        msg = str(exc_info.value)
        assert "incurred" in msg, (
            f"validator error {msg!r} must name the missing numerator column"
        )
        assert "loss_ratio" in msg, (
            f"validator error {msg!r} must name the constraint label"
        )

    def test_apply_optimiser_ratio_numerator_column_missing_raises(self):
        """Mirror of :meth:`test_numerator_column_missing_raises` (which
        runs through ``OnlineOptimiser``) but for the apply path.

        Schema validation must run BEFORE the C2 stub on the apply path
        too: a missing numerator column produces a ``ValueError`` naming
        the column AND the constraint label, NOT a generic
        ``NotImplementedError`` from the ratio stub. Pinning this stops
        a future refactor from accidentally re-ordering the apply-path
        gates so the C2 stub fires first and swallows the more useful
        schema error."""
        df = make_ratio_df(n_quotes=30).drop("incurred")
        optimiser = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.5},
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",   # not in df anymore
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*incurred|incurred.*loss_ratio",
        ):
            optimiser.apply(df)

    def test_apply_optimiser_ratio_denominator_column_missing_raises(self):
        """Same as the numerator test but for the denominator column.
        Pinned for symmetry with the OnlineOptimiser denominator test
        and to cover the second branch of the apply-path schema check."""
        df = make_ratio_df(n_quotes=30).drop("premium")
        optimiser = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.5},
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",   # not in df anymore
                    "max": 0.65,
                },
            },
        )
        with pytest.raises(
            ValueError,
            match=r"loss_ratio.*premium|premium.*loss_ratio",
        ):
            optimiser.apply(df)


# ---------------------------------------------------------------------------
# 4. Stub errors at solve / apply / frontier (placeholder for C2).
# ---------------------------------------------------------------------------


class TestRatioStubErrorsAtSolve:
    """Every entry point that would actually run a solve / apply with a
    ratio constraint raises ``NotImplementedError`` for C1.

    The error class is pinned at ``NotImplementedError`` (per the C1 spec
    design rule 4) for clarity — a future C2 implementation will simply
    remove this raise. The error message must mention either ``C2`` or
    ``not yet supported`` / ``not yet implemented`` AND name the offending
    constraint label.
    """

    def test_online_solve_from_grid_with_ratio_raises_value_error(self):
        """The pre-built QuoteGrid path does not carry numerator/denominator
        column metadata, so ratio-constraint solves on a grid raise
        ``ValueError`` (not ``NotImplementedError``) — the user must pass
        a DataFrame for ratio linearisation. Online's DataFrame solve path
        was lifted in C2, so the only remaining stub is this grid path.
        """
        df = make_ratio_df(n_quotes=30)
        # Build a grid via a numeric sum-constraint solve so we have a
        # real grid to pass back.
        warmup = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.0}},
            max_iter=1,
        )
        grid = warmup.solve(df).grid

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.65,
                },
            },
        )
        with pytest.raises(ValueError, match=r"loss_ratio"):
            solver.solve(grid)

    # NOTE: ``test_online_frontier_with_ratio_raises`` (C1-era stub
    # state-machine pin) was removed when C4 lit up the ratio frontier
    # path. The replacement regression guard lives in
    # ``test_ratio_frontier_c4.py::TestRatioFrontierStubsRemovedForOnlineOnly::test_online_frontier_with_ratio_no_longer_raises``.

    # NOTE: ``test_ratebook_solve_with_ratio_raises`` and
    # ``test_ratebook_frontier_with_ratio_raises`` (C1-era stub
    # state-machine pins) were removed when C5 lit up the ratebook
    # ratio paths. The replacement regression guards live in
    # ``test_ratio_ratebook_c5.py::TestRatebookRatioStubsRemoved``.

    # NOTE: the C1-era stub-pin tests for ApplyOptimiser.apply,
    # apply_from_grid, and mixed-apply were removed when C6 lit up the
    # apply ratio paths. The replacement regression guards live in
    # ``test_ratio_apply_c6.py::TestApplyOptimiserRatioStubsRemoved``.


# ---------------------------------------------------------------------------
# 5. Sum-constraint regression guard.
# ---------------------------------------------------------------------------


class TestSumConstraintsUnchanged:
    """Adding the ratio path must not regress sum-constraint behaviour.
    These tests duplicate small parts of A1 to give a quick canary
    against the C1 impl agent's changes — full A1 coverage still runs in
    test_constraint_keys_a1.py."""

    def test_sum_constraint_solve_still_works(self):
        """A pure sum-constraint solve still completes and respects the
        constraint within the standard tolerance."""
        df = make_small_df(n_quotes=100, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
            max_iter=200,
        )
        result = solver.solve(df)
        baseline = result.baseline_constraints["volume"]
        assert result.total_constraints["volume"] >= baseline * 0.9 * 0.98

    def test_sum_constraint_frontier_still_works(self):
        """A pure sum-constraint frontier still produces the right
        number of points."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.0}},
            max_iter=50,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 0.95)},
            n_points_per_dim=3,
        )
        assert result.n_points == 3

    def test_sum_constraint_validator_unchanged(self):
        """The validator still accepts the four direction keys for sum
        constraints with no ratio-detection false positives."""
        _validate_constraint_dict({"volume": {"min": 100.0}})
        _validate_constraint_dict({"loss_ratio": {"max": 1.5}})
        _validate_constraint_dict({"volume": {"min_pct": 0.9}})
        _validate_constraint_dict({"loss_ratio": {"max_pct": 1.05}})

    def test_sum_constraint_apply_still_works(self):
        """ApplyOptimiser.apply with a pure sum constraint still produces
        an ApplyResult — no NotImplementedError."""
        df = make_small_df(n_quotes=50, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.05},
            objective="expected_income",
            constraints={"volume": {"min": 1.0}},
        )
        result = applier.apply(df)
        # Sanity: result has the expected attribute surface.
        assert hasattr(result, "total_objective")
        assert hasattr(result, "total_constraints")
        assert "volume" in result.total_constraints

    def test_sum_constraint_ratebook_still_works(self):
        """Ratebook solve with a pure sum constraint still succeeds."""
        n = 30
        df = make_small_df(n_quotes=n, n_steps=5)
        factors = make_factors(n)
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=20,
        )
        result = solver.solve(df, factors)
        assert hasattr(result, "factor_tables")
