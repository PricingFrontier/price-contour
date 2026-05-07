"""Tests for ``ApplyOptimiser.with_explainer_columns``.

Pins the contract from ``PRICE_CONTOUR_APPLY_EXPLAINABILITY_SPEC.md``:
the helper returns the input DataFrame with optimiser-consistent
explainer columns appended. The selected candidate, sign convention,
ratio linearisation, and baseline marker all reconcile with
:meth:`ApplyOptimiser.apply` by construction.
"""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from helpers import make_small_df
from price_contour.solver import _spec_direction
from test_ratio_solve_c2 import make_ratio_solve_df, make_retention_df


# Floating-point tolerance for the Lagrangian reconstruction invariant.
# Apply's argmax scores in f32 (objective/constraints and lambdas are
# cast before accumulation), so the explainer columns intentionally
# mirror that precision.
SCORE_RTOL = 1e-6
SCORE_ABS = 1e-6


def _sum_constraint_applier(
    df: pl.DataFrame, constraint_thresholds: dict[str, float] | None = None
) -> pc.ApplyOptimiser:
    """Build an ApplyOptimiser by solving for converged lambdas first.

    Tests that need realistic non-zero lambdas (so the lambda-term
    columns aren't trivially zero) go through this helper.
    """
    constraints = {
        "volume": {"min_pct": (constraint_thresholds or {}).get("volume", 0.92)},
        "loss_ratio": {
            "max_pct": (constraint_thresholds or {}).get("loss_ratio", 1.05)
        },
    }
    solver = pc.OnlineOptimiser(
        objective="expected_income",
        constraints=constraints,
        max_iter=200,
    )
    solve_result = solver.solve(df)
    return pc.ApplyOptimiser(
        lambdas=solve_result.lambdas,
        objective="expected_income",
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# 1. Schema preservation
# ---------------------------------------------------------------------------


class TestSchemaPreservation:
    """Spec test 1: returned DataFrame has same row count and original
    columns as input, with explainer columns appended."""

    def test_row_count_preserved(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        assert out.height == df.height

    def test_original_columns_preserved(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        for col in df.columns:
            assert col in out.columns
            assert out[col].equals(df[col]), f"input column {col} mutated"

    def test_appended_columns_exist(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        # core columns
        for col in ("decision_score", "selected", "is_baseline"):
            assert col in out.columns
        # per-constraint columns
        for name in ("volume", "loss_ratio"):
            assert f"linearised_{name}" in out.columns
            assert f"lambda_term_{name}" in out.columns

    def test_appended_column_dtypes(self) -> None:
        df = make_small_df(n_quotes=10, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        assert out["decision_score"].dtype == pl.Float64
        assert out["selected"].dtype == pl.Boolean
        assert out["is_baseline"].dtype == pl.Boolean
        assert out["linearised_volume"].dtype == pl.Float32
        assert out["lambda_term_volume"].dtype == pl.Float32


# ---------------------------------------------------------------------------
# 2. selected matches apply(df)
# ---------------------------------------------------------------------------


class TestSelectedMatchesApply:
    """Spec test 2: ``selected`` matches ``ApplyOptimiser.apply(df)``."""

    def test_selected_matches_apply_per_quote(self) -> None:
        df = make_small_df(n_quotes=50, n_steps=5)
        applier = _sum_constraint_applier(df)
        apply_result = applier.apply(df)
        out = applier.with_explainer_columns(df)

        selected_rows = out.filter(pl.col("selected"))
        # exactly one row per quote
        assert selected_rows.height == apply_result.dataframe.height
        # selected scenario_index matches apply's optimal_step
        merged = (
            selected_rows.select(["quote_id", "scenario_index"])
            .sort("quote_id")
            .join(
                apply_result.dataframe.select(["quote_id", "optimal_step"]).sort(
                    "quote_id"
                ),
                on="quote_id",
            )
        )
        for row in merged.iter_rows(named=True):
            assert row["scenario_index"] == row["optimal_step"], (
                f"quote {row['quote_id']}: explainer selected "
                f"scenario_index={row['scenario_index']} but apply chose "
                f"optimal_step={row['optimal_step']}"
            )

    def test_exactly_one_selected_per_quote(self) -> None:
        df = make_small_df(n_quotes=50, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        per_quote = out.group_by("quote_id").agg(pl.col("selected").sum())
        for row in per_quote.iter_rows(named=True):
            assert row["selected"] == 1, (
                f"quote {row['quote_id']}: expected exactly 1 selected row, "
                f"got {row['selected']}"
            )


# ---------------------------------------------------------------------------
# 3. decision_score reconstructs from objective + lambda_terms
# ---------------------------------------------------------------------------


class TestDecisionScoreReconstruction:
    """Spec test 3: ``decision_score == objective + sum(lambda_term_*)``."""

    def test_score_invariant_sum_constraints(self) -> None:
        df = make_small_df(n_quotes=30, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)

        reconstructed = (
            out["expected_income"]
            + out["lambda_term_volume"]
            + out["lambda_term_loss_ratio"]
        )
        for actual, expected in zip(
            out["decision_score"].to_list(),
            reconstructed.cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(expected, rel=SCORE_RTOL, abs=SCORE_ABS)

    def test_score_invariant_no_constraints(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={},
            objective="expected_income",
            constraints={},
        )
        out = applier.with_explainer_columns(df)
        # decision_score == objective when there are no constraints
        for actual, expected in zip(
            out["decision_score"].to_list(),
            out["expected_income"].cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(expected, rel=SCORE_RTOL, abs=SCORE_ABS)


# ---------------------------------------------------------------------------
# 4. Sum constraints — correct linearised_<name> and lambda_term_<name>
# ---------------------------------------------------------------------------


class TestSumConstraints:
    """Spec test 4: sum constraints get correct linearised + lambda_term."""

    def test_linearised_equals_original_for_sum(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        # linearised_<name> for sum constraints == original column at
        # apply's f32 precision.
        for name in ("volume", "loss_ratio"):
            for actual, expected in zip(
                out[f"linearised_{name}"].to_list(),
                out[name].to_list(),
            ):
                assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)

    def test_lambda_term_sign_convention_min(self) -> None:
        """+lambda for min direction."""
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 2.5},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
        )
        out = applier.with_explainer_columns(df)
        # lambda_term_volume == +2.5 * volume
        for actual, vol in zip(
            out["lambda_term_volume"].to_list(),
            out["volume"].cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(2.5 * vol, rel=1e-6, abs=1e-7)

    def test_lambda_term_sign_convention_max(self) -> None:
        """-lambda for max direction."""
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 1.7},
            objective="expected_income",
            constraints={"loss_ratio": {"max_pct": 1.05}},
        )
        out = applier.with_explainer_columns(df)
        # lambda_term_loss_ratio == -1.7 * loss_ratio
        for actual, lr in zip(
            out["lambda_term_loss_ratio"].to_list(),
            out["loss_ratio"].cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(-1.7 * lr, rel=1e-6, abs=1e-7)

    def test_constraint_without_explicit_lambda_defaults_to_zero(self) -> None:
        """ApplyOptimiser permits a partial ``lambdas`` dict — missing
        keys default to zero in the apply path. The explainer should
        match: lambda_term_<name> = 0 * linearised_<name> = 0."""
        df = make_small_df(n_quotes=10, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 1.5},  # 'loss_ratio' deliberately omitted
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.9},
                "loss_ratio": {"max_pct": 1.05},
            },
        )
        out = applier.with_explainer_columns(df)
        # loss_ratio has no lambda -> term must be exactly zero everywhere
        for v in out["lambda_term_loss_ratio"].to_list():
            assert v == 0.0
        # volume term should be its usual +1.5 * volume
        for actual, vol in zip(
            out["lambda_term_volume"].to_list(),
            out["volume"].cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(1.5 * vol, rel=1e-6, abs=1e-7)


# ---------------------------------------------------------------------------
# 5. Ratio constraints — correct linearised_<name> and lambda_term_<name>
# ---------------------------------------------------------------------------


class TestRatioConstraints:
    """Spec test 5: ratio constraints get correct linearised + lambda_term."""

    def test_linearised_ratio_equals_num_minus_L_denom(self) -> None:
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max_pct": 0.95,
            }
        }
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.3},
            objective="income",
            constraints=constraints,
        )
        out = applier.with_explainer_columns(df)

        # Compute expected L = 0.95 * (Σ_baseline incurred / Σ_baseline premium)
        baseline = df.filter(pl.col("scenario_value") == 1.0)
        baseline_lr = float(baseline["incurred"].sum()) / float(
            baseline["premium"].sum()
        )
        L = 0.95 * baseline_lr

        # linearised = incurred - L * premium
        for actual, num, denom in zip(
            out["linearised_loss_ratio"].to_list(),
            out["incurred"].cast(pl.Float64).to_list(),
            out["premium"].cast(pl.Float64).to_list(),
        ):
            expected = num - L * denom
            assert actual == pytest.approx(expected, rel=1e-5, abs=1e-5)

    def test_score_invariant_ratio_constraint(self) -> None:
        """For ratio constraints the invariant holds against the
        linearised value, NOT the actual ratio."""
        df = make_ratio_solve_df(n_quotes=20, n_steps=5)
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max_pct": 0.95,
            }
        }
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.4},
            objective="income",
            constraints=constraints,
        )
        out = applier.with_explainer_columns(df)

        reconstructed = out["income"] + out["lambda_term_loss_ratio"]
        for actual, expected in zip(
            out["decision_score"].to_list(),
            reconstructed.cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(expected, rel=SCORE_RTOL, abs=SCORE_ABS)

    def test_ratio_label_not_leaked_as_extra_column(self) -> None:
        """The ratio label is used as an internal synthetic-column name
        by the linearisation; it must not appear in the output as a
        phantom column the caller didn't supply."""
        df = make_ratio_solve_df(n_quotes=10, n_steps=5)
        # 'loss_ratio' is NOT a column in df (raw cols: income, incurred,
        # premium). After linearisation the synthetic column is named
        # 'loss_ratio'; the explainer must drop it from the output.
        assert "loss_ratio" not in df.columns
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.2},
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max_pct": 0.95,
                }
            },
        )
        out = applier.with_explainer_columns(df)
        assert "loss_ratio" not in out.columns
        assert "linearised_loss_ratio" in out.columns


# ---------------------------------------------------------------------------
# 6. & 7. Baseline marking
# ---------------------------------------------------------------------------


class TestBaselineMarking:
    """Spec tests 6 & 7: exact baseline at scenario_value==1.0 (or
    nearest if missing)."""

    def test_exact_baseline_at_scenario_value_one(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        # make_small_df produces scenario_values [0.8, 0.9, 1.0, 1.1, 1.2]
        # — so 1.0 IS present at scenario_index=2.
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        baseline_rows = out.filter(pl.col("is_baseline"))
        for row in baseline_rows.iter_rows(named=True):
            assert row["scenario_value"] == pytest.approx(1.0)
            assert row["scenario_index"] == 2

    def test_nearest_baseline_when_exact_missing(self) -> None:
        # Build a grid where scenario_value=1.0 is absent: [0.8, 0.9, 1.1, 1.2].
        # 0.9 and 1.1 are exactly equidistant from 1.0 in IEEE 754 f32 (both
        # at distance 0x3DCCCCCE ≈ 0.100000024); 0.95 and 1.05 are NOT
        # equidistant in f32, which is why we don't use them. The spec's
        # tie-break (lowest scenario_index wins on |sv - 1.0| tie) is
        # being exercised here, not the nearest-but-no-tie case.
        n_quotes = 5
        rows = []
        mults = [0.8, 0.9, 1.1, 1.2]
        for q in range(n_quotes):
            for j, mult in enumerate(mults):
                rows.append(
                    {
                        "quote_id": f"Q{q}",
                        "scenario_index": j,
                        "scenario_value": mult,
                        "expected_income": 100.0 * mult,
                        "volume": 1.0 - 0.1 * j,
                    }
                )
        df = pl.DataFrame(
            rows,
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.5}},
        )
        out = applier.with_explainer_columns(df)
        baseline_rows = out.filter(pl.col("is_baseline"))

        # Tie at distance ~0.1 between scenario_index=1 (0.9) and
        # scenario_index=2 (1.1); spec says lowest scenario_index wins
        # -> scenario_index=1.
        for row in baseline_rows.iter_rows(named=True):
            assert row["scenario_index"] == 1, (
                f"quote {row['quote_id']}: expected baseline at "
                f"scenario_index=1 (scenario_value=0.9), got "
                f"scenario_index={row['scenario_index']}"
            )

    def test_nearest_baseline_no_tie(self) -> None:
        """Asymmetric distances around 1.0: 0.85 (dist 0.15) vs 1.05
        (dist 0.05). 1.05 wins as the unambiguous nearest."""
        n_quotes = 3
        rows = []
        mults = [0.7, 0.85, 1.05, 1.2]
        for q in range(n_quotes):
            for j, mult in enumerate(mults):
                rows.append(
                    {
                        "quote_id": f"Q{q}",
                        "scenario_index": j,
                        "scenario_value": mult,
                        "expected_income": 100.0 * mult,
                        "volume": 1.0 - 0.1 * j,
                    }
                )
        df = pl.DataFrame(
            rows,
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.5}},
        )
        out = applier.with_explainer_columns(df)
        baseline_rows = out.filter(pl.col("is_baseline"))
        for row in baseline_rows.iter_rows(named=True):
            assert row["scenario_index"] == 2, (
                f"quote {row['quote_id']}: expected baseline at "
                f"scenario_index=2 (scenario_value=1.05), got "
                f"scenario_index={row['scenario_index']}"
            )

    def test_exactly_one_baseline_per_quote(self) -> None:
        df = make_small_df(n_quotes=30, n_steps=5)
        applier = _sum_constraint_applier(df)
        out = applier.with_explainer_columns(df)
        per_quote = out.group_by("quote_id").agg(pl.col("is_baseline").sum())
        for row in per_quote.iter_rows(named=True):
            assert row["is_baseline"] == 1


# ---------------------------------------------------------------------------
# 8. Tie-breaking on selected — explicit tie construction
# ---------------------------------------------------------------------------


class TestSelectedTieBreaking:
    """Spec test 8: construct an exact decision_score tie for one quote;
    selected==True is on the lowest scenario_index. This matches apply's
    strict-greater argmax in argmax.rs."""

    def test_explicit_tie_lowest_scenario_index_wins(self) -> None:
        # One quote with two scenarios that produce identical Lagrangian:
        # objective + (-lambda) * loss_ratio == objective + (-lambda) * loss_ratio
        # at scenarios j=0 and j=1. We pick obj/lr values such that both
        # rows yield exactly 100.0 at lambda=2.0:
        #   row 0: obj=120, lr=10 -> 120 - 2*10 = 100
        #   row 1: obj= 80, lr=-10 -> 80 - 2*(-10) = 100   (wait — negative lr)
        # Easier: obj=120, lr=10 vs obj=140, lr=20: 100 vs 100.
        # Then add a clearly-worse third scenario so the winner isn't
        # by-default index 0 just because the others are awful.
        rows = [
            {
                "quote_id": "Q0",
                "scenario_index": 0,
                "scenario_value": 0.9,
                "obj": 120.0,
                "loss_ratio": 10.0,
            },
            {
                "quote_id": "Q0",
                "scenario_index": 1,
                "scenario_value": 1.0,
                "obj": 140.0,
                "loss_ratio": 20.0,
            },
            {
                "quote_id": "Q0",
                "scenario_index": 2,
                "scenario_value": 1.1,
                "obj": 50.0,
                "loss_ratio": 0.0,
            },
        ]
        df = pl.DataFrame(
            rows,
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "obj": pl.Float32,
                "loss_ratio": pl.Float32,
            },
        )
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 2.0},
            objective="obj",
            constraints={"loss_ratio": {"max": 100.0}},
        )
        out = applier.with_explainer_columns(df)

        # Confirm the tie really exists at f32 precision (the apply
        # argmax runs in f32; if the test grid happens to break the
        # tie at f32 precision the assertion below tells us nothing).
        scores = out["decision_score"].to_list()
        assert scores[0] == pytest.approx(scores[1], rel=1e-6, abs=1e-6), (
            f"test setup invalid: rows 0 and 1 should tie, got {scores}"
        )
        assert scores[2] < scores[0] - 1.0  # row 2 strictly worse

        selected = out.filter(pl.col("selected"))
        assert selected.height == 1
        assert selected["scenario_index"][0] == 0, (
            f"expected lowest scenario_index (0) to win on tie, got "
            f"{selected['scenario_index'][0]}"
        )

    def test_decision_score_uses_apply_f32_precision(self) -> None:
        """A lambda that differs from 1.0 only below f32 precision should
        produce the same f32 tie that apply sees. This prevents the
        explainer from reporting a higher decision_score on an unselected
        row because it accidentally scored in f64."""
        df = pl.DataFrame(
            [
                {
                    "quote_id": "Q0",
                    "scenario_index": 0,
                    "scenario_value": 1.0,
                    "obj": 0.0,
                    "volume": 0.0,
                },
                {
                    "quote_id": "Q0",
                    "scenario_index": 1,
                    "scenario_value": 1.1,
                    "obj": -1.0,
                    "volume": 1.0,
                },
            ],
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "obj": pl.Float32,
                "volume": pl.Float32,
            },
        )
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 1.00000005},
            objective="obj",
            constraints={"volume": {"min": 0.0}},
        )

        apply_result = applier.apply(df)
        out = applier.with_explainer_columns(df)

        assert apply_result.dataframe["optimal_step"][0] == 0
        assert out["lambda_term_volume"].to_list() == [0.0, 1.0]
        assert out["decision_score"].to_list() == [0.0, 0.0]
        selected = out.filter(pl.col("selected"))
        assert selected["scenario_index"][0] == 0


# ---------------------------------------------------------------------------
# 9. Custom column names
# ---------------------------------------------------------------------------


class TestCustomColumnNames:
    """Spec test 9: custom column names for quote_id, scenario_index,
    scenario_value, and objective work."""

    def test_custom_column_names_round_trip(self) -> None:
        df = make_small_df(n_quotes=10, n_steps=5).rename(
            {
                "quote_id": "policy_id",
                "scenario_index": "step_idx",
                "scenario_value": "multiplier",
                "expected_income": "ei",
            }
        )
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.5},
            objective="ei",
            constraints={"volume": {"min_pct": 0.9}},
            quote_id="policy_id",
            scenario_index="step_idx",
            scenario_value="multiplier",
        )
        out = applier.with_explainer_columns(df)
        # original columns preserved with their custom names
        for col in ("policy_id", "step_idx", "multiplier", "ei", "volume"):
            assert col in out.columns
        # explainer columns appended
        for col in (
            "decision_score",
            "selected",
            "is_baseline",
            "linearised_volume",
            "lambda_term_volume",
        ):
            assert col in out.columns
        # selected reconciles
        per_quote = out.group_by("policy_id").agg(pl.col("selected").sum())
        for row in per_quote.iter_rows(named=True):
            assert row["selected"] == 1

    def test_categorical_quote_id_round_trip(self) -> None:
        """Apply already accepts Categorical quote ids; the explainer's
        selected lookup must preserve that contract when joining the
        apply output back to the caller's frame."""
        df = make_small_df(n_quotes=10, n_steps=5).with_columns(
            pl.col("quote_id").cast(pl.Categorical)
        )
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.5},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
        )

        out = applier.with_explainer_columns(df)

        assert out.schema["quote_id"] == pl.Categorical
        per_quote = out.group_by("quote_id").agg(pl.col("selected").sum())
        for row in per_quote.iter_rows(named=True):
            assert row["selected"] == 1


# ---------------------------------------------------------------------------
# 10. apply() behaviour unchanged
# ---------------------------------------------------------------------------


class TestApplyUnchanged:
    """Spec test 10: ``ApplyOptimiser.apply(df)`` behaviour and output
    remain unchanged after introducing with_explainer_columns."""

    def test_apply_output_identical_before_and_after_explainer_call(self) -> None:
        df = make_small_df(n_quotes=20, n_steps=5)
        applier = _sum_constraint_applier(df)
        before = applier.apply(df).dataframe.clone()
        # Call the new method, then re-run apply.
        _ = applier.with_explainer_columns(df)
        after = applier.apply(df).dataframe
        assert before.equals(after)


# ---------------------------------------------------------------------------
# 11. No-constraints case
# ---------------------------------------------------------------------------


class TestNoConstraints:
    """Spec test 11: with constraints == {} and lambdas == {}, only
    decision_score, selected, is_baseline are appended; decision_score
    equals the objective column; no lambda_term_* or linearised_* cols."""

    def test_no_constraints_minimal_output(self) -> None:
        df = make_small_df(n_quotes=15, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={},
            objective="expected_income",
            constraints={},
        )
        out = applier.with_explainer_columns(df)

        assert "decision_score" in out.columns
        assert "selected" in out.columns
        assert "is_baseline" in out.columns

        for col in out.columns:
            assert not col.startswith("lambda_term_"), (
                f"unexpected lambda_term column: {col}"
            )
            assert not col.startswith("linearised_"), (
                f"unexpected linearised column: {col}"
            )

        # decision_score == objective (cast to f64)
        for actual, expected in zip(
            out["decision_score"].to_list(),
            out["expected_income"].cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(expected, rel=SCORE_RTOL, abs=SCORE_ABS)

    def test_no_constraints_selected_picks_max_objective(self) -> None:
        df = make_small_df(n_quotes=15, n_steps=5)
        applier = pc.ApplyOptimiser(
            lambdas={},
            objective="expected_income",
            constraints={},
        )
        out = applier.with_explainer_columns(df)
        # With no constraints, selected == argmax(objective) per quote.
        for qid in out["quote_id"].unique():
            q_df = out.filter(pl.col("quote_id") == qid)
            argmax_idx = int(q_df["expected_income"].arg_max())
            selected_idx = int(q_df.filter(pl.col("selected"))["scenario_index"][0])
            assert selected_idx == argmax_idx, (
                f"quote {qid}: selected {selected_idx} != argmax {argmax_idx}"
            )


# ---------------------------------------------------------------------------
# 12. Multiple ratio constraints
# ---------------------------------------------------------------------------


class TestMultipleRatioConstraints:
    """Spec test 12: two ratio constraints each get their own
    linearised_<name> with the correct L; decision_score reconciles."""

    def test_two_ratio_constraints_distinct_L_values(self) -> None:
        # Combine make_ratio_solve_df (income/incurred/premium) + the
        # retention dataset's columns to get two distinct ratio
        # constraints over the same quote universe.
        n_quotes, n_steps = 20, 5
        ratio_df = make_ratio_solve_df(n_quotes=n_quotes, n_steps=n_steps)
        retention_df = make_retention_df(n_quotes=n_quotes, n_steps=n_steps)
        # Both fixtures use the same quote_id/scenario_index/scenario_value
        # axes; merge on those keys to get one DataFrame with both ratio
        # numerator/denominator pairs.
        df = ratio_df.join(
            retention_df.select(["quote_id", "scenario_index", "kept", "exposed"]),
            on=["quote_id", "scenario_index"],
            how="inner",
        )

        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max_pct": 0.95,
            },
            "retention": {
                "numerator": "kept",
                "denominator": "exposed",
                "min_pct": 0.95,
            },
        }
        applier = pc.ApplyOptimiser(
            lambdas={"loss_ratio": 0.3, "retention": 0.7},
            objective="income",
            constraints=constraints,
        )
        out = applier.with_explainer_columns(df)

        # Both linearised columns present
        assert "linearised_loss_ratio" in out.columns
        assert "linearised_retention" in out.columns

        # Compute expected L for each from baseline rows
        baseline = df.filter(pl.col("scenario_value") == 1.0)
        lr_baseline = float(baseline["incurred"].sum()) / float(
            baseline["premium"].sum()
        )
        ret_baseline = float(baseline["kept"].sum()) / float(baseline["exposed"].sum())
        L_lr = 0.95 * lr_baseline
        L_ret = 0.95 * ret_baseline

        # linearised_loss_ratio == incurred - L_lr * premium
        for actual, num, denom in zip(
            out["linearised_loss_ratio"].to_list(),
            out["incurred"].cast(pl.Float64).to_list(),
            out["premium"].cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(num - L_lr * denom, rel=1e-5, abs=1e-5)

        # linearised_retention == kept - L_ret * exposed
        for actual, num, denom in zip(
            out["linearised_retention"].to_list(),
            out["kept"].cast(pl.Float64).to_list(),
            out["exposed"].cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(num - L_ret * denom, rel=1e-5, abs=1e-5)

        # Score invariant holds across both constraints:
        # loss_ratio is max -> -lambda; retention is min -> +lambda.
        reconstructed = (
            out["income"] + out["lambda_term_loss_ratio"] + out["lambda_term_retention"]
        )
        for actual, expected in zip(
            out["decision_score"].to_list(),
            reconstructed.cast(pl.Float64).to_list(),
        ):
            assert actual == pytest.approx(expected, rel=SCORE_RTOL, abs=SCORE_ABS)


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class TestErrorSemantics:
    """Validation rejects bad input the same way apply() does, plus an
    explicit collision check on the appended explainer columns."""

    def test_rejects_non_dataframe(self) -> None:
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
        )
        with pytest.raises(TypeError, match="Expected pl.DataFrame"):
            applier.with_explainer_columns([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_missing_objective_column(self) -> None:
        df = make_small_df(n_quotes=5, n_steps=5).drop("expected_income")
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            applier.with_explainer_columns(df)

    def test_rejects_missing_constraint_column(self) -> None:
        df = make_small_df(n_quotes=5, n_steps=5).drop("volume")
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            applier.with_explainer_columns(df)

    def test_rejects_collision_with_appended_columns(self) -> None:
        df = make_small_df(n_quotes=5, n_steps=5).with_columns(
            pl.lit(0.0).alias("decision_score")
        )
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
        )
        with pytest.raises(ValueError, match="decision_score"):
            applier.with_explainer_columns(df)

    def test_rejects_collision_on_per_constraint_column(self) -> None:
        df = make_small_df(n_quotes=5, n_steps=5).with_columns(
            pl.lit(0.0).alias("lambda_term_volume")
        )
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.9}},
        )
        with pytest.raises(ValueError, match="lambda_term_volume"):
            applier.with_explainer_columns(df)


# ---------------------------------------------------------------------------
# Helper: _spec_direction
# ---------------------------------------------------------------------------


class TestSpecDirectionHelper:
    """Direct coverage for the ``_spec_direction`` helper. Validators
    upstream guarantee one direction key per spec, so the defensive
    raise is unreachable through normal call sites — test it directly
    so the branch isn't a black box."""

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ({"min": 0.5}, "min"),
            ({"min_pct": 0.9}, "min"),
            ({"max": 100.0}, "max"),
            ({"max_pct": 1.05}, "max"),
            (
                {"numerator": "n", "denominator": "d", "min_pct": 0.9},
                "min",
            ),
            (
                {"numerator": "n", "denominator": "d", "max": 0.6},
                "max",
            ),
        ],
    )
    def test_direction_inferred_from_spec_keys(
        self, spec: dict[str, object], expected: str
    ) -> None:
        assert _spec_direction(spec) == expected

    def test_raises_when_no_direction_key(self) -> None:
        with pytest.raises(ValueError, match="no direction key"):
            _spec_direction({"numerator": "n", "denominator": "d"})
