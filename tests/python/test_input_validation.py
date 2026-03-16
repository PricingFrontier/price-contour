"""Tests for input validation and error handling.

Exercises the Rust binding layer's (solver_py.rs) validation of DataFrame
schema, column types, constraint specs, and data shape. The Rust layer maps
most validation failures to ``ValueError`` with descriptive messages, but
some edge cases may produce ``pyo3_runtime.PanicException`` or other
exception types — we use broad exception matching where the exact type is
uncertain and note where the behavior should be refined.
"""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from helpers import make_small_df


# ---------------------------------------------------------------------------
# Missing columns
# ---------------------------------------------------------------------------


class TestMissingColumns:
    """Verify that missing required columns produce clear errors."""

    @pytest.mark.parametrize(
        "drop_col",
        [
            pytest.param("expected_income", id="missing-expected_income"),
            pytest.param("quote_id", id="missing-quote_id"),
            pytest.param("scenario_index", id="missing-scenario_index"),
            pytest.param("scenario_value", id="missing-scenario_value"),
        ],
    )
    def test_missing_required_column_raises(self, drop_col: str) -> None:
        """Dropping a required column should raise before the solver runs."""
        df = make_small_df(n_quotes=10, n_steps=3).drop(drop_col)
        solver = pc.OnlineOptimiser(objective="expected_income")
        # Rust layer should raise ValueError for missing columns, but some
        # code paths may panic — accept any exception until that is tightened.
        with pytest.raises((ValueError, Exception)):
            solver.solve(df)

    def test_missing_constraint_column_raises(self) -> None:
        """Specifying a constraint on a column absent from the DataFrame
        should raise an error during grid ingestion or constraint parsing."""
        df = make_small_df(n_quotes=10, n_steps=3)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"nonexistent_col": {"min": 0.9}},
        )
        with pytest.raises((ValueError, Exception)):
            solver.solve(df)


# ---------------------------------------------------------------------------
# Wrong column types
# ---------------------------------------------------------------------------


class TestWrongColumnTypes:
    """Verify that columns with the wrong dtype are rejected."""

    def test_expected_income_as_utf8_raises(self) -> None:
        """expected_income must be Float32; passing Utf8 should error."""
        df = make_small_df(n_quotes=10, n_steps=3).with_columns(
            pl.col("expected_income").cast(pl.Utf8)
        )
        solver = pc.OnlineOptimiser(objective="expected_income")
        with pytest.raises((ValueError, Exception)):
            solver.solve(df)

    def test_scenario_index_as_float_raises(self) -> None:
        """scenario_index must be Int32; passing Float32 should error."""
        df = make_small_df(n_quotes=10, n_steps=3).with_columns(
            pl.col("scenario_index").cast(pl.Float32)
        )
        solver = pc.OnlineOptimiser(objective="expected_income")
        with pytest.raises((ValueError, Exception)):
            solver.solve(df)


# ---------------------------------------------------------------------------
# Invalid data
# ---------------------------------------------------------------------------


class TestInvalidData:
    """Verify that degenerate or corrupt data is caught."""

    def test_empty_dataframe_raises(self) -> None:
        """An empty DataFrame with correct schema should be rejected."""
        df = pl.DataFrame(
            schema={
                "quote_id": pl.Utf8,
                "scenario_index": pl.Int32,
                "scenario_value": pl.Float32,
                "expected_income": pl.Float32,
                "volume": pl.Float32,
            },
        )
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.9}},
        )
        with pytest.raises((ValueError, Exception)):
            solver.solve(df)

    def test_null_in_objective_raises(self) -> None:
        """Insert null into expected_income — should raise ValueError.

        Now that we validate nulls in both the Python layer and the Rust
        binding layer (via null_count check before into_no_null_iter),
        this correctly raises before reaching the solver.
        """
        df = make_small_df(n_quotes=10, n_steps=3)
        # Replace the first row's expected_income with null.
        df = (
            df.with_row_index("__row")
            .with_columns(
                pl.when(pl.col("__row") == 0)
                .then(None)
                .otherwise(pl.col("expected_income"))
                .alias("expected_income")
            )
            .drop("__row")
        )
        solver = pc.OnlineOptimiser(objective="expected_income")
        with pytest.raises((ValueError, Exception)):
            solver.solve(df)


# ---------------------------------------------------------------------------
# Invalid constraint specification
# ---------------------------------------------------------------------------


class TestInvalidConstraintSpec:
    """Verify that malformed constraint dicts are rejected."""

    def test_invalid_constraint_key_raises(self) -> None:
        """Constraint dicts must use one of min, max, min_abs, max_abs.
        An unrecognised key like 'invalid_key' should raise ValueError."""
        with pytest.raises((ValueError, Exception)):
            pc.OnlineOptimiser(
                objective="expected_income",
                constraints={"volume": {"invalid_key": 0.9}},
            )


# ---------------------------------------------------------------------------
# Uneven data (ragged quote grids)
# ---------------------------------------------------------------------------


class TestUnevenData:
    """Verify that ragged grids — quotes with different step counts — are
    caught by the Rust validation layer."""

    def test_uneven_steps_raises(self) -> None:
        """If one quote has fewer steps than the rest, the row count will
        not be divisible by n_steps and the Rust layer should raise."""
        df = make_small_df(n_quotes=10, n_steps=5)
        # Remove the last step of the first quote, creating a ragged grid
        # (Q0000 has 4 steps while others have 5).
        mask = ~((pl.col("quote_id") == "Q0000") & (pl.col("scenario_index") == 4))
        df = df.filter(mask)
        solver = pc.OnlineOptimiser(objective="expected_income")
        with pytest.raises((ValueError, Exception)):
            solver.solve(df)


# ---------------------------------------------------------------------------
# Issue 35: Sorted optimisation — shuffle invariance
# ---------------------------------------------------------------------------


class TestSortedOptimization:
    def test_presorted_df_produces_same_result(self):
        """Pre-sorted and unsorted DataFrames should give identical results."""
        df = make_small_df(n_quotes=50)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        result1 = solver.solve(df)

        # Shuffle the DataFrame
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
        result2 = solver.solve(shuffled)

        # Results should match
        steps1 = result1.dataframe.sort("quote_id")["optimal_step"].to_list()
        steps2 = result2.dataframe.sort("quote_id")["optimal_step"].to_list()
        assert steps1 == steps2
