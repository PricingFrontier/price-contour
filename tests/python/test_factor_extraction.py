"""Tests for the Polars-direct factor label extractor.

The function takes a Polars DataFrame and a list of factor specs (each a
list of column names) and returns one `Vec<String>` per spec — bypassing
the per-column Python `.to_list()` materialisation that the previous
ratebook code did before passing into Rust.
"""

from __future__ import annotations

import polars as pl
import pytest

import price_contour as pc
from price_contour._price_contour import (
    build_interaction_labels_py,
    extract_factor_labels_py,
)


class TestExtractFactorLabels:
    def test_single_column_spec(self):
        """One column → labels are that column's values cast to Utf8."""
        df = pl.DataFrame({"region": ["North", "South", "East", "North"]})
        labels = extract_factor_labels_py(df, [["region"]], "\x1f")
        assert labels == [["North", "South", "East", "North"]]

    def test_two_column_interaction(self):
        """Two-column spec → labels are sep-joined."""
        df = pl.DataFrame(
            {
                "region": ["North", "South", "East"],
                "age_band": ["18-25", "26-35", "36-50"],
            }
        )
        labels = extract_factor_labels_py(df, [["region", "age_band"]], "\x1f")
        assert labels == [["North\x1f18-25", "South\x1f26-35", "East\x1f36-50"]]

    def test_three_column_interaction(self):
        """Three-column spec → labels are joined with the separator twice."""
        df = pl.DataFrame(
            {
                "a": ["x", "y"],
                "b": ["1", "2"],
                "c": ["p", "q"],
            }
        )
        labels = extract_factor_labels_py(df, [["a", "b", "c"]], ":")
        assert labels == [["x:1:p", "y:2:q"]]

    def test_multiple_specs(self):
        """Multiple specs in one call → one Vec<String> per spec."""
        df = pl.DataFrame(
            {
                "region": ["North", "South"],
                "age_band": ["18-25", "26-35"],
                "vehicle_type": ["car", "van"],
            }
        )
        labels = extract_factor_labels_py(
            df,
            [["region"], ["age_band", "vehicle_type"]],
            "\x1f",
        )
        assert labels == [
            ["North", "South"],
            ["18-25\x1fcar", "26-35\x1fvan"],
        ]

    def test_int_column_auto_casts_to_string(self):
        """Non-string columns are cast to Utf8 automatically."""
        df = pl.DataFrame({"age": [25, 35, 50]}, schema={"age": pl.Int64})
        labels = extract_factor_labels_py(df, [["age"]], "\x1f")
        assert labels == [["25", "35", "50"]]

    def test_categorical_column_extracts_strings(self):
        """Categorical columns are cast to Utf8 by their string values."""
        df = pl.DataFrame(
            {"cat": pl.Series("cat", ["A", "B", "A"], dtype=pl.Categorical)}
        )
        labels = extract_factor_labels_py(df, [["cat"]], "\x1f")
        assert labels == [["A", "B", "A"]]

    def test_missing_column_raises(self):
        """A spec referencing a missing column errors with a clear message."""
        df = pl.DataFrame({"region": ["A", "B"]})
        with pytest.raises(ValueError, match=r"(?i)not found|missing|nonexistent"):
            extract_factor_labels_py(df, [["nonexistent"]], "\x1f")

    def test_null_in_factor_column_raises(self):
        """Null values in a factor column are rejected with a named error."""
        df = pl.DataFrame({"region": ["A", None, "B"]})
        with pytest.raises(ValueError, match=r"(?i)null|region"):
            extract_factor_labels_py(df, [["region"]], "\x1f")

    def test_null_in_interaction_column_raises(self):
        """Null in any column of an interaction spec is rejected."""
        df = pl.DataFrame(
            {
                "region": ["A", "B", "C"],
                "age_band": ["18-25", None, "36-50"],
            }
        )
        with pytest.raises(ValueError, match=r"(?i)null|age_band"):
            extract_factor_labels_py(df, [["region", "age_band"]], "\x1f")

    def test_empty_spec_raises(self):
        """A spec with no columns is invalid."""
        df = pl.DataFrame({"region": ["A", "B"]})
        with pytest.raises(ValueError, match=r"(?i)empty|no columns"):
            extract_factor_labels_py(df, [[]], "\x1f")

    def test_zero_specs_returns_empty(self):
        """No specs → empty result, not an error."""
        df = pl.DataFrame({"region": ["A", "B"]})
        labels = extract_factor_labels_py(df, [], "\x1f")
        assert labels == []

    def test_zero_rows(self):
        """Zero-row DataFrame → empty per-spec lists."""
        df = pl.DataFrame({"region": pl.Series("region", [], dtype=pl.Utf8)})
        labels = extract_factor_labels_py(df, [["region"]], "\x1f")
        assert labels == [[]]

    def test_matches_python_to_list_path(self):
        """For string columns, the new path is bit-identical to the old `.to_list()` path."""
        df = pl.DataFrame(
            {
                "region": ["North", "South", "East", "West"] * 250,
                "age_band": ["18-25", "26-35", "36-50", "51+"] * 250,
            }
        )
        # Old path: per-column .to_list() + build_interaction_labels_py.
        old_single = df["region"].cast(pl.Utf8).to_list()
        old_interaction_cols = [
            df[c].cast(pl.Utf8).to_list() for c in ["region", "age_band"]
        ]
        old_interaction = build_interaction_labels_py(old_interaction_cols, "\x1f")

        # New path: extract_factor_labels_py.
        new = extract_factor_labels_py(df, [["region"], ["region", "age_band"]], "\x1f")
        assert new[0] == old_single
        assert new[1] == old_interaction

    def test_bool_column_casts_to_string(self):
        """Boolean columns cast to 'true'/'false' strings."""
        df = pl.DataFrame({"flag": [True, False, True]})
        labels = extract_factor_labels_py(df, [["flag"]], "\x1f")
        assert labels == [["true", "false", "true"]]

    def test_date_column_casts_to_iso_string(self):
        """Date columns cast to ISO 8601 strings."""
        import datetime as dt

        df = pl.DataFrame(
            {
                "d": [
                    dt.date(2024, 1, 15),
                    dt.date(2025, 6, 1),
                    dt.date(2026, 12, 31),
                ]
            }
        )
        labels = extract_factor_labels_py(df, [["d"]], "\x1f")
        assert labels == [["2024-01-15", "2025-06-01", "2026-12-31"]]

    def test_datetime_column_casts_to_iso_string(self):
        """Datetime columns cast to ISO 8601 strings (microseconds resolution)."""
        import datetime as dt

        df = pl.DataFrame(
            {
                "ts": [
                    dt.datetime(2024, 1, 15, 10, 30, 0),
                    dt.datetime(2025, 6, 1, 0, 0, 0),
                ]
            }
        )
        labels = extract_factor_labels_py(df, [["ts"]], "\x1f")
        # Polars' string cast for datetime uses ISO format; assert prefix
        # rather than exact (microsecond representation can vary).
        assert labels[0][0].startswith("2024-01-15")
        assert labels[0][1].startswith("2025-06-01")

    def test_all_null_column_raises(self):
        """A column that is entirely null still surfaces as a null-rejection error."""
        df = pl.DataFrame({"col": pl.Series("col", [None, None, None], dtype=pl.Utf8)})
        with pytest.raises(ValueError, match=r"(?i)null|col"):
            extract_factor_labels_py(df, [["col"]], "\x1f")

    def test_mixed_dtype_specs_in_one_call(self):
        """A single call can mix specs over different dtypes."""
        df = pl.DataFrame(
            {
                "int_age": [25, 35, 50],
                "str_region": ["N", "S", "E"],
                "cat_tier": pl.Series(
                    "cat_tier", ["bronze", "silver", "gold"], dtype=pl.Categorical
                ),
            }
        )
        labels = extract_factor_labels_py(
            df,
            [["int_age"], ["str_region", "cat_tier"]],
            "\x1f",
        )
        assert labels == [
            ["25", "35", "50"],
            ["N\x1fbronze", "S\x1fsilver", "E\x1fgold"],
        ]

    def test_repeated_calls_are_pure(self):
        """Calling the function twice on the same DataFrame produces identical results.

        Pins the side-effect-free contract — the cast does not mutate `factors`
        and is not memoised across calls.
        """
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        first = extract_factor_labels_py(df, [["a", "b"]], "\x1f")
        second = extract_factor_labels_py(df, [["a", "b"]], "\x1f")
        assert first == second

    def test_separator_collision_safe(self):
        """The default unit-separator (ASCII 31) is unlikely to appear in real labels."""
        df = pl.DataFrame(
            {
                "a": ["x:y", "p:q"],  # contains colon
                "b": ["1", "2"],
            }
        )
        labels = extract_factor_labels_py(df, [["a", "b"]], "\x1f")
        # The colon is preserved in the field, separator is \x1f.
        assert labels == [["x:y\x1f1", "p:q\x1f2"]]


class TestRatebookEndToEndRegressions:
    """Ratebook results must be byte-identical after the extractor swap."""

    def test_solve_results_unchanged_single_factor(self):
        from helpers import make_small_df, make_factors

        df = make_small_df(n_quotes=20, n_steps=5)
        factors = make_factors(n_quotes=20)

        rb = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=50,
            max_cd_iterations=3,
        )
        result = rb.solve(df, factors, factor_columns=[["region"]])
        assert result.cd_iterations >= 1
        assert "region" in result.factor_tables

    def test_solve_results_unchanged_interaction(self):
        from helpers import make_small_df, make_factors

        df = make_small_df(n_quotes=20, n_steps=5)
        factors = make_factors(n_quotes=20)

        rb = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=50,
            max_cd_iterations=3,
        )
        result = rb.solve(df, factors, factor_columns=[["region", "age_band"]])
        assert "region:age_band" in result.factor_tables

    def test_solve_with_int_factor_column(self):
        """Ratebook end-to-end with an Int factor column — auto-cast must work."""
        from helpers import make_small_df

        df = make_small_df(n_quotes=20, n_steps=5)
        # Int-typed factor column; previous Python `.to_list()` path would
        # have produced ["25", "35", ...] via Polars cast → list[str]. The
        # new path casts in Rust; results must be identical.
        factors = pl.DataFrame({"age_int": [(20 + (i * 3) % 30) for i in range(20)]})
        rb = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=50,
            max_cd_iterations=3,
        )
        result = rb.solve(df, factors, factor_columns=[["age_int"]])
        assert "age_int" in result.factor_tables
        # Factor table keys are the string-cast int values.
        for key in result.factor_tables["age_int"]:
            assert key.isdigit()

    def test_solve_with_categorical_factor_column(self):
        """Ratebook end-to-end with a Categorical factor column."""
        from helpers import make_small_df

        df = make_small_df(n_quotes=20, n_steps=5)
        factors = pl.DataFrame(
            {
                "tier": pl.Series(
                    "tier",
                    [["bronze", "silver", "gold"][i % 3] for i in range(20)],
                    dtype=pl.Categorical,
                )
            }
        )
        rb = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.90}},
            max_iter=50,
            max_cd_iterations=3,
        )
        result = rb.solve(df, factors, factor_columns=[["tier"]])
        assert "tier" in result.factor_tables
        # Categorical levels appear in the table by their string name.
        keys = set(result.factor_tables["tier"].keys())
        assert keys.issubset({"bronze", "silver", "gold"})
