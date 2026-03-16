"""Tests for the RatebookOptimiser (coordinate descent loop)."""

from __future__ import annotations


import price_contour as pc
from price_contour.ratebook import RatebookOptimiser, RatebookResult
from helpers import make_small_df, make_factors


class TestRatebook:
    def test_single_factor_cd(self):
        """Single factor CD = single grouped solve result."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=100,
        )
        result = opt.solve(df, factors)

        assert isinstance(result, RatebookResult)
        assert "region" in result.factor_tables
        assert len(result.factor_tables["region"]) == 4  # 4 regions
        assert result.total_objective > 0
        assert len(result.per_factor_results) == 1

    def test_two_factor_cd_converges(self):
        """Two-factor CD converges on synthetic problem."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=100,
        )
        result = opt.solve(df, factors)

        assert "region" in result.factor_tables
        assert "age_band" in result.factor_tables
        assert result.cd_iterations >= 1
        assert result.total_objective > 0

    def test_to_rating_entries_valid(self):
        """to_rating_entries() produces valid DataFrames."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)
        entries = result.to_rating_entries()

        assert "region" in entries
        assert "age_band" in entries
        region_df = entries["region"]
        assert "region" in region_df.columns
        assert "factor" in region_df.columns
        assert region_df.shape[0] == 4  # 4 unique regions

    def test_clamp_rate_below_threshold(self):
        """Clamp rate should be low with a sufficiently wide grid."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            factor_columns=[["region"]],
            candidate_min=0.80,
            candidate_max=1.20,
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)
        assert result.clamp_rate < 0.5, f"clamp rate too high: {result.clamp_rate}"

    def test_auto_discover_selects_factors(self):
        """Auto-discover selects factors from the factors DataFrame."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)

        # Should have discovered at least one factor
        assert len(result.factor_tables) >= 1

        # Discovered factors must be a subset of the available columns
        assert set(result.factor_tables.keys()) <= {"region", "age_band"}, (
            f"unexpected factors discovered: {set(result.factor_tables.keys())}"
        )

        # Each factor table should have the correct number of levels
        for name, table in result.factor_tables.items():
            expected_levels = factors[name].n_unique()
            assert len(table) == expected_levels, (
                f"factor {name!r} has {len(table)} levels, expected {expected_levels}"
            )

    def test_summary_structure(self):
        """summary() produces valid params/metrics/artifacts."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)
        s = opt.summary(result)

        assert set(s.keys()) == {"params", "metrics", "artifacts"}
        assert isinstance(s["params"]["n_factors"], int)
        assert isinstance(s["metrics"]["total_objective"], float)
        assert isinstance(s["artifacts"]["factor_tables"], dict)
        assert isinstance(s["artifacts"]["rating_entries"], dict)

    def test_cd_objective_nondecreasing(self):
        """CD iterations should produce non-decreasing objective values."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=100,
        )
        result = opt.solve(df, factors)

        # per_factor_results has one entry per (cd_iteration, factor).
        # With 2 factors and 3 CD iterations, there are up to 6 entries.
        # Group by CD sweep: every 2 entries is one CD iteration.
        n_factors = 2
        objectives_per_cd = []
        for i in range(0, len(result.per_factor_results), n_factors):
            sweep = result.per_factor_results[i : i + n_factors]
            if sweep:
                objectives_per_cd.append(sweep[-1].total_objective)

        # Each CD sweep should produce an objective no worse than the previous
        for i in range(len(objectives_per_cd) - 1):
            assert objectives_per_cd[i] <= objectives_per_cd[i + 1] + 1e-3, (
                f"CD objective decreased at iteration {i + 1}: "
                f"{objectives_per_cd[i]:.4f} > {objectives_per_cd[i + 1]:.4f}"
            )

    def test_factor_values_within_candidate_range(self):
        """All factor values should be within the candidate range."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            factor_columns=[["region"]],
            candidate_min=0.80,
            candidate_max=1.20,
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)

        for factor_name, table in result.factor_tables.items():
            for level, value in table.items():
                assert 0.80 <= value <= 1.20, (
                    f"factor {factor_name!r} level {level!r} has value {value}, "
                    f"outside candidate range [0.80, 1.20]"
                )

    def test_interaction_factor_solve(self):
        """Solving with an interaction factor should converge."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        solver = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region", "age_band"]],  # interaction
            max_cd_iterations=2,
            max_iter=100,
        )
        result = solver.solve(df, factors)

        assert isinstance(result, RatebookResult)
        # Interaction factor table should have a single key like "region:age_band"
        assert len(result.factor_tables) == 1
        key = list(result.factor_tables.keys())[0]
        # The interaction key is formed by joining the column names with ":"
        assert "region" in key and "age_band" in key
        # Should have entries (the interaction levels)
        assert len(result.factor_tables[key]) > 0
        assert result.total_objective > 0

    def test_ratebook_to_apply_roundtrip(self):
        """Ratebook lambdas can be used with ApplyOptimiser on the same data."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = make_factors(n)

        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"]],
            max_cd_iterations=1,
            max_iter=100,
        )
        result = opt.solve(df, factors)

        # Use the ratebook's lambdas with ApplyOptimiser
        applier = pc.ApplyOptimiser(
            lambdas=result.lambdas,
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        apply_result = applier.apply(df)

        # Both should produce positive objectives
        assert apply_result.total_objective > 0
        assert result.total_objective > 0

        # The apply result won't exactly match (ratebook uses factor-grouped
        # steps, apply uses per-quote steps), but both should be in the
        # same order of magnitude
        scale = abs(result.total_objective) or 1.0
        diff = abs(apply_result.total_objective - result.total_objective) / scale
        assert diff < 0.50, (
            f"apply objective {apply_result.total_objective:.2f} differs from "
            f"ratebook objective {result.total_objective:.2f} by {diff:.1%}"
        )
