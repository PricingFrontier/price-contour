"""Tests for serialisation (save/load) and summary helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

import price_contour as pc
from price_contour.frontier import frontier_summary
from price_contour.ratebook import RatebookOptimiser
from helpers import make_small_df


class TestApplySerialisation:
    def test_save_load_roundtrip(self):
        """ApplyOptimiser save → load produces same results."""
        df = make_small_df(n_quotes=50, n_steps=5)

        # Solve to get lambdas
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=200,
        )
        solve_result = solver.solve(df)

        # Create applier and apply
        applier = pc.ApplyOptimiser(
            lambdas=solve_result.lambdas,
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        result_before = applier.apply(df)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            applier.save(path)

            loaded = pc.ApplyOptimiser.load(path)

        result_after = loaded.apply(df)

        assert abs(result_before.total_objective - result_after.total_objective) < 1e-6
        assert result_before.lambdas == result_after.lambdas

    def test_save_load_preserves_config(self):
        """Save/load preserves all configuration fields."""
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.5, "loss_ratio": 0.3},
            objective="my_obj",
            constraints={"volume": {"min": 0.9}, "loss_ratio": {"max": 1.05}},
            quote_id="qid",
            scenario_index="step",
            scenario_value="mult",
            chunk_size=100_000,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            applier.save(path)
            loaded = pc.ApplyOptimiser.load(path)

        assert loaded.lambdas == {"volume": 0.5, "loss_ratio": 0.3}
        assert loaded.objective == "my_obj"
        assert loaded.constraints == {
            "volume": {"min": 0.9},
            "loss_ratio": {"max": 1.05},
        }
        assert loaded.quote_id == "qid"
        assert loaded.scenario_index == "step"
        assert loaded.scenario_value == "mult"
        assert loaded.chunk_size == 100_000


class TestFrontierSummary:
    def test_frontier_summary_structure(self):
        """frontier_summary() structure is valid."""
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            max_iter=100,
        )
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (0.85, 1.0)},
            n_points_per_dim=3,
        )
        summary = frontier_summary(result, selected_index=1)

        assert set(summary.keys()) == {"params", "metrics", "artifacts"}
        params = summary["params"]
        assert isinstance(params["frontier_n_points"], int)
        assert params["frontier_n_points"] == 3

        metrics = summary["metrics"]
        assert isinstance(metrics["selected_total_objective"], float)
        assert "selected_threshold_volume" in metrics
        assert "selected_lambda_volume" in metrics

        artifacts = summary["artifacts"]
        assert isinstance(artifacts["frontier"], pl.DataFrame)


class TestRatebookSummary:
    def test_ratebook_summary_structure(self):
        """Ratebook summary() structure is valid."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = pl.DataFrame(
            {
                "region": [["N", "S", "E", "W"][i % 4] for i in range(n)],
            }
        )
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
        assert isinstance(s["metrics"]["baseline_objective"], float)
        assert "factor_tables" in s["artifacts"]
        assert "rating_entries" in s["artifacts"]
        assert isinstance(s["artifacts"]["rating_entries"]["region"], pl.DataFrame)


class TestRatebookSave:
    def _solve_ratebook(self, n=50):
        df = make_small_df(n_quotes=n)
        factors = pl.DataFrame(
            {
                "region": [["N", "S", "E", "W"][i % 4] for i in range(n)],
                "age_band": [["young", "mid", "old"][i % 3] for i in range(n)],
            }
        )
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        return opt.solve(df, factors)

    def test_save_creates_config_and_factor_files(self):
        """save() creates config.json + one JSON per factor."""
        result = self._solve_ratebook()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "params"
            result.save(out)

            assert (out / "config.json").exists()
            assert (out / "region.json").exists()
            assert (out / "age_band.json").exists()

    def test_save_config_contents(self):
        """config.json has factor_order and lambdas, no metrics."""
        result = self._solve_ratebook()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "params"
            result.save(out)

            import json

            config = json.loads((out / "config.json").read_text())
            assert config["factor_order"] == ["region", "age_band"]
            assert isinstance(config["lambdas"], dict)
            # No metrics in config
            assert "total_objective" not in config
            assert "uplift_pct" not in config
            assert "clamp_rate" not in config

    def test_save_factor_json_contents(self):
        """Each factor JSON has columns list and table dict."""
        result = self._solve_ratebook()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "params"
            result.save(out)

            import json

            region = json.loads((out / "region.json").read_text())
            assert region["columns"] == ["region"]
            assert isinstance(region["table"], dict)
            assert set(region["table"].keys()) == {"E", "N", "S", "W"}
            for v in region["table"].values():
                assert isinstance(v, float)

    def test_save_factor_values_match_result(self):
        """Saved factor values match the in-memory result."""
        result = self._solve_ratebook()

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "params"
            result.save(out)

            import json

            for factor_name, table in result.factor_tables.items():
                filename = factor_name.replace(":", "_") + ".json"
                saved = json.loads((out / filename).read_text())
                assert saved["table"] == table

    def test_save_interaction_factor(self):
        """Interaction factors use compound key and multi-column columns list."""
        n = 50
        df = make_small_df(n_quotes=n)
        factors = pl.DataFrame(
            {
                "region": [["N", "S"][i % 2] for i in range(n)],
                "age": [["young", "old"][i % 2] for i in range(n)],
            }
        )
        opt = RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
            factor_columns=[["region", "age"]],
            max_cd_iterations=1,
            max_iter=50,
        )
        result = opt.solve(df, factors)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "params"
            result.save(out)

            import json

            interaction = json.loads((out / "region_age.json").read_text())
            assert interaction["columns"] == ["region", "age"]
            # Keys should be compound "val1:val2"
            for key in interaction["table"]:
                assert ":" in key


class TestApplySerialisationErrors:
    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            pc.ApplyOptimiser.load("/no/such/file.json")

    def test_load_corrupted_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        with pytest.raises(Exception):  # json.JSONDecodeError
            pc.ApplyOptimiser.load(bad)

    def test_load_missing_lambdas_key_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"objective": "x"}')
        with pytest.raises(KeyError):
            pc.ApplyOptimiser.load(bad)

    def test_load_empty_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{}")
        with pytest.raises(KeyError):
            pc.ApplyOptimiser.load(bad)

    def test_save_creates_parent_dirs(self, tmp_path):
        applier = pc.ApplyOptimiser(
            lambdas={"volume": 0.5},
            objective="expected_income",
        )
        deep = tmp_path / "deep" / "nested" / "config.json"
        applier.save(deep)
        assert deep.exists()

    def test_apply_mismatched_lambda_keys(self):
        """Lambdas for wrong constraint name should behave like zero lambdas."""
        df = make_small_df(n_quotes=20, n_steps=5)
        mismatched = pc.ApplyOptimiser(
            lambdas={"wrong_name": 1.0},
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        zero = pc.ApplyOptimiser(
            lambdas={"volume": 0.0},
            objective="expected_income",
            constraints={"volume": {"min": 0.90}},
        )
        # Both should behave as unconstrained since effective lambda=0
        result_mismatched = mismatched.apply(df)
        result_zero = zero.apply(df)
        assert abs(result_mismatched.total_objective - result_zero.total_objective) < 1e-3
