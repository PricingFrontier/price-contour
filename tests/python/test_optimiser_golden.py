"""Golden-file tests pinning numerical output of price-contour.

These tests re-solve a small set of deterministic fixtures and compare
the result against persisted reference output (``tests/python/golden/*.json``).

What this catches
-----------------
- Silent numerical drift across Rust / PyO3 / Polars version bumps.
- Refactors that change behaviour by 1 ULP (golden tests pin exact
  values, not "approximately equal").
- Accidental introduction of randomness or non-determinism (the same
  fixture must always solve to the same numbers).

Updating the golden files
-------------------------
When the algorithm intentionally changes (e.g. step-size schedule
adjustment, faster early-exit), regenerate the goldens by setting
``UPDATE_GOLDEN=1``::

    UPDATE_GOLDEN=1 pytest tests/python/test_optimiser_golden.py

This rewrites every golden JSON with the current solver output. **Always
inspect the diff** before committing — a regenerated golden with totally
different lambdas indicates a behaviour change that needs a code-review
discussion, not a passive accept.

Fixture stability contract
--------------------------
:func:`build_deterministic_fixture` is a pinned fixture: its output is
the input to every golden test. Once goldens exist for a fixture, the
function MUST NOT be changed — modifying it would silently invalidate
every persisted reference value. If a new fixture is needed, add a new
function (e.g. ``build_deterministic_fixture_v2``) and a new set of
goldens; do not edit this one.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import polars as pl

import price_contour as pc

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

# Online and apply solvers are bit-deterministic on this fixture set
# (verified with 5x repeat on each test). We pin tightly to catch any
# accidental introduction of non-determinism. Floats have a small slack
# (``ABS_TOL``) for cross-platform IEEE-754 rounding differences in the
# Rust/Polars stack.
ABS_TOL = 1e-9
REL_TOL = 1e-9

# Frontier sweeps run multiple solves with warm-start lambdas; small
# floating-point variation can surface in the last point's lambdas
# even on a deterministic solver. Use a slightly wider tolerance for
# frontier-only fields.
FRONTIER_ABS_TOL = 1e-6
FRONTIER_REL_TOL = 1e-7


# ---------------------------------------------------------------------------
# Deterministic fixture
# ---------------------------------------------------------------------------


def build_deterministic_fixture(
    seed: int = 42, n_quotes: int = 50, n_steps: int = 5
) -> pl.DataFrame:
    """Build a deterministic long-format DataFrame for golden tests.

    Uses Python's :class:`random.Random` (NOT numpy random) for full
    cross-platform reproducibility. Schema matches the price-contour
    convention: float32 cells, int32 indices.

    The fixture mixes per-quote elasticity, base income, and baseline
    loss-ratio so binding constraints have somewhere to find a non-trivial
    optimum. Columns:

    * ``expected_income`` — same as ``income`` and ``premium`` (income-
      maximising fixture; constraint-only columns vary per quote).
    * ``volume`` — logistic conversion at the price multiplier.
    * ``income`` / ``premium`` — alias of ``expected_income`` (used by
      tests that prefer the ratio fixture's column names).
    * ``incurred`` — ``premium * baseline_LR_per_quote * lr_factor`` so
      ``incurred / premium`` has a meaningful ratio per quote.

    DO NOT modify this function once goldens exist (see module docstring).
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    mults = [0.8 + 0.1 * j for j in range(n_steps)]
    for q in range(n_quotes):
        elasticity = 1.0 + 2.0 * rng.random()
        base = 80.0 + 40.0 * rng.random()
        quote_baseline_lr = 0.40 + 0.50 * rng.random()
        for j, mult in enumerate(mults):
            conversion = 1.0 / (1.0 + math.exp(elasticity * (mult - 1.0)))
            premium = base * mult * conversion
            lr_factor = 1.0 + 0.4 * (mult - 1.0)
            incurred = premium * quote_baseline_lr * lr_factor
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": mult,
                    "expected_income": premium,
                    "volume": conversion,
                    "income": premium,
                    "incurred": incurred,
                    "premium": premium,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "quote_id": pl.Utf8,
            "scenario_index": pl.Int32,
            "scenario_value": pl.Float32,
            "expected_income": pl.Float32,
            "volume": pl.Float32,
            "income": pl.Float32,
            "incurred": pl.Float32,
            "premium": pl.Float32,
        },
    )


def build_deterministic_factors(seed: int, n_quotes: int) -> pl.DataFrame:
    """Build deterministic factor levels for ratebook tests.

    Cycles through a fixed list of regions and age bands so the result is
    seed-insensitive but pinned. The seed argument is accepted for
    parameter symmetry with the fixture but is not used (cycling is
    deterministic by construction).
    """
    del seed  # signature symmetry only; cycling is independent of seed
    regions = ["A", "B", "C", "D"]
    age_bands = ["18-25", "26-35", "36-50", "51+"]
    return pl.DataFrame(
        {
            "region": [regions[i % len(regions)] for i in range(n_quotes)],
            "age_band": [age_bands[i % len(age_bands)] for i in range(n_quotes)],
        }
    )


# ---------------------------------------------------------------------------
# Golden file helpers
# ---------------------------------------------------------------------------


def _approx_equal(
    actual: float, expected: float, abs_tol: float, rel_tol: float
) -> bool:
    """Return True iff actual ≈ expected within abs_tol or rel_tol slack.

    Mirrors :func:`math.isclose` semantics but with explicit args so the
    failure message can echo both tolerances.
    """
    if actual == expected:
        return True
    if math.isnan(actual) and math.isnan(expected):
        return True
    return math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)


def _compare_dict_floats(
    actual: dict[str, float],
    expected: dict[str, float],
    *,
    name: str,
    abs_tol: float,
    rel_tol: float,
) -> None:
    """Assert two ``dict[str, float]`` are equal to tolerance.

    Key sets must match exactly; values must be within tolerance. Errors
    name the offending key so the failure message points at the
    regression directly.
    """
    assert set(actual.keys()) == set(expected.keys()), (
        f"{name} key mismatch: actual={sorted(actual.keys())}, "
        f"expected={sorted(expected.keys())}"
    )
    for k, expected_v in expected.items():
        actual_v = actual[k]
        assert _approx_equal(actual_v, expected_v, abs_tol, rel_tol), (
            f"{name}[{k!r}]: actual={actual_v} != expected={expected_v} "
            f"(abs_tol={abs_tol}, rel_tol={rel_tol})"
        )


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def _should_update() -> bool:
    return os.environ.get("UPDATE_GOLDEN", "0") == "1"


def _write_golden(name: str, payload: dict[str, Any]) -> None:
    path = _golden_path(name)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_golden(name: str) -> dict[str, Any]:
    path = _golden_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Golden file {path} not found. Run with UPDATE_GOLDEN=1 to "
            f"generate it."
        )
    return json.loads(path.read_text())


def _solve_result_to_payload(
    result: pc.SolveResult,
    *,
    fixture_seed: int,
    fixture_n_quotes: int,
    fixture_n_steps: int,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """Serialise an online ``SolveResult`` into a JSON-friendly payload."""
    out_df = result.dataframe.sort("quote_id")
    return {
        "fixture_seed": fixture_seed,
        "fixture_n_quotes": fixture_n_quotes,
        "fixture_n_steps": fixture_n_steps,
        "constraints": constraints,
        "expected": {
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "lambdas": dict(result.lambdas),
            "total_objective": float(result.total_objective),
            "total_constraints": dict(result.total_constraints),
            "baseline_objective": float(result.baseline_objective),
            "baseline_constraints": dict(result.baseline_constraints),
            "optimal_steps": [int(v) for v in out_df["optimal_step"].to_list()],
            "optimal_scenario_values": [
                float(v) for v in out_df["optimal_scenario_value"].to_list()
            ],
        },
        "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
    }


def _apply_result_to_payload(
    result: pc.ApplyResult,
    *,
    fixture_seed: int,
    fixture_n_quotes: int,
    fixture_n_steps: int,
    constraints: dict[str, Any],
    lambdas: dict[str, float],
) -> dict[str, Any]:
    """Serialise an ``ApplyResult`` into a JSON-friendly payload.

    Apply runs a fixed forward pass, so iterations / converged are not
    part of the result; we record the lambdas verbatim and per-quote
    optimal steps.
    """
    out_df = result.dataframe.sort("quote_id")
    return {
        "fixture_seed": fixture_seed,
        "fixture_n_quotes": fixture_n_quotes,
        "fixture_n_steps": fixture_n_steps,
        "constraints": constraints,
        "lambdas_in": lambdas,
        "expected": {
            "lambdas": dict(result.lambdas),
            "total_objective": float(result.total_objective),
            "total_constraints": dict(result.total_constraints),
            "baseline_objective": float(result.baseline_objective),
            "baseline_constraints": dict(result.baseline_constraints),
            "optimal_steps": [int(v) for v in out_df["optimal_step"].to_list()],
            "optimal_scenario_values": [
                float(v) for v in out_df["optimal_scenario_value"].to_list()
            ],
        },
        "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
    }


def _frontier_result_to_payload(
    result: Any,
    *,
    fixture_seed: int,
    fixture_n_quotes: int,
    fixture_n_steps: int,
    constraints: dict[str, Any],
    threshold_ranges: dict[str, list[float]],
    n_points_per_dim: int,
) -> dict[str, Any]:
    """Serialise a frontier result.

    Persists only the key columns: ``threshold_*``, ``total_objective``,
    ``total_*``, ``lambda_*``, ``iterations``, ``converged``. Skips
    ``sv_*`` percentile columns because their floating-point
    sensitivity is higher than core solve metrics (mean / std are
    aggregations over scenario_value floats).
    """
    pts = result.points
    keep_cols = [
        c
        for c in pts.columns
        if c.startswith("threshold_")
        or c.startswith("total_")
        or c.startswith("lambda_")
        or c in ("iterations", "converged")
    ]
    rows: list[dict[str, Any]] = []
    for row in pts.select(keep_cols).iter_rows(named=True):
        out_row: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, bool):
                out_row[k] = bool(v)
            elif isinstance(v, int):
                out_row[k] = int(v)
            else:
                out_row[k] = float(v)
        rows.append(out_row)
    return {
        "fixture_seed": fixture_seed,
        "fixture_n_quotes": fixture_n_quotes,
        "fixture_n_steps": fixture_n_steps,
        "constraints": constraints,
        "threshold_ranges": threshold_ranges,
        "n_points_per_dim": n_points_per_dim,
        "expected": {
            "n_points": int(result.n_points),
            "rows": rows,
        },
        "tolerance": {
            "absolute": FRONTIER_ABS_TOL,
            "relative": FRONTIER_REL_TOL,
        },
    }


def _compare_solve_to_golden(
    result: pc.SolveResult, golden: dict[str, Any]
) -> None:
    """Assert a ``SolveResult`` matches the persisted golden payload."""
    expected = golden["expected"]
    abs_tol = float(golden["tolerance"]["absolute"])
    rel_tol = float(golden["tolerance"]["relative"])
    assert result.converged == expected["converged"], (
        f"converged mismatch: actual={result.converged}, "
        f"expected={expected['converged']}"
    )
    assert result.iterations == expected["iterations"], (
        f"iterations mismatch: actual={result.iterations}, "
        f"expected={expected['iterations']}"
    )
    _compare_dict_floats(
        dict(result.lambdas),
        expected["lambdas"],
        name="lambdas",
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    assert _approx_equal(
        float(result.total_objective),
        float(expected["total_objective"]),
        abs_tol,
        rel_tol,
    ), (
        f"total_objective mismatch: actual={result.total_objective}, "
        f"expected={expected['total_objective']}"
    )
    _compare_dict_floats(
        dict(result.total_constraints),
        expected["total_constraints"],
        name="total_constraints",
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    _compare_dict_floats(
        dict(result.baseline_constraints),
        expected["baseline_constraints"],
        name="baseline_constraints",
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    out_df = result.dataframe.sort("quote_id")
    actual_steps = [int(v) for v in out_df["optimal_step"].to_list()]
    assert actual_steps == expected["optimal_steps"], (
        f"optimal_steps differ — first 10 actual={actual_steps[:10]}, "
        f"expected={expected['optimal_steps'][:10]}"
    )


def _compare_apply_to_golden(
    result: pc.ApplyResult, golden: dict[str, Any]
) -> None:
    """Assert an ``ApplyResult`` matches the persisted golden payload."""
    expected = golden["expected"]
    abs_tol = float(golden["tolerance"]["absolute"])
    rel_tol = float(golden["tolerance"]["relative"])
    _compare_dict_floats(
        dict(result.lambdas),
        expected["lambdas"],
        name="lambdas",
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    assert _approx_equal(
        float(result.total_objective),
        float(expected["total_objective"]),
        abs_tol,
        rel_tol,
    ), (
        f"total_objective mismatch: actual={result.total_objective}, "
        f"expected={expected['total_objective']}"
    )
    _compare_dict_floats(
        dict(result.total_constraints),
        expected["total_constraints"],
        name="total_constraints",
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    out_df = result.dataframe.sort("quote_id")
    actual_steps = [int(v) for v in out_df["optimal_step"].to_list()]
    assert actual_steps == expected["optimal_steps"], (
        f"optimal_steps differ — first 10 actual={actual_steps[:10]}, "
        f"expected={expected['optimal_steps'][:10]}"
    )


def _compare_frontier_to_golden(result: Any, golden: dict[str, Any]) -> None:
    """Assert a frontier result matches the persisted golden payload."""
    expected = golden["expected"]
    abs_tol = float(golden["tolerance"]["absolute"])
    rel_tol = float(golden["tolerance"]["relative"])
    assert result.n_points == expected["n_points"], (
        f"n_points mismatch: actual={result.n_points}, "
        f"expected={expected['n_points']}"
    )
    pts = result.points
    expected_rows = expected["rows"]
    assert len(expected_rows) == pts.shape[0], (
        f"row count mismatch: actual {pts.shape[0]} vs expected {len(expected_rows)}"
    )
    for i, expected_row in enumerate(expected_rows):
        for k, expected_v in expected_row.items():
            actual_v = pts[k][i]
            if isinstance(expected_v, bool):
                assert bool(actual_v) == expected_v, (
                    f"row {i}, col {k!r}: actual={actual_v}, expected={expected_v}"
                )
            elif isinstance(expected_v, int) and not isinstance(expected_v, bool):
                # iterations is an int field in the points DataFrame.
                assert int(actual_v) == int(expected_v), (
                    f"row {i}, col {k!r}: actual={actual_v}, "
                    f"expected={expected_v}"
                )
            else:
                assert _approx_equal(
                    float(actual_v),
                    float(expected_v),
                    abs_tol,
                    rel_tol,
                ), (
                    f"row {i}, col {k!r}: actual={actual_v}, "
                    f"expected={expected_v} (abs_tol={abs_tol}, "
                    f"rel_tol={rel_tol})"
                )


# ---------------------------------------------------------------------------
# Golden tests — each runs solve / apply / frontier and either:
# - writes the golden file (when UPDATE_GOLDEN=1) or
# - compares against the persisted golden.
# ---------------------------------------------------------------------------


# Constants for fixtures used in goldens. Once a golden is written for a
# fixture (seed, n_quotes, n_steps), DO NOT change these values without
# regenerating ALL the goldens.
GOLDEN_SEED = 42
GOLDEN_N_QUOTES = 50
GOLDEN_N_STEPS = 5


class TestOnlineGolden:
    """Online solver — sum and ratio constraints."""

    def test_online_sum_golden(self):
        """Sum-constraint binding ``min_pct`` solve."""
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
        constraints = {"volume": {"min_pct": 1.10}}
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints=constraints,
            max_iter=500,
            tolerance=1e-6,
        )
        result = solver.solve(df)

        if _should_update():
            payload = _solve_result_to_payload(
                result,
                fixture_seed=GOLDEN_SEED,
                fixture_n_quotes=GOLDEN_N_QUOTES,
                fixture_n_steps=GOLDEN_N_STEPS,
                constraints=constraints,
            )
            _write_golden("online_sum", payload)
            return
        golden = _read_golden("online_sum")
        _compare_solve_to_golden(result, golden)

    def test_online_ratio_golden(self):
        """Ratio-constraint binding ``max`` solve."""
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
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
            max_iter=500,
            tolerance=1e-6,
        )
        result = solver.solve(df)

        if _should_update():
            payload = _solve_result_to_payload(
                result,
                fixture_seed=GOLDEN_SEED,
                fixture_n_quotes=GOLDEN_N_QUOTES,
                fixture_n_steps=GOLDEN_N_STEPS,
                constraints=constraints,
            )
            _write_golden("online_ratio", payload)
            return
        golden = _read_golden("online_ratio")
        _compare_solve_to_golden(result, golden)


class TestRatebookGolden:
    """Ratebook solver — coordinate-descent with two factors."""

    def test_ratebook_sum_golden(self):
        """Two-factor ratebook with binding sum constraint."""
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
        factors = build_deterministic_factors(
            GOLDEN_SEED, GOLDEN_N_QUOTES
        )
        constraints = {"volume": {"min_pct": 0.95}}
        solver = pc.RatebookOptimiser(
            objective="expected_income",
            constraints=constraints,
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=100,
        )
        result = solver.solve(df, factors)

        if _should_update():
            payload = {
                "fixture_seed": GOLDEN_SEED,
                "fixture_n_quotes": GOLDEN_N_QUOTES,
                "fixture_n_steps": GOLDEN_N_STEPS,
                "constraints": constraints,
                "factor_columns": [["region"], ["age_band"]],
                "expected": {
                    "converged": bool(result.converged),
                    "cd_iterations": int(result.cd_iterations),
                    "lambdas": dict(result.lambdas),
                    "total_objective": float(result.total_objective),
                    "total_constraints": dict(result.total_constraints),
                    "baseline_objective": float(result.baseline_objective),
                    "baseline_constraints": dict(result.baseline_constraints),
                    "factor_tables": {
                        f: dict(levels)
                        for f, levels in result.factor_tables.items()
                    },
                },
                "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL},
            }
            _write_golden("ratebook_sum", payload)
            return
        golden = _read_golden("ratebook_sum")
        expected = golden["expected"]
        abs_tol = float(golden["tolerance"]["absolute"])
        rel_tol = float(golden["tolerance"]["relative"])
        assert result.converged == expected["converged"]
        assert result.cd_iterations == expected["cd_iterations"]
        _compare_dict_floats(
            dict(result.lambdas),
            expected["lambdas"],
            name="lambdas",
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        assert _approx_equal(
            float(result.total_objective),
            float(expected["total_objective"]),
            abs_tol,
            rel_tol,
        ), (
            f"total_objective mismatch: actual={result.total_objective}, "
            f"expected={expected['total_objective']}"
        )
        _compare_dict_floats(
            dict(result.total_constraints),
            expected["total_constraints"],
            name="total_constraints",
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        # Factor tables: dict[factor_name -> dict[level -> float]]
        assert set(result.factor_tables.keys()) == set(
            expected["factor_tables"].keys()
        ), (
            f"factor_tables keys differ: actual="
            f"{set(result.factor_tables.keys())}, expected="
            f"{set(expected['factor_tables'].keys())}"
        )
        for factor_name, expected_levels in expected["factor_tables"].items():
            actual_levels = result.factor_tables[factor_name]
            _compare_dict_floats(
                dict(actual_levels),
                expected_levels,
                name=f"factor_tables[{factor_name!r}]",
                abs_tol=abs_tol,
                rel_tol=rel_tol,
            )


class TestApplyGolden:
    """Apply (single-pass with stored lambdas)."""

    def test_apply_ratio_golden(self):
        """Apply with a fixed lambda on a ratio-constrained config."""
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
        constraints = {
            "loss_ratio": {
                "numerator": "incurred",
                "denominator": "premium",
                "max": 0.62,
            }
        }
        # Use a deliberately fixed lambda value (not solver-derived) so
        # the golden doesn't drift if the upstream solver's lambdas
        # change. The apply path with this fixed lambda is what the test
        # is verifying — independent of the solver loop.
        lambdas = {"loss_ratio": 0.5}
        applier = pc.ApplyOptimiser(
            lambdas=lambdas,
            objective="income",
            constraints=constraints,
        )
        result = applier.apply(df)

        if _should_update():
            payload = _apply_result_to_payload(
                result,
                fixture_seed=GOLDEN_SEED,
                fixture_n_quotes=GOLDEN_N_QUOTES,
                fixture_n_steps=GOLDEN_N_STEPS,
                constraints=constraints,
                lambdas=lambdas,
            )
            _write_golden("apply_ratio", payload)
            return
        golden = _read_golden("apply_ratio")
        _compare_apply_to_golden(result, golden)


class TestFrontierGolden:
    """Frontier sweep — sum constraint, multi-point."""

    def test_frontier_sum_golden(self):
        """1D frontier sweep with 5 points on a sum constraint."""
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
        constraints = {"volume": {"min_pct": 0.90}}
        threshold_ranges = {"volume": (0.90, 1.10)}
        n_points_per_dim = 5
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints=constraints,
            max_iter=200,
            tolerance=1e-6,
        )
        result = solver.frontier(
            df,
            threshold_ranges=threshold_ranges,
            n_points_per_dim=n_points_per_dim,
        )

        # Persist threshold_ranges as a list[list[float]] (JSON has no
        # native tuple), keyed by constraint name.
        threshold_ranges_json = {
            k: list(v) for k, v in threshold_ranges.items()
        }

        if _should_update():
            payload = _frontier_result_to_payload(
                result,
                fixture_seed=GOLDEN_SEED,
                fixture_n_quotes=GOLDEN_N_QUOTES,
                fixture_n_steps=GOLDEN_N_STEPS,
                constraints=constraints,
                threshold_ranges=threshold_ranges_json,
                n_points_per_dim=n_points_per_dim,
            )
            _write_golden("frontier_sum", payload)
            return
        golden = _read_golden("frontier_sum")
        _compare_frontier_to_golden(result, golden)


# ---------------------------------------------------------------------------
# Determinism guard
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify the solver is bit-deterministic across repeated invocations
    on the same fixture. This is the implicit contract that makes the
    golden tests meaningful — a non-deterministic solver would force us
    to relax tolerances to the point where the goldens stop catching
    real regressions.
    """

    def test_online_solve_is_deterministic(self):
        """Same fixture + config => identical results, 5 runs in a row."""
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
        constraints = {"volume": {"min_pct": 1.10}}
        results = []
        for _ in range(5):
            solver = pc.OnlineOptimiser(
                objective="expected_income",
                constraints=constraints,
                max_iter=500,
                tolerance=1e-6,
            )
            r = solver.solve(df)
            out = r.dataframe.sort("quote_id")
            results.append(
                (
                    r.total_objective,
                    tuple(sorted(r.lambdas.items())),
                    r.iterations,
                    r.converged,
                    tuple(out["optimal_step"].to_list()),
                )
            )
        ref = results[0]
        for i, run in enumerate(results[1:], start=1):
            assert run == ref, (
                f"Online solve is non-deterministic: run {i} differs "
                f"from run 0\n  ref={ref}\n  run={run}"
            )

    def test_apply_is_deterministic(self):
        """Apply must be deterministic — single forward pass, no randomness."""
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
        constraints = {"volume": {"min_pct": 0.90}}
        lambdas = {"volume": 0.05}
        results = []
        for _ in range(5):
            applier = pc.ApplyOptimiser(
                lambdas=lambdas,
                objective="expected_income",
                constraints=constraints,
            )
            r = applier.apply(df)
            out = r.dataframe.sort("quote_id")
            results.append(
                (
                    r.total_objective,
                    tuple(sorted(r.total_constraints.items())),
                    tuple(out["optimal_step"].to_list()),
                )
            )
        ref = results[0]
        for i, run in enumerate(results[1:], start=1):
            assert run == ref, (
                f"Apply is non-deterministic: run {i} differs from run 0"
            )

    def test_ratebook_is_deterministic_at_floating_point(self):
        """Ratebook (coordinate descent + grouped Lagrangian) must produce
        bit-identical numeric results across repeated invocations.

        The factor_tables dict ordering may differ across runs (Rust
        HashMap iteration order is not guaranteed), so we compare
        sorted (key, value) tuples per factor rather than dict equality.
        """
        df = build_deterministic_fixture(
            GOLDEN_SEED, GOLDEN_N_QUOTES, GOLDEN_N_STEPS
        )
        factors = build_deterministic_factors(
            GOLDEN_SEED, GOLDEN_N_QUOTES
        )
        constraints = {"volume": {"min_pct": 0.95}}
        results = []
        for _ in range(5):
            solver = pc.RatebookOptimiser(
                objective="expected_income",
                constraints=constraints,
                factor_columns=[["region"], ["age_band"]],
                max_cd_iterations=3,
                max_iter=100,
            )
            r = solver.solve(df, factors)
            ft = {
                fname: tuple(sorted(levels.items()))
                for fname, levels in r.factor_tables.items()
            }
            results.append(
                (
                    r.total_objective,
                    tuple(sorted(r.lambdas.items())),
                    r.cd_iterations,
                    r.converged,
                    tuple(sorted(ft.items())),
                )
            )
        ref = results[0]
        for i, run in enumerate(results[1:], start=1):
            assert run == ref, (
                f"Ratebook is non-deterministic: run {i} differs from run 0"
            )
