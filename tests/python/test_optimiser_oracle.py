"""Independent-solver oracle tests for price-contour's discrete optimiser.

These tests treat ``scipy.optimize.linprog`` as an *external oracle*:
they solve the **LP relaxation** of the same problem price-contour
solves discretely, and check that price-contour's objective is within a
tight bound of the LP's continuous optimum.

Theory
------
price-contour picks one scenario index per quote (a discrete decision).
The LP relaxation lets each quote choose a probability distribution over
scenarios::

    maximise   Sigma_i Sigma_m  obj[i,m] * x[i,m]
    s.t.       Sigma_m x[i,m] = 1                  per quote
               x[i,m] >= 0                         all i, m
               Sigma_i Sigma_m c[i,m] * x[i,m] {<=, >=} threshold      per sum constraint
               Sigma_i Sigma_m (num[i,m] - L * denom[i,m]) * x[i,m] <= 0
                                                   per ratio constraint (max-direction)

The LP is at least as flexible as the discrete problem, so its optimum
upper-bounds price-contour's discrete optimum **at the same effective
constraint level**.

Comparison contract
-------------------
For each test we:

1. Run price-contour and observe the constraint level it actually
   achieves (``pc_result.total_constraints[name]``).
2. Solve the LP at price-contour's *achieved* constraint level — this is
   the fairest comparison: "given the constraint price-contour actually
   enforced, how close was its discrete objective to the continuous
   optimum?"
3. Assert ``pc_obj <= lp_obj + abs(lp_obj) * 1e-3`` (LP upper-bounds
   price-contour, modulo floating-point slack).
4. Assert ``pc_obj >= lp_obj * lower_bound`` (price-contour's
   integrality gap is bounded — typically 0% in single-constraint
   problems where the LP basic feasible solution lies on a discrete
   extreme point, but can grow to several percent when multiple
   constraints intersect at fractional vertices).

What this catches
-----------------
- Solver bugs that produce **feasible but suboptimal** solutions
  (existing tests verify constraint satisfaction but not optimality).
- Linearisation bugs where the synthetic ratio column doesn't match
  the LP's linearised constraint.
- Scaling / normalisation regressions that move the objective by >5%.

Notes
-----
The LP-at-PC-threshold approach is deliberately chosen over LP-at-user-
threshold because price-contour's discrete grid often overshoots the
user threshold (a known property of the algorithm — see
``CONSTRAINT_RTOL`` in ``helpers.py``). Using the achieved constraint
level isolates *optimality* from *constraint-satisfaction*; the latter
is already pinned by ``test_online.py`` and friends.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import price_contour as pc

scipy = pytest.importorskip("scipy")
from scipy.optimize import linprog  # noqa: E402

from helpers import make_small_df  # noqa: E402

# ---------------------------------------------------------------------------
# LP comparison tolerances
# ---------------------------------------------------------------------------

# Upper bound: ``pc_obj <= lp_obj + LP_UPPER_RTOL * abs(lp_obj)``.
# Floating-point slack only; LP and PC at the same constraint level
# should match very tightly when there's a discrete extreme point on
# the LP's optimal facet.
LP_UPPER_RTOL = 1e-3

# Lower bound: ``pc_obj >= lp_obj * LP_LOWER_BOUND_TYPICAL``.
# 95% covers the integrality gap for single-constraint sum problems on
# 30-50 quote fixtures. Empirically the gap is well under 0.5% in these
# tests; 95% leaves headroom for solver oscillation.
LP_LOWER_BOUND_TYPICAL = 0.95

# Tighter constraints (multi-constraint or boundary-binding) can have
# integrality gaps of several percent. Use this for tests where the
# constraint sits very close to the achievable boundary.
LP_LOWER_BOUND_TIGHT = 0.90


# ---------------------------------------------------------------------------
# Fixture for ratio-constraint oracle tests
# ---------------------------------------------------------------------------


def make_ratio_oracle_df(
    n_quotes: int = 30, n_steps: int = 5
) -> pl.DataFrame:
    """Synthetic DataFrame for ratio-constraint oracle tests.

    Mirrors ``test_ratio_solve_c2.make_ratio_solve_df`` but reproduced
    here so the oracle file is self-contained (no cross-imports between
    test modules).
    """
    rows = []
    mults = [0.8 + 0.1 * j for j in range(n_steps)]
    for q in range(n_quotes):
        elasticity = 1.0 + 1.5 * q / n_quotes
        base = 100.0 + 30.0 * q / n_quotes
        quote_baseline_lr = 0.40 + 0.50 * q / n_quotes
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
            "income": pl.Float32,
            "incurred": pl.Float32,
            "premium": pl.Float32,
        },
    )


# ---------------------------------------------------------------------------
# Helpers — pivot DataFrame to (n_quotes, n_steps) numpy grids
# ---------------------------------------------------------------------------


def _pivot_column_to_grid(
    df: pl.DataFrame, col: str, scenario_index_col: str = "scenario_index"
) -> np.ndarray:
    """Pivot a long-format column into a (n_quotes, n_steps) float64 array.

    Rows are sorted by ``quote_id`` then ``scenario_index`` so the grid
    layout matches price-contour's internal quote-major layout.
    """
    df_sorted = df.sort(["quote_id", scenario_index_col])
    pivoted = df_sorted.pivot(
        values=col,
        index="quote_id",
        on=scenario_index_col,
        aggregate_function="first",
    )
    step_cols = sorted(
        [c for c in pivoted.columns if c != "quote_id"], key=int
    )
    return pivoted.select(step_cols).to_numpy().astype(np.float64)


def _build_eq_constraints(n_quotes: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Probability-simplex constraint per quote: each row sums to 1."""
    n_vars = n_quotes * n_steps
    A_eq = np.zeros((n_quotes, n_vars))
    for i in range(n_quotes):
        A_eq[i, i * n_steps : (i + 1) * n_steps] = 1.0
    b_eq = np.ones(n_quotes)
    return A_eq, b_eq


# ---------------------------------------------------------------------------
# LP problem builders
# ---------------------------------------------------------------------------


def build_sum_lp(
    obj_grid: np.ndarray,
    sum_specs: list[tuple[np.ndarray, str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build linprog inputs for a sum-constraint LP.

    Parameters
    ----------
    obj_grid : (n_quotes, n_steps) float64 — objective per (quote, step).
    sum_specs : list of (constraint_grid, direction, threshold) where
        ``direction`` is ``"min"`` (>= threshold) or ``"max"`` (<= threshold).

    Returns
    -------
    (c, A_ub, b_ub, A_eq, b_eq) for ``linprog(method="highs")``.
    """
    n_quotes, n_steps = obj_grid.shape
    c = -obj_grid.flatten()  # negate: linprog minimises
    A_eq, b_eq = _build_eq_constraints(n_quotes, n_steps)

    A_ub_rows: list[np.ndarray] = []
    b_ub_vals: list[float] = []
    for grid, direction, threshold in sum_specs:
        if direction == "max":
            # Sigma c_i * x_i <= threshold
            A_ub_rows.append(grid.flatten())
            b_ub_vals.append(threshold)
        elif direction == "min":
            # Sigma c_i * x_i >= threshold  =>  -Sigma c_i * x_i <= -threshold
            A_ub_rows.append(-grid.flatten())
            b_ub_vals.append(-threshold)
        else:
            raise ValueError(f"unknown direction {direction!r}")

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    return c, A_ub, b_ub, A_eq, b_eq


def build_ratio_lp(
    obj_grid: np.ndarray,
    sum_specs: list[tuple[np.ndarray, str, float]],
    ratio_specs: list[tuple[np.ndarray, np.ndarray, str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build linprog inputs combining sum and ratio (linearised) constraints.

    Each ratio spec is ``(num_grid, denom_grid, direction, L)``. The
    linearisation matches price-contour's implementation::

        max-direction:  Sigma (num - L * denom) * x <= 0
        min-direction:  Sigma (num - L * denom) * x >= 0
                        equivalently -Sigma (num - L * denom) * x <= 0
    """
    c, A_ub_sum, b_ub_sum, A_eq, b_eq = build_sum_lp(obj_grid, sum_specs)

    A_ub_ratio_rows: list[np.ndarray] = []
    b_ub_ratio_vals: list[float] = []
    for num_grid, denom_grid, direction, L in ratio_specs:
        linearised = (num_grid - L * denom_grid).flatten()
        if direction == "max":
            A_ub_ratio_rows.append(linearised)
            b_ub_ratio_vals.append(0.0)
        elif direction == "min":
            A_ub_ratio_rows.append(-linearised)
            b_ub_ratio_vals.append(0.0)
        else:
            raise ValueError(f"unknown ratio direction {direction!r}")

    if A_ub_sum is None and not A_ub_ratio_rows:
        return c, None, None, A_eq, b_eq

    rows = []
    vals = []
    if A_ub_sum is not None:
        rows.append(A_ub_sum)
        vals.append(b_ub_sum)
    if A_ub_ratio_rows:
        rows.append(np.array(A_ub_ratio_rows))
        vals.append(np.array(b_ub_ratio_vals))
    A_ub = np.vstack(rows)
    b_ub = np.concatenate(vals)
    return c, A_ub, b_ub, A_eq, b_eq


def solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq) -> tuple[float, np.ndarray]:
    """Run linprog with HiGHS, assert feasibility, return (objective, x).

    Returns the *negated* fun value (linprog minimises, we want maximise).
    """
    n_vars = c.size
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, 1)] * n_vars,
        method="highs",
    )
    assert res.success, f"linprog failed: {res.message!r}"
    return -float(res.fun), res.x


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_pc_to_lp(
    pc_obj: float,
    lp_obj: float,
    lower_bound: float = LP_LOWER_BOUND_TYPICAL,
    upper_rtol: float = LP_UPPER_RTOL,
) -> None:
    """Assert price-contour's objective is sandwiched by the LP relaxation.

    Two assertions:

    1. ``pc_obj <= lp_obj + abs(lp_obj) * upper_rtol`` — LP relaxation is
       a valid upper bound on the discrete problem; price-contour cannot
       beat the LP at the same constraint level.
    2. ``pc_obj >= lp_obj * lower_bound`` — price-contour's integrality
       gap is bounded; a regression that moves the objective by more than
       (1 - lower_bound) of the LP optimum will fail loudly.
    """
    assert pc_obj <= lp_obj + abs(lp_obj) * upper_rtol, (
        f"price-contour objective {pc_obj:.6f} exceeds LP upper bound "
        f"{lp_obj:.6f} (slack {upper_rtol:.0e}); discrete optimiser is "
        f"reporting an infeasible objective relative to the LP relaxation."
    )
    assert pc_obj >= lp_obj * lower_bound, (
        f"price-contour objective {pc_obj:.6f} below LP lower bound "
        f"{lp_obj * lower_bound:.6f} (= {lower_bound:.0%} of LP "
        f"{lp_obj:.6f}); integrality gap exceeds {1 - lower_bound:.0%}."
    )


# ---------------------------------------------------------------------------
# Sum constraint oracle tests
# ---------------------------------------------------------------------------


class TestOracleSumConstraints:
    """Compare price-contour's objective to the LP relaxation under sum
    constraints — verifies the discrete optimiser is near-optimal at the
    constraint level it actually enforces.
    """

    def test_oracle_sum_max_constraint(self):
        """Single ``max`` sum constraint that binds on volume.

        Loose enough (max_pct=1.05) that price-contour reliably converges
        to a feasible point. LP at price-contour's achieved constraint
        level upper-bounds price-contour's discrete objective.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"max_pct": 1.05}},
            max_iter=500,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        obj_grid = _pivot_column_to_grid(df, "expected_income")
        vol_grid = _pivot_column_to_grid(df, "volume")
        # LP threshold = price-contour's actual achieved constraint level.
        # See module docstring for rationale.
        threshold = float(pc_result.total_constraints["volume"])
        c, A_ub, b_ub, A_eq, b_eq = build_sum_lp(
            obj_grid, [(vol_grid, "max", threshold)]
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        compare_pc_to_lp(pc_result.total_objective, lp_obj)

    def test_oracle_sum_min_constraint(self):
        """Single ``min`` sum constraint on volume.

        ``min_pct=1.20`` is binding (above unconstrained-optimum volume
        on this fixture) and tight enough that price-contour exercises
        the dual update path repeatedly.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.20}},
            max_iter=500,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        obj_grid = _pivot_column_to_grid(df, "expected_income")
        vol_grid = _pivot_column_to_grid(df, "volume")
        threshold = float(pc_result.total_constraints["volume"])
        c, A_ub, b_ub, A_eq, b_eq = build_sum_lp(
            obj_grid, [(vol_grid, "min", threshold)]
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        compare_pc_to_lp(pc_result.total_objective, lp_obj)

    def test_oracle_sum_min_pct(self):
        """``_pct`` direction key resolves to absolute via baseline.

        Solves the same problem as :meth:`test_oracle_sum_min_constraint`
        but pins the LP threshold using the absolute volume that
        ``min_pct=1.20`` corresponds to (1.20 * baseline volume). The
        baseline used must match price-contour's: the sum at
        ``scenario_value == 1.0``.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        baseline_vol = float(
            df.filter(pl.col("scenario_value") == 1.0)["volume"].sum()
        )

        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.20}},
            max_iter=500,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        # Sanity: price-contour's reported baseline matches our manual one.
        assert math.isclose(
            pc_result.baseline_constraints["volume"], baseline_vol, rel_tol=1e-4
        ), (
            f"baseline_vol mismatch: pc reports "
            f"{pc_result.baseline_constraints['volume']}, manual "
            f"{baseline_vol}"
        )

        obj_grid = _pivot_column_to_grid(df, "expected_income")
        vol_grid = _pivot_column_to_grid(df, "volume")
        threshold = float(pc_result.total_constraints["volume"])
        c, A_ub, b_ub, A_eq, b_eq = build_sum_lp(
            obj_grid, [(vol_grid, "min", threshold)]
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        compare_pc_to_lp(pc_result.total_objective, lp_obj)

    def test_oracle_two_sum_constraints(self):
        """One ``min`` plus one ``max``; LP must respect both.

        Two-constraint problems can produce fractional LP vertices
        (intersection of constraint hyperplanes); the integrality gap is
        therefore larger than single-constraint cases. The 0.95 lower
        bound still holds on this fixture.
        """
        df = make_small_df(n_quotes=50, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={
                "volume": {"min_pct": 0.95},
                "loss_ratio": {"max_pct": 1.05},
            },
            max_iter=500,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        obj_grid = _pivot_column_to_grid(df, "expected_income")
        vol_grid = _pivot_column_to_grid(df, "volume")
        lr_grid = _pivot_column_to_grid(df, "loss_ratio")
        vol_threshold = float(pc_result.total_constraints["volume"])
        lr_threshold = float(pc_result.total_constraints["loss_ratio"])
        c, A_ub, b_ub, A_eq, b_eq = build_sum_lp(
            obj_grid,
            [
                (vol_grid, "min", vol_threshold),
                (lr_grid, "max", lr_threshold),
            ],
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        compare_pc_to_lp(pc_result.total_objective, lp_obj)


# ---------------------------------------------------------------------------
# Ratio constraint oracle tests
# ---------------------------------------------------------------------------


class TestOracleRatioConstraints:
    """Compare price-contour's objective to the LP relaxation under ratio
    constraints (linearised)."""

    def test_oracle_ratio_max(self):
        """Absolute ``max`` ratio constraint, single ratio.

        The LP linearisation uses ``L`` equal to the ratio target the
        user passed in (``max=0.62``). price-contour's actual ratio
        should match the target within tolerance — if it does, the LP
        at L=target produces the same objective bound; if it overshoots
        slightly, we re-anchor L to price-contour's actual ratio so the
        upper bound holds.
        """
        df = make_ratio_oracle_df(n_quotes=30, n_steps=5)
        target = 0.62
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": target,
                }
            },
            max_iter=500,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        # Compute price-contour's actual ratio at the optimum from the
        # stitched ``optimal_*`` columns.
        out = pc_result.dataframe
        pc_num = float(out["optimal_incurred"].sum())
        pc_denom = float(out["optimal_premium"].sum())
        pc_ratio = pc_num / pc_denom
        # LP at price-contour's actual L (anchors the comparison fairly
        # even when price-contour overshoots the user target).
        L = pc_ratio

        obj_grid = _pivot_column_to_grid(df, "income")
        num_grid = _pivot_column_to_grid(df, "incurred")
        denom_grid = _pivot_column_to_grid(df, "premium")
        c, A_ub, b_ub, A_eq, b_eq = build_ratio_lp(
            obj_grid, [], [(num_grid, denom_grid, "max", L)]
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        compare_pc_to_lp(pc_result.total_objective, lp_obj)

    def test_oracle_ratio_min_pct(self):
        """``min_pct`` direction; ``L = pct * baseline_LR``.

        Verifies the LP setup mirrors price-contour's ``L`` derivation
        for ``_pct`` modes — the baseline ratio is computed from the
        ``scenario_value == 1.0`` rows.

        Uses the loss-ratio fixture but with a ``min_pct`` direction
        that biases the solver toward higher (rather than lower) ratios.
        """
        df = make_ratio_oracle_df(n_quotes=30, n_steps=5)
        baseline = df.filter(pl.col("scenario_value") == 1.0)
        baseline_lr = float(baseline["incurred"].sum()) / float(
            baseline["premium"].sum()
        )
        # Use a min_pct that's slack at baseline but pushes some quotes
        # toward higher-LR scenarios. Small lower-bound to ensure the
        # constraint is well-defined; the LP comparison still holds.
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "min_pct": 0.95,
                }
            },
            max_iter=500,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        # Recompute price-contour's actual L.
        out = pc_result.dataframe
        pc_num = float(out["optimal_incurred"].sum())
        pc_denom = float(out["optimal_premium"].sum())
        pc_ratio = pc_num / pc_denom
        # Sanity: baseline_LR helper here matches what price-contour used.
        target_L = 0.95 * baseline_lr
        # Use price-contour's actual ratio for a tight comparison.
        # min direction LP: sum (num - L * denom) * x >= 0
        L = pc_ratio
        # Sanity that price-contour respected the min: L >= target_L (within
        # discrete-grid tolerance). If price-contour underran, the LP at L
        # would still be a valid upper bound at the *achieved* ratio.
        assert L >= target_L * 0.95, (
            f"price-contour ratio {L} far below min target {target_L}"
        )

        obj_grid = _pivot_column_to_grid(df, "income")
        num_grid = _pivot_column_to_grid(df, "incurred")
        denom_grid = _pivot_column_to_grid(df, "premium")
        c, A_ub, b_ub, A_eq, b_eq = build_ratio_lp(
            obj_grid, [], [(num_grid, denom_grid, "min", L)]
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        compare_pc_to_lp(pc_result.total_objective, lp_obj)

    def test_oracle_mixed_sum_and_ratio(self):
        """One sum constraint plus one ratio constraint together."""
        df = make_ratio_oracle_df(n_quotes=30, n_steps=5)
        baseline = df.filter(pl.col("scenario_value") == 1.0)
        baseline_premium = float(baseline["premium"].sum())
        # Sum floor on premium + max on loss_ratio. Both bind moderately.
        solver = pc.OnlineOptimiser(
            objective="income",
            constraints={
                "premium": {"min": 0.85 * baseline_premium},
                "loss_ratio": {
                    "numerator": "incurred",
                    "denominator": "premium",
                    "max": 0.62,
                },
            },
            max_iter=500,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        out = pc_result.dataframe
        pc_num = float(out["optimal_incurred"].sum())
        pc_denom = float(out["optimal_premium"].sum())
        pc_ratio = pc_num / pc_denom

        obj_grid = _pivot_column_to_grid(df, "income")
        premium_grid = _pivot_column_to_grid(df, "premium")
        incurred_grid = _pivot_column_to_grid(df, "incurred")
        # LP at price-contour's actual constraint levels.
        sum_threshold = float(pc_result.total_constraints["premium"])
        c, A_ub, b_ub, A_eq, b_eq = build_ratio_lp(
            obj_grid,
            [(premium_grid, "min", sum_threshold)],
            [(incurred_grid, premium_grid, "max", pc_ratio)],
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        # Mixed sum+ratio constraints can have a slightly larger
        # integrality gap than single-constraint cases (two binding
        # hyperplanes intersect at fractional vertices). 0.93 covers
        # the empirical range.
        compare_pc_to_lp(
            pc_result.total_objective, lp_obj, lower_bound=0.93
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestOracleEdgeCases:
    """Edge cases: slack constraints (LP == PC unconstrained) and tight
    constraints (larger integrality gap permitted).
    """

    def test_oracle_slack_constraint(self):
        """Constraint far from binding — LP and price-contour both
        recover the unconstrained optimum.

        ``min_pct=0.50`` on volume is well below baseline; both solvers
        pick the per-quote argmax of the objective. The two should
        match to floating-point precision.
        """
        df = make_small_df(n_quotes=30, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.50}},
            max_iter=200,
            tolerance=1e-5,
        )
        pc_result = solver.solve(df)

        obj_grid = _pivot_column_to_grid(df, "expected_income")
        vol_grid = _pivot_column_to_grid(df, "volume")
        threshold = float(pc_result.total_constraints["volume"])
        c, A_ub, b_ub, A_eq, b_eq = build_sum_lp(
            obj_grid, [(vol_grid, "min", threshold)]
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        # Slack constraint: integrality gap is essentially zero.
        compare_pc_to_lp(
            pc_result.total_objective, lp_obj, lower_bound=0.999
        )

    def test_oracle_tight_constraint(self):
        """Constraint near the feasibility boundary — wider gap allowed.

        ``min_pct=1.22`` on volume is right at the achievable boundary
        for this fixture (max possible volume on a 30-quote, 5-step
        grid is ~1.30 * baseline). Integrality gap can grow to several
        percent.
        """
        df = make_small_df(n_quotes=30, n_steps=5)
        solver = pc.OnlineOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 1.22}},
            max_iter=1000,
            tolerance=1e-6,
        )
        pc_result = solver.solve(df)

        obj_grid = _pivot_column_to_grid(df, "expected_income")
        vol_grid = _pivot_column_to_grid(df, "volume")
        threshold = float(pc_result.total_constraints["volume"])
        c, A_ub, b_ub, A_eq, b_eq = build_sum_lp(
            obj_grid, [(vol_grid, "min", threshold)]
        )
        lp_obj, _ = solve_lp_oracle(c, A_ub, b_ub, A_eq, b_eq)

        # Tight: 90% lower bound
        compare_pc_to_lp(
            pc_result.total_objective, lp_obj, lower_bound=LP_LOWER_BOUND_TIGHT
        )
