"""A one-constraint solve bisects its λ, as the 1-D frontier does (DESIGN_DECISIONS §20).

The subgradient method can stop far short of a bound the grid can meet; with one
constraint the total is monotone in λ, so ``solve()`` bisects it instead. The
solve and the frontier point at its bound are then one method and one answer.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import price_contour as pc

# The bisection's own caps: the λ=0 probe, the first bracket probe, the
# bracket-expansion doublings and the halvings (§16.2).
MAX_PROBES = 1 + 1 + 32 + 60

HISTORY_KEYS = {
    "iteration",
    "phase",
    "bracket_lo",
    "bracket_hi",
    "lambdas",
    "total_objective",
    "total_constraints",
    "all_constraints_satisfied",
    "accepted",
    "max_lambda_change",
}


def _book(n_quotes: int = 5000, n_steps: int = 9, seed: int = 7) -> pl.DataFrame:
    """Quotes with their own premium, elasticity and conversion: income peaks
    above the volume each quote can give, so a volume minimum trades one for
    the other quote by quote."""
    rng = np.random.default_rng(seed)
    values = np.linspace(0.8, 1.2, n_steps).astype(np.float32)
    base = rng.uniform(200, 800, n_quotes)
    elasticity = rng.uniform(2, 9, n_quotes)
    conversion_at_one = rng.uniform(0.3, 0.9, n_quotes)
    quote = np.repeat(np.arange(n_quotes), n_steps)
    value = np.tile(values, n_quotes)
    conversion = 1 / (
        1 + (1 / conversion_at_one[quote] - 1) * np.exp(elasticity[quote] * (value - 1))
    )
    return pl.DataFrame(
        {
            "quote_id": [f"q{i:05d}" for i in quote],
            "scenario_index": pl.Series(
                np.tile(np.arange(n_steps), n_quotes), dtype=pl.Int32
            ),
            "scenario_value": pl.Series(value, dtype=pl.Float32),
            "expected_income": pl.Series(
                conversion * base[quote] * (value - 0.75), dtype=pl.Float32
            ),
            "volume": pl.Series(conversion, dtype=pl.Float32),
            "count": pl.Series(np.ones_like(conversion), dtype=pl.Float32),
        }
    )


def _volume_reach(df: pl.DataFrame) -> tuple[float, float]:
    """The volume each quote's best-income step gives, and the most the grid allows."""
    best = (
        df.sort(["quote_id", "expected_income"], descending=[False, True])
        .group_by("quote_id", maintain_order=True)
        .first()["volume"]
        .cast(pl.Float64)
        .sum()
    )
    most = (
        df.group_by("quote_id")
        .agg(pl.col("volume").max())["volume"]
        .cast(pl.Float64)
        .sum()
    )
    return float(best), float(most)


@pytest.fixture(scope="module")
def book() -> pl.DataFrame:
    return _book()


@pytest.fixture(scope="module")
def binding_bound(book: pl.DataFrame) -> float:
    best, most = _volume_reach(book)
    return 0.5 * (best + most)


def test_a_binding_minimum_is_met_where_the_subgradient_method_stopped_short(
    book: pl.DataFrame, binding_bound: float
) -> None:
    # With 0.10.0's subgradient method this solve ended 19% short at 50
    # iterations and 10% short at 1,000.
    result = pc.OnlineOptimiser(
        constraints={"volume": {"min": binding_bound}}, max_iter=50, tolerance=1e-6
    ).solve(book)

    assert result.solver_path == "bisection"
    assert result.converged is True
    assert result.non_convergence_reason is None
    assert result.total_constraints["volume"] >= binding_bound
    assert result.lambdas["volume"] > 0
    assert result.iterations <= MAX_PROBES


def test_the_solve_is_its_frontiers_first_point(
    book: pl.DataFrame, binding_bound: float
) -> None:
    solver = pc.OnlineOptimiser(constraints={"volume": {"min": binding_bound}})
    result = solver.solve(book)
    _best, most = _volume_reach(book)
    frontier = solver.frontier(
        result.grid,
        threshold_ranges={"volume": (binding_bound, most)},
        n_points_per_dim=3,
        initial_lambdas=result.lambdas,
    )
    first = frontier.points.row(0, named=True)

    # One method: the point's steps are the solve's. Its warm start moves where the
    # bisection starts, so its λ agrees to the bisection's resolution.
    assert first["solver_path"] == "bisection"
    assert first["total_objective"] == result.total_objective
    assert first["total_volume"] == result.total_constraints["volume"]
    assert first["lambda_volume"] == pytest.approx(result.lambdas["volume"], rel=1e-6)


def test_a_bisection_history_is_its_probes(
    book: pl.DataFrame, binding_bound: float
) -> None:
    result = pc.OnlineOptimiser(
        constraints={"volume": {"min": binding_bound}}, record_history=True
    ).solve(book)
    history = result.history

    assert history is not None
    assert len(history) == result.iterations
    assert all(set(record) == HISTORY_KEYS for record in history)
    assert [record["iteration"] for record in history] == list(range(len(history)))
    assert history[0]["phase"] == "bracket"
    assert history[0]["lambdas"] == {"volume": 0.0}
    for record in history:
        assert record["phase"] in {"bracket", "bisect"}
        assert record["max_lambda_change"] is None
        assert isinstance(record["accepted"], bool)
        bracketed = record["bracket_lo"] is not None
        assert bracketed == (record["phase"] == "bisect")
        assert bracketed == (record["bracket_hi"] is not None)
    answer = [record for record in history if record["accepted"]][-1]
    assert answer["lambdas"] == result.lambdas
    assert answer["total_constraints"] == result.total_constraints
    assert answer["total_objective"] == result.total_objective


def test_a_subgradient_history_has_the_same_keys(
    book: pl.DataFrame, binding_bound: float
) -> None:
    result = pc.OnlineOptimiser(
        constraints={"volume": {"min": binding_bound}, "count": {"max": 1e9}},
        max_iter=5,
        record_history=True,
    ).solve(book)
    history = result.history

    assert result.solver_path == "subgradient"
    assert history is not None
    assert len(history) == result.iterations
    for record in history:
        assert set(record) == HISTORY_KEYS
        assert record["phase"] == "subgradient"
        assert record["bracket_lo"] is None and record["bracket_hi"] is None
        assert record["accepted"] is None
        assert math.isfinite(record["max_lambda_change"])


def test_a_subgradient_solve_out_of_iterations_says_so(
    book: pl.DataFrame, binding_bound: float
) -> None:
    result = pc.OnlineOptimiser(
        constraints={"volume": {"min": binding_bound}, "count": {"max": 1e9}},
        max_iter=2,
    ).solve(book)

    assert result.converged is False
    assert result.non_convergence_reason == "iteration_budget_exhausted"


def test_a_bound_beyond_the_grid_is_above_the_envelope(book: pl.DataFrame) -> None:
    _best, most = _volume_reach(book)
    result = pc.OnlineOptimiser(constraints={"volume": {"min": most * 1.01}}).solve(
        book
    )

    assert result.solver_path == "bisection"
    assert result.converged is False
    assert result.non_convergence_reason == "above_envelope"
    assert result.total_constraints["volume"] < most * 1.01


def test_a_met_bound_is_the_unconstrained_optimum(book: pl.DataFrame) -> None:
    best, _most = _volume_reach(book)
    result = pc.OnlineOptimiser(constraints={"volume": {"min": 0.5 * best}}).solve(book)
    unconstrained = pc.OnlineOptimiser().solve(book)

    assert result.converged is True
    assert result.lambdas == {"volume": 0.0}
    assert result.total_objective == pytest.approx(
        unconstrained.total_objective, rel=1e-12
    )


def _ratio_book(n_quotes: int = 200, n_steps: int = 5) -> pl.DataFrame:
    """Quotes whose loss ratio rises with the price, each from its own baseline."""
    rows = []
    for q in range(n_quotes):
        elasticity = 1.0 + 1.5 * q / n_quotes
        base = 100.0 + 30.0 * q / n_quotes
        quote_lr = 0.40 + 0.50 * q / n_quotes
        for j in range(n_steps):
            value = 0.8 + 0.1 * j
            premium = base * value / (1.0 + math.exp(elasticity * (value - 1.0)))
            rows.append(
                {
                    "quote_id": f"Q{q:04d}",
                    "scenario_index": j,
                    "scenario_value": value,
                    "income": premium,
                    "incurred": premium * quote_lr * (1.0 + 0.4 * (value - 1.0)),
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


_LOSS_RATIO = {"numerator": "incurred", "denominator": "premium"}


def test_one_ratio_constraint_bisects_and_replays_each_probes_ratio() -> None:
    df = _ratio_book()
    result = pc.OnlineOptimiser(
        objective="income",
        constraints={"loss_ratio": {**_LOSS_RATIO, "max": 0.64}},
        record_history=True,
    ).solve(df)
    history = result.history

    assert result.solver_path == "bisection"
    assert result.converged is True
    assert result.total_constraints["loss_ratio"] <= 0.64
    assert history is not None
    answer = [record for record in history if record["accepted"]][-1]
    # Each probe's total is the actual ratio at its λ, the answer's the result's.
    assert answer["total_constraints"]["loss_ratio"] == pytest.approx(
        result.total_constraints["loss_ratio"], rel=1e-9
    )
    assert all(
        0.0 < record["total_constraints"]["loss_ratio"] < 1.0 for record in history
    )


def test_a_python_orchestrated_one_constraint_sweep_reports_bisection_points() -> None:
    df = _ratio_book()
    frontier = pc.OnlineOptimiser(
        objective="income", constraints={"loss_ratio": {**_LOSS_RATIO, "max": 0.64}}
    ).frontier(
        df,
        threshold_ranges={"loss_ratio": (0.63, 0.65)},
        n_points_per_dim=3,
        record_history=True,
    )

    assert set(frontier.points["solver_path"]) == {"bisection"}
    assert frontier.points["non_convergence_reason"].null_count() == 3
    history = frontier.point_history
    assert history is not None
    assert set(history["phase"]) <= {"bracket", "bisect"}
    assert history["accepted"].null_count() == 0
    assert history["max_lambda_change"].null_count() == history.height


def test_summary_records_the_path_and_types_each_column() -> None:
    df = _ratio_book()
    solver = pc.OnlineOptimiser(
        objective="income",
        constraints={"loss_ratio": {**_LOSS_RATIO, "max": 0.64}},
        record_history=True,
    )
    summary = solver.summary(solver.solve(df))
    convergence = summary["artifacts"]["convergence"]

    assert summary["artifacts"]["summary"]["solver_path"] == "bisection"
    assert summary["artifacts"]["summary"]["non_convergence_reason"] is None
    assert convergence.schema["max_lambda_change"] == pl.Float64
    assert convergence.schema["accepted"] == pl.Boolean
    assert convergence.schema["bracket_lo"] == pl.Float64


def _one_quote(rows: list[tuple[int, float, float, float]]) -> pl.DataFrame:
    """One quote's rows, as given: (scenario_index, objective, numerator, denominator)."""
    return pl.DataFrame(
        {
            "quote_id": ["Q0"] * len(rows),
            "scenario_index": pl.Series([row[0] for row in rows], dtype=pl.Int32),
            "scenario_value": pl.Series(
                [0.9 + 0.1 * row[0] for row in rows], dtype=pl.Float32
            ),
            "income": pl.Series([row[1] for row in rows], dtype=pl.Float32),
            "incurred": pl.Series([row[2] for row in rows], dtype=pl.Float32),
            "premium": pl.Series([row[3] for row in rows], dtype=pl.Float32),
        }
    )


# By hand: loss ratios 0.3, 0.4, 0.6 at objectives 1, 2, 3. Under a 0.5 maximum the
# best choice is step 1 (objective 2, ratio 0.4); below 0.3 nothing is reachable.
_THREE_STEPS = [(0, 1.0, 30.0, 100.0), (1, 2.0, 40.0, 100.0), (2, 3.0, 60.0, 100.0)]


@pytest.mark.parametrize("warm", [None, 6e8, 1e9, 1e12])
@pytest.mark.parametrize(
    "incurred",
    [
        pytest.param((30.0, 40.0, 60.0), id="a_wide_interval"),
        # Every step's Lagrangian meets at one λ, so only the bisection's finest
        # resolution tells the middle step from its neighbours.
        pytest.param((20.0, 40.0, 60.0), id="a_triple_tie"),
    ],
)
def test_a_distant_warm_start_finds_the_cold_answer(
    warm: float | None, incurred: tuple[float, float, float]
) -> None:
    rows = [(step, step + 1.0, incurred[step], 100.0) for step in range(3)]
    solver = pc.OnlineOptimiser(
        objective="income", constraints={"loss_ratio": {**_LOSS_RATIO, "max": 0.5}}
    )
    cold = solver.solve(_one_quote(rows))
    result = solver.solve(
        _one_quote(rows), lambdas=None if warm is None else {"loss_ratio": warm}
    )

    assert result.converged is True
    assert result.total_objective == cold.total_objective == 2.0
    assert result.total_constraints["loss_ratio"] == pytest.approx(0.4)


def test_a_sweep_past_an_unreachable_bound_keeps_its_precision() -> None:
    """The first point's bound is beyond the grid; its capped λ must not coarsen the next."""
    frontier = pc.OnlineOptimiser(
        objective="income", constraints={"loss_ratio": {**_LOSS_RATIO, "max": 0.5}}
    ).frontier(
        _one_quote(_THREE_STEPS),
        threshold_ranges={"loss_ratio": (0.2, 0.8)},
        n_points_per_dim=3,
    )
    rows = frontier.points.select(
        "threshold_loss_ratio", "converged", "non_convergence_reason", "total_objective"
    ).rows()

    assert rows == [
        (pytest.approx(0.2), False, "above_envelope", 1.0),
        (pytest.approx(0.5), True, None, 2.0),
        (pytest.approx(0.8), True, None, 3.0),
    ]


def test_a_replayed_record_breaks_ties_as_the_solver_does() -> None:
    """At the answer's λ the two steps tie; the solver takes the lowest scenario index
    (ratio 0.4), and so must the history's replay, whatever the rows' order."""
    reversed_rows = _one_quote([(1, 3.0, 6.0, 10.0), (0, 1.0, 4.0, 10.0)])
    result = pc.OnlineOptimiser(
        objective="income",
        constraints={"loss_ratio": {**_LOSS_RATIO, "max": 0.5}},
        record_history=True,
    ).solve(reversed_rows)
    answer = [record for record in result.history if record["accepted"]][-1]

    assert result.total_constraints["loss_ratio"] == pytest.approx(0.4)
    assert answer["total_constraints"]["loss_ratio"] == pytest.approx(0.4)
    assert answer["all_constraints_satisfied"] is True


@pytest.mark.parametrize("warm", [math.nan, math.inf])
def test_a_non_finite_warm_start_is_refused(warm: float) -> None:
    rows = [(step, step + 1.0, (30.0, 40.0, 60.0)[step], 100.0) for step in range(3)]
    solver = pc.OnlineOptimiser(
        objective="income", constraints={"loss_ratio": {**_LOSS_RATIO, "max": 0.5}}
    )

    with pytest.raises(ValueError, match="a warm-start lambda must be finite"):
        solver.solve(_one_quote(rows), lambdas={"loss_ratio": warm})


def test_a_non_finite_frontier_warm_start_is_refused(book: pl.DataFrame) -> None:
    best, most = _volume_reach(book)
    solver = pc.OnlineOptimiser(constraints={"volume": {"min": best}})

    with pytest.raises(ValueError, match="a warm-start lambda must be finite"):
        solver.frontier(
            book,
            threshold_ranges={"volume": (best, most)},
            n_points_per_dim=2,
            initial_lambdas={"volume": math.nan},
        )
