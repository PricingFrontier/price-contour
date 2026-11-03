"""Per-point convergence traces of a frontier sweep (DESIGN_DECISIONS §16)."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import price_contour as pc
from price_contour._grid_utils import build_grid
from helpers import make_factors, make_small_df

ONLINE_KEYS = {"objective": "expected_income", "max_iter": 60}


def _grid(df: pl.DataFrame, constraints: list[str]) -> pc.QuoteGrid:
    return build_grid(
        df,
        constraint_columns=constraints,
        quote_id="quote_id",
        scenario_index="scenario_index",
        scenario_value="scenario_value",
        objective="expected_income",
    )


def _points_trace(history: pl.DataFrame, index: int) -> pl.DataFrame:
    return history.filter(pl.col("point_index") == index)


def _expected_subgradient_rows(
    index: int, solve_history: list[dict], names: list[str]
) -> pl.DataFrame:
    rows = [
        {
            "point_index": index,
            "step": record["iteration"],
            "phase": "subgradient",
            "bracket_lo": None,
            "bracket_hi": None,
            **{f"lambda_{n}": record["lambdas"][n] for n in names},
            "total_objective": record["total_objective"],
            **{f"total_{n}": record["total_constraints"][n] for n in names},
            "all_satisfied": record["all_constraints_satisfied"],
            "accepted": None,
            "max_lambda_change": record["max_lambda_change"],
        }
        for record in solve_history
    ]
    return pl.DataFrame(rows, schema=pc.frontier_point_history_schema("online", names))


class TestBisectionTrace:
    """A 1-D sweep records every bisection probe and nothing else."""

    @pytest.fixture(scope="class")
    def sweeps(self) -> tuple[pc.FrontierResult, pc.FrontierResult]:
        df = make_small_df(n_quotes=200, n_steps=7)
        grid = _grid(df, ["volume"])
        solver = pc.OnlineOptimiser(
            constraints={"volume": {"min_pct": 0.9}}, **ONLINE_KEYS
        )
        # λ=0 reaches 1.233 of baseline and the envelope 1.310: the range has
        # interior targets and, past 1.31, unreachable ones.
        kwargs = {"threshold_ranges": {"volume": (1.24, 1.36)}, "n_points_per_dim": 8}
        return (
            solver.frontier(grid, **kwargs),
            solver.frontier(grid, record_history=True, **kwargs),
        )

    def test_untraced_sweep_has_no_history(self, sweeps):
        plain, _ = sweeps
        assert plain.point_history is None

    def test_tracing_changes_no_point(self, sweeps):
        plain, traced = sweeps
        assert_frame_equal(traced.points, plain.points)

    def test_history_has_the_published_schema(self, sweeps):
        _, traced = sweeps
        assert traced.point_history.schema == pl.Schema(
            pc.frontier_point_history_schema("online", ["volume"])
        )

    def test_each_point_ends_at_its_last_accepted_probe(self, sweeps):
        _, traced = sweeps
        points = traced.points
        history = traced.point_history
        assert history["point_index"].unique().sort().to_list() == list(
            range(points.height)
        )
        assert not points["converged"].all(), (
            "the range must include unreachable targets"
        )
        for index, point in enumerate(points.iter_rows(named=True)):
            trace = _points_trace(history, index)
            assert trace.height == point["iterations"], "tracing adds no probe"
            assert trace["step"].to_list() == list(range(trace.height))
            assert set(trace["phase"]) <= {"bracket", "bisect"}
            answer = trace.filter(pl.col("accepted")).row(-1, named=True)
            assert answer["lambda_volume"] == point["lambda_volume"]
            assert answer["total_objective"] == point["total_objective"]
            assert answer["total_volume"] == point["total_volume"]
            if point["converged"]:
                assert answer["all_satisfied"]
            else:
                # The highest-λ probe of an unreachable target, which meets nothing.
                assert not trace["all_satisfied"].any()
                assert trace["accepted"].to_list()[-1]

    def test_bisect_probes_halve_their_bracket(self, sweeps):
        _, traced = sweeps
        bisect = traced.point_history.filter(pl.col("phase") == "bisect")
        assert bisect.height > 0
        midpoints = 0.5 * (bisect["bracket_lo"] + bisect["bracket_hi"])
        assert (bisect["lambda_volume"] == midpoints).all()
        bracket = traced.point_history.filter(pl.col("phase") == "bracket")
        assert (
            bracket["bracket_lo"].is_null().all()
            and bracket["bracket_hi"].is_null().all()
        )
        assert traced.point_history["max_lambda_change"].is_null().all()

    def test_satisfied_is_the_one_sided_target(self, sweeps):
        _, traced = sweeps
        joined = traced.point_history.join(
            traced.points.with_row_index("point_index").select(
                pl.col("point_index").cast(pl.Int64), "bound_volume"
            ),
            on="point_index",
            maintain_order="left",
        )
        assert (
            joined["all_satisfied"]
            == (joined["total_volume"] >= joined["bound_volume"])
        ).all()


class TestSubgradientTrace:
    """A multi-constraint point records the iterations its solve records."""

    NAMES = ["volume", "loss_ratio"]

    def test_parallel_sweep_points_trace_their_own_solve(self):
        df = make_small_df(n_quotes=100, n_steps=5)
        grid = _grid(df, self.NAMES)
        constraints = {"volume": {"min": None}, "loss_ratio": {"max": None}}
        ranges = {"volume": (40.0, 46.0), "loss_ratio": (60.0, 64.0)}
        initial = {"volume": 0.05, "loss_ratio": 0.0}
        solver = pc.OnlineOptimiser(constraints=constraints, **ONLINE_KEYS)
        result = solver.frontier(
            grid,
            threshold_ranges=ranges,
            n_points_per_dim=2,
            initial_lambdas=initial,
            parallel=True,
            record_history=True,
        )
        history = result.point_history
        assert history.schema == pl.Schema(
            pc.frontier_point_history_schema("online", self.NAMES)
        )
        for index, point in enumerate(result.points.iter_rows(named=True)):
            standalone = pc.OnlineOptimiser(
                constraints={
                    "volume": {"min": point["bound_volume"]},
                    "loss_ratio": {"max": point["bound_loss_ratio"]},
                },
                record_history=True,
                **ONLINE_KEYS,
            ).solve(grid, lambdas=initial)
            expected = _expected_subgradient_rows(index, standalone.history, self.NAMES)
            assert_frame_equal(_points_trace(history, index), expected)
            assert expected.height == point["iterations"]

    def test_orchestrated_sweep_points_trace_their_own_solve(self):
        # loss_ratio has no range, so the sweep is orchestrated in Python and
        # each point warm-starts from the previous one's lambdas.
        df = make_small_df(n_quotes=100, n_steps=5)
        constraints = {"volume": {"min": None}, "loss_ratio": {"max": 63.0}}
        solver = pc.OnlineOptimiser(constraints=constraints, **ONLINE_KEYS)
        result = solver.frontier(
            df,
            threshold_ranges={"volume": (40.0, 46.0)},
            n_points_per_dim=3,
            record_history=True,
        )
        plain = solver.frontier(
            df, threshold_ranges={"volume": (40.0, 46.0)}, n_points_per_dim=3
        )
        assert plain.point_history is None
        assert_frame_equal(result.points, plain.points)
        previous: dict[str, float] | None = None
        for index, point in enumerate(result.points.iter_rows(named=True)):
            standalone = pc.OnlineOptimiser(
                constraints={
                    "volume": {"min": point["threshold_volume"]},
                    "loss_ratio": {"max": 63.0},
                },
                record_history=True,
                **ONLINE_KEYS,
            ).solve(df, lambdas=previous)
            previous = dict(standalone.lambdas)
            expected = _expected_subgradient_rows(index, standalone.history, self.NAMES)
            assert_frame_equal(_points_trace(result.point_history, index), expected)


class TestRatebookTrace:
    """A ratebook point records its coordinate descent's inner solves."""

    @pytest.fixture(scope="class")
    def setting(self):
        df = make_small_df(n_quotes=80, n_steps=7)
        factors = make_factors(n_quotes=80)
        optimiser = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.95}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=60,
        )
        return df, factors, optimiser

    def test_tracing_changes_no_point(self, setting):
        df, factors, optimiser = setting
        kwargs = {"threshold_ranges": {"volume": (0.9, 1.0)}, "n_points_per_dim": 3}
        plain = optimiser.frontier(df, factors, **kwargs)
        traced = optimiser.frontier(df, factors, record_history=True, **kwargs)
        assert plain.point_history is None
        assert_frame_equal(traced.points, plain.points)
        assert traced.point_history.schema == pl.Schema(
            pc.frontier_point_history_schema("ratebook", ["volume"])
        )
        assert traced.point_history["point_index"].unique().sort().to_list() == [
            0,
            1,
            2,
        ]

    def test_a_point_traces_its_own_solve(self, setting):
        df, factors, optimiser = setting
        initial = {"volume": 0.2}
        traced = optimiser.frontier(
            df,
            factors,
            threshold_ranges={"volume": (0.97, 0.97)},
            n_points_per_dim=1,
            initial_lambdas=initial,
            record_history=True,
        )
        standalone = pc.RatebookOptimiser(
            objective="expected_income",
            constraints={"volume": {"min_pct": 0.97}},
            factor_columns=[["region"], ["age_band"]],
            max_cd_iterations=3,
            max_iter=60,
        ).solve(df, factors, lambdas=initial)
        expected = pl.DataFrame(
            [
                {
                    "point_index": 0,
                    "cd_iteration": record.cd_iteration,
                    "factor": record.factor,
                    "factor_index": record.factor_index,
                    "total_objective": record.total_objective,
                    "total_volume": record.total_constraints["volume"],
                    "lambda_volume": record.lambdas["volume"],
                    "clamp_rate": record.clamp_rate,
                    "inner_iterations": record.inner_iterations,
                    "inner_converged": record.inner_converged,
                }
                for record in standalone.per_factor_results
            ],
            schema=pc.frontier_point_history_schema("ratebook", ["volume"]),
        )
        assert_frame_equal(traced.point_history, expected)


def test_schemas_are_the_documented_columns_in_order():
    """DESIGN_DECISIONS §16.2, written out rather than read from the helper."""
    f64, i64, boolean, string = pl.Float64(), pl.Int64(), pl.Boolean(), pl.String()
    assert list(pc.frontier_point_history_schema("online", ["a", "b"]).items()) == [
        ("point_index", i64),
        ("step", i64),
        ("phase", string),
        ("bracket_lo", f64),
        ("bracket_hi", f64),
        ("lambda_a", f64),
        ("lambda_b", f64),
        ("total_objective", f64),
        ("total_a", f64),
        ("total_b", f64),
        ("all_satisfied", boolean),
        ("accepted", boolean),
        ("max_lambda_change", f64),
    ]
    assert list(pc.frontier_point_history_schema("ratebook", ["a", "b"]).items()) == [
        ("point_index", i64),
        ("cd_iteration", i64),
        ("factor", string),
        ("factor_index", i64),
        ("total_objective", f64),
        ("total_a", f64),
        ("total_b", f64),
        ("lambda_a", f64),
        ("lambda_b", f64),
        ("clamp_rate", f64),
        ("inner_iterations", i64),
        ("inner_converged", boolean),
    ]


def test_schema_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        pc.frontier_point_history_schema("grouped", ["volume"])
