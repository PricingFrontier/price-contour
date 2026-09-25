"""FrontierResult wrapper and frontier_summary helper."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import polars as pl

from price_contour._price_contour import FrontierResult

__all__ = ["FrontierResult", "FrontierResultLike", "frontier_summary"]


@runtime_checkable
class FrontierResultLike(Protocol):
    """Protocol for frontier results (both online and ratebook).

    ``FrontierResult`` (the Rust online sweep), the Python-orchestrated
    online sweep and ``RatebookFrontierResult`` (``RatebookOptimiser.frontier``)
    all satisfy it. ``points`` follows ``frontier_points_schema(mode,
    constraint_names)``: ratebook points carry ``clamp_rate`` and clamp
    counts instead of the online ``sv_*`` distribution columns.
    """

    @property
    def points(self) -> pl.DataFrame:
        """DataFrame with one row per frontier point."""
        ...

    @property
    def n_points(self) -> int:
        """Number of frontier points."""
        ...

    @property
    def constraint_names(self) -> list[str]:
        """Constraint names, in column order."""
        ...


def frontier_summary(
    frontier_result: FrontierResultLike, selected_index: int
) -> dict[str, Any]:
    """Package a frontier result into MLflow-ready dicts.

    Parameters
    ----------
    frontier_result : FrontierResultLike
        Result from sweep_frontier or RatebookOptimiser.frontier.
        Any object satisfying the ``FrontierResultLike`` protocol.
    selected_index : int
        Index of the selected frontier point (row in the points DataFrame).

    Returns
    -------
    dict with keys: params, metrics, artifacts
    """
    df = frontier_result.points
    n = df.shape[0]
    if not (0 <= selected_index < n):
        raise IndexError(
            f"selected_index {selected_index} out of range for frontier with {n} points"
        )
    selected_row = df.row(selected_index, named=True)

    params: dict[str, Any] = {
        "frontier_n_points": n,
        "frontier_selected_index": selected_index,
    }

    metrics: dict[str, float] = {
        "selected_total_objective": selected_row["total_objective"],
        "selected_iterations": float(selected_row["iterations"]),
        "selected_converged": float(selected_row["converged"]),
    }

    # Per-constraint values of the selected point, by constraint name (not
    # by parsing column prefixes, which a constraint name could collide with).
    for name in frontier_result.constraint_names:
        for prefix in ("threshold", "bound", "total", "lambda"):
            metrics[f"selected_{prefix}_{name}"] = float(
                selected_row[f"{prefix}_{name}"]
            )

    artifacts: dict[str, Any] = {
        "frontier": df,
    }

    return {
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts,
    }
