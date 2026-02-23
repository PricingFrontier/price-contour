"""FrontierResult wrapper and frontier_summary helper."""

from __future__ import annotations

from typing import Any

from price_contour._price_contour import FrontierResult

__all__ = ["FrontierResult", "frontier_summary"]


def frontier_summary(frontier_result: FrontierResult, selected_index: int) -> dict[str, Any]:
    """Package a frontier result into MLflow-ready dicts.

    Parameters
    ----------
    frontier_result : FrontierResult
        Result from sweep_frontier.
    selected_index : int
        Index of the selected frontier point (row in the points DataFrame).

    Returns
    -------
    dict with keys: params, metrics, artifacts
    """
    df = frontier_result.points
    n = df.shape[0]
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

    # Add threshold and constraint values from selected point
    for col in df.columns:
        if col.startswith("threshold_"):
            name = col.removeprefix("threshold_")
            metrics[f"selected_threshold_{name}"] = float(selected_row[col])
        elif col.startswith("total_") and col != "total_objective":
            name = col.removeprefix("total_")
            metrics[f"selected_total_{name}"] = float(selected_row[col])
        elif col.startswith("lambda_"):
            name = col.removeprefix("lambda_")
            metrics[f"selected_lambda_{name}"] = float(selected_row[col])

    artifacts: dict[str, Any] = {
        "frontier": df,
    }

    return {
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts,
    }
