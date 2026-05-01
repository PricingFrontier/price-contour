"""Shared helpers for ratio-aware result wrappers.

Both :class:`OnlineOptimiser` and :class:`ApplyOptimiser` wrap their
inner Rust result with a Python-side decorator that:

* stitches in ``optimal_<numerator>`` / ``optimal_<denominator>``
  columns via a ``(quote_id, optimal_step)`` join against the original
  input DataFrame, and
* recomputes ``total_constraints`` / ``baseline_constraints`` for each
  ratio label as the actual ratio
  ``Sigma_optimal num / Sigma_optimal denom`` rather than the linearised
  total Rust returns.

The :func:`_safe_ratio_from_columns` helper handles the 0/0 sentinel
convention; :func:`_stitch_optimal_ratio_columns` handles the dedup
+ join + rename recipe; together they live here so both wrappers and
the ratebook ``_stitch_optimum_columns`` helper share one
implementation.
"""

from __future__ import annotations

import polars as pl


def _safe_ratio_from_columns(
    df: pl.DataFrame, numerator_col: str, denominator_col: str
) -> float:
    """Return ``Sigma df[num] / Sigma df[denom]`` with sentinel handling.

    Mirrors the test-side ``actual_ratio_at_optimum`` convention:

    * If the denominator sum is zero, return ``float('nan')`` (the
      tests' convention; covers the genuine 0/0 ambiguity).
    * If the result would be ±inf (numerator non-zero, denominator
      zero) the implementation also returns NaN — float division in
      Python handles this naturally only if we let it through, but
      treating both as NaN keeps the contract simple and matches the
      ``test_near_zero_optimal_denominator_handled_gracefully``
      acceptance criterion (non-finite sentinel OR zero are all OK).

    If ``Sigma denom == 0`` at the optimum, returns ``nan``. This
    sentinel propagates into ``total_constraints`` and ``summary()``
    output, surfacing the divide-by-zero loud rather than silently
    reporting zero.
    """
    num_total = float(df[numerator_col].sum())
    denom_total = float(df[denominator_col].sum())
    if denom_total == 0.0:
        return float("nan")
    return num_total / denom_total


def _stitch_optimal_ratio_columns(
    *,
    base_df: pl.DataFrame,
    original_df: pl.DataFrame,
    ratio_columns: list[tuple[str, str, str]],
    quote_id_col: str,
    scenario_index_col: str,
) -> pl.DataFrame:
    """Join numerator / denominator columns onto a chosen-step result
    DataFrame so the actual-ratio computation can read
    ``optimal_<num>`` / ``optimal_<denom>``.

    The Rust solver materialises ``optimal_<col>`` only for the
    constraint columns embedded in the grid — for ratio constraints
    this is the synthetic linearised column under the ratio's display
    label, NOT the underlying numerator / denominator. We pull those
    in via a join on ``(quote_id, optimal_step == scenario_index)``
    against the original input DataFrame.

    Skips ratios whose numerator / denominator already appear as
    ``optimal_<col>`` (e.g. when a sum constraint and a ratio
    constraint share a column) so the join doesn't introduce duplicate
    columns. Uses an inner join + height check so a missing chosen
    step surfaces as ``RuntimeError`` instead of silently filling NaN.
    """
    existing_optimal_cols = set(base_df.columns)
    wanted: list[str] = []
    seen: set[str] = set()
    for _label, numerator, denominator in ratio_columns:
        for col in (numerator, denominator):
            if col in seen:
                continue
            if f"optimal_{col}" in existing_optimal_cols:
                seen.add(col)
                continue
            seen.add(col)
            wanted.append(col)
    if not wanted:
        return base_df
    lookup = original_df.select([quote_id_col, scenario_index_col, *wanted]).rename(
        {scenario_index_col: "optimal_step"}
    )
    joined = base_df.join(
        lookup,
        on=[quote_id_col, "optimal_step"],
        how="inner",
    )
    if joined.height != base_df.height:
        raise RuntimeError(
            f"ratio-column stitch join produced {joined.height} rows "
            f"but expected {base_df.height} (one per chosen quote step); "
            f"this indicates a (quote_id, optimal_step) row in the "
            f"result has no matching row in the input DataFrame."
        )
    rename_map = {col: f"optimal_{col}" for col in wanted}
    return joined.rename(rename_map)
