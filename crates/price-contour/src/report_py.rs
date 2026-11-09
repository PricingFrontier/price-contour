//! Bindings for the frontier impact report (`docs/DESIGN_DECISIONS.md` §19.2).
//!
//! One call evaluates every point against a single baseline and returns only
//! aggregates: no per-quote buffer crosses to Python, and each point's
//! evaluation is released before the next.

use std::sync::Arc;

use price_contour_core::report::{
    column_totals, delta_summary, display_summary, group_impacts, step_counts, step_movement,
    transitions, DeltaSummary, DisplaySummary, GroupImpacts, StepMovement, Transition,
};
use price_contour_core::{GroupMapping, PriceContourError, QuoteGrid};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cancel_py::core_error;
use crate::factor_context_py::PyRatebookFactorContexts;
use crate::grid_py::PyQuoteGrid;
use crate::grouped_py::{PyRatebookEvaluation, QuoteBase};
use crate::ratebook_helpers_py::PyFactorContext;

struct PointAggregates {
    movement: StepMovement,
    display: DisplaySummary,
    deltas: Vec<DeltaSummary>,
    groups: Vec<GroupImpacts>,
    point_counts: Vec<u64>,
    transitions: Vec<Transition>,
    sensitivity: Vec<Vec<f64>>,
}

/// Aggregates for every frontier point: one dict per entry of `points`, each a
/// list of per-factor rate values in `contexts` order. `group_contexts` are
/// the mappings to break totals down by.
#[pyfunction]
#[pyo3(signature = (
    grid,
    contexts,
    group_contexts,
    points,
    *,
    base_contexts = None,
    existing_values = None,
    display_values = None,
    movement_band = None,
    sensitivity_grids = Vec::new(),
    n_bins = 40,
))]
#[allow(clippy::too_many_arguments)]
pub fn report_points_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    contexts: Vec<PyRef<'_, PyFactorContext>>,
    group_contexts: Vec<PyRef<'_, PyFactorContext>>,
    points: Vec<Vec<Vec<f32>>>,
    base_contexts: Option<PyRef<'_, PyRatebookFactorContexts>>,
    existing_values: Option<Vec<Vec<f32>>>,
    display_values: Option<Vec<f64>>,
    movement_band: Option<f64>,
    sensitivity_grids: Vec<PyRef<'_, PyQuoteGrid>>,
    n_bins: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let grid_arc = Arc::clone(&grid.inner);
    let n_steps = grid_arc.n_steps;
    if let Some(display) = display_values.as_ref() {
        if display.len() != n_steps {
            return Err(PyValueError::new_err(format!(
                "{} display values for {n_steps} steps",
                display.len()
            )));
        }
    }
    let sensitivity: Vec<Arc<QuoteGrid>> = sensitivity_grids
        .iter()
        .map(|g| Arc::clone(&g.inner))
        .collect();
    for other in &sensitivity {
        if other.n_quotes != grid_arc.n_quotes
            || other.n_steps != n_steps
            || other.constraints.len() != grid_arc.constraints.len()
        {
            return Err(PyValueError::new_err(
                "a sensitivity grid must have the shape of the main grid",
            ));
        }
    }
    // Without a display, a step reads as its own scenario value.
    let display_given = display_values.is_some();
    let display: Vec<f64> = display_values
        .unwrap_or_else(|| grid_arc.scenario_values.iter().map(|&v| v as f64).collect());
    let mappings: Vec<Arc<GroupMapping>> =
        contexts.iter().map(|c| Arc::clone(c.mapping())).collect();
    let group_mappings: Vec<Arc<GroupMapping>> = group_contexts
        .iter()
        .map(|c| Arc::clone(c.mapping()))
        .collect();

    // One baseline for the whole report.
    let quote_base = QuoteBase::resolve(
        py,
        base_contexts.as_deref(),
        existing_values,
        &mappings,
        &grid_arc,
    )?;
    let uniform: Vec<u32>;
    let baseline_steps: &[u32] = match quote_base.as_ref() {
        Some(base) => base.baseline_steps(),
        None => {
            uniform = vec![grid_arc.baseline_step() as u32; grid_arc.n_quotes];
            &uniform
        }
    };
    let first_sv = grid_arc.scenario_values[0];
    let last_sv = grid_arc.scenario_values[n_steps - 1];
    let (clamped_low, clamped_high) = match quote_base.as_ref() {
        Some(base) => (
            base.products().iter().filter(|&&p| p < first_sv).count() as u64,
            base.products().iter().filter(|&&p| p > last_sv).count() as u64,
        ),
        None => (
            if 1.0 < first_sv {
                grid_arc.n_quotes as u64
            } else {
                0
            },
            if 1.0 > last_sv {
                grid_arc.n_quotes as u64
            } else {
                0
            },
        ),
    };

    let baseline_counts = step_counts(baseline_steps, n_steps);

    let mut out = Vec::with_capacity(points.len());
    for factor_values in &points {
        let evaluation = PyRatebookEvaluation::evaluate(
            py,
            Arc::clone(&grid_arc),
            &mappings,
            factor_values,
            quote_base.clone(),
            None,
        )?;
        let steps = evaluation.optimal_steps();
        let aggregates = py
            .detach(|| -> Result<PointAggregates, PriceContourError> {
                let mut columns: Vec<&[f32]> = vec![grid_arc.objective.as_slice()];
                columns.extend(grid_arc.constraints.iter().map(|c| c.as_slice()));
                Ok(PointAggregates {
                    movement: step_movement(steps, baseline_steps, n_steps)?,
                    display: display_summary(&display, steps, baseline_steps)?,
                    deltas: columns
                        .iter()
                        .map(|c| delta_summary(c, n_steps, steps, baseline_steps, n_bins))
                        .collect::<Result<_, _>>()?,
                    groups: group_mappings
                        .iter()
                        .map(|m| {
                            group_impacts(
                                &grid_arc,
                                m,
                                steps,
                                baseline_steps,
                                &display,
                                movement_band,
                            )
                        })
                        .collect::<Result<_, _>>()?,
                    point_counts: step_counts(steps, n_steps),
                    transitions: transitions(&grid_arc, steps, baseline_steps)?,
                    sensitivity: sensitivity
                        .iter()
                        .map(|g| column_totals(g, steps))
                        .collect::<Result<_, _>>()?,
                })
            })
            .map_err(|e| core_error("Frontier report error", e))?;

        let record = PyDict::new(py);
        record.set_item("total_objective", evaluation.total_objective_value())?;
        record.set_item("total_constraints", evaluation.total_constraint_values())?;
        record.set_item("baseline_objective", evaluation.baseline_objective_value())?;
        record.set_item(
            "baseline_constraints",
            evaluation.baseline_constraint_values(),
        )?;
        record.set_item("baseline_clamped_low", clamped_low)?;
        record.set_item("baseline_clamped_high", clamped_high)?;
        record.set_item("movement", movement_dict(py, &aggregates, display_given)?)?;
        record.set_item(
            "scenario",
            scenario_dict(py, &aggregates, &baseline_counts)?,
        )?;
        record.set_item("sensitivity", aggregates.sensitivity.clone())?;
        record.set_item("deltas", deltas_list(py, &aggregates.deltas)?)?;
        record.set_item("groups", groups_list(py, &aggregates.groups)?)?;
        out.push(record.into_any().unbind());
    }
    Ok(out)
}

fn scenario_dict<'py>(
    py: Python<'py>,
    aggregates: &PointAggregates,
    baseline_counts: &[u64],
) -> PyResult<Bound<'py, PyDict>> {
    let display = &aggregates.display;
    let dict = PyDict::new(py);
    dict.set_item("baseline_mean", display.baseline_mean)?;
    dict.set_item("point_mean", display.point_mean)?;
    dict.set_item("baseline_max", display.baseline_max)?;
    dict.set_item("point_max", display.point_max)?;
    dict.set_item("baseline_counts", baseline_counts.to_vec())?;
    dict.set_item("point_counts", aggregates.point_counts.clone())?;
    let cells: Vec<(u32, u32, u64, Vec<f64>)> = aggregates
        .transitions
        .iter()
        .map(|t| (t.from, t.to, t.quotes, t.changes.clone()))
        .collect();
    dict.set_item("transitions", cells)?;
    Ok(dict)
}

fn movement_dict<'py>(
    py: Python<'py>,
    aggregates: &PointAggregates,
    display_given: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let movement = &aggregates.movement;
    let dict = PyDict::new(py);
    dict.set_item("up", movement.up)?;
    dict.set_item("down", movement.down)?;
    dict.set_item("unchanged", movement.unchanged)?;
    dict.set_item("changes", movement.changes.clone())?;
    dict.set_item("counts", movement.counts.clone())?;
    dict.set_item("percentiles_steps", movement.percentiles.to_vec())?;
    dict.set_item(
        "percentiles_display",
        display_given.then(|| aggregates.display.percentiles.to_vec()),
    )?;
    dict.set_item("display_moved_up", aggregates.display.moved_up)?;
    dict.set_item("display_moved_down", aggregates.display.moved_down)?;
    Ok(dict)
}

fn deltas_list<'py>(py: Python<'py>, deltas: &[DeltaSummary]) -> PyResult<Vec<Bound<'py, PyDict>>> {
    deltas
        .iter()
        .map(|delta| {
            let dict = PyDict::new(py);
            dict.set_item("percentiles", delta.percentiles.to_vec())?;
            dict.set_item("edges", delta.edges.clone())?;
            dict.set_item("counts", delta.counts.clone())?;
            dict.set_item("below", delta.below)?;
            dict.set_item("above", delta.above)?;
            dict.set_item("moved_up", delta.moved_up)?;
            dict.set_item("moved_down", delta.moved_down)?;
            Ok(dict)
        })
        .collect()
}

fn groups_list<'py>(py: Python<'py>, groups: &[GroupImpacts]) -> PyResult<Vec<Bound<'py, PyDict>>> {
    groups
        .iter()
        .map(|group| {
            let split = |flat: &[f64]| -> Vec<Vec<f64>> {
                flat.chunks(group.n_columns).map(<[f64]>::to_vec).collect()
            };
            let dict = PyDict::new(py);
            dict.set_item("n_quotes", group.n_quotes.clone())?;
            dict.set_item("step_change_sum", group.step_change_sum.clone())?;
            dict.set_item("baseline", split(&group.baseline))?;
            dict.set_item("point", split(&group.point))?;
            dict.set_item("display_baseline_sum", group.display_baseline_sum.clone())?;
            dict.set_item("display_point_sum", group.display_point_sum.clone())?;
            dict.set_item("abs_change_sum", group.abs_change_sum.clone())?;
            dict.set_item("abs_change_max", group.abs_change_max.clone())?;
            dict.set_item("n_beyond_band", group.n_beyond_band.clone())?;
            dict.set_item("n_at_first", group.n_at_first.clone())?;
            dict.set_item("n_at_last", group.n_at_last.clone())?;
            Ok(dict)
        })
        .collect()
}
