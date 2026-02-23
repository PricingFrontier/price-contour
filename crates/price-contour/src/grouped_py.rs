use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use price_contour_core::{
    build_group_mapping, solve_grouped, ConstraintDirection, ConstraintSpec, GroupedSolveResult,
    SolverConfig,
};

use crate::grid_py::PyQuoteGrid;

/// Python-visible grouped solve result.
#[pyclass(name = "GroupedSolveResult")]
pub struct PyGroupedSolveResult {
    inner: GroupedSolveResult,
    constraint_names: Vec<String>,
    group_labels: Vec<String>,
}

#[pymethods]
impl PyGroupedSolveResult {
    #[getter]
    fn optimal_factor_values(&self) -> HashMap<String, f32> {
        self.group_labels
            .iter()
            .zip(self.inner.optimal_factor_values.iter())
            .map(|(label, &val)| (label.clone(), val))
            .collect()
    }

    #[getter]
    fn optimal_steps_per_quote(&self) -> Vec<u32> {
        self.inner.optimal_steps_per_quote.clone()
    }

    #[getter]
    fn lambdas(&self) -> HashMap<String, f64> {
        self.constraint_names
            .iter()
            .zip(self.inner.lambdas.iter())
            .map(|(name, &lam)| (name.clone(), lam))
            .collect()
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn total_objective(&self) -> f64 {
        self.inner.total_objective
    }

    #[getter]
    fn total_constraints(&self) -> HashMap<String, f64> {
        self.constraint_names
            .iter()
            .zip(self.inner.total_constraints.iter())
            .map(|(name, &val)| (name.clone(), val))
            .collect()
    }

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.inner.baseline_objective
    }

    #[getter]
    fn baseline_constraints(&self) -> HashMap<String, f64> {
        self.constraint_names
            .iter()
            .zip(self.inner.baseline_constraints.iter())
            .map(|(name, &val)| (name.clone(), val))
            .collect()
    }

    #[getter]
    fn clamp_rate(&self) -> f32 {
        self.inner.clamp_rate
    }

    #[getter]
    fn group_labels(&self) -> Vec<String> {
        self.group_labels.clone()
    }
}

/// Parse constraints for grouped solve. Supports the same dict format as online,
/// but uses the grid's baseline totals for relative thresholds.
fn parse_grouped_constraints(
    constraints: &HashMap<String, HashMap<String, f64>>,
    grid: &price_contour_core::QuoteGrid,
) -> PyResult<Vec<ConstraintSpec>> {
    let (_, baseline_totals) = grid.baseline_totals();

    let mut specs = Vec::new();
    for (name, spec_dict) in constraints {
        let constraint_idx = grid
            .constraint_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Constraint '{}' not found. Available: {:?}",
                    name, grid.constraint_names
                ))
            })?;

        if let Some(&frac) = spec_dict.get("min") {
            specs.push(ConstraintSpec {
                name: name.clone(),
                direction: ConstraintDirection::Min,
                threshold: baseline_totals[constraint_idx] * frac,
            });
        } else if let Some(&frac) = spec_dict.get("max") {
            specs.push(ConstraintSpec {
                name: name.clone(),
                direction: ConstraintDirection::Max,
                threshold: baseline_totals[constraint_idx] * frac,
            });
        } else if let Some(&abs_val) = spec_dict.get("min_abs") {
            specs.push(ConstraintSpec {
                name: name.clone(),
                direction: ConstraintDirection::Min,
                threshold: abs_val,
            });
        } else if let Some(&abs_val) = spec_dict.get("max_abs") {
            specs.push(ConstraintSpec {
                name: name.clone(),
                direction: ConstraintDirection::Max,
                threshold: abs_val,
            });
        } else {
            return Err(PyValueError::new_err(format!(
                "Constraint '{}' must specify one of: min, max, min_abs, max_abs",
                name
            )));
        }
    }
    Ok(specs)
}

#[pyfunction]
#[pyo3(signature = (
    grid,
    group_labels,
    residuals,
    candidates,
    constraints = None,
    max_iter = 50,
    chunk_size = 500_000,
    tolerance = 1e-6,
    lambdas = None,
    record_history = false,
))]
pub fn solve_grouped_py(
    grid: &PyQuoteGrid,
    group_labels: Vec<String>,
    residuals: Vec<f32>,
    candidates: Vec<f32>,
    constraints: Option<HashMap<String, HashMap<String, f64>>>,
    max_iter: usize,
    chunk_size: usize,
    tolerance: f64,
    lambdas: Option<HashMap<String, f64>>,
    record_history: bool,
) -> PyResult<PyGroupedSolveResult> {
    let constraints = constraints.unwrap_or_default();

    if group_labels.len() != grid.inner.n_quotes {
        return Err(PyValueError::new_err(format!(
            "group_labels length {} != n_quotes {}",
            group_labels.len(),
            grid.inner.n_quotes
        )));
    }
    if residuals.len() != grid.inner.n_quotes {
        return Err(PyValueError::new_err(format!(
            "residuals length {} != n_quotes {}",
            residuals.len(),
            grid.inner.n_quotes
        )));
    }

    let group_mapping = build_group_mapping(&group_labels);
    let specs = parse_grouped_constraints(&constraints, &grid.inner)?;

    let config = SolverConfig {
        max_iter,
        chunk_size,
        tolerance,
        record_history,
        ..Default::default()
    };

    let initial_lambdas: Option<Vec<f64>> = lambdas.map(|lam_dict| {
        specs
            .iter()
            .map(|spec| *lam_dict.get(&spec.name).unwrap_or(&0.0))
            .collect()
    });

    let result_group_labels = group_mapping.group_labels.clone();

    let result = solve_grouped(
        &grid.inner,
        &group_mapping,
        &residuals,
        &candidates,
        &specs,
        &config,
        initial_lambdas.as_deref(),
    )
    .map_err(|e| PyValueError::new_err(format!("Grouped solver error: {e}")))?;

    let constraint_names = specs.iter().map(|s| s.name.clone()).collect();

    Ok(PyGroupedSolveResult {
        inner: result,
        constraint_names,
        group_labels: result_group_labels,
    })
}
