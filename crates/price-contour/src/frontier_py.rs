use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{
    sweep_frontier, ConstraintDirection, ConstraintSpec, FrontierConfig, FrontierResult,
    SolverConfig,
};

use crate::grid_py::PyQuoteGrid;

/// Python-visible frontier result.
#[pyclass(name = "FrontierResult")]
pub struct PyFrontierResult {
    inner: FrontierResult,
}

#[pymethods]
impl PyFrontierResult {
    /// Return the frontier as a Polars DataFrame with columns:
    /// threshold_*, total_objective, total_*, lambda_*, iterations, converged
    #[getter]
    fn points(&self, py: Python) -> PyResult<Py<PyAny>> {
        let n_constraints = self.inner.constraint_names.len();

        let mut columns: Vec<Column> = Vec::new();

        // Threshold columns
        for (k, name) in self.inner.constraint_names.iter().enumerate() {
            let vals: Vec<f64> = self.inner.points.iter().map(|p| p.thresholds[k]).collect();
            columns.push(Column::new(format!("threshold_{name}").into(), &vals));
        }

        // Total objective
        let obj_vals: Vec<f64> = self.inner.points.iter().map(|p| p.total_objective).collect();
        columns.push(Column::new("total_objective".into(), &obj_vals));

        // Total constraints
        for k in 0..n_constraints {
            let name = &self.inner.constraint_names[k];
            let vals: Vec<f64> = self
                .inner
                .points
                .iter()
                .map(|p| p.total_constraints[k])
                .collect();
            columns.push(Column::new(format!("total_{name}").into(), &vals));
        }

        // Lambda columns
        for k in 0..n_constraints {
            let name = &self.inner.constraint_names[k];
            let vals: Vec<f64> = self.inner.points.iter().map(|p| p.lambdas[k]).collect();
            columns.push(Column::new(format!("lambda_{name}").into(), &vals));
        }

        // Iterations and converged
        let iter_vals: Vec<i64> = self
            .inner
            .points
            .iter()
            .map(|p| p.iterations as i64)
            .collect();
        columns.push(Column::new("iterations".into(), &iter_vals));

        let conv_vals: Vec<bool> = self.inner.points.iter().map(|p| p.converged).collect();
        columns.push(Column::new("converged".into(), &conv_vals));

        let df = DataFrame::new(columns)
            .map_err(|e| PyValueError::new_err(format!("DataFrame build failed: {e}")))?;
        let py_df: Py<PyAny> = PyDataFrame(df).into_pyobject(py)?.into();
        Ok(py_df)
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.points.len()
    }

    #[getter]
    fn constraint_names(&self) -> Vec<String> {
        self.inner.constraint_names.clone()
    }
}

#[pyfunction]
#[pyo3(signature = (
    grid,
    constraints,
    threshold_ranges,
    n_points_per_dim = 10,
    max_iter = 50,
    chunk_size = 500_000,
    tolerance = 1e-6,
))]
pub fn sweep_frontier_py(
    grid: &PyQuoteGrid,
    constraints: HashMap<String, HashMap<String, f64>>,
    threshold_ranges: HashMap<String, (f64, f64)>,
    n_points_per_dim: usize,
    max_iter: usize,
    chunk_size: usize,
    tolerance: f64,
) -> PyResult<PyFrontierResult> {
    let (_, baseline_totals) = grid.inner.baseline_totals();

    // Build specs_template and ranges, ordered consistently
    let mut specs_template = Vec::new();
    let mut ranges = Vec::new();

    for (name, spec_dict) in &constraints {
        let constraint_idx = grid
            .inner
            .constraint_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Constraint '{}' not found. Available: {:?}",
                    name, grid.inner.constraint_names
                ))
            })?;

        let direction = if spec_dict.contains_key("min") || spec_dict.contains_key("min_abs") {
            ConstraintDirection::Min
        } else if spec_dict.contains_key("max") || spec_dict.contains_key("max_abs") {
            ConstraintDirection::Max
        } else {
            return Err(PyValueError::new_err(format!(
                "Constraint '{}' must specify min, max, min_abs, or max_abs",
                name
            )));
        };

        specs_template.push(ConstraintSpec {
            name: name.clone(),
            direction,
            threshold: 0.0, // replaced per frontier point
        });

        // Get threshold range for this constraint
        let (lo, hi) = threshold_ranges.get(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "No threshold_range for constraint '{}'", name
            ))
        })?;

        // Convert relative to absolute if needed
        let (abs_lo, abs_hi) = if spec_dict.contains_key("min") || spec_dict.contains_key("max") {
            (baseline_totals[constraint_idx] * lo, baseline_totals[constraint_idx] * hi)
        } else {
            (*lo, *hi)
        };

        ranges.push((abs_lo, abs_hi));
    }

    let frontier_config = FrontierConfig {
        n_points_per_dim,
        threshold_ranges: ranges,
    };

    let solver_config = SolverConfig {
        max_iter,
        chunk_size,
        tolerance,
        ..Default::default()
    };

    let result = sweep_frontier(&grid.inner, &specs_template, &frontier_config, &solver_config)
        .map_err(|e| PyValueError::new_err(format!("Frontier error: {e}")))?;

    Ok(PyFrontierResult { inner: result })
}
