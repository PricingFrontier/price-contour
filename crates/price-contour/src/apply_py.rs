use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{apply_lambdas, ApplyResult, QuoteGrid};

use crate::grid_py::PyQuoteGrid;
use crate::solver_py::{build_result_dataframe, ingest_dataframe, parse_constraints};
use crate::utils::{order_lambdas, zip_to_dict};

/// Python-visible apply result.
#[pyclass(name = "ApplyResult")]
pub struct PyApplyResult {
    inner: ApplyResult,
    grid: Arc<QuoteGrid>,
    constraint_names: Vec<String>,
    result_df: Option<Py<PyAny>>,
}

#[pymethods]
impl PyApplyResult {
    #[getter]
    fn lambdas(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.lambdas)
    }

    #[getter]
    fn total_objective(&self) -> f64 {
        self.inner.total_objective
    }

    #[getter]
    fn total_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.total_constraints)
    }

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.inner.baseline_objective
    }

    #[getter]
    fn baseline_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.baseline_constraints)
    }

    #[getter]
    fn dataframe(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(ref cached) = self.result_df {
            return Ok(cached.clone_ref(py));
        }
        let df = build_result_dataframe(&self.inner.optimal_steps, &self.grid)?;
        let py_df = PyDataFrame(df).into_pyobject(py)?.into();
        self.result_df = Some(py_df);
        Ok(self.result_df.as_ref().unwrap().clone_ref(py))
    }
}

#[pyfunction]
#[pyo3(signature = (
    df,
    lambdas,
    quote_id = "quote_id",
    scenario_index = "scenario_index",
    scenario_value = "scenario_value",
    objective = "expected_income",
    constraints = None,
    chunk_size = 500_000,
))]
#[allow(clippy::too_many_arguments)]
pub fn apply_lambdas_py(
    py: Python<'_>,
    df: PyDataFrame,
    lambdas: HashMap<String, f64>,
    quote_id: &str,
    scenario_index: &str,
    scenario_value: &str,
    objective: &str,
    constraints: Option<HashMap<String, HashMap<String, f64>>>,
    chunk_size: usize,
) -> PyResult<PyApplyResult> {
    let constraints = constraints.unwrap_or_default();
    let mut constraint_cols: Vec<String> = constraints.keys().cloned().collect();
    constraint_cols.sort();

    let grid = Arc::new(ingest_dataframe(
        &df.0,
        quote_id,
        scenario_index,
        scenario_value,
        objective,
        &constraint_cols,
    )?);

    let specs = parse_constraints(constraints, &grid)?;
    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    // Order lambdas to match specs
    let lambda_vec = order_lambdas(&lambdas, &constraint_names);

    let result = py
        .detach(|| apply_lambdas(&grid, &specs, &lambda_vec, Some(chunk_size)))
        .map_err(|e| PyValueError::new_err(format!("Apply error: {e}")))?;

    Ok(PyApplyResult {
        inner: result,
        grid,
        constraint_names,
        result_df: None,
    })
}

/// Single-pass Lagrangian apply on an existing QuoteGrid (no re-ingestion).
///
/// This avoids re-building the grid from a DataFrame — useful when the grid
/// is already in memory (e.g. after a `solve()` or `frontier()` call).
#[pyfunction]
#[pyo3(signature = (grid, lambdas, constraints, chunk_size = 500_000))]
pub fn apply_from_grid_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    lambdas: HashMap<String, f64>,
    constraints: HashMap<String, HashMap<String, f64>>,
    chunk_size: usize,
) -> PyResult<PyApplyResult> {
    let specs = parse_constraints(constraints, &grid.inner)?;
    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    let lambda_vec = order_lambdas(&lambdas, &constraint_names);

    let grid_arc = Arc::clone(&grid.inner);
    let result = py
        .detach(|| apply_lambdas(&grid_arc, &specs, &lambda_vec, Some(chunk_size)))
        .map_err(|e| PyValueError::new_err(format!("Apply error: {e}")))?;

    Ok(PyApplyResult {
        inner: result,
        grid: Arc::clone(&grid.inner),
        constraint_names,
        result_df: None,
    })
}
