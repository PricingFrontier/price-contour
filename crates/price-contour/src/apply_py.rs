use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{apply_lambdas, ApplyResult, QuoteGrid};

use crate::grid_py::PyQuoteGrid;
use crate::solver_py::{build_result_dataframe, ingest_dataframe, parse_constraints};

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
        self.constraint_names
            .iter()
            .zip(self.inner.lambdas.iter())
            .map(|(name, &lam)| (name.clone(), lam))
            .collect()
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
pub fn apply_lambdas_py(
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
    let constraint_cols: Vec<String> = constraints.keys().cloned().collect();

    let grid = Arc::new(ingest_dataframe(
        &df.0,
        quote_id,
        scenario_index,
        scenario_value,
        objective,
        &constraint_cols,
    )?);

    let specs = parse_constraints(constraints, &grid)?;

    // Order lambdas to match specs
    let lambda_vec: Vec<f64> = specs
        .iter()
        .map(|spec| *lambdas.get(&spec.name).unwrap_or(&0.0))
        .collect();

    let result = apply_lambdas(&grid, &specs, &lambda_vec, Some(chunk_size))
        .map_err(|e| PyValueError::new_err(format!("Apply error: {e}")))?;

    let constraint_names = specs.iter().map(|s| s.name.clone()).collect();

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
    grid: &PyQuoteGrid,
    lambdas: HashMap<String, f64>,
    constraints: HashMap<String, HashMap<String, f64>>,
    chunk_size: usize,
) -> PyResult<PyApplyResult> {
    let specs = parse_constraints(constraints, &grid.inner)?;

    let lambda_vec: Vec<f64> = specs
        .iter()
        .map(|spec| *lambdas.get(&spec.name).unwrap_or(&0.0))
        .collect();

    let result = apply_lambdas(&grid.inner, &specs, &lambda_vec, Some(chunk_size))
        .map_err(|e| PyValueError::new_err(format!("Apply error: {e}")))?;

    let constraint_names = specs.iter().map(|s| s.name.clone()).collect();

    Ok(PyApplyResult {
        inner: result,
        grid: Arc::clone(&grid.inner),
        constraint_names,
        result_df: None,
    })
}
