use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{apply_lambdas, ApplyResult};

use crate::solver_py::{ingest_dataframe, parse_constraints};

/// Python-visible apply result.
#[pyclass(name = "ApplyResult")]
pub struct PyApplyResult {
    inner: ApplyResult,
    grid: price_contour_core::QuoteGrid,
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
        let df = build_apply_result_dataframe(&self.inner, &self.grid)?;
        let py_df = PyDataFrame(df).into_pyobject(py)?.into();
        self.result_df = Some(py_df);
        Ok(self.result_df.as_ref().unwrap().clone_ref(py))
    }
}

/// Build result DataFrame from ApplyResult + QuoteGrid.
fn build_apply_result_dataframe(
    result: &ApplyResult,
    grid: &price_contour_core::QuoteGrid,
) -> PyResult<DataFrame> {
    let n = grid.n_quotes;
    let m = grid.n_steps;

    let mut opt_multipliers = Vec::with_capacity(n);
    let mut opt_objectives = Vec::with_capacity(n);
    let mut opt_constraint_vals: Vec<Vec<f32>> =
        vec![Vec::with_capacity(n); grid.constraint_names.len()];

    for q in 0..n {
        let step = result.optimal_steps[q] as usize;
        let idx = q * m + step;
        opt_multipliers.push(grid.multipliers[step]);
        opt_objectives.push(grid.objective[idx]);
        for (k, con) in grid.constraints.iter().enumerate() {
            opt_constraint_vals[k].push(con[idx]);
        }
    }

    let mut columns: Vec<Column> = vec![
        Column::new("quote_id".into(), &grid.quote_ids),
        Column::new(
            "optimal_step".into(),
            result
                .optimal_steps
                .iter()
                .map(|&s| s as i32)
                .collect::<Vec<i32>>(),
        ),
        Column::new("optimal_multiplier".into(), &opt_multipliers),
        Column::new("optimal_objective".into(), &opt_objectives),
    ];

    for (k, name) in grid.constraint_names.iter().enumerate() {
        columns.push(Column::new(
            format!("optimal_{name}").into(),
            &opt_constraint_vals[k],
        ));
    }

    DataFrame::new(columns)
        .map_err(|e| PyValueError::new_err(format!("DataFrame build failed: {e}")))
}

#[pyfunction]
#[pyo3(signature = (
    df,
    lambdas,
    quote_id = "quote_id",
    scenario_step = "scenario_step",
    multiplier = "multiplier",
    objective = "expected_income",
    constraints = None,
    chunk_size = 500_000,
))]
pub fn apply_lambdas_py(
    df: PyDataFrame,
    lambdas: HashMap<String, f64>,
    quote_id: &str,
    scenario_step: &str,
    multiplier: &str,
    objective: &str,
    constraints: Option<HashMap<String, HashMap<String, f64>>>,
    chunk_size: usize,
) -> PyResult<PyApplyResult> {
    let constraints = constraints.unwrap_or_default();
    let constraint_cols: Vec<String> = constraints.keys().cloned().collect();

    let grid = ingest_dataframe(
        &df.0,
        quote_id,
        scenario_step,
        multiplier,
        objective,
        &constraint_cols,
    )?;

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
