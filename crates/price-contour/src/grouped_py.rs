use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{
    build_group_mapping, solve_grouped, GroupedSolveResult, QuoteGrid, SolverConfig,
};

use crate::grid_py::PyQuoteGrid;
use crate::solver_py::{build_result_dataframe, parse_constraints};
use crate::utils::{order_lambdas, zip_to_dict};

/// Python-visible grouped solve result.
#[pyclass(name = "GroupedSolveResult")]
pub struct PyGroupedSolveResult {
    inner: GroupedSolveResult,
    grid: Arc<QuoteGrid>,
    constraint_names: Vec<String>,
    group_labels: Vec<String>,
    result_df: Option<Py<PyAny>>,
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
        zip_to_dict(&self.constraint_names, &self.inner.lambdas)
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
    fn clamp_rate(&self) -> f32 {
        self.inner.clamp_rate
    }

    #[getter]
    fn group_labels(&self) -> Vec<String> {
        self.group_labels.clone()
    }

    #[getter]
    fn history(&self, py: Python) -> Option<Vec<HashMap<String, Py<PyAny>>>> {
        self.inner.history.as_ref().map(|h| {
            h.records
                .iter()
                .map(|rec| {
                    let mut d: HashMap<String, Py<PyAny>> = HashMap::new();
                    d.insert(
                        "iteration".into(),
                        rec.iteration.into_pyobject(py).unwrap().unbind().into(),
                    );
                    d.insert(
                        "total_objective".into(),
                        rec.total_objective
                            .into_pyobject(py)
                            .unwrap()
                            .unbind()
                            .into(),
                    );
                    d.insert(
                        "max_lambda_change".into(),
                        rec.max_lambda_change
                            .into_pyobject(py)
                            .unwrap()
                            .unbind()
                            .into(),
                    );
                    d.insert(
                        "all_constraints_satisfied".into(),
                        rec.all_constraints_satisfied
                            .into_pyobject(py)
                            .unwrap()
                            .to_owned()
                            .unbind()
                            .into(),
                    );

                    let lam_dict = zip_to_dict(&self.constraint_names, &rec.lambdas);
                    d.insert(
                        "lambdas".into(),
                        lam_dict.into_pyobject(py).unwrap().unbind().into(),
                    );

                    let con_dict = zip_to_dict(&self.constraint_names, &rec.total_constraints);
                    d.insert(
                        "total_constraints".into(),
                        con_dict.into_pyobject(py).unwrap().unbind().into(),
                    );

                    d
                })
                .collect()
        })
    }

    #[getter]
    fn dataframe(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(ref cached) = self.result_df {
            return Ok(cached.clone_ref(py));
        }
        let df = build_result_dataframe(&self.inner.optimal_steps_per_quote, &self.grid)?;
        let py_df = PyDataFrame(df).into_pyobject(py)?.into();
        self.result_df = Some(py_df);
        Ok(self.result_df.as_ref().unwrap().clone_ref(py))
    }
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
    tolerance = 1e-5,
    lambdas = None,
    record_history = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_grouped_py(
    py: Python<'_>,
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
    let specs = parse_constraints(constraints, &grid.inner)?;

    let config = SolverConfig {
        max_iter,
        chunk_size,
        tolerance,
        record_history,
        ..Default::default()
    };

    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    let initial_lambdas: Option<Vec<f64>> =
        lambdas.map(|lam_dict| order_lambdas(&lam_dict, &constraint_names));

    let result_group_labels = group_mapping.group_labels.clone();

    let grid_arc = Arc::clone(&grid.inner);
    let result = py
        .detach(|| {
            solve_grouped(
                &grid_arc,
                &group_mapping,
                &residuals,
                &candidates,
                &specs,
                &config,
                initial_lambdas.as_deref(),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("Grouped solver error: {e}")))?;

    Ok(PyGroupedSolveResult {
        inner: result,
        grid: grid_arc,
        constraint_names,
        group_labels: result_group_labels,
        result_df: None,
    })
}
