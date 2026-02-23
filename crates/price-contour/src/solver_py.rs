use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{
    ConstraintDirection, ConstraintSpec, QuoteGrid, SolveResult, SolverConfig, solve_online,
};

/// Ingest a PyDataFrame, validate columns, sort, and build a QuoteGrid.
pub(crate) fn ingest_dataframe(
    df: &DataFrame,
    quote_id_col: &str,
    scenario_step_col: &str,
    multiplier_col: &str,
    objective_col: &str,
    constraint_cols: &[String],
) -> PyResult<QuoteGrid> {
    // Sort by (quote_id, scenario_step)
    let df = df
        .sort(
            [quote_id_col, scenario_step_col],
            SortMultipleOptions::default(),
        )
        .map_err(|e| PyValueError::new_err(format!("Sort failed: {e}")))?;

    // Extract multiplier grid from first quote's steps
    let n_rows = df.height();
    let step_series = df
        .column(scenario_step_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_step_col}")))?;
    let steps_ca = step_series
        .i32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_step_col} must be Int32")))?;

    // Count unique quotes and determine n_steps
    let qid_series = df
        .column(quote_id_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {quote_id_col}")))?;
    let qid_ca = qid_series
        .str()
        .map_err(|_| PyValueError::new_err(format!("{quote_id_col} must be Utf8")))?;

    // Determine n_steps from first quote
    let first_qid = qid_ca
        .get(0)
        .ok_or_else(|| PyValueError::new_err("Empty DataFrame"))?;
    let mut n_steps: usize = 0;
    for i in 0..n_rows {
        if qid_ca.get(i) == Some(first_qid) {
            n_steps += 1;
        } else {
            break;
        }
    }
    if n_steps == 0 {
        return Err(PyValueError::new_err("Could not determine n_steps"));
    }
    if n_rows % n_steps != 0 {
        return Err(PyValueError::new_err(format!(
            "Row count {n_rows} not divisible by n_steps {n_steps}"
        )));
    }
    let n_quotes = n_rows / n_steps;

    // Extract multiplier grid (from first quote's rows)
    let mult_series = df
        .column(multiplier_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {multiplier_col}")))?;
    let mult_ca = mult_series
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{multiplier_col} must be Float32")))?;
    let multipliers: Vec<f32> = (0..n_steps)
        .map(|i| mult_ca.get(i).unwrap_or(0.0))
        .collect();

    // Extract objective as flat Vec<f32>
    let obj_series = df
        .column(objective_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {objective_col}")))?;
    let obj_ca = obj_series
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{objective_col} must be Float32")))?;
    let objective: Vec<f32> = obj_ca.into_no_null_iter().collect();

    // Extract constraints
    let mut constraints: Vec<Vec<f32>> = Vec::with_capacity(constraint_cols.len());
    for col_name in constraint_cols {
        let series = df
            .column(col_name)
            .map_err(|_| PyValueError::new_err(format!("Missing column: {col_name}")))?;
        let ca = series
            .f32()
            .map_err(|_| PyValueError::new_err(format!("{col_name} must be Float32")))?;
        constraints.push(ca.into_no_null_iter().collect());
    }

    // Extract quote_ids (one per quote — take every n_steps-th)
    let quote_ids: Vec<String> = (0..n_quotes)
        .map(|q| {
            qid_ca
                .get(q * n_steps)
                .unwrap_or("")
                .to_string()
        })
        .collect();

    // Validate step sequence for each quote
    for q in 0..n_quotes.min(10) {
        for j in 0..n_steps {
            let idx = q * n_steps + j;
            let step_val = steps_ca.get(idx).unwrap_or(-1);
            if step_val != j as i32 {
                return Err(PyValueError::new_err(format!(
                    "Quote {} step {} has scenario_step={}, expected {}",
                    quote_ids[q], j, step_val, j
                )));
            }
        }
    }

    Ok(QuoteGrid {
        n_quotes,
        n_steps,
        multipliers,
        objective,
        constraints,
        constraint_names: constraint_cols.to_vec(),
        quote_ids,
    })
}

/// Build result DataFrame from optimal_steps + QuoteGrid.
pub(crate) fn build_result_dataframe(result: &SolveResult, grid: &QuoteGrid) -> PyResult<DataFrame> {
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

    DataFrame::new(columns).map_err(|e| PyValueError::new_err(format!("DataFrame build failed: {e}")))
}

/// Python-visible solve result wrapping the core SolveResult.
#[pyclass(name = "SolveResult")]
pub struct PySolveResult {
    inner: SolveResult,
    grid: QuoteGrid,
    constraint_names: Vec<String>,
    result_df: Option<Py<PyAny>>,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn lambdas(&self) -> HashMap<String, f64> {
        self.constraint_names
            .iter()
            .zip(self.inner.lambdas.iter())
            .map(|(name, &lam)| (name.clone(), lam))
            .collect()
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
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
    fn dataframe(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(ref cached) = self.result_df {
            return Ok(cached.clone_ref(py));
        }
        let df = build_result_dataframe(&self.inner, &self.grid)?;
        let py_df = PyDataFrame(df).into_pyobject(py)?.into();
        self.result_df = Some(py_df);
        Ok(self.result_df.as_ref().unwrap().clone_ref(py))
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
    fn history(&self, py: Python) -> Option<Vec<HashMap<String, Py<PyAny>>>> {
        self.inner.history.as_ref().map(|h| {
            h.records
                .iter()
                .map(|rec| {
                    let mut d: HashMap<String, Py<PyAny>> = HashMap::new();
                    d.insert("iteration".into(), rec.iteration.into_pyobject(py).unwrap().unbind().into());
                    d.insert("total_objective".into(), rec.total_objective.into_pyobject(py).unwrap().unbind().into());
                    d.insert("max_lambda_change".into(), rec.max_lambda_change.into_pyobject(py).unwrap().unbind().into());
                    d.insert("all_constraints_satisfied".into(), rec.all_constraints_satisfied.into_pyobject(py).unwrap().to_owned().unbind().into());

                    let lam_dict: HashMap<String, f64> = self.constraint_names
                        .iter()
                        .zip(rec.lambdas.iter())
                        .map(|(name, &v)| (name.clone(), v))
                        .collect();
                    d.insert("lambdas".into(), lam_dict.into_pyobject(py).unwrap().unbind().into());

                    let con_dict: HashMap<String, f64> = self.constraint_names
                        .iter()
                        .zip(rec.total_constraints.iter())
                        .map(|(name, &v)| (name.clone(), v))
                        .collect();
                    d.insert("total_constraints".into(), con_dict.into_pyobject(py).unwrap().unbind().into());

                    d
                })
                .collect()
        })
    }

    #[getter]
    fn multipliers(&self) -> Vec<f32> {
        self.grid.multipliers.clone()
    }

    #[getter]
    fn n_quotes(&self) -> usize {
        self.grid.n_quotes
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.grid.n_steps
    }
}

/// Parse constraint dict from Python: {"volume": {"min": 0.9}, "loss_ratio": {"max": 1.05}}
pub(crate) fn parse_constraints(
    constraints: HashMap<String, HashMap<String, f64>>,
    grid: &QuoteGrid,
) -> PyResult<Vec<ConstraintSpec>> {
    let (_, baseline_totals) = grid.baseline_totals();

    let mut specs = Vec::new();
    for (name, spec_dict) in &constraints {
        let constraint_idx = grid
            .constraint_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Constraint '{}' not found in DataFrame columns. Available: {:?}",
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
    df,
    quote_id = "quote_id",
    scenario_step = "scenario_step",
    multiplier = "multiplier",
    objective = "expected_income",
    constraints = None,
    max_iter = 50,
    chunk_size = 500_000,
    tolerance = 1e-6,
    lambdas = None,
    record_history = false,
))]
pub fn solve_online_py(
    df: PyDataFrame,
    quote_id: &str,
    scenario_step: &str,
    multiplier: &str,
    objective: &str,
    constraints: Option<HashMap<String, HashMap<String, f64>>>,
    max_iter: usize,
    chunk_size: usize,
    tolerance: f64,
    lambdas: Option<HashMap<String, f64>>,
    record_history: bool,
) -> PyResult<PySolveResult> {
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

    let config = SolverConfig {
        max_iter,
        chunk_size,
        tolerance,
        record_history,
        ..Default::default()
    };

    // Build initial_lambdas from the dict, ordered to match specs
    let initial_lambdas: Option<Vec<f64>> = lambdas.map(|lam_dict| {
        specs
            .iter()
            .map(|spec| *lam_dict.get(&spec.name).unwrap_or(&0.0))
            .collect()
    });

    let result = solve_online(&grid, &specs, &config, initial_lambdas.as_deref())
        .map_err(|e| PyValueError::new_err(format!("Solver error: {e}")))?;

    let constraint_names = specs.iter().map(|s| s.name.clone()).collect();

    Ok(PySolveResult {
        inner: result,
        grid,
        constraint_names,
        result_df: None,
    })
}
