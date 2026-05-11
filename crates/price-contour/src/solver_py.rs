use std::collections::HashMap;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{
    fingerprint_quote_ids, solve_online, ConstraintDirection, ConstraintSpec, QuoteGrid,
    SolveResult, SolverConfig,
};

use crate::constraint_parsing::validate_constraints_dict;
use crate::grid_py::PyQuoteGrid;
use crate::quote_id::quote_id_str_iter;
use crate::utils::{order_lambdas, zip_to_dict};

/// Check if a DataFrame is already sorted by (col1, col2) where col1 is a
/// quote_id-shaped column (Utf8 or Categorical) and col2 is Int32. Returns
/// false on any error or null values. O(n) scan — much cheaper than the
/// O(n log n) sort it can skip.
pub(crate) fn is_df_sorted(df: &DataFrame, col1: &str, col2: &str) -> bool {
    let n = df.height();
    if n <= 1 {
        return true;
    }

    let (Ok(c1), Ok(s2)) = (df.column(col1), df.column(col2)) else {
        return false;
    };
    let Ok(mut iter1) = quote_id_str_iter(c1, col1) else {
        return false;
    };
    let Ok(ca2) = s2.i32() else {
        return false;
    };

    // Owned `String` carry-over: comparing successive iterator items
    // through a `Box<dyn Iterator>` is awkward to spell against the
    // borrow checker, so we stage `prev1` as an owned String and reuse
    // its capacity via `clear() + push_str()`. Total allocation cost is
    // O(n_distinct_quote_ids) at worst (one buffer growth each time the
    // new id is longer than the existing capacity); for fixed-width id
    // schemes (UUIDs, padded sequence numbers) it's a single allocation.
    // The sorted-input fast path is the common one and bails on the
    // first mismatch.
    let Some(Some(first1)) = iter1.next() else {
        return false;
    };
    let Some(first2) = ca2.get(0) else {
        return false;
    };
    let mut prev1 = first1.to_string();
    let mut prev2 = first2;

    for i in 1..n {
        let Some(Some(curr1)) = iter1.next() else {
            return false;
        };
        let Some(curr2) = ca2.get(i) else {
            return false;
        };
        match prev1.as_str().cmp(curr1) {
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Less => {
                prev1.clear();
                prev1.push_str(curr1);
                prev2 = curr2;
            }
            std::cmp::Ordering::Equal => {
                if prev2 > curr2 {
                    return false;
                }
                prev2 = curr2;
            }
        }
    }
    true
}

/// Ingest a PyDataFrame, validate columns, sort, and build a QuoteGrid.
pub(crate) fn ingest_dataframe(
    df: &DataFrame,
    quote_id_col: &str,
    scenario_index_col: &str,
    scenario_value_col: &str,
    objective_col: &str,
    constraint_cols: &[String],
) -> PyResult<QuoteGrid> {
    // Skip sort if data is already in (quote_id, scenario_index) order.
    // O(n) check vs O(n log n) sort — common case for scored DataFrames.
    let df = if is_df_sorted(df, quote_id_col, scenario_index_col) {
        df.clone()
    } else {
        df.sort(
            [quote_id_col, scenario_index_col],
            SortMultipleOptions::default(),
        )
        .map_err(|e| PyValueError::new_err(format!("Sort failed: {e}")))?
    };

    // Extract scenario_value grid from first quote's steps
    let n_rows = df.height();
    if n_rows == 0 {
        return Err(PyValueError::new_err("Empty DataFrame"));
    }
    let step_series = df
        .column(scenario_index_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_index_col}")))?;
    let steps_ca = step_series
        .i32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_index_col} must be Int32")))?;

    // Count unique quotes and determine n_steps. Iterates the quote_id
    // column once via the shared accessor so the one-shot path supports
    // both Utf8 and Categorical inputs identically.
    let qid_col = df
        .column(quote_id_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {quote_id_col}")))?;

    // Determine n_steps from the first quote: count how many leading rows
    // share the same string as row 0.
    let n_steps: usize = {
        let mut probe_iter = quote_id_str_iter(qid_col, quote_id_col)?;
        let first_qid = probe_iter
            .next()
            .ok_or_else(|| PyValueError::new_err("Empty DataFrame"))?
            .ok_or_else(|| PyValueError::new_err("Empty DataFrame"))?
            .to_string();
        let mut count: usize = 1;
        for next_opt in probe_iter {
            match next_opt {
                Some(s) if s == first_qid.as_str() => count += 1,
                _ => break,
            }
        }
        count
    };
    if n_steps == 0 {
        return Err(PyValueError::new_err("Could not determine n_steps"));
    }
    if n_rows % n_steps != 0 {
        return Err(PyValueError::new_err(format!(
            "Row count {n_rows} not divisible by n_steps {n_steps}"
        )));
    }
    let n_quotes = n_rows / n_steps;

    // Extract scenario_value grid (from first quote's rows)
    let mult_series = df
        .column(scenario_value_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_value_col}")))?;
    let mult_ca = mult_series
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_value_col} must be Float32")))?;
    let scenario_values: Vec<f32> = (0..n_steps)
        .map(|i| {
            mult_ca.get(i).ok_or_else(|| {
                PyValueError::new_err(format!("Null {scenario_value_col} at row {i}"))
            })
        })
        .collect::<PyResult<Vec<f32>>>()?;

    // Extract objective as flat Vec<f32>
    let obj_series = df
        .column(objective_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {objective_col}")))?;
    let obj_ca = obj_series
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{objective_col} must be Float32")))?;
    if obj_ca.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Column '{}' contains null values",
            objective_col
        )));
    }
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
        if ca.null_count() > 0 {
            return Err(PyValueError::new_err(format!(
                "Column '{}' contains null values",
                col_name
            )));
        }
        constraints.push(ca.into_no_null_iter().collect());
    }

    // Extract quote_ids: one per quote (the head of each n_steps block).
    // Iterating the column once and skipping n_steps - 1 rows after each
    // capture is identical-cost for Utf8 and avoids the random-access path
    // that Categorical would otherwise pay (per-row dict lookup × n_rows).
    let mut quote_ids: Vec<String> = Vec::with_capacity(n_quotes);
    {
        let mut iter = quote_id_str_iter(qid_col, quote_id_col)?;
        for q in 0..n_quotes {
            let row = q * n_steps;
            let qid = iter
                .next()
                .ok_or_else(|| PyValueError::new_err(format!("iter exhausted at row {row}")))?
                .ok_or_else(|| PyValueError::new_err(format!("Null {quote_id_col} at row {row}")))?
                .to_string();
            quote_ids.push(qid);
            // Skip the remaining n_steps - 1 rows of this quote.
            for _ in 1..n_steps {
                let _ = iter.next();
            }
        }
    }

    // Validate step sequence (first 10 quotes only, for performance on large grids)
    for (q, qid) in quote_ids.iter().enumerate().take(n_quotes.min(10)) {
        for j in 0..n_steps {
            let idx = q * n_steps + j;
            let step_val = steps_ca.get(idx).unwrap_or(-1);
            if step_val != j as i32 {
                return Err(PyValueError::new_err(format!(
                    "Quote {qid} step {j} has scenario_index={step_val}, expected {j}",
                )));
            }
        }
    }

    let quote_id_fingerprint = fingerprint_quote_ids(&quote_ids);
    let grid = QuoteGrid {
        n_quotes,
        n_steps,
        scenario_values,
        objective,
        constraints,
        constraint_names: constraint_cols.to_vec(),
        quote_ids,
        quote_id_fingerprint,
    };
    grid.validate()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(grid)
}

/// Build result DataFrame from optimal_steps + QuoteGrid.
///
/// Shared by both `PySolveResult.dataframe` and `PyApplyResult.dataframe`.
pub(crate) fn build_result_dataframe(
    optimal_steps: &[u32],
    grid: &QuoteGrid,
) -> PyResult<DataFrame> {
    let n = grid.n_quotes;
    let m = grid.n_steps;

    let mut opt_scenario_values = Vec::with_capacity(n);
    let mut opt_objectives = Vec::with_capacity(n);
    let mut opt_constraint_vals: Vec<Vec<f32>> =
        vec![Vec::with_capacity(n); grid.constraint_names.len()];

    for (q, &opt_step) in optimal_steps.iter().enumerate().take(n) {
        let step = opt_step as usize;
        let idx = q * m + step;
        opt_scenario_values.push(grid.scenario_values[step]);
        opt_objectives.push(grid.objective[idx]);
        for (k, con) in grid.constraints.iter().enumerate() {
            opt_constraint_vals[k].push(con[idx]);
        }
    }

    let mut columns: Vec<Column> = vec![
        Column::new("quote_id".into(), &grid.quote_ids),
        Column::new(
            "optimal_step".into(),
            optimal_steps
                .iter()
                .map(|&s| s as i32)
                .collect::<Vec<i32>>(),
        ),
        Column::new("optimal_scenario_value".into(), &opt_scenario_values),
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

/// Python-visible solve result wrapping the core SolveResult.
#[pyclass(name = "SolveResult")]
pub struct PySolveResult {
    inner: SolveResult,
    grid: Arc<QuoteGrid>,
    constraint_names: Vec<String>,
    result_df: Option<Py<PyAny>>,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn lambdas(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.lambdas)
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
        zip_to_dict(&self.constraint_names, &self.inner.total_constraints)
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

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.inner.baseline_objective
    }

    #[getter]
    fn baseline_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.baseline_constraints)
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
    fn scenario_values(&self) -> Vec<f32> {
        self.grid.scenario_values.clone()
    }

    #[getter]
    fn n_quotes(&self) -> usize {
        self.grid.n_quotes
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.grid.n_steps
    }

    /// Return the underlying QuoteGrid (Arc-shared, zero-copy).
    #[getter]
    fn grid(&self) -> PyQuoteGrid {
        PyQuoteGrid {
            inner: Arc::clone(&self.grid),
        }
    }
}

/// Parse constraint dict from Python.
///
/// Direction keys:
/// * ``min`` / ``max``         → absolute thresholds.
/// * ``min_pct`` / ``max_pct`` → fraction-of-baseline thresholds.
///
/// We walk `grid.constraint_names` (NOT the user `HashMap`) when
/// emitting specs so `specs[k]` aligns with `grid.constraints[k]` —
/// the inner solver in `argmax.rs` indexes by position. Iterating the
/// `HashMap` directly would produce nondeterministic ordering and
/// silently mix up which constraint each lambda controls.
///
/// Validation (multi-key, NaN/inf, unknown name) is
/// shared with `frontier_py::sweep_frontier_py` via
/// `crate::constraint_parsing::validate_constraints_dict` so the two
/// entry points can never drift.
pub(crate) fn parse_constraints(
    constraints: HashMap<String, HashMap<String, Option<f64>>>,
    grid: &QuoteGrid,
) -> PyResult<Vec<ConstraintSpec>> {
    let (_, baseline_totals) = grid.baseline_totals();

    validate_constraints_dict(&constraints, grid)?;

    // Walk grid order so spec[k] aligns with grid.constraints[k].
    // Extension point: ratio constraints (C1) detect via `numerator` /
    // `denominator` keys before the direction-key match below.
    //
    // ``None`` values are frontier-only markers (B1); the from-grid solve
    // path lands here, and solve() cannot pick a threshold from thin air,
    // so we raise a ``ValueError`` that names the constraint and points
    // the user at ``frontier()``. The Python ``OnlineOptimiser.solve()``
    // shim catches this earlier for the DataFrame path; this is the
    // belt-and-braces backstop for callers that bypass it.
    let mut specs = Vec::with_capacity(constraints.len());
    for (constraint_idx, name) in grid.constraint_names.iter().enumerate() {
        let Some(spec_dict) = constraints.get(name) else {
            continue;
        };

        let (direction, raw_threshold, is_pct) = if let Some(value_opt) = spec_dict.get("min") {
            (ConstraintDirection::Min, value_opt, false)
        } else if let Some(value_opt) = spec_dict.get("max") {
            (ConstraintDirection::Max, value_opt, false)
        } else if let Some(value_opt) = spec_dict.get("min_pct") {
            (ConstraintDirection::Min, value_opt, true)
        } else if let Some(value_opt) = spec_dict.get("max_pct") {
            (ConstraintDirection::Max, value_opt, true)
        } else {
            // validate_constraints_dict already requires exactly one
            // direction key, so this branch is unreachable.
            continue;
        };

        let Some(value) = raw_threshold else {
            return Err(PyValueError::new_err(format!(
                "Constraint '{}' has no threshold (value is None); \
                 solve() requires a numeric threshold per constraint. \
                 Use frontier() with a matching threshold_ranges entry to sweep \
                 this constraint, or supply a numeric threshold.",
                name
            )));
        };

        let threshold = if is_pct {
            baseline_totals[constraint_idx] * *value
        } else {
            *value
        };

        specs.push(ConstraintSpec {
            name: name.clone(),
            direction,
            threshold,
        });
    }
    Ok(specs)
}

#[pyfunction]
#[pyo3(signature = (
    df,
    quote_id = "quote_id",
    scenario_index = "scenario_index",
    scenario_value = "scenario_value",
    objective = "expected_income",
    constraints = None,
    max_iter = 50,
    tolerance = 1e-5,
    lambdas = None,
    record_history = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_online_py(
    py: Python<'_>,
    df: PyDataFrame,
    quote_id: &str,
    scenario_index: &str,
    scenario_value: &str,
    objective: &str,
    constraints: Option<HashMap<String, HashMap<String, Option<f64>>>>,
    max_iter: usize,
    tolerance: f64,
    lambdas: Option<HashMap<String, f64>>,
    record_history: bool,
) -> PyResult<PySolveResult> {
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

    let config = SolverConfig {
        max_iter,
        tolerance,
        record_history,
        ..Default::default()
    };

    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    // Build initial_lambdas from the dict, ordered to match specs
    let initial_lambdas: Option<Vec<f64>> =
        lambdas.map(|lam_dict| order_lambdas(&lam_dict, &constraint_names));

    let result = py
        .detach(|| solve_online(&grid, &specs, &config, initial_lambdas.as_deref()))
        .map_err(|e| PyValueError::new_err(format!("Solver error: {e}")))?;

    Ok(PySolveResult {
        inner: result,
        grid,
        constraint_names,
        result_df: None,
    })
}

/// Solve from a pre-built QuoteGrid (skips DataFrame ingestion).
#[pyfunction]
#[pyo3(signature = (
    grid,
    constraints = None,
    max_iter = 50,
    tolerance = 1e-5,
    lambdas = None,
    record_history = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_from_grid_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    constraints: Option<HashMap<String, HashMap<String, Option<f64>>>>,
    max_iter: usize,
    tolerance: f64,
    lambdas: Option<HashMap<String, f64>>,
    record_history: bool,
) -> PyResult<PySolveResult> {
    let constraints = constraints.unwrap_or_default();

    let specs = parse_constraints(constraints, &grid.inner)?;

    let config = SolverConfig {
        max_iter,
        tolerance,
        record_history,
        ..Default::default()
    };

    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    let initial_lambdas: Option<Vec<f64>> =
        lambdas.map(|lam_dict| order_lambdas(&lam_dict, &constraint_names));

    let grid_arc = Arc::clone(&grid.inner);
    let result = py
        .detach(|| solve_online(&grid_arc, &specs, &config, initial_lambdas.as_deref()))
        .map_err(|e| PyValueError::new_err(format!("Solver error: {e}")))?;

    Ok(PySolveResult {
        inner: result,
        grid: Arc::clone(&grid.inner),
        constraint_names,
        result_df: None,
    })
}
