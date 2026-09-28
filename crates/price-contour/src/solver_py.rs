use std::collections::HashMap;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use price_contour_core::constants::{CANCEL_POLL_QUOTES, RECONSTRUCT_PAR_GRAIN};
use price_contour_core::{
    check_cancelled, fingerprint_quote_ids, solve_online, CancelFlag, ConstraintDirection,
    ConstraintSpec, QuoteGrid, SolveResult, SolverConfig,
};

use crate::cancel_py::core_error;
use crate::constraint_parsing::validate_constraints_dict;
use crate::grid_py::PyQuoteGrid;
use crate::quote_id::quote_id_str_iter;
use crate::utils::{order_lambdas, zip_to_dict, OrderedDict};

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
/// Shared by `PySolveResult.dataframe`, `PyApplyResult.dataframe` and the
/// ratebook `quote_results`. Columns are gathered in parallel (order is
/// preserved) into owned vectors that move into Polars without a second copy.
pub(crate) fn build_result_dataframe(
    optimal_steps: &[u32],
    grid: &QuoteGrid,
) -> PyResult<DataFrame> {
    build_result_dataframe_polling(optimal_steps, grid, None)
}

/// [`build_result_dataframe`], polling `cancel` once per block of rows in
/// every column (never once per column), raising `Cancelled` once it is set.
/// Holds no Python state, so callers run it with the GIL released.
pub(crate) fn build_result_dataframe_polling(
    optimal_steps: &[u32],
    grid: &QuoteGrid,
    cancel: Option<&CancelFlag>,
) -> PyResult<DataFrame> {
    let n = grid.n_quotes;
    let m = grid.n_steps;
    if optimal_steps.len() != n {
        return Err(PyValueError::new_err(format!(
            "optimal_steps length {} != n_quotes {n}",
            optimal_steps.len()
        )));
    }
    let check = || check_cancelled(cancel).map_err(|e| core_error("DataFrame build failed", e));

    // Each column is filled in parallel blocks; a block skips its work once
    // the flag is set, and `check` turns that into `Cancelled`.
    fn fill<T: Copy + Default + Send>(
        n: usize,
        cancel: Option<&CancelFlag>,
        f: &(dyn Fn(usize) -> T + Sync),
    ) -> Vec<T> {
        let mut out = vec![T::default(); n];
        out.par_chunks_mut(RECONSTRUCT_PAR_GRAIN)
            .enumerate()
            .for_each(|(block, slice)| {
                if check_cancelled(cancel).is_err() {
                    return;
                }
                let start = block * RECONSTRUCT_PAR_GRAIN;
                for (i, v) in slice.iter_mut().enumerate() {
                    *v = f(start + i);
                }
            });
        out
    }
    let fill_f32 = |f: &(dyn Fn(usize) -> f32 + Sync)| -> PyResult<Vec<f32>> {
        let out = fill(n, cancel, f);
        check()?;
        Ok(out)
    };
    let gather = |values: &[f32]| fill_f32(&|q| values[q * m + optimal_steps[q] as usize]);

    check()?;
    let mut quote_ids = StringChunkedBuilder::new("quote_id".into(), n);
    for block in grid.quote_ids.chunks(CANCEL_POLL_QUOTES) {
        check()?;
        for id in block {
            quote_ids.append_value(id);
        }
    }
    let opt_steps: Vec<i32> = fill(n, cancel, &|q| optimal_steps[q] as i32);
    check()?;
    let opt_scenario_values = fill_f32(&|q| grid.scenario_values[optimal_steps[q] as usize])?;

    let mut columns: Vec<Column> = vec![
        quote_ids.finish().into_column(),
        Int32Chunked::from_vec("optimal_step".into(), opt_steps).into_column(),
        Float32Chunked::from_vec("optimal_scenario_value".into(), opt_scenario_values)
            .into_column(),
        Float32Chunked::from_vec("optimal_objective".into(), gather(&grid.objective)?)
            .into_column(),
    ];
    for (name, values) in grid.constraint_names.iter().zip(grid.constraints.iter()) {
        columns.push(
            Float32Chunked::from_vec(format!("optimal_{name}").into(), gather(values)?)
                .into_column(),
        );
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
    /// Absolute bound of each constraint, in `constraint_names` order.
    constraint_bounds: Vec<f64>,
    result_df: Option<Py<PyAny>>,
}

#[pymethods]
impl PySolveResult {
    #[getter]
    fn lambdas(&self) -> OrderedDict {
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
    fn total_constraints(&self) -> OrderedDict {
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
    fn baseline_constraints(&self) -> OrderedDict {
        zip_to_dict(&self.constraint_names, &self.inner.baseline_constraints)
    }

    /// Absolute bound each constraint was solved against: the threshold for
    /// `min`/`max`, `baseline × fraction` for `min_pct`/`max_pct`.
    #[getter]
    fn constraint_bounds(&self) -> OrderedDict {
        zip_to_dict(&self.constraint_names, &self.constraint_bounds)
    }

    /// Scenario value of the baseline step (nearest 1.0).
    #[getter]
    fn baseline_scenario_value(&self) -> f32 {
        self.grid.scenario_values[self.grid.baseline_step()]
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
    parse_constraints_polling(constraints, grid, None)
}

/// [`parse_constraints`] whose baseline scan (needed only for `min_pct` /
/// `max_pct`) polls `cancel`. Holds no Python state, so callers run it with
/// the GIL released.
pub(crate) fn parse_constraints_polling(
    constraints: HashMap<String, HashMap<String, Option<f64>>>,
    grid: &QuoteGrid,
    cancel: Option<&CancelFlag>,
) -> PyResult<Vec<ConstraintSpec>> {
    validate_constraints_dict(&constraints, grid)?;

    // First pass: every input error, before any O(n_quotes) work, so a
    // malformed call reports its error rather than `Cancelled`.
    let mut parsed: Vec<(usize, &String, ConstraintDirection, f64, bool)> =
        Vec::with_capacity(constraints.len());
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
        parsed.push((constraint_idx, name, direction, *value, is_pct));
    }

    // The baseline scan is O(n_quotes); only a pct bound needs it.
    let baseline_totals = if parsed.iter().any(|&(.., is_pct)| is_pct) {
        match cancel {
            None => grid.baseline_totals().1,
            Some(flag) => {
                grid.baseline_totals_cancellable(flag)
                    .map_err(|e| core_error("Constraint parsing error", e))?
                    .1
            }
        }
    } else {
        Vec::new()
    };

    let mut specs = Vec::with_capacity(parsed.len());
    for (constraint_idx, name, direction, value, is_pct) in parsed {
        let threshold = if is_pct {
            let baseline = baseline_totals[constraint_idx];
            if baseline == 0.0 {
                return Err(PyValueError::new_err(format!(
                    "min_pct/max_pct on '{name}' is undefined: baseline total is 0"
                )));
            }
            baseline * value
        } else {
            value
        };

        specs.push(ConstraintSpec {
            name: name.clone(),
            direction,
            threshold,
        });
    }
    Ok(specs)
}

/// The names of the constraints `constraints` specifies, in grid order —
/// the order of the specs [`parse_constraints`] returns. Needs no baseline
/// scan, so callers can check lambda keys against it first.
pub(crate) fn spec_names(
    constraints: &HashMap<String, HashMap<String, Option<f64>>>,
    grid: &QuoteGrid,
) -> Vec<String> {
    grid.constraint_names
        .iter()
        .filter(|name| constraints.contains_key(*name))
        .cloned()
        .collect()
}

/// Absolute bound of each parsed constraint, in spec order.
pub(crate) fn spec_bounds(specs: &[ConstraintSpec]) -> Vec<f64> {
    specs.iter().map(|spec| spec.threshold).collect()
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
    let initial_lambdas: Option<Vec<f64>> = lambdas
        .map(|lam_dict| order_lambdas(&lam_dict, &constraint_names))
        .transpose()
        .map_err(PyValueError::new_err)?;

    let result = py
        .detach(|| solve_online(&grid, &specs, &config, initial_lambdas.as_deref()))
        .map_err(|e| PyValueError::new_err(format!("Solver error: {e}")))?;

    Ok(PySolveResult {
        inner: result,
        grid,
        constraint_names,
        constraint_bounds: spec_bounds(&specs),
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

    let initial_lambdas: Option<Vec<f64>> = lambdas
        .map(|lam_dict| order_lambdas(&lam_dict, &constraint_names))
        .transpose()
        .map_err(PyValueError::new_err)?;

    let grid_arc = Arc::clone(&grid.inner);
    let result = py
        .detach(|| solve_online(&grid_arc, &specs, &config, initial_lambdas.as_deref()))
        .map_err(|e| PyValueError::new_err(format!("Solver error: {e}")))?;

    Ok(PySolveResult {
        inner: result,
        grid: Arc::clone(&grid.inner),
        constraint_names,
        constraint_bounds: spec_bounds(&specs),
        result_df: None,
    })
}
