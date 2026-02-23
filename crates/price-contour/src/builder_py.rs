use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::QuoteGridBuilder;

use crate::grid_py::PyQuoteGrid;

/// Python-visible builder that accepts Polars DataFrame chunks and
/// constructs a QuoteGrid incrementally.
#[pyclass(name = "QuoteGridBuilder")]
pub struct PyQuoteGridBuilder {
    inner: Option<QuoteGridBuilder>,
    // Column name configuration
    quote_id_col: String,
    scenario_step_col: String,
    multiplier_col: String,
    objective_col: String,
    constraint_cols: Vec<String>,
    // Lazily initialised on first append
    initialised: bool,
    consumed: bool,
}

#[pymethods]
impl PyQuoteGridBuilder {
    #[new]
    #[pyo3(signature = (
        constraint_columns,
        *,
        quote_id = "quote_id",
        scenario_step = "scenario_step",
        multiplier_col = "multiplier",
        objective = "expected_income",
    ))]
    fn new(
        constraint_columns: Vec<String>,
        quote_id: &str,
        scenario_step: &str,
        multiplier_col: &str,
        objective: &str,
    ) -> Self {
        Self {
            inner: None,
            quote_id_col: quote_id.to_string(),
            scenario_step_col: scenario_step.to_string(),
            multiplier_col: multiplier_col.to_string(),
            objective_col: objective.to_string(),
            constraint_cols: constraint_columns,
            initialised: false,
            consumed: false,
        }
    }

    /// Append a chunk of data. The first call extracts n_steps and multipliers
    /// from the chunk; subsequent calls validate consistency.
    fn append(&mut self, df: PyDataFrame) -> PyResult<()> {
        if self.consumed {
            return Err(PyValueError::new_err("builder already consumed by build()"));
        }
        let df = &df.0;

        // Sort by (quote_id, scenario_step)
        let df = df
            .sort(
                [&self.quote_id_col, &self.scenario_step_col],
                SortMultipleOptions::default(),
            )
            .map_err(|e| PyValueError::new_err(format!("Sort failed: {e}")))?;

        // If first call, initialise the builder from the chunk
        if !self.initialised {
            let (n_steps, multipliers) = extract_grid_info(&df, &self.quote_id_col, &self.multiplier_col)?;
            let builder = QuoteGridBuilder::new(n_steps, multipliers, self.constraint_cols.clone())
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            self.inner = Some(builder);
            self.initialised = true;
        }

        let builder = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("builder consumed by build()"))?;

        // Extract columns
        let (quote_ids, objective, constraints) = extract_chunk(
            &df,
            &self.quote_id_col,
            &self.scenario_step_col,
            &self.objective_col,
            &self.constraint_cols,
            builder.n_quotes(), // for error context only
        )?;

        builder
            .append(&quote_ids, &objective, &constraints)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok(())
    }

    /// Consume the builder and return a QuoteGrid.
    fn build(&mut self) -> PyResult<PyQuoteGrid> {
        if self.consumed {
            return Err(PyValueError::new_err("builder already consumed by build()"));
        }
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("builder already consumed or no data appended"))?;
        self.consumed = true;
        let grid = builder
            .build()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(PyQuoteGrid { inner: grid })
    }

    #[getter]
    fn n_quotes(&self) -> usize {
        self.inner.as_ref().map_or(0, |b| b.n_quotes())
    }
}

/// Extract n_steps and sorted multipliers from the first quote in a sorted DataFrame.
fn extract_grid_info(
    df: &DataFrame,
    quote_id_col: &str,
    multiplier_col: &str,
) -> PyResult<(usize, Vec<f32>)> {
    let n_rows = df.height();
    if n_rows == 0 {
        return Err(PyValueError::new_err("empty DataFrame"));
    }

    let qid_series = df
        .column(quote_id_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {quote_id_col}")))?;
    let qid_ca = qid_series
        .str()
        .map_err(|_| PyValueError::new_err(format!("{quote_id_col} must be Utf8")))?;

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

    let mult_series = df
        .column(multiplier_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {multiplier_col}")))?;
    let mult_ca = mult_series
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{multiplier_col} must be Float32")))?;
    let multipliers: Vec<f32> = (0..n_steps)
        .map(|i| mult_ca.get(i).unwrap_or(0.0))
        .collect();

    Ok((n_steps, multipliers))
}

/// Extract columns from a sorted DataFrame chunk.
fn extract_chunk(
    df: &DataFrame,
    quote_id_col: &str,
    scenario_step_col: &str,
    objective_col: &str,
    constraint_cols: &[String],
    _prior_quotes: usize,
) -> PyResult<(Vec<String>, Vec<f32>, Vec<Vec<f32>>)> {
    let n_rows = df.height();

    let qid_series = df
        .column(quote_id_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {quote_id_col}")))?;
    let qid_ca = qid_series
        .str()
        .map_err(|_| PyValueError::new_err(format!("{quote_id_col} must be Utf8")))?;

    // Determine n_steps from first quote in this chunk
    let first_qid = qid_ca
        .get(0)
        .ok_or_else(|| PyValueError::new_err("Empty chunk"))?;
    let mut n_steps: usize = 0;
    for i in 0..n_rows {
        if qid_ca.get(i) == Some(first_qid) {
            n_steps += 1;
        } else {
            break;
        }
    }
    if n_steps == 0 || n_rows % n_steps != 0 {
        return Err(PyValueError::new_err(format!(
            "Row count {n_rows} not divisible by n_steps {n_steps}"
        )));
    }
    let n_quotes = n_rows / n_steps;

    // Extract quote IDs (one per quote)
    let quote_ids: Vec<String> = (0..n_quotes)
        .map(|q| qid_ca.get(q * n_steps).unwrap_or("").to_string())
        .collect();

    // Validate step sequence
    let step_series = df
        .column(scenario_step_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_step_col}")))?;
    let steps_ca = step_series
        .i32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_step_col} must be Int32")))?;
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

    // Extract objective
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

    Ok((quote_ids, objective, constraints))
}
