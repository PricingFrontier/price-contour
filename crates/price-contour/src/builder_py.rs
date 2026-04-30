use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::QuoteGridBuilder;

use crate::grid_py::PyQuoteGrid;

/// Extracted chunk: (quote_ids, objective_values, constraint_columns).
type ChunkData = (Vec<String>, Vec<f32>, Vec<Vec<f32>>);

/// Tolerance for comparing scenario_value floats against the canonical grid.
///
/// Upstream pipelines typically materialise scenario_values from a fixed
/// grid, so chunks from the same pipeline will have bit-identical values.
/// This tolerance covers benign rounding when chunks come from different
/// parquet writers or scaling steps without flagging genuine mismatches.
const SCENARIO_VALUE_TOL: f32 = 1e-6;

/// Python-visible builder that accepts Polars DataFrame chunks and constructs
/// a `QuoteGrid` incrementally.
///
/// **Per-chunk contract:** each chunk's rows must already be grouped by
/// `quote_id` (each quote occupies `n_steps` contiguous rows) and ordered by
/// `scenario_index` within each group. The builder does **not** sort chunks
/// — that would require materialising and rewriting the whole chunk and
/// defeats the memory-saving purpose of chunked ingestion. Across chunks,
/// the global order can be arbitrary; the canonical sort by `quote_id`
/// happens once at `build()` time, in-place over the unified grid.
///
/// **`n_steps`:** if provided to the constructor, it locks the contract and
/// every chunk is validated against it. If omitted, it is auto-detected from
/// the first chunk's first quote — convenient when the first chunk is fully
/// formed, but unsafe if the streaming source can hand you a partial first
/// quote (e.g., a parquet row group that cuts mid-quote). Pipelines that may
/// receive partial first chunks should pass `n_steps` explicitly.
#[pyclass(name = "QuoteGridBuilder")]
pub struct PyQuoteGridBuilder {
    inner: Option<QuoteGridBuilder>,
    // Column name configuration
    quote_id_col: String,
    scenario_index_col: String,
    scenario_value_col: String,
    objective_col: String,
    constraint_cols: Vec<String>,
    // Established on first append (or at construction if `n_steps` was passed).
    n_steps: Option<usize>,
    scenario_values: Option<Vec<f32>>,
    consumed: bool,
}

#[pymethods]
impl PyQuoteGridBuilder {
    #[new]
    #[pyo3(signature = (
        constraint_columns,
        *,
        quote_id = "quote_id",
        scenario_index = "scenario_index",
        scenario_value_col = "scenario_value",
        objective = "expected_income",
        n_steps = None,
    ))]
    pub(crate) fn new(
        constraint_columns: Vec<String>,
        quote_id: &str,
        scenario_index: &str,
        scenario_value_col: &str,
        objective: &str,
        n_steps: Option<usize>,
    ) -> PyResult<Self> {
        if let Some(0) = n_steps {
            return Err(PyValueError::new_err("n_steps must be > 0 if provided"));
        }
        Ok(Self {
            inner: None,
            quote_id_col: quote_id.to_string(),
            scenario_index_col: scenario_index.to_string(),
            scenario_value_col: scenario_value_col.to_string(),
            objective_col: objective.to_string(),
            constraint_cols: constraint_columns,
            n_steps,
            scenario_values: None,
            consumed: false,
        })
    }

    /// Append a chunk of data.
    ///
    /// First call extracts `scenario_values` from the chunk's first quote
    /// (and `n_steps`, unless it was provided to the constructor). Every
    /// chunk — including the first — is then validated row-by-row: each
    /// quote occupies exactly `n_steps` contiguous rows in `scenario_index`
    /// order, and every `scenario_value` matches the canonical grid.
    pub(crate) fn append(&mut self, df: PyDataFrame) -> PyResult<()> {
        if self.consumed {
            return Err(PyValueError::new_err("builder already consumed by build()"));
        }
        let df = df.0;

        if df.height() == 0 {
            // Empty chunks are a no-op — useful when streaming and a row-group
            // happens to be empty after filtering.
            return Ok(());
        }

        // Establish n_steps and the canonical scenario_values on the first
        // non-empty chunk. After this point, both are immutable for the
        // lifetime of the builder.
        if self.scenario_values.is_none() {
            let (n_steps, scenario_values) = derive_grid_metadata(
                &df,
                &self.quote_id_col,
                &self.scenario_index_col,
                &self.scenario_value_col,
                self.n_steps,
            )?;
            self.n_steps = Some(n_steps);
            self.scenario_values = Some(scenario_values.clone());
            let builder =
                QuoteGridBuilder::new(n_steps, scenario_values, self.constraint_cols.clone())
                    .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            self.inner = Some(builder);
        }
        let n_steps = self.n_steps.unwrap();
        let scenario_values = self.scenario_values.as_ref().unwrap();

        let (quote_ids, objective, constraints) = extract_chunk(
            &df,
            &self.quote_id_col,
            &self.scenario_index_col,
            &self.scenario_value_col,
            &self.objective_col,
            &self.constraint_cols,
            n_steps,
            scenario_values,
        )?;

        let builder = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("builder consumed by build()"))?;

        builder
            .append(&quote_ids, &objective, &constraints)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok(())
    }

    /// Consume the builder and return a `QuoteGrid` sorted by `quote_id`.
    pub(crate) fn build(&mut self) -> PyResult<PyQuoteGrid> {
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
        Ok(PyQuoteGrid::new(grid))
    }

    #[getter]
    fn n_quotes(&self) -> usize {
        self.inner.as_ref().map_or(0, |b| b.n_quotes())
    }
}

/// Derive `n_steps` and the canonical `scenario_values` from the first chunk.
///
/// If `n_steps_hint` is `Some`, it's used directly (the caller has committed
/// to the contract upfront). Otherwise it is inferred as
/// `max(scenario_index) + 1` over the whole chunk — robust against
/// interleaved layouts that would otherwise be silently misinterpreted as
/// `n_steps = 1`.
///
/// Either way, the first quote is then required to occupy rows `0..n_steps`
/// with `scenario_index = 0..n_steps`, and `scenario_values` is read from
/// those rows. Caller must have verified the DataFrame is non-empty.
fn derive_grid_metadata(
    df: &DataFrame,
    quote_id_col: &str,
    scenario_index_col: &str,
    scenario_value_col: &str,
    n_steps_hint: Option<usize>,
) -> PyResult<(usize, Vec<f32>)> {
    let n_rows = df.height();

    let step_ca = df
        .column(scenario_index_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_index_col}")))?
        .i32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_index_col} must be Int32")))?;

    let n_steps = if let Some(hint) = n_steps_hint {
        hint
    } else {
        // Single linear pass to find max(scenario_index); also catches negative
        // values and nulls early so they surface as a clean error.
        let mut max_step: i32 = -1;
        for i in 0..n_rows {
            let s = step_ca.get(i).ok_or_else(|| {
                PyValueError::new_err(format!("Null {scenario_index_col} at row {i}"))
            })?;
            if s < 0 {
                return Err(PyValueError::new_err(format!(
                    "{scenario_index_col} at row {i} is {s}; must be >= 0"
                )));
            }
            if s > max_step {
                max_step = s;
            }
        }
        if max_step < 0 {
            return Err(PyValueError::new_err(
                "could not determine n_steps from chunk (no scenario_index values)",
            ));
        }
        (max_step as usize) + 1
    };

    if n_rows < n_steps {
        return Err(PyValueError::new_err(format!(
            "first chunk has only {n_rows} rows but n_steps={n_steps}; \
             first quote cannot be observed (consider passing n_steps=… explicitly \
             if streaming partial first chunks)"
        )));
    }

    let qid_ca = df
        .column(quote_id_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {quote_id_col}")))?
        .str()
        .map_err(|_| PyValueError::new_err(format!("{quote_id_col} must be Utf8")))?;
    let first_qid = qid_ca
        .get(0)
        .ok_or_else(|| PyValueError::new_err(format!("Null {quote_id_col} at row 0")))?;

    // The first quote must occupy the first `n_steps` rows in scenario_index
    // order. This catches interleaved layouts where step-0 rows for many
    // quotes appear before any step-1 row.
    for j in 0..n_steps {
        let row_qid = qid_ca
            .get(j)
            .ok_or_else(|| PyValueError::new_err(format!("Null {quote_id_col} at row {j}")))?;
        if row_qid != first_qid {
            return Err(PyValueError::new_err(format!(
                "First quote '{first_qid}' should occupy rows 0..{n_steps} but row {j} \
                 has {quote_id_col}='{row_qid}'. Each quote must occupy {n_steps} \
                 contiguous rows in {scenario_index_col} order"
            )));
        }
        let row_step = step_ca.get(j).ok_or_else(|| {
            PyValueError::new_err(format!("Null {scenario_index_col} at row {j}"))
        })?;
        if row_step != j as i32 {
            return Err(PyValueError::new_err(format!(
                "First quote '{first_qid}': row {j} has {scenario_index_col}={row_step}, \
                 expected {j}"
            )));
        }
    }

    let mult_ca = df
        .column(scenario_value_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_value_col}")))?
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_value_col} must be Float32")))?;
    let scenario_values: Vec<f32> = (0..n_steps)
        .map(|i| {
            mult_ca.get(i).ok_or_else(|| {
                PyValueError::new_err(format!("Null {scenario_value_col} at row {i}"))
            })
        })
        .collect::<PyResult<Vec<f32>>>()?;

    Ok((n_steps, scenario_values))
}

/// Extract per-quote columns from a chunk DataFrame, validating the full
/// per-chunk layout contract: `n_steps` contiguous rows per quote with
/// `scenario_index = 0..n_steps`, and `scenario_value` matching the
/// canonical grid on every row.
///
/// One linear pass over the chunk handles all the validation; this replaces
/// the old O(n log n) Polars sort with a strict O(n) check that catches
/// interleaved layouts, mismatched scenario grids, and out-of-order rows.
#[allow(clippy::too_many_arguments)]
fn extract_chunk(
    df: &DataFrame,
    quote_id_col: &str,
    scenario_index_col: &str,
    scenario_value_col: &str,
    objective_col: &str,
    constraint_cols: &[String],
    n_steps: usize,
    canonical_scenario_values: &[f32],
) -> PyResult<ChunkData> {
    let n_rows = df.height();
    if !n_rows.is_multiple_of(n_steps) {
        return Err(PyValueError::new_err(format!(
            "chunk row count {n_rows} not divisible by n_steps {n_steps} \
             (each quote must occupy exactly {n_steps} contiguous rows)"
        )));
    }
    let n_quotes = n_rows / n_steps;

    let qid_ca = df
        .column(quote_id_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {quote_id_col}")))?
        .str()
        .map_err(|_| PyValueError::new_err(format!("{quote_id_col} must be Utf8")))?;
    let steps_ca = df
        .column(scenario_index_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_index_col}")))?
        .i32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_index_col} must be Int32")))?;
    let mult_ca = df
        .column(scenario_value_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_value_col}")))?
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_value_col} must be Float32")))?;

    // Single linear scan: pull one quote_id per group, validate that all
    // n_steps rows in that group share that quote_id, that scenario_index
    // runs 0..n_steps in order, and that scenario_value matches the
    // canonical grid bit-for-bit (within tolerance).
    let mut quote_ids: Vec<String> = Vec::with_capacity(n_quotes);
    for q in 0..n_quotes {
        let block_start = q * n_steps;
        let qid = qid_ca.get(block_start).ok_or_else(|| {
            PyValueError::new_err(format!("Null {quote_id_col} at row {block_start}"))
        })?;
        for (j, &expected_sv) in canonical_scenario_values.iter().enumerate().take(n_steps) {
            let idx = block_start + j;
            let row_qid = qid_ca.get(idx).ok_or_else(|| {
                PyValueError::new_err(format!("Null {quote_id_col} at row {idx}"))
            })?;
            if row_qid != qid {
                return Err(PyValueError::new_err(format!(
                    "Quote rows not contiguous: row {idx} has {quote_id_col}='{row_qid}', \
                     expected '{qid}' (each quote must occupy {n_steps} consecutive rows)"
                )));
            }
            let step_val = steps_ca.get(idx).ok_or_else(|| {
                PyValueError::new_err(format!("Null {scenario_index_col} at row {idx}"))
            })?;
            if step_val != j as i32 {
                return Err(PyValueError::new_err(format!(
                    "Quote '{qid}' row {idx}: {scenario_index_col}={step_val}, expected {j} \
                     (rows for each quote must be in scenario_index order 0..{n_steps})"
                )));
            }
            let mult_val = mult_ca.get(idx).ok_or_else(|| {
                PyValueError::new_err(format!("Null {scenario_value_col} at row {idx}"))
            })?;
            // (NaN - NaN).abs() is NaN, which compares false against any
            // finite tolerance — without an explicit check, a NaN at row > 0
            // would slip through the consistency comparison.
            if !mult_val.is_finite() || (mult_val - expected_sv).abs() > SCENARIO_VALUE_TOL {
                return Err(PyValueError::new_err(format!(
                    "Quote '{qid}' row {idx}: {scenario_value_col}={mult_val}, expected \
                     {expected_sv} for step {j} (all chunks must share the same scenario grid)"
                )));
            }
        }
        quote_ids.push(qid.to_string());
    }

    let obj_ca = df
        .column(objective_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {objective_col}")))?
        .f32()
        .map_err(|_| PyValueError::new_err(format!("{objective_col} must be Float32")))?;
    if obj_ca.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Column '{objective_col}' contains null values"
        )));
    }
    let objective: Vec<f32> = obj_ca.into_no_null_iter().collect();

    let mut constraints: Vec<Vec<f32>> = Vec::with_capacity(constraint_cols.len());
    for col_name in constraint_cols {
        let ca = df
            .column(col_name)
            .map_err(|_| PyValueError::new_err(format!("Missing column: {col_name}")))?
            .f32()
            .map_err(|_| PyValueError::new_err(format!("{col_name} must be Float32")))?;
        if ca.null_count() > 0 {
            return Err(PyValueError::new_err(format!(
                "Column '{col_name}' contains null values"
            )));
        }
        constraints.push(ca.into_no_null_iter().collect());
    }

    Ok((quote_ids, objective, constraints))
}
