use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{apply_lambdas, ApplyResult, ConstraintSpec, QuoteGrid};

use crate::builder_py::PyQuoteGridBuilder;
use crate::grid_py::PyQuoteGrid;
use crate::parquet_grid_py::{read_parquet_in_aligned_chunks, validate_column_names};
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
    constraints: Option<HashMap<String, HashMap<String, Option<f64>>>>,
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
        .detach(|| apply_lambdas(&grid, &specs, &lambda_vec))
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
#[pyo3(signature = (grid, lambdas, constraints))]
pub fn apply_from_grid_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    lambdas: HashMap<String, f64>,
    constraints: HashMap<String, HashMap<String, Option<f64>>>,
) -> PyResult<PyApplyResult> {
    let specs = parse_constraints(constraints, &grid.inner)?;
    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    let lambda_vec = order_lambdas(&lambdas, &constraint_names);

    let grid_arc = Arc::clone(&grid.inner);
    let result = py
        .detach(|| apply_lambdas(&grid_arc, &specs, &lambda_vec))
        .map_err(|e| PyValueError::new_err(format!("Apply error: {e}")))?;

    Ok(PyApplyResult {
        inner: result,
        grid: Arc::clone(&grid.inner),
        constraint_names,
        result_df: None,
    })
}

/// Aggregate result of a chunked, streaming-output apply.
///
/// Carries the same totals/baselines/lambdas as the one-shot
/// [`PyApplyResult`], but the per-quote rows are NOT held in memory — they
/// have been written to `output_path` as the apply ran. Callers who need
/// the per-row output read it back via `pl.read_parquet(output_path)` (or
/// stream it lazily via `pl.scan_parquet`).
#[pyclass(name = "ChunkedApplyResult")]
pub struct PyChunkedApplyResult {
    constraint_names: Vec<String>,
    lambdas_vec: Vec<f64>,
    total_objective_val: f64,
    total_constraints_vec: Vec<f64>,
    baseline_objective_val: f64,
    baseline_constraints_vec: Vec<f64>,
    output_path_val: String,
}

#[pymethods]
impl PyChunkedApplyResult {
    #[getter]
    fn lambdas(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.lambdas_vec)
    }

    #[getter]
    fn total_objective(&self) -> f64 {
        self.total_objective_val
    }

    #[getter]
    fn total_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.total_constraints_vec)
    }

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.baseline_objective_val
    }

    #[getter]
    fn baseline_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.baseline_constraints_vec)
    }

    #[getter]
    fn output_path(&self) -> &str {
        &self.output_path_val
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkedApplyResult(total_objective={:.6}, output_path={:?}, n_constraints={})",
            self.total_objective_val,
            self.output_path_val,
            self.constraint_names.len(),
        )
    }
}

/// Apply fixed lambdas to a parquet input, streaming per-quote results to a
/// parquet output and returning aggregate totals.
///
/// **Memory:** the input parquet IO buffer is bounded by `chunk_size`, the
/// output parquet is written incrementally one row group per chunk, and
/// the **whole-portfolio** per-quote `optimal_steps` array is never
/// materialised — only one chunk's `optimal_steps` is alive at a time
/// (`chunk_size / n_steps` u32 entries) and gets dropped along with the
/// chunk's mini-grid after the row group has been written. The peak
/// resident set is therefore O(chunk_size × n_columns × 4 bytes) + the
/// BatchedWriter's internal buffers, regardless of the input file size.
///
/// `chunk_size` is rounded down to a multiple of `n_steps` so every slice
/// boundary falls between quotes. The chunked reader contract from
/// [`build_grid_from_parquet_chunked_py`] applies: rows must already be
/// grouped by `quote_id` with `scenario_index` running `0..n_steps` within
/// each group, and every `scenario_value` must match the canonical grid
/// (the per-row builder validation enforces this).
///
/// The output parquet schema is:
///   - `quote_id` (Utf8) — identical to input.
///   - `optimal_step` (Int32) — argmax index in `0..n_steps`.
///   - `optimal_scenario_value` (Float32).
///   - `optimal_objective` (Float32).
///   - `optimal_<name>` (Float32) — one per constraint, in sorted
///     constraint-name order (matching the in-memory apply path).
#[pyfunction]
#[pyo3(signature = (
    parquet_in,
    parquet_out,
    lambdas,
    constraints,
    chunk_size,
    *,
    quote_id = "quote_id",
    scenario_index = "scenario_index",
    scenario_value = "scenario_value",
    objective = "expected_income",
    n_steps = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn apply_lambdas_to_parquet_chunked_py(
    py: Python<'_>,
    parquet_in: &str,
    parquet_out: &str,
    lambdas: HashMap<String, f64>,
    constraints: HashMap<String, HashMap<String, Option<f64>>>,
    chunk_size: usize,
    quote_id: &str,
    scenario_index: &str,
    scenario_value: &str,
    objective: &str,
    n_steps: Option<usize>,
) -> PyResult<PyChunkedApplyResult> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if let Some(0) = n_steps {
        return Err(PyValueError::new_err("n_steps must be > 0 if provided"));
    }
    reject_same_input_output(parquet_in, parquet_out)?;

    // Note: ratio constraint specs (containing string `numerator` /
    // `denominator` keys) cannot reach this function — PyO3 deserialises
    // `constraints` into `HashMap<String, HashMap<String, Option<f64>>>`
    // and rejects string values at the boundary with `TypeError:
    // argument 'constraints': must be real number, not str`. The
    // chunked path therefore can't accept ratios by construction; the
    // type system enforces this. The Python wrapper docstring directs
    // ratio-constraint callers to `ApplyOptimiser.apply(df)`.

    // Sort constraint keys before any error message that references them
    // (matches `ApplyOptimiser.__init__`'s error format) and before storing
    // them in the grid (the grid's constraint_names is sorted-order so
    // results align across one-shot and chunked paths).
    let mut sorted_constraint_keys: Vec<String> = constraints.keys().cloned().collect();
    sorted_constraint_keys.sort();
    // Strict-match: every lambda key must correspond to a known constraint.
    // Collecting all extras and sorting them mirrors `ApplyOptimiser`'s
    // error wording exactly and is deterministic regardless of HashMap
    // iteration order.
    let mut extras: Vec<String> = lambdas
        .keys()
        .filter(|k| !constraints.contains_key(*k))
        .cloned()
        .collect();
    if !extras.is_empty() {
        extras.sort();
        return Err(PyValueError::new_err(format!(
            "Lambda keys {extras:?} do not match any constraint. Valid \
             constraint keys are {sorted_constraint_keys:?}."
        )));
    }

    validate_column_names(
        quote_id,
        scenario_index,
        scenario_value,
        objective,
        &sorted_constraint_keys,
    )?;

    // The grid stores constraint columns in sorted-name order; we already
    // sorted these for the strict-match error above, so just rename.
    let constraint_cols: Vec<String> = sorted_constraint_keys;

    // Cleanup-on-error: if the streaming pipeline fails after we've started
    // writing the output parquet, the file on disk has partial row groups
    // and no valid footer (Polars writes the footer only on `finish`).
    // Best-effort delete on error so we never leave a corrupt artefact.
    let output_touched = Arc::new(AtomicBool::new(false));
    let result = run_chunked_apply(
        py,
        parquet_in,
        parquet_out,
        lambdas,
        constraints,
        chunk_size,
        quote_id,
        scenario_index,
        scenario_value,
        objective,
        n_steps,
        constraint_cols,
        Arc::clone(&output_touched),
    );
    if result.is_err() && output_touched.load(Ordering::SeqCst) {
        let _ = std::fs::remove_file(parquet_out);
    }
    result
}

fn reject_same_input_output(parquet_in: &str, parquet_out: &str) -> PyResult<()> {
    let in_path = Path::new(parquet_in);
    let out_path = Path::new(parquet_out);
    if in_path == out_path {
        return Err(PyValueError::new_err(
            "parquet_out must be different from parquet_in; refusing to overwrite input parquet",
        ));
    }
    if let (Ok(in_canon), Ok(out_canon)) = (
        std::fs::canonicalize(in_path),
        std::fs::canonicalize(out_path),
    ) {
        if in_canon == out_canon {
            return Err(PyValueError::new_err(
                "parquet_out must be different from parquet_in; refusing to overwrite input parquet",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_chunked_apply(
    py: Python<'_>,
    parquet_in: &str,
    parquet_out: &str,
    lambdas: HashMap<String, f64>,
    constraints: HashMap<String, HashMap<String, Option<f64>>>,
    chunk_size: usize,
    quote_id: &str,
    scenario_index: &str,
    scenario_value_col: &str,
    objective: &str,
    n_steps: Option<usize>,
    constraint_cols: Vec<String>,
    output_touched: Arc<AtomicBool>,
) -> PyResult<PyChunkedApplyResult> {
    // The state holds the writer, accumulators, and parsed specs. It is
    // dropped (writer included) automatically on early-return — but we
    // still need the explicit `finish()` call before extracting totals,
    // which `state.finalize()` does. The outer cleanup-on-error wrapper
    // handles the disk file when an error escapes here.
    let mut state: Option<ChunkedApplyState<'_>> = None;

    py.detach(|| {
        read_parquet_in_aligned_chunks(
            parquet_in,
            chunk_size,
            n_steps,
            quote_id,
            scenario_index,
            scenario_value_col,
            objective,
            &constraint_cols,
            |df, resolved_n_steps| {
                if state.is_none() {
                    state = Some(ChunkedApplyState::new(
                        parquet_out,
                        constraint_cols.clone(),
                        constraints.clone(),
                        lambdas.clone(),
                        quote_id,
                        scenario_index,
                        scenario_value_col,
                        objective,
                        resolved_n_steps,
                        Arc::clone(&output_touched),
                    ));
                }
                state.as_mut().unwrap().process_chunk(df)
            },
        )
    })?;

    let state = state.ok_or_else(|| {
        PyValueError::new_err("no chunks were processed; output parquet was not written")
    })?;
    state.finalize()
}

/// Loop-state for the streaming apply. Encapsulates the per-chunk pipeline
/// (build mini-grid → apply → write row group → accumulate totals) and the
/// caching of derived state (specs, lambda vector, output parquet writer)
/// from the first chunk.
struct ChunkedApplyState<'a> {
    parquet_out: String,
    constraint_cols: Vec<String>,
    constraints: HashMap<String, HashMap<String, Option<f64>>>,
    lambdas: HashMap<String, f64>,
    quote_id: &'a str,
    scenario_index: &'a str,
    scenario_value_col: &'a str,
    objective: &'a str,
    n_steps: usize,

    // Derived from the first chunk and reused thereafter.
    specs: Option<Vec<ConstraintSpec>>,
    constraint_names: Option<Vec<String>>,
    lambda_vec: Option<Vec<f64>>,

    // Output writer is opened lazily on the first chunk so its schema can
    // be read off the first mini-result DataFrame.
    writer: Option<polars::io::parquet::write::BatchedWriter<File>>,

    // Aggregate accumulators in f64 for portfolio-scale precision.
    total_objective: f64,
    total_constraints: Vec<f64>,
    baseline_objective: f64,
    baseline_constraints: Vec<f64>,
    output_touched: Arc<AtomicBool>,
}

impl<'a> ChunkedApplyState<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        parquet_out: &str,
        constraint_cols: Vec<String>,
        constraints: HashMap<String, HashMap<String, Option<f64>>>,
        lambdas: HashMap<String, f64>,
        quote_id: &'a str,
        scenario_index: &'a str,
        scenario_value_col: &'a str,
        objective: &'a str,
        n_steps: usize,
        output_touched: Arc<AtomicBool>,
    ) -> Self {
        let n_cons = constraint_cols.len();
        Self {
            parquet_out: parquet_out.to_string(),
            constraint_cols,
            constraints,
            lambdas,
            quote_id,
            scenario_index,
            scenario_value_col,
            objective,
            n_steps,
            specs: None,
            constraint_names: None,
            lambda_vec: None,
            writer: None,
            total_objective: 0.0,
            total_constraints: vec![0.0; n_cons],
            baseline_objective: 0.0,
            baseline_constraints: vec![0.0; n_cons],
            output_touched,
        }
    }

    fn process_chunk(&mut self, df: DataFrame) -> PyResult<()> {
        // Build a mini-grid from this chunk via the shared chunked-builder
        // pipeline. This reuses the per-row scenario_value / scenario_index
        // validation, so a corrupt chunk surfaces a clear error here rather
        // than producing garbage output downstream.
        let mut builder = PyQuoteGridBuilder::new(
            self.constraint_cols.clone(),
            self.quote_id,
            self.scenario_index,
            self.scenario_value_col,
            self.objective,
            Some(self.n_steps),
        )?;
        builder.append(PyDataFrame(df))?;
        let mini = builder.build()?;
        let grid = &mini.inner;

        // Parse constraint specs once on the first chunk; reuse on all
        // subsequent chunks.
        //
        // Subtle correctness note: `parse_constraints` resolves
        // `min_pct`/`max_pct` against the grid's per-quote baselines. Run
        // on the first chunk's mini-grid, the resulting `spec.threshold`
        // values reflect ONLY that chunk's baseline totals, not the whole
        // file's — and they would be wrong if downstream math relied on
        // them. The math doesn't: `apply_lambdas` reads `spec.direction`
        // and the user-supplied `lambdas`, never `spec.threshold`. The
        // user-facing `total_constraints` and `baseline_constraints` are
        // accumulated chunk-by-chunk from per-quote sums, so they reflect
        // the full file and are independent of any spec-threshold drift.
        if self.specs.is_none() {
            let parsed = parse_constraints(self.constraints.clone(), grid)?;
            let names: Vec<String> = parsed.iter().map(|s| s.name.clone()).collect();
            let lvec = order_lambdas(&self.lambdas, &names);
            self.specs = Some(parsed);
            self.constraint_names = Some(names);
            self.lambda_vec = Some(lvec);
        }
        let specs = self.specs.as_ref().unwrap();
        let lambda_vec = self.lambda_vec.as_ref().unwrap();

        let result = apply_lambdas(grid, specs, lambda_vec)
            .map_err(|e| PyValueError::new_err(format!("Apply error: {e}")))?;

        // Emit the per-quote rows for this chunk. Reuses `build_result_dataframe`
        // so the output schema is identical to `PyApplyResult.dataframe`.
        let chunk_df = build_result_dataframe(&result.optimal_steps, grid)?;

        if self.writer.is_none() {
            let file = File::create(&self.parquet_out).map_err(|e| {
                PyValueError::new_err(format!("Failed to open output parquet for write: {e}"))
            })?;
            self.output_touched.store(true, Ordering::SeqCst);
            let writer = ParquetWriter::new(file)
                .batched(chunk_df.schema().as_ref())
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to initialise parquet writer: {e}"))
                })?;
            self.writer = Some(writer);
        }
        self.writer
            .as_mut()
            .unwrap()
            .write_batch(&chunk_df)
            .map_err(|e| {
                PyValueError::new_err(format!("Failed to write parquet row group: {e}"))
            })?;
        // chunk_df, mini, and result fall out of scope here; their backing
        // buffers (objective/constraint Vecs, optimal_steps Vec) are dropped
        // before the next chunk is read, so peak memory stays bounded by
        // chunk_size + the writer's row-group buffer.

        // Accumulate aggregate totals in f64.
        self.total_objective += result.total_objective;
        for (i, &v) in result.total_constraints.iter().enumerate() {
            self.total_constraints[i] += v;
        }
        self.baseline_objective += result.baseline_objective;
        for (i, &v) in result.baseline_constraints.iter().enumerate() {
            self.baseline_constraints[i] += v;
        }
        Ok(())
    }

    fn finalize(self) -> PyResult<PyChunkedApplyResult> {
        // Close the output writer; if no chunks were processed, error
        // (consistent with empty-grid handling elsewhere).
        let Some(writer) = self.writer else {
            return Err(PyValueError::new_err(
                "no chunks were processed; output parquet was not written",
            ));
        };
        writer.finish().map_err(|e| {
            PyValueError::new_err(format!("Failed to finalise parquet writer: {e}"))
        })?;

        let constraint_names = self.constraint_names.unwrap_or_default();
        let lambda_vec = self.lambda_vec.unwrap_or_default();
        Ok(PyChunkedApplyResult {
            constraint_names,
            lambdas_vec: lambda_vec,
            total_objective_val: self.total_objective,
            total_constraints_vec: self.total_constraints,
            baseline_objective_val: self.baseline_objective,
            baseline_constraints_vec: self.baseline_constraints,
            output_path_val: self.parquet_out,
        })
    }
}
