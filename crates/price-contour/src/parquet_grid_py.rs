use std::fs::File;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::QuoteGrid;

use crate::builder_py::PyQuoteGridBuilder;
use crate::grid_py::PyQuoteGrid;
use crate::solver_py::ingest_dataframe;

/// Build a QuoteGrid directly from a Parquet file on disk.
///
/// Reads the file in Rust (no Python DataFrame intermediate), sorts by
/// (quote_id, scenario_index), and constructs the grid. The caller
/// (typically haute) sinks its lazy plan to parquet first, then passes
/// the path here so the entire pipeline avoids materialising a large
/// DataFrame in Python.
///
/// **Memory:** loads the entire required-column subset of the parquet into
/// memory at once. For very large files that may exceed available memory,
/// use [`build_grid_from_parquet_chunked_py`] instead, which streams the
/// file in fixed-size row slices via Polars' `with_slice` API.
#[pyfunction]
#[pyo3(signature = (
    path,
    constraint_columns,
    *,
    quote_id = "quote_id",
    scenario_index = "scenario_index",
    scenario_value = "scenario_value",
    objective = "expected_income",
))]
pub fn build_grid_from_parquet_py(
    py: Python<'_>,
    path: &str,
    constraint_columns: Vec<String>,
    quote_id: &str,
    scenario_index: &str,
    scenario_value: &str,
    objective: &str,
) -> PyResult<PyQuoteGrid> {
    validate_column_names(
        quote_id,
        scenario_index,
        scenario_value,
        objective,
        &constraint_columns,
    )?;

    let needed_columns = build_projection(
        quote_id,
        scenario_index,
        scenario_value,
        objective,
        &constraint_columns,
    );

    // Release the GIL across parquet IO + ingest. The body is pure Rust:
    // `read_parquet_slice` constructs Polars buffers (no Python objects),
    // `ingest_dataframe` walks the column data into a `QuoteGrid`. Errors
    // become `PyValueError`s lazily — `PyValueError::new_err(String)` is
    // safe to construct without the GIL in pyo3 0.26 (it stores the
    // message until restored to the interpreter).
    py.detach(|| -> PyResult<QuoteGrid> {
        let (total_rows, metadata) = open_metadata(path)?;
        let df = read_parquet_slice(path, &metadata, 0, total_rows, &needed_columns)?;
        ingest_dataframe(
            &df,
            quote_id,
            scenario_index,
            scenario_value,
            objective,
            &constraint_columns,
        )
    })
    .map(PyQuoteGrid::new)
}

/// Build a QuoteGrid by streaming a Parquet file in fixed-size row slices.
///
/// **Memory:** the parquet IO buffer scales with `chunk_size` (only one
/// slice resident at a time), not with the file's total row count. The
/// final `QuoteGrid` itself is still O(total_rows × n_columns × 4 bytes),
/// because every quote ends up resident in flat Rust vectors — that is
/// inherent to the solver's data layout. The win over the one-shot path is
/// avoiding a doubled peak from a Polars sort buffer and keeping per-IO
/// memory bounded.
///
/// Each slice is read via Polars' `ParquetReader::with_slice` so only the
/// row groups overlapping the slice range are deserialised; column
/// projection means only the four schema columns plus the requested
/// constraint columns are decoded.
///
/// `chunk_size` is rounded **down** to a multiple of `n_steps` so every
/// slice boundary falls between quotes — no carry buffer is needed. The
/// first slice of `chunk_size` rows is also used to auto-detect `n_steps`
/// (unless `n_steps` is supplied explicitly), so `chunk_size` must be at
/// least `n_steps`. For maximum safety against partial first chunks, pass
/// `n_steps` explicitly.
///
/// The resulting grid is sorted by `quote_id` (the underlying builder
/// performs an in-place sort at `build()` time), so the order of quotes in
/// the parquet does not matter for correctness — only the per-quote layout
/// (`n_steps` rows in `scenario_index` order) matters.
#[pyfunction]
#[pyo3(signature = (
    path,
    constraint_columns,
    chunk_size,
    *,
    quote_id = "quote_id",
    scenario_index = "scenario_index",
    scenario_value = "scenario_value",
    objective = "expected_income",
    n_steps = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn build_grid_from_parquet_chunked_py(
    py: Python<'_>,
    path: &str,
    constraint_columns: Vec<String>,
    chunk_size: usize,
    quote_id: &str,
    scenario_index: &str,
    scenario_value: &str,
    objective: &str,
    n_steps: Option<usize>,
) -> PyResult<PyQuoteGrid> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if let Some(0) = n_steps {
        return Err(PyValueError::new_err("n_steps must be > 0 if provided"));
    }
    validate_column_names(
        quote_id,
        scenario_index,
        scenario_value,
        objective,
        &constraint_columns,
    )?;

    // Release the GIL across the entire chunked-read + per-row validation
    // + cycle-permutation sort. The closure body is pure Rust: it
    // constructs `PyDataFrame` (a #[repr(transparent)] newtype around a
    // Polars `DataFrame`, no Python objects) and calls `PyQuoteGridBuilder`
    // methods which don't touch the interpreter. Without `py.detach` here
    // every other Python thread stalls through the entire ingest, including
    // any concurrent `build_grid_from_parquet_chunked` call.
    py.detach(|| -> PyResult<PyQuoteGrid> {
        // The builder is initialised on the first chunk once `n_steps` has
        // been resolved by the shared helper, then every subsequent chunk
        // appends through the same instance. Locking `n_steps` on the
        // builder skips its own auto-detection and gets us a single
        // consistent contract.
        let mut builder: Option<PyQuoteGridBuilder> = None;

        read_parquet_in_aligned_chunks(
            path,
            chunk_size,
            n_steps,
            quote_id,
            scenario_index,
            scenario_value,
            objective,
            &constraint_columns,
            |df, resolved_n_steps| {
                if builder.is_none() {
                    builder = Some(PyQuoteGridBuilder::new(
                        constraint_columns.clone(),
                        quote_id,
                        scenario_index,
                        scenario_value,
                        objective,
                        Some(resolved_n_steps),
                    )?);
                }
                builder.as_mut().unwrap().append(PyDataFrame(df))
            },
        )?;

        let mut builder =
            builder.ok_or_else(|| PyValueError::new_err("no chunks were processed"))?;
        builder.build()
    })
}

/// Read parquet metadata once so subsequent slice reads can reuse it.
pub(crate) fn open_metadata(path: &str) -> PyResult<(usize, FileMetadataRef)> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open parquet file: {e}")))?;
    let mut reader = ParquetReader::new(file);
    let total = reader
        .num_rows()
        .map_err(|e| PyValueError::new_err(format!("Failed to read parquet: {e}")))?;
    let metadata = reader
        .get_metadata()
        .map_err(|e| PyValueError::new_err(format!("Failed to read parquet metadata: {e}")))?
        .clone();
    Ok((total, metadata))
}

/// Project just the columns the grid needs and read a single contiguous
/// row range. The cached `metadata` avoids re-parsing the parquet footer
/// on every chunk.
pub(crate) fn read_parquet_slice(
    path: &str,
    metadata: &FileMetadataRef,
    offset: usize,
    length: usize,
    columns: &[String],
) -> PyResult<DataFrame> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open parquet file: {e}")))?;
    let mut reader = ParquetReader::new(file);
    reader.set_metadata(metadata.clone());
    reader
        .with_slice(Some((offset, length)))
        .with_columns(Some(columns.to_vec()))
        .finish()
        .map_err(|e| PyValueError::new_err(format!("Failed to read parquet: {e}")))
}

/// Reject column-name configurations that would collide in the parquet
/// projection — duplicate constraint names, or a constraint column named
/// the same as one of the schema columns. Without this check, Polars
/// errors deep inside the column-fetch with an opaque message.
pub(crate) fn validate_column_names(
    quote_id: &str,
    scenario_index: &str,
    scenario_value_col: &str,
    objective: &str,
    constraint_columns: &[String],
) -> PyResult<()> {
    let schema_cols = [quote_id, scenario_index, scenario_value_col, objective];

    // Schema columns themselves must be distinct (e.g., user can't pass
    // quote_id="x" AND objective="x").
    for i in 0..schema_cols.len() {
        for j in (i + 1)..schema_cols.len() {
            if schema_cols[i] == schema_cols[j] {
                return Err(PyValueError::new_err(format!(
                    "schema columns must be distinct, but '{}' is used twice",
                    schema_cols[i]
                )));
            }
        }
    }

    // Constraint columns must not collide with schema columns or each other.
    let mut seen = std::collections::HashSet::with_capacity(constraint_columns.len());
    for c in constraint_columns {
        if schema_cols.contains(&c.as_str()) {
            return Err(PyValueError::new_err(format!(
                "constraint column '{c}' collides with a schema column \
                 (quote_id/scenario_index/scenario_value/objective)"
            )));
        }
        if !seen.insert(c.as_str()) {
            return Err(PyValueError::new_err(format!(
                "constraint column '{c}' is listed more than once"
            )));
        }
    }
    Ok(())
}

pub(crate) fn build_projection(
    quote_id: &str,
    scenario_index: &str,
    scenario_value_col: &str,
    objective: &str,
    constraint_columns: &[String],
) -> Vec<String> {
    let mut cols = Vec::with_capacity(4 + constraint_columns.len());
    cols.push(quote_id.to_string());
    cols.push(scenario_index.to_string());
    cols.push(scenario_value_col.to_string());
    cols.push(objective.to_string());
    cols.extend(constraint_columns.iter().cloned());
    cols
}

/// Stream a parquet file in aligned chunks, invoking `callback` for each
/// chunk's DataFrame. The chunked-parquet ingest contract — column-name
/// validation, metadata caching, `n_steps` detection, divisibility check,
/// and quote-aligned slicing — is handled here once so every
/// chunked-parquet entry point shares a single implementation.
///
/// Returns the resolved `n_steps` so the caller can configure downstream
/// state (e.g., `PyQuoteGridBuilder` with `n_steps` locked).
///
/// Caller is responsible for:
///   - Validating `chunk_size > 0` and `n_steps_hint != Some(0)` upfront.
///   - Calling `validate_column_names` for collision detection (which has
///     more context-specific error wording per entry point).
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_parquet_in_aligned_chunks<F>(
    path: &str,
    chunk_size: usize,
    n_steps_hint: Option<usize>,
    quote_id: &str,
    scenario_index: &str,
    scenario_value_col: &str,
    objective: &str,
    constraint_columns: &[String],
    mut callback: F,
) -> PyResult<usize>
where
    F: FnMut(DataFrame, usize) -> PyResult<()>,
{
    let (total_rows, metadata) = open_metadata(path)?;
    if total_rows == 0 {
        return Err(PyValueError::new_err("parquet file has no rows"));
    }

    let needed_columns = build_projection(
        quote_id,
        scenario_index,
        scenario_value_col,
        objective,
        constraint_columns,
    );

    let probe_len = chunk_size.min(total_rows);
    let probe = read_parquet_slice(path, &metadata, 0, probe_len, &needed_columns)?;

    let n_steps = match n_steps_hint {
        Some(ns) => ns,
        None => detect_n_steps(
            &probe,
            scenario_index,
            total_rows,
            path,
            &metadata,
            &needed_columns,
        )?,
    };

    if !total_rows.is_multiple_of(n_steps) {
        return Err(PyValueError::new_err(format!(
            "parquet has {total_rows} rows, not divisible by n_steps={n_steps} \
             (each quote must occupy exactly n_steps rows)"
        )));
    }

    let aligned = (chunk_size / n_steps) * n_steps;
    if aligned == 0 {
        return Err(PyValueError::new_err(format!(
            "chunk_size {chunk_size} is smaller than n_steps {n_steps}; \
             must process at least one complete quote per chunk"
        )));
    }

    // First chunk: trim the probe to its largest aligned-multiple-of-n_steps
    // prefix that fits within `aligned`. The trailing rows of the probe
    // (if any) are NOT carried over — they're re-read by the next slice
    // via `with_slice` starting at `first_len`. Bounded duplicate IO of
    // at most `n_steps - 1` rows total.
    let probe_aligned = probe.height() - (probe.height() % n_steps);
    let first_len = aligned.min(probe_aligned);
    if first_len > 0 {
        callback(probe.slice(0, first_len), n_steps)?;
    }

    let mut offset = first_len;
    while offset < total_rows {
        let len = aligned.min(total_rows - offset);
        let df = read_parquet_slice(path, &metadata, offset, len, &needed_columns)?;
        callback(df, n_steps)?;
        offset += len;
    }

    Ok(n_steps)
}

/// Detect `n_steps` from a probe DataFrame, confirming with a one-row
/// peek past the probe if the probe didn't observe a quote boundary
/// itself.
///
/// A correct sorted probe runs `scenario_index` from `0..n_steps`, then
/// resets to `0` for the next quote. The position of the first reset is
/// `n_steps`. When no reset is observed within the probe, the inference
/// `n_steps = max + 1` is only a lower bound, so we read one extra row
/// (`probe.height()`) and confirm its `scenario_index == 0`. The peek
/// is skipped when the probe IS the whole file (no more rows to peek
/// at).
///
/// `pub(crate)` so the chunked apply path can reuse this exact detection
/// logic — keeps the chunked-parquet contract consistent across
/// read/apply.
pub(crate) fn detect_n_steps(
    probe: &DataFrame,
    scenario_index_col: &str,
    total_rows: usize,
    path: &str,
    metadata: &FileMetadataRef,
    needed_columns: &[String],
) -> PyResult<usize> {
    let h = probe.height();
    if h == 0 {
        return Err(PyValueError::new_err("probe DataFrame is empty"));
    }
    let step_ca = probe
        .column(scenario_index_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_index_col}")))?
        .i32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_index_col} must be Int32")))?;

    let first = step_ca
        .get(0)
        .ok_or_else(|| PyValueError::new_err(format!("Null {scenario_index_col} at row 0")))?;
    if first != 0 {
        return Err(PyValueError::new_err(format!(
            "first row of parquet has {scenario_index_col}={first}, expected 0 \
             (rows must be sorted by (quote_id, scenario_index) per the parquet contract)"
        )));
    }

    // Walk forward looking for the first scenario_index drop (= start of next quote).
    let mut prev = first;
    let mut max_step = first;
    for i in 1..h {
        let cur = step_ca.get(i).ok_or_else(|| {
            PyValueError::new_err(format!("Null {scenario_index_col} at row {i}"))
        })?;
        if cur < prev {
            // Drop found at row i — n_steps = i (number of rows in the first quote).
            return Ok(i);
        }
        if cur > max_step {
            max_step = cur;
        }
        prev = cur;
    }

    let inferred = (max_step as usize) + 1;

    if h == total_rows {
        // Probe IS the whole file — there is exactly one quote, no further rows
        // to confirm against.
        return Ok(inferred);
    }

    // Probe didn't see a drop but more rows exist. Peek one row past the probe;
    // it must be the start of the next quote (scenario_index = 0).
    let peek = read_parquet_slice(path, metadata, h, 1, needed_columns)?;
    let peek_ca = peek
        .column(scenario_index_col)
        .map_err(|_| PyValueError::new_err(format!("Missing column: {scenario_index_col}")))?
        .i32()
        .map_err(|_| PyValueError::new_err(format!("{scenario_index_col} must be Int32")))?;
    let peek_first = peek_ca
        .get(0)
        .ok_or_else(|| PyValueError::new_err(format!("Null {scenario_index_col} at row {h}")))?;
    if peek_first != 0 {
        return Err(PyValueError::new_err(format!(
            "auto-detected n_steps={inferred} but row {h} has \
             {scenario_index_col}={peek_first}, expected 0 (start of next quote). \
             chunk_size is too small to confirm n_steps; pass n_steps explicitly \
             or use a larger chunk_size that contains at least two complete quotes"
        )));
    }
    Ok(inferred)
}
