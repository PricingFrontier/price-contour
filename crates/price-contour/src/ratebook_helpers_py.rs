use std::sync::Arc;

use polars::prelude::*;
use price_contour_core::{build_group_mapping, GroupMapping};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Threshold above which we switch to Rayon parallel iteration.
pub(crate) const PAR_THRESHOLD: usize = 100_000;

/// Cached per-factor group structure: an Arc-wrapped `GroupMapping` with
/// pre-computed `group_of: Vec<u32>` (per-quote group index) and
/// `group_labels: Vec<String>` (one label per group).
///
/// **Why this exists:** the ratebook CD orchestrator calls
/// `solve_grouped_py`, `compute_residuals_py`, and `update_multipliers_py`
/// once per (frontier point × CD iteration × factor) — typically 100–500
/// times per portfolio sweep. Without a context, every call re-ferries
/// the per-quote `Vec<String>` group_labels through PyO3 (each
/// `PyString → owned String` allocation, 100k+ strings per call) and
/// re-builds the `GroupMapping` HashMap inside Rust. Wrapping the
/// expensive build once and passing the same `Arc<GroupMapping>` by
/// reference on subsequent calls eliminates both costs.
#[pyclass(name = "FactorContext", frozen)]
pub struct PyFactorContext {
    inner: Arc<GroupMapping>,
}

impl PyFactorContext {
    pub(crate) fn mapping(&self) -> &Arc<GroupMapping> {
        &self.inner
    }

    /// Wrap an existing `Arc<GroupMapping>` without rebuilding it.
    /// Used by `RatebookFactorContexts::_factor_contexts_for_solver`
    /// to hand the same shared mapping to the solver path that the
    /// chunked builder already constructed.
    pub(crate) fn from_arc(inner: Arc<GroupMapping>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFactorContext {
    /// Construct a context from per-quote group labels, building the
    /// group mapping eagerly.
    #[staticmethod]
    fn from_labels(labels: Vec<String>) -> Self {
        let mapping = build_group_mapping(&labels);
        Self {
            inner: Arc::new(mapping),
        }
    }

    #[getter]
    fn n_groups(&self) -> usize {
        self.inner.n_groups
    }

    #[getter]
    fn n_quotes(&self) -> usize {
        self.inner.group_of.len()
    }

    #[getter]
    fn group_labels(&self) -> Vec<String> {
        self.inner.group_labels.clone()
    }
}

/// Compute labels for a single factor spec.
///
/// One column → the string-cast values directly.
/// More than one column → per-row joined string with `separator` between fields.
pub(crate) fn build_spec_labels(
    df: &DataFrame,
    spec: &[String],
    separator: &str,
) -> PyResult<Vec<String>> {
    if spec.is_empty() {
        return Err(PyValueError::new_err(
            "factor spec is empty (no columns); each spec must list at least \
             one column",
        ));
    }
    if spec.len() == 1 {
        return cast_column_to_strings(df, &spec[0]);
    }

    // Multi-column interaction: cast each column to Utf8 once, then walk
    // rows in lock-step joining the values with `separator`. We hold the
    // casted Series alive for the duration of the join so the underlying
    // string buffers don't get freed mid-iteration.
    let casted: Vec<Series> = spec
        .iter()
        .map(|col_name| cast_column_to_string_series(df, col_name))
        .collect::<PyResult<_>>()?;
    for (i, s) in casted.iter().enumerate() {
        if s.null_count() > 0 {
            return Err(PyValueError::new_err(format!(
                "Factor column '{}' contains null values",
                spec[i]
            )));
        }
    }

    let str_chunked: Vec<&StringChunked> = casted
        .iter()
        .map(|s| {
            s.str()
                .expect("casted to String above; downcast cannot fail")
        })
        .collect();

    let n_rows = df.height();

    // Build the per-row joined strings. Use rayon above PAR_THRESHOLD so
    // the 50M-row interaction case doesn't choke on single-threaded
    // string concatenation; below the threshold, sequential beats
    // rayon's overhead.
    let labels = if n_rows > PAR_THRESHOLD {
        (0..n_rows)
            .into_par_iter()
            .map(|i| join_row(&str_chunked, i, separator))
            .collect()
    } else {
        (0..n_rows)
            .map(|i| join_row(&str_chunked, i, separator))
            .collect()
    };
    Ok(labels)
}

/// Join one row's values across all string-cast columns with `separator`.
/// Caller has already verified no nulls so `.get(i)` cannot be `None`.
#[inline]
fn join_row(columns: &[&StringChunked], row: usize, separator: &str) -> String {
    let mut s = columns[0]
        .get(row)
        .expect("null check at call site forbids None")
        .to_string();
    for ca in columns.iter().skip(1) {
        s.push_str(separator);
        s.push_str(ca.get(row).expect("null check at call site forbids None"));
    }
    s
}

/// Cast a column to Utf8 and collect into a `Vec<String>`. Single-column
/// fast path that avoids the multi-column lock-step iteration.
///
/// Above PAR_THRESHOLD the per-element `to_string()` (each row allocates a
/// `String`) is parallelised with rayon — without this the single-factor
/// path on a 50M-row column was 2-4× slower than the multi-column path it
/// shares a function with, due to lost parallelism.
fn cast_column_to_strings(df: &DataFrame, column_name: &str) -> PyResult<Vec<String>> {
    let casted = cast_column_to_string_series(df, column_name)?;
    if casted.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Factor column '{column_name}' contains null values"
        )));
    }
    let ca = casted
        .str()
        .expect("casted to String above; downcast cannot fail");
    let n = ca.len();
    let labels = if n > PAR_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|i| {
                ca.get(i)
                    .expect("null check above forbids None")
                    .to_string()
            })
            .collect()
    } else {
        ca.into_no_null_iter().map(|s| s.to_string()).collect()
    };
    Ok(labels)
}

/// Cast a DataFrame column to a Utf8 Series, mapping Polars errors to clear
/// Python-side messages naming the offending column.
fn cast_column_to_string_series(df: &DataFrame, column_name: &str) -> PyResult<Series> {
    let col = df.column(column_name).map_err(|_| {
        PyValueError::new_err(format!(
            "Factor column '{column_name}' not found in factors DataFrame"
        ))
    })?;
    let casted = col.cast(&DataType::String).map_err(|e| {
        PyValueError::new_err(format!(
            "Failed to cast factor column '{column_name}' to Utf8: {e}"
        ))
    })?;
    Ok(casted.take_materialized_series())
}

