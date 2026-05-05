use std::sync::Arc;

use polars::prelude::*;
use price_contour_core::{build_group_mapping, GroupMapping};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

/// Threshold above which we switch to Rayon parallel iteration.
const PAR_THRESHOLD: usize = 100_000;

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

/// Build one `FactorContext` per spec from a Polars DataFrame in a single
/// pass. Mirrors the per-spec label extraction in
/// `extract_factor_labels_py` but eagerly builds the `GroupMapping` so
/// downstream consumers don't pay for the rebuild on every solver call.
#[pyfunction]
#[pyo3(signature = (factors, factor_specs, separator = "\x1f"))]
pub fn build_factor_contexts_py(
    py: Python<'_>,
    factors: PyDataFrame,
    factor_specs: Vec<Vec<String>>,
    separator: &str,
) -> PyResult<Vec<PyFactorContext>> {
    let df = factors.0;
    let sep = separator.to_string();

    py.detach(|| {
        factor_specs
            .iter()
            .map(|spec| {
                let labels = build_spec_labels(&df, spec, &sep)?;
                Ok(PyFactorContext {
                    inner: Arc::new(build_group_mapping(&labels)),
                })
            })
            .collect()
    })
}

/// Build interaction labels by joining multiple string columns with a separator.
///
/// Each column is a `Vec<String>` of length n_quotes. The result is a `Vec<String>`
/// where `result[i] = columns[0][i] + sep + columns[1][i] + ...`.
#[pyfunction]
pub fn build_interaction_labels_py(
    py: Python<'_>,
    columns: Vec<Vec<String>>,
    separator: &str,
) -> Vec<String> {
    if columns.is_empty() {
        return vec![];
    }
    let n = columns[0].len();
    let sep = separator.to_owned();

    py.detach(|| {
        if n > PAR_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut s = columns[0][i].clone();
                    for col in columns.iter().skip(1) {
                        s.push_str(&sep);
                        s.push_str(&col[i]);
                    }
                    s
                })
                .collect()
        } else {
            (0..n)
                .map(|i| {
                    let mut s = columns[0][i].clone();
                    for col in columns.iter().skip(1) {
                        s.push_str(&sep);
                        s.push_str(&col[i]);
                    }
                    s
                })
                .collect()
        }
    })
}

/// Extract per-quote factor labels for one or more factor specs directly from
/// a Polars DataFrame.
///
/// Each `spec` is a list of column names; a single-column spec produces the
/// column's string-cast values; a multi-column spec produces a `Vec<String>`
/// of separator-joined values, mirroring the contract of
/// [`build_interaction_labels_py`].
///
/// **Why this exists:** the previous ratebook code did
/// `factors[col].cast(pl.Utf8).to_list()` per column. The `.to_list()` step
/// allocates a Python `PyUnicode` wrapper per element (~49 bytes overhead
/// each) before PyO3 unwraps the strings back into a Rust `Vec<String>`.
/// For 50M-quote portfolios that's gigabytes of transient Python overhead.
/// This function casts inside Rust and only ever produces the final
/// `Vec<String>`s, eliminating that wrapping cost.
///
/// **Memory caveat for multi-column interaction specs:** to build a joined
/// label per row, we cast each column to Utf8 once and hold all the casted
/// `Series` alive for the duration of the join (so the underlying string
/// buffers stay valid while we walk rows in lock-step). For a K-column
/// interaction on `N` rows, that's `K × N × avg_string_len` of casted
/// buffers held concurrently, vs the legacy Python path which processed
/// columns one at a time. For the common K=1-2 case this is fine; for
/// deep interactions on very large `N`, the Rust-side peak is higher
/// than legacy by a factor of K. The PyUnicode-elision win still
/// dominates net memory for typical workloads.
///
/// Nulls in any factor column are rejected with a clear error naming the
/// offending column. Non-string columns are auto-cast to Utf8 via Polars.
#[pyfunction]
#[pyo3(signature = (factors, factor_specs, separator = "\x1f"))]
pub fn extract_factor_labels_py(
    py: Python<'_>,
    factors: PyDataFrame,
    factor_specs: Vec<Vec<String>>,
    separator: &str,
) -> PyResult<Vec<Vec<String>>> {
    let df = factors.0;
    let sep = separator.to_string();

    py.detach(|| {
        factor_specs
            .iter()
            .map(|spec| build_spec_labels(&df, spec, &sep))
            .collect()
    })
}

/// Compute labels for a single factor spec.
///
/// One column → the string-cast values directly.
/// More than one column → per-row joined string with `separator` between fields.
fn build_spec_labels(df: &DataFrame, spec: &[String], separator: &str) -> PyResult<Vec<String>> {
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
    // the 50M-row interaction case (the original motivation for this
    // function) doesn't regress vs the previous `build_interaction_labels_py`
    // path. Below the threshold, sequential is faster than rayon's overhead.
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_build_interaction_labels_basic() {
        let columns = vec![vec!["a".into(), "b".into()], vec!["1".into(), "2".into()]];
        let result = build_interaction_labels_impl(&columns, "\x1f");
        assert_eq!(result, vec!["a\x1f1", "b\x1f2"]);
    }

    #[test]
    fn test_build_interaction_labels_three_cols() {
        let columns = vec![
            vec!["a".into(), "b".into()],
            vec!["1".into(), "2".into()],
            vec!["x".into(), "y".into()],
        ];
        let result = build_interaction_labels_impl(&columns, ":");
        assert_eq!(result, vec!["a:1:x", "b:2:y"]);
    }

    #[test]
    fn test_build_interaction_labels_empty() {
        let columns: Vec<Vec<String>> = vec![];
        let result = build_interaction_labels_impl(&columns, "\x1f");
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_interaction_labels_single_col() {
        let columns = vec![vec!["a".into(), "b".into()]];
        let result = build_interaction_labels_impl(&columns, "\x1f");
        assert_eq!(result, vec!["a", "b"]);
    }

    fn build_interaction_labels_impl(columns: &[Vec<String>], separator: &str) -> Vec<String> {
        if columns.is_empty() {
            return vec![];
        }
        let n = columns[0].len();
        (0..n)
            .map(|i| {
                let mut s = columns[0][i].clone();
                for col in columns.iter().skip(1) {
                    s.push_str(separator);
                    s.push_str(&col[i]);
                }
                s
            })
            .collect()
    }
}
