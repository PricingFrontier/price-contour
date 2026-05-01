use std::collections::HashMap;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

/// Threshold above which we switch to Rayon parallel iteration.
const PAR_THRESHOLD: usize = 100_000;

/// Compute residuals: for each quote, residual = overall_mult[i] / factor_table[label[i]].
///
/// If the factor value is 0, the residual defaults to 1.0.
/// Missing keys in `factor_table` default to 1.0.
#[pyfunction]
pub fn compute_residuals_py(
    py: Python<'_>,
    overall_mult: Vec<f32>,
    group_labels: Vec<String>,
    factor_table: HashMap<String, f64>,
) -> Vec<f32> {
    let n = overall_mult.len();
    py.detach(|| {
        if n > PAR_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let om = overall_mult[i];
                    let fv = *factor_table.get(&group_labels[i]).unwrap_or(&1.0) as f32;
                    if fv != 0.0 {
                        om / fv
                    } else {
                        1.0
                    }
                })
                .collect()
        } else {
            overall_mult
                .iter()
                .zip(group_labels.iter())
                .map(|(&om, label)| {
                    let fv = *factor_table.get(label).unwrap_or(&1.0) as f32;
                    if fv != 0.0 {
                        om / fv
                    } else {
                        1.0
                    }
                })
                .collect()
        }
    })
}

/// Update multipliers: for each quote, new_mult = old_mult / old_fv * new_fv.
///
/// If old_fv is 0, new_mult = new_fv.
/// Missing keys in `old_table` default to 0.0; in `new_table` default to 1.0.
#[pyfunction]
pub fn update_multipliers_py(
    py: Python<'_>,
    overall_mult: Vec<f32>,
    group_labels: Vec<String>,
    old_table: HashMap<String, f64>,
    new_table: HashMap<String, f64>,
) -> Vec<f32> {
    let n = overall_mult.len();
    py.detach(|| {
        if n > PAR_THRESHOLD {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let om = overall_mult[i];
                    let label = &group_labels[i];
                    let fv_old = *old_table.get(label).unwrap_or(&0.0) as f32;
                    let fv_new = *new_table.get(label).unwrap_or(&1.0) as f32;
                    if fv_old != 0.0 {
                        om / fv_old * fv_new
                    } else {
                        fv_new
                    }
                })
                .collect()
        } else {
            overall_mult
                .iter()
                .zip(group_labels.iter())
                .map(|(&om, label)| {
                    let fv_old = *old_table.get(label).unwrap_or(&0.0) as f32;
                    let fv_new = *new_table.get(label).unwrap_or(&1.0) as f32;
                    if fv_old != 0.0 {
                        om / fv_old * fv_new
                    } else {
                        fv_new
                    }
                })
                .collect()
        }
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
    use super::*;

    #[test]
    fn test_compute_residuals_basic() {
        let overall_mult = vec![2.0f32, 3.0, 4.0];
        let labels = vec!["a".into(), "b".into(), "a".into()];
        let mut table = HashMap::new();
        table.insert("a".to_string(), 2.0);
        table.insert("b".to_string(), 1.5);

        // a: 2.0 / 2.0 = 1.0, b: 3.0 / 1.5 = 2.0, a: 4.0 / 2.0 = 2.0
        let result = compute_residuals_impl(&overall_mult, &labels, &table);
        assert_eq!(result, vec![1.0f32, 2.0, 2.0]);
    }

    #[test]
    fn test_compute_residuals_missing_label() {
        let overall_mult = vec![2.0f32];
        let labels = vec!["missing".into()];
        let table = HashMap::new();

        // missing key defaults to 1.0: 2.0 / 1.0 = 2.0
        let result = compute_residuals_impl(&overall_mult, &labels, &table);
        assert_eq!(result, vec![2.0f32]);
    }

    #[test]
    fn test_compute_residuals_zero_fv() {
        let overall_mult = vec![2.0f32];
        let labels = vec!["z".into()];
        let mut table = HashMap::new();
        table.insert("z".to_string(), 0.0);

        // zero factor value => residual = 1.0
        let result = compute_residuals_impl(&overall_mult, &labels, &table);
        assert_eq!(result, vec![1.0f32]);
    }

    #[test]
    fn test_update_multipliers_basic() {
        let overall_mult = vec![2.0f32, 3.0];
        let labels = vec!["a".into(), "b".into()];
        let mut old_table = HashMap::new();
        old_table.insert("a".to_string(), 1.0);
        old_table.insert("b".to_string(), 1.5);
        let mut new_table = HashMap::new();
        new_table.insert("a".to_string(), 2.0);
        new_table.insert("b".to_string(), 3.0);

        // a: 2.0 / 1.0 * 2.0 = 4.0, b: 3.0 / 1.5 * 3.0 = 6.0
        let result = update_multipliers_impl(&overall_mult, &labels, &old_table, &new_table);
        assert_eq!(result, vec![4.0f32, 6.0]);
    }

    #[test]
    fn test_update_multipliers_zero_old() {
        let overall_mult = vec![2.0f32];
        let labels = vec!["a".into()];
        let mut old_table = HashMap::new();
        old_table.insert("a".to_string(), 0.0);
        let mut new_table = HashMap::new();
        new_table.insert("a".to_string(), 5.0);

        // old is 0.0 => result = new_fv = 5.0
        let result = update_multipliers_impl(&overall_mult, &labels, &old_table, &new_table);
        assert_eq!(result, vec![5.0f32]);
    }

    #[test]
    fn test_update_multipliers_missing_old() {
        let overall_mult = vec![2.0f32];
        let labels = vec!["missing".into()];
        let old_table = HashMap::new();
        let new_table = HashMap::new();

        // missing old defaults to 0.0 => result = new_fv default = 1.0
        let result = update_multipliers_impl(&overall_mult, &labels, &old_table, &new_table);
        assert_eq!(result, vec![1.0f32]);
    }

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

    // Pure-Rust helpers for testing without Python GIL

    fn compute_residuals_impl(
        overall_mult: &[f32],
        group_labels: &[String],
        factor_table: &HashMap<String, f64>,
    ) -> Vec<f32> {
        overall_mult
            .iter()
            .zip(group_labels.iter())
            .map(|(&om, label)| {
                let fv = *factor_table.get(label).unwrap_or(&1.0) as f32;
                if fv != 0.0 {
                    om / fv
                } else {
                    1.0
                }
            })
            .collect()
    }

    fn update_multipliers_impl(
        overall_mult: &[f32],
        group_labels: &[String],
        old_table: &HashMap<String, f64>,
        new_table: &HashMap<String, f64>,
    ) -> Vec<f32> {
        overall_mult
            .iter()
            .zip(group_labels.iter())
            .map(|(&om, label)| {
                let fv_old = *old_table.get(label).unwrap_or(&0.0) as f32;
                let fv_new = *new_table.get(label).unwrap_or(&1.0) as f32;
                if fv_old != 0.0 {
                    om / fv_old * fv_new
                } else {
                    fv_new
                }
            })
            .collect()
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
