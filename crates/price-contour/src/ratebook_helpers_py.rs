use std::collections::HashMap;

use pyo3::prelude::*;
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
