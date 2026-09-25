use std::collections::HashMap;

use indexmap::IndexMap;

/// A name-keyed map that keeps insertion order across the FFI boundary, so
/// every dict the bindings return (`lambdas`, `total_constraints`, ...) lists
/// its keys in constraint order, deterministically.
pub type OrderedDict = IndexMap<String, f64>;

/// Zip constraint/label names with their corresponding values, preserving the
/// order of `names`.
pub fn zip_to_dict(names: &[String], values: &[f64]) -> OrderedDict {
    names
        .iter()
        .zip(values.iter())
        .map(|(n, &v)| (n.clone(), v))
        .collect()
}

/// Order lambda values from a name-keyed map into a Vec matching `names`.
///
/// A name missing from `lambda_dict` starts at 0.0 (the documented warm-start
/// default). A key that names no constraint is an error: it is almost always a
/// typo, and silently ignoring it would start that constraint at 0.0 instead.
///
/// Returns a plain `String` error (not a `PyErr`) so this helper stays free of
/// Python symbols and unit-testable without an interpreter; callers convert
/// with `PyValueError::new_err`.
pub fn order_lambdas(
    lambda_dict: &HashMap<String, f64>,
    names: &[String],
) -> Result<Vec<f64>, String> {
    let mut unknown: Vec<&str> = lambda_dict
        .keys()
        .filter(|key| !names.iter().any(|n| n == *key))
        .map(String::as_str)
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return Err(format!(
            "lambdas given for unknown constraint(s) {unknown:?}; known constraints: {names:?}"
        ));
    }
    Ok(names
        .iter()
        .map(|name| lambda_dict.get(name).copied().unwrap_or(0.0))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zip_to_dict_keeps_name_order() {
        let names = vec!["zeta".to_string(), "alpha".to_string(), "mid".to_string()];
        let dict = zip_to_dict(&names, &[1.0, 2.0, 3.0]);
        let keys: Vec<&String> = dict.keys().collect();
        assert_eq!(keys, vec!["zeta", "alpha", "mid"]);
        assert_eq!(dict["alpha"], 2.0);
    }

    #[test]
    fn test_zip_to_dict_empty() {
        let dict = zip_to_dict(&[], &[]);
        assert!(dict.is_empty());
    }

    #[test]
    fn test_order_lambdas() {
        let mut dict = HashMap::new();
        dict.insert("b".to_string(), 2.0);
        dict.insert("a".to_string(), 1.0);
        let names = vec!["a".to_string(), "b".to_string()];
        assert_eq!(order_lambdas(&dict, &names).unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_order_lambdas_missing_key_starts_at_zero() {
        let dict = HashMap::new();
        let names = vec!["a".to_string()];
        assert_eq!(order_lambdas(&dict, &names).unwrap(), vec![0.0]);
    }

    #[test]
    fn test_order_lambdas_rejects_unknown_key() {
        let mut dict = HashMap::new();
        dict.insert("volum".to_string(), 0.5);
        let names = vec!["volume".to_string()];
        let err = order_lambdas(&dict, &names).unwrap_err();
        assert!(err.contains("\"volum\""), "{err}");
    }
}
