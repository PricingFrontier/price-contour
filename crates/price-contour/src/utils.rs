use std::collections::HashMap;

/// Zip constraint/label names with their corresponding values into a HashMap.
///
/// Used throughout the PyO3 bindings to convert parallel Vec<String> + Vec<f64>
/// into the Python-friendly Dict[str, float] representation.
pub fn zip_to_dict(names: &[String], values: &[f64]) -> HashMap<String, f64> {
    names
        .iter()
        .zip(values.iter())
        .map(|(n, &v)| (n.clone(), v))
        .collect()
}

/// Order lambda values from a name-keyed HashMap into a Vec matching the
/// given `names` ordering. Missing keys default to 0.0.
pub fn order_lambdas(lambda_dict: &HashMap<String, f64>, names: &[String]) -> Vec<f64> {
    names
        .iter()
        .map(|name| *lambda_dict.get(name).unwrap_or(&0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zip_to_dict() {
        let names = vec!["a".to_string(), "b".to_string()];
        let values = vec![1.0, 2.0];
        let dict = zip_to_dict(&names, &values);
        assert_eq!(dict.len(), 2);
        assert_eq!(dict["a"], 1.0);
        assert_eq!(dict["b"], 2.0);
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
        let ordered = order_lambdas(&dict, &names);
        assert_eq!(ordered, vec![1.0, 2.0]);
    }

    #[test]
    fn test_order_lambdas_missing_key() {
        let dict = HashMap::new();
        let names = vec!["a".to_string()];
        let ordered = order_lambdas(&dict, &names);
        assert_eq!(ordered, vec![0.0]);
    }
}
