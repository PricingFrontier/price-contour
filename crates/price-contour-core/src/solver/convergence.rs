use crate::data::{ConstraintDirection, ConstraintSpec};

/// Check whether all constraints are satisfied within tolerance.
///
/// For each constraint, computes an absolute tolerance as
/// `threshold * tolerance * 10` and checks the directional bound.
pub fn all_constraints_satisfied(specs: &[ConstraintSpec], totals: &[f64], tolerance: f64) -> bool {
    specs.iter().enumerate().all(|(k, spec)| {
        let tol = spec.threshold.abs() * tolerance * 10.0;
        match spec.direction {
            ConstraintDirection::Min => totals[k] >= spec.threshold - tol,
            ConstraintDirection::Max => totals[k] <= spec.threshold + tol,
        }
    })
}

/// Select the final lambda values when the solver has not converged.
///
/// If a feasible solution was found during iteration, return those lambdas.
/// Otherwise, return the ergodic average (running mean) of all lambda iterates.
pub fn select_final_lambdas(
    best_feasible_obj: f64,
    best_lambdas: Vec<f64>,
    lambda_sum: &[f64],
    iterations: usize,
) -> Vec<f64> {
    if best_feasible_obj > f64::NEG_INFINITY {
        best_lambdas
    } else {
        lambda_sum
            .iter()
            .map(|&s| s / iterations.max(1) as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ConstraintDirection, ConstraintSpec};

    #[test]
    fn test_all_satisfied_min() {
        let specs = vec![ConstraintSpec {
            name: "volume".into(),
            direction: ConstraintDirection::Min,
            threshold: 100.0,
        }];
        // 99.5 is within tolerance (100 * 1e-5 * 10 = 0.01)
        assert!(all_constraints_satisfied(&specs, &[99.995], 1e-5));
        // 98 is NOT within tolerance
        assert!(!all_constraints_satisfied(&specs, &[98.0], 1e-5));
    }

    #[test]
    fn test_all_satisfied_max() {
        let specs = vec![ConstraintSpec {
            name: "lr".into(),
            direction: ConstraintDirection::Max,
            threshold: 50.0,
        }];
        assert!(all_constraints_satisfied(&specs, &[50.0004], 1e-5));
        assert!(!all_constraints_satisfied(&specs, &[51.0], 1e-5));
    }

    #[test]
    fn test_select_final_lambdas_feasible() {
        let best = vec![1.0, 2.0];
        let sum = vec![10.0, 20.0];
        let result = select_final_lambdas(100.0, best.clone(), &sum, 10);
        assert_eq!(result, best);
    }

    #[test]
    fn test_select_final_lambdas_averaged() {
        let best = vec![1.0, 2.0];
        let sum = vec![10.0, 20.0];
        let result = select_final_lambdas(f64::NEG_INFINITY, best, &sum, 10);
        assert_eq!(result, vec![1.0, 2.0]);
    }
}
