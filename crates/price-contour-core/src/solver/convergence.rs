use crate::constants::{
    RATIO_LINEARISED_TOLERANCE_MULTIPLIER, SUM_TOLERANCE_MULTIPLIER, ZERO_THRESHOLD_EPSILON,
};
use crate::data::{ConstraintDirection, ConstraintSpec};

/// Check whether all constraints are satisfied within tolerance.
///
/// For sum constraints (`threshold.abs() >= ZERO_THRESHOLD_EPSILON`),
/// uses `|threshold| * tolerance * SUM_TOLERANCE_MULTIPLIER` as the
/// absolute tolerance — relative to the threshold magnitude, matching
/// the long-standing solver contract.
///
/// For synthetic constraints with `threshold.abs() < ZERO_THRESHOLD_EPSILON`
/// (e.g. the ratio-constraint linearisation `Σ (num − L·denom) ≤ 0`),
/// the threshold itself carries no scale information. The natural
/// scale of the values being summed is the baseline magnitude, but the
/// linearised baseline can be small for ratio targets near baseline LR.
/// The tolerance therefore uses
/// `max(|baseline_total|, 1.0) * tolerance * RATIO_LINEARISED_TOLERANCE_MULTIPLIER`
/// — a generous constant matching the typical ratio of underlying
/// values (`Σ num`, `Σ denom`) to the synthetic baseline
/// (`(baseline_LR − L) * Σ denom`), which is commonly 100–1000× for
///   binding ratio targets near baseline.
pub fn all_constraints_satisfied(
    specs: &[ConstraintSpec],
    totals: &[f64],
    baseline_constraints: &[f64],
    tolerance: f64,
) -> bool {
    specs.iter().enumerate().all(|(k, spec)| {
        let tol = if spec.threshold.abs() < ZERO_THRESHOLD_EPSILON {
            baseline_constraints[k].abs().max(1.0)
                * tolerance
                * RATIO_LINEARISED_TOLERANCE_MULTIPLIER
        } else {
            spec.threshold.abs() * tolerance * SUM_TOLERANCE_MULTIPLIER
        };
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
        let baseline = vec![100.0];
        // 99.5 is within tolerance (100 * 1e-5 * 10 = 0.01)
        assert!(all_constraints_satisfied(&specs, &[99.995], &baseline, 1e-5));
        // 98 is NOT within tolerance
        assert!(!all_constraints_satisfied(&specs, &[98.0], &baseline, 1e-5));
    }

    #[test]
    fn test_all_satisfied_max() {
        let specs = vec![ConstraintSpec {
            name: "lr".into(),
            direction: ConstraintDirection::Max,
            threshold: 50.0,
        }];
        let baseline = vec![50.0];
        assert!(all_constraints_satisfied(&specs, &[50.0004], &baseline, 1e-5));
        assert!(!all_constraints_satisfied(&specs, &[51.0], &baseline, 1e-5));
    }

    #[test]
    fn test_all_satisfied_zero_threshold_uses_generous_baseline_scale() {
        // Synthetic ratio-linearisation constraint: threshold=0, baseline
        // values O(100). The synthetic baseline is small relative to the
        // underlying num/denom scales, so we apply a generous *1000
        // multiplier instead of *10. Tolerance becomes
        // 100 * 1e-5 * 1000 = 1.0 — within reach of discrete
        // optimisation noise.
        let specs = vec![ConstraintSpec {
            name: "loss_ratio".into(),
            direction: ConstraintDirection::Max,
            threshold: 0.0,
        }];
        let baseline = vec![100.0];
        // 0.5 is well within tolerance (1.0 absolute)
        assert!(all_constraints_satisfied(&specs, &[0.5], &baseline, 1e-5));
        // 50 is NOT within tolerance
        assert!(!all_constraints_satisfied(&specs, &[50.0], &baseline, 1e-5));
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
