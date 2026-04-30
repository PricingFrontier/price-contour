use crate::constants::{SUBGRADIENT_ALPHA, SUBGRADIENT_MIN_STEP};
use crate::data::{ConstraintDirection, ConstraintSpec};

/// Update lambdas using subgradient method with per-constraint scale factors.
///
/// `scale_factors[k]` adjusts the step size for constraint k, compensating for
/// the ratio between objective and constraint magnitudes. Computed once from
/// baseline totals as `(baseline_obj / n_quotes) / (baseline_con_k / n_quotes + eps)`.
///
/// `baseline_constraints[k]` provides a fallback magnitude for synthetic
/// constraints with `threshold == 0` (e.g. ratio linearisation), where the
/// threshold itself carries no scale information; without this fallback the
/// effective step would be dominated by `scale_factors[k]` alone, producing
/// large oscillations.
///
/// Sign convention:
/// - Min constraint (total >= threshold): gap = threshold - total (positive when violated)
/// - Max constraint (total <= threshold): gap = total - threshold (positive when violated)
///
/// Returns the maximum absolute change across all lambdas.
pub fn update_lambdas_subgradient(
    lambdas: &mut [f64],
    specs: &[ConstraintSpec],
    totals: &[f64],
    baseline_constraints: &[f64],
    scale_factors: &[f64],
    iteration: usize,
) -> f64 {
    let base_step = (SUBGRADIENT_ALPHA / ((iteration + 1) as f64).sqrt()).max(SUBGRADIENT_MIN_STEP);
    let mut max_change: f64 = 0.0;

    for (k, spec) in specs.iter().enumerate() {
        let gap = match spec.direction {
            ConstraintDirection::Min => spec.threshold - totals[k],
            ConstraintDirection::Max => totals[k] - spec.threshold,
        };

        // Normalize gap: divide by threshold magnitude, multiply by scale factor.
        // This makes the step size independent of the absolute scale of the
        // constraint values, working equally well whether volume is ~0.5 per
        // quote or ~500 per quote. For synthetic constraints (threshold=0,
        // e.g. ratio linearisation) we fall back to the baseline magnitude
        // to recover an analogous normalisation.
        let threshold_scale = spec
            .threshold
            .abs()
            .max(baseline_constraints[k].abs())
            .max(1.0);
        let effective_step = base_step * scale_factors[k] / threshold_scale;

        let old = lambdas[k];
        lambdas[k] = (lambdas[k] + effective_step * gap).max(0.0);
        max_change = max_change.max((lambdas[k] - old).abs());
    }

    max_change
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scales(n: usize) -> Vec<f64> {
        vec![1.0; n]
    }

    #[test]
    fn test_min_constraint_violated() {
        let mut lambdas = vec![0.0];
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 100.0,
        }];
        let totals = vec![90.0];
        let change = update_lambdas_subgradient(
            &mut lambdas,
            &specs,
            &totals,
            &[100.0],
            &default_scales(1),
            0,
        );
        assert!(
            lambdas[0] > 0.0,
            "lambda should increase when min constraint is violated"
        );
        assert!(change > 0.0);
    }

    #[test]
    fn test_min_constraint_satisfied() {
        let mut lambdas = vec![1.0];
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 100.0,
        }];
        let totals = vec![110.0];
        update_lambdas_subgradient(
            &mut lambdas,
            &specs,
            &totals,
            &[100.0],
            &default_scales(1),
            0,
        );
        assert!(
            lambdas[0] < 1.0,
            "lambda should decrease when min constraint is satisfied"
        );
    }

    #[test]
    fn test_max_constraint_violated() {
        let mut lambdas = vec![0.0];
        let specs = vec![ConstraintSpec {
            name: "loss_ratio".to_string(),
            direction: ConstraintDirection::Max,
            threshold: 100.0,
        }];
        let totals = vec![110.0];
        update_lambdas_subgradient(
            &mut lambdas,
            &specs,
            &totals,
            &[100.0],
            &default_scales(1),
            0,
        );
        assert!(
            lambdas[0] > 0.0,
            "lambda should increase when max constraint is violated"
        );
    }

    #[test]
    fn test_max_constraint_satisfied() {
        let mut lambdas = vec![1.0];
        let specs = vec![ConstraintSpec {
            name: "loss_ratio".to_string(),
            direction: ConstraintDirection::Max,
            threshold: 100.0,
        }];
        let totals = vec![90.0];
        update_lambdas_subgradient(
            &mut lambdas,
            &specs,
            &totals,
            &[100.0],
            &default_scales(1),
            0,
        );
        assert!(lambdas[0] < 1.0);
    }

    #[test]
    fn test_non_negative_projection() {
        let mut lambdas = vec![0.01];
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 100.0,
        }];
        let totals = vec![1_000_000.0];
        update_lambdas_subgradient(
            &mut lambdas,
            &specs,
            &totals,
            &[100.0],
            &default_scales(1),
            0,
        );
        assert!(
            lambdas[0] >= 0.0,
            "lambda must be non-negative, got {}",
            lambdas[0]
        );
    }

    #[test]
    fn test_diminishing_step_size() {
        let specs = vec![ConstraintSpec {
            name: "v".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 100.0,
        }];
        let totals = vec![90.0];

        let mut lambdas_early = vec![0.0];
        update_lambdas_subgradient(
            &mut lambdas_early,
            &specs,
            &totals,
            &[100.0],
            &default_scales(1),
            0,
        );

        let mut lambdas_late = vec![0.0];
        update_lambdas_subgradient(
            &mut lambdas_late,
            &specs,
            &totals,
            &[100.0],
            &default_scales(1),
            99,
        );

        assert!(
            lambdas_early[0] > lambdas_late[0],
            "earlier iteration should produce larger update"
        );
    }

    #[test]
    fn test_scale_factor_amplifies() {
        let specs = vec![ConstraintSpec {
            name: "v".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 100.0,
        }];
        let totals = vec![90.0];

        let mut lambdas_scaled = vec![0.0];
        update_lambdas_subgradient(&mut lambdas_scaled, &specs, &totals, &[100.0], &[10.0], 0);

        let mut lambdas_unscaled = vec![0.0];
        update_lambdas_subgradient(&mut lambdas_unscaled, &specs, &totals, &[100.0], &[1.0], 0);

        assert!(
            lambdas_scaled[0] > lambdas_unscaled[0],
            "higher scale factor should produce larger lambda update"
        );
    }

    #[test]
    fn test_threshold_zero_falls_back_to_baseline_magnitude() {
        // Synthetic ratio-linearisation constraint: threshold=0. Without the
        // baseline-magnitude fallback, threshold_scale collapses to 1.0 and
        // the effective step explodes by ~scale_factors[k]. With the fix,
        // step size is bounded by max(0, |baseline|, 1.0).
        let mut lambdas = vec![0.0];
        let specs = vec![ConstraintSpec {
            name: "loss_ratio".to_string(),
            direction: ConstraintDirection::Max,
            threshold: 0.0,
        }];
        let totals = vec![10.0]; // synthetic constraint violated by 10
        let baseline = vec![100.0]; // underlying scale
        let scale_factors = vec![1.0];
        update_lambdas_subgradient(&mut lambdas, &specs, &totals, &baseline, &scale_factors, 0);
        // Without baseline fallback, threshold_scale would be 1.0 and the
        // step would be ~SUBGRADIENT_ALPHA = 0.1 * 1 * 10 = 1.0.
        // With baseline fallback, threshold_scale = 100 → step ≈ 0.01.
        assert!(
            lambdas[0] < 0.5,
            "with baseline fallback, lambda step should be modest (got {})",
            lambdas[0]
        );
    }
}
