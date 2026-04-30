use crate::data::{ApplyResult, ConstraintSpec, QuoteGrid};
use crate::error::{PriceContourError, Result};

use super::argmax::{compute_lambda_signs_f32, lagrangian_argmax_pass};

/// Single-pass Lagrangian argmax with fixed lambdas (no iteration).
///
/// One forward pass with rayon-parallel argmax — no lambda updates.
/// Internal parallelism is handled by rayon grain sizes.
pub fn apply_lambdas(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    lambdas: &[f64],
) -> Result<ApplyResult> {
    grid.validate()?;

    if lambdas.len() != specs.len() {
        return Err(PriceContourError::DimensionMismatch(format!(
            "lambdas length {} != specs count {}",
            lambdas.len(),
            specs.len()
        )));
    }
    if specs.len() != grid.constraints.len() {
        return Err(PriceContourError::DimensionMismatch(format!(
            "specs count {} != grid constraints count {}",
            specs.len(),
            grid.constraints.len()
        )));
    }

    let n_quotes = grid.n_quotes;

    let lambda_signs_f32 = compute_lambda_signs_f32(specs, lambdas);

    // Single-pass argmax over all quotes (rayon parallelism inside)
    let (optimal_steps, total_objective, total_constraints) =
        lagrangian_argmax_pass(grid, &lambda_signs_f32, 0, n_quotes);

    let (baseline_objective, baseline_constraints) = grid.baseline_totals();

    Ok(ApplyResult {
        optimal_steps,
        lambdas: lambdas.to_vec(),
        total_objective,
        total_constraints,
        baseline_objective,
        baseline_constraints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::*;
    use crate::solver::solve_online;
    use approx::assert_abs_diff_eq;

    fn make_test_grid() -> (QuoteGrid, Vec<ConstraintSpec>) {
        let n = 200;
        let m = 5;
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];

        for q in 0..n {
            let elasticity = 1.0 + 4.0 * (q as f32) / (n as f32);
            let base = 50.0 + 100.0 * (q as f32) / (n as f32);
            for j in 0..m {
                let mult = 0.8 + 0.1 * j as f32;
                let conversion = 1.0 / (1.0 + (elasticity * (mult - 1.0)).exp());
                obj[q * m + j] = base * mult * conversion;
                vol[q * m + j] = conversion;
            }
        }

        let grid = QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            objective: obj,
            constraints: vec![vol],
            constraint_names: vec!["volume".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
        };

        let (_, baseline_cons) = grid.baseline_totals();
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: baseline_cons[0] * 0.90,
        }];

        (grid, specs)
    }

    #[test]
    fn test_apply_zero_lambdas_picks_max_objective() {
        let (grid, specs) = make_test_grid();
        let zero_lambdas = vec![0.0; specs.len()];
        let result = apply_lambdas(&grid, &specs, &zero_lambdas).unwrap();

        // With zero lambdas, each quote picks max-objective step (same as unconstrained)
        for q in 0..grid.n_quotes {
            let base = q * grid.n_steps;
            let best = (0..grid.n_steps)
                .max_by(|&a, &b| {
                    grid.objective[base + a]
                        .partial_cmp(&grid.objective[base + b])
                        .unwrap()
                })
                .unwrap();
            assert_eq!(result.optimal_steps[q], best as u32, "quote {q}");
        }
    }

    #[test]
    fn test_apply_with_solve_lambdas_reproduces_steps() {
        let (grid, specs) = make_test_grid();
        let config = SolverConfig {
            max_iter: 200,
            ..Default::default()
        };
        let solve_result = solve_online(&grid, &specs, &config, None).unwrap();

        let apply_result = apply_lambdas(&grid, &specs, &solve_result.lambdas).unwrap();

        // Same lambdas → same optimal steps
        assert_eq!(apply_result.optimal_steps, solve_result.optimal_steps);
        assert_abs_diff_eq!(
            apply_result.total_objective,
            solve_result.total_objective,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_apply_has_baselines() {
        let (grid, specs) = make_test_grid();
        let result = apply_lambdas(&grid, &specs, &[0.0]).unwrap();

        let (expected_obj, expected_cons) = grid.baseline_totals();
        assert_abs_diff_eq!(result.baseline_objective, expected_obj, epsilon = 1e-6);
        assert_abs_diff_eq!(
            result.baseline_constraints[0],
            expected_cons[0],
            epsilon = 1e-6
        );
    }

    // -----------------------------------------------------------------------
    // Issue 33: Error path tests for apply_lambdas
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_rejects_lambdas_specs_mismatch() {
        // lambdas.len() != specs.len() should error
        let (grid, specs) = make_test_grid();
        // Pass 2 lambdas but only 1 constraint spec
        let err = apply_lambdas(&grid, &specs, &[0.0, 0.0]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("lambdas") || msg.contains("length") || msg.contains("specs"),
            "error should mention lambdas/specs mismatch: {msg}"
        );
    }
}
