use crate::constants::*;
use crate::data::{ApplyResult, ConstraintSpec, QuoteGrid};
use crate::error::Result;

use super::argmax::{compute_lambda_signs_f32, lagrangian_argmax_pass};

/// Single-pass Lagrangian argmax with fixed lambdas (no iteration).
///
/// Same parallel pattern as the solve loop, but just one forward pass —
/// no lambda updates.
pub fn apply_lambdas(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    lambdas: &[f64],
    chunk_size: Option<usize>,
) -> Result<ApplyResult> {
    grid.validate()?;

    let n_quotes = grid.n_quotes;
    let n_constraints = specs.len();
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);

    let lambda_signs_f32 = compute_lambda_signs_f32(specs, lambdas);

    let mut optimal_steps = vec![0u32; n_quotes];
    let mut total_objective: f64 = 0.0;
    let mut total_constraints = vec![0.0f64; n_constraints];

    // Process quotes in chunks
    let mut quote_offset = 0;
    while quote_offset < n_quotes {
        let chunk_end = (quote_offset + chunk_size).min(n_quotes);

        let (chunk_steps, chunk_obj, chunk_cons) =
            lagrangian_argmax_pass(grid, &lambda_signs_f32, quote_offset, chunk_end);

        total_objective += chunk_obj;
        for k in 0..n_constraints {
            total_constraints[k] += chunk_cons[k];
        }

        optimal_steps[quote_offset..chunk_end].copy_from_slice(&chunk_steps);
        quote_offset = chunk_end;
    }

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
        let result = apply_lambdas(&grid, &specs, &zero_lambdas, None).unwrap();

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

        let apply_result =
            apply_lambdas(&grid, &specs, &solve_result.lambdas, None).unwrap();

        // Same lambdas → same optimal steps
        assert_eq!(apply_result.optimal_steps, solve_result.optimal_steps);
        assert!((apply_result.total_objective - solve_result.total_objective).abs() < 1e-6);
    }

    #[test]
    fn test_apply_has_baselines() {
        let (grid, specs) = make_test_grid();
        let result = apply_lambdas(&grid, &specs, &vec![0.0], None).unwrap();

        let (expected_obj, expected_cons) = grid.baseline_totals();
        assert!((result.baseline_objective - expected_obj).abs() < 1e-6);
        assert!((result.baseline_constraints[0] - expected_cons[0]).abs() < 1e-6);
    }
}
