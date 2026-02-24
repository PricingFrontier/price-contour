use rayon::prelude::*;

use crate::constants::*;
use crate::data::{ApplyResult, ConstraintDirection, ConstraintSpec, QuoteGrid};
use crate::error::Result;

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
    let n_steps = grid.n_steps;
    let n_constraints = specs.len();
    let chunk_size = chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE);

    // Pre-compute lambda signs: +1 for Min, -1 for Max
    let lambda_signs: Vec<f64> = specs
        .iter()
        .zip(lambdas.iter())
        .map(|(spec, &lam)| match spec.direction {
            ConstraintDirection::Min => lam,
            ConstraintDirection::Max => -lam,
        })
        .collect();

    let mut optimal_steps = vec![0u32; n_quotes];
    let mut total_objective: f64 = 0.0;
    let mut total_constraints = vec![0.0f64; n_constraints];

    // Process quotes in chunks
    let mut quote_offset = 0;
    while quote_offset < n_quotes {
        let chunk_end = (quote_offset + chunk_size).min(n_quotes);
        let chunk_len = chunk_end - quote_offset;

        let mut chunk_steps = vec![0u32; chunk_len];

        const PAR_GRAIN: usize = 4096;

        let partials: Vec<(f64, Vec<f64>)> = chunk_steps
            .par_chunks_mut(PAR_GRAIN)
            .enumerate()
            .map(|(grain_idx, step_slice)| {
                let sub_start = quote_offset + grain_idx * PAR_GRAIN;
                let sub_len = step_slice.len();
                let mut partial_obj: f64 = 0.0;
                let mut partial_cons = vec![0.0f64; n_constraints];

                for local_i in 0..sub_len {
                    let q = sub_start + local_i;
                    let base = q * n_steps;

                    let mut best_step: usize = 0;
                    let mut best_lagrangian = f64::NEG_INFINITY;

                    for j in 0..n_steps {
                        let idx = base + j;
                        let mut l = grid.objective[idx] as f64;
                        for (k, &sign_lam) in lambda_signs.iter().enumerate() {
                            l += sign_lam * grid.constraints[k][idx] as f64;
                        }
                        if l > best_lagrangian {
                            best_lagrangian = l;
                            best_step = j;
                        }
                    }

                    step_slice[local_i] = best_step as u32;

                    let opt_idx = base + best_step;
                    partial_obj += grid.objective[opt_idx] as f64;
                    for k in 0..n_constraints {
                        partial_cons[k] += grid.constraints[k][opt_idx] as f64;
                    }
                }

                (partial_obj, partial_cons)
            })
            .collect();

        // Reduce
        for (p_obj, p_cons) in &partials {
            total_objective += p_obj;
            for k in 0..n_constraints {
                total_constraints[k] += p_cons[k];
            }
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
