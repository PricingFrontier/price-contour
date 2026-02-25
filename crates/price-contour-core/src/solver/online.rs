use crate::constants::*;
use crate::data::{
    ConstraintDirection, ConstraintSpec, IterationHistory, IterationRecord, QuoteGrid, SolveResult,
    SolverConfig,
};
use crate::error::Result;
use crate::solver::lambda::update_lambdas_subgradient;

use super::argmax::{compute_lambda_signs_f32, lagrangian_argmax_pass};

/// Solve the online optimisation problem via Lagrangian dual decomposition.
pub fn solve_online(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    config: &SolverConfig,
    initial_lambdas: Option<&[f64]>,
) -> Result<SolveResult> {
    grid.validate()?;

    let n_constraints = specs.len();
    let n_quotes = grid.n_quotes;

    // Compute scale factors for subgradient: ratio of objective scale to each
    // constraint's scale. This ensures the step size adapts to the magnitude
    // difference between objective and constraint values.
    let (baseline_obj, baseline_cons) = grid.baseline_totals();
    let baseline_objective = baseline_obj;
    let baseline_constraints = baseline_cons.clone();
    let obj_per_quote = baseline_obj / n_quotes.max(1) as f64;
    let scale_factors: Vec<f64> = baseline_cons
        .iter()
        .map(|&bc| {
            let con_per_quote = bc / n_quotes.max(1) as f64;
            obj_per_quote / (con_per_quote.abs() + 1e-10)
        })
        .collect();

    // Initialise lambdas
    let mut lambdas = match initial_lambdas {
        Some(init) => init.to_vec(),
        None => vec![0.0; n_constraints],
    };

    let mut optimal_steps = vec![0u32; n_quotes];
    let mut total_objective: f64 = 0.0;
    let mut total_constraints = vec![0.0f64; n_constraints];
    let mut converged = false;
    let mut iterations = 0;

    // Lambda averaging for ergodic convergence (standard fix for oscillation
    // in discrete Lagrangian relaxation — all quotes can flip simultaneously,
    // causing large swings; the running average smooths this out).
    let mut lambda_sum = vec![0.0f64; n_constraints];
    let mut best_lambdas = lambdas.clone();
    let mut best_feasible_obj = f64::NEG_INFINITY;

    let mut history_records: Vec<IterationRecord> = if config.record_history {
        Vec::with_capacity(config.max_iter)
    } else {
        Vec::new()
    };

    for iter in 0..config.max_iter {
        // Reset totals
        total_objective = 0.0;
        total_constraints.fill(0.0);

        // Pre-compute signed lambdas once per iteration (reused across chunks)
        let lambda_signs_f32 = compute_lambda_signs_f32(specs, &lambdas);

        // Process quotes in chunks
        let mut quote_offset = 0;
        while quote_offset < n_quotes {
            let chunk_end = (quote_offset + config.chunk_size).min(n_quotes);

            let (chunk_steps, chunk_obj, chunk_cons) =
                lagrangian_argmax_pass(grid, &lambda_signs_f32, quote_offset, chunk_end);

            total_objective += chunk_obj;
            for k in 0..n_constraints {
                total_constraints[k] += chunk_cons[k];
            }

            // Copy optimal steps
            optimal_steps[quote_offset..chunk_end].copy_from_slice(&chunk_steps);

            quote_offset = chunk_end;
        }

        // Track best feasible solution
        let all_satisfied = specs.iter().enumerate().all(|(k, spec)| {
            let tol = spec.threshold.abs() * CONSTRAINT_SATISFACTION_TOL;
            match spec.direction {
                ConstraintDirection::Min => total_constraints[k] >= spec.threshold - tol,
                ConstraintDirection::Max => total_constraints[k] <= spec.threshold + tol,
            }
        });

        if all_satisfied && total_objective > best_feasible_obj {
            best_feasible_obj = total_objective;
            best_lambdas = lambdas.clone();
        }

        // Update lambdas
        let max_lambda_change = update_lambdas_subgradient(
            &mut lambdas,
            specs,
            &total_constraints,
            &scale_factors,
            iter,
        );

        // Record history if requested
        if config.record_history {
            history_records.push(IterationRecord {
                iteration: iter,
                lambdas: lambdas.clone(),
                total_objective,
                total_constraints: total_constraints.clone(),
                max_lambda_change,
                all_constraints_satisfied: all_satisfied,
            });
        }

        // Accumulate for averaging
        for k in 0..n_constraints {
            lambda_sum[k] += lambdas[k];
        }

        iterations = iter + 1;

        if all_satisfied && max_lambda_change < LAMBDA_CONVERGENCE_TOL {
            converged = true;
            break;
        }
    }

    // If we didn't converge via the main loop, use the averaged lambdas
    // for a final pass to get a better solution.
    if !converged {
        // Use the best feasible lambdas if we found any, otherwise use averaged
        let final_lambdas = if best_feasible_obj > f64::NEG_INFINITY {
            best_lambdas
        } else {
            lambda_sum
                .iter()
                .map(|&s| s / iterations.max(1) as f64)
                .collect()
        };

        let lambda_signs_f32 = compute_lambda_signs_f32(specs, &final_lambdas);

        // Final argmax pass with the chosen lambdas
        total_objective = 0.0;
        total_constraints.fill(0.0);

        let mut quote_offset = 0;
        while quote_offset < n_quotes {
            let chunk_end = (quote_offset + config.chunk_size).min(n_quotes);

            let (chunk_steps, chunk_obj, chunk_cons) =
                lagrangian_argmax_pass(grid, &lambda_signs_f32, quote_offset, chunk_end);

            total_objective += chunk_obj;
            for k in 0..n_constraints {
                total_constraints[k] += chunk_cons[k];
            }
            optimal_steps[quote_offset..chunk_end].copy_from_slice(&chunk_steps);
            quote_offset = chunk_end;
        }

        lambdas = final_lambdas;
    }

    let history = if config.record_history {
        Some(IterationHistory {
            records: history_records,
        })
    } else {
        None
    };

    Ok(SolveResult {
        optimal_steps,
        lambdas,
        iterations,
        converged,
        total_objective,
        total_constraints,
        baseline_objective,
        baseline_constraints,
        history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::*;

    /// Unconstrained solve: with no constraints, every quote picks the step
    /// with highest objective.
    #[test]
    fn test_unconstrained() {
        let grid = QuoteGrid {
            n_quotes: 3,
            n_steps: 4,
            scenario_values: vec![0.9, 0.95, 1.0, 1.05],
            objective: vec![
                1.0, 3.0, 2.0, 0.5, // quote 0: best at step 1
                5.0, 4.0, 3.0, 2.0, // quote 1: best at step 0
                0.1, 0.2, 0.3, 0.9, // quote 2: best at step 3
            ],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
        };
        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };
        let result = solve_online(&grid, &[], &config, None).unwrap();
        assert_eq!(result.optimal_steps, vec![1, 0, 3]);
        assert!((result.total_objective - (3.0 + 5.0 + 0.9)).abs() < 1e-6);
    }

    /// Single min constraint with heterogeneous quotes — each has a different
    /// elasticity so different lambdas shift different quotes, enabling the
    /// subgradient to find intermediate solutions.
    #[test]
    fn test_single_min_constraint() {
        let n = 200;
        let m = 5;
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];

        for q in 0..n {
            // Varying elasticity: quotes 0..199 have elasticity 1.0..5.0
            let elasticity = 1.0 + 4.0 * (q as f32) / (n as f32);
            // Varying base premium: 50..150
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
        let threshold = baseline_cons[0] * 0.90;

        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold,
        }];

        let config = SolverConfig {
            max_iter: 200,
            ..Default::default()
        };
        let result = solve_online(&grid, &specs, &config, None).unwrap();

        // With heterogeneous quotes, some shift to lower scenario values while others
        // stay high — total volume should approximately satisfy the constraint.
        assert!(
            result.total_constraints[0] >= threshold * 0.98,
            "volume constraint not approximately satisfied: {} < {} (98% of threshold)",
            result.total_constraints[0],
            threshold * 0.98
        );
        // Objective should be less than unconstrained max
        let unconstrained_result = solve_online(&grid, &[], &SolverConfig { max_iter: 1, ..Default::default() }, None).unwrap();
        assert!(
            result.total_objective <= unconstrained_result.total_objective + 1e-6,
            "constrained objective should not exceed unconstrained"
        );
    }

    /// Warm-start: providing initial lambdas should converge faster.
    #[test]
    fn test_warm_start_converges_faster() {
        let n = 100;
        let m = 5;
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];

        for q in 0..n {
            for j in 0..m {
                let mult = 0.8 + 0.1 * j as f32;
                obj[q * m + j] = 100.0 * mult * (1.0 / (1.0 + (3.0 * (mult - 1.0)).exp()));
                vol[q * m + j] = 1.0 / (1.0 + (3.0 * (mult - 1.0)).exp());
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
            threshold: baseline_cons[0] * 0.9,
        }];

        let config = SolverConfig::default();

        // Cold start
        let cold = solve_online(&grid, &specs, &config, None).unwrap();

        // Warm start with lambdas from cold solve
        let warm = solve_online(&grid, &specs, &config, Some(&cold.lambdas)).unwrap();

        assert!(
            warm.iterations <= cold.iterations,
            "warm start ({} iters) should converge no slower than cold start ({} iters)",
            warm.iterations,
            cold.iterations
        );
    }

    #[test]
    fn test_record_history_true() {
        let grid = QuoteGrid {
            n_quotes: 3,
            n_steps: 4,
            scenario_values: vec![0.9, 0.95, 1.0, 1.05],
            objective: vec![
                1.0, 3.0, 2.0, 0.5,
                5.0, 4.0, 3.0, 2.0,
                0.1, 0.2, 0.3, 0.9,
            ],
            constraints: vec![vec![
                1.0, 0.9, 0.8, 0.7,
                1.0, 0.9, 0.8, 0.7,
                1.0, 0.9, 0.8, 0.7,
            ]],
            constraint_names: vec!["volume".to_string()],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
        };
        let (_, bc) = grid.baseline_totals();
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: bc[0] * 0.9,
        }];
        let config = SolverConfig {
            max_iter: 10,
            record_history: true,
            ..Default::default()
        };
        let result = solve_online(&grid, &specs, &config, None).unwrap();
        let history = result.history.as_ref().expect("history should be Some");
        assert_eq!(history.records.len(), result.iterations);
        assert_eq!(history.records[0].iteration, 0);
    }

    #[test]
    fn test_record_history_false() {
        let grid = QuoteGrid {
            n_quotes: 3,
            n_steps: 4,
            scenario_values: vec![0.9, 0.95, 1.0, 1.05],
            objective: vec![
                1.0, 3.0, 2.0, 0.5,
                5.0, 4.0, 3.0, 2.0,
                0.1, 0.2, 0.3, 0.9,
            ],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
        };
        let config = SolverConfig {
            max_iter: 1,
            record_history: false,
            ..Default::default()
        };
        let result = solve_online(&grid, &[], &config, None).unwrap();
        assert!(result.history.is_none());
    }

    #[test]
    fn test_baselines_on_result() {
        let grid = QuoteGrid {
            n_quotes: 2,
            n_steps: 3,
            scenario_values: vec![0.9, 1.0, 1.1],
            objective: vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec!["Q0".into(), "Q1".into()],
        };
        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };
        let result = solve_online(&grid, &[], &config, None).unwrap();
        // Baseline at step 1 (mult=1.0): 20 + 50 = 70
        assert!((result.baseline_objective - 70.0).abs() < 1e-6);
    }
}
