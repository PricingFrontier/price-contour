use crate::data::{
    ConstraintSpec, IterationHistory, IterationRecord, QuoteGrid, SolveResult, SolverConfig,
};
use crate::error::{PriceContourError, Result};
use crate::solver::convergence::{all_constraints_satisfied, select_final_lambdas};
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

    if specs.len() != grid.constraints.len() {
        return Err(PriceContourError::DimensionMismatch(format!(
            "specs count {} != grid constraints count {}",
            specs.len(),
            grid.constraints.len()
        )));
    }
    if let Some(il) = initial_lambdas {
        if il.len() != specs.len() {
            return Err(PriceContourError::DimensionMismatch(format!(
                "initial_lambdas length {} != specs count {}",
                il.len(),
                specs.len()
            )));
        }
    }

    let (baseline_obj, baseline_cons, scale_factors) = grid.compute_scale_factors();
    solve_online_with_precomputed(
        grid,
        specs,
        config,
        initial_lambdas,
        baseline_obj,
        baseline_cons,
        scale_factors,
    )
}

/// Internal solve variant that accepts pre-computed baselines and scale factors.
///
/// This avoids redundant `compute_scale_factors()` calls when the caller
/// (e.g. `sweep_frontier`) invokes the solver many times on the same grid.
pub(crate) fn solve_online_with_precomputed(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    config: &SolverConfig,
    initial_lambdas: Option<&[f64]>,
    baseline_obj: f64,
    baseline_cons: Vec<f64>,
    scale_factors: Vec<f64>,
) -> Result<SolveResult> {
    let n_constraints = specs.len();
    let n_quotes = grid.n_quotes;

    let baseline_objective = baseline_obj;
    let baseline_constraints = baseline_cons;

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
        // Pre-compute signed lambdas once per iteration
        let lambda_signs_f32 = compute_lambda_signs_f32(specs, &lambdas);

        // Single-pass argmax over all quotes (rayon parallelism inside)
        let (all_steps, iter_obj, iter_cons) =
            lagrangian_argmax_pass(grid, &lambda_signs_f32, 0, n_quotes);

        optimal_steps.copy_from_slice(&all_steps);
        total_objective = iter_obj;
        total_constraints = iter_cons;

        // Track best feasible solution
        let all_satisfied = all_constraints_satisfied(
            specs,
            &total_constraints,
            &baseline_constraints,
            config.tolerance,
        );

        if all_satisfied && total_objective > best_feasible_obj {
            best_feasible_obj = total_objective;
            best_lambdas = lambdas.clone();
        }

        // Accumulate for averaging (before update, so we average the lambdas
        // that were actually used for this iteration's argmax pass)
        for k in 0..n_constraints {
            lambda_sum[k] += lambdas[k];
        }

        // Clone lambdas before update for history recording
        let pre_update_lambdas = if config.record_history {
            Some(lambdas.clone())
        } else {
            None
        };

        // Update lambdas
        let max_lambda_change = update_lambdas_subgradient(
            &mut lambdas,
            specs,
            &total_constraints,
            &baseline_constraints,
            &scale_factors,
            iter,
        );

        // Record history if requested
        if let Some(hist_lambdas) = pre_update_lambdas {
            history_records.push(IterationRecord {
                iteration: iter,
                lambdas: hist_lambdas,
                total_objective,
                total_constraints: total_constraints.clone(),
                max_lambda_change,
                all_constraints_satisfied: all_satisfied,
            });
        }

        iterations = iter + 1;

        if all_satisfied && max_lambda_change < config.tolerance {
            converged = true;
            break;
        }
    }

    // If we didn't converge via the main loop, use the averaged lambdas
    // for a final pass to get a better solution.
    if !converged {
        let final_lambdas =
            select_final_lambdas(best_feasible_obj, best_lambdas, &lambda_sum, iterations);

        let lambda_signs_f32 = compute_lambda_signs_f32(specs, &final_lambdas);

        // Final single-pass argmax with the chosen lambdas
        let (all_steps, final_obj, final_cons) =
            lagrangian_argmax_pass(grid, &lambda_signs_f32, 0, n_quotes);

        optimal_steps.copy_from_slice(&all_steps);
        total_objective = final_obj;
        total_constraints = final_cons;

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
    use approx::assert_abs_diff_eq;

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
            quote_id_fingerprint: 0,
        };
        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };
        let result = solve_online(&grid, &[], &config, None).unwrap();
        assert_eq!(result.optimal_steps, vec![1, 0, 3]);
        assert_abs_diff_eq!(result.total_objective, 3.0 + 5.0 + 0.9, epsilon = 1e-6);
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
            quote_id_fingerprint: 0,
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
        // Objective should be less than unconstrained max.
        // Build a grid without constraints for the unconstrained comparison.
        let unconstrained_grid = QuoteGrid {
            n_quotes: grid.n_quotes,
            n_steps: grid.n_steps,
            scenario_values: grid.scenario_values.clone(),
            objective: grid.objective.clone(),
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: grid.quote_ids.clone(),
            quote_id_fingerprint: 0,
        };
        let unconstrained_result = solve_online(
            &unconstrained_grid,
            &[],
            &SolverConfig {
                max_iter: 1,
                ..Default::default()
            },
            None,
        )
        .unwrap();
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
            quote_id_fingerprint: 0,
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
            objective: vec![1.0, 3.0, 2.0, 0.5, 5.0, 4.0, 3.0, 2.0, 0.1, 0.2, 0.3, 0.9],
            constraints: vec![vec![
                1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7,
            ]],
            constraint_names: vec!["volume".to_string()],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
            quote_id_fingerprint: 0,
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
            objective: vec![1.0, 3.0, 2.0, 0.5, 5.0, 4.0, 3.0, 2.0, 0.1, 0.2, 0.3, 0.9],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
            quote_id_fingerprint: 0,
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
            quote_id_fingerprint: 0,
        };
        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };
        let result = solve_online(&grid, &[], &config, None).unwrap();
        // Baseline at step 1 (mult=1.0): 20 + 50 = 70
        assert_abs_diff_eq!(result.baseline_objective, 70.0, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // Issue 32: Multi-constraint test (min volume + max loss_ratio)
    // -----------------------------------------------------------------------

    /// Build a heterogeneous grid with 2 constraints (volume and loss_ratio),
    /// used by multi-constraint tests.
    fn make_two_constraint_grid(n: usize, m: usize) -> QuoteGrid {
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];
        let mut lr = vec![0.0f32; n * m];

        for q in 0..n {
            let elasticity = 1.0 + 4.0 * (q as f32) / (n as f32);
            let base = 50.0 + 100.0 * (q as f32) / (n as f32);
            for j in 0..m {
                let mult = 0.8 + 0.1 * j as f32;
                let conversion = 1.0 / (1.0 + (elasticity * (mult - 1.0)).exp());
                obj[q * m + j] = base * mult * conversion;
                vol[q * m + j] = conversion;
                lr[q * m + j] = 0.6 / mult * (1.0 + 0.1 * (mult - 1.0));
            }
        }

        QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            objective: obj,
            constraints: vec![vol, lr],
            constraint_names: vec!["volume".to_string(), "loss_ratio".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    #[test]
    fn test_two_constraints_min_and_max() {
        // 30 quotes, 5 steps, 2 constraints: volume (Min) and loss_ratio (Max).
        let n = 30;
        let m = 5;
        let grid = make_two_constraint_grid(n, m);

        let (_, baseline_cons) = grid.baseline_totals();
        let vol_threshold = baseline_cons[0] * 0.90;
        let lr_threshold = baseline_cons[1] * 1.10;

        let specs = vec![
            ConstraintSpec {
                name: "volume".to_string(),
                direction: ConstraintDirection::Min,
                threshold: vol_threshold,
            },
            ConstraintSpec {
                name: "loss_ratio".to_string(),
                direction: ConstraintDirection::Max,
                threshold: lr_threshold,
            },
        ];

        let config = SolverConfig {
            max_iter: 300,
            ..Default::default()
        };
        let result = solve_online(&grid, &specs, &config, None).unwrap();

        // Should produce a valid result with correct structure
        assert_eq!(result.lambdas.len(), 2);
        assert_eq!(result.total_constraints.len(), 2);
        assert!(result.iterations > 0);

        // Volume should approximately satisfy the min constraint (allow 5% slack
        // because discrete Lagrangian relaxation on small grids may not converge
        // perfectly with multiple constraints).
        assert!(
            result.total_constraints[0] >= vol_threshold * 0.95,
            "volume constraint not approximately satisfied: {} < {} (95% of threshold {})",
            result.total_constraints[0],
            vol_threshold * 0.95,
            vol_threshold
        );
    }

    // -----------------------------------------------------------------------
    // Issue 33: Error path tests for solve_online
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_rejects_specs_grid_mismatch() {
        // specs.len() != grid.constraints.len() should error
        let grid = QuoteGrid {
            n_quotes: 3,
            n_steps: 3,
            scenario_values: vec![0.9, 1.0, 1.1],
            objective: vec![1.0; 9],
            constraints: vec![vec![1.0; 9]], // 1 constraint
            constraint_names: vec!["volume".to_string()],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
            quote_id_fingerprint: 0,
        };

        // 2 specs but only 1 constraint in grid
        let specs = vec![
            ConstraintSpec {
                name: "volume".to_string(),
                direction: ConstraintDirection::Min,
                threshold: 1.0,
            },
            ConstraintSpec {
                name: "extra".to_string(),
                direction: ConstraintDirection::Max,
                threshold: 1.0,
            },
        ];

        let config = SolverConfig::default();
        let err = solve_online(&grid, &specs, &config, None).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("specs count") || msg.contains("mismatch"),
            "error should mention specs/grid mismatch: {msg}"
        );
    }

    #[test]
    fn test_solve_rejects_wrong_initial_lambdas_len() {
        // initial_lambdas with wrong length should error
        let grid = QuoteGrid {
            n_quotes: 3,
            n_steps: 3,
            scenario_values: vec![0.9, 1.0, 1.1],
            objective: vec![1.0; 9],
            constraints: vec![vec![1.0; 9]],
            constraint_names: vec!["volume".to_string()],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
            quote_id_fingerprint: 0,
        };

        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 1.0,
        }];

        let config = SolverConfig::default();
        // Pass 2 lambdas but only 1 constraint
        let err = solve_online(&grid, &specs, &config, Some(&[0.0, 0.0])).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("initial_lambdas") || msg.contains("length"),
            "error should mention initial_lambdas length: {msg}"
        );
    }
}
