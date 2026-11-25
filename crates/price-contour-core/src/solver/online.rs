use crate::data::{
    ConstraintSpec, IterationHistory, IterationRecord, QuoteGrid, SolveResult, SolverConfig,
};
use crate::error::{PriceContourError, Result};
use crate::frontier::{
    bisection_residual_tolerance, solve_point_1d_bisection, NonConvergenceReason, SolveHistory,
    SolverPath,
};
use crate::solver::convergence::{all_constraints_satisfied, select_final_lambdas};
use crate::solver::lambda::update_lambdas_subgradient;

use super::argmax::{compute_lambda_signs_f32, lagrangian_argmax_pass};

/// Solve the online optimisation problem via Lagrangian dual decomposition.
///
/// Exactly one constraint is solved by bisecting its λ, as a 1-D frontier
/// point is (DESIGN_DECISIONS §20): its total is monotone in λ, and the
/// subgradient method can stop far short of a bound the grid can meet. Any
/// other number of constraints runs the subgradient method.
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

    if let [spec] = specs {
        return solve_online_1d(grid, spec, config, initial_lambdas.map(|l| l[0]));
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

/// One constraint: the 1-D frontier's bisection on λ, with its residual
/// tolerance (`tolerance × max(|bound|, |baseline total|)`). `warm` is the
/// first upper probe in place of 1.0; `max_iter` bounds nothing here, as
/// the bisection's own caps bound its probes.
fn solve_online_1d(
    grid: &QuoteGrid,
    spec: &ConstraintSpec,
    config: &SolverConfig,
    warm: Option<f64>,
) -> Result<SolveResult> {
    let (baseline_objective, baseline_constraints) = grid.baseline_totals();
    let tolerance = bisection_residual_tolerance(
        spec.threshold,
        baseline_constraints[0].abs(),
        config.tolerance,
    );
    let outcome = solve_point_1d_bisection(grid, spec, warm, tolerance, config.record_history)?;
    Ok(SolveResult {
        optimal_steps: outcome.optimal_steps,
        lambdas: vec![outcome.lambda],
        iterations: outcome.probes,
        converged: outcome.non_convergence.is_none(),
        total_objective: outcome.total_objective,
        total_constraints: vec![outcome.total_constraint],
        baseline_objective,
        baseline_constraints,
        solver_path: SolverPath::Bisection,
        non_convergence_reason: outcome.non_convergence,
        history: outcome.history.map(SolveHistory::Bisection),
    })
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
        Some(SolveHistory::Subgradient(IterationHistory {
            records: history_records,
        }))
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
        solver_path: SolverPath::Subgradient,
        non_convergence_reason: (!converged)
            .then_some(NonConvergenceReason::IterationBudgetExhausted),
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
        // One constraint: the history is the bisection's probes, one per iteration.
        let Some(SolveHistory::Bisection(probes)) = result.history.as_ref() else {
            panic!("a one-constraint solve records its bisection probes");
        };
        assert_eq!(probes.len(), result.iterations);
        assert_eq!(probes[0].lambda, 0.0);
    }

    /// Two constraints run the subgradient method and record its iterations.
    #[test]
    fn test_two_constraint_history_is_the_subgradient_iterations() {
        let grid = QuoteGrid {
            n_quotes: 3,
            n_steps: 4,
            scenario_values: vec![0.9, 0.95, 1.0, 1.05],
            objective: vec![1.0, 3.0, 2.0, 0.5, 5.0, 4.0, 3.0, 2.0, 0.1, 0.2, 0.3, 0.9],
            constraints: vec![
                vec![1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7, 1.0, 0.9, 0.8, 0.7],
                vec![1.0; 12],
            ],
            constraint_names: vec!["volume".to_string(), "count".to_string()],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
            quote_id_fingerprint: 0,
        };
        let (_, bc) = grid.baseline_totals();
        let specs = vec![
            ConstraintSpec {
                name: "volume".to_string(),
                direction: ConstraintDirection::Min,
                threshold: bc[0] * 0.9,
            },
            ConstraintSpec {
                name: "count".to_string(),
                direction: ConstraintDirection::Max,
                threshold: 3.0,
            },
        ];
        let config = SolverConfig {
            max_iter: 10,
            record_history: true,
            ..Default::default()
        };
        let result = solve_online(&grid, &specs, &config, None).unwrap();
        assert_eq!(result.solver_path, SolverPath::Subgradient);
        let Some(SolveHistory::Subgradient(history)) = result.history.as_ref() else {
            panic!("a two-constraint solve records its subgradient iterations");
        };
        assert_eq!(history.records.len(), result.iterations);
        assert_eq!(history.records[0].iteration, 0);
        assert_eq!(
            result.non_convergence_reason,
            (!result.converged).then_some(NonConvergenceReason::IterationBudgetExhausted)
        );
    }

    /// A grid where each quote has its own elasticity and premium, so a volume
    /// minimum moves quotes one at a time as its λ rises.
    fn heterogeneous_volume_grid(n: usize) -> QuoteGrid {
        let m = 9;
        let scenario_values: Vec<f32> = (0..m).map(|j| 0.8 + 0.05 * j as f32).collect();
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];
        for q in 0..n {
            let elasticity = 2.0 + 7.0 * ((q * 37) % n) as f32 / n as f32;
            let base = 200.0 + 600.0 * ((q * 53) % n) as f32 / n as f32;
            for (j, &mult) in scenario_values.iter().enumerate() {
                let conversion = 1.0 / (1.0 + (elasticity * (mult - 1.0)).exp());
                obj[q * m + j] = base * (mult - 0.75) * conversion;
                vol[q * m + j] = conversion;
            }
        }
        QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values,
            objective: obj,
            constraints: vec![vol],
            constraint_names: vec!["volume".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i:05}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    /// The bound half way between the volume the unconstrained optimum gives
    /// and the most the grid allows: binding, and reachable.
    fn binding_volume_bound(grid: &QuoteGrid) -> f64 {
        let m = grid.n_steps;
        let (mut at_best, mut most) = (0.0f64, 0.0f64);
        for q in 0..grid.n_quotes {
            let row = q * m..(q + 1) * m;
            let objective = &grid.objective[row.clone()];
            let volume = &grid.constraints[0][row];
            let best = (0..m)
                .max_by(|&a, &b| objective[a].total_cmp(&objective[b]))
                .unwrap();
            at_best += volume[best] as f64;
            most += volume.iter().copied().fold(f32::MIN, f32::max) as f64;
        }
        0.5 * (at_best + most)
    }

    /// One constraint is solved as the 1-D frontier solves its point at the
    /// same bound (DESIGN_DECISIONS §20): it meets the bound where the
    /// subgradient method stops short, and gives the point's λ and totals.
    #[test]
    fn test_one_constraint_solve_is_the_frontier_point_at_its_bound() {
        use crate::frontier::{sweep_frontier, FrontierConfig};

        let grid = heterogeneous_volume_grid(2000);
        let threshold = binding_volume_bound(&grid);
        let spec = ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold,
        };
        let config = SolverConfig {
            record_history: true,
            ..Default::default()
        };
        let solved = solve_online(&grid, std::slice::from_ref(&spec), &config, None).unwrap();

        assert_eq!(solved.solver_path, SolverPath::Bisection);
        assert!(solved.converged);
        assert_eq!(solved.non_convergence_reason, None);
        assert!(solved.total_constraints[0] >= threshold);
        assert!(solved.lambdas[0] > 0.0, "the bound binds");

        let frontier = sweep_frontier(
            &grid,
            std::slice::from_ref(&spec),
            &FrontierConfig {
                n_points_per_dim: 1,
                threshold_ranges: vec![(threshold, threshold)],
                max_total_points: None,
                parallel: false,
            },
            &config,
            None,
        )
        .unwrap();
        let point = &frontier.points[0];
        assert_eq!(solved.lambdas, point.lambdas);
        assert_eq!(solved.total_objective, point.total_objective);
        assert_eq!(solved.total_constraints, point.total_constraints);
        assert_eq!(solved.iterations, point.iterations);

        let Some(SolveHistory::Bisection(probes)) = solved.history.as_ref() else {
            panic!("a one-constraint solve records its bisection probes");
        };
        assert_eq!(probes.len(), solved.iterations);
        let answer = probes.iter().rev().find(|probe| probe.accepted).unwrap();
        assert_eq!(answer.lambda, solved.lambdas[0]);
        assert_eq!(answer.total_constraint, solved.total_constraints[0]);
        assert_eq!(answer.total_objective, solved.total_objective);
    }

    /// A warm start moves where the bracket starts; the answer still meets the bound.
    #[test]
    fn test_one_constraint_warm_start_meets_the_bound() {
        let grid = heterogeneous_volume_grid(500);
        let threshold = binding_volume_bound(&grid);
        let specs = [ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold,
        }];
        let config = SolverConfig::default();
        let cold = solve_online(&grid, &specs, &config, None).unwrap();
        let warm = solve_online(&grid, &specs, &config, Some(&cold.lambdas)).unwrap();

        assert!(warm.converged);
        assert!(warm.total_constraints[0] >= threshold);
        assert!(warm.iterations <= cold.iterations);
    }

    fn one_quote_grid(objective: Vec<f32>, constraint: Vec<f32>) -> QuoteGrid {
        let n_steps = objective.len();
        QuoteGrid {
            n_quotes: 1,
            n_steps,
            scenario_values: (0..n_steps).map(|j| 0.9 + 0.1 * j as f32).collect(),
            objective,
            constraints: vec![constraint],
            constraint_names: vec!["c".to_string()],
            quote_ids: vec!["Q0".into()],
            quote_id_fingerprint: 0,
        }
    }

    /// A warm start far above the answer costs halvings, not precision: by hand,
    /// step 1 (objective 2) wins once λ > 0.05 and step 0 (objective 1) once
    /// λ > 0.1, so the smallest λ meeting `c ≤ 0` chooses step 1, whatever the start.
    #[test]
    fn test_one_constraint_warm_start_far_above_finds_the_cold_answer() {
        let grid = one_quote_grid(vec![1.0, 2.0, 3.0], vec![-20.0, -10.0, 10.0]);
        let specs = [ConstraintSpec {
            name: "c".to_string(),
            direction: ConstraintDirection::Max,
            threshold: 0.0,
        }];
        let config = SolverConfig::default();
        for warm in [None, Some(1e9), Some(1e5)] {
            let result =
                solve_online(&grid, &specs, &config, warm.map(|w| vec![w]).as_deref()).unwrap();
            assert!(result.converged, "warm start {warm:?}");
            assert_eq!(result.optimal_steps, vec![1], "warm start {warm:?}");
            assert_eq!(result.total_objective, 2.0, "warm start {warm:?}");
            assert!(
                (result.lambdas[0] - 0.05).abs() < 1e-6,
                "warm start {warm:?}"
            );
        }
    }

    /// A tiny warm start never stops the bracket short of a λ the cold start
    /// reaches: by hand, step 1 (c = 1) wins once λ > 10.
    #[test]
    fn test_one_constraint_warm_start_far_below_still_reaches_the_bound() {
        let grid = one_quote_grid(vec![100.0, 90.0], vec![0.0, 1.0]);
        let specs = [ConstraintSpec {
            name: "c".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.5,
        }];
        for warm in [None, Some(1e-10), Some(0.3)] {
            let result = solve_online(
                &grid,
                &specs,
                &SolverConfig::default(),
                warm.map(|w| vec![w]).as_deref(),
            )
            .unwrap();
            assert!(result.converged, "warm start {warm:?}");
            assert_eq!(result.total_constraints, vec![1.0], "warm start {warm:?}");
            // The argmax compares in f32, whose step at 10 is about 1e-6.
            assert!(
                (result.lambdas[0] - 10.0).abs() < 1e-5,
                "warm start {warm:?}"
            );
        }
    }

    /// A bound beyond what the grid allows is not met: the solve says why and
    /// reports the highest-λ probe, never a silent undershoot.
    #[test]
    fn test_one_constraint_bound_beyond_the_grid_is_above_the_envelope() {
        let grid = heterogeneous_volume_grid(200);
        let most: f64 = (0..grid.n_quotes)
            .map(|q| {
                grid.constraints[0][q * grid.n_steps..(q + 1) * grid.n_steps]
                    .iter()
                    .copied()
                    .fold(f32::MIN, f32::max) as f64
            })
            .sum();
        let specs = [ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: most * 1.01,
        }];
        let result = solve_online(&grid, &specs, &SolverConfig::default(), None).unwrap();

        assert!(!result.converged);
        assert_eq!(
            result.non_convergence_reason,
            Some(NonConvergenceReason::AboveEnvelope)
        );
        assert!(result.total_constraints[0] < most * 1.01);
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
