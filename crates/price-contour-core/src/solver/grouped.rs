use crate::constants::*;
use crate::data::{
    ConstraintDirection, ConstraintSpec, GroupMapping, GroupedSolveResult, IterationHistory,
    IterationRecord, QuoteGrid, SolverConfig,
};
use crate::error::Result;
use crate::solver::lambda::update_lambdas_subgradient;

/// Find the step index whose scenario_value is nearest to `target`.
/// Returns (step_index, was_clamped).
fn nearest_step(scenario_values: &[f32], target: f32) -> (usize, bool) {
    let n = scenario_values.len();
    if target <= scenario_values[0] {
        return (0, target < scenario_values[0]);
    }
    if target >= scenario_values[n - 1] {
        return (n - 1, target > scenario_values[n - 1]);
    }
    // Binary search for nearest
    match scenario_values.binary_search_by(|m| m.partial_cmp(&target).unwrap()) {
        Ok(i) => (i, false),
        Err(i) => {
            // i is insertion point; compare i-1 and i
            let below = scenario_values[i - 1];
            let above = scenario_values[i];
            if (target - below).abs() <= (above - target).abs() {
                (i - 1, false)
            } else {
                (i, false)
            }
        }
    }
}

/// Grouped Lagrangian solve: per-group argmax over candidate factor values,
/// with remapping from (residual * candidate) to the nearest grid step.
pub fn solve_grouped(
    grid: &QuoteGrid,
    group_mapping: &GroupMapping,
    residuals: &[f32],
    candidates: &[f32],
    specs: &[ConstraintSpec],
    config: &SolverConfig,
    initial_lambdas: Option<&[f64]>,
) -> Result<GroupedSolveResult> {
    grid.validate()?;

    let n_quotes = grid.n_quotes;
    let n_steps = grid.n_steps;
    let n_constraints = specs.len();
    let n_groups = group_mapping.n_groups;
    let n_candidates = candidates.len();

    // Compute baselines and scale factors
    let (baseline_obj, baseline_cons) = grid.baseline_totals();
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

    let mut best_lambdas = lambdas.clone();
    let mut best_feasible_obj = f64::NEG_INFINITY;
    let mut lambda_sum = vec![0.0f64; n_constraints];
    let mut converged = false;
    let mut iterations = 0;

    // Current per-group best candidate index
    let mut group_best_candidate = vec![0usize; n_groups];
    // Per-quote optimal step (after remapping)
    let mut optimal_steps = vec![0u32; n_quotes];
    let mut total_objective: f64 = 0.0;
    let mut total_constraints = vec![0.0f64; n_constraints];
    let mut clamp_count: u64 = 0;
    let mut total_remaps: u64 = 0;

    let mut history_records: Vec<IterationRecord> = if config.record_history {
        Vec::with_capacity(config.max_iter)
    } else {
        Vec::new()
    };

    for iter in 0..config.max_iter {
        // Pre-compute lambda signs
        let lambda_signs: Vec<f64> = specs
            .iter()
            .zip(lambdas.iter())
            .map(|(spec, &lam)| match spec.direction {
                ConstraintDirection::Min => lam,
                ConstraintDirection::Max => -lam,
            })
            .collect();

        // Allocate group_L: (n_groups, n_candidates) — Lagrangian accumulated per group per candidate
        let mut group_l = vec![vec![0.0f64; n_candidates]; n_groups];
        // Track per-group per-candidate remapped indices for later reconstruction
        // We don't store all N*P indices; instead we recompute in the final pass

        clamp_count = 0;
        total_remaps = 0;

        // For each quote, for each candidate, remap and accumulate
        for i in 0..n_quotes {
            let g = group_mapping.group_of[i] as usize;
            let base = i * n_steps;

            for (j, &cand) in candidates.iter().enumerate() {
                let target = residuals[i] * cand;
                let (k, clamped) = nearest_step(&grid.scenario_values, target);
                if clamped {
                    clamp_count += 1;
                }
                total_remaps += 1;

                let idx = base + k;
                let mut l = grid.objective[idx] as f64;
                for (c, &sign_lam) in lambda_signs.iter().enumerate() {
                    l += sign_lam * grid.constraints[c][idx] as f64;
                }
                group_l[g][j] += l;
            }
        }

        // Argmax per group
        for g in 0..n_groups {
            let mut best_j = 0;
            let mut best_val = f64::NEG_INFINITY;
            for j in 0..n_candidates {
                if group_l[g][j] > best_val {
                    best_val = group_l[g][j];
                    best_j = j;
                }
            }
            group_best_candidate[g] = best_j;
        }

        // Reconstruct per-quote steps and compute totals
        total_objective = 0.0;
        total_constraints.fill(0.0);

        for i in 0..n_quotes {
            let g = group_mapping.group_of[i] as usize;
            let cand = candidates[group_best_candidate[g]];
            let target = residuals[i] * cand;
            let (k, _) = nearest_step(&grid.scenario_values, target);
            optimal_steps[i] = k as u32;

            let idx = i * n_steps + k;
            total_objective += grid.objective[idx] as f64;
            for c in 0..n_constraints {
                total_constraints[c] += grid.constraints[c][idx] as f64;
            }
        }

        // Check constraint satisfaction
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

        for k in 0..n_constraints {
            lambda_sum[k] += lambdas[k];
        }

        iterations = iter + 1;

        if all_satisfied && max_lambda_change < LAMBDA_CONVERGENCE_TOL {
            converged = true;
            break;
        }
    }

    // If not converged, do a final pass with best/averaged lambdas
    if !converged {
        let final_lambdas = if best_feasible_obj > f64::NEG_INFINITY {
            best_lambdas.clone()
        } else {
            lambda_sum
                .iter()
                .map(|&s| s / iterations.max(1) as f64)
                .collect()
        };

        let lambda_signs: Vec<f64> = specs
            .iter()
            .zip(final_lambdas.iter())
            .map(|(spec, &lam)| match spec.direction {
                ConstraintDirection::Min => lam,
                ConstraintDirection::Max => -lam,
            })
            .collect();

        let mut group_l = vec![vec![0.0f64; n_candidates]; n_groups];
        for i in 0..n_quotes {
            let g = group_mapping.group_of[i] as usize;
            let base = i * n_steps;
            for (j, &cand) in candidates.iter().enumerate() {
                let target = residuals[i] * cand;
                let (k, _) = nearest_step(&grid.scenario_values, target);
                let idx = base + k;
                let mut l = grid.objective[idx] as f64;
                for (c, &sign_lam) in lambda_signs.iter().enumerate() {
                    l += sign_lam * grid.constraints[c][idx] as f64;
                }
                group_l[g][j] += l;
            }
        }

        for g in 0..n_groups {
            let mut best_j = 0;
            let mut best_val = f64::NEG_INFINITY;
            for j in 0..n_candidates {
                if group_l[g][j] > best_val {
                    best_val = group_l[g][j];
                    best_j = j;
                }
            }
            group_best_candidate[g] = best_j;
        }

        total_objective = 0.0;
        total_constraints.fill(0.0);
        for i in 0..n_quotes {
            let g = group_mapping.group_of[i] as usize;
            let cand = candidates[group_best_candidate[g]];
            let target = residuals[i] * cand;
            let (k, _) = nearest_step(&grid.scenario_values, target);
            optimal_steps[i] = k as u32;
            let idx = i * n_steps + k;
            total_objective += grid.objective[idx] as f64;
            for c in 0..n_constraints {
                total_constraints[c] += grid.constraints[c][idx] as f64;
            }
        }

        lambdas = final_lambdas;
    }

    // Build per-group optimal factor values
    let optimal_factor_values: Vec<f32> = group_best_candidate
        .iter()
        .map(|&j| candidates[j])
        .collect();

    let clamp_rate = if total_remaps > 0 {
        clamp_count as f32 / total_remaps as f32
    } else {
        0.0
    };

    let history = if config.record_history {
        Some(IterationHistory {
            records: history_records,
        })
    } else {
        None
    };

    Ok(GroupedSolveResult {
        optimal_factor_values,
        optimal_steps_per_quote: optimal_steps,
        lambdas,
        iterations,
        converged,
        total_objective,
        total_constraints,
        baseline_objective: baseline_obj,
        baseline_constraints: baseline_cons,
        clamp_rate,
        history,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::*;
    use crate::solver::solve_online;

    fn make_heterogeneous_grid(n: usize, m: usize) -> QuoteGrid {
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];
        let mults: Vec<f32> = (0..m).map(|j| 0.8 + 0.1 * j as f32).collect();

        for q in 0..n {
            let elasticity = 1.0 + 4.0 * (q as f32) / (n as f32);
            let base = 50.0 + 100.0 * (q as f32) / (n as f32);
            for j in 0..m {
                let mult = mults[j];
                let conversion = 1.0 / (1.0 + (elasticity * (mult - 1.0)).exp());
                obj[q * m + j] = base * mult * conversion;
                vol[q * m + j] = conversion;
            }
        }

        QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: mults,
            objective: obj,
            constraints: vec![vol],
            constraint_names: vec!["volume".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
        }
    }

    #[test]
    fn test_nearest_step() {
        let mults = vec![0.8f32, 0.9, 1.0, 1.1, 1.2];
        assert_eq!(nearest_step(&mults, 1.0), (2, false));  // exact match
        assert_eq!(nearest_step(&mults, 0.87), (1, false));  // closer to 0.9
        assert_eq!(nearest_step(&mults, 0.7), (0, true));   // clamped below
        assert_eq!(nearest_step(&mults, 1.3), (4, true));   // clamped above
        assert_eq!(nearest_step(&mults, 0.97), (2, false));  // closer to 1.0
        // Equidistant case: just check it returns a valid index
        let (idx, _) = nearest_step(&mults, 0.85);
        assert!(idx == 0 || idx == 1, "equidistant should pick adjacent index");
    }

    #[test]
    fn test_all_distinct_groups_matches_online() {
        // Each quote in its own group with residual=1.0 and candidates=scenario_values
        // should behave like the online solver (identity remap).
        let n = 50;
        let m = 5;
        let grid = make_heterogeneous_grid(n, m);

        let labels: Vec<String> = (0..n).map(|i| format!("G{i}")).collect();
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![1.0f32; n];
        let candidates = grid.scenario_values.clone();

        let (_, bc) = grid.baseline_totals();
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: bc[0] * 0.90,
        }];

        let config = SolverConfig {
            max_iter: 200,
            ..Default::default()
        };

        let grouped_result =
            solve_grouped(&grid, &group_mapping, &residuals, &candidates, &specs, &config, None)
                .unwrap();
        let online_result = solve_online(&grid, &specs, &config, None).unwrap();

        // Both should produce similar objectives (may not be exactly equal due to
        // algorithm differences, but should be close)
        let diff = (grouped_result.total_objective - online_result.total_objective).abs();
        let scale = online_result.total_objective.abs().max(1.0);
        assert!(
            diff / scale < 0.05,
            "grouped vs online objective diff too large: {} vs {}",
            grouped_result.total_objective,
            online_result.total_objective
        );
    }

    #[test]
    fn test_single_group() {
        // All quotes in one group: should pick a single factor for the whole portfolio
        let n = 20;
        let m = 5;
        let grid = make_heterogeneous_grid(n, m);

        let labels = vec!["ALL".to_string(); n];
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![1.0f32; n];
        let candidates: Vec<f32> = (0..21).map(|i| 0.8 + 0.02 * i as f32).collect();

        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };

        let result =
            solve_grouped(&grid, &group_mapping, &residuals, &candidates, &[], &config, None)
                .unwrap();

        assert_eq!(result.optimal_factor_values.len(), 1);
        // All quotes should have the same factor value
        let fv = result.optimal_factor_values[0];
        assert!(fv >= 0.8 && fv <= 1.2, "factor value out of range: {fv}");
    }

    #[test]
    fn test_clamp_rate_with_extreme_residuals() {
        let n = 10;
        let m = 5;
        let grid = make_heterogeneous_grid(n, m);

        let labels: Vec<String> = (0..n).map(|i| format!("G{i}")).collect();
        let group_mapping = build_group_mapping(&labels);
        // Very large residuals push targets outside grid
        let residuals = vec![3.0f32; n];
        let candidates = vec![1.0f32];

        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };

        let result =
            solve_grouped(&grid, &group_mapping, &residuals, &candidates, &[], &config, None)
                .unwrap();

        assert!(
            result.clamp_rate > 0.0,
            "expected clamping with extreme residuals, got clamp_rate={}",
            result.clamp_rate
        );
    }

    #[test]
    fn test_clamp_rate_zero_with_wide_grid() {
        let n = 10;
        let m = 5;
        let grid = make_heterogeneous_grid(n, m);

        let labels: Vec<String> = (0..n).map(|i| format!("G{i}")).collect();
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![1.0f32; n];
        // Candidates well within grid range
        let candidates = vec![1.0f32];

        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };

        let result =
            solve_grouped(&grid, &group_mapping, &residuals, &candidates, &[], &config, None)
                .unwrap();

        assert!(
            (result.clamp_rate - 0.0).abs() < 1e-6,
            "expected zero clamp rate, got {}",
            result.clamp_rate
        );
    }
}
