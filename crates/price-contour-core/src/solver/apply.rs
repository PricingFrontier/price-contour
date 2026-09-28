use crate::cancel::CancelFlag;
use crate::data::{ApplyResult, ConstraintSpec, QuoteGrid};
use crate::error::{PriceContourError, Result};

use super::argmax::{
    compute_lambda_signs_f32, lagrangian_argmax_pass, lagrangian_argmax_pass_cancellable,
};

/// Result of one Lagrangian argmax pass at fixed lambdas, without baselines.
///
/// Returned by `apply_lambdas_no_baselines` so callers that already have
/// the per-grid baselines (e.g. the frontier sweep) avoid re-running the
/// O(n_quotes * n_constraints) baseline pass on every probe.
pub struct ApplyPass {
    pub optimal_steps: Vec<u32>,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
}

/// Single-pass Lagrangian argmax with fixed lambdas — no baseline pass,
/// no `grid.validate()`.
///
/// **Precondition**: caller must have validated the grid (via
/// [`QuoteGrid::validate`] or by reaching here from a public entry point
/// like [`apply_lambdas`] or `sweep_frontier` that validates at the
/// boundary). Hot loops (frontier bisection, repeated probes) call this
/// directly to skip the per-call validation pass.
pub fn apply_lambdas_no_baselines(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    lambdas: &[f64],
) -> Result<ApplyPass> {
    apply_pass(grid, specs, lambdas, None)
}

fn apply_pass(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    lambdas: &[f64],
    cancel: Option<&CancelFlag>,
) -> Result<ApplyPass> {
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

    let (optimal_steps, total_objective, total_constraints) = match cancel {
        None => lagrangian_argmax_pass(grid, &lambda_signs_f32, 0, n_quotes),
        Some(flag) => {
            lagrangian_argmax_pass_cancellable(grid, &lambda_signs_f32, 0, n_quotes, flag)?
        }
    };

    Ok(ApplyPass {
        optimal_steps,
        total_objective,
        total_constraints,
    })
}

/// Single-pass Lagrangian argmax with fixed lambdas (no iteration).
///
/// One forward pass with rayon-parallel argmax — no lambda updates.
/// Internal parallelism is handled by rayon grain sizes. Validates the
/// grid at the public boundary so internals can skip per-call validation.
pub fn apply_lambdas(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    lambdas: &[f64],
) -> Result<ApplyResult> {
    apply_lambdas_polling(grid, specs, lambdas, None)
}

/// [`apply_lambdas`] that stops with `Err(Cancelled)` once `cancel` is set.
/// Both the argmax and the baseline totals poll the flag; an uncancelled
/// result is bit-identical to [`apply_lambdas`].
pub fn apply_lambdas_cancellable(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    lambdas: &[f64],
    cancel: &CancelFlag,
) -> Result<ApplyResult> {
    apply_lambdas_polling(grid, specs, lambdas, Some(cancel))
}

fn apply_lambdas_polling(
    grid: &QuoteGrid,
    specs: &[ConstraintSpec],
    lambdas: &[f64],
    cancel: Option<&CancelFlag>,
) -> Result<ApplyResult> {
    grid.validate()?;
    let pass = apply_pass(grid, specs, lambdas, cancel)?;
    let (baseline_objective, baseline_constraints) = match cancel {
        None => grid.baseline_totals(),
        Some(flag) => grid.baseline_totals_cancellable(flag)?,
    };

    Ok(ApplyResult {
        optimal_steps: pass.optimal_steps,
        lambdas: lambdas.to_vec(),
        total_objective: pass.total_objective,
        total_constraints: pass.total_constraints,
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
            quote_id_fingerprint: 0,
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

    // -----------------------------------------------------------------------
    // Cancellation (DESIGN_DECISIONS §14)
    // -----------------------------------------------------------------------

    fn large_grid(n: usize) -> (QuoteGrid, Vec<ConstraintSpec>) {
        let m = 7;
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];
        for q in 0..n {
            for j in 0..m {
                let mult = 0.85 + 0.05 * j as f32;
                let conversion = 1.0 / (1.0 + ((q % 13) as f32 * 0.3 * (mult - 1.0)).exp());
                obj[q * m + j] = (50.0 + (q % 101) as f32) * mult * conversion;
                vol[q * m + j] = conversion;
            }
        }
        let grid = QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: (0..m).map(|j| 0.85 + 0.05 * j as f32).collect(),
            objective: obj,
            constraints: vec![vol],
            constraint_names: vec!["volume".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        };
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.0,
        }];
        (grid, specs)
    }

    #[test]
    fn cancellable_apply_is_bit_identical_when_not_cancelled() {
        // Several argmax grains and several baseline poll blocks.
        let (grid, specs) = large_grid(crate::constants::CANCEL_POLL_QUOTES * 2 + 4099);
        let lambdas = [0.7];
        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (plain, tokened) = single_thread.install(|| {
            (
                apply_lambdas(&grid, &specs, &lambdas).unwrap(),
                apply_lambdas_cancellable(&grid, &specs, &lambdas, &CancelFlag::new()).unwrap(),
            )
        });
        assert_eq!(plain.optimal_steps, tokened.optimal_steps);
        assert_eq!(
            plain.total_objective.to_bits(),
            tokened.total_objective.to_bits()
        );
        assert_eq!(
            plain.total_constraints[0].to_bits(),
            tokened.total_constraints[0].to_bits()
        );
        assert_eq!(
            plain.baseline_objective.to_bits(),
            tokened.baseline_objective.to_bits()
        );
        assert_eq!(
            plain.baseline_constraints[0].to_bits(),
            tokened.baseline_constraints[0].to_bits()
        );

        // Multi-threaded, the per-quote choices are identical.
        let tokened_parallel =
            apply_lambdas_cancellable(&grid, &specs, &lambdas, &CancelFlag::new()).unwrap();
        assert_eq!(plain.optimal_steps, tokened_parallel.optimal_steps);
    }

    #[test]
    fn pre_cancelled_apply_returns_cancelled() {
        let (grid, specs) = make_test_grid();
        let flag = CancelFlag::new();
        flag.cancel();
        let err = apply_lambdas_cancellable(&grid, &specs, &[0.0], &flag).unwrap_err();
        assert!(matches!(err, PriceContourError::Cancelled), "{err}");
    }

    #[test]
    fn apply_input_errors_win_over_cancellation() {
        // A malformed call reports the input error, not Cancelled.
        let (grid, specs) = make_test_grid();
        let flag = CancelFlag::new();
        flag.cancel();
        let err = apply_lambdas_cancellable(&grid, &specs, &[0.0, 0.0], &flag).unwrap_err();
        assert!(
            matches!(err, PriceContourError::DimensionMismatch(_)),
            "{err}"
        );
    }
}
