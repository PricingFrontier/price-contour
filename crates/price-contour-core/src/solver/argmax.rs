use rayon::prelude::*;

use crate::cancel::{is_cancelled, CancelFlag};
use crate::constants::ARGMAX_PAR_GRAIN;
use crate::data::{ConstraintDirection, ConstraintSpec, QuoteGrid};
use crate::error::Result;

/// Pre-compute signed lambdas in f32.
///
/// For Min constraints: positive lambda (penalty for violating lower bound).
/// For Max constraints: negated lambda (penalty for violating upper bound).
/// Keeping these in f32 avoids per-element casts in the inner loop.
pub fn compute_lambda_signs_f32(specs: &[ConstraintSpec], lambdas: &[f64]) -> Vec<f32> {
    specs
        .iter()
        .zip(lambdas.iter())
        .map(|(spec, &lam)| match spec.direction {
            ConstraintDirection::Min => lam as f32,
            ConstraintDirection::Max => -(lam as f32),
        })
        .collect()
}

/// Parallel Lagrangian argmax over a quote range.
///
/// For each quote in `[quote_start, quote_end)`, computes:
///   L(q, j) = objective[q,j] + Σ_k (signed_lambda_k × constraint_k[q,j])
/// and selects the step j that maximises L.
///
/// Returns (optimal_steps, total_objective, total_constraints).
///
/// The inner loop is structured constraint-outer, step-inner for SIMD
/// auto-vectorisation (`a[j] += c * b[j]`). Computation is in f32;
/// only the final accumulation at the optimal step uses f64.
pub fn lagrangian_argmax_pass(
    grid: &QuoteGrid,
    lambda_signs_f32: &[f32],
    quote_start: usize,
    quote_end: usize,
) -> (Vec<u32>, f64, Vec<f64>) {
    argmax_pass_polling(grid, lambda_signs_f32, quote_start, quote_end, None)
}

/// [`lagrangian_argmax_pass`], polling `cancel` once per
/// [`ARGMAX_PAR_GRAIN`] quotes. Once the flag is set the remaining grains are
/// skipped and the call returns `Err(Cancelled)`; otherwise the result is
/// bit-identical (same grains, same fold and reduce).
pub fn lagrangian_argmax_pass_cancellable(
    grid: &QuoteGrid,
    lambda_signs_f32: &[f32],
    quote_start: usize,
    quote_end: usize,
    cancel: &CancelFlag,
) -> Result<(Vec<u32>, f64, Vec<f64>)> {
    cancel.check()?;
    let pass = argmax_pass_polling(grid, lambda_signs_f32, quote_start, quote_end, Some(cancel));
    cancel.check()?;
    Ok(pass)
}

fn argmax_pass_polling(
    grid: &QuoteGrid,
    lambda_signs_f32: &[f32],
    quote_start: usize,
    quote_end: usize,
    cancel: Option<&CancelFlag>,
) -> (Vec<u32>, f64, Vec<f64>) {
    let chunk_size = quote_end - quote_start;
    let n_steps = grid.n_steps;
    let n_constraints = grid.constraints.len();

    let mut steps = vec![0u32; chunk_size];

    // fold+reduce: each rayon partition accumulates into its own (obj, cons),
    // then reduce merges them. No intermediate Vec<(f64, Vec<f64>)> allocated.
    let (total_obj, total_cons) = steps
        .par_chunks_mut(ARGMAX_PAR_GRAIN)
        .enumerate()
        .fold(
            || (0.0f64, vec![0.0f64; n_constraints]),
            |(mut partial_obj, mut partial_cons), (chunk_idx, step_slice)| {
                // A cancelled pass skips its remaining grains; the caller
                // discards the partial result.
                if is_cancelled(cancel) {
                    return (partial_obj, partial_cons);
                }
                let sub_start = quote_start + chunk_idx * ARGMAX_PAR_GRAIN;
                let sub_len = step_slice.len();
                // Reusable buffer for per-quote Lagrangian values (n_steps × 4 bytes).
                // Allocated once per rayon partition, reused across all quotes in the grain.
                let mut lagrangians = vec![0.0f32; n_steps];

                // Index-based loops are intentional here: they produce the exact
                // memory access pattern LLVM needs for SIMD auto-vectorization.
                // Converting to iterator chains would obscure the axpy pattern.
                #[allow(clippy::needless_range_loop)]
                for local_i in 0..sub_len {
                    let q = sub_start + local_i;
                    let base = q * n_steps;

                    // Build Lagrangian in f32: objective + sum(signed_lambda × constraint).
                    // Constraint-outer, step-inner gives LLVM a simple axpy pattern
                    // (a[j] += c * b[j]) it can auto-vectorize with SIMD.
                    lagrangians.copy_from_slice(&grid.objective[base..base + n_steps]);
                    for (k, &sign_lam) in lambda_signs_f32.iter().enumerate() {
                        let con_slice = &grid.constraints[k][base..base + n_steps];
                        for j in 0..n_steps {
                            lagrangians[j] += sign_lam * con_slice[j];
                        }
                    }

                    // Argmax in f32
                    let mut best_step: usize = 0;
                    let mut best_val = lagrangians[0];
                    for j in 1..n_steps {
                        if lagrangians[j] > best_val {
                            best_val = lagrangians[j];
                            best_step = j;
                        }
                    }

                    step_slice[local_i] = best_step as u32;

                    // Accumulate in f64 only at the optimal step — precision
                    // where it matters (portfolio-level totals).
                    let opt_idx = base + best_step;
                    partial_obj += grid.objective[opt_idx] as f64;
                    for k in 0..n_constraints {
                        partial_cons[k] += grid.constraints[k][opt_idx] as f64;
                    }
                }

                (partial_obj, partial_cons)
            },
        )
        .reduce(
            || (0.0f64, vec![0.0f64; n_constraints]),
            |(mut obj_a, mut cons_a), (obj_b, cons_b)| {
                obj_a += obj_b;
                for k in 0..cons_a.len() {
                    cons_a[k] += cons_b[k];
                }
                (obj_a, cons_a)
            },
        );

    (steps, total_obj, total_cons)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::QuoteGrid;
    use approx::assert_abs_diff_eq;

    fn make_test_grid() -> QuoteGrid {
        // 3 quotes, 4 steps
        QuoteGrid {
            n_quotes: 3,
            n_steps: 4,
            scenario_values: vec![0.9, 0.95, 1.0, 1.05],
            objective: vec![
                1.0, 3.0, 2.0, 0.5, // quote 0: best at step 1
                5.0, 4.0, 3.0, 2.0, // quote 1: best at step 0
                0.1, 0.2, 0.3, 0.9, // quote 2: best at step 3
            ],
            constraints: vec![vec![
                1.0, 0.9, 0.8, 0.7, // quote 0
                1.0, 0.9, 0.8, 0.7, // quote 1
                1.0, 0.9, 0.8, 0.7, // quote 2
            ]],
            constraint_names: vec!["volume".to_string()],
            quote_ids: vec!["Q0".into(), "Q1".into(), "Q2".into()],
            quote_id_fingerprint: 0,
        }
    }

    #[test]
    fn test_zero_lambdas_picks_max_objective() {
        let grid = make_test_grid();
        let lambda_signs = vec![0.0f32]; // zero lambda -> no constraint influence

        let (steps, obj, _cons) = lagrangian_argmax_pass(&grid, &lambda_signs, 0, 3);

        assert_eq!(steps, vec![1, 0, 3]); // max objective per quote
        assert_abs_diff_eq!(obj, 3.0 + 5.0 + 0.9, epsilon = 1e-6);
    }

    #[test]
    fn test_subrange_of_quotes() {
        let grid = make_test_grid();
        let lambda_signs = vec![0.0f32];

        // Only process quotes 1..3
        let (steps, obj, _cons) = lagrangian_argmax_pass(&grid, &lambda_signs, 1, 3);

        assert_eq!(steps.len(), 2);
        assert_eq!(steps, vec![0, 3]); // quote 1 best at 0, quote 2 best at 3
        assert_abs_diff_eq!(obj, 5.0 + 0.9, epsilon = 1e-6);
    }

    #[test]
    fn test_compute_lambda_signs_f32() {
        let specs = vec![
            ConstraintSpec {
                name: "vol".into(),
                direction: ConstraintDirection::Min,
                threshold: 100.0,
            },
            ConstraintSpec {
                name: "lr".into(),
                direction: ConstraintDirection::Max,
                threshold: 50.0,
            },
        ];
        let lambdas = vec![2.0, 3.0];
        let signs = compute_lambda_signs_f32(&specs, &lambdas);
        assert_eq!(signs, vec![2.0f32, -3.0f32]);
    }

    #[test]
    fn a_set_flag_skips_every_grain() {
        // Polling happens before each grain, so a flag set before the pass
        // leaves every step unwritten and every total zero; the public
        // cancellable pass turns that into Err(Cancelled).
        let n = ARGMAX_PAR_GRAIN * 3 + 5;
        let grid = QuoteGrid {
            n_quotes: n,
            n_steps: 2,
            scenario_values: vec![1.0, 1.1],
            objective: (0..n).flat_map(|_| [0.0f32, 1.0]).collect(),
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        };
        let flag = CancelFlag::new();
        let (steps, obj, _) = argmax_pass_polling(&grid, &[], 0, n, Some(&flag));
        assert!(steps.iter().all(|&s| s == 1));
        assert_abs_diff_eq!(obj, n as f64, epsilon = 1e-9);

        flag.cancel();
        let (steps, obj, _) = argmax_pass_polling(&grid, &[], 0, n, Some(&flag));
        assert!(steps.iter().all(|&s| s == 0));
        assert_eq!(obj, 0.0);
        assert!(matches!(
            lagrangian_argmax_pass_cancellable(&grid, &[], 0, n, &flag),
            Err(crate::error::PriceContourError::Cancelled)
        ));
    }

    #[test]
    fn a_flag_set_mid_pass_skips_every_later_grain() {
        // One thread runs the grains in order, polling once before each: a
        // flag that trips on the third poll lets grains 0 and 1 run and
        // skips all the rest.
        let n = ARGMAX_PAR_GRAIN * 8;
        let grid = QuoteGrid {
            n_quotes: n,
            n_steps: 2,
            scenario_values: vec![1.0, 1.1],
            objective: (0..n).flat_map(|_| [0.0f32, 1.0]).collect(),
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        };
        let one_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let flag = CancelFlag::new();
        flag.cancel_after_polls(3);
        let (steps, obj, _) =
            one_thread.install(|| argmax_pass_polling(&grid, &[], 0, n, Some(&flag)));
        assert!(steps[..2 * ARGMAX_PAR_GRAIN].iter().all(|&s| s == 1));
        assert!(steps[2 * ARGMAX_PAR_GRAIN..].iter().all(|&s| s == 0));
        assert_abs_diff_eq!(obj, (2 * ARGMAX_PAR_GRAIN) as f64, epsilon = 1e-9);

        let flag = CancelFlag::new();
        flag.cancel_after_polls(3);
        let result =
            one_thread.install(|| lagrangian_argmax_pass_cancellable(&grid, &[], 0, n, &flag));
        assert!(matches!(
            result,
            Err(crate::error::PriceContourError::Cancelled)
        ));
    }
}
