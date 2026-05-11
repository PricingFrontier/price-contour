use std::cell::RefCell;

use rayon::prelude::*;

use crate::constants::*;

// Thread-local scratch for the precomputed `target_idx` buffer. Re-used
// across `solve_grouped` calls on the same calling thread, which removes
// the per-call `mmap`/zero-fill for the 20 MB Vec<u32> on the typical
// ratebook frontier (165 calls × 5 M cells × 4 bytes ≈ 3.3 GB of allocator
// traffic if we re-allocated every call). Each call resizes to the
// current `cells` and overwrites every entry in the parallel precompute
// pass, so the buffer's prior contents do not leak into the result.
thread_local! {
    static TARGET_IDX_SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}
use crate::data::{
    ConstraintDirection, ConstraintSpec, GroupMapping, GroupedSolveResult, IterationHistory,
    IterationRecord, QuoteGrid, SolverConfig,
};
use crate::error::{PriceContourError, Result};
use crate::solver::convergence::{all_constraints_satisfied, select_final_lambdas};
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
    match scenario_values.binary_search_by(|m| m.total_cmp(&target)) {
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

/// Fill an existing buffer with signed lambdas, reusing the allocation
/// across subgradient iterations of one `solve_grouped` call.
#[inline]
fn fill_lambda_signs(out: &mut [f64], specs: &[ConstraintSpec], lambdas: &[f64]) {
    debug_assert_eq!(out.len(), specs.len());
    debug_assert_eq!(lambdas.len(), specs.len());
    for (slot, (spec, &lam)) in out.iter_mut().zip(specs.iter().zip(lambdas.iter())) {
        *slot = match spec.direction {
            ConstraintDirection::Min => lam,
            ConstraintDirection::Max => -lam,
        };
    }
}

/// Accumulate Lagrangian values per group per candidate.
///
/// Returns (group_l flat matrix [n_groups × n_candidates], clamp_count, total_remaps).
/// The matrix is stored row-major: group_l[g * n_candidates + j].
///
/// Uses rayon fold+reduce when the per-thread memory is within the cap (4 MB per
/// fold identity). Falls back to sequential for degenerate inputs.
fn accumulate_group_lagrangians(
    grid: &QuoteGrid,
    group_mapping: &GroupMapping,
    residuals: &[f32],
    candidates: &[f32],
    lambda_signs: &[f64],
    n_groups: usize,
) -> (Vec<f64>, u64, u64) {
    let n_steps = grid.n_steps;
    let n_candidates = candidates.len();
    let n_quotes = grid.n_quotes;

    // Memory check: each fold identity allocates n_groups * n_candidates * 8 bytes.
    // Cap per-identity at 4 MB to prevent degenerate inputs from exhausting memory.
    let per_identity_bytes = n_groups * n_candidates * std::mem::size_of::<f64>();
    const MAX_IDENTITY_BYTES: usize = 4 * 1024 * 1024;

    if per_identity_bytes > MAX_IDENTITY_BYTES {
        // Sequential fallback for degenerate group × candidate sizes
        let mut group_l = vec![0.0f64; n_groups * n_candidates];
        let mut clamp_count = 0u64;
        let mut total_remaps = 0u64;

        for (i, &res) in residuals.iter().enumerate().take(n_quotes) {
            let g = group_mapping.group_of[i] as usize;
            let base = i * n_steps;
            for (j, &cand) in candidates.iter().enumerate() {
                let target = res * cand;
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
                group_l[g * n_candidates + j] += l;
            }
        }

        (group_l, clamp_count, total_remaps)
    } else {
        // Parallel: each thread gets its own group_l matrix
        (0..n_quotes)
            .into_par_iter()
            .with_min_len(GROUPED_PAR_GRAIN)
            .fold(
                || (vec![0.0f64; n_groups * n_candidates], 0u64, 0u64),
                |(mut local_gl, mut local_clamp, mut local_remaps), i| {
                    let g = group_mapping.group_of[i] as usize;
                    let base = i * n_steps;
                    for (j, &cand) in candidates.iter().enumerate() {
                        let target = residuals[i] * cand;
                        let (k, clamped) = nearest_step(&grid.scenario_values, target);
                        if clamped {
                            local_clamp += 1;
                        }
                        local_remaps += 1;
                        let idx = base + k;
                        let mut l = grid.objective[idx] as f64;
                        for (c, &sign_lam) in lambda_signs.iter().enumerate() {
                            l += sign_lam * grid.constraints[c][idx] as f64;
                        }
                        local_gl[g * n_candidates + j] += l;
                    }
                    (local_gl, local_clamp, local_remaps)
                },
            )
            .reduce(
                || (vec![0.0f64; n_groups * n_candidates], 0u64, 0u64),
                |(mut gl_a, cc_a, tr_a), (gl_b, cc_b, tr_b)| {
                    for idx in 0..gl_a.len() {
                        gl_a[idx] += gl_b[idx];
                    }
                    (gl_a, cc_a + cc_b, tr_a + tr_b)
                },
            )
    }
}

/// λ-independent caches for the grouped solver's inner Lagrangian loop.
///
/// Per-(group, candidate) Lagrangians under the Min/Max sign convention
/// expand into:
///
///   group_l[g, j] = Σ_{i ∈ group g} (obj[i, k_{i,j}] + Σ_c sign_c · λ_c · cons_c[i, k_{i,j}])
///                 = a_table[g, j] + Σ_c (sign_c · λ_c) · b_table[c, g, j]
///
/// where `k_{i,j} = nearest_step(scenario_values, residuals[i] · candidates[j])`.
/// Both inner sums are linear in λ, so we can precompute them ONCE per
/// `solve_grouped` call and collapse every subgradient iteration to an
/// `n_groups × n_candidates × (1 + K)` axpy — a 4 KB scratch table at the
/// typical ratebook size, vs the `n_quotes × n_candidates` (5 M-cell)
/// per-iter sweep the old kernel did.
///
/// Cached fields:
///
/// * `target_idx[i, j] = k_{i,j}` (`Vec<u32>`, length `n_quotes × n_candidates`).
///   Still needed by `reconstruct_from_tables` to emit `optimal_steps`
///   (which records the grid step index, not a candidate index).
/// * `a_table[g, j] = Σ_{i ∈ g} obj[i, k_{i,j}]` (`Vec<f64>`, length
///   `n_groups × n_candidates`).
/// * `b_table[(g · n_candidates + j) · n_constraints + c] = Σ_{i ∈ g}
///   cons_c[i, k_{i,j}]` — flat layout with stride 1 over `c` so the
///   K-loop in `accumulate_from_tables` reads constraint values
///   contiguously per (g, j) cell.
///
/// `target_idx` is large (5 M × 4 B at the typical size) and lives in the
/// thread-local pool to skip per-call `mmap` traffic; `a_table` /
/// `b_table` are KB-scale and re-allocated each call.
struct RemapTables {
    target_idx: Vec<u32>,
    a_table: Vec<f64>,
    b_table: Vec<f64>,
    n_groups: usize,
    n_candidates: usize,
    n_constraints: usize,
    clamp_count: u64,
    total_remaps: u64,
}

impl RemapTables {
    /// Estimated transient memory in bytes for the cache, used by the
    /// `MAX_REMAP_TABLE_BYTES` gate to decide whether to materialise it
    /// or fall back to per-iteration recompute.
    fn estimate_bytes(
        n_quotes: usize,
        n_groups: usize,
        n_candidates: usize,
        n_constraints: usize,
    ) -> usize {
        let target_bytes = n_quotes
            .saturating_mul(n_candidates)
            .saturating_mul(std::mem::size_of::<u32>());
        let aggregate_cells = n_groups.saturating_mul(n_candidates);
        let a_bytes = aggregate_cells.saturating_mul(std::mem::size_of::<f64>());
        let b_bytes = aggregate_cells
            .saturating_mul(n_constraints)
            .saturating_mul(std::mem::size_of::<f64>());
        target_bytes.saturating_add(a_bytes).saturating_add(b_bytes)
    }
}

impl Drop for RemapTables {
    /// Return the `target_idx` buffer to the thread-local pool so the next
    /// `solve_grouped` call on this thread can re-use the allocation
    /// instead of paying for a fresh `mmap`. The buffer is dropped (freed)
    /// only when the thread ends or the pool is replaced. Capping the
    /// retained capacity at the current length keeps a single
    /// pathologically-large solve from permanently inflating the
    /// thread-local resident set.
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.target_idx);
        let _ = TARGET_IDX_SCRATCH.try_with(|cell| {
            let mut slot = cell.borrow_mut();
            // Keep the buffer with the larger capacity so a steady-state
            // sweep never reallocates after the first call. If a tiny
            // call happens after a huge one we accept the larger buffer
            // (cheap to re-use; freeing here would defeat the pool).
            if buf.capacity() > slot.capacity() {
                *slot = buf;
            }
        });
    }
}

/// Precompute the λ-independent caches in a single parallel pass.
///
/// For each (quote i, candidate j) pair we run `nearest_step` once,
/// gather `obj[grid_idx]` and `cons_c[grid_idx]`, write `target_idx[i, j]
/// = k`, and accumulate the gathered values into the per-(group g,
/// candidate j) sums `a_table[g, j]` and `b_table[(g, j), c]`. This is
/// the only place the kernel sweeps all `n_quotes × n_candidates` cells —
/// every subgradient iteration of `solve_grouped` then reads the small
/// per-(g, j) tables instead of re-touching grid memory.
///
/// `target_idx` comes from the thread-local `TARGET_IDX_SCRATCH` pool so
/// steady-state sweeps avoid the multi-MB allocator round-trip; `a_table`
/// / `b_table` are KB-scale so we just allocate per-call.
///
/// Per-fold task identity is `(a_local, b_local, clamp_local)`. The
/// combined per-task footprint is `n_groups × n_candidates × (1 + K) × 8`
/// bytes; if that exceeds `MAX_IDENTITY_BYTES` (4 MB) we drop to a
/// single-threaded sequential walk to bound concurrent memory.
fn precompute_remap_tables(
    grid: &QuoteGrid,
    group_mapping: &GroupMapping,
    residuals: &[f32],
    candidates: &[f32],
) -> RemapTables {
    let n_quotes = grid.n_quotes;
    let n_candidates = candidates.len();
    let n_steps = grid.n_steps;
    let n_groups = group_mapping.n_groups;
    let n_constraints = grid.constraints.len();
    let scenario_values = &grid.scenario_values;
    let group_of = group_mapping.group_of.as_slice();
    let cells = n_quotes * n_candidates;

    // Pull the scratch buffer; if the pool is empty (first call on this
    // thread, or a prior call retained a smaller buffer) we get an empty
    // Vec. `resize(cells, 0)` zero-fills any growth — the parallel pass
    // overwrites every cell so the zero-fill is wasted but bounded
    // (≤cells × 4 bytes, only the first few calls until capacity stabilises).
    let mut target_idx = TARGET_IDX_SCRATCH
        .try_with(|cell| std::mem::take(&mut *cell.borrow_mut()))
        .unwrap_or_default();
    target_idx.clear();
    target_idx.resize(cells, 0u32);

    let objective = grid.objective.as_slice();
    let constraint_cols: Vec<&[f32]> = grid.constraints.iter().map(|c| c.as_slice()).collect();

    let a_size = n_groups * n_candidates;
    let b_size = a_size * n_constraints;

    // Targets `res × candidates[j]` are monotone in `j` whenever the
    // candidates are sorted ascending and this quote's `res ≥ 0`. The
    // ratebook path naturally has non-negative residuals (`overall_mult /
    // factor_value`), so it keeps the two-pointer remap that walks
    // `scenario_values` once per quote instead of running a `log₂(n_steps)`
    // binary search for every (i, j) pair. Public grouped solves may pass
    // negative residuals; those quote rows fall back to `nearest_step`.
    let candidates_ascending = candidates.windows(2).all(|w| w[0] <= w[1]);

    // Single per-quote kernel. Writes `target_slot` (this quote's row of
    // target_idx) and accumulates this quote's contribution into the
    // (a_local, b_local) per-thread reductions. Returns the per-quote
    // clamp count. Captured by reference inside the rayon fold below.
    let process_quote =
        |i: usize, target_slot: &mut [u32], a_local: &mut [f64], b_local: &mut [f64]| -> u64 {
            let g = group_of[i] as usize;
            let res = residuals[i];
            let monotone_fast_path = candidates_ascending && res >= 0.0;
            let base = i * n_steps;
            let a_row = &mut a_local[g * n_candidates..(g + 1) * n_candidates];
            let b_row_base = g * n_candidates * n_constraints;
            let mut local_clamp = 0u64;
            // Two-pointer state for the monotone-candidates fast path. `k`
            // is the current "below" step index; we never decrement it
            // within a quote. Outside the fast path we fall through to
            // `nearest_step` per (i, j).
            let n_steps_minus_one = n_steps.saturating_sub(1);
            let first_sv = scenario_values[0];
            let last_sv = scenario_values[n_steps_minus_one];
            let mut k_ptr: usize = 0;
            for (j, &cand) in candidates.iter().enumerate() {
                let target = res * cand;
                let (k, clamped) = if monotone_fast_path {
                    if target <= first_sv {
                        (0, target < first_sv)
                    } else if target >= last_sv {
                        (n_steps_minus_one, target > last_sv)
                    } else {
                        while k_ptr < n_steps_minus_one && scenario_values[k_ptr + 1] <= target {
                            k_ptr += 1;
                        }
                        let below = scenario_values[k_ptr];
                        let above = scenario_values[k_ptr + 1];
                        if (target - below).abs() <= (above - target).abs() {
                            (k_ptr, false)
                        } else {
                            (k_ptr + 1, false)
                        }
                    }
                } else {
                    nearest_step(scenario_values, target)
                };
                target_slot[j] = k as u32;
                let grid_idx = base + k;
                a_row[j] += objective[grid_idx] as f64;
                let b_cell = b_row_base + j * n_constraints;
                for (c, col) in constraint_cols.iter().enumerate() {
                    b_local[b_cell + c] += col[grid_idx] as f64;
                }
                if clamped {
                    local_clamp += 1;
                }
            }
            local_clamp
        };

    let per_identity_bytes = (a_size + b_size) * std::mem::size_of::<f64>();
    const MAX_IDENTITY_BYTES: usize = 4 * 1024 * 1024;

    let (a_table, b_table, clamp_count) = if per_identity_bytes > MAX_IDENTITY_BYTES {
        // Sequential fallback for pathological group × candidate × constraint
        // sizes where per-thread A+B identities would dominate memory.
        let mut a = vec![0.0f64; a_size];
        let mut b = vec![0.0f64; b_size];
        let mut clamp = 0u64;
        for (i, slot) in target_idx.chunks_mut(n_candidates).enumerate() {
            clamp += process_quote(i, slot, &mut a, &mut b);
        }
        (a, b, clamp)
    } else {
        target_idx
            .par_chunks_mut(n_candidates)
            .with_min_len(GROUPED_PAR_GRAIN)
            .enumerate()
            .fold(
                || (vec![0.0f64; a_size], vec![0.0f64; b_size], 0u64),
                |(mut a_local, mut b_local, mut clamp), (i, target_slot)| {
                    clamp += process_quote(i, target_slot, &mut a_local, &mut b_local);
                    (a_local, b_local, clamp)
                },
            )
            .reduce(
                || (vec![0.0f64; a_size], vec![0.0f64; b_size], 0u64),
                |(mut a1, mut b1, c1), (a2, b2, c2)| {
                    for idx in 0..a1.len() {
                        a1[idx] += a2[idx];
                    }
                    for idx in 0..b1.len() {
                        b1[idx] += b2[idx];
                    }
                    (a1, b1, c1 + c2)
                },
            )
    };

    let total_remaps = cells as u64;

    RemapTables {
        target_idx,
        a_table,
        b_table,
        n_groups,
        n_candidates,
        n_constraints,
        clamp_count,
        total_remaps,
    }
}

/// Compute per-(group, candidate) Lagrangians from the affine-decomposition
/// cache.
///
/// Reads the precomputed `a_table[g, j]` and `b_table[(g, j), c]` and
/// returns `group_l[g, j] = a_table[g, j] + Σ_c sign_c · λ_c · b_table[…]`.
/// This is `O(n_groups · n_candidates · (1 + K))` work — typically a few
/// thousand f64 ops at the standard ratebook size — instead of the
/// `O(n_quotes · n_candidates · (1 + K))` per-iter sweep the original
/// kernel did. The expensive per-quote gather has already been amortised
/// by `precompute_remap_tables`.
///
/// Specialised arms for `K = 0` and `K = 1` (the dominant ratebook case)
/// keep the inner body free of an extra constraint loop. Result is
/// numerically equivalent to the legacy `accumulate_group_lagrangians`
/// up to f64 reduction-order rounding (we reorder the sum into
/// `Σ_i a_i + λ · Σ_i b_i` instead of `Σ_i (a_i + λ · b_i)`; the totals
/// differ at most by a few ULPs).
fn accumulate_from_tables(tables: &RemapTables, lambda_signs: &[f64]) -> Vec<f64> {
    let n_groups = tables.n_groups;
    let n_candidates = tables.n_candidates;
    let n_constraints = tables.n_constraints;
    let a_table = tables.a_table.as_slice();
    let b_table = tables.b_table.as_slice();
    let cells = n_groups * n_candidates;

    let mut group_l = vec![0.0f64; cells];

    match lambda_signs.len() {
        0 => {
            group_l.copy_from_slice(a_table);
        }
        1 => {
            let sign_lam0 = lambda_signs[0];
            // Stride 1 in b_table because n_constraints == 1, so b_table[g·n_cand + j]
            // is the per-(g, j) constraint sum. The two slices are equal-length
            // and aligned: the compiler emits a tight FMA loop here.
            for idx in 0..cells {
                group_l[idx] = a_table[idx] + sign_lam0 * b_table[idx];
            }
        }
        _ => {
            for idx in 0..cells {
                let mut l = a_table[idx];
                let b_offset = idx * n_constraints;
                for (c, &sign_lam) in lambda_signs.iter().enumerate() {
                    l += sign_lam * b_table[b_offset + c];
                }
                group_l[idx] = l;
            }
        }
    }

    group_l
}

/// Compute portfolio-level totals (`total_objective`,
/// `total_constraints`) from the affine cache.
///
/// `a_table[g, j_star]` is already `Σ_{i ∈ g} obj[i, k_{i, j_star}]` —
/// exactly the per-quote sum the legacy `reconstruct_and_accumulate`
/// recomputes via gather every inner iteration. Same for
/// `b_table[(g, j_star), c]` and the per-constraint totals. So once the
/// cache is built we can drop the O(n_quotes) per-iter gather and read
/// the totals back as an O(n_groups × (1 + K)) scalar sum — typically
/// hundreds of f64 ops per iter, not millions.
///
/// Result is numerically equivalent to the legacy gather up to f64
/// reduction-order rounding (a_table aggregates per-quote contributions
/// in precompute's chunk order, the legacy path aggregates in
/// reconstruct's chunk order — algebraically the same value, may differ
/// at the ULP level).
fn compute_totals_from_tables(
    tables: &RemapTables,
    group_best_candidate: &[usize],
) -> (f64, Vec<f64>) {
    let n_groups = tables.n_groups;
    let n_candidates = tables.n_candidates;
    let n_constraints = tables.n_constraints;
    let a_table = tables.a_table.as_slice();
    let b_table = tables.b_table.as_slice();

    let mut total_obj = 0.0f64;
    let mut total_cons = vec![0.0f64; n_constraints];

    for (g, &j_star) in group_best_candidate.iter().enumerate().take(n_groups) {
        let cell = g * n_candidates + j_star;
        total_obj += a_table[cell];
        let cons_offset = cell * n_constraints;
        for (c, total) in total_cons.iter_mut().enumerate().take(n_constraints) {
            *total += b_table[cons_offset + c];
        }
    }

    (total_obj, total_cons)
}

/// Populate per-quote optimal step indices from the cache.
///
/// Used once at the very end of `solve_grouped` (just before building
/// the result) — there's no need to recompute these every inner
/// iteration since only the final `group_best_candidate` actually ships
/// in the result. The body is a parallel u32 lookup with no f32→f64
/// promotion or accumulation, so it's effectively free compared to the
/// gather work the legacy path did per iter.
fn extract_optimal_steps_from_tables(
    tables: &RemapTables,
    group_of: &[u32],
    group_best_candidate: &[usize],
    optimal_steps: &mut [u32],
) {
    let n_candidates = tables.n_candidates;
    let target_idx = tables.target_idx.as_slice();
    optimal_steps
        .par_chunks_mut(RECONSTRUCT_PAR_GRAIN)
        .enumerate()
        .for_each(|(chunk_idx, step_slice)| {
            let start = chunk_idx * RECONSTRUCT_PAR_GRAIN;
            for (local_i, step_out) in step_slice.iter_mut().enumerate() {
                let i = start + local_i;
                let g = group_of[i] as usize;
                let j_star = group_best_candidate[g];
                let cell = i * n_candidates + j_star;
                *step_out = target_idx[cell];
            }
        });
}

/// Per-group argmax: select the candidate with highest accumulated Lagrangian.
fn argmax_groups(
    group_l: &[f64],
    n_groups: usize,
    n_candidates: usize,
    group_best_candidate: &mut [usize],
) {
    for (g, best_cand) in group_best_candidate.iter_mut().enumerate().take(n_groups) {
        let mut best_j = 0;
        let mut best_val = f64::NEG_INFINITY;
        let row_offset = g * n_candidates;
        for j in 0..n_candidates {
            if group_l[row_offset + j] > best_val {
                best_val = group_l[row_offset + j];
                best_j = j;
            }
        }
        *best_cand = best_j;
    }
}

/// Reconstruct per-quote optimal steps from group selections and accumulate totals.
fn reconstruct_and_accumulate(
    grid: &QuoteGrid,
    group_mapping: &GroupMapping,
    residuals: &[f32],
    candidates: &[f32],
    group_best_candidate: &[usize],
    optimal_steps: &mut [u32],
) -> (f64, Vec<f64>) {
    let n_constraints = grid.constraints.len();
    let n_steps = grid.n_steps;

    optimal_steps
        .par_chunks_mut(RECONSTRUCT_PAR_GRAIN)
        .enumerate()
        .fold(
            || (0.0f64, vec![0.0f64; n_constraints]),
            |(mut obj, mut cons), (chunk_idx, step_slice)| {
                let start = chunk_idx * RECONSTRUCT_PAR_GRAIN;
                for (local_i, step_out) in step_slice.iter_mut().enumerate() {
                    let i = start + local_i;
                    let g = group_mapping.group_of[i] as usize;
                    let cand = candidates[group_best_candidate[g]];
                    let target = residuals[i] * cand;
                    let (k, _) = nearest_step(&grid.scenario_values, target);
                    *step_out = k as u32;
                    let idx = i * n_steps + k;
                    obj += grid.objective[idx] as f64;
                    for (c, con_total) in cons.iter_mut().enumerate().take(n_constraints) {
                        *con_total += grid.constraints[c][idx] as f64;
                    }
                }
                (obj, cons)
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
        )
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

    if specs.len() != grid.constraints.len() {
        return Err(PriceContourError::DimensionMismatch(format!(
            "specs count {} != grid constraints count {}",
            specs.len(),
            grid.constraints.len()
        )));
    }
    if residuals.len() != grid.n_quotes {
        return Err(PriceContourError::DimensionMismatch(format!(
            "residuals length {} != n_quotes {}",
            residuals.len(),
            grid.n_quotes
        )));
    }
    if group_mapping.group_of.len() != grid.n_quotes {
        return Err(PriceContourError::DimensionMismatch(format!(
            "group_mapping.group_of length {} != n_quotes {}",
            group_mapping.group_of.len(),
            grid.n_quotes
        )));
    }
    if candidates.is_empty() {
        return Err(PriceContourError::InvalidValue(
            "candidates must not be empty".into(),
        ));
    }

    let n_quotes = grid.n_quotes;
    let n_constraints = specs.len();
    let n_groups = group_mapping.n_groups;
    let n_candidates = candidates.len();

    // Compute baselines and scale factors
    let (baseline_obj, baseline_cons, scale_factors) = grid.compute_scale_factors();

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

    // Build the λ-independent remap cache once if it fits within the
    // configured memory budget. Above the cap, fall back to the original
    // per-iteration recompute path (still correct, just slower) so that
    // pathologically large grids don't OOM here.
    let cache_bytes = RemapTables::estimate_bytes(n_quotes, n_groups, n_candidates, n_constraints);
    let tables: Option<RemapTables> = if cache_bytes <= MAX_REMAP_TABLE_BYTES {
        Some(precompute_remap_tables(
            grid,
            group_mapping,
            residuals,
            candidates,
        ))
    } else {
        None
    };

    // Reuse a single signs buffer across all subgradient iterations so
    // we don't pay for the per-iter allocation in `compute_lambda_signs`.
    let mut lambda_signs = vec![0.0f64; n_constraints];

    for iter in 0..config.max_iter {
        fill_lambda_signs(&mut lambda_signs, specs, &lambdas);

        let group_l = match &tables {
            Some(t) => {
                clamp_count = t.clamp_count;
                total_remaps = t.total_remaps;
                accumulate_from_tables(t, &lambda_signs)
            }
            None => {
                let (gl, iter_clamp, iter_remaps) = accumulate_group_lagrangians(
                    grid,
                    group_mapping,
                    residuals,
                    candidates,
                    &lambda_signs,
                    n_groups,
                );
                clamp_count = iter_clamp;
                total_remaps = iter_remaps;
                gl
            }
        };

        argmax_groups(&group_l, n_groups, n_candidates, &mut group_best_candidate);

        // Cached path computes totals directly from a_table / b_table;
        // legacy path recomputes via per-quote gather (and also fills
        // `optimal_steps` in-line, which the cached path defers to a
        // single one-time pass after the loop).
        let (iter_obj, iter_cons) = match &tables {
            Some(t) => compute_totals_from_tables(t, &group_best_candidate),
            None => reconstruct_and_accumulate(
                grid,
                group_mapping,
                residuals,
                candidates,
                &group_best_candidate,
                &mut optimal_steps,
            ),
        };
        total_objective = iter_obj;
        total_constraints = iter_cons;

        // Check constraint satisfaction
        let all_satisfied =
            all_constraints_satisfied(specs, &total_constraints, &baseline_cons, config.tolerance);

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
            &baseline_cons,
            &scale_factors,
            iter,
        );

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

    // If not converged, do a final pass with best/averaged lambdas
    if !converged {
        let final_lambdas =
            select_final_lambdas(best_feasible_obj, best_lambdas, &lambda_sum, iterations);

        fill_lambda_signs(&mut lambda_signs, specs, &final_lambdas);

        let group_l = match &tables {
            Some(t) => {
                clamp_count = t.clamp_count;
                total_remaps = t.total_remaps;
                accumulate_from_tables(t, &lambda_signs)
            }
            None => {
                let (gl, final_clamp, final_remaps) = accumulate_group_lagrangians(
                    grid,
                    group_mapping,
                    residuals,
                    candidates,
                    &lambda_signs,
                    n_groups,
                );
                clamp_count = final_clamp;
                total_remaps = final_remaps;
                gl
            }
        };

        argmax_groups(&group_l, n_groups, n_candidates, &mut group_best_candidate);

        let (iter_obj, iter_cons) = match &tables {
            Some(t) => compute_totals_from_tables(t, &group_best_candidate),
            None => reconstruct_and_accumulate(
                grid,
                group_mapping,
                residuals,
                candidates,
                &group_best_candidate,
                &mut optimal_steps,
            ),
        };
        total_objective = iter_obj;
        total_constraints = iter_cons;

        lambdas = final_lambdas;
    }

    // Cached path defers the per-quote `optimal_steps` population to a
    // single pass at the end of the solve — avoiding the O(n_quotes)
    // gather work the legacy path repeated every iteration. The legacy
    // path has already filled `optimal_steps` in-line during
    // `reconstruct_and_accumulate`, so it skips this branch.
    if let Some(t) = &tables {
        extract_optimal_steps_from_tables(
            t,
            &group_mapping.group_of,
            &group_best_candidate,
            &mut optimal_steps,
        );
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
    use approx::assert_abs_diff_eq;

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
            quote_id_fingerprint: 0,
        }
    }

    #[test]
    fn test_nearest_step() {
        let mults = vec![0.8f32, 0.9, 1.0, 1.1, 1.2];
        assert_eq!(nearest_step(&mults, 1.0), (2, false)); // exact match
        assert_eq!(nearest_step(&mults, 0.87), (1, false)); // closer to 0.9
        assert_eq!(nearest_step(&mults, 0.7), (0, true)); // clamped below
        assert_eq!(nearest_step(&mults, 1.3), (4, true)); // clamped above
        assert_eq!(nearest_step(&mults, 0.97), (2, false)); // closer to 1.0
                                                            // Equidistant case: just check it returns a valid index
        let (idx, _) = nearest_step(&mults, 0.85);
        assert!(
            idx == 0 || idx == 1,
            "equidistant should pick adjacent index"
        );
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

        let grouped_result = solve_grouped(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &specs,
            &config,
            None,
        )
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

    fn make_unconstrained_grid(n: usize, m: usize) -> QuoteGrid {
        let mut obj = vec![0.0f32; n * m];
        let mults: Vec<f32> = (0..m).map(|j| 0.8 + 0.1 * j as f32).collect();

        for q in 0..n {
            let elasticity = 1.0 + 4.0 * (q as f32) / (n as f32);
            let base = 50.0 + 100.0 * (q as f32) / (n as f32);
            for j in 0..m {
                let mult = mults[j];
                let conversion = 1.0 / (1.0 + (elasticity * (mult - 1.0)).exp());
                obj[q * m + j] = base * mult * conversion;
            }
        }

        QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: mults,
            objective: obj,
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    #[test]
    fn test_single_group() {
        // All quotes in one group: should pick a single factor for the whole portfolio
        let n = 20;
        let m = 5;
        let grid = make_unconstrained_grid(n, m);

        let labels = vec!["ALL".to_string(); n];
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![1.0f32; n];
        let candidates: Vec<f32> = (0..21).map(|i| 0.8 + 0.02 * i as f32).collect();

        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };

        let result = solve_grouped(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &[],
            &config,
            None,
        )
        .unwrap();

        assert_eq!(result.optimal_factor_values.len(), 1);
        // All quotes should have the same factor value
        let fv = result.optimal_factor_values[0];
        assert!((0.8..=1.2).contains(&fv), "factor value out of range: {fv}");
    }

    #[test]
    fn test_clamp_rate_with_extreme_residuals() {
        let n = 10;
        let m = 5;
        let grid = make_unconstrained_grid(n, m);

        let labels: Vec<String> = (0..n).map(|i| format!("G{i}")).collect();
        let group_mapping = build_group_mapping(&labels);
        // Very large residuals push targets outside grid
        let residuals = vec![3.0f32; n];
        let candidates = vec![1.0f32];

        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };

        let result = solve_grouped(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &[],
            &config,
            None,
        )
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
        let grid = make_unconstrained_grid(n, m);

        let labels: Vec<String> = (0..n).map(|i| format!("G{i}")).collect();
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![1.0f32; n];
        // Candidates well within grid range
        let candidates = vec![1.0f32];

        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };

        let result = solve_grouped(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &[],
            &config,
            None,
        )
        .unwrap();

        assert_abs_diff_eq!(result.clamp_rate, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_negative_residual_uses_nearest_step_fallback() {
        // Sorted candidates with a negative residual produce descending
        // targets. The two-pointer remap is only valid for monotone
        // increasing targets, so this pins the fallback to the legacy
        // nearest-step path.
        let grid = QuoteGrid {
            n_quotes: 1,
            n_steps: 5,
            scenario_values: vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            objective: vec![-100.0, 100.0, 0.0, -100.0, -100.0],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec!["Q0".to_string()],
            quote_id_fingerprint: 0,
        };
        let labels = vec!["G".to_string()];
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![-1.0f32];
        let candidates = vec![0.0f32, 1.0, 2.0];
        let config = SolverConfig {
            max_iter: 1,
            ..Default::default()
        };

        let result = solve_grouped(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &[],
            &config,
            None,
        )
        .unwrap();

        assert_abs_diff_eq!(result.total_objective, 100.0);
        assert_abs_diff_eq!(result.optimal_factor_values[0], 1.0);
        assert_eq!(result.optimal_steps_per_quote, vec![1]);
    }

    // -----------------------------------------------------------------------
    // Issue 33: Error path tests for solve_grouped
    // -----------------------------------------------------------------------

    #[test]
    fn test_grouped_rejects_empty_candidates() {
        let n = 10;
        let m = 5;
        let grid = make_heterogeneous_grid(n, m);

        let labels: Vec<String> = (0..n).map(|i| format!("G{i}")).collect();
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![1.0f32; n];
        let candidates: Vec<f32> = vec![]; // empty!

        let (_, bc) = grid.baseline_totals();
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: bc[0] * 0.90,
        }];

        let config = SolverConfig::default();
        let err = solve_grouped(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &specs,
            &config,
            None,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("candidates") || msg.contains("empty"),
            "error should mention empty candidates: {msg}"
        );
    }

    #[test]
    fn test_remap_cache_matches_legacy_within_tolerance() {
        // The affine-decomposition cache reorders the per-(group,
        // candidate) reduction from `Σ_i (obj_i + λ · cons_i)` to
        // `Σ_i obj_i + λ · Σ_i cons_i`, so f64 results differ by at most
        // a few ULPs vs the legacy gather kernel. Pin the algebraic
        // equivalence with a relative tolerance keyed off the max
        // absolute value across the result. clamp_count / total_remaps
        // are integer and must still match exactly.
        let n = 200;
        let m = 7;
        let grid = make_heterogeneous_grid(n, m);

        // A mix of single-letter group labels so we exercise both
        // multi-quote groups and the group_of indirection.
        let group_letters = ["A", "B", "C", "D"];
        let labels: Vec<String> = (0..n)
            .map(|i| group_letters[i % group_letters.len()].to_string())
            .collect();
        let group_mapping = build_group_mapping(&labels);

        // Non-trivial residuals (not all 1.0) so the remap actually
        // moves and the cached target_idx exercises a range of steps.
        let residuals: Vec<f32> = (0..n).map(|i| 0.85 + 0.003 * i as f32).collect();
        let candidates: Vec<f32> = (0..11).map(|j| 0.75 + 0.05 * j as f32).collect();

        // Pick lambda_signs deliberately non-zero on every constraint
        // so the inner λ-dependent reduction is non-trivial.
        let lambda_signs = vec![0.37f64; grid.constraints.len()];

        // Legacy path
        let (group_l_legacy, clamp_legacy, remaps_legacy) = accumulate_group_lagrangians(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &lambda_signs,
            group_mapping.n_groups,
        );

        // Cached path
        let tables = precompute_remap_tables(&grid, &group_mapping, &residuals, &candidates);
        let group_l_cached = accumulate_from_tables(&tables, &lambda_signs);

        assert_eq!(group_l_legacy.len(), group_l_cached.len());
        let scale = group_l_legacy
            .iter()
            .map(|v| v.abs())
            .fold(1.0_f64, f64::max);
        let tol = scale * 1e-12;
        for (idx, (a, b)) in group_l_legacy.iter().zip(group_l_cached.iter()).enumerate() {
            assert_abs_diff_eq!(*a, *b, epsilon = tol);
            let _ = idx; // suppress unused warning in non-failure path
        }
        assert_eq!(clamp_legacy, tables.clamp_count, "clamp_count drift");
        assert_eq!(remaps_legacy, tables.total_remaps, "total_remaps drift");

        // Reconstruct: pick a synthetic group_best_candidate that's not
        // all-zeros so we hit varied (i, j_star) lookups.
        let group_best_candidate: Vec<usize> = (0..group_mapping.n_groups)
            .map(|g| (g * 3) % candidates.len())
            .collect();

        let mut steps_legacy = vec![0u32; n];
        let (obj_legacy, cons_legacy) = reconstruct_and_accumulate(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &group_best_candidate,
            &mut steps_legacy,
        );

        // Cached totals come from a_table / b_table — algebraically
        // equal to the legacy gather but in a different f64 reduction
        // order, so accept ULP-level drift just like the group_l check
        // above.
        let (obj_cached, cons_cached) = compute_totals_from_tables(&tables, &group_best_candidate);

        let mut steps_cached = vec![0u32; n];
        extract_optimal_steps_from_tables(
            &tables,
            &group_mapping.group_of,
            &group_best_candidate,
            &mut steps_cached,
        );

        // optimal_steps must match exactly (integer indices, no f64
        // arithmetic involved on either path).
        assert_eq!(steps_legacy, steps_cached, "optimal_steps drift");

        let obj_scale = obj_legacy.abs().max(1.0);
        assert_abs_diff_eq!(obj_legacy, obj_cached, epsilon = obj_scale * 1e-12);

        assert_eq!(cons_legacy.len(), cons_cached.len());
        let cons_scale = cons_legacy.iter().map(|v| v.abs()).fold(1.0_f64, f64::max);
        let cons_tol = cons_scale * 1e-12;
        for (a, b) in cons_legacy.iter().zip(cons_cached.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = cons_tol);
        }
    }

    #[test]
    fn test_remap_cache_estimate_includes_aggregate_tables() {
        let n_quotes = 100;
        let n_groups = 7;
        let n_candidates = 11;
        let n_constraints = 3;

        let expected = n_quotes * n_candidates * std::mem::size_of::<u32>()
            + n_groups * n_candidates * std::mem::size_of::<f64>()
            + n_groups * n_candidates * n_constraints * std::mem::size_of::<f64>();

        assert_eq!(
            RemapTables::estimate_bytes(n_quotes, n_groups, n_candidates, n_constraints),
            expected
        );
    }

    #[test]
    fn test_grouped_rejects_residuals_length_mismatch() {
        let n = 10;
        let m = 5;
        let grid = make_heterogeneous_grid(n, m);

        let labels: Vec<String> = (0..n).map(|i| format!("G{i}")).collect();
        let group_mapping = build_group_mapping(&labels);
        let residuals = vec![1.0f32; n + 5]; // wrong length
        let candidates = vec![1.0f32];

        let (_, bc) = grid.baseline_totals();
        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: bc[0] * 0.90,
        }];

        let config = SolverConfig::default();
        let err = solve_grouped(
            &grid,
            &group_mapping,
            &residuals,
            &candidates,
            &specs,
            &config,
            None,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("residuals") || msg.contains("n_quotes"),
            "error should mention residuals length mismatch: {msg}"
        );
    }
}
