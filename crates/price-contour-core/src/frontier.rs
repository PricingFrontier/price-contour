use rayon::prelude::*;

use crate::data::{ConstraintDirection, ConstraintSpec, QuoteGrid, SolverConfig};
use crate::error::Result;
use crate::solver::apply_lambdas_no_baselines;
use crate::solver::solve_online_with_precomputed;

/// Hard upper bound on lambda during 1-D bisection bracket expansion.
///
/// At this value the constraint penalty dominates the Lagrangian by ~1e9
/// per unit of constraint violation, so any feasible target has long
/// since been satisfied. If the bracket-expansion phase hits this cap
/// without satisfying the target, the target is treated as infeasible.
const FRONTIER_LAMBDA_BISECTION_CAP: f64 = 1.0e9;

/// Hard cap on bisection halving steps after the bracket is established.
///
/// f64 has 52 mantissa bits, so a bracket of `[0, 1e9]` underflows the
/// representable interval at iter ≈ 53 (`1e9 / 2^53 ≈ 1.1e-7`). The
/// `bracket_tolerance` early-break in [`solve_point_1d_bisection`] usually
/// fires well before this — the cap is purely defensive against
/// pathological brackets.
const FRONTIER_LAMBDA_BISECTION_MAX_ITER: usize = 56;

/// Cap on bracket-expansion doublings before declaring infeasibility.
///
/// Doubling from `lo=1.0` reaches the lambda cap (`1e9`) at iter 30
/// (`2^30 ≈ 1.07e9`); the `hi >= cap` guard inside the expand loop fires
/// first. Holding this cap at 32 keeps a small safety margin without
/// allowing the loop to run further than reachable.
const FRONTIER_BRACKET_EXPAND_MAX_ITER: usize = 32;

/// Configuration for the efficient frontier sweep.
#[derive(Debug, Clone)]
pub struct FrontierConfig {
    /// Number of equally-spaced points per constraint dimension.
    pub n_points_per_dim: usize,
    /// Per-constraint (lo, hi) absolute threshold range.
    pub threshold_ranges: Vec<(f64, f64)>,
    /// Safety cap on total cartesian-product points (prevents accidental blow-up).
    pub max_total_points: Option<usize>,
    /// When true, solve each frontier point in parallel using rayon.
    /// Each point solves independently without warm-starting from neighbours.
    pub parallel: bool,
}

/// Distribution statistics for per-quote optimal scenario values.
#[derive(Debug, Clone)]
pub struct ScenarioValueStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub p5: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub p95: f64,
    pub max: f64,
    pub pct_increase: f64,
    pub pct_decrease: f64,
}

/// Which solver produced a frontier point.
///
/// 1-D sweeps go through bisection (log-time, monotone). Multi-constraint
/// sweeps go through the iterative subgradient solver. The two paths
/// have different work units and different convergence semantics, and
/// downstream consumers can use this enum to disambiguate without
/// re-deriving from the constraint count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverPath {
    /// Iterative subgradient solver — one or more constraints; the
    /// `iterations` count is the number of lambda-update rounds.
    Subgradient,
    /// 1-D lambda bisection — exactly one swept constraint; the
    /// `iterations` count is the number of fixed-lambda apply probes
    /// (lambda=0 baseline + bracket-expand doublings + bisection halvings).
    Bisection,
}

/// Why a frontier point reported `converged=false`.
///
/// Distinguishes "the target is structurally unreachable" from
/// "we ran out of budget before getting there". The former is a user-
/// data signal (raise the threshold range, or accept the envelope clamp);
/// the latter is a solver-config signal (raise `max_iter` / loosen
/// `tolerance`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonConvergenceReason {
    /// Bisection bracket expansion hit `FRONTIER_LAMBDA_BISECTION_CAP`
    /// without satisfying the target — the target is above (Min) or
    /// below (Max) the achievable envelope.
    AboveEnvelope,
    /// Bisection bracket expansion ran out of doublings before reaching
    /// the lambda cap. Pathological grids only; in practice `AboveEnvelope`
    /// fires first.
    BracketExpansionExhausted,
    /// Subgradient solver exhausted `solver_config.max_iter` without
    /// settling — lambda updates never converged within tolerance.
    IterationBudgetExhausted,
}

/// A single solved point on the efficient frontier.
#[derive(Debug, Clone)]
pub struct FrontierPoint {
    /// Threshold values used for this point, one per constraint.
    pub thresholds: Vec<f64>,
    /// Total objective at the optimal steps.
    pub total_objective: f64,
    /// Total constraints at the optimal steps.
    pub total_constraints: Vec<f64>,
    /// Converged Lagrange multipliers.
    pub lambdas: Vec<f64>,
    /// Solver work for this point. Units depend on `solver_path`:
    /// subgradient = lambda-update iterations; bisection = apply probes.
    /// The two are NOT comparable across paths.
    pub iterations: usize,
    /// Whether this point's solve found a feasible result. `true` means:
    /// (subgradient) lambda updates settled with all constraints
    /// satisfied within tolerance; (bisection) the returned lambda
    /// satisfies the target one-sidedly. `false` should always be
    /// accompanied by a `non_convergence_reason`.
    pub converged: bool,
    /// Which solver path produced this point. See [`SolverPath`].
    pub solver_path: SolverPath,
    /// Reason the solver reported `converged=false`, or `None` when
    /// converged. See [`NonConvergenceReason`].
    pub non_convergence_reason: Option<NonConvergenceReason>,
    /// Distribution statistics over per-quote optimal scenario values.
    pub sv_stats: ScenarioValueStats,
}

/// Aggregate result of a frontier sweep across constraint threshold combinations.
#[derive(Debug, Clone)]
pub struct FrontierResult {
    /// All solved frontier points, in cartesian-product order.
    pub points: Vec<FrontierPoint>,
    /// Constraint names matching the template ordering.
    pub constraint_names: Vec<String>,
    /// Count of points where the solver converged.
    pub n_converged: usize,
}

/// Linear interpolation percentile on a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = p * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil().min((n - 1) as f64) as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Compute scenario value distribution stats from optimal steps.
///
/// Mean, std, min, max, and increase/decrease counts are computed in a
/// single O(n) pass without sorting. Percentiles require sorting, which
/// is done once afterwards.
fn compute_sv_stats(optimal_steps: &[u32], grid: &QuoteGrid) -> ScenarioValueStats {
    let n = optimal_steps.len();
    if n == 0 {
        return ScenarioValueStats {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            p5: 0.0,
            p25: 0.0,
            median: 0.0,
            p75: 0.0,
            p95: 0.0,
            max: 0.0,
            pct_increase: 0.0,
            pct_decrease: 0.0,
        };
    }

    // Single O(n) pass for mean, min, max, increase/decrease counts
    let mut sum: f64 = 0.0;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let mut n_increase: usize = 0;
    let mut n_decrease: usize = 0;

    let mut vals: Vec<f64> = Vec::with_capacity(n);
    for &step in optimal_steps {
        let v = grid.scenario_values[step as usize] as f64;
        vals.push(v);
        sum += v;
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
        if v > 1.0 {
            n_increase += 1;
        }
        if v < 1.0 {
            n_decrease += 1;
        }
    }

    let mean = sum / n as f64;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    // Sort only for percentile computation
    vals.sort_unstable_by(|a, b| a.total_cmp(b));

    ScenarioValueStats {
        mean,
        std: variance.sqrt(),
        min: min_val,
        p5: percentile(&vals, 0.05),
        p25: percentile(&vals, 0.25),
        median: percentile(&vals, 0.50),
        p75: percentile(&vals, 0.75),
        p95: percentile(&vals, 0.95),
        max: max_val,
        pct_increase: n_increase as f64 / n as f64,
        pct_decrease: n_decrease as f64 / n as f64,
    }
}

/// Greedy nearest-neighbour ordering through normalised threshold space.
fn nn_order(points: &[Vec<f64>], ranges: &[(f64, f64)]) -> Vec<usize> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }

    // Normalise each dimension to [0, 1]
    let normalised: Vec<Vec<f64>> = points
        .iter()
        .map(|p| {
            p.iter()
                .zip(ranges.iter())
                .map(|(&val, &(lo, hi))| {
                    if (hi - lo).abs() < 1e-15 {
                        0.5
                    } else {
                        (val - lo) / (hi - lo)
                    }
                })
                .collect()
        })
        .collect();

    // Start from point closest to origin (baseline = lowest thresholds)
    let start = (0..n)
        .min_by(|&a, &b| {
            let da: f64 = normalised[a].iter().map(|x| x * x).sum();
            let db: f64 = normalised[b].iter().map(|x| x * x).sum();
            da.total_cmp(&db)
        })
        .unwrap_or(0);

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut current = start;

    for _ in 0..n {
        visited[current] = true;
        order.push(current);

        // Find nearest unvisited
        let mut best_dist = f64::INFINITY;
        let mut best_next = 0;
        for j in 0..n {
            if visited[j] {
                continue;
            }
            let dist: f64 = normalised[current]
                .iter()
                .zip(normalised[j].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_next = j;
            }
        }
        current = best_next;
    }

    order
}

/// Generate a linspace from lo to hi with n points.
fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    (0..n)
        .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
        .collect()
}

/// Generate cartesian product of multiple 1D grids.
fn cartesian_product(grids: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if grids.is_empty() {
        return vec![vec![]];
    }
    let mut result = vec![vec![]];
    for grid in grids {
        let mut new_result = Vec::new();
        for existing in &result {
            for &val in grid {
                let mut v = existing.clone();
                v.push(val);
                new_result.push(v);
            }
        }
        result = new_result;
    }
    result
}

/// Outcome of a single bisection solve — the per-point bisection result
/// before it's repackaged into a `FrontierPoint` (which carries the
/// `SolverPath`/`NonConvergenceReason` enums up to the caller).
struct BisectionOutcome {
    optimal_steps: Vec<u32>,
    total_objective: f64,
    total_constraint: f64,
    lambda: f64,
    /// Probe count: lambda=0 + bracket-expand + bisect halvings.
    probes: usize,
    /// `Some(reason)` iff the bisection couldn't satisfy the target —
    /// the result still carries the highest-lambda probe so callers
    /// see the envelope clamp rather than a silent undershoot.
    non_convergence: Option<NonConvergenceReason>,
}

/// Solve a single 1-D frontier point by bisecting over `lambda`.
///
/// For a single-constraint frontier, the lambda → total-constraint mapping
/// is monotone (non-decreasing for `Min`, non-increasing for `Max`) — every
/// quote independently picks an `argmax` step that, as lambda climbs, only
/// moves toward favouring the constraint. This monotonicity lets us bisect
/// lambda directly via fixed-lambda argmax probes, which is dramatically
/// faster and more reliable than the iterative subgradient solver: no
/// 1/√t step decay, no warm-start sensitivity, log-time convergence
/// regardless of how high lambda has to go.
///
/// The contract differs from the subgradient path:
///
/// - **One-sided satisfaction.** `Min` returns the smallest lambda for
///   which `total ≥ target`; `Max` returns the smallest lambda for which
///   `total ≤ target`. Discrete-choice means equality is rarely exactly
///   reachable, but one-sided satisfaction is the right notion for
///   constraints anyway (you don't want to over-constrain).
/// - **Convergence.** Convergence is reported via `BisectionOutcome.non_convergence`:
///   `None` means the returned lambda satisfies the target one-sidedly;
///   `Some(AboveEnvelope)` / `Some(BracketExpansionExhausted)` means we
///   couldn't bracket the target (returns the highest-lambda apply
///   result so callers see the envelope ceiling rather than a silent
///   undershoot).
///
/// `warm_lo` is an optional lower bound on lambda from a neighbour point
/// in a sorted sweep. For a sorted Min sweep with monotone-increasing
/// targets the previous point's lambda is a free lower bracket — so the
/// bisection skips the doublings below it. Pass `None` to cold-start
/// from `lambda=0`.
///
/// `target_residual_tolerance` is the absolute residual on `total - target`
/// that ends bisection early once the bracket has shrunk enough that
/// further halvings can't improve the user-visible answer.
fn solve_point_1d_bisection(
    grid: &QuoteGrid,
    spec: &ConstraintSpec,
    warm_lo: Option<f64>,
    target_residual_tolerance: f64,
) -> Result<BisectionOutcome> {
    let target = spec.threshold;
    let direction = spec.direction;

    // Probe at lambda=0: this is either already-satisfied (lambda=0 wins)
    // or the side of the bracket the target sits across from. We always
    // probe lambda=0 even when warm_lo > 0 — it's needed to detect the
    // already-feasible short-circuit, which the warm-start path can't.
    let zero_pass = apply_at_lambda(grid, spec, 0.0)?;
    let mut probes = 1usize;
    if direction.is_satisfied(zero_pass.total_constraints[0], target) {
        return Ok(BisectionOutcome {
            optimal_steps: zero_pass.optimal_steps,
            total_objective: zero_pass.total_objective,
            total_constraint: zero_pass.total_constraints[0],
            lambda: 0.0,
            probes,
            non_convergence: None,
        });
    }

    // Bracket the target: starting from `warm_lo` (or 1.0), double lambda
    // until the target is one-sidedly satisfied or we hit the cap. The
    // doubling guarantees O(log) probes to reach any achievable lambda
    // within the cap, regardless of magnitude.
    let mut hi = warm_lo.filter(|&w| w > 0.0).unwrap_or(1.0);
    let mut hi_pass = apply_at_lambda(grid, spec, hi)?;
    probes += 1;
    let mut expand_iters = 0usize;
    while !direction.is_satisfied(hi_pass.total_constraints[0], target) {
        if hi >= FRONTIER_LAMBDA_BISECTION_CAP {
            // Target is above (Min) / below (Max) the achievable envelope.
            // Return the highest-lambda apply with the AboveEnvelope reason
            // so callers see the actual ceiling, not a silent undershoot.
            return Ok(BisectionOutcome {
                optimal_steps: hi_pass.optimal_steps,
                total_objective: hi_pass.total_objective,
                total_constraint: hi_pass.total_constraints[0],
                lambda: hi,
                probes,
                non_convergence: Some(NonConvergenceReason::AboveEnvelope),
            });
        }
        if expand_iters >= FRONTIER_BRACKET_EXPAND_MAX_ITER {
            // Pathological case: doubling from `warm_lo` hasn't reached
            // the cap. In practice unreachable from any `warm_lo <= cap`
            // (32 doublings cover 1.0 → ~4e9), but defensive against a
            // future caller passing `warm_lo` close to the cap.
            return Ok(BisectionOutcome {
                optimal_steps: hi_pass.optimal_steps,
                total_objective: hi_pass.total_objective,
                total_constraint: hi_pass.total_constraints[0],
                lambda: hi,
                probes,
                non_convergence: Some(NonConvergenceReason::BracketExpansionExhausted),
            });
        }
        hi = (hi * 2.0).min(FRONTIER_LAMBDA_BISECTION_CAP);
        hi_pass = apply_at_lambda(grid, spec, hi)?;
        probes += 1;
        expand_iters += 1;
    }

    // Bisect [lo, hi] to the smallest lambda that one-sidedly satisfies
    // `target`. The post-loop `hi` is the satisfying endpoint by loop
    // invariant. Two stopping conditions, in priority order:
    //   1. Target residual within tolerance — once `total` is within
    //      `target_residual_tolerance` of `target`, further halvings
    //      can't improve the user-visible answer.
    //   2. Bracket width within f64 precision — falls out of (1) for
    //      well-behaved problems but guards pathological tolerances.
    let mut lo = 0.0f64;
    let mut hi_lambda = hi;
    let mut best_optimal_steps = hi_pass.optimal_steps;
    let mut best_total_obj = hi_pass.total_objective;
    let mut best_total_cons = hi_pass.total_constraints[0];
    let bracket_tolerance = (hi_lambda - lo).max(1.0) * 1e-9;
    for _ in 0..FRONTIER_LAMBDA_BISECTION_MAX_ITER {
        // Residual termination: the satisfying endpoint is close enough
        // to the target that further bisection won't change the
        // user-visible answer.
        let residual = direction.signed_residual(best_total_cons, target).abs();
        if residual <= target_residual_tolerance {
            break;
        }
        if hi_lambda - lo <= bracket_tolerance {
            break;
        }
        let mid = 0.5 * (lo + hi_lambda);
        let mid_pass = apply_at_lambda(grid, spec, mid)?;
        probes += 1;
        if direction.is_satisfied(mid_pass.total_constraints[0], target) {
            hi_lambda = mid;
            best_optimal_steps = mid_pass.optimal_steps;
            best_total_obj = mid_pass.total_objective;
            best_total_cons = mid_pass.total_constraints[0];
        } else {
            lo = mid;
        }
    }

    Ok(BisectionOutcome {
        optimal_steps: best_optimal_steps,
        total_objective: best_total_obj,
        total_constraint: best_total_cons,
        lambda: hi_lambda,
        probes,
        non_convergence: None,
    })
}

/// Single fixed-lambda apply probe. Skips the per-call baseline pass
/// since the bisection caller doesn't need baselines on the result.
#[inline]
fn apply_at_lambda(
    grid: &QuoteGrid,
    spec: &ConstraintSpec,
    lambda: f64,
) -> Result<crate::solver::ApplyPass> {
    apply_lambdas_no_baselines(grid, std::slice::from_ref(spec), &[lambda])
}

/// Sweep the efficient frontier by solving at each threshold combination.
/// Uses warm-start from the nearest previously solved point (NN ordering).
///
/// **1-D fast path.** When `specs_template` has exactly one constraint,
/// each frontier point is solved by bisecting over lambda via
/// [`solve_point_1d_bisection`]. The lambda → total-constraint mapping is
/// monotone in 1-D, so bisection converges in O(log) probes regardless of
/// how high lambda must go — fixing a class of cases where the iterative
/// subgradient solver's 1/√t step decay would undershoot high targets
/// even at 10 000+ iterations. Multi-constraint sweeps stay on the
/// subgradient solver because the lambda space is no longer 1-D-monotone.
///
/// If `initial_lambdas` is provided, the first point in the NN ordering
/// warm-starts from these lambdas instead of cold-starting from zero.
/// This is useful when a prior `solve_online` has already been run on the
/// same grid — its lambdas will be close to the first frontier point.
/// (The 1-D bisection path ignores `initial_lambdas` because bisection
/// doesn't benefit from a warm start; bracket expansion is already
/// O(log) regardless.)
pub fn sweep_frontier(
    grid: &QuoteGrid,
    specs_template: &[ConstraintSpec],
    frontier_config: &FrontierConfig,
    solver_config: &SolverConfig,
    initial_lambdas: Option<&[f64]>,
) -> Result<FrontierResult> {
    grid.validate()?;

    let n_constraints = specs_template.len();
    if frontier_config.threshold_ranges.len() != n_constraints {
        return Err(crate::error::PriceContourError::DimensionMismatch(format!(
            "threshold_ranges length {} != specs count {}",
            frontier_config.threshold_ranges.len(),
            n_constraints
        )));
    }

    // Generate per-dimension grids
    let dim_grids: Vec<Vec<f64>> = frontier_config
        .threshold_ranges
        .iter()
        .map(|&(lo, hi)| linspace(lo, hi, frontier_config.n_points_per_dim))
        .collect();

    // Generate all threshold combinations
    let threshold_combos = cartesian_product(&dim_grids);

    if let Some(cap) = frontier_config.max_total_points {
        if threshold_combos.len() > cap {
            return Err(crate::error::PriceContourError::InvalidValue(format!(
                "Frontier would generate {} points (exceeds max_total_points={}). \
                     Reduce n_points_per_dim or increase max_total_points.",
                threshold_combos.len(),
                cap
            )));
        }
    }

    let n_points = threshold_combos.len();

    if n_points == 0 {
        return Ok(FrontierResult {
            points: vec![],
            constraint_names: specs_template.iter().map(|s| s.name.clone()).collect(),
            n_converged: 0,
        });
    }

    // Dispatch: use the 1-D bisection fast path when there is exactly
    // one swept constraint, otherwise fall back to the iterative
    // subgradient solver. The Python wrapper at
    // `python/price_contour/solver.py::OnlineOptimiser.frontier` already
    // routes any unswept-axis or ratio constraint into the Python
    // orchestrator, so by the time we reach here every constraint has
    // a threshold range and `n_constraints == 1` means a genuinely
    // 1-D lambda search.
    //
    // `n_constraints >= 2` with degenerate axes (lo == hi for one or
    // more) is a valid request — e.g. "evaluate at this single point
    // under the full multi-axis subgradient solver" or "scan one
    // axis with the other pinned to a constant". The math handles
    // pinned axes correctly: the solver still has to find a λ that
    // satisfies the pinned threshold; it just doesn't sweep it. We
    // accept a small perf wart on multi-axis sweeps that could be
    // collapsed to bisection upstream — the wart is documented but
    // not enforced, because enforcement breaks legitimate callers
    // (e.g. `test_subgradient_iteration_budget_exhausted_reason_emitted`
    // pins both axes to a corner point to force non-convergence).
    let use_bisection = n_constraints == 1;

    // Pre-compute baselines once. The subgradient path additionally
    // needs scale factors; the bisection path doesn't, so skip that
    // O(n_quotes) pass when the dispatch is bisection-only.
    let baseline_obj: f64;
    let baseline_cons: Vec<f64>;
    let scale_factors: Vec<f64>;
    if use_bisection {
        let (b_obj, b_cons) = grid.baseline_totals();
        baseline_obj = b_obj;
        baseline_cons = b_cons;
        scale_factors = Vec::new();
    } else {
        let (b_obj, b_cons, sf) = grid.compute_scale_factors();
        baseline_obj = b_obj;
        baseline_cons = b_cons;
        scale_factors = sf;
    }

    let names: Vec<String> = specs_template.iter().map(|s| s.name.clone()).collect();

    // Build a `FrontierPoint` from a subgradient SolveResult.
    //
    // The mirror "build from BisectionOutcome" lives inside
    // `run_1d_bisection_sweep` — the bisection path has only one
    // caller (here) so inlining the conversion keeps that helper
    // self-contained without an extra closure parameter.
    let subgradient_to_point = |thresholds: Vec<f64>,
                                result: crate::data::SolveResult,
                                sv_stats: ScenarioValueStats|
     -> FrontierPoint {
        let non_convergence_reason = if result.converged {
            None
        } else {
            Some(NonConvergenceReason::IterationBudgetExhausted)
        };
        FrontierPoint {
            thresholds,
            total_objective: result.total_objective,
            total_constraints: result.total_constraints,
            lambdas: result.lambdas,
            iterations: result.iterations,
            converged: result.converged,
            solver_path: SolverPath::Subgradient,
            non_convergence_reason,
            sv_stats,
        }
    };

    let points: Vec<FrontierPoint> = if use_bisection {
        debug_assert_eq!(n_constraints, 1);
        run_1d_bisection_sweep(
            grid,
            specs_template,
            &threshold_combos,
            initial_lambdas,
            frontier_config.parallel,
            solver_config.tolerance,
            baseline_cons[0].abs(),
        )
    } else if frontier_config.parallel {
        // Parallel subgradient: each point solves independently without
        // warm-starting; initial lambdas are broadcast to all points.
        let init_lam = initial_lambdas.map(|l| l.to_vec());
        let mut indexed: Vec<(usize, FrontierPoint)> = threshold_combos
            .par_iter()
            .enumerate()
            .filter_map(|(idx, thresholds)| {
                let specs = build_specs(specs_template, thresholds);
                let result = solve_online_with_precomputed(
                    grid,
                    &specs,
                    solver_config,
                    init_lam.as_deref(),
                    baseline_obj,
                    baseline_cons.clone(),
                    scale_factors.clone(),
                )
                .ok()?;
                let sv_stats = compute_sv_stats(&result.optimal_steps, grid);
                Some((
                    idx,
                    subgradient_to_point(thresholds.clone(), result, sv_stats),
                ))
            })
            .collect();
        indexed.sort_by_key(|(idx, _)| *idx);
        indexed.into_iter().map(|(_, p)| p).collect()
    } else {
        // Sequential subgradient: NN ordering so adjacent points
        // warm-start from each other.
        let order = nn_order(&threshold_combos, &frontier_config.threshold_ranges);
        let mut indexed_points = Vec::with_capacity(n_points);
        let mut prev_lambdas: Option<Vec<f64>> = initial_lambdas.map(|l| l.to_vec());

        for &idx in &order {
            let thresholds = &threshold_combos[idx];
            let specs = build_specs(specs_template, thresholds);
            let result = match solve_online_with_precomputed(
                grid,
                &specs,
                solver_config,
                prev_lambdas.as_deref(),
                baseline_obj,
                baseline_cons.clone(),
                scale_factors.clone(),
            ) {
                Ok(r) => r,
                Err(_) => continue,
            };
            prev_lambdas = Some(result.lambdas.clone());
            let sv_stats = compute_sv_stats(&result.optimal_steps, grid);
            indexed_points.push((
                idx,
                subgradient_to_point(thresholds.clone(), result, sv_stats),
            ));
        }
        indexed_points.sort_by_key(|(idx, _)| *idx);
        indexed_points.into_iter().map(|(_, p)| p).collect()
    };

    let n_converged = points.iter().filter(|p| p.converged).count();

    Ok(FrontierResult {
        points,
        constraint_names: names,
        n_converged,
    })
}

/// Build per-point ConstraintSpecs by overlaying threshold values onto
/// the template. Hot path; called once per frontier point.
#[inline]
fn build_specs(template: &[ConstraintSpec], thresholds: &[f64]) -> Vec<ConstraintSpec> {
    template
        .iter()
        .zip(thresholds.iter())
        .map(|(t, &threshold)| ConstraintSpec {
            name: t.name.clone(),
            direction: t.direction,
            threshold,
        })
        .collect()
}

/// Convert a bisection outcome into a `FrontierPoint`. Single caller
/// (the bisection sweep below); kept as a free helper to mirror the
/// `subgradient_to_point` closure in `sweep_frontier`.
#[inline]
fn bisection_outcome_to_point(
    thresholds: Vec<f64>,
    outcome: BisectionOutcome,
    sv_stats: ScenarioValueStats,
) -> FrontierPoint {
    FrontierPoint {
        thresholds,
        total_objective: outcome.total_objective,
        total_constraints: vec![outcome.total_constraint],
        lambdas: vec![outcome.lambda],
        iterations: outcome.probes,
        converged: outcome.non_convergence.is_none(),
        solver_path: SolverPath::Bisection,
        non_convergence_reason: outcome.non_convergence,
        sv_stats,
    }
}

/// Per-point bisection target-residual tolerance.
///
/// Translates the user's `solver_config.tolerance` (a fractional
/// tolerance for the subgradient solver) into an absolute residual on
/// `total - target` for the bisection's early-stop criterion. Scales
/// by `max(|target|, |baseline|)` so the tolerance stays meaningful
/// across the regimes:
///
/// * non-degenerate baseline (the typical case) → `tol × baseline`
///   matches the subgradient solver's relative-to-baseline contract;
/// * differential / signed-sum constraint where `baseline ≈ 0` → falls
///   back to `tol × |target|` so a target of 1 000 doesn't get a
///   `2e-16` tolerance;
/// * both zero (degenerate) → floor at `f64::EPSILON` so the loop
///   still terminates.
#[inline]
fn bisection_residual_tolerance(target: f64, baseline_abs: f64, relative_tolerance: f64) -> f64 {
    let scale = target.abs().max(baseline_abs);
    (relative_tolerance * scale).max(f64::EPSILON)
}

/// Run a 1-D bisection sweep — sequential mode warm-starts each point
/// from the previous (sorted) point's lambda, parallel mode cold-starts
/// every point.
///
/// Sequential mode sorts threshold combos in monotone-direction order so
/// the previous point's lambda is a valid lower bound on the current
/// point's lambda. For Min, the sweep walks targets ascending (tighter
/// targets need higher lambda); for Max, descending (lower targets need
/// higher lambda). Both leave the emitted `FrontierResult` in
/// cartesian-product order.
fn run_1d_bisection_sweep(
    grid: &QuoteGrid,
    specs_template: &[ConstraintSpec],
    threshold_combos: &[Vec<f64>],
    initial_lambdas: Option<&[f64]>,
    parallel: bool,
    relative_tolerance: f64,
    baseline_abs: f64,
) -> Vec<FrontierPoint> {
    debug_assert_eq!(specs_template.len(), 1);
    let n_points = threshold_combos.len();

    if parallel {
        // Cold-start every point — bracket-expand is O(log) so the loss
        // from skipping warm-start is small relative to the parallel win.
        let warm = initial_lambdas.and_then(|l| l.first().copied());
        let mut indexed: Vec<(usize, FrontierPoint)> = threshold_combos
            .par_iter()
            .enumerate()
            .filter_map(|(idx, thresholds)| {
                let specs = build_specs(specs_template, thresholds);
                let tol = bisection_residual_tolerance(
                    specs[0].threshold,
                    baseline_abs,
                    relative_tolerance,
                );
                let outcome = solve_point_1d_bisection(grid, &specs[0], warm, tol).ok()?;
                let sv_stats = compute_sv_stats(&outcome.optimal_steps, grid);
                Some((
                    idx,
                    bisection_outcome_to_point(thresholds.clone(), outcome, sv_stats),
                ))
            })
            .collect();
        indexed.sort_by_key(|(idx, _)| *idx);
        return indexed.into_iter().map(|(_, p)| p).collect();
    }

    // Sequential: walk targets in monotone-direction order so each point
    // warm-starts from the previous point's lambda.
    let direction = specs_template[0].direction;
    let mut order: Vec<usize> = (0..n_points).collect();
    order.sort_by(|&a, &b| {
        let ta = threshold_combos[a][0];
        let tb = threshold_combos[b][0];
        match direction {
            ConstraintDirection::Min => ta.total_cmp(&tb),
            ConstraintDirection::Max => tb.total_cmp(&ta),
        }
    });

    let mut indexed_points = Vec::with_capacity(n_points);
    let mut warm = initial_lambdas.and_then(|l| l.first().copied());
    for &idx in &order {
        let thresholds = &threshold_combos[idx];
        let specs = build_specs(specs_template, thresholds);
        let tol =
            bisection_residual_tolerance(specs[0].threshold, baseline_abs, relative_tolerance);
        let outcome = match solve_point_1d_bisection(grid, &specs[0], warm, tol) {
            Ok(o) => o,
            Err(_) => continue,
        };
        // Warm-start gating, two cases:
        //
        // 1. `non_convergence.is_some()` (above-envelope) — the returned
        //    lambda is at the cap. Propagating it would over-tighten the
        //    bracket for the (still infeasible) successor and waste
        //    bisection halvings.
        // 2. `outcome.lambda == 0.0` (already-feasible) — the lambda=0
        //    short-circuit fired. A `0.0` warm_lo is filtered out inside
        //    `solve_point_1d_bisection` anyway (we'd cold-start at 1.0),
        //    so propagating it gains nothing; explicit guard is purely
        //    documentary.
        //
        // Anything else propagates: a converged-with-positive-lambda
        // outcome is a valid lower bracket for the next sorted target.
        if outcome.non_convergence.is_none() && outcome.lambda > 0.0 {
            warm = Some(outcome.lambda);
        }
        let sv_stats = compute_sv_stats(&outcome.optimal_steps, grid);
        indexed_points.push((
            idx,
            bisection_outcome_to_point(thresholds.clone(), outcome, sv_stats),
        ));
    }
    indexed_points.sort_by_key(|(idx, _)| *idx);
    indexed_points.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::*;
    use approx::assert_abs_diff_eq;

    fn make_test_grid() -> QuoteGrid {
        let n = 100;
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

        QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            objective: obj,
            constraints: vec![vol],
            constraint_names: vec!["volume".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    #[test]
    fn test_nn_order_visits_all() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![0.0, 1.0],
        ];
        let ranges = vec![(0.0, 1.0), (0.0, 1.0)];
        let order = nn_order(&points, &ranges);
        assert_eq!(order.len(), 4);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_1d_frontier() {
        let grid = make_test_grid();
        let (_, baseline_cons) = grid.baseline_totals();

        let specs_template = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.0, // will be replaced
        }];

        let frontier_config = FrontierConfig {
            n_points_per_dim: 5,
            threshold_ranges: vec![(baseline_cons[0] * 0.85, baseline_cons[0] * 1.0)],
            max_total_points: Some(10_000),
            parallel: false,
        };

        let solver_config = SolverConfig {
            max_iter: 100,
            ..Default::default()
        };

        let result = sweep_frontier(
            &grid,
            &specs_template,
            &frontier_config,
            &solver_config,
            None,
        )
        .unwrap();

        assert_eq!(result.points.len(), 5);
        for p in &result.points {
            assert!(p.total_objective > 0.0);
            assert_eq!(p.thresholds.len(), 1);
            assert_eq!(p.lambdas.len(), 1);
        }
    }

    /// Build a 2-D fixture (two constraints) so warm-start tests exercise
    /// the subgradient solver path. The 1-D bisection fast path bypasses
    /// the iterative update schedule entirely, so any test that wants to
    /// validate warm-start *must* sweep at least two constraint
    /// dimensions.
    fn make_test_grid_2d() -> QuoteGrid {
        let n = 100;
        let m = 5;
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];
        let mut loss = vec![0.0f32; n * m];
        for q in 0..n {
            let elasticity = 1.0 + 4.0 * (q as f32) / (n as f32);
            let base = 50.0 + 100.0 * (q as f32) / (n as f32);
            for j in 0..m {
                let mult = 0.8 + 0.1 * j as f32;
                let conversion = 1.0 / (1.0 + (elasticity * (mult - 1.0)).exp());
                obj[q * m + j] = base * mult * conversion;
                vol[q * m + j] = conversion;
                // Loss ratio increases with mult (to give the second
                // constraint a non-trivial trade-off vs volume).
                loss[q * m + j] = 0.5 + 0.3 * (mult - 0.8);
            }
        }
        QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            objective: obj,
            constraints: vec![vol, loss],
            constraint_names: vec!["volume".to_string(), "loss".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    #[test]
    fn test_warm_start_reduces_iterations() {
        // Multi-constraint sweep so the subgradient solver runs (the 1-D
        // bisection fast path bypasses warm-start entirely — see the
        // dispatch comment in `sweep_frontier`). With two constraints we
        // exercise the iterative path that warm-start actually helps.
        let grid = make_test_grid_2d();
        let (_, baseline_cons) = grid.baseline_totals();

        let specs_template = vec![
            ConstraintSpec {
                name: "volume".to_string(),
                direction: ConstraintDirection::Min,
                threshold: 0.0,
            },
            ConstraintSpec {
                name: "loss".to_string(),
                direction: ConstraintDirection::Max,
                threshold: 0.0,
            },
        ];

        let frontier_config = FrontierConfig {
            n_points_per_dim: 4,
            threshold_ranges: vec![
                (baseline_cons[0] * 0.85, baseline_cons[0] * 1.0),
                (baseline_cons[1] * 1.0, baseline_cons[1] * 1.10),
            ],
            max_total_points: Some(10_000),
            parallel: false,
        };

        let solver_config = SolverConfig {
            max_iter: 100,
            ..Default::default()
        };

        let result = sweep_frontier(
            &grid,
            &specs_template,
            &frontier_config,
            &solver_config,
            None,
        )
        .unwrap();

        // At least some points should converge in fewer iterations than
        // max — warm-start from the previous NN-ordered point shrinks
        // the per-point iteration budget on average.
        let avg_iters: f64 = result
            .points
            .iter()
            .map(|p| p.iterations as f64)
            .sum::<f64>()
            / result.points.len() as f64;
        assert!(
            avg_iters < solver_config.max_iter as f64,
            "warm-start should reduce average iterations: avg={avg_iters}"
        );
    }

    #[test]
    fn test_cartesian_product() {
        let grids = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
        let product = cartesian_product(&grids);
        assert_eq!(product.len(), 6);
    }

    #[test]
    fn test_initial_lambdas_warm_start() {
        // Same rationale as `test_warm_start_reduces_iterations`: use a
        // 2-D fixture so the subgradient path runs and `initial_lambdas`
        // is meaningful (the 1-D bisection path ignores warm-starts).
        let grid = make_test_grid_2d();
        let (_, baseline_cons) = grid.baseline_totals();

        let specs = vec![
            ConstraintSpec {
                name: "volume".to_string(),
                direction: ConstraintDirection::Min,
                threshold: baseline_cons[0] * 0.90,
            },
            ConstraintSpec {
                name: "loss".to_string(),
                direction: ConstraintDirection::Max,
                threshold: baseline_cons[1] * 1.05,
            },
        ];

        let config = SolverConfig {
            max_iter: 200,
            ..Default::default()
        };

        // First, solve to get lambdas
        let solve_result = crate::solver::solve_online(&grid, &specs, &config, None).unwrap();

        // Run frontier without initial_lambdas
        let frontier_config = FrontierConfig {
            n_points_per_dim: 4,
            threshold_ranges: vec![
                (baseline_cons[0] * 0.85, baseline_cons[0] * 1.0),
                (baseline_cons[1] * 1.0, baseline_cons[1] * 1.10),
            ],
            max_total_points: Some(10_000),
            parallel: false,
        };

        let cold_result = sweep_frontier(&grid, &specs, &frontier_config, &config, None).unwrap();
        let cold_total_iters: usize = cold_result.points.iter().map(|p| p.iterations).sum();

        // Run frontier with initial_lambdas from the prior solve
        let warm_result = sweep_frontier(
            &grid,
            &specs,
            &frontier_config,
            &config,
            Some(&solve_result.lambdas),
        )
        .unwrap();
        let warm_total_iters: usize = warm_result.points.iter().map(|p| p.iterations).sum();

        // Warm-started frontier should use no more total iterations than cold
        assert!(
            warm_total_iters <= cold_total_iters,
            "warm-start ({warm_total_iters}) should use <= iterations than cold ({cold_total_iters})"
        );
    }

    // -----------------------------------------------------------------------
    // Issue 34: Frontier helper (linspace, cartesian_product) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_linspace_basic() {
        let result = linspace(0.0, 1.0, 5);
        assert_eq!(result.len(), 5);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[4], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_linspace_single_point() {
        let result = linspace(0.5, 1.5, 1);
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_linspace_lo_equals_hi() {
        let result = linspace(1.0, 1.0, 5);
        assert_eq!(result.len(), 5);
        for v in &result {
            assert_abs_diff_eq!(*v, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cartesian_product_empty() {
        let result = cartesian_product(&[]);
        assert_eq!(result.len(), 1); // one empty combination
        assert!(result[0].is_empty());
    }

    // -----------------------------------------------------------------------
    // Issue 6: high-target 1D frontier convergence.
    //
    // The diminishing 1/sqrt(t) step schedule in `update_lambdas_subgradient`
    // makes the iterative solver climb lambda extremely slowly when the
    // target requires lambda values orders of magnitude larger than the
    // step size at iter=0. Bisection over lambda (using `apply_lambdas` per
    // probe) is monotone in 1D constraints and converges in O(log) probes
    // regardless of how high the lambda has to go.
    //
    // Acceptance: every feasible 1D `Min` target satisfies `total >= target`
    // at the returned lambda, with `converged=true`. Targets above the
    // achievable envelope return `converged=false` AND the highest-lambda
    // result we could compute (loud failure, not a silent low-lambda
    // undershoot).
    // -----------------------------------------------------------------------

    /// Build a 1D fixture where the lambda required for a high `Min`
    /// threshold is large (~100s+) — exposing the same shape as the
    /// haute frontier reproducer.
    ///
    /// Each quote's `volume` (the constraint) has a wide envelope:
    /// scenario step 0 gives a tiny volume per quote, the highest step
    /// gives ~10× more. To push the portfolio total above a high
    /// fraction of the achievable maximum, lambda has to be large
    /// because shifting each marginal quote up requires accepting a
    /// large objective hit.
    fn make_high_lambda_grid() -> QuoteGrid {
        let n = 200;
        let m = 11;
        let mut obj = vec![0.0f32; n * m];
        let mut vol = vec![0.0f32; n * m];
        for q in 0..n {
            // Each quote has a different objective curve. Higher steps
            // give more volume but lower objective.
            let base_obj = 100.0 + 50.0 * (q as f32 / n as f32);
            for j in 0..m {
                // Volume increases steeply with step.
                vol[q * m + j] = 1.0 + 9.0 * (j as f32 / (m - 1) as f32);
                // Objective decreases with step — penalty for grabbing volume.
                obj[q * m + j] = base_obj * (1.0 - 0.5 * (j as f32 / (m - 1) as f32));
            }
        }
        QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: (0..m).map(|j| 0.8 + 0.05 * j as f32).collect(),
            objective: obj,
            constraints: vec![vol],
            constraint_names: vec!["volume".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    /// Maximum achievable portfolio total for the constraint by picking,
    /// per quote, the step with the largest constraint value. This is the
    /// upper envelope a `Min` constraint can ever reach.
    fn max_achievable_constraint(grid: &QuoteGrid) -> f64 {
        let m = grid.n_steps;
        let mut max_total = 0.0f64;
        for q in 0..grid.n_quotes {
            let mut quote_max = grid.constraints[0][q * m] as f64;
            for j in 1..m {
                let v = grid.constraints[0][q * m + j] as f64;
                if v > quote_max {
                    quote_max = v;
                }
            }
            max_total += quote_max;
        }
        max_total
    }

    #[test]
    fn test_high_target_1d_min_frontier_converges() {
        // Sweep targets up to ~95 % of the achievable envelope. A subgradient
        // solver with default max_iter undershoots these by hundreds; the
        // bisection path satisfies them.
        let grid = make_high_lambda_grid();
        let (_, baseline_cons) = grid.baseline_totals();
        let max_envelope = max_achievable_constraint(&grid);

        // Lower bound: 5 % above the lambda=0 baseline (so lambda must move).
        // Upper bound: 95 % of the max — feasible but requires high lambda.
        let lo = baseline_cons[0] * 1.05;
        let hi = baseline_cons[0] + 0.95 * (max_envelope - baseline_cons[0]);

        let specs_template = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.0,
        }];
        let frontier_config = FrontierConfig {
            n_points_per_dim: 10,
            threshold_ranges: vec![(lo, hi)],
            max_total_points: Some(10_000),
            parallel: false,
        };
        let solver_config = SolverConfig {
            max_iter: 50, // The user's repro uses 50 — bisection should
            // close every feasible point regardless of this budget.
            ..Default::default()
        };

        let result = sweep_frontier(
            &grid,
            &specs_template,
            &frontier_config,
            &solver_config,
            None,
        )
        .unwrap();

        assert_eq!(result.points.len(), 10);
        // Every point should converge AND satisfy its target one-sidedly
        // (`total >= threshold` for Min). Tolerate a small per-target
        // residual to absorb the discrete-choice gap (each quote selects
        // one of n_steps values, so total is a step function in lambda).
        let envelope = max_envelope;
        for p in &result.points {
            assert_eq!(p.total_constraints.len(), 1);
            let target = p.thresholds[0];
            let actual = p.total_constraints[0];
            // One-sided satisfaction: actual >= target (within a 0.5 %
            // residual relative to the envelope, which is the discrete
            // step granularity).
            let residual = target - actual;
            assert!(
                residual <= envelope * 0.005,
                "point at target={target:.2} undershoots: actual={actual:.2}, \
                 residual={residual:.2}, envelope={envelope:.2}, lambda={}",
                p.lambdas[0]
            );
            assert!(
                p.converged,
                "point at target={target:.2} should converge, got actual={actual:.2}, \
                 lambda={}",
                p.lambdas[0]
            );
        }
    }

    #[test]
    fn test_1d_min_frontier_below_lambda_zero_returns_lambda_zero() {
        // If the target is below the lambda=0 (max-objective) total, no
        // constraint pressure is needed — lambda=0 already satisfies.
        // The bisection path's first probe at lambda=0 short-circuits
        // here, returning lambda=0 in one apply call.
        let grid = make_high_lambda_grid();
        // Compute the actual lambda=0 total (max-objective steps): the
        // bisection's lambda=0 baseline.
        let zero_spec = ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.0,
        };
        let zero_total =
            crate::solver::apply_lambdas(&grid, std::slice::from_ref(&zero_spec), &[0.0])
                .unwrap()
                .total_constraints[0];

        let specs_template = vec![zero_spec];
        // Both endpoints comfortably below the lambda=0 total — lambda
        // must stay at 0.
        let frontier_config = FrontierConfig {
            n_points_per_dim: 3,
            threshold_ranges: vec![(zero_total * 0.5, zero_total * 0.9)],
            max_total_points: Some(10),
            parallel: false,
        };
        let solver_config = SolverConfig::default();

        let result = sweep_frontier(
            &grid,
            &specs_template,
            &frontier_config,
            &solver_config,
            None,
        )
        .unwrap();

        for p in &result.points {
            assert!(
                p.converged,
                "below-lambda-zero target {} should converge at lambda=0",
                p.thresholds[0]
            );
            assert_abs_diff_eq!(p.lambdas[0], 0.0, epsilon = 1e-9);
            assert!(p.total_constraints[0] >= p.thresholds[0]);
        }
    }

    #[test]
    fn test_1d_min_frontier_infeasible_target_marked_not_converged() {
        // A target above the achievable envelope must NOT silently report
        // a low-lambda undershoot. The bisection path detects the
        // infeasibility (its bracket-expansion runs to the cap) and
        // returns the highest-lambda apply with converged=false.
        let grid = make_high_lambda_grid();
        let max_envelope = max_achievable_constraint(&grid);

        let specs_template = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.0,
        }];
        // Above the achievable maximum.
        let infeasible_target = max_envelope * 1.01;
        let frontier_config = FrontierConfig {
            n_points_per_dim: 1,
            threshold_ranges: vec![(infeasible_target, infeasible_target)],
            max_total_points: Some(10),
            parallel: false,
        };
        let solver_config = SolverConfig::default();

        let result = sweep_frontier(
            &grid,
            &specs_template,
            &frontier_config,
            &solver_config,
            None,
        )
        .unwrap();

        let p = &result.points[0];
        assert!(
            !p.converged,
            "infeasible target {} should NOT report converged=true (got actual={}, lambda={})",
            p.thresholds[0], p.total_constraints[0], p.lambdas[0]
        );
        // The reported total should be at most the envelope — i.e. the
        // best lambda we tried, not a low-lambda undershoot.
        assert!(
            p.total_constraints[0] >= max_envelope * 0.99,
            "infeasible-target point should report the high-lambda apply result \
             (≈envelope), got {}",
            p.total_constraints[0]
        );
    }

    #[test]
    fn test_high_target_1d_max_frontier_converges() {
        // Symmetric Max-direction test: high lambda needed to push total
        // DOWN below a low target.
        let grid = make_high_lambda_grid();
        let (_, baseline_cons) = grid.baseline_totals();

        // For Max constraint, lambda=0 gives the maximum total (no penalty).
        // Targets BELOW baseline require lambda > 0 to push total down.
        let baseline = baseline_cons[0];
        // Cover from 95 % of baseline (mild) down to 30 % of baseline
        // (aggressive — needs high lambda).
        let lo = baseline * 0.30;
        let hi = baseline * 0.95;

        let specs_template = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Max,
            threshold: 0.0,
        }];
        let frontier_config = FrontierConfig {
            n_points_per_dim: 8,
            threshold_ranges: vec![(lo, hi)],
            max_total_points: Some(10_000),
            parallel: false,
        };
        let solver_config = SolverConfig {
            max_iter: 50,
            ..Default::default()
        };

        let result = sweep_frontier(
            &grid,
            &specs_template,
            &frontier_config,
            &solver_config,
            None,
        )
        .unwrap();

        // For Max direction, one-sided satisfaction is `actual <= target`.
        for p in &result.points {
            let target = p.thresholds[0];
            let actual = p.total_constraints[0];
            let residual = actual - target;
            assert!(
                residual <= baseline * 0.005,
                "point at target={target:.2} overshoots: actual={actual:.2}, \
                 residual={residual:.2}, lambda={}",
                p.lambdas[0]
            );
            assert!(
                p.converged,
                "Max point at target={target:.2} should converge, got actual={actual:.2}, \
                 lambda={}",
                p.lambdas[0]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Boundary-case tests for solve_point_1d_bisection (Test #2)
    //
    // Direct unit tests of the bisection internals so a regression that
    // breaks the lambda=0 short-circuit, the cap detection, or the
    // residual-based termination fires here without needing the whole
    // sweep_frontier pipeline.
    // -----------------------------------------------------------------------

    fn min_spec() -> ConstraintSpec {
        ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.0,
        }
    }

    fn zero_lambda_total(grid: &QuoteGrid, spec: &ConstraintSpec) -> f64 {
        crate::solver::apply_lambdas(grid, std::slice::from_ref(spec), &[0.0])
            .unwrap()
            .total_constraints[0]
    }

    #[test]
    fn test_bisection_target_at_zero_total_short_circuits() {
        // target == lambda=0 baseline → already-feasible branch fires
        // (lambda=0 satisfies for Min).
        let grid = make_high_lambda_grid();
        let mut spec = min_spec();
        spec.threshold = zero_lambda_total(&grid, &spec);

        let outcome = solve_point_1d_bisection(&grid, &spec, None, 1e-6).unwrap();
        assert!(outcome.non_convergence.is_none());
        assert_eq!(outcome.lambda, 0.0);
        assert_eq!(
            outcome.probes, 1,
            "should short-circuit on zero-lambda probe"
        );
    }

    #[test]
    fn test_bisection_target_at_envelope_max_converges() {
        // target == envelope max → reachable in principle, but discrete
        // steps mean the bisection should converge or report
        // BracketExpansionExhausted / AboveEnvelope cleanly. Either way
        // `actual` must sit at-or-just-under env_hi, never far below.
        let grid = make_high_lambda_grid();
        let env_hi = max_achievable_constraint(&grid);
        let mut spec = min_spec();
        spec.threshold = env_hi;

        let outcome = solve_point_1d_bisection(&grid, &spec, None, 1e-6).unwrap();
        let env_span = env_hi - zero_lambda_total(&grid, &spec);
        assert!(
            outcome.total_constraint >= env_hi - env_span * 0.01,
            "boundary target at env_hi gave actual={} far below env_hi={}",
            outcome.total_constraint,
            env_hi
        );
    }

    #[test]
    fn test_bisection_target_one_ulp_above_envelope_marked_infeasible() {
        let grid = make_high_lambda_grid();
        let env_hi = max_achievable_constraint(&grid);
        let mut spec = min_spec();
        spec.threshold = f64::from_bits(env_hi.to_bits() + 1);
        // Just above representable max — must be flagged.

        let outcome = solve_point_1d_bisection(&grid, &spec, None, 1e-6).unwrap();
        assert!(
            outcome.non_convergence.is_some(),
            "1-ULP-above-envelope target should be flagged; got converged with \
             actual={}, lambda={}",
            outcome.total_constraint,
            outcome.lambda
        );
        // The reported lambda should be the cap, the actual should be
        // env_hi (or close to it) — loud-failure contract.
        let env_span = env_hi - zero_lambda_total(&grid, &spec);
        assert!(
            outcome.total_constraint >= env_hi - env_span * 0.01,
            "infeasible-boundary point reports actual far from env_hi: \
             actual={}, env_hi={}",
            outcome.total_constraint,
            env_hi
        );
    }

    #[test]
    fn test_bisection_warm_start_does_not_change_converged_lambda() {
        // Warm-start with a small lo and a large lo — both should
        // converge to the same lambda (within bracket-tolerance) and
        // total. Pins the contract in the dispatch loop's warm-start.
        let grid = make_high_lambda_grid();
        let baseline = grid.baseline_totals().1[0];
        let env_hi = max_achievable_constraint(&grid);
        let mut spec = min_spec();
        spec.threshold = baseline + 0.4 * (env_hi - baseline);

        let cold = solve_point_1d_bisection(&grid, &spec, None, 1e-6).unwrap();
        let warm = solve_point_1d_bisection(&grid, &spec, Some(0.5), 1e-6).unwrap();

        let env_span = env_hi - baseline;
        assert!(
            (cold.total_constraint - warm.total_constraint).abs() <= env_span * 0.005,
            "warm-start changed converged total: cold={}, warm={}",
            cold.total_constraint,
            warm.total_constraint
        );
        assert!(cold.non_convergence.is_none() && warm.non_convergence.is_none());
    }

    #[test]
    fn test_bisection_emits_above_envelope_reason() {
        // Specifically pin the reason enum value, not just `is_some`.
        let grid = make_high_lambda_grid();
        let env_hi = max_achievable_constraint(&grid);
        let mut spec = min_spec();
        spec.threshold = env_hi * 1.10;

        let outcome = solve_point_1d_bisection(&grid, &spec, None, 1e-6).unwrap();
        assert_eq!(
            outcome.non_convergence,
            Some(NonConvergenceReason::AboveEnvelope),
        );
    }

    #[test]
    fn test_bisection_warm_start_reduces_probes_for_high_target() {
        // A high-target case where cold-start has to do many bracket-
        // expand doublings (1.0 → 2.0 → 4.0 → ... → ~700). Warm-starting
        // from a near-optimal lambda lets the bracket form much faster.
        // Pins the *benefit* of warm-start, not just its invariance —
        // a regression that disables `warm_lo` pass-through in
        // run_1d_bisection_sweep would not fail any other test.
        let grid = make_high_lambda_grid();
        let baseline = grid.baseline_totals().1[0];
        let env_hi = max_achievable_constraint(&grid);
        let mut spec = min_spec();
        spec.threshold = baseline + 0.85 * (env_hi - baseline);

        let cold = solve_point_1d_bisection(&grid, &spec, None, 1e-6).unwrap();
        // Warm with the cold's converged lambda — this is exactly what
        // the sequential sweep does for a neighbour with the same target.
        let warm = solve_point_1d_bisection(&grid, &spec, Some(cold.lambda), 1e-6).unwrap();

        assert!(
            warm.probes < cold.probes,
            "warm-start should reduce probe count (cold={}, warm={})",
            cold.probes,
            warm.probes
        );
    }

    #[test]
    fn test_bisection_handles_zero_baseline_constraint() {
        // Differential / signed-sum constraint where the lambda=0
        // baseline sums to zero. Without the per-target tolerance scale
        // (CQ#1) the residual tolerance would collapse to f64::EPSILON
        // and the bisection loop would not terminate on residual.
        let n = 50;
        let m = 5;
        let mut obj = vec![0.0f32; n * m];
        let mut signed = vec![0.0f32; n * m];
        for q in 0..n {
            for j in 0..m {
                let mult = 0.8 + 0.1 * j as f32;
                // Symmetric signed values per quote: equal positive and
                // negative magnitudes summing to zero across all steps.
                signed[q * m + j] = (j as f32) - 2.0; // -2, -1, 0, 1, 2
                                                      // Objective peaks at step 2 (signed = 0), so lambda=0
                                                      // gives signed_total = sum_q(0) = 0.
                obj[q * m + j] = 100.0 * (1.0 - (mult - 1.0).abs());
            }
        }
        let grid = QuoteGrid {
            n_quotes: n,
            n_steps: m,
            scenario_values: vec![0.8, 0.9, 1.0, 1.1, 1.2],
            objective: obj,
            constraints: vec![signed],
            constraint_names: vec!["signed".to_string()],
            quote_ids: (0..n).map(|i| format!("Q{i}")).collect(),
            quote_id_fingerprint: 0,
        };
        let baseline_abs = grid.baseline_totals().1[0].abs();
        // Target 50 (well above the lambda=0 zero baseline) — needs
        // lambda > 0 to push the constraint up.
        let spec = ConstraintSpec {
            name: "signed".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 50.0,
        };
        let tol = bisection_residual_tolerance(spec.threshold, baseline_abs, 1e-5);
        // The tolerance should scale by |target|=50, not collapse to EPS.
        assert!(
            tol > 1e-10,
            "zero-baseline tolerance should fall back to target scale; got {tol}"
        );
        let outcome = solve_point_1d_bisection(&grid, &spec, None, tol).unwrap();
        // Either converged or genuinely infeasible — never a silent
        // EPSILON-tolerance grind that runs to the iter cap.
        assert!(
            outcome.probes < FRONTIER_LAMBDA_BISECTION_MAX_ITER + 4,
            "zero-baseline grid took {} probes (cap={})",
            outcome.probes,
            FRONTIER_LAMBDA_BISECTION_MAX_ITER
        );
    }

    #[test]
    fn test_bracket_expand_cap_reaches_lambda_cap_from_unit_warm() {
        // Pins the constants relationship: doubling from `warm_lo=1.0`
        // for `FRONTIER_BRACKET_EXPAND_MAX_ITER` iterations must exceed
        // `FRONTIER_LAMBDA_BISECTION_CAP` so the cap branch fires before
        // the iter cap branch (the iter cap is purely defensive against
        // a future caller passing warm_lo near the cap). If a future
        // refactor tightens BRACKET_EXPAND_MAX_ITER below log2(CAP), the
        // dead-code defensive branch becomes load-bearing — this test
        // breaks loudly so the constant relationship gets re-examined.
        let max_reachable = (1u128 << FRONTIER_BRACKET_EXPAND_MAX_ITER) as f64;
        assert!(
            max_reachable > FRONTIER_LAMBDA_BISECTION_CAP,
            "BRACKET_EXPAND_MAX_ITER ({}) doublings from 1.0 reach {} which \
             does not exceed FRONTIER_LAMBDA_BISECTION_CAP ({}). The \
             AboveEnvelope cap branch may no longer fire first; revisit the \
             defensive BracketExpansionExhausted branch.",
            FRONTIER_BRACKET_EXPAND_MAX_ITER,
            max_reachable,
            FRONTIER_LAMBDA_BISECTION_CAP
        );
        // Symmetric: bisection-iter cap should be ≥ log2(cap) + a small
        // safety margin so well-behaved sweeps never run to the cap.
        let bisection_precision = 1u128 << FRONTIER_LAMBDA_BISECTION_MAX_ITER;
        assert!(
            (bisection_precision as f64) > FRONTIER_LAMBDA_BISECTION_CAP,
            "BISECTION_MAX_ITER ({}) too small to halve the [0, CAP] bracket \
             to f64 precision",
            FRONTIER_LAMBDA_BISECTION_MAX_ITER
        );
    }
}
