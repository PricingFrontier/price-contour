use crate::data::{ConstraintSpec, QuoteGrid, SolverConfig};
use crate::error::Result;
use crate::solver::solve_online;

pub struct FrontierConfig {
    pub n_points_per_dim: usize,
    pub threshold_ranges: Vec<(f64, f64)>,
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

pub struct FrontierPoint {
    pub thresholds: Vec<f64>,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub lambdas: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub sv_stats: ScenarioValueStats,
}

pub struct FrontierResult {
    pub points: Vec<FrontierPoint>,
    pub constraint_names: Vec<String>,
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
fn compute_sv_stats(optimal_steps: &[u32], grid: &QuoteGrid) -> ScenarioValueStats {
    let n = optimal_steps.len();
    if n == 0 {
        return ScenarioValueStats {
            mean: 0.0, std: 0.0, min: 0.0,
            p5: 0.0, p25: 0.0, median: 0.0, p75: 0.0, p95: 0.0,
            max: 0.0, pct_increase: 0.0, pct_decrease: 0.0,
        };
    }

    let mut vals: Vec<f64> = optimal_steps
        .iter()
        .map(|&step| grid.scenario_values[step as usize] as f64)
        .collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let sum: f64 = vals.iter().sum();
    let mean = sum / n as f64;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    let n_increase = vals.iter().filter(|&&v| v > 1.0).count();
    let n_decrease = vals.iter().filter(|&&v| v < 1.0).count();

    ScenarioValueStats {
        mean,
        std: variance.sqrt(),
        min: vals[0],
        p5: percentile(&vals, 0.05),
        p25: percentile(&vals, 0.25),
        median: percentile(&vals, 0.50),
        p75: percentile(&vals, 0.75),
        p95: percentile(&vals, 0.95),
        max: vals[n - 1],
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
            da.partial_cmp(&db).unwrap()
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

/// Sweep the efficient frontier by solving at each threshold combination.
/// Uses warm-start from the nearest previously solved point (NN ordering).
///
/// If `initial_lambdas` is provided, the first point in the NN ordering
/// warm-starts from these lambdas instead of cold-starting from zero.
/// This is useful when a prior `solve_online` has already been run on the
/// same grid — its lambdas will be close to the first frontier point.
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
    let n_points = threshold_combos.len();

    if n_points == 0 {
        return Ok(FrontierResult {
            points: vec![],
            constraint_names: specs_template.iter().map(|s| s.name.clone()).collect(),
        });
    }

    // NN ordering for warm-start efficiency
    let order = nn_order(&threshold_combos, &frontier_config.threshold_ranges);

    let mut points = Vec::with_capacity(n_points);
    let mut prev_lambdas: Option<Vec<f64>> = initial_lambdas.map(|l| l.to_vec());

    for &idx in &order {
        let thresholds = &threshold_combos[idx];

        // Build specs with current thresholds
        let specs: Vec<ConstraintSpec> = specs_template
            .iter()
            .zip(thresholds.iter())
            .map(|(template, &threshold)| ConstraintSpec {
                name: template.name.clone(),
                direction: template.direction,
                threshold,
            })
            .collect();

        let result = solve_online(
            grid,
            &specs,
            solver_config,
            prev_lambdas.as_deref(),
        )?;

        prev_lambdas = Some(result.lambdas.clone());
        let sv_stats = compute_sv_stats(&result.optimal_steps, grid);

        points.push((
            idx,
            FrontierPoint {
                thresholds: thresholds.clone(),
                total_objective: result.total_objective,
                total_constraints: result.total_constraints,
                lambdas: result.lambdas,
                iterations: result.iterations,
                converged: result.converged,
                sv_stats,
            },
        ));
    }

    // Re-sort by original index order (cartesian product order)
    points.sort_by_key(|(idx, _)| *idx);
    let points: Vec<FrontierPoint> = points.into_iter().map(|(_, p)| p).collect();

    Ok(FrontierResult {
        points,
        constraint_names: specs_template.iter().map(|s| s.name.clone()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::*;

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
        };

        let solver_config = SolverConfig {
            max_iter: 100,
            ..Default::default()
        };

        let result = sweep_frontier(&grid, &specs_template, &frontier_config, &solver_config, None).unwrap();

        assert_eq!(result.points.len(), 5);
        for p in &result.points {
            assert!(p.total_objective > 0.0);
            assert_eq!(p.thresholds.len(), 1);
            assert_eq!(p.lambdas.len(), 1);
        }
    }

    #[test]
    fn test_warm_start_reduces_iterations() {
        let grid = make_test_grid();
        let (_, baseline_cons) = grid.baseline_totals();

        let specs_template = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: 0.0,
        }];

        let frontier_config = FrontierConfig {
            n_points_per_dim: 5,
            threshold_ranges: vec![(baseline_cons[0] * 0.85, baseline_cons[0] * 1.0)],
        };

        let solver_config = SolverConfig {
            max_iter: 100,
            ..Default::default()
        };

        let result = sweep_frontier(&grid, &specs_template, &frontier_config, &solver_config, None).unwrap();

        // At least some points should converge in fewer iterations than max
        let avg_iters: f64 =
            result.points.iter().map(|p| p.iterations as f64).sum::<f64>() / result.points.len() as f64;
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
        let grid = make_test_grid();
        let (_, baseline_cons) = grid.baseline_totals();

        let specs = vec![ConstraintSpec {
            name: "volume".to_string(),
            direction: ConstraintDirection::Min,
            threshold: baseline_cons[0] * 0.90,
        }];

        let config = SolverConfig {
            max_iter: 200,
            ..Default::default()
        };

        // First, solve to get lambdas
        let solve_result = crate::solver::solve_online(&grid, &specs, &config, None).unwrap();

        // Run frontier without initial_lambdas
        let frontier_config = FrontierConfig {
            n_points_per_dim: 5,
            threshold_ranges: vec![(baseline_cons[0] * 0.85, baseline_cons[0] * 1.0)],
        };

        let cold_result = sweep_frontier(&grid, &specs, &frontier_config, &config, None).unwrap();
        let cold_total_iters: usize = cold_result.points.iter().map(|p| p.iterations).sum();

        // Run frontier with initial_lambdas from the prior solve
        let warm_result = sweep_frontier(
            &grid, &specs, &frontier_config, &config,
            Some(&solve_result.lambdas),
        ).unwrap();
        let warm_total_iters: usize = warm_result.points.iter().map(|p| p.iterations).sum();

        // Warm-started frontier should use no more total iterations than cold
        assert!(
            warm_total_iters <= cold_total_iters,
            "warm-start ({warm_total_iters}) should use <= iterations than cold ({cold_total_iters})"
        );
    }
}
