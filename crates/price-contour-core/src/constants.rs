/// Maximum number of Lagrangian iterations before the solver stops.
pub const DEFAULT_MAX_ITER: usize = 50;

/// Convergence tolerance: the solver stops when the maximum lambda change
/// falls below this threshold and all constraints are satisfied.
pub const DEFAULT_TOLERANCE: f64 = 1e-5;

/// Number of quotes processed per memory chunk in the argmax pass.
/// Controls the peak working-set size for parallel processing.
pub const DEFAULT_CHUNK_SIZE: usize = 500_000;

/// Base step-size multiplier for the subgradient lambda update rule.
/// Larger values converge faster but risk oscillation.
pub const SUBGRADIENT_ALPHA: f64 = 0.1;

/// Floor on the subgradient step size to prevent stalling near zero.
pub const SUBGRADIENT_MIN_STEP: f64 = 1e-8;

/// Upper bound on the per-constraint scale factor to prevent numerical
/// blow-up when a constraint baseline is near zero.
pub const MAX_SCALE_FACTOR: f64 = 1000.0;

/// Small epsilon added to constraint baselines before dividing, to avoid
/// division by zero when computing scale factors.
pub const SCALE_EPSILON: f64 = 1e-10;

/// Rayon parallel grain size for per-quote argmax. Each grain processes
/// this many quotes before merging partial results.
pub const ARGMAX_PAR_GRAIN: usize = 4096;

/// Rayon parallel grain size for grouped accumulation (more work per item
/// due to candidate iteration).
pub const GROUPED_PAR_GRAIN: usize = 1024;

/// Rayon parallel grain size for the reconstruction pass in the grouped solver.
pub const RECONSTRUCT_PAR_GRAIN: usize = 4096;
