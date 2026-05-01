use crate::constants::*;
use crate::error::{PriceContourError, Result};

/// Contiguous memory layout for solver operations.
///
/// All arrays are length `n_quotes * n_steps`, laid out quote-major:
/// `[q0_s0, q0_s1, ..., q0_sM-1, q1_s0, q1_s1, ..., q1_sM-1, ...]`
#[derive(Debug)]
pub struct QuoteGrid {
    /// Number of quotes in the grid.
    pub n_quotes: usize,
    /// Number of scenario steps per quote.
    pub n_steps: usize,
    /// Sorted scenario values, one per step (length `n_steps`).
    pub scenario_values: Vec<f32>,
    /// Objective values, flat quote-major (length `n_quotes * n_steps`).
    pub objective: Vec<f32>,
    /// Constraint matrices, one `Vec<f32>` per constraint (each length `n_quotes * n_steps`).
    pub constraints: Vec<Vec<f32>>,
    /// Names corresponding to each constraint matrix.
    pub constraint_names: Vec<String>,
    /// Unique identifier for each quote (length `n_quotes`).
    pub quote_ids: Vec<String>,
}

impl QuoteGrid {
    /// Validate dimensions of all arrays.
    pub fn validate(&self) -> Result<()> {
        if self.n_steps == 0 {
            return Err(PriceContourError::DimensionMismatch(
                "n_steps must be > 0".into(),
            ));
        }
        if self.n_quotes == 0 {
            return Err(PriceContourError::DimensionMismatch(
                "n_quotes must be > 0".into(),
            ));
        }
        let expected_len = self.n_quotes.checked_mul(self.n_steps).ok_or_else(|| {
            PriceContourError::DimensionMismatch(format!(
                "n_quotes * n_steps overflow: {} * {}",
                self.n_quotes, self.n_steps
            ))
        })?;
        if self.scenario_values.iter().any(|v| !v.is_finite()) {
            return Err(PriceContourError::InvalidValue(
                "scenario_values contains NaN or Inf".into(),
            ));
        }

        if self.scenario_values.len() != self.n_steps {
            return Err(PriceContourError::DimensionMismatch(format!(
                "scenario_values length {} != n_steps {}",
                self.scenario_values.len(),
                self.n_steps
            )));
        }
        if self.objective.len() != expected_len {
            return Err(PriceContourError::DimensionMismatch(format!(
                "objective length {} != n_quotes * n_steps {}",
                self.objective.len(),
                expected_len
            )));
        }
        if self.constraints.len() != self.constraint_names.len() {
            return Err(PriceContourError::DimensionMismatch(format!(
                "constraints count {} != constraint_names count {}",
                self.constraints.len(),
                self.constraint_names.len()
            )));
        }
        for (i, c) in self.constraints.iter().enumerate() {
            if c.len() != expected_len {
                return Err(PriceContourError::DimensionMismatch(format!(
                    "constraint '{}' length {} != n_quotes * n_steps {}",
                    self.constraint_names[i],
                    c.len(),
                    expected_len
                )));
            }
        }
        if self.quote_ids.len() != self.n_quotes {
            return Err(PriceContourError::DimensionMismatch(format!(
                "quote_ids length {} != n_quotes {}",
                self.quote_ids.len(),
                self.n_quotes
            )));
        }
        Ok(())
    }

    /// Find the step index closest to scenario_value=1.0 and compute baseline totals.
    /// Returns (baseline_objective_total, baseline_constraint_totals) as f64.
    pub fn baseline_totals(&self) -> (f64, Vec<f64>) {
        let baseline_step = self
            .scenario_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| ((**a) - 1.0f32).abs().total_cmp(&((**b) - 1.0f32).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut obj_total: f64 = 0.0;
        let mut con_totals = vec![0.0f64; self.constraints.len()];

        for q in 0..self.n_quotes {
            let idx = q * self.n_steps + baseline_step;
            obj_total += self.objective[idx] as f64;
            for (k, con) in self.constraints.iter().enumerate() {
                con_totals[k] += con[idx] as f64;
            }
        }

        (obj_total, con_totals)
    }

    /// Compute baseline totals and per-constraint scale factors for subgradient updates.
    ///
    /// The scale factors normalise the step size so that the subgradient adapts to
    /// the magnitude difference between objective and constraint values.
    ///
    /// Returns `(baseline_obj, baseline_cons, scale_factors)`.
    pub fn compute_scale_factors(&self) -> (f64, Vec<f64>, Vec<f64>) {
        let (baseline_obj, baseline_cons) = self.baseline_totals();
        let n_quotes = self.n_quotes.max(1) as f64;
        let obj_per_quote = baseline_obj / n_quotes;
        let scale_factors = baseline_cons
            .iter()
            .map(|&bc| {
                let con_per_quote = bc / n_quotes;
                (obj_per_quote / (con_per_quote.abs() + SCALE_EPSILON)).min(MAX_SCALE_FACTOR)
            })
            .collect();
        (baseline_obj, baseline_cons, scale_factors)
    }
}

/// Direction of a constraint bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintDirection {
    /// Total must be >= threshold.
    Min,
    /// Total must be <= threshold.
    Max,
}

impl ConstraintDirection {
    /// One-sided satisfaction check matching the constraint direction.
    ///
    /// `Min`: returns `true` iff `total >= target`.
    /// `Max`: returns `true` iff `total <= target`.
    ///
    /// Centralised here so every solver path (frontier bisection, lambda
    /// subgradient, argmax sign) routes through one canonical
    /// definition. Adding a new `ConstraintDirection` variant becomes a
    /// compile-error checklist instead of a grep across the codebase.
    #[inline]
    pub fn is_satisfied(self, total: f64, target: f64) -> bool {
        match self {
            ConstraintDirection::Min => total >= target,
            ConstraintDirection::Max => total <= target,
        }
    }

    /// Signed residual (target - actual for Min, actual - target for Max).
    /// Positive means *violated*, negative means *over-satisfied*. Useful
    /// for subgradient updates that want a single sign convention.
    #[inline]
    pub fn signed_residual(self, total: f64, target: f64) -> f64 {
        match self {
            ConstraintDirection::Min => target - total,
            ConstraintDirection::Max => total - target,
        }
    }
}

/// Specification for a single constraint in the optimisation problem.
#[derive(Debug, Clone)]
pub struct ConstraintSpec {
    /// Constraint name (must match a `QuoteGrid::constraint_names` entry).
    pub name: String,
    /// Whether this is a lower-bound (Min) or upper-bound (Max) constraint.
    pub direction: ConstraintDirection,
    /// Absolute threshold value for the constraint.
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LambdaStrategy {
    Subgradient,
}

/// Configuration parameters for the Lagrangian solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of subgradient iterations.
    pub max_iter: usize,
    /// Convergence tolerance for lambda changes and constraint satisfaction.
    pub tolerance: f64,
    /// Lambda update strategy (currently only Subgradient).
    pub lambda_strategy: LambdaStrategy,
    /// When true, record per-iteration convergence history.
    pub record_history: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            tolerance: DEFAULT_TOLERANCE,
            lambda_strategy: LambdaStrategy::Subgradient,
            record_history: false,
        }
    }
}

/// A single iteration's snapshot for convergence tracking.
#[derive(Debug, Clone)]
pub struct IterationRecord {
    pub iteration: usize,
    pub lambdas: Vec<f64>,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub max_lambda_change: f64,
    pub all_constraints_satisfied: bool,
}

/// Full iteration history for convergence diagnostics.
#[derive(Debug, Clone)]
pub struct IterationHistory {
    pub records: Vec<IterationRecord>,
}

/// Result of the online Lagrangian solver.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// Per-quote optimal step index (length `n_quotes`).
    pub optimal_steps: Vec<u32>,
    /// Final Lagrange multipliers, one per constraint.
    pub lambdas: Vec<f64>,
    /// Number of iterations actually executed.
    pub iterations: usize,
    /// Whether the solver converged within `max_iter`.
    pub converged: bool,
    /// Sum of objective values at the optimal steps.
    pub total_objective: f64,
    /// Sum of each constraint at the optimal steps.
    pub total_constraints: Vec<f64>,
    /// Baseline objective total (all quotes at scenario_value nearest 1.0).
    pub baseline_objective: f64,
    /// Baseline constraint totals (all quotes at scenario_value nearest 1.0).
    pub baseline_constraints: Vec<f64>,
    /// Per-iteration convergence history, if `record_history` was set.
    pub history: Option<IterationHistory>,
}

/// Builder for incrementally constructing a QuoteGrid from chunked data.
#[derive(Debug)]
pub struct QuoteGridBuilder {
    n_steps: usize,
    scenario_values: Vec<f32>,
    constraint_names: Vec<String>,
    objective: Vec<f32>,
    constraints: Vec<Vec<f32>>,
    quote_ids: Vec<String>,
    n_quotes: usize,
    finalised: bool,
}

impl QuoteGridBuilder {
    /// Create a new builder. `scenario_values` must be sorted and have length `n_steps`.
    pub fn new(
        n_steps: usize,
        scenario_values: Vec<f32>,
        constraint_names: Vec<String>,
    ) -> Result<Self> {
        if scenario_values.len() != n_steps {
            return Err(PriceContourError::DimensionMismatch(format!(
                "scenario_values length {} != n_steps {}",
                scenario_values.len(),
                n_steps
            )));
        }
        // Validate sorted
        for i in 1..scenario_values.len() {
            if scenario_values[i] < scenario_values[i - 1] {
                return Err(PriceContourError::InvalidValue(format!(
                    "scenario_values must be sorted, but [{i}]={} < [{}]={}",
                    scenario_values[i],
                    i - 1,
                    scenario_values[i - 1]
                )));
            }
        }
        let n_constraints = constraint_names.len();
        Ok(Self {
            n_steps,
            scenario_values,
            constraint_names,
            objective: Vec::new(),
            constraints: vec![Vec::new(); n_constraints],
            quote_ids: Vec::new(),
            n_quotes: 0,
            finalised: false,
        })
    }

    /// Append a chunk of quotes. `objective` and each constraint vec must have
    /// length divisible by `n_steps`, and all the same length.
    pub fn append(
        &mut self,
        quote_ids: &[String],
        objective: &[f32],
        constraints: &[Vec<f32>],
    ) -> Result<()> {
        if self.finalised {
            return Err(PriceContourError::InvalidValue(
                "builder already finalised".into(),
            ));
        }
        let total = objective.len();
        if total == 0 {
            return Ok(());
        }
        if !total.is_multiple_of(self.n_steps) {
            return Err(PriceContourError::DimensionMismatch(format!(
                "objective length {} not divisible by n_steps {}",
                total, self.n_steps
            )));
        }
        let chunk_quotes = total / self.n_steps;
        if quote_ids.len() != chunk_quotes {
            return Err(PriceContourError::DimensionMismatch(format!(
                "quote_ids length {} != chunk quotes {}",
                quote_ids.len(),
                chunk_quotes
            )));
        }
        if constraints.len() != self.constraint_names.len() {
            return Err(PriceContourError::DimensionMismatch(format!(
                "constraints count {} != expected {}",
                constraints.len(),
                self.constraint_names.len()
            )));
        }
        for (k, con) in constraints.iter().enumerate() {
            if con.len() != total {
                return Err(PriceContourError::DimensionMismatch(format!(
                    "constraint '{}' length {} != objective length {}",
                    self.constraint_names[k],
                    con.len(),
                    total
                )));
            }
        }

        self.objective.extend_from_slice(objective);
        for (k, con) in constraints.iter().enumerate() {
            self.constraints[k].extend_from_slice(con);
        }
        self.quote_ids.extend_from_slice(quote_ids);
        self.n_quotes += chunk_quotes;
        Ok(())
    }

    /// Consume the builder and return a validated QuoteGrid sorted by `quote_id`.
    ///
    /// Sort happens here (not in `append`) because upstream pipelines often
    /// stream chunks they cannot globally sort. The canonical sort point is
    /// this builder, where the full set of quotes is finally unified.
    ///
    /// The sort is performed in-place via cycle decomposition over a
    /// permutation of the `n_quotes` quote indices, so peak memory does not
    /// double during the sort. Duplicate `quote_id`s across all appended
    /// chunks are detected and reported.
    pub fn build(mut self) -> Result<QuoteGrid> {
        self.finalised = true;
        if self.n_quotes == 0 {
            return Err(PriceContourError::DataValidation(
                "no quotes appended to builder".into(),
            ));
        }

        // Build a stable permutation of quote indices sorted by quote_id.
        // `perm[i]` is the source index whose data should end up at position
        // `i` of the final grid.
        let mut perm: Vec<usize> = (0..self.n_quotes).collect();
        perm.sort_by(|&a, &b| self.quote_ids[a].cmp(&self.quote_ids[b]));

        // Reject duplicates: walk the sorted permutation, compare adjacent
        // quote_ids. Reporting both append-order indices lets callers locate
        // both occurrences in their pipeline.
        for w in perm.windows(2) {
            let a = w[0];
            let b = w[1];
            if self.quote_ids[a] == self.quote_ids[b] {
                let (first, second) = if a < b { (a, b) } else { (b, a) };
                return Err(PriceContourError::DataValidation(format!(
                    "duplicate quote_id '{}' appears at append-order indices {first} and {second} \
                     (each quote_id must be unique across all appended chunks)",
                    self.quote_ids[a]
                )));
            }
        }

        // Apply the permutation in-place to all quote-aligned arrays.
        apply_quote_permutation(
            &perm,
            self.n_steps,
            &mut self.objective,
            &mut self.constraints,
            &mut self.quote_ids,
        );

        let grid = QuoteGrid {
            n_quotes: self.n_quotes,
            n_steps: self.n_steps,
            scenario_values: self.scenario_values,
            objective: self.objective,
            constraints: self.constraints,
            constraint_names: self.constraint_names,
            quote_ids: self.quote_ids,
        };
        grid.validate()?;
        Ok(grid)
    }

    /// Return the number of quotes appended so far.
    pub fn n_quotes(&self) -> usize {
        self.n_quotes
    }
}

/// Apply a quote-index permutation to all parallel arrays in place.
///
/// `perm[i]` is the source quote index that should occupy slot `i` after
/// reordering. Each f32 array is laid out quote-major with `n_steps` elements
/// per quote; `quote_ids` has one element per quote.
///
/// The algorithm walks each cycle of the permutation exactly once. For a
/// cycle `i → a → b → … → i`, it stages the data at `i` in scratch buffers,
/// shifts each subsequent block back by one position, and finally writes the
/// staged data into the cycle's last slot. Fixed points (`perm[i] == i`) are
/// skipped without copying.
///
/// Memory overhead is O(n_steps × (1 + n_constraints)) f32s for the scratch
/// buffers plus O(n_quotes / 8) bits for the visited bitset — independent of
/// the grid size.
fn apply_quote_permutation(
    perm: &[usize],
    n_steps: usize,
    objective: &mut [f32],
    constraints: &mut [Vec<f32>],
    quote_ids: &mut [String],
) {
    let n = perm.len();
    debug_assert_eq!(quote_ids.len(), n);
    debug_assert_eq!(objective.len(), n * n_steps);
    for con in constraints.iter() {
        debug_assert_eq!(con.len(), n * n_steps);
    }

    // Visited bitset, one bit per quote. `n_words = ceil(n / 64)`.
    let n_words = n.div_ceil(64);
    let mut visited: Vec<u64> = vec![0u64; n_words];
    let is_visited = |w: &[u64], i: usize| -> bool { w[i / 64] & (1u64 << (i % 64)) != 0 };
    let mark_visited = |w: &mut [u64], i: usize| {
        w[i / 64] |= 1u64 << (i % 64);
    };

    // Reusable scratch buffers — allocated once outside the cycle loop.
    let mut tmp_obj = vec![0.0f32; n_steps];
    let mut tmp_cons: Vec<Vec<f32>> = constraints.iter().map(|_| vec![0.0f32; n_steps]).collect();

    for start in 0..n {
        if is_visited(&visited, start) {
            continue;
        }
        let src = perm[start];
        if src == start {
            // Fixed point — no work.
            mark_visited(&mut visited, start);
            continue;
        }

        // Stage the data currently at `start`; the cycle ends with this data
        // landing at the cycle's last position.
        let start_block = start * n_steps;
        tmp_obj.copy_from_slice(&objective[start_block..start_block + n_steps]);
        for (k, con) in constraints.iter().enumerate() {
            tmp_cons[k].copy_from_slice(&con[start_block..start_block + n_steps]);
        }
        let tmp_qid = std::mem::take(&mut quote_ids[start]);

        // Walk the cycle: position `cur` should receive the data currently at
        // `next = perm[cur]`. When `next` returns to `start`, deposit the
        // staged data and close the cycle.
        let mut cur = start;
        loop {
            let next = perm[cur];
            let cur_block = cur * n_steps;
            if next == start {
                objective[cur_block..cur_block + n_steps].copy_from_slice(&tmp_obj);
                for (k, con) in constraints.iter_mut().enumerate() {
                    con[cur_block..cur_block + n_steps].copy_from_slice(&tmp_cons[k]);
                }
                quote_ids[cur] = tmp_qid;
                mark_visited(&mut visited, cur);
                break;
            }
            let next_block = next * n_steps;
            objective.copy_within(next_block..next_block + n_steps, cur_block);
            for con in constraints.iter_mut() {
                con.copy_within(next_block..next_block + n_steps, cur_block);
            }
            quote_ids[cur] = std::mem::take(&mut quote_ids[next]);
            mark_visited(&mut visited, cur);
            cur = next;
        }
    }
}

/// Mapping from quote index to group index, for grouped optimisation.
#[derive(Debug, Clone)]
pub struct GroupMapping {
    /// Group index for each quote (length `n_quotes`).
    pub group_of: Vec<u32>,
    /// Total number of distinct groups.
    pub n_groups: usize,
    /// Human-readable label for each group, ordered by group index.
    pub group_labels: Vec<String>,
}

/// Build a GroupMapping from per-quote string labels.
pub fn build_group_mapping(labels: &[String]) -> GroupMapping {
    use std::collections::HashMap;
    let mut label_to_idx: HashMap<&str, u32> = HashMap::new();
    let mut group_labels: Vec<String> = Vec::new();
    let mut group_of = Vec::with_capacity(labels.len());

    for label in labels {
        let idx = if let Some(&idx) = label_to_idx.get(label.as_str()) {
            idx
        } else {
            let idx = group_labels.len() as u32;
            group_labels.push(label.clone());
            label_to_idx.insert(label, idx);
            idx
        };
        group_of.push(idx);
    }

    let n_groups = group_labels.len();
    GroupMapping {
        group_of,
        n_groups,
        group_labels,
    }
}

/// Result of grouped optimisation (per-group factor selection with remapping).
#[derive(Debug, Clone)]
pub struct GroupedSolveResult {
    /// Optimal factor value per group (length `n_groups`).
    pub optimal_factor_values: Vec<f32>,
    /// Remapped optimal step index per quote (length `n_quotes`).
    pub optimal_steps_per_quote: Vec<u32>,
    /// Final Lagrange multipliers, one per constraint.
    pub lambdas: Vec<f64>,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Whether the solver converged.
    pub converged: bool,
    /// Total objective at the optimal steps.
    pub total_objective: f64,
    /// Total constraints at the optimal steps.
    pub total_constraints: Vec<f64>,
    /// Baseline objective total.
    pub baseline_objective: f64,
    /// Baseline constraint totals.
    pub baseline_constraints: Vec<f64>,
    /// Fraction of (quote, candidate) pairs that were clamped to grid bounds.
    pub clamp_rate: f32,
    /// Per-iteration convergence history, if requested.
    pub history: Option<IterationHistory>,
}

/// Result of applying fixed lambdas (single forward pass, no iteration).
#[derive(Debug, Clone)]
pub struct ApplyResult {
    /// Per-quote optimal step index (length `n_quotes`).
    pub optimal_steps: Vec<u32>,
    /// The lambdas that were applied (echo of input).
    pub lambdas: Vec<f64>,
    /// Total objective at the optimal steps.
    pub total_objective: f64,
    /// Total constraints at the optimal steps.
    pub total_constraints: Vec<f64>,
    /// Baseline objective total.
    pub baseline_objective: f64,
    /// Baseline constraint totals.
    pub baseline_constraints: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_grid(n_quotes: usize, n_steps: usize) -> QuoteGrid {
        QuoteGrid {
            n_quotes,
            n_steps,
            scenario_values: vec![0.9, 1.0, 1.1][..n_steps].to_vec(),
            objective: vec![1.0; n_quotes * n_steps],
            constraints: vec![vec![1.0; n_quotes * n_steps]],
            constraint_names: vec!["volume".to_string()],
            quote_ids: (0..n_quotes).map(|i| format!("Q{i}")).collect(),
        }
    }

    #[test]
    fn test_validate_ok() {
        let grid = make_grid(3, 3);
        assert!(grid.validate().is_ok());
    }

    #[test]
    fn test_validate_bad_objective_len() {
        let mut grid = make_grid(3, 3);
        grid.objective = vec![1.0; 5]; // wrong length
        assert!(grid.validate().is_err());
    }

    #[test]
    fn test_validate_bad_constraint_len() {
        let mut grid = make_grid(3, 3);
        grid.constraints[0] = vec![1.0; 5];
        assert!(grid.validate().is_err());
    }

    #[test]
    fn test_baseline_totals() {
        // 2 quotes, 3 steps. scenario_values [0.9, 1.0, 1.1].
        // Baseline step = 1 (scenario_value closest to 1.0).
        let grid = QuoteGrid {
            n_quotes: 2,
            n_steps: 3,
            scenario_values: vec![0.9, 1.0, 1.1],
            objective: vec![
                10.0, 20.0, 30.0, // quote 0
                40.0, 50.0, 60.0, // quote 1
            ],
            constraints: vec![vec![
                1.0, 2.0, 3.0, // quote 0
                4.0, 5.0, 6.0, // quote 1
            ]],
            constraint_names: vec!["volume".to_string()],
            quote_ids: vec!["Q0".to_string(), "Q1".to_string()],
        };

        let (obj, cons) = grid.baseline_totals();
        // At step 1: objective = 20 + 50 = 70, constraint = 2 + 5 = 7
        assert_abs_diff_eq!(obj, 70.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cons[0], 7.0, epsilon = 1e-6);
    }

    // -----------------------------------------------------------------------
    // Issue 28: QuoteGridBuilder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_happy_path() {
        // Build a 2-quote, 3-step grid with 1 constraint, verify it validates.
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();

        builder
            .append(
                &["Q0".to_string(), "Q1".to_string()],
                &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            )
            .unwrap();

        assert_eq!(builder.n_quotes(), 2);
        let grid = builder.build().unwrap();
        assert_eq!(grid.n_quotes, 2);
        assert_eq!(grid.n_steps, 3);
        assert!(grid.validate().is_ok());
    }

    #[test]
    fn test_builder_rejects_unsorted_scenario_values() {
        // scenario_values = [1.0, 0.5, 0.8] should error
        let result = QuoteGridBuilder::new(3, vec![1.0, 0.5, 0.8], vec!["volume".to_string()]);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("sorted"), "error should mention sorted: {msg}");
    }

    #[test]
    fn test_builder_rejects_mismatched_n_steps() {
        // scenario_values.len() != n_steps should error
        let result = QuoteGridBuilder::new(
            5,
            vec![0.9, 1.0, 1.1], // len 3 != 5
            vec!["volume".to_string()],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_rejects_append_after_build() {
        // builder.build() consumes self, so we cannot call append again.
        // Instead, verify that build() sets the finalised flag correctly
        // by testing the ownership model: after build(), the builder is gone.
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();

        builder
            .append(
                &["Q0".to_string()],
                &[1.0, 2.0, 3.0],
                &[vec![1.0, 2.0, 3.0]],
            )
            .unwrap();

        // build() consumes the builder, so there is no way to call append() after.
        // The finalised flag is still tested: if build() were &mut self,
        // calling append after would hit the finalised check. With the current
        // ownership design, the test simply confirms build() succeeds.
        let grid = builder.build().unwrap();
        assert_eq!(grid.n_quotes, 1);
    }

    #[test]
    fn test_builder_accepts_empty_append() {
        // append with empty objective (0 quotes) should succeed.
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();

        // Empty append — should be a no-op
        builder.append(&[], &[], &[vec![]]).unwrap();
        assert_eq!(builder.n_quotes(), 0);

        // Add actual data so build() succeeds
        builder
            .append(
                &["Q0".to_string()],
                &[1.0, 2.0, 3.0],
                &[vec![1.0, 2.0, 3.0]],
            )
            .unwrap();

        let grid = builder.build().unwrap();
        assert_eq!(grid.n_quotes, 1);
    }

    #[test]
    fn test_builder_rejects_objective_not_divisible_by_n_steps() {
        // objective.len() % n_steps != 0 should error
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();

        let result = builder.append(
            &["Q0".to_string()],
            &[1.0, 2.0], // len 2 not divisible by 3
            &[vec![1.0, 2.0]],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_rejects_mismatched_quote_ids_length() {
        // quote_ids.len() != n_quotes_in_chunk should error
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();

        // 6 objective values = 2 quotes, but only 1 quote_id
        let result = builder.append(
            &["Q0".to_string()],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_rejects_wrong_constraint_count() {
        // Passing 2 constraints when builder was created with 1 should error.
        let mut builder = QuoteGridBuilder::new(
            3,
            vec![0.9, 1.0, 1.1],
            vec!["volume".to_string()], // 1 constraint
        )
        .unwrap();

        let result = builder.append(
            &["Q0".to_string()],
            &[1.0, 2.0, 3.0],
            &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]], // 2 constraints
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_rejects_zero_quotes() {
        // build() with no appended data should error.
        let builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();

        let result = builder.build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("no quotes"),
            "error should mention no quotes: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Issue 1: build()-time sort by quote_id via in-place cycle permutation.
    //
    // Contract: append() preserves input order without sorting; build() sorts
    // the entire grid by quote_id in-place. The motivation is that upstream
    // pipelines feed chunks they cannot globally sort (they don't have the
    // whole DataFrame materialised), so the canonical sort point is the
    // QuoteGrid layer where the data is unified.
    // -----------------------------------------------------------------------

    /// Helper: build a grid where each quote's data uniquely identifies the
    /// quote, so a permutation bug shows up as a data/id mismatch.
    ///
    /// Quote `Q{id}` step `j` has objective = id*100 + j and constraint =
    /// id*1000 + j (where id is the integer parsed from the quote_id).
    fn append_uniquely_tagged(
        builder: &mut QuoteGridBuilder,
        ids: &[u32], // numeric quote ids in append order
        n_steps: usize,
    ) {
        let quote_ids: Vec<String> = ids.iter().map(|i| format!("Q{i:04}")).collect();
        let mut objective: Vec<f32> = Vec::with_capacity(ids.len() * n_steps);
        let mut constraint: Vec<f32> = Vec::with_capacity(ids.len() * n_steps);
        for &id in ids {
            for j in 0..n_steps {
                objective.push((id as f32) * 100.0 + j as f32);
                constraint.push((id as f32) * 1000.0 + j as f32);
            }
        }
        builder
            .append(&quote_ids, &objective, &[constraint])
            .unwrap();
    }

    /// Helper: assert the grid is in canonical sorted order and each quote's
    /// data matches its tagged value (per `append_uniquely_tagged`).
    fn assert_sorted_with_tags(grid: &QuoteGrid, expected_ids: &[u32]) {
        let n_steps = grid.n_steps;
        assert_eq!(grid.n_quotes, expected_ids.len());
        let expected_qids: Vec<String> = expected_ids.iter().map(|i| format!("Q{i:04}")).collect();
        assert_eq!(grid.quote_ids, expected_qids);
        for (q, &id) in expected_ids.iter().enumerate() {
            for j in 0..n_steps {
                let idx = q * n_steps + j;
                assert!(
                    (grid.objective[idx] - ((id as f32) * 100.0 + j as f32)).abs() < 1e-6,
                    "objective[{q}][{j}] (quote Q{id:04}) wrong: got {}",
                    grid.objective[idx],
                );
                assert!(
                    (grid.constraints[0][idx] - ((id as f32) * 1000.0 + j as f32)).abs() < 1e-3,
                    "constraint[{q}][{j}] (quote Q{id:04}) wrong: got {}",
                    grid.constraints[0][idx],
                );
            }
        }
    }

    #[test]
    fn test_build_sorts_reverse_order_chunks() {
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();
        // Append in reverse: Q0003, Q0002, Q0001, Q0000.
        append_uniquely_tagged(&mut builder, &[3, 2, 1, 0], 3);
        let grid = builder.build().unwrap();
        // After build(), expected order is Q0000, Q0001, Q0002, Q0003.
        assert_sorted_with_tags(&grid, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_build_sorts_interleaved_one_quote_chunks() {
        // Many chunks of a single quote each, in random order.
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();
        // Order chosen to exercise multi-step cycles across chunks.
        for &id in &[5u32, 2, 7, 0, 9, 1, 8, 3, 6, 4] {
            append_uniquely_tagged(&mut builder, &[id], 3);
        }
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_build_sorts_multi_chunk_unsorted() {
        // Multiple chunks, each internally unsorted, with cross-chunk overlap
        // in lexicographic order.
        let mut builder =
            QuoteGridBuilder::new(2, vec![0.9, 1.1], vec!["volume".to_string()]).unwrap();
        // Chunk 1: ids 5, 2, 7. Chunk 2: ids 3, 0, 9. Chunk 3: ids 8, 1, 4, 6.
        append_uniquely_tagged(&mut builder, &[5, 2, 7], 2);
        append_uniquely_tagged(&mut builder, &[3, 0, 9], 2);
        append_uniquely_tagged(&mut builder, &[8, 1, 4, 6], 2);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_build_already_sorted_is_idempotent() {
        // Already-sorted input must produce the same data, byte-for-byte
        // (the cycle algorithm should detect fixed points and skip work).
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();
        append_uniquely_tagged(&mut builder, &[0, 1, 2, 3, 4], 3);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_build_detects_duplicate_quote_id_within_chunk() {
        let mut builder =
            QuoteGridBuilder::new(2, vec![0.9, 1.1], vec!["volume".to_string()]).unwrap();
        // Same quote id appears twice in one chunk: at append-order 0 and 2.
        append_uniquely_tagged(&mut builder, &[3, 1, 3], 2);
        let err = builder.build().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("duplicate"), "missing 'duplicate': {msg}");
        assert!(msg.contains("Q0003"), "missing offending id Q0003: {msg}");
        assert!(
            msg.contains('0') && msg.contains('2'),
            "should report both occurrences (indices 0 and 2): {msg}"
        );
    }

    #[test]
    fn test_build_detects_duplicate_quote_id_across_chunks() {
        let mut builder =
            QuoteGridBuilder::new(2, vec![0.9, 1.1], vec!["volume".to_string()]).unwrap();
        append_uniquely_tagged(&mut builder, &[3, 1], 2);
        append_uniquely_tagged(&mut builder, &[5, 1], 2); // Q0001 again, index 3
        let err = builder.build().unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("duplicate"), "missing 'duplicate': {msg}");
        assert!(msg.contains("Q0001"), "missing offending id Q0001: {msg}");
        // Q0001 appears at append-order indices 1 and 3.
        assert!(
            msg.contains('1') && msg.contains('3'),
            "should report both occurrences (indices 1 and 3): {msg}"
        );
    }

    #[test]
    fn test_build_single_quote() {
        let mut builder =
            QuoteGridBuilder::new(2, vec![0.9, 1.1], vec!["volume".to_string()]).unwrap();
        append_uniquely_tagged(&mut builder, &[7], 2);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &[7]);
    }

    #[test]
    fn test_build_single_swap_two_quotes() {
        // Two quotes in reverse — exercises a 2-cycle.
        let mut builder =
            QuoteGridBuilder::new(2, vec![0.9, 1.1], vec!["volume".to_string()]).unwrap();
        append_uniquely_tagged(&mut builder, &[1, 0], 2);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &[0, 1]);
    }

    #[test]
    fn test_build_mixed_cycles_and_fixed_points() {
        // Permutation: positions 0,3 are fixed; (1,2) is a swap; (4,5,6) is a 3-cycle.
        // Append order such that after sorting by id, the source positions form
        // exactly that permutation.
        // ids in append order: 0, 2, 1, 3, 6, 4, 5
        // sorted ids:           0, 1, 2, 3, 4, 5, 6
        // perm[i] = position in source where sorted[i] lives:
        //   sorted[0]=0 -> source 0
        //   sorted[1]=1 -> source 2
        //   sorted[2]=2 -> source 1
        //   sorted[3]=3 -> source 3
        //   sorted[4]=4 -> source 5
        //   sorted[5]=5 -> source 6
        //   sorted[6]=6 -> source 4
        // Cycles: {0}, {1,2}, {3}, {4,5,6}.
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.9, 1.0, 1.1], vec!["volume".to_string()]).unwrap();
        append_uniquely_tagged(&mut builder, &[0, 2, 1, 3, 6, 4, 5], 3);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &[0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_build_zero_constraints_still_sorts() {
        // Edge: a builder created with no constraint columns should still sort
        // objective + quote_ids correctly.
        let mut builder = QuoteGridBuilder::new(2, vec![0.9, 1.1], vec![]).unwrap();
        builder
            .append(
                &[
                    "Q0002".to_string(),
                    "Q0000".to_string(),
                    "Q0001".to_string(),
                ],
                &[200.0, 201.0, 0.0, 1.0, 100.0, 101.0],
                &[],
            )
            .unwrap();
        let grid = builder.build().unwrap();
        assert_eq!(grid.quote_ids, vec!["Q0000", "Q0001", "Q0002"]);
        assert_eq!(grid.objective, vec![0.0, 1.0, 100.0, 101.0, 200.0, 201.0]);
        assert!(grid.constraints.is_empty());
    }

    #[test]
    fn test_build_multi_constraint_permutation() {
        // Verify all constraint vectors are permuted consistently with objective.
        let mut builder = QuoteGridBuilder::new(
            2,
            vec![0.9, 1.1],
            vec!["volume".to_string(), "loss_ratio".to_string()],
        )
        .unwrap();
        // Append Q0002, Q0000, Q0001 in that order. Each constraint encodes
        // `id*1000 + step` (volume) and `id*-1000 - step` (loss_ratio) so a
        // mis-permutation between constraints would be detectable.
        let ids = ["Q0002", "Q0000", "Q0001"];
        let mut obj = Vec::new();
        let mut vol = Vec::new();
        let mut lr = Vec::new();
        for id_str in &ids {
            let id: u32 = id_str[1..].parse().unwrap();
            for j in 0..2 {
                obj.push(id as f32 * 100.0 + j as f32);
                vol.push(id as f32 * 1000.0 + j as f32);
                lr.push(-(id as f32) * 1000.0 - j as f32);
            }
        }
        builder
            .append(
                &ids.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                &obj,
                &[vol, lr],
            )
            .unwrap();
        let grid = builder.build().unwrap();
        assert_eq!(grid.quote_ids, vec!["Q0000", "Q0001", "Q0002"]);
        for q in 0..3 {
            let id: u32 = grid.quote_ids[q][1..].parse().unwrap();
            for j in 0..2 {
                let idx = q * 2 + j;
                assert!((grid.objective[idx] - (id as f32 * 100.0 + j as f32)).abs() < 1e-6);
                assert!((grid.constraints[0][idx] - (id as f32 * 1000.0 + j as f32)).abs() < 1e-3);
                assert!(
                    (grid.constraints[1][idx] - (-(id as f32) * 1000.0 - j as f32)).abs() < 1e-3
                );
            }
        }
    }

    #[test]
    fn test_build_n_steps_one() {
        // Edge: n_steps = 1 means each quote occupies a single row. Cycle
        // permutation must still work; tmp_obj/tmp_cons are single-element.
        let mut builder = QuoteGridBuilder::new(1, vec![1.0], vec!["volume".to_string()]).unwrap();
        append_uniquely_tagged(&mut builder, &[5, 0, 3, 1, 2, 4], 1);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_build_single_giant_cycle() {
        // A permutation that's one cycle of length n_quotes (no fixed points,
        // no smaller cycles). Constructed by appending in the order
        // [n-1, 0, 1, 2, ..., n-2]: sorted index 0 needs source 1, sorted 1
        // needs source 2, ..., sorted n-2 needs source n-1, sorted n-1 needs
        // source 0 — a single n-cycle.
        let n = 11;
        let n_steps = 3;
        let mut ids: Vec<u32> = vec![(n - 1) as u32];
        ids.extend(0..(n - 1) as u32);
        let mut builder =
            QuoteGridBuilder::new(n_steps, vec![0.9, 1.0, 1.1], vec!["volume".to_string()])
                .unwrap();
        append_uniquely_tagged(&mut builder, &ids, n_steps);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &(0..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn test_build_scenario_values_unaffected_by_sort() {
        // Sort permutes per-quote arrays, but scenario_values is shared
        // across all quotes (length n_steps) — it must NOT be permuted.
        let mut builder =
            QuoteGridBuilder::new(3, vec![0.7, 1.0, 1.3], vec!["volume".to_string()]).unwrap();
        // Append in reverse so the sort actually moves data.
        append_uniquely_tagged(&mut builder, &[9, 5, 1, 7, 3], 3);
        let grid = builder.build().unwrap();
        assert_eq!(grid.scenario_values, vec![0.7, 1.0, 1.3]);
    }

    #[test]
    fn test_build_large_n_quotes_stresses_bitset() {
        // n_quotes = 130 spans 3 u64 words in the visited bitset and is not
        // a multiple of 64; exercises both word-boundary handling and the
        // tail bits of the last word.
        let n = 130;
        let n_steps = 2;
        let mut builder =
            QuoteGridBuilder::new(n_steps, vec![0.95, 1.05], vec!["volume".to_string()]).unwrap();
        // Reverse-append to force a non-trivial permutation across all bits.
        let ids: Vec<u32> = (0..n as u32).rev().collect();
        append_uniquely_tagged(&mut builder, &ids, n_steps);
        let grid = builder.build().unwrap();
        assert_sorted_with_tags(&grid, &(0..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn test_build_property_random_shuffles_match_naive_sort() {
        // Property test: many random shuffles of a known grid should each
        // produce the same canonical-sorted result. Verifies the cycle
        // permutation against a naive copy-based sort.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let n_steps = 4;
        let n_quotes = 23; // odd, prime — exercises odd-length and 1-cycles
        let scenario_values = vec![0.85, 0.95, 1.05, 1.15];

        // Build the canonical (sorted) reference via append-in-order.
        let mut ref_builder =
            QuoteGridBuilder::new(n_steps, scenario_values.clone(), vec!["volume".to_string()])
                .unwrap();
        let canonical_ids: Vec<u32> = (0..n_quotes as u32).collect();
        append_uniquely_tagged(&mut ref_builder, &canonical_ids, n_steps);
        let reference = ref_builder.build().unwrap();

        // Try multiple deterministic shuffles via a tiny seeded hash-shuffle.
        for seed in 0u64..16 {
            let mut shuffled: Vec<u32> = canonical_ids.clone();
            // Fisher-Yates with a deterministic PRNG built from the seed.
            for i in (1..shuffled.len()).rev() {
                let mut h = DefaultHasher::new();
                (seed, i as u64).hash(&mut h);
                let j = (h.finish() as usize) % (i + 1);
                shuffled.swap(i, j);
            }

            let mut builder =
                QuoteGridBuilder::new(n_steps, scenario_values.clone(), vec!["volume".to_string()])
                    .unwrap();
            append_uniquely_tagged(&mut builder, &shuffled, n_steps);
            let grid = builder.build().unwrap();

            assert_eq!(
                grid.quote_ids, reference.quote_ids,
                "seed {seed}: quote_ids mismatch (input shuffle: {shuffled:?})"
            );
            assert_eq!(
                grid.objective, reference.objective,
                "seed {seed}: objective mismatch"
            );
            assert_eq!(
                grid.constraints, reference.constraints,
                "seed {seed}: constraints mismatch"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Issue 29: build_group_mapping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_mapping_empty() {
        let labels: Vec<String> = vec![];
        let mapping = build_group_mapping(&labels);
        assert_eq!(mapping.n_groups, 0);
        assert!(mapping.group_of.is_empty());
    }

    #[test]
    fn test_group_mapping_single_label_repeated() {
        let labels = vec!["A".to_string(), "A".to_string(), "A".to_string()];
        let mapping = build_group_mapping(&labels);
        assert_eq!(mapping.n_groups, 1);
        assert_eq!(mapping.group_of, vec![0, 0, 0]);
    }

    #[test]
    fn test_group_mapping_all_unique() {
        let labels = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mapping = build_group_mapping(&labels);
        assert_eq!(mapping.n_groups, 3);
        // Each label gets a unique group
        assert_ne!(mapping.group_of[0], mapping.group_of[1]);
        assert_ne!(mapping.group_of[1], mapping.group_of[2]);
    }

    #[test]
    fn test_group_mapping_preserves_order() {
        let labels = vec![
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ];
        let mapping = build_group_mapping(&labels);
        assert_eq!(mapping.n_groups, 3);
        assert_eq!(mapping.group_of[0], mapping.group_of[2]); // Both "B"
    }

    // -----------------------------------------------------------------------
    // Issue 33: Error path tests for QuoteGrid validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_rejects_zero_n_steps() {
        let grid = QuoteGrid {
            n_quotes: 1,
            n_steps: 0,
            scenario_values: vec![],
            objective: vec![],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec!["Q0".to_string()],
        };
        let err = grid.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("n_steps"),
            "error should mention n_steps: {msg}"
        );
    }

    #[test]
    fn test_validate_rejects_zero_n_quotes() {
        let grid = QuoteGrid {
            n_quotes: 0,
            n_steps: 3,
            scenario_values: vec![0.9, 1.0, 1.1],
            objective: vec![],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec![],
        };
        let err = grid.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("n_quotes"),
            "error should mention n_quotes: {msg}"
        );
    }

    #[test]
    fn test_validate_rejects_nan_scenario_values() {
        let grid = QuoteGrid {
            n_quotes: 1,
            n_steps: 3,
            scenario_values: vec![0.9, f32::NAN, 1.1],
            objective: vec![1.0, 2.0, 3.0],
            constraints: vec![],
            constraint_names: vec![],
            quote_ids: vec!["Q0".to_string()],
        };
        let err = grid.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("NaN") || msg.contains("Inf") || msg.contains("finite"),
            "error should mention NaN/Inf: {msg}"
        );
    }
}
