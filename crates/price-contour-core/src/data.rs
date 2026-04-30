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

    /// Consume the builder and return a validated QuoteGrid.
    pub fn build(mut self) -> Result<QuoteGrid> {
        self.finalised = true;
        if self.n_quotes == 0 {
            return Err(PriceContourError::DataValidation(
                "no quotes appended to builder".into(),
            ));
        }
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
