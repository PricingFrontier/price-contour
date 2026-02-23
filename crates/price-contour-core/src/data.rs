use crate::constants::*;
use crate::error::{PriceContourError, Result};

/// Contiguous memory layout for solver operations.
/// All arrays are length N*M, laid out quote-major:
/// [q0_s0, q0_s1, ..., q0_sM-1, q1_s0, q1_s1, ..., q1_sM-1, ...]
pub struct QuoteGrid {
    pub n_quotes: usize,
    pub n_steps: usize,
    pub multipliers: Vec<f32>,
    pub objective: Vec<f32>,
    pub constraints: Vec<Vec<f32>>,
    pub constraint_names: Vec<String>,
    pub quote_ids: Vec<String>,
}

impl QuoteGrid {
    /// Validate dimensions of all arrays.
    pub fn validate(&self) -> Result<()> {
        let expected_len = self.n_quotes * self.n_steps;

        if self.multipliers.len() != self.n_steps {
            return Err(PriceContourError::DimensionMismatch(format!(
                "multipliers length {} != n_steps {}",
                self.multipliers.len(),
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

    /// Find the step index closest to multiplier=1.0 and compute baseline totals.
    /// Returns (baseline_objective_total, baseline_constraint_totals) as f64.
    pub fn baseline_totals(&self) -> (f64, Vec<f64>) {
        let baseline_step = self
            .multipliers
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                ((**a) - 1.0f32)
                    .abs()
                    .partial_cmp(&((**b) - 1.0f32).abs())
                    .unwrap()
            })
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstraintDirection {
    Min,
    Max,
}

#[derive(Debug, Clone)]
pub struct ConstraintSpec {
    pub name: String,
    pub direction: ConstraintDirection,
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LambdaStrategy {
    Subgradient,
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub max_iter: usize,
    pub chunk_size: usize,
    pub tolerance: f64,
    pub lambda_strategy: LambdaStrategy,
    pub record_history: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iter: DEFAULT_MAX_ITER,
            chunk_size: DEFAULT_CHUNK_SIZE,
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

pub struct SolveResult {
    pub optimal_steps: Vec<u32>,
    pub lambdas: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub baseline_objective: f64,
    pub baseline_constraints: Vec<f64>,
    pub history: Option<IterationHistory>,
}

/// Result of applying fixed lambdas (single forward pass, no iteration).
pub struct ApplyResult {
    pub optimal_steps: Vec<u32>,
    pub lambdas: Vec<f64>,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub baseline_objective: f64,
    pub baseline_constraints: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(n_quotes: usize, n_steps: usize) -> QuoteGrid {
        QuoteGrid {
            n_quotes,
            n_steps,
            multipliers: vec![0.9, 1.0, 1.1][..n_steps].to_vec(),
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
        // 2 quotes, 3 steps. Multipliers [0.9, 1.0, 1.1].
        // Baseline step = 1 (multiplier closest to 1.0).
        let grid = QuoteGrid {
            n_quotes: 2,
            n_steps: 3,
            multipliers: vec![0.9, 1.0, 1.1],
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
        assert!((obj - 70.0).abs() < 1e-6);
        assert!((cons[0] - 7.0).abs() < 1e-6);
    }
}
