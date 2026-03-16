pub mod constants;
pub mod data;
pub mod error;
pub mod frontier;
pub mod solver;

pub use data::{
    build_group_mapping, ApplyResult, ConstraintDirection, ConstraintSpec, GroupMapping,
    GroupedSolveResult, IterationHistory, IterationRecord, LambdaStrategy, QuoteGrid,
    QuoteGridBuilder, SolveResult, SolverConfig,
};
pub use error::{PriceContourError, Result};
pub use frontier::{
    sweep_frontier, FrontierConfig, FrontierPoint, FrontierResult, ScenarioValueStats,
};
pub use solver::{
    apply_lambdas, compute_lambda_signs_f32, lagrangian_argmax_pass, solve_grouped, solve_online,
};
