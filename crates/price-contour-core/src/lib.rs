pub mod constants;
pub mod data;
pub mod error;
pub mod factor_context;
pub mod frontier;
pub mod solver;

pub use data::{
    build_group_mapping, fingerprint_quote_ids, ApplyResult, ConstraintDirection, ConstraintSpec,
    GroupMapping, GroupedSolveResult, IterationHistory, IterationRecord, LambdaStrategy, QuoteGrid,
    QuoteGridBuilder, SolveResult, SolverConfig,
};
pub use factor_context::{FactorContextBuilder, FactorContextsBuilt};
pub use error::{PriceContourError, Result};
pub use frontier::{
    sweep_frontier, FrontierConfig, FrontierPoint, FrontierResult, NonConvergenceReason,
    ScenarioValueStats, SolverPath,
};
pub use solver::{
    apply_lambdas, compute_lambda_signs_f32, lagrangian_argmax_pass, solve_grouped, solve_online,
};
