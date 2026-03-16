mod apply;
mod argmax;
pub mod convergence;
mod grouped;
mod lambda;
mod online;

pub use apply::apply_lambdas;
pub use argmax::{compute_lambda_signs_f32, lagrangian_argmax_pass};
pub use convergence::{all_constraints_satisfied, select_final_lambdas};
pub use grouped::solve_grouped;
pub use lambda::update_lambdas_subgradient;
pub use online::solve_online;
pub(crate) use online::solve_online_with_precomputed;
