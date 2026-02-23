use thiserror::Error;

#[derive(Error, Debug)]
pub enum PriceContourError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Missing column: {0}")]
    MissingColumn(String),

    #[error("Convergence failure after {iterations} iterations: {message}")]
    ConvergenceFailure { iterations: usize, message: String },

    #[error("Data validation: {0}")]
    DataValidation(String),
}

pub type Result<T> = std::result::Result<T, PriceContourError>;
