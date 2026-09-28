use thiserror::Error;

#[derive(Error, Debug)]
pub enum PriceContourError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Data validation: {0}")]
    DataValidation(String),

    /// The caller's [`crate::CancelFlag`] was set while the call ran.
    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, PriceContourError>;
