use thiserror::Error;

#[derive(Error, Debug)]
pub enum PriceContourError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Data validation: {0}")]
    DataValidation(String),
}

pub type Result<T> = std::result::Result<T, PriceContourError>;
