//! Test hook for the frame boundary between Python and the native module
//! (DESIGN_DECISIONS §15.1).

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

/// Test hook: each column of `df` as the native module receives it, as
/// `(name, dtype, n_chunks)`. String, Categorical and Enum are named
/// exactly; other dtypes use Polars' display name. Lets the tests pin what
/// the bridge delivers (an Enum stays an Enum; chunks are not merged).
/// Not part of the API.
#[pyfunction]
pub fn _received_columns(df: PyDataFrame) -> Vec<(String, String, usize)> {
    df.0.columns()
        .iter()
        .map(|c| {
            let dtype = match c.dtype() {
                DataType::String => "String".to_string(),
                DataType::Categorical(..) => "Categorical".to_string(),
                DataType::Enum(..) => "Enum".to_string(),
                other => other.to_string(),
            };
            (c.name().to_string(), dtype, c.n_chunks())
        })
        .collect()
}
