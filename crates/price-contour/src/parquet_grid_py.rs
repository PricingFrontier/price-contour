use std::fs::File;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grid_py::PyQuoteGrid;
use crate::solver_py::ingest_dataframe;

/// Build a QuoteGrid directly from a Parquet file on disk.
///
/// Reads the file in Rust (no Python DataFrame intermediate), sorts by
/// (quote_id, scenario_index), and constructs the grid.  The caller
/// (typically haute) sinks its lazy plan to parquet first, then passes
/// the path here so the entire pipeline avoids materialising a large
/// DataFrame in Python.
#[pyfunction]
#[pyo3(signature = (
    path,
    constraint_columns,
    *,
    quote_id = "quote_id",
    scenario_index = "scenario_index",
    scenario_value_col = "scenario_value",
    objective = "expected_income",
))]
pub fn build_grid_from_parquet_py(
    path: &str,
    constraint_columns: Vec<String>,
    quote_id: &str,
    scenario_index: &str,
    scenario_value_col: &str,
    objective: &str,
) -> PyResult<PyQuoteGrid> {
    let file = File::open(path)
        .map_err(|e| PyValueError::new_err(format!("Failed to open parquet file: {e}")))?;

    let df = ParquetReader::new(file)
        .finish()
        .map_err(|e| PyValueError::new_err(format!("Failed to read parquet: {e}")))?;

    let grid = ingest_dataframe(
        &df,
        quote_id,
        scenario_index,
        scenario_value_col,
        objective,
        &constraint_columns,
    )?;

    Ok(PyQuoteGrid::new(grid))
}
