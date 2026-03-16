mod apply_py;
mod builder_py;
mod frontier_py;
mod grid_py;
mod grouped_py;
mod parquet_grid_py;
mod solver_py;

use pyo3::prelude::*;

#[pymodule]
fn _price_contour(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<solver_py::PySolveResult>()?;
    m.add_function(wrap_pyfunction!(solver_py::solve_online_py, m)?)?;
    m.add_function(wrap_pyfunction!(solver_py::solve_from_grid_py, m)?)?;
    m.add_class::<apply_py::PyApplyResult>()?;
    m.add_function(wrap_pyfunction!(apply_py::apply_lambdas_py, m)?)?;
    m.add_function(wrap_pyfunction!(apply_py::apply_from_grid_py, m)?)?;
    m.add_class::<builder_py::PyQuoteGridBuilder>()?;
    m.add_class::<grid_py::PyQuoteGrid>()?;
    m.add_class::<grouped_py::PyGroupedSolveResult>()?;
    m.add_function(wrap_pyfunction!(grouped_py::solve_grouped_py, m)?)?;
    m.add_class::<frontier_py::PyFrontierResult>()?;
    m.add_function(wrap_pyfunction!(frontier_py::sweep_frontier_py, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_grid_py::build_grid_from_parquet_py, m)?)?;
    Ok(())
}
