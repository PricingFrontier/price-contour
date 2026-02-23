mod apply_py;
mod solver_py;

use pyo3::prelude::*;

#[pymodule]
fn _price_contour(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<solver_py::PySolveResult>()?;
    m.add_function(wrap_pyfunction!(solver_py::solve_online_py, m)?)?;
    m.add_class::<apply_py::PyApplyResult>()?;
    m.add_function(wrap_pyfunction!(apply_py::apply_lambdas_py, m)?)?;
    Ok(())
}
