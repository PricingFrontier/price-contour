use pyo3::prelude::*;

use price_contour_core::QuoteGrid;

/// Python-visible opaque handle to a QuoteGrid.
#[pyclass(name = "QuoteGrid")]
pub struct PyQuoteGrid {
    pub(crate) inner: QuoteGrid,
}

#[pymethods]
impl PyQuoteGrid {
    #[getter]
    fn n_quotes(&self) -> usize {
        self.inner.n_quotes
    }

    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    #[getter]
    fn multipliers(&self) -> Vec<f32> {
        self.inner.multipliers.clone()
    }

    #[getter]
    fn constraint_names(&self) -> Vec<String> {
        self.inner.constraint_names.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "QuoteGrid(n_quotes={}, n_steps={}, constraints={:?})",
            self.inner.n_quotes, self.inner.n_steps, self.inner.constraint_names
        )
    }
}
