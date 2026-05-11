use std::sync::Arc;

use pyo3::prelude::*;

use price_contour_core::QuoteGrid;

/// Python-visible opaque handle to a QuoteGrid.
#[pyclass(name = "QuoteGrid")]
pub struct PyQuoteGrid {
    pub(crate) inner: Arc<QuoteGrid>,
}

impl PyQuoteGrid {
    /// Create a new PyQuoteGrid wrapping a QuoteGrid in Arc.
    pub fn new(grid: QuoteGrid) -> Self {
        Self {
            inner: Arc::new(grid),
        }
    }
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
    fn scenario_values(&self) -> Vec<f32> {
        self.inner.scenario_values.clone()
    }

    #[getter]
    fn constraint_names(&self) -> Vec<String> {
        self.inner.constraint_names.clone()
    }

    #[getter]
    fn quote_ids(&self) -> Vec<String> {
        self.inner.quote_ids.clone()
    }

    /// 64-bit fingerprint of `quote_ids`, used by the ratebook factor-context
    /// path to validate alignment between the grid and prebuilt contexts in
    /// constant time. See [`price_contour_core::fingerprint_quote_ids`].
    #[getter]
    fn quote_id_fingerprint(&self) -> u64 {
        self.inner.quote_id_fingerprint
    }

    fn __repr__(&self) -> String {
        format!(
            "QuoteGrid(n_quotes={}, n_steps={}, constraints={:?})",
            self.inner.n_quotes, self.inner.n_steps, self.inner.constraint_names
        )
    }
}
