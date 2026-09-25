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

/// The baseline step of a list of scenario values (the single baseline rule,
/// `price_contour_core::baseline_step`), for Python-side code that works on
/// a DataFrame rather than a built grid.
#[pyfunction]
pub fn _baseline_step_index(scenario_values: Vec<f32>) -> PyResult<usize> {
    if scenario_values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "scenario_values must not be empty",
        ));
    }
    Ok(price_contour_core::baseline_step(&scenario_values))
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

    /// Index of the baseline step: the scenario value nearest 1.0 (f32),
    /// lowest index on a tie. See `price_contour_core::baseline_step`.
    #[getter]
    fn baseline_step(&self) -> usize {
        self.inner.baseline_step()
    }

    /// Scenario value of the baseline step.
    #[getter]
    fn baseline_scenario_value(&self) -> f32 {
        self.inner.scenario_values[self.inner.baseline_step()]
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
