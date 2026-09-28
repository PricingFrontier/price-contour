//! Python face of cooperative cancellation (DESIGN_DECISIONS §14).

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use price_contour_core::{CancelFlag, PriceContourError};

create_exception!(
    price_contour,
    Cancelled,
    PyRuntimeError,
    "Raised by a call whose CancelToken was cancelled while it ran."
);

/// A thread-safe, one-way cancellation flag for `apply_from_grid` and
/// `RatebookOptimiser.evaluate`. `frozen`: every method takes `&self`, so any
/// thread may call `cancel()` while a call holding the token runs.
#[pyclass(name = "CancelToken", frozen, module = "price_contour")]
pub struct PyCancelToken {
    flag: CancelFlag,
}

impl PyCancelToken {
    pub(crate) fn flag(&self) -> CancelFlag {
        self.flag.clone()
    }
}

#[pymethods]
impl PyCancelToken {
    #[new]
    fn new() -> Self {
        Self {
            flag: CancelFlag::new(),
        }
    }

    /// Ask every call holding this token to stop. Idempotent; cannot be undone.
    fn cancel(&self) {
        self.flag.cancel();
    }

    #[getter]
    fn cancelled(&self) -> bool {
        self.flag.is_cancelled()
    }

    fn __repr__(&self) -> String {
        let cancelled = if self.flag.is_cancelled() {
            "True"
        } else {
            "False"
        };
        format!("CancelToken(cancelled={cancelled})")
    }
}

/// The flag behind an optional Python token.
pub(crate) fn flag_of(token: Option<&Bound<'_, PyCancelToken>>) -> Option<CancelFlag> {
    token.map(|t| t.get().flag())
}

/// Convert a core error: `Cancelled` becomes the `Cancelled` exception, every
/// other error the `ValueError` the bindings have always raised, prefixed with
/// `context`.
pub(crate) fn core_error(context: &str, error: PriceContourError) -> PyErr {
    match error {
        PriceContourError::Cancelled => cancelled_error(),
        other => PyValueError::new_err(format!("{context}: {other}")),
    }
}

pub(crate) fn cancelled_error() -> PyErr {
    Cancelled::new_err("the call was cancelled through its CancelToken")
}
