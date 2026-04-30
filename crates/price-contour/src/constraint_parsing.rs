//! Shared constraint-dict parsing and validation for the solver and
//! frontier PyO3 entry points.
//!
//! Both `solver_py::parse_constraints` and `frontier_py::sweep_frontier_py`
//! receive a `HashMap<String, HashMap<String, Option<f64>>>` from Python
//! and need to:
//!
//! 1. Surface a migration `ValueError` for the removed `min_abs` /
//!    `max_abs` keys (mentioning both old and new key names so the
//!    message stays in sync with the Python-side validator).
//! 2. Reject zero or multiple direction keys per constraint
//!    (`{"volume": {"min": 100, "max": 200}}` is ambiguous).
//! 3. Reject NaN / inf threshold values when the value is numeric. A
//!    `None` value is permitted (B1 frontier-only marker); the finite
//!    check is guarded by an `if let Some(v)` so it does not fire for
//!    `None`.
//! 4. Confirm the constraint name is a column in the grid.
//! 5. Iterate `grid.constraint_names` (NOT the user `HashMap`) when
//!    emitting the spec vector so spec[k] aligns with grid column k —
//!    `argmax.rs` assumes that alignment.
//!
//! The two call sites differ in how they consume the parsed entries:
//! the solver materialises absolute thresholds (and rejects `None` —
//! solve() requires a fixed threshold); the frontier emits a template
//! with threshold=0 plus a (lo, hi) range and is happy with `None` (the
//! sweep supplies the value). Both share `validate_constraints_dict`
//! for (1)–(4) and `direction_for` / `is_pct_key` for direction
//! extraction.

use std::collections::HashMap;

use price_contour_core::{ConstraintDirection, QuoteGrid};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// The four valid post-A1 direction keys.
pub(crate) const VALID_KEYS: [&str; 4] = ["min", "max", "min_pct", "max_pct"];

/// Validate a user-supplied constraints dict against the grid.
///
/// Surfaces the same errors the Python-side `_validate_constraint_dict`
/// would, with messages that stay close enough to keep the regex tests
/// (e.g. `RE_MIN_ABS_REMOVED`) passing on either side.
///
/// Does NOT walk the dict in any particular order for emission — the
/// caller is responsible for iterating `grid.constraint_names` when
/// building the spec vector. We only validate here.
pub(crate) fn validate_constraints_dict(
    constraints: &HashMap<String, HashMap<String, Option<f64>>>,
    grid: &QuoteGrid,
) -> PyResult<()> {
    for (name, spec_dict) in constraints {
        // Surface the migration hint BEFORE the multi-key / unknown-key
        // errors so users see the rename path immediately.
        if spec_dict.contains_key("min_abs") {
            return Err(PyValueError::new_err(
                "'min_abs' has been renamed to 'min'; \
                 the previous fraction-of-baseline 'min' is now 'min_pct'",
            ));
        }
        if spec_dict.contains_key("max_abs") {
            return Err(PyValueError::new_err(
                "'max_abs' has been renamed to 'max'; \
                 the previous fraction-of-baseline 'max' is now 'max_pct'",
            ));
        }

        if !grid.constraint_names.iter().any(|n| n == name) {
            return Err(PyValueError::new_err(format!(
                "Constraint '{}' not found in DataFrame columns. Available: {:?}",
                name, grid.constraint_names
            )));
        }

        // Count direction keys — exactly one is required. Mirrors the
        // Python validator's "must have exactly one key" rule so users
        // going through the from-grid Rust path don't silently get the
        // first-match-wins behaviour the old code had.
        let direction_keys: Vec<&str> = VALID_KEYS
            .iter()
            .copied()
            .filter(|k| spec_dict.contains_key(*k))
            .collect();

        if direction_keys.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Constraint '{}' must specify one of: min, max, min_pct, max_pct",
                name
            )));
        }
        if direction_keys.len() > 1 {
            return Err(PyValueError::new_err(format!(
                "Constraint '{}' must have exactly one key from {:?}, got {:?}",
                name, VALID_KEYS, direction_keys
            )));
        }

        // Reject NaN / inf threshold values when numeric. ``None`` is a
        // valid frontier-only marker (B1) and skips the finite check —
        // the value is supplied by the sweep, not the caller.
        let key = direction_keys[0];
        if let Some(value) = spec_dict[key] {
            if !value.is_finite() {
                // Use Python-style lowercase casing (`nan`/`inf`/`-inf`) so
                // the error message matches the Python validator verbatim.
                let value_repr = if value.is_nan() {
                    "nan"
                } else if value == f64::INFINITY {
                    "inf"
                } else {
                    "-inf"
                };
                return Err(PyValueError::new_err(format!(
                    "Constraint '{}' value for '{}' must be a finite number, got {}",
                    name, key, value_repr
                )));
            }
        }
    }
    Ok(())
}

/// Returns the constraint direction implied by which of the four keys
/// is present. Caller has already validated that exactly one is set.
///
/// Extension point: ratio constraints (C1) detect via `numerator` /
/// `denominator` keys before the direction-key match below.
pub(crate) fn direction_for(spec_dict: &HashMap<String, Option<f64>>) -> ConstraintDirection {
    if spec_dict.contains_key("min") || spec_dict.contains_key("min_pct") {
        ConstraintDirection::Min
    } else {
        ConstraintDirection::Max
    }
}

/// True if the spec dict uses a fraction-of-baseline key (`min_pct` /
/// `max_pct`); the absolute keys (`min` / `max`) return false.
pub(crate) fn is_pct_key(spec_dict: &HashMap<String, Option<f64>>) -> bool {
    spec_dict.contains_key("min_pct") || spec_dict.contains_key("max_pct")
}
