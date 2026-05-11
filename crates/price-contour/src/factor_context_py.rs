//! Python-facing surface for ratebook factor contexts.
//!
//! Two public constructors land here:
//!
//! * `RatebookFactorContexts.from_dataframe(...)` — classmethod that
//!   builds contexts from a single in-memory `pl.DataFrame`.
//! * `build_ratebook_factor_contexts_from_parquet_chunked(...)` —
//!   top-level function that streams a parquet file in fixed-size row
//!   slices.
//!
//! Both route through `FactorContextBuilder` so the
//! deterministic-remap, reorder, and fingerprint logic exists in
//! exactly one place. The internal `FactorContextBuilder` is
//! `pub(crate)` only — there's no separate public Rust builder API
//! (the spec explicitly defers that until there's a second consumer).

use std::sync::Arc;

use polars::prelude::*;
use price_contour_core::{FactorContextBuilder, FactorContextsBuilt, GroupMapping};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_polars::PyDataFrame;

use crate::parquet_grid_py::{open_metadata, read_parquet_slice};
use crate::quote_id::quote_id_str_iter;
use crate::ratebook_helpers_py::{build_spec_labels, PyFactorContext};

/// Opaque Python-facing handle to a set of ratebook factor contexts.
///
/// The wrapper owns one `Arc<GroupMapping>` per factor spec plus the
/// quote-id fingerprint and the `factor_specs` it was built for. The
/// Rust-side group mappings flow through to the existing
/// `run_cd_pass_py` / `solve_grouped_py` paths via the
/// `_factor_contexts_for_solver` accessor.
///
/// **Why opaque.** We deliberately do not expose `contexts: list[FactorContext]`
/// as a public field. `FactorContext` is internal-only; exposing a list
/// of them would commit us to a public API surface that ties our
/// internal `GroupMapping` shape to a backwards-compatibility
/// guarantee. Read-only metadata (counts, fingerprint, factor_specs)
/// is enough for the public API; the actual mappings stay behind the
/// type and only the solver path inside this crate sees them.
#[pyclass(name = "RatebookFactorContexts", frozen)]
pub struct PyRatebookFactorContexts {
    factor_specs: Vec<Vec<String>>,
    separator: String,
    n_quotes: usize,
    quote_id_fingerprint: Option<u64>,
    /// One `Arc<GroupMapping>` per factor spec, in spec order.
    mappings: Vec<Arc<GroupMapping>>,
}

impl PyRatebookFactorContexts {
    /// Internal constructor used by both Python-facing entry points
    /// (`from_dataframe` and the parquet helper) after the core
    /// builder has produced its `FactorContextsBuilt`.
    fn from_built(
        factor_specs: Vec<Vec<String>>,
        separator: String,
        built: FactorContextsBuilt,
    ) -> Self {
        let n_quotes = built.group_mappings.first().map_or(0, |m| m.group_of.len());
        let mappings = built
            .group_mappings
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();
        Self {
            factor_specs,
            separator,
            n_quotes,
            quote_id_fingerprint: built.quote_id_fingerprint,
            mappings,
        }
    }
}

#[pymethods]
impl PyRatebookFactorContexts {
    #[getter]
    fn factor_specs(&self) -> Vec<Vec<String>> {
        self.factor_specs.clone()
    }

    #[getter]
    fn n_factors(&self) -> usize {
        self.mappings.len()
    }

    #[getter]
    fn n_quotes(&self) -> usize {
        self.n_quotes
    }

    #[getter]
    fn separator(&self) -> String {
        self.separator.clone()
    }

    /// `None` when the contexts were built without provable quote
    /// order (no `quote_id` column and no `expected_quote_ids`).
    /// `solve(QuoteGrid, contexts)` rejects contexts with no
    /// fingerprint.
    #[getter]
    fn quote_id_fingerprint(&self) -> Option<u64> {
        self.quote_id_fingerprint
    }

    fn __repr__(&self) -> String {
        format!(
            "RatebookFactorContexts(n_factors={}, n_quotes={}, fingerprint={})",
            self.mappings.len(),
            self.n_quotes,
            self.quote_id_fingerprint
                .map(|fp| format!("0x{fp:016x}"))
                .unwrap_or_else(|| "None".to_string()),
        )
    }

    /// Internal accessor used by `ratebook.py` to plug the contexts
    /// into the existing `run_cd_pass_py` / `solve_grouped_py` calls.
    /// Returns a fresh `list[FactorContext]` where each element shares
    /// an `Arc<GroupMapping>` with this object — no per-call rebuild.
    ///
    /// Leading underscore signals "private; do not rely on" — this is
    /// not part of the documented public API and may change.
    fn _factor_contexts_for_solver(&self, py: Python<'_>) -> PyResult<Vec<Py<PyFactorContext>>> {
        self.mappings
            .iter()
            .map(|m| Py::new(py, PyFactorContext::from_arc(Arc::clone(m))))
            .collect()
    }

    /// Build factor contexts from a single in-memory `pl.DataFrame`.
    ///
    /// Routes through the same internal builder as the parquet
    /// helper, so the dataframe and chunked paths produce bit-identical
    /// `factor_tables` for the same logical input.
    ///
    /// `quote_id` selects the column whose values become the
    /// alignment fingerprint. Pass `None` for legacy callers that
    /// already align rows positionally to the quote grid — they must
    /// supply `expected_quote_ids` to get a fingerprint, otherwise
    /// `solve(QuoteGrid, contexts)` will reject the contexts.
    #[classmethod]
    #[pyo3(signature = (
        factors,
        factor_specs,
        *,
        quote_id = Some("quote_id"),
        separator = "\x1f",
        expected_quote_ids = None,
        expected_n_quotes = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_dataframe(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        factors: PyDataFrame,
        factor_specs: Vec<Vec<String>>,
        quote_id: Option<&str>,
        separator: &str,
        expected_quote_ids: Option<Vec<String>>,
        expected_n_quotes: Option<usize>,
    ) -> PyResult<Self> {
        cross_validate_expected(&expected_quote_ids, expected_n_quotes)?;

        let df = factors.0;
        let separator_owned = separator.to_string();
        let factor_specs_owned = factor_specs.clone();
        let expected_ref = expected_quote_ids.as_deref();

        py.detach(move || -> PyResult<PyRatebookFactorContexts> {
            // Extract per-factor labels from the DataFrame in one pass.
            // Mirrors `build_factor_contexts_py`'s extraction so the
            // dataframe path remains byte-identical to legacy.
            let mut per_factor_labels: Vec<Vec<String>> = Vec::with_capacity(factor_specs.len());
            for spec in &factor_specs {
                let labels = build_spec_labels(&df, spec, &separator_owned)?;
                per_factor_labels.push(labels);
            }

            // Extract quote_ids if requested. When `quote_id` is None
            // (legacy callers without an id column), the builder is fed
            // labels positionally — `expected_quote_ids` becomes the
            // sole proof of alignment.
            let quote_ids: Option<Vec<String>> = match quote_id {
                Some(col_name) => Some(extract_quote_ids_column(&df, col_name)?),
                None => None,
            };

            let mut builder = FactorContextBuilder::new(factor_specs_owned);
            builder
                .append(quote_ids.as_deref(), per_factor_labels)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            let built = builder
                .build(expected_ref, expected_n_quotes)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;

            Ok(PyRatebookFactorContexts::from_built(
                factor_specs,
                separator_owned,
                built,
            ))
        })
    }
}

/// Stream a parquet file in fixed-size row slices, feeding each chunk
/// into a `FactorContextBuilder`. Memory peak for the parquet decode
/// buffer is bounded by `chunk_size`; only the projected columns
/// (`quote_id` + factor columns) are decoded.
///
/// Unlike `build_grid_from_parquet_chunked_py`, factor parquets have
/// one row per quote, so no `n_steps` alignment is needed and
/// `chunk_size` is simply a row count.
#[pyfunction]
#[pyo3(signature = (
    path,
    factor_specs,
    chunk_size,
    *,
    quote_id = Some("quote_id"),
    separator = "\x1f",
    expected_quote_ids = None,
    expected_n_quotes = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn build_ratebook_factor_contexts_from_parquet_chunked_py(
    py: Python<'_>,
    path: &str,
    factor_specs: Vec<Vec<String>>,
    chunk_size: usize,
    quote_id: Option<&str>,
    separator: &str,
    expected_quote_ids: Option<Vec<String>>,
    expected_n_quotes: Option<usize>,
) -> PyResult<PyRatebookFactorContexts> {
    if chunk_size == 0 {
        return Err(PyValueError::new_err("chunk_size must be > 0"));
    }
    if factor_specs.is_empty() {
        return Err(PyValueError::new_err(
            "factor_specs is empty; at least one factor spec is required",
        ));
    }
    for (i, spec) in factor_specs.iter().enumerate() {
        if spec.is_empty() {
            return Err(PyValueError::new_err(format!(
                "factor_specs[{i}] is empty; each spec must list at least one column"
            )));
        }
    }
    cross_validate_expected(&expected_quote_ids, expected_n_quotes)?;

    // Collect the parquet projection: every column referenced by any
    // factor spec, plus `quote_id` if supplied. We deduplicate so the
    // projection list is minimal and `validate_column_names` doesn't
    // reject a column that happens to be referenced by two specs.
    let mut needed: Vec<String> = Vec::new();
    if let Some(qid_col) = quote_id {
        needed.push(qid_col.to_string());
    }
    for spec in &factor_specs {
        for col in spec {
            if !needed.contains(col) {
                needed.push(col.clone());
            }
        }
    }

    let separator_owned = separator.to_string();
    let factor_specs_clone = factor_specs.clone();
    let expected_ref = expected_quote_ids.clone();

    py.detach(move || -> PyResult<PyRatebookFactorContexts> {
        let (total_rows, metadata) = open_metadata(path)?;
        if total_rows == 0 {
            return Err(PyValueError::new_err("parquet file has no rows"));
        }

        // Sanity-check column collisions (e.g. a factor spec naming
        // `quote_id`). We piggyback on the existing helper by passing
        // dummies for the four schema-column slots; the only mode we
        // care about is "no two columns share a name", which the
        // helper's de-duplication enforces.
        validate_factor_projection(quote_id, &factor_specs)?;

        let mut builder = FactorContextBuilder::new(factor_specs_clone);

        let mut offset = 0;
        while offset < total_rows {
            let len = chunk_size.min(total_rows - offset);
            let df = read_parquet_slice(path, &metadata, offset, len, &needed)?;
            append_dataframe_chunk(&mut builder, &df, quote_id, &factor_specs, &separator_owned)?;
            offset += len;
        }

        let built = builder
            .build(expected_ref.as_deref(), expected_n_quotes)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok(PyRatebookFactorContexts::from_built(
            factor_specs,
            separator_owned,
            built,
        ))
    })
}

/// Shared cross-validation for the two public constructors: if both
/// `expected_quote_ids` and `expected_n_quotes` are supplied, their
/// lengths must agree.
fn cross_validate_expected(
    expected_quote_ids: &Option<Vec<String>>,
    expected_n_quotes: Option<usize>,
) -> PyResult<()> {
    if let (Some(ids), Some(n)) = (expected_quote_ids.as_ref(), expected_n_quotes) {
        if ids.len() != n {
            return Err(PyValueError::new_err(format!(
                "expected_quote_ids has {} entries but expected_n_quotes is {n}",
                ids.len()
            )));
        }
    }
    Ok(())
}

/// Reject obviously-malformed projections: the `quote_id` column (if
/// supplied) shouldn't also appear in any factor spec, and within the
/// projection no factor column should collide with `quote_id`. The
/// underlying parquet reader rejects unknown columns naturally.
fn validate_factor_projection(
    quote_id: Option<&str>,
    factor_specs: &[Vec<String>],
) -> PyResult<()> {
    let Some(qid) = quote_id else {
        return Ok(());
    };
    for (i, spec) in factor_specs.iter().enumerate() {
        for col in spec {
            if col == qid {
                return Err(PyValueError::new_err(format!(
                    "factor_specs[{i}] references the quote_id column '{qid}'; \
                     a factor spec must not reuse the quote_id column"
                )));
            }
        }
    }
    Ok(())
}

/// Extract the `quote_id` column from a chunk DataFrame as a
/// `Vec<String>`. Cast Utf8 or Categorical to String, rejecting nulls
/// with a clear error naming the offending column.
fn extract_quote_ids_column(df: &DataFrame, column_name: &str) -> PyResult<Vec<String>> {
    let col = df.column(column_name).map_err(|_| {
        PyValueError::new_err(format!(
            "quote_id column '{column_name}' not found in factor source"
        ))
    })?;
    if col.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "quote_id column '{column_name}' contains null values"
        )));
    }
    let iter = quote_id_str_iter(col, column_name)?;
    let mut out = Vec::with_capacity(df.height());
    for (i, item) in iter.enumerate() {
        let s = item.ok_or_else(|| {
            PyValueError::new_err(format!(
                "Null in quote_id column '{column_name}' at row {i}"
            ))
        })?;
        out.push(s.to_string());
    }
    Ok(out)
}

/// Append one chunk of factor data to the builder. Extracts the
/// `quote_id` column (if supplied) plus the per-factor labels via the
/// shared `build_spec_labels` helper.
fn append_dataframe_chunk(
    builder: &mut FactorContextBuilder,
    df: &DataFrame,
    quote_id: Option<&str>,
    factor_specs: &[Vec<String>],
    separator: &str,
) -> PyResult<()> {
    let quote_ids: Option<Vec<String>> = match quote_id {
        Some(col_name) => Some(extract_quote_ids_column(df, col_name)?),
        None => None,
    };
    let mut chunk_labels: Vec<Vec<String>> = Vec::with_capacity(factor_specs.len());
    for spec in factor_specs {
        let labels = build_spec_labels(df, spec, separator)?;
        chunk_labels.push(labels);
    }
    builder
        .append(quote_ids.as_deref(), chunk_labels)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(())
}
