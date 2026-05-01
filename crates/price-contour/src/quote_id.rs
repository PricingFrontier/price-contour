//! Quote-id column helpers.
//!
//! `quote_id` may be supplied as either Polars `Utf8` (the long-standing
//! default) or `Categorical` / `Enum` (dictionary-encoded). The
//! Categorical path exists so downstream pipelines can hand us
//! long-format DataFrames where each id is materialised once in the
//! dictionary plus per-row integer codes — bypassing the n_rows ×
//! per-string overhead the Utf8 path incurs at parquet decode time.
//!
//! **Chunk-local dictionary safety.** Each Categorical chunk owns its own
//! string dictionary; the same physical code in two different chunks may
//! refer to different strings. This module ONLY ever yields `&str` values
//! (via `iter_str` / `StringChunked::iter`), never raw codes, so callers
//! that copy the strings out (e.g. into a `Vec<String>`) are insulated
//! from cross-chunk code aliasing.

use polars::datatypes::CategoricalPhysical;
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A sequential iterator over a quote_id column's string values.
///
/// Returned items are `Option<&str>` (matching `StringChunked::iter()` and
/// `CategoricalChunked::iter_str()`); `None` indicates a Polars null in
/// that row, which callers must reject explicitly.
///
/// The iterator borrows from the underlying chunked array (and, for
/// Categorical, from its dictionary mapping). Successive `next()` calls
/// can yield items that coexist for the duration of the iteration —
/// neither variant is a streaming iterator.
pub(crate) type QuoteIdStrIter<'a> = Box<dyn Iterator<Item = Option<&'a str>> + 'a>;

/// Build a `QuoteIdStrIter` over a Polars column, accepting Utf8 or
/// Categorical / Enum and rejecting everything else with a clear error.
///
/// `col_name` is used in the error message so callers don't have to
/// repeat the column-name plumbing.
pub(crate) fn quote_id_str_iter<'a>(
    col: &'a Column,
    col_name: &str,
) -> PyResult<QuoteIdStrIter<'a>> {
    let physical = match col.dtype() {
        DataType::String => {
            // Utf8 path: zero-copy iter directly off the StringChunked.
            // `.unwrap()` is safe — the dtype check above guarantees the cast.
            return Ok(Box::new(col.str().unwrap().iter()));
        }
        DataType::Categorical(cats, _) => cats.physical(),
        DataType::Enum(frozen_cats, _) => frozen_cats.physical(),
        other => {
            return Err(PyValueError::new_err(format!(
                "{col_name} must be Utf8 (String) or Categorical, got {other:?}. \
                 Numeric, binary, and other dtypes are not supported as quote_id columns."
            )));
        }
    };

    // Categorical/Enum dispatch on physical code width. Polars picks the
    // smallest physical type that fits the dictionary cardinality (U8 for
    // ≤256 unique strings, U16 for ≤65k, U32 otherwise — there is no U64
    // variant in 0.52).
    match physical {
        CategoricalPhysical::U8 => Ok(Box::new(col.cat8().unwrap().iter_str())),
        CategoricalPhysical::U16 => Ok(Box::new(col.cat16().unwrap().iter_str())),
        CategoricalPhysical::U32 => Ok(Box::new(col.cat32().unwrap().iter_str())),
    }
}
