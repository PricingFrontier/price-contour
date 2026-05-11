//! Chunked builder for ratebook factor contexts.
//!
//! Mirrors `QuoteGridBuilder`'s contract: chunks are appended in whatever
//! order they arrive, and the canonical reorder (to `expected_quote_ids`)
//! happens once at `build()` time via cycle-following permutation. After
//! reorder, group indices are renumbered so that labels are assigned in
//! first-encounter order over the post-sort quote traversal — this
//! reproduces the dataframe-mode insertion-order semantics established by
//! `build_group_mapping`, giving byte-identical `factor_tables` on both
//! the dataframe and chunked-parquet paths.
//!
//! The builder is `pub(crate)` for tests/benches in this crate plus the
//! `price-contour` PyO3 crate which is the only production consumer. The
//! public Python surface (`RatebookFactorContexts.from_dataframe`,
//! `build_ratebook_factor_contexts_from_parquet_chunked`) wraps this
//! type; no third public construction surface is exposed.

use std::collections::HashMap;

use crate::data::{fingerprint_quote_ids, GroupMapping};
use crate::error::{PriceContourError, Result};

/// Per-factor build state held across chunks. Owned in `FactorContextBuilder`
/// as `Vec<PerFactorState>` (one entry per factor spec). Each chunk's
/// `append` pushes one new index into `group_of` per row and grows
/// `group_labels` / `label_to_idx` on first encounter.
struct PerFactorState {
    label_to_idx: HashMap<String, u32>,
    group_labels: Vec<String>,
    group_of: Vec<u32>,
}

impl PerFactorState {
    fn new() -> Self {
        Self {
            label_to_idx: HashMap::new(),
            group_labels: Vec::new(),
            group_of: Vec::new(),
        }
    }

    /// Look up `label`; assign a fresh index on miss.
    #[inline]
    fn intern(&mut self, label: String) -> u32 {
        if let Some(&idx) = self.label_to_idx.get(label.as_str()) {
            return idx;
        }
        let idx = self.group_labels.len() as u32;
        self.group_labels.push(label.clone());
        self.label_to_idx.insert(label, idx);
        idx
    }
}

/// Outcome of `FactorContextBuilder::build`.
#[derive(Debug)]
pub struct FactorContextsBuilt {
    /// Per-factor group mappings, ordered by `factor_specs`. Each
    /// mapping's `group_of` array is in the final quote order (either
    /// the source append order, or `expected_quote_ids` order if
    /// supplied) and the label numbering has been remapped to
    /// first-encounter order over that traversal.
    pub group_mappings: Vec<GroupMapping>,
    /// The ordered quote IDs corresponding to the rows of every
    /// `group_of` array. Length = `n_quotes`. Empty when the builder
    /// was driven without `quote_id` data — in that case the contexts
    /// have no provable order and `quote_id_fingerprint` is `None`.
    pub quote_ids: Vec<String>,
    /// `Some(hash)` when quote-id alignment can be verified against a
    /// `QuoteGrid` (i.e. quote IDs were tracked end-to-end). `None`
    /// when the builder was driven by labels only with no per-row
    /// quote ID, in which case the contexts must be paired with a
    /// matching `expected_quote_ids` at `build()` time to gain a
    /// fingerprint — otherwise downstream solvers reject the contexts
    /// because order cannot be proven.
    pub quote_id_fingerprint: Option<u64>,
}

/// Internal chunked builder for ratebook factor contexts.
///
/// The label-index assignment during `append` is in **append-encounter
/// order** — labels seen for the first time in chunk 3 get the next
/// available index regardless of what `expected_quote_ids` will be at
/// `build()`. The renumbering in `build()` is what makes the output
/// match dataframe-mode byte-for-byte.
pub struct FactorContextBuilder {
    factor_specs: Vec<Vec<String>>,
    per_factor: Vec<PerFactorState>,
    quote_ids: Vec<String>,
    /// `true` when the caller has been feeding `quote_id`s with each
    /// chunk. Once a chunk arrives without one, the contract is
    /// inconsistent and `append` rejects it.
    quote_ids_supplied: Option<bool>,
    n_quotes: usize,
    finalised: bool,
}

impl FactorContextBuilder {
    /// Construct a new builder for the given factor specs. `factor_specs`
    /// is record-keeping only — the builder does not extract labels
    /// from columns; callers feed pre-extracted per-factor label
    /// vectors via `append`.
    pub fn new(factor_specs: Vec<Vec<String>>) -> Self {
        let n_factors = factor_specs.len();
        Self {
            factor_specs,
            per_factor: (0..n_factors).map(|_| PerFactorState::new()).collect(),
            quote_ids: Vec::new(),
            quote_ids_supplied: None,
            n_quotes: 0,
            finalised: false,
        }
    }

    pub fn n_factors(&self) -> usize {
        self.per_factor.len()
    }

    pub fn n_quotes(&self) -> usize {
        self.n_quotes
    }

    pub fn factor_specs(&self) -> &[Vec<String>] {
        &self.factor_specs
    }

    /// Append one chunk.
    ///
    /// `chunk_labels[f]` is the per-row label sequence for factor `f`
    /// in this chunk. Every factor must have the same chunk length.
    /// `quote_ids_chunk`, when supplied, must have the same length and
    /// is appended to the running quote-id list for later reorder /
    /// validation.
    ///
    /// **Quote-id consistency.** If the first chunk was appended with
    /// a quote-id slice, every subsequent chunk must also supply one;
    /// likewise if the first chunk was driven without IDs, no later
    /// chunk may introduce them. This stops a builder from ending up
    /// half-aligned, half-positional.
    pub fn append(
        &mut self,
        quote_ids_chunk: Option<&[String]>,
        chunk_labels: Vec<Vec<String>>,
    ) -> Result<()> {
        if self.finalised {
            return Err(PriceContourError::InvalidValue(
                "FactorContextBuilder already finalised".into(),
            ));
        }
        if chunk_labels.len() != self.per_factor.len() {
            return Err(PriceContourError::DimensionMismatch(format!(
                "chunk_labels has {} factors, builder expects {}",
                chunk_labels.len(),
                self.per_factor.len()
            )));
        }

        // Verify every factor's slice has the same row count, and that
        // count agrees with `quote_ids_chunk` if supplied.
        let chunk_rows = chunk_labels.first().map(|v| v.len()).unwrap_or(0);
        for (f_idx, col) in chunk_labels.iter().enumerate() {
            if col.len() != chunk_rows {
                return Err(PriceContourError::DimensionMismatch(format!(
                    "factor {f_idx} chunk has {} rows but factor 0 has {chunk_rows} \
                     (all factors in a chunk must share the same row count)",
                    col.len()
                )));
            }
        }
        if let Some(qids) = quote_ids_chunk {
            if qids.len() != chunk_rows {
                return Err(PriceContourError::DimensionMismatch(format!(
                    "quote_ids chunk has {} rows but factor labels have {chunk_rows}",
                    qids.len()
                )));
            }
        }

        // Enforce quote-id consistency across the lifetime of the
        // builder. The first append establishes the contract; later
        // appends must match.
        let chunk_has_qids = quote_ids_chunk.is_some();
        match self.quote_ids_supplied {
            None => {
                self.quote_ids_supplied = Some(chunk_has_qids);
            }
            Some(prev) if prev != chunk_has_qids => {
                return Err(PriceContourError::InvalidValue(format!(
                    "FactorContextBuilder quote_id contract changed mid-stream \
                     (previous chunks {}supplied quote_ids; this chunk {}supplied them)",
                    if prev { "" } else { "did not " },
                    if chunk_has_qids { "" } else { "did not " }
                )));
            }
            _ => {}
        }

        // Empty chunks are a no-op.
        if chunk_rows == 0 {
            return Ok(());
        }

        if let Some(qids) = quote_ids_chunk {
            self.quote_ids.extend_from_slice(qids);
        }

        // Intern labels per factor in append order. The renumbering in
        // build() will produce the final group indices; for now we
        // just keep a self-consistent map.
        for (f, col) in chunk_labels.into_iter().enumerate() {
            let state = &mut self.per_factor[f];
            state.group_of.reserve(chunk_rows);
            for label in col {
                let idx = state.intern(label);
                state.group_of.push(idx);
            }
        }

        self.n_quotes += chunk_rows;
        Ok(())
    }

    /// Finalise the builder.
    ///
    /// * `expected_quote_ids` — when supplied, the builder verifies
    ///   that every appended quote_id matches one in `expected_quote_ids`
    ///   (count, no duplicates, no missing IDs, no unknowns), then
    ///   reorders every factor's `group_of` array to match the
    ///   `expected_quote_ids` order. The fingerprint is computed over
    ///   `expected_quote_ids`.
    /// * `expected_n_quotes` — when supplied, the builder confirms
    ///   the total appended quote count equals it. If
    ///   `expected_quote_ids` is also supplied, both must agree.
    ///
    /// The renumbering step at the end traverses each reordered
    /// `group_of[f]` from index 0 and assigns final group IDs in
    /// first-encounter order, producing the same group-label ordering
    /// dataframe-mode would produce on a DataFrame whose rows were in
    /// the same final quote order.
    pub fn build(
        mut self,
        expected_quote_ids: Option<&[String]>,
        expected_n_quotes: Option<usize>,
    ) -> Result<FactorContextsBuilt> {
        if self.finalised {
            return Err(PriceContourError::InvalidValue(
                "FactorContextBuilder already finalised".into(),
            ));
        }
        self.finalised = true;

        // Cross-validate expected_n_quotes vs expected_quote_ids.
        if let (Some(ids), Some(n)) = (expected_quote_ids, expected_n_quotes) {
            if ids.len() != n {
                return Err(PriceContourError::DimensionMismatch(format!(
                    "expected_quote_ids has {} entries but expected_n_quotes is {n}",
                    ids.len()
                )));
            }
        }
        if let Some(n) = expected_n_quotes {
            if n != self.n_quotes {
                return Err(PriceContourError::DimensionMismatch(format!(
                    "expected_n_quotes {n} != appended quote count {}",
                    self.n_quotes
                )));
            }
        }

        if self.n_quotes == 0 {
            return Err(PriceContourError::DataValidation(
                "no rows appended to FactorContextBuilder".into(),
            ));
        }

        let quote_ids_supplied = self.quote_ids_supplied.unwrap_or(false);

        // Build the post-reorder quote order and (optionally) a
        // source->target permutation for every factor's group_of array.
        let (final_quote_ids, permutation, fingerprint) = match expected_quote_ids {
            Some(expected) => {
                let perm = if quote_ids_supplied {
                    // The permutation builder verifies set-equality
                    // between source and expected (rejecting missing
                    // and unexpected with precise error messages), so
                    // the count check is implicit in this branch.
                    Some(build_reorder_permutation(&self.quote_ids, expected)?)
                } else {
                    // Positional-trust path: caller asserts row order
                    // already matches `expected`. We can only enforce a
                    // count check here.
                    if expected.len() != self.n_quotes {
                        return Err(PriceContourError::DimensionMismatch(format!(
                            "expected_quote_ids has {} entries but builder accumulated \
                             {} rows (positional-trust mode: no quote_id column was \
                             supplied, so row count must match expected_quote_ids exactly)",
                            expected.len(),
                            self.n_quotes
                        )));
                    }
                    None
                };
                let fp = fingerprint_quote_ids(expected);
                (expected.to_vec(), perm, Some(fp))
            }
            None => {
                if quote_ids_supplied {
                    let fp = fingerprint_quote_ids(&self.quote_ids);
                    (std::mem::take(&mut self.quote_ids), None, Some(fp))
                } else {
                    // Neither expected_quote_ids nor per-chunk quote_ids
                    // were supplied. Quote order is unknown; downstream
                    // solvers will reject contexts with no fingerprint.
                    (Vec::new(), None, None)
                }
            }
        };

        // Apply the reorder permutation (if any) and renumber labels
        // in first-encounter order over the final quote traversal.
        let mut group_mappings = Vec::with_capacity(self.per_factor.len());
        for state in self.per_factor.drain(..) {
            let PerFactorState {
                mut group_of,
                group_labels,
                ..
            } = state;
            if let Some(perm) = permutation.as_ref() {
                apply_permutation_u32(perm, &mut group_of);
            }
            let (final_group_of, final_labels) = renumber_first_encounter(group_of, group_labels);
            let n_groups = final_labels.len();
            group_mappings.push(GroupMapping {
                group_of: final_group_of,
                n_groups,
                group_labels: final_labels,
            });
        }

        Ok(FactorContextsBuilt {
            group_mappings,
            quote_ids: final_quote_ids,
            quote_id_fingerprint: fingerprint,
        })
    }
}

/// Build the permutation `perm[i] = j` meaning: the row at source-index
/// `j` belongs at target-index `i` after reorder. Rejects duplicate,
/// unexpected, and missing quote IDs with concrete error messages.
fn build_reorder_permutation(source: &[String], expected: &[String]) -> Result<Vec<usize>> {
    // Map source quote_id -> source index. Rejects duplicates in source.
    let mut source_index: HashMap<&str, usize> = HashMap::with_capacity(source.len());
    for (i, id) in source.iter().enumerate() {
        if let Some(prev) = source_index.insert(id.as_str(), i) {
            return Err(PriceContourError::DataValidation(format!(
                "duplicate quote_id '{id}' in factor source at indices {prev} and {i}"
            )));
        }
    }

    // For each target slot, find the source row carrying that quote_id.
    let mut perm = Vec::with_capacity(expected.len());
    for (target_idx, eid) in expected.iter().enumerate() {
        match source_index.remove(eid.as_str()) {
            Some(src) => perm.push(src),
            None => {
                return Err(PriceContourError::DataValidation(format!(
                    "quote_id '{eid}' expected at position {target_idx} but \
                     not found in factor source"
                )));
            }
        }
    }
    // Any remaining IDs in `source_index` were not in `expected`.
    if !source_index.is_empty() {
        // Report up to 3 examples to keep the error short.
        let mut extras: Vec<&str> = source_index.keys().copied().collect();
        extras.sort_unstable();
        let preview: Vec<&str> = extras.iter().take(3).copied().collect();
        return Err(PriceContourError::DataValidation(format!(
            "factor source contains {} quote_id(s) not in expected_quote_ids \
             (e.g. {:?})",
            extras.len(),
            preview
        )));
    }

    Ok(perm)
}

/// Apply `perm` to `data` out-of-place into a fresh `Vec<u32>`. Used for
/// factor `group_of` reorder where `data[perm[i]]` becomes `out[i]`.
///
/// We allocate a fresh buffer (O(n) extra memory per factor) rather than
/// the cycle-following in-place permutation used by `QuoteGridBuilder`.
/// Each factor's `group_of` is one `Vec<u32>` (4 bytes per quote), so for
/// 1M quotes × 3 factors the temporary peak is 12 MB total — well under
/// the wider quote-grid arrays. Cycle decomposition would save the
/// allocation but adds bitmap state and is harder to get right; the
/// trade-off doesn't pay here.
fn apply_permutation_u32(perm: &[usize], data: &mut Vec<u32>) {
    debug_assert_eq!(perm.len(), data.len());
    let mut out = vec![0u32; data.len()];
    for (target, &src) in perm.iter().enumerate() {
        out[target] = data[src];
    }
    *data = out;
}

/// Renumber `group_of` in first-encounter order. The input numbering
/// reflects the append-time encounter sequence; the output reflects the
/// post-reorder traversal. `label_table_in` is indexed by the OLD
/// numbering; we rebuild a parallel `label_table_out` indexed by the
/// new numbering.
fn renumber_first_encounter(
    group_of: Vec<u32>,
    label_table_in: Vec<String>,
) -> (Vec<u32>, Vec<String>) {
    let n_old = label_table_in.len();
    // remap[old_idx] = new_idx, or u32::MAX if not yet assigned.
    let mut remap: Vec<u32> = vec![u32::MAX; n_old];
    let mut label_table_out: Vec<String> = Vec::with_capacity(n_old);
    let mut out = Vec::with_capacity(group_of.len());

    // Move label strings out of the input table by index when we
    // first encounter each old index, so we don't pay for cloning
    // every label string.
    let mut label_table_in_opt: Vec<Option<String>> =
        label_table_in.into_iter().map(Some).collect();

    for old in group_of {
        let old_usize = old as usize;
        let new_idx = remap[old_usize];
        if new_idx == u32::MAX {
            let assigned = label_table_out.len() as u32;
            remap[old_usize] = assigned;
            // .take() leaves None behind; safe because each label is
            // taken at most once (the first encounter of `old_usize`).
            let label = label_table_in_opt[old_usize]
                .take()
                .expect("each old index is interned at most once");
            label_table_out.push(label);
            out.push(assigned);
        } else {
            out.push(new_idx);
        }
    }

    (out, label_table_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn test_single_factor_single_chunk() {
        let mut b = FactorContextBuilder::new(vec![vec![s("region")]]);
        b.append(
            None,
            vec![vec![s("North"), s("South"), s("North"), s("East")]],
        )
        .unwrap();
        // No expected_quote_ids and no quote_id_chunks => fingerprint None.
        let built = b.build(None, None).unwrap();
        assert_eq!(built.group_mappings.len(), 1);
        assert_eq!(built.quote_id_fingerprint, None);
        let m = &built.group_mappings[0];
        // First-encounter order: North=0, South=1, East=2.
        assert_eq!(m.group_labels, vec!["North", "South", "East"]);
        assert_eq!(m.group_of, vec![0, 1, 0, 2]);
        assert_eq!(m.n_groups, 3);
    }

    #[test]
    fn test_multi_factor_chunked_appends() {
        // Two factors, three chunks, same labels in different chunks.
        let mut b = FactorContextBuilder::new(vec![vec![s("region")], vec![s("age_band")]]);
        b.append(
            None,
            vec![vec![s("North"), s("South")], vec![s("A"), s("B")]],
        )
        .unwrap();
        b.append(
            None,
            vec![vec![s("North"), s("East")], vec![s("A"), s("C")]],
        )
        .unwrap();
        b.append(None, vec![vec![s("West")], vec![s("B")]]).unwrap();
        let built = b.build(None, None).unwrap();
        assert_eq!(
            built.group_mappings[0].group_labels,
            vec!["North", "South", "East", "West"]
        );
        assert_eq!(built.group_mappings[1].group_labels, vec!["A", "B", "C"]);
        assert_eq!(built.group_mappings[0].group_of, vec![0, 1, 0, 2, 3]);
        assert_eq!(built.group_mappings[1].group_of, vec![0, 1, 0, 2, 1]);
    }

    #[test]
    fn test_reorder_to_expected_quote_ids() {
        let mut b = FactorContextBuilder::new(vec![vec![s("region")]]);
        // Append in source order Q2, Q0, Q1 with labels A, B, C.
        b.append(
            Some(&[s("Q2"), s("Q0"), s("Q1")]),
            vec![vec![s("A"), s("B"), s("C")]],
        )
        .unwrap();
        let expected = vec![s("Q0"), s("Q1"), s("Q2")];
        let built = b.build(Some(&expected), None).unwrap();
        let m = &built.group_mappings[0];
        // After reorder: Q0 has B, Q1 has C, Q2 has A.
        // Renumber first-encounter over Q0,Q1,Q2: B->0, C->1, A->2.
        assert_eq!(m.group_labels, vec!["B", "C", "A"]);
        assert_eq!(m.group_of, vec![0, 1, 2]);
        assert_eq!(built.quote_ids, expected);
        assert_eq!(
            built.quote_id_fingerprint,
            Some(fingerprint_quote_ids(&expected))
        );
    }

    #[test]
    fn test_renumber_matches_dataframe_mode_after_reorder() {
        // Parity target: after the reorder, the renumbering should give
        // the same labels-by-index as calling build_group_mapping on the
        // reordered label sequence directly (which is what dataframe mode
        // produces when fed a quote-id-sorted DataFrame).
        use crate::data::build_group_mapping;

        let source_quote_ids = vec![s("Q5"), s("Q2"), s("Q9"), s("Q0"), s("Q3")];
        let source_labels = vec![s("red"), s("blue"), s("red"), s("green"), s("blue")];
        let expected_quote_ids = vec![s("Q0"), s("Q2"), s("Q3"), s("Q5"), s("Q9")];
        // After reorder by expected_quote_ids:
        //   Q0 -> green, Q2 -> blue, Q3 -> blue, Q5 -> red, Q9 -> red
        let post_sort_labels = vec![s("green"), s("blue"), s("blue"), s("red"), s("red")];
        let reference = build_group_mapping(&post_sort_labels);

        let mut b = FactorContextBuilder::new(vec![vec![s("colour")]]);
        b.append(Some(&source_quote_ids), vec![source_labels])
            .unwrap();
        let built = b.build(Some(&expected_quote_ids), None).unwrap();
        let m = &built.group_mappings[0];
        assert_eq!(m.group_labels, reference.group_labels);
        assert_eq!(m.group_of, reference.group_of);
    }

    #[test]
    fn test_rejects_duplicate_quote_ids() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(
            Some(&[s("Q0"), s("Q1"), s("Q0")]),
            vec![vec![s("a"), s("b"), s("c")]],
        )
        .unwrap();
        // Duplicate detection happens at build() against expected IDs.
        let err = b
            .build(Some(&[s("Q0"), s("Q1"), s("Q0")]), None)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("duplicate"), "msg: {msg}");
    }

    #[test]
    fn test_rejects_missing_quote_ids() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(Some(&[s("Q0"), s("Q1")]), vec![vec![s("a"), s("b")]])
            .unwrap();
        // Expected wants Q2 which we never saw.
        let err = b.build(Some(&[s("Q0"), s("Q2")]), None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Q2"), "msg should name missing id: {msg}");
    }

    #[test]
    fn test_rejects_unexpected_quote_ids() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(
            Some(&[s("Q0"), s("Q1"), s("Q9")]),
            vec![vec![s("a"), s("b"), s("c")]],
        )
        .unwrap();
        let err = b.build(Some(&[s("Q0"), s("Q1")]), None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not in expected"), "msg: {msg}");
    }

    #[test]
    fn test_rejects_no_rows() {
        let b = FactorContextBuilder::new(vec![vec![s("f")]]);
        let err = b.build(None, None).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no rows"), "msg: {msg}");
    }

    #[test]
    fn test_positional_trust_no_quote_id_column() {
        // Caller asserts factors rows are already in expected order; no
        // per-row quote_id column was extracted. Builder honours this and
        // assigns the expected fingerprint.
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(None, vec![vec![s("a"), s("b"), s("a")]]).unwrap();
        let expected = vec![s("Q0"), s("Q1"), s("Q2")];
        let built = b.build(Some(&expected), None).unwrap();
        assert_eq!(
            built.quote_id_fingerprint,
            Some(fingerprint_quote_ids(&expected))
        );
        // No reorder applied; first-encounter on the original sequence.
        assert_eq!(built.group_mappings[0].group_labels, vec!["a", "b"]);
        assert_eq!(built.group_mappings[0].group_of, vec![0, 1, 0]);
    }

    #[test]
    fn test_no_quote_ids_no_expected_yields_none_fingerprint() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(None, vec![vec![s("a"), s("b")]]).unwrap();
        let built = b.build(None, None).unwrap();
        assert_eq!(built.quote_id_fingerprint, None);
        assert!(built.quote_ids.is_empty());
    }

    #[test]
    fn test_expected_n_quotes_mismatch() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(None, vec![vec![s("a"), s("b")]]).unwrap();
        let err = b.build(None, Some(5)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("5") && msg.contains("2"), "msg: {msg}");
    }

    #[test]
    fn test_expected_quote_ids_and_n_quotes_disagree() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(Some(&[s("Q0")]), vec![vec![s("a")]]).unwrap();
        let err = b.build(Some(&[s("Q0")]), Some(7)).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("7"), "msg: {msg}");
    }

    #[test]
    fn test_quote_id_contract_changes_mid_stream() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(Some(&[s("Q0")]), vec![vec![s("a")]]).unwrap();
        let err = b.append(None, vec![vec![s("b")]]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("contract changed"), "msg: {msg}");
    }

    #[test]
    fn test_mismatched_chunk_shapes_across_factors() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f1")], vec![s("f2")]]);
        let err = b
            .append(None, vec![vec![s("a"), s("b")], vec![s("x")]])
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("rows"), "msg: {msg}");
    }

    #[test]
    fn test_empty_chunk_noop() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(None, vec![vec![]]).unwrap();
        // Second append with real data — the empty chunk neither
        // committed to a quote-id contract nor incremented n_quotes.
        b.append(None, vec![vec![s("a")]]).unwrap();
        let built = b.build(None, None).unwrap();
        assert_eq!(built.group_mappings[0].group_of, vec![0]);
    }

    #[test]
    fn test_chunk_size_one() {
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        for label in ["a", "b", "a", "c"] {
            b.append(None, vec![vec![s(label)]]).unwrap();
        }
        let built = b.build(None, None).unwrap();
        assert_eq!(built.group_mappings[0].group_labels, vec!["a", "b", "c"]);
        assert_eq!(built.group_mappings[0].group_of, vec![0, 1, 0, 2]);
    }

    #[test]
    fn test_reorder_with_chunked_input() {
        // The reorder works across chunks: append two chunks of unsorted
        // quote IDs and verify build() reorders correctly.
        let mut b = FactorContextBuilder::new(vec![vec![s("f")]]);
        b.append(Some(&[s("Q3"), s("Q1")]), vec![vec![s("x"), s("y")]])
            .unwrap();
        b.append(Some(&[s("Q0"), s("Q2")]), vec![vec![s("z"), s("y")]])
            .unwrap();
        let expected = vec![s("Q0"), s("Q1"), s("Q2"), s("Q3")];
        let built = b.build(Some(&expected), None).unwrap();
        // After reorder: Q0->z, Q1->y, Q2->y, Q3->x.
        // First-encounter: z=0, y=1, x=2.
        assert_eq!(built.group_mappings[0].group_labels, vec!["z", "y", "x"]);
        assert_eq!(built.group_mappings[0].group_of, vec![0, 1, 1, 2]);
    }
}
