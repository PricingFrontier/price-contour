//! Canonical evaluation of a ratebook solution.
//!
//! Every reported ratebook number — solve totals, frontier rows and the
//! per-quote results — comes from [`evaluate_ratebook`]. The coordinate-descent
//! loop keeps an incrementally updated f32 multiplier as a search device; that
//! value is never reported, because no consumer can reproduce it from the
//! published factor tables. This kernel recomputes each quote's factor product
//! from the tables themselves, so `evaluate(tables)` is exactly reproducible.
//! See `docs/DESIGN_DECISIONS.md` §13.1.

use rayon::prelude::*;

use crate::constants::RECONSTRUCT_PAR_GRAIN;
use crate::data::{GroupMapping, QuoteGrid};
use crate::error::{PriceContourError, Result};
use crate::solver::grouped::nearest_step;

/// One ratebook solution evaluated per quote.
#[derive(Debug, Clone)]
pub struct RatebookEvaluation {
    /// Per-quote product of its factor values, in f32, multiplied left to
    /// right in factor order starting from 1.0 (length `n_quotes`).
    pub factor_product: Vec<f32>,
    /// Per-quote grid step the product maps to (length `n_quotes`).
    pub optimal_steps: Vec<u32>,
    /// Per-quote flag: the product lies strictly below the first scenario value.
    pub clamped_low: Vec<bool>,
    /// Per-quote flag: the product lies strictly above the last scenario value.
    pub clamped_high: Vec<bool>,
    /// Number of quotes with `clamped_low`.
    pub n_clamped_low: u64,
    /// Number of quotes with `clamped_high`.
    pub n_clamped_high: u64,
    /// Objective summed in f64 over every quote at its step.
    pub total_objective: f64,
    /// Each constraint summed in f64 over every quote at its step, in grid
    /// constraint order.
    pub total_constraints: Vec<f64>,
}

/// Per-chunk partial sums, combined sequentially in chunk order so the totals
/// are bit-identical regardless of thread count or scheduling.
struct ChunkTotals {
    objective: f64,
    constraints: Vec<f64>,
    n_clamped_low: u64,
    n_clamped_high: u64,
}

/// Evaluate a ratebook solution: each quote is priced at the grid step
/// nearest the product of its factor values.
///
/// * `mappings[f]` maps each quote to its level (group) of factor `f`.
/// * `factor_values[f][g]` is the rate of level `g` of factor `f`.
///
/// The step rule is the grouped solver's `nearest_step`: a product at or
/// below the first scenario value maps to step 0, at or above the last maps
/// to the last step, and an exact midpoint maps to the lower step.
pub fn evaluate_ratebook(
    grid: &QuoteGrid,
    mappings: &[&GroupMapping],
    factor_values: &[&[f32]],
) -> Result<RatebookEvaluation> {
    grid.validate()?;
    validate_inputs(grid, mappings, factor_values)?;

    let n_quotes = grid.n_quotes;
    let n_steps = grid.n_steps;
    let n_constraints = grid.constraints.len();
    let scenario_values = grid.scenario_values.as_slice();
    let first_sv = scenario_values[0];
    let last_sv = scenario_values[n_steps - 1];
    let objective = grid.objective.as_slice();
    let constraint_cols: Vec<&[f32]> = grid.constraints.iter().map(|c| c.as_slice()).collect();

    let mut factor_product = vec![0.0f32; n_quotes];
    let mut optimal_steps = vec![0u32; n_quotes];
    let mut clamped_low = vec![false; n_quotes];
    let mut clamped_high = vec![false; n_quotes];

    let partials: Vec<ChunkTotals> = factor_product
        .par_chunks_mut(RECONSTRUCT_PAR_GRAIN)
        .zip(optimal_steps.par_chunks_mut(RECONSTRUCT_PAR_GRAIN))
        .zip(clamped_low.par_chunks_mut(RECONSTRUCT_PAR_GRAIN))
        .zip(clamped_high.par_chunks_mut(RECONSTRUCT_PAR_GRAIN))
        .enumerate()
        .map(|(chunk_idx, (((products, steps), lows), highs))| {
            let start = chunk_idx * RECONSTRUCT_PAR_GRAIN;
            let mut totals = ChunkTotals {
                objective: 0.0,
                constraints: vec![0.0; n_constraints],
                n_clamped_low: 0,
                n_clamped_high: 0,
            };
            for local_i in 0..products.len() {
                let i = start + local_i;
                let mut product = 1.0f32;
                for (mapping, values) in mappings.iter().zip(factor_values.iter()) {
                    product *= values[mapping.group_of[i] as usize];
                }
                let (step, _) = nearest_step(scenario_values, product);
                let low = product < first_sv;
                let high = product > last_sv;

                products[local_i] = product;
                steps[local_i] = step as u32;
                lows[local_i] = low;
                highs[local_i] = high;
                totals.n_clamped_low += u64::from(low);
                totals.n_clamped_high += u64::from(high);

                let idx = i * n_steps + step;
                totals.objective += objective[idx] as f64;
                for (total, col) in totals.constraints.iter_mut().zip(constraint_cols.iter()) {
                    *total += col[idx] as f64;
                }
            }
            totals
        })
        .collect();

    let mut total_objective = 0.0f64;
    let mut total_constraints = vec![0.0f64; n_constraints];
    let mut n_clamped_low = 0u64;
    let mut n_clamped_high = 0u64;
    for partial in &partials {
        total_objective += partial.objective;
        for (total, part) in total_constraints.iter_mut().zip(partial.constraints.iter()) {
            *total += part;
        }
        n_clamped_low += partial.n_clamped_low;
        n_clamped_high += partial.n_clamped_high;
    }

    Ok(RatebookEvaluation {
        factor_product,
        optimal_steps,
        clamped_low,
        clamped_high,
        n_clamped_low,
        n_clamped_high,
        total_objective,
        total_constraints,
    })
}

fn validate_inputs(
    grid: &QuoteGrid,
    mappings: &[&GroupMapping],
    factor_values: &[&[f32]],
) -> Result<()> {
    if mappings.is_empty() {
        return Err(PriceContourError::InvalidValue(
            "ratebook evaluation needs at least one factor".into(),
        ));
    }
    if mappings.len() != factor_values.len() {
        return Err(PriceContourError::DimensionMismatch(format!(
            "{} factor mappings but {} factor value tables",
            mappings.len(),
            factor_values.len()
        )));
    }
    for (f, (mapping, values)) in mappings.iter().zip(factor_values.iter()).enumerate() {
        if mapping.group_of.len() != grid.n_quotes {
            return Err(PriceContourError::DimensionMismatch(format!(
                "factor {f} maps {} quotes but the grid has {}",
                mapping.group_of.len(),
                grid.n_quotes
            )));
        }
        if values.len() != mapping.n_groups {
            return Err(PriceContourError::DimensionMismatch(format!(
                "factor {f} has {} levels but {} values",
                mapping.n_groups,
                values.len()
            )));
        }
        if let Some((g, v)) = values
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite() || **v <= 0.0)
        {
            return Err(PriceContourError::InvalidValue(format!(
                "factor {f} level {g} ('{}') has value {v}; factor values must be finite and > 0",
                mapping
                    .group_labels
                    .get(g)
                    .map(String::as_str)
                    .unwrap_or("?")
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::build_group_mapping;

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    /// Objective at (quote q, step k) = 100·q + k; constraint = 10·q + k.
    /// Every (quote, step) cell is distinct, so a wrong step shows up in the totals.
    fn indexed_grid(scenario_values: Vec<f32>, n_quotes: usize) -> QuoteGrid {
        let n_steps = scenario_values.len();
        let mut objective = Vec::with_capacity(n_quotes * n_steps);
        let mut volume = Vec::with_capacity(n_quotes * n_steps);
        for q in 0..n_quotes {
            for k in 0..n_steps {
                objective.push((100 * q + k) as f32);
                volume.push((10 * q + k) as f32);
            }
        }
        QuoteGrid {
            n_quotes,
            n_steps,
            scenario_values,
            objective,
            constraints: vec![volume],
            constraint_names: vec!["volume".into()],
            quote_ids: (0..n_quotes).map(|q| format!("Q{q}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    #[test]
    fn hand_computed_two_factors() {
        // Grid [0.8, 1.0, 1.2]; factor A levels {a: 1.2, b: 0.8}, factor B
        // levels {x: 1.0, y: 1.25}. Products are exact in f32:
        //   Q0 a·x = 1.2  → step 2
        //   Q1 b·x = 0.8  → step 0
        //   Q2 a·y = 1.5  → step 2, clamped high
        let grid = indexed_grid(vec![0.8, 1.0, 1.2], 3);
        let a = build_group_mapping(&labels(&["a", "b", "a"]));
        let b = build_group_mapping(&labels(&["x", "x", "y"]));
        let eval = evaluate_ratebook(&grid, &[&a, &b], &[&[1.2, 0.8], &[1.0, 1.25]]).unwrap();

        assert_eq!(eval.optimal_steps, vec![2, 0, 2]);
        assert_eq!(
            eval.factor_product,
            vec![1.2f32 * 1.0, 0.8f32 * 1.0, 1.2f32 * 1.25]
        );
        assert_eq!(eval.clamped_low, vec![false, false, false]);
        assert_eq!(eval.clamped_high, vec![false, false, true]);
        assert_eq!((eval.n_clamped_low, eval.n_clamped_high), (0, 1));
        // objective: (0+2) + (100+0) + (200+2) = 304; volume: 2 + 10 + 22 = 34
        assert_eq!(eval.total_objective, 304.0);
        assert_eq!(eval.total_constraints, vec![34.0]);
    }

    #[test]
    fn product_is_sequential_f32_in_factor_order() {
        let grid = indexed_grid(vec![0.5, 1.0, 1.5], 1);
        let fa = build_group_mapping(&labels(&["l"]));
        let fb = build_group_mapping(&labels(&["l"]));
        let fc = build_group_mapping(&labels(&["l"]));
        let (va, vb, vc) = (1.1f32, 0.93f32, 1.07f32);
        let eval = evaluate_ratebook(&grid, &[&fa, &fb, &fc], &[&[va], &[vb], &[vc]]).unwrap();
        assert_eq!(eval.factor_product[0], ((1.0f32 * va) * vb) * vc);
    }

    #[test]
    fn exact_midpoint_goes_to_lower_step() {
        // 1.0 is exactly halfway between 0.75 and 1.25.
        let grid = indexed_grid(vec![0.75, 1.25], 1);
        let f = build_group_mapping(&labels(&["l"]));
        let eval = evaluate_ratebook(&grid, &[&f], &[&[1.0]]).unwrap();
        assert_eq!(eval.optimal_steps, vec![0]);
        assert_eq!((eval.n_clamped_low, eval.n_clamped_high), (0, 0));
    }

    #[test]
    fn clamp_flags_are_strict_at_both_ends() {
        // Products: 0.5 (below), 0.75 (on the low edge), 1.25 (on the high
        // edge), 2.0 (above).
        let grid = indexed_grid(vec![0.75, 1.0, 1.25], 4);
        let f = build_group_mapping(&labels(&["lo", "at_lo", "at_hi", "hi"]));
        let eval = evaluate_ratebook(&grid, &[&f], &[&[0.5, 0.75, 1.25, 2.0]]).unwrap();
        assert_eq!(eval.optimal_steps, vec![0, 0, 2, 2]);
        assert_eq!(eval.clamped_low, vec![true, false, false, false]);
        assert_eq!(eval.clamped_high, vec![false, false, false, true]);
        assert_eq!((eval.n_clamped_low, eval.n_clamped_high), (1, 1));
    }

    #[test]
    fn one_step_grid() {
        let grid = indexed_grid(vec![1.0], 3);
        let f = build_group_mapping(&labels(&["a", "b", "c"]));
        let eval = evaluate_ratebook(&grid, &[&f], &[&[0.9, 1.0, 1.1]]).unwrap();
        assert_eq!(eval.optimal_steps, vec![0, 0, 0]);
        assert_eq!(eval.clamped_low, vec![true, false, false]);
        assert_eq!(eval.clamped_high, vec![false, false, true]);
    }

    #[test]
    fn rejects_no_factors() {
        let grid = indexed_grid(vec![1.0], 1);
        let err = evaluate_ratebook(&grid, &[], &[]).unwrap_err();
        assert!(format!("{err}").contains("at least one factor"), "{err}");
    }

    #[test]
    fn rejects_mapping_value_count_mismatch() {
        let grid = indexed_grid(vec![1.0], 1);
        let f = build_group_mapping(&labels(&["a"]));
        let err = evaluate_ratebook(&grid, &[&f], &[]).unwrap_err();
        assert!(
            format!("{err}").contains("1 factor mappings but 0"),
            "{err}"
        );
    }

    #[test]
    fn rejects_mapping_for_other_quote_count() {
        let grid = indexed_grid(vec![1.0], 2);
        let f = build_group_mapping(&labels(&["a"]));
        let err = evaluate_ratebook(&grid, &[&f], &[&[1.0]]).unwrap_err();
        assert!(
            format!("{err}").contains("maps 1 quotes but the grid has 2"),
            "{err}"
        );
    }

    #[test]
    fn rejects_level_value_count_mismatch() {
        let grid = indexed_grid(vec![1.0], 2);
        let f = build_group_mapping(&labels(&["a", "b"]));
        let err = evaluate_ratebook(&grid, &[&f], &[&[1.0]]).unwrap_err();
        assert!(
            format!("{err}").contains("has 2 levels but 1 values"),
            "{err}"
        );
    }

    #[test]
    fn rejects_non_positive_and_non_finite_values() {
        let grid = indexed_grid(vec![1.0], 1);
        let f = build_group_mapping(&labels(&["young"]));
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let err = evaluate_ratebook(&grid, &[&f], &[&[bad]]).unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("'young'") && msg.contains("finite and > 0"),
                "{msg}"
            );
        }
    }

    #[test]
    fn totals_are_bit_identical_across_thread_counts() {
        // Several parallel chunks, values that do not sum exactly in f64.
        let n_quotes = RECONSTRUCT_PAR_GRAIN * 5 + 17;
        let sv = vec![0.8f32, 0.9, 1.0, 1.1, 1.2];
        let mut grid = indexed_grid(sv, n_quotes);
        for (i, v) in grid.objective.iter_mut().enumerate() {
            *v = 1.0 / (1.0 + i as f32 * 0.37);
        }
        let level_labels: Vec<String> = (0..n_quotes).map(|i| format!("L{}", i % 7)).collect();
        let f = build_group_mapping(&level_labels);
        let values: Vec<f32> = (0..7).map(|g| 0.85 + 0.05 * g as f32).collect();

        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| evaluate_ratebook(&grid, &[&f], &[&values]).unwrap())
        };
        let single = run(1);
        let many = run(8);
        assert_eq!(
            single.total_objective.to_bits(),
            many.total_objective.to_bits()
        );
        assert_eq!(
            single.total_constraints[0].to_bits(),
            many.total_constraints[0].to_bits()
        );
        assert_eq!(single.optimal_steps, many.optimal_steps);
    }
}
