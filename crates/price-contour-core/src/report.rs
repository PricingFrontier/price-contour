//! Per-point aggregates for a frontier impact report
//! (`docs/DESIGN_DECISIONS.md` §19.2, §19.3).
//!
//! Everything here reduces per-quote data to a fixed-size summary. Reductions
//! over groups run on fixed chunks combined in chunk order, so the result does
//! not depend on the thread count.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::constants::RECONSTRUCT_PAR_GRAIN;
use crate::data::{GroupMapping, QuoteGrid};
use crate::error::{PriceContourError, Result};

/// The percentiles a report records, in order.
pub const PERCENTILES: [u64; 7] = [1, 5, 25, 50, 75, 95, 99];

/// Chunks processed at once before being merged and released, which bounds
/// how many accumulators exist at one time.
const BATCH_CHUNKS: usize = 32;
/// An accumulator larger than this (a high-cardinality segment column) makes
/// the reduction sequential.
const MAX_ACCUMULATOR_BYTES: usize = 4 * 1024 * 1024;

/// Totals per group of one mapping. `baseline` and `point` are flat,
/// `[group * n_columns + column]`, with the objective in column 0 and the
/// constraints after it in grid order.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupImpacts {
    pub n_columns: usize,
    pub n_quotes: Vec<u64>,
    pub step_change_sum: Vec<i64>,
    pub baseline: Vec<f64>,
    pub point: Vec<f64>,
    /// Sum of each quote's display value at its baseline step.
    pub display_baseline_sum: Vec<f64>,
    /// Sum of each quote's display value at its point step.
    pub display_point_sum: Vec<f64>,
    /// Sum of each quote's absolute display change.
    pub abs_change_sum: Vec<f64>,
    /// The largest absolute display change in the group.
    pub abs_change_max: Vec<f64>,
    /// Quotes whose absolute display change exceeds the band.
    pub n_beyond_band: Vec<u64>,
    /// Quotes at the first step at the point.
    pub n_at_first: Vec<u64>,
    /// Quotes at the last step at the point.
    pub n_at_last: Vec<u64>,
}

/// The display value (a step's human-readable value) at baseline and at the
/// point, and its per-quote change.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplaySummary {
    pub percentiles: [f64; 7],
    pub baseline_mean: f64,
    pub point_mean: f64,
    pub baseline_max: f64,
    pub point_max: f64,
    /// Sum of the display change over quotes whose step went up.
    pub moved_up: f64,
    /// Sum of the display change over quotes whose step went down.
    pub moved_down: f64,
}

/// Quotes that went from one baseline step to one point step.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub from: u32,
    pub to: u32,
    pub quotes: u64,
    /// Summed change in every column, objective first.
    pub changes: Vec<f64>,
}

/// How far quotes moved, in steps.
#[derive(Debug, Clone, PartialEq)]
pub struct StepMovement {
    pub up: u64,
    pub down: u64,
    pub unchanged: u64,
    /// The step changes that occur, ascending.
    pub changes: Vec<i32>,
    /// Quotes at each entry of `changes`.
    pub counts: Vec<u64>,
    pub percentiles: [i32; 7],
}

/// The distribution of one column's per-quote change.
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaSummary {
    pub percentiles: [f64; 7],
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
    pub below: u64,
    pub above: u64,
    /// Sum of the change over quotes whose step went up.
    pub moved_up: f64,
    /// Sum of the change over quotes whose step went down.
    pub moved_down: f64,
}

fn check_steps(steps: &[u32], baseline_steps: &[u32], n_quotes: usize) -> Result<()> {
    if steps.len() != n_quotes || baseline_steps.len() != n_quotes {
        return Err(PriceContourError::DimensionMismatch(format!(
            "{} steps and {} baseline steps for {n_quotes} quotes",
            steps.len(),
            baseline_steps.len()
        )));
    }
    Ok(())
}

/// The 0-based index of the nearest-rank percentile `p` among `n` sorted values:
/// the value at 1-based rank `ceil(p × n / 100)`.
fn rank_index(p: u64, n: usize) -> usize {
    ((p * n as u64).div_ceil(100).max(1) - 1) as usize
}

struct GroupAccumulator {
    n_columns: usize,
    n_quotes: Vec<u64>,
    step_change_sum: Vec<i64>,
    baseline: Vec<f64>,
    point: Vec<f64>,
    display_baseline_sum: Vec<f64>,
    display_point_sum: Vec<f64>,
    abs_change_sum: Vec<f64>,
    abs_change_max: Vec<f64>,
    n_beyond_band: Vec<u64>,
    n_at_first: Vec<u64>,
    n_at_last: Vec<u64>,
}

/// What a group reduction reads for each quote.
struct GroupInputs<'a> {
    n_steps: usize,
    group_of: &'a [u32],
    columns: &'a [&'a [f32]],
    steps: &'a [u32],
    baseline_steps: &'a [u32],
    display: &'a [f64],
    band: Option<f64>,
}

impl GroupAccumulator {
    fn new(n_groups: usize, n_columns: usize) -> Self {
        Self {
            n_columns,
            n_quotes: vec![0; n_groups],
            step_change_sum: vec![0; n_groups],
            baseline: vec![0.0; n_groups * n_columns],
            point: vec![0.0; n_groups * n_columns],
            display_baseline_sum: vec![0.0; n_groups],
            display_point_sum: vec![0.0; n_groups],
            abs_change_sum: vec![0.0; n_groups],
            abs_change_max: vec![0.0; n_groups],
            n_beyond_band: vec![0; n_groups],
            n_at_first: vec![0; n_groups],
            n_at_last: vec![0; n_groups],
        }
    }

    fn add_quotes(&mut self, quotes: std::ops::Range<usize>, inputs: &GroupInputs<'_>) {
        let last = inputs.n_steps - 1;
        for q in quotes {
            let g = inputs.group_of[q] as usize;
            let (step, start) = (inputs.steps[q] as usize, inputs.baseline_steps[q] as usize);
            self.n_quotes[g] += 1;
            self.step_change_sum[g] += step as i64 - start as i64;
            let cell = g * self.n_columns;
            for (c, column) in inputs.columns.iter().enumerate() {
                self.baseline[cell + c] += column[q * inputs.n_steps + start] as f64;
                self.point[cell + c] += column[q * inputs.n_steps + step] as f64;
            }
            let (before, after) = (inputs.display[start], inputs.display[step]);
            let moved = (after - before).abs();
            self.display_baseline_sum[g] += before;
            self.display_point_sum[g] += after;
            self.abs_change_sum[g] += moved;
            self.abs_change_max[g] = self.abs_change_max[g].max(moved);
            self.n_beyond_band[g] += u64::from(inputs.band.is_some_and(|band| moved > band));
            self.n_at_first[g] += u64::from(step == 0);
            self.n_at_last[g] += u64::from(step == last);
        }
    }

    fn merge(&mut self, other: &Self) {
        for (a, b) in self.n_quotes.iter_mut().zip(&other.n_quotes) {
            *a += b;
        }
        for (a, b) in self.step_change_sum.iter_mut().zip(&other.step_change_sum) {
            *a += b;
        }
        for (a, b) in self.baseline.iter_mut().zip(&other.baseline) {
            *a += b;
        }
        for (a, b) in self.point.iter_mut().zip(&other.point) {
            *a += b;
        }
        for (a, b) in self
            .display_baseline_sum
            .iter_mut()
            .zip(&other.display_baseline_sum)
        {
            *a += b;
        }
        for (a, b) in self
            .display_point_sum
            .iter_mut()
            .zip(&other.display_point_sum)
        {
            *a += b;
        }
        for (a, b) in self.abs_change_sum.iter_mut().zip(&other.abs_change_sum) {
            *a += b;
        }
        for (a, b) in self.abs_change_max.iter_mut().zip(&other.abs_change_max) {
            *a = a.max(*b);
        }
        for (a, b) in self.n_beyond_band.iter_mut().zip(&other.n_beyond_band) {
            *a += b;
        }
        for (a, b) in self.n_at_first.iter_mut().zip(&other.n_at_first) {
            *a += b;
        }
        for (a, b) in self.n_at_last.iter_mut().zip(&other.n_at_last) {
            *a += b;
        }
    }
}

/// Per group of `mapping`: quotes, the sum of the step change, the baseline and
/// point totals of every column, and the display statistics. `display` has one
/// value per step; `band` is the absolute display change beyond which a quote
/// counts as past the band.
pub fn group_impacts(
    grid: &QuoteGrid,
    mapping: &GroupMapping,
    steps: &[u32],
    baseline_steps: &[u32],
    display: &[f64],
    band: Option<f64>,
) -> Result<GroupImpacts> {
    let n_quotes = grid.n_quotes;
    check_steps(steps, baseline_steps, n_quotes)?;
    if mapping.group_of.len() != n_quotes {
        return Err(PriceContourError::DimensionMismatch(format!(
            "mapping covers {} quotes but the grid has {n_quotes}",
            mapping.group_of.len()
        )));
    }
    if display.len() != grid.n_steps {
        return Err(PriceContourError::DimensionMismatch(format!(
            "{} display values for {} steps",
            display.len(),
            grid.n_steps
        )));
    }
    let mut columns: Vec<&[f32]> = vec![grid.objective.as_slice()];
    columns.extend(grid.constraints.iter().map(|c| c.as_slice()));
    let n_columns = columns.len();
    let n_groups = mapping.n_groups;
    let inputs = GroupInputs {
        n_steps: grid.n_steps,
        group_of: mapping.group_of.as_slice(),
        columns: &columns,
        steps,
        baseline_steps,
        display,
        band,
    };

    let mut total = GroupAccumulator::new(n_groups, n_columns);
    let accumulator_bytes = n_groups.saturating_mul(72 + 16 * n_columns);
    if accumulator_bytes > MAX_ACCUMULATOR_BYTES {
        total.add_quotes(0..n_quotes, &inputs);
    } else {
        let batch_quotes = BATCH_CHUNKS * RECONSTRUCT_PAR_GRAIN;
        for batch_start in (0..n_quotes).step_by(batch_quotes) {
            let batch_end = (batch_start + batch_quotes).min(n_quotes);
            let starts: Vec<usize> = (batch_start..batch_end)
                .step_by(RECONSTRUCT_PAR_GRAIN)
                .collect();
            let parts: Vec<GroupAccumulator> = starts
                .par_iter()
                .map(|&start| {
                    let end = (start + RECONSTRUCT_PAR_GRAIN).min(batch_end);
                    let mut part = GroupAccumulator::new(n_groups, n_columns);
                    part.add_quotes(start..end, &inputs);
                    part
                })
                .collect();
            for part in &parts {
                total.merge(part);
            }
        }
    }
    Ok(GroupImpacts {
        n_columns,
        n_quotes: total.n_quotes,
        step_change_sum: total.step_change_sum,
        baseline: total.baseline,
        point: total.point,
        display_baseline_sum: total.display_baseline_sum,
        display_point_sum: total.display_point_sum,
        abs_change_sum: total.abs_change_sum,
        abs_change_max: total.abs_change_max,
        n_beyond_band: total.n_beyond_band,
        n_at_first: total.n_at_first,
        n_at_last: total.n_at_last,
    })
}

/// Quotes on each step, in step order.
pub fn step_counts(steps: &[u32], n_steps: usize) -> Vec<u64> {
    let mut counts = vec![0u64; n_steps];
    for &step in steps {
        counts[step as usize] += 1;
    }
    counts
}

/// Each column's total with every quote held at `steps`, objective first.
/// Summed in quote order, so it does not depend on the thread count.
pub fn column_totals(grid: &QuoteGrid, steps: &[u32]) -> Result<Vec<f64>> {
    check_steps(steps, steps, grid.n_quotes)?;
    let mut columns: Vec<&[f32]> = vec![grid.objective.as_slice()];
    columns.extend(grid.constraints.iter().map(|c| c.as_slice()));
    Ok(columns
        .iter()
        .map(|column| {
            steps
                .iter()
                .enumerate()
                .map(|(q, &step)| column[q * grid.n_steps + step as usize] as f64)
                .sum()
        })
        .collect())
}

/// Every (baseline step, point step) pair with quotes, ascending, with the
/// summed change in each column.
pub fn transitions(
    grid: &QuoteGrid,
    steps: &[u32],
    baseline_steps: &[u32],
) -> Result<Vec<Transition>> {
    check_steps(steps, baseline_steps, grid.n_quotes)?;
    let mut columns: Vec<&[f32]> = vec![grid.objective.as_slice()];
    columns.extend(grid.constraints.iter().map(|c| c.as_slice()));
    let n_steps = grid.n_steps;
    let mut cells: HashMap<(u32, u32), (u64, Vec<f64>)> = HashMap::new();
    // Quote order fixes each cell's sum; the map's order is sorted away below.
    for (q, (&step, &start)) in steps.iter().zip(baseline_steps).enumerate() {
        let cell = cells
            .entry((start, step))
            .or_insert_with(|| (0, vec![0.0; columns.len()]));
        cell.0 += 1;
        for (c, column) in columns.iter().enumerate() {
            cell.1[c] += column[q * n_steps + step as usize] as f64
                - column[q * n_steps + start as usize] as f64;
        }
    }
    let mut out: Vec<Transition> = cells
        .into_iter()
        .map(|((from, to), (quotes, changes))| Transition {
            from,
            to,
            quotes,
            changes,
        })
        .collect();
    out.sort_by_key(|t| (t.from, t.to));
    Ok(out)
}

/// How far each quote's step moved from its baseline step.
pub fn step_movement(
    steps: &[u32],
    baseline_steps: &[u32],
    n_steps: usize,
) -> Result<StepMovement> {
    check_steps(steps, baseline_steps, steps.len())?;
    let offset = n_steps as i64 - 1;
    let mut tally = vec![0u64; 2 * n_steps - 1];
    for (&step, &start) in steps.iter().zip(baseline_steps) {
        tally[(step as i64 - start as i64 + offset) as usize] += 1;
    }
    let mut changes = Vec::new();
    let mut counts = Vec::new();
    for (i, &count) in tally.iter().enumerate() {
        if count > 0 {
            changes.push(i as i64 - offset);
            counts.push(count);
        }
    }
    let n = steps.len();
    let mut percentiles = [0i32; 7];
    for (slot, &p) in percentiles.iter_mut().zip(PERCENTILES.iter()) {
        let target = rank_index(p, n) as u64;
        let mut seen = 0u64;
        for (&change, &count) in changes.iter().zip(&counts) {
            seen += count;
            if seen > target {
                *slot = change as i32;
                break;
            }
        }
    }
    let band = |wanted: std::cmp::Ordering| -> u64 {
        changes
            .iter()
            .zip(&counts)
            .filter(|(c, _)| c.cmp(&&0) == wanted)
            .map(|(_, n)| *n)
            .sum()
    };
    Ok(StepMovement {
        up: band(std::cmp::Ordering::Greater),
        down: band(std::cmp::Ordering::Less),
        unchanged: band(std::cmp::Ordering::Equal),
        changes: changes.iter().map(|&c| c as i32).collect(),
        counts,
        percentiles,
    })
}

fn sorted_percentiles(sorted: &[f64]) -> [f64; 7] {
    let mut out = [0.0f64; 7];
    for (slot, &p) in out.iter_mut().zip(PERCENTILES.iter()) {
        *slot = sorted[rank_index(p, sorted.len())];
    }
    out
}

fn sorted_deltas(mut deltas: Vec<f64>) -> Vec<f64> {
    deltas.par_sort_unstable_by(f64::total_cmp);
    deltas
}

/// Bins on `[lo, hi]` over already-sorted values: `(edges, counts, below, above)`.
fn histogram(sorted: &[f64], lo: f64, hi: f64, n_bins: usize) -> (Vec<f64>, Vec<u64>, u64, u64) {
    if lo == hi {
        let below = sorted.iter().filter(|&&v| v < lo).count() as u64;
        let above = sorted.iter().filter(|&&v| v > hi).count() as u64;
        return (
            vec![lo, hi],
            vec![sorted.len() as u64 - below - above],
            below,
            above,
        );
    }
    let span = hi - lo;
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| {
            if i == n_bins {
                hi
            } else {
                lo + span * i as f64 / n_bins as f64
            }
        })
        .collect();
    let mut counts = vec![0u64; n_bins];
    let (mut below, mut above) = (0u64, 0u64);
    for &v in sorted {
        if v < lo {
            below += 1;
        } else if v > hi {
            above += 1;
        } else {
            let mut bin = (((v - lo) / span * n_bins as f64) as usize).min(n_bins - 1);
            while bin > 0 && v < edges[bin] {
                bin -= 1;
            }
            while bin + 1 < n_bins && v >= edges[bin + 1] {
                bin += 1;
            }
            counts[bin] += 1;
        }
    }
    (edges, counts, below, above)
}

/// Percentiles, a histogram and the direction sums of one column's per-quote
/// change. `cells` is laid out `[quote * n_steps + step]`.
pub fn delta_summary(
    cells: &[f32],
    n_steps: usize,
    steps: &[u32],
    baseline_steps: &[u32],
    n_bins: usize,
) -> Result<DeltaSummary> {
    check_steps(steps, baseline_steps, steps.len())?;
    if n_bins == 0 {
        return Err(PriceContourError::InvalidValue("n_bins must be > 0".into()));
    }
    if cells.len() != steps.len() * n_steps {
        return Err(PriceContourError::DimensionMismatch(format!(
            "{} cells for {} quotes of {n_steps} steps",
            cells.len(),
            steps.len()
        )));
    }
    let (mut moved_up, mut moved_down) = (0.0f64, 0.0f64);
    let mut deltas = Vec::with_capacity(steps.len());
    for (q, (&step, &start)) in steps.iter().zip(baseline_steps).enumerate() {
        let delta =
            cells[q * n_steps + step as usize] as f64 - cells[q * n_steps + start as usize] as f64;
        match step.cmp(&start) {
            std::cmp::Ordering::Greater => moved_up += delta,
            std::cmp::Ordering::Less => moved_down += delta,
            std::cmp::Ordering::Equal => {}
        }
        deltas.push(delta);
    }
    let sorted = sorted_deltas(deltas);
    let percentiles = sorted_percentiles(&sorted);
    let (edges, counts, below, above) = histogram(&sorted, percentiles[0], percentiles[6], n_bins);
    Ok(DeltaSummary {
        percentiles,
        edges,
        counts,
        below,
        above,
        moved_up,
        moved_down,
    })
}

/// The display value at baseline and at the point, and the percentiles and
/// direction sums of `display[step] − display[baseline_step]`, taken per quote.
pub fn display_summary(
    display: &[f64],
    steps: &[u32],
    baseline_steps: &[u32],
) -> Result<DisplaySummary> {
    check_steps(steps, baseline_steps, steps.len())?;
    let n = steps.len() as f64;
    let (mut before_sum, mut after_sum) = (0.0f64, 0.0f64);
    let (mut before_max, mut after_max) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let (mut moved_up, mut moved_down) = (0.0f64, 0.0f64);
    let mut deltas = Vec::with_capacity(steps.len());
    for (&step, &start) in steps.iter().zip(baseline_steps) {
        let (before, after) = (display[start as usize], display[step as usize]);
        before_sum += before;
        after_sum += after;
        before_max = before_max.max(before);
        after_max = after_max.max(after);
        let delta = after - before;
        match step.cmp(&start) {
            std::cmp::Ordering::Greater => moved_up += delta,
            std::cmp::Ordering::Less => moved_down += delta,
            std::cmp::Ordering::Equal => {}
        }
        deltas.push(delta);
    }
    Ok(DisplaySummary {
        percentiles: sorted_percentiles(&sorted_deltas(deltas)),
        baseline_mean: before_sum / n,
        point_mean: after_sum / n,
        baseline_max: before_max,
        point_max: after_max,
        moved_up,
        moved_down,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::build_group_mapping;

    /// Cell (q, k): objective 100q + k, volume 10q + k.
    fn indexed_grid(n_quotes: usize, n_steps: usize) -> QuoteGrid {
        let mut objective = Vec::new();
        let mut volume = Vec::new();
        for q in 0..n_quotes {
            for k in 0..n_steps {
                objective.push((100 * q + k) as f32);
                volume.push((10 * q + k) as f32);
            }
        }
        QuoteGrid {
            n_quotes,
            n_steps,
            scenario_values: (0..n_steps).map(|k| 0.5 + k as f32 * 0.5).collect(),
            objective,
            constraints: vec![volume],
            constraint_names: vec!["volume".into()],
            quote_ids: (0..n_quotes).map(|q| format!("Q{q}")).collect(),
            quote_id_fingerprint: 0,
        }
    }

    fn labels(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn group_impacts_by_hand() {
        let grid = indexed_grid(3, 3);
        let mapping = build_group_mapping(&labels(&["a", "b", "a"]));
        let display = [0.0, 1.0, 3.0];
        let got =
            group_impacts(&grid, &mapping, &[2, 0, 1], &[0, 1, 1], &display, Some(2.5)).unwrap();
        assert_eq!(got.n_columns, 2);
        assert_eq!(got.n_quotes, vec![2, 1]);
        assert_eq!(got.step_change_sum, vec![2, -1]);
        // a: quotes 0 and 2. Objective baseline 0 + 201, point 2 + 201.
        assert_eq!(got.baseline, vec![201.0, 21.0, 101.0, 11.0]);
        assert_eq!(got.point, vec![203.0, 23.0, 100.0, 10.0]);
        // a: display 0 -> 3 and 1 -> 1. b: display 1 -> 0.
        assert_eq!(got.display_baseline_sum, vec![1.0, 1.0]);
        assert_eq!(got.display_point_sum, vec![4.0, 0.0]);
        assert_eq!(got.abs_change_sum, vec![3.0, 1.0]);
        assert_eq!(got.abs_change_max, vec![3.0, 1.0]);
        assert_eq!(got.n_beyond_band, vec![1, 0]);
        assert_eq!(got.n_at_first, vec![0, 1]);
        assert_eq!(got.n_at_last, vec![1, 0]);
    }

    #[test]
    fn a_change_exactly_at_the_band_is_not_beyond_it() {
        let grid = indexed_grid(1, 2);
        let mapping = build_group_mapping(&labels(&["a"]));
        let got = group_impacts(&grid, &mapping, &[1], &[0], &[0.0, 1.0], Some(1.0)).unwrap();
        assert_eq!(got.n_beyond_band, vec![0]);
        let none = group_impacts(&grid, &mapping, &[1], &[0], &[0.0, 1.0], None).unwrap();
        assert_eq!(none.n_beyond_band, vec![0]);
    }

    #[test]
    fn step_counts_and_transitions_by_hand() {
        assert_eq!(step_counts(&[0, 2, 2], 3), vec![1, 0, 2]);
        let grid = indexed_grid(3, 3);
        let got = transitions(&grid, &[2, 0, 2], &[0, 1, 0]).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!((got[0].from, got[0].to, got[0].quotes), (0, 2, 2));
        // Each of the two quotes gains 2 in the objective and 2 in volume.
        assert_eq!(got[0].changes, vec![4.0, 4.0]);
        assert_eq!((got[1].from, got[1].to, got[1].quotes), (1, 0, 1));
        assert_eq!(got[1].changes, vec![-1.0, -1.0]);
    }

    #[test]
    fn column_totals_read_the_grid_at_fixed_steps() {
        let grid = indexed_grid(2, 3);
        // Objective 0 + 2 and 100 + 0; volume 0 + 2 and 10 + 0.
        assert_eq!(
            column_totals(&grid, &[2, 0]).unwrap(),
            vec![2.0 + 100.0, 2.0 + 10.0]
        );
        assert!(column_totals(&grid, &[0]).is_err());
    }

    #[test]
    fn group_impacts_do_not_depend_on_threads_or_chunks() {
        let n = RECONSTRUCT_PAR_GRAIN * 40 + 17;
        let mut grid = indexed_grid(n, 3);
        for (i, v) in grid.objective.iter_mut().enumerate() {
            *v = 1.0 / (1.0 + i as f32 * 0.37);
        }
        let names: Vec<String> = (0..n).map(|i| format!("g{}", i % 7)).collect();
        let mapping = build_group_mapping(&names);
        let steps: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
        let starts = vec![1u32; n];
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    group_impacts(
                        &grid,
                        &mapping,
                        &steps,
                        &starts,
                        &[0.25, 1.0, 2.0],
                        Some(0.5),
                    )
                    .unwrap()
                })
        };
        let (one, many) = (run(1), run(8));
        assert_eq!(one, many);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&one.point), bits(&many.point));
        assert_eq!(bits(&one.display_point_sum), bits(&many.display_point_sum));
    }

    #[test]
    fn a_large_group_count_is_reduced_sequentially_to_the_same_totals() {
        // Integer-valued cells sum exactly in any order.
        let n = 300_000;
        let grid = indexed_grid(n, 2);
        let mapping = GroupMapping {
            group_of: (0..n as u32).collect(),
            n_groups: n,
            group_labels: (0..n).map(|i| i.to_string()).collect(),
        };
        let steps = vec![1u32; n];
        let starts = vec![0u32; n];
        let got = group_impacts(&grid, &mapping, &steps, &starts, &[0.0, 1.0], None).unwrap();
        assert_eq!(got.n_quotes, vec![1; n]);
        assert_eq!(got.step_change_sum, vec![1; n]);
        assert_eq!(got.point[2], 101.0);
        assert_eq!(got.baseline[2], 100.0);
    }

    #[test]
    fn group_impacts_reject_mismatched_lengths() {
        let grid = indexed_grid(2, 2);
        let mapping = build_group_mapping(&labels(&["a"]));
        let display = [0.0, 1.0];
        assert!(group_impacts(&grid, &mapping, &[0, 0], &[0, 0], &display, None).is_err());
        let mapping = build_group_mapping(&labels(&["a", "b"]));
        assert!(group_impacts(&grid, &mapping, &[0], &[0, 0], &display, None).is_err());
        assert!(group_impacts(&grid, &mapping, &[0, 0], &[0, 0], &[0.0], None).is_err());
    }

    #[test]
    fn rank_is_nearest_rank_with_no_interpolation() {
        assert_eq!(rank_index(50, 4), 1); // ceil(2) = rank 2
        assert_eq!(rank_index(50, 5), 2); // ceil(2.5) = rank 3
        assert_eq!(rank_index(1, 1), 0);
        assert_eq!(rank_index(99, 1), 0);
        assert_eq!(rank_index(99, 100), 98);
    }

    #[test]
    fn step_movement_by_hand() {
        let steps = [2, 0, 1, 1, 3];
        let starts = [1, 1, 1, 0, 3];
        let got = step_movement(&steps, &starts, 4).unwrap();
        assert_eq!((got.up, got.down, got.unchanged), (2, 1, 2));
        assert_eq!(got.changes, vec![-1, 0, 1]);
        assert_eq!(got.counts, vec![1, 2, 2]);
        // Sorted changes [-1, 0, 0, 1, 1]: p50 is rank 3.
        assert_eq!(got.percentiles[3], 0);
        assert_eq!(got.percentiles[0], -1);
        assert_eq!(got.percentiles[6], 1);
    }

    #[test]
    fn delta_summary_by_hand() {
        // Two steps per quote; the change is cell 1 − cell 0.
        let cells = [0.0f32, 1.0, 0.0, 3.0, 5.0, 4.0, 2.0, 2.0];
        let got = delta_summary(&cells, 2, &[1, 1, 1, 0], &[0, 0, 0, 0], 2).unwrap();
        // Changes 1, 3, -1, 0 (the last quote stays), sorted -1, 0, 1, 3.
        assert_eq!(got.percentiles, [-1.0, -1.0, -1.0, 0.0, 1.0, 3.0, 3.0]);
        assert_eq!(got.moved_up, 1.0 + 3.0 - 1.0);
        assert_eq!(got.moved_down, 0.0);
        assert_eq!(got.edges, vec![-1.0, 1.0, 3.0]);
        assert_eq!(got.counts, vec![2, 2]);
        assert_eq!((got.below, got.above), (0, 0));
    }

    #[test]
    fn delta_summary_splits_by_direction_and_counts_tails() {
        let n = 200;
        let mut cells = Vec::new();
        for q in 0..n {
            cells.push(0.0f32);
            cells.push(q as f32);
        }
        let steps = vec![1u32; n];
        let starts = vec![0u32; n];
        let got = delta_summary(&cells, 2, &steps, &starts, 10).unwrap();
        assert_eq!(got.percentiles[0], 1.0); // rank 2 of 0..199
        assert_eq!(got.percentiles[6], 197.0); // rank 198
        assert_eq!(got.below, 1);
        assert_eq!(got.above, 2);
        assert_eq!(
            got.counts.iter().sum::<u64>() + got.below + got.above,
            n as u64
        );

        let back = delta_summary(&cells, 2, &starts, &steps, 10).unwrap();
        assert_eq!(back.moved_up, 0.0);
        assert_eq!(back.moved_down, -((0..n).sum::<usize>() as f64));
    }

    #[test]
    fn a_constant_distribution_has_one_bin() {
        let cells = [1.0f32, 1.0, 2.0, 2.0];
        let got = delta_summary(&cells, 2, &[1, 1], &[0, 0], 40).unwrap();
        assert_eq!(got.edges, vec![0.0, 0.0]);
        assert_eq!(got.counts, vec![2]);
    }

    #[test]
    fn display_changes_are_taken_per_quote() {
        let display = [0.0, 0.1, 0.5];
        let got = display_summary(&display, &[1, 2, 0], &[0, 1, 0]).unwrap();
        // Changes 0.1, 0.4, 0.0, sorted 0, 0.1, 0.4.
        assert_eq!(got.percentiles[3], 0.1);
        assert_eq!(got.percentiles[6], 0.5 - 0.1);
        assert_eq!(got.baseline_mean, (0.0 + 0.1 + 0.0) / 3.0);
        assert_eq!(got.point_mean, (0.1 + 0.5 + 0.0) / 3.0);
        assert_eq!((got.baseline_max, got.point_max), (0.1, 0.5));
        assert_eq!(got.moved_up, 0.1 + (0.5 - 0.1));
        assert_eq!(got.moved_down, 0.0);
    }
}
