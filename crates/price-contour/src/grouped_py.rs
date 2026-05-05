use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use rayon::prelude::*;

use price_contour_core::{
    solve_grouped, GroupMapping, GroupedSolveResult, QuoteGrid, SolverConfig,
};

use crate::grid_py::PyQuoteGrid;
use crate::ratebook_helpers_py::PyFactorContext;
use crate::solver_py::{build_result_dataframe, parse_constraints};
use crate::utils::{order_lambdas, zip_to_dict};

/// Python-visible grouped solve result.
#[pyclass(name = "GroupedSolveResult")]
pub struct PyGroupedSolveResult {
    inner: GroupedSolveResult,
    grid: Arc<QuoteGrid>,
    constraint_names: Vec<String>,
    group_labels: Vec<String>,
    result_df: Option<Py<PyAny>>,
}

#[pymethods]
impl PyGroupedSolveResult {
    #[getter]
    fn optimal_factor_values(&self) -> HashMap<String, f32> {
        self.group_labels
            .iter()
            .zip(self.inner.optimal_factor_values.iter())
            .map(|(label, &val)| (label.clone(), val))
            .collect()
    }

    /// Per-group optimal factor values as a flat `Vec<f32>` indexed by
    /// group index (matches the order of `group_labels`). The hot ratebook
    /// CD orchestrator uses this getter to skip the
    /// `HashMap<String, f32>` allocation + Python dict materialisation
    /// that `optimal_factor_values` does on every call.
    #[getter]
    fn optimal_factor_values_by_group(&self) -> Vec<f32> {
        self.inner.optimal_factor_values.clone()
    }

    #[getter]
    fn optimal_steps_per_quote(&self) -> Vec<u32> {
        self.inner.optimal_steps_per_quote.clone()
    }

    #[getter]
    fn lambdas(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.lambdas)
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn total_objective(&self) -> f64 {
        self.inner.total_objective
    }

    #[getter]
    fn total_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.total_constraints)
    }

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.inner.baseline_objective
    }

    #[getter]
    fn baseline_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.inner.baseline_constraints)
    }

    #[getter]
    fn clamp_rate(&self) -> f32 {
        self.inner.clamp_rate
    }

    #[getter]
    fn group_labels(&self) -> Vec<String> {
        self.group_labels.clone()
    }

    #[getter]
    fn history(&self, py: Python) -> Option<Vec<HashMap<String, Py<PyAny>>>> {
        self.inner.history.as_ref().map(|h| {
            h.records
                .iter()
                .map(|rec| {
                    let mut d: HashMap<String, Py<PyAny>> = HashMap::new();
                    d.insert(
                        "iteration".into(),
                        rec.iteration.into_pyobject(py).unwrap().unbind().into(),
                    );
                    d.insert(
                        "total_objective".into(),
                        rec.total_objective
                            .into_pyobject(py)
                            .unwrap()
                            .unbind()
                            .into(),
                    );
                    d.insert(
                        "max_lambda_change".into(),
                        rec.max_lambda_change
                            .into_pyobject(py)
                            .unwrap()
                            .unbind()
                            .into(),
                    );
                    d.insert(
                        "all_constraints_satisfied".into(),
                        rec.all_constraints_satisfied
                            .into_pyobject(py)
                            .unwrap()
                            .to_owned()
                            .unbind()
                            .into(),
                    );

                    let lam_dict = zip_to_dict(&self.constraint_names, &rec.lambdas);
                    d.insert(
                        "lambdas".into(),
                        lam_dict.into_pyobject(py).unwrap().unbind().into(),
                    );

                    let con_dict = zip_to_dict(&self.constraint_names, &rec.total_constraints);
                    d.insert(
                        "total_constraints".into(),
                        con_dict.into_pyobject(py).unwrap().unbind().into(),
                    );

                    d
                })
                .collect()
        })
    }

    #[getter]
    fn dataframe(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(ref cached) = self.result_df {
            return Ok(cached.clone_ref(py));
        }
        let df = build_result_dataframe(&self.inner.optimal_steps_per_quote, &self.grid)?;
        let py_df = PyDataFrame(df).into_pyobject(py)?.into();
        self.result_df = Some(py_df);
        Ok(self.result_df.as_ref().unwrap().clone_ref(py))
    }
}

#[pyfunction]
#[pyo3(signature = (
    grid,
    context,
    residuals,
    candidates,
    constraints = None,
    max_iter = 50,
    tolerance = 1e-5,
    lambdas = None,
    record_history = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn solve_grouped_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    context: &PyFactorContext,
    residuals: Vec<f32>,
    candidates: Vec<f32>,
    constraints: Option<HashMap<String, HashMap<String, Option<f64>>>>,
    max_iter: usize,
    tolerance: f64,
    lambdas: Option<HashMap<String, f64>>,
    record_history: bool,
) -> PyResult<PyGroupedSolveResult> {
    let constraints = constraints.unwrap_or_default();
    let group_mapping_arc = Arc::clone(context.mapping());

    if group_mapping_arc.group_of.len() != grid.inner.n_quotes {
        return Err(PyValueError::new_err(format!(
            "context n_quotes {} != grid n_quotes {}",
            group_mapping_arc.group_of.len(),
            grid.inner.n_quotes
        )));
    }
    if residuals.len() != grid.inner.n_quotes {
        return Err(PyValueError::new_err(format!(
            "residuals length {} != n_quotes {}",
            residuals.len(),
            grid.inner.n_quotes
        )));
    }

    let specs = parse_constraints(constraints, &grid.inner)?;

    let config = SolverConfig {
        max_iter,
        tolerance,
        record_history,
        ..Default::default()
    };

    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    let initial_lambdas: Option<Vec<f64>> =
        lambdas.map(|lam_dict| order_lambdas(&lam_dict, &constraint_names));

    let result_group_labels = group_mapping_arc.group_labels.clone();

    let grid_arc = Arc::clone(&grid.inner);
    let result = py
        .detach(|| {
            solve_grouped(
                &grid_arc,
                &group_mapping_arc,
                &residuals,
                &candidates,
                &specs,
                &config,
                initial_lambdas.as_deref(),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("Grouped solver error: {e}")))?;

    Ok(PyGroupedSolveResult {
        inner: result,
        grid: grid_arc,
        constraint_names,
        group_labels: result_group_labels,
        result_df: None,
    })
}

/// Result of a full ratebook CD pass run inside Rust. Carries the per-
/// factor optimal factor values plus the aggregate metrics the Python
/// orchestrator needs to assemble a `RatebookResult`. Also retains the
/// last grouped solve's `optimal_steps_per_quote` so the orchestrator
/// can build the per-quote results DataFrame for ratio reporting.
#[pyclass(name = "RatebookCDResult")]
pub struct PyRatebookCDResult {
    factor_values: Vec<Vec<f32>>,
    lambdas: Vec<f64>,
    constraint_names: Vec<String>,
    total_objective: f64,
    total_constraints: Vec<f64>,
    baseline_objective: f64,
    baseline_constraints: Vec<f64>,
    cd_iterations: usize,
    converged: bool,
    avg_clamp_rate: f32,
    grid: Arc<QuoteGrid>,
    optimal_steps_per_quote: Vec<u32>,
    result_df: Option<Py<PyAny>>,
    /// Per-(cd_iter × factor) `total_objective` values, one per inner
    /// `solve_grouped` call. Preserved so the orchestrator can rebuild
    /// lightweight `per_factor_results` records for backwards-
    /// compatibility with code that inspects per-call convergence.
    per_call_total_objectives: Vec<f64>,
    /// Per-(cd_iter × factor) λ vectors, ordered by constraint index
    /// (same order as `constraint_names`).
    per_call_lambdas: Vec<Vec<f64>>,
}

#[pymethods]
impl PyRatebookCDResult {
    /// Per-factor `Vec<f32>` of optimal factor values, indexed by
    /// `(factor_idx, group_idx)`. Caller stitches back to the
    /// `dict[str, dict[str, float]]` shape using each context's
    /// `group_labels`.
    #[getter]
    fn factor_values(&self) -> Vec<Vec<f32>> {
        self.factor_values.clone()
    }

    #[getter]
    fn lambdas(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.lambdas)
    }

    #[getter]
    fn total_objective(&self) -> f64 {
        self.total_objective
    }

    #[getter]
    fn total_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.total_constraints)
    }

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.baseline_objective
    }

    #[getter]
    fn baseline_constraints(&self) -> HashMap<String, f64> {
        zip_to_dict(&self.constraint_names, &self.baseline_constraints)
    }

    #[getter]
    fn cd_iterations(&self) -> usize {
        self.cd_iterations
    }

    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    #[getter]
    fn clamp_rate(&self) -> f32 {
        self.avg_clamp_rate
    }

    #[getter]
    fn optimal_steps_per_quote(&self) -> Vec<u32> {
        self.optimal_steps_per_quote.clone()
    }

    /// Per-call total_objective in solve order (`(cd_iter, factor)`
    /// loop). Length = `cd_iterations × n_factors` (or earlier if CD
    /// terminated). Used by the orchestrator to populate
    /// `RatebookResult.per_factor_results` with lightweight
    /// monotonicity-checking objects.
    #[getter]
    fn per_call_total_objectives(&self) -> Vec<f64> {
        self.per_call_total_objectives.clone()
    }

    /// Per-call λ vectors in solve order. Each inner `Vec<f64>` is
    /// ordered to match `lambdas`'s constraint-name order. Length =
    /// number of grouped solves run during the CD pass.
    #[getter]
    fn per_call_lambdas(&self) -> Vec<HashMap<String, f64>> {
        self.per_call_lambdas
            .iter()
            .map(|lam| zip_to_dict(&self.constraint_names, lam))
            .collect()
    }

    /// Per-quote results DataFrame for the last grouped solve in the
    /// CD pass — needed by the ratebook orchestrator's ratio-reporting
    /// path (`_stitch_optimal_ratio_columns`). Built lazily on first
    /// access and cached.
    #[getter]
    fn dataframe(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(ref cached) = self.result_df {
            return Ok(cached.clone_ref(py));
        }
        let df = build_result_dataframe(&self.optimal_steps_per_quote, &self.grid)?;
        let py_df: Py<PyAny> = PyDataFrame(df).into_pyobject(py)?.into();
        let cloned = py_df.clone_ref(py);
        self.result_df = Some(py_df);
        Ok(cloned)
    }
}

/// Threshold above which the per-quote residual / multiplier loops drop
/// to rayon parallel iteration. Mirrors the helper-level threshold in
/// `ratebook_helpers_py::PAR_THRESHOLD`; small portfolios stay scalar.
const CD_PAR_THRESHOLD: usize = 100_000;

/// Run a full ratebook CD pass entirely in Rust.
///
/// Replaces the Python `for cd_iter: for f_idx: compute_residuals_py +
/// solve_grouped_py + update_multipliers_py + bookkeeping` loop with a
/// single PyO3 entry. Within the loop:
///
/// * residuals are computed in-place from `overall_mult` and the
///   current per-factor `factor_values` (group-indexed) — no Python
///   round-trip.
/// * `solve_grouped` runs against the existing affine-cache kernel.
/// * `overall_mult` is updated in-place using the new factor values.
/// * `last_lambdas` is threaded as a `Vec<f64>` between calls (no dict
///   round-trip).
///
/// The entire body runs inside `py.detach`, so 100k-element float
/// buffers stay Rust-side across CD iterations.
#[pyfunction]
#[pyo3(signature = (
    grid,
    contexts,
    candidates,
    constraints = None,
    max_iter = 50,
    tolerance = 1e-5,
    max_cd_iterations = 3,
    cd_tolerance = 1e-3,
    lambdas = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn run_cd_pass_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    contexts: Vec<PyRef<'_, PyFactorContext>>,
    candidates: Vec<f32>,
    constraints: Option<HashMap<String, HashMap<String, Option<f64>>>>,
    max_iter: usize,
    tolerance: f64,
    max_cd_iterations: usize,
    cd_tolerance: f64,
    lambdas: Option<HashMap<String, f64>>,
) -> PyResult<PyRatebookCDResult> {
    let constraints = constraints.unwrap_or_default();
    let n_factors = contexts.len();
    if n_factors == 0 {
        return Err(PyValueError::new_err("contexts must not be empty"));
    }

    let n_quotes = grid.inner.n_quotes;
    let group_mappings: Vec<Arc<GroupMapping>> =
        contexts.iter().map(|c| Arc::clone(c.mapping())).collect();
    for (f_idx, gm) in group_mappings.iter().enumerate() {
        if gm.group_of.len() != n_quotes {
            return Err(PyValueError::new_err(format!(
                "context[{f_idx}] n_quotes {} != grid n_quotes {}",
                gm.group_of.len(),
                n_quotes
            )));
        }
    }

    let specs = parse_constraints(constraints, &grid.inner)?;
    let constraint_names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();

    let initial_lambdas: Option<Vec<f64>> =
        lambdas.map(|lam_dict| order_lambdas(&lam_dict, &constraint_names));

    let config = SolverConfig {
        max_iter,
        tolerance,
        ..Default::default()
    };

    let grid_arc = Arc::clone(&grid.inner);

    // Persistent buffers for the whole CD pass. `overall_mult` and
    // `factor_values` mirror the Python orchestrator's mutable state;
    // `residuals_buf` is the per-factor scratch passed into
    // `solve_grouped`.
    let mut overall_mult = vec![1.0f32; n_quotes];
    let mut factor_values: Vec<Vec<f32>> = group_mappings
        .iter()
        .map(|gm| vec![1.0f32; gm.n_groups])
        .collect();
    let mut residuals_buf = vec![0.0f32; n_quotes];

    let mut last_lambdas: Option<Vec<f64>> = initial_lambdas;
    let mut cd_iter = 0usize;
    let mut cd_converged = false;
    let mut last_result: Option<GroupedSolveResult> = None;
    let mut clamp_sum = 0.0f64;
    let mut clamp_count = 0u64;
    let mut per_call_total_objectives: Vec<f64> = Vec::new();
    let mut per_call_lambdas: Vec<Vec<f64>> = Vec::new();

    let cd_tolerance_f32 = cd_tolerance as f32;

    let solver_outcome: Result<(), price_contour_core::PriceContourError> = py.detach(|| {
        for iter_idx in 1..=max_cd_iterations {
            cd_iter = iter_idx;
            let mut max_change: f32 = 0.0;

            for (f_idx, gm) in group_mappings.iter().enumerate() {
                let group_of = gm.group_of.as_slice();
                let n_groups_f = gm.n_groups;

                // residuals = overall_mult / factor_values[f_idx][group_of[i]]
                {
                    let old_values = factor_values[f_idx].as_slice();
                    let om = overall_mult.as_slice();
                    if n_quotes > CD_PAR_THRESHOLD {
                        residuals_buf
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(i, slot)| {
                                let g = group_of[i] as usize;
                                let fv = old_values[g];
                                *slot = if fv != 0.0 { om[i] / fv } else { 1.0 };
                            });
                    } else {
                        for (i, slot) in residuals_buf.iter_mut().enumerate() {
                            let g = group_of[i] as usize;
                            let fv = old_values[g];
                            *slot = if fv != 0.0 { om[i] / fv } else { 1.0 };
                        }
                    }
                }

                // Run the inner Lagrangian solve.
                let result = solve_grouped(
                    &grid_arc,
                    gm,
                    &residuals_buf,
                    &candidates,
                    &specs,
                    &config,
                    last_lambdas.as_deref(),
                )?;

                // New factor values per group. Mirror the Python rule:
                // a 0.0 from solve_grouped means "no active candidate
                // for this group" and we keep the prior value.
                let new_values: Vec<f32> = (0..n_groups_f)
                    .map(|g| {
                        let nv = result.optimal_factor_values[g];
                        if nv != 0.0 {
                            nv
                        } else {
                            factor_values[f_idx][g]
                        }
                    })
                    .collect();

                // Track max_change across all groups in this factor.
                for (g, &nv) in new_values.iter().enumerate() {
                    let change = (nv - factor_values[f_idx][g]).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }

                // Update overall_mult in place: new_mult[i] =
                // overall_mult[i] / old_fv * new_fv (or new_fv when
                // old_fv == 0).
                {
                    let old_values = factor_values[f_idx].as_slice();
                    let new_slice = new_values.as_slice();
                    if n_quotes > CD_PAR_THRESHOLD {
                        overall_mult
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(i, slot)| {
                                let g = group_of[i] as usize;
                                let fv_old = old_values[g];
                                let fv_new = new_slice[g];
                                *slot = if fv_old != 0.0 {
                                    *slot / fv_old * fv_new
                                } else {
                                    fv_new
                                };
                            });
                    } else {
                        for (i, slot) in overall_mult.iter_mut().enumerate() {
                            let g = group_of[i] as usize;
                            let fv_old = old_values[g];
                            let fv_new = new_slice[g];
                            *slot = if fv_old != 0.0 {
                                *slot / fv_old * fv_new
                            } else {
                                fv_new
                            };
                        }
                    }
                }

                factor_values[f_idx] = new_values;
                per_call_total_objectives.push(result.total_objective);
                per_call_lambdas.push(result.lambdas.clone());
                last_lambdas = Some(result.lambdas.clone());
                clamp_sum += result.clamp_rate as f64;
                clamp_count += 1;
                last_result = Some(result);
            }

            if max_change < cd_tolerance_f32 {
                cd_converged = true;
                break;
            }
        }
        Ok(())
    });

    solver_outcome.map_err(|e| PyValueError::new_err(format!("Grouped solver error: {e}")))?;

    let last = last_result
        .ok_or_else(|| PyValueError::new_err("CD loop produced no grouped solve results"))?;

    let avg_clamp_rate = if clamp_count > 0 {
        (clamp_sum / clamp_count as f64) as f32
    } else {
        0.0
    };

    Ok(PyRatebookCDResult {
        factor_values,
        lambdas: last.lambdas,
        constraint_names,
        total_objective: last.total_objective,
        total_constraints: last.total_constraints,
        baseline_objective: last.baseline_objective,
        baseline_constraints: last.baseline_constraints,
        cd_iterations: cd_iter,
        converged: cd_converged,
        avg_clamp_rate,
        grid: grid_arc,
        optimal_steps_per_quote: last.optimal_steps_per_quote,
        result_df: None,
        per_call_total_objectives,
        per_call_lambdas,
    })
}
