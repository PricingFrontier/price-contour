use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use rayon::prelude::*;

use polars::prelude::{BooleanChunked, Float32Chunked, IntoColumn, NewChunkedArray};
use price_contour_core::{
    evaluate_ratebook, solve_grouped, GroupMapping, GroupedSolveResult, QuoteGrid,
    RatebookEvaluation, SolverConfig,
};

use crate::grid_py::PyQuoteGrid;
use crate::ratebook_helpers_py::PyFactorContext;
use crate::solver_py::{build_result_dataframe, parse_constraints, spec_bounds};
use crate::utils::{order_lambdas, zip_to_dict, OrderedDict};

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
    fn lambdas(&self) -> OrderedDict {
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
    fn total_constraints(&self) -> OrderedDict {
        zip_to_dict(&self.constraint_names, &self.inner.total_constraints)
    }

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.inner.baseline_objective
    }

    #[getter]
    fn baseline_constraints(&self) -> OrderedDict {
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

    let initial_lambdas: Option<Vec<f64>> = lambdas
        .map(|lam_dict| order_lambdas(&lam_dict, &constraint_names))
        .transpose()
        .map_err(PyValueError::new_err)?;

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

/// A ratebook solution evaluated per quote by the canonical kernel
/// (`price_contour_core::evaluate_ratebook`). Returned by
/// `evaluate_ratebook_py` and carried by every `RatebookCDResult`, so solve
/// totals, frontier rows and per-quote results all come from one place.
#[pyclass(name = "RatebookEvaluation")]
pub struct PyRatebookEvaluation {
    inner: RatebookEvaluation,
    grid: Arc<QuoteGrid>,
    baseline_objective: f64,
    baseline_constraints: Vec<f64>,
    quote_results: Option<Py<PyAny>>,
}

impl PyRatebookEvaluation {
    fn evaluate(
        py: Python<'_>,
        grid: Arc<QuoteGrid>,
        mappings: &[Arc<GroupMapping>],
        factor_values: &[Vec<f32>],
    ) -> PyResult<Self> {
        let (inner, (baseline_objective, baseline_constraints)) = py
            .detach(|| {
                let mapping_refs: Vec<&GroupMapping> =
                    mappings.iter().map(|m| m.as_ref()).collect();
                let value_refs: Vec<&[f32]> = factor_values.iter().map(|v| v.as_slice()).collect();
                evaluate_ratebook(&grid, &mapping_refs, &value_refs)
                    .map(|evaluation| (evaluation, grid.baseline_totals()))
            })
            .map_err(|e| PyValueError::new_err(format!("Ratebook evaluation error: {e}")))?;
        Ok(Self {
            inner,
            grid,
            baseline_objective,
            baseline_constraints,
            quote_results: None,
        })
    }
}

#[pymethods]
impl PyRatebookEvaluation {
    #[getter]
    fn total_objective(&self) -> f64 {
        self.inner.total_objective
    }

    #[getter]
    fn total_constraints(&self) -> OrderedDict {
        zip_to_dict(&self.grid.constraint_names, &self.inner.total_constraints)
    }

    #[getter]
    fn baseline_objective(&self) -> f64 {
        self.baseline_objective
    }

    #[getter]
    fn baseline_constraints(&self) -> OrderedDict {
        zip_to_dict(&self.grid.constraint_names, &self.baseline_constraints)
    }

    #[getter]
    fn n_quotes(&self) -> usize {
        self.grid.n_quotes
    }

    #[getter]
    fn n_quotes_clamped_low(&self) -> u64 {
        self.inner.n_clamped_low
    }

    #[getter]
    fn n_quotes_clamped_high(&self) -> u64 {
        self.inner.n_clamped_high
    }

    #[getter]
    fn scenario_values(&self) -> Vec<f32> {
        self.grid.scenario_values.clone()
    }

    #[getter]
    fn baseline_scenario_value(&self) -> f32 {
        self.grid.scenario_values[self.grid.baseline_step()]
    }

    /// Per-quote results in grid quote order: the online result columns
    /// (`quote_id`, `optimal_step`, `optimal_scenario_value`,
    /// `optimal_objective`, `optimal_<c>`) followed by `factor_product`,
    /// `clamped_low` and `clamped_high`. Built on first access and cached.
    #[getter]
    fn quote_results(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(ref cached) = self.quote_results {
            return Ok(cached.clone_ref(py));
        }
        let mut df = build_result_dataframe(&self.inner.optimal_steps, &self.grid)?;
        let extra = [
            Float32Chunked::from_slice("factor_product".into(), &self.inner.factor_product)
                .into_column(),
            BooleanChunked::from_slice("clamped_low".into(), &self.inner.clamped_low).into_column(),
            BooleanChunked::from_slice("clamped_high".into(), &self.inner.clamped_high)
                .into_column(),
        ];
        df.hstack_mut(&extra)
            .map_err(|e| PyValueError::new_err(format!("DataFrame build failed: {e}")))?;
        let py_df: Py<PyAny> = PyDataFrame(df).into_pyobject(py)?.into();
        self.quote_results = Some(py_df.clone_ref(py));
        Ok(py_df)
    }
}

/// Evaluate ratebook factor tables per quote (the canonical kernel).
///
/// `factor_values[f][g]` is the rate of level `g` of `contexts[f]`, in that
/// context's `group_labels` order.
#[pyfunction]
pub fn evaluate_ratebook_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    contexts: Vec<PyRef<'_, PyFactorContext>>,
    factor_values: Vec<Vec<f32>>,
) -> PyResult<PyRatebookEvaluation> {
    let mappings: Vec<Arc<GroupMapping>> =
        contexts.iter().map(|c| Arc::clone(c.mapping())).collect();
    PyRatebookEvaluation::evaluate(py, Arc::clone(&grid.inner), &mappings, &factor_values)
}

/// One inner grouped solve of the CD pass.
struct CdCall {
    cd_iteration: usize,
    factor_index: usize,
    total_objective: f64,
    total_constraints: Vec<f64>,
    lambdas: Vec<f64>,
    clamp_rate: f32,
    iterations: usize,
    converged: bool,
}

/// Result of a full ratebook CD pass run inside Rust. Carries the per-factor
/// optimal factor values, the final λ, one record per inner grouped solve,
/// and the canonical evaluation of the final factor tables (which supplies
/// every reported total).
#[pyclass(name = "RatebookCDResult")]
pub struct PyRatebookCDResult {
    factor_values: Vec<Vec<f32>>,
    lambdas: Vec<f64>,
    constraint_names: Vec<String>,
    constraint_bounds: Vec<f64>,
    cd_iterations: usize,
    converged: bool,
    avg_clamp_rate: f32,
    calls: Vec<CdCall>,
    evaluation: Py<PyRatebookEvaluation>,
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

    /// λ from the last inner grouped solve.
    #[getter]
    fn lambdas(&self) -> OrderedDict {
        zip_to_dict(&self.constraint_names, &self.lambdas)
    }

    /// Absolute bound each constraint was solved against.
    #[getter]
    fn constraint_bounds(&self) -> OrderedDict {
        zip_to_dict(&self.constraint_names, &self.constraint_bounds)
    }

    #[getter]
    fn cd_iterations(&self) -> usize {
        self.cd_iterations
    }

    /// Coordinate-descent convergence: the largest factor-value change over
    /// the last full pass fell below `cd_tolerance`. Not a feasibility check.
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }

    /// Unweighted mean of the inner solves' clamp rates (a search-space
    /// diagnostic; see `docs/DESIGN_DECISIONS.md` §13.11).
    #[getter]
    fn clamp_rate(&self) -> f32 {
        self.avg_clamp_rate
    }

    /// The canonical evaluation of the final factor tables.
    #[getter]
    fn evaluation(&self, py: Python) -> Py<PyRatebookEvaluation> {
        self.evaluation.clone_ref(py)
    }

    /// One dict per inner grouped solve, in (CD pass, factor) order, with
    /// explicit `cd_iteration` (1-based) and `factor_index` fields.
    #[getter]
    fn per_call_records(&self, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        self.calls
            .iter()
            .map(|call| {
                let record = pyo3::types::PyDict::new(py);
                record.set_item("cd_iteration", call.cd_iteration)?;
                record.set_item("factor_index", call.factor_index)?;
                record.set_item("total_objective", call.total_objective)?;
                record.set_item(
                    "total_constraints",
                    zip_to_dict(&self.constraint_names, &call.total_constraints),
                )?;
                record.set_item(
                    "lambdas",
                    zip_to_dict(&self.constraint_names, &call.lambdas),
                )?;
                record.set_item("clamp_rate", call.clamp_rate)?;
                record.set_item("inner_iterations", call.iterations)?;
                record.set_item("inner_converged", call.converged)?;
                Ok(record.into_any().unbind())
            })
            .collect()
    }
}

/// Threshold above which the per-quote residual / multiplier loops drop
/// to rayon parallel iteration. Mirrors the helper-level threshold in
/// `ratebook_helpers_py::PAR_THRESHOLD`; small portfolios stay scalar.
const CD_PAR_THRESHOLD: usize = 100_000;

/// Run a full ratebook CD pass entirely in Rust, then evaluate the final
/// factor tables with the canonical kernel.
///
/// Within the loop:
///
/// * residuals are computed in-place from `overall_mult` and the
///   current per-factor `factor_values` (group-indexed) — no Python
///   round-trip.
/// * `solve_grouped` runs against the existing affine-cache kernel.
/// * `overall_mult` is updated in-place using the new factor values. It is
///   the search's working multiplier only; reported totals come from the
///   final `evaluate_ratebook` call.
/// * `last_lambdas` is threaded as a `Vec<f64>` between calls (no dict
///   round-trip).
///
/// The loop runs inside `py.detach`, so the float buffers stay Rust-side
/// across CD iterations.
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
    if max_cd_iterations == 0 {
        return Err(PyValueError::new_err("max_cd_iterations must be >= 1"));
    }
    if let Some(bad) = candidates.iter().find(|c| !c.is_finite() || **c <= 0.0) {
        return Err(PyValueError::new_err(format!(
            "candidate factor values must be finite and > 0; got {bad}"
        )));
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

    let initial_lambdas: Option<Vec<f64>> = lambdas
        .map(|lam_dict| order_lambdas(&lam_dict, &constraint_names))
        .transpose()
        .map_err(PyValueError::new_err)?;

    let config = SolverConfig {
        max_iter,
        tolerance,
        ..Default::default()
    };

    let grid_arc = Arc::clone(&grid.inner);

    // Persistent buffers for the whole CD pass. Factor values start at 1.0
    // and candidates are validated > 0, so no factor value is ever 0.
    let mut overall_mult = vec![1.0f32; n_quotes];
    let mut factor_values: Vec<Vec<f32>> = group_mappings
        .iter()
        .map(|gm| vec![1.0f32; gm.n_groups])
        .collect();
    let mut residuals_buf = vec![0.0f32; n_quotes];

    let mut last_lambdas: Option<Vec<f64>> = initial_lambdas;
    let mut cd_iter = 0usize;
    let mut cd_converged = false;
    let mut calls: Vec<CdCall> = Vec::with_capacity(max_cd_iterations * n_factors);

    let cd_tolerance_f32 = cd_tolerance as f32;

    let solver_outcome: Result<(), price_contour_core::PriceContourError> = py.detach(|| {
        for iter_idx in 1..=max_cd_iterations {
            cd_iter = iter_idx;
            let mut max_change: f32 = 0.0;

            for (f_idx, gm) in group_mappings.iter().enumerate() {
                let group_of = gm.group_of.as_slice();

                // residuals = overall_mult / factor_values[f_idx][group_of[i]]
                {
                    let old_values = factor_values[f_idx].as_slice();
                    let om = overall_mult.as_slice();
                    let residual = |i: usize| om[i] / old_values[group_of[i] as usize];
                    if n_quotes > CD_PAR_THRESHOLD {
                        residuals_buf
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(i, slot)| *slot = residual(i));
                    } else {
                        for (i, slot) in residuals_buf.iter_mut().enumerate() {
                            *slot = residual(i);
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
                let new_values = result.optimal_factor_values.clone();

                for (&nv, &ov) in new_values.iter().zip(factor_values[f_idx].iter()) {
                    max_change = max_change.max((nv - ov).abs());
                }

                // overall_mult[i] = overall_mult[i] / old_fv * new_fv
                {
                    let old_values = factor_values[f_idx].as_slice();
                    let new_slice = new_values.as_slice();
                    let update = |i: usize, slot: &mut f32| {
                        let g = group_of[i] as usize;
                        *slot = *slot / old_values[g] * new_slice[g];
                    };
                    if n_quotes > CD_PAR_THRESHOLD {
                        overall_mult
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(i, slot)| update(i, slot));
                    } else {
                        for (i, slot) in overall_mult.iter_mut().enumerate() {
                            update(i, slot);
                        }
                    }
                }

                factor_values[f_idx] = new_values;
                last_lambdas = Some(result.lambdas.clone());
                calls.push(CdCall {
                    cd_iteration: iter_idx,
                    factor_index: f_idx,
                    total_objective: result.total_objective,
                    total_constraints: result.total_constraints,
                    lambdas: result.lambdas,
                    clamp_rate: result.clamp_rate,
                    iterations: result.iterations,
                    converged: result.converged,
                });
            }

            if max_change < cd_tolerance_f32 {
                cd_converged = true;
                break;
            }
        }
        Ok(())
    });

    solver_outcome.map_err(|e| PyValueError::new_err(format!("Grouped solver error: {e}")))?;

    let final_lambdas = last_lambdas
        .ok_or_else(|| PyValueError::new_err("CD loop produced no grouped solve results"))?;
    let avg_clamp_rate =
        (calls.iter().map(|c| c.clamp_rate as f64).sum::<f64>() / calls.len() as f64) as f32;

    let evaluation =
        PyRatebookEvaluation::evaluate(py, Arc::clone(&grid_arc), &group_mappings, &factor_values)?;

    Ok(PyRatebookCDResult {
        factor_values,
        lambdas: final_lambdas,
        constraint_bounds: spec_bounds(&specs),
        constraint_names,
        cd_iterations: cd_iter,
        converged: cd_converged,
        avg_clamp_rate,
        calls,
        evaluation: Py::new(py, evaluation)?,
    })
}
