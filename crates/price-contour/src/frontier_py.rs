use std::collections::HashMap;
use std::sync::Arc;

use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use price_contour_core::{
    sweep_frontier, ConstraintSpec, FrontierConfig, FrontierResult, SolverConfig,
};

use crate::constraint_parsing::{direction_for, is_pct_key, validate_constraints_dict};
use crate::grid_py::PyQuoteGrid;
use crate::utils::order_lambdas;

/// Python-visible frontier result.
#[pyclass(name = "FrontierResult")]
pub struct PyFrontierResult {
    inner: FrontierResult,
}

#[pymethods]
impl PyFrontierResult {
    /// Return the frontier as a Polars DataFrame with columns:
    /// threshold_*, total_objective, total_*, lambda_*, iterations, converged
    #[getter]
    fn points(&self, py: Python) -> PyResult<Py<PyAny>> {
        let n_constraints = self.inner.constraint_names.len();

        let mut columns: Vec<Column> = Vec::new();

        // Threshold columns
        for (k, name) in self.inner.constraint_names.iter().enumerate() {
            let vals: Vec<f64> = self.inner.points.iter().map(|p| p.thresholds[k]).collect();
            columns.push(Column::new(format!("threshold_{name}").into(), &vals));
        }

        // Total objective
        let obj_vals: Vec<f64> = self
            .inner
            .points
            .iter()
            .map(|p| p.total_objective)
            .collect();
        columns.push(Column::new("total_objective".into(), &obj_vals));

        // Total constraints
        for k in 0..n_constraints {
            let name = &self.inner.constraint_names[k];
            let vals: Vec<f64> = self
                .inner
                .points
                .iter()
                .map(|p| p.total_constraints[k])
                .collect();
            columns.push(Column::new(format!("total_{name}").into(), &vals));
        }

        // Lambda columns
        for k in 0..n_constraints {
            let name = &self.inner.constraint_names[k];
            let vals: Vec<f64> = self.inner.points.iter().map(|p| p.lambdas[k]).collect();
            columns.push(Column::new(format!("lambda_{name}").into(), &vals));
        }

        // Iterations and converged
        let iter_vals: Vec<i64> = self
            .inner
            .points
            .iter()
            .map(|p| p.iterations as i64)
            .collect();
        columns.push(Column::new("iterations".into(), &iter_vals));

        let conv_vals: Vec<bool> = self.inner.points.iter().map(|p| p.converged).collect();
        columns.push(Column::new("converged".into(), &conv_vals));

        // Scenario value distribution stats — helper macro avoids complex type
        macro_rules! sv_col {
            ($name:expr, $field:ident) => {
                let vals: Vec<f64> = self
                    .inner
                    .points
                    .iter()
                    .map(|p| p.sv_stats.$field)
                    .collect();
                columns.push(Column::new($name.into(), &vals));
            };
        }
        sv_col!("sv_mean", mean);
        sv_col!("sv_std", std);
        sv_col!("sv_min", min);
        sv_col!("sv_p5", p5);
        sv_col!("sv_p25", p25);
        sv_col!("sv_median", median);
        sv_col!("sv_p75", p75);
        sv_col!("sv_p95", p95);
        sv_col!("sv_max", max);
        sv_col!("sv_pct_increase", pct_increase);
        sv_col!("sv_pct_decrease", pct_decrease);

        let df = DataFrame::new(columns)
            .map_err(|e| PyValueError::new_err(format!("DataFrame build failed: {e}")))?;
        let py_df: Py<PyAny> = PyDataFrame(df).into_pyobject(py)?.into();
        Ok(py_df)
    }

    #[getter]
    fn n_points(&self) -> usize {
        self.inner.points.len()
    }

    #[getter]
    fn n_converged(&self) -> usize {
        self.inner.n_converged
    }

    #[getter]
    fn constraint_names(&self) -> Vec<String> {
        self.inner.constraint_names.clone()
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // PyO3 binding: each arg is a Python keyword argument
#[pyo3(signature = (
    grid,
    constraints,
    threshold_ranges,
    n_points_per_dim = 10,
    max_iter = 50,
    tolerance = 1e-5,
    initial_lambdas = None,
    max_total_points = 10_000,
    parallel = false,
))]
pub fn sweep_frontier_py(
    py: Python<'_>,
    grid: &PyQuoteGrid,
    constraints: HashMap<String, HashMap<String, Option<f64>>>,
    threshold_ranges: HashMap<String, (f64, f64)>,
    n_points_per_dim: usize,
    max_iter: usize,
    tolerance: f64,
    initial_lambdas: Option<HashMap<String, f64>>,
    max_total_points: usize,
    parallel: bool,
) -> PyResult<PyFrontierResult> {
    let (_, baseline_totals) = grid.inner.baseline_totals();

    // Shared validation: migration errors, multi-key, NaN/inf, unknown
    // names. ``None`` thresholds are accepted here — the sweep supplies
    // the value per point, so the inner threshold is irrelevant and the
    // ``threshold_ranges`` entry below is what matters. Mirrors the
    // solver path so the two entry points cannot drift on error wording.
    validate_constraints_dict(&constraints, &grid.inner)?;

    // Frontier-specific extra check (D1): a ``None``-threshold
    // constraint MUST have a matching ``threshold_ranges`` entry — its
    // numeric value is supplied by the sweep, so the marker is
    // meaningless without a range. A numeric-threshold constraint is
    // allowed to omit its range; the Python wrapper redirects such
    // calls into the orchestrated path where the constructor value is
    // held fixed across every frontier point. Once we reach this Rust
    // path we still enforce that every named constraint has a range —
    // the caller is the all-swept fast path; mixed swept/unswept
    // dispatches don't reach here.
    let mut has_unswept_numeric = false;
    for (name, spec) in constraints.iter() {
        if threshold_ranges.contains_key(name) {
            continue;
        }
        let is_none_threshold = spec
            .values()
            .any(|v: &Option<f64>| v.is_none());
        if is_none_threshold {
            return Err(PyValueError::new_err(format!(
                "No threshold_range for constraint '{}'",
                name
            )));
        }
        has_unswept_numeric = true;
    }
    if has_unswept_numeric {
        // Defensive: the Python ``OnlineOptimiser.frontier`` wrapper
        // detects unswept numeric axes and routes the call to the
        // Python orchestrator (which holds the constructor value
        // fixed). Reaching the Rust fast path with unswept axes means
        // a direct caller bypassed the wrapper; surface a clear error
        // rather than silently dropping the unswept axis.
        return Err(PyValueError::new_err(
            "Rust frontier fast path requires every constraint to have \
             a threshold_ranges entry; unswept numeric thresholds must \
             be dispatched via OnlineOptimiser.frontier (Python wrapper) \
             which holds the constructor value fixed across the sweep."
                .to_string(),
        ));
    }
    if threshold_ranges.is_empty() && !constraints.is_empty() {
        // Unreachable for normal flows (the unswept-numeric loop above
        // would already have errored), but defensively reject the
        // zero-axes-swept case so the message style stays consistent.
        return Err(PyValueError::new_err(
            "No threshold_range entries supplied — frontier requires \
             at least one threshold_ranges entry"
                .to_string(),
        ));
    }

    // Walk grid order (NOT the user HashMap) so specs_template[k]
    // aligns with grid.constraints[k] — the inner solver indexes by
    // position. HashMap iteration order would be nondeterministic and
    // silently mis-pair lambdas with constraint names.
    //
    // Extension point: ratio constraints (C1) detect via `numerator` /
    // `denominator` keys before the direction-key match below.
    let mut specs_template = Vec::with_capacity(constraints.len());
    let mut ranges = Vec::with_capacity(constraints.len());
    let mut ordered_names: Vec<String> = Vec::with_capacity(constraints.len());
    // Per-output-spec scale factor used when reporting thresholds back to
    // Python. The reported ``threshold_<name>`` column must match the
    // units of the user-supplied ``threshold_ranges`` entry verbatim:
    // * ``min`` / ``max``         → absolute units, scale = 1.
    // * ``min_pct`` / ``max_pct`` → fractions of baseline; the inner
    //   solver scaled them up by baseline, so divide back down by
    //   baseline before reporting. Applies whether the constraint's
    //   threshold is numeric or ``None`` — the reporting unit follows
    //   the key, not the threshold value.
    let mut report_scales: Vec<f64> = Vec::with_capacity(constraints.len());

    for (constraint_idx, name) in grid.inner.constraint_names.iter().enumerate() {
        let Some(spec_dict) = constraints.get(name) else {
            continue;
        };
        ordered_names.push(name.clone());

        specs_template.push(ConstraintSpec {
            name: name.clone(),
            direction: direction_for(spec_dict),
            threshold: 0.0, // replaced per frontier point
        });

        let (lo, hi) = threshold_ranges[name];

        // Threshold ranges follow the constraint key:
        // * ``min`` / ``max``         → already absolute, pass through.
        // * ``min_pct`` / ``max_pct`` → fractions of baseline; scale up.
        let pct = is_pct_key(spec_dict);
        let (abs_lo, abs_hi) = if pct {
            (
                baseline_totals[constraint_idx] * lo,
                baseline_totals[constraint_idx] * hi,
            )
        } else {
            (lo, hi)
        };

        ranges.push((abs_lo, abs_hi));

        // For pct constraints, the inner solver received absolute
        // thresholds (frac × baseline). Scale the recorded threshold
        // back to the user-supplied fraction so the reported column
        // matches the user's input units verbatim. Absolute (``min`` /
        // ``max``) constraints already match user units and need no
        // rescaling.
        let baseline = baseline_totals[constraint_idx];
        let scale = if pct && baseline != 0.0 {
            1.0 / baseline
        } else {
            1.0
        };
        report_scales.push(scale);
    }

    let frontier_config = FrontierConfig {
        n_points_per_dim,
        threshold_ranges: ranges,
        max_total_points: Some(max_total_points),
        parallel,
    };

    let solver_config = SolverConfig {
        max_iter,
        tolerance,
        ..Default::default()
    };

    // Convert initial_lambdas dict to Vec<f64> in specs_template order
    let initial_lambda_vec: Option<Vec<f64>> =
        initial_lambdas.map(|lam_dict| order_lambdas(&lam_dict, &ordered_names));

    let grid_arc = Arc::clone(&grid.inner);
    let mut result = py
        .detach(|| {
            sweep_frontier(
                &grid_arc,
                &specs_template,
                &frontier_config,
                &solver_config,
                initial_lambda_vec.as_deref(),
            )
        })
        .map_err(|e| PyValueError::new_err(format!("Frontier error: {e}")))?;

    // Rescale recorded thresholds for ``_pct`` constraints back to the
    // user-supplied fraction units. The inner solver operates on
    // absolute thresholds; the reported column must match the units of
    // ``threshold_ranges`` verbatim, so divide by baseline for any
    // ``min_pct`` / ``max_pct`` axis (numeric or ``None``).
    if report_scales.iter().any(|&s| s != 1.0) {
        for point in result.points.iter_mut() {
            for (k, t) in point.thresholds.iter_mut().enumerate() {
                *t *= report_scales[k];
            }
        }
    }

    Ok(PyFrontierResult { inner: result })
}
