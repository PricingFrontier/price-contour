# price-contour: Frontier Improvements for Haute Integration

These four changes address friction points discovered while building Haute's efficient frontier workflow. Each is backward-compatible and independent — they can be implemented in any order.

---

## 1. Frontier warm-start from prior solve

### Problem

When Haute runs the frontier, it calls `solver.solve()` first (producing optimal lambdas), then immediately calls `solver.frontier()` on the same `QuoteGrid`. The frontier's internal sweep (`sweep_frontier` in `frontier.rs`) warm-starts each point from the *previous* point's lambdas — but the very first point in the NN ordering always cold-starts from `None` (zero lambdas).

This throws away the lambdas from the solve that just ran. The first frontier point starts from scratch, typically needing 50-150 iterations to converge, when it could converge in 1-5 iterations if warm-started from the nearby solve result.

### Current code

`frontier.rs`, line 159:
```rust
let mut prev_lambdas: Option<Vec<f64>> = None;  // always starts cold
```

`solver.py`, `OnlineOptimiser.frontier()`:
```python
def frontier(self, df_or_grid, *, threshold_ranges, n_points_per_dim=10):
    # No initial_lambdas parameter
```

### Proposed change

**Rust (`frontier.rs`):** Accept `initial_lambdas` in the frontier config and use it as the starting point:

```rust
pub fn sweep_frontier(
    grid: &QuoteGrid,
    constraint_specs: &[ConstraintSpec],
    solver_config: &SolverConfig,
    threshold_ranges: &[(f64, f64)],
    n_points_per_dim: usize,
    initial_lambdas: Option<&[f64]>,  // NEW
) -> Result<FrontierResult, SolverError> {
    // ...
    let mut prev_lambdas: Option<Vec<f64>> = initial_lambdas.map(|l| l.to_vec());
    // rest unchanged
}
```

**PyO3 binding (`frontier_py.rs`):** Add `initial_lambdas` parameter:

```rust
#[pyo3(signature = (
    grid, constraints, threshold_ranges, n_points_per_dim = 10,
    max_iter = 50, chunk_size = 500_000, tolerance = 1e-6,
    initial_lambdas = None,  // NEW
))]
pub fn sweep_frontier_py(
    // ...existing params...
    initial_lambdas: Option<HashMap<String, f64>>,
) -> PyResult<PyFrontierResult> {
```

Convert the HashMap to a Vec<f64> in constraint order (same pattern as `solve_online_py` handles its `lambdas` parameter).

**Python API (`solver.py`):** Thread the parameter through:

```python
def frontier(
    self,
    df_or_grid: pl.DataFrame | QuoteGrid,
    *,
    threshold_ranges: dict[str, tuple[float, float]],
    n_points_per_dim: int = 10,
    initial_lambdas: dict[str, float] | None = None,  # NEW
) -> FrontierResult:
```

### Haute-side usage

Two-line change in `_optimiser_service.py`:

```python
frontier_result = solver.frontier(
    quote_grid,
    threshold_ranges=ranges,
    n_points_per_dim=frontier_steps,
    initial_lambdas=solve_result.lambdas,  # warm-start from solve
)
```

### Effort

Small. One new parameter threaded through three layers. Fully backward-compatible (defaults to `None`).

---

## 2. Single-pass apply on a QuoteGrid (`apply_from_grid`)

### Problem

When a user selects a frontier point in Haute's UI, the backend needs to re-evaluate the solver at that point's lambdas to get per-quote results (optimal scenario values, objective/constraint breakdowns). Currently Haute calls `solver.solve(quote_grid, lambdas=selected_lambdas)`, which runs the full iterative solver (up to `max_iter` iterations) even though the lambdas are already known and fixed.

This is wasteful. The Lagrangian argmax with fixed lambdas is a single O(N) forward pass — no iteration needed. The Rust core already has `apply_lambdas()` in `solver/apply.rs` that does exactly this on a `QuoteGrid`. But the Python `ApplyOptimiser.apply()` method only accepts a `pl.DataFrame` — it re-ingests data from scratch rather than reusing the in-memory `QuoteGrid`.

### Current code

`solver/apply.rs` has `apply_lambdas(grid, specs, lambdas, chunk_size)` — a single-pass function that takes a `QuoteGrid` reference.

`apply_py.rs` has `ApplyOptimiser.apply()` which builds a new grid from a DataFrame input.

There is no Python-accessible way to call `apply_lambdas` on an existing `QuoteGrid`.

### Proposed change

**PyO3 binding — new function in `apply_py.rs` (or a new file):**

```rust
#[pyfunction]
#[pyo3(signature = (grid, lambdas, constraints, chunk_size = 500_000))]
pub fn apply_from_grid_py(
    grid: &PyQuoteGrid,
    lambdas: HashMap<String, f64>,
    constraints: HashMap<String, HashMap<String, f64>>,
    chunk_size: usize,
) -> PyResult<PyApplyResult> {
    // 1. Parse constraints into ConstraintSpecs (reuse parse_constraints from solver_py.rs)
    // 2. Convert lambdas HashMap to Vec<f64> in constraint order
    // 3. Call apply_lambdas(&grid.inner, &specs, &lambda_vec, Some(chunk_size))
    // 4. Wrap result in PyApplyResult (with grid Arc for the dataframe builder)
}
```

Register in `lib.rs`:
```rust
m.add_function(wrap_pyfunction!(apply_from_grid_py, m)?)?;
```

**Python API — expose in `apply.py` or `__init__.py`:**

```python
from price_contour._price_contour import apply_from_grid_py

def apply_from_grid(
    grid: QuoteGrid,
    lambdas: dict[str, float],
    constraints: dict[str, dict[str, float]],
    chunk_size: int = 500_000,
) -> ApplyResult:
    """Single-pass Lagrangian apply on an existing QuoteGrid. No iteration."""
    return apply_from_grid_py(grid, lambdas, constraints, chunk_size)
```

### Haute-side usage

In `optimiser.py`, `select_frontier_point()`:

```python
# BEFORE: full re-solve (up to 50 iterations)
new_result = solver.solve(quote_grid, lambdas=new_lambdas)

# AFTER: single forward pass
from price_contour import apply_from_grid
new_result = apply_from_grid(
    quote_grid,
    lambdas=new_lambdas,
    constraints=job.get("config", {}).get("constraints", {}),
)
```

### Impact

Frontier point selection goes from O(max_iter * N) to O(N). For a typical solve with `max_iter=50` and 100K quotes, this is roughly a 50x speedup on the select operation. The user clicks a frontier point and sees results in milliseconds instead of seconds.

### Effort

Small. ~30 lines of PyO3 binding. The Rust core function already exists. The main work is wiring up `parse_constraints` (which is already a shared helper in `solver_py.rs`) and building the `PyApplyResult` (same pattern as existing `ApplyOptimiser.apply()`). Fully backward-compatible — new function, nothing changes for existing callers.

---

## 3. Scenario value stats per frontier point

### Problem

Each `FrontierPoint` currently contains only aggregate metrics: `total_objective`, `total_constraints`, `lambdas`, `iterations`, `converged`. It has no information about the distribution of per-quote optimal scenario values.

In Haute's frontier UI, the analyst wants to compare not just "total margin vs total volume" across frontier points, but also how the price distribution changes: "at this point, are we giving big discounts to a few quotes or small discounts to many?" This requires scenario value distribution stats (mean, std, percentiles).

Currently, Haute must re-solve at a selected point and then compute these stats from the result DataFrame. This means:
- The analyst can only see stats for one point at a time (the selected one)
- Each selection requires an API call + re-solve
- The frontier chart cannot show distribution previews (tooltips, sparklines)

### Current code

`frontier.rs`, `FrontierPoint`:
```rust
pub struct FrontierPoint {
    pub thresholds: Vec<f64>,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub lambdas: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    // No scenario value information
}
```

Inside `sweep_frontier`, after each `solve_online` call, the `SolveResult` contains `optimal_steps` (the per-quote argmax indices). The scenario values for those steps are readily available from the grid. The stats are trivially computable but are thrown away.

### Proposed change

**Rust (`data.rs` or `frontier.rs`):** Add stats to `FrontierPoint`:

```rust
pub struct FrontierPoint {
    pub thresholds: Vec<f64>,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub lambdas: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    // NEW: scenario value distribution stats
    pub sv_mean: f64,
    pub sv_std: f64,
    pub sv_min: f64,
    pub sv_p5: f64,
    pub sv_p25: f64,
    pub sv_median: f64,
    pub sv_p75: f64,
    pub sv_p95: f64,
    pub sv_max: f64,
    pub sv_pct_increase: f64,  // fraction of quotes with scenario_value > 1.0
    pub sv_pct_decrease: f64,  // fraction with scenario_value < 1.0
}
```

**Rust (`frontier.rs`):** After each `solve_online` call in the sweep loop, compute stats from `result.optimal_steps` and `grid.scenario_values`:

```rust
fn compute_sv_stats(optimal_steps: &[u32], grid: &QuoteGrid) -> ScenarioValueStats {
    let n = optimal_steps.len();
    let mut vals: Vec<f64> = optimal_steps.iter()
        .map(|&step| grid.scenario_value(step) as f64)
        .collect();
    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let sum: f64 = vals.iter().sum();
    let mean = sum / n as f64;
    let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    ScenarioValueStats {
        mean,
        std: variance.sqrt(),
        min: vals[0],
        p5: percentile(&vals, 0.05),
        p25: percentile(&vals, 0.25),
        median: percentile(&vals, 0.50),
        p75: percentile(&vals, 0.75),
        p95: percentile(&vals, 0.95),
        max: vals[n - 1],
        pct_increase: vals.iter().filter(|&&v| v > 1.0).count() as f64 / n as f64,
        pct_decrease: vals.iter().filter(|&&v| v < 1.0).count() as f64 / n as f64,
    }
}
```

**PyO3 binding (`frontier_py.rs`):** Add the new columns to the `points()` getter when building the Polars DataFrame. The columns would be named `sv_mean`, `sv_std`, `sv_p5`, etc.

### Haute-side usage

The frontier chart in `OptimiserPreview.tsx` can show scenario value distribution info for every point:
- Tooltips on hover: "mean scenario value: 1.03, 62% increases, 38% decreases"
- Color encoding: points with wider spread (high std) could appear differently
- The detail card can show stats immediately on click without an API call

The `_compute_scenario_value_stats` call in `select_frontier_point` becomes unnecessary for the overview — it's only needed if the full histogram is required.

### Effort

Medium. ~50 lines of Rust for the stats computation + percentile helper. ~10 lines in the PyO3 binding to add DataFrame columns. Backward-compatible (additive columns only).

---

## 4. Ratebook frontier

### Problem

`RatebookOptimiser` has no `frontier()` method. Haute's frontier computation is gated on `mode == "online"` — ratebook users get no efficient frontier at all.

Ratebook mode uses coordinate descent (CD) over factor groups, which is fundamentally different from the online Lagrangian solver. Each "point" in a ratebook frontier requires a full CD solve (multiple passes over all factor groups until convergence), not just a single Lagrangian iteration.

### Why Python (not Rust)

The online frontier is implemented in Rust because each point is a lightweight `solve_online` call on the same grid. Ratebook frontier is better implemented in Python because:

1. Each point calls `self.solve(grid, factors)` which is already a Python method with complex setup (factor column alignment, DataFrame preparation)
2. The CD orchestration is already in Python
3. The number of frontier points would be small (5-7) due to the cost per point
4. Python overhead is negligible compared to the per-point solve time

### Proposed change

**Python API (`ratebook.py`):**

```python
class RatebookOptimiser:
    def frontier(
        self,
        df_or_grid: pl.DataFrame | QuoteGrid,
        factors: pl.DataFrame,
        *,
        threshold_ranges: dict[str, tuple[float, float]],
        n_points_per_dim: int = 5,
        factor_columns: list[list[str]] | None = None,
    ) -> FrontierResult:
        """Sweep the efficient frontier by running coordinate descent at each threshold.

        Each frontier point is a full CD solve with modified constraint bounds.
        Results are warm-started from adjacent points using NN ordering.
        """
        # 1. Generate threshold grid (same as online frontier)
        # 2. NN-order the points for warm-starting
        # 3. Loop: for each point, modify self._constraints with the threshold,
        #    call self.solve(grid, factors, factor_columns=factor_columns),
        #    store result
        # 4. Build FrontierResult from collected points
```

The key implementation details:
- Reuse the NN ordering logic from the Rust frontier (or implement a simple version in Python — it's just a greedy nearest-neighbour on normalised threshold coordinates)
- Warm-start by passing `lambdas` from the previous point: this requires `RatebookOptimiser.solve()` to accept a `lambdas` kwarg (same as `OnlineOptimiser.solve()` already does)
- Default to fewer points (`n_points_per_dim=5`) since each is expensive
- Return a standard `FrontierResult` so the same Haute frontend code works for both modes

### Prerequisite

`RatebookOptimiser.solve()` may need a `lambdas` keyword argument for warm-starting. Check whether the underlying Rust ratebook solver accepts initial lambdas — if not, that's an additional Rust change.

### Haute-side usage

Remove the `if mode == "online"` guard in `_optimiser_service.py`. Both modes produce frontier data. The ratebook frontier call would also pass `factors_df`:

```python
if mode == "ratebook":
    frontier_result = solver.frontier(
        quote_grid, factors_df,
        threshold_ranges=ranges,
        n_points_per_dim=min(frontier_steps, 7),  # cap for performance
    )
```

### Effort

Medium-large. ~80 lines of Python for the frontier method + NN ordering. May need a small Rust change if warm-starting isn't supported in the ratebook solver. Testing requires ratebook test fixtures.

### Performance consideration

A single ratebook solve typically takes 5-30 seconds (depending on data size and number of factor groups). With 5 frontier points, that's 25-150 seconds additional after the initial solve. This should:
- Show a progress indicator ("Computing frontier: 3/5 points...")
- Be cancellable
- Be documented as potentially slow for large datasets
