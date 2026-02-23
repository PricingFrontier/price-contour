# Price Contour — Architecture

A high-performance Rust library with Python bindings for insurance price optimisation. Takes pre-computed objective and constraint values as Polars DataFrames, solves for optimal pricing via Lagrangian dual decomposition, and provides live scoring and efficient frontier generation.

```
uv add price-contour
```

```python
import price_contour
```

---

## High-Level Architecture

```
haute nodes (data prep, model scoring, UI hooks)
  │
  │  Polars DataFrames (long format)
  ▼
Python (price_contour/)
  ├── solver.py        # OnlineOptimiser, RatebookOptimiser
  ├── apply.py         # OptimiserApply: live scoring with stored lambdas
  └── frontier.py      # Efficient frontier sweep
        │
        ▼
PyO3 + pyo3-polars bindings (crates/price-contour/src/)
  ├── solver_py.rs     # Solver entry points
  ├── apply_py.rs      # Apply entry points
  └── frontier_py.rs   # Frontier entry points
        │
        ▼
Pure Rust core (crates/price-contour-core/src/)
  ├── solver/          # Lagrangian dual decomposition, coordinate descent
  ├── spline/          # Optional cubic spline fitting
  ├── frontier/        # Efficient frontier generation
  ├── apply/           # Live scoring logic
  └── data/            # Internal data structures & memory layout
```

Data preparation (expanding quotes across price points, scoring models, computing
price-derived features) is handled by haute. This library receives the already-scored
DataFrame and focuses purely on optimisation.

### Why pyo3-polars (not numpy)?

The interface to this library is Polars DataFrames — long-format tables of quotes and scenarios. `pyo3-polars` provides zero-copy DataFrame/Series passing between Python and Rust. This is the natural choice when the boundary is tabular data, not matrices.

`rustystats` correctly uses numpy because GLM fitting is matrix algebra. Different libraries, different data models.

---

## Workspace Layout

```
price-contour/
├── Cargo.toml                     # Workspace root
├── crates/
│   ├── price-contour-core/        # Pure Rust: algorithms, no Python deps
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── data.rs            # QuoteGrid, internal memory layout
│   │       ├── solver/
│   │       │   ├── mod.rs
│   │       │   ├── online.rs      # Lagrangian dual decomposition
│   │       │   ├── ratebook.rs    # Grouped Lagrangian (per-level argmax)
│   │       │   └── lambda.rs      # Lambda update strategies (subgradient, bisection)
│   │       ├── spline/
│   │       │   ├── mod.rs
│   │       │   └── cubic.rs       # Batched cubic spline fitting
│   │       ├── frontier/
│   │       │   └── mod.rs         # Efficient frontier sweep
│   │       ├── apply/
│   │       │   └── mod.rs         # Live scoring with stored lambdas
│   │       └── error.rs           # Error types
│   └── price-contour/             # PyO3 + pyo3-polars bindings
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs             # PyModule registration
│           ├── solver_py.rs       # Solver bindings
│           ├── apply_py.rs        # Apply bindings
│           └── frontier_py.rs     # Frontier bindings
├── python/
│   └── price_contour/
│       ├── __init__.py            # Public API
│       ├── solver.py              # OnlineOptimiser
│       ├── ratebook.py            # RatebookOptimiser (CD loop, structure selection)
│       ├── apply.py               # OptimiserApply
│       └── frontier.py            # EfficientFrontier
├── tests/                         # Rust integration tests
├── python/tests/                  # Python tests
└── pyproject.toml                 # maturin build config
```

---

## Input Format

The library accepts a **long-format Polars DataFrame** — one row per quote per scenario step:

| Column | Type | Description |
|---|---|---|
| `quote_id` | str/int | Identifies each quote/risk. Column name is configurable. |
| `scenario_step` | int | Step index (0, 1, 2, ..., M-1). Column name is configurable. |
| `multiplier` | f32 | The price multiplier at this step (e.g. 0.80, 0.81, ..., 1.20). |
| `objective` | f32 | Objective function value at this step (e.g. expected income). Column name is configurable. |
| `constraint_*` | f32 | One column per constraint (e.g. `volume`, `loss_ratio`). Names are configurable. |

### Example

```
┌──────────┬───────────────┬────────────┬───────────┬────────┐
│ quote_id │ scenario_step │ multiplier │ objective │ volume │
│ str      │ i32           │ f32        │ f32       │ f32    │
╞══════════╪═══════════════╪════════════╪═══════════╪════════╡
│ Q001     │ 0             │ 0.80       │ 85.2      │ 0.95   │
│ Q001     │ 1             │ 0.90       │ 92.1      │ 0.88   │
│ Q001     │ 2             │ 1.00       │ 100.0     │ 0.80   │
│ Q001     │ 3             │ 1.10       │ 105.3     │ 0.70   │
│ Q001     │ 4             │ 1.20       │ 108.1     │ 0.58   │
│ Q002     │ 0             │ 0.80       │ 42.0      │ 0.97   │
│ Q002     │ 1             │ 0.90       │ 45.8      │ 0.91   │
│ ...      │ ...           │ ...        │ ...       │ ...    │
└──────────┴───────────────┴────────────┴───────────┴────────┘
```

### Why long format?

- Natural output of the prep/expansion step (cross-join quote × multiplier grid)
- Works directly with Polars groupby/sort operations
- No column-name encoding of step indices
- Easy to add more steps or constraints without schema changes

### Internal representation

On ingestion, the library validates, sorts by `(quote_id, scenario_step)`, and builds a contiguous `QuoteGrid` — a struct-of-arrays layout optimised for the solver's access patterns:

```rust
/// Contiguous memory layout for solver operations.
/// All arrays are length N*M, laid out quote-major:
/// [q0_s0, q0_s1, ..., q0_sM-1, q1_s0, q1_s1, ..., q1_sM-1, ...]
pub struct QuoteGrid {
    pub n_quotes: usize,            // N
    pub n_steps: usize,             // M
    pub multipliers: Vec<f32>,      // (M,) — shared multiplier grid
    pub objective: Vec<f32>,        // (N*M,) — contiguous, quote-major
    pub constraints: Vec<Vec<f32>>, // K × (N*M,) — one flat vec per constraint
    pub quote_ids: Vec<String>,     // (N,) — original quote IDs, in order
}
```

Quote-major layout means each quote's M steps are contiguous in memory. This is optimal for the per-quote argmax in the Lagrangian inner loop.

---

## Core Algorithm: Online Optimisation

### Problem

```
Maximise   Σ_i objective_i(m_i)
Subject to Σ_i constraint_k_i(m_i) ≥ threshold_k   for each constraint k
           m_i ∈ {multiplier grid}                    for each quote i
```

### Lagrangian Dual Decomposition

For fixed lambda values, each quote's subproblem is independent:

```
For each quote i:
  L_i(j) = objective_i(j) + Σ_k λ_k × constraint_k_i(j)
  best_j = argmax_j L_i(j)
```

This is embarrassingly parallel and vectorises perfectly.

### Algorithm

```
Initialise: λ = zeros(K)

Outer loop (max_iter iterations):

  // Process quotes in chunks for bounded memory
  For each chunk of C quotes:

    1. Compute Lagrangian for each quote × step:
       L[i, j] = objective[i*M + j] + Σ_k λ_k × constraint_k[i*M + j]

    2. Find optimal step per quote:
       best[i] = argmax over j of L[i, j]

    3. Accumulate portfolio totals:
       total_obj += Σ_i objective[i*M + best[i]]
       total_con_k += Σ_i constraint_k[i*M + best[i]]

  // Update lambdas
  For each constraint k:
    λ_k = max(0, λ_k + step_size × (threshold_k - total_con_k))

  Check convergence → stop if all constraints satisfied and λ stable.

Output: λ values (K,), optimal_step per quote (N,)
```

### Chunked + Parallel Execution

```
Outer loop
  └── chunk_iter (sequential, bounded memory)
        └── Rayon par_chunks within each chunk
              └── SIMD-friendly inner loop (Lagrangian + argmax)
```

The chunk size controls peak memory. Within each chunk, Rayon parallelises across quotes. The inner loop (multiply-accumulate + argmax over M steps) is auto-vectorised by LLVM.

### Lambda Update Strategies

Two strategies, selectable by the user:

1. **Subgradient** (default): Simple, `λ_k += α/√t × (threshold - total)`. Good for most cases.
2. **Bisection**: Binary search per lambda for the value that makes the constraint bind. Tighter convergence, more sweeps.

---

## Core Algorithm: Ratebook Optimisation

### Key Insight: Grouped Lagrangian

The ratebook solver is structurally the same as the online solver, but with a **group constraint**: all quotes sharing the same factor level must pick the same candidate value.

In the online solver, each quote independently picks its best step. In the ratebook solver, all quotes at (say) `vehicle_age=3` must share the same multiplier — we do a **per-level argmax** instead of a per-quote argmax. The Lagrangian for a level is the sum of Lagrangians across all quotes at that level.

This means the core Lagrangian engine is shared between online and ratebook modes. The only difference is the grouping.

### Responsibility Split: price_contour owns the full ratebook solve

Price_contour owns the entire ratebook optimisation — coordinate descent across factors, structure selection, and the per-factor grouped Lagrangian solve. Haute's only job is to provide a **scoring callback** that, given a factor table, returns the scored long-format DataFrame.

This keeps ratebook logic centralised in one library rather than split across two.

### Input Format

Two DataFrames:

**Scored DataFrame** — same long format as online (one row per quote per scenario step):

| Column | Type | Description |
|---|---|---|
| `quote_id` | str/int | Identifies each quote |
| `scenario_step` | int | Step index (0, 1, ..., M-1) |
| `multiplier` | f32 | Price multiplier at this step |
| `objective` | f32 | Objective value at this step |
| `constraint_*` | f32 | Constraint values at this step |

**Factors DataFrame** — quote-level factor assignments (one row per quote):

| Column | Type | Description |
|---|---|---|
| `quote_id` | str/int | Identifies each quote |
| `factor_col_1` | str/int | Factor value (e.g. `vehicle_age`) |
| `factor_col_2` | str/int | Factor value (e.g. `region`) |
| ... | ... | One column per candidate rating factor |

Factor attributes are quote-level, not step-level — storing them separately avoids duplicating them across every step row. The Python layer joins `factor_level` onto the scored DataFrame by `quote_id` as needed.

### Modes

**Explicit structure**: haute passes `factor_columns` — a list of column names (or list of lists for interactions). Price_contour runs coordinate descent over the specified factors.

**Auto structure** (default): haute passes the full `factors` DataFrame with all candidate columns. Price_contour discovers which main effects and interactions improve the objective:

```
1. Solve online (no grouping) → baseline objective

2. Screen main effects (reduced iterations):
   For each column in factors:
     Solve ratebook grouped by that column
     Record objective lift over baseline
     Record level distinctness (variance of optimal multipliers)
   Rank by lift, keep factors above threshold

3. Screen interactions (reduced iterations):
   For each pair of selected main effects:
     Solve ratebook grouped by the pair (composite factor_level)
     Compare lift to sum of individual main-effect lifts
     Check level distinctness within each main effect
     Check minimum cell volume (credibility)
   Keep interactions where marginal lift justifies extra levels

4. Coordinate descent with selected structure
```

Screening solves use fewer iterations (e.g. 10 instead of 50) since we're ranking relative lift, not finding exact multipliers. Screening solves are independent and parallelise trivially.

For a typical insurance portfolio (10–20 candidate factors), screening adds ~30–50% overhead vs an explicit structure solve.

### Coordinate Descent

Price_contour runs the coordinate descent loop internally:

```
For each CD iteration:
  For each factor (or interaction) in the selected structure:
    1. Derive factor_level from the factor column(s) in the factors DataFrame
    2. Join factor_level onto the scored DataFrame by quote_id
    3. Solve the grouped Lagrangian (same algorithm as below)
    4. Update the factor table with optimal multipliers per level

  Request re-scoring from haute via callback (factor table changed)
  Check CD convergence → stop if factor table stable
```

Haute provides a scoring callback: given the current factor table, re-expand quotes, score models, and return the updated scored DataFrame. Price_contour drives the loop.

### Grouped Lagrangian Algorithm

For a single factor solve (one step of coordinate descent):

```
Initialise: λ = zeros(K)  (or warm-start from previous factor's solve)

Outer loop (max_iter iterations):

  For each chunk of quotes:

    1. Compute Lagrangian per quote per step (same as online)

    2. Accumulate Lagrangian by factor_level:
       For each level l, for each step j:
         group_L[l, j] = Σ_{i in level l} L[i, j]

    3. Find optimal step per level:
       best_step[l] = argmax over j of group_L[l, j]

    4. Map back to quotes: best[i] = best_step[level_of(i)]

    5. Accumulate portfolio totals using best[i]

  Update lambdas (same as online)

Output: optimal_step per level, λ values
```

---

## Optional: Spline Fitting

When enabled, cubic splines are fitted through the discrete price points before optimisation. This gives a smooth landscape and allows analytical gradient computation.

```rust
pub struct SplineFitter {
    knots: Vec<f32>,    // The M multiplier values as knot positions
    degree: usize,      // Default: 3 (cubic)
}

impl SplineFitter {
    /// Fit splines for all quotes in a batch.
    /// Because the knot positions are the same for all quotes
    /// (the shared multiplier grid), the spline basis matrix is
    /// computed once and applied to all quotes via a matrix multiply.
    pub fn fit_batch(&self, grid: &QuoteGrid) -> SplineCoefficients { ... }
}
```

The key insight: uniform knot positions across all quotes means the spline basis matrix is computed once and applied as a batched matrix multiply — not a per-quote loop.

**When to use splines:**
- Many steps (M > 20): splines smooth out noise
- Live scoring: store spline coefficients as artifacts, evaluate at any multiplier (not just grid points)

**When to skip:**
- Few steps (M ≤ 10): discrete grid is fine
- Speed priority: skipping splines saves the fitting step

---

## Apply (Live Scoring)

Given stored lambdas from a batch solve and a single quote's objective/constraint values at M steps:

```rust
pub fn apply(
    objective: &[f32],        // (M,) — single quote's objective at each step
    constraints: &[&[f32]],   // K × (M,) — single quote's constraints
    lambdas: &[f32],          // (K,) — stored from batch solve
    multipliers: &[f32],      // (M,) — multiplier grid
) -> ApplyResult {
    // Evaluate Lagrangian at each step
    // L(j) = objective(j) + Σ_k λ_k × constraint_k(j)
    // Return the step with the highest Lagrangian value
}
```

This is microseconds — one pass over M values. The cost is dominated by whatever upstream model scoring produces the input values.

---

## Efficient Frontier

Sweep constraint thresholds to produce the Pareto curve:

```rust
pub fn frontier(
    grid: &QuoteGrid,
    threshold_range: &[(f32, f32)],  // (min, max) per constraint
    n_points: usize,                  // e.g. 20
    config: &SolverConfig,
) -> FrontierResult {
    // For each threshold combination:
    //   solve → record (total_objective, total_constraints, lambdas)
    // Return the full Pareto curve
}
```

Each point on the frontier is a full solve. With 20 points and ~1.5s solve time: ~30s for the frontier.

---

## Output Format

### Solver Output

Returns a Polars DataFrame:

| Column | Type | Description |
|---|---|---|
| `quote_id` | str/int | Original quote ID |
| `optimal_step` | i32 | Index of the optimal scenario step |
| `optimal_multiplier` | f32 | Multiplier value at optimal step |
| `optimal_objective` | f32 | Objective value at optimal step |
| `optimal_{constraint}` | f32 | Each constraint's value at optimal step |

Plus a `SolveResult` metadata object:

```python
result.lambdas          # dict[str, float] — Lagrange multipliers per constraint
result.iterations       # int — number of outer iterations
result.converged        # bool
result.total_objective  # float — portfolio-level objective at optimum
result.total_constraints  # dict[str, float] — portfolio-level constraint values
result.dataframe        # pl.DataFrame — the per-quote results above
```

### Frontier Output

```python
frontier.points         # pl.DataFrame with columns: threshold_*, objective, constraint_*, lambda_*
frontier.n_points       # int
```

### Apply Output

```python
apply_result.optimal_multiplier   # float
apply_result.optimal_step         # int
apply_result.lagrangian_values    # list[float] — L(j) at each step (for diagnostics)
```

---

## Python API

```python
import polars as pl
import price_contour as pc

# --- Online Optimisation ---

df = pl.read_parquet("scored_quotes.parquet")
# Columns: quote_id, scenario_step, multiplier, expected_income, volume, loss_ratio

solver = pc.OnlineOptimiser(
    quote_id="quote_id",
    scenario_step="scenario_step",
    multiplier="multiplier",
    objective="expected_income",
    constraints={
        "volume": {"min": 0.90},          # portfolio volume ≥ 90% of baseline
        "loss_ratio": {"max": 1.05},      # portfolio loss ratio ≤ 105% of baseline
    },
    spline=False,               # disable spline fitting (use discrete grid)
    chunk_size=500_000,         # quotes per memory chunk
    max_iter=50,
    lambda_strategy="subgradient",  # or "bisection"
)

result = solver.solve(df)

# Warm-start: reuse lambdas from a previous solve (faster convergence)
result2 = solver.solve(df, lambdas=result.lambdas)

result.lambdas              # {"volume": 0.42, "loss_ratio": 1.13}
result.converged            # True
result.dataframe            # pl.DataFrame with optimal_step, optimal_multiplier, ...

# --- Efficient Frontier ---

frontier = solver.frontier(
    df,
    sweep="volume",                    # which constraint to sweep
    threshold_range=(0.85, 0.98),
    n_points=20,
)

frontier.points    # pl.DataFrame: threshold_volume, total_objective, total_volume, ...

# --- Live Scoring (Apply) ---

applier = pc.OptimiserApply(
    lambdas=result.lambdas,
    multipliers=result.multipliers,     # the grid used in the solve
    objective="expected_income",
    constraints=["volume", "loss_ratio"],
)

# Single quote — pass a 1-row-per-step DataFrame
quote_df = pl.DataFrame({
    "scenario_step": [0, 1, 2, 3, 4],
    "multiplier": [0.80, 0.90, 1.00, 1.10, 1.20],
    "expected_income": [85.2, 92.1, 100.0, 105.3, 108.1],
    "volume": [0.95, 0.88, 0.80, 0.70, 0.58],
    "loss_ratio": [0.58, 0.60, 0.63, 0.66, 0.70],
})

score = applier.score(quote_df)
score.optimal_multiplier    # 1.05
score.optimal_step          # 2

# Batch apply — pass many quotes
batch_scores = applier.score_batch(large_df)  # returns pl.DataFrame

# --- Ratebook Optimisation ---
# Price_contour owns the full ratebook solve: structure selection,
# coordinate descent, and per-factor grouped Lagrangian.
# Haute provides a scoring callback for re-scoring after factor updates.

ratebook_solver = pc.RatebookOptimiser(
    quote_id="quote_id",
    scenario_step="scenario_step",
    multiplier="multiplier",
    objective="expected_income",
    constraints={
        "volume": {"min": 0.90},
        "loss_ratio": {"max": 1.05},
    },
    chunk_size=500_000,
    max_iter=50,
)

# Factors DataFrame — one row per quote, one column per candidate factor
# quote_id, vehicle_age, region, driver_age_band, ...
factors = pl.read_parquet("quote_factors.parquet")

# --- Auto structure (default) ---
# Price_contour discovers which factors and interactions improve the objective.
result = ratebook_solver.solve(
    df,
    factors=factors,
    # Auto structure controls
    max_interaction_order=2,       # don't try 3-way interactions
    max_main_effects=10,           # keep top 10 from screening
    max_interactions=15,           # test at most 15 pairs
    min_cell_volume=500,           # credibility floor per level
    screening_iterations=10,       # reduced iterations for screening solves
    # Coordinate descent controls
    max_cd_iterations=3,
    # Scoring callback — haute provides this
    score_fn=haute_score_callback,  # (factor_table) → scored pl.DataFrame
)

result.selected_structure   # [["vehicle_age"], ["region"], ["vehicle_age", "region"]]
result.structure_report     # pl.DataFrame: factor, lift, n_levels, min_cell_volume, distinctness
result.level_results        # pl.DataFrame: factor, factor_level, optimal_multiplier
result.lambdas              # constraint shadow prices
result.converged            # bool

# --- Explicit structure ---
# Haute already knows the shape — skip auto discovery.
result = ratebook_solver.solve(
    df,
    factors=factors,
    factor_columns=[
        ["vehicle_age"],               # main effect
        ["region"],                    # main effect
        ["vehicle_age", "region"],     # interaction
    ],
    max_cd_iterations=3,
    score_fn=haute_score_callback,
)
```

---

## Rust Dependencies

### price-contour-core

```toml
[dependencies]
rayon = "1.10"          # Parallel chunk processing
thiserror = "2.0"       # Error types
```

No ndarray, no nalgebra — the data structures are flat `Vec<f32>` for maximum control over memory layout and cache behaviour. The inner loops are simple enough that raw slices + SIMD-friendly iteration outperform matrix abstractions.

### price-contour (bindings)

```toml
[dependencies]
price-contour-core = { path = "../price-contour-core" }
pyo3 = { version = "0.23", features = ["extension-module"] }
pyo3-polars = "0.20"     # Native Polars DataFrame interop
polars = "0.46"           # For internal DataFrame operations
```

---

## Memory Budget

Target: bounded peak memory regardless of portfolio size.

For N=5M quotes, M=50 steps, K=3 constraints, chunk_size=500K:

| Allocation | Size |
|---|---|
| Full objective array (N×M f32) | 1.0 GB |
| Full constraint arrays (K×N×M f32) | 3.0 GB |
| Lagrangian buffer (chunk×M f32) | 100 MB |
| Best-step indices (N i32) | 20 MB |
| **Total** | **~4.1 GB** |

For smaller portfolios (e.g. 500K quotes): ~400 MB total.

### Streaming mode (future)

For portfolios too large to fit in memory, support reading chunks from Parquet directly. The solver only needs one chunk at a time for the inner loop — accumulate totals across chunks, then update lambdas.

---

## Performance Targets

| Operation | Portfolio | Time |
|---|---|---|
| Single solve (50 iterations) | 5M quotes, 50 steps | ~1.5s |
| Frontier (20 points) | 5M quotes, 50 steps | ~30s |
| Apply (single quote) | 1 quote, 50 steps | <1μs |
| Apply (batch) | 5M quotes, 50 steps | ~200ms |

The solver is compute-bound (multiply-accumulate + argmax), not memory-bound. Rayon parallelism + SIMD auto-vectorisation should approach memory bandwidth limits.

---

## Constraint Types

### Portfolio-level (sum) constraints

The primary constraint type. The constraint is on the portfolio aggregate:

```
Σ_i constraint_k_i(m_i) ≥ threshold_k    (min constraint)
Σ_i constraint_k_i(m_i) ≤ threshold_k    (max constraint)
```

### Threshold modes

**Relative (default)**: The threshold is a fraction of the baseline value (portfolio total at multiplier=1.0, or the nearest step). The library computes the baseline from the input data automatically.

```python
constraints={
    "volume": {"min": 0.90},          # portfolio volume ≥ 90% of baseline
    "loss_ratio": {"max": 1.05},      # portfolio loss ratio ≤ 105% of baseline
}
```

**Absolute**: The threshold is an explicit value. Use when the constraint isn't naturally relative to a baseline.

```python
constraints={
    "volume": {"min_abs": 45000},     # portfolio volume ≥ 45,000
    "loss_ratio": {"max_abs": 0.65},  # portfolio loss ratio ≤ 0.65
}
```

The library identifies the baseline by finding the step closest to `multiplier=1.0` for each quote and summing the constraint column across all quotes at that step.

### Per-quote constraints (future)

Clip the multiplier range per quote (e.g. max 10% price change per individual risk). Handled by restricting the feasible set per quote, not by adding Lagrange multipliers.

---

## Key Design Decisions

1. **Long-format Polars input**: natural for the prep pipeline, no wide-column encoding hacks.
2. **pyo3-polars for bindings**: DataFrame in, DataFrame out — zero-copy where possible.
3. **f32 default**: sufficient precision for insurance pricing, halves memory.
4. **Flat Vec\<f32\> internals**: no ndarray/nalgebra. Raw slices give maximum control over memory layout for SIMD-friendly loops.
5. **Quote-major memory layout**: each quote's M steps are contiguous, optimal for per-quote argmax.
6. **Chunk + Rayon parallelism**: chunks bound memory, Rayon parallelises within chunks.
7. **Splines optional**: discrete grid is the default; splines available when the user needs smooth curves or live scoring artifacts.
8. **Two-crate workspace**: pure Rust core (testable, no Python deps) + thin PyO3 bindings layer. Follows the rustystats pattern.
9. **Config-driven column names**: quote_id, scenario_step, objective, constraint names are all configurable — no hardcoded column assumptions.
10. **Price_contour owns ratebook fully**: coordinate descent, structure selection (auto and explicit), and the grouped Lagrangian solve all live in price_contour. Haute provides scored DataFrames via a callback — it doesn't need to know about factors or CD iteration.
11. **Separate factors DataFrame**: factor assignments are quote-level, passed as a separate DataFrame rather than duplicated across every step row in the scored data.

---

## Artifact Serialization

Solver artifacts (for live scoring via `OptimiserApply`) are saved/loaded as:

- **Lambdas + config** (scalars): JSON. Human-readable, easy to inspect and version.
- **Spline coefficients** (per-quote arrays, when spline mode is used): Parquet. Columnar, compressed, fast to load.

```python
# Save
result.save("artifacts/optimiser_v1")
# Creates:
#   artifacts/optimiser_v1/config.json    (lambdas, multiplier grid, constraint names)
#   artifacts/optimiser_v1/splines.parquet (optional, only if spline=True)

# Load
applier = pc.OptimiserApply.load("artifacts/optimiser_v1")
```

---

## Resolved Design Decisions

| Decision | Resolution |
|---|---|
| Prep/expansion | Handled by haute, not this library |
| Artifact format | JSON for scalars, Parquet for arrays |
| Baseline thresholds | Library computes baseline from data; supports both relative and absolute thresholds |
| Ratebook input | Scored DF (same as online) + separate quote-level factors DF; price_contour derives factor_level and joins by quote_id |
| Ratebook structure | Auto discovery (default) or explicit factor_columns; price_contour owns the full loop |
| Coordinate descent | Lives in price_contour, not haute; haute provides a scoring callback |
| Warm-start | Supported — pass `lambdas=` to `solve()` for faster convergence |
