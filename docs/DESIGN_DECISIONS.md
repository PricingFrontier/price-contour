# Price Contour — Design Decisions

This document captures the finalised design decisions for the next phase of price-contour development. It covers the GridBuilder API, grouped Lagrangian solver, multi-dimensional efficient frontier, ratebook optimisation with banding integration, and the two-input design for ratebook mode.

These decisions supersede the earlier `optimisation_design.md` (which assumed numpy and a different node structure). The `architecture.md` remains the canonical reference for the current codebase — this document extends it with the decisions for features not yet implemented.

---

## 1. GridBuilder — Incremental Data Ingestion

### Problem

The current `ingest_dataframe()` in `solver_py.rs` takes a single Polars DataFrame, sorts it, and builds the entire `QuoteGrid` in one shot. For large portfolios processed through haute's chunked pipeline executor, the full scored DataFrame never exists in memory at once — it's produced in chunks. Requiring materialisation of the full DataFrame before calling the solver defeats the purpose of chunked execution.

### Decision

Add a `QuoteGridBuilder` that accepts data incrementally, chunk by chunk. Each chunk is a Polars DataFrame in the standard long format. The builder extracts the relevant f32 columns and appends them to the growing Rust-owned `Vec<f32>` arrays, then the Python DataFrame chunk is freed.

### API

```rust
// Rust core
pub struct QuoteGridBuilder {
    n_steps: usize,
    multipliers: Vec<f32>,
    objective: Vec<f32>,
    constraints: Vec<Vec<f32>>,
    constraint_names: Vec<String>,
    quote_ids: Vec<String>,
    n_quotes: usize,
}

impl QuoteGridBuilder {
    pub fn new(n_steps: usize, multipliers: Vec<f32>, constraint_names: Vec<String>) -> Self;
    pub fn append(&mut self, /* chunk arrays */) -> Result<()>;
    pub fn build(self) -> Result<QuoteGrid>;
}
```

```python
# Python API
builder = pc.QuoteGridBuilder(
    quote_id="quote_id",
    scenario_step="scenario_step",
    multiplier="multiplier",
    objective="expected_income",
    constraints=["volume", "loss_ratio"],
)

for chunk_df in chunked_pipeline:
    builder.append(chunk_df)

grid = builder.build()  # Returns opaque handle to Rust-owned QuoteGrid

result = solver.solve(grid)  # Accepts QuoteGrid directly
```

### Key Properties

- **Multiplier grid extracted from first chunk.** The builder validates that subsequent chunks use the same grid.
- **n_steps validated per chunk.** Every chunk must have rows divisible by n_steps, and step sequences must be correct.
- **Memory profile:** Peak memory = one Python chunk (transient) + accumulated f32 grid (persistent). For 5M quotes × 50 steps × 4 columns (obj + 3 constraints): ~4 GB of f32. The Python chunks are freed after each `append()`.
- **The `solve()` method accepts either a `QuoteGrid` (from builder) or a `pl.DataFrame` (existing one-shot path).** The one-shot path remains for convenience with small datasets.

### Why Not Streaming Chunks Through the Solver?

The solver needs random access to the full grid during each iteration — every lambda update requires a full portfolio sweep. True streaming (process once, discard) would require a fundamentally different algorithm. The GridBuilder pattern gives us chunked *ingestion* while keeping the solver's random-access requirement.

---

## 2. Two-Input Ratebook Design

### Problem

Ratebook optimisation needs two pieces of information per quote:
1. **Objective/constraint values at each multiplier step** — the scored long-format DataFrame (N×M rows), same as online mode.
2. **Risk factor assignments** — which factor level each quote belongs to (e.g. `age_band="25-34"`, `region="South East"`). These are quote-level, not step-level.

The original design considered extracting factor columns from the scored DataFrame. This was rejected because:
- The scored DataFrame is in long format (N×M rows) — factor columns would be duplicated M times per quote.
- Factor columns are string-typed band assignments from upstream banding nodes, not numeric values. Including them in the scored DataFrame pollutes the f32 optimisation grid.
- Keeping them separate is cleaner and avoids unnecessary memory duplication.

### Decision

The optimiser accepts **two separate DataFrames** in ratebook mode:

| Input | Format | Size | Passed via |
|---|---|---|---|
| Scored DataFrame | Long format (N×M rows), same as online | Large, chunked | `GridBuilder.append()` |
| Factors DataFrame | One row per quote (N rows), string/categorical columns | Small, passed once | `GridBuilder.set_factors()` or constructor |

### Factors DataFrame Format

| Column | Type | Description |
|---|---|---|
| `quote_id` | str | Quote identifier (join key) |
| `age_band` | str | Factor level from banding node |
| `region` | str | Factor level from banding node |
| `vehicle_group` | str | Factor level from banding node |
| ... | str | One column per candidate rating factor |

### How Factors Are Used

1. The factors DataFrame is passed once to the `QuoteGridBuilder` (or directly to the solver).
2. price-contour builds a `GroupMapping` — a `Vec<u32>` of length N mapping each quote to its group index for the currently active factor.
3. When doing coordinate descent, the solver re-derives the `GroupMapping` for each factor being optimised.
4. For interactions (e.g. `age_band × region`), the solver creates composite group keys by concatenating factor levels.

### Memory

The factors DataFrame is small (N rows × F string columns). It's held in Rust as a `Vec<Vec<String>>` — one inner vec per factor column, each of length N. For 5M quotes × 10 factors × ~10 bytes per string: ~500 MB. This is manageable and doesn't need chunking.

---

## 3. Grouped Lagrangian Solver

### Problem

Online optimisation does per-quote argmax — each quote independently picks its best multiplier step. Ratebook optimisation requires that all quotes sharing the same factor level receive the **same factor adjustment** — that's what makes it a rating table. The question is how to express this as a solver primitive.

### Important Distinction: Factor Value vs Overall Multiplier

A rating table assigns each factor level a single value (e.g. `age_factor["25-34"] = 1.05`). But two quotes at the same age band have different overall multipliers because their other factors differ:

```
Quote A: age 25-34, London     → overall = 1.05 × 1.12 × ... = 1.18
Quote B: age 25-34, rural      → overall = 1.05 × 0.88 × ... = 0.92
```

The grouped solver must find the best **factor value** per group, not the best **overall multiplier** per group. These are different things because each quote's residual (product of all other factors) translates the same factor value to a different position in the scored grid.

### Decision

Implement a `solve_grouped()` function with a **remapping step**. For each (quote, candidate factor value), compute the target overall multiplier (`residual × candidate`), find the nearest step in the scored grid, and look up the pre-computed objective/constraint values. The grouped argmax then selects the best candidate factor value per group.

### Algorithm

```
Input:
  grid: QuoteGrid (scored at overall multiplier steps)
  group_mapping: GroupMapping (N,) — maps quote → group index
  residuals: Vec<f32> (N,) — per-quote residual (product of other factors)
  candidates: Vec<f32> (P,) — candidate factor values to try
  specs: constraint specs
  config: SolverConfig

Initialise: λ = zeros(K)

Outer loop (max_iter):

  // Reset group accumulators — G groups × P candidate steps
  group_L[g, j] = 0.0  for all g, j

  For each chunk of quotes:

    For each quote i in chunk:
      g = group_of[i]

      For each candidate factor step j:

        1. Remap: target_overall = residuals[i] × candidates[j]
           k = nearest step in grid.multipliers to target_overall

        2. Look up pre-computed values:
           idx = i * M + k
           obj_val = grid.objective[idx]
           con_vals = grid.constraints[c][idx] for each c

        3. Compute Lagrangian:
           L = obj_val + Σ_c λ_c × con_vals[c]

        4. Accumulate into group:
           group_L[g][j] += L

  // After all chunks:
  For each group g:
    best_j[g] = argmax_j group_L[g, j]
    optimal_factor_value[g] = candidates[best_j[g]]

  // Compute portfolio totals at the selected factor values
  For each quote i:
    g = group_of[i]
    target_overall = residuals[i] × optimal_factor_value[g]
    k = nearest step in grid.multipliers
    total_obj += grid.objective[i * M + k]
    total_con[c] += grid.constraints[c][i * M + k]

  Update lambdas (same machinery as online)
  Check convergence

Output: optimal_factor_value per group (G,), λ values (K,)
```

### Key Properties

- **Fused remap + Lagrangian + accumulate.** Steps 1–4 fuse into a single pass per quote with **O(1) extra memory per quote**. No materialised remapped grid. The only persistent allocation is `group_L` (G × P), which is tiny.
- **Fully chunkable.** Each chunk of quotes contributes to the `group_L` accumulators. After all chunks, the argmax is taken per group. Same chunking pattern as `solve_online()`.
- **Nearest-step lookup.** Binary search in the sorted multiplier grid: O(log M) per lookup. Since M is typically 20–100, this is ~6 comparisons — negligible vs the Lagrangian arithmetic.
- **G is small.** For a typical factor, G = 5–50 levels. The `group_L` matrix is tiny (e.g. 50 groups × 50 candidates × 8 bytes = 20 KB).
- **Quantisation error is small.** For M=50 steps over [0.80, 1.20], grid spacing = 0.008. Max error from nearest-step = 0.004 (0.4% of multiplier). Negligible for insurance pricing.

### Out-of-Range Handling

If `residual × candidate` falls outside the scored grid range (e.g. residual=1.15, candidate=1.15 → target=1.32 but grid max=1.20), the lookup clamps to the grid boundary. This is handled by:

1. **Wider scored grid for ratebook.** Recommend scoring at e.g. [0.60, 1.50] instead of [0.80, 1.20] when ratebook mode is intended. The price scenario node in haute would use a wider range.
2. **Clamping diagnostics.** The solver reports what percentage of lookups hit the boundary, so the user knows if the grid is too narrow.
3. **Natural constraint.** CD tends toward moderate factor values — extreme values perform poorly in the Lagrangian because they push many quotes to grid boundaries where values are suboptimal.

### Rust Structs

```rust
pub struct GroupMapping {
    pub group_of: Vec<u32>,        // (N,) — maps quote index → group index
    pub n_groups: usize,            // G
    pub group_labels: Vec<String>,  // (G,) — human-readable group names
}

pub struct GroupedSolveResult {
    pub optimal_factor_values: Vec<f32>,  // (G,) — factor value per group
    pub lambdas: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub baseline_objective: f64,
    pub baseline_constraints: Vec<f64>,
    pub clamp_rate: f32,  // fraction of lookups that hit grid boundary
}
```

### Relationship to Online Solver

The online solver is the degenerate case: every quote is its own group, residuals are all 1.0, and candidates = grid multipliers. In that case, `residual × candidate = candidate = grid step`, so the remapping is an identity operation and the algorithm reduces to per-quote argmax.

We keep the online solver as a separate code path because:
- No remapping overhead (identity is free)
- Per-quote argmax is more cache-friendly than group accumulation
- No `group_L` matrix needed

Both implementations share the lambda update code.

---

## 4. Multi-Dimensional Efficient Frontier

### Problem

The current architecture describes a single-constraint sweep. Real pricing problems have 2-3 constraints (e.g. volume retention, loss ratio, average premium change). A best-in-class frontier explores the **joint** constraint space.

### Decision

Implement multi-dimensional frontier generation. For K constraints, sweep a grid of threshold combinations and solve at each point. Use **nearest-neighbour warm-start ordering** to minimise total solve time.

### Algorithm

```
Input:
  grid: QuoteGrid
  constraints: K constraints, each with (min_threshold, max_threshold)
  n_points_per_dim: int (e.g. 10)
  solver_config: SolverConfig

1. Generate threshold grid:
   For each constraint k, create linspace(min_k, max_k, n_points_per_dim)
   Total grid points: n_points_per_dim^K

2. Order grid points by nearest-neighbour path:
   Start from a central point (baseline thresholds)
   Greedily visit the nearest unvisited point (Euclidean in normalised threshold space)

3. For each grid point (in warm-start order):
   Solve with thresholds from this grid point
   Warm-start with lambdas from the previous point in the path
   Record: thresholds, total_objective, total_constraints, lambdas, iterations

4. Filter to Pareto-optimal points (optional)

Output: FrontierResult with all grid points
```

### Warm-Start Performance

With nearest-neighbour ordering, adjacent solves have similar thresholds, so lambdas from the previous solve are close to optimal. Empirically, warm-started solves converge in 3–5 iterations vs 50 for cold-start. This makes the frontier ~10× faster than naive independent solves.

### Scaling with K

| K | Grid points (n=10) | Est. time (5M quotes) |
|---|---|---|
| 1 | 10 | ~5s |
| 2 | 100 | ~50s |
| 3 | 1,000 | ~8 min |

For K ≥ 3, the grid becomes large. Options:
- **Reduce n_points_per_dim** (e.g. 5 instead of 10 → 125 points for K=3)
- **Hierarchical approach**: coarse grid first, refine around the Pareto front
- **Adaptive sampling**: Latin hypercube or Sobol sequence instead of regular grid

The API supports all these via a `threshold_grid` parameter that accepts either `n_points_per_dim` (regular grid) or an explicit list of threshold combinations.

### Rust Structs

```rust
pub struct FrontierConfig {
    pub n_points_per_dim: usize,
    pub threshold_ranges: Vec<(f64, f64)>,  // (min, max) per constraint
    pub solver_config: SolverConfig,
}

pub struct FrontierPoint {
    pub thresholds: Vec<f64>,
    pub total_objective: f64,
    pub total_constraints: Vec<f64>,
    pub lambdas: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

pub struct FrontierResult {
    pub points: Vec<FrontierPoint>,
    pub n_constraints: usize,
}
```

### Python API

```python
frontier = solver.frontier(
    grid,
    threshold_ranges={
        "volume": (0.85, 0.98),
        "loss_ratio": (0.95, 1.10),
    },
    n_points_per_dim=15,
)

frontier.points      # pl.DataFrame: threshold_volume, threshold_loss_ratio,
                     #   total_objective, total_volume, total_loss_ratio,
                     #   lambda_volume, lambda_loss_ratio, iterations, converged
frontier.n_points    # int
```

### Frontier for Ratebook

The same frontier machinery works with the grouped solver. Each grid point runs a full grouped Lagrangian solve instead of an online solve. The warm-start ordering is equally effective because lambda proximity still holds.

---

## 5. Ratebook Optimisation — Coordinate Descent

### Problem

A ratebook has multiple rating factors (e.g. age band, region, vehicle group). Each factor has levels with associated multipliers. The optimiser must find the optimal multiplier for every level of every factor, subject to portfolio-level constraints.

The key challenge: when optimising one factor, each quote's other factors create a different **residual** (product of all other factor values). The same age_band factor value maps to different overall multiplier positions for different quotes. The solver must handle this correctly while remaining chunkable and fast.

### Decision

Use coordinate descent (CD): optimise one factor at a time using the grouped Lagrangian solver (with remapping, see section 3), cycling through factors until convergence.

### Why No Re-Scoring?

The scored grid contains pre-computed objective/constraint values at M overall multiplier steps. When CD adjusts a factor value, the grouped solver uses the **remapping** step to translate `residual × candidate_factor_value` to the nearest scored grid position. The pre-computed values are looked up directly — no model re-evaluation needed.

The cost per CD factor iteration is dominated by the grouped Lagrangian solve, which is the same complexity as an online solve pass: O(N × P) where P is the number of candidate factor values. For 5M quotes × 50 candidates, this is ~250M lookups + Lagrangian arithmetic — sub-second on modern hardware.

### Algorithm

```
Input:
  grid: QuoteGrid (N×M scored at overall multiplier steps)
  factors: FactorsFrame (N rows, F factor columns)
  factor_columns: list of factor specs (main effects + interactions)
  candidates: Vec<f32> (P,) — candidate factor values (e.g. [0.70, 0.72, ..., 1.40])
  constraints: constraint specs
  max_cd_iterations: int (default 3)

Initialise:
  factor_table[f][level] = 1.0  for all factors and levels
  overall_mult[i] = 1.0  for all quotes (product of all factor values)
  For each factor spec, build GroupMapping from factors DataFrame

CD outer loop (max_cd_iterations):
  For each factor_spec f in factor_columns:

    1. Compute residuals (chunkable, O(N)):
       For each quote i:
         residuals[i] = overall_mult[i] / factor_table[f][level_of(i, f)]

    2. Build GroupMapping for this factor (or interaction):
       - Main effect: group_of[i] = level index of quote i for factor f
       - Interaction: group_of[i] = composite level index

    3. Run grouped Lagrangian solve with remapping:
       result = solve_grouped(
           grid, group_mapping, residuals, candidates,
           constraints, config, lambdas=previous_lambdas
       )
       This is fully chunked internally (see section 3).

    4. Update factor table:
       For each group g:
         old_value = factor_table[f][label_of(g)]
         new_value = result.optimal_factor_values[g]
         factor_table[f][label_of(g)] = new_value

    5. Update overall multipliers (chunkable, O(N)):
       For each quote i:
         g = group_of[i]
         overall_mult[i] *= new_value[g] / old_value[g]

  Check CD convergence:
    If max change in any factor's values < tolerance → stop

Output:
  factor_table: {factor_name: {level: optimal_value}}
  Final lambdas
  Portfolio metrics (objective, constraints at final overall_mult)
```

### Chunked Execution Throughout

Every step of the CD algorithm is chunkable:

| Step | What it does | Memory per chunk |
|---|---|---|
| Residual computation | `overall_mult[i] / factor_value` | O(chunk_size) f32 |
| Grouped Lagrangian | Remap + Lagrangian + group accumulate | O(1) per quote (fused) |
| Group argmax | `argmax_j group_L[g,j]` | O(G × P) — tiny, ~20 KB |
| Factor table update | Update G values | O(G) — trivial |
| Overall mult update | `overall_mult[i] *= ratio` | O(chunk_size) f32 |

The `overall_mult` and `group_of` arrays (N f32 and N u32 respectively) persist across CD iterations — they're the ratebook's working state. Everything else is transient per chunk.

### Memory Profile (5M quotes, 10 factors, 50 steps)

| Allocation | Size | Lifetime |
|---|---|---|
| QuoteGrid (obj + 3 constraints) | ~4.0 GB | Permanent |
| Factor level indices (10 factors × N u32) | ~200 MB | Permanent |
| Overall multipliers (N f32) | ~20 MB | Permanent |
| Residuals (N f32) | ~20 MB | Per-factor, recomputed |
| group_L accumulators (G × P f64) | ~20 KB | Per-factor, reset |
| **Total** | **~4.25 GB** | |

Essentially the same as online, plus ~220 MB for the factor metadata.

### Wider Scored Grid for Ratebook

Because CD combines factor values with residuals (`residual × candidate`), the target overall multiplier can fall outside the original scored grid range. For example:
- Residual = 1.15 (other factors are net-positive)
- Candidate factor value = 1.15
- Target overall = 1.32, but scored grid max = 1.20

**Recommendation:** Score a wider grid for ratebook mode. If the online grid is [0.80, 1.20], use [0.60, 1.50] for ratebook. The price scenario node in haute should support configuring the range. A wider grid with the same step count (M=50) gives coarser resolution but covers more factor combinations. Alternatively, increase M (e.g. M=80 over [0.60, 1.50]) at the cost of more scoring upfront and more grid memory.

The solver reports a **clamp rate** (fraction of lookups that hit the grid boundary) so the user knows if the grid is too narrow.

### Interaction Support

- **1-way (main effect):** Group by a single factor column. G = number of unique levels.
- **2-way (interaction):** Group by the Cartesian product of two factor columns. G = L1 × L2 (or fewer if some combinations don't exist in the data). The residual for a 2-way interaction factor is the product of all OTHER factors (excluding both columns in the interaction).
- **N-way:** Same principle, composite key from N columns. G grows combinatorially but is bounded by the actual number of distinct combinations in the portfolio.

The user specifies which interactions to include. Structure selection (below) can discover them automatically.

### Output Format — RatingStep Entries

The ratebook output must integrate with haute's rating step nodes. A rating step node applies factors via a left-join lookup table. The optimiser output is formatted as rating step entries:

**1-way factor:**

| age_band | factor |
|---|---|
| 18-24 | 1.15 |
| 25-34 | 1.02 |
| 35-44 | 0.95 |
| 45-54 | 0.98 |
| 55+ | 1.08 |

**2-way interaction:**

| age_band | region | factor |
|---|---|---|
| 18-24 | London | 1.22 |
| 18-24 | South East | 1.10 |
| 25-34 | London | 1.05 |
| ... | ... | ... |

These are Polars DataFrames that can be directly used as rating step lookup tables in haute.

```python
result.to_rating_entries()
# Returns: dict[str, pl.DataFrame]
# Keys: factor spec name (e.g. "age_band", "age_band×region")
# Values: DataFrames in rating step format
```

---

## 6. Structure Selection

### Problem

The user may not know which factors and interactions improve the objective. Trying all possible combinations is expensive. We need an automated way to discover which factors matter.

### Decision

Implement structure selection as an optional step before coordinate descent. Two modes:

### Explicit Structure

The user specifies exactly which factors and interactions to include:

```python
result = ratebook_solver.solve(
    grid,
    factors=factors_df,
    factor_columns=[
        ["age_band"],                    # main effect
        ["region"],                      # main effect
        ["age_band", "region"],          # 2-way interaction
    ],
)
```

### Auto Structure

price-contour discovers the best structure:

```
Phase 1 — Screen main effects (reduced iterations, e.g. 10):
  For each factor column in factors_df:
    Build GroupMapping, run grouped solve
    Record: objective lift over baseline, level distinctness (variance of optimal multipliers)
  Rank by lift, keep factors above threshold (or top-N)

Phase 2 — Screen interactions (reduced iterations):
  For each pair of selected main effects:
    Build composite GroupMapping, run grouped solve
    Compare lift to sum of individual main-effect lifts (marginal gain)
    Check minimum cell volume (credibility) per composite level
  Keep interactions where marginal lift exceeds threshold

Phase 3 — Full CD with selected structure
```

### Key Properties

- **Screening is cheap.** Each screening solve uses 10 iterations (not 50) because we only need relative ranking, not exact multipliers.
- **Screening solves are independent.** They can run in parallel (across factors) with Rayon.
- **Cell volume checks prevent overfitting.** If a composite level (e.g. age 18-24 × rural Scotland) has too few quotes, the interaction term isn't credible.
- **The user can override.** Auto-selected structure is presented for review; the user can add or remove factors before the final CD run.

---

## 7. Banding Integration (Haute ↔ Price Contour)

### Context

Haute has banding nodes that define how continuous or categorical variables are grouped into rating factor levels. For example, a banding node might define:

- `driver_age` → `age_band`: 18-24, 25-34, 35-44, 45-54, 55+
- `postcode` → `region`: London, South East, Midlands, North, Scotland, Wales

The banding node outputs string-typed columns (e.g. `age_band = "25-34"`). Downstream rating step nodes use these as lookup keys for factor tables.

### Decision

The optimiser node in haute extracts available factor columns from upstream banding nodes. The user selects which factors to optimise in the optimiser UI. The factors DataFrame is built from the banding node outputs — it's the N-row DataFrame with just the `quote_id` and selected banding columns.

### Flow

```
[Banding Node]  →  factors_df (N rows: quote_id, age_band, region, ...)
                        ↓
[Model Score + Pipeline]  →  scored_df (N×M rows: quote_id, scenario_step, multiplier, objective, constraints)
                        ↓
[Optimiser Node]  ←  both DataFrames
    │
    │  Calls price-contour with scored_df (chunked) + factors_df (once)
    │
    ↓
  RatebookResult  →  to_rating_entries()  →  rating step lookup tables
                        ↓
[Rating Step Node]  ←  applies optimised factors via left-join
```

### Factor Level Source

The factor levels available in the optimiser come from the banding definitions, not from the data. Haute's `extractBandingLevels()` function scans upstream banding nodes and returns the defined levels for each factor. This ensures the optimiser knows all possible levels (including any with zero volume in the current portfolio).

### What price-contour Sees

price-contour receives the factors DataFrame and treats factor columns as opaque string labels. It doesn't know or care that they came from banding nodes — it just groups quotes by matching string values. The banding integration is purely a haute concern.

---

## 8. Module Structure (Planned)

### Rust Core (`crates/price-contour-core/src/`)

```
data/
  mod.rs
  grid.rs          # QuoteGrid (existing, from data.rs)
  builder.rs       # QuoteGridBuilder (new)
  types.rs         # ConstraintSpec, SolverConfig, results (existing, from data.rs)
  group.rs         # GroupMapping, FactorsFrame (new)

solver/
  mod.rs
  online.rs        # solve_online (existing)
  grouped.rs       # solve_grouped (new)
  lambda.rs        # Lambda update strategies (existing)
  apply.rs         # apply_lambdas (existing)

frontier/
  mod.rs           # multi-dimensional frontier sweep (new)
  ordering.rs      # nearest-neighbour warm-start path (new)
```

### PyO3 Bindings (`crates/price-contour/src/`)

```
lib.rs             # PyModule registration
solver_py.rs       # solve_online_py, solve_grouped_py (extend existing)
builder_py.rs      # PyQuoteGridBuilder (new)
frontier_py.rs     # frontier_py (new)
apply_py.rs        # apply_lambdas_py (existing)
```

### Python Layer (`python/price_contour/`)

```
__init__.py        # Public API exports
solver.py          # OnlineOptimiser (existing)
ratebook.py        # RatebookOptimiser — CD loop, structure selection (new)
apply.py           # ApplyOptimiser (new, extends current)
frontier.py        # EfficientFrontier types (new)
builder.py         # QuoteGridBuilder Python wrapper (new)
```

---

## 9. Serialisation & MLflow

### Problem

Two distinct serialisation needs:

1. **MLflow logging** — params, metrics, and artifacts for experiment tracking and comparison.
2. **Production application** — a self-contained artifact that can be loaded by a rating engine or `ApplyOptimiser` to apply the optimised result to new data.

These are different audiences. MLflow stores everything needed for experiment management (searchable params, comparable metrics). The production artifact stores only what's needed for application.

### Decision: Separate Concerns

**Metrics live in MLflow only.** The production artifact does not duplicate metrics (objective, uplift, clamp rate, etc.) — those are for experiment comparison, not application.

**Production artifacts are JSON files in a parameters folder.** One file per concern, human-readable, diffable, independently inspectable.

### Online Solver — Apply Artifact

The online solver's production artifact is a single `config.json` containing the lambdas and constraint config needed by `ApplyOptimiser`:

```
artifacts/online_v1/
  config.json
```

```json
{
  "lambdas": {"volume": 0.0, "loss_ratio": 52.76},
  "objective": "expected_income",
  "constraints": {"volume": {"min": 0.90}, "loss_ratio": {"max": 1.05}},
  "quote_id": "quote_id",
  "scenario_step": "scenario_step",
  "multiplier": "multiplier",
  "chunk_size": 500000
}
```

```python
# Save
applier = pc.ApplyOptimiser(lambdas=result.lambdas, ...)
applier.save("artifacts/online_v1/config.json")

# Load (in live pipeline)
applier = pc.ApplyOptimiser.load("artifacts/online_v1/config.json")
apply_result = applier.apply(new_df)
```

### Ratebook Solver — Parameters Folder

The ratebook solver's production artifact is a **parameters folder** with one JSON per factor plus a config file:

```
parameters/ratebook_v1/
  config.json
  region.json
  age_band.json
  vehicle_type.json
  region_age_band.json     # interaction (if present)
```

**config.json** — operational config only, no metrics:

```json
{
  "objective": "expected_income",
  "constraints": {"volume": {"min": 0.90}, "loss_ratio": {"max": 1.05}},
  "lambdas": {"volume": 0.0, "loss_ratio": 7.66},
  "factors": ["region", "age_band", "vehicle_type"],
  "factor_order": ["region", "age_band", "vehicle_type"]
}
```

**Each factor JSON** — self-describing lookup table:

```json
{
  "columns": ["region"],
  "table": {
    "London": 0.8653,
    "South East": 0.8400,
    "Midlands": 0.8245,
    "North": 0.8100,
    "Scotland": 0.8245,
    "Wales": 0.8653
  }
}
```

**Interaction factor** — compound keys, multiple columns:

```json
{
  "columns": ["region", "age_band"],
  "table": {
    "London:17-25": 1.02,
    "London:26-35": 0.98,
    "South East:17-25": 1.05
  }
}
```

#### Save / Load API

```python
# After solve
result = ratebook_solver.solve(df, factors)

# Save parameters folder
result.save("parameters/ratebook_v1/")

# Load and apply (in live pipeline or rating engine)
applier = pc.RatebookApplier.load("parameters/ratebook_v1/")
adjustments = applier.apply(new_factors_df)
# → DataFrame with [factor columns..., adjustment] per quote
```

#### Application Logic

`RatebookApplier.apply()` is a pure lookup + multiply:

1. Load config.json to know factor order
2. For each quote, look up its factor value from each factor JSON
3. Overall adjustment = product of all factor lookups
4. Return per-quote adjustment multiplier

No solver, no scored grid, no lambdas needed. The factor tables ARE the production artifact.

#### Key Properties

- **One file per factor** — independently inspectable, diffable across versions, swappable without re-optimising.
- **Factor order preserved** — the list in config.json encodes the CD iteration order (deterministic).
- **Human-readable JSON** — a pricing actuary can open `region.json` and see exactly what the optimiser decided.
- **Rating engine integration** — each factor JSON maps directly to a rating step lookup table. `to_rating_entries()` produces the same shape as Polars DataFrames.

### MLflow Logging

MLflow logging uses the `summary()` methods which produce params/metrics/artifacts dicts. The parameters folder is logged as an MLflow artifact directory.

#### Online

```python
summary = solver.summary(result)
mlflow.log_params(summary["params"])       # objective, max_iter, n_quotes, etc.
mlflow.log_metrics(summary["metrics"])     # total_objective, uplift_pct, lambdas, etc.
mlflow.log_dict(summary["artifacts"]["lambdas"], "lambdas.json")
mlflow.log_dict(summary["artifacts"]["config"], "config.json")
if summary["artifacts"]["convergence"] is not None:
    summary["artifacts"]["convergence"].write_parquet("/tmp/convergence.parquet")
    mlflow.log_artifact("/tmp/convergence.parquet")
```

#### Frontier

```python
summary = frontier_summary(frontier_result, selected_index=chosen_idx)
mlflow.log_params(summary["params"])       # frontier_n_points, selected_index
mlflow.log_metrics(summary["metrics"])     # selected_total_objective, thresholds, lambdas
summary["artifacts"]["frontier"].write_parquet("/tmp/frontier.parquet")
mlflow.log_artifact("/tmp/frontier.parquet")
```

#### Ratebook

```python
summary = ratebook_solver.summary(result)
mlflow.log_params(summary["params"])       # n_factors, candidate_range, cd_iterations, etc.
mlflow.log_metrics(summary["metrics"])     # total_objective, uplift_pct, clamp_rate, etc.

# Log the parameters folder as an artifact directory
result.save("/tmp/parameters/")
mlflow.log_artifacts("/tmp/parameters/", artifact_path="parameters")
```

The parameters folder appears in MLflow's artifact browser with each factor JSON visible and downloadable individually.

---

## 10. Optimiser Node — UI Configuration & Solver Arguments

This section maps every solver parameter to a UI element in haute's optimiser node, following the patterns established by ModellingConfig, BandingEditor, and RatingStepEditor.

### Node Inputs

The optimiser node has **one input edge** in online mode and **two input edges** in ratebook mode:

| Input | Source | Format | Required |
|---|---|---|---|
| Scored data | Upstream pipeline (model score → price scenario) | Long-format DataFrame (N×M rows) | Always |
| Factors data | Upstream banding node | One row per quote (N rows), string factor columns | Ratebook only |

### Section 1: Mode Selection

| Parameter | UI Element | Options | Default | Maps to |
|---|---|---|---|---|
| Mode | Radio group | Online, Ratebook | Online | Determines which solver class is used and which config sections are visible |

### Section 2: Column Mappings

Populated from `upstreamColumns` (discovered from the scored data input edge). Dropdowns filter by dtype — string columns for `quote_id`, numeric (f32) columns for objective/constraints.

| Parameter | UI Element | Filter | Default | Required |
|---|---|---|---|---|
| Quote ID column | Dropdown | String/Utf8 columns | `"quote_id"` | Yes |
| Scenario step column | Dropdown | Int32 columns | `"scenario_step"` | Yes |
| Multiplier column | Dropdown | Float32 columns | `"multiplier"` | Yes |
| Objective column | Dropdown | Float32 columns | `"expected_income"` | Yes |

### Section 3: Constraints

Dynamic form — user adds/removes constraints. Each constraint maps a column to a threshold spec. Follows the pattern of BandingEditor's factor tabs.

| Component | UI Element | Notes |
|---|---|---|
| Add constraint | Button (+ icon) | Adds a new constraint row |
| Constraint column | Dropdown (Float32 columns from upstream) | e.g. `volume`, `loss_ratio` |
| Direction | Toggle: Min / Max | Min = "at least X% of baseline", Max = "at most X% of baseline" |
| Threshold mode | Toggle: Relative / Absolute | Relative uses baseline ratio, absolute uses raw value |
| Threshold value | Number input | Relative: 0.0–2.0 (e.g. 0.90 = 90% of baseline). Absolute: free value |
| Remove | Delete button (per row) | |

Maps to: `constraints = {"volume": {"min": 0.90}, "loss_ratio": {"max": 1.05}}`

### Section 4: Solver Tuning (Collapsible, Defaults Fine)

| Parameter | UI Element | Range | Default | Notes |
|---|---|---|---|---|
| Max iterations | Number input | 10–500 | 50 | Increase if `converged=False` |
| Convergence tolerance | Number input (scientific) | 1e-8 – 1e-3 | 1e-6 | Lower = tighter, more iterations |
| Chunk size | Number input | 50,000 – 5,000,000 | 500,000 | Controls peak memory per iteration |
| Record history | Toggle | — | Off | Enables per-iteration convergence tracking |

### Section 5: Frontier Configuration (Collapsible)

Frontier generation is available in both online and ratebook modes.

| Parameter | UI Element | Range | Default | Notes |
|---|---|---|---|---|
| Enable frontier | Toggle | — | On | Whether to generate the efficient frontier |
| Points per dimension | Slider | 5–30 | 10 | Total grid = n^K. For K=2, n=10 → 100 solves |
| Threshold ranges | Per-constraint (min, max) inputs | Based on constraint direction | (0.85, 0.98) typical | Auto-populated from constraint specs. User adjusts range endpoints. |

After frontier generation, the user picks a point on the Pareto curve in the results panel. The selected point's lambdas become the deployed artifact.

### Section 6: Ratebook Configuration (Visible When Mode = Ratebook)

#### Factor Selection

Factor columns are discovered from the **factors input edge** (upstream banding node outputs). Follows the RatingStepEditor pattern — `extractBandingLevels(allNodes)` scans upstream banding nodes.

| Component | UI Element | Notes |
|---|---|---|
| Available factors | Multi-select checkbox list | All banding output columns from upstream. Each shows column name + number of levels. |
| Main effects | Selected factors list | Checked factors become main effects |
| Interactions | Pair builder | User selects pairs (or triples) from main effects to create interaction terms. Shows estimated cell count per interaction. |

Maps to: `factor_columns = [["age_band"], ["region"], ["age_band", "region"]]`

#### Structure Discovery Mode

| Parameter | UI Element | Options | Default |
|---|---|---|---|
| Structure mode | Radio group | Explicit (user-defined above), Auto-discover | Explicit |

When Auto-discover is selected, additional controls appear:

| Parameter | UI Element | Range | Default | Notes |
|---|---|---|---|---|
| Max interaction order | Dropdown | 1, 2, 3 | 2 | 1 = main effects only |
| Max main effects | Number input | 3–20 | 10 | Keep top N by lift |
| Max interactions to screen | Number input | 5–50 | 15 | Caps the number of pairs tested |
| Min cell volume | Number input | 50–10,000 | 500 | Credibility floor per level |
| Screening iterations | Number input | 5–50 | 10 | Reduced iterations for ranking |

#### CD and Factor Grid

| Parameter | UI Element | Range | Default | Notes |
|---|---|---|---|---|
| Max CD iterations | Number input | 1–10 | 3 | Outer loops cycling through factors |
| Factor value range | Min/Max inputs | 0.50–2.00 | 0.70 – 1.40 | Range of candidate factor values |
| Factor value steps | Number input | 10–100 | 50 | Number of candidate values in range |

Maps to: `candidates = linspace(0.70, 1.40, 50)`

**Wider scored grid note:** When ratebook mode is selected, the price scenario node upstream should use a wider multiplier range (e.g. [0.60, 1.50]) than online mode ([0.80, 1.20]). The optimiser UI should display a warning if the scored grid range looks too narrow relative to the factor value range — specifically, if `min(grid) > factor_min × 0.7` or `max(grid) < factor_max × 1.3`.

### Section 7: MLflow Logging (Collapsible)

Follows the ModellingConfig pattern exactly.

| Parameter | UI Element | Default | Notes |
|---|---|---|---|
| Experiment path | Text input | `"/optimisation"` | MLflow experiment path |
| Model name | Text input | `"optimiser_v1"` | Artifact name in MLflow |

### Section 8: Run Controls

| Component | UI Element | Behaviour |
|---|---|---|
| Run Optimisation | Primary button | Starts the batch solve. Disabled until required fields are set. |
| Progress | Progress bar + status text | Polls backend (same pattern as ModellingConfig training). Shows: iteration count, objective value, constraint satisfaction, elapsed time. |
| Cancel | Secondary button (during run) | Stops the solve early, returns best feasible solution found so far. |

### Results Panel (Post-Solve)

After the solve completes, the results panel shows multiple tabs (same pattern as ModellingConfig's post-training results):

| Tab | Content | Both modes? |
|---|---|---|
| **Summary** | Converged/iterations, total objective, baseline objective, uplift %, constraint totals vs thresholds, lambdas | Yes |
| **Frontier** | Interactive Pareto chart. User clicks/drags to select a point. Updates summary + impact tabs. | Yes |
| **Impact** | Before/after multiplier distribution (histogram). Per-segment breakdown if segments available. Mean/median/p5/p95 of optimal multipliers. | Yes |
| **Constraints** | Per-constraint: total vs threshold, slack, shadow price (lambda). Binding vs non-binding indicators. | Yes |
| **Factors** | Per-factor level table showing optimal multiplier values. Heatmap for 2-way interactions. Before/after comparison if re-running. | Ratebook only |
| **Convergence** | Line charts: objective per iteration, lambda trajectories, constraint totals per iteration. Only if `record_history=True`. | Yes |
| **Log to MLflow** | Button + experiment/model name fields. Logs params, metrics, artifacts. | Yes |

### Config TypedDict (Python Backend)

```python
class OptimiserConfig(TypedDict, total=False):
    # Mode
    mode: str  # "online" | "ratebook"

    # Column mappings
    quote_id: str
    scenario_step: str
    multiplier: str
    objective: str

    # Constraints
    constraints: dict[str, dict[str, float]]
    # e.g. {"volume": {"min": 0.90}, "loss_ratio": {"max": 1.05}}

    # Solver tuning
    max_iter: int
    tolerance: float
    chunk_size: int
    record_history: bool

    # Frontier
    frontier_enabled: bool
    frontier_n_points: int
    frontier_threshold_ranges: dict[str, tuple[float, float]]

    # Ratebook-specific
    factor_columns: list[list[str]] | None  # None = auto
    max_cd_iterations: int
    candidate_min: float
    candidate_max: float
    candidate_steps: int

    # Auto structure (when factor_columns is None)
    max_interaction_order: int
    max_main_effects: int
    max_interactions: int
    min_cell_volume: int
    screening_iterations: int

    # MLflow
    mlflow_experiment: str
    model_name: str
```

---

## 11. Implementation Order

### Phase 1: QuoteGridBuilder

- `QuoteGridBuilder` in Rust core
- PyO3 bindings (`PyQuoteGridBuilder`)
- Python wrapper with `append()` / `build()` API
- Modify `OnlineOptimiser.solve()` to accept `QuoteGrid` or `DataFrame`
- Tests: chunked ingestion produces same grid as one-shot

### Phase 2: Grouped Lagrangian Solver

- `GroupMapping` struct and builder
- `solve_grouped()` in Rust core with remapping — fused remap + Lagrangian + group accumulate inner loop
- Nearest-step binary search in sorted multiplier grid
- Clamp-rate tracking for out-of-range diagnostics
- `GroupedSolveResult` struct
- PyO3 bindings (accepts residuals + candidates + group mapping)
- Tests: grouped with residuals=1.0 and candidates=grid.multipliers = online solve; single group = portfolio-wide argmax; known remapping cases

### Phase 3: Multi-Dimensional Frontier

- `FrontierConfig`, `FrontierPoint`, `FrontierResult` structs
- Nearest-neighbour ordering in Rust
- Frontier sweep function (calls `solve_online` or `solve_grouped` per point)
- PyO3 bindings returning Polars DataFrame
- Python `frontier()` method on `OnlineOptimiser`
- Tests: 1D frontier matches manual sweep, warm-start reduces iterations

### Phase 4: Ratebook Optimiser (Python CD Loop)

- `RatebookOptimiser` class in Python
- CD loop: residual computation → `solve_grouped` with remapping → factor table update → overall mult update
- `overall_mult` and factor table tracking across CD iterations
- `to_rating_entries()` output formatter
- Structure selection (screening + auto-discovery)
- Tests: CD with one factor = single grouped solve, CD with two factors converges on known problem, clamp rate below threshold with appropriate grid width

### Phase 5: Config Serialisation & MLflow Extensions

- `frontier_summary()` and ratebook `summary()` methods
- `ApplyOptimiser.save()` / `.load()`
- Ratebook apply via factor table lookup
- Tests: round-trip save/load, summary dict structure

### Phase 6: Polish & PyPI

- Public API review and cleanup
- Documentation
- CI pipeline
- PyPI publish

---

## 12. Decisions Log

| # | Decision | Rationale |
|---|---|---|
| 1 | **GridBuilder for chunked ingestion** | haute's pipeline executor produces data in chunks. GridBuilder lets each chunk be ingested and freed, keeping peak Python memory low while accumulating compact f32 arrays in Rust. |
| 2 | **Two separate DataFrames for ratebook** | Factor columns are string-typed, quote-level. Duplicating them across M step rows wastes memory and pollutes the numeric grid. Passing them separately is cleaner. |
| 3 | **Grouped Lagrangian with remapping** | The grouped solver finds the best *factor value* per group, not the best *overall multiplier*. A remapping step translates `residual × candidate` to the nearest scored grid position per quote. This fuses into the inner loop with O(1) extra memory per quote. Lambda update code is shared with online. |
| 4 | **Fast CD without re-scoring** | The scored grid contains pre-computed values at overall multiplier steps. CD uses remapping to look up values at `residual × candidate_factor_value` — no model re-evaluation needed. Quantisation error from nearest-step lookup is ~0.4% (negligible). Wider scored grid recommended for ratebook. |
| 5 | **Multi-dimensional frontier with warm-start ordering** | Insurance problems typically have 2-3 constraints. Nearest-neighbour ordering through the threshold grid ensures each solve warm-starts from a nearby solution, reducing per-point iterations from ~50 to ~3-5. |
| 6 | **Banding integration via factors DataFrame** | Haute's banding nodes define factor levels. The factors DataFrame is built from banding outputs. price-contour treats factor columns as opaque strings — the banding integration is purely a haute concern. |
| 7 | **Output in rating step entries format** | `to_rating_entries()` produces DataFrames that slot directly into haute's rating step nodes as lookup tables. This closes the loop: banding defines levels → optimiser finds multipliers → rating step applies them. |
| 8 | **Structure selection as optional screening** | Auto-discovery adds ~30-50% overhead but saves the user from guessing which factors matter. Screening uses reduced iterations (10 vs 50) since only relative ranking matters. |
| 9 | **CD loop in Python, grouped solve in Rust** | The CD loop is simple orchestration (iterate over factors, call solver, check convergence). Putting it in Python keeps it readable and flexible. The expensive inner solve (grouped Lagrangian) runs in Rust. |
| 10 | **price-contour owns the full ratebook solve** | Coordinate descent, structure selection, and the grouped Lagrangian all live in price-contour. Haute provides DataFrames and consumes results — it doesn't need to understand the optimisation algorithm. |
| 11 | **Frontier works for both online and ratebook** | The frontier sweep calls either `solve_online` or `solve_grouped` per point. Same warm-start ordering, same result structure. The user gets frontier visualisation in both modes. |
| 12 | **Interaction support via composite group keys** | 2-way interactions create composite levels (e.g. `"25-34|London"`). The grouped solver doesn't know or care that the group key is composite — it just groups by matching strings. This naturally supports N-way interactions. |
| 13 | **f32 for all grid data, f64 for lambdas/totals** | Grid data (objective, constraints) uses f32 for half the memory. Lambdas and portfolio totals use f64 to avoid accumulation errors across millions of quotes. This matches the existing codebase. |
| 14 | **Column mappings via dropdowns, not text inputs** | Following haute's existing pattern (ModellingConfig, BandingEditor), column names are selected from `upstreamColumns` with dtype filtering. Prevents typos and makes schema mismatches impossible. |
| 15 | **Constraint builder as dynamic form** | Constraints are the most complex user input. A row-based builder (add/remove, column dropdown, direction toggle, threshold input) is more usable than raw JSON. Follows the BandingEditor factor-tabs pattern. |
| 16 | **Factor selection from upstream banding nodes** | Following RatingStepEditor's `extractBandingLevels()` pattern, factor columns are discovered from upstream banding nodes rather than typed manually. The user sees checkboxes with level counts, not raw column names. |
| 17 | **Frontier point selection in results panel** | The frontier is generated as part of the solve, then the user interactively picks a point on the Pareto curve. The selected point's lambdas are what gets logged to MLflow and deployed. This separates "explore the tradeoff space" from "commit to a solution". |
| 18 | **Async solve with polling** | The solve runs as an async backend job (same pattern as ModellingConfig training). The UI polls for progress (iteration count, objective, constraint satisfaction, elapsed time). User can cancel early and get the best feasible solution found so far. |
