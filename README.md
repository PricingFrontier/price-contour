<div align="center">

# Price Contour

### High-performance insurance price optimisation via Lagrangian dual decomposition.

<br>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Rust](https://img.shields.io/badge/core-Rust-DEA584?style=flat-square&logo=rust&logoColor=white)](https://rust-lang.org)
[![Polars](https://img.shields.io/badge/data-Polars-CD792C?style=flat-square)](https://pola.rs)
[![AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)

</div>

---

Price Contour finds optimal price scenario values across a portfolio of insurance risks subject to business constraints. Give it a scored dataset with objective and constraint values at discrete price points, and it returns the scenario value per quote that maximises your objective while respecting every constraint.

The core algorithm is Lagrangian dual decomposition, implemented in Rust for speed and exposed to Python via zero-copy Polars DataFrames. A portfolio of 1M+ risks solves in seconds.

---

## Quick start

```bash
uv add price-contour
```

```python
import polars as pl
import price_contour as pc

# Long-format DataFrame: one row per (quote, price_scenario)
# with pre-computed objective and constraint values
df = pl.read_parquet("scored_quotes.parquet")

optimiser = pc.OnlineOptimiser(
    objective="income",
    constraints={"volume": {"min_pct": 0.90}},  # retain at least 90% of baseline volume
    quote_id="quote_id",
    scenario_index="scenario_index",
    scenario_value="scenario_value",
)

result = optimiser.solve(df)

print(result.converged)        # True
print(result.iterations)       # 23
print(result.lambdas)          # {'volume': 0.147}
print(result.total_objective)  # 1_284_302.5

# Per-quote optimal scenario values as a Polars DataFrame
out = result.dataframe
print(out.head())
# ┌──────────┬──────────────┬────────────────────┬─────────────────────┬──────────────────┐
# │ quote_id │ optimal_step │ optimal_scenario_value │ optimal_income      │ optimal_volume   │
# ╞══════════╪══════════════╪════════════════════╪═════════════════════╪══════════════════╡
# │ Q001     │ 14           │ 1.07               │ 42.30               │ 0.82             │
# │ Q002     │ 11           │ 0.98               │ 18.55               │ 0.91             │
# └──────────┴──────────────┴────────────────────┴─────────────────────┴──────────────────┘
```

---

## What it does

Price Contour operates on **pre-computed scenario data**. It does not fit models or generate demand curves. Upstream, your pricing pipeline scores every quote at a grid of price scenario values (e.g. 0.8, 0.85, 0.9, ..., 1.2) and computes what the expected income, volume, loss ratio, etc. would be at each point. Price Contour then selects the optimal scenario value per quote across the portfolio.

The input is a long-format Polars DataFrame:

| quote_id | scenario_index | scenario_value | income | volume | loss_ratio |
|----------|----------------|----------------|--------|--------|------------|
| Q001     | 0             | 0.80       | 85.2   | 0.95   | 0.62       |
| Q001     | 1             | 0.90       | 92.1   | 0.88   | 0.59       |
| Q001     | 2             | 1.00       | 100.0  | 0.80   | 0.60       |
| Q002     | 0             | 0.80       | 42.0   | 0.97   | 0.58       |
| ...      | ...           | ...        | ...    | ...    | ...        |

The output is one optimal scenario value per quote, chosen to maximise portfolio-level income while keeping portfolio-level volume above 90% of baseline (or whatever constraints you set).

---

## Three optimisation modes

### Online optimisation

Find the optimal scenario value per individual quote. Each quote independently picks its best price point, coordinated by shared Lagrange multipliers that enforce portfolio-level constraints.

```python
optimiser = pc.OnlineOptimiser(
    objective="income",
    constraints={
        "volume": {"min_pct": 0.90},                # sum constraint
        "loss_ratio": {                              # ratio constraint
            "numerator":   "incurred",
            "denominator": "premium",
            "max":         0.65,
        },
    },
)
result = optimiser.solve(df)
print(result.lambdas)            # {'volume': 0.147, 'loss_ratio': 1.21}
print(result.total_constraints)  # {'volume': 5400.0, 'loss_ratio': 0.6498}
```

Both sum and ratio constraints work in all three optimisation modes (online, ratebook, apply) and in the efficient-frontier sweep.

### Ratebook optimisation

Find optimal rating factors across rating dimensions. Instead of individual scenario values, find the best factor value for each level of each rating factor (e.g. age band, region, vehicle power), applied uniformly to all quotes sharing that level.

```python
optimiser = pc.RatebookOptimiser(
    objective="income",
    constraints={"volume": {"min_pct": 0.90}},
    factor_columns=[["age_band"], ["region"], ["vehicle_power"]],
)

result = optimiser.solve(df, factors=factor_df)

print(result.factor_tables)
# {'age_band': {'18-25': 1.15, '26-35': 1.02, '36-50': 0.95, '51+': 0.98},
#  'region': {'London': 1.08, 'South East': 1.01, 'North': 0.93},
#  'vehicle_power': {'Low': 0.97, 'Medium': 1.0, 'High': 1.06}}

# Save to disk
result.save("parameters/")

# Convert to rating-step DataFrames
tables = result.to_rating_entries()
```

### Live scoring with stored lambdas

Apply pre-computed Lagrange multipliers to new quotes in a single forward pass, with no iteration. Use this in production to score individual quotes using lambdas learned from a batch solve.

```python
# Batch solve (offline)
result = optimiser.solve(df_portfolio)
lambdas = result.lambdas

# Live scoring (per-quote, no iteration)
applier = pc.ApplyOptimiser(
    lambdas=lambdas,
    objective="income",
    constraints={"volume": {"min_pct": 0.90}},
)
applier.save("config/applier.json")

# Later, in production:
applier = pc.ApplyOptimiser.load("config/applier.json")
live_result = applier.apply(df_single_quote)
optimal_scenario_value = live_result.dataframe["optimal_scenario_value"][0]
```

---

## Efficient frontier

Sweep constraint thresholds to generate the Pareto frontier - the trade-off curve between your objective and constraints. Each point on the frontier is a full portfolio solve at a different constraint target.

```python
frontier = optimiser.frontier(
    df,
    threshold_ranges={"volume": (0.85, 1.0)},
    n_points_per_dim=20,
)

# DataFrame with one row per frontier point
print(frontier.points)
# ┌──────────────────┬─────────────────┬──────────────┬───────────────┬────────────┬───────────┬─────────┬─────────────────┐
# │ threshold_volume │ total_objective │ total_volume │ lambda_volume │ iterations │ converged │ sv_mean │ sv_pct_increase │
# ╞══════════════════╪═════════════════╪══════════════╪═══════════════╪════════════╪═══════════╪═════════╪═════════════════╡
# │ 0.85             │ 1_350_102       │ 0.851        │ 0.089         │ 18         │ true      │ 1.04    │ 0.62            │
# │ 0.86             │ 1_342_891       │ 0.861        │ 0.102         │ 21         │ true      │ 1.03    │ 0.58            │
# │ ...              │ ...             │ ...          │ ...           │ ...        │ ...       │ ...     │ ...             │
# └──────────────────┴─────────────────┴──────────────┴───────────────┴────────────┴───────────┴─────────┴─────────────────┘
```

Adjacent points are warm-started from each other (nearest-neighbour traversal of the threshold grid), so the full frontier solves much faster than running each point independently. Each point also includes scenario value distribution statistics (`sv_mean`, `sv_std`, percentiles, `sv_pct_increase`/`sv_pct_decrease`).

**Sweeping a ratio target** — declare the constraint with `None` so the constructor doesn't fix it, then supply the range to `frontier()`:

```python
optimiser = pc.OnlineOptimiser(
    objective="income",
    constraints={
        "loss_ratio": {
            "numerator":   "incurred",
            "denominator": "premium",
            "max":         None,       # frontier supplies the target
        },
    },
)
frontier = optimiser.frontier(
    df,
    threshold_ranges={"loss_ratio": (0.55, 0.75)},
    n_points_per_dim=10,
)
# points["threshold_loss_ratio"] = [0.55, 0.572, ..., 0.75]  (user units, verbatim)
# points["total_loss_ratio"]     = actual Σ incurred / Σ premium at each optimum
```

**Mixed sweep** — sweep multiple constraints at once via the cartesian product:

```python
frontier = optimiser.frontier(
    df,
    threshold_ranges={
        "volume":     (8000, 12000),    # absolute units
        "loss_ratio": (0.55, 0.75),     # absolute ratio targets
    },
    n_points_per_dim=10,
)
# 10 × 10 = 100 frontier points
```

Constraints with numeric thresholds may be omitted from `threshold_ranges` — they are held fixed at the constructor value across the sweep. `None` thresholds *must* have a range entry.

---

## Constraint format

Constraints are specified as a dictionary. There are two shapes:

**Sum constraints** apply to a single column. The dict key is the column name in your DataFrame, the value specifies direction and threshold. Use ``min`` / ``max`` for absolute thresholds and ``min_pct`` / ``max_pct`` for thresholds expressed as a fraction of baseline (the portfolio totals at scenario_value = 1.0):

```python
constraints = {
    "volume":  {"min_pct": 0.90},     # portfolio volume >= 90% of baseline
    "premium": {"min": 1_000_000},    # absolute: portfolio premium >= 1M
    "claims":  {"max_pct": 1.05},     # portfolio claims <= 105% of baseline
}
```

**Ratio constraints** apply to a ratio of two summed columns (e.g. loss ratio = `Σ incurred / Σ premium`). The dict key is a display label (does NOT need to be a column); `numerator` and `denominator` name the columns:

```python
constraints = {
    "loss_ratio": {
        "numerator":   "incurred",
        "denominator": "premium",
        "max":         0.65,           # portfolio loss ratio <= 0.65
    },
    "combined_ratio": {
        "numerator":   "claims_plus_expenses",
        "denominator": "premium",
        "max_pct":     1.10,           # <= 110% of baseline combined ratio
    },
}
```

Internally, ratio constraints are linearised as `Σ (num − L·denom) ≤ 0` and handed to the same Lagrangian solver. Setting `Σ_baseline denom == 0` raises `ValueError` for `_pct` modes (baseline ratio undefined). If `Σ_optimum denom == 0` at the chosen step set, the ratio reported in `total_constraints[label]` and `summary()` is `nan` (sentinel; the divide is undefined, not silently zero).

**`None` thresholds** mark frontier-only constraints — the threshold is supplied by the sweep range:

```python
constraints = {
    "loss_ratio": {
        "numerator":   "incurred",
        "denominator": "premium",
        "max":         None,           # frontier supplies the target
    },
}

frontier = optimiser.frontier(
    df,
    threshold_ranges={"loss_ratio": (0.55, 0.75)},
    n_points_per_dim=10,
)
```

`solve()` rejects `None` thresholds; `frontier()` requires a `threshold_ranges` entry for every `None` constraint. Numeric-threshold constraints are optional in `threshold_ranges` — omitted ones are held fixed at their constructor value across the sweep.

`points["threshold_<name>"]` reports the user-supplied range value verbatim (absolute units for `min`/`max`, fractions of baseline for `min_pct`/`max_pct`); `points["total_<name>"]` reports the actual aggregate at the optimum (the actual ratio for ratio constraints).

---

## Direct Parquet loading

For large datasets, build the internal grid directly from a Parquet file without materialising a DataFrame in Python memory:

```python
grid = pc.build_grid_from_parquet(
    "scored_quotes.parquet",
    constraint_columns=["volume", "loss_ratio"],
    objective="income",
)
result = optimiser.solve(grid)
```

For parquets that exceed available memory in their raw form, use the streaming variant. The IO buffer is bounded by `chunk_size`; the file is read in row slices via Polars' `with_slice` pushdown so only the row groups overlapping each slice are deserialised, and column projection means only the four schema columns plus the requested constraint columns are decoded:

```python
grid = pc.build_grid_from_parquet_chunked(
    "huge_scored_quotes.parquet",
    constraint_columns=["volume", "loss_ratio"],
    chunk_size=500_000,         # rows per IO slice; rounded down to a multiple of n_steps
    objective="income",
    # n_steps=20,               # optional: lock upfront if your first slice could be partial
)
result = optimiser.solve(grid)
```

The final `QuoteGrid` is still O(n_quotes × n_steps × n_columns × 4 bytes) — that's inherent to the solver's flat data layout — but the parquet decode buffer never exceeds `chunk_size` rows. Use this when the parquet itself doesn't fit in RAM, not as a way to avoid loading the grid.

---

## Incremental grid building

For datasets streamed from upstream pipelines (e.g. when chunks arrive out-of-order or before the full dataset is materialised anywhere), build the grid incrementally:

```python
builder = pc.QuoteGridBuilder(
    ["volume", "loss_ratio"],
    quote_id="quote_id",
    scenario_index="scenario_index",
    scenario_value="scenario_value",
    objective="income",
    # n_steps=20,               # optional: lock upfront for streaming sources
)

for chunk in upstream:          # any iterable of pl.DataFrame
    builder.append(chunk)

grid = builder.build()
result = optimiser.solve(grid)
```

**Per-chunk contract:** each chunk's rows must already be grouped by `quote_id` (each quote occupies `n_steps` contiguous rows in `scenario_index` order). Within a chunk this is validated row-by-row (including a `scenario_value` consistency check against the canonical grid). Across chunks the order is arbitrary — the builder performs an in-place sort by `quote_id` at `build()` time using cycle-following permutation, so peak memory does not double during the sort. Duplicate `quote_id`s across all appended chunks are detected and reported with both append-order indices.

The optional `n_steps` kwarg lets streaming pipelines that may receive a partial first chunk lock the contract upfront, skipping the auto-detection probe.

---

## Streaming apply to disk

For live scoring on inputs too large to hold in RAM, stream a parquet through `apply` and write per-quote results to a parquet output one row group per chunk:

```python
result = pc.apply_lambdas_to_parquet_chunked(
    parquet_in="huge_scored_quotes.parquet",
    parquet_out="scored_results.parquet",
    lambdas={"volume": 0.147, "loss_ratio": 1.21},
    constraints={
        "volume": {"min_pct": 0.90},
        "loss_ratio": {"max_pct": 1.05},
    },
    chunk_size=500_000,
)

# Aggregate totals on the result; per-quote rows are in the output parquet.
print(result.total_objective)         # 1_284_302.5
print(result.total_constraints)       # {'volume': 5400.0, 'loss_ratio': 0.6498}
print(result.output_path)             # 'scored_results.parquet'

# Read back per-quote results lazily.
opt = pl.scan_parquet(result.output_path)
```

The **whole-portfolio** `optimal_steps` array is never materialised — only one chunk's `optimal_steps` is alive at a time (`chunk_size / n_steps` entries), and gets dropped along with the chunk's mini-grid after the row group has been written. Aggregate totals accumulate in f64 across chunks. On any error the partial output is best-effort deleted so callers never observe a corrupt artefact, and the input/output paths are checked for equality so the input parquet can't be silently overwritten. Lambda keys not matching any constraint are rejected up front (matching `ApplyOptimiser`). Ratio constraints are rejected on this path — use `ApplyOptimiser.apply(df)` on a DataFrame instead, since the per-chunk mini-grid can't carry the raw numerator/denominator columns.

---

## MLflow integration

Both `OnlineOptimiser` and `RatebookOptimiser` produce MLflow-ready summaries:

```python
result = optimiser.solve(df)
summary = optimiser.summary(result)

import mlflow
mlflow.log_params(summary["params"])
mlflow.log_metrics(summary["metrics"])
mlflow.log_dict(summary["artifacts"]["lambdas"], "lambdas.json")
mlflow.log_dict(summary["artifacts"]["config"], "config.json")
```

---

## How it works

### The algorithm

Price Contour solves the constrained optimisation problem:

```
Maximise    sum_i  objective(quote_i, scenario_value_i)
Subject to  sum_i  constraint_k(quote_i, scenario_value_i) >= threshold_k   for all k
            scenario_value_i in {discrete grid}
```

This is a combinatorial problem (each quote picks from M discrete scenario values). Lagrangian dual decomposition relaxes the coupling constraints into the objective using dual variables (lambdas), decomposing it into N independent per-quote subproblems:

```
For fixed lambdas:
    Each quote picks:  argmax_m [ objective(i, m) + sum_k lambda_k * constraint_k(i, m) ]

These are independent and embarrassingly parallel.
```

The outer loop updates lambdas via the subgradient method with adaptive step sizes, iterating until all constraints are satisfied and lambdas converge.

### Performance

The Rust core uses:

- **Quote-major memory layout** - each quote's M scenario values are contiguous, optimising the per-quote argmax inner loop for cache locality
- **Rayon parallelism** - the argmax across quotes is parallelised in grain sizes of 4096 quotes
- **Adaptive step scaling** - per-constraint scale factors normalise for differing magnitudes, so the algorithm works equally well for constraints ranging from 0.1 to 1,000,000
- **Lambda averaging** - smooths the oscillations inherent in discrete Lagrangian relaxation where all quotes can flip simultaneously

### Ratebook mode

For ratebook optimisation, coordinate descent iterates over rating factors. For each factor, a grouped Lagrangian solve finds the best discrete factor value per group (e.g. per age band), with the individual quote scenario value computed as the product of all factor values times a per-quote residual. The inner grouped solve uses the same Lagrangian machinery with remapping to the nearest grid point.

---

## Architecture

```
price-contour/
├── crates/
│   ├── price-contour-core/        # Pure Rust: algorithms, data structures, solver
│   │   └── src/
│   │       ├── data.rs            # QuoteGrid, SolverConfig, SolveResult, GroupMapping
│   │       ├── solver/
│   │       │   ├── online.rs      # Lagrangian dual decomposition
│   │       │   ├── grouped.rs     # Grouped solve (ratebook inner loop)
│   │       │   ├── argmax.rs      # Per-quote Lagrangian argmax (parallel)
│   │       │   ├── lambda.rs      # Subgradient lambda updates
│   │       │   └── apply.rs       # Fixed-lambda forward pass
│   │       ├── frontier.rs        # Efficient frontier sweeping
│   │       ├── constants.rs       # Solver defaults
│   │       └── error.rs           # Error types
│   └── price-contour/             # PyO3 bindings (thin wrappers)
│       └── src/
│           ├── solver_py.rs       # DataFrame ingestion + solve
│           ├── grouped_py.rs      # Grouped solve bindings
│           ├── apply_py.rs        # Apply bindings
│           ├── frontier_py.rs     # Frontier bindings
│           ├── builder_py.rs      # QuoteGridBuilder bindings
│           ├── grid_py.rs         # QuoteGrid bindings
│           └── parquet_grid_py.rs # Parquet → QuoteGrid loader
├── python/
│   └── price_contour/
│       ├── solver.py              # OnlineOptimiser, ratio linearisation, validation
│       ├── ratebook.py            # RatebookOptimiser + RatebookResult
│       ├── apply.py               # ApplyOptimiser + apply_from_grid
│       ├── frontier.py            # FrontierResult helpers + frontier_summary
│       ├── builder.py             # QuoteGridBuilder wrapper
│       ├── _ratio_results.py      # Shared ratio reporting (actual ratios + column stitching)
│       └── _frontier_helpers.py   # Shared frontier orchestrator (used by online + ratebook)
├── tests/
│   └── python/                    # Integration tests
├── notebooks/                     # Demo notebooks
├── docs/                          # Design documentation
└── scripts/                       # Utility scripts
```

The pure-Rust core (`price-contour-core`) has no Python dependencies and can be tested independently with `cargo test`. The PyO3 crate (`price-contour`) is a thin binding layer that converts between Polars DataFrames and the internal `QuoteGrid` representation with zero-copy where possible.

---

## Development

```bash
# Clone
git clone https://github.com/PricingFrontier/price-contour.git
cd price-contour

# Install in development mode (compiles Rust, links Python)
uv sync --all-groups
maturin develop

# Run Rust tests
cargo test

# Run Python tests
pytest

# Rebuild after Rust changes
maturin develop
```

**Requirements:** Rust toolchain (stable), Python 3.10+, maturin.

---

## API reference

### OnlineOptimiser

| Method | Description |
|---|---|
| `solve(df_or_grid, *, lambdas=None)` | Run full optimisation. Returns `SolveResult`. Ratio constraints require a DataFrame (the linearisation needs raw numerator/denominator columns); a pre-built `QuoteGrid` with ratio constraints raises `ValueError`. |
| `frontier(df_or_grid, *, threshold_ranges, n_points_per_dim=10, initial_lambdas=None)` | Sweep the efficient frontier. Returns `FrontierResult`. Numeric thresholds are optional in `threshold_ranges` (held fixed if omitted); `None` thresholds require a range. |
| `summary(result)` | Package result into MLflow-ready `params`, `metrics`, `artifacts` dicts. |
| `config_dict()` | Serialisable solver configuration. |

### RatebookOptimiser

| Method | Description |
|---|---|
| `solve(df_or_grid, factors, *, factor_columns=None, lambdas=None)` | Run ratebook optimisation via coordinate descent. Returns `RatebookResult`. |
| `frontier(df_or_grid, factors, *, threshold_ranges, n_points_per_dim=5, factor_columns=None, initial_lambdas=None)` | Sweep the efficient frontier via coordinate descent at each threshold. Returns `FrontierResult`. |
| `summary(result)` | Package result into MLflow-ready dicts. |

### ApplyOptimiser

| Method | Description |
|---|---|
| `apply(df)` | Single-pass scoring with fixed lambdas. Returns `ApplyResult`. For ratio constraints, `min_pct`/`max_pct` resolve `L = pct × baseline_LR` from the **apply-time** DataFrame (live-scoring contract), not the solve-time baseline. |
| `save(path)` | Save config + lambdas to JSON. Ratio specs round-trip verbatim. |
| `ApplyOptimiser.load(path)` | Load from saved JSON. Rejects unknown keys. |

### QuoteGridBuilder

| Method | Description |
|---|---|
| `QuoteGridBuilder(constraint_columns, *, quote_id, scenario_index, scenario_value, objective, n_steps=None)` | Construct a builder. `n_steps` may be passed upfront to skip auto-detection from the first chunk — useful for streaming sources where the first chunk may be partial. |
| `append(df)` | Add a chunk of quotes. Rows must be grouped by `quote_id` with `scenario_index` running `0..n_steps` in order. Per-row validation rejects layout violations and `scenario_value` drift across chunks. |
| `build()` | Finalise and return a `QuoteGrid`. Sorts by `quote_id` in-place via cycle-following permutation (no 2× memory peak). Rejects duplicate `quote_id`s with both append-order indices in the error. |

### SolveResult

| Property | Type | Description |
|---|---|---|
| `converged` | `bool` | Whether the solver converged. |
| `iterations` | `int` | Number of iterations taken. |
| `lambdas` | `dict[str, float]` | Final Lagrange multipliers (shadow prices) per constraint. |
| `total_objective` | `float` | Portfolio-level objective at optimal solution. |
| `total_constraints` | `dict[str, float]` | Portfolio-level constraint totals. |
| `baseline_objective` | `float` | Objective at scenario_value = 1.0. |
| `baseline_constraints` | `dict[str, float]` | Constraints at scenario_value = 1.0. |
| `dataframe` | `pl.DataFrame` | Per-quote results with optimal scenario values. |
| `history` | `list[dict] \| None` | Per-iteration convergence records (if `record_history=True`). |
| `n_quotes` | `int` | Number of quotes in the grid. |
| `n_steps` | `int` | Number of scenario value steps. |
| `scenario_values` | `list[float]` | The scenario value grid. |
| `grid` | `QuoteGrid` | The internal grid (reusable for subsequent solves or apply). |

### ApplyResult

| Property | Type | Description |
|---|---|---|
| `total_objective` | `float` | Portfolio-level objective. |
| `total_constraints` | `dict[str, float]` | Portfolio-level constraint totals. |
| `baseline_objective` | `float` | Objective at scenario_value = 1.0. |
| `baseline_constraints` | `dict[str, float]` | Constraints at scenario_value = 1.0. |
| `lambdas` | `dict[str, float]` | Applied Lagrange multipliers. |
| `dataframe` | `pl.DataFrame` | Per-quote results with optimal scenario values. |

### ChunkedApplyResult

Returned by `apply_lambdas_to_parquet_chunked`. Carries the same aggregate totals as `ApplyResult` but the per-quote rows live only in the output parquet — only one chunk's `optimal_steps` (`chunk_size / n_steps` entries) is alive at any time, then dropped after the row group is written.

| Property | Type | Description |
|---|---|---|
| `total_objective` | `float` | Portfolio-level objective at the optimum (summed across chunks in f64). |
| `total_constraints` | `dict[str, float]` | Portfolio-level constraint totals. |
| `baseline_objective` | `float` | Objective at scenario_value = 1.0. |
| `baseline_constraints` | `dict[str, float]` | Constraints at scenario_value = 1.0. |
| `lambdas` | `dict[str, float]` | Applied Lagrange multipliers. |
| `output_path` | `str` | Path to the streamed-output parquet. Read back via `pl.read_parquet` or `pl.scan_parquet`. |

### FrontierResult

| Property | Type | Description |
|---|---|---|
| `points` | `pl.DataFrame` | One row per frontier point with `threshold_*`, `total_objective`, `total_*`, `lambda_*`, `iterations`, `converged`, and scenario value statistics (`sv_mean`, `sv_std`, `sv_min`, `sv_p5`–`sv_p95`, `sv_max`, `sv_pct_increase`, `sv_pct_decrease`). |
| `n_points` | `int` | Number of frontier points. |

### RatebookResult

| Property | Type | Description |
|---|---|---|
| `factor_tables` | `dict[str, dict[str, float]]` | Factor name to level-value mapping. |
| `lambdas` | `dict[str, float]` | Final Lagrange multipliers. |
| `total_objective` | `float` | Portfolio-level objective at optimal solution. |
| `total_constraints` | `dict[str, float]` | Portfolio-level constraint totals. |
| `baseline_objective` | `float` | Objective at scenario_value = 1.0. |
| `baseline_constraints` | `dict[str, float]` | Constraints at scenario_value = 1.0. |
| `converged` | `bool` | Whether coordinate descent converged. |
| `cd_iterations` | `int` | Coordinate descent iterations. |
| `clamp_rate` | `float` | Fraction of remappings that hit a grid boundary. |
| `per_factor_results` | `list[GroupedSolveResult]` | Per-factor inner solve results. |
| `save(path)` | | Save factor tables to a directory (one JSON per factor). |
| `to_rating_entries()` | `dict[str, pl.DataFrame]` | Convert to rating-step DataFrames. |

### Utility functions

| Function | Description |
|---|---|
| `build_grid_from_parquet(path, constraint_columns, *, ...)` | Build a `QuoteGrid` directly from a Parquet file. Loads the projected columns whole; column projection prunes everything outside `constraint_columns` + the four schema columns. Sum constraints only — ratio constraints require a DataFrame. |
| `build_grid_from_parquet_chunked(path, constraint_columns, chunk_size, *, n_steps=None, ...)` | Stream a Parquet file in fixed-size row slices via Polars' `with_slice` pushdown. Memory peak for the parquet decode buffer is bounded by `chunk_size`; the final `QuoteGrid` is still O(total_rows). `chunk_size` is rounded down to a multiple of `n_steps` so every slice ends on a quote boundary. Use when the parquet itself doesn't fit in RAM. |
| `apply_lambdas_to_parquet_chunked(parquet_in, parquet_out, lambdas, constraints, chunk_size, *, n_steps=None, ...)` | Stream a parquet through `apply` and write per-quote results to `parquet_out` one row group per chunk. Returns `ChunkedApplyResult` with aggregate totals; per-quote rows live in the output parquet. The input/output paths are checked for equality (refuses to overwrite the input), and any error best-effort-deletes the partial output. |
| `apply_from_grid(grid, lambdas, constraints)` | Single-pass Lagrangian apply on an existing `QuoteGrid`. Returns `ApplyResult`. Sum constraints only; ratio constraints raise `ValueError` (use `ApplyOptimiser.apply(df)` on a DataFrame instead — the grid path can't carry numerator/denominator columns for linearisation). |
| `frontier_summary(frontier_result, selected_index)` | Package a frontier result into MLflow-ready `params`, `metrics`, `artifacts` dicts. |

---

## License

Price Contour is licensed under the [GNU Affero General Public License v3.0](LICENSE).
