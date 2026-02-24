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

Price Contour finds optimal price multipliers across a portfolio of insurance risks subject to business constraints. Give it a scored dataset with objective and constraint values at discrete price points, and it returns the multiplier per quote that maximises your objective while respecting every constraint.

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
    constraints={"volume": {"min": 0.90}},  # retain at least 90% of baseline volume
    quote_id="quote_id",
    scenario_step="scenario_step",
    multiplier="multiplier",
)

result = optimiser.solve(df)

print(result.converged)        # True
print(result.iterations)       # 23
print(result.lambdas)          # {'volume': 0.147}
print(result.total_objective)  # 1_284_302.5

# Per-quote optimal multipliers as a Polars DataFrame
out = result.dataframe
print(out.head())
# ┌──────────┬──────────────┬────────────────────┬─────────────────────┬──────────────────┐
# │ quote_id │ optimal_step │ optimal_multiplier │ optimal_income      │ optimal_volume   │
# ╞══════════╪══════════════╪════════════════════╪═════════════════════╪══════════════════╡
# │ Q001     │ 14           │ 1.07               │ 42.30               │ 0.82             │
# │ Q002     │ 11           │ 0.98               │ 18.55               │ 0.91             │
# └──────────┴──────────────┴────────────────────┴─────────────────────┴──────────────────┘
```

---

## What it does

Price Contour operates on **pre-computed scenario data**. It does not fit models or generate demand curves. Upstream, your pricing pipeline scores every quote at a grid of price multipliers (e.g. 0.8, 0.85, 0.9, ..., 1.2) and computes what the expected income, volume, loss ratio, etc. would be at each point. Price Contour then selects the optimal multiplier per quote across the portfolio.

The input is a long-format Polars DataFrame:

| quote_id | scenario_step | multiplier | income | volume | loss_ratio |
|----------|---------------|------------|--------|--------|------------|
| Q001     | 0             | 0.80       | 85.2   | 0.95   | 0.62       |
| Q001     | 1             | 0.90       | 92.1   | 0.88   | 0.59       |
| Q001     | 2             | 1.00       | 100.0  | 0.80   | 0.60       |
| Q002     | 0             | 0.80       | 42.0   | 0.97   | 0.58       |
| ...      | ...           | ...        | ...    | ...    | ...        |

The output is one optimal multiplier per quote, chosen to maximise portfolio-level income while keeping portfolio-level volume above 90% of baseline (or whatever constraints you set).

---

## Three optimisation modes

### Online optimisation

Find the optimal multiplier per individual quote. Each quote independently picks its best price point, coordinated by shared Lagrange multipliers that enforce portfolio-level constraints.

```python
optimiser = pc.OnlineOptimiser(
    objective="income",
    constraints={"volume": {"min": 0.90}},
)
result = optimiser.solve(df)
```

### Ratebook optimisation

Find optimal rating factors across rating dimensions. Instead of individual multipliers, find the best factor value for each level of each rating factor (e.g. age band, region, vehicle power), applied uniformly to all quotes sharing that level.

```python
optimiser = pc.RatebookOptimiser(
    objective="income",
    constraints={"volume": {"min": 0.90}},
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
    constraints={"volume": {"min": 0.90}},
)
applier.save("config/applier.json")

# Later, in production:
applier = pc.ApplyOptimiser.load("config/applier.json")
live_result = applier.apply(df_single_quote)
optimal_multiplier = live_result.dataframe["optimal_multiplier"][0]
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
# ┌──────────────────┬─────────────────┬──────────────┬───────────────┬────────────┬───────────┐
# │ threshold_volume │ total_objective │ total_volume │ lambda_volume │ iterations │ converged │
# ╞══════════════════╪═════════════════╪══════════════╪═══════════════╪════════════╪═══════════╡
# │ 0.85             │ 1_350_102       │ 0.851        │ 0.089         │ 18         │ true      │
# │ 0.86             │ 1_342_891       │ 0.861        │ 0.102         │ 21         │ true      │
# │ ...              │ ...             │ ...          │ ...           │ ...        │ ...       │
# └──────────────────┴─────────────────┴──────────────┴───────────────┴────────────┴───────────┘
```

Adjacent points are warm-started from each other (nearest-neighbour traversal of the threshold grid), so the full frontier solves much faster than running each point independently.

---

## Constraint format

Constraints are specified as a dictionary. Keys are column names in your DataFrame, values specify the direction and threshold relative to the baseline (the portfolio totals at multiplier = 1.0):

```python
constraints = {
    "volume": {"min": 0.90},            # portfolio volume >= 90% of baseline
    "loss_ratio": {"max": 1.05},        # portfolio loss ratio <= 105% of baseline
    "premium": {"min_abs": 1_000_000},  # absolute: portfolio premium >= 1M
}
```

---

## Incremental grid building

For large datasets that don't fit in memory at once, build the internal grid incrementally:

```python
builder = pc.QuoteGridBuilder(
    ["volume", "loss_ratio"],
    quote_id="quote_id",
    scenario_step="scenario_step",
    multiplier_col="multiplier",
    objective="income",
)

for chunk in data_source.iter_chunks(100_000):
    builder.append(chunk)

grid = builder.build()
result = optimiser.solve(grid)
```

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
Maximise    sum_i  objective(quote_i, multiplier_i)
Subject to  sum_i  constraint_k(quote_i, multiplier_i) >= threshold_k   for all k
            multiplier_i in {discrete grid}
```

This is a combinatorial problem (each quote picks from M discrete multipliers). Lagrangian dual decomposition relaxes the coupling constraints into the objective using dual variables (lambdas), decomposing it into N independent per-quote subproblems:

```
For fixed lambdas:
    Each quote picks:  argmax_m [ objective(i, m) + sum_k lambda_k * constraint_k(i, m) ]

These are independent and embarrassingly parallel.
```

The outer loop updates lambdas via the subgradient method with adaptive step sizes, iterating until all constraints are satisfied and lambdas converge.

### Performance

The Rust core uses:

- **Quote-major memory layout** - each quote's M scenario values are contiguous, optimising the per-quote argmax inner loop for cache locality
- **Rayon parallelism** - the argmax across quotes is parallelised within chunks of 4096 quotes
- **Chunked processing** - large portfolios are processed in chunks (default 500K quotes) to bound memory usage
- **Adaptive step scaling** - per-constraint scale factors normalise for differing magnitudes, so the algorithm works equally well for constraints ranging from 0.1 to 1,000,000
- **Lambda averaging** - smooths the oscillations inherent in discrete Lagrangian relaxation where all quotes can flip simultaneously

### Ratebook mode

For ratebook optimisation, coordinate descent iterates over rating factors. For each factor, a grouped Lagrangian solve finds the best discrete factor value per group (e.g. per age band), with the individual quote multiplier computed as the product of all factor values times a per-quote residual. The inner grouped solve uses the same Lagrangian machinery with remapping to the nearest grid point.

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
│           └── grid_py.rs         # QuoteGrid bindings
├── python/
│   └── price_contour/
│       ├── solver.py              # OnlineOptimiser
│       ├── ratebook.py            # RatebookOptimiser + RatebookResult
│       ├── apply.py               # ApplyOptimiser
│       ├── frontier.py            # FrontierResult helpers
│       └── builder.py             # QuoteGridBuilder wrapper
└── tests/
    └── python/                    # Integration tests
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
| `solve(df_or_grid, *, lambdas=None)` | Run full optimisation. Returns `SolveResult`. |
| `frontier(df_or_grid, *, threshold_ranges, n_points_per_dim=10)` | Sweep the efficient frontier. Returns `FrontierResult`. |
| `summary(result)` | Package result into MLflow-ready `params`, `metrics`, `artifacts` dicts. |
| `config_dict()` | Serialisable solver configuration. |

### RatebookOptimiser

| Method | Description |
|---|---|
| `solve(df_or_grid, factors, *, factor_columns=None)` | Run ratebook optimisation via coordinate descent. Returns `RatebookResult`. |
| `summary(result)` | Package result into MLflow-ready dicts. |

### ApplyOptimiser

| Method | Description |
|---|---|
| `apply(df)` | Single-pass scoring with fixed lambdas. Returns `ApplyResult`. |
| `save(path)` | Save config + lambdas to JSON. |
| `ApplyOptimiser.load(path)` | Load from saved JSON. |

### QuoteGridBuilder

| Method | Description |
|---|---|
| `append(df)` | Add a chunk of quotes. |
| `build()` | Finalise and return a `QuoteGrid`. |

### SolveResult

| Property | Type | Description |
|---|---|---|
| `converged` | `bool` | Whether the solver converged. |
| `iterations` | `int` | Number of iterations taken. |
| `lambdas` | `dict[str, float]` | Final Lagrange multipliers per constraint. |
| `total_objective` | `float` | Portfolio-level objective at optimal solution. |
| `total_constraints` | `dict[str, float]` | Portfolio-level constraint totals. |
| `baseline_objective` | `float` | Objective at multiplier = 1.0. |
| `baseline_constraints` | `dict[str, float]` | Constraints at multiplier = 1.0. |
| `dataframe` | `pl.DataFrame` | Per-quote results with optimal multipliers. |
| `history` | `list[dict] \| None` | Per-iteration convergence records (if `record_history=True`). |
| `n_quotes` | `int` | Number of quotes in the grid. |
| `n_steps` | `int` | Number of multiplier steps. |
| `multipliers` | `list[float]` | The multiplier grid. |

### RatebookResult

| Property | Type | Description |
|---|---|---|
| `factor_tables` | `dict[str, dict[str, float]]` | Factor name to level-value mapping. |
| `converged` | `bool` | Whether coordinate descent converged. |
| `cd_iterations` | `int` | Coordinate descent iterations. |
| `clamp_rate` | `float` | Fraction of remappings that hit a grid boundary. |
| `save(path)` | | Save factor tables to a directory (one JSON per factor). |
| `to_rating_entries()` | `dict[str, pl.DataFrame]` | Convert to rating-step DataFrames. |

---

## License

Price Contour is licensed under the [GNU Affero General Public License v3.0](LICENSE).
