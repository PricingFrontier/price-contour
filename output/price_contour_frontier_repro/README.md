# price-contour frontier convergence repro

This folder contains a standalone reproduction of an issue found when running Haute's optimiser UI against `price-contour==0.3.2`.

The important files are:

- `haute_price_contour_frontier_repro.parquet` - projected optimiser input from `rating/main.py`.
- `haute_price_contour_frontier_repro_stats.json` - schema and envelope stats from the exported parquet.
- `reproduce_frontier_issue.py` - standalone reproduction script. It imports only `polars` and `price_contour`.

## Data shape

The parquet contains only the columns passed to `price-contour`:

| column | dtype | role |
| --- | --- | --- |
| `quote_id` | `Categorical` | quote id |
| `scenario_index` | `Int32` | scenario index |
| `premium_multiplier` | `Float32` | scenario value |
| `expected_margin` | `Float32` | objective |
| `conversion_prediction` | `Float32` | constraint |

The dataset has:

- `100,000` quotes
- `21` scenarios per quote
- `2,100,000` rows
- no nulls in objective or constraint columns
- scenario values from `0.2` to `1.4`

The per-quote scenario envelope for `conversion_prediction` is roughly:

- min: `189.03745`
- max: `99,695.734` to `99,695.748`, depending on summation path/float precision

The configured frontier range from Haute was:

```python
{"conversion_prediction": (189.03746032714844, 99695.75)}
```

Note the final endpoint is about `0.0025` above the exact maximum observed by `price-contour` fixed-lambda application. That tiny endpoint rounding issue is not the main bug: the previous high frontier targets are clearly feasible and still fail by thousands.

## Reproduction

From this folder:

```powershell
python reproduce_frontier_issue.py haute_price_contour_frontier_repro.parquet
```

From the Haute repo root:

```powershell
uv run python output\price_contour_frontier_repro\reproduce_frontier_issue.py output\price_contour_frontier_repro\haute_price_contour_frontier_repro.parquet
```

The minimal `price-contour` call is:

```python
from price_contour import OnlineOptimiser, build_grid_from_parquet

grid = build_grid_from_parquet(
    "haute_price_contour_frontier_repro.parquet",
    ["conversion_prediction"],
    quote_id="quote_id",
    scenario_index="scenario_index",
    scenario_value="premium_multiplier",
    objective="expected_margin",
)

solver = OnlineOptimiser(
    objective="expected_margin",
    constraints={"conversion_prediction": {"min": 0.0}},
    max_iter=50,
    tolerance=1e-6,
)

base = solver.solve(grid)
frontier = solver.frontier(
    grid,
    threshold_ranges={"conversion_prediction": (189.03746032714844, 99695.75)},
    n_points_per_dim=15,
    initial_lambdas=base.lambdas,
)
print(frontier.points)
```

## Observed behaviour

With `max_iter=50`, only `3` of `15` points converge. Those first three converge because their thresholds are below the unconstrained optimum, so lambda remains `0`.

Selected points from `frontier.points`:

| point | target | actual | gap | lambda | converged |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | `189.037` | `21,438.160` | `+21,249.122` | `0.000` | true |
| 3 | `21,511.904` | `21,483.972` | `-27.933` | `0.128` | false |
| 7 | `49,942.394` | `42,671.466` | `-7,270.927` | `49.869` | false |
| 10 | `71,265.261` | `57,112.205` | `-14,153.055` | `95.172` | false |
| 13 | `92,588.128` | `68,225.912` | `-24,362.215` | `151.149` | false |
| 14 | `99,695.750` | `71,326.128` | `-28,369.622` | `172.537` | false |

Increasing `max_iter` helps only slowly:

| max_iter | converged points | final target | final actual | final gap | final lambda |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 3 / 15 | `99,695.750` | `71,326.128` | `-28,369.622` | `172.537` |
| 500 | 3 / 15 | `99,695.750` | `82,037.572` | `-17,658.178` | `294.655` |
| 2,000 | 5 / 15 | `99,695.750` | `86,260.252` | `-13,435.498` | `387.958` |
| 10,000 | 7 / 15 | `99,695.750` | `89,917.427` | `-9,778.323` | `525.394` |

Point 13 is the key non-endpoint example:

- target: `92,588.128`
- bisection via fixed-lambda application finds a satisfying lambda around `702.822`
- `frontier(max_iter=10_000)` only reaches lambda `393.095`
- actual is still `86,434.099`, missing by `6,154.029`

So this is not caused by the final target being infinitesimally above the max.

## Proof the targets are reachable

The script also uses `apply_from_grid` to binary-search lambdas. This proves that most of the high frontier targets are reachable with the same grid and same objective/constraint data.

Selected bisection results:

| target | lambda by bisection | actual at lambda | gap |
| ---: | ---: | ---: | ---: |
| `49,942.394` | `70.354` | `49,942.460` | `+0.067` |
| `71,265.261` | `172.132` | `71,265.320` | `+0.060` |
| `85,480.505` | `366.539` | `85,480.534` | `+0.029` |
| `92,588.128` | `702.822` | `92,588.167` | `+0.039` |

Fixed-lambda application also shows the upper end is close to reachable:

| lambda | total conversion |
| ---: | ---: |
| `0` | `21,438.160` |
| `172.537` | `71,326.128` |
| `294.655` | `82,037.572` |
| `1,000` | `94,957.873` |
| `5,000` | `99,183.780` |
| `10,000` | `99,575.371` |
| `20,000` | `99,681.761` |
| `110,000` | `99,695.748` |

## Interpretation

For this one-dimensional absolute `min` constraint, the lambda-to-total-constraint mapping is monotone non-decreasing. The current frontier solve appears to rely on the iterative Lagrangian solver's update schedule. For high thresholds, the lambda needed to satisfy the target is much larger than the frontier solver reaches within practical iteration counts.

The issue is therefore not Haute calculating the range incorrectly and not Haute passing the wrong columns. The same failure reproduces from a standalone parquet using only `price_contour`.

There are two separate behaviours worth fixing or making explicit:

1. High absolute frontier targets are feasible but the frontier solver's lambda search undershoots badly.
2. Some points can be close to or one-sided-satisfy the target but still report `converged=false`, suggesting the convergence criterion is not aligned with frontier target residual/satisfaction.

## Suggested price-contour fix direction

For single swept sum constraints, consider a dedicated frontier path based on fixed-lambda application:

1. Compute the unconstrained lambda-0 result.
2. Compute or estimate the per-quote achievable envelope.
3. For each target:
   - if lambda 0 already satisfies it, return lambda 0 and converged;
   - if the target is outside the envelope, return a loud infeasible/endpoint-clamped status rather than a misleading non-converged low-lambda point;
   - otherwise bracket lambda by exponential search in the correct sign direction;
   - binary-search lambda using `apply_from_grid` until the one-sided target residual is within tolerance or the interval is exhausted.
4. Emit the apply result at the chosen lambda as the frontier point.

Because quote choices are discrete, exact equality is not always possible. The convergence criterion should allow one-sided satisfaction for `min` constraints (`actual >= target`) and `max` constraints (`actual <= target`), with a target-unit residual tolerance and perhaps a separate flag for residual magnitude.

Acceptance criteria for this repro:

- Point 13 (`target=92588.12767573765`) should satisfy the target within a small residual and not be thousands short.
- The final endpoint should either be clamped to the true reachable max or clearly marked infeasible by about `0.0025`, not returned as a non-converged point `28k` short.
- Increasing `max_iter` should not be required to get a usable one-dimensional frontier.
