"""Generate synthetic insurance pricing dataset for testing price_contour.

Produces a long-format Polars DataFrame matching the schema in architecture.md:
  quote_id, scenario_index, scenario_value, expected_income, volume, loss_ratio

Usage:
    python scripts/generate_test_data.py
    python scripts/generate_test_data.py --n-quotes 100000 --n-steps 41 --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate synthetic test quotes")
parser.add_argument("--n-quotes", type=int, default=1_000_000, help="Number of quotes")
parser.add_argument("--n-steps", type=int, default=21, help="Number of scenario value steps")
parser.add_argument("--mult-lo", type=float, default=0.80, help="Lowest scenario value")
parser.add_argument("--mult-hi", type=float, default=1.20, help="Highest scenario value")
parser.add_argument("--seed", type=int, default=12345, help="Random seed")
parser.add_argument(
    "--output",
    type=str,
    default="data/test_quotes.parquet",
    help="Output parquet path (relative to repo root)",
)
args = parser.parse_args()

N = args.n_quotes
M = args.n_steps
SEED = args.seed
OUTPUT = Path(__file__).resolve().parent.parent / args.output

rng = np.random.default_rng(SEED)

print(f"Generating {N:,} quotes × {M} steps = {N * M:,} rows …")
t0 = time.perf_counter()

# ---------------------------------------------------------------------------
# 1.  Scenario value grid (shared across all quotes)
# ---------------------------------------------------------------------------

scenario_values = np.linspace(args.mult_lo, args.mult_hi, M, dtype=np.float32)

# ---------------------------------------------------------------------------
# 2.  Per-quote random characteristics  (N,)
# ---------------------------------------------------------------------------

base_premium = rng.lognormal(mean=7.0, sigma=0.6, size=N).astype(np.float32)
base_loss_cost = base_premium * rng.uniform(0.55, 0.75, size=N).astype(np.float32)
elasticity = rng.uniform(1.5, 5.0, size=N).astype(np.float32)
midpoint = rng.normal(1.0, 0.03, size=N).astype(np.float32)

# ---------------------------------------------------------------------------
# 3.  Expand to (N, M) via broadcasting and compute derived columns
# ---------------------------------------------------------------------------

# shapes: base_premium[:, None] → (N,1);  scenario_values[None, :] → (1,M)
m = scenario_values[None, :]  # (1, M)

bp = base_premium[:, None]      # (N, 1)
blc = base_loss_cost[:, None]   # (N, 1)
el = elasticity[:, None]        # (N, 1)
mp = midpoint[:, None]          # (N, 1)

# Conversion probability — logistic curve, decreasing with price
conversion = (1.0 / (1.0 + np.exp(el * (m - mp)))).astype(np.float32)

# Small per-cell noise to prevent perfectly smooth curves
noise_income = rng.normal(1.0, 0.005, size=(N, M)).astype(np.float32)
noise_lr = rng.normal(1.0, 0.005, size=(N, M)).astype(np.float32)

# Expected income = base_premium × scenario_value × conversion_prob
expected_income = (bp * m * conversion * noise_income).astype(np.float32)

# Volume = conversion probability (the retention/conversion metric)
volume = conversion  # already f32

# Loss ratio = (base_loss_cost / (base_premium × scenario_value)) × adverse selection factor
# Adverse selection: as price rises, good risks leave, pushing loss ratio up
adverse_selection = (1.0 + 0.1 * (m - 1.0)).astype(np.float32)
loss_ratio = ((blc / (bp * m)) * adverse_selection * noise_lr).astype(np.float32)

print(f"  Arrays computed in {time.perf_counter() - t0:.1f}s")

# ---------------------------------------------------------------------------
# 4.  Flatten to long format and build Polars DataFrame
# ---------------------------------------------------------------------------

t1 = time.perf_counter()

# Quote IDs: Q0000001 .. Q{N}
quote_ids = np.arange(1, N + 1)
# Repeat each quote_id M times
quote_id_col = np.repeat(quote_ids, M)
# Format as strings
quote_id_str = [f"Q{qid:07d}" for qid in quote_id_col]

# Scenario indices: tile [0..M-1] N times
scenario_index_col = np.tile(np.arange(M, dtype=np.int32), N)

# Scenario value: tile the grid N times
scenario_value_col = np.tile(scenario_values, N)

# Flatten row-major (quote-major): each quote's M steps are contiguous
expected_income_col = expected_income.ravel()
volume_col = volume.ravel()
loss_ratio_col = loss_ratio.ravel()

print(f"  Flattened in {time.perf_counter() - t1:.1f}s")

t2 = time.perf_counter()

df = pl.DataFrame(
    {
        "quote_id": quote_id_str,
        "scenario_index": scenario_index_col,
        "scenario_value": scenario_value_col,
        "expected_income": expected_income_col,
        "volume": volume_col,
        "loss_ratio": loss_ratio_col,
    },
    schema={
        "quote_id": pl.Utf8,
        "scenario_index": pl.Int32,
        "scenario_value": pl.Float32,
        "expected_income": pl.Float32,
        "volume": pl.Float32,
        "loss_ratio": pl.Float32,
    },
)

print(f"  DataFrame built in {time.perf_counter() - t2:.1f}s")

# ---------------------------------------------------------------------------
# 5.  Write parquet
# ---------------------------------------------------------------------------

t3 = time.perf_counter()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.write_parquet(OUTPUT, compression="zstd")
file_size_mb = OUTPUT.stat().st_size / (1024 * 1024)
print(f"  Parquet written in {time.perf_counter() - t3:.1f}s → {OUTPUT}")
print(f"  File size: {file_size_mb:.0f} MB")

# ---------------------------------------------------------------------------
# 6.  Verification / summary statistics
# ---------------------------------------------------------------------------

print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
print(f"Rows: {len(df):,}")
print(f"Columns: {df.columns}")
print(f"\nSchema:\n{df.schema}\n")
print(df.describe())

# Spot-check: first quote
q1 = df.filter(pl.col("quote_id") == "Q0000001")
print(f"\n--- Quote Q0000001 ({len(q1)} rows) ---")
print(q1)

# Check peaked shape: expected_income should rise then fall
incomes = q1["expected_income"].to_list()
peak_idx = incomes.index(max(incomes))
print(f"  Peak expected_income at step {peak_idx} (scenario_value={q1['scenario_value'][peak_idx]:.2f})")
if peak_idx > 0 and peak_idx < M - 1:
    print("  ✓ Income has interior peak (not monotonic)")
else:
    print("  ⚠ Income peak at boundary — may still be valid for extreme elasticity")

# Check volume monotonically decreases (ignoring tiny noise violations)
volumes = q1["volume"].to_list()
mono_violations = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i - 1] + 0.01)
if mono_violations == 0:
    print("  ✓ Volume approximately monotonically decreasing")
else:
    print(f"  ⚠ Volume has {mono_violations} monotonicity violations > 0.01")

# Portfolio-level summary at scenario_value ≈ 1.0
baseline_step = int(np.argmin(np.abs(scenario_values - 1.0)))
baseline = df.filter(pl.col("scenario_index") == baseline_step)
print(f"\n--- Portfolio summary at step {baseline_step} (scenario_value={scenario_values[baseline_step]:.2f}) ---")
print(f"  Mean expected_income: {baseline['expected_income'].mean():.1f}")
print(f"  Mean volume:          {baseline['volume'].mean():.4f}")
print(f"  Mean loss_ratio:      {baseline['loss_ratio'].mean():.4f}")
print(f"  Total expected_income: {baseline['expected_income'].sum():.0f}")
print(f"  Total volume:          {baseline['volume'].sum():.0f}")
