# Performance optimisation methodology

A transferable process for finding and shipping high-leverage perf wins on Rust+Python (or any heavy-language-bound) libraries. Distilled from the work that took the ratebook efficient-frontier sweep from 62.06 s → 1.51 s (41×) across nine fixes — see `MEMORY.md` in `~/.claude/projects/.../memory/` for the per-fix records.

The specific patterns at the end are useful, but the *methodology* is the transferable part.

---

## 1. Establish a reproducible baseline before touching code

Before any optimisation: a standalone repro with a single timer, a fixed dataset, and a one-line invocation. Without this, every change becomes an argument about whether timings are comparable.

Concretely:
- A script that runs the hot path end-to-end and prints one wall-clock number per phase.
- A fixed input small enough to iterate quickly but representative (the 100k × 21 × 3 ratebook profile took ~60 s baseline, fast enough for tens of iterations per session).
- The script reads no env vars, takes no flags that change the algorithm — only optional args for things like dataset size.

If you don't have this for your library, build it before anything else.

## 2. Profile before forming hypotheses

Run `cProfile` once before the first fix and once after each major fix. Every time, the profile reshuffles what's actually dominant.

**Don't optimise based on intuition.** I would have spent a day on the helper FFI cost (`compute_residuals_py` etc.) before profiling showed it was 7% of the time and the kernel was 95%. Fix 9 (CD loop in Rust) only became the right move *after* eight earlier fixes had collapsed the kernel cost down to where the helpers were proportionally large.

Tools that worked:
- `cProfile` for Python-entry profiling — gave me the "this Rust function is 95% of time" picture quickly.
- Per-phase timing via the user's repro script — let me see when the bottleneck shifted between phases (precompute vs inner iters vs reconstruct).
- Run-to-run distributions, not single runs — a 30 s benchmark with 30% variance is uninterpretable from one sample.

For Rust-only kernels you'd reach for `cargo flamegraph`, `perf record`, or `samply`. I didn't end up needing them because the Python entry profile was always informative enough.

## 3. Spawn a team of agents in parallel

This is the move that compounds the most. For each major decision point, spawn 4 agents in parallel:

| Agent | Role | What it adds |
|---|---|---|
| **Critique** | Tear apart the current state / proposed plan | Catches assumptions you've internalised |
| **Library research** | Find what mature libraries do for this *exact* problem shape | Borrows from people who've solved harder versions |
| **Algorithmic brainstorm** | Generate 8–10 *meaningfully different* angles | Forces breadth before depth |
| **Code-level reader** | Read the actual hot files line-by-line, surface concrete file:line wins | Catches things the higher-level agents won't see |

**The pattern that matters: convergent insights from independent agents = high confidence.** The biggest single win (Fix 4, affine λ decomposition, 3× speedup) came up *independently* in the algorithmic agent and the critique agent, framed differently. When two agents who didn't talk to each other point at the same thing, take it seriously. When only one suggests something, treat it as a hypothesis to test.

**Self-contained prompts.** Each agent starts with no memory of the conversation. Every prompt should include:
- Current state with concrete numbers ("3.33 s frontier, 14 ms per call"), not "we've optimised some things".
- What's been tried and rejected (so they don't re-suggest dead ends).
- Profile breakdown (so they target the actual bottleneck).
- The exact question — ranked shortlist with effort + risk + speedup estimates.

The "what's been tried and rejected" is critical. After Fix 2 (warm-start step offset) was reverted, every subsequent agent prompt included "Tried and reverted: ... saved as memory note". Without that, two of three subsequent rounds re-suggested it.

## 4. Verify, don't trust — at every step

Three different verification levels, picked deliberately per fix:

- **Bit-identical** for pure refactors (Fix 1's λ-independent cache). New `test_remap_cache_matches_legacy_bit_identical` test compared exact f64 bits between the optimised and legacy paths. If a refactor claims to be "just faster, same answer," prove it — bit-equality testing catches mistakes a perf benchmark won't.
- **Tolerance-based** for algorithmically equivalent rearrangements (Fix 4's affine decomposition reorders the f64 reduction). Test asserts `assert_abs_diff_eq!(epsilon = scale × 1e-12)`.
- **End-to-end** when results legitimately change (Fix 3's λ extrapolation finds different local optima). Full test suite still passes + final values match within reasonable tolerance.

**Always run the full test suite at every step.** Cheap insurance against accidental algorithmic changes.

## 5. Be willing to revert

Fix 2 (subgradient step-size offset for warm starts) was suggested by the very first critique agent with high confidence. I implemented it. Three variants tested, every one regressed: smaller subgradient steps inside the inner solver produced less-converged factor tables, which inflated the *outer* CD iteration count — a multiplicative slowdown.

I reverted it cleanly and **saved a memory note** so future sessions wouldn't waste time re-attempting it.

If you're not willing to revert, you'll keep bad fixes alive because of sunk cost. Make the revert easy (small commits, atomic changes) and treat reverting as a normal outcome, not a failure.

## 6. Watch for variance

I made a measurement mistake mid-process: declared "Phase 1 micro-optimisations regressed performance" based on 5 runs that turned out to be cold-cache runs. After 10+ warm runs the actual baseline was 24% faster than my "baseline" measurement. The optimisations were neutral, not regressing.

Rules I now follow:
- Run 10+ samples, not 5.
- Discard the first 1–2 runs (cache warmup, JIT, allocator settling).
- Compare medians and ranges, not means.
- If you change two things at once, you can't tell which moved the needle. Keep changes atomic.

## 7. Save findings to memory between sessions

Each major fix and each rejected hypothesis got a `~/.claude/projects/.../memory/*.md` entry with frontmatter (`name`, `description`, `type: project | feedback`). The "tried and reverted" memory for Fix 2 is what stopped subsequent agents re-suggesting it for the rest of the project.

The pattern I used: lead with the rule/finding, then a `**Why:**` line and a `**How to apply:**` line. Future-you (or future agents) needs the *why* to judge edge cases, not just the verdict.

## 8. Re-profile after every major fix — the bottleneck shifts

After each fix, the dominant cost is different. The order I tackled things wasn't planned upfront — each fix's profile told me what to do next.

```
Pre-Fix 1:  95% solve_grouped (per-iter Lagrangian gather)
After Fix 1:  95% solve_grouped (now per-call precompute, gather is cached)
After Fix 4:  67% solve_grouped, 25% helpers (affine decomp made inner loop ~free)
After Fix 5:  67% solve_grouped (helper FFI cleaner, but still ~25%)
After Fix 6:  60% solve_grouped (totals from A/B, no per-iter gather)
After Fix 9:  98% in run_cd_pass_py (entire CD pass = one Rust call)
```

**Each fix unmasks the next.** Don't try to plan a 10-fix roadmap upfront — the world will reshape after each one. Plan one fix ahead, ship it, re-profile, plan the next.

## 9. Know when to stop

I stopped at 41× for two reasons: the next levers all involved big surgery (multi-frontier-point parallelism, GPU offload, f32 accumulation with precision risk), and the workload didn't need more. Remaining levers were documented in memory for a future session.

Symptoms you're hitting diminishing returns:
- Each fix takes more code than the last for the same speedup.
- The remaining bottleneck is "memory bandwidth" or "cache miss latency" — these need hardware-level attention.
- Risk of changing the answer (precision, parallelism reordering) starts to dominate.

---

## Patterns that recurred (transferable across libraries)

These came up enough times that I'd look for them on day one of a new optimisation pass.

### Cache λ-independent work
Anywhere an inner loop iterates with a parameter that updates while *part* of the work doesn't depend on the parameter, lift the invariant work out. Per-iter cost goes from full work to incremental work.

> Fix 1: precomputed `target_idx` (nearest-step indices) once per `solve_grouped` call instead of every subgradient iteration. ~1.7×.

### Affine decomposition
When per-iter cost is `f(state, λ)` and `f` is linear in `λ` (or any small parameter), precompute the linear coefficients and replace the iteration with a tiny SAXPY.

> Fix 4: `group_l[g, j] = a_table[g, j] + Σ_c sign_c · λ_c · b_table[(g, j), c]`. Inner loop O(n_quotes × n_candidates × (1+K)) → O(n_groups × n_candidates × (1+K)). ~3×.

This is the highest-leverage pattern when it applies — look for it whenever a hot loop has a "current parameter" structure with linear dependence.

### Eliminate per-iter gathers via aggregate sums
If a per-iter loop sums gathered values over a fixed grouping, precompute the per-group sums once. Per-iter becomes O(n_groups), not O(n_total).

> Fix 6: `total_objective = Σ_g a_table[g, j_star]` from cached A/B instead of per-quote re-gather every iter. Reconstruct collapsed from O(n_quotes) per iter to O(n_groups).

### Cross-FFI persistent state via PyClass
When two FFI calls share large data (e.g., 100k strings as group labels), wrap it in a `#[pyclass]` and pass by reference instead of marshalling per call. PyO3's element-by-element `Vec<String>` extraction is shockingly slow at 100k elements; the canonical workaround is `#[pyclass]` + an `Arc<...>` member.

> Fix 5: `FactorContext` PyClass holding `Arc<GroupMapping>`. Helpers FFI dropped 4×.

### Move loops to the heavy-language side
A loop that does N round-trips of M-element data ferries N×M bytes through FFI overhead. Move the loop to the heavy language and amortise to one round-trip.

> Fix 9: ported the entire CD outer loop into Rust as `run_cd_pass_py`. ~1.7× from this alone, on top of everything else.

### Two-pointer for monotone gather
When per-iter binary searches sweep targets that turn out to be monotone, replace with a linear advance.

> Fix 7: `targets = res × candidates[j]` is monotone in `j` for sorted candidates and non-negative residuals. Replaced n_candidates × log(n_steps) binary searches per quote with a single linear scan. ~25% on the precompute kernel.

### Thread-local buffer pool
For transient allocations >1 MB, the allocator goes through `mmap`/`munmap`, microseconds per call. A `thread_local! { static SCRATCH: RefCell<Vec<...>> }` returned via `Drop` skips this. Cheap addition, surprising win at high call frequency.

> Fix 1's pool: `TARGET_IDX_SCRATCH` for the 20 MB step-index buffer. Steady-state sweeps avoid the per-call mmap.

### Frontier-level / orchestrator-level hoist
When per-call work depends only on inputs that don't vary across the outer loop, hoist it. The orchestrator knows what's invariant; the per-call function doesn't.

> Fix 8: built `factor_contexts` once at the top of `frontier()` and passed via private kwarg to every `self.solve()` call. ~10%.

### Verification levels
Pick deliberately per fix:
- Bit-identical (`a.to_bits() == b.to_bits()`) for refactors that claim to compute exactly the same f64.
- ULP tolerance (`assert_abs_diff_eq!(epsilon = scale × 1e-12)`) for algorithmically equivalent rearrangements that drift in the last bits.
- End-to-end test green for changes that legitimately alter the answer (different local optima, different convergence trajectory).

---

## A concrete sequence to apply on a new library

If I were starting this exercise on a new library tomorrow:

1. **Hour 1:** Build the repro. One script, one timer, fixed dataset, deterministic.
2. **Hour 2:** Run cProfile + flamegraph. Print the top 10 cumulative-time entries. Where is 80% of the time?
3. **Hour 3:** Spawn the four-agent review with the profile data. Read the 4 reports.
4. **Hour 4:** Implement the highest-confidence-lowest-risk fix. Verify (tests + benchmark). Re-profile.
5. **Day 2:** Repeat. Save memory notes for failed attempts. Track the speedup ladder.
6. **When the bottleneck has moved layers** (kernel → FFI → orchestration → algorithm), pause and re-do the four-agent review with the *new* profile. The recommendations change as the shape changes.
7. **Stop when each fix is < 1.5× and > 100 LOC.** Diminishing returns territory.

The 41× wasn't 9 plans executed. It was 9 *one-step-ahead* fixes, each surfaced by re-profiling, re-spawning agents, and trusting convergent recommendations over single-source ones.

---

## Cross-references

Per-fix details live in the project memory:

- `feedback_subgradient_offset_failed.md` — the one that didn't work, kept so it doesn't get re-attempted.
- `project_ratebook_frontier_extrapolation.md` — first-order λ extrapolation between frontier points (Fix 3).
- `project_grouped_solver_affine_decomp.md` — affine A/B decomposition (Fix 4), the highest-leverage single fix.
- `project_factor_context_caching.md` — `FactorContext` PyClass + frontier-level hoist (Fix 5 + Fix 8).
- `project_grouped_solver_totals_and_two_pointer.md` — totals from A/B + two-pointer remap (Fix 6 + Fix 7).
- `project_cd_loop_in_rust.md` — entire CD pass moved into one Rust call (Fix 9).
