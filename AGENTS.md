# Engineering priorities

- Prioritise numerical correctness, maintainability, and engineering quality over quick fixes or band-aids.
- Build new functionality consistently with the existing codebase. Reuse and extend existing abstractions when that produces a clearer design.
- DRY is important: flag and remove repetition, including logic duplicated between Rust and Python (the Rust version is canonical).
- Well-tested code is non-negotiable; err towards more edge cases, not fewer.
- Engineered enough: neither fragile and hacky nor prematurely abstracted. Prefer explicit over clever.
- Do not add speculative or silent fallbacks. Invalid input and unexpected states fail loudly with an actionable message; never return a default (0.0, an empty list, NaN) that masks an error.
- Preserve existing user changes and avoid unrelated edits.
- Make sure changes to behaviour are defined in the design documents (below) before changing code.

# Design principles

- **Rust computes, Python orchestrates.** Hot loops and data-parallel work live in `crates/price-contour-core/`. The PyO3 bindings in `crates/price-contour/` are thin: type conversion and delegation only. `python/price_contour/` owns the user-facing API, validation, and result presentation.
- **One code path.** A quantity (a baseline, a step rule, a total) is computed in one place and reused. Parallel implementations drift.
- **Deterministic, reproducible results.** The same inputs give bit-identical outputs: ordered dicts and columns, deterministic reductions, documented tie rules.
- **Explicit contracts.** Result shapes, column names and dtypes are documented and pinned by tests; nothing a consumer needs is private, positional or inferred.

Design documents, updated before the code they describe: `README.md` (the public API), `docs/architecture.md`, `docs/DESIGN_DECISIONS.md`, and `docs/optimisation_design.md`.

# GitHub access

- In the managed sandbox, GitHub CLI credential storage may be inaccessible and
  `gh auth status` can falsely report that the active token is invalid.
- Always run `gh auth status` and GitHub network operations outside the sandbox
  (using the normal escalation mechanism) before concluding that authentication
  is missing or invalid.
- If a sandboxed GitHub authentication check fails, immediately repeat it
  outside the sandbox. Do not ask the user to re-authenticate based only on the
  sandboxed result.

# Models and delegation

- Claude Code sessions run on Opus 5.5 for all work: judgment, implementation,
  and any subagent. There are no cheaper worker tiers to route through.
- Keep work in the main session. Spawn a subagent only for independent work
  that saves meaningful wall-clock time, such as a broad read-only search. Do
  not spawn one for a task a few direct tool calls can finish.
- Give a subagent a self-contained prompt: exact scope, inputs, constraints,
  expected output, and verification command. Subagents do not spawn further
  agents, and never run more than two at once.
- The main session inspects every subagent diff and its evidence itself, and
  owns planning, test design, integration, and the completion decision.
- Do not enable Fast mode for routine repository work. Use it only when the user
  explicitly prioritises latency over cost.

# Fix and tweak workflow

1. Establish a narrow scope and preserve unrelated user changes. Inspect the relevant code and define expected behaviour, risks, acceptance criteria, and a verification strategy before editing.
2. For a bug, reproduce it with the smallest failing regression test before implementing the fix. For new behaviour, add the smallest non-overlapping tests that prove the acceptance criteria. Numerical changes get a hand-computed or reference-oracle test, not only a self-consistency test. Cover boundaries, invalid input, ties, empty and one-element inputs, and past regressions where relevant; prefer extending an existing test module or parameterisation over creating a redundant test matrix.
3. Keep fixes in the main session; delegate only as described in "Models and delegation".
4. Work in tight red-green-refactor loops. Run the new or failing test first, make the smallest coherent implementation, rerun the targeted test, then clean up without broadening scope.
5. Inspect the actual diff and run the affected tests and checks below once the change is complete. Never accept a subagent summary in place of reviewing its changes and evidence.
6. Before the PR, follow "Before a pull request" below. Work is complete only after the acceptance criteria are met, the review findings are resolved, and CI is green.

# Targeted verification

Run only the affected tests locally:

1. While fixing a bug or adding a test, run that single test.
2. When the change is complete, run the affected test modules and the checks for
   touched files once. Do not rerun them after every small edit, and do not widen
   the run to neighbouring modules "to be safe".
3. After any Rust change, rebuild the extension before running Python tests:
   `uv run maturin develop --release` (about 2 minutes incremental). It writes
   `python/price_contour/_price_contour.abi3.so` in place, so an editable
   install in haute picks it up immediately.
4. When the change alters the public Python API or a result's shape, also run
   haute's affected real-library tests against this checkout (for example
   `uv run --no-sync pytest tests/test_optimiser_routes_real_library.py -q` from
   `../haute`).

Never run the full Rust (`cargo test --workspace`) or Python (`uv run pytest tests/python`)
suite locally unless the user asks for it. CI is the full gate.

Useful commands:

- Targeted Rust test: `cargo test -p price-contour-core <test_name>`
- Targeted Python test: `uv run pytest tests/python/test_relevant.py::test_name -q`
- Rust checks: `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`
- Touched Python files: `uvx ruff check <files>` and `uvx ruff format --check <files>`

# Releases

- The version lives in `pyproject.toml`. Follow semantic versioning with the
  pre-1.0 convention: a change to a public result's shape, a public signature,
  a reported number, or a newly raised error is a minor bump (0.x → 0.x+1); a
  pure bug fix or addition that no consumer can observe is a patch bump.
- Commit messages are conventional (`feat:`, `fix:`, `refactor:`, `perf:`,
  `test:`, `docs:`, `ci:`, `chore:`, `build:`), enforced by pre-commit.

# Before a pull request

1. Run the affected tests and checks for the whole change (above).
2. Have Codex review the whole branch diff once ("Code review with Codex" below).
   Resolve or rebut each finding, rerun only the tests the fixes touch, and
   resume the same review thread to confirm. Do not review each commit or
   package separately.
3. Open the PR, or push to the existing one, then watch its checks with `gh`
   until they finish. On a failure, read the failing job's log, reproduce only
   that test locally if the cause is unclear, fix it, and push again. Rerun an
   environmental flake with `gh run rerun <id> --failed` instead of pushing a
   change.

# Code review with Codex

Reviews use Codex, a different model family from the author, in a read-only
sandbox: `gpt-6-astra` at `xhigh` effort. Keep review state (event log, thread id,
review text) in a scratch directory outside the repository.

- **Start:** `codex exec --json --sandbox read-only --skip-git-repo-check -c model=gpt-6-astra -c model_reasoning_effort=xhigh "<prompt>" < /dev/null > review.ndjson`.
  The thread id is in the `thread.started` event; the review is the text of the
  last `item.completed` event whose item type is `agent_message`.
- **Resume after fixes:** `codex exec --sandbox read-only -c model=gpt-6-astra -c model_reasoning_effort=xhigh resume <thread-id> --json --skip-git-repo-check "<prompt>" < /dev/null`.
  Say what changed and why any finding was rebutted, and ask Codex to state for
  each prior finding whether it is addressed, flag new issues, and not re-flag
  a finding rebutted with a reason.
- **The prompt** names the diff (`git diff origin/main...HEAD` for a branch),
  the intent, the design documents changed, and the verification already run
  with its results. It asks for a review against the checklist below, citing
  `file:line`, with a severity for each finding and one final tag on its own
  line: `APPROVED`, `REQUEST_CHANGES`, or `NEEDS_REWORK` (structural problems).

Review priorities, in order: numerical correctness (wrong results, silent
failure, non-determinism); conformance to the changed design documents, where
drift between documents and code in either direction is a finding; API and
contract stability for consumers; test quality; practical concerns
(performance and memory on real book sizes, messages a user can act on).
Not priorities: documentation compliance for its own sake when the change
updates the document, environment limitations, theoretical edge cases real
inputs do not produce, and repeating a finding already addressed or rebutted.

Checklist:

1. **Function:** the logic matches the requirements and design documents; error
   scenarios and edge cases are handled.
2. **Code quality:** clippy- and ruff-clean; no duplication, and existing
   helpers are reused; no needless complexity; clear names; comments explain
   constraints rather than narrate; no oversized modules.
3. **Architecture:** Rust computes and Python orchestrates; bindings stay thin;
   one code path per quantity; the code and the design documents agree.
4. **Contracts:** public result shapes, column names, dtypes and orderings are
   documented and pinned by tests; nothing a consumer needs is private or
   positional; the `.pyi` stubs match the bindings.
5. **Error handling:** fail loud, with no fallback or default that masks an
   error; messages are clear and actionable; no empty catches or broad excepts.
6. **Numerics:** precision (f32 storage, f64 accumulation) is deliberate;
   reductions are deterministic; tie and boundary rules are documented and
   tested.
7. **Performance:** memory is bounded on multi-million-quote grids; no
   unnecessary copies across the FFI boundary; nothing unnecessary runs in the
   solver's hot loops.
8. **Tests, reviewed as code:** they assert observable behaviour rather than
   wiring, and a test that would pass against a buggy implementation is a Major
   finding; numerical tests compare against hand-computed values or an
   independent oracle; failure paths are tested.

Severity: **Critical** blocks a merge (silently wrong results, data corruption,
a breaking interface change without a version bump); **Major** must be fixed
(wrong logic, significant slowdown, missing error handling or a silent failure
path, build errors); **Minor** should be fixed (style, missing documentation,
duplication, a missed edge case); **Suggestion** is optional.

Approval gate: the requirements are implemented; no Critical or Major finding is
open; the build passes; the affected tests pass; new logic has behavioural
tests; and the affected design documents are updated.

Handling a review: read the code at each cited `file:line`; fix the legitimate
findings; rebut an incorrect one with the reason; weigh each finding's cost
against its risk and push back on one that grows the scope without matching
risk. Rerun only the tests the fixes touch, then resume the thread once. Bring
`NEEDS_REWORK` to the user before making large changes.

**Plan or design review** uses the same command. Codex reviews the design
document deltas together with the plan for correctness (fail loud, one code
path, determinism), whether a developer can build it without guessing, whether
an engineer independent of the implementer could write the failing tests from
it, and practical risks. Findings are tagged P1 (blocks implementation) or P2,
and the review ends with a tag.

**A second opinion** on a design choice, a stuck bug, or a conclusion about to be
presented also uses the same command. The prompt gives the question and your
draft position and asks Codex to disagree where warranted, separating what it
verified in the repository from what it inferred, and to end with a short bottom
line. It is advisory: no tags, and nothing is gated on the answer. Don't use it
for a question only the user can answer. When Codex disagrees, tell the user.
