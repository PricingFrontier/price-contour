# CI/CD Setup Guide

Step-by-step guide to activate the full CI/CD pipeline for price-contour.

---

## Prerequisites

- Admin access to the `PricingFrontier/price-contour` GitHub repository
- A PyPI account with owner/maintainer access to the `price-contour` package (or ability to create it)
- Rust toolchain and Python 3.10+ installed locally

---

## 1. Local: Install pre-commit hooks

Pre-commit hooks run `cargo fmt`, `cargo clippy`, `ruff`, and conventional commit linting before every commit.

```bash
# Install the pre-commit framework
pip install pre-commit

# Install the hooks into your local repo
cd price-contour
pre-commit install                        # pre-commit hooks (fmt, clippy, ruff)
pre-commit install --hook-type commit-msg  # commit message linting

# Verify it works
pre-commit run --all-files
```

**What this does:**
- On every `git commit`, these run automatically:
  - `cargo fmt --all -- --check` — rejects unformatted Rust
  - `cargo clippy --workspace --all-targets -- -D warnings` — rejects Rust warnings
  - `ruff check --fix` — auto-fixes Python lint issues
  - `ruff format` — auto-formats Python
- On every commit message, validates conventional commit format (e.g., `feat:`, `fix:`, `refactor:`)

**If a hook fails:** fix the issue, `git add` the fixed files, and commit again. Clippy can be slow on first run (~30s) but is cached after that.

---

## 2. GitHub: Push the workflow files

The workflow files are already in the repo at:

```
.pre-commit-config.yaml
.github/workflows/ci.yml        # CI on every push/PR
.github/workflows/publish.yml   # Auto-publish on version bump
.github/workflows/bench.yml     # Benchmark tracking
.github/workflows/fuzz.yml      # Weekly fuzz testing
```

Push them to `main`:

```bash
git add .pre-commit-config.yaml .github/
git commit -m "ci: add CI/CD pipeline with pre-commit hooks"
git push origin main
```

The CI workflow will trigger immediately on this push. The publish workflow will detect no version bump and skip.

---

## 3. GitHub: Create the `pypi` environment

This is required for the publish workflow's OIDC trusted publishing.

1. Go to **github.com/PricingFrontier/price-contour/settings/environments**
2. Click **New environment**
3. Name it exactly: `pypi`
4. Under **Deployment protection rules** (optional but recommended):
   - Enable **Required reviewers** and add yourself
   - This means every publish requires manual approval — prevents accidental releases
5. Click **Save protection rules**

---

## 4. PyPI: Claim the package name (first publish only)

If `price-contour` has never been published to PyPI, you need to do an initial manual publish to claim the name.

```bash
# Build a wheel locally
maturin build --release --out dist/

# Upload to PyPI (will prompt for credentials)
pip install twine
twine upload dist/*.whl
```

If the package already exists on PyPI, skip this step.

---

## 5. PyPI: Configure trusted publishing

This lets GitHub Actions publish to PyPI without API tokens.

1. Go to **pypi.org/manage/project/price-contour/settings/publishing/**
   - If you just claimed the name: go to **pypi.org** → Your projects → price-contour → Settings → Publishing
2. Under **Add a new publisher**, fill in:
   - **Owner:** `PricingFrontier`
   - **Repository name:** `price-contour`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`
3. Click **Add**

**What this does:** PyPI trusts GitHub Actions runs from this specific repo/workflow/environment combo. The workflow uses OIDC (no secrets needed) to prove its identity.

---

## 6. GitHub: Enable GitHub Pages (for benchmarks)

The benchmark workflow stores results on a `gh-pages` branch and can serve an interactive chart.

1. Go to **github.com/PricingFrontier/price-contour/settings/pages**
2. Under **Source**, select **Deploy from a branch**
3. Branch: `gh-pages`, folder: `/ (root)`
4. Click **Save**

The `gh-pages` branch will be auto-created on the first benchmark push to `main`. After that, benchmark charts are visible at:

```
https://pricingfrontier.github.io/price-contour/dev/bench/
```

---

## 7. Verify CI works

### Test the CI workflow

Create a test branch and open a PR:

```bash
git checkout -b test/ci-verification
echo "" >> README.md
git add README.md
git commit -m "test: verify CI pipeline"
git push origin test/ci-verification
```

Go to the repo on GitHub, open a PR from `test/ci-verification` → `main`. You should see:

- **Rust checks** job: fmt, clippy, tests (~1-2 min)
- **Python 3.10/3.11/3.12/3.13** jobs: build extension + pytest (~3-5 min each, running in parallel)
- **Benchmarks** job: criterion benchmarks with PR comment showing results

All should pass green. Close the PR without merging (or merge if you want).

### Test the publish workflow

To verify the publish pipeline without actually releasing:

1. Edit `pyproject.toml` and bump the version (e.g., `0.1.0` → `0.1.1`)
2. Commit and push to `main`:

```bash
git checkout main
# Edit pyproject.toml version
git add pyproject.toml
git commit -m "build: bump version to 0.1.1"
git push origin main
```

3. Go to **Actions** tab on GitHub. You should see:
   - **CI** running (Rust + Python tests)
   - **Publish to PyPI** running with jobs:
     - `check-version` — detects the bump
     - `build-wheels` — 4 parallel jobs (Linux, macOS Intel, macOS ARM, Windows)
     - `build-sdist` — source distribution
     - `publish` — uploads to PyPI (will need approval if you set up required reviewers)
   - **Benchmarks** running

4. If you set up required reviewers on the `pypi` environment, go to the publish workflow run and approve the deployment.

5. After publish completes, verify at **pypi.org/project/price-contour/** that the new version is live.

6. Check **github.com/PricingFrontier/price-contour/releases** for the auto-created release with attached wheels.

---

## 8. Verify fuzz testing

The fuzz workflow runs weekly (Sundays at 03:00 UTC). To test it immediately:

1. Go to **Actions** → **Fuzz Testing** → **Run workflow** (top right)
2. Click **Run workflow** on the `main` branch
3. Four parallel jobs will run, each fuzzing for 5 minutes
4. If no crashes are found, all jobs pass green
5. Corpus artifacts are uploaded for each target (visible in the job summary)

---

## How it all works together

### Day-to-day development

```
Developer commits code
  → pre-commit hooks run (fmt, clippy, ruff)
  → push to branch
  → open PR
  → CI runs (Rust checks + Python 3.10-3.13 tests + benchmarks)
  → review + merge
```

### Releasing a new version

```
Bump version in pyproject.toml
  → merge to main
  → CI runs and passes
  → Publish workflow detects version bump
  → Builds wheels for 4 platforms
  → Publishes to PyPI via OIDC (manual approval if configured)
  → Creates GitHub Release with changelog and wheel attachments
```

### Ongoing quality

```
Every Sunday at 03:00 UTC
  → 4 fuzz targets run for 5 minutes each
  → Crashes uploaded as artifacts if found

Every push to main
  → Criterion benchmarks run
  → Results stored on gh-pages
  → 30%+ regressions flagged with PR comments
```

---

## Workflow summary

| Workflow | Trigger | Duration | What it does |
|----------|---------|----------|-------------|
| **CI** | Push / PR | ~5 min | fmt, clippy, cargo test, pytest × 4 Python versions |
| **Publish** | Push to main (version bump) | ~10 min | Build 4-platform wheels + sdist → PyPI + GitHub Release |
| **Benchmarks** | Push / PR | ~3 min | Criterion benchmarks, tracked over time on gh-pages |
| **Fuzz** | Weekly / manual | ~20 min | 4 fuzz targets × 5 min, crashes uploaded |

---

## Troubleshooting

### Pre-commit hooks are slow

Clippy compiles the entire workspace on first run. Subsequent runs use the cargo cache and are much faster. If it's too slow for your workflow:

```bash
# Skip hooks for a single commit (use sparingly)
git commit --no-verify -m "wip: work in progress"
```

### CI fails with "target-cpu" errors

The CI workflows override `.cargo/config.toml`'s `target-cpu=native` with `x86-64-v3` (AVX2). If a CI runner doesn't support AVX2 (unlikely for GitHub-hosted runners), change the env var to `x86-64` (baseline SSE2) in the workflow file.

### Publish workflow doesn't trigger

The version detection compares `pyproject.toml` between HEAD and HEAD~1. If you squash-merge a PR that includes a version bump, HEAD~1 is the previous merge commit — this works correctly. If you rebase or amend, ensure the version change is in the final commit on `main`.

### PyPI publish fails with "403 Forbidden"

This means the OIDC trust is not configured correctly. Verify:
1. The trusted publisher on PyPI matches exactly: owner `PricingFrontier`, repo `price-contour`, workflow `publish.yml`, environment `pypi`
2. The GitHub environment is named exactly `pypi` (case-sensitive)
3. The workflow has `permissions: id-token: write`

### Fuzz target crashes

If a fuzz run finds a crash:
1. The job fails and uploads the crashing input as an artifact
2. Download the crash artifact from the workflow run
3. Reproduce locally:
   ```bash
   cd crates/price-contour-core
   cargo +nightly fuzz run <target> fuzz/artifacts/<target>/<crash-file>
   ```
4. Fix the bug, add a regression test, and push

### Benchmark regressions

The benchmark action comments on PRs when performance degrades >30%. To investigate:
1. Check the benchmark chart at `https://pricingfrontier.github.io/price-contour/dev/bench/`
2. Compare the PR's benchmark numbers against the baseline
3. Note: CI runner performance varies ~10-15%, so small regressions may be noise

---

## File reference

| File | Purpose |
|------|---------|
| `.pre-commit-config.yaml` | Local pre-commit hook configuration |
| `.github/workflows/ci.yml` | CI: Rust checks + Python test matrix |
| `.github/workflows/publish.yml` | Auto-publish wheels on version bump |
| `.github/workflows/bench.yml` | Criterion benchmark tracking |
| `.github/workflows/fuzz.yml` | Weekly fuzz testing |
