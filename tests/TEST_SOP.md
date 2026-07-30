# ComfyUI-Sigmax Test SOP

## 1. Purpose

This document defines the mandatory validation workflow for ComfyUI-Sigmax.

The repository is a Python-based ComfyUI custom-node project. Its primary correctness risks
are mathematical schedule drift, ambiguous profile resolution, double shifting, inert node
controls, ComfyUI API incompatibility, and unintended global side effects.

Tests must be designed to expose those failures, not merely produce a green result.

## 2. Current Bootstrap State

The package, project-local quality runner, cross-platform full-gate wrappers, and CI workflow
contract now exist. The canonical package namespace, pytest, Ruff, mypy, pre-commit,
detect-secrets, branch coverage, wheel inventory, Windows/WSL wrappers, and Windows/Ubuntu
Python matrix were established in M0-04 through M0-06.

The pure numerical, artifact, capability, core-independence, and deterministic property
lanes now exist. The ComfyUI host fixture, model golden/parity suites, and product nodes do not yet
exist. Until their roadmap owners create them:

- the OS-specific full-gate wrapper is mandatory acceptance evidence;
- direct commands remain available for targeted diagnosis;
- missing future gates remain `NOT_IMPLEMENTED`, not passed;
- documentation-only work may use the exception in Section 5.

## 3. Required Reading Order

Before running acceptance validation, read:

1. `tests/TEST_SOP.md`
2. `tests/E2E_TESTING_NOTICE.md`
3. `tests/E2E_TESTING_SOP.md`
4. `tests/CI_TEST_MATRIX.md`
5. The active roadmap item and its `.planning/` plan

## 4. Acceptance Rule

Non-documentation work is accepted only when:

- targeted tests for the changed behavior pass;
- every applicable full-gate stage passes;
- required parity and host E2E lanes pass;
- critical no-skip seams execute without skips;
- CI and local commands exercise the same underlying stage definitions;
- results are recorded in a dated command log;
- the implementation record maps evidence to each acceptance criterion.

Do not treat a missing, skipped, or unavailable gate as a pass.

## 5. Documentation-Only Exception

The full executable gate is optional when every changed file is prose-only documentation and
the change does not modify:

- Python, JavaScript, or other executable code;
- test code or fixtures;
- scripts, hooks, or CI workflows;
- dependency or package manifests;
- model/profile schemas consumed at runtime;
- configuration or generated artifacts;
- node definitions, workflow JSON, or runtime behavior.

Documentation-only validation must still check:

- referenced paths and names;
- stale project-specific terms;
- Markdown whitespace and fence balance;
- roadmap acceptance requirements;
- consistency among `AGENTS.md`, `ROADMAP.md`, and test SOPs.
- consistency with `tests/CI_TEST_MATRIX.md`.

## 6. Documentation Baseline Checks

When Git is initialized:

```powershell
git diff --check
git status --short
```

Before Git exists, use a lightweight repository scan:

```powershell
$files = @(
  "AGENTS.md",
  "ROADMAP.md",
  "tests/TEST_SOP.md",
  "tests/E2E_TESTING_NOTICE.md",
  "tests/E2E_TESTING_SOP.md",
  "tests/CI_TEST_MATRIX.md"
)

$files | ForEach-Object {
  if (-not (Test-Path -LiteralPath $_)) {
    throw "Missing required document: $_"
  }
}

```

For a repository migration, run a separate search for every legacy project name, route prefix,
environment-variable prefix, and item-code prefix identified in the active implementation
plan. Any intentional match must be explicitly reviewed; do not place the legacy terms in this
canonical SOP merely to drive the scan.

## 7. Environment Policy

### 7.1 Python

- Minimum supported version: Python 3.10.
- The foundation matrix covers Python 3.10 and 3.13; expand it only with explicit evidence.
- Use a repository-local environment:
  - Windows: `.venv`
  - Linux/WSL: `.venv-wsl`
- Use the same interpreter for setup, hooks, tests, and reports.
- Record:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

For real ComfyUI host tests, also record the exact interpreter used by the host.

### 7.2 Node.js

Node.js is not part of the default gate while the repository has no web extension.

If a browser UI is later added:

- Node.js 18+ becomes mandatory for that lane;
- package-lock integrity and Playwright instructions must be added to
  `tests/E2E_TESTING_SOP.md`;
- the roadmap and implementation plan must explicitly activate the frontend gate.

### 7.3 Optional Heavy Dependencies

Diffusers, full ComfyUI hosts, GPU libraries, and model weights must be isolated:

- closed-form core tests must not require them;
- parity dependencies must be pinned;
- GPU/model-weight tests must use explicit markers and cannot run accidentally;
- model files and caches must remain ignored and outside release artifacts.

## 8. Problem-First Bugfix Model

Every bugfix follows `Reproduce -> Pin -> Sweep`.

### Reproduce

Capture the smallest credible pre-fix failure:

- failing unit/property test;
- failing golden vector;
- authoritative parity mismatch;
- ComfyUI import/registration failure;
- host workflow failure;
- stateful sampler step mismatch.

### Pin

Add a regression test that:

- fails against the broken behavior;
- targets the root cause;
- passes only after the fix;
- asserts the final contract, not a nearby happy path.

### Sweep

Run all applicable full-gate stages after the targeted regression passes.

The record must keep the original failure and corrected rerun. A green full gate without
reproduction and pinning is insufficient bugfix evidence.

## 9. Planned Full Validation Gate

M0 provides equivalent Windows and Linux/WSL entry scripts:

```powershell
powershell -File scripts/run_full_tests_windows.ps1
```

```bash
bash scripts/run_full_tests_linux.sh
```

Both wrappers call `scripts/run_full_gate.py`, which is the canonical stage ordering. Direct
commands remain useful for targeted diagnosis, but acceptance uses the OS wrapper.

The common runner executes `core-independence` after static/type checks and before pytest.

Gate classes, triggers, job roles, and artifact requirements are defined in
`tests/CI_TEST_MATRIX.md`. A targeted `fast` run is never acceptance evidence by itself.

### Stage 0 - Environment and Dependency Preflight

Validate:

- supported Python version;
- project-local interpreter;
- supported PyTorch version when installed;
- optional dependency availability only for selected lanes;
- supported ComfyUI path/version for host tests;
- no incompatible Node requirement in a Python-only run.
- CI policy/configuration contracts when workflows or launchers exist.

Canonical command:

```powershell
python scripts/preflight_check.py
```

### Stage 1 - Secret Scan and Repository Hygiene

```powershell
python -m pre_commit run detect-secrets --all-files
python -m pre_commit run --all-files --show-diff-on-failure
git status --short
```

Rules:

- run hooks serially on Windows;
- hook-modified files are a failure until reviewed and the hook reruns clean;
- ignored internal documents must not be staged;
- secrets, model weights, caches, and private paths must not enter public artifacts.

### Stage 2 - Formatting, Linting, and Typing

```powershell
python -m ruff format --check .
python -m ruff check .
python -m mypy comfyui_sigmax tests scripts
```

The paths, Python floor, lint selection, and strictness are defined in `pyproject.toml`.

### Stage 3 - Unit, Pure-Core, and Property Tests

The current canonical runner is:

```powershell
python -m pytest
python -m pytest --cov=comfyui_sigmax --cov-branch
```

The pure-core/property-specific command is:

```powershell
python scripts/check_core_independence.py
python -m pytest tests/test_*.py tests/property
```

Required risk coverage:

- invalid steps and dimensions;
- float32/float64 behavior;
- sigma domains;
- base grids;
- exponential and direct-ratio shifts;
- terminal policies;
- start/end and denoise slicing;
- conflicting transforms;
- non-finite and non-monotonic outputs;
- fingerprint stability;
- metadata serialization;
- optional dependency absence.
- deterministic property and metamorphic relations.

The pure-core lane must pass without ComfyUI or Diffusers installed.

Mutation testing of critical pure math is a scheduled or milestone gate after the core exists;
it does not run in every developer inner loop.

### Stage 4 - Golden Schedule Tests

```powershell
python -m pytest tests/golden
```

Required Krea 2 coverage:

- Turbo at 4, 8, 12, and 16 steps;
- RAW at 28 and 52 steps;
- RAW square resolutions: 256, 512, 768, 1024, and 1280;
- RAW landscape and portrait cases;
- terminal zero;
- full vector, not only selected points.

Tier 1 golden tests are no-skip seams.

Initial comparison tolerances:

- float64 maximum absolute error: `1e-8`;
- float32 maximum absolute error: `1e-6`.

Tolerance changes require a documented numerical reason and updated reference evidence.

### Stage 5 - Authoritative Parity Tests

```powershell
python -m pytest tests/parity -m parity
```

Parity tests must record:

- official source revision;
- optional Diffusers version;
- ComfyUI version when relevant;
- dtype and device;
- expected and actual vectors;
- maximum and mean error;
- schedule fingerprint.

Parity lanes may use isolated optional environments. An evidence-pinned structural profile
declaration may land before these lanes exist only when it makes no Tier 1 parity or host
support claim and the roadmap retains separate blocking golden, framework-parity, and host
items. A profile cannot be promoted to Tier 1 official parity without their passing evidence.

Pinned parity results must be retained as machine-readable vectors and human-readable error
summaries when CI exists.

Never execute an unreviewed reference repository directly. Prefer extracted formulas, pinned
known-safe adapters, or disposable sandbox execution after explicit approval.

### Stage 6 - ComfyUI Adapter and Node Integration

```powershell
python -m pytest tests/integration
```

Required coverage:

- node mappings and display mappings;
- input/output schema;
- model/profile resolution precedence;
- ambiguous Krea 2 variant handling;
- strict-mode errors;
- no double shift;
- no inert inputs;
- idempotent namespaced registration;
- absence of automatic global PyTorch patches;
- workflow metadata serialization;
- supported ComfyUI API matrix.

### Stage 7 - Real ComfyUI Host E2E

Follow `tests/E2E_TESTING_SOP.md`.

Host E2E is mandatory when changes affect:

- node import or registration;
- node input/output schema;
- workflow load/save behavior;
- ComfyUI model inspection;
- schedule execution through ComfyUI;
- sampler execution;
- host compatibility.

Pure schedule-core changes may omit host E2E when the plan explains why no adapter, node, or
workflow contract is affected. A dependency-free structural profile may also omit host E2E
when it performs no host discovery/import/execution and explicitly defers host compatibility
claims to a blocking later roadmap item. Profile changes that affect resolution, registration,
or host execution require host E2E.

### Stage 8 - Packaging and Clean-Install Checks

Current baseline:

```powershell
python -m build --wheel --outdir .tmp/dist
```

Verify:

- clean environment install/import;
- optional dependencies remain optional;
- node package import has no forbidden side effects;
- licenses and attribution are included;
- the wheel contains only the declared package and required metadata;
- internal planning, research, caches, tests, and local paths are absent;
- runtime dependency metadata is empty until an approved runtime dependency is introduced.
- internal files, model weights, caches, and secrets are excluded.

### Stage 9 - Test Health, Coverage, and Debt Governance

When source and coverage tooling exist:

- collect statement and branch coverage;
- compare against the reviewed ratchet floor;
- report high-risk hotspot families separately;
- enforce skip/xfail/quarantine metadata and expiry;
- retain first-attempt flake and retry evidence;
- run scheduled mutation/property/fuzz lanes when required by the active milestone.

Do not create a bootstrap coverage target before observing the imported codebase. Coverage
percentage cannot replace golden, parity, integration, or host evidence.

## 10. Test Layout

Planned repository layout:

```text
tests/
  unit/
  property/
  golden/
  parity/
  contract/
  ci/
  integration/
  e2e/
  mutation/
  packaging/
  fixtures/
  workflows/
```

Tests should use markers such as:

- `parity`
- `comfyui`
- `e2e`
- `gpu`
- `model_weights`
- `slow`
- `mutation`
- `latest_host`

Default CPU validation must not select `gpu` or `model_weights`.

## 11. Model and Image Validation

### 11.1 Correctness Hierarchy

Use this order:

1. closed-form unit tests;
2. complete golden vectors;
3. authoritative numerical parity;
4. step-level sampler parity;
5. ComfyUI host behavior;
6. optional image-level comparison.

An attractive image cannot override a numerical parity failure.

### 11.2 Optional GPU/Image Lane

When explicitly required, record:

- checkpoint hash and precision;
- text encoder and VAE hashes;
- prompt and negative prompt;
- seed;
- dimensions;
- steps, CFG, sampler, and schedule fingerprint;
- GPU, PyTorch, and ComfyUI versions.

Quantized and BF16 results must be evaluated separately.

Image metrics and blind review are supplemental evidence and must not be described as proof of
official schedule correctness.

## 12. ComfyUI Compatibility Matrix

M0/M4 must define a supported host matrix. At minimum, validation should distinguish:

- current pinned/known-good ComfyUI revision;
- latest tested ComfyUI revision;
- unsupported revisions.

Compatibility failures must be actionable and must not trigger a silent generic fallback.

Known-good host failures are blocking. Latest-host failures must produce a compatibility
report and cannot silently redefine the supported matrix.

## 13. Windows and WSL Guardrails

### Windows

- Prefer `.venv`.
- Run pre-commit serially.
- Use a repo-local cache when global cache locks occur:

```powershell
$env:PRE_COMMIT_HOME = "$PWD\.tmp\pre-commit-win"
python -m pre_commit run --all-files --show-diff-on-failure
```

- Include at least one test path containing non-ASCII characters.
- Record file-lock failures and corrected reruns.

### Linux/WSL

- Prefer `.venv-wsl`.
- Fail clearly when Python or required system libraries are missing.
- Use a repository-local writable temporary directory when a mounted Windows path causes
  permission problems.

Do not mix Windows and WSL virtual environments.

## 14. Evidence and Traceability

Accepted non-documentation work requires a repository-local command log under `.planning/`.

Each relevant entry records:

- date and timezone;
- workspace and branch/baseline;
- OS and shell;
- Python, PyTorch, ComfyUI, Diffusers, and Node versions when applicable;
- exact command;
- exit status;
- materially relevant redacted output;
- PASS, FAIL, SKIPPED, or NOT_APPLICABLE;
- reason and replacement evidence for any non-applicable lane.

The implementation record must link every acceptance requirement to command-log evidence.

## 15. Failure Policy

If a required stage fails:

1. preserve the failure evidence;
2. diagnose and fix the root cause;
3. rerun the targeted stage;
4. rerun downstream dependent stages;
5. update the command log and implementation record;
6. do not mark the roadmap item complete until all required evidence passes.

After three materially different failed attempts at the same blocker, stop, document the
attempts, and request direction.

## 16. Coverage, Skip, and Flake Governance

The authoritative policy is `tests/CI_TEST_MATRIX.md`.

- Establish coverage from measured source, then ratchet.
- Critical math and host boundary modules require named targeted suites in addition to an
  overall percentage.
- Tier 1 parity, ambiguity, ownership, registration, import-safety, and package-leakage tests
  cannot silently skip.
- Every skip, xfail, retry, or quarantine requires structured ownership, reason, and review
  metadata.
- A retry may provide diagnostics but does not erase a first-attempt P0 regression.

## 17. CI Workflow Contract

When workflows are introduced, they are test subjects.

Before workflow acceptance, automated contract tests must verify:

- minimal permissions and approved action/dependency pinning;
- canonical local/CI command parity;
- required OS/Python/host matrices;
- diagnostic artifact upload on failure;
- no accidental Node/Playwright dependency in Python-only lanes;
- no path filter or condition can bypass a required P0 test seam;
- unavailable lanes report `NOT_IMPLEMENTED`, never pass.
