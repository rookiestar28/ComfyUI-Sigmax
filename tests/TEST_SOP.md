# ComfyUI-Sigmax Test SOP

<!-- CURRENT-TEST-GOVERNANCE:START -->
## Current Test Governance

This section supersedes narrower platform-selection, pre-push, and Hosted CI requirements later in
this document.

- Pure text/documentation changes and version-field-only `pyproject.toml` updates require no test
  contract, independent review, Full Gate, or E2E run. A `pyproject.toml` change that affects
  dependencies, build behavior, tool configuration, entry points, packaging, or runtime semantics
  is not version-only and is not exempt.
- For non-exempt implementation work, a passing Windows Full Gate is the authoritative
  repository-wide acceptance result. Push and Hosted CI are not acceptance prerequisites, and
  evidence need not be bound to a pushed commit. Linux/WSL and Hosted CI are optional diagnostics
  unless the active item explicitly requires additional platform, release, publication, or live-host
  evidence.
- Item-scoped parity, host, GPU, security, release, or publication checks remain additive when their
  actual risk boundary is in scope; none replaces the Windows Full Gate.
<!-- CURRENT-TEST-GOVERNANCE:END -->

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

The repository now also ships a scoped ComfyUI frontend extension for the Krea 2 experimental
variant policy. The default full gate runs its dependency-free Node.js policy tests and syntax
check after core-independence validation. This is a deterministic frontend-policy gate, not a
substitute for real-browser ComfyUI integration evidence.

The pure numerical, artifact, capability, core-independence, deterministic property, Krea 2
variant-resolution, Turbo golden-vector, Krea 2 RAW golden-vector, and Turbo/RAW framework
parity lanes now exist. Validated product nodes, adapter/integration tests, the isolated
real-ComfyUI H1 harness, and the M2-05 strict Turbo and M3-06 RAW H2 workflow lanes now exist.
The accepted M5-01 through M5-05 model-free H3 surface now includes deterministic native-Euler
proof, the portable sampler-state contract, deterministic full/partial and in-process resume
probes, caller-RNG stochastic Flow Euler with pinned Diffusers-v0.39.0 expression parity, and
advanced-workflow compatibility decisions/receipts. These are internal test/controller contracts,
not a public sampler. They do not establish model-backed stochastic execution, persisted latent/RNG
state, cross-process resume, general advanced-workflow execution, or image-quality behavior.
The accepted M6-05 MiniMax H3 slice also has a separate pinned-ComfyUI 0.30.0 model-free H1/H2
contract for explicit FL2VA and Ref2VA node execution; first/repeat H1/H2 evidence passed without
loading H3 weights. It does not replace the separately authorized model-host gate. M4-13 closed
its public transition-count vocabulary and downstream schema requalification with passing
WSL/Windows full gates and exact pinned-host H1/H2. GitHub Actions hosted CI was explicitly
waived for M4-13 on 2026-08-06 because the user's quota was exhausted; no hosted pass is claimed.
Unsupported model-backed sampler and advanced-workflow capabilities remain unclaimed; no active
roadmap successor owns them. M7-09 owns an explicit, local optional-heavy H4 lane under its frozen
plan; it is separate from the default CPU/full gate.
On 2026-08-07 the user waived M7-09 scoring as an acceptance blocker. Its execution/provenance
receipt may close the local item, but the waiver does not create a prompt-adherence, image-quality,
or profile-promotion claim. Until later roadmap owners create any other heavy gates:

- the Windows Full Gate is mandatory repository-wide acceptance evidence;
- direct commands remain available for targeted diagnosis;
- missing future gates remain `NOT_IMPLEMENTED`, not passed;
- documentation-only and version-field-only `pyproject.toml` work uses the exception in Section 5
  and does not run the full gate.

## 3. Required Reading Order

Before running acceptance validation, read:

1. `tests/TEST_SOP.md`
2. `tests/E2E_TESTING_NOTICE.md`
3. `tests/E2E_TESTING_SOP.md`
4. `tests/CI_TEST_MATRIX.md`

## 4. Acceptance Rule

Non-documentation work is accepted only when:

- targeted tests for the changed behavior pass;
- every applicable full-gate stage passes;
- required parity and host E2E lanes pass;
- critical no-skip seams execute without skips;
- the Windows Full Gate output is retained for the reviewed change;
- any optional Hosted CI result is reported accurately as supplemental evidence and is not required
  for acceptance or bound to a pushed commit;
- the public PR, issue, or change description maps evidence to each acceptance criterion.

Do not treat a missing, skipped, or unavailable gate as a pass.

## 5. Documentation and Version-Only Exception

Pure prose that is unrelated to executable behavior is not a test contract. A version-field-only
change to `pyproject.toml` is also exempt. Neither change requires a plan, independent reviewer,
pytest/CI/hook contract, Full Gate, or E2E run. Do not add automated acceptance assertions that
freeze document wording, headings, line counts, link layout, file inventory, or narrative
freshness. The exemption applies when the change does not modify:

- Python, JavaScript, or other executable code;
- test code or fixtures;
- scripts, hooks, or CI workflows;
- dependency declarations or package/build behavior beyond the `pyproject.toml` version field;
- model/profile schemas consumed at runtime;
- configuration or generated artifacts;
- node definitions, workflow JSON, or runtime behavior.

Review the touched prose directly. Lightweight hygiene such as `git diff --check`, link/path
inspection, or Markdown rendering may be used when useful, but it is not promoted into pytest.
When a document describes an executable API, script, package, schema, or release boundary, test
that executable subject at its source rather than asserting the prose that describes it.

## 6. Documentation Hygiene (Non-contract)

These commands help reviewers inspect prose-only changes; they are not automated acceptance
contracts for document content.

When Git is initialized:

```powershell
git diff --check
git status --short
```

Before Git exists, use a lightweight repository scan:

```powershell
$files = @(
  "README.md",
  "CONTRIBUTING.md",
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

Node.js 18 or newer is mandatory for the default full gate because the repository ships a scoped
ComfyUI frontend extension. Hosted CI selects Node.js 20.

The `frontend-policy` stage runs `scripts/run_frontend_policy_tests.py`, which uses only Node's
built-in test runner and syntax checker against fixed repository files. It does not run npm,
require a package lock, launch ComfyUI in a browser, or provide Playwright E2E evidence. The
preflight report does not own this check; the frontend-policy runner independently fails with an
actionable message when Node is absent or older than version 18.

Real-browser scope and the distinction between accepted item-specific browser evidence and a
reusable automated browser lane are defined in `tests/E2E_TESTING_SOP.md`.

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

## 9. Full Validation Gate

M0 provides a Windows acceptance entrypoint and an optional Linux/WSL diagnostic entrypoint:

```powershell
powershell -File scripts/run_full_tests_windows.ps1
```

```bash
bash scripts/run_full_tests_linux.sh
```

Both wrappers call `scripts/run_full_gate.py`, which is the canonical stage ordering. Direct
commands and the Linux/WSL wrapper remain useful for targeted diagnosis, but repository-wide
acceptance uses the Windows wrapper.

The common runner executes `core-independence` and `frontend-policy` after static/type checks and
before parity-contract and pytest stages.

Gate classes, triggers, job roles, and artifact requirements are defined in
`tests/CI_TEST_MATRIX.md`. A targeted `fast` run is never acceptance evidence by itself.

### Stage 0 - Environment and Dependency Preflight

Validate:

- supported Python version;
- project-local interpreter;
- supported PyTorch version when installed;
- optional dependency availability only for selected lanes;
- supported ComfyUI path/version for host tests;
- the frontend-policy runner's Node.js 18+ requirement;
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

### Stage 2A - Core Independence and Frontend Policy

```powershell
python scripts/check_core_independence.py
python scripts/run_frontend_policy_tests.py
```

Core independence blocks accidental ComfyUI or Diffusers imports in the pure layers. Frontend
policy requires Node.js 18+ and verifies the experimental Krea 2 widget policy plus JavaScript
syntax without npm or a browser. A real-browser change may require the separate evidence defined
in `tests/E2E_TESTING_SOP.md`.

### Stage 3 - Unit, Pure-Core, and Property Tests

The current canonical runner is:

```powershell
python -m pytest
python -m pytest --cov=comfyui_sigmax --cov-branch
```

The pure-core/property-specific command is:

```powershell
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

For release-facing work, also build and inspect both wheel and sdist through the canonical
non-publishing audit, using fresh repository-local ignored destinations:

```powershell
python scripts/run_release_audit.py --dist-dir .tmp/release-audit-dist --output .tmp/release-audit.json
```

The `sigmax.release-audit/1` report must pass its separate tracked-file, dependency,
provenance/license, Registry metadata, secret-scan, and archive sections. This does not replace
the wheel clean-install check or authorize publishing.

For a Comfy Registry candidate, stage the intended files and validate the exact Git-indexed ZIP:

```powershell
python scripts/validate_registry_artifact.py --archive .tmp/comfy-registry/comfyui-sigmax-1.1.0.zip --check-manifest --observe-registry --output .tmp/comfy-registry/report.json
```

The `sigmax.registry-artifact-report/1` must bind the canonical
`sigmax.registry-release-manifest/1`, pass member and source hash audits, import from a normalized
renamed directory, and record
`publication_performed: false`. Windows and WSL must produce the same archive SHA-256. This
validation neither replaces the release audit nor authorizes publication.

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

M7-09's frozen protocol defines fixed RAW/Turbo cases, control and identity-ablation arms, model
component hashes, a randomized blind ballot, independent reviewers, variance/confidence reporting,
and a predeclared regression decision. The user may explicitly waive the scoring phase as an
acceptance blocker; when waived, record the waiver and retain the execution receipt, while keeping
quality and promotion claims unmade. Its local H4 receipt is not a hosted-CI result.

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
- The preflight emits stable `venv.*`, `cache.*`, `tooling.*`, `filesystem.*`, `unicode.*`,
  `temp.*`, and `optional.*` issue codes. Follow the single remediation attached to the first
  blocking issue; never mix a global `pre-commit` executable with the selected local venv.

### Linux/WSL

- Prefer `.venv-wsl`.
- Fail clearly when Python or required system libraries are missing.
- Use a repository-local writable temporary directory when a mounted Windows path causes
  permission problems.
- The Linux wrapper isolates the selected `.venv-wsl` tool path and exports repository-local
  cache/temp roots before preflight. A foreign PATH `pre-commit` is diagnosed rather than used.

Both wrappers run `sigmax.environment-diagnostics/1` and set `PRE_COMMIT_HOME`,
`SIGMAX_TEMP_ROOT`, `TMPDIR`, `TMP`, and `TEMP` below `.tmp`. The preflight performs an owned
write/read/rename/delete cycle with a non-ASCII filename and a read-only SQLite integrity check
when the pre-commit database exists. It never deletes a cache automatically.

On a WSL mounted workspace, the same preflight also exercises anonymous temporary-file
semantics. When the mount cannot sustain pytest's default file-descriptor capture, the Linux
wrapper records the `pytest.capture_sys` mitigation and runs pytest with `--capture=sys`; an
unmitigated direct preflight reports `temp.incompatible` instead of allowing a zero-test run.

Fixed optional lanes can be diagnosed without importing their modules:

```powershell
python scripts/preflight_check.py --optional-lane plot
python scripts/preflight_check.py --optional-lane reference
```

Install only the reported reviewed extra in the selected local venv, then rerun preflight.

Do not mix Windows and WSL virtual environments.

## 14. Evidence and Traceability

Accepted non-documentation work requires retained CI output or an attached redacted verification
report. The public PR, issue, or change description must identify:

- date and timezone;
- workspace and branch/baseline;
- OS and shell;
- Python, PyTorch, ComfyUI, Diffusers, and Node versions when applicable;
- exact command;
- exit status;
- materially relevant redacted output;
- PASS, FAIL, SKIPPED, or NOT_APPLICABLE;
- reason and replacement evidence for any non-applicable lane.

The public change description must link every acceptance requirement to verification evidence.

## 15. Failure Policy

If a required stage fails:

1. preserve the failure evidence;
2. diagnose and fix the root cause;
3. rerun the targeted stage;
4. rerun downstream dependent stages;
5. update the retained verification report and public change description;
6. do not declare the change complete until all required evidence passes.

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
- no accidental Node/Playwright dependency in pure-core or isolated parity environments;
- no path filter or condition can bypass a required P0 test seam;
- unavailable lanes report `NOT_IMPLEMENTED`, never pass, unless an explicit item-level waiver is
  recorded by the user; a waiver changes the acceptance decision, not the underlying lane result.
