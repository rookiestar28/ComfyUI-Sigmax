# ComfyUI-Sigmax End-to-End Testing SOP

<!-- CURRENT-TEST-GOVERNANCE:START -->
## Current Test Governance

- Pure text/documentation changes and version-field-only `pyproject.toml` updates require no test
  contract, Full Gate, or host E2E run. Behavior-bearing `pyproject.toml` changes are not exempt.
- For non-exempt work, a passing Windows Full Gate is the authoritative repository-wide result.
  Push and Hosted CI are not prerequisites and need not bind evidence to a pushed commit.
- Real-host, parity, GPU, release, or publication evidence remains item-scoped and additive when the
  affected boundary explicitly requires it. Linux/WSL and Hosted CI repetitions are otherwise
  optional diagnostics.
<!-- CURRENT-TEST-GOVERNANCE:END -->

## 1. Scope

This SOP defines end-to-end validation against a real ComfyUI host.

It covers:

- custom-node discovery and import;
- node registration and schema exposure;
- workflow load/save behavior;
- schedule execution through ComfyUI;
- model/profile inspection when fixtures permit;
- sampler execution when implemented;
- compatibility across supported ComfyUI revisions.

It does not require real Krea 2 model weights for the default host lane. Heavy GPU/model tests
are separate, explicit lanes.

Strictly documentation-only changes do not run the full gate or any host E2E lane. Pure prose
unrelated to executable behavior is not an E2E or automated test contract; review it directly
under the documentation-only exception in `tests/TEST_SOP.md`. When prose describes host-visible
behavior, validate the underlying node, schema, workflow, or host behavior rather than the prose.

## 2. Current Status

The cross-platform host E2E harness and canonical entrypoints are implemented.

Current implemented scope:

- H0 package/import safety;
- H1 isolated real-host registration and public schema validation;
- M2-05 strict official eight-step Turbo H2 schedule execution, artifact, receipt, and
  no-double-shift evidence;
- M3-06 RAW square/non-square/portrait H2 schedule execution, requested/effective geometry,
  dynamic-mu fingerprints, metadata reload, no-double-shift evidence, strict-auto runtime
  rejection, and invalid-step prequeue HTTP 400 rejection;
- M5-01 deterministic native-Euler H3 controlled execution, step/count parity, deterministic
  rerun, artifact-linked succeeded receipt, and explicit partial-denoise rejection.
- M4-13 native Windows requalification of the public H3 `steps` schema on exact pinned ComfyUI
  0.30.0, including first/repeat H1 and explicit FL2VA/Ref2VA H2 execution.
- M4-13 is closed from passing local WSL/native Windows and pinned-host evidence. GitHub Actions
  hosted CI was explicitly waived for that item on 2026-08-06 because the quota was exhausted;
  no hosted result is claimed. This item-level waiver does not authorize model-weight/GPU
  execution; M7-09 received separate explicit H4/model-weight/GPU authorization on 2026-08-06
  under its frozen local evaluation plan.
- M7-09 has an optional-heavy H4 receipt lane for four fixed Krea 2 RAW/Turbo cases. Its model
  hashes, control/ablation arms, blind ballot, and threshold review are separate from the default
  CPU H1/H2/H3 gate. The user-authorized 2026-08-07 scoring waiver may close local execution
  acceptance without a review receipt; it never authorizes an image-quality, adherence, or
  promotion claim.
- a blocking Node.js frontend-policy gate for the scoped Krea 2 experimental-variant widget
  behavior, plus the accepted M4-11 item-specific browser evidence.

Remaining H3 capabilities and H4 retain their later activation rules below. A missing later lane
is never a pass. The frontend-policy gate is not H1/H2 and does not substitute for reusable
real-browser regression coverage.

## 3. Test Lanes

### H0 - Package Import Smoke

Purpose:

- verify Python package import;
- verify optional dependency absence does not break core import;
- detect automatic global side effects.

H0 is necessary but is not end-to-end.

### H1 - Real ComfyUI Host Registration

Purpose:

- start a supported ComfyUI checkout;
- load ComfyUI-Sigmax as a custom node;
- verify node mappings and display mappings;
- query exposed object information;
- detect import errors, duplicate IDs, and registry conflicts.

H1 is the minimum real-host E2E lane.

### H2 - Workflow Contract and Execution

Purpose:

- load versioned workflow fixtures;
- validate node input/output schemas;
- execute CPU-safe or lightweight schedule workflows;
- verify sigma vectors and metadata fingerprints;
- verify strict errors for ambiguous/invalid profiles;
- save and reload workflow metadata.

H2 is mandatory for node and workflow behavior changes.

### H3 - Sampler and Advanced Workflow Integration

Purpose:

- execute deterministic and future stochastic samplers;
- test partial denoise, image-to-image, inpainting, ControlNet/model patches, interruption, and
  resume where supported;
- detect double shifting and sampler-state errors.

H3 becomes mandatory for M5 and any advanced integration change.

### H4 - Optional GPU and Real-Model Conformance

Purpose:

- run approved Krea 2 or other model profiles with pinned model hashes;
- compare selected step-level or image-level behavior;
- validate quantized and BF16 environments separately.

H4 is not part of the default CPU gate unless a roadmap item explicitly requires it.

## 4. Prerequisites

The M0/M4 harness must provide:

- a pinned or explicitly selected ComfyUI checkout;
- a project-local Python environment compatible with that checkout;
- an isolated custom-node installation/link strategy;
- a unique temporary user/output directory;
- no access to secrets or unrelated user model directories;
- a deterministic test client for host endpoints;
- fixtures that do not require downloading model weights by default.

Recommended environment variables:

```text
COMFYUI_ROOT
SIGMAX_COMFYUI_PYTHON
SIGMAX_E2E_TMP
SIGMAX_COMFYUI_REVISION
```

These names are authoritative. `COMFYUI_ROOT` selects the reviewed checkout;
`SIGMAX_COMFYUI_PYTHON` selects its compatible isolated interpreter.

The selected host's installed dependency versions are evidence fields, not hidden gates. In
particular, `comfy-aimdo` must be recorded but must not be compared with a historical exact
version; a current ComfyUI-recommended release such as `0.4.13` is valid when host startup,
registration, schema, and H1/H2 assertions pass. Exact package pins used by isolated parity
reproduction remain separate lanes and do not replace host compatibility evidence.

### 4.1 Model-Free Host Fixture Architecture

Adapt the official ComfyUI execution-test pattern:

- expose a dedicated Sigmax testing custom-node pack;
- use lightweight CPU tensors and versioned schedule/profile fixtures;
- build graphs through public workflow/API contracts;
- assert actual node execution, sigma outputs, metadata, warnings, and failures;
- avoid loading Krea or other production model weights in H1/H2.

The fixture pack must be test-only, namespaced, and excluded from release artifacts.

Package import, schema serialization, host launch, node registration, and workflow execution
remain separate assertions.

## 5. Safety Requirements

- Never point destructive cleanup at a user's normal ComfyUI output, input, model, or user
  directory.
- Resolve and verify every temporary path before recursive cleanup.
- Use an isolated host port.
- Do not load untrusted custom nodes alongside the test subject unless compatibility with a
  named node pack is the explicit test.
- Do not expose the test host beyond loopback.
- Do not place secrets, private model paths, or user workflow data in logs.
- Do not download or execute external reference code as part of E2E setup without explicit
  approval and prior inspection.
- Terminate the test host after the run, including after failure.

## 6. Canonical Commands

### Windows PowerShell

```powershell
python --version
python -c "import sys; print(sys.executable)"

$env:COMFYUI_ROOT = "C:\path\to\pinned\ComfyUI"
$env:SIGMAX_COMFYUI_PYTHON = "C:\path\to\host-venv\Scripts\python.exe"
powershell -File scripts/run_comfyui_e2e_windows.ps1
```

### Linux/WSL

```bash
python3 --version
export COMFYUI_ROOT="/path/to/pinned/ComfyUI"
export SIGMAX_COMFYUI_PYTHON="/path/to/host-venv/bin/python"
bash scripts/run_comfyui_e2e_linux.sh
```

The wrappers use `.venv` on Windows and `.venv-wsl` on Linux/WSL for the Sigmax validation
driver. The separately selected host interpreter owns ComfyUI-only dependencies.

The scripts must:

1. validate the selected ComfyUI root and revision;
2. validate the project-local interpreter;
3. create an isolated temp/user/output environment;
4. expose ComfyUI-Sigmax without modifying the user's normal custom-node installation;
5. start ComfyUI on loopback with a unique port;
6. poll a bounded readiness endpoint while also watching for early process exit;
7. run H1, the implemented M2-05 Turbo plus M3-06 RAW H2 lanes, the activated M5-01
   deterministic native-Euler H3 lane, and the accepted pinned M6-05 MiniMax H3 model-free
   H1/H2 contract when its host revision is available; run other later lanes only after activation;
8. collect redacted logs and results;
9. request graceful shutdown when supported, wait a bounded interval, then terminate the
   verified process tree if required;
10. verify the port is released and all owned temporary paths are cleaned or retained
    intentionally as failure artifacts;
11. return nonzero on any failed assertion, shutdown, or cleanup failure.

Fixed sleeps are not readiness checks. A retry loop must have a deadline, retain the last
connection/process error, and fail immediately if the host exits.

## 7. H1 Required Assertions

H1 must verify:

- ComfyUI starts without a ComfyUI-Sigmax import exception;
- all expected namespaced node IDs are present;
- no duplicate node ID is silently overwritten;
- declared node display names map to the expected classes;
- unrelated scheduler/solver registry entries remain unchanged;
- importing the extension does not replace `torch.nn.Module.__call__`;
- core import succeeds without optional Diffusers support unless a selected node requires it;
- startup logs contain no unhandled warning classified as a Sigmax compatibility failure.
- the host binds only to the selected loopback address and port;
- the process and port are gone after the lane completes.

## 8. H2 Required Assertions

Each workflow fixture must declare:

- fixture version;
- expected node IDs and schema;
- selected model profile and evidence level;
- dimensions and steps;
- expected sigma vector or fingerprint;
- expected warnings or strict failures.

H2 must include:

1. Krea 2 Turbo official schedule at 8 steps.
2. Krea 2 RAW official schedule at a reference square resolution.
3. Krea 2 RAW official schedule at a non-square resolution.
4. Krea 2 RAW framework-reference schedule at a portrait resolution.
5. Ambiguous Krea 2 `auto` selection through a terminal runtime rejection.
6. Invalid steps through structured prequeue HTTP 400 rejection.
7. Metadata save/reload.
8. Registration reload/idempotency when supported.
9. Native versus external-sigma ownership without double shifting.
10. A changed control producing a changed executed result or explicit error.

For successful workflows, assert final sigma values and metadata, not only queue acceptance.

For rejected workflows, assert:

- the intended runtime or prequeue boundary;
- error category;
- stable machine-readable reason when available;
- prompt-ID presence for runtime rejection and absence for prequeue rejection;
- absence of partial output or silent fallback.

## 9. H3 Required Assertions

When sampler support exists, add:

- deterministic step-level parity;
- fixed-seed reproducibility;
- partial-denoise start/end behavior;
- terminal sigma behavior;
- interruption/cleanup behavior;
- stochastic mode changing executed math, not only metadata;
- no double shift when a model already owns sampling semantics;
- compatibility or explicit rejection for model patches and advanced workflows.

An execution request being accepted is not sufficient. Verify the resulting latent/schedule
contract or other deterministic final artifact.

## 10. H4 Optional Real-Model Protocol

H4 requires explicit authorization for model files and compute.

Record:

- checkpoint name, hash, license, and precision;
- model profile ID/version;
- ComfyUI, Python, PyTorch, and GPU environment;
- text encoder and VAE hashes when used;
- prompt, seed, resolution, steps, CFG, sampler, and schedule fingerprint;
- baseline/reference configuration;
- step-level errors or image comparison method;
- output retention and cleanup policy.

Do not commit model files or generated benchmark images unless they are small, licensed,
public-safe fixtures intentionally approved for the repository.

## 11. Browser E2E Policy

The repository now has a scoped ComfyUI frontend extension. Its deterministic policy is a
blocking default-gate stage implemented with Node.js 18+ `node:test` and syntax validation. It
asserts that both experimental Krea 2 variants force `strict_official=false`, disable the widget,
and restore official-variant behavior without mutating unrelated widgets.

M4-11 acceptance also records bounded item-specific browser evidence. The repository does not
yet provide a reusable automated Playwright lane that launches the supported ComfyUI frontend;
that general browser-regression lane remains `NOT_IMPLEMENTED`, not passed or silently covered by
the Node.js policy test.

Any later browser behavior that cannot be completely proved by the pure policy module must:

- add a dedicated Playwright/browser procedure and deterministic environment;
- test real ComfyUI frontend extension registration and user-visible state;
- require Node.js 18+ plus an intentional package/lockfile policy when external packages are
  introduced;
- keep browser E2E separate from real-host backend/node execution;
- update `tests/TEST_SOP.md`, this SOP, and `tests/CI_TEST_MATRIX.md` in the same change.

## 12. Compatibility Matrix

Each host run must identify:

| Field | Required |
| --- | --- |
| ComfyUI revision/version | Yes |
| ComfyUI-Sigmax revision | Yes |
| Python executable/version | Yes |
| OS and shell | Yes |
| PyTorch version | When loaded |
| Diffusers version | When selected |
| Device/dtype | When relevant |
| Workflow fixture version | Yes |

At least one pinned known-good ComfyUI revision is mandatory. A latest-revision lane may be
non-blocking until reviewed and promoted.

Recommended initial roles:

- known-good ComfyUI on Ubuntu: blocking H1/H2;
- known-good ComfyUI on Windows: blocking before release and for platform-sensitive changes;
- latest reviewed ComfyUI: scheduled/manual compatibility signal;
- unsupported revisions: fail with an actionable compatibility error.

## 13. Artifacts and Logs

Store temporary E2E artifacts under ignored paths such as:

```text
.tmp/e2e/
```

The retained redacted verification report must include:

- exact command;
- start/end time;
- host revision;
- selected lanes;
- exit status;
- redacted relevant host output;
- assertion summary;
- cleanup status.

Do not retain secrets, raw private paths, user workflows, model weights, or unrelated ComfyUI
logs.

Host artifacts should include:

- revision and interpreter metadata;
- selected loopback address and port;
- startup/readiness timing;
- redacted stdout/stderr;
- `/object_info` contract result;
- workflow fixture IDs and fingerprints;
- assertion, shutdown, port-release, and cleanup summaries.

## 14. Failure and Cleanup Procedure

On failure:

1. preserve the minimal redacted failure artifact;
2. stop the host;
3. verify the temporary directory is inside the repository before cleanup;
4. retain the failed verification output;
5. diagnose and rerun the smallest failing lane;
6. rerun downstream lanes after correction;
7. do not mark E2E passed if host shutdown or bounded cleanup fails.

For a host-visible bugfix, the public PR, issue, or change description must show:

- pre-fix host failure;
- post-fix targeted assertion;
- final applicable H1/H2/H3 sweep;
- full `tests/TEST_SOP.md` gate.

## 15. Flake and Retry Policy

- P0 H1/H2 contract assertions do not pass by retrying the entire test.
- Connection polling is allowed only during bounded host readiness.
- A diagnostic rerun must preserve and report the first failure separately.
- Any quarantined host test needs an owner, reason, linked issue/ADR, review date, and
  replacement evidence as defined in `tests/CI_TEST_MATRIX.md`.
- Repeated port, process, or cleanup failures are harness defects and block host acceptance.
