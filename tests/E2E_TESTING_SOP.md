# ComfyUI-Sigmax End-to-End Testing SOP

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

Remaining H3 capabilities and H4 retain their later activation rules below. A missing later lane
is never a pass.

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
7. run H1, the implemented M2-05 Turbo plus M3-06 RAW H2 lanes, and the activated M5-01
   deterministic native-Euler H3 lane; run other later lanes only after activation;
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

Playwright/browser E2E is currently `NOT_APPLICABLE`.

If a web extension is added:

- create a dedicated browser section or SOP;
- require Node.js 18+ and a lockfile;
- test real ComfyUI frontend extension registration;
- assert user-visible state and payload contracts;
- keep browser E2E separate from real-host backend/node execution;
- update `tests/TEST_SOP.md`, this SOP, and the roadmap in the same change.

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
