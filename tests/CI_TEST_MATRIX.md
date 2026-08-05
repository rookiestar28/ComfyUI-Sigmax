# ComfyUI-Sigmax CI and Test Matrix

## 1. Purpose

This document maps regression risks to executable test layers and CI lanes. It prevents a
workflow from being considered comprehensive merely because it is green.

Command definitions remain in `tests/TEST_SOP.md`. Real-host lifecycle requirements remain in
`tests/E2E_TESTING_SOP.md`.

## 2. Current Activation State

| Capability | State | Activation dependency |
| --- | --- | --- |
| Documentation checks | Implemented in pytest | M0-07 public contract |
| CI configuration self-tests | Implemented | M0-06 |
| Pure-core tests | Implemented | M1-07 |
| Deterministic property tests | Implemented | M1-07 |
| Krea 2 Turbo golden vectors | Implemented | M2-02 |
| Krea 2 RAW golden vectors | Implemented | M3-03 |
| Krea 2 variant-resolution contracts | Implemented | M3-04 |
| Framework parity tests | Implemented | M2-03, M3-05 |
| Native ComfyUI parity tests | Implemented | M2-04 |
| MiniMax H3 Diffusers/native parity lanes | Contract, source pins, complete 4/8/12/16/20 fixtures, and isolated CI jobs implemented; local CPython 3.13 CPU reports pass, hosted runtime artifacts pending | M6-05 subordinate qualification |
| Numerical benchmark matrix | Implemented | M7-02 |
| Optional image benchmark protocol | Implemented; execution unapproved | M7-03 |
| Dependency compatibility matrix | Local/pinned plus release/HEAD latest-host evidence implemented; official container explicitly unavailable/non-blocking | M7-04 |
| Co-installation mutation matrix | Ten deterministic synthetic first/repeat scenarios implemented; no external pack code executed | M7-08 |
| Performance budgets | Windows/WSL pure first/repeat limits plus two pinned-host startup observations implemented | M7-05 |
| Environment guardrails | Versioned venv/cache/lock/Unicode/temp/optional diagnostics run before every full gate | M7-06 |
| Stable public contracts | Source-derived fingerprinted node/schema/reason manifest plus migration policy implemented | M8-01 |
| Security and release audit | Tracked/dependency/provenance/Registry sections plus wheel/sdist boundary inspection implemented | M8-02 |
| Comfy Registry artifact | Deterministic Git-indexed ZIP, manifest/source binding, normalized install, and read-only observation implemented | M8-06 |
| Adapter/integration tests | Implemented | M4-01 through M4-10 |
| Real ComfyUI H1 | Implemented | M2-05 harness |
| Real ComfyUI H2 | Turbo and RAW/auto implemented | M2-05; M3-06 |
| Real ComfyUI H3 | M5-01 deterministic native-Euler H3 implemented; partial-denoise execution is rejected | M5-01 |
| MiniMax H3 model-free host contract | Explicit-variant scheduler/probe/verifier and optional pinned 0.30.0 runner implemented; host execution pending explicit authorization | M6-05 subordinate qualification |
| Frontend policy | Implemented in the default full gate with Node.js 18+; hosted CI uses Node.js 20 | M4-11 amendment |
| Browser E2E | Bounded M4-11 item-specific browser evidence accepted; reusable automated Playwright lane `NOT_IMPLEMENTED` | Activate when frontend behavior requires maintained real-browser regression coverage |
| GPU/real-model tests | Optional and unapproved | Explicit requirement and authorization |
| CI workflows | Configured; hosted four-cell full gate and three pinned parity jobs passed at commit `070cb12fe444fed138b8ea601dcde3657a460647` | M0-06 plus accepted hosted evidence |

An unavailable lane is never a pass.

## 3. Gate Classes

| Gate | Intended use | Required content | Acceptance authority |
| --- | --- | --- | --- |
| `fast` | Developer inner loop | Changed-area targeted tests | Never sufficient alone |
| `pr-core` | Every pull request | Policy, static, pure core, golden, packaging as available | Blocking |
| `pr-host` | Node, adapter, workflow, or host changes | H0, H1, H2 on known-good host | Blocking when applicable |
| `full-local` | Pre-push and implementation acceptance | Every applicable implemented stage | Blocking |
| `scheduled-compat` | Nightly/weekly | Latest-host, wider matrix, mutation/fuzz | Review required |
| `release` | Release candidate | Full gate, known-good host, clean install, audit, parity | Blocking |
| `optional-heavy` | Approved GPU/real-model run | Explicit H4 protocol | Blocking only when required |

Change-aware selection may accelerate `fast`. It must not remove a required `pr-core`,
`pr-host`, `full-local`, or `release` lane.

## 4. Regression Risk Matrix

| Risk | Primary automated tests | Required cross-check |
| --- | --- | --- |
| Base-grid formula drift | Unit + property + golden vector | Pinned official/Diffusers differential parity |
| Shift parameterization mix-up | Table-driven unit + negative contracts | `mu` versus direct-ratio metamorphic relation |
| Terminal/slicing off-by-one | Boundary/property tests | Full, partial, empty, invalid-range cases |
| Float/dtype instability | Float64/float32 parameter matrix | Tolerance and fingerprint normalization |
| RAW dimension/sequence drift | Property/table cases | Square, landscape, portrait, endpoint, invalid, out-of-range |
| RAW/Turbo ambiguity | Resolver negative tests | Strict host workflow fails closed |
| Evidence/profile override drift | Schema and serialization tests | Official override becomes `modified` |
| Double shift/wrong ownership | Adapter integration matrix | H2 native, external-sigma, and patch workflows |
| Inert node control | Input-to-executed-path contract | H2 result or error changes with the control |
| Node ID/schema drift | V3 schema snapshot | H1 `/object_info` and reload/idempotency |
| Frozen public-contract drift | Manifest generator check + strict loader negatives | Migration-policy review and package inventory |
| Global import side effect | Fresh subprocess import | PyTorch call identity and optional dependency absence |
| Workflow metadata drift | JSON round-trip tests | H2 save/reload fingerprint and profile version |
| Sampler step drift | Step-level differential/property tests | Fixed seed/state, partial denoise, terminal behavior |
| Host API drift | Known-good plus latest-host lanes | Actionable compatibility error; no generic fallback |
| Package leakage | Build/clean-install inspection | No internal docs, caches, weights, secrets, or private paths |
| Release archive leakage | Canonical release audit + malicious ZIP/TAR tests | Fresh wheel and sdist semantic inventories |
| Registry candidate drift | `.comfyignore` ZIP validator + malicious member tests | Windows/WSL byte identity and renamed-directory import |
| Platform/process failure | Windows/Linux + Unicode path | Port, readiness, shutdown, and cleanup verification |

## 5. CI Job Role Matrix

Exact Python and dependency versions are frozen by M0/M7. This matrix defines roles, not
unreviewed version numbers.

| Job | PR | Push | Scheduled | OS/profile | Must produce |
| --- | --- | --- | --- | --- | --- |
| `policy-contract` | Yes | Yes | No | Ubuntu, one supported Python | Workflow/config self-tests |
| `quality-security` | Yes | Yes | No | Ubuntu | Secret/static/type/dependency results |
| `core` | Yes | Yes | Yes | Ubuntu Python min/max; Windows representative | JUnit + branch coverage |
| `golden` | Yes | Yes | Yes | Ubuntu + representative Windows | Full vectors + fingerprints |
| `parity-pinned` | When Tier 1 affected | Yes | Yes | Isolated pinned optional env | Max/mean error artifact |
| `adapter-contract` | When adapter affected | Yes | Yes | Ubuntu + Windows representative | Schema/registration report |
| `host-known-good` | When host affected | Yes | Yes | Pinned ComfyUI CPU host | Redacted host log + applicable H1/H2/H3 evidence |
| `host-latest` | No | No | Yes/manual | Latest reviewed ComfyUI | Compatibility report |
| `package` | Yes | Yes | Yes | Supported Python matrix | Wheel/sdist inventory, Registry ZIP, clean installs |
| `mutation-property` | No | No | Yes/manual | Pure-core environment | Survivors + reproducible seeds |
| `optional-heavy` | No | No | Manual/release only | Approved GPU/model matrix | H4 record |

Known-good Tier 1 and release lanes are blocking. A latest-host failure opens compatibility
review and cannot silently redefine supported behavior.

## 6. Test-Layer Boundaries

### Pure Core

- No ComfyUI or Diffusers import.
- No network, GPU, model weight, or host process.
- Deterministic by default.
- Static import-root checks plus isolated all-module import blockers.
- Deterministic property/metamorphic relations run in the default gate.

### Framework Parity

- Pinned optional environment.
- No unreviewed reference checkout execution.
- Complete vectors and error statistics, not selected-point assertions.

### Native ComfyUI Parity

- Exact reviewed ComfyUI revision and behavior-bearing blobs.
- Isolated Python 3.13 dependency lock; no default-runtime dependency change.
- Offline CPU import of the actual `ModelSamplingFlux` and registered `simple` scheduler.
- Complete 4/8/12/16-step vectors, bounded table-quantization policy, and canonical evidence.
- No model weights, host server, workflow execution, or copied GPL implementation.

### Adapter Contract

- May import reviewed ComfyUI APIs or controlled stubs.
- Must prove registration, schema, ownership, and error behavior.
- Cannot substitute for a real-host lane.

### Real Host

- Real pinned ComfyUI process on loopback.
- Isolated user/input/output/temp/custom-node roots.
- Lightweight test nodes/profiles and CPU workflows by default.
- Public HTTP/WebSocket interfaces and `/object_info`.

### Heavy Conformance

- Explicit authorization, pinned hashes, isolated caches, and separate artifacts.
- Supplemental to mathematical parity, never its replacement.

## 7. Coverage Governance

When source code exists:

1. Collect statement and branch coverage in the normal core test job.
2. Establish an observed reviewed baseline; do not invent a percentage before measurement.
3. Store the accepted floor in a versioned policy file.
4. Ratchet upward; reducing the floor requires an approved rollback record.
5. Report repository totals and high-risk module families separately.
6. Upload machine-readable coverage data and a human-readable summary.
7. Require new branches in critical modules to have behavioral evidence or a reviewed
   exception.

Planned hotspot families:

- grids/shifts/transforms/slicing;
- profile schema/resolution/evidence;
- ComfyUI registration/ownership;
- sampler state/stepping/randomness;
- import/package/optional-dependency boundaries.

Coverage is a gap detector, not proof of numerical correctness. Deterministic property tests,
complete Krea 2 Turbo 4/8/12/16-step goldens, authoritative Turbo differential parity, native
ComfyUI Turbo schedule parity are implemented. Complete RAW 28/52-step geometry goldens are
also implemented, together with pinned authoritative/framework parity over all 14
recipe/geometry cases. Pure Krea variant-resolution tests enforce strong-evidence conflicts,
suggestion-only weak signals, and family-only model/tensor signals. Scheduled mutation
evidence remains mandatory when its roadmap stage activates. MiniMax H3 keeps its Diffusers
endpoint-inclusive and native ComfyUI `simple` vectors separate; the source-pinned native
adapter/report contract is present, but hosted Python 3.13 regeneration and authorized model-host
evidence remain open.

## 8. Skip, XFail, Retry, and Quarantine Policy

Tier 1 golden/parity, ambiguity fail-closed, double-shift, registration, import-safety, and
package-leakage tests are no-skip seams.

Every permitted skip, xfail, or quarantine record must include:

- owner;
- reason;
- linked issue or ADR;
- introduced date;
- review/expiry date;
- affected test IDs;
- replacement evidence;
- promotion/removal condition.

Rules:

- Unexpected pass is a failure until the expectation is removed.
- Expired entries fail the policy check.
- Retry counts and first-attempt failures are reported separately.
- Retries may diagnose infrastructure flakiness; they cannot turn a first-attempt P0
  behavioral regression into passing acceptance evidence.
- Flaky tests remain owned test debt and may not disappear silently.

## 9. CI Security and Reproducibility

Every workflow must:

- declare minimal permissions;
- cancel superseded unprivileged PR runs;
- avoid write tokens and secrets when executing fork code;
- pin actions according to the approved immutable/version policy;
- use bounded, locked, or constrained dependency inputs;
- audit declared production dependencies separately from CI tools;
- avoid cache sharing across incompatible trust boundaries;
- use deterministic locale, timezone, and seeds where relevant;
- retain redacted failure artifacts with bounded retention;
- avoid model downloads in default PR jobs;
- never publish artifacts from an unreviewed or failed release gate.

## 10. CI Self-Tests

CI configuration is executable behavior. Current contracts verify or reserve the following
requirements for their owning lane:

- required workflows and jobs exist;
- job commands reference canonical repo scripts;
- local and CI gates invoke the same underlying stages;
- workflow matrices stay inside supported policy;
- action refs and permissions comply with policy;
- host/parity jobs upload diagnostics even on failure;
- the required Node.js frontend-policy stage remains separate from pure-core and optional
  Playwright dependencies;
- required P0 suites cannot be filtered or skipped;
- scheduled/latest/optional lanes cannot masquerade as known-good evidence;
- full-gate scripts fail when hooks mutate the worktree or index;
- missing prerequisites fail with actionable output.

These tests must be introduced before or with the first executable workflow.

## 11. Artifact Contract

As applicable, jobs retain:

- JUnit XML;
- branch-coverage JSON/XML and summary;
- golden/parity vectors, versions, fingerprints, and max/mean errors;
- property/fuzz seeds and minimized failing examples;
- mutation survivors;
- redacted ComfyUI startup/execution logs;
- workflow fixture version and result metadata;
- package inventory and dependency audit result.

Artifacts must not contain secrets, user data, model files, private paths, or unredacted
environment dumps.

## 12. Activation Sequence

```text
M0-01 Git identity
└── M0-02 audited source import
    └── M0-03 import-safety hardening
        └── M0-04 package/dependency layout
            └── M0-05 test runner, policies, and quality tooling
                └── M0-06 local/CI orchestration and workflow self-tests
                    └── M0-07 public documentation
```

Later layers activate with their production subjects:

- M1: pure core/property/fingerprint;
- M2/M3: golden and authoritative parity;
- M4: adapter, schema, H1/H2;
- M5: step-level/H3;
- M7: compatibility, performance, mutation, image supplements;
- M8: release/upgrade/rollback.
