# Compatibility

## Current Status

ComfyUI-Sigmax 1.0.0 is the stable public-contract baseline. The Krea 2 Turbo structural profile
and formula-composed schedule builder are implemented, along with `krea2.raw.official`
geometry, sequence-length, and dynamic-`mu` derivation. A static dependency-free ComfyUI adapter
normalizes reviewed public host/node schemas and derives model/host/sampler capability evidence.
A pure `sigmax.node-registration/1` catalog validates legacy/current, V3, `/object_info`, and
Node Definition JSON v2 registration shapes with explicit namespaced IDs and fail-closed
collisions. Its built-in catalog contains the statically validated legacy/current
`Sigmax.AdvancedFlowMatchScheduler`, an experimental explicit `UNIT_FLOW` constructor returning
`sigmax.advanced-flowmatch-node/1` information; `Sigmax.Krea2SigmaScheduler`, which explicitly
constructs Turbo or RAW sigmas and returns
structured `sigmax.krea2-sigma-node/1` schedule information, plus
`Sigmax.ModelAwareSigmaScheduler`, which requires a Krea 2 MODEL, exposes family-only Auto
ambiguity, and returns exact-profile capability decisions under
`sigmax.model-aware-sigma-node/1`; plus the read-only `Sigmax.ProfileInspector` and
`Sigmax.ScheduleComparison`, and `Sigmax.ScheduleInspector`, which return
`sigmax.profile-inspector/1`, `sigmax.schedule-comparison/1`, and
`sigmax.schedule-inspector/1` reports. Separate `Sigmax.RawWorkflowOutput` and
`Sigmax.TurboWorkflowOutput` nodes publish verified model-free artifact bundles through prompt
history. Real-host import and model-free workflow execution are validated on pinned ComfyUI
`0.29.0` revision `e651b7bef55a5376343dcb1c0edb79f0142c985e`. Controlled deterministic
native-Euler sampler steps are validated there through complete eight-step latent traces,
independent flow-equation recomputation, exact execution counts, deterministic reruns, and an
artifact-linked success receipt. RAW native model execution, real Krea model weights, GPU
execution, image generation, stochastic/resumable semantics, partial-denoise execution, and
advanced workflows are **not yet validated**. Complete Turbo golden vectors, authoritative
framework parity, and native ComfyUI schedule parity are validated at 4, 8, 12, and 16 steps.
RAW authoritative and framework schedule parity is validated across 14 complete 28/52-step
geometry cases.
The dependency-free `ProfileSchemaV1` contract is frozen for these validated external-sigma
profiles, including separate source/framework/weight license provenance. The immutable
exact-key `ProfileRegistry` and explicit inheritance/conflict policy are implemented without
file or plugin loading. Native or patched schedule-ownership schemas, external document
parsing and native/patched ownership schemas are not yet implemented.

The separate `sigmax.generic-flowmatch-profile/1` fixed/dynamic declarations describe explicit
schedule structure only. They are not compatibility rows, do not enter the concrete model
registry, and cannot promote a model family, host, sampler, or image-quality claim.

The package metadata declares a ComfyUI floor of `0.29.0`. Adapter fixtures and pinned source
review define the exact static-contract window, while the repository H1/H2 harness defines the
real-host node/workflow E2E window at `0.29.0` revision
`e651b7bef55a5376343dcb1c0edb79f0142c985e`; this remains narrower than the packaging
declaration. The H3 deterministic-Euler proof uses the same exact host revision. Other revisions
require separate evidence.

Versioned execution receipts and portable artifact/receipt bundles are implemented as
dependency-free pure contracts. They record explicit status, counts, component identities, RNG
ownership, compatibility, and fingerprints. The RAW/Turbo output nodes create canonical
`not_executed` receipts after host schedule construction and inspection. They do not claim that
a model or sampler ran. The separate M5-01 H3 lane validates a controlled native-Euler execution
trace first and then builds a `succeeded` receipt against the unchanged Turbo artifact; that
receipt does not upgrade the H2 workflow receipts or establish real-model execution.

Workflow metadata supports copy-on-write attachment to official ComfyUI workflow forms `0.4`
and `1`. It preserves unrelated graph and `extra` data and verifies the embedded Sigmax metadata
envelope. Metadata parsing alone does not validate node/link/widget compatibility; the pinned
H2 lane separately proves metadata reload for every published RAW workflow.

The separate `comfyui_sigmax.workflows` validator now checks canonical model-free Turbo/RAW
fixtures against the pinned 0.29.0 legacy/v2 schema baseline or caller-observed live
`/object_info`. It reports stable issue kinds, package/node/host versions, and strict
known-good versus observational latest-host results. Its optional HTTP loader is
literal-loopback-only and bounded. A clean validation report remains schema evidence only.
The separate `H2_RAW_M3_06` lane in `scripts/run_comfyui_e2e.py` executes the square, landscape,
and portrait RAW graphs to completed history on the pinned host, verifies workflow metadata
reload and canonical bundles, and proves two fail-closed boundaries: an ambiguous variant
produces a terminal runtime rejection with no partial output, while invalid steps produce a
structured prequeue HTTP 400 rejection without a prompt ID.

## Validated Foundation Environments

The current package, quality gates, tests, and wheel inventory have been validated locally on:

| Environment | Python | Evidence scope |
| --- | --- | --- |
| Windows | 3.13 | Core independence, deterministic property, Turbo/RAW golden and parity contracts, package, quality, unit, coverage, and wheel gates |
| WSL2/Linux path | 3.10 | Core independence, deterministic property, Turbo/RAW golden and parity contracts, package, quality, unit, coverage, and wheel gates |
| Hosted Linux parity lanes | 3.13 | Exact Diffusers 0.39.0, NumPy 2.3.4, and Torch 2.9.0 Turbo/RAW report regeneration |
| Pinned ComfyUI host | 3.13 | H1 import/schema safety and H2 model-free Turbo/RAW workflow execution on ComfyUI 0.29.0 revision `e651b7b…` |
| Latest ComfyUI release observation | 3.13 | Non-blocking H1/H2/H3 first/repeat execution on official v0.29.2 revision `32212244…`, Torch 2.13.0+CPU, no model weights |
| Latest ComfyUI HEAD observation | 3.13 | Non-blocking H1/H2/H3 first/repeat execution on pinned HEAD `5cc026f5…`, Torch 2.13.0+CPU, no model weights |

The supported Python floor is 3.10. Python versions or operating systems not listed above may
work, but do not yet have repository acceptance evidence.

## Dependency Boundary

| Component | Current policy |
| --- | --- |
| Runtime Python dependencies | None |
| Development tools | Version-bounded `dev` extra |
| Diffusers | Optional `reference` extra, currently `>=0.39,<0.40` |
| Matplotlib | Optional `plot` extra, currently `>=3.10,<3.12`; lazy and outside core |
| Numerical benchmark matrix | Packaged canonical JSON; dependency-free loader; no model weights |
| Dependency compatibility matrix | Packaged canonical JSON; fixed local runner; no implicit acquisition |
| Co-installation mutation matrix | Packaged canonical JSON; fixed synthetic operations; no external code |
| Performance budget matrix | Integer-unit first/repeat limits; Windows/WSL pure lanes plus pinned host startup |
| Environment diagnostics | Local venv/cache/lock/Unicode/temp/optional checks before Windows/WSL full gates |
| ComfyUI | Optional host; not imported by the package shell or pure adapter |
| Node/browser tooling | Not required by the current Python-only foundation |
| Model weights and GPU runtime | Not downloaded or exercised |

Diffusers is intended as a pinned parity reference or optional backend. Closed-form schedule
construction does not require it at runtime. The canonical pure lane requires both ComfyUI and
Diffusers to be absent, blocks attempted imports, and enumerates every core and profile module
in Python isolated mode.

Canonical schedule and comparison reports remain standard-library-only. PNG/SVG rendering is a
separate optional path: Python 3.10 resolves the compatible Matplotlib 3.10 line, while current
Python 3.11+ environments may use 3.11. Plot output is not required to import Sigmax, build or
compare reports, execute nodes, or validate artifacts and receipts.

## Dependency Compatibility Evidence

The versioned `sigmax.dependency-compatibility-matrix/1` resource distinguishes blocking
`known_good`/`supported` lanes from `latest_informational` observations. Latest observations
cannot silently expand support, and unavailable lanes are never PASS.

The fixed invariant contract currently passes twice on native Windows Python 3.13.9 and WSL
Python 3.10.12 with identical source/test-selection identities and zero mandatory runtime
dependencies. It binds Turbo/RAW goldens and parity, workflow schemas, capability/receipt
conformance, serialization identities, and the M7-02 numerical matrix.

The pinned framework baseline remains Diffusers 0.39.0 with Torch 2.9.0. The pinned real-host
baseline remains ComfyUI 0.29.0 revision
`e651b7bef55a5376343dcb1c0edb79f0142c985e`, Python 3.13, Torch 2.13.0, and experimental
numbered API `v0_0_2`. These are accepted source-evidence references, not newly rerun external
lanes.

Python 3.10 reaches upstream end of life in October 2026, so its floor is a dated compatibility
commitment that must be reviewed at each release. Official ComfyUI v0.29.2
and pinned HEAD `5cc026f5…` passed separate non-blocking CPU/no-model H1/H2/H3 first/repeat
observations. The official Comfy Org CI-container row remains `unavailable`: its mutable tag and
source were observed, but an immutable registry digest could not be obtained. No third-party
image substitutes for that lane.

## Co-Installation and Host-Mutation Evidence

The versioned `sigmax.host-mutation-snapshot/1` contract and
`sigmax.co-installation-mutation-matrix/1` resource contain ten first/repeat synthetic
observations bound to the exact built-in node catalog and accepted dependency compatibility
matrix. They allow clean/reload-idempotent behavior and unrelated node or scheduler additions.
They reject replacement of an existing node or scheduler, an external `Sigmax.` namespace
registration, a PyTorch call-path change, shared model-patch mutation, and repeated or
model-native-plus-external double-shift behavior.

The runner applies only fixed declarative repository-owned operations to immutable in-memory
snapshots. It does not import ComfyUI or Torch, mutate a live host, execute `reference/` or
third-party code, or claim compatibility with a named node pack. A named-pack observation
requires a separate reviewed plan and explicit approval.

## Performance Evidence

`sigmax.performance-budget-matrix/1` separates normative regression limits from
machine-specific observations. Windows Python 3.13 and WSL Python 3.10 execute fixed Turbo/RAW
schedule latency and peak-allocation workloads, isolated package startup, and an instrumented
CPU tensor boundary. The boundary requires one tensor construction, one host round-trip, and
zero explicit device transfers. Two fresh pinned ComfyUI 0.29.0 CPU/no-model processes provide
first/repeat readiness evidence under a 30-second ceiling.

These limits detect major regressions; they are not cross-machine wall-clock guarantees. GPU,
model weights, latest-host, and official-container performance are `not_evaluated` and cannot
borrow a PASS from CPU or synthetic evidence.

## Environment Failure Diagnostics

`sigmax.environment-diagnostics/1` runs before the canonical Windows and WSL gates. It validates
the selected `.venv`/`.venv-wsl`, repository-local pre-commit cache and temp roots, read-only
SQLite cache integrity, pre-commit executable consistency, file replace/delete behavior, a
non-ASCII filename round-trip, and fixed optional-lane availability. Failures use stable issue
codes and an exact remediation sequence; the diagnostic never repairs or deletes a cache. WSL
mounted paths that fail anonymous temp-file semantics use an explicit `pytest.capture_sys`
mitigation, while an unmitigated direct run fails as `temp.incompatible`.

## Planned Validation Tiers

Compatibility claims will progress through separate lanes:

1. pure schedule and deterministic property tests — implemented;
2. authoritative golden tests — Turbo and RAW implemented;
3. authoritative framework parity tests — Turbo and RAW implemented with Diffusers 0.39.0;
4. native ComfyUI schedule parity — Turbo implemented against a pinned host revision;
5. real ComfyUI host import and model-free node/workflow integration — implemented for the
   pinned 0.29.0 revision;
6. fixed-seed sampler and latent-level comparison;
7. approved model/GPU workflows;
8. latest-host compatibility signals.

Passing a lower tier does not imply a higher tier.

New or changed model profiles must satisfy the
[model profile contribution guide](PROFILE_CONTRIBUTION_GUIDE.md). In particular, source,
framework, and weight licenses remain separate; evidence level is scoped to the executed proof;
and a validator-clean fixture does not imply supported-host, sampler, model-weight, GPU, or image
conformance.

The implemented property lane checks mathematical and serialization invariants. The Krea 2
Turbo structural profile pins evidence and the golden lane compares the production builder
against complete independent float64 and float32 fixtures. The initial bounds are `1e-8` and
`1e-6`, respectively.

The authoritative parity report additionally compares complete production vectors against
the pinned Krea formula in float64 and an actual Diffusers 0.39.0
`FlowMatchEulerDiscreteScheduler` run in float32. Across 4, 8, 12, and 16 steps, the largest
observed Krea error is `1.1102230246251565e-16` and the largest Diffusers error is
`5.960464477539063e-08`. Exact source revisions, dependency versions, CPU/dtype, mean and
maximum errors, tolerances, and fingerprints are stored in the committed report and
regenerated by a separate hosted CI lane.

The native ComfyUI parity report separately imports the actual pinned `ModelSamplingFlux`
and registered `simple` scheduler in an isolated CPU environment. The 4-, 8-, and 16-step
cases use exact table positions and stay within `1e-6`; the 12-step case records the expected
10,000-table integer-index quantization under a `2e-4` bound. This is native schedule
evidence, not a real-host, node, sampler-step, checkpoint, or image-level claim.

The RAW structural profile records the resolution-linear exponential shift from sequence
length 256 / `mu=0.5` through 6400 / `mu=1.15`, with explicit upstream-unclamped
extrapolation. It also keeps the 52-step official-full and 28-step Diffusers-reference
guidance recipes separate. Requested dimensions are retained, effective dimensions are
rounded upward to 16, and the packed image sequence length and unclamped dynamic `mu` are
calculated by dependency-free pure functions. Exact named 28- and 52-step builders are
validated against 14 complete independent float64/float32 golden cases across square,
landscape, and portrait geometry. The canonical RAW parity report separately executes all 14
cases against the pinned Krea formulas and Diffusers 0.39.0. The largest observed Krea error
is `9.992007221626409e-16`; the largest Diffusers float32 error is
`1.1920928955078125e-07`. Both remain below the enforced `1e-8` and `1e-6` bounds.
Canonical float64 evidence uses a declared 15-significant-digit normalization so sub-bound
platform `libm` noise does not alter report bytes.

## Current Known Limitations

- The eight node mappings are `Sigmax.AdvancedFlowMatchScheduler`,
  `Sigmax.Krea2SigmaScheduler`, `Sigmax.ModelAwareSigmaScheduler`, `Sigmax.ProfileInspector`,
  `Sigmax.RawWorkflowOutput`, `Sigmax.ScheduleComparison`, `Sigmax.ScheduleInspector`, and
  `Sigmax.TurboWorkflowOutput`; their schemas load on the pinned host and the four model-free
  publication workflows complete.
- Inspectors are read-only: profile inspection requires explicit Krea variant resolution and a
  bounded native sampling class, while schedule inspection accepts only implemented Sigmax
  schemas and requires connected SIGMAS to match the advertised output fingerprint. Schedule
  comparison aligns only verified equal-domain/equal-length inputs and reports mismatches
  without conversion or resampling.
- The advanced FlowMatch node is an `experimental` external `UNIT_FLOW` constructor with explicit
  linear endpoints, one exponential-mu or direct-ratio shift, terminal policy, and slicing. It
  is not evidence that the result is compatible with an arbitrary model.
- Krea-specific variant resolution and a bounded static ComfyUI evidence adapter exist, but no
  live transport or automatic model-file inspection exists; family-only Auto mode therefore
  rejects as ambiguous.
- The pure schedule/artifact/capability core, dedicated Turbo/RAW profiles, and explicit Krea 2
  sigma-scheduler nodes exist, but no cross-family generic fallback or full sampler is exposed.
- Filename and local-header matches are suggestions only; the shared ComfyUI model class and
  common tensor keys are family-only and cannot resolve RAW versus Turbo.
- Real-host evidence is deliberately limited to exact ComfyUI 0.29.0 revision
  `e651b7bef55a5376343dcb1c0edb79f0142c985e` and model-free graphs; it does not generalize to
  later hosts or model/sampler/image execution.
- No current node can manually assert a successful execution receipt; successful real-host
  receipt production remains pending an executed and validated sampler path.
- The current `v0_0_2` V3 API is discoverable but experimental; activation is rejected when a
  stable numbered API is required.
- macOS and native hosted Ubuntu evidence are not yet available.
- Image-quality comparisons are not correctness evidence and have not begun.

For the intended component boundaries, see [Architecture](ARCHITECTURE.md).
