# Compatibility

## Current Status

ComfyUI-Sigmax is a pre-alpha development foundation. The Krea 2 Turbo structural profile
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
`Sigmax.ScheduleInspector`, which return `sigmax.profile-inspector/1` and
`sigmax.schedule-inspector/1` reports. Live ComfyUI host integration, RAW
native-ComfyUI parity, model weights, GPU
execution, and sampler-step behavior are
**not yet validated**. Complete Turbo golden vectors, authoritative framework parity, and
native ComfyUI schedule parity are validated at 4, 8, 12, and 16 steps. RAW authoritative
and framework schedule parity is validated across 14 complete 28/52-step geometry cases.
The dependency-free `ProfileSchemaV1` contract is frozen for these validated external-sigma
profiles, including separate source/framework/weight license provenance. The immutable
exact-key `ProfileRegistry` and explicit inheritance/conflict policy are implemented without
file or plugin loading. Native or patched schedule-ownership schemas, external document
parsing and native/patched ownership schemas are not yet implemented.

The package metadata declares a ComfyUI floor of `0.29.0`. Adapter contract fixtures and pinned
source review currently define an exact static-contract window of `0.29.0`; this is narrower than
the packaging declaration and is not real-host node/workflow evidence. No release should infer
working host execution from either declaration alone.

## Validated Foundation Environments

The current package, quality gates, tests, and wheel inventory have been validated locally on:

| Environment | Python | Evidence scope |
| --- | --- | --- |
| Windows | 3.13 | Core independence, deterministic property, Turbo/RAW golden and parity contracts, package, quality, unit, coverage, and wheel gates |
| WSL2/Linux path | 3.10 | Core independence, deterministic property, Turbo/RAW golden and parity contracts, package, quality, unit, coverage, and wheel gates |
| Hosted Linux parity lanes | 3.13 | Exact Diffusers 0.39.0, NumPy 2.3.4, and Torch 2.9.0 Turbo/RAW report regeneration |

The supported Python floor is 3.10. Python versions or operating systems not listed above may
work, but do not yet have repository acceptance evidence.

## Dependency Boundary

| Component | Current policy |
| --- | --- |
| Runtime Python dependencies | None |
| Development tools | Version-bounded `dev` extra |
| Diffusers | Optional `reference` extra, currently `>=0.39,<0.40` |
| ComfyUI | Optional host; not imported by the package shell or pure adapter |
| Node/browser tooling | Not required by the current Python-only foundation |
| Model weights and GPU runtime | Not downloaded or exercised |

Diffusers is intended as a pinned parity reference or optional backend. Closed-form schedule
construction does not require it at runtime. The canonical pure lane requires both ComfyUI and
Diffusers to be absent, blocks attempted imports, and enumerates every core and profile module
in Python isolated mode.

## Planned Validation Tiers

Compatibility claims will progress through separate lanes:

1. pure schedule and deterministic property tests — implemented;
2. authoritative golden tests — Turbo and RAW implemented;
3. authoritative framework parity tests — Turbo and RAW implemented with Diffusers 0.39.0;
4. native ComfyUI schedule parity — Turbo implemented against a pinned host revision;
5. real ComfyUI host import and node integration;
6. fixed-seed sampler and latent-level comparison;
7. approved model/GPU workflows;
8. latest-host compatibility signals.

Passing a lower tier does not imply a higher tier.

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

- The five node mappings are `Sigmax.AdvancedFlowMatchScheduler`,
  `Sigmax.Krea2SigmaScheduler`, `Sigmax.ModelAwareSigmaScheduler`, `Sigmax.ProfileInspector`,
  and `Sigmax.ScheduleInspector`; their pure behavior and static schemas are covered without
  claiming real-host loading or workflow execution.
- Inspectors are read-only: profile inspection requires explicit Krea variant resolution and a
  bounded native sampling class, while schedule inspection accepts only implemented Sigmax
  schemas and requires connected SIGMAS to match the advertised output fingerprint.
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
- No ComfyUI version has completed real-host node/workflow E2E validation.
- The current `v0_0_2` V3 API is discoverable but experimental; activation is rejected when a
  stable numbered API is required.
- macOS and native hosted Ubuntu evidence are not yet available.
- Image-quality comparisons are not correctness evidence and have not begun.

For the intended component boundaries, see [Architecture](ARCHITECTURE.md).
