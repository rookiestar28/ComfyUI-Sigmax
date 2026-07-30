# Architecture

## Status

This document separates the **implemented foundation** from the **planned product
architecture**. Planned components are design targets and do not imply working ComfyUI nodes
or validated model support.

## Vocabulary

Sigmax uses four terms deliberately:

- **Model sampling semantics** map an external sigma or time value to the model's internal
  conditioning.
- **Schedule** is the ordered sigma or timestep sequence.
- **Sampler** is the numerical integration method that updates the latent.
- **Profile** is a versioned model-specific contract combining supported semantics, schedule,
  sampler compatibility, evidence, and recommendations.

A Diffusers class that owns both schedule creation and step behavior cannot automatically be
reduced to a ComfyUI `SIGMAS` output. Controls that affect `step()` belong in a sampler
implementation.

## Implemented Foundation

The repository currently implements:

```text
comfyui_sigmax/
  __init__.py   side-effect-free package contract
  py.typed      typing marker
  core/
    base_grids.py
                Krea reciprocal-step and generic descending linear grids
    schedule_contracts.py
                ownership, sigma-domain, and transform-chain preflight
    request_result.py
                immutable requested/effective inputs and structural results
    shifts.py
                exponential-mu, direct-ratio, and explicit identity transforms
    terminal_slicing.py
                terminal append/preserve, step ranges, and denoise-tail slicing
    validation.py
                complete external-schedule numerical and policy validation
    fingerprints.py
                canonical projections and numerical/construction identities
    artifacts.py
                immutable artifact assembly and strict canonical JSON transport
    execution_receipts.py
                execution evidence and portable artifact/receipt bundle transport
    workflow_metadata.py
                portable workflow requirements and non-destructive graph attachment
    capabilities.py
                model/profile/sampler declarations and execution preflight decisions
  profiles/
    schema_v1.py
                frozen profile contract, provenance separation, and canonical fingerprint
    registry.py
                exact-key immutable registry, conflicts, and explicit inheritance
    resolution.py
                pure profile/model/host/sampler capability composition
    krea2_common.py
                shared evidence, guidance, dimension policy, and packed-image geometry
    krea2_raw.py
                immutable RAW profile, dynamic-mu provenance, and exact-recipe builder
    krea2_turbo.py
                immutable official-recipe declaration and structural schedule builder
    krea2_variant.py
                pure evidence normalization and fail-closed RAW/Turbo resolution
  workflows/
    validation.py
                static/live workflow schema comparison and canonical machine reports
    fixtures.json
                model-free canonical Turbo/RAW graphs and ordered widget contracts
    host_baseline.json
                pinned legacy and Node Definition v2 host schemas

scripts/
  preflight_check.py          local environment validation
  check_core_independence.py  isolated pure-layer optional-framework blocker
  parity/                     independent official adapter and report contract
  run_krea2_turbo_parity.py   isolated pinned Diffusers execution
  run_krea2_raw_parity.py     isolated RAW dynamic-shift and scheduler execution
  run_comfyui_e2e.py          isolated pinned-host H1, Turbo/RAW H2, and native-Euler H3
  run_full_gate.py            canonical ordered acceptance gate
  OS wrappers                 repo-local environment selection

tests/
  numerical, artifact, capability, property, import, package, quality, CI,
  documentation contracts, and independent Turbo golden vectors

comfyui_sigmax/nodes/
  advanced_flowmatch_scheduler.py  explicit configurable unit-flow SIGMAS node
  inspectors.py  bounded read-only profile and schedule reports
  krea2_sigma_scheduler.py  thin explicit RAW/Turbo SIGMAS product node
  model_aware_sigma_scheduler.py  bounded MODEL probe and exact capability-gated profile node
  raw_workflow_output.py  verified named RAW artifact/receipt history boundary
  turbo_workflow_output.py  verified strict Turbo artifact/receipt history boundary
```

The dependency-free `adapters/registration.py` module owns the immutable node catalog and
wire-schema projections. The package exports the validated
`Sigmax.AdvancedFlowMatchScheduler`, `Sigmax.Krea2SigmaScheduler`, and
`Sigmax.ModelAwareSigmaScheduler`, `Sigmax.ProfileInspector`,
`Sigmax.RawWorkflowOutput`, `Sigmax.ScheduleComparison`, `Sigmax.ScheduleInspector`, and
`Sigmax.TurboWorkflowOutput` mappings. Importing them does not load Torch or ComfyUI, patch
PyTorch, import Diffusers, or alter host process state.

The full gate proves this boundary before the test suite: a clean project-local environment
must resolve neither ComfyUI nor Diffusers, and a `python -I` subprocess installs explicit
import blockers before enumerating and importing every core and profile module. Static AST
checks allow only Python standard-library and `comfyui_sigmax` import roots. Deterministic
property and metamorphic tests exercise mathematical invariants without either optional
framework.

The implemented pure-core preflight requires exactly one ownership mode:

- `MODEL_NATIVE`: the host/model sampling object owns the schedule;
- `EXTERNAL_SIGMAS`: Sigmax owns a complete external sigma chain;
- `MODEL_PATCH`: an explicit host adapter replaces the model sampling object.

Native and patched ownership accept no external transform chain. External ownership uses
explicit domains and the ordered stages `PRIMARY_TIME_SHIFT`, `OPTIONAL_SPACING`, `TERMINAL`,
and `SLICE`. Domain discontinuity, stage regression, and duplicate shift/spacing transforms
fail before numerical execution.

Immutable request/result contracts now bind that preflight to:

- requested and effective steps/dimensions;
- base-grid identity and output domain;
- terminal and slicing policy;
- source/profile provenance and evidence level;
- warnings and explicit requested-to-effective overrides;
- the structural sigma tuple and validated final domain.

Complete external schedules can now be checked for finite values, strict decrease, exact
transition count, domain bounds, and terminal policy. Validated results can be projected into
immutable schedule artifacts with separate numerical and construction identities, then
transported as bounded canonical JSON with strict untrusted-input rejection.

Execution evidence is a separate immutable layer. `ExecutionReceipt` binds the artifact's
construction/numerical identities to effective inputs, profile, complete compatibility decision,
host/model/sampler identities, explicit RNG ownership, requested/effective transition and
model-evaluation counts, and one truthful final status. `PortableExecutionBundle` carries the
unchanged schedule-artifact envelope beside the receipt envelope and verifies their fingerprints
and effective-input cross-links. Neither contract runs a sampler or infers success from
construction or static capability evidence.

Workflow metadata is a third reference layer. Its canonical projection records package, node,
host/API, profile, compatibility, artifact, and receipt requirements, while the surrounding
ComfyUI graph remains host-owned passthrough data. Attachment changes only the namespaced
`extra.comfyui_sigmax` member of supported version `0.4` and version `1` mappings. It does not
convert or validate nodes, links, widgets, models, or subgraphs.

The adjacent dependency-free `workflows/validation.py` boundary owns static/live schema
comparison. It loads packaged canonical Turbo/RAW model-free graphs, normalizes legacy
`/object_info` and Node Definition JSON v2 through the adapter, preserves a separate explicit
widget order, and emits fingerprinted `sigmax.workflow-validation-report/1` results. Known-good
findings block; latest-host findings remain explicitly observational. The optional acquisition
helper accepts only bounded literal-loopback `/object_info`. It does not import node
implementations, launch ComfyUI, execute a graph, or replace real-host H1/H2.

The separate `scripts/run_comfyui_e2e.py` boundary stages the extension into owned isolated
state and launches pinned ComfyUI `0.29.0` revision
`e651b7bef55a5376343dcb1c0edb79f0142c985e`. H1 verifies import safety, the eight-node
registry, and live schemas. H2 executes the accepted Turbo graph and three orientation-complete
RAW graphs to completed history. RAW history verification cross-checks requested/effective
geometry, sequence length, recipe/evidence, dynamic `mu`, canonical artifact and receipt
fingerprints, external ownership, one shift, and submitted workflow metadata reload. Its
receipt status stays `not_executed` because no model or sampler step runs.

M5-01 activates a separate H3 proof in the same owned host lifecycle. A namespaced test-only
node pack, excluded from the wheel and production registry, invokes the pinned host's actual
`comfy.k_diffusion.sampling.sample_euler` on a nontrivial controlled flow-velocity fixture. The
driver validates every sigma, input, velocity, denoised value, output state, transition and
model-evaluation count against an independent framework-free Euler oracle, requires an identical
second run, then binds a `succeeded` receipt to the existing Turbo schedule artifact. Existing
H2 receipts remain `not_executed`; the H3 proof does not add a duplicate public sampler or imply
model-weight, GPU, image, stochastic, resumable, partial-denoise-execution, or advanced-workflow
support.

The first numerical builders are now implemented:

- Krea reciprocal-step returns the non-terminal unit-flow values from `1` through `1 / steps`;
- generic linear endpoint construction returns a finite strictly descending grid in an
  explicit non-opaque domain.

Terminal zero is not part of either builder. It remains a later terminal-stage operation.

The first pointwise shifts are also implemented:

- Krea/Flux exponential `mu`, including an explicit positive exponent;
- discrete-flow direct ratio, named separately from `mu`;
- an explicit no-shift identity path.

They accept only finite unit-flow tuples and reject model-native or other domains. Exact zero
and one endpoints are preserved, and algebraically equivalent stable evaluation prevents
extreme finite controls from overflowing. Krea 2 RAW resolution-to-`mu` derivation is
implemented in its evidence-pinned profile layer; generic dynamic-shift policy remains
future profile/registry work.

Terminal and slicing operations are now implemented separately:

- append terminal zero or preserve an already terminal-inclusive vector;
- interpret `N + 1` sigmas as `N` transitions;
- retain strict `start_step:end_step + 1` transition ranges;
- calculate ComfyUI-compatible partial-denoise construction counts and retain the requested
  terminal-inclusive tail;
- represent zero denoise as an explicit empty no-execution vector.

Manual empty or out-of-range requests fail. ComfyUI's later `force_full_denoise` endpoint
replacement remains a host/sampler execution policy and is not hidden in terminal
construction.

Framework-independent capability declarations now define the compatibility boundary before
host integration:

- `ModelCapabilities` declares accepted prediction types, sigma domains, schedule ownerships,
  and support for partial denoise or per-token timesteps.
- `ProfileCapabilities` binds one model family/variant to prediction, domain, ownership,
  terminal behavior, permitted deterministic/stochastic modes, noise owners, sampler state,
  feature support, and reference sampler identifiers.
- `SamplerCapabilities` declares accepted semantics plus terminal, state, noise, and execution
  requirements.
- `CompatibilityDecision` returns a canonical `ALLOW`, `WARN`, or `REJECT` result with stable
  reason codes across every considered capability dimension.

The execution gate raises before sampler work on `REJECT`. A compatible non-reference sampler
is a warning rather than an unsupported claim.

The dependency-free `profiles/schema_v1.py` module freezes `ProfileSchemaV1` under
`sigmax.model-profile/1`. It composes identity, schedule construction, recipes, detection,
capabilities, artifact versions, bounded extension fields, and known limitations. Software
source, framework, and model-weight provenance use distinct versioned types with independent
license declarations, so one artifact's license cannot be inherited by another. Cross-field
validation rejects inconsistent domains, transforms, terminal semantics, recipe sources,
capabilities, or provenance identities. `profile_schema_fingerprint()` hashes a canonical
typed projection without importing ComfyUI or Diffusers.

The dependency-free `profiles/registry.py` module implements `ProfileRegistry` as a
copy-on-write canonical tuple of complete schemas. Exact `ProfileKey` lookup prevents
implicit version or namespace fallback. External conflicts reject by default; explicit
compare-and-swap replacement can affect only an existing external entry, never a built-in.
Inheritance resolves at registration time by comparing every top-level schema field with an
already registered parent. The declared canonical override set must equal the actual
semantic difference, and inherited external children require `modified` evidence.

The dependency-free `profiles/resolution.py` module composes a registered profile with normalized
model identity/capabilities, host lifecycle evidence, sampler capabilities, and requested execution
features. It returns the versioned `sigmax.capability-resolution/1` decision, retains the exact
profile key/fingerprint and host revision, and namespaces the existing core compatibility reasons.
Only confirmed identity is executable. Required host capabilities must be `landed`; missing,
`experimental`, and `unsupported` lifecycle evidence reject before execution. The module performs
no host probing, schedule construction, or sampling.

The dependency-free `adapters/comfyui.py` module is the public ComfyUI evidence boundary. It
accepts an already loaded API module object and reads only `ComfyAPI.VERSION`,
`ComfyAPI.STABLE`, `ComfyExtension`, `io`, and `ui`; it never imports host-controlled module text.
It normalizes both the V1-compatible `/object_info` projection used for legacy and V3 nodes and
the documented Node Definition JSON v2 form into immutable canonical node definitions.
`/system_stats`, `/features`, a trusted pinned revision, node lifecycle flags, concrete SIGMAS
inputs, and sampler combo options become `sigmax.comfyui-adapter/1` evidence plus the existing
`HostCapabilities` contract. Missing, malformed, outside-window, or experimental required API
surfaces fail with stable reason/action data before registration or sampling.

The adjacent `adapters/registration.py` boundary uses `sigmax.node-registration/1`. It discovers
legacy/current classes from public V1 fields and V3 classes through `GET_SCHEMA()` plus
`GET_NODE_INFO_V1()`, validates documented Node Definition JSON v2 payloads, and creates fresh
mapping, display-name, `/object_info`, and v2 projections. Explicit `Sigmax.<Name>` IDs make
identity independent of installation-directory normalization. Copy-on-write registration is
idempotent for an identical definition and rejects conflicts without mutating the prior registry
or overwriting unrelated scheduler IDs.

The first concrete profile, `krea2.turbo.official`, is implemented in
`profiles/krea2_turbo.py`. It pins the official Krea source and framework corroboration,
declares fixed exponential `mu = 1.15`, eight default steps, terminal zero, deterministic
ComfyUI Euler capabilities, guidance conventions, and ceil-to-16 dimensions. Its builder
composes existing pure-core primitives and records dimension changes and non-reference step
counts explicitly. It does not inspect ComfyUI, choose a checkpoint variant, or execute a
sampler. Its schedule output now has authoritative formula and pinned-framework parity;
native host/sampler execution remains a later layer.

The first model golden lane stores complete 4/8/12/16-step vectors separately from product
code. Its generator uses precision-80 `Decimal` evaluation of an algebraically simplified
official formula, rounds once to binary64, and explicitly quantizes binary32 with `struct`.
The generator cannot import Sigmax or optional frameworks. An additional eight-step
official-direct binary64 calculation cross-checks the fixture, while production uses the
stable log-odds implementation.

The authoritative parity lane remains separate from the golden oracle. A standard-library
Krea adapter reproduces the pinned official formula, while an isolated optional environment
executes Diffusers 0.39.0 with exact NumPy and Torch pins. A strict canonical report records
complete terminal-inclusive vectors, source revisions, dependency versions, dtype/device,
errors, tolerances, and fingerprints. The default wheel imports neither framework; the
canonical gate validates the report contract, and hosted CI regenerates it byte for byte.
This establishes formula and framework schedule parity without claiming native ComfyUI host
or sampler parity.

The native ComfyUI schedule lane is a third, isolated boundary. It verifies the exact host
revision and source blobs, sets reviewed CPU mode before host imports, instantiates the real
`ModelSamplingFlux`, and dispatches the registered `simple` scheduler. Its committed report
stores complete 4/8/12/16-step vectors and an explicit quantization reason for non-divisible
table indexing. This lane executes GPL framework code as a reference but copies no framework
implementation into the MIT package. It still does not start a host, register a node, execute
Euler latent steps, or load model weights.

The second concrete profile, `krea2.raw.official`, is implemented in
`profiles/krea2_raw.py` on shared declarations extracted to `profiles/krea2_common.py`. It
records dynamic exponential-shift endpoints, upstream-unclamped extrapolation, dimensions,
terminal behavior, capabilities, and two separately evidenced guidance/step recipes.
`resolve_krea2_image_geometry()` retains requested and ceil-to-16 effective dimensions plus
the packed image grid, while `derive_krea2_raw_shift()` calculates the image sequence length
and official unclamped affine `mu`. `build_krea2_raw_schedule()` composes one of the two exact
named recipes through existing pure grid, shift, terminal, and validation stages.

The independent RAW golden lane stores complete 28/52-step float64 and float32 vectors for
five square and two orientation cases. Its precision-80 Decimal oracle independently repeats
geometry and affine-`mu` math and imports neither product nor optional framework code.

A separate RAW parity lane executes the pinned Krea geometry, affine-`mu`, and timestep
formulas through a standard-library adapter and executes Diffusers 0.39.0's actual
`calculate_shift` definition plus `FlowMatchEulerDiscreteScheduler` in an isolated
environment. Its canonical report contains all 14 terminal-inclusive cases, complete
vectors, effective geometry, image sequence length, calculated `mu`, source identities,
dependency versions, errors, tolerances, and fingerprints. Float64 report vectors are
normalized to a declared 15 significant digits so Windows and Linux `libm` differences below
the authoritative tolerance cannot change canonical evidence bytes. This proves
authoritative and framework schedule parity without claiming native ComfyUI, host,
sampler-step, checkpoint, or image execution.

The Krea-specific variant resolver is also implemented in the pure profile layer. It resolves
only explicit selection, trusted profile/framework metadata, or exact verified official
SHA-256 evidence. Local header and filename signals remain suggestions, while tensor keys and
the shared ComfyUI model class remain family-only. Conflicting resolving evidence is never
hidden by precedence, and strict official mode fails closed.

## Planned Layered Design

```text
ComfyUI nodes and diagnostics
            |
ComfyUI model/host adapters
            |
Profile resolver and registry
            |
Pure schedule engine
            |
Optional sampler backends
```

### Pure schedule engine

Responsibilities implemented or completed by the end of the pure-core stage:

- construct explicit base grids;
- compose the implemented named and domain-checked time shifts;
- apply at most one compatible optional spacing transform;
- compose the implemented terminal and slicing policies;
- validate monotonicity, finiteness, length, and domain;
- emit deterministic metadata and fingerprints.

This layer must not require ComfyUI or Diffusers for closed-form schedule formulas.

### Profile resolver and registry

Dedicated Turbo and RAW profiles carry model identity, variant, evidence level, sigma domain,
base-grid construction, shift parameterization, terminal policy, sampler compatibility, and
provenance. The implemented Krea-specific evidence resolver exposes status, confidence,
decisive source, normalized evidence, and warnings. The generic `ProfileSchemaV1` contract is
implemented together with the exact-key `ProfileRegistry` and explicit inheritance policy.
Generic capability resolution is implemented as a pure composition layer over those contracts.
Static model/host/sampler evidence collection and ComfyUI public-surface probing are implemented
in `adapters/comfyui.py`. Pure node registration and schema discovery are implemented in
`adapters/registration.py`. The repository H1/H2 harness owns pinned-host loading and model-free
workflow execution; remote hosts and model/sampler/image execution remain later work.

### ComfyUI adapters and nodes

The implemented adapter reuses Krea 2's existing trust boundary and never assumes that one
internal model class uniquely identifies a checkpoint variant. Its exact initial
static-contract window is ComfyUI 0.29.0; current numbered API `v0_0_2` is retained as
experimental rather than promoted to stable.

Pinned ComfyUI loading treats `NODE_CLASS_MAPPINGS` and `comfy_entrypoint` as mutually exclusive
branches. Sigmax therefore exposes one mixed mapping projection: legacy/current classes keep
their public V1 definitions, while V3 classes remain recognizable through `GET_SCHEMA()` and
`GET_NODE_INFO_V1()`. It does not expose an inert V3 entrypoint beside mappings.

The first catalog entry is the legacy/current `Sigmax.Krea2SigmaScheduler`. Its pure function
selects only the validated Turbo builder or one of the two named RAW recipes, enforces
strict-official mode, and applies the existing terminal-inclusive manual slice. The node boundary
then converts the tuple through host-provided Torch at execution time. Its second output is
deterministic `sigmax.krea2-sigma-node/1` JSON that separates complete-construction and selected
output fingerprints. It constructs sigmas only; it does not sample or patch the model.

The second catalog entry is `Sigmax.ModelAwareSigmaScheduler`. Its
`model_aware_sigma_scheduler.py` trust boundary uses static bounded reads of the public MODEL
wrapper, underlying class, model-config class, and primitive `unet_config.image_model` field. It
never invokes model methods or serializes foreign objects. Those signals can establish the Krea 2
family but cannot distinguish RAW from Turbo, so `Auto` exposes an ambiguous resolution and stops.
Explicit selection follows the existing evidence precedence, resolves an exact built-in
`ProfileKey`, and runs `resolve_profile_capabilities()` against the profile's reference sampler
and a pinned, visibly labeled ComfyUI `static_contract`. Only ALLOW/WARN reaches the M4-01 builder.
Deterministic `sigmax.model-aware-sigma-node/1` JSON carries the full decision and stable reason
codes.

The third catalog entry is `Sigmax.AdvancedFlowMatchScheduler`. Its
`advanced_flowmatch_scheduler.py` pure boundary accepts only the constructible `UNIT_FLOW`
sigma/time domain, builds a descending linear endpoint grid, and executes exactly one
`PRIMARY_TIME_SHIFT` using exponential-mu or direct-ratio parameterization. A single
mode-dependent value prevents mutually exclusive UI controls from becoming inert. Terminal
policy and terminal-inclusive slicing follow as separate declared stages. The builder creates
typed `ScheduleRequest` and `ScheduleResult` contracts, validates the complete and selected
vectors, and emits deterministic `sigmax.advanced-flowmatch-node/1` JSON. The node is
experimental external-sigma construction, not a sampler, model profile, domain converter, or
model patch.

The fourth through sixth catalog entries are `Sigmax.ProfileInspector`,
`Sigmax.ScheduleComparison`, and `Sigmax.ScheduleInspector`. `inspectors.py` reuses the exact
model-aware construction result and
a bounded static `model_sampling` class read for deterministic
`sigmax.profile-inspector/1` reports. Schedule inspection accepts only the three implemented
schedule-information schemas, applies strict JSON/collection limits, normalizes model-aware
nesting, and recomputes the connected SIGMAS output fingerprint before emitting
`sigmax.schedule-inspector/1`. Schedule comparison reuses that verified producer boundary and
emits `sigmax.schedule-comparison/1`, aligning only equal-domain/equal-length schedules by sigma
index and otherwise returning an explicit non-comparable reason. All three nodes are read-only
and exclude foreign objects, tensors, paths, prompts, and arbitrary host metadata from reports.

The seventh and eighth catalog entries are `Sigmax.RawWorkflowOutput` and
`Sigmax.TurboWorkflowOutput`. Each is a V1 empty-output execution root that independently rebuilds
the connected complete named schedule, accepts only exact float64 values or host float32
quantization, and publishes one bounded canonical artifact/receipt bundle through prompt
history. RAW selects only the immutable 28- or 52-step recipe and re-verifies requested/effective
geometry and dynamic shift evidence. Both nodes declare external-sigma ownership, exactly one
time shift, no double shift, and a truthful `not_executed` receipt.

### Sampler strategy

The first integration target is ComfyUI's native deterministic Euler sampler with an explicit
validated sigma sequence. Sigmax should reuse the native sampler whenever it is mathematically
equivalent.

A separate full sampler is planned only for behavior that cannot be encoded by sigmas alone,
such as scheduler-owned stochastic noise, stateful indices, or per-token timesteps. Alternative
integrators remain profile-scoped and experimental until independently validated.

## Data Flow

```text
MODEL + explicit profile/variant + dimensions + steps
                         |
                  profile resolution
                         |
                schedule specification
                         |
       base grid -> shift -> optional transform
                         |
             terminal/slicing/validation
                         |
              SIGMAS + SCHEDULE_INFO
                         |
              compatible ComfyUI sampler
```

Every automatic or overridden decision must be reflected in `SCHEDULE_INFO`.

## Invariants

- No silent `mu = 0` fallback.
- No silent RAW/Turbo variant choice.
- No dummy sigmas for unsupported models.
- No double shifting.
- No automatic global PyTorch or ComfyUI monkey patches.
- No user control without an executed effect.
- Official, framework-reference, community, and experimental evidence remain distinct.
- Numerical parity precedes subjective quality optimization.

## Related Documents

- [Profile specification](PROFILE_SPEC.md)
- [Compatibility](COMPATIBILITY.md)
- [Contributing](../CONTRIBUTING.md)
