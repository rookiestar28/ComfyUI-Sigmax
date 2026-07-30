# Model Profile Schema v1 Specification

## Status

The runtime contract identified by `sigmax.model-profile/1` is **frozen**. Additive or
incompatible contract changes require a new schema identifier; profile content continues to
use its own independently versioned `profile_version`.

Schema v1 freezes the validated externally supplied sigma path used by Krea 2 Turbo and RAW.
It deliberately rejects `MODEL_NATIVE` and `MODEL_PATCH` ownership rather than implying that
those future integration paths already have a stable profile contract.

## Purpose

A model profile is a versioned, evidence-bearing description of how Sigmax should construct
and validate a schedule for one model family or variant. Profiles prevent model-specific
values from becoming undocumented global defaults.

Profiles do not contain model weights and do not authorize model downloads.

## Frozen Runtime Schema

`ProfileSchemaV1` is an immutable, dependency-free contract. Its required areas are:

| Area | v1 declaration |
| --- | --- |
| Identity | Schema/profile versions, stable profile ID, display name, model family, and variant |
| Evidence | Evidence level and a `primary_source_id` that resolves to pinned source provenance |
| Schedule | Prediction type, sigma domain, external ownership, base grid, ordered transforms, terminal, and slicing |
| Recipes | Named guidance conventions, bounded step ranges, reference steps, and source identity |
| Detection | Ordered evidence methods, minimum confirmation confidence, strict default, and ambiguity policy |
| Compatibility | Exact `ModelCapabilities`, `ProfileCapabilities`, and reference `SamplerCapabilities` contracts |
| Artifacts | Supported construction-envelope, numerical-projection, and execution-receipt schema versions |
| Provenance | Separately versioned software-source, framework, and model-weight records and licenses |
| Extension data | Canonically ordered typed `ProfileField` parameters and public known limitations |

The provenance types are intentionally distinct:

- `SoftwareSourceProvenance` pins source code with a 40-hex revision and its own
  `LicenseDeclaration`;
- `FrameworkProvenance` pins each corroborating framework and its independently declared
  license;
- `ModelWeightProvenance` pins the public model resource version and SHA-256 while declaring
  its separate weight license.

A framework or source license never implies a model-weight license. Public locators must use
HTTPS, secret-like fields and private local paths are rejected, and all identifiers and lists
must be canonical and bounded.

`profile_schema_projection()` produces a deterministic typed projection. Floating values are
encoded by IEEE-754 binary64 bits, and `profile_schema_fingerprint()` hashes the canonical
projection as a `sha256:` identity. The projection is a fingerprint contract, not a general
JSON profile loader or an authorization to download weights.

## Registry and Inheritance

`ProfileRegistry` is an immutable copy-on-write snapshot of complete `ProfileSchemaV1`
objects. `ProfileKey` requires an exact namespaced `profile_id` and exact numeric
`profile_version`; lookup has no implicit latest version, range, prefix match, alias,
case-folding, or fallback.

The package's `builtin_profile_registry()` contains the exact RAW and Turbo built-ins.
External registration follows these rules:

- a new exact key is accepted only after the complete schema validates;
- an identical external key, fingerprint, and inheritance declaration is idempotent;
- different content under the same key rejects with `ConflictPolicy.REJECT`;
- `ConflictPolicy.REPLACE_EXTERNAL` is an explicit compare-and-swap operation that requires
  the exact existing fingerprint;
- external registration cannot replace a built-in under any policy, even when bytes are
  identical.

`ProfileInheritance` contains an already registered parent key and a canonical
`overridden_fields` tuple. The child is still a complete schema; no partial dictionary is
merged at lookup time. Profile ID and version establish the child identity. Every other
top-level semantic difference must be declared exactly—missing, extra, duplicate, unknown,
protected, or unchanged override names fail. An inherited external child must use
`modified` evidence, so an official parent cannot silently lend official evidence to changed
behavior. Requiring the parent to exist in the previous immutable snapshot makes cycles
unrepresentable.

Registry construction performs no file discovery, dynamic import, network access, model
download, host inspection, or schedule execution. External document parsing and
model/host/sampler evidence resolution remain separate future boundaries.

## Implemented Core Vocabulary

Exactly one ownership mode is required:

| Ownership | Meaning |
| --- | --- |
| `MODEL_NATIVE` | The model/host sampling object owns schedule construction |
| `EXTERNAL_SIGMAS` | Sigmax supplies the complete external sigma sequence |
| `MODEL_PATCH` | An explicit adapter replaces the model sampling object |

The accepted initial sigma domains are:

- `UNIT_FLOW`;
- `MODEL_NATIVE`, an opaque host-owned domain;
- `CONTINUOUS_EDM`;
- `DISCRETE_TRAINING_INDEX`.

External transformations declare input and output domains and follow this order:

```text
PRIMARY_TIME_SHIFT -> OPTIONAL_SPACING -> TERMINAL -> SLICE
```

The primary shift and optional spacing stages each occur at most once. Native and patched
ownership cannot also carry an external transform chain, because that would risk double
shifting.

The implemented immutable request/result layer also separates:

- `requested_inputs` from `effective_inputs`;
- source/profile `provenance` from user `overrides`;
- the requested base grid and transform chain from the structural sigma result;
- user-facing `warnings` from explicit value changes.

Any effective change to steps, width, or height requires a matching override record. Numerical
validity is checked separately from structural construction. Validated results can be encoded
as versioned canonical schedule artifacts with separate numerical and construction
fingerprints.

The Krea reciprocal-step base-grid identifier is now backed by a pure builder. Its output is
the non-terminal vector:

```text
1, (steps - 1) / steps, ..., 1 / steps
```

Appending terminal zero in the separate terminal stage reproduces the official unshifted
`steps + 1` vector. Terminal preservation, terminal-inclusive step ranges, and
ComfyUI-compatible denoise-tail slicing are now pure-core operations. A generic strictly
descending linear endpoint builder is also available for explicitly declared non-opaque
domains; it is not an automatic model default.

## Implemented Capability Preflight

The capability layer keeps host discovery, profile requirements, and sampler behavior
separate:

| Contract | Declares |
| --- | --- |
| `ModelCapabilities` | Model family/variant, accepted prediction and sigma domains, accepted schedule ownership, and optional execution features |
| `ProfileCapabilities` | Required prediction/domain/ownership, terminal policy, execution modes, noise owners, allowed sampler state, features, and reference samplers |
| `SamplerCapabilities` | Accepted semantics, terminal requirement, deterministic/stochastic behavior, noise ownership, state requirements, and feature support |

Every evaluation considers the complete canonical capability dimension set and produces a
`CompatibilityDecision`:

- `ALLOW` means all declared semantics are compatible and the sampler is a profile reference;
- `WARN` means the combination is compatible but the sampler is not a declared reference;
- `REJECT` contains stable reason codes and fails at `require_compatible()` before execution.

The decision vocabulary includes model family and variant, prediction type, sigma domain,
schedule ownership, terminal requirement, execution behavior, noise ownership, sampler state,
partial denoise, per-token timesteps, and reference-sampler status.

## Implemented Capability Resolution

`resolve_profile_capabilities()` composes one exact `RegisteredProfile` with:

- normalized `ModelIdentityEvidence` plus `ModelCapabilities`;
- versioned `HostCapabilities` whose entries retain `landed`, `experimental`, or `unsupported`
  lifecycle;
- one `SamplerCapabilities` declaration;
- one `ExecutionFeatureRequest`.

The immutable `ProfileCapabilityDecision` uses `sigmax.capability-resolution/1`, retains the exact
profile key/fingerprint and host identity/revision, records every required host capability, and
namespaces the existing core reason codes under `core.*`. Confirmed identity is required;
suggested, ambiguous, conflicting, and unknown identity never becomes confirmed because a
capability string matches. Missing, experimental, or unsupported required host capabilities
reject. The resolver does not inspect a live host, construct sigmas, or execute a sampler; host
evidence collection is supplied separately by `sigmax.comfyui-adapter/1`.

The adapter's immutable projection normalizes the public V1-compatible `/object_info` form and
Node Definition JSON v2, preserves deprecated/experimental lifecycle, and derives capability
evidence only from observed input types and sampler options. The exact initial static-contract
window is ComfyUI 0.29.0. Current `comfy_api.v0_0_2` reports `STABLE = False`, so a stable numbered
API requirement rejects actionably. This evidence boundary performs no network access,
registration, schedule construction, or sampling and does not constitute real-host E2E.

Node registration is a separate pure boundary under `sigmax.node-registration/1`. It preserves
legacy/current, V3, deprecation, and experimental metadata while serializing `/object_info` and
Node Definition JSON v2 projections. Registration requires explicit `Sigmax.<Name>` IDs and
rejects collisions; it does not select profiles, inspect model files, or imply host execution.

`Sigmax.Krea2SigmaScheduler` is the first consumer. Its `sigmax.krea2-sigma-node/1` information
records the selected built-in profile and exact recipe/evidence, requested/effective dimensions,
fixed or resolution-derived shift, manual output range, warnings, and complete/output numerical
fingerprints. Strict mode rejects modified Turbo steps and the RAW framework-reference recipe.
The node requires explicit variant selection and never promotes filename/model-class suggestions.

`Sigmax.ModelAwareSigmaScheduler` is the exact-profile consumer under
`sigmax.model-aware-sigma-node/1`. A required MODEL is inspected only for bounded public Krea 2
family signals. Family-only evidence leaves `Auto` ambiguous; it never selects a generic fallback
profile
or promotes a shared class/config to RAW or Turbo. Explicit RAW/Turbo selection retains
`explicit_selection` as the decisive source, resolves the exact built-in `ProfileKey`, and
evaluates the complete `sigmax.capability-resolution/1` decision before delegating schedule math
to the first node. Output records the unchanged registered-profile evidence separately from any
modified recipe evidence and labels pinned ComfyUI compatibility evidence as `static_contract`.

## Required Conceptual Fields

| Area | Required information |
| --- | --- |
| Identity | Stable profile ID, display name, profile version, model family, and variant |
| Evidence | Classification, primary source, and reference revision or version |
| Matching | Supported model metadata plus minimum identification confidence |
| Sampling | Prediction type and sigma/time domain |
| Base grid | Construction rule, endpoint policy, count, and terminal policy |
| Shift | Parameterization, fixed/dynamic mode, and required inputs |
| Sampler | Reference sampler and explicitly labeled alternatives |
| Guidance | Guidance convention and model-specific recommendation where authoritative |
| Validation | Length, finiteness, monotonicity, domain, and tolerance rules |
| Provenance | Engine/profile versions, overrides, and warnings |

## Evidence Classification

- `official`: reproduced from an authoritative model implementation or technical reference.
- `framework_reference`: reproduced from a pinned framework implementation.
- `community_recommended`: reproducible community configuration with cited evidence.
- `experimental`: exploratory behavior without parity or recommendation claims.
- `modified`: a resolved profile whose user overrides changed reference behavior.

Evidence level is part of the result, not just profile documentation.

## Sigma Domains

Every profile must declare the domain accepted by its grids and transforms. Initial domain
concepts include:

- unit flow time, normally decreasing from `1.0` to `0.0`;
- model-native values supplied by a host sampling object;
- continuous EDM-style sigma;
- discrete training-index values.

A transform may run only when its declared input domain matches the current schedule domain.

## Shift Parameterization

The profile must name the formula and its parameters. A generic field called only `shift` is
insufficient because models and frameworks use incompatible parameterizations.

Implemented pure-core forms include:

- exponential `mu` shifting;
- direct-ratio shifting;
- an explicit no-shift policy;

Krea 2 RAW resolution-derived numerical selection is implemented in its evidence-pinned
profile layer. It preserves requested and ceil-to-16 effective dimensions, calculates the
packed image sequence length, and evaluates the declared affine `mu` without clamping. The
exponential and direct controls are distinct: at exponent `1`, their schedules are equivalent
only when `direct_ratio = exp(mu)`.

Missing shift configuration must be an error in strict mode, not a hidden zero.

## First Concrete Profile: Krea 2 Turbo

The immutable `krea2.turbo.official` profile version `1` is bound to
`KREA2_TURBO_SCHEMA`, a validated `ProfileSchemaV1`. It declares:

- `official` evidence with a pinned Krea repository revision as the primary source;
- pinned Diffusers and ComfyUI framework references;
- Krea 2 `turbo` and flow-velocity model semantics;
- `EXTERNAL_SIGMAS` ownership in the `UNIT_FLOW` domain;
- the `krea.reciprocal_step` base grid;
- fixed exponential `mu = 1.15`;
- eight default steps and terminal zero;
- deterministic `comfy.euler` as its reference sampler;
- Krea guidance `0.0`, equivalent to standard ComfyUI CFG `1.0`;
- ceil-to-multiple-of-16 image dimensions.

`build_krea2_turbo_schedule()` composes the existing grid, shift, terminal, validation, and
request/result contracts. Dimension alignment is recorded as requested-to-effective
overrides. A non-eight-step request preserves the formula but changes evidence to `modified`
and emits a warning. The profile is a structural official-recipe implementation; fixed
golden vectors and authoritative Krea/Diffusers numerical parity are now enforced. Real
ComfyUI host and sampler execution remain required before a native-host product claim.

## Krea 2 RAW Profile and Geometry Derivation

The immutable `krea2.raw.official` profile version `1` declares:

- Krea 2 `raw` flow-velocity semantics and externally owned unit-flow sigmas;
- `krea.reciprocal_step`, exponential `mu`, terminal zero, and deterministic `comfy.euler`;
- resolution-linear shift endpoints from sequence length 256 / `mu=0.5` to 6400 /
  `mu=1.15`;
- upstream-unclamped extrapolation behavior;
- ceil-to-multiple-of-16 image dimensions;
- `krea2.raw.official-full-52` with Krea guidance 3.5 / ComfyUI CFG 4.5;
- `krea2.raw.diffusers-reference-28` with Krea guidance 4.5 / ComfyUI CFG 5.5;
- pinned official Krea and framework references.

`resolve_krea2_image_geometry()` accepts positive integer pixel dimensions and retains:

- requested width and height;
- each effective dimension rounded upward to 16;
- packed grid width and height;
- image-only packed sequence length.

`derive_krea2_raw_shift()` binds that geometry to the official profile and records the
calculated `mu` plus whether the value extrapolates beyond the 256-to-6400 sequence interval.
The upstream affine formula is deliberately unclamped. `build_krea2_raw_schedule()` does not
accept arbitrary steps: it requires either the named 28-step framework recipe or named
52-step official-full recipe. Complete independent float64/float32 goldens cover five square
plus landscape/portrait geometry cases. A separate pinned parity report executes the Krea
formula and Diffusers 0.39.0 dynamic-shift/scheduler paths for the same 14 complete cases.
Native ComfyUI, host, sampler-step, checkpoint, and image evidence remain separate.

## Matching and Variant Resolution

The Krea-specific pure resolver implements this evidence order:

1. explicit user-selected profile and variant;
2. trusted metadata retained by a profiled loader;
3. trusted Diffusers `Krea2Pipeline.is_distilled` metadata;
4. verified official checkpoint SHA-256;
5. local safetensors header suggestion;
6. filename suggestion;
7. family-only tensor keys or model class.

An internal model class alone must not be treated as sufficient when multiple variants share
that class. Local headers and filenames do not resolve official identity. Conflicting strong
evidence remains unresolved, and strict official mode raises rather than guessing. The
resolver retains normalized reason codes rather than private paths or arbitrary metadata.

The exact official single-file identities currently recognized are:

- RAW: `f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7`;
- Turbo: `78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3`.

Converted, quantized, fine-tuned, repackaged, or metadata-rewritten files require explicit or
other trusted evidence; similar filenames do not inherit these identities.

## Result Metadata

A generated schedule is expected to expose:

- resolved profile and version;
- evidence level and reference;
- detection method and confidence;
- dimensions and derived sequence length when applicable;
- steps, base grid, shift parameterization, and computed values;
- transform chain, terminal and slicing policies;
- user overrides and warnings;
- engine version and deterministic schedule fingerprint.

## Validation Requirements

An official or framework-reference profile is incomplete without:

- fixed golden vectors at representative step counts;
- formula-level unit tests;
- authoritative numerical parity tests;
- explicit float tolerances;
- at least one integration workflow;
- documented detection limits and known incompatibilities.

See [Architecture](ARCHITECTURE.md) for layer ownership and
[Compatibility](COMPATIBILITY.md) for current support status.
