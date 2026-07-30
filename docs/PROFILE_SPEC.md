# Provisional Model Profile Specification

## Status

This specification is **provisional** and **not frozen**. The foundational ownership, domain,
transform-stage, artifact, and capability vocabulary is implemented, while resolved profile
serialization and schema versioning may change as Krea 2 reference profiles are validated.

## Purpose

A model profile is a versioned, evidence-bearing description of how Sigmax should construct
and validate a schedule for one model family or variant. Profiles prevent model-specific
values from becoming undocumented global defaults.

Profiles do not contain model weights and do not authorize model downloads.

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
partial denoise, per-token timesteps, and reference-sampler status. Profile resolution will
construct these declarations later; the pure core does not inspect a host model or guess a
variant.

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

The immutable `krea2.turbo.official` profile version `1` is implemented before the generic
profile schema is frozen. It declares:

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
The upstream affine formula is deliberately unclamped. No `build_krea2_raw_schedule()` API
exists at this stage; complete sigma vectors, goldens, parity, and variant resolution require
their own numerical and integration evidence.

## Matching and Variant Resolution

Resolution priority is planned as:

1. explicit user-selected profile and variant;
2. trusted metadata retained by a profiled loader;
3. verified checkpoint metadata or hash;
4. compatible internal model configuration;
5. filename suggestion;
6. visible generic fallback, when technically valid.

An internal model class alone must not be treated as sufficient when multiple variants share
that class. Strict official mode must fail rather than guess.

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
