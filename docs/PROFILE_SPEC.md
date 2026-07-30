# Provisional Model Profile Specification

## Status

This specification is **provisional** and **not frozen**. The foundational ownership, domain,
and transform-stage vocabulary is implemented, while profile fields, serialization format,
and schema versioning may change as the pure schedule engine and Krea 2 reference profiles are
validated.

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

Initial planned forms include:

- exponential `mu` shifting;
- direct-ratio shifting;
- an explicit no-shift policy;
- resolution-derived shifting when supported by an authoritative profile.

Missing shift configuration must be an error in strict mode, not a hidden zero.

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
