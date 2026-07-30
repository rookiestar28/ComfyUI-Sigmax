# Model Profile Contribution Guide

## Status and authority

This guide defines the evidence required to propose, review, and accept a model profile in
ComfyUI-Sigmax. It applies to built-in profiles, external-profile examples, and changes to an
existing profile.

The frozen runtime fields and validation rules remain authoritative in the
[model profile schema v1 specification](PROFILE_SPEC.md). This guide does not create a profile
loader, authorize model downloads, or expand the current runtime ownership modes.

A profile contribution is an evidence package, not a collection of preferred settings.
Attractive images, filenames, copied configuration, or a schedule that merely looks plausible
cannot establish mathematical correctness or model compatibility.

## 1. Submission package

Submit all of the following as one reviewable change:

| Area | Required deliverable |
| --- | --- |
| Identity | Stable namespaced profile ID, numeric profile version, model family, variant, and display name |
| Provenance | Separately pinned software-source, framework, and model-weight resources with separate licenses |
| Mathematics | Base grid, domains, ordered transforms, formulas, parameters, terminal policy, and slicing |
| Recipes | Named step/guidance recipes with their own evidence and source identity |
| Capabilities | Complete model, profile, reference-sampler, and requested-execution semantics |
| Detection | Ordered resolving, suggestion-only, and family-only signals with fail-closed ambiguity |
| Numerical evidence | Independent complete goldens, formula tests, parity reports, tolerances, and fingerprints |
| Workflow evidence | A canonical validator-clean workflow fixture and applicable supported-host results |
| Safety | Dependency, attribution, secret/path, packaging, and untrusted-reference review |
| Documentation | Known limitations, support tier, migration impact, regeneration commands, and changelog entry |

Every claimed value must trace to a declared source or be labeled non-authoritative. Review may
accept a lower evidence level; it must never silently promote weak evidence.

## 2. Identity and versioning

Declare:

- `schema_id = sigmax.model-profile/1` and schema version `1`;
- one stable namespaced `profile_id`;
- one exact numeric `profile_version`;
- a public display name, model family, and unambiguous variant;
- the primary source ID used by the profile-level evidence claim;
- the expected profile fingerprint after canonical projection.

Use a new profile version whenever accepted profile content changes schedule construction,
recipes, detection, capabilities, provenance, artifact compatibility, or limitations. Do not
reuse a version for different bytes. Do not rely on “latest,” prefix, case-folding, aliases, or
implicit fallback.

An inherited profile is still a complete schema. It must name the exact parent and every changed
top-level field. It may not replace a built-in, hide undeclared differences, or inherit an
official label after changing behavior; an inherited external child uses `modified` evidence.

Breaking schema, node, artifact, or schedule semantics require a separately reviewed major
version and migration plan. A profile version bump cannot disguise a schema break.

## 3. Provenance and licensing

Keep the three resource layers independent:

| Layer | Required identity | Required license evidence |
| --- | --- | --- |
| Inference software source (`SoftwareSourceProvenance`) | Public HTTPS repository/resource, resource version when available, exact 40-hex revision, and behavior-bearing locators | Versioned SPDX-like or `LicenseRef-*` identifier, public license name, and HTTPS license URL |
| Corroborating framework (`FrameworkProvenance`) | Public HTTPS resource, exact framework version/revision, and behavior-bearing locators | Its own versioned license declaration |
| Model weights (`ModelWeightProvenance`) | Public HTTPS resource, immutable resource version/revision, and lowercase SHA-256 identity | The weight resource's own versioned license declaration and applicable use/redistribution limits |

The repository license covers the contributed code. It does not grant rights to upstream source
code, framework code, model weights, datasets, trademarks, or generated media. A software or
framework license never implies a model-weight license.

Also:

- state whether code or formulas were independently implemented, adapted, or copied;
- retain required attribution and notices;
- identify incompatible, non-redistributable, gated, or unclear terms before implementation;
- do not add weights, credentials, access tokens, cookies, private URLs, or gated-resource
  contents to commits, fixtures, logs, or workflow metadata;
- do not treat a public hash as permission to download or redistribute its resource.

Missing or incompatible license evidence is a rejection, not an invitation to guess.

## 4. Mathematical schedule contract

Document the complete construction path in execution order.

### Prediction, domain, and ownership

Declare:

- prediction type;
- input and output sigma/time domain;
- schedule ownership;
- which component owns any model-native shift;
- why the proposed path cannot apply the same shift twice.

Schema v1 accepts complete externally supplied sigmas. Do not claim `MODEL_NATIVE` or
`MODEL_PATCH` support under v1. Do not describe a sigma-only node as a sampler.

### Base grid

Provide:

- a stable base-grid identifier;
- a closed-form formula or fully specified algorithm;
- index direction, endpoints, length, and whether terminal is excluded;
- all parameters and their units;
- requested versus effective input behavior;
- valid ranges and error behavior.

### Ordered transforms

For each transform, declare:

- stable identifier and stage;
- exact formula;
- input and output domains;
- parameters, precision, and valid ranges;
- whether it is fixed, dynamic, resolution-derived, or optional;
- extrapolation or clamping policy;
- proof that the transform runs exactly once.

The supported external order is:

```text
PRIMARY_TIME_SHIFT -> OPTIONAL_SPACING -> TERMINAL -> SLICE
```

Missing parameters, incompatible domains, duplicate stages, or an unexplained second shift must
fail. A generic field named only `shift` is insufficient.

### Terminal and slicing

Declare:

- append/preserve policy and effective terminal value;
- transition-count convention;
- terminal-inclusive start/end behavior;
- denoise-tail semantics when supported;
- empty, zero-denoise, partial, and out-of-range behavior.

### Geometry and dynamic parameters

If dimensions affect the schedule, retain both requested and effective width/height and specify:

- alignment/rounding rules;
- latent or packed-grid calculation;
- sequence-length formula;
- dynamic parameter endpoints and interpolation formula;
- orientation behavior;
- extrapolation/clamping policy.

Cover square, landscape, portrait, boundary, invalid, and out-of-range cases where applicable.

## 5. Recipes, guidance, and evidence levels

Each named recipe declares:

- stable recipe ID and source ID;
- evidence level;
- exact/default/reference step policy;
- whether modified steps are allowed;
- model guidance convention and value;
- host guidance convention and value;
- material exclusions and limitations.

Use only these evidence values:

| Evidence | Required meaning |
| --- | --- |
| `official` | Reproduced from a pinned authoritative model implementation or technical reference |
| `framework_reference` | Reproduced from a pinned framework implementation |
| `community_recommended` | Reproducible community configuration with a cited public source |
| `experimental` | Exploratory behavior with no parity, support, or recommendation claim |
| `modified` | Reference behavior changed by an override, inheritance, step change, formula change, or other material deviation |

Evidence is scoped. An official source may coexist with a framework-reference recipe; one does
not upgrade the other. If a user or contributor changes reference behavior, the resulting
evidence is `modified`, even when the parent profile is official.

## 6. Capability and sampler declarations

Provide complete, mutually compatible declarations for:

- model family and variant;
- prediction type and sigma domain;
- allowed schedule ownership;
- terminal requirement;
- deterministic and stochastic execution behavior;
- noise ownership;
- RNG ownership by schedule, model, and sampler;
- sampler state requirements;
- partial denoise;
- per-token timesteps;
- reference sampler identifiers;
- required host capabilities and lifecycle (`landed`, `experimental`, or unsupported).

Name alternatives separately from reference samplers. A compatible alternative may produce a
warning; it does not become a reference merely because execution succeeds.

State requested and effective transitions and model evaluations. A schedule artifact does not
prove that a sampler or model executed. A successful execution receipt requires evidence from
the actual step path.

## 7. Detection and ambiguity

Define disjoint, ordered evidence classes:

1. resolving evidence that may confirm the exact family and variant;
2. suggestion-only evidence that may inform the user but cannot resolve;
3. family-only evidence that cannot choose between variants.

For every signal, document source, trust boundary, normalization, confidence, collision risk,
and known conversions that invalidate it.

Requirements:

- exact official weight identities may resolve only the exact reviewed resource;
- converted, quantized, fine-tuned, repackaged, or metadata-rewritten files require their own
  evidence;
- filenames and local headers are suggestions unless a separately reviewed trusted format says
  otherwise;
- shared model classes, configs, and tensor keys are family-only when variants share them;
- conflicting strong evidence remains unresolved;
- unknown or ambiguous evidence requires explicit selection in strict mode;
- no private path or arbitrary model metadata may enter normalized evidence or serialized output.

Detection must fail closed. Capability compatibility cannot turn weak identity evidence into a
confirmed variant.

## 8. Numerical and conformance evidence

### Independent goldens

Provide complete terminal-inclusive vectors for representative recipes and inputs:

- independently implemented oracle, with its method described;
- float64 values and explicit IEEE-754 float32 projections when the host emits float32;
- expected length, finiteness, monotonicity, domain, endpoints, terminal, and slicing;
- square/non-square or other material input dimensions;
- deterministic numerical and construction fingerprints;
- a reviewed regeneration command that fails without partial output.

Do not derive expected values by calling the production builder under test.

### Formula tests

Pin:

- exact base-grid positions;
- every transform formula and order;
- dynamic parameter derivation;
- terminal and slicing boundaries;
- invalid input and unsupported-domain failures;
- override-to-`modified` behavior;
- cross-platform exact-bit behavior when fingerprints depend on exact binary64 output.

### Authoritative and framework parity

Record:

- exact source revision and behavior-bearing locators;
- framework and dependency versions;
- CPU/GPU device and dtype;
- complete case matrix and vectors;
- maximum and mean errors;
- explicit absolute/relative tolerances and their rationale;
- source/report fingerprints;
- dirty-source, wrong-version, and missing-dependency rejection;
- regeneration isolation and license review.

Image quality is supplemental and cannot replace numerical parity.

### Evidence tier

Claim only the highest tier actually executed:

1. pure/formula;
2. independent golden;
3. authoritative/framework parity;
4. native-host schedule parity;
5. real-host node/workflow execution;
6. sampler/latent execution;
7. approved model/GPU conformance.

Passing a lower tier does not imply a higher tier.

## 9. Workflow and supported-host evidence

Every contribution includes at least one canonical workflow fixture that:

- uses stable namespaced node IDs;
- identifies exact package, node, host/API, profile, and recipe versions;
- contains no secret, private path, model weight, or user data;
- retains requested/effective inputs and applicable geometry;
- embeds verified construction/numerical identities and receipt references;
- passes the pinned-static and literal-loopback live validator;
- documents expected warnings or strict failures.

If the contribution changes node, adapter, workflow, or host behavior, real supported-host H1/H2
is blocking. Assert completed history and final values/fingerprints, not queue acceptance.
Include:

- import, schema, registration, and reload/idempotency evidence;
- metadata save/reload;
- ownership and no-double-shift proof;
- changed-control execution or an explicit error;
- runtime and prequeue rejection boundaries when applicable;
- bounded loopback, shutdown, port release, redaction, and owned-state cleanup.

Model weights are not required for the default model-free H1/H2 lane. Any GPU/weight run requires
explicit authorization, pinned hashes, isolated caches, and a separate heavy-conformance record.

## 10. Security, dependency, and packaging review

The contribution must:

- keep closed-form schedule math independent of ComfyUI and Diffusers;
- isolate and version-bound optional references;
- add no mandatory runtime dependency without explicit architectural approval;
- avoid network, model download, file discovery, dynamic import, or host inspection during
  registry construction;
- treat external repositories and artifacts as untrusted/read-only;
- exclude weights, caches, build output, internal evidence, private paths, and credentials from
  packages;
- pass secret scanning, attribution/license review, clean-environment import, and wheel inventory;
- identify Unicode, path, platform, precision, and optional-dependency failure modes.

Never execute an external setup script, hook, container, binary, or package lifecycle command
merely to inspect a candidate profile.

## 11. Documentation and limitations

Update:

- profile schema documentation;
- compatibility/support tier;
- architecture when ownership or data flow changes;
- public usage examples;
- changelog;
- attribution/notice material when required.

List all known limitations as public, bounded statements. Include unsupported host versions,
unverified devices/dtypes, ambiguous detection, unsupported recipes, extrapolation behavior,
weight variants, quantization, partial denoise, stochastic/state behavior, and missing
sampler/model/image evidence as applicable.

Do not use “supported,” “official,” “native,” or “compatible” without naming the evidence scope.

## 12. Pull request checklist

Copy this checklist into the pull request and link each item to a file, test, or retained
evidence artifact.

### Identity and provenance

- [ ] Stable profile ID/version, family, variant, display name, primary source, and expected
      profile fingerprint are declared.
- [ ] Software source is pinned to a public revision and separately licensed.
- [ ] Every corroborating framework is pinned and separately licensed.
- [ ] Model weights use an immutable version/revision and SHA-256 with their own license.
- [ ] Attribution, adaptation/copy status, and redistribution restrictions are recorded.

### Schedule and capabilities

- [ ] Prediction/domain/ownership, base grid, transforms/order, terminal, slicing, and dynamic
      geometry are fully specified.
- [ ] Recipes declare steps, guidance conversions, evidence, and modification policy.
- [ ] Model/profile/sampler/execution capabilities and host lifecycle requirements are complete.
- [ ] Requested/effective transition and model-evaluation semantics are explicit.
- [ ] No model-native/external double shift or inert control is possible.

### Detection and evidence

- [ ] Resolving, suggestion-only, and family-only signals are disjoint and ordered.
- [ ] Ambiguous/conflicting/unknown evidence fails closed.
- [ ] Evidence level matches the weakest claim actually made; material overrides are `modified`.
- [ ] Known conversions, quantizations, fine-tunes, and detection limitations are documented.

### Tests and workflows

- [ ] Independent complete float64/float32 goldens and formula/negative tests are present.
- [ ] Pinned parity records include complete vectors, device/dtype, tolerances, max/mean errors,
      fingerprints, and regeneration instructions.
- [ ] A canonical workflow passes static and live validation with verified metadata.
- [ ] Applicable supported-host execution and fail-closed rejection paths pass.
- [ ] Full applicable test, security, type, coverage, package, and platform gates pass with no
      unexplained skip or flake.

### Release and review

- [ ] Public docs, compatibility tier, limitations, changelog, licenses, and notices are updated.
- [ ] No secrets, private paths, user data, weights, caches, or internal evidence are publishable.
- [ ] Migration/versioning and rollback are documented.
- [ ] Review evidence maps every acceptance requirement to an objective pass/fail result.

## 13. Automatic rejection conditions

Reject or return a contribution for revision when any of these is true:

- provenance or any applicable license layer is missing, unclear, or incompatible;
- an official/reference claim lacks an exact pinned authoritative source;
- expected vectors come only from the implementation under test;
- only selected points or screenshots are supplied instead of complete numerical evidence;
- domains, ownership, transform order, terminal, slicing, or dynamic geometry are implicit;
- arbitrary model-specific values are hidden in a generic scheduler;
- ambiguous identity silently falls back or filename/header evidence is promoted;
- an override retains official evidence instead of becoming `modified`;
- capabilities claim sampler/model execution that did not occur;
- workflow validation or applicable supported-host evidence is missing;
- a second shift, global framework patch, inert control, secret, private path, weight, or
  unapproved dependency is introduced;
- required tests are skipped, flaky, partial, or missing reproducible command/evidence details.

Review can request stronger evidence or narrow the declared support tier. It cannot waive
truthful provenance, licensing, fail-closed identity, or numerical reproducibility.

## Related specifications

- [Model profile schema v1](PROFILE_SPEC.md)
- [Compatibility and validation tiers](COMPATIBILITY.md)
- [Architecture](ARCHITECTURE.md)
- [Schedule artifact specification](SCHEDULE_ARTIFACT_SPEC.md)
- [Execution receipt specification](EXECUTION_RECEIPT_SPEC.md)
- [Workflow metadata specification](WORKFLOW_METADATA_SPEC.md)
- [Workflow validation specification](WORKFLOW_VALIDATION_SPEC.md)
- [Contributing](../CONTRIBUTING.md)
