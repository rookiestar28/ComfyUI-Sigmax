# Changelog

All notable user-visible changes will be documented in this file.

The project follows semantic versioning for package releases. Model profile versions are
tracked independently from the frozen profile-schema version.

## [Unreleased]

### Added

- A source-derived, fingerprinted `sigmax.public-contract-manifest/1` freezing built-in node IDs,
  public profile/capability/artifact/receipt schemas, stable capability reason codes, and the
  versioned compatibility and migration policy.
- Side-effect-free `comfyui_sigmax` package shell with intentionally empty node mappings.
- Python packaging metadata and typed-package marker.
- Repository-local quality, test, coverage, and wheel validation.
- Cross-platform Windows and Linux/WSL full-gate wrappers.
- Versioned environment diagnostics for local venv, cache integrity/locks, tool conflicts,
  Unicode and temp-path operations, and selected optional dependencies.
- Minimal-permission continuous-integration workflow contract.
- Public architecture, profile-schema, compatibility, and contribution documentation.
- An objective model profile contribution guide covering separate source/framework/weight
  provenance and licenses, mathematical/domain/capability declarations, fail-closed detection,
  evidence levels, complete golden/parity proof, validator-clean workflows, supported-host
  applicability, security, limitations, review checklists, and automatic rejection conditions.
- Framework-independent schedule ownership, sigma-domain, transform-stage, and double-shift
  preflight contracts.
- Immutable request/result contracts for requested/effective inputs, base grids, terminal and
  slicing policy, provenance, warnings, overrides, and structural sigma values.
- Exact dependency-free Krea reciprocal-step and generic descending linear base-grid builders.
- Dependency-free exponential-`mu`, direct-ratio, and explicit no-shift unit-flow transforms
  with stable endpoint and extreme finite-control behavior.
- Explicit terminal append/preserve, terminal-inclusive step-range slicing, and
  ComfyUI-compatible partial-denoise tail policies with strict boundary checks.
- Versioned canonical schedule artifact specification with separate numerical and
  construction identities, typed IEEE-754 tokens, and cross-process golden fixtures.
- Dependency-free complete-schedule validation, typed float encoding, bounded canonical
  projections, and numerical/construction SHA-256 fingerprint functions.
- Immutable schedule artifacts with versioned canonical UTF-8 transport, complete effective
  construction metadata, strict untrusted-input parsing, and dual-fingerprint verification.
- Immutable execution receipts and portable artifact/receipt bundles with explicit status,
  host/model/sampler identities, RNG ownership, transition/model-evaluation counts, strict
  canonical transport, and construction/numerical cross-link verification.
- Canonical immutable artifact/receipt schedule and comparison reports with exact typed values,
  signed deltas, execution evidence, and an optional lazy headless PNG/SVG plotting extra.
- A packaged dependency-free numerical benchmark matrix summarizing capability-filtered
  Turbo/RAW parity, H2 artifact/receipt workflows, H3 deterministic Euler, source identities,
  runtime/count/repeat evidence, and explicit non-evaluated BF16/quantized lanes.
- A strict dependency compatibility matrix with repeated Windows/WSL invariant evidence,
  exact pinned Diffusers/ComfyUI/API baselines, latest-host separation, immutable-container
  requirements, and explicit non-PASS external lanes.
- Immutable `sigmax.host-mutation-snapshot/1` contracts and a strict ten-row synthetic
  co-installation matrix covering reload idempotence, registry collisions,
  PyTorch/model-patch changes, and model-native/external double-shift detection without
  executing third-party code.
- Integer-unit `sigmax.performance-budget-matrix/1` evidence for schedule latency/allocation,
  exact CPU tensor-boundary operations, isolated startup, and repeated pinned ComfyUI readiness,
  with unsupported GPU/latest/container lanes explicitly unevaluated.
- Versioned workflow metadata with package/node/host requirements, profile and compatibility
  evidence, artifact/receipt references, and non-destructive ComfyUI 0.4/1 graph attachment.
- Immutable model, profile, sampler, and execution-feature capability declarations with
  canonical allow/warn/reject decisions, stable reason codes, and a fail-before-execution
  compatibility gate.
- An ordered pure-layer independence gate that requires absent optional frameworks, blocks
  ComfyUI/Diffusers imports under Python isolated mode, imports every core and profile module,
  and checks static import roots.
- Dependency-free deterministic property/metamorphic coverage for grids, shifts, terminal
  structure, and capability-decision stability.
- Wheel inventory enforcement for the pure-core package.
- An immutable, evidence-pinned Krea 2 Turbo official-recipe profile with fixed exponential
  `mu = 1.15`, guidance and dimension conventions, deterministic Euler capability
  declarations, and a dependency-free structural schedule builder.
- Complete 4/8/12/16-step Krea 2 Turbo float64 and float32 golden vectors, generated by an
  independent precision-80 Decimal oracle and separately cross-checked at eight steps; golden
  status is reported separately from framework and native-ComfyUI parity.
- Authoritative Krea 2 Turbo framework parity harness with pinned Krea source, exact
  Diffusers 0.39.0 / NumPy 2.3.4 / Torch 2.9.0 execution, complete canonical report vectors,
  enforced error bounds, schedule fingerprints, fail-closed dependency checks, and isolated
  hosted-CI regeneration.
- Native-ComfyUI parity against pinned `ModelSamplingFlux` and registered `simple` behavior,
  with exact source/dependency checks, complete 4/8/12/16-step vectors, explicit integer-table
  quantization policy, fail-closed canonical evidence, and a separate hosted regeneration
  lane.
- Pinned native-ComfyUI deterministic Euler parity with complete eight-step latent traces,
  independent flow-equation recomputation, exact transition/model-evaluation counts,
  deterministic reruns, fail-closed source/dependency checks, a release-excluded real-host H3
  probe, single-shift/external-sigma verification, and artifact-linked `succeeded` execution
  receipts.
- Immutable `krea2.raw.official` structural profile with resolution-linear exponential-shift
  endpoints, upstream-unclamped extrapolation, ceil-to-16 dimensions, deterministic Euler
  capabilities, and separate `krea2.raw.official-full-52` and
  `krea2.raw.diffusers-reference-28` guidance recipes.
- Immutable Krea 2 image-geometry and RAW shift-derivation results that retain requested and
  ceil-to-16 effective dimensions, calculate packed image sequence length, and reproduce the
  official unclamped resolution-linear `mu`.
- Exact-recipe Krea 2 RAW schedule construction plus 14 complete independent 28/52-step
  float64 and float32 golden cases generated by a precision-80 standard-library oracle.
- Authoritative Krea 2 RAW parity across 14 complete recipe/geometry cases, with an
  independent pinned Krea adapter, executable Diffusers 0.39.0 dynamic-shift and scheduler
  paths, canonical vectors and error evidence, fail-closed regeneration, and an isolated
  hosted-CI lane.
- Executable square, non-aligned landscape, and non-aligned portrait Krea 2 RAW workflows with
  `Sigmax.RawWorkflowOutput`, canonical model-free artifact/receipt publication, metadata
  reload, static/live schema validation, and isolated H1/H2 execution on pinned ComfyUI
  `0.29.0` revision `e651b7bef55a5376343dcb1c0edb79f0142c985e`.
- Immutable Krea 2 variant-evidence and resolution contracts with explicit/trusted/verified
  resolution, suggestion-only local headers and filenames, family-only tensor/model-class
  signals, exact official file hashes, conflict detection, and strict fail-closed behavior.
- Frozen dependency-free `ProfileSchemaV1` (`sigmax.model-profile/1`) with strict cross-field
  schedule, recipe, detection, capability, and artifact-version validation; separately
  versioned and licensed `SoftwareSourceProvenance`, `FrameworkProvenance`, and
  `ModelWeightProvenance`; and deterministic typed projection plus
  `profile_schema_fingerprint`.
- Immutable exact-key `ProfileRegistry` snapshots with namespaced numeric-version keys,
  deterministic built-ins, idempotent identical registration, fail-closed conflicts,
  fingerprint-guarded `REPLACE_EXTERNAL`, unconditional built-in protection, and explicit
  complete-schema inheritance whose declared differences must match and use `modified`
  evidence.
- Versioned `sigmax.capability-resolution/1` composition of exact registered profiles, confirmed
  model identity/capabilities, explicit host lifecycle evidence, sampler capabilities, and
  execution-feature requests; required missing, experimental, or unsupported host capabilities
  and all weak/unresolved model identity states fail closed before sampling.
- Dependency-free `sigmax.comfyui-adapter/1` normalization of public numbered-API manifests,
  `/system_stats`, `/features`, V1-compatible `/object_info`, and Node Definition JSON v2;
  lifecycle-aware external-SIGMAS, Euler, partial-denoise, and model evidence; an exact ComfyUI
  0.29.0 static-contract window; and actionable rejection of missing, malformed, outside-window,
  or experimental required APIs.
- Immutable `sigmax.node-registration/1` node catalogs with explicit `Sigmax.<Name>` IDs,
  copy-on-write idempotent registration, fail-closed collision handling, installation-directory
  independence, mixed legacy/current and V3 mapping discovery, lifecycle metadata preservation,
  and fresh `/object_info` plus Node Definition JSON v2 wire projections.
- The first user-facing `Sigmax.Krea2SigmaScheduler` legacy/current node with explicit RAW/Turbo
  selection, required geometry, strict-official enforcement, named RAW recipes,
  terminal-inclusive manual slicing, execution-time-only host Torch conversion, and deterministic
  `sigmax.krea2-sigma-node/1` schedule information with complete/output fingerprints.
- `Sigmax.ModelAwareSigmaScheduler` with a required MODEL, bounded public Krea 2 family probing,
  visible Auto ambiguity, exact built-in profile lookup, existing evidence precedence, complete
  capability gating, stable reason codes, a pinned `static_contract` host label, M4-01 numerical
  delegation, and deterministic `sigmax.model-aware-sigma-node/1` information.
- `Sigmax.AdvancedFlowMatchScheduler` with explicit `UNIT_FLOW` linear endpoints, mutually
  exclusive exponential-mu/direct-ratio parameterization through one active shift value,
  terminal and slicing stages, typed request/result validation, execution-time-only Torch
  conversion, and deterministic `sigmax.advanced-flowmatch-node/1` information.
- Read-only `Sigmax.ProfileInspector` and `Sigmax.ScheduleInspector` nodes with bounded static
  native sampling-class evidence, exact capability/profile reports, controlled versioned
  schedule-info parsing, connected-SIGMAS fingerprint verification, and deterministic
  `sigmax.profile-inspector/1` / `sigmax.schedule-inspector/1` output.

### Security

- Unfinished external scheduler code and unrelated global patches are excluded from the
  runtime import path.
- Runtime dependency and built-wheel inventories are enforced by tests.
- Artifact transport rejects oversized inputs, BOMs, invalid UTF-8, duplicate JSON keys,
  floating literals, non-standard constants, unknown fields, non-canonical bytes, stale
  fingerprints, secret-like metadata fields, and private local paths.

### Known limitations

- No cross-family generic profile fallback or sampler is implemented.
- Real ComfyUI model-free Turbo/RAW workflows are validated only on the pinned supported host;
  controlled deterministic native-Euler sampler steps are validated there, while real Krea
  model-weight, GPU, image, stochastic, resumable, partial-denoise-execution, and
  advanced-workflow validation remain pending.

[Unreleased]: https://github.com/rookiestar28/ComfyUI-Sigmax/commits/main
