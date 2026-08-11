# Changelog

All notable user-visible changes will be documented in this file.

The project follows semantic versioning for package releases. Model profile versions are
tracked independently from the frozen profile-schema version.

## [Unreleased]

### Added

- A MiniMax H3 Base qualification slice with `Sigmax.MiniMaxH3SigmaScheduler`, explicit FL2VA and
  Ref2VA selection, endpoint-inclusive Diffusers video sigmas, model-free host workflow/schema
  preflight, and pinned Diffusers/native ComfyUI parity. The Sigmax node owns the external video
  schedule while the generated workflow configures ComfyUI's upstream `MiniMaxH3SigmaShift` with
  matching video/audio values `12.0`/`3.0`; audio remapping and derivative correction remain
  model-owned.
- Source-qualified Anima Base, Aesthetic, and Turbo schedules with
  `Sigmax.AnimaSigmaScheduler`. Variants are explicit, use the fixed framework-reference shift
  `3.0`, and remain schedule-only without weight loading or image-quality claims.
- Source-qualified Wan 2.1 and 2.2 schedule matrices with `Sigmax.WanSigmaScheduler`, covering
  explicit T2V/I2V/TI2V generation, source, and resolution lanes. Wan 2.2 A14B boundaries remain
  caller-owned metadata; Sigmax does not route experts or implement a video sampler.
- Extended `Sigmax.WanSigmaScheduler` with official-native FLF2V, VACE, S2V, Wan 2.2 Animate,
  and Wan Animate 2 Base/Distilled task profiles. Each profile preserves explicit source, task,
  resolution, and documented shift/step recipe ownership; A14B boundary metadata remains
  caller-owned, with no expert-routing, model-execution, or video-quality claim.
- Completed model-free Wan compatibility and release qualification on the pinned ComfyUI 0.29.0
  known-good lane and the current ComfyUI 0.31.0 lane, covering first/repeat registration,
  schema, schedule/metadata, package, and co-installation checks. These lanes do not establish
  real-model execution or video-quality parity.
- Source-qualified LTXV 0.9.8, LTX-2 19B, and LTX-2.3 22B schedules with
  `Sigmax.LTXSigmaScheduler`, separating adaptive Dev token shifts from immutable distilled Stage
  1/2 vectors.
- Original Qwen Image schedule modes with `Sigmax.QwenImageSigmaScheduler`: pinned ComfyUI fixed
  shift `1.15` and Diffusers dynamic shift with required `image_seq_len`. Later Qwen variants are
  outside this support claim.
- HunyuanImage 2.1 Base and Distilled schedules with `Sigmax.HunyuanImage21SigmaScheduler`,
  explicit fixed shifts `5.0`/`4.0`, separate recipes, and fail-closed variant selection. The
  Distilled lane remains publisher-schedule-only without native-host qualification.
- A Lumina-Image 2.0 schedule slice with `Sigmax.Lumina2SigmaScheduler`, preserving the
  source-qualified fixed unit-flow ratio `6.0`, original 50-step recipe, independent goldens, and
  model-free host validation.
- An original AuraFlow v0.2 schedule slice with `Sigmax.AuraFlowSigmaScheduler`. It exposes the
  source-qualified fixed unit-flow ratio `1.73`, an explicit 50-step recipe, independent golden
  and parity vectors, canonical workflow metadata, and pinned-host model-free validation. Other
  AuraFlow variants, model execution, and image-quality claims remain out of scope.

- An original Stable Diffusion 3 schedule slice with `Sigmax.SD3SigmaScheduler`. It exposes
  explicit, non-composable publisher-reference `1.0` and ComfyUI/Diffusers fixed `3.0` modes,
  source-qualified golden vectors, workflow fixtures, and pinned-host model-free validation;
  SD3.5, Turbo, model execution, and image-quality claims remain out of scope.
- An experimental `Sigmax.Krea2ConditioningRebalance` node for explicit RAW/Turbo
  `(batch, sequence, 30720)` conditioning tensors. It applies versioned community-derived tap
  reweighting with fixed per-sample RMS restoration, preserves conditioning metadata, and emits
  bounded provenance/fingerprint information. This does not claim prompt-adherence or image-
  quality improvement.
- Verified Z-Image Base and Turbo profiles plus `Sigmax.ZImageSigmaScheduler`. The profiles use
  explicit variants, fixed direct-ratio shifts, complete independent float64/float32 goldens,
  four-source evidence, canonical workflow fixtures, and pinned-host validation without adding a
  mandatory runtime dependency.
- A verified FLUX.1-schnell profile plus `Sigmax.Flux1SchnellSigmaScheduler` for the publisher's
  explicit one-to-four-step unshifted schedule, with independent goldens, four-source evidence,
  canonical workflow execution, and Windows/WSL gate evidence.
- An explicit `krea2.raw-turbo-lora.experimental` profile with separate RAW-derived and fixed
  Turbo `mu` modes, arbitrary bounded steps, independent goldens, and pinned-host workflows. It
  remains an experimental schedule for a compatible RAW-to-Turbo model-difference LoRA and does
  not load the LoRA or claim an official step count.
- A scoped ComfyUI frontend extension for `Sigmax.Krea2SigmaScheduler` that forces
  `strict_official=false` and disables that widget while either experimental LoRA variant is
  selected. Its deterministic policy and JavaScript syntax run in the default full gate with
  Node.js 18 or newer.
- A dependency-free `sigmax.image-benchmark-protocol/1` with four fixed Turbo/RAW cases,
  construction/numerical/receipt and prompt/settings identities, explicit unapproved execution,
  typed component/output hash evidence, and deterministic balanced blind ballot/reveal contracts.
  Image metrics and reviewer preferences are permanently `supplemental_only` and cannot replace
  mathematical parity.
- A dependency-free `Checkpoint Evidence Inspector` (`Sigmax.CheckpointEvidenceInspector`) for
  ComfyUI-allowlisted local safetensors.
  It reads only the bounded header, validates tensor structure without loading payloads, accessing
  an accelerator, or using the network, and emits canonical confidence/reason-code evidence that
  never confirms a variant from weak metadata, filename, or family-only structure.
- Fingerprint-bound `Schedule Slice`, `Schedule Concatenate`, and `Schedule Resample` nodes with
  terminal-inclusive bounds, exact shared-boundary enforcement, explicit normalized-index
  interpolation, strict domain/monotonicity validation, and `modified` output provenance.
- Dependency-free `sigmax.generic-flowmatch-profile/1` declarations for exact, explicitly
  selected `flowmatch.generic.fixed` framework-reference and
  `flowmatch.generic.dynamic` experimental schedule structures. They remain outside the concrete
  model `ProfileRegistry`, carry no model compatibility or official-recipe claim, and add no
  runtime dependency.

### Changed

- Finalized the unreleased `Sigmax.MiniMaxH3SigmaScheduler` input contract by removing the
  user-visible `already_shifted` option. H3 schedules always enter through the unshifted input
  path, while the pure core retains its fail-closed double-shift guard. Workflows saved from an
  earlier development build may need the H3 scheduler node refreshed or recreated; numerical
  schedule behavior is unchanged.
- A bounded local Krea 2 H4 lane completed model execution and artifact-provenance verification.
  Blind scoring and threshold review were explicitly waived, so this creates no prompt-adherence,
  image-quality, or profile-promotion claim.
- Removed pytest contracts for pure prose documents. README, changelog, compatibility, contributor,
  and test-governance wording can evolve without line-count, heading, link-layout, or narrative
  string assertions; executable APIs, schemas, CI workflows, packaging, security, and release
  artifacts remain tested at their source boundaries.
- MiniMax H3 Base's unreleased public scheduler and workflow contract now names the control
  `steps`: `steps=N` constructs `N+1` endpoint-inclusive Diffusers video sigmas and reports
  `N` transitions/model evaluations. Existing development workflows using `grid_points` or the
  removed `already_shifted` widget must refresh/recreate that node; the source-facing parity
  builder retains `grid_points` only at its explicit framework boundary. FL2VA/Ref2VA selection,
  video shift `12.0`, model-owned audio shift `3.0`, native `simple` separation, and the
  no-double-shift guard are unchanged.
- Reduced the public documentation surface to the concise product README, compatibility guide,
  contribution guide, changelog, and test-governance documents. Detailed architecture, schema,
  evidence, and release contracts continue to be enforced by source, generated manifests, tests,
  and local governance instead of duplicated public specification files.
- The canonical full gate now runs the frontend policy after core-independence checks and before
  parity and pytest stages. Hosted CI provisions Node.js 20 for this Node.js 18+ gate.

### Fixed

- Stabilized Windows and Linux/WSL foundation gates around Comfy Registry artifacts, subprocess
  environment inheritance, local tool isolation, and current post-release contract inventories.
- Bounded the mounted-workspace WSL startup probe so host filesystem latency is measured without
  turning transient machine load into a schedule or profile behavior change.

## [1.0.0] - 2026-08-01

### Added

- A source-derived, fingerprinted `sigmax.public-contract-manifest/1` freezing built-in node IDs,
  public profile/capability/artifact/receipt schemas, stable capability reason codes, and the
  versioned compatibility and migration policy.
- A canonical non-publishing `sigmax.release-audit/1` covering tracked-file hygiene, separated
  dependency/provenance/license/Registry reviews, and bounded wheel/sdist archive inspection.
- A deterministic `.comfyignore`-filtered Comfy Registry ZIP validator with an embedded
  `sigmax.registry-release-manifest/1`, safe member inspection, cross-platform reproducibility,
  renamed-directory clean import, and optional unauthenticated read-only Registry observation.
- A version 1.0 user guide covering installation, verified node selection, Turbo/RAW examples,
  workflow metadata, compatibility, troubleshooting, migration, upgrade/rollback, security, and
  known limitations, plus synchronized contributor release-contract guidance.
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

[Unreleased]: https://github.com/rookiestar28/ComfyUI-Sigmax/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rookiestar28/ComfyUI-Sigmax/tree/v1.0.0
