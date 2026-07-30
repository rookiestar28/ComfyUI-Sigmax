# ComfyUI-Sigmax

ComfyUI-Sigmax is a planned model-aware sigma schedule and sampler toolkit for ComfyUI. Its
first validation targets are Krea 2 Turbo and Krea 2 RAW; the longer-term design supports
versioned profiles for other flow-matching and diffusion model families.

> **Status: pre-alpha foundation.** The first user-facing `Krea 2 Sigma Scheduler` node is
> implemented as a statically validated legacy/current ComfyUI contract. The
> current repository provides a side-effect-free package shell, packaging metadata, quality
> gates, framework-independent schedule primitives, canonical schedule-artifact
> serialization, a frozen `ProfileSchemaV1`, and evidence-pinned Krea 2 Turbo and RAW
> structural profiles. It also
> provides typed model/profile/sampler capability preflight and complete independent
> 4/8/12/16-step Turbo golden vectors plus authoritative parity against pinned Krea code and
> Diffusers 0.39.0, plus native-ComfyUI Turbo schedule parity. Complete independent RAW
> 28/52-step golden vectors and executable authoritative/framework parity now cover square,
> landscape, and portrait geometry. A pure
> fail-closed resolver distinguishes authoritative/verified variant evidence from visible
> header and filename suggestions. A dependency-free ComfyUI adapter now normalizes reviewed
> public `/system_stats`, `/features`, `/object_info`, and Node Definition v2 evidence into
> immutable model/host/sampler contracts. A pure `sigmax.node-registration/1` catalog also
> discovers legacy/current and V3 node definitions, validates Node Definition v2 wire schemas,
> and produces collision-safe namespaced mappings. Its built-in catalog now contains
> `Sigmax.AdvancedFlowMatchScheduler`, `Sigmax.Krea2SigmaScheduler`, and
> `Sigmax.ModelAwareSigmaScheduler`, `Sigmax.ProfileInspector`, and
> `Sigmax.ScheduleComparison`, and `Sigmax.ScheduleInspector`; RAW native-ComfyUI and
> real-host node/workflow parity remain
> pending. The
> complete core and profile
> layer are dependency-free and have enforced
> isolation/property/golden/parity contract lanes.

## Why Sigmax

Sampling systems often mix four different concerns:

1. how a model interprets time or sigma;
2. how the sigma sequence is constructed;
3. which numerical sampler integrates model predictions;
4. which settings are authoritative for a specific model variant.

Sigmax will keep those concerns separate. A schedule must come from an explicit, traceable
profile; experimental transforms must remain distinguishable from reference behavior; and
features that require stateful sampler steps must not be represented as inert sigma controls.

## Planned Product Shape

The intended toolkit includes:

- official-parity Krea 2 Turbo and resolution-aware Krea 2 RAW schedules;
- a pure, independently testable schedule engine;
- versioned model profiles with evidence and provenance;
- simple model-aware and advanced schedule nodes;
- schedule inspection, comparison, metadata, and fingerprints;
- native ComfyUI Euler reuse where it is mathematically equivalent;
- separate full-sampler implementations only where sigma tensors are insufficient;
- numerical golden, parity, property, and integration test lanes.

These are roadmap targets, not claims about the current package.

## Current Repository Surface

The import contract is intentionally minimal:

```python
import comfyui_sigmax

assert sorted(comfyui_sigmax.NODE_CLASS_MAPPINGS) == [
    "Sigmax.AdvancedFlowMatchScheduler",
    "Sigmax.Krea2SigmaScheduler",
    "Sigmax.ModelAwareSigmaScheduler",
    "Sigmax.ProfileInspector",
    "Sigmax.ScheduleComparison",
    "Sigmax.ScheduleInspector",
    "Sigmax.TurboWorkflowOutput",
]
assert comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS == {
    "Sigmax.AdvancedFlowMatchScheduler": "Advanced FlowMatch Scheduler",
    "Sigmax.Krea2SigmaScheduler": "Krea 2 Sigma Scheduler",
    "Sigmax.ModelAwareSigmaScheduler": "Model-Aware Sigma Scheduler",
    "Sigmax.ProfileInspector": "Profile Inspector",
    "Sigmax.ScheduleComparison": "Schedule Comparison",
    "Sigmax.ScheduleInspector": "Schedule Inspector",
    "Sigmax.TurboWorkflowOutput": "Turbo Workflow Output",
}
```

Only validated product nodes enter these mappings. Runtime dependencies remain empty; schedule
nodes load host-provided Torch only when they execute, the Turbo workflow output publishes a
canonical model-free artifact/receipt bundle, and Diffusers remains isolated to an exactly
pinned parity environment.

The pure core currently exposes explicit schedule ownership, sigma domains, transform stages,
pre-execution compatibility validation, and immutable request/result structures:

```python
from comfyui_sigmax.core import (
    ScheduleInputs,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    apply_terminal_policy,
    exponential_mu_shift,
    krea_reciprocal_step_grid,
)

ownership = ScheduleOwnership.EXTERNAL_SIGMAS
domain = SigmaDomain.UNIT_FLOW
requested_inputs = ScheduleInputs(steps=8, width=1024, height=1024)
base_grid = krea_reciprocal_step_grid(8)
shifted = exponential_mu_shift(base_grid, mu=1.15)
sigmas = apply_terminal_policy(
    shifted,
    policy=TerminalPolicy.APPEND_ZERO,
    domain=domain,
)
```

Model-native, externally constructed, and model-patched schedules are mutually exclusive.
External transforms cannot be silently applied to native or patched ownership. Requested and
effective inputs are stored separately, and effective step or dimension changes require an
explicit override record.

The Krea builder returns only the unshifted, non-terminal base grid. Exponential `mu`,
direct-ratio, and explicit no-shift transforms are separate, unit-flow-only operations. The
terminal stage remains separate so terminal zero cannot be appended or transformed twice.
Terminal-inclusive start/end ranges and ComfyUI-compatible partial-denoise tail slicing are
also explicit; invalid or empty manual ranges fail instead of silently changing execution.

Complete external schedules can be validated for finite values, domain bounds, exact
transition count, strict decrease, and terminal-zero policy. The core also provides typed
IEEE-754 float tokens, bounded NFC canonical projections, and separate numerical and
construction SHA-256 fingerprints that reproduce the published artifact fixtures.

Validated external `ScheduleResult` values can now be built into immutable
`ScheduleArtifact` objects and transported as strict canonical UTF-8 JSON:

```python
from comfyui_sigmax.core import (
    build_schedule_artifact,
    deserialize_schedule_artifact,
    serialize_schedule_artifact,
)

# `result` and `metadata` are validated immutable core contracts.
artifact = build_schedule_artifact(result, metadata=metadata, precision="float64")
payload = serialize_schedule_artifact(artifact)
restored = deserialize_schedule_artifact(payload)

assert restored == artifact
```

The parser bounds input size, rejects duplicate keys, BOMs, invalid UTF-8, JSON floating
literals, non-standard constants, unknown schema fields, and non-canonical encodings. It
recomputes both numerical and construction fingerprints before returning an artifact.

Execution outcomes use a separate immutable `ExecutionReceipt`; schedule construction never
implies successful model or sampler execution. Receipts bind explicit host/model/sampler,
compatibility, RNG-ownership, transition/model-evaluation counts, and final status evidence to
the existing construction and numerical fingerprints:

```python
from comfyui_sigmax.core import (
    PortableExecutionBundle,
    build_execution_receipt,
    serialize_portable_execution_bundle,
)

# `metadata` is explicit validated runtime evidence.
receipt = build_execution_receipt(artifact, metadata=metadata)
bundle = PortableExecutionBundle(artifact=artifact, receipt=receipt)
portable_payload = serialize_portable_execution_bundle(bundle)
```

The receipt statuses are `not_executed`, `succeeded`, `failed`, and `interrupted`. Success
requires complete counts and a non-rejected compatibility decision; failure/interruption require
stable reason codes. The current package does not execute a sampler and does not expose a node
that can manually assert success.

Portable workflow metadata records exact package, node, host/API, profile, compatibility,
artifact, and receipt requirements without embedding complete artifacts, receipts, prompts, or
machine-local data:

```python
from comfyui_sigmax.core import (
    attach_workflow_metadata,
    extract_workflow_metadata,
)

saved_workflow = attach_workflow_metadata(workflow, metadata)
restored_metadata = extract_workflow_metadata(saved_workflow)
assert restored_metadata == metadata
```

Metadata uses the versioned `sigmax.workflow-metadata/1` contract under the
`extra.comfyui_sigmax` namespace. Attachment supports ComfyUI workflow versions `0.4` and `1`,
preserves nodes, links, widgets, subgraphs, positions, and unrelated `extra` members, and rejects
conflicting or malformed existing metadata. It does not validate the surrounding graph or claim
live-host compatibility.

The adjacent workflow validator supplies that static schema boundary for the packaged canonical
Turbo and RAW fixtures:

```python
from comfyui_sigmax.workflows import (
    WorkflowValidationLane,
    fetch_live_object_info,
    validate_live_workflow_fixtures,
    validate_pinned_workflow_fixtures,
)

static_report = validate_pinned_workflow_fixtures()
assert static_report.gate_passed

object_info = fetch_live_object_info()
live_report = validate_live_workflow_fixtures(
    object_info=object_info,
    host_version="0.29.0",
    host_revision="reviewed-host-revision",
    lane=WorkflowValidationLane.KNOWN_GOOD,
)
```

It checks node/input presence, linked and positional input types, widget slots, fixed combo
values, lifecycle flags, stable package identity, and M4-07 metadata. Reports use
`sigmax.workflow-validation-report/1`; known-good findings block, while latest-host findings stay
explicitly observational. The live loader accepts only a bounded literal-loopback
`/object_info` endpoint. This is not real-host workflow load or execution evidence.

Model, profile, sampler, and requested execution features also have immutable capability
contracts:

```python
from comfyui_sigmax.core import (
    CompatibilityLevel,
    ExecutionFeatureRequest,
    ModelCapabilities,
    ProfileCapabilities,
    SamplerCapabilities,
    evaluate_compatibility,
    require_compatible,
)

# `model`, `profile`, and `sampler` are validated capability declarations.
decision = evaluate_compatibility(
    model=model,
    profile=profile,
    sampler=sampler,
    request=ExecutionFeatureRequest(),
)
assert decision.level in {
    CompatibilityLevel.ALLOW,
    CompatibilityLevel.WARN,
}
require_compatible(decision)
```

The preflight checks model family and variant, prediction and sigma domains, schedule
ownership, terminal requirements, deterministic or stochastic behavior, noise ownership,
sampler state, partial denoise, and per-token timesteps. Its stable reason codes distinguish
`ALLOW`, `WARN`, and `REJECT`; rejected combinations fail before host or sampler execution.

The frozen profile contract exposes deterministic schema identities without importing host
frameworks:

```python
from comfyui_sigmax.profiles import (
    KREA2_RAW_SCHEMA,
    KREA2_TURBO_SCHEMA,
    ProfileSchemaV1,
    profile_schema_fingerprint,
)

assert isinstance(KREA2_TURBO_SCHEMA, ProfileSchemaV1)
assert KREA2_TURBO_SCHEMA.schema_id == "sigmax.model-profile/1"
assert KREA2_RAW_SCHEMA.model_variant == "raw"
schema_identity = profile_schema_fingerprint(KREA2_TURBO_SCHEMA)
assert schema_identity.startswith("sha256:")
```

`ProfileSchemaV1` validates identity, grid/transform/terminal/slicing semantics, recipes,
detection, compatibility capabilities, and artifact versions. Software source, framework,
and model-weight provenance remain separately versioned and separately licensed. Schema v1
freezes the validated external-sigma path; native/patch ownership contracts remain later
work.

Complete schemas can now be placed in an immutable exact-key registry:

```python
from comfyui_sigmax.profiles import (
    KREA2_TURBO_SCHEMA,
    ProfileKey,
    builtin_profile_registry,
)

registry = builtin_profile_registry()
entry = registry.resolve(ProfileKey.from_schema(KREA2_TURBO_SCHEMA))
assert entry.schema is KREA2_TURBO_SCHEMA
```

`ProfileRegistry` uses exact namespaced IDs and numeric versions; it performs no “latest”,
prefix, or fallback selection. External conflicts reject by default. Explicit
compare-and-swap replacement is limited to existing external entries and requires the exact
old fingerprint; an external registration can never replace an official built-in.
Inheritance names an already registered parent and the exact canonical top-level fields that
changed. The child remains a complete validated schema, and inherited external profiles must
use `modified` evidence.

Registered profiles can be resolved against normalized model, host, and sampler evidence
without importing ComfyUI or executing sampling:

```python
from comfyui_sigmax.core import ExecutionFeatureRequest
from comfyui_sigmax.profiles import (
    HostCapabilities,
    HostCapabilityEvidence,
    HostCapabilityLifecycle,
    ModelCapabilityEvidence,
    model_identity_from_krea2_resolution,
    resolve_krea2_variant,
    resolve_profile_capabilities,
)

identity = model_identity_from_krea2_resolution(resolve_krea2_variant(explicit_variant="turbo"))
model = ModelCapabilityEvidence(
    evidence_version="1",
    identity=identity,
    capabilities=entry.schema.model_capabilities,
)
host = HostCapabilities(
    evidence_version="1",
    host_id="comfyui",
    host_version="0.29.0",
    host_revision="e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
    capabilities=(
        HostCapabilityEvidence(
            capability_id="sampler.comfy.euler",
            lifecycle=HostCapabilityLifecycle.LANDED,
        ),
        HostCapabilityEvidence(
            capability_id="schedule.external_sigmas",
            lifecycle=HostCapabilityLifecycle.LANDED,
        ),
    ),
)
decision = resolve_profile_capabilities(
    registered_profile=entry,
    model=model,
    host=host,
    sampler=entry.schema.reference_sampler_capabilities,
    request=ExecutionFeatureRequest(),
)
assert decision.schema_id == "sigmax.capability-resolution/1"
```

Only confirmed model identity can pass. Suggested, ambiguous, conflicting, or unknown identity
stays unresolved even if a declared capability variant happens to match. Every required host
capability retains a `landed`, `experimental`, `unsupported`, or missing result; required
experimental capabilities fail closed because they are not stable host APIs.

The dependency-free `comfyui_sigmax.adapters` boundary now supplies the static host-evidence
side under `sigmax.comfyui-adapter/1`. It probes only public attributes of an already loaded
numbered Comfy API module, normalizes the V1-compatible `/object_info` projection and documented
Node Definition JSON v2, and derives external-SIGMAS, Euler, and partial-denoise lifecycle
evidence from actual node inputs and combo options. The current ComfyUI `v0_0_2` API remains
`experimental`; it cannot satisfy `require_stable_numbered_api()`. The initial static-contract
window is exactly ComfyUI `0.29.0`; this is not a real-host node/workflow E2E claim. Network
access, host module loading, and sampling remain separate later layers.

The adjacent pure registration boundary uses `sigmax.node-registration/1`. It requires explicit
stable IDs in the `Sigmax.<Name>` form, such as `Sigmax.ExampleNode`, and rejects conflicting duplicate registrations without
overwriting unrelated nodes, and returns fresh legacy/current mapping, `/object_info`, and Node
Definition JSON v2 projections. V3 classes are discovered through `GET_SCHEMA()` and
`GET_NODE_INFO_V1()` and may share the mapping projection with legacy classes. Activation through
the current experimental numbered API is rejected when stability is required. This design avoids
the pinned loader's mutually exclusive `NODE_CLASS_MAPPINGS`/`comfy_entrypoint` branches and is
independent of the normalized installation-directory name. The built-in catalog now exposes
`Sigmax.AdvancedFlowMatchScheduler`, `Sigmax.Krea2SigmaScheduler`, and
`Sigmax.ModelAwareSigmaScheduler`, `Sigmax.ProfileInspector`, and
`Sigmax.ScheduleComparison`, and `Sigmax.ScheduleInspector`.

The `Krea 2 Sigma Scheduler` requires an explicit `Turbo` or `RAW` choice, width and height,
steps, strict-official mode, and terminal-inclusive start/end slicing. Strict mode accepts only
Turbo 8-step and RAW official 52-step construction; relaxed mode additionally allows modified
Turbo step counts and the named RAW 28-step framework-reference recipe. It returns a ComfyUI
`SIGMAS` tensor plus deterministic `sigmax.krea2-sigma-node/1` JSON containing the exact profile,
recipe/evidence, requested/effective geometry, applied shift, selected range, warnings, and
complete/output fingerprints. This is a sigma scheduler, not a sampler: it does not execute
Euler, apply guidance, patch model sampling, or claim live-host workflow validation.

The `Model-Aware Sigma Scheduler` requires a ComfyUI `MODEL` and exposes `Auto`, `Turbo`, and
`RAW`. Its bounded public probe can confirm the Krea 2 family but does not inspect filenames,
weights, private host state, or infer a variant from the shared Krea 2 class. Consequently,
family-only `Auto` selection fails visibly as ambiguous instead of guessing or using a generic
fallback. Explicit RAW/Turbo selection follows the existing `explicit_selection` precedence,
resolves the exact built-in `ProfileRegistry` entry, and gates M4-01 construction through the
complete capability resolver. Its deterministic `sigmax.model-aware-sigma-node/1` information
contains stable reason codes, the profile key/fingerprint/evidence, the full capability decision,
and a clearly labeled pinned `static_contract` host record. This is also not a sampler and does
not claim real-host validation.

The `Advanced FlowMatch Scheduler` (`Sigmax.AdvancedFlowMatchScheduler`) constructs an explicitly
external `UNIT_FLOW` schedule from a finite descending linear endpoint grid. It selects exactly
one primary shift parameterization—`exponential_mu` or `direct_ratio`—and uses one
mode-dependent `shift_value`, so mutually exclusive parameter controls cannot become inert.
Their identity values (`mu = 0`, `ratio = 1`) provide an explicit no-shift result. Terminal
append/preserve behavior and terminal-inclusive slicing execute after the primary shift in the
declared order. The node returns deterministic `sigmax.advanced-flowmatch-node/1` information
and fingerprints. Its provenance is `experimental`: it is not a sampler, registered generic
model profile, cross-model compatibility claim, or model patch.

`Sigmax.ProfileInspector` is a read-only exact-profile view for an explicitly selected Krea 2
variant. It reports bounded model identity confidence, the connected native sampling class
(`ModelSamplingFlux` when present), the `comfy.euler` reference sampler, requested/effective
dimensions, computed shift, capability decision, provenance, warnings, and fingerprints as
deterministic `sigmax.profile-inspector/1` JSON. It uses static bounded MODEL reads and never
serializes or invokes the foreign model.

`Sigmax.ScheduleInspector` accepts connected `SIGMAS` plus one implemented versioned
`schedule_info` projection. It bounds and validates the JSON, normalizes direct Krea,
model-aware, or advanced FlowMatch information, recomputes the selected sigma fingerprint, and
fails if it does not match the advertised output identity. Only a matching read-only
`sigmax.schedule-inspector/1` report is returned; the node does not modify the schedule.

`Sigmax.ScheduleComparison` accepts two such verified `SIGMAS`/`schedule_info` pairs. Matching
domains and lengths are compared by terminal-inclusive sigma index, with absolute differences
and symmetric relative differences (`abs(a-b) / max(abs(a), abs(b))`, or zero when both are
zero), plus source transform metadata and aggregate maxima/means. Length or domain mismatch
returns a deterministic non-comparable `sigmax.schedule-comparison/1` report; it never truncates,
resamples, or converts a schedule.

The first concrete profile is an immutable, evidence-pinned declaration of the official
Krea 2 Turbo recipe:

```python
from comfyui_sigmax.profiles import (
    KREA2_TURBO_PROFILE,
    build_krea2_turbo_schedule,
)

assert KREA2_TURBO_PROFILE.profile_id == "krea2.turbo.official"
result = build_krea2_turbo_schedule(width=1025, height=1024)
assert result.effective_inputs.width == 1040
assert len(result.sigmas) == 9
```

The profile declares unit-flow external schedule ownership, the Krea reciprocal-step grid,
fixed exponential `mu = 1.15`, terminal zero, deterministic ComfyUI Euler capabilities,
Krea-guidance `0.0` / ComfyUI-CFG `1.0`, and ceil-to-16 dimensions. Its references are pinned
to immutable Krea, Diffusers, and ComfyUI revisions. Non-eight-step construction is allowed
only as an explicitly `modified` result.

The immutable `KREA2_RAW_PROFILE` separately declares the RAW variant without pretending that
RAW is Turbo with more steps:

```python
from comfyui_sigmax.profiles import (
    KREA2_RAW_PROFILE,
    build_krea2_raw_schedule,
    derive_krea2_raw_shift,
)

assert KREA2_RAW_PROFILE.profile_id == "krea2.raw.official"
assert KREA2_RAW_PROFILE.shift_policy.base_image_seq_len == 256
assert KREA2_RAW_PROFILE.shift_policy.max_image_seq_len == 6400

derivation = derive_krea2_raw_shift(1025, 767)
assert (
    derivation.geometry.requested_width,
    derivation.geometry.requested_height,
) == (1025, 767)
assert (
    derivation.geometry.effective_width,
    derivation.geometry.effective_height,
) == (1040, 768)
assert derivation.geometry.image_seq_len == 3120

schedule = build_krea2_raw_schedule(width=1025, height=767)
assert len(schedule.sigmas) == 53
```

Its resolution-linear exponential-`mu` policy records endpoints `0.5` and `1.15`, upstream
unclamped extrapolation, ceil-to-16 dimensions, terminal zero, and deterministic ComfyUI
Euler capabilities. It keeps `krea2.raw.official-full-52` (Krea guidance 3.5 / ComfyUI CFG
4.5) distinct from `krea2.raw.diffusers-reference-28` (4.5 / 5.5).
`derive_krea2_raw_shift()` retains requested and effective pixel dimensions, rounds each
dimension upward to 16, calculates the packed image sequence length, and derives official
unclamped RAW `mu`. `build_krea2_raw_schedule()` composes an exact named 28- or 52-step
recipe; it does not accept an arbitrary silently modified step count. Automatic variant
resolution never trusts the shared ComfyUI model class or filename alone:

```python
from comfyui_sigmax.profiles import (
    KREA2_TURBO_OFFICIAL_SHA256,
    Krea2Variant,
    resolve_krea2_variant,
)

resolution = resolve_krea2_variant(
    checkpoint_sha256=KREA2_TURBO_OFFICIAL_SHA256,
    filename="renamed.safetensors",
)
assert resolution.resolved_variant is Krea2Variant.TURBO
```

Explicit selection, trusted Sigmax/Diffusers metadata, and exact official file hashes can
resolve a variant. Local safetensors metadata and filenames only produce visible suggestions
in flexible mode; tensor keys and the shared ComfyUI `Krea2` class confirm only the family.
Strict official mode raises on suggestions, ambiguity, or conflicting strong evidence.

The committed `tests/golden/krea2_raw_v1.json` fixture contains 14 complete terminal-inclusive
RAW cases: both named recipes across 256, 512, 768, 1024, and 1280 square resolutions plus
1360x768 landscape and 768x1360 portrait. Its independent precision-80 Decimal generator
recalculates geometry, sequence length, affine `mu`, and both float64/float32 vectors without
importing Sigmax or optional frameworks.

The committed `tests/golden/krea2_turbo_v1.json` fixture freezes complete terminal-inclusive
4-, 8-, 12-, and 16-step float64 and IEEE-754 float32 vectors. A standard-library-only
high-precision Decimal generator imports no Sigmax, ComfyUI, Diffusers, NumPy, or PyTorch
code; regeneration must match the canonical fixture byte for byte. The eight-step vector also
passes a separate implementation of Krea's official direct expression.

The committed `tests/parity/fixtures/krea2_turbo_parity_v1.json` report adds independent
execution evidence. It compares the production builder with pinned Krea official code at
float64 and the actual Diffusers 0.39.0 FlowMatch scheduler at float32 for 4, 8, 12, and 16
steps. The report records complete vectors, immutable source revisions, exact dependency
versions, CPU/dtype, tolerances, maximum and mean absolute errors, and schedule fingerprints.
Its largest observed Diffusers error is `5.960464477539063e-08`, below the enforced `1e-6`
float32 tolerance. Only the 8-step result retains official Turbo evidence; the other step
counts are explicitly modified differential cases.

The separate `tests/parity/fixtures/krea2_raw_parity_v1.json` report executes all 14 RAW
cases: the named 28- and 52-step recipes at 256, 512, 768, 1024, and 1280 square geometry,
plus 1360x768 landscape and 768x1360 portrait. It independently executes the pinned Krea
geometry, affine-`mu`, and timestep formulas at float64 and the actual Diffusers 0.39.0
`FlowMatchEulerDiscreteScheduler` at float32. The largest observed errors are
`9.992007221626409e-16` against Krea and `1.1920928955078125e-07` against Diffusers, below
the enforced `1e-8` and `1e-6` tolerances. The report records complete vectors, requested and
effective dimensions, image sequence length, calculated `mu`, source identities, dependency
versions, fingerprints, and error statistics. Its float64 evidence vectors use a declared
15-significant-digit normalization to remove sub-tolerance platform `libm` noise.

The canonical gate runs `scripts/check_core_independence.py` before pytest. It requires a
clean dev environment, launches Python isolated mode, blocks `comfy` and `diffusers`, and
imports every core and profile module. Static import-boundary and deterministic
property/metamorphic/golden tests provide complementary evidence. These checks prove
framework independence and detect formula drift. The default gate validates the committed
Turbo and RAW parity reports without importing optional frameworks; dedicated hosted CI jobs
recreate them in isolated exactly pinned Diffusers environments.

The separate
`tests/parity/fixtures/krea2_turbo_comfy_native_parity_v1.json` report executes the actual
pinned ComfyUI `ModelSamplingFlux` and registered `simple` scheduler on CPU. It records
complete 4-, 8-, 12-, and 16-step float32 vectors, source blobs, environment versions, and
fingerprints. Exact table-position cases remain within `1e-6`; the 12-step case has an
explicit `2e-4` bound for ComfyUI's documented 10,000-entry integer-index quantization. This
proves native schedule parity only: no checkpoint, Euler latent integration, host process, or
workflow is executed.

## Development Setup

Python 3.10 or newer is required.

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
powershell -File scripts/run_full_tests_windows.ps1
```

Linux or WSL:

```bash
python3 -m venv .venv-wsl
.venv-wsl/bin/python -m pip install -e '.[dev]'
bash scripts/run_full_tests_linux.sh
```

See [Contributing](CONTRIBUTING.md) for the development workflow and
[Compatibility](docs/COMPATIBILITY.md) for the distinction between validated environments
and planned host/model support.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Model profile schema v1 specification](docs/PROFILE_SPEC.md)
- [Schedule artifact specification](docs/SCHEDULE_ARTIFACT_SPEC.md)
- [Execution receipt and portable bundle specification](docs/EXECUTION_RECEIPT_SPEC.md)
- [Workflow metadata specification](docs/WORKFLOW_METADATA_SPEC.md)
- [Workflow validation specification](docs/WORKFLOW_VALIDATION_SPEC.md)
- [Compatibility](docs/COMPATIBILITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Repository identity](docs/REPOSITORY_IDENTITY.md)
- [Base import and attribution manifest](docs/BASE_IMPORT_MANIFEST.md)

## License and Attribution

ComfyUI-Sigmax is distributed under the [MIT License](LICENSE.TXT). The project retains
attribution for its audited source lineage in [NOTICE](NOTICE).
