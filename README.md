# ComfyUI-Sigmax

ComfyUI-Sigmax is a planned model-aware sigma schedule and sampler toolkit for ComfyUI. Its
first validation targets are Krea 2 Turbo and Krea 2 RAW; the longer-term design supports
versioned profiles for other flow-matching and diffusion model families.

> **Status: pre-alpha foundation.** No user-facing ComfyUI nodes are implemented yet. The
> current repository provides a side-effect-free package shell, packaging metadata, quality
> gates, framework-independent schedule primitives, canonical schedule-artifact
> serialization, and evidence-pinned Krea 2 Turbo and RAW structural profiles. It also
> provides typed model/profile/sampler capability preflight and complete independent
> 4/8/12/16-step Turbo golden vectors plus authoritative parity against pinned Krea code and
> Diffusers 0.39.0, plus native-ComfyUI Turbo schedule parity. Complete independent RAW
> 28/52-step golden vectors and executable authoritative/framework parity now cover square,
> landscape, and portrait geometry. A pure
> fail-closed resolver distinguishes authoritative/verified variant evidence from visible
> header and filename suggestions. No ComfyUI nodes are exposed, and RAW native-ComfyUI or
> host parity remains pending. The complete core and profile
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

assert comfyui_sigmax.NODE_CLASS_MAPPINGS == {}
assert comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS == {}
```

Empty mappings prevent unfinished nodes from being registered. Runtime dependencies are also
empty; Diffusers is used only in a separate, exactly pinned parity environment.

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
- [Provisional model profile specification](docs/PROFILE_SPEC.md)
- [Schedule artifact specification](docs/SCHEDULE_ARTIFACT_SPEC.md)
- [Compatibility](docs/COMPATIBILITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Repository identity](docs/REPOSITORY_IDENTITY.md)
- [Base import and attribution manifest](docs/BASE_IMPORT_MANIFEST.md)

## License and Attribution

ComfyUI-Sigmax is distributed under the [MIT License](LICENSE.TXT). The project retains
attribution for its audited source lineage in [NOTICE](NOTICE).
