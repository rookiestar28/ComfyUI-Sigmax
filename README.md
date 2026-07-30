# ComfyUI-Sigmax

ComfyUI-Sigmax is a planned model-aware sigma schedule and sampler toolkit for ComfyUI. Its
first validation targets are Krea 2 Turbo and Krea 2 RAW; the longer-term design supports
versioned profiles for other flow-matching and diffusion model families.

> **Status: pre-alpha foundation.** No user-facing ComfyUI nodes are implemented yet. The
> current repository provides a side-effect-free package shell, packaging metadata, quality
> gates, framework-independent schedule primitives, and canonical schedule-artifact
> serialization. It also provides typed model/profile/sampler capability preflight, but does
> not yet expose resolved model profiles, parity-validated Krea 2 schedules, or ComfyUI nodes.

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
empty; Diffusers is an optional reference dependency used for later parity research.

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
