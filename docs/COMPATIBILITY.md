# Compatibility

## Current Status

ComfyUI-Sigmax is a pre-alpha development foundation. ComfyUI host integration, user-facing
nodes, Krea 2 schedules, model weights, GPU execution, and numerical model parity are
**not yet validated**.

The package metadata declares a ComfyUI floor of `0.29.0`, but that is a packaging target
rather than current host-compatibility evidence. No release should infer working host support
from the declaration alone.

## Validated Foundation Environments

The current package, quality gates, tests, and wheel inventory have been validated locally on:

| Environment | Python | Evidence scope |
| --- | --- | --- |
| Windows | 3.13 | Core independence, deterministic property, package, quality, unit, coverage, and wheel gates |
| WSL2/Linux path | 3.10 | Core independence, deterministic property, package, quality, unit, coverage, and wheel gates |

The supported Python floor is 3.10. Python versions or operating systems not listed above may
work, but do not yet have repository acceptance evidence.

## Dependency Boundary

| Component | Current policy |
| --- | --- |
| Runtime Python dependencies | None |
| Development tools | Version-bounded `dev` extra |
| Diffusers | Optional `reference` extra, currently `>=0.39,<0.40` |
| ComfyUI | Planned host dependency; not imported by the package shell |
| Node/browser tooling | Not required by the current Python-only foundation |
| Model weights and GPU runtime | Not downloaded or exercised |

Diffusers is intended as a pinned parity reference or optional backend. Closed-form schedule
construction does not require it at runtime. The canonical core lane requires both ComfyUI and
Diffusers to be absent, blocks attempted imports, and enumerates every core module in Python
isolated mode.

## Planned Validation Tiers

Compatibility claims will progress through separate lanes:

1. pure schedule and deterministic property tests — implemented;
2. authoritative golden and framework parity tests;
3. real ComfyUI host import and node integration;
4. fixed-seed sampler and latent-level comparison;
5. approved model/GPU workflows;
6. latest-host compatibility signals.

Passing a lower tier does not imply a higher tier.

The implemented property lane checks mathematical and serialization invariants. It is not an
official model parity lane; Krea 2 profile goldens and authoritative comparison remain
unimplemented until M2/M3.

## Current Known Limitations

- Node mappings are intentionally empty.
- No Krea 2 RAW or Turbo variant resolution exists.
- The pure schedule/artifact/capability core exists, but no resolved Krea 2 profile, public
  sigma-scheduler node, or full sampler is exposed.
- No ComfyUI version has completed real-host E2E validation.
- macOS and native hosted Ubuntu evidence are not yet available.
- Image-quality comparisons are not correctness evidence and have not begun.

For the intended component boundaries, see [Architecture](ARCHITECTURE.md).
