# Changelog

All notable user-visible changes will be documented in this file.

The project follows semantic versioning for package releases. Model profile versions will be
tracked independently once the profile schema is implemented.

## [Unreleased]

### Added

- Side-effect-free `comfyui_sigmax` package shell with intentionally empty node mappings.
- Python packaging metadata and typed-package marker.
- Repository-local quality, test, coverage, and wheel validation.
- Cross-platform Windows and Linux/WSL full-gate wrappers.
- Minimal-permission continuous-integration workflow contract.
- Public architecture, provisional profile, compatibility, and contribution documentation.
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
- Immutable model, profile, sampler, and execution-feature capability declarations with
  canonical allow/warn/reject decisions, stable reason codes, and a fail-before-execution
  compatibility gate.
- An ordered core-independence gate that requires absent optional frameworks, blocks
  ComfyUI/Diffusers imports under Python isolated mode, imports every core module, and checks
  static import roots.
- Dependency-free deterministic property/metamorphic coverage for grids, shifts, terminal
  structure, and capability-decision stability.
- Wheel inventory enforcement for the pure-core package.

### Security

- Unfinished external scheduler code and unrelated global patches are excluded from the
  runtime import path.
- Runtime dependency and built-wheel inventories are enforced by tests.
- Artifact transport rejects oversized inputs, BOMs, invalid UTF-8, duplicate JSON keys,
  floating literals, non-standard constants, unknown fields, non-canonical bytes, stale
  fingerprints, secret-like metadata fields, and private local paths.

### Known limitations

- No user-facing ComfyUI node, model profile, sigma schedule, or sampler is implemented.
- Real ComfyUI host, model-weight, GPU, and numerical parity validation remain pending.

[Unreleased]: https://github.com/rookiestar28/ComfyUI-Sigmax/commits/main
