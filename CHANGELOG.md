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

### Security

- Unfinished external scheduler code and unrelated global patches are excluded from the
  runtime import path.
- Runtime dependency and built-wheel inventories are enforced by tests.

### Known limitations

- No user-facing ComfyUI node, model profile, sigma schedule, or sampler is implemented.
- Real ComfyUI host, model-weight, GPU, and numerical parity validation remain pending.

[Unreleased]: https://github.com/rookiestar28/ComfyUI-Sigmax/commits/main
