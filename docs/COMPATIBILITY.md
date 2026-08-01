# Compatibility

This page summarizes the supported user-facing boundary for ComfyUI-Sigmax 1.0.0.

## Environment

| Component | Supported boundary |
| --- | --- |
| Python | 3.10 or newer |
| ComfyUI package requirement | 0.29.0 or newer |
| Validated host baseline | ComfyUI 0.29.0 |
| Operating systems covered by project gates | Windows and Linux/WSL |
| Mandatory additional Python packages | None |

A newer ComfyUI version may work, but it is not automatically promoted to the validated host
baseline. When behavior changes after a ComfyUI update, first reproduce the workflow on the
validated baseline.

## Supported model profiles

| Profile | Supported schedule | User selection |
| --- | --- | --- |
| Krea 2 Turbo | Fixed-shift external sigma schedule, 8-step official recipe | Select `Turbo` explicitly when model evidence is ambiguous |
| Krea 2 RAW | Resolution-aware external sigma schedule, official 52-step and framework-reference 28-step recipes | Select `RAW` explicitly and provide the actual width and height |
| Z-Image Base | Fixed-ratio external sigma schedule, 28-50 steps | Select `Base` explicitly |
| Z-Image Turbo | Fixed-ratio external sigma schedule, 8-step official recipe | Select `Turbo` explicitly |
| FLUX.1-schnell | Unshifted external sigma schedule, 1-4 steps | Use the dedicated FLUX.1-schnell node |

The generic advanced FlowMatch node constructs explicit schedule math only. It is experimental
and does not establish compatibility with an arbitrary model.

## Usage boundary

- Sigmax produces sigma schedules; it does not replace the numerical sampler.
- External Sigmax sigmas must not be shifted or scheduled a second time.
- Krea 2 `Auto` selection fails closed when evidence identifies the family but not RAW versus
  Turbo.
- Checkpoint header inspection does not load weights and cannot confirm a model variant from weak
  filename or structural evidence.
- Slicing, concatenating, or resampling a schedule changes its evidence status to modified.
- Model weights must be obtained separately under their own licenses and access terms.

## Not currently claimed

- Real-model GPU execution or image-quality parity.
- Stochastic, resumable, or interrupted sampler state.
- Partial-denoise execution and advanced model-patch workflows.
- Automatic compatibility with unlisted model families.
- Guaranteed compatibility with every future ComfyUI release.

If a workflow falls outside these boundaries, treat it as unvalidated rather than silently
substituting another profile or schedule.
