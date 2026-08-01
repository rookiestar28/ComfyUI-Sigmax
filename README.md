# ComfyUI-Sigmax

ComfyUI-Sigmax provides model-aware sigma schedules for ComfyUI. It includes verified profiles
for Krea 2 Turbo, Krea 2 RAW, Z-Image Base, Z-Image Turbo, and FLUX.1-schnell, plus tools for
inspecting and editing schedules.

## Features

- Explicit model and variant selection with no silent generic fallback.
- Verified Krea 2, Z-Image, and FLUX.1-schnell schedule recipes.
- Schedule slicing, concatenation, resampling, inspection, and comparison.
- Checkpoint header inspection without loading model weights.
- Versioned schedule information and fingerprints for reproducible workflows.
- No mandatory third-party runtime dependencies beyond the libraries provided by ComfyUI.

Sigmax builds `SIGMAS` for ComfyUI custom sampling. It does not replace the sampler and does not
download models.

## Installation

### ComfyUI Manager

Search for `ComfyUI-Sigmax` in ComfyUI Manager, install it, and restart ComfyUI. If it is not
available in Manager, use the Git installation below.

### Git

Stop ComfyUI, open a terminal in `ComfyUI/custom_nodes`, and run:

```bash
git clone https://github.com/rookiestar28/ComfyUI-Sigmax comfyui-sigmax
```

Restart ComfyUI. The expected package entry point is:

```text
ComfyUI/custom_nodes/comfyui-sigmax/__init__.py
```

Python 3.10 or newer and ComfyUI 0.29.0 or newer are required. Do not install the repository's
development or parity dependencies into ComfyUI just to use the nodes.

## Use in ComfyUI

Search the node menu for `Sigmax`. The package registers 14 namespaced nodes.

### Build a model schedule

| Model | Node | Recommended settings |
| --- | --- | --- |
| Krea 2 Turbo | `Sigmax.Krea2SigmaScheduler` | `Turbo`, 8 steps, Euler, CFG 1.0 |
| Krea 2 RAW | `Sigmax.Krea2SigmaScheduler` | `RAW`, 52 steps with CFG 4.5, or 28 steps with CFG 5.5 |
| Z-Image Base | `Sigmax.ZImageSigmaScheduler` | `Base`, 28-50 steps, default 50, CFG 4.0 |
| Z-Image Turbo | `Sigmax.ZImageSigmaScheduler` | `Turbo`, 8 steps, CFG 1.0 |
| FLUX.1-schnell | `Sigmax.Flux1SchnellSigmaScheduler` | 1-4 steps, default 4, CFG 1.0 |

For normal use:

1. Add the scheduler node for the selected model.
2. Select the exact variant and keep `strict_official` enabled.
3. For Krea 2, enter the actual output width and height.
4. Connect the node's `SIGMAS` output directly to a custom-sampling path that accepts external
   sigmas.
5. Do not pass the result through another scheduler or apply another time shift.
6. Read `schedule_info` when checking the selected recipe, dimensions, step range, or warnings.

`Sigmax.ModelAwareSigmaScheduler` can inspect a connected Krea 2 model, but shared filenames and
model structure may identify only the family. If `Auto` reports ambiguity, select `Turbo` or
`RAW` explicitly.

### Inspect or modify a schedule

- `Sigmax.ProfileInspector` and `Sigmax.ScheduleInspector` show profile and schedule details.
- `Sigmax.ScheduleComparison` compares two schedules.
- `Sigmax.ScheduleSlice`, `Sigmax.ScheduleConcatenate`, and `Sigmax.ScheduleResample` create a
  modified schedule while retaining traceable schedule information.
- `Sigmax.CheckpointEvidenceInspector` reads only the bounded header of a local `.safetensors`
  file. Its suggested variant is advisory, not confirmation.
- `Sigmax.AdvancedFlowMatchScheduler` is an experimental schedule constructor, not a universal
  model profile or sampler.

## Update or remove

For a Git installation:

```bash
cd ComfyUI/custom_nodes/comfyui-sigmax
git pull --ff-only
```

Restart ComfyUI after updating. Back up important workflows before changing versions. To remove
Sigmax, stop ComfyUI, remove or move only the `comfyui-sigmax` directory, and restart ComfyUI.

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| Nodes do not appear | Restart ComfyUI, inspect the startup log, and confirm there is only one direct `comfyui-sigmax` folder under `custom_nodes`. |
| Import fails | Confirm Python 3.10+ and ComfyUI 0.29.0+, then remove duplicate or nested installations. |
| `Auto` rejects Krea 2 | Select `Turbo` or `RAW` explicitly. Do not rely on the filename alone. |
| The schedule appears shifted twice | Remove the second scheduler or model time shift; use the Sigmax `SIGMAS` output once. |
| Krea 2 RAW dimensions differ | Check the requested and effective dimensions in `schedule_info`; RAW dimensions are normalized to the supported grid. |
| A newer ComfyUI release behaves differently | Review the supported boundary in [Compatibility](docs/COMPATIBILITY.md). |

ComfyUI custom nodes execute Python code inside the host process. Install reviewed sources only.
Model weights are not included, and current validation does not claim real-model GPU image
quality, stochastic sampling, resume behavior, or arbitrary-model compatibility.
