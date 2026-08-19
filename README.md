# ComfyUI-Sigmax
ComfyUI-Sigmax provides model-aware sigma schedules for ComfyUI, with supported Krea 2, Z-Image, FLUX.1-schnell, Qwen Image, SD3, AuraFlow v0.2, Lumina-Image 2.0, HunyuanImage 2.1, MiniMax H3 Base FL2VA/Ref2VA, Anima, Wan 2.1/2.2, and LTX profiles plus editing tools.

## Table of contents

- [Features](#features)
- [Installation](#installation): [ComfyUI Manager](#comfyui-manager) · [Git](#git)
- [Use in ComfyUI](#use-in-comfyui)
  - [Build a model schedule](#build-a-model-schedule)
    - [Image model families](#image-model-families)
    - [Video and audio-video model families](#video-and-audio-video-model-families)
  - [Inspect or modify a schedule](#inspect-or-modify-a-schedule)
- [Update or remove](#update-or-remove)
- [Troubleshooting](#troubleshooting)

## Features

- Explicit model and variant selection with no silent generic fallback.
- Supported Krea 2, Z-Image, FLUX.1-schnell, Qwen Image, SD3, AuraFlow v0.2, Lumina-Image 2.0, HunyuanImage 2.1, MiniMax H3 Base FL2VA/Ref2VA, Anima, Wan 2.1/2.2, and LTX recipes.
- Twenty-four namespaced nodes for schedule construction, model-aware selection, inspection, comparison, editing, checkpoint evidence, and experimental Krea 2 conditioning.
- Schedule slicing, concatenation, resampling, inspection, and comparison.
- Experimental Krea 2 `CONDITIONING` tap rebalancing for explicitly selected RAW or Turbo workflows, with fixed RMS preservation and no scheduler/model patching.
- MiniMax H3 Base workflow construction with explicit FL2VA/Ref2VA selection, upstream model-shift integration, and model-free `/object_info` schema preflight.
- Checkpoint header inspection without loading model weights.
- Versioned schedule information and fingerprints for reproducible workflows.
- No mandatory third-party runtime dependencies beyond the libraries provided by ComfyUI.

Sigmax builds `SIGMAS` for ComfyUI custom sampling. It does not replace the sampler and does not download models.

## Installation

### ComfyUI Manager

Search for `ComfyUI-Sigmax` in ComfyUI Manager, install it, and restart ComfyUI. If it is not available in Manager, use the Git installation below.

### Git

Stop ComfyUI, open a terminal in `ComfyUI/custom_nodes`, and run:

```bash
git clone https://github.com/rookiestar28/ComfyUI-Sigmax comfyui-sigmax
```

Restart ComfyUI. The expected package entry point is:

```text
ComfyUI/custom_nodes/comfyui-sigmax/__init__.py
```

Python 3.10 or newer and ComfyUI 0.29.0 or newer are required. Do not install the repository's development or parity dependencies into ComfyUI just to use the nodes.

## Use in ComfyUI

Search the node menu for `Sigmax`. The package registers 24 namespaced nodes.

### Build a model schedule

#### Image model families

| Model | Node | Recommended settings |
| --- | --- | --- |
| Krea 2 Turbo | `Sigmax.Krea2SigmaScheduler` | `Turbo`, 8 steps, Euler, CFG 1.0 |
| Krea 2 RAW | `Sigmax.Krea2SigmaScheduler` | `RAW`, 52 steps with CFG 4.5, or 28 steps with CFG 5.5 |
| Krea 2 RAW + RAW-to-Turbo LoRA | `Sigmax.Krea2SigmaScheduler` | Experimental RAW/Turbo `mu`; user-selected steps; community starting point: 12 steps, Euler, CFG 1.0 |
| Z-Image Base | `Sigmax.ZImageSigmaScheduler` | `Base`, 28-50 steps, default 50, CFG 4.0 |
| Z-Image Turbo | `Sigmax.ZImageSigmaScheduler` | `Turbo`, 8 steps, CFG 1.0 |
| FLUX.1-schnell | `Sigmax.Flux1SchnellSigmaScheduler` | 1-4 steps, default 4, CFG 1.0 |
| Original Qwen Image | `Sigmax.QwenImageSigmaScheduler` | `Comfy Fixed`, 50 steps, or `Diffusers Dynamic` with explicit `image_seq_len`; host true CFG 4.0 |
| Original Stable Diffusion 3 | `Sigmax.SD3SigmaScheduler` | Select `Publisher Reference (1.0)` at 50 steps or `Comfy/Diffusers Fixed (3.0)` at 28 steps; source mode is required |
| Original AuraFlow v0.2 | `Sigmax.AuraFlowSigmaScheduler` | `Official Fixed (1.73)`, 50 steps, CFG 3.5; source mode is explicit |
| Lumina-Image 2.0 | `Sigmax.Lumina2SigmaScheduler` | `Official Fixed (6.0)`, 50 steps, CFG 4.0; source mode is explicit |
| HunyuanImage 2.1 | `Sigmax.HunyuanImage21SigmaScheduler` | `Base (5.0)`, 50 steps, CFG 3.5, or `Distilled (4.0)`, 8 steps, CFG 3.25; variant is explicit |
| Anima Base v1.0 / Aesthetic / Turbo | `Sigmax.AnimaSigmaScheduler` | `Base`, 30-50 steps, default 50, CFG 4.5; `Aesthetic` uses the same recipe; `Turbo`, 8-12 steps, CFG 1.0; fixed shift 3.0 |

Connect an image scheduler's `SIGMAS` directly to a custom-sampling path that accepts external sigmas. Keep `strict_official` enabled for official variants and use `schedule_info` to confirm the recipe, dimensions, step range, and warnings. Do not pass the result through another scheduler or apply another time shift.

- **Krea 2:** Select `Turbo` or `RAW` explicitly and enter the actual output width/height. `Sigmax.ModelAwareSigmaScheduler` may identify only the family, so resolve an ambiguous `Auto` result manually. `Sigmax.Krea2ConditioningRebalance` is an optional experimental `CONDITIONING` transform with identity at strength `0`; select `RAW`/`Turbo` and its profile explicitly. The RAW-to-Turbo LoRA schedule modes force `strict_official` off, do not load or scale a LoRA, and are only for a compatible LoRA applied to RAW.
- **Z-Image:** Select `Base` or `Turbo` explicitly; the node supplies the validated schedule only and does not load a checkpoint or choose a sampler.
- **FLUX.1-schnell:** Use the dedicated 1-4-step schedule for the schnell family; other FLUX variants are not implied.
- **Qwen Image:** Only the original text-to-image family is covered. `Comfy Fixed` mirrors `1.15`; `Diffusers Dynamic` requires explicit `image_seq_len`. Later variants and image-quality parity are outside scope.
- **Stable Diffusion 3:** Only original SD3 Medium is covered. Choose the publisher `1.0` or pinned ComfyUI/Diffusers `3.0` source mode explicitly; SD3.5, Turbo, ControlNet, execution, and quality claims are excluded.
- **AuraFlow:** Support is limited to original fal AuraFlow v0.2 with fixed ratio `1.73` and 50 steps; other versions, finetunes, dynamic shifts, execution, and quality claims are excluded.
- **Lumina-Image 2.0:** Support is limited to the original Alpha-VLLM text-to-image schedule with fixed ratio `6.0` and 50 steps; video, mGPT, editing paths, dynamic shifts, execution, and quality claims are excluded.
- **HunyuanImage 2.1:** Base and Distilled use explicit direct-ratio paths (`5.0`/`4.0`). Base is the pinned ComfyUI-compatible lane; Distilled remains publisher-schedule-only. Refiner, encoders, conditioning, weights, and quality claims are excluded.
- **Anima:** Select Base, Aesthetic, or Turbo explicitly. Aesthetic shares the Base recipe; all three paths construct schedules only and do not load model components.

#### Video and audio-video model families

| Model | Node | Recommended settings |
| --- | --- | --- |
| MiniMax H3 Base | `Sigmax.MiniMaxH3SigmaScheduler` | Select FL2VA or Ref2VA explicitly. `h3_endpoint` remains the default pure schedule; nine additional ComfyUI-native scheduler choices are experimental and require the already-shifted H3 `MODEL` |
| Wan 2.1 T2V / I2V | `Sigmax.WanSigmaScheduler` | Select generation, task, source, and resolution explicitly; official T2V 50 steps (`5.0`), official I2V 480P/720P 40 steps (`3.0`/`5.0`) |
| Wan 2.1 FLF2V / VACE | `Sigmax.WanSigmaScheduler` | Official-native FLF2V 14B 720P uses 50 steps (`16.0`); VACE 1.3B/14B use 50 steps (`16.0`) |
| Wan 2.2 TI2V 5B / A14B T2V/I2V | `Sigmax.WanSigmaScheduler` | Native or Diffusers-reference source lanes; TI2V 5B uses `5.0`; A14B T2V/I2V use `12.0`/`5.0` with caller-owned boundary metadata |
| Wan 2.2 S2V / Animate | `Sigmax.WanSigmaScheduler` | Official-native S2V 14B uses 40 steps (`3.0`); Animate 14B uses 20 steps (`5.0`) |
| Wan Animate 2 Base / Distilled | `Sigmax.WanSigmaScheduler` | Official-native Base uses 40 steps (`5.0`); Distilled uses 10 steps (`5.0`); select the task explicitly |
| LTXV 0.9.8 / LTX-2 19B / LTX-2.3 22B | `Sigmax.LTXSigmaScheduler` | Dev adaptive token shift (20/40/30 default steps) or explicit LTX-2/LTX-2.3 distilled Stage 1/2 vectors; generation and stage are explicit |

Connect a video scheduler's `SIGMAS` directly to the matching custom-sampling path. Do not add another scheduler or time shift, and inspect `schedule_info` for the selected generation mode, stage, resolution, boundary ownership, and warnings.

- **MiniMax H3:** Select Base FL2VA or Ref2VA explicitly. The `scheduler` menu contains `h3_endpoint`, `simple`, `sgm_uniform`, `karras`, `exponential`, `ddim_uniform`, `beta`, `normal`, `linear_quadratic`, and `kl_optimal`. The default `h3_endpoint` path is the existing pure endpoint-inclusive schedule and needs no `MODEL`. Every other choice is an experimental ComfyUI-native lane: connect the `MODEL` after upstream `MiniMaxH3SigmaShift`, keep its video/audio shifts aligned with the selected Base or Turbo recipe, and do not add a separate `BasicScheduler` or another time shift. These native choices are functional compatibility options, not MiniMax recommendations or quality, speed, memory, NFE, or acceleration claims. The complete nine-scheduler/two-host validation matrix is tracked separately.
- **Wan 2.1/2.2 and Wan Animate 2:** Select generation, task, source, and resolution explicitly. Wan 2.1 I2V requires `480P` or `720P`; Wan 2.2 A14B boundaries are caller-owned metadata and never route experts. FLF2V, VACE, S2V, Animate, and Wan Animate 2 rows are official-native schedule math only; Diffusers-reference lanes describe scheduler construction only. Execution, weights, expert routing, conditioning, and video-quality parity are excluded.
- **LTX:** Select LTXV 0.9.8, LTX-2 19B, or LTX-2.3 22B plus generation/stage explicitly. Dev mode derives one token-count shift; distilled modes use immutable publisher vectors. Sigmax does not load video weights, run encoders, or claim video-quality parity.

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
| MiniMax H3 native scheduler asks for `MODEL` or reports a shift mismatch | Connect the H3 model after `MiniMaxH3SigmaShift`. Use `12.0`/`3.0` for Base or the exact selected experimental Turbo recipe values; do not add a separate `BasicScheduler` or second shift. |
| A newer ComfyUI release behaves differently | Review the supported boundary in [Compatibility](docs/COMPATIBILITY.md). |

ComfyUI custom nodes execute Python code inside the host process. Install reviewed sources only.
Model weights are not included. A bounded local Krea 2 H4 lane verified execution and artifact
provenance, but blind scoring was waived; it does not establish image-quality parity or general
GPU compatibility. Stochastic sampling, resume behavior, and arbitrary-model compatibility are
not claimed.
