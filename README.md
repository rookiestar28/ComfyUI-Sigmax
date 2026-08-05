# ComfyUI-Sigmax
ComfyUI-Sigmax provides model-aware sigma schedules for ComfyUI, with verified Krea 2, Z-Image, FLUX.1-schnell, Qwen Image, SD3, AuraFlow v0.2, Lumina-Image 2.0, HunyuanImage 2.1, MiniMax H3 Base (qualification slice), Anima, Wan 2.1/2.2, and LTX profiles plus editing tools.
## Features

- Explicit model and variant selection with no silent generic fallback.
- Verified Krea 2, Z-Image, FLUX.1-schnell, Qwen Image, SD3, AuraFlow v0.2, Lumina-Image 2.0, HunyuanImage 2.1, MiniMax H3 Base qualification, Anima, Wan 2.1/2.2, and LTX recipes.
- Schedule slicing, concatenation, resampling, inspection, and comparison.
- Experimental Krea 2 `CONDITIONING` tap rebalancing for explicitly selected RAW or Turbo workflows, with fixed RMS preservation and no scheduler/model patching.
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
| MiniMax H3 Base | `Sigmax.MiniMaxH3SigmaScheduler` | Select `H3 Base FL2VA` or `H3 Base Ref2VA`; endpoint-inclusive video sigmas, 20 grid points by default; audio remapping remains model-owned |
| Anima Base v1.0 / Aesthetic / Turbo | `Sigmax.AnimaSigmaScheduler` | `Base`, 30-50 steps, default 50, CFG 4.5; `Aesthetic` uses the same recipe; `Turbo`, 8-12 steps, CFG 1.0; fixed shift 3.0 |
| Wan 2.1 T2V / I2V | `Sigmax.WanSigmaScheduler` | Select generation, task, source, and resolution explicitly; official T2V 50 steps (`5.0`), official I2V 480P/720P 40 steps (`3.0`/`5.0`) |
| Wan 2.2 TI2V 5B / A14B T2V/I2V | `Sigmax.WanSigmaScheduler` | Native or Diffusers-reference source lanes; TI2V 5B uses `5.0`; A14B T2V/I2V use `12.0`/`5.0` with caller-owned boundary metadata |
| LTXV 0.9.8 / LTX-2 19B / LTX-2.3 22B | `Sigmax.LTXSigmaScheduler` | Dev adaptive token shift (20/40/30 default steps) or explicit LTX-2/LTX-2.3 distilled Stage 1/2 vectors; generation and stage are explicit |

For normal use:

1. Add the scheduler node for the selected model.
2. Select the exact variant and keep `strict_official` enabled.
3. For Krea 2, enter the actual output width and height.
4. Connect the node's `SIGMAS` output directly to a custom-sampling path that accepts external sigmas.
5. Do not pass the result through another scheduler or apply another time shift.
6. Read `schedule_info` when checking the selected recipe, dimensions, step range, or warnings.

The Qwen Image node covers the original text-to-image family only: `Comfy Fixed` mirrors `1.15`, while `Diffusers Dynamic` requires `image_seq_len`; later variants and image-quality parity are out of scope.

The SD3 node covers only the original Stability AI SD3 Medium text-to-image schedule. Its two source-qualified modes preserve the publisher 1.0 versus pinned ComfyUI/Diffusers 3.0 conflict;
SD3.5, Turbo, ControlNet, model execution, and image quality are outside this support claim.

The AuraFlow node covers only original fal AuraFlow v0.2 with fixed ratio `1.73` and 50 steps; other versions, finetunes, dynamic shifts, execution, and image quality are outside scope.

The Lumina-Image 2.0 node covers only original Alpha-VLLM text-to-image with fixed ratio `6.0` and 50 steps; video, mGPT, editing paths, dynamic shifts, execution, and image quality are outside scope.

The HunyuanImage 2.1 node constructs schedule-only Base and Distilled direct-ratio paths (`5.0`/`4.0`). Base is the pinned ComfyUI-compatible lane; Distilled remains publisher-schedule-only until a native host path is qualified. Refiner, encoders, conditioning, weights, and image quality are outside scope.

The MiniMax H3 node is a narrow Base qualification slice: FL2VA/Ref2VA are explicit, only the Diffusers endpoint-inclusive video lane (shift `12.0`) is exposed, and native `simple` plus model-owned audio mapping (shift `3.0`) remain separate. The model-free workflow helper can preflight the generated graph against a caller-supplied ComfyUI `/object_info` schema, including Ref2VA autogrow inputs, without loading weights or submitting a prompt. Context-IR, Regenerate-2K, sparse attention, hosted/API behavior, weights, samplers, and quality claims are outside scope.

The Wan node constructs schedule-only unit-flow sigmas for the released 2.1/2.2 matrix; 2.1 I2V requires `480P` or `720P`, and 2.2 A14B boundaries are caller-owned metadata (never expert routing).
Diffusers-reference lanes describe scheduler math only; video execution, weights, and quality parity are outside scope.

The LTX node constructs schedule-only unit-flow sigmas for pinned LTXV 0.9.8, LTX-2 19B, and LTX-2.3 22B lanes. Dev mode derives one token-count shift; distilled modes use immutable
publisher vectors. It does not load video weights, run encoders, or claim video-quality parity.

`Sigmax.ModelAwareSigmaScheduler` can inspect a connected Krea 2 model, but weak evidence may
identify only the family; if `Auto` reports ambiguity, select `Turbo` or `RAW` explicitly.

For experimental conditioning, connect Krea 2 `CONDITIONING` to
`Sigmax.Krea2ConditioningRebalance`, select `RAW`/`Turbo` and a profile; it changes only the primary
tensor, preserves metadata, and makes no prompt-adherence or image-quality claim. Strength `0` is identity.

For a RAW checkpoint with a compatible RAW-to-Turbo model-difference LoRA, choose
`LoRA Experimental (RAW mu)` or `LoRA Experimental (Turbo mu)` and set the desired steps. The
node automatically forces `strict_official` off and disables that widget for either
Experimental selection. The former derives `mu` from width/height; the latter fixes `mu = 1.15`.
This path is experimental: the scheduler does not load or scale the LoRA, enforce CFG/sampler
settings, or claim that 12 steps is official. Apply the LoRA to RAW only; do not stack equivalent
RAW-to-Turbo LoRAs or apply one to the Turbo checkpoint.

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
Model weights are not included, and current validation does not claim real-model GPU image quality, stochastic sampling, resume behavior, or arbitrary-model compatibility.
