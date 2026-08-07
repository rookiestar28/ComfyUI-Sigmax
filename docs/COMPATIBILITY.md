# Compatibility

This page summarizes the supported user-facing boundary for tagged ComfyUI-Sigmax 1.0.0 and the
current unreleased development additions described below.

## Environment

| Component | Supported boundary |
| --- | --- |
| Python | 3.10 or newer |
| ComfyUI package requirement | 0.29.0 or newer |
| General validated host baseline | ComfyUI 0.29.0 |
| MiniMax H3 validated host baseline | ComfyUI 0.30.0 with the upstream H3 nodes |
| Operating systems covered by project gates | Windows and Linux/WSL |
| Mandatory additional Python packages | None |
| Host runtime dependency policy | Record the selected host's compatible package versions; current ComfyUI-recommended `comfy-aimdo` versions (including 0.4.13) are accepted without an exact-version gate |

A newer ComfyUI version may work, but is not automatically promoted to the validated host baseline; reproduce workflows on that baseline first after an update.

## Supported model profiles

| Profile | Supported schedule | User selection |
| --- | --- | --- |
| Krea 2 Turbo | Fixed-shift external sigma schedule, 8-step official recipe | Select `Turbo` explicitly when model evidence is ambiguous |
| Krea 2 RAW | Resolution-aware external sigma schedule, official 52-step and framework-reference 28-step recipes | Select `RAW` explicitly and provide the actual width and height |
| Z-Image Base | Fixed-ratio external sigma schedule, 28-50 steps | Select `Base` explicitly |
| Z-Image Turbo | Fixed-ratio external sigma schedule, 8-step official recipe | Select `Turbo` explicitly |
| FLUX.1-schnell | Unshifted external sigma schedule, 1-4 steps | Use the dedicated FLUX.1-schnell node |
| Original Qwen Image | ComfyUI fixed `1.15` shift, or Diffusers dynamic shift with explicit `image_seq_len`; original T2I only | Use `Sigmax.QwenImageSigmaScheduler`; select the mode explicitly |
| Original Stable Diffusion 3 | Original SD3 Medium T2I only; explicit publisher-reference `1.0` or ComfyUI/Diffusers fixed `3.0` shift | Use `Sigmax.SD3SigmaScheduler`; select the source mode explicitly |
| Original AuraFlow v0.2 | Fixed unit-flow `1.73` ratio shift, original 50-step recipe | Use `Sigmax.AuraFlowSigmaScheduler`; select `Official Fixed (1.73)` explicitly |
| Lumina-Image 2.0 | Fixed unit-flow `6.0` ratio shift, original 50-step recipe | Use `Sigmax.Lumina2SigmaScheduler`; select `Official Fixed (6.0)` explicitly |
| HunyuanImage 2.1 Base | Fixed unit-flow `5.0` ratio shift, official 50-step recipe; schedule-only | Use `Sigmax.HunyuanImage21SigmaScheduler`; select `Base (5.0)` explicitly |
| HunyuanImage 2.1 Distilled | Fixed unit-flow `4.0` ratio shift, official 8-step publisher recipe; native host path unqualified | Use `Sigmax.HunyuanImage21SigmaScheduler`; select `Distilled (4.0)` explicitly |
| MiniMax H3 Base FL2VA/Ref2VA | Diffusers endpoint-inclusive external video sigmas with shift `12.0`; model-owned audio mapping with shift `3.0`; post-v1.0.0 qualification slice | Use `Sigmax.MiniMaxH3SigmaScheduler` with ComfyUI's upstream `MiniMaxH3SigmaShift`; select FL2VA or Ref2VA explicitly |
| Anima Base / Aesthetic | Fixed unit-flow rational `3.0` shift, 30-50 step framework-reference recipe; schedule-only | Use `Sigmax.AnimaSigmaScheduler`; select `Base` or `Aesthetic` explicitly |
| Anima Turbo | Fixed unit-flow rational `3.0` shift, 8-12 step framework-reference recipe; schedule-only | Use `Sigmax.AnimaSigmaScheduler`; select `Turbo` explicitly |
| Wan 2.1 T2V | Source-qualified unit-flow direct-ratio shift (`5.0` official, `8.0` ComfyUI-native, `3.0` Diffusers reference), 50-step recipes | Use `Sigmax.WanSigmaScheduler`; select generation, task, source, and `None` resolution explicitly |
| Wan 2.1 I2V | Resolution-qualified official/Diffusers-reference paths: 480P `3.0` and 720P `5.0`, 40-step recipes | Use `Sigmax.WanSigmaScheduler`; select `I2V` and the actual `480P` or `720P` class |
| Wan 2.2 TI2V 5B | Native and Diffusers-reference unit-flow `5.0` shift, 50-step recipes | Use `Sigmax.WanSigmaScheduler`; select `TI2V` and the source explicitly |
| Wan 2.2 A14B T2V/I2V | Source-qualified unit-flow paths with caller-owned boundaries; native T2V/I2V `12.0`/`5.0`, Diffusers references `3.0` | Use `Sigmax.WanSigmaScheduler`; boundary output is metadata only and never routes experts |
| LTXV 0.9.8 / LTX-2 19B / LTX-2.3 22B | Dev token-count adaptive shift; LTX-2/LTX-2.3 distilled Stage 1/2 immutable vectors; schedule-only | Use `Sigmax.LTXSigmaScheduler`; select generation and stage explicitly |

The experimental `Sigmax.Krea2ConditioningRebalance` node accepts explicit RAW or Turbo
`CONDITIONING` tensors with shape `(batch, sequence, 30720)`. It preserves standard ComfyUI
metadata and performs no scheduler or model mutation. Its profiles are community-derived and
experimental; this boundary does not claim prompt adherence or image-quality improvement.

The generic advanced FlowMatch node constructs explicit schedule math only. It is experimental
and does not establish compatibility with an arbitrary model.

MiniMax H3 Base FL2VA/Ref2VA is an accepted post-v1.0.0 development profile on `dev`, not part
of the tagged 1.0.0 boundary. Public `steps` counts transitions and produces `steps + 1`
endpoint-inclusive video sigmas. The Sigmax scheduler owns that externally shifted video lane;
ComfyUI's upstream `MiniMaxH3SigmaShift` supplies the matching video/audio shifts to the model so
audio remapping remains model-owned. These are complementary responsibilities, not two schedule
transforms.

## Usage boundary

- Sigmax produces sigma schedules; it does not replace the numerical sampler.
- External Sigmax sigmas must not be shifted or scheduled a second time.
- Krea 2 `Auto` selection fails closed when evidence identifies the family but not RAW versus
  Turbo.
- Checkpoint header inspection does not load weights and cannot confirm a model variant from weak
  filename or structural evidence.
- Slicing, concatenating, or resampling a schedule changes its evidence status to modified.
- Model weights must be obtained separately under their own licenses and access terms.
- Qwen Image dynamic mode requires an explicit packed image sequence length; it never falls back
  to the fixed mode when that input is absent.
- SD3 source modes are intentionally non-composable. The publisher 1.0 and ComfyUI/Diffusers 3.0
  values are separate evidence lanes, not a silent default selection.
- AuraFlow v0.2 uses one fixed model-native ratio shift. Do not apply a second shift or pass
  already-shifted sigmas to the node.
- Lumina-Image 2.0 uses one fixed model-native ratio shift. Do not apply a second shift or pass
  already-shifted sigmas to the node.
- HunyuanImage 2.1 uses one direct-ratio shift per explicit variant. The Base lane is host-compatible
  at the pinned baseline; Distilled is schedule-only and must not be presented as native host parity.
- HunyuanImage 2.1 model weights remain under Tencent's community license and are not distributed by
  Sigmax; this package does not include model code, weights, encoders, or conditioning.
- MiniMax H3 workflows must keep the Sigmax external video schedule and upstream
  `MiniMaxH3SigmaShift` values aligned at `12.0`/`3.0`. Do not add `BasicScheduler`, apply another
  shift, or change only the model-side controls. The current Sigmax profile does not expose
  arbitrary H3 shift overrides.
- Anima applies one fixed rational `3.0` shift and rejects already-shifted composition. The node
  does not load Anima weights, run conditioning, or establish image-quality or prompt-adherence claims;
  Anima weight files remain under CircleStone Labs and applicable derivative licenses.
- Wan profiles keep ComfyUI-native, official-native, and Diffusers-reference shift ownership
  separate. A Wan 2.1 I2V resolution is mandatory, and unsupported derivatives fail closed.
- Wan 2.2 A14B boundaries are descriptive caller-owned split metadata; Sigmax does not select high
  or low experts, load video weights, patch the model, or implement a video sampler.
- Wan Diffusers-reference rows document scheduler sigma/timestep construction only; they do not
  claim UniPC solver parity when consumed by an Euler sampler.
- LTX generation and stage are explicit; Dev token shifting and distilled vectors are separate
  lanes. LTX profiles do not load video weights or claim video-quality parity.
- Host E2E receipts record ComfyUI, Python, PyTorch, and runtime package versions for diagnosis;
  historical parity pins are reproducibility metadata, not a blocker for a current recommended host.

## Not currently claimed

- General real-model GPU compatibility or image-quality parity. One bounded local Krea 2 H4
  execution/provenance lane completed, but blind scoring and threshold review were explicitly
  waived, so it does not support a quality or profile-promotion claim.
- Stochastic, resumable, or interrupted sampler state.
- General partial-denoise execution or advanced model-patch compatibility beyond the explicitly
  documented MiniMax H3 host workflow.
- Automatic compatibility with unlisted model families.
- Guaranteed compatibility with every future ComfyUI release.
- AuraFlow v0.1/v0.3, PonyFlow, Wan derivatives (FLF2V/VACE/Fun-Control and similar), community
  finetunes, or real-model image-quality parity.

If a workflow falls outside these boundaries, treat it as unvalidated rather than silently
substituting another profile or schedule.
