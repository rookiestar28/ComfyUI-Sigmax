# Compatibility

This page summarizes the supported user-facing boundary for tagged ComfyUI-Sigmax 1.0.0 and the
current 1.0.2 source-tree additions described below. The 1.0.2 package version is present in the
current tree; it is not a claim that a corresponding public tag or Registry publication exists.

## Environment

| Component | Supported boundary |
| --- | --- |
| Python | 3.10 or newer |
| ComfyUI package requirement | 0.29.0 or newer |
| General validated host baseline | ComfyUI 0.29.0 (pinned known-good lane) |
| Current Wan qualification lane | ComfyUI 0.31.0 on Windows (model-free H1/H2 registration, schema, schedule, and metadata checks) |
| MiniMax H3 qualified host roles | ComfyUI 0.30.0 and 0.32.0 with the upstream H3 nodes; the complete ten-scheduler model-free matrix is validated on both exact host roles |
| Operating systems covered by project gates | Windows and Linux/WSL |
| Mandatory additional Python packages | None |
| Host runtime dependency policy | Record the selected host's compatible package versions; current ComfyUI-recommended `comfy-aimdo` versions (including 0.4.13) are accepted without an exact-version gate |

A newer ComfyUI version may work, but is not automatically promoted to the validated host baseline; reproduce workflows on that baseline first after an update.

The current ComfyUI 0.31.0 lane is limited to model-free Wan qualification and does not establish
real-model execution or video-quality parity.

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
| MiniMax H3 Base FL2VA/Ref2VA | Pure endpoint-inclusive `h3_endpoint` default plus nine experimental ComfyUI-native scheduler choices; optional community Turbo recipes are readiness-only; Base video/audio shifts remain `12.0`/`3.0` | Use `Sigmax.MiniMaxH3SigmaScheduler`; native choices require the H3 `MODEL` after upstream `MiniMaxH3SigmaShift`; Turbo requires an exact recipe selector |
| Anima Base / Aesthetic | Fixed unit-flow rational `3.0` shift, 30-50 step framework-reference recipe; schedule-only | Use `Sigmax.AnimaSigmaScheduler`; select `Base` or `Aesthetic` explicitly |
| Anima Turbo | Fixed unit-flow rational `3.0` shift, 8-12 step framework-reference recipe; schedule-only | Use `Sigmax.AnimaSigmaScheduler`; select `Turbo` explicitly |
| Wan 2.1 T2V | Source-qualified unit-flow direct-ratio shift (`5.0` official, `8.0` ComfyUI-native, `3.0` Diffusers reference), 50-step recipes | Use `Sigmax.WanSigmaScheduler`; select generation, task, source, and `None` resolution explicitly |
| Wan 2.1 I2V | Resolution-qualified official/Diffusers-reference paths: 480P `3.0` and 720P `5.0`, 40-step recipes | Use `Sigmax.WanSigmaScheduler`; select `I2V` and the actual `480P` or `720P` class |
| Wan 2.2 TI2V 5B | Native and Diffusers-reference unit-flow `5.0` shift, 50-step recipes | Use `Sigmax.WanSigmaScheduler`; select `TI2V` and the source explicitly |
| Wan 2.2 A14B T2V/I2V | Source-qualified unit-flow paths with caller-owned boundaries; native T2V/I2V `12.0`/`5.0`, Diffusers references `3.0` | Use `Sigmax.WanSigmaScheduler`; boundary output is metadata only and never routes experts |
| Wan 2.1 FLF2V 14B 720P | Official-native unit-flow `16.0` shift, 50-step recipe | Use `Sigmax.WanSigmaScheduler`; select `FLF2V`, `Official native`, and `720P` explicitly |
| Wan 2.1 VACE 1.3B / 14B | Official-native unit-flow `16.0` shift, 50-step recipes | Use `Sigmax.WanSigmaScheduler`; select the VACE model size and `Official native` explicitly |
| Wan 2.2 S2V 14B | Official-native unit-flow `3.0` shift, 40-step recipe | Use `Sigmax.WanSigmaScheduler`; select `S2V` and `Official native` explicitly |
| Wan 2.2 Animate 14B | Official-native unit-flow `5.0` shift, 20-step recipe | Use `Sigmax.WanSigmaScheduler`; select `Animate` and `Official native` explicitly |
| Wan Animate 2 Base / Distilled 14B | Official-native unit-flow `5.0` shift, 40-step Base or 10-step Distilled recipes | Use `Sigmax.WanSigmaScheduler`; select `Wan Animate 2` and the explicit Base/Distilled task |
| LTXV 0.9.8 / LTX-2 19B / LTX-2.3 22B | Dev token-count adaptive shift; LTX-2/LTX-2.3 distilled Stage 1/2 immutable vectors; schedule-only | Use `Sigmax.LTXSigmaScheduler`; select generation and stage explicitly |

The experimental `Sigmax.Krea2ConditioningRebalance` node accepts explicit RAW or Turbo
`CONDITIONING` tensors with shape `(batch, sequence, 30720)`. It preserves standard ComfyUI
metadata and performs no scheduler or model mutation. Its profiles are community-derived and
experimental; this boundary does not claim prompt adherence or image-quality improvement.

The generic advanced FlowMatch node constructs explicit schedule math only. It is experimental
and does not establish compatibility with an arbitrary model.

MiniMax H3 Base FL2VA/Ref2VA is an accepted post-v1.0.0 profile in the current 1.0.2 source
tree, not part of the tagged 1.0.0 boundary. Public `steps` counts transitions and produces `steps + 1`
endpoint-inclusive video sigmas. The Sigmax scheduler owns that externally shifted video lane;
ComfyUI's upstream `MiniMaxH3SigmaShift` supplies the matching video/audio shifts to the model so
audio remapping remains model-owned. These are complementary responsibilities, not two schedule
transforms.

The same node now exposes a BasicScheduler-style selector with the fixed order `h3_endpoint`,
`simple`, `sgm_uniform`, `karras`, `exponential`, `ddim_uniform`, `beta`, `normal`,
`linear_quadratic`, and `kl_optimal`. `h3_endpoint` preserves the dependency-free compatibility
path. The other nine values delegate to the installed ComfyUI scheduler using a validated,
already-shifted H3 model and are explicitly experimental. This support means that the dispatch
path is executable; it is not an official MiniMax recommendation and makes no image-quality,
speed, memory, NFE, or acceleration claim. Full cross-host validation of every native choice is a
completed development check on the exact 0.30.0 and 0.32.0 host roles; this remains functional
experimental evidence, not a quality recommendation.

The optional `turbo` selector exposes four source-qualified community recipes: 544p FL2VA at 4 or
8 NFE, 768p FL2VA at 4 NFE, and 544p Ref2VA at 4 NFE. The 544p recipes use video/audio shifts
`12.0`/`3.0`; the 768p recipe uses `6.0`/`3.0`. The selector constructs recipe-owned sigmas and
readiness receipts only. It does not load or patch a LoRA, attention backend, or model, and it
does not claim an official MiniMax method, quality, speed, memory, NFE, or acceleration result.

The current source tree also contains the dependency-free M5-02 sampler-state schema-v1 contract.
It represents a sampler capability declaration, scheduler/begin cursor, solver order, timestep
spacing, random-source ownership, optional per-token time, requested/effective counts, immutable
step history, lifecycle status, and exact execution-receipt binding. This is a pure portable
contract and has no production ComfyUI node.

The source tree adds internal, experimental deterministic and stochastic Flow Euler controllers
over that state contract. The deterministic path supports full, partial, interrupt-boundary, and
in-process resume probes; the stochastic path preserves the pinned Diffusers v0.39.0 expression
order and parity boundary. Both keep the Torch/Comfy adapter optional and remain model-free
contracts: they do not add a public `SAMPLER` node, replace the native `euler` workflow path,
persist latent/RNG state, or claim real-model quality or acceleration.

## Usage boundary

- Public Sigmax nodes produce sigma schedules; they do not replace the workflow's numerical
  sampler. The internal experimental M5-03 controller is not registered as a public node.
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
- MiniMax H3 `h3_endpoint` workflows use the Sigmax external video schedule without a `MODEL`
  input. For any of the nine native choices, connect the model after upstream
  `MiniMaxH3SigmaShift`; keep its video/audio values aligned at Base `12.0`/`3.0` or the exact
  selected experimental Turbo recipe values. Do not add a separate `BasicScheduler`, apply
  another shift, or change only the model-side controls. The node fails closed for missing or
  incompatible models and does not expose arbitrary H3 shift overrides.
- Anima applies one fixed rational `3.0` shift and rejects already-shifted composition. The node
  does not load Anima weights, run conditioning, or establish image-quality or prompt-adherence claims;
  Anima weight files remain under CircleStone Labs and applicable derivative licenses.
- Wan profiles keep ComfyUI-native, official-native, and Diffusers-reference shift ownership
  separate. A Wan 2.1 I2V resolution is mandatory, and unsupported derivatives fail closed.
  The released FLF2V, VACE, S2V, Wan 2.2 Animate, and Wan Animate 2 rows are official-native
  schedule lanes only; they do not imply model loading, conditioning, expert routing, or quality
  parity.
- Wan 2.2 A14B boundaries are descriptive caller-owned split metadata; Sigmax does not select high
  or low experts, load video weights, patch the model, or implement a video sampler.
- Automatic Wan 2.2 A14B expert dispatch and controlled shift/quality experiments remain outside
  the current supported schedule-only boundary.
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
- Executed model-backed stochastic sampling, persisted latent/RNG state, host interruption
  plumbing, or cross-process resume. M5-03/M5-04 prove only model-free state/controller and
  Diffusers-expression parity boundaries over supplied state.
- General partial-denoise execution or advanced model-patch compatibility beyond the explicitly
  documented MiniMax H3 host workflow.
- Automatic compatibility with unlisted model families.
- Guaranteed compatibility with every future ComfyUI release.
- AuraFlow v0.1/v0.3, PonyFlow, unsupported Wan derivatives (including Fun-Control and community
  finetunes), or real-model image-quality parity.

If a workflow falls outside these boundaries, treat it as unvalidated rather than silently
substituting another profile or schedule.
