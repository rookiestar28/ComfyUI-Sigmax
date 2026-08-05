"""Pure native-ComfyUI workflow contracts for MiniMax H3 Base.

This module only constructs an API prompt.  It never probes the filesystem, loads a model, or
imports ComfyUI.  The graph keeps the external video sigma schedule separate from H3's
model-owned audio coordinate mapping and velocity correction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_DEFAULT_GRID_POINTS,
    MINIMAX_H3_MAX_GRID_POINTS,
    MINIMAX_H3_VIDEO_SHIFT,
)

MiniMaxH3PublicVariant = Literal["H3 Base FL2VA", "H3 Base Ref2VA"]
WorkflowPrompt = dict[str, dict[str, object]]

MINIMAX_H3_HOST_MIN_VERSION: Final = "0.30.0"
MINIMAX_H3_CANVAS_MULTIPLE: Final = 32
MINIMAX_H3_MAX_PIXELS: Final = 768 * 1344
MINIMAX_H3_MIN_FRAMES: Final = 5
MINIMAX_H3_MAX_FRAMES: Final = 3_600
MINIMAX_H3_MAX_REFERENCE_IMAGES: Final = 9
MINIMAX_H3_MAX_SEED: Final = 0xFFFF_FFFF_FFFF_FFFF
MINIMAX_H3_SAMPLER: Final = "euler"
MINIMAX_H3_REFERENCE_IMAGE_SIZE: Final = "match"
MINIMAX_H3_TEXT_ENCODER: Final = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
MINIMAX_H3_VIDEO_VAE: Final = "minimax_h3_video_vae_fp16.safetensors"
MINIMAX_H3_AUDIO_VAE: Final = "minimax_h3_audio_vae_fp32.safetensors"

_FL2VA: Final[MiniMaxH3PublicVariant] = "H3 Base FL2VA"
_REF2VA: Final[MiniMaxH3PublicVariant] = "H3 Base Ref2VA"
_VARIANTS: Final = (_FL2VA, _REF2VA)
_WINDOWS_DRIVE: Final = re.compile(r"^[A-Za-z]:[\\/]")

_MODEL_ID = "1"
_CLIP_ID = "2"
_VIDEO_VAE_ID = "3"
_AUDIO_VAE_ID = "4"
_SHIFT_ID = "5"
_CONDITION_ID = "6"
_SCHEDULE_ID = "7"
_SAMPLER_ID = "8"
_GUIDER_ID = "9"
_NOISE_ID = "10"
_SAMPLE_ID = "11"
_SPLIT_ID = "12"
_VIDEO_DECODE_ID = "13"
_AUDIO_DECODE_ID = "14"
_VIDEO_PREVIEW_ID = "15"
_AUDIO_PREVIEW_ID = "16"
_FIRST_DYNAMIC_ID = 17


def _require_variant(value: object) -> MiniMaxH3PublicVariant:
    if value not in _VARIANTS:
        raise ScheduleContractError("MiniMax H3 workflow variant must be selected explicitly")
    return value


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScheduleContractError(f"MiniMax H3 workflow {field} must be non-empty text")
    if "\x00" in value:
        raise ScheduleContractError(f"MiniMax H3 workflow {field} contains a NUL byte")
    return value


def _require_host_relative_name(
    value: object,
    *,
    field: str,
    suffix: str | None = None,
) -> str:
    name = _require_text(value, field=field)
    normalized = name.replace("\\", "/")
    if (
        normalized.startswith("/")
        or _WINDOWS_DRIVE.match(name) is not None
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ScheduleContractError(f"MiniMax H3 workflow {field} must be host-relative")
    if suffix is not None and not normalized.casefold().endswith(suffix.casefold()):
        raise ScheduleContractError(
            f"MiniMax H3 workflow {field} must use the {suffix} artifact suffix"
        )
    return name


def _require_int(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum or value > maximum:
        raise ScheduleContractError(
            f"MiniMax H3 workflow {field} must be an integer in [{minimum}, {maximum}]"
        )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3ModelFiles:
    """Host-relative H3 artifact names; no artifact is inspected by this contract."""

    diffusion_model: str
    text_encoder: str = MINIMAX_H3_TEXT_ENCODER
    video_vae: str = MINIMAX_H3_VIDEO_VAE
    audio_vae: str = MINIMAX_H3_AUDIO_VAE

    def __post_init__(self) -> None:
        _require_host_relative_name(
            self.diffusion_model, field="diffusion_model", suffix=".safetensors"
        )
        _require_host_relative_name(self.text_encoder, field="text_encoder", suffix=".safetensors")
        _require_host_relative_name(self.video_vae, field="video_vae", suffix=".safetensors")
        _require_host_relative_name(self.audio_vae, field="audio_vae", suffix=".safetensors")


def default_minimax_h3_model_files(
    variant: MiniMaxH3PublicVariant,
) -> MiniMaxH3ModelFiles:
    """Return the ComfyUI-format BF16 diffusion filename for one explicit H3 variant."""

    selected = _require_variant(variant)
    stem = "minimax_h3_fl2va" if selected == _FL2VA else "minimax_h3_ref2va"
    return MiniMaxH3ModelFiles(diffusion_model=f"H3/{stem}_bf16.safetensors")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3WorkflowSpec:
    """Validated inputs for a model-backed H3 host graph, without touching model state."""

    variant: MiniMaxH3PublicVariant
    prompt: str
    width: int = 1344
    height: int = 768
    length: int = 124
    grid_points: int = MINIMAX_H3_DEFAULT_GRID_POINTS
    seed: int = 0
    sampler_name: str = MINIMAX_H3_SAMPLER
    ref_image_size: str = MINIMAX_H3_REFERENCE_IMAGE_SIZE
    first_frame: str | None = None
    last_frame: str | None = None
    reference_images: tuple[str, ...] = ()
    model_files: MiniMaxH3ModelFiles | None = None

    def __post_init__(self) -> None:
        selected = _require_variant(self.variant)
        _require_text(self.prompt, field="prompt")
        width = _require_int(
            self.width,
            field="width",
            minimum=MINIMAX_H3_CANVAS_MULTIPLE,
            maximum=16_384,
        )
        height = _require_int(
            self.height,
            field="height",
            minimum=MINIMAX_H3_CANVAS_MULTIPLE,
            maximum=16_384,
        )
        if width % MINIMAX_H3_CANVAS_MULTIPLE or height % MINIMAX_H3_CANVAS_MULTIPLE:
            raise ScheduleContractError(
                "MiniMax H3 workflow width and height must be multiples of 32"
            )
        if width * height > MINIMAX_H3_MAX_PIXELS:
            raise ScheduleContractError(
                "MiniMax H3 workflow canvas exceeds the native 768*1344 pixel envelope"
            )
        length = _require_int(
            self.length,
            field="length",
            minimum=MINIMAX_H3_MIN_FRAMES,
            maximum=MINIMAX_H3_MAX_FRAMES,
        )
        if (length - MINIMAX_H3_MIN_FRAMES) % 17:
            raise ScheduleContractError(
                "MiniMax H3 workflow length must follow the 17k+5 frame grid at 24 fps"
            )
        _require_int(
            self.grid_points,
            field="grid_points",
            minimum=2,
            maximum=MINIMAX_H3_MAX_GRID_POINTS,
        )
        _require_int(self.seed, field="seed", minimum=0, maximum=MINIMAX_H3_MAX_SEED)
        if self.sampler_name != MINIMAX_H3_SAMPLER:
            raise ScheduleContractError(
                "MiniMax H3 host workflow currently exposes the pinned euler sampler only"
            )
        if self.ref_image_size not in {"match", "max"}:
            raise ScheduleContractError("MiniMax H3 ref_image_size must be match or max")
        if not isinstance(self.reference_images, tuple):
            raise ScheduleContractError("MiniMax H3 reference_images must be an immutable tuple")
        if selected == _REF2VA:
            if not 1 <= len(self.reference_images) <= MINIMAX_H3_MAX_REFERENCE_IMAGES:
                raise ScheduleContractError(
                    "MiniMax H3 Ref2VA requires one to nine explicit reference images"
                )
            if self.first_frame is not None or self.last_frame is not None:
                raise ScheduleContractError(
                    "MiniMax H3 Ref2VA cannot combine keyframes with reference images"
                )
        elif self.reference_images:
            raise ScheduleContractError(
                "MiniMax H3 FL2VA cannot receive Ref2VA reference image inputs"
            )
        for field, value in (("first_frame", self.first_frame), ("last_frame", self.last_frame)):
            if value is not None:
                _require_host_relative_name(value, field=field)
        for index, value in enumerate(self.reference_images):
            _require_host_relative_name(value, field=f"reference_images[{index}]")
        if self.model_files is not None and not isinstance(self.model_files, MiniMaxH3ModelFiles):
            raise ScheduleContractError("MiniMax H3 model_files must be MiniMaxH3ModelFiles")
        if self.model_files is not None:
            diffusion_name = self.model_files.diffusion_model.replace("\\", "/").casefold()
            if selected == _FL2VA and "ref2va" in diffusion_name and "fl2va" not in diffusion_name:
                raise ScheduleContractError(
                    "MiniMax H3 FL2VA workflow received a contradictory Ref2VA diffusion artifact"
                )
            if selected == _REF2VA and "fl2va" in diffusion_name and "ref2va" not in diffusion_name:
                raise ScheduleContractError(
                    "MiniMax H3 Ref2VA workflow received a contradictory FL2VA diffusion artifact"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3HostWorkflowContract:
    """Machine-readable ownership statement attached to a generated graph."""

    schema: str
    host_min_version: str
    variant: MiniMaxH3PublicVariant
    schedule_node_id: str
    native_shift_node_id: str
    sampler_node_id: str
    schedule_ownership: str
    audio_ownership: str
    external_video_shift_applied_once: bool
    external_audio_schedule: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3HostWorkflow:
    """Generated API graph plus the non-executable ownership contract."""

    spec: MiniMaxH3WorkflowSpec
    model_files: MiniMaxH3ModelFiles
    prompt: WorkflowPrompt
    contract: MiniMaxH3HostWorkflowContract


def _link(node_id: str, output: int = 0) -> list[object]:
    return [node_id, output]


def _node(class_type: str, inputs: dict[str, object]) -> dict[str, object]:
    return {"class_type": class_type, "inputs": inputs}


def _load_image_node(node_id: str, filename: str) -> dict[str, object]:
    return _node("LoadImage", {"image": filename})


def _condition_node(
    spec: MiniMaxH3WorkflowSpec,
    *,
    next_node_id: int,
) -> tuple[dict[str, object], dict[str, dict[str, object]], int]:
    nodes: dict[str, dict[str, object]] = {}
    inputs: dict[str, object]
    if spec.variant == _FL2VA:
        inputs = {
            "clip": _link(_CLIP_ID),
            "vae": _link(_VIDEO_VAE_ID),
            "prompt": spec.prompt,
            "width": spec.width,
            "height": spec.height,
            "length": spec.length,
        }
        for key, filename in (("first_frame", spec.first_frame), ("last_frame", spec.last_frame)):
            if filename is not None:
                node_id = str(next_node_id)
                nodes[node_id] = _load_image_node(node_id, filename)
                inputs[key] = _link(node_id)
                next_node_id += 1
        return _node("MiniMaxH3ImageToVideo", inputs), nodes, next_node_id

    inputs = {
        "clip": _link(_CLIP_ID),
        "vae": _link(_VIDEO_VAE_ID),
        "audio_vae": _link(_AUDIO_VAE_ID),
        "prompt": spec.prompt,
        "width": spec.width,
        "height": spec.height,
        "length": spec.length,
        "ref_image_size": spec.ref_image_size,
    }
    for index, filename in enumerate(spec.reference_images):
        node_id = str(next_node_id)
        nodes[node_id] = _load_image_node(node_id, filename)
        inputs[f"ref_images.ref_image_{index}"] = _link(node_id)
        next_node_id += 1
    return _node("MiniMaxH3ReferenceToVideo", inputs), nodes, next_node_id


def build_minimax_h3_host_workflow(spec: MiniMaxH3WorkflowSpec) -> MiniMaxH3HostWorkflow:
    """Build a deterministic native H3 API graph without loading or inspecting any artifact."""

    if not isinstance(spec, MiniMaxH3WorkflowSpec):
        raise ScheduleContractError("MiniMax H3 host workflow requires MiniMaxH3WorkflowSpec")
    files = spec.model_files or default_minimax_h3_model_files(spec.variant)
    prompt: WorkflowPrompt = {
        _MODEL_ID: _node(
            "UNETLoader",
            {"unet_name": files.diffusion_model, "weight_dtype": "default"},
        ),
        _CLIP_ID: _node(
            "CLIPLoader",
            {"clip_name": files.text_encoder, "type": "minimax"},
        ),
        _VIDEO_VAE_ID: _node("VAELoader", {"vae_name": files.video_vae}),
        _AUDIO_VAE_ID: _node("VAELoader", {"vae_name": files.audio_vae}),
        _SHIFT_ID: _node(
            "MiniMaxH3SigmaShift",
            {
                "model": _link(_MODEL_ID),
                "shift_video": MINIMAX_H3_VIDEO_SHIFT,
                "shift_audio": MINIMAX_H3_AUDIO_SHIFT,
            },
        ),
    }
    condition, input_nodes, next_node_id = _condition_node(
        spec,
        next_node_id=_FIRST_DYNAMIC_ID,
    )
    prompt[_CONDITION_ID] = condition
    prompt.update(input_nodes)
    prompt[_SCHEDULE_ID] = _node(
        "Sigmax.MiniMaxH3SigmaScheduler",
        {
            "variant": spec.variant,
            "grid_points": spec.grid_points,
            "start_step": 0,
            "end_step": -1,
            "already_shifted": False,
        },
    )
    prompt[_SAMPLER_ID] = _node("KSamplerSelect", {"sampler_name": spec.sampler_name})
    prompt[_GUIDER_ID] = _node(
        "BasicGuider",
        {"model": _link(_SHIFT_ID), "conditioning": _link(_CONDITION_ID)},
    )
    prompt[_NOISE_ID] = _node("RandomNoise", {"noise_seed": spec.seed})
    prompt[_SAMPLE_ID] = _node(
        "SamplerCustomAdvanced",
        {
            "noise": _link(_NOISE_ID),
            "guider": _link(_GUIDER_ID),
            "sampler": _link(_SAMPLER_ID),
            "sigmas": _link(_SCHEDULE_ID),
            "latent_image": _link(_CONDITION_ID, 1),
        },
    )
    prompt[_SPLIT_ID] = _node("LTXVSeparateAVLatent", {"av_latent": _link(_SAMPLE_ID)})
    prompt[_VIDEO_DECODE_ID] = _node(
        "VAEDecode",
        {"samples": _link(_SPLIT_ID), "vae": _link(_VIDEO_VAE_ID)},
    )
    prompt[_AUDIO_DECODE_ID] = _node(
        "VAEDecodeAudio",
        {"samples": _link(_SPLIT_ID, 1), "vae": _link(_AUDIO_VAE_ID)},
    )
    prompt[_VIDEO_PREVIEW_ID] = _node("PreviewImage", {"images": _link(_VIDEO_DECODE_ID)})
    prompt[_AUDIO_PREVIEW_ID] = _node("PreviewAny", {"source": _link(_AUDIO_DECODE_ID)})
    # Keep the generated graph's key ordering deterministic even when dynamic LoadImage nodes exist.
    prompt = {node_id: prompt[node_id] for node_id in sorted(prompt, key=lambda value: int(value))}
    contract = MiniMaxH3HostWorkflowContract(
        schema="sigmax.minimax-h3-host-workflow/1",
        host_min_version=MINIMAX_H3_HOST_MIN_VERSION,
        variant=spec.variant,
        schedule_node_id=_SCHEDULE_ID,
        native_shift_node_id=_SHIFT_ID,
        sampler_node_id=_SAMPLE_ID,
        schedule_ownership="external_video_only",
        audio_ownership="model_native",
        external_video_shift_applied_once=True,
        external_audio_schedule=False,
    )
    # next_node_id is intentionally calculated above so malformed future dynamic-input changes
    # remain visible to a reviewer; it is not a hidden execution side effect.
    del next_node_id
    return MiniMaxH3HostWorkflow(
        spec=spec,
        model_files=files,
        prompt=prompt,
        contract=contract,
    )


def build_minimax_h3_host_workflow_prompt(spec: MiniMaxH3WorkflowSpec) -> WorkflowPrompt:
    """Return only the JSON-compatible API prompt for one validated H3 workflow spec."""

    return build_minimax_h3_host_workflow(spec).prompt


__all__ = [
    "MINIMAX_H3_AUDIO_VAE",
    "MINIMAX_H3_CANVAS_MULTIPLE",
    "MINIMAX_H3_HOST_MIN_VERSION",
    "MINIMAX_H3_MAX_FRAMES",
    "MINIMAX_H3_MAX_PIXELS",
    "MINIMAX_H3_MAX_REFERENCE_IMAGES",
    "MINIMAX_H3_MAX_SEED",
    "MINIMAX_H3_REFERENCE_IMAGE_SIZE",
    "MINIMAX_H3_SAMPLER",
    "MINIMAX_H3_TEXT_ENCODER",
    "MINIMAX_H3_VIDEO_VAE",
    "MiniMaxH3HostWorkflow",
    "MiniMaxH3HostWorkflowContract",
    "MiniMaxH3ModelFiles",
    "MiniMaxH3PublicVariant",
    "MiniMaxH3WorkflowSpec",
    "WorkflowPrompt",
    "build_minimax_h3_host_workflow",
    "build_minimax_h3_host_workflow_prompt",
    "default_minimax_h3_model_files",
]
