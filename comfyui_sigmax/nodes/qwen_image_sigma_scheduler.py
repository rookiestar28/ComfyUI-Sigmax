"""Thin explicit original Qwen Image SIGMAS node."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Final

from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    numerical_fingerprint,
    slice_step_range,
    validate_sigma_schedule,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint
from comfyui_sigmax.profiles.qwen_image import (
    QWEN_IMAGE_COMFY_FIXED_PROFILE,
    QWEN_IMAGE_DIFFUSERS_DYNAMIC_PROFILE,
    QwenImageShiftMode,
    build_qwen_image_schedule,
    calculate_qwen_image_mu,
)

QWEN_IMAGE_SIGMA_NODE_ID: Final = "Sigmax.QwenImageSigmaScheduler"
QWEN_IMAGE_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.qwen-image-sigma-node/1"
_MAX_STEPS: Final = 10_000
_MAX_IMAGE_SEQ_LEN: Final = 1_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenImageSigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    mode: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.mode not in {"Comfy Fixed", "Diffusers Dynamic"}:
            raise ScheduleContractError("Qwen Image node mode is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("Qwen Image node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("Qwen Image node result is incomplete")


def _mode(value: object) -> tuple[str, QwenImageShiftMode]:
    if value == "Comfy Fixed":
        return "Comfy Fixed", QwenImageShiftMode.COMFY_FIXED
    if value == "Diffusers Dynamic":
        return "Diffusers Dynamic", QwenImageShiftMode.DIFFUSERS_DYNAMIC
    raise ScheduleContractError("mode must be Comfy Fixed or Diffusers Dynamic")


def _positive_steps(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    return value


def _sequence_length(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= _MAX_IMAGE_SEQ_LEN
    ):
        raise ScheduleContractError(
            f"image_seq_len must be an integer between 0 and {_MAX_IMAGE_SEQ_LEN}"
        )
    return value


def _slice_bounds(
    *, start_step: object, end_step: object, available_steps: int
) -> tuple[int, int | None]:
    if not isinstance(start_step, int) or isinstance(start_step, bool) or start_step < 0:
        raise ScheduleContractError("start_step must be a non-negative integer")
    if not isinstance(end_step, int) or isinstance(end_step, bool) or end_step < -1:
        raise ScheduleContractError("end_step must be -1 or a non-negative integer")
    end = None if end_step == -1 else end_step
    if start_step >= available_steps:
        raise ScheduleContractError("start_step must be below the constructed step count")
    if end is not None and (end <= start_step or end > available_steps):
        raise ScheduleContractError(
            "end_step must exceed start_step and not exceed the constructed step count"
        )
    return start_step, end


def _canonical_info(value: dict[str, object]) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError(
            "Qwen Image schedule information must be canonical JSON"
        ) from exc


def build_qwen_image_sigma_schedule(
    *,
    mode: object,
    steps: object,
    image_seq_len: object,
    strict_official: object,
    start_step: object,
    end_step: object,
) -> QwenImageSigmaNodeResult:
    """Build and slice one original-Qwen schedule without host imports."""

    public_mode, internal_mode = _mode(mode)
    count = _positive_steps(steps)
    sequence_length = _sequence_length(image_seq_len)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")
    complete = build_qwen_image_schedule(
        mode=internal_mode,
        steps=count,
        image_seq_len=sequence_length,
        strict_official=strict_official,
    )
    start, end = _slice_bounds(
        start_step=start_step,
        end_step=end_step,
        available_steps=complete.effective_inputs.steps,
    )
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    profile = (
        QWEN_IMAGE_COMFY_FIXED_PROFILE
        if internal_mode is QwenImageShiftMode.COMFY_FIXED
        else QWEN_IMAGE_DIFFUSERS_DYNAMIC_PROFILE
    )
    evidence = complete.request.provenance.evidence
    recipe = (
        profile.profile_id
        if evidence is not None and evidence.value != "modified"
        else f"{profile.profile_id}.modified-{count}"
    )
    effective_end = count if end is None else end
    shift: dict[str, object]
    if internal_mode is QwenImageShiftMode.COMFY_FIXED:
        shift = {"dynamic": False, "kind": "fixed_direct_ratio", "ratio": 1.15}
    else:
        mu = calculate_qwen_image_mu(sequence_length)
        shift = {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "dynamic": True,
            "image_seq_len": sequence_length,
            "kind": "exponential_mu",
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "mu": mu,
        }
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "guidance": {"host_true_cfg": 4.0, "model_guidance": 0.0},
        "profile": {
            "evidence": evidence.value,
            "id": profile.profile_id,
            "recipe": recipe,
            "variant": "original",
            "version": profile.profile_version,
        },
        "schema": QWEN_IMAGE_SIGMA_NODE_SCHEMA_ID,
        "shift": shift,
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_official": strict_official,
        "warnings": list(complete.warnings),
    }
    return QwenImageSigmaNodeResult(
        mode=public_mode,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_qwen_image_sigma_output_info(
    result: QwenImageSigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to actual host tensor values."""

    if not isinstance(result, QwenImageSigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("Qwen Image output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("Qwen Image schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class QwenImageSigmaScheduler:
    """Construct explicit original-Qwen external sigmas without model patching."""

    DESCRIPTION = "Builds an explicit original Qwen Image fixed or dynamic sigma schedule."
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False
    EXPERIMENTAL = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "mode": (("Comfy Fixed", "Diffusers Dynamic"),),
                "steps": ("INT", {"default": 50, "min": 1, "max": _MAX_STEPS, "step": 1}),
                "image_seq_len": (
                    "INT",
                    {"default": 0, "min": 0, "max": _MAX_IMAGE_SEQ_LEN, "step": 1},
                ),
                "strict_official": ("BOOLEAN", {"default": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS - 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": _MAX_STEPS}),
            }
        }

    def build(
        self,
        mode: object,
        steps: object,
        image_seq_len: object,
        strict_official: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        result = build_qwen_image_sigma_schedule(
            mode=mode,
            steps=steps,
            image_seq_len=image_seq_len,
            strict_official=strict_official,
            start_step=start_step,
            end_step=end_step,
        )
        try:
            torch = importlib.import_module("torch")
            float_tensor = torch.__dict__["FloatTensor"]
        except (ImportError, KeyError) as exc:
            # CRITICAL: keep Torch execution-only; package imports remain dependency-free.
            raise RuntimeError("ComfyUI host execution requires Torch FloatTensor support") from exc
        tensor = float_tensor(result.sigmas)
        try:
            host_values = tuple(float(value) for value in tensor.tolist())
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("ComfyUI host tensor must expose numeric tolist output") from exc
        return tensor, bind_qwen_image_sigma_output_info(result, output_sigmas=host_values)


__all__ = [
    "QWEN_IMAGE_SIGMA_NODE_ID",
    "QWEN_IMAGE_SIGMA_NODE_SCHEMA_ID",
    "QwenImageSigmaNodeResult",
    "QwenImageSigmaScheduler",
    "bind_qwen_image_sigma_output_info",
    "build_qwen_image_sigma_schedule",
]
