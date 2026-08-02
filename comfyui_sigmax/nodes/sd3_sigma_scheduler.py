"""Thin explicit original Stable Diffusion 3 SIGMAS node."""

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
from comfyui_sigmax.profiles.sd3 import (
    SD3_COMFY_DIFFUSERS_PROFILE,
    SD3_PUBLISHER_REFERENCE_PROFILE,
    SD3ShiftMode,
    build_sd3_schedule,
)

SD3_SIGMA_NODE_ID: Final = "Sigmax.SD3SigmaScheduler"
SD3_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.sd3-sigma-node/1"
_MAX_STEPS: Final = 10_000
_PUBLISHER_MODE: Final = "Publisher Reference (1.0)"
_FRAMEWORK_MODE: Final = "Comfy/Diffusers Fixed (3.0)"
_MODES: Final = (_PUBLISHER_MODE, _FRAMEWORK_MODE)


@dataclass(frozen=True, slots=True, kw_only=True)
class SD3SigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    mode: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ScheduleContractError("SD3 node mode is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("SD3 node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("SD3 node result is incomplete")


def _mode(value: object) -> tuple[str, SD3ShiftMode]:
    if value == _PUBLISHER_MODE:
        return _PUBLISHER_MODE, SD3ShiftMode.PUBLISHER_REFERENCE
    if value == _FRAMEWORK_MODE:
        return _FRAMEWORK_MODE, SD3ShiftMode.COMFY_DIFFUSERS_FIXED
    raise ScheduleContractError(
        "mode must be Publisher Reference (1.0) or Comfy/Diffusers Fixed (3.0)"
    )


def _positive_steps(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
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
        raise ScheduleContractError("SD3 schedule information must be canonical JSON") from exc


def build_sd3_sigma_schedule(
    *,
    mode: object,
    steps: object,
    strict_source: object,
    start_step: object,
    end_step: object,
    already_shifted: object = False,
) -> SD3SigmaNodeResult:
    """Build and slice one explicit original-SD3 schedule without host imports."""

    public_mode, internal_mode = _mode(mode)
    count = _positive_steps(steps)
    if not isinstance(strict_source, bool):
        raise ScheduleContractError("strict_source must be boolean")
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    complete = build_sd3_schedule(
        mode=internal_mode,
        steps=count,
        strict_source=strict_source,
        already_shifted=already_shifted,
    )
    start, end = _slice_bounds(
        start_step=start_step,
        end_step=end_step,
        available_steps=complete.effective_inputs.steps,
    )
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    profile = (
        SD3_PUBLISHER_REFERENCE_PROFILE
        if internal_mode is SD3ShiftMode.PUBLISHER_REFERENCE
        else SD3_COMFY_DIFFUSERS_PROFILE
    )
    evidence = complete.request.provenance.evidence
    recipe = (
        profile.profile_id
        if evidence is not None and evidence.value != "modified"
        else f"{profile.profile_id}.modified-{count}"
    )
    effective_end = count if end is None else end
    ratio = 1.0 if internal_mode is SD3ShiftMode.PUBLISHER_REFERENCE else 3.0
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "guidance": {
            "host_cfg_scale": 5.0 if internal_mode is SD3ShiftMode.PUBLISHER_REFERENCE else 7.0,
            "model_cfg_scale": 5.0 if internal_mode is SD3ShiftMode.PUBLISHER_REFERENCE else 7.0,
        },
        "profile": {
            "evidence": evidence.value,
            "id": profile.profile_id,
            "recipe": recipe,
            "variant": "original",
            "version": profile.profile_version,
        },
        "schema": SD3_SIGMA_NODE_SCHEMA_ID,
        "shift": {"kind": "direct_ratio", "ratio": ratio},
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_source": strict_source,
        "warnings": list(complete.warnings),
    }
    return SD3SigmaNodeResult(
        mode=public_mode,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_sd3_sigma_output_info(
    result: SD3SigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to the actual host tensor values."""

    if not isinstance(result, SD3SigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("SD3 output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("SD3 schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class SD3SigmaScheduler:
    """Construct explicit original-SD3 external sigmas without model patching."""

    DESCRIPTION = "Builds an explicit original Stable Diffusion 3 sigma schedule with a source-qualified shift."
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
                "mode": (_MODES,),
                "steps": ("INT", {"default": 28, "min": 1, "max": _MAX_STEPS, "step": 1}),
                "strict_source": ("BOOLEAN", {"default": False}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS - 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": _MAX_STEPS}),
                "already_shifted": ("BOOLEAN", {"default": False}),
            }
        }

    def build(
        self,
        mode: object,
        steps: object,
        strict_source: object,
        start_step: object,
        end_step: object,
        already_shifted: object = False,
    ) -> tuple[object, str]:
        result = build_sd3_sigma_schedule(
            mode=mode,
            steps=steps,
            strict_source=strict_source,
            start_step=start_step,
            end_step=end_step,
            already_shifted=already_shifted,
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
        return tensor, bind_sd3_sigma_output_info(result, output_sigmas=host_values)


__all__ = [
    "SD3_SIGMA_NODE_ID",
    "SD3_SIGMA_NODE_SCHEMA_ID",
    "SD3SigmaNodeResult",
    "SD3SigmaScheduler",
    "bind_sd3_sigma_output_info",
    "build_sd3_sigma_schedule",
]
