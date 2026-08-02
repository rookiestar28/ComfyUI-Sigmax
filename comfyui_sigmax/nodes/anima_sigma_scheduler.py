"""Thin explicit Anima v1 SIGMAS node."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Final

from comfyui_sigmax.core import (
    EvidenceLevel,
    ScheduleContractError,
    SigmaDomain,
    numerical_fingerprint,
    slice_step_range,
    validate_sigma_schedule,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint
from comfyui_sigmax.profiles.anima import (
    ANIMA_AESTHETIC_PROFILE,
    ANIMA_BASE_PROFILE,
    ANIMA_TURBO_PROFILE,
    AnimaProfile,
    AnimaVariant,
    build_anima_schedule,
)

ANIMA_SIGMA_NODE_ID: Final = "Sigmax.AnimaSigmaScheduler"
ANIMA_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.anima-sigma-node/1"
_MAX_STEPS: Final = 10_000
_BASE_MODE: Final = "Base (3.0)"
_AESTHETIC_MODE: Final = "Aesthetic (3.0)"
_TURBO_MODE: Final = "Turbo (3.0)"
_MODES: Final = (_BASE_MODE, _AESTHETIC_MODE, _TURBO_MODE)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnimaSigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    variant: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.variant not in _MODES:
            raise ScheduleContractError("Anima node variant is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("Anima node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("Anima node result is incomplete")


def _variant(value: object) -> tuple[str, AnimaVariant, AnimaProfile]:
    if value == _BASE_MODE:
        return _BASE_MODE, AnimaVariant.BASE, ANIMA_BASE_PROFILE
    if value == _AESTHETIC_MODE:
        return _AESTHETIC_MODE, AnimaVariant.AESTHETIC, ANIMA_AESTHETIC_PROFILE
    if value == _TURBO_MODE:
        return _TURBO_MODE, AnimaVariant.TURBO, ANIMA_TURBO_PROFILE
    raise ScheduleContractError("variant must be Base (3.0), Aesthetic (3.0), or Turbo (3.0)")


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
        raise ScheduleContractError("Anima schedule information must be canonical JSON") from exc


def build_anima_sigma_schedule(
    *,
    variant: object,
    steps: object,
    strict_source: object,
    start_step: object,
    end_step: object,
    already_shifted: object = False,
) -> AnimaSigmaNodeResult:
    """Build and slice an explicit Anima schedule without host imports."""

    public_variant, internal_variant, profile = _variant(variant)
    count = _positive_steps(steps)
    if not isinstance(strict_source, bool) or not isinstance(already_shifted, bool):
        raise ScheduleContractError("strict_source and already_shifted must be boolean")
    complete = build_anima_schedule(
        variant=internal_variant,
        steps=count,
        strict_source=strict_source,
        already_shifted=already_shifted,
    )
    start, end = _slice_bounds(start_step=start_step, end_step=end_step, available_steps=count)
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    evidence = complete.request.provenance.evidence
    recipe = (
        profile.profile_id
        if evidence is EvidenceLevel.FRAMEWORK_REFERENCE
        else f"{profile.profile_id}.modified-{count}"
    )
    effective_end = count if end is None else end
    guidance = profile.schema.recipes[0].guidance
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "guidance": {
            "host_cfg_scale": guidance.host_value,
            "model_cfg_scale": guidance.model_value,
        },
        "host_compatibility": "comfyui_fixed_flow",
        "profile": {
            "evidence": evidence.value,
            "id": profile.profile_id,
            "recipe": recipe,
            "variant": profile.schema.model_variant,
            "version": profile.profile_version,
        },
        "schema": ANIMA_SIGMA_NODE_SCHEMA_ID,
        "shift": {"kind": "rational", "multiplier": 1.0, "shift": 3.0},
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_source": strict_source,
        "warnings": list(complete.warnings),
    }
    return AnimaSigmaNodeResult(
        variant=public_variant,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_anima_sigma_output_info(
    result: AnimaSigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to actual host tensor values."""

    if not isinstance(result, AnimaSigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("Anima output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("Anima schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class AnimaSigmaScheduler:
    """Construct explicit Anima v1 external sigmas without model patching."""

    DESCRIPTION = "Builds an explicit Anima v1 fixed-shift sigma schedule."
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
                "variant": (_MODES,),
                "steps": ("INT", {"default": 50, "min": 1, "max": _MAX_STEPS, "step": 1}),
                "strict_source": ("BOOLEAN", {"default": False}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS - 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": _MAX_STEPS}),
                "already_shifted": ("BOOLEAN", {"default": False}),
            }
        }

    def build(
        self,
        variant: object,
        steps: object,
        strict_source: object,
        start_step: object,
        end_step: object,
        already_shifted: object = False,
    ) -> tuple[object, str]:
        result = build_anima_sigma_schedule(
            variant=variant,
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
        return tensor, bind_anima_sigma_output_info(result, output_sigmas=host_values)


__all__ = [
    "ANIMA_SIGMA_NODE_ID",
    "ANIMA_SIGMA_NODE_SCHEMA_ID",
    "AnimaSigmaNodeResult",
    "AnimaSigmaScheduler",
    "bind_anima_sigma_output_info",
    "build_anima_sigma_schedule",
]
