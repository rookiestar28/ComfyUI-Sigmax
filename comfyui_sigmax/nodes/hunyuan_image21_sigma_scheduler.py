"""Thin explicit HunyuanImage 2.1 Base/Distilled SIGMAS node."""

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
from comfyui_sigmax.profiles.hunyuan_image21 import (
    HUNYUAN_IMAGE21_BASE_PROFILE,
    HUNYUAN_IMAGE21_DISTILLED_PROFILE,
    HunyuanImage21Profile,
    HunyuanImage21Variant,
    build_hunyuan_image21_schedule,
)

HUNYUAN_IMAGE21_SIGMA_NODE_ID: Final = "Sigmax.HunyuanImage21SigmaScheduler"
HUNYUAN_IMAGE21_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.hunyuan-image-2-1-sigma-node/1"
_MAX_STEPS: Final = 10_000
_BASE_MODE: Final = "Base (5.0)"
_DISTILLED_MODE: Final = "Distilled (4.0)"
_MODES: Final = (_BASE_MODE, _DISTILLED_MODE)


@dataclass(frozen=True, slots=True, kw_only=True)
class HunyuanImage21SigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    variant: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.variant not in _MODES:
            raise ScheduleContractError("HunyuanImage 2.1 node variant is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("HunyuanImage 2.1 node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("HunyuanImage 2.1 node result is incomplete")


def _variant(
    value: object,
) -> tuple[str, HunyuanImage21Variant, float, HunyuanImage21Profile]:
    if value == _BASE_MODE:
        return _BASE_MODE, HunyuanImage21Variant.BASE, 5.0, HUNYUAN_IMAGE21_BASE_PROFILE
    if value == _DISTILLED_MODE:
        return (
            _DISTILLED_MODE,
            HunyuanImage21Variant.DISTILLED,
            4.0,
            HUNYUAN_IMAGE21_DISTILLED_PROFILE,
        )
    raise ScheduleContractError("variant must be Base (5.0) or Distilled (4.0)")


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
        raise ScheduleContractError(
            "HunyuanImage 2.1 schedule information must be canonical JSON"
        ) from exc


def build_hunyuan_image21_sigma_schedule(
    *,
    variant: object,
    steps: object,
    strict_source: object,
    start_step: object,
    end_step: object,
    already_shifted: object = False,
) -> HunyuanImage21SigmaNodeResult:
    """Build and slice an explicit HunyuanImage 2.1 schedule without host imports."""

    public_variant, internal_variant, ratio, profile = _variant(variant)
    count = _positive_steps(steps)
    if not isinstance(strict_source, bool) or not isinstance(already_shifted, bool):
        raise ScheduleContractError("strict_source and already_shifted must be boolean")
    complete = build_hunyuan_image21_schedule(
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
        if evidence is not None and evidence.value != "modified"
        else f"{profile.profile_id}.modified-{count}"
    )
    effective_end = count if end is None else end
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "guidance": {
            "host_cfg_scale": 3.5 if internal_variant is HunyuanImage21Variant.BASE else 3.25,
            "model_cfg_scale": 3.5 if internal_variant is HunyuanImage21Variant.BASE else 3.25,
        },
        "host_compatibility": "native_base"
        if internal_variant is HunyuanImage21Variant.BASE
        else "publisher_schedule_only",
        "profile": {
            "evidence": evidence.value,
            "id": profile.profile_id,
            "recipe": recipe,
            "variant": profile.schema.model_variant,
            "version": profile.profile_version,
        },
        "schema": HUNYUAN_IMAGE21_SIGMA_NODE_SCHEMA_ID,
        "shift": {"kind": "direct_ratio", "multiplier": 1.0, "ratio": ratio},
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_source": strict_source,
        "warnings": list(complete.warnings),
    }
    return HunyuanImage21SigmaNodeResult(
        variant=public_variant,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_hunyuan_image21_sigma_output_info(
    result: HunyuanImage21SigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to actual host tensor values."""

    if not isinstance(result, HunyuanImage21SigmaNodeResult) or len(output_sigmas) != len(
        result.sigmas
    ):
        raise ScheduleContractError("HunyuanImage 2.1 output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("HunyuanImage 2.1 schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class HunyuanImage21SigmaScheduler:
    """Construct explicit HunyuanImage 2.1 external sigmas without model patching."""

    DESCRIPTION = "Builds an explicit HunyuanImage 2.1 Base or Distilled sigma schedule."
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
        result = build_hunyuan_image21_sigma_schedule(
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
        return tensor, bind_hunyuan_image21_sigma_output_info(result, output_sigmas=host_values)


__all__ = [
    "HUNYUAN_IMAGE21_SIGMA_NODE_ID",
    "HUNYUAN_IMAGE21_SIGMA_NODE_SCHEMA_ID",
    "HunyuanImage21SigmaNodeResult",
    "HunyuanImage21SigmaScheduler",
    "bind_hunyuan_image21_sigma_output_info",
    "build_hunyuan_image21_sigma_schedule",
]
