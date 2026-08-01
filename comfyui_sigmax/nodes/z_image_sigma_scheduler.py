"""Thin explicit Z-Image SIGMAS node over the dependency-free profile builder."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    numerical_fingerprint,
    slice_step_range,
    validate_sigma_schedule,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint
from comfyui_sigmax.profiles.z_image import (
    Z_IMAGE_BASE_PROFILE,
    Z_IMAGE_TURBO_PROFILE,
    ZImageVariant,
    build_z_image_schedule,
)

Z_IMAGE_SIGMA_NODE_ID: Final = "Sigmax.ZImageSigmaScheduler"
Z_IMAGE_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.z-image-sigma-node/1"
_MAX_STEPS: Final = 10_000


class ZImageSigmaVariant(str, Enum):
    """User-facing explicit variant choices."""

    BASE = "Base"
    TURBO = "Turbo"


@dataclass(frozen=True, slots=True, kw_only=True)
class ZImageSigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    variant: ZImageSigmaVariant
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.variant, ZImageSigmaVariant):
            raise ScheduleContractError("Z-Image node variant is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("Z-Image node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("Z-Image node result is incomplete")


def _variant(value: object) -> tuple[ZImageSigmaVariant, ZImageVariant]:
    if not isinstance(value, str):
        raise ScheduleContractError("variant must be Base or Turbo")
    try:
        public = ZImageSigmaVariant(value)
    except ValueError as exc:
        raise ScheduleContractError("variant must be Base or Turbo") from exc
    internal = ZImageVariant.BASE if public is ZImageSigmaVariant.BASE else ZImageVariant.TURBO
    return public, internal


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
        raise ScheduleContractError("Z-Image schedule information must be canonical JSON") from exc


def build_z_image_sigma_schedule(
    *,
    variant: object,
    steps: object,
    strict_official: object,
    start_step: object,
    end_step: object,
) -> ZImageSigmaNodeResult:
    """Build and slice one explicit Z-Image schedule without host imports."""

    public_variant, internal_variant = _variant(variant)
    count = _positive_steps(steps)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")
    complete = build_z_image_schedule(
        variant=internal_variant, steps=count, strict_official=strict_official
    )
    start, end = _slice_bounds(
        start_step=start_step, end_step=end_step, available_steps=complete.effective_inputs.steps
    )
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    profile = (
        Z_IMAGE_BASE_PROFILE if internal_variant is ZImageVariant.BASE else Z_IMAGE_TURBO_PROFILE
    )
    evidence = complete.request.provenance.evidence
    recipe = (
        f"z_image.{internal_variant.value}.official"
        if evidence.value == "official"
        else f"z_image.{internal_variant.value}.modified-{count}"
    )
    effective_end = count if end is None else end
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "profile": {
            "evidence": evidence.value,
            "id": profile.profile_id,
            "recipe": recipe,
            "variant": internal_variant.value,
            "version": profile.profile_version,
        },
        "schema": Z_IMAGE_SIGMA_NODE_SCHEMA_ID,
        "shift": {
            "dynamic": False,
            "kind": "fixed_direct_ratio",
            "ratio": profile.fixed_shift_ratio,
        },
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_official": strict_official,
        "warnings": list(complete.warnings),
    }
    return ZImageSigmaNodeResult(
        variant=public_variant,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_z_image_sigma_output_info(
    result: ZImageSigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to the actual values returned by the host tensor implementation."""

    if not isinstance(result, ZImageSigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("Z-Image output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("Z-Image schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class ZImageSigmaScheduler:
    """Construct explicit Z-Image Base or Turbo external sigmas for custom sampling."""

    DESCRIPTION = "Builds a pinned Z-Image Base or Turbo sigma schedule without model patching."
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "variant": (("Base", "Turbo"),),
                "steps": ("INT", {"default": 50, "min": 1, "max": _MAX_STEPS, "step": 1}),
                "strict_official": ("BOOLEAN", {"default": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS - 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": _MAX_STEPS}),
            }
        }

    def build(
        self,
        variant: object,
        steps: object,
        strict_official: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        result = build_z_image_sigma_schedule(
            variant=variant,
            steps=steps,
            strict_official=strict_official,
            start_step=start_step,
            end_step=end_step,
        )
        try:
            torch = importlib.import_module("torch")
            float_tensor = torch.__dict__["FloatTensor"]
        except (ImportError, KeyError) as exc:
            # CRITICAL: keep Torch execution-only; package imports must remain dependency-free.
            raise RuntimeError("ComfyUI host execution requires Torch FloatTensor support") from exc
        tensor = float_tensor(result.sigmas)
        try:
            host_values = tuple(float(value) for value in tensor.tolist())
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("ComfyUI host tensor must expose numeric tolist output") from exc
        return tensor, bind_z_image_sigma_output_info(result, output_sigmas=host_values)
