"""Thin explicit FLUX.1-schnell SIGMAS node."""

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
from comfyui_sigmax.profiles.flux1_schnell import (
    FLUX1_SCHNELL_PROFILE,
    build_flux1_schnell_schedule,
)

FLUX1_SCHNELL_SIGMA_NODE_ID: Final = "Sigmax.Flux1SchnellSigmaScheduler"
FLUX1_SCHNELL_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.flux1-schnell-sigma-node/1"
_MAX_STEPS: Final = 10_000


@dataclass(frozen=True, slots=True, kw_only=True)
class Flux1SchnellSigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("FLUX.1-schnell node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("FLUX.1-schnell node result is incomplete")


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
            "FLUX.1-schnell schedule information must be canonical JSON"
        ) from exc


def build_flux1_schnell_sigma_schedule(
    *,
    steps: object,
    strict_official: object,
    start_step: object,
    end_step: object,
) -> Flux1SchnellSigmaNodeResult:
    """Build and slice FLUX.1-schnell sigmas without host imports."""

    count = _positive_steps(steps)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")
    complete = build_flux1_schnell_schedule(steps=count, strict_official=strict_official)
    start, end = _slice_bounds(
        start_step=start_step,
        end_step=end_step,
        available_steps=complete.effective_inputs.steps,
    )
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    evidence = complete.request.provenance.evidence
    recipe = (
        "flux1.schnell.official"
        if evidence.value == "official"
        else f"flux1.schnell.modified-{count}"
    )
    effective_end = count if end is None else end
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "guidance": {"host_cfg": 1.0, "model_guidance": 0.0},
        "profile": {
            "evidence": evidence.value,
            "id": FLUX1_SCHNELL_PROFILE.profile_id,
            "recipe": recipe,
            "variant": "schnell",
            "version": FLUX1_SCHNELL_PROFILE.profile_version,
        },
        "schema": FLUX1_SCHNELL_SIGMA_NODE_SCHEMA_ID,
        "shift": {"dynamic": False, "kind": "none"},
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_official": strict_official,
        "warnings": list(complete.warnings),
    }
    return Flux1SchnellSigmaNodeResult(
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_flux1_schnell_sigma_output_info(
    result: Flux1SchnellSigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to the actual host tensor values."""

    if not isinstance(result, Flux1SchnellSigmaNodeResult) or len(output_sigmas) != len(
        result.sigmas
    ):
        raise ScheduleContractError("FLUX.1-schnell output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("FLUX.1-schnell schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class Flux1SchnellSigmaScheduler:
    """Construct explicit FLUX.1-schnell external sigmas without model patching."""

    DESCRIPTION = "Builds the pinned unshifted FLUX.1-schnell sigma schedule."
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "steps": ("INT", {"default": 4, "min": 1, "max": _MAX_STEPS, "step": 1}),
                "strict_official": ("BOOLEAN", {"default": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS - 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": _MAX_STEPS}),
            }
        }

    def build(
        self,
        steps: object,
        strict_official: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        result = build_flux1_schnell_sigma_schedule(
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
        return tensor, bind_flux1_schnell_sigma_output_info(result, output_sigmas=host_values)
