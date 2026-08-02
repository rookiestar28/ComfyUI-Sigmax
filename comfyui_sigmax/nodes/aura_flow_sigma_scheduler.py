"""Thin explicit original AuraFlow v0.2 SIGMAS node."""

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
from comfyui_sigmax.profiles.aura_flow import (
    AURAFLOW_V02_PROFILE,
    AuraFlowShiftMode,
    build_aura_flow_schedule,
)

AURAFLOW_SIGMA_NODE_ID: Final = "Sigmax.AuraFlowSigmaScheduler"
AURAFLOW_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.aura-flow-sigma-node/1"
_MAX_STEPS: Final = 10_000
_MODE: Final = "Official Fixed (1.73)"


@dataclass(frozen=True, slots=True, kw_only=True)
class AuraFlowSigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    mode: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.mode != _MODE:
            raise ScheduleContractError("AuraFlow node mode is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("AuraFlow node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("AuraFlow node result is incomplete")


def _mode(value: object) -> tuple[str, AuraFlowShiftMode]:
    if value == _MODE:
        return _MODE, AuraFlowShiftMode.OFFICIAL_FIXED
    raise ScheduleContractError("mode must be Official Fixed (1.73)")


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
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("AuraFlow schedule information must be canonical JSON") from exc


def build_aura_flow_sigma_schedule(
    *,
    mode: object,
    steps: object,
    strict_source: object,
    start_step: object,
    end_step: object,
    already_shifted: object = False,
) -> AuraFlowSigmaNodeResult:
    """Build and slice an explicit original AuraFlow schedule without host imports."""

    public_mode, internal_mode = _mode(mode)
    count = _positive_steps(steps)
    if not isinstance(strict_source, bool):
        raise ScheduleContractError("strict_source must be boolean")
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    complete = build_aura_flow_schedule(
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
    evidence = complete.request.provenance.evidence
    recipe = (
        AURAFLOW_V02_PROFILE.profile_id
        if evidence is not None and evidence.value != "modified"
        else f"{AURAFLOW_V02_PROFILE.profile_id}.modified-{count}"
    )
    effective_end = count if end is None else end
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas,
                domain=complete.final_domain,
                precision="float64",
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "guidance": {"host_cfg_scale": 3.5, "model_cfg_scale": 3.5},
        "profile": {
            "evidence": evidence.value,
            "id": AURAFLOW_V02_PROFILE.profile_id,
            "recipe": recipe,
            "variant": "v0.2",
            "version": AURAFLOW_V02_PROFILE.profile_version,
        },
        "schema": AURAFLOW_SIGMA_NODE_SCHEMA_ID,
        "shift": {"kind": "direct_ratio", "multiplier": 1.0, "ratio": 1.73},
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_source": strict_source,
        "warnings": list(complete.warnings),
    }
    return AuraFlowSigmaNodeResult(
        mode=public_mode,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_aura_flow_sigma_output_info(
    result: AuraFlowSigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to actual host tensor values."""

    if not isinstance(result, AuraFlowSigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("AuraFlow output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("AuraFlow schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class AuraFlowSigmaScheduler:
    """Construct explicit original AuraFlow v0.2 external sigmas without model patching."""

    DESCRIPTION = "Builds an explicit original AuraFlow v0.2 fixed-shift sigma schedule."
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
                "mode": ((_MODE,),),
                "steps": ("INT", {"default": 50, "min": 1, "max": _MAX_STEPS, "step": 1}),
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
        result = build_aura_flow_sigma_schedule(
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
        return tensor, bind_aura_flow_sigma_output_info(result, output_sigmas=host_values)


__all__ = [
    "AURAFLOW_SIGMA_NODE_ID",
    "AURAFLOW_SIGMA_NODE_SCHEMA_ID",
    "AuraFlowSigmaNodeResult",
    "AuraFlowSigmaScheduler",
    "bind_aura_flow_sigma_output_info",
    "build_aura_flow_sigma_schedule",
]
