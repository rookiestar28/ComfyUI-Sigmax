"""Configurable external unit-flow SIGMAS node over the dependency-free core."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    BaseGridSpec,
    EvidenceLevel,
    Provenance,
    ScheduleContractError,
    ScheduleInputs,
    ScheduleOwnership,
    ScheduleRequest,
    ScheduleResult,
    SigmaDomain,
    SliceSpec,
    TerminalPolicy,
    TransformContract,
    TransformStage,
    apply_terminal_policy,
    direct_ratio_shift,
    exponential_mu_shift,
    linear_endpoint_grid,
    slice_step_range,
    validate_sigma_schedule,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint

ADVANCED_FLOWMATCH_NODE_ID: Final = "Sigmax.AdvancedFlowMatchScheduler"
ADVANCED_FLOWMATCH_NODE_SCHEMA_ID: Final = "sigmax.advanced-flowmatch-node/1"
_ENGINE_VERSION: Final = "0.1.0.dev0"
_SOURCE_ID: Final = "sigmax.advanced-flowmatch-scheduler"
_BASE_GRID_ID: Final = "sigmax.linear_endpoint"
_MAX_STEPS: Final = 10_000
_MIN_SHIFT_VALUE: Final = -20.0
_MAX_SHIFT_VALUE: Final = 20.0


class AdvancedFlowMatchShiftMode(str, Enum):
    """Mutually exclusive primary unit-flow shift parameterizations."""

    EXPONENTIAL_MU = "exponential_mu"
    DIRECT_RATIO = "direct_ratio"


@dataclass(frozen=True, slots=True, kw_only=True)
class AdvancedFlowMatchNodeResult:
    """Pure advanced scheduler output before host tensor conversion."""

    shift_mode: AdvancedFlowMatchShiftMode
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.shift_mode, AdvancedFlowMatchShiftMode):
            raise ScheduleContractError("advanced FlowMatch shift mode is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("advanced FlowMatch result must use UNIT_FLOW")
        if not isinstance(self.sigmas, tuple) or len(self.sigmas) < 2:
            raise ScheduleContractError("advanced FlowMatch result requires sigma transitions")
        validate_sigma_schedule(
            self.sigmas,
            domain=self.domain,
            expected_steps=len(self.sigmas) - 1,
            require_terminal_zero=False,
        )
        if not isinstance(self.schedule_info_json, str) or not self.schedule_info_json:
            raise ScheduleContractError("advanced FlowMatch result requires schedule information")


def _canonical_info(projection: dict[str, object]) -> str:
    try:
        return json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("schedule information is not canonical JSON") from exc


def _positive_steps(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 2 or value > _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 2 and {_MAX_STEPS}")
    return value


def _unit_endpoint(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ScheduleContractError(f"{label} must be finite and within UNIT_FLOW [0, 1]")
    return float(value)


def _domain(value: object) -> SigmaDomain:
    if value != SigmaDomain.UNIT_FLOW.value:
        raise ScheduleContractError("domain must be UNIT_FLOW")
    return SigmaDomain.UNIT_FLOW


def _shift_mode(value: object) -> AdvancedFlowMatchShiftMode:
    if not isinstance(value, str):
        raise ScheduleContractError("shift_mode must be exponential_mu or direct_ratio")
    try:
        return AdvancedFlowMatchShiftMode(value)
    except ValueError as exc:
        raise ScheduleContractError("shift_mode must be exponential_mu or direct_ratio") from exc


def _shift_value(value: object, *, mode: AdvancedFlowMatchShiftMode) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not _MIN_SHIFT_VALUE <= float(value) <= _MAX_SHIFT_VALUE
    ):
        raise ScheduleContractError(
            f"shift_value must be finite and between {_MIN_SHIFT_VALUE:g} and {_MAX_SHIFT_VALUE:g}"
        )
    normalized = float(value)
    if mode is AdvancedFlowMatchShiftMode.DIRECT_RATIO and normalized <= 0.0:
        raise ScheduleContractError("direct_ratio shift_value must be greater than zero")
    return normalized


def _terminal_policy(value: object) -> TerminalPolicy:
    if value == "append_zero":
        return TerminalPolicy.APPEND_ZERO
    if value == "preserve":
        return TerminalPolicy.PRESERVE
    raise ScheduleContractError("terminal_policy must be append_zero or preserve")


def _slice_bounds(
    *,
    start_step: object,
    end_step: object,
    available_steps: int,
) -> tuple[int, int | None]:
    if not isinstance(start_step, int) or isinstance(start_step, bool) or start_step < 0:
        raise ScheduleContractError("start_step must be a non-negative integer")
    if not isinstance(end_step, int) or isinstance(end_step, bool) or end_step < -1:
        raise ScheduleContractError("end_step must be -1 or a non-negative integer")
    effective_end = None if end_step == -1 else end_step
    if start_step >= available_steps:
        raise ScheduleContractError("start_step must be below the constructed step count")
    if effective_end is not None and (
        effective_end <= start_step or effective_end > available_steps
    ):
        raise ScheduleContractError(
            "end_step must exceed start_step and not exceed the constructed step count"
        )
    return start_step, effective_end


def build_advanced_flowmatch_schedule(
    *,
    domain: object,
    steps: object,
    sigma_start: object,
    sigma_end: object,
    shift_mode: object,
    shift_value: object,
    terminal_policy: object,
    start_step: object,
    end_step: object,
) -> AdvancedFlowMatchNodeResult:
    """Build one explicit external unit-flow schedule without host imports."""

    selected_domain = _domain(domain)
    requested_steps = _positive_steps(steps)
    start = _unit_endpoint(sigma_start, label="sigma_start")
    end = _unit_endpoint(sigma_end, label="sigma_end")
    if start <= end:
        raise ScheduleContractError("sigma_start must be greater than sigma_end")

    selected_shift = _shift_mode(shift_mode)
    selected_shift_value = _shift_value(shift_value, mode=selected_shift)
    selected_terminal = _terminal_policy(terminal_policy)
    if selected_terminal is TerminalPolicy.APPEND_ZERO and end == 0.0:
        raise ScheduleContractError(
            "append_zero requires sigma_end greater than zero to avoid duplicate terminal zero"
        )

    # Preserve needs steps + 1 grid values; append_zero supplies the final value itself.
    base_points = (
        requested_steps + 1 if selected_terminal is TerminalPolicy.PRESERVE else requested_steps
    )
    base = linear_endpoint_grid(
        points=base_points,
        start=start,
        end=end,
        domain=selected_domain,
    )
    if selected_shift is AdvancedFlowMatchShiftMode.EXPONENTIAL_MU:
        shifted = exponential_mu_shift(
            base,
            mu=selected_shift_value,
            domain=selected_domain,
        )
    else:
        shifted = direct_ratio_shift(
            base,
            ratio=selected_shift_value,
            domain=selected_domain,
        )

    transform = TransformContract(
        name=f"sigmax.{selected_shift.value}",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=selected_domain,
        output_domain=selected_domain,
    )
    complete = apply_terminal_policy(
        shifted,
        policy=selected_terminal,
        domain=selected_domain,
    )
    validate_sigma_schedule(
        complete,
        domain=selected_domain,
        expected_steps=requested_steps,
        require_terminal_zero=selected_terminal is TerminalPolicy.APPEND_ZERO,
    )
    selected_start, selected_end = _slice_bounds(
        start_step=start_step,
        end_step=end_step,
        available_steps=requested_steps,
    )
    output = slice_step_range(
        complete,
        start_step=selected_start,
        end_step=selected_end,
    )

    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=requested_steps),
        sigma_domain=selected_domain,
        provenance=Provenance(
            engine_version=_ENGINE_VERSION,
            evidence=EvidenceLevel.EXPERIMENTAL,
            source=_SOURCE_ID,
        ),
        base_grid=BaseGridSpec(
            identifier=_BASE_GRID_ID,
            output_domain=selected_domain,
        ),
        transforms=(transform,),
        terminal_policy=selected_terminal,
        slicing=SliceSpec(
            start_step=selected_start,
            end_step=selected_end,
            denoise=1.0,
        ),
    )
    ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=requested_steps),
        sigmas=output,
        final_domain=selected_domain,
    )

    effective_end = requested_steps if selected_end is None else selected_end
    projection: dict[str, object] = {
        "base_grid": {
            "identifier": _BASE_GRID_ID,
            "points": base_points,
            "sigma_end": end,
            "sigma_start": start,
        },
        "domain": {
            "sigma": selected_domain.value,
            "time": selected_domain.value,
        },
        "fingerprints": {
            "complete": sigma_output_fingerprint(complete, domain=selected_domain),
            "output": sigma_output_fingerprint(output, domain=selected_domain),
        },
        "ownership": ScheduleOwnership.EXTERNAL_SIGMAS.value,
        "provenance": {
            "evidence": EvidenceLevel.EXPERIMENTAL.value,
            "profile_id": None,
            "source": _SOURCE_ID,
        },
        "schema": ADVANCED_FLOWMATCH_NODE_SCHEMA_ID,
        "shift": {
            "kind": selected_shift.value,
            "value": selected_shift_value,
        },
        "slicing": {
            "available_steps": requested_steps,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": selected_start,
        },
        "terminal": {
            "policy": str(terminal_policy),
            "value": complete[-1],
        },
        "transform_order": [
            TransformStage.PRIMARY_TIME_SHIFT.value,
            TransformStage.TERMINAL.value,
            TransformStage.SLICE.value,
        ],
    }
    return AdvancedFlowMatchNodeResult(
        shift_mode=selected_shift,
        domain=selected_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(projection),
    )


class AdvancedFlowMatchScheduler:
    """Construct explicit configurable unit-flow sigmas for ComfyUI."""

    DESCRIPTION = (
        "Builds an experimental external UNIT_FLOW sigma schedule with one explicit "
        "shift parameterization; it does not sample or patch a model."
    )
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        """Return a fresh deterministic legacy/current ComfyUI input schema."""

        return {
            "required": {
                "domain": (("UNIT_FLOW",),),
                "steps": (
                    "INT",
                    {"default": 20, "min": 2, "max": _MAX_STEPS, "step": 1},
                ),
                "sigma_start": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "sigma_end": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "shift_mode": (("exponential_mu", "direct_ratio"),),
                "shift_value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": _MIN_SHIFT_VALUE,
                        "max": _MAX_SHIFT_VALUE,
                        "step": 0.01,
                    },
                ),
                "terminal_policy": (("append_zero", "preserve"),),
                "start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": _MAX_STEPS - 1, "step": 1},
                ),
                "end_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": _MAX_STEPS, "step": 1},
                ),
            }
        }

    def build(
        self,
        domain: object,
        steps: object,
        sigma_start: object,
        sigma_end: object,
        shift_mode: object,
        shift_value: object,
        terminal_policy: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        """Build pure output first, then convert through host-provided Torch."""

        result = build_advanced_flowmatch_schedule(
            domain=domain,
            steps=steps,
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            shift_mode=shift_mode,
            shift_value=shift_value,
            terminal_policy=terminal_policy,
            start_step=start_step,
            end_step=end_step,
        )
        try:
            torch = importlib.import_module("torch")
            float_tensor = torch.__dict__["FloatTensor"]
        except (ImportError, KeyError) as exc:
            # CRITICAL: keep Torch execution-only; package imports must remain dependency-free.
            raise RuntimeError("ComfyUI host execution requires Torch FloatTensor support") from exc
        return (float_tensor(result.sigmas), result.schedule_info_json)
