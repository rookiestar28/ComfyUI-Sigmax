"""Dependency-free terminal and terminal-inclusive slicing policies.

Behavior references:

- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py
- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_custom_sampler.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
"""

from __future__ import annotations

import math

from comfyui_sigmax.core.request_result import TerminalPolicy
from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    SigmaDomain,
)


def _require_numeric_tuple(
    values: tuple[float, ...],
    *,
    minimum_length: int,
) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) < minimum_length:
        raise ScheduleContractError(
            f"schedule values must be a numeric tuple with at least {minimum_length} values"
        )
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ScheduleContractError("schedule values must contain only numeric values")
    return values


def _require_positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleContractError(f"{label} must be a positive integer")
    return value


def _require_denoise(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ScheduleContractError("denoise must be finite and between 0 and 1")
    return float(value)


def apply_terminal_policy(
    values: tuple[float, ...],
    *,
    policy: TerminalPolicy,
    domain: SigmaDomain,
) -> tuple[float, ...]:
    """Append terminal zero or explicitly preserve an existing terminal."""

    schedule = _require_numeric_tuple(values, minimum_length=1)
    if domain is SigmaDomain.MODEL_NATIVE or not isinstance(domain, SigmaDomain):
        raise ScheduleContractError(
            "terminal policy cannot mutate the opaque MODEL_NATIVE or an invalid domain"
        )
    if not isinstance(policy, TerminalPolicy):
        raise ScheduleContractError("policy must be a TerminalPolicy value")
    if policy is TerminalPolicy.PRESERVE:
        return schedule
    if schedule[-1] == 0.0:
        raise ScheduleContractError("schedule already has a zero terminal value")
    return (*schedule, 0.0)


def slice_step_range(
    values: tuple[float, ...],
    *,
    start_step: int = 0,
    end_step: int | None = None,
) -> tuple[float, ...]:
    """Retain an explicit transition range from a terminal-inclusive vector."""

    schedule = _require_numeric_tuple(values, minimum_length=2)
    available_steps = len(schedule) - 1
    if (
        not isinstance(start_step, int)
        or isinstance(start_step, bool)
        or not 0 <= start_step < available_steps
    ):
        raise ScheduleContractError(
            "start_step must be a non-negative integer below the available step count"
        )

    effective_end = available_steps if end_step is None else end_step
    if (
        not isinstance(effective_end, int)
        or isinstance(effective_end, bool)
        or effective_end <= start_step
        or effective_end > available_steps
    ):
        raise ScheduleContractError(
            "end_step must be an integer greater than start_step "
            "and no greater than the available step count"
        )
    if start_step == 0 and effective_end == available_steps:
        return schedule
    return schedule[start_step : effective_end + 1]


def denoise_construction_steps(requested_steps: int, denoise: float) -> int:
    """Return ComfyUI's construction step count for a requested denoise amount."""

    steps = _require_positive_integer(requested_steps, label="requested_steps")
    amount = _require_denoise(denoise)
    if amount == 0.0:
        return 0
    if amount == 1.0:
        return steps
    return int(steps / amount)


def slice_denoise_tail(
    values: tuple[float, ...],
    *,
    requested_steps: int,
    denoise: float,
) -> tuple[float, ...]:
    """Retain ComfyUI's requested terminal-inclusive tail after denoise construction."""

    construction_steps = denoise_construction_steps(requested_steps, denoise)
    if construction_steps == 0:
        if values != ():
            raise ScheduleContractError("zero denoise requires an empty schedule vector")
        return values

    schedule = _require_numeric_tuple(values, minimum_length=2)
    expected_length = construction_steps + 1
    if len(schedule) != expected_length:
        raise ScheduleContractError(
            "schedule length must equal denoise construction steps plus terminal"
        )
    if construction_steps == requested_steps:
        return schedule
    return schedule[-(requested_steps + 1) :]
