"""Dependency-free constructors for unshifted, non-terminal base grids."""

from __future__ import annotations

import math

from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    SigmaDomain,
)


def _require_integer_count(value: object, *, minimum: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        if minimum == 1:
            raise ScheduleContractError(f"{label} must be a positive integer")
        raise ScheduleContractError(f"{label} must be an integer with at least two values")
    return value


def krea_reciprocal_step_grid(
    steps: int,
    *,
    domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
) -> tuple[float, ...]:
    """Build Krea's non-terminal unshifted grid from one through one-over-steps."""

    count = _require_integer_count(steps, minimum=1, label="steps")
    if domain is not SigmaDomain.UNIT_FLOW:
        raise ScheduleContractError("Krea reciprocal-step grid requires the UNIT_FLOW domain")
    return tuple((count - index) / count for index in range(count))


def flowmatch_reciprocal_step_grid(
    steps: int,
    *,
    domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
) -> tuple[float, ...]:
    """Build a neutral terminal-free FlowMatch grid from one through one-over-steps."""

    count = _require_integer_count(steps, minimum=1, label="steps")
    if domain is not SigmaDomain.UNIT_FLOW:
        raise ScheduleContractError("FlowMatch reciprocal-step grid requires the UNIT_FLOW domain")
    return tuple((count - index) / count for index in range(count))


def linear_endpoint_grid(
    *,
    points: int,
    start: float,
    end: float,
    domain: SigmaDomain,
) -> tuple[float, ...]:
    """Build a generic finite descending grid that includes both endpoints."""

    count = _require_integer_count(points, minimum=2, label="points")
    if domain is SigmaDomain.MODEL_NATIVE or not isinstance(domain, SigmaDomain):
        raise ScheduleContractError(
            "linear endpoint grid cannot construct the opaque MODEL_NATIVE domain"
        )
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(float(start))
        or not math.isfinite(float(end))
    ):
        raise ScheduleContractError("linear grid endpoints must be finite numbers")
    if start <= end:
        raise ScheduleContractError("linear grid start must be greater than end")

    interval = (float(end) - float(start)) / (count - 1)
    return tuple(float(start) + interval * index for index in range(count))
