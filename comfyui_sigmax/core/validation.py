"""Numerical validation for complete externally constructed sigma schedules."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import pairwise

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError, SigmaDomain


def validate_sigma_schedule(
    sigmas: Iterable[float],
    *,
    domain: SigmaDomain,
    expected_steps: int,
    require_terminal_zero: bool,
) -> tuple[float, ...]:
    """Validate and return one immutable terminal-inclusive sigma schedule.

    ``expected_steps`` is the number of integration transitions, so the schedule must contain
    exactly ``expected_steps + 1`` values.
    """

    if (
        not isinstance(expected_steps, int)
        or isinstance(expected_steps, bool)
        or expected_steps <= 0
    ):
        raise ScheduleContractError("expected_steps must be a positive integer")
    if not isinstance(domain, SigmaDomain):
        raise ScheduleContractError("unsupported sigma domain")
    if domain is SigmaDomain.MODEL_NATIVE:
        raise ScheduleContractError("MODEL_NATIVE is opaque and cannot be externally validated")
    if not isinstance(require_terminal_zero, bool):
        raise ScheduleContractError("require_terminal_zero must be boolean")

    raw_values = tuple(sigmas)
    if len(raw_values) != expected_steps + 1:
        raise ScheduleContractError(
            f"schedule length must be expected_steps + 1 ({expected_steps + 1})"
        )

    values: list[float] = []
    for value in raw_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ScheduleContractError("sigmas must contain numeric values")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ScheduleContractError("sigmas must contain only finite values")
        values.append(normalized)

    if domain is SigmaDomain.UNIT_FLOW:
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ScheduleContractError("sigma value is outside the UNIT_FLOW domain")
    elif domain in {
        SigmaDomain.CONTINUOUS_EDM,
        SigmaDomain.DISCRETE_TRAINING_INDEX,
    } and any(value < 0.0 for value in values):
        raise ScheduleContractError(f"sigma value is outside the {domain.value} domain")

    if any(current <= following for current, following in pairwise(values)):
        raise ScheduleContractError("sigmas must be strictly decreasing")

    if require_terminal_zero and values[-1] != 0.0:
        raise ScheduleContractError("schedule must end with terminal zero")

    return tuple(values)
