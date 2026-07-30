"""Dependency-free pointwise transforms for unit-flow schedules.

Formula references:

- https://github.com/krea-ai/krea-2/blob/main/sampling.py
- https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_sampling.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
"""

from __future__ import annotations

import math

from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    SigmaDomain,
)


def _require_unit_flow_values(
    values: tuple[float, ...],
    *,
    domain: SigmaDomain,
) -> tuple[float, ...]:
    if domain is not SigmaDomain.UNIT_FLOW:
        raise ScheduleContractError("shift transforms require the UNIT_FLOW domain")
    if not isinstance(values, tuple) or not values:
        raise ScheduleContractError("shift values must be a non-empty tuple")

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ScheduleContractError(
                "shift values must contain finite unit-flow numbers in [0, 1]"
            )
    return values


def _require_finite_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ScheduleContractError(f"{label} must be a finite number")
    return float(value)


def _logistic(log_odds: float) -> float:
    if log_odds >= 0.0:
        return 1.0 / (1.0 + math.exp(-log_odds))
    exponential = math.exp(log_odds)
    return exponential / (1.0 + exponential)


def exponential_mu_shift(
    values: tuple[float, ...],
    *,
    mu: float,
    exponent: float = 1.0,
    domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
) -> tuple[float, ...]:
    """Apply the Krea/Flux exponential-mu time shift to unit-flow values."""

    unit_values = _require_unit_flow_values(values, domain=domain)
    finite_mu = _require_finite_number(mu, label="mu")
    finite_exponent = _require_finite_number(exponent, label="exponent")
    if finite_exponent <= 0.0:
        raise ScheduleContractError("exponent must be greater than zero")

    def transform(value: float) -> float:
        if value == 0.0 or value == 1.0:
            return float(value)
        log_inverse_odds = math.log((1.0 - value) / value)
        return _logistic(finite_mu - finite_exponent * log_inverse_odds)

    return tuple(transform(float(value)) for value in unit_values)


def direct_ratio_shift(
    values: tuple[float, ...],
    *,
    ratio: float,
    domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
) -> tuple[float, ...]:
    """Apply the discrete-flow direct-ratio time shift to unit-flow values."""

    unit_values = _require_unit_flow_values(values, domain=domain)
    finite_ratio = _require_finite_number(ratio, label="ratio")
    if finite_ratio <= 0.0:
        raise ScheduleContractError("ratio must be greater than zero")

    def transform(value: float) -> float:
        if value == 0.0 or value == 1.0:
            return float(value)
        if finite_ratio > 1.0:
            return 1.0 / (1.0 + (1.0 - value) / (finite_ratio * value))
        numerator = finite_ratio * value
        return numerator / ((1.0 - value) + numerator)

    return tuple(transform(float(value)) for value in unit_values)


def no_shift(
    values: tuple[float, ...],
    *,
    domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
) -> tuple[float, ...]:
    """Validate and explicitly preserve an unshifted unit-flow schedule."""

    return _require_unit_flow_values(values, domain=domain)
