"""Independent pinned-source parity for the M4-18 ComfyUI simple six-step recipe."""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest
from comfyui_sigmax.profiles.wan import (
    WAN_ANIMATE2_COMFY_OPTIMIZED_6_PROFILE,
    WanProfileId,
    WanResolution,
    build_wan_schedule,
)

pytestmark = pytest.mark.parity


def _pinned_comfy_reference() -> tuple[float, ...]:
    with localcontext() as context:
        context.prec = 80
        table_length = 1000
        steps = 6
        values = []
        for index in range(steps):
            table_index = table_length - 1 - int(index * table_length / steps)
            timestep = Decimal(table_index + 1) / Decimal(table_length)
            values.append(float(Decimal(5) * timestep / (Decimal(1) + Decimal(4) * timestep)))
        return (*values, 0.0)


def test_exact_comfyui_simple_six_step_parity() -> None:
    expected = _pinned_comfy_reference()
    actual = build_wan_schedule(
        profile=WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6,
        steps=6,
        resolution=WanResolution.P480,
        strict_source=True,
    ).sigmas
    errors = tuple(abs(left - right) for left, right in zip(actual, expected, strict=True))
    assert max(errors) <= 1e-15
    assert sum(errors) / len(errors) <= 2e-16
    assert WAN_ANIMATE2_COMFY_OPTIMIZED_6_PROFILE.schema.frameworks[0].revision == (
        "76135e557da1ec7dcb270160f01e597565e3e003"  # pragma: allowlist secret
    )
