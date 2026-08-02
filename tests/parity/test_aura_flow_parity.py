"""Independent clean-room parity checks for the pinned AuraFlow v0.2 equation."""

from __future__ import annotations

import struct
from itertools import pairwise
from typing import cast

import pytest
from comfyui_sigmax.profiles.aura_flow import (
    AURAFLOW_V02_PROFILE,
    AuraFlowShiftMode,
    build_aura_flow_schedule,
)

pytestmark = pytest.mark.parity

_COMFYUI_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_DIFFUSERS_REVISION = "3c468926ffd12b69baa4316e27b09306b8da19a6"  # pragma: allowlist secret
_HF_REVISION = "ea13150f559b7f85d2c5959297f7de10325584b4"  # pragma: allowlist secret


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


def _pinned_flowmatch_equation(steps: int, ratio: float) -> tuple[float, ...]:
    values = tuple((steps - index) / steps for index in range(steps))
    shifted = tuple(
        0.0
        if value == 0.0
        else 1.0
        if value == 1.0
        else 1.0 / (1.0 + (1.0 - value) / (ratio * value))
        for value in values
    )
    return (*shifted, 0.0)


@pytest.mark.parametrize("steps", (8, 28, 50))
def test_auraflow_matches_pinned_equation_float64(steps: int) -> None:
    result = build_aura_flow_schedule(
        mode=AuraFlowShiftMode.OFFICIAL_FIXED,
        steps=steps,
        strict_source=False,
    )
    expected = _pinned_flowmatch_equation(steps, 1.73)
    assert result.sigmas == pytest.approx(expected, abs=1e-15)
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(current > following for current, following in pairwise(result.sigmas))


def test_auraflow_matches_pinned_equation_float32() -> None:
    result = build_aura_flow_schedule(
        mode=AuraFlowShiftMode.OFFICIAL_FIXED,
        steps=50,
        strict_source=True,
    )
    expected = tuple(_float32(value) for value in _pinned_flowmatch_equation(50, 1.73))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_auraflow_reference_revisions_and_lanes_are_pinned() -> None:
    sources = {reference.lane: reference for reference in AURAFLOW_V02_PROFILE.references}
    assert sources["comfyui_implementation"].revision == _COMFYUI_REVISION
    assert sources["diffusers_framework"].revision == _DIFFUSERS_REVISION
    assert sources["official_huggingface"].revision == _HF_REVISION
