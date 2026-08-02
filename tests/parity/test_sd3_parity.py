"""Clean-room parity checks for the pinned original-SD3 schedule equations."""

from __future__ import annotations

import struct
from itertools import pairwise
from typing import cast

import pytest
from comfyui_sigmax.profiles.sd3 import (
    SD3_COMFY_DIFFUSERS_PROFILE,
    SD3_PUBLISHER_REFERENCE_PROFILE,
    SD3ShiftMode,
    build_sd3_schedule,
)

pytestmark = pytest.mark.parity

_COMFYUI_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_DIFFUSERS_REVISION = "cc92165331e1b20afc1a47e03f63e8f3a930f8cc"  # pragma: allowlist secret
_PUBLISHER_REVISION = "8565799a3b41eb0c7ba976d18375f0f753f56402"  # pragma: allowlist secret


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


def _pinned_flowmatch_equation(steps: int, ratio: float) -> tuple[float, ...]:
    """Independent clean-room expression of the pinned FlowMatch direct-ratio path."""

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


@pytest.mark.parametrize(
    ("mode", "steps", "ratio"),
    (
        (SD3ShiftMode.PUBLISHER_REFERENCE, 50, 1.0),
        (SD3ShiftMode.COMFY_DIFFUSERS_FIXED, 28, 3.0),
    ),
)
def test_sd3_matches_pinned_framework_equation_float64(
    mode: SD3ShiftMode, steps: int, ratio: float
) -> None:
    result = build_sd3_schedule(mode=mode, steps=steps, strict_source=True)
    expected = _pinned_flowmatch_equation(steps, ratio)
    assert result.sigmas == pytest.approx(expected, abs=1e-15)
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(current > following for current, following in pairwise(result.sigmas))


@pytest.mark.parametrize(
    ("mode", "steps", "ratio"),
    (
        (SD3ShiftMode.PUBLISHER_REFERENCE, 50, 1.0),
        (SD3ShiftMode.COMFY_DIFFUSERS_FIXED, 28, 3.0),
    ),
)
def test_sd3_matches_pinned_framework_equation_float32(
    mode: SD3ShiftMode, steps: int, ratio: float
) -> None:
    result = build_sd3_schedule(mode=mode, steps=steps, strict_source=True)
    expected = tuple(_float32(value) for value in _pinned_flowmatch_equation(steps, ratio))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_sd3_reference_revisions_and_profile_lanes_are_pinned() -> None:
    publisher_sources = {
        reference.lane: reference for reference in SD3_PUBLISHER_REFERENCE_PROFILE.references
    }
    framework_sources = {
        reference.lane: reference for reference in SD3_COMFY_DIFFUSERS_PROFILE.references
    }
    assert publisher_sources["official_github"].revision == _PUBLISHER_REVISION
    assert framework_sources["comfyui_implementation"].revision == _COMFYUI_REVISION
    assert framework_sources["diffusers_framework"].revision == _DIFFUSERS_REVISION
