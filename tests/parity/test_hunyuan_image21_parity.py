"""Independent clean-room parity checks for HunyuanImage 2.1 equations."""

from __future__ import annotations

import struct
from itertools import pairwise
from typing import cast

import pytest
from comfyui_sigmax.profiles.hunyuan_image21 import (
    HUNYUAN_IMAGE21_BASE_PROFILE,
    HUNYUAN_IMAGE21_DISTILLED_PROFILE,
    HunyuanImage21Variant,
    build_hunyuan_image21_schedule,
)

pytestmark = pytest.mark.parity

_PUBLISHER_REVISION = "307df8801d176740dafb67b2872c831cb9362cf9"  # pragma: allowlist secret
_HF_REVISION = "e435da11d9e8795a25e224c5ba27b099ed45c55b"  # pragma: allowlist secret
_COMFYUI_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


def _pinned_flowmatch_equation(steps: int, ratio: float) -> tuple[float, ...]:
    values = tuple((steps - index) / steps for index in range(steps))
    shifted = tuple(ratio * value / (1.0 + (ratio - 1.0) * value) for value in values)
    return (*shifted, 0.0)


@pytest.mark.parametrize(
    ("variant", "steps", "ratio"),
    [
        (HunyuanImage21Variant.BASE, 8, 5.0),
        (HunyuanImage21Variant.BASE, 50, 5.0),
        (HunyuanImage21Variant.DISTILLED, 8, 4.0),
        (HunyuanImage21Variant.DISTILLED, 36, 4.0),
    ],
)
def test_hunyuan_image21_matches_pinned_equation_float64(
    variant: HunyuanImage21Variant,
    steps: int,
    ratio: float,
) -> None:
    result = build_hunyuan_image21_schedule(variant=variant, steps=steps, strict_source=False)
    expected = _pinned_flowmatch_equation(steps, ratio)
    assert result.sigmas == pytest.approx(expected, abs=1e-15)
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(current > following for current, following in pairwise(result.sigmas))


@pytest.mark.parametrize("variant", (HunyuanImage21Variant.BASE, HunyuanImage21Variant.DISTILLED))
def test_hunyuan_image21_matches_pinned_equation_float32(variant: HunyuanImage21Variant) -> None:
    steps = 50 if variant is HunyuanImage21Variant.BASE else 8
    ratio = 5.0 if variant is HunyuanImage21Variant.BASE else 4.0
    result = build_hunyuan_image21_schedule(variant=variant, steps=steps, strict_source=True)
    expected = tuple(_float32(value) for value in _pinned_flowmatch_equation(steps, ratio))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_hunyuan_image21_reference_revisions_and_lanes_are_pinned() -> None:
    for profile in (HUNYUAN_IMAGE21_BASE_PROFILE, HUNYUAN_IMAGE21_DISTILLED_PROFILE):
        sources = {reference.lane: reference for reference in profile.references}
        assert sources["official_github"].revision == _PUBLISHER_REVISION
        assert sources["official_huggingface"].revision == _HF_REVISION
        assert sources["comfyui_implementation"].revision == _COMFYUI_REVISION


def test_hunyuan_image21_distilled_native_host_limitation_is_explicit() -> None:
    assert any(
        "native" in value.lower() and "distilled" in value.lower()
        for value in HUNYUAN_IMAGE21_DISTILLED_PROFILE.schema.known_limitations
    )
