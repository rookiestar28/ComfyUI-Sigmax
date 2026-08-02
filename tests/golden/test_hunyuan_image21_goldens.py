"""Independent golden vectors for HunyuanImage 2.1 Base/Distilled shifts."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import pytest
from comfyui_sigmax.profiles.hunyuan_image21 import (
    HunyuanImage21Variant,
    build_hunyuan_image21_schedule,
)


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "hunyuan_image21_v1.json").read_text()).get("cases", []),
)
def test_hunyuan_image21_float64_golden(case: dict[str, object]) -> None:
    result = build_hunyuan_image21_schedule(
        variant=HunyuanImage21Variant(str(case["variant"])),
        steps=cast(int, case["steps"]),
    )
    assert result.sigmas == pytest.approx(
        tuple(cast(float, value) for value in cast(list[object], case["float64"])),
        abs=1e-15,
    )


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "hunyuan_image21_v1.json").read_text()).get("cases", []),
)
def test_hunyuan_image21_float32_golden(case: dict[str, object]) -> None:
    result = build_hunyuan_image21_schedule(
        variant=HunyuanImage21Variant(str(case["variant"])),
        steps=cast(int, case["steps"]),
    )
    expected = tuple(_float32(cast(float, value)) for value in cast(list[object], case["float64"]))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-7)
