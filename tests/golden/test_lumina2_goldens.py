"""Independent golden vectors for the Lumina-Image 2.0 fixed shift."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import pytest
from comfyui_sigmax.profiles.lumina2 import Lumina2ShiftMode, build_lumina2_schedule


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "lumina2_v2.json").read_text()).get("cases", []),
)
def test_lumina2_float64_golden(case: dict[str, object]) -> None:
    result = build_lumina2_schedule(
        mode=Lumina2ShiftMode(str(case["mode"])),
        steps=cast(int, case["steps"]),
        strict_source=False,
    )
    assert result.sigmas == pytest.approx(
        tuple(cast(float, value) for value in cast(list[object], case["float64"])),
        abs=1e-15,
    )


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "lumina2_v2.json").read_text()).get("cases", []),
)
def test_lumina2_float32_golden(case: dict[str, object]) -> None:
    result = build_lumina2_schedule(
        mode=Lumina2ShiftMode(str(case["mode"])),
        steps=cast(int, case["steps"]),
        strict_source=False,
    )
    expected = tuple(_float32(cast(float, value)) for value in cast(list[object], case["float64"]))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-7)
