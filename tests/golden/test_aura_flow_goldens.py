"""Independent golden vectors for the original AuraFlow v0.2 fixed shift."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

import pytest
from comfyui_sigmax.profiles.aura_flow import AuraFlowShiftMode, build_aura_flow_schedule


@pytest.mark.parametrize(
    "case",
    json.loads((Path(__file__).parent / "aura_flow_v0_2.json").read_text()).get("cases", []),
)
def test_auraflow_float64_golden(case: dict[str, object]) -> None:
    result = build_aura_flow_schedule(
        mode=AuraFlowShiftMode(str(case["mode"])),
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
    json.loads((Path(__file__).parent / "aura_flow_v0_2.json").read_text()).get("cases", []),
)
def test_auraflow_float32_golden(case: dict[str, object]) -> None:
    result = build_aura_flow_schedule(
        mode=AuraFlowShiftMode(str(case["mode"])),
        steps=cast(int, case["steps"]),
        strict_source=False,
    )
    expected = tuple(_float32(cast(float, value)) for value in cast(list[object], case["float64"]))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-7)
