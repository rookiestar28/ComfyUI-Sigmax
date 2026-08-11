"""Independent float64/float32 golden vectors for M6-11 Wan Animate profiles."""

from __future__ import annotations

import importlib
import json
import math
import struct
from itertools import pairwise
from pathlib import Path
from typing import cast

import pytest

_PAYLOAD = json.loads(Path(__file__).with_name("wan_m6_11_v1.json").read_text(encoding="utf-8"))


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


@pytest.mark.parametrize(
    ("profile_id", "case"),
    _PAYLOAD["profiles"].items(),
    ids=tuple(_PAYLOAD["profiles"]),
)
def test_m6_11_float64_and_float32_vectors_match_independent_reference(
    profile_id: str,
    case: dict[str, str],
) -> None:
    module = importlib.import_module("comfyui_sigmax.profiles.wan")
    vector = _PAYLOAD["vectors"][case["vector"]]
    expected64 = tuple(cast(float, value) for value in vector["float64"])
    expected32 = tuple(cast(float, value) for value in vector["float32"])
    result = module.build_wan_schedule(
        profile=module.WanProfileId(profile_id),
        steps=len(expected64) - 1,
        resolution=module.WanResolution(case["resolution"]),
        strict_source=True,
    )
    actual32 = tuple(_float32(value) for value in result.sigmas)
    errors64 = tuple(
        abs(actual - expected) for actual, expected in zip(result.sigmas, expected64, strict=True)
    )
    errors32 = tuple(
        abs(actual - expected) for actual, expected in zip(actual32, expected32, strict=True)
    )
    assert len(result.sigmas) == len(expected64) == len(expected32)
    assert max(errors64) <= float(_PAYLOAD["tolerances"]["float64_max_abs"])
    assert sum(errors64) / len(errors64) <= float(_PAYLOAD["tolerances"]["float64_max_abs"])
    assert max(errors32) <= float(_PAYLOAD["tolerances"]["float32_max_abs"])
    assert sum(errors32) / len(errors32) <= float(_PAYLOAD["tolerances"]["float32_max_abs"])
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result.sigmas)
    assert all(left > right for left, right in pairwise(result.sigmas))
    assert result.sigmas[0] == 1.0 and result.sigmas[-1] == 0.0


def test_m6_11_golden_source_pins_are_exact() -> None:
    module = importlib.import_module("comfyui_sigmax.profiles.wan")
    assert _PAYLOAD["schema"] == "sigmax.wan-m6-11-golden/1"
    assert _PAYLOAD["source_revisions"] == {
        "official_wan22": module.WAN22_REPOSITORY_REVISION,
        "official_wan_animate2": module.WAN_ANIMATE2_REPOSITORY_REVISION,
    }
    assert len(_PAYLOAD["profiles"]) == 3
