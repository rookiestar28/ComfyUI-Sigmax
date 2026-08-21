"""Float64 and float32 golden evidence for M4-18."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import cast

from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles.wan import (
    WAN_ANIMATE2_COMFY_WORKFLOW_REVISION,
    WAN_ANIMATE2_COMFYUI_REVISION,
    WanProfileId,
    WanResolution,
    build_wan_schedule,
)

_PAYLOAD = json.loads(Path(__file__).with_name("wan_m4_18_v1.json").read_text(encoding="utf-8"))


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", value))[0])


def test_m4_18_golden_vectors_and_fingerprints_are_exact() -> None:
    result = build_wan_schedule(
        profile=WanProfileId(_PAYLOAD["profile_id"]),
        steps=_PAYLOAD["steps"],
        resolution=WanResolution(_PAYLOAD["resolution"]),
        strict_source=True,
    )
    expected64 = tuple(_PAYLOAD["sigmas"]["float64"])
    expected32 = tuple(_PAYLOAD["sigmas"]["float32"])
    actual32 = tuple(_float32(value) for value in result.sigmas)
    assert max(abs(a - b) for a, b in zip(result.sigmas, expected64, strict=True)) <= 1e-15
    assert max(abs(a - b) for a, b in zip(actual32, expected32, strict=True)) <= 1e-7
    assert (
        numerical_fingerprint(
            result.sigmas,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float64",
        )
        == _PAYLOAD["fingerprints"]["float64"]
    )
    assert (
        numerical_fingerprint(
            result.sigmas,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float32",
        )
        == _PAYLOAD["fingerprints"]["float32"]
    )


def test_m4_18_golden_source_pins_are_current() -> None:
    assert _PAYLOAD["schema"] == "sigmax.wan-m4-18-golden/1"
    pins = _PAYLOAD["source_revisions"]
    assert "".join(pins["comfyui_chunks"]) == WAN_ANIMATE2_COMFYUI_REVISION
    assert "".join(pins["workflow_templates_chunks"]) == WAN_ANIMATE2_COMFY_WORKFLOW_REVISION
