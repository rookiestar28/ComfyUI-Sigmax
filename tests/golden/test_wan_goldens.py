"""Numerical golden vectors for the source-qualified Wan family."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

import pytest


def _module() -> Any:
    import importlib

    return importlib.import_module("comfyui_sigmax.profiles.wan")


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


_FIXTURE = Path(__file__).with_name("wan_v1.json")
_PAYLOAD = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_wan_golden_source_revisions_are_pinned() -> None:
    module = _module()
    assert _PAYLOAD["schema"] == "sigmax.wan-golden/1"
    assert _PAYLOAD["source_revisions"] == {
        "official_wan21": module.WAN21_REPOSITORY_REVISION,
        "official_wan22": module.WAN22_REPOSITORY_REVISION,
        "comfyui": module.WAN_COMFYUI_REVISION,
        "diffusers": module.WAN_DIFFUSERS_REVISION,
    }
    assert len(_PAYLOAD["cases"]) == 13


def _profile_id(module: Any, value: str) -> Any:
    return module.WanProfileId(value)


def _resolution(module: Any, value: str) -> Any:
    return module.WanResolution(value)


@pytest.mark.parametrize("case", _PAYLOAD.get("cases", []), ids=lambda item: item["profile"])
def test_wan_float64_vectors_match_clean_room_reference(case: dict[str, object]) -> None:
    module = _module()
    profile = _profile_id(module, cast(str, case["profile"]))
    result = module.build_wan_schedule(
        profile=profile,
        steps=cast(int, case["steps"]),
        resolution=_resolution(module, cast(str, case["resolution"])),
        strict_source=True,
    )
    expected = tuple(cast(float, value) for value in cast(list[object], case["float64"]))
    assert len(cast(list[object], case["float32"])) == len(expected)
    assert result.sigmas == pytest.approx(expected, abs=1e-15)
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0


@pytest.mark.parametrize("case", _PAYLOAD.get("cases", []), ids=lambda item: item["profile"])
def test_wan_float32_vectors_preserve_float64_reference(case: dict[str, object]) -> None:
    module = _module()
    profile = _profile_id(module, cast(str, case["profile"]))
    result = module.build_wan_schedule(
        profile=profile,
        steps=cast(int, case["steps"]),
        resolution=_resolution(module, cast(str, case["resolution"])),
        strict_source=True,
    )
    expected = tuple(cast(float, value) for value in cast(list[object], case["float32"]))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("case", _PAYLOAD.get("cases", []), ids=lambda item: item["profile"])
def test_wan_golden_case_declares_profile_shift_and_terminal_contract(
    case: dict[str, object],
) -> None:
    module = _module()
    profile = _profile_id(module, cast(str, case["profile"]))
    schema = module._SCHEMAS_BY_PROFILE[profile]
    ratio = next(field.value for field in schema.parameters if field.name == "shift")
    assert ratio == case["ratio"]
    assert schema.base_grid.identifier == "flowmatch.reciprocal_step"
    assert schema.terminal.policy.name == "APPEND_ZERO"
    assert schema.terminal.value == 0.0
    assert schema.slicing.supports_step_range is True
