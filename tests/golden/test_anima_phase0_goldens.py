"""Phase 0 RED golden vectors for Anima's fixed rational shift."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

import pytest


def _module() -> Any:
    import importlib

    return importlib.import_module("comfyui_sigmax.profiles.anima")


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


_FIXTURE = Path(__file__).with_name("anima_v1.json")


def test_anima_golden_source_revisions_are_pinned() -> None:
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema"] == "sigmax.anima-golden/1"
    assert fixture["source_revisions"] == {
        "official_model_card": "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b",
        "official_diffusers_base": "073c3a9db359c31ad0e8aa268d15775473c2176c",
        "comfyui": "5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25",
    }


@pytest.mark.parametrize(
    "case",
    json.loads(_FIXTURE.read_text(encoding="utf-8")).get("cases", []),
)
def test_anima_float64_vectors_are_complete(case: dict[str, object]) -> None:
    module = _module()
    builder = getattr(module, "build_anima_schedule", None)
    assert callable(builder)
    variant = module.AnimaVariant[str(case["variant"]).upper()]
    result = builder(variant=variant, steps=cast(int, case["steps"]))
    expected = tuple(cast(float, value) for value in cast(list[object], case["float64"]))
    assert len(cast(list[object], case["float32"])) == len(expected)
    assert result.sigmas == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    "case",
    json.loads(_FIXTURE.read_text(encoding="utf-8")).get("cases", []),
)
def test_anima_float32_vectors_preserve_float64_reference(case: dict[str, object]) -> None:
    module = _module()
    builder = getattr(module, "build_anima_schedule", None)
    assert callable(builder)
    variant = module.AnimaVariant[str(case["variant"]).upper()]
    result = builder(variant=variant, steps=cast(int, case["steps"]))
    expected = tuple(cast(float, value) for value in cast(list[object], case["float32"]))
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-6)
