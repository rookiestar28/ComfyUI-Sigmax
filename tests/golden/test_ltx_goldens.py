"""Numerical golden vectors for the source-qualified LTX family."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles import ltx

_FIXTURE = Path(__file__).with_name("ltx_v1.json")
_PAYLOAD = cast(dict[str, Any], json.loads(_FIXTURE.read_text(encoding="utf-8")))


def _float32(value: float) -> float:
    return cast(float, struct.unpack("<f", struct.pack("<f", float(value)))[0])


def test_ltx_golden_source_revisions_are_pinned() -> None:
    assert _PAYLOAD["schema"] == "sigmax.ltx-golden/1"
    assert _PAYLOAD["source_revisions"] == {
        "ltxv_repository": ltx.LTXV_REPOSITORY_REVISION,
        "ltxv_model": ltx.LTXV_MODEL_REVISION,
        "ltx2_repository": ltx.LTX2_REPOSITORY_REVISION,
        "ltx2_model": ltx.LTX2_MODEL_REVISION,
        "ltx23_model": ltx.LTX23_MODEL_REVISION,
        "comfyui": ltx.LTX_COMFYUI_REVISION,
        "diffusers": ltx.LTX_DIFFUSERS_REVISION,
    }


@pytest.mark.parametrize("case", _PAYLOAD["cases"], ids=lambda item: item["profile"])
def test_ltx_float64_vectors_and_fingerprints_match_golden(case: dict[str, Any]) -> None:
    result = ltx.build_ltx_schedule(
        profile=ltx.LTXProfileId(case["profile"]),
        steps=case["steps"],
        token_count=case["token_count"],
    )
    expected = tuple(cast(float, value) for value in case["float64"])
    assert result.sigmas == pytest.approx(expected, abs=1e-15)
    assert (
        numerical_fingerprint(result.sigmas, domain=SigmaDomain.UNIT_FLOW, precision="float64")
        == case["fingerprint"]
    )


@pytest.mark.parametrize("case", _PAYLOAD["cases"], ids=lambda item: item["profile"])
def test_ltx_float32_vectors_match_golden(case: dict[str, Any]) -> None:
    result = ltx.build_ltx_schedule(
        profile=ltx.LTXProfileId(case["profile"]),
        steps=case["steps"],
        token_count=case["token_count"],
    )
    expected = tuple(cast(float, value) for value in case["float32"])
    actual = tuple(_float32(value) for value in result.sigmas)
    assert actual == pytest.approx(expected, abs=1e-7)
