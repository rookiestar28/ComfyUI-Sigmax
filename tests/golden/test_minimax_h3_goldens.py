"""Pinned MiniMax H3 numerical goldens for both schedule lanes and audio diagnostics."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_BASE_FL2VA_PROFILE,
    MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT,
    MINIMAX_H3_COMFYUI_REVISION,
    MINIMAX_H3_DIFFUSERS_REVISION,
    MINIMAX_H3_HF_REVISION,
    MiniMaxH3Variant,
    build_minimax_h3_comfyui_simple_schedule,
    build_minimax_h3_schedule,
    map_minimax_h3_audio_coordinate,
)

FIXTURE = Path(__file__).with_name("minimax_h3_v1.json")


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_fixture_is_source_pinned_and_complete() -> None:
    fixture = _fixture()
    assert fixture["schema"] == "sigmax.minimax-h3-golden/1"
    assert fixture["source_revisions"] == {
        "comfyui": MINIMAX_H3_COMFYUI_REVISION,
        "comfyui_h3_implementation": MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT,
        "diffusers": MINIMAX_H3_DIFFUSERS_REVISION,
        "huggingface": MINIMAX_H3_HF_REVISION,
    }
    assert len(MINIMAX_H3_BASE_FL2VA_PROFILE.references) == 4


def test_diffusers_endpoint_golden_matches_float64_and_float32() -> None:
    case = _fixture()["diffusers_endpoint"]
    float64 = build_minimax_h3_schedule(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        grid_points=case["grid_points"],
        precision="float64",
    )
    float32 = build_minimax_h3_schedule(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        grid_points=case["grid_points"],
        precision="float32",
    )
    assert float64.sigmas == pytest.approx(tuple(case["float64"]), abs=1e-15)
    assert float32.sigmas == tuple(case["float32"])
    assert (
        numerical_fingerprint(float64.sigmas, domain=SigmaDomain.UNIT_FLOW, precision="float64")
        == case["float64_fingerprint"]
    )
    assert (
        numerical_fingerprint(float32.sigmas, domain=SigmaDomain.UNIT_FLOW, precision="float64")
        == case["float32_fingerprint"]
    )
    assert all(left > right for left, right in pairwise(float32.sigmas))


def test_comfyui_simple_golden_is_not_the_diffusers_grid() -> None:
    case = _fixture()["comfyui_simple"]
    result = build_minimax_h3_comfyui_simple_schedule(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        transitions=case["transitions"],
    )
    assert result.sigmas == tuple(case["sigmas"])
    assert (
        numerical_fingerprint(result.sigmas, domain=SigmaDomain.UNIT_FLOW, precision="float64")
        == case["fingerprint"]
    )
    assert result.request.base_grid is not None
    assert result.request.base_grid.identifier == "comfyui.discrete_flow_1000"


def test_audio_mapping_golden_preserves_model_ownership() -> None:
    for case in _fixture()["audio_mapping_float64"]:
        result = map_minimax_h3_audio_coordinate(case["video"], precision="float64")
        assert result.base_coordinate == pytest.approx(case["base"], abs=1e-15)
        assert result.audio_sigma == pytest.approx(case["audio"], abs=1e-15)
        assert result.derivative == pytest.approx(case["derivative"], abs=1e-15)
