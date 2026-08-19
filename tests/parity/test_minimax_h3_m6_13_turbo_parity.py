"""Independent Decimal parity oracle for the pinned ModelTC MiniMax H3 Turbo recipe grid."""

from __future__ import annotations

import json
import struct
from decimal import Decimal, localcontext
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.profiles.minimax_h3_turbo import (
    MINIMAX_H3_TURBO_MODELTECH_REVISION,
    MINIMAX_H3_TURBO_PROFILES,
    build_minimax_h3_turbo_schedule,
)

FIXTURE_PATH = Path(__file__).parents[1] / "golden" / "minimax_h3_turbo_v1.json"
_F32 = ">f"


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _f32(value: Decimal) -> float:
    return float(struct.unpack(_F32, struct.pack(_F32, float(value)))[0])


def _reference_vector(nfe: int, shift: float, precision: str) -> tuple[float, ...]:
    # Clean-room transcription of ModelTC's q_i=(N-i)/N direct-ratio contract; no project
    # scheduler helper is called here so the parity check can detect a shared implementation bug.
    with localcontext() as context:
        context.prec = 80
        n = Decimal(nfe)
        ratio = Decimal(str(shift))
        values: list[float] = []
        for index in range(nfe):
            q = (n - Decimal(index)) / n
            value = ratio * q / (Decimal(1) + (ratio - Decimal(1)) * q)
            values.append(float(value) if precision == "float64" else _f32(value))
        values.append(0.0)
        return tuple(values)


def _error(actual: tuple[float, ...], expected: tuple[float, ...]) -> tuple[float, float]:
    errors = [abs(left - right) for left, right in zip(actual, expected, strict=True)]
    return max(errors), sum(errors) / len(errors)


def test_pinned_modeltc_revision_and_fixture_schema() -> None:
    fixture = _fixture()
    assert "".join(fixture["source_revision_chunks"]) == MINIMAX_H3_TURBO_MODELTECH_REVISION
    assert fixture["schema"] == "sigmax.minimax-h3-turbo-golden/1"
    assert fixture["tolerances"] == {"float64_max_abs": "1e-8", "float32_max_abs": "1e-6"}


@pytest.mark.parametrize("profile", MINIMAX_H3_TURBO_PROFILES, ids=lambda p: p.recipe_id)
def test_modeltc_decimal_parity_has_bounded_max_and_mean_error(profile: Any) -> None:
    for nfe in profile.allowed_nfe:
        result64 = build_minimax_h3_turbo_schedule(profile.recipe_id, nfe=nfe, precision="float64")
        result32 = build_minimax_h3_turbo_schedule(profile.recipe_id, nfe=nfe, precision="float32")
        reference_video64 = _reference_vector(nfe, profile.video_shift, "float64")
        reference_audio64 = _reference_vector(nfe, profile.audio_shift, "float64")
        reference_video32 = _reference_vector(nfe, profile.video_shift, "float32")
        reference_audio32 = _reference_vector(nfe, profile.audio_shift, "float32")
        assert _error(result64.video_sigmas, reference_video64) <= (1e-15, 1e-15)
        assert _error(result64.audio_sigmas, reference_audio64) <= (1e-15, 1e-15)
        assert _error(result32.video_sigmas, reference_video32) <= (1e-6, 1e-6)
        assert _error(result32.audio_sigmas, reference_audio32) <= (1e-6, 1e-6)
        assert all(left >= right for left, right in pairwise(result64.video_sigmas))
        assert all(left >= right for left, right in pairwise(result64.audio_sigmas))
