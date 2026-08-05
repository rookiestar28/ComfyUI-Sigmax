"""MiniMax H3 pure-core, paired-coordinate, and source-parity contracts."""

from __future__ import annotations

from itertools import pairwise

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_BASE_FL2VA_PROFILE,
    MINIMAX_H3_BASE_FL2VA_SCHEMA,
    MINIMAX_H3_BASE_REF2VA_PROFILE,
    MINIMAX_H3_BASE_REF2VA_SCHEMA,
    MINIMAX_H3_COMFYUI_REVISION,
    MINIMAX_H3_DIFFUSERS_REVISION,
    MINIMAX_H3_HF_REVISION,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3ScheduleLane,
    MiniMaxH3Variant,
    MiniMaxH3VelocityDirection,
    build_minimax_h3_comfyui_simple_schedule,
    build_minimax_h3_schedule,
    map_minimax_h3_audio_coordinate,
    minimax_h3_velocity_conversion_sign,
)


def test_minimax_h3_profiles_are_explicit_and_source_pinned() -> None:
    assert MINIMAX_H3_BASE_FL2VA_SCHEMA.profile_id == "minimax-h3.base_fl2va"
    assert MINIMAX_H3_BASE_REF2VA_SCHEMA.profile_id == "minimax-h3.base_ref2va"
    assert MINIMAX_H3_BASE_FL2VA_PROFILE.variant is MiniMaxH3Variant.BASE_FL2VA
    assert MINIMAX_H3_BASE_REF2VA_PROFILE.variant is MiniMaxH3Variant.BASE_REF2VA
    assert MINIMAX_H3_BASE_FL2VA_SCHEMA.evidence.value == "framework_reference"
    assert MINIMAX_H3_HF_REVISION in {
        reference.revision for reference in MINIMAX_H3_BASE_FL2VA_PROFILE.references
    }
    assert MINIMAX_H3_DIFFUSERS_REVISION in {
        framework.revision for framework in MINIMAX_H3_BASE_FL2VA_SCHEMA.frameworks
    }
    assert MINIMAX_H3_COMFYUI_REVISION in {
        framework.revision for framework in MINIMAX_H3_BASE_FL2VA_SCHEMA.frameworks
    }
    assert MINIMAX_H3_BASE_FL2VA_SCHEMA.software_sources[0].license.identifier == (
        "LicenseRef-MiniMax-H3-Community"
    )
    assert any(
        field.name == "license_boundary" and field.value == "code_only_no_weight_redistribution"
        for field in MINIMAX_H3_BASE_FL2VA_SCHEMA.parameters
    )
    assert "weight redistribution" in " ".join(MINIMAX_H3_BASE_FL2VA_SCHEMA.known_limitations)


def _expected_shift(value: float, ratio: float) -> float:
    return ratio * value / (1.0 + (ratio - 1.0) * value)


def test_minimax_h3_diffusers_lane_matches_float64_closed_form_and_counts() -> None:
    result = build_minimax_h3_schedule(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        grid_points=4,
        precision="float64",
    )
    assert result.request.provenance.source_revision == MINIMAX_H3_DIFFUSERS_REVISION
    assert result.request.requested_inputs.steps == 3
    assert result.effective_inputs.steps == 3
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas == pytest.approx(
        (*(_expected_shift(value, MINIMAX_H3_VIDEO_SHIFT) for value in (1.0, 2 / 3, 1 / 3)), 0.0),
        abs=1e-15,
    )
    assert result.sigmas[-1] == 0.0
    assert all(left > right for left, right in pairwise(result.sigmas))


def test_minimax_h3_float32_lane_has_finite_deterministic_terminal_and_fingerprint() -> None:
    first = build_minimax_h3_schedule(
        variant=MiniMaxH3Variant.BASE_REF2VA,
        grid_points=20,
        precision="float32",
    )
    second = build_minimax_h3_schedule(
        variant=MiniMaxH3Variant.BASE_REF2VA,
        grid_points=20,
        precision="float32",
    )
    assert first.sigmas == second.sigmas
    assert len(first.sigmas) == 20
    assert first.sigmas[-1] == 0.0
    assert all(0.0 <= sigma <= 1.0 for sigma in first.sigmas)
    assert all(left > right for left, right in pairwise(first.sigmas))


def test_minimax_h3_float32_dedup_records_an_explicit_step_override() -> None:
    result = build_minimax_h3_schedule(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        grid_points=10_000,
        precision="float32",
    )
    if result.effective_inputs.steps != result.request.requested_inputs.steps:
        assert result.overrides
        assert result.overrides[0].field == "steps"
        assert result.warnings
    else:
        assert result.overrides == ()


def test_minimax_h3_comfyui_simple_lane_is_explicitly_distinct() -> None:
    diffusers = build_minimax_h3_schedule(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        grid_points=20,
        precision="float32",
    )
    native = build_minimax_h3_comfyui_simple_schedule(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        transitions=20,
    )
    assert native.request.provenance.source_revision == MINIMAX_H3_COMFYUI_REVISION
    assert native.request.base_grid is not None
    assert native.request.base_grid.identifier == "comfyui.discrete_flow_1000"
    assert native.sigmas[-1] == 0.0
    assert native.request.requested_inputs.steps == 20
    assert native.sigmas != diffusers.sigmas
    assert MiniMaxH3ScheduleLane.COMFYUI_SIMPLE.value == "comfyui_simple"


@pytest.mark.parametrize(
    ("video_sigma", "expected_audio", "expected_derivative"),
    [(0.0, 0.0, 0.25), (1.0, 1.0, 4.0)],
)
def test_minimax_h3_audio_mapping_has_explicit_endpoint_limits(
    video_sigma: float,
    expected_audio: float,
    expected_derivative: float,
) -> None:
    mapping = map_minimax_h3_audio_coordinate(video_sigma)
    assert mapping.audio_sigma == expected_audio
    assert mapping.base_coordinate == video_sigma
    assert mapping.derivative == expected_derivative


def test_minimax_h3_audio_mapping_is_paired_and_not_an_external_schedule() -> None:
    mapping = map_minimax_h3_audio_coordinate(0.5, precision="float64")
    expected_base = 0.5 / (MINIMAX_H3_VIDEO_SHIFT + 0.5 * (1.0 - MINIMAX_H3_VIDEO_SHIFT))
    expected_audio = (
        MINIMAX_H3_AUDIO_SHIFT
        * expected_base
        / (1.0 + (MINIMAX_H3_AUDIO_SHIFT - 1.0) * expected_base)
    )
    assert mapping.base_coordinate == pytest.approx(expected_base, abs=1e-15)
    assert mapping.audio_sigma == pytest.approx(expected_audio, abs=1e-15)
    assert mapping.derivative > 0.0


def test_minimax_h3_velocity_direction_requires_explicit_sign_adapter() -> None:
    assert (
        minimax_h3_velocity_conversion_sign(
            MiniMaxH3VelocityDirection.DATA_WARD,
            MiniMaxH3VelocityDirection.NOISE_WARD,
        )
        == -1.0
    )
    assert (
        minimax_h3_velocity_conversion_sign(
            MiniMaxH3VelocityDirection.DATA_WARD,
            MiniMaxH3VelocityDirection.DATA_WARD,
        )
        == 1.0
    )
    with pytest.raises(ScheduleContractError, match="explicit"):
        minimax_h3_velocity_conversion_sign("data_ward", MiniMaxH3VelocityDirection.NOISE_WARD)  # type: ignore[arg-type]
