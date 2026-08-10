"""Phase 2 boundary, slicing, and provenance contracts for Wan 2.2."""

from __future__ import annotations

import json
from itertools import pairwise

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain, slice_step_range
from comfyui_sigmax.nodes.wan_sigma_scheduler import build_wan_sigma_schedule
from comfyui_sigmax.profiles.wan import (
    WanProfileId,
    WanResolution,
    build_wan_schedule,
    derive_wan_boundary,
)


@pytest.mark.parametrize(
    ("profile", "boundary", "steps"),
    [
        (WanProfileId.WAN22_T2V_A14B_NATIVE, 0.875, 40),
        (WanProfileId.WAN22_I2V_A14B_NATIVE, 0.9, 40),
        (WanProfileId.WAN22_T2V_A14B_DIFFUSERS, 0.875, 40),
        (WanProfileId.WAN22_I2V_A14B_DIFFUSERS, 0.9, 40),
    ],
)
def test_a14b_boundary_is_deterministic_and_caller_owned(
    profile: WanProfileId, boundary: float, steps: int
) -> None:
    first = build_wan_schedule(profile=profile, steps=steps)
    second = build_wan_schedule(profile=profile, steps=steps)
    assert first.boundary == second.boundary
    assert first.boundary is not None
    assert first.boundary.normalized == pytest.approx(boundary)
    assert first.boundary.routing_owner == "caller"
    assert first.boundary.model_dispatch is False
    assert first.boundary.transition_index < steps
    assert first.sigmas[first.boundary.transition_index] <= boundary
    if first.boundary.transition_index:
        assert first.sigmas[first.boundary.transition_index - 1] > boundary


def test_boundary_equality_is_at_or_above_and_off_by_one_is_rejected() -> None:
    exact = derive_wan_boundary(sigmas=(1.0, 0.95, 0.875, 0.7, 0.0), normalized_boundary=0.875)
    assert exact.transition_index == 2
    assert exact.crossing == "at_or_above"
    with pytest.raises(ScheduleContractError, match="descending"):
        derive_wan_boundary(sigmas=(1.0, 0.8, 0.81, 0.0), normalized_boundary=0.875)


def test_a14b_boundary_survives_explicit_slice_without_model_routing() -> None:
    complete = build_wan_sigma_schedule(
        generation="Wan 2.2",
        task="T2V A14B",
        source="Official native",
        resolution="None",
        steps=40,
        strict_source=True,
        start_step=0,
        end_step=-1,
    )
    sliced = build_wan_sigma_schedule(
        generation="Wan 2.2",
        task="T2V A14B",
        source="Official native",
        resolution="None",
        steps=40,
        strict_source=True,
        start_step=10,
        end_step=26,
    )
    assert sliced.sigmas == slice_step_range(complete.sigmas, start_step=10, end_step=26)
    assert sliced.boundary_step == complete.boundary_step
    assert sliced.domain is SigmaDomain.UNIT_FLOW
    assert '"model_dispatch":false' in sliced.schedule_info_json
    assert '"routing_owner":"caller"' in sliced.schedule_info_json
    guidance = json.loads(sliced.schedule_info_json)["guidance"]
    assert guidance == {
        "cfg_high": 4.0,
        "cfg_low": 3.0,
        "host_cfg_scale": 3.0,
        "model_cfg_scale": 3.0,
    }


def test_wan_strict_modified_and_resolution_contracts_are_explicit() -> None:
    modified = build_wan_schedule(
        profile=WanProfileId.WAN21_I2V_480P_OFFICIAL,
        steps=41,
        resolution=WanResolution.P480,
        strict_source=False,
    )
    assert modified.warnings
    assert modified.request.provenance.evidence.value == "modified"
    with pytest.raises(ScheduleContractError, match="pinned"):
        build_wan_schedule(
            profile=WanProfileId.WAN21_I2V_480P_OFFICIAL,
            steps=41,
            resolution=WanResolution.P480,
            strict_source=True,
        )
    with pytest.raises(ScheduleContractError, match="required"):
        build_wan_schedule(
            profile=WanProfileId.WAN21_I2V_480P_OFFICIAL,
            steps=40,
            resolution=WanResolution.NONE,
        )


def test_wan_schedule_vectors_remain_finite_and_strictly_descending() -> None:
    for profile in WanProfileId:
        resolution = (
            WanResolution.P480
            if profile
            in {
                WanProfileId.WAN21_I2V_480P_OFFICIAL,
                WanProfileId.WAN21_I2V_480P_DIFFUSERS,
            }
            else (
                WanResolution.P720
                if profile
                in {
                    WanProfileId.WAN21_FLF2V_14B_720P_OFFICIAL,
                    WanProfileId.WAN21_I2V_720P_OFFICIAL,
                    WanProfileId.WAN21_I2V_720P_DIFFUSERS,
                }
                else WanResolution.NONE
            )
        )
        result = build_wan_schedule(profile=profile, steps=8, resolution=resolution)
        assert result.sigmas[0] == 1.0
        assert result.sigmas[-1] == 0.0
        assert all(left > right for left, right in pairwise(result.sigmas))
