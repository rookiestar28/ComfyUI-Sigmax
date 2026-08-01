"""Public node contract for explicit Z-Image schedule construction."""

from __future__ import annotations

import json

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.z_image_sigma_scheduler import (
    Z_IMAGE_SIGMA_NODE_ID,
    Z_IMAGE_SIGMA_NODE_SCHEMA_ID,
    ZImageSigmaScheduler,
    build_z_image_sigma_schedule,
)


def test_node_schema_and_builtin_registration_are_stable() -> None:
    inputs = ZImageSigmaScheduler.INPUT_TYPES()

    assert Z_IMAGE_SIGMA_NODE_ID == "Sigmax.ZImageSigmaScheduler"
    assert Z_IMAGE_SIGMA_NODE_SCHEMA_ID == "sigmax.z-image-sigma-node/1"
    assert ZImageSigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")
    assert ZImageSigmaScheduler.RETURN_NAMES == ("sigmas", "schedule_info")
    assert ZImageSigmaScheduler.FUNCTION == "build"
    assert ZImageSigmaScheduler.CATEGORY == "Sigmax/scheduling"
    assert inputs["required"]["variant"][0] == ("Base", "Turbo")
    assert set(inputs["required"]) == {
        "variant",
        "steps",
        "strict_official",
        "start_step",
        "end_step",
    }
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS[Z_IMAGE_SIGMA_NODE_ID] is ZImageSigmaScheduler
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[Z_IMAGE_SIGMA_NODE_ID]
        == "Z-Image Sigma Scheduler"
    )


@pytest.mark.parametrize(
    ("variant", "steps", "profile_id", "ratio"),
    (
        ("Base", 50, "z_image.base.official", 6.0),
        ("Turbo", 8, "z_image.turbo.official", 3.0),
    ),
)
def test_strict_node_output_is_provenance_bound(
    variant: str,
    steps: int,
    profile_id: str,
    ratio: float,
) -> None:
    result = build_z_image_sigma_schedule(
        variant=variant,
        steps=steps,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)

    assert len(result.sigmas) == steps + 1
    assert result.sigmas[-1] == 0.0
    assert info["schema"] == Z_IMAGE_SIGMA_NODE_SCHEMA_ID
    assert info["profile"]["id"] == profile_id
    assert info["profile"]["evidence"] == "official"
    assert info["shift"] == {"dynamic": False, "kind": "fixed_direct_ratio", "ratio": ratio}
    assert info["fingerprints"]["complete"].startswith("sha256:")
    assert info["fingerprints"]["output"].startswith("sha256:")


def test_slicing_preserves_the_complete_fingerprint() -> None:
    full = build_z_image_sigma_schedule(
        variant="Turbo", steps=8, strict_official=True, start_step=0, end_step=-1
    )
    sliced = build_z_image_sigma_schedule(
        variant="Turbo", steps=8, strict_official=True, start_step=2, end_step=6
    )
    full_info = json.loads(full.schedule_info_json)
    sliced_info = json.loads(sliced.schedule_info_json)

    assert sliced.sigmas == full.sigmas[2:7]
    assert sliced_info["fingerprints"]["complete"] == full_info["fingerprints"]["complete"]
    assert sliced_info["fingerprints"]["output"] != full_info["fingerprints"]["output"]


@pytest.mark.parametrize(
    "changes",
    (
        {"variant": "unknown"},
        {"steps": 0},
        {"strict_official": 1},
        {"start_step": -1},
        {"end_step": 0},
    ),
)
def test_invalid_node_requests_fail_closed(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "variant": "Turbo",
        "steps": 8,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
    }
    arguments.update(changes)
    with pytest.raises(ScheduleContractError):
        build_z_image_sigma_schedule(**arguments)
