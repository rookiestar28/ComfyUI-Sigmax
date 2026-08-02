"""RED contract coverage for the explicit original SD3 SIGMAS node."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.sd3_sigma_scheduler import (
    SD3_SIGMA_NODE_ID,
    SD3_SIGMA_NODE_SCHEMA_ID,
    SD3SigmaScheduler,
    build_sd3_sigma_schedule,
)


def test_sd3_node_is_registered_with_explicit_source_modes() -> None:
    assert SD3_SIGMA_NODE_ID == "Sigmax.SD3SigmaScheduler"
    assert NODE_CLASS_MAPPINGS[SD3_SIGMA_NODE_ID] is SD3SigmaScheduler
    inputs = SD3SigmaScheduler.INPUT_TYPES()["required"]
    assert inputs["mode"][0] == ("Publisher Reference (1.0)", "Comfy/Diffusers Fixed (3.0)")
    assert SD3SigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")


@pytest.mark.parametrize(
    ("mode", "profile_id", "shift"),
    [
        ("Publisher Reference (1.0)", "sd3.publisher-reference.official", 1.0),
        ("Comfy/Diffusers Fixed (3.0)", "sd3.comfy-diffusers-fixed.framework-reference", 3.0),
    ],
)
def test_sd3_node_metadata_is_canonical_and_source_explicit(
    mode: str, profile_id: str, shift: float
) -> None:
    result = build_sd3_sigma_schedule(
        mode=mode,
        steps=50,
        strict_source=False,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == SD3_SIGMA_NODE_SCHEMA_ID
    assert info["profile"]["id"] == profile_id
    assert info["shift"] == {"kind": "direct_ratio", "ratio": shift}
    assert info["slicing"]["output_steps"] == 50


def test_sd3_node_rejects_implicit_or_invalid_mode_before_output() -> None:
    with pytest.raises(ScheduleContractError, match="mode"):
        build_sd3_sigma_schedule(
            mode="auto",
            steps=50,
            strict_source=False,
            start_step=0,
            end_step=-1,
        )
