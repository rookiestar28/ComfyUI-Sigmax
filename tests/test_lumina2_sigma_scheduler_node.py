"""RED contract coverage for the explicit Lumina-Image 2.0 SIGMAS node."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.lumina2_sigma_scheduler import (
    LUMINA2_SIGMA_NODE_ID,
    LUMINA2_SIGMA_NODE_SCHEMA_ID,
    Lumina2SigmaScheduler,
    build_lumina2_sigma_schedule,
)


def test_lumina2_node_is_registered_with_one_explicit_mode() -> None:
    assert LUMINA2_SIGMA_NODE_ID == "Sigmax.Lumina2SigmaScheduler"
    assert NODE_CLASS_MAPPINGS[LUMINA2_SIGMA_NODE_ID] is Lumina2SigmaScheduler
    inputs = Lumina2SigmaScheduler.INPUT_TYPES()["required"]
    assert inputs["mode"][0] == ("Official Fixed (6.0)",)
    assert Lumina2SigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")


def test_lumina2_node_metadata_is_canonical_and_source_explicit() -> None:
    result = build_lumina2_sigma_schedule(
        mode="Official Fixed (6.0)",
        steps=50,
        strict_source=False,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == LUMINA2_SIGMA_NODE_SCHEMA_ID
    assert info["profile"]["id"] == "lumina2.v2.official"
    assert info["shift"] == {"kind": "direct_ratio", "multiplier": 1.0, "ratio": 6.0}
    assert info["slicing"]["output_steps"] == 50


def test_lumina2_node_rejects_implicit_or_invalid_mode_before_output() -> None:
    with pytest.raises(ScheduleContractError, match="mode"):
        build_lumina2_sigma_schedule(
            mode="auto",
            steps=50,
            strict_source=False,
            start_step=0,
            end_step=-1,
        )
