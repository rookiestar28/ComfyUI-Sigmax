"""RED contract coverage for the HunyuanImage 2.1 SIGMAS node."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.hunyuan_image21_sigma_scheduler import (
    HUNYUAN_IMAGE21_SIGMA_NODE_ID,
    HUNYUAN_IMAGE21_SIGMA_NODE_SCHEMA_ID,
    HunyuanImage21SigmaScheduler,
    build_hunyuan_image21_sigma_schedule,
)


def test_hunyuan_image21_node_is_registered_with_two_explicit_variants() -> None:
    assert HUNYUAN_IMAGE21_SIGMA_NODE_ID == "Sigmax.HunyuanImage21SigmaScheduler"
    assert NODE_CLASS_MAPPINGS[HUNYUAN_IMAGE21_SIGMA_NODE_ID] is HunyuanImage21SigmaScheduler
    inputs = HunyuanImage21SigmaScheduler.INPUT_TYPES()["required"]
    assert inputs["variant"][0] == ("Base (5.0)", "Distilled (4.0)")
    assert HunyuanImage21SigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")


@pytest.mark.parametrize(
    ("variant", "profile_id", "ratio", "steps"),
    [
        ("Base (5.0)", "hunyuan-image-2-1.base.official", 5.0, 50),
        ("Distilled (4.0)", "hunyuan-image-2-1.distilled.official", 4.0, 8),
    ],
)
def test_hunyuan_image21_node_metadata_is_canonical(
    variant: str,
    profile_id: str,
    ratio: float,
    steps: int,
) -> None:
    result = build_hunyuan_image21_sigma_schedule(
        variant=variant,
        steps=steps,
        strict_source=True,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == HUNYUAN_IMAGE21_SIGMA_NODE_SCHEMA_ID
    assert info["profile"]["id"] == profile_id
    assert info["shift"] == {"kind": "direct_ratio", "multiplier": 1.0, "ratio": ratio}
    assert info["slicing"]["output_steps"] == steps


def test_hunyuan_image21_node_rejects_implicit_variant() -> None:
    with pytest.raises(ScheduleContractError, match="variant"):
        build_hunyuan_image21_sigma_schedule(
            variant="auto",
            steps=50,
            strict_source=False,
            start_step=0,
            end_step=-1,
        )
