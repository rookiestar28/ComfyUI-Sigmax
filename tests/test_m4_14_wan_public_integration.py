"""M4-14 RED contracts for the expanded public Wan node/workflow surface."""

from __future__ import annotations

import importlib
import json
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.wan_sigma_scheduler import build_wan_sigma_schedule
from comfyui_sigmax.workflows import load_canonical_workflow_fixtures, load_pinned_host_baseline

_NEW_CASES = (
    (
        "wan21-flf2v-720p-official-50",
        "Wan 2.1",
        "FLF2V",
        "Official native",
        "720P",
        50,
        "wan2.1.flf2v.14b.720p.official-native",
    ),
    (
        "wan21-vace-1-3b-official-50",
        "Wan 2.1",
        "VACE 1.3B",
        "Official native",
        "None",
        50,
        "wan2.1.vace.1.3b.official-native",
    ),
    (
        "wan21-vace-14b-official-50",
        "Wan 2.1",
        "VACE 14B",
        "Official native",
        "None",
        50,
        "wan2.1.vace.14b.official-native",
    ),
    (
        "wan22-s2v-14b-official-40",
        "Wan 2.2",
        "S2V",
        "Official native",
        "None",
        40,
        "wan2.2.s2v.14b.official-native",
    ),
    (
        "wan22-animate-14b-official-20",
        "Wan 2.2",
        "Animate",
        "Official native",
        "None",
        20,
        "wan2.2.animate.14b.official-native",
    ),
    (
        "wan-animate2-base-14b-official-40",
        "Wan Animate 2",
        "Animate Base",
        "Official native",
        "None",
        40,
        "wan-animate2.14b.base.official-native",
    ),
    (
        "wan-animate2-distilled-14b-official-10",
        "Wan Animate 2",
        "Animate Distilled",
        "Official native",
        "None",
        10,
        "wan-animate2.14b.distilled.official-native",
    ),
)


def test_m4_14_expands_axes_without_changing_the_released_prefix() -> None:
    module = importlib.import_module("comfyui_sigmax.nodes.wan_sigma_scheduler")

    assert module._GENERATIONS[:2] == ("Wan 2.1", "Wan 2.2")
    assert module._GENERATIONS[-1] == "Wan Animate 2"
    assert module._TASKS[:5] == ("T2V", "I2V", "TI2V", "T2V A14B", "I2V A14B")
    assert module._TASKS[5:] == (
        "FLF2V",
        "VACE 1.3B",
        "VACE 14B",
        "S2V",
        "Animate",
        "Animate Base",
        "Animate Distilled",
        "Animate Optimized",
    )
    assert "shift" not in module.WanSigmaScheduler.INPUT_TYPES()["required"]
    assert "model" not in module.WanSigmaScheduler.INPUT_TYPES()["required"]


@pytest.mark.parametrize(
    "fixture_id,generation,task,source,resolution,steps,profile_id",
    _NEW_CASES,
)
def test_m4_14_new_public_selections_resolve_to_qualified_profiles(
    fixture_id: str,
    generation: str,
    task: str,
    source: str,
    resolution: str,
    steps: int,
    profile_id: str,
) -> None:
    del fixture_id
    result = build_wan_sigma_schedule(
        generation=generation,
        task=task,
        source=source,
        resolution=resolution,
        steps=steps,
        strict_source=True,
        start_step=0,
        end_step=-1,
        already_shifted=False,
    )
    info = json.loads(result.schedule_info_json)
    assert info["profile"]["id"] == profile_id
    assert info["profile"]["evidence"] == "official"
    assert info["boundary"] == {"model_dispatch": False, "routing_owner": "caller", "step": -1}
    assert info["slicing"]["output_steps"] == steps
    assert info["shift"]["ratio"] in {3.0, 5.0, 16.0}
    assert info["solver_ownership"] in {"unipc.multistep", "flow_dpm.multistep"}


@pytest.mark.parametrize(
    "generation,task,source,resolution",
    (
        ("Wan 2.1", "VACE", "Official native", "None"),
        ("Wan Animate 2", "Animate", "Official native", "None"),
        ("Wan 2.2", "S2V", "Diffusers reference", "None"),
        ("Wan Animate 2", "Animate Base", "Diffusers reference", "None"),
    ),
)
def test_m4_14_unqualified_or_blocked_new_combinations_fail_closed(
    generation: str, task: str, source: str, resolution: str
) -> None:
    with pytest.raises(ScheduleContractError, match="unsupported"):
        build_wan_sigma_schedule(
            generation=generation,
            task=task,
            source=source,
            resolution=resolution,
            steps=20,
            strict_source=True,
            start_step=0,
            end_step=-1,
            already_shifted=False,
        )


@pytest.mark.parametrize(
    "fixture_id,generation,task,source,resolution,steps,profile_id",
    _NEW_CASES,
)
def test_m4_14_new_workflow_fixtures_and_host_options_exist(
    fixture_id: str,
    generation: str,
    task: str,
    source: str,
    resolution: str,
    steps: int,
    profile_id: str,
) -> None:
    fixtures = {item.identifier: item for item in load_canonical_workflow_fixtures()}
    assert fixture_id in fixtures
    fixture = fixtures[fixture_id]
    assert fixture.variant.startswith("Wan")
    assert fixture.profile.identifier == profile_id
    workflow = cast(dict[str, object], fixture.workflow)
    nodes = cast(list[dict[str, Any]], workflow["nodes"])
    scheduler = next(node for node in nodes if node["id"] == 1)
    assert scheduler["widgets_values"] == [
        generation,
        task,
        source,
        resolution,
        steps,
        True,
        0,
        -1,
        False,
    ]

    baseline = load_pinned_host_baseline()
    object_info = cast(dict[str, Any], baseline.object_info)
    required = object_info["Sigmax.WanSigmaScheduler"]["input"]["required"]
    assert generation in required["generation"][0]
    assert task in required["task"][0]
