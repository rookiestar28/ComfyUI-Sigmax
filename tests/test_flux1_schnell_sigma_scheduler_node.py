"""Public node contract for explicit FLUX.1-schnell sigma construction."""

from __future__ import annotations

import json
from typing import Any, cast

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.flux1_schnell_sigma_scheduler import (
    FLUX1_SCHNELL_SIGMA_NODE_ID,
    FLUX1_SCHNELL_SIGMA_NODE_SCHEMA_ID,
    Flux1SchnellSigmaScheduler,
    build_flux1_schnell_sigma_schedule,
)


def test_node_schema_and_builtin_registration_are_stable() -> None:
    inputs = Flux1SchnellSigmaScheduler.INPUT_TYPES()

    assert FLUX1_SCHNELL_SIGMA_NODE_ID == "Sigmax.Flux1SchnellSigmaScheduler"
    assert FLUX1_SCHNELL_SIGMA_NODE_SCHEMA_ID == "sigmax.flux1-schnell-sigma-node/1"
    assert Flux1SchnellSigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")
    assert Flux1SchnellSigmaScheduler.RETURN_NAMES == ("sigmas", "schedule_info")
    assert Flux1SchnellSigmaScheduler.FUNCTION == "build"
    assert set(inputs["required"]) == {
        "steps",
        "strict_official",
        "start_step",
        "end_step",
    }
    assert cast(dict[str, Any], inputs["required"]["steps"][1])["default"] == 4
    assert (
        comfyui_sigmax.NODE_CLASS_MAPPINGS[FLUX1_SCHNELL_SIGMA_NODE_ID]
        is Flux1SchnellSigmaScheduler
    )


def test_strict_node_output_is_provenance_bound() -> None:
    result = build_flux1_schnell_sigma_schedule(
        steps=4,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)

    assert result.sigmas == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert info["schema"] == FLUX1_SCHNELL_SIGMA_NODE_SCHEMA_ID
    assert info["profile"] == {
        "evidence": "official",
        "id": "flux1.schnell.official",
        "recipe": "flux1.schnell.official",
        "variant": "schnell",
        "version": "1",
    }
    assert info["shift"] == {"dynamic": False, "kind": "none"}
    assert info["guidance"] == {"host_cfg": 1.0, "model_guidance": 0.0}
    assert info["fingerprints"]["complete"].startswith("sha256:")


def test_slicing_preserves_complete_fingerprint() -> None:
    full = build_flux1_schnell_sigma_schedule(
        steps=4, strict_official=True, start_step=0, end_step=-1
    )
    sliced = build_flux1_schnell_sigma_schedule(
        steps=4, strict_official=True, start_step=1, end_step=3
    )
    full_info = json.loads(full.schedule_info_json)
    sliced_info = json.loads(sliced.schedule_info_json)

    assert sliced.sigmas == (0.75, 0.5, 0.25)
    assert sliced_info["fingerprints"]["complete"] == full_info["fingerprints"]["complete"]
    assert sliced_info["fingerprints"]["output"] != full_info["fingerprints"]["output"]


@pytest.mark.parametrize(
    "changes",
    (
        {"steps": 0},
        {"strict_official": 1},
        {"start_step": -1},
        {"end_step": 0},
    ),
)
def test_invalid_node_requests_fail_closed(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "steps": 4,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
    }
    arguments.update(changes)
    with pytest.raises(ScheduleContractError):
        build_flux1_schnell_sigma_schedule(**arguments)
