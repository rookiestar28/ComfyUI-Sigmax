"""Phase 0 RED contracts for the thin Wan SIGMAS node."""

from __future__ import annotations

import importlib
import importlib.util
import json
from typing import Any

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain


def _node_module() -> Any:
    assert importlib.util.find_spec("comfyui_sigmax.nodes.wan_sigma_scheduler") is not None
    return importlib.import_module("comfyui_sigmax.nodes.wan_sigma_scheduler")


def test_wan_node_is_namespaced_and_registered() -> None:
    module = _node_module()
    node_id = getattr(module, "WAN_SIGMA_NODE_ID", None)
    node_class = getattr(module, "WanSigmaScheduler", None)
    assert node_id == "Sigmax.WanSigmaScheduler"
    assert node_class is not None
    assert NODE_CLASS_MAPPINGS[node_id] is node_class
    assert module.WAN_SIGMA_NODE_SCHEMA_ID == "sigmax.wan-sigma-node/1"


def test_wan_node_exposes_explicit_axes_and_boundary_output() -> None:
    module = _node_module()
    node_class = module.WanSigmaScheduler
    inputs = node_class.INPUT_TYPES()["required"]
    assert "generation" in inputs and "task" in inputs and "source" in inputs
    assert "resolution" in inputs
    assert node_class.RETURN_TYPES == ("SIGMAS", "INT", "STRING")
    assert "shift" not in inputs
    assert "model" not in inputs


def test_wan_node_metadata_declares_boundary_without_routing() -> None:
    module = _node_module()
    result = module.build_wan_sigma_schedule(
        generation="Wan 2.2",
        task="T2V A14B",
        source="Official native",
        resolution="None",
        steps=40,
        strict_source=True,
        start_step=0,
        end_step=-1,
        already_shifted=False,
    )
    assert result.domain is SigmaDomain.UNIT_FLOW
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == "sigmax.wan-sigma-node/1"
    assert info["boundary"]["routing_owner"] == "caller"
    assert info["boundary"]["model_dispatch"] is False
    assert info["slicing"]["output_steps"] == 40


def test_wan_node_rejects_implicit_derivatives_and_second_shift() -> None:
    module = _node_module()
    builder = module.build_wan_sigma_schedule
    with pytest.raises(ScheduleContractError, match=r"profile|generation|task"):
        builder(
            generation="Wan",
            task="Fun-Control",
            source="Official native",
            resolution="None",
            steps=40,
            strict_source=False,
            start_step=0,
            end_step=-1,
            already_shifted=False,
        )
    with pytest.raises(ScheduleContractError, match="already shifted"):
        builder(
            generation="Wan 2.1",
            task="T2V",
            source="Official native",
            resolution="None",
            steps=50,
            strict_source=False,
            start_step=0,
            end_step=-1,
            already_shifted=True,
        )
