"""Phase 0 RED contract for the thin Anima SIGMAS node."""

from __future__ import annotations

import importlib
import importlib.util
import json
from typing import Any

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain


def _node_module() -> Any:
    assert importlib.util.find_spec("comfyui_sigmax.nodes.anima_sigma_scheduler") is not None
    return importlib.import_module("comfyui_sigmax.nodes.anima_sigma_scheduler")


def test_anima_node_is_namespaced_and_registered() -> None:
    module = _node_module()
    node_id = getattr(module, "ANIMA_SIGMA_NODE_ID", None)
    node_class = getattr(module, "AnimaSigmaScheduler", None)
    assert node_id == "Sigmax.AnimaSigmaScheduler"
    assert node_class is not None
    assert NODE_CLASS_MAPPINGS[node_id] is node_class
    assert module.ANIMA_SIGMA_NODE_SCHEMA_ID == "sigmax.anima-sigma-node/1"


def test_anima_node_has_explicit_variants_and_only_sigmas_plus_info_outputs() -> None:
    module = _node_module()
    node_class = module.AnimaSigmaScheduler
    inputs = node_class.INPUT_TYPES()["required"]
    assert inputs["variant"][0] == ("Base (3.0)", "Aesthetic (3.0)", "Turbo (3.0)")
    assert node_class.RETURN_TYPES == ("SIGMAS", "STRING")
    assert "base_shift" not in inputs
    assert "max_shift" not in inputs


def test_anima_node_metadata_declares_profile_and_transform_order() -> None:
    module = _node_module()
    builder = getattr(module, "build_anima_sigma_schedule", None)
    assert callable(builder)
    result = builder(
        variant="Base (3.0)",
        steps=50,
        strict_source=True,
        start_step=0,
        end_step=-1,
        already_shifted=False,
    )
    assert result.domain is SigmaDomain.UNIT_FLOW
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == "sigmax.anima-sigma-node/1"
    assert info["profile"]["id"] == "anima.base.framework-reference"
    assert info["shift"] == {"kind": "rational", "multiplier": 1.0, "shift": 3.0}
    assert info["slicing"]["output_steps"] == 50


def test_anima_node_rejects_implicit_variant_and_second_shift() -> None:
    module = _node_module()
    builder = module.build_anima_sigma_schedule
    with pytest.raises(ScheduleContractError, match="variant"):
        builder(
            variant="auto",
            steps=50,
            strict_source=False,
            start_step=0,
            end_step=-1,
            already_shifted=False,
        )
    with pytest.raises(ScheduleContractError, match="already shifted"):
        builder(
            variant="Base (3.0)",
            steps=50,
            strict_source=False,
            start_step=0,
            end_step=-1,
            already_shifted=True,
        )
