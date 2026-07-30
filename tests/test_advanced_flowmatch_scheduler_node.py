"""Contracts for the configurable unit-flow SIGMAS scheduler node."""

from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import FrozenInstanceError
from itertools import pairwise
from types import SimpleNamespace
from typing import Any, cast

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.nodes import advanced_flowmatch_scheduler as scheduler_module
from comfyui_sigmax.nodes.advanced_flowmatch_scheduler import (
    ADVANCED_FLOWMATCH_NODE_ID,
    ADVANCED_FLOWMATCH_NODE_SCHEMA_ID,
    AdvancedFlowMatchNodeResult,
    AdvancedFlowMatchScheduler,
    AdvancedFlowMatchShiftMode,
    build_advanced_flowmatch_schedule,
)


def _arguments(**changes: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "domain": "UNIT_FLOW",
        "steps": 4,
        "sigma_start": 1.0,
        "sigma_end": 0.1,
        "shift_mode": "exponential_mu",
        "shift_value": 0.0,
        "terminal_policy": "append_zero",
        "start_step": 0,
        "end_step": -1,
    }
    arguments.update(changes)
    return arguments


def _info(result: AdvancedFlowMatchNodeResult) -> dict[str, Any]:
    decoded = json.loads(result.schedule_info_json)
    assert isinstance(decoded, dict)
    return decoded


def test_node_declares_explicit_stable_schema_without_inert_mode_controls() -> None:
    inputs = AdvancedFlowMatchScheduler.INPUT_TYPES()

    assert ADVANCED_FLOWMATCH_NODE_ID == "Sigmax.AdvancedFlowMatchScheduler"
    assert ADVANCED_FLOWMATCH_NODE_SCHEMA_ID == "sigmax.advanced-flowmatch-node/1"
    assert AdvancedFlowMatchScheduler.RETURN_TYPES == ("SIGMAS", "STRING")
    assert AdvancedFlowMatchScheduler.RETURN_NAMES == ("sigmas", "schedule_info")
    assert AdvancedFlowMatchScheduler.FUNCTION == "build"
    assert AdvancedFlowMatchScheduler.CATEGORY == "Sigmax/scheduling"
    assert AdvancedFlowMatchScheduler.OUTPUT_NODE is False
    assert inputs["required"]["domain"][0] == ("UNIT_FLOW",)
    assert inputs["required"]["shift_mode"][0] == (
        "exponential_mu",
        "direct_ratio",
    )
    assert inputs["required"]["terminal_policy"][0] == (
        "append_zero",
        "preserve",
    )
    assert set(inputs["required"]) == {
        "domain",
        "steps",
        "sigma_start",
        "sigma_end",
        "shift_mode",
        "shift_value",
        "terminal_policy",
        "start_step",
        "end_step",
    }
    assert "mu" not in inputs["required"]
    assert "ratio" not in inputs["required"]
    assert inputs == AdvancedFlowMatchScheduler.INPUT_TYPES()
    assert inputs is not AdvancedFlowMatchScheduler.INPUT_TYPES()


def test_builtin_mapping_registers_advanced_scheduler() -> None:
    assert (
        comfyui_sigmax.NODE_CLASS_MAPPINGS[ADVANCED_FLOWMATCH_NODE_ID] is AdvancedFlowMatchScheduler
    )
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[ADVANCED_FLOWMATCH_NODE_ID]
        == "Advanced FlowMatch Scheduler"
    )


def test_linear_grid_and_exponential_mu_formula_execute_every_numeric_control() -> None:
    result = build_advanced_flowmatch_schedule(
        **_arguments(steps=4, sigma_start=0.9, sigma_end=0.3, shift_value=math.log(2.0))
    )
    base = (0.9, 0.7, 0.5, 0.3)
    expected = (*(2.0 / (2.0 + (1.0 / value) - 1.0) for value in base), 0.0)
    info = _info(result)

    assert result.shift_mode is AdvancedFlowMatchShiftMode.EXPONENTIAL_MU
    assert result.sigmas == pytest.approx(expected, rel=1e-14, abs=1e-15)
    assert info["base_grid"] == {
        "identifier": "sigmax.linear_endpoint",
        "points": 4,
        "sigma_end": 0.3,
        "sigma_start": 0.9,
    }
    assert info["domain"] == {
        "sigma": "UNIT_FLOW",
        "time": "UNIT_FLOW",
    }
    assert info["shift"] == {
        "kind": "exponential_mu",
        "value": math.log(2.0),
    }


def test_direct_ratio_formula_is_distinct_and_identity_values_are_explicit() -> None:
    direct = build_advanced_flowmatch_schedule(
        **_arguments(shift_mode="direct_ratio", shift_value=3.0)
    )
    base = (1.0, 0.7, 0.4, 0.1)
    expected = (
        *(
            value if value in {0.0, 1.0} else 1.0 / (1.0 + (1.0 - value) / (3.0 * value))
            for value in base
        ),
        0.0,
    )
    exponential_identity = build_advanced_flowmatch_schedule(**_arguments(shift_value=0.0))
    ratio_identity = build_advanced_flowmatch_schedule(
        **_arguments(shift_mode="direct_ratio", shift_value=1.0)
    )

    assert direct.shift_mode is AdvancedFlowMatchShiftMode.DIRECT_RATIO
    assert direct.sigmas == pytest.approx(expected, rel=1e-14, abs=1e-15)
    assert exponential_identity.sigmas == pytest.approx((*base, 0.0))
    assert ratio_identity.sigmas == pytest.approx((*base, 0.0))
    assert _info(direct)["shift"] == {"kind": "direct_ratio", "value": 3.0}


@pytest.mark.parametrize(
    ("terminal_policy", "expected"),
    (
        ("append_zero", (1.0, 0.7, 0.4, 0.1, 0.0)),
        ("preserve", (1.0, 0.775, 0.55, 0.325, 0.1)),
    ),
)
def test_terminal_policy_preserves_requested_transition_count(
    terminal_policy: str,
    expected: tuple[float, ...],
) -> None:
    result = build_advanced_flowmatch_schedule(
        **_arguments(
            shift_mode="direct_ratio",
            shift_value=1.0,
            terminal_policy=terminal_policy,
        )
    )
    info = _info(result)

    assert result.sigmas == pytest.approx(expected)
    assert len(result.sigmas) == 5
    assert info["terminal"] == {
        "policy": terminal_policy,
        "value": result.sigmas[-1],
    }
    assert info["slicing"]["available_steps"] == 4


def test_slicing_runs_after_shift_and_terminal_and_has_separate_fingerprint() -> None:
    full = build_advanced_flowmatch_schedule(
        **_arguments(shift_mode="direct_ratio", shift_value=2.0)
    )
    sliced = build_advanced_flowmatch_schedule(
        **_arguments(
            shift_mode="direct_ratio",
            shift_value=2.0,
            start_step=1,
            end_step=3,
        )
    )
    info = _info(sliced)

    assert sliced.sigmas == full.sigmas[1:4]
    assert info["slicing"] == {
        "available_steps": 4,
        "end_step": 3,
        "output_steps": 2,
        "start_step": 1,
    }
    assert info["fingerprints"]["complete"] != info["fingerprints"]["output"]


def test_result_is_finite_strictly_decreasing_experimental_and_deterministic() -> None:
    first = build_advanced_flowmatch_schedule(**_arguments(shift_value=1.15))
    second = build_advanced_flowmatch_schedule(**_arguments(shift_value=1.15))
    info = _info(first)

    assert first == second
    assert all(math.isfinite(value) for value in first.sigmas)
    assert all(left > right for left, right in pairwise(first.sigmas))
    assert info["schema"] == ADVANCED_FLOWMATCH_NODE_SCHEMA_ID
    assert info["provenance"]["evidence"] == "experimental"
    assert info["provenance"]["profile_id"] is None
    assert info["ownership"] == "EXTERNAL_SIGMAS"
    assert info["transform_order"] == [
        "PRIMARY_TIME_SHIFT",
        "TERMINAL",
        "SLICE",
    ]
    assert info["fingerprints"]["complete"].startswith("sha256:")
    assert first.schedule_info_json == second.schedule_info_json
    with pytest.raises(FrozenInstanceError):
        first.sigmas = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    (
        {"domain": "MODEL_NATIVE"},
        {"domain": 1},
        {"steps": 0},
        {"steps": True},
        {"steps": 10_001},
        {"sigma_start": 0.0},
        {"sigma_start": 1.1},
        {"sigma_start": math.inf},
        {"sigma_end": -0.1},
        {"sigma_end": 1.0},
        {"sigma_start": 0.5, "sigma_end": 0.5},
        {"shift_mode": "none"},
        {"shift_mode": 1},
        {"shift_value": math.nan},
        {"shift_mode": "direct_ratio", "shift_value": 0.0},
        {"terminal_policy": "other"},
        {"terminal_policy": 1},
        {"terminal_policy": "append_zero", "sigma_end": 0.0},
        {"start_step": -1},
        {"start_step": True},
        {"start_step": 4},
        {"end_step": -2},
        {"end_step": False},
        {"start_step": 3, "end_step": 3},
        {"start_step": 3, "end_step": 5},
    ),
)
def test_invalid_or_ambiguous_inputs_fail_before_tensor_conversion(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ScheduleContractError):
        build_advanced_flowmatch_schedule(**_arguments(**changes))


@pytest.mark.parametrize(
    "changes",
    (
        {"shift_mode": cast(Any, "other")},
        {"domain": SigmaDomain.CONTINUOUS_EDM},
        {"sigmas": cast(Any, [1.0, 0.0])},
        {"sigmas": (0.0,)},
        {"schedule_info_json": ""},
        {"schedule_info_json": cast(Any, 1)},
    ),
)
def test_result_contract_rejects_invalid_values(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "shift_mode": AdvancedFlowMatchShiftMode.EXPONENTIAL_MU,
        "domain": SigmaDomain.UNIT_FLOW,
        "sigmas": (1.0, 0.0),
        "schedule_info_json": "{}",
    }
    arguments.update(changes)

    with pytest.raises(ScheduleContractError):
        AdvancedFlowMatchNodeResult(**cast(Any, arguments))


def test_information_projection_rejects_non_json_values() -> None:
    with pytest.raises(ScheduleContractError, match="canonical JSON"):
        scheduler_module._canonical_info({"invalid": object()})


def test_runtime_node_converts_only_at_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, ...]] = []

    def float_tensor(values: tuple[float, ...]) -> object:
        calls.append(tuple(values))
        return SimpleNamespace(values=tuple(values), device="cpu")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(FloatTensor=float_tensor))
    output = AdvancedFlowMatchScheduler().build(**_arguments())

    tensor = cast(SimpleNamespace, output[0])
    assert tensor.device == "cpu"
    assert calls == [tensor.values]
    assert json.loads(output[1])["schema"] == ADVANCED_FLOWMATCH_NODE_SCHEMA_ID


@pytest.mark.parametrize("failure", ("missing_module", "missing_float_tensor"))
def test_runtime_node_fails_actionably_without_torch_contract(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    if failure == "missing_module":

        def missing_module(name: str) -> object:
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", missing_module)
    else:
        monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace())

    with pytest.raises(RuntimeError, match="requires Torch FloatTensor"):
        AdvancedFlowMatchScheduler().build(**_arguments())
