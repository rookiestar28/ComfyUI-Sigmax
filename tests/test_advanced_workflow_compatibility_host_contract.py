"""M5-05 exact-host model-free compatibility harness tests."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "run_comfyui_e2e.py"
H3_TEST_PACK = ROOT / "tests" / "fixtures" / "comfyui_h3_nodes" / "__init__.py"


def _harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sigmax_m5_05_host_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("ComfyUI E2E harness is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fp(character: str) -> str:
    return "sha256:" + character * 64


def _trace() -> dict[str, object]:
    names = (
        "deterministic_controller",
        "deterministic_resume",
        "native_interruption",
        "native_missing_capability",
        "pure_inpainting_rejected",
        "stochastic_rejected",
    )
    hex_chars = "abcdef"
    return {
        "cleanup": True,
        "decision_fingerprints": {name: _fp(hex_chars[index]) for index, name in enumerate(names)},
        "decision_levels": {
            "deterministic_controller": "allow",
            "deterministic_resume": "allow",
            "native_interruption": "warn",
            "native_missing_capability": "reject",
            "pure_inpainting_rejected": "reject",
            "stochastic_rejected": "reject",
        },
        "expected_rejections": 3,
        "global_mutation": False,
        "model_weights_used": False,
        "receipt_fingerprints": {
            name: _fp(hex_chars[5 - index]) for index, name in enumerate(names)
        },
        "receipt_statuses": {
            "deterministic_controller": "not_executed",
            "deterministic_resume": "resumable",
            "native_interruption": "interrupted",
            "native_missing_capability": "rejected",
            "pure_inpainting_rejected": "rejected",
            "stochastic_rejected": "rejected",
        },
        "registry_mutation": False,
        "round_trip_stable": True,
        "schema": "sigmax.advanced-workflow-host-contract/1",
        "status": "succeeded",
        "python_version": "3.13.9",
        "torch_version": "2.9.0+cpu",
    }


def _history(*, prompt_id: str = "m5-05-prompt") -> dict[str, object]:
    harness = _harness()
    trace = _trace()
    canonical = json.dumps(
        trace,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        prompt_id: {
            "outputs": {"13": {"sigmax_advanced_workflow_compatibility_contract": [canonical]}},
            "prompt": [0, "fixture", harness.build_advanced_workflow_contract_api_prompt()],
            "status": {"completed": True, "status_str": "success"},
        }
    }


def test_advanced_host_prompt_flag_and_probe_registration_are_explicit() -> None:
    harness = _harness()
    assert harness.build_advanced_workflow_contract_api_prompt() == {
        "13": {"class_type": "SigmaxTest.AdvancedWorkflowCompatibilityProbe", "inputs": {}}
    }
    assert (
        harness._parser()
        .parse_args(["--advanced-workflow-contract-only"])
        .advanced_workflow_contract_only
    )
    tree = ast.parse(H3_TEST_PACK.read_text(encoding="utf-8"))
    probe = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AdvancedWorkflowCompatibilityProbe"
    )
    assert "resolve_advanced_workflow" in {
        node.func.id
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "SigmaxTest.AdvancedWorkflowCompatibilityProbe" in H3_TEST_PACK.read_text(
        encoding="utf-8"
    )


def test_advanced_host_history_verifier_accepts_expected_matrix() -> None:
    harness = _harness()
    summary = harness.verify_advanced_workflow_contract_history(
        _history(),
        prompt_id="m5-05-prompt",
    )
    assert summary == _trace()
    transition = harness.build_verified_host_repeat_transition(
        lane="H2_ADVANCED_WORKFLOW_CONTRACT_M5_05",
        first_summary=summary,
        repeat_summary=summary,
    )
    assert transition["transition"] == "pass_to_pass"
    assert transition["accepted"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cleanup", False),
        ("global_mutation", True),
        ("model_weights_used", True),
        ("round_trip_stable", False),
        ("python_version", r"C:\private\python.exe"),
    ],
)
def test_advanced_host_history_verifier_rejects_tampered_evidence(
    field: str,
    value: object,
) -> None:
    history = _history()
    entry = cast(dict[str, Any], history["m5-05-prompt"])
    outputs = cast(dict[str, Any], entry["outputs"])
    trace = copy.deepcopy(_trace())
    trace[field] = value
    outputs["13"]["sigmax_advanced_workflow_compatibility_contract"] = [
        json.dumps(
            trace,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    ]

    with pytest.raises(ScheduleContractError, match="advanced-workflow"):
        _harness().verify_advanced_workflow_contract_history(
            history,
            prompt_id="m5-05-prompt",
        )
