"""M5-03 exact-host deterministic Flow Euler harness tests."""

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
    spec = importlib.util.spec_from_file_location("sigmax_m5_03_host_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("ComfyUI E2E harness is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fingerprint(character: str) -> str:
    return "sha256:" + character * 64


def _trace() -> dict[str, object]:
    return {
        "full_effective_model_evaluations": 3,
        "full_effective_transitions": 3,
        "full_result_fingerprint": (
            "sha256:a24cbd32a60a671ef5c69d3def61caeacc550d224922d94d73d87caa7c360cc7"
        ),
        "full_scheduler_indexes": [0, 1, 2],
        "global_mutation": False,
        "model_weights_used": False,
        "negative_rejections": {
            "invalid_terminal_evaluator_calls": 0,
            "resume_mismatch_evaluator_calls": 0,
        },
        "native_full_max_abs_error_hex": "0x0.0p+0",
        "native_full_mean_abs_error_hex": "0x0.0p+0",
        "native_partial_max_abs_error_hex": "0x0.0p+0",
        "partial_effective_model_evaluations": 2,
        "partial_effective_transitions": 2,
        "partial_result_fingerprint": (
            "sha256:6b2967fc320ce24cd4beeb1c70863742461c78462106bef5f0491b2cbe3582b8"
        ),
        "partial_scheduler_indexes": [1, 2],
        "python_version": "3.13.9",
        "resumed_matches_full": True,
        "resumed_result_fingerprint": (
            "sha256:a24cbd32a60a671ef5c69d3def61caeacc550d224922d94d73d87caa7c360cc7"
        ),
        "sampler_execution_performed": True,
        "schedule_fingerprint": (
            "sha256:d63e9988942758f87cf65135dbfd48371536d466abfa38f942652c72674a772f"
        ),
        "schema": "sigmax.flow-euler-host-contract/1",
        "status": "succeeded",
        "terminal_model_evaluations": 0,
        "torch_version": "2.13.0+cpu",
    }


def _history(*, prompt_id: str = "m5-03-prompt") -> dict[str, object]:
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
            "outputs": {"11": {"sigmax_flow_euler_contract": [canonical]}},
            "prompt": [0, "fixture", harness.build_flow_euler_contract_api_prompt()],
            "status": {"completed": True, "status_str": "success"},
        }
    }


def test_flow_euler_host_prompt_and_parser_flag_are_explicit() -> None:
    harness = _harness()

    assert harness.build_flow_euler_contract_api_prompt() == {
        "11": {"class_type": "SigmaxTest.FlowEulerContractProbe", "inputs": {}}
    }
    assert harness._parser().parse_args(["--flow-euler-contract-only"]).flow_euler_contract_only


def test_flow_euler_probe_is_registered_and_calls_only_model_free_native_euler() -> None:
    tree = ast.parse(H3_TEST_PACK.read_text(encoding="utf-8"))
    probe = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowEulerContractProbe"
    )
    calls = {
        node.func.id
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "sample_euler" in calls
    assert "SigmaxTest.FlowEulerContractProbe" in H3_TEST_PACK.read_text(encoding="utf-8")
    assert "common_ksampler" not in calls


def test_flow_euler_host_history_verifier_accepts_bounded_exact_contract() -> None:
    harness = _harness()
    summary = harness.verify_flow_euler_contract_history(
        _history(),
        prompt_id="m5-03-prompt",
    )

    assert summary == _trace()
    assert summary["resumed_matches_full"] is True
    assert summary["terminal_model_evaluations"] == 0
    transition = harness.build_verified_host_repeat_transition(
        lane="H2_FLOW_EULER_CONTRACT_M5_03",
        first_summary=summary,
        repeat_summary=summary,
    )
    assert transition["transition"] == "pass_to_pass"
    assert transition["accepted"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampler_execution_performed", False),
        ("global_mutation", True),
        ("model_weights_used", True),
        ("terminal_model_evaluations", 1),
        ("resumed_result_fingerprint", _fingerprint("9")),
        ("native_full_max_abs_error_hex", "0x1.0p-10"),
        ("python_version", r"C:\private\python.exe"),
    ],
)
def test_flow_euler_host_history_verifier_rejects_tampered_evidence(
    field: str,
    value: object,
) -> None:
    history = _history()
    entry = cast(dict[str, Any], history["m5-03-prompt"])
    outputs = cast(dict[str, Any], entry["outputs"])
    trace = copy.deepcopy(_trace())
    trace[field] = value
    outputs["11"]["sigmax_flow_euler_contract"] = [
        json.dumps(
            trace, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    ]

    with pytest.raises(ScheduleContractError, match="flow-euler contract"):
        _harness().verify_flow_euler_contract_history(history, prompt_id="m5-03-prompt")
