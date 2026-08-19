"""M5-04 exact-host caller-RNG stochastic Flow Euler harness tests."""

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
    spec = importlib.util.spec_from_file_location("sigmax_m5_04_host_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("ComfyUI E2E harness is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fp(character: str) -> str:
    return "sha256:" + character * 64


def _trace() -> dict[str, object]:
    return {
        "alternate_result_fingerprint": _fp("a"),
        "different_seed_diverges": True,
        "full_effective_model_evaluations": 3,
        "full_effective_noise_draws": 3,
        "full_effective_transitions": 3,
        "full_result_fingerprint": _fp("b"),
        "global_mutation": False,
        "model_weights_used": False,
        "noise_fingerprints": [_fp("c"), _fp("d"), _fp("e")],
        "python_version": "3.13.9",
        "repeat_result_fingerprint": _fp("b"),
        "same_seed_repeat": True,
        "sampler_execution_performed": True,
        "schedule_fingerprint": _fp("f"),
        "schema": "sigmax.stochastic-flow-euler-host-contract/1",
        "status": "succeeded",
        "terminal_noise_draw": True,
        "torch_version": "2.9.0+cpu",
    }


def _history(*, prompt_id: str = "m5-04-prompt") -> dict[str, object]:
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
            "outputs": {"12": {"sigmax_stochastic_flow_euler_contract": [canonical]}},
            "prompt": [0, "fixture", harness.build_stochastic_flow_euler_contract_api_prompt()],
            "status": {"completed": True, "status_str": "success"},
        }
    }


def test_stochastic_host_prompt_flag_and_probe_registration_are_explicit() -> None:
    harness = _harness()

    assert harness.build_stochastic_flow_euler_contract_api_prompt() == {
        "12": {"class_type": "SigmaxTest.StochasticFlowEulerContractProbe", "inputs": {}}
    }
    assert (
        harness._parser()
        .parse_args(["--stochastic-flow-euler-contract-only"])
        .stochastic_flow_euler_contract_only
    )
    tree = ast.parse(H3_TEST_PACK.read_text(encoding="utf-8"))
    probe = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StochasticFlowEulerContractProbe"
    )
    assert "execute_stochastic_flow_euler" in {
        node.func.id
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "SigmaxTest.StochasticFlowEulerContractProbe" in H3_TEST_PACK.read_text(encoding="utf-8")


def test_stochastic_host_history_verifier_accepts_repeat_divergence_contract() -> None:
    harness = _harness()
    summary = harness.verify_stochastic_flow_euler_contract_history(
        _history(),
        prompt_id="m5-04-prompt",
    )
    assert summary == _trace()
    transition = harness.build_verified_host_repeat_transition(
        lane="H2_STOCHASTIC_FLOW_EULER_CONTRACT_M5_04",
        first_summary=summary,
        repeat_summary=summary,
    )
    assert transition["transition"] == "pass_to_pass"
    assert transition["accepted"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_mutation", True),
        ("model_weights_used", True),
        ("terminal_noise_draw", False),
        ("same_seed_repeat", False),
        ("alternate_result_fingerprint", _fp("b")),
        ("python_version", r"C:\private\python.exe"),
    ],
)
def test_stochastic_host_history_verifier_rejects_tampered_evidence(
    field: str,
    value: object,
) -> None:
    history = _history()
    entry = cast(dict[str, Any], history["m5-04-prompt"])
    outputs = cast(dict[str, Any], entry["outputs"])
    trace = copy.deepcopy(_trace())
    trace[field] = value
    outputs["12"]["sigmax_stochastic_flow_euler_contract"] = [
        json.dumps(
            trace, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    ]

    with pytest.raises(ScheduleContractError, match="stochastic-flow-euler"):
        _harness().verify_stochastic_flow_euler_contract_history(
            history,
            prompt_id="m5-04-prompt",
        )
