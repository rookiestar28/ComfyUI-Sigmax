"""M5-02 model-free supported-host contract harness tests."""

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
    spec = importlib.util.spec_from_file_location("sigmax_m5_02_host_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("ComfyUI E2E harness is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace() -> dict[str, object]:
    return {
        "bound_snapshot_fingerprint": (
            "sha256:6ba2e932400f6f63be7d0e9e626ea79d3add14493bcd313b102443ec789d61ae"
        ),
        "execution_receipt_fingerprint": (
            "sha256:a93ea0f52a55111376ec3c9c78a2b452a53e72df7989f9cbfe0439a7ded6775e"
        ),
        "global_mutation": False,
        "history_length": 0,
        "initial_snapshot_fingerprint": (
            "sha256:25a5a7a47f7a5220ccb04d2c7819ac13bb30c84d0f7a36b43041229f67f8faff"
        ),
        "python_version": "3.13.5",
        "receipt_bound": True,
        "receipt_status": "not_executed",
        "round_trip_stable": True,
        "sampler_execution_performed": False,
        "schema": "sigmax.sampler-state-host-contract/1",
        "spec_fingerprint": (
            "sha256:b0433c362287832b9e92868894ea03d4cb78520a90ef3054aee824da14c86887"
        ),
        "status": "ready",
    }


def _history(*, prompt_id: str = "m5-02-prompt") -> dict[str, object]:
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
            "outputs": {"10": {"sigmax_sampler_state_contract": [canonical]}},
            "prompt": [0, "fixture", harness.build_sampler_state_contract_api_prompt()],
            "status": {"completed": True, "status_str": "success"},
        }
    }


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def test_host_contract_prompt_and_parser_flag_are_explicit() -> None:
    harness = _harness()

    assert harness.build_sampler_state_contract_api_prompt() == {
        "10": {
            "class_type": "SigmaxTest.SamplerStateContractProbe",
            "inputs": {},
        }
    }
    arguments = harness._parser().parse_args(["--sampler-state-contract-only"])
    assert arguments.sampler_state_contract_only is True


def test_host_contract_probe_is_registered_and_contains_no_sampler_call() -> None:
    tree = ast.parse(H3_TEST_PACK.read_text(encoding="utf-8"))
    probe = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SamplerStateContractProbe"
    )
    calls = {_call_name(node) for node in ast.walk(probe) if isinstance(node, ast.Call)}
    source = H3_TEST_PACK.read_text(encoding="utf-8")

    assert "SigmaxTest.SamplerStateContractProbe" in source
    assert not {"sample", "sample_euler", "common_ksampler"} & calls


def test_host_contract_history_verifier_proves_bounded_nonexecution() -> None:
    harness = _harness()
    summary = harness.verify_sampler_state_contract_history(
        _history(),
        prompt_id="m5-02-prompt",
    )

    assert summary == _trace()
    assert summary["sampler_execution_performed"] is False
    assert summary["global_mutation"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampler_execution_performed", True),
        ("global_mutation", True),
        ("spec_fingerprint", "sha256:" + "0" * 64),
        ("python_version", r"A:\\private\\python.exe"),
    ],
)
def test_host_contract_history_verifier_rejects_false_or_private_evidence(
    field: str,
    value: object,
) -> None:
    history = _history()
    entry = cast(dict[str, Any], history["m5-02-prompt"])
    outputs = cast(dict[str, Any], entry["outputs"])
    trace = copy.deepcopy(_trace())
    trace[field] = value
    outputs["10"]["sigmax_sampler_state_contract"] = [
        json.dumps(
            trace, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    ]

    with pytest.raises(ScheduleContractError, match="sampler-state contract"):
        _harness().verify_sampler_state_contract_history(
            history,
            prompt_id="m5-02-prompt",
        )
