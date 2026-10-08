"""Receipt verification rejects vector, metadata and native-source drift."""

from __future__ import annotations

import argparse
import json
import struct
from typing import Any

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.qwen_image21_sigma_scheduler import build_qwen_image21_sigma_schedule
from comfyui_sigmax.nodes.qwen_image_sigma_scheduler import (
    bind_qwen_image_sigma_output_info,
    build_qwen_image_sigma_schedule,
)
from scripts.run_qwen_alignment_e2e import (
    QWEN21_CASES,
    native_prompt,
    qwen21_prompt,
    run,
    verify_native_history,
    verify_qwen21_history,
)


def _history() -> dict[str, Any]:
    result = build_qwen_image_sigma_schedule(
        mode="Comfy Native",
        steps=1,
        image_seq_len=0,
        strict_official=False,
        start_step=0,
        end_step=-1,
    )
    trace = {
        "sigmas": [1.0, 0.0],
        "native_sigmas": [1.0, 0.0],
        "schedule_info": json.loads(result.schedule_info_json),
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "tolerance": 2e-7,
        "mu": 1.15,
        "dtype": "float32",
        "device": "cpu",
    }
    return {
        "p": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"2": {"sigmax_qwen_native": [json.dumps(trace)]}},
        }
    }


def test_explicit_native_prompt_and_verified_complete_trace() -> None:
    assert native_prompt(steps=1)["1"]["inputs"]["strict_official"] is False
    assert verify_native_history(_history(), "p", steps=1)["status"] == "succeeded"


@pytest.mark.parametrize(
    "mutation", ["status", "vector", "native", "fingerprint", "mu", "error", "dtype"]
)
def test_native_receipt_rejects_material_drift(mutation: str) -> None:
    history = _history()
    trace = json.loads(history["p"]["outputs"]["2"]["sigmax_qwen_native"][0])
    if mutation == "status":
        history["p"]["status"]["completed"] = False
    elif mutation == "vector":
        trace["sigmas"] = [1.0]
    elif mutation == "native":
        trace["native_sigmas"][1] = 0.1
    elif mutation == "fingerprint":
        trace["schedule_info"]["fingerprints"]["output"] = "invalid"
    elif mutation == "mu":
        trace["mu"] = 0.69
    elif mutation == "error":
        trace["max_abs_error"] = 0.1
    else:
        trace["dtype"] = "float64"
    history["p"]["outputs"]["2"]["sigmax_qwen_native"] = [json.dumps(trace)]
    with pytest.raises(ScheduleContractError):
        verify_native_history(history, "p", steps=1)


def _qwen21_history(changes: dict[str, Any]) -> dict[str, Any]:
    inputs = qwen21_prompt(changes)["1"]["inputs"]
    result = build_qwen_image21_sigma_schedule(**inputs)
    vector = tuple(struct.unpack("!f", struct.pack("!f", x))[0] for x in result.sigmas)
    trace: dict[str, Any] = {
        "sigmas": vector,
        "schedule_info": json.loads(
            bind_qwen_image_sigma_output_info(result, output_sigmas=vector)
        ),
        "dtype": "float32",
        "device": "cpu",
    }
    if inputs["mode"] == "Comfy Native":
        trace.update(
            native_sigmas=vector, max_abs_error=0.0, mean_abs_error=0.0, tolerance=2e-7, mu=0.69
        )
    else:
        trace["reference_kind"] = "independent_source_golden_checked_by_runner"
    return {
        "p": {
            "status": {"status_str": "success", "completed": True},
            "outputs": {"2": {"sigmax_qwen_native": [json.dumps(trace)]}},
        }
    }


@pytest.mark.parametrize("changes", QWEN21_CASES)
def test_qwen21_receipt_complete_vector_and_independent_golden(changes: dict[str, Any]) -> None:
    assert (
        verify_qwen21_history(_qwen21_history(changes), "p", changes=changes)["status"]
        == "succeeded"
    )


@pytest.mark.parametrize(
    "mutation", ["vector", "authority", "fingerprint", "mu", "error", "dtype", "native"]
)
def test_qwen21_native_receipt_rejects_drift(mutation: str) -> None:
    changes = QWEN21_CASES[0]
    history = _qwen21_history(changes)
    trace = json.loads(history["p"]["outputs"]["2"]["sigmax_qwen_native"][0])
    if mutation == "vector":
        trace["sigmas"].pop()
    elif mutation == "authority":
        trace["schedule_info"]["target"]["source"] = "Dimensions"
    elif mutation == "fingerprint":
        trace["schedule_info"]["fingerprints"]["output"] = "invalid"
    elif mutation == "mu":
        trace["mu"] = 1.15
    elif mutation == "error":
        trace["max_abs_error"] = 0.1
    elif mutation == "dtype":
        trace["dtype"] = "float64"
    else:
        trace["native_sigmas"][0] = 0.5
    history["p"]["outputs"]["2"]["sigmax_qwen_native"] = [json.dumps(trace)]
    with pytest.raises(ScheduleContractError):
        verify_qwen21_history(history, "p", changes=changes)


def test_dynamic_receipt_rejects_framework_execution_claim() -> None:
    changes = QWEN21_CASES[3]
    history = _qwen21_history(changes)
    trace = json.loads(history["p"]["outputs"]["2"]["sigmax_qwen_native"][0])
    trace["reference_kind"] = "actual_diffusers_execution"
    history["p"]["outputs"]["2"]["sigmax_qwen_native"] = [json.dumps(trace)]
    with pytest.raises(ScheduleContractError):
        verify_qwen21_history(history, "p", changes=changes)


def test_qwen21_opt_in_rejects_the_known_good_original_only_host() -> None:
    import sys
    from pathlib import Path

    args = argparse.Namespace(
        comfyui_root=str(Path(__file__).resolve().parents[1]),
        host_python=sys.executable,
        host_version="0.30.0",
        expected_revision="14b05228cef127ce529bc0c08660770d4af3e9a8",  # pragma: allowlist secret
        include_qwen21=True,
    )
    with pytest.raises(ScheduleContractError, match=r"2\.1 requires"):
        run(args)
