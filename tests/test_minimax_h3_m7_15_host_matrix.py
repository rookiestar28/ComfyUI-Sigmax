"""M7-15 exact supported-host ten-scheduler matrix contracts."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles.minimax_h3_scheduler_contract import (
    MINIMAX_H3_NATIVE_SCHEDULERS,
)
from scripts import run_comfyui_e2e as harness


def test_m7_15_matrix_is_exact_deterministic_and_public_contract_bounded() -> None:
    first = harness.build_minimax_h3_m7_15_cases()
    second = harness.build_minimax_h3_m7_15_cases()

    assert first == second
    assert len(first) == 126
    assert len({case.case_id for case in first}) == 126
    assert all(case.scheduler in MINIMAX_H3_NATIVE_SCHEDULERS for case in first)
    assert all(case.dtype == "float32" for case in first)

    base_full = [case for case in first if case.lane == "base_full"]
    base_slice = [case for case in first if case.lane == "base_slice"]
    turbo = [case for case in first if case.lane == "turbo_full"]
    assert (len(base_full), len(base_slice), len(turbo)) == (72, 18, 36)
    assert {case.steps for case in base_full} == {2, 4, 8, 20}
    assert {(case.start_step, case.end_step) for case in base_slice} == {(1, 4)}
    assert {case.steps for case in turbo} == {4, 8}
    assert {case.recipe_id for case in turbo} == {
        "h3.fl2va.lightx2v-turbo-4-v0.1-544p",
        "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
        "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
        "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
    }
    assert not any(
        case.recipe_id == "h3.fl2va.lightx2v-turbo-8-v1.0-544p" and case.steps == 4
        for case in turbo
    )


def _case(case_id: str) -> Any:
    return next(case for case in harness.build_minimax_h3_m7_15_cases() if case.case_id == case_id)


@pytest.mark.parametrize(
    "case_id",
    [
        "base.fl2va.karras.s2.full",
        "base.ref2va.ddim_uniform.s8.slice-1-4",
        ("turbo.fl2va.h3.fl2va.lightx2v-turbo-4-v1.0-768p.linear_quadratic.s4.full"),
    ],
)
def test_m7_15_prompt_binds_exact_case_model_scheduler_recipe_and_probe(case_id: str) -> None:
    case = _case(case_id)
    prompt = harness.build_minimax_h3_native_matrix_h2_api_prompt(case)

    source = cast(dict[str, Any], prompt["1"])
    schedule = cast(dict[str, Any], prompt["2"])
    probe = cast(dict[str, Any], prompt["5"])
    assert source == {
        "class_type": "SigmaxTest.MiniMaxH3NativeModelSource",
        "inputs": {
            "audio_shift": case.audio_shift,
            "source_mode": "h3",
            "video_shift": case.video_shift,
        },
    }
    assert schedule["class_type"] == "Sigmax.MiniMaxH3SigmaScheduler"
    assert schedule["inputs"] == {
        "end_step": case.end_step,
        "model": ["1", 0],
        **({"turbo": case.recipe_id} if case.recipe_id is not None else {}),
        "scheduler": case.scheduler,
        "start_step": case.start_step,
        "steps": case.steps,
        "variant": case.variant,
    }
    assert probe["class_type"] == "SigmaxTest.MiniMaxH3NativeScheduleProbe"
    assert probe["inputs"] == {
        "audio_shift": case.audio_shift,
        "case_id": case.case_id,
        "end_step": case.end_step,
        "model": ["1", 0],
        "recipe_id": case.recipe_id or "",
        "schedule_info": ["2", 1],
        "scheduler": case.scheduler,
        "sigmas": ["2", 0],
        "start_step": case.start_step,
        "steps": case.steps,
        "variant": case.variant,
        "video_shift": case.video_shift,
    }


def _matrix_history(
    case: Any,
    *,
    host_version: str = "0.32.0",
    max_abs_error: float = 0.0,
) -> dict[str, Any]:
    raw = [0.95, 0.8, 0.6, 0.4, 0.2, 0.0]
    normalized = raw[-(case.steps + 1) :]
    available = len(normalized) - 1
    effective_end = available if case.end_step == -1 else case.end_step
    output = normalized[case.start_step : effective_end + 1]
    task = "fl2va" if case.variant == "H3 Base FL2VA" else "ref2va"
    sampling_api = (
        "model_sampling_discrete_flow_h3_v030"
        if host_version == "0.30.0"
        else "model_sampling_av_v032"
    )
    trace = {
        "basic_scheduler_sigmas": normalized,
        "case_id": case.case_id,
        "finite": True,
        "max_abs_error": max_abs_error,
        "mean_abs_error": 0.0,
        "monotonic_nonincreasing": True,
        "raw_reference_sigmas": raw,
        "reference_sigmas": output,
        "schedule_info": {
            "lane": "m4_17_comfyui_native_scheduler",
            "mode": "experimental_comfyui_native_scheduler",
            "scheduler": {
                "counts": {
                    "actual_sigmas": len(output),
                    "actual_transitions": len(output) - 1,
                    "raw_sigmas": len(raw),
                    "requested_steps": case.steps,
                },
                "dtype": "float32",
                "fingerprints": {
                    "contract": "sha256:" + "a" * 64,
                    "output": "sha256:" + "b" * 64,
                },
                "host": {"observed_version": host_version},
                "model_task": task,
                "owner": "comfyui_native",
                "recipe_id": case.recipe_id,
                "sampling_api": sampling_api,
                "scheduler": case.scheduler,
                "shift": {
                    "already_applied": True,
                    "audio": case.audio_shift,
                    "video": case.video_shift,
                },
                "slicing": {
                    "end_step": effective_end,
                    "start_step": case.start_step,
                },
                "terminal": {
                    "included": output[-1] == 0.0,
                    "value": output[-1],
                },
            },
        },
        "scheduler": case.scheduler,
        "schema": "sigmax.minimax-h3-native-matrix-trace/1",
        "sigmas": output,
        "steps": case.steps,
    }
    prompt = harness.build_minimax_h3_native_matrix_h2_api_prompt(case)
    return {
        "prompt-matrix": {
            "prompt": [0, "prompt-matrix", prompt, {}, ["5"]],
            "outputs": {
                "5": {
                    "sigmax_minimax_h3_native_h2": [
                        json.dumps(
                            trace,
                            allow_nan=False,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    ]
                }
            },
            "status": {"completed": True, "messages": [], "status_str": "success"},
        }
    }


@pytest.mark.parametrize(
    "case_id",
    [
        "base.fl2va.beta.s4.full",
        "base.ref2va.ddim_uniform.s8.slice-1-4",
        ("turbo.fl2va.h3.fl2va.lightx2v-turbo-8-v1.0-544p.kl_optimal.s8.full"),
    ],
)
def test_m7_15_history_verifier_returns_full_numerical_case_evidence(case_id: str) -> None:
    case = _case(case_id)
    summary = harness.verify_minimax_h3_native_matrix_h2_history(
        _matrix_history(case), prompt_id="prompt-matrix", case=case
    )

    assert summary["case"] == case.projection()
    assert summary["status"] == "succeeded"
    assert summary["max_abs_error"] == 0.0
    assert summary["mean_abs_error"] == 0.0
    assert summary["raw_reference_sigmas"]
    assert summary["basic_scheduler_sigmas"]
    assert summary["reference_sigmas"] == summary["sigmas"]
    sigma_values = cast(list[object], summary["sigmas"])
    assert summary["actual_sigmas"] == len(sigma_values)
    assert summary["actual_transitions"] == len(sigma_values) - 1
    assert summary["dtype"] == "float32"
    assert summary["contract_fingerprint"] == "sha256:" + "a" * 64
    assert summary["output_fingerprint"] == "sha256:" + "b" * 64


def test_m7_15_history_verifier_rejects_any_nonzero_same_object_difference() -> None:
    case = _case("base.fl2va.simple.s4.full")
    with pytest.raises(ScheduleContractError, match="drifted"):
        harness.verify_minimax_h3_native_matrix_h2_history(
            _matrix_history(case, max_abs_error=1e-7),
            prompt_id="prompt-matrix",
            case=case,
        )


@pytest.mark.parametrize(
    ("case_id", "expected_reason"),
    [
        ("missing_model", "MODEL_REQUIRED"),
        ("non_h3_model", "MODEL_FAMILY_MISMATCH"),
        ("base_shift_mismatch", "SHIFT_MISMATCH"),
        ("turbo_shift_mismatch", "SHIFT_MISMATCH"),
    ],
)
def test_m7_15_negative_graphs_are_runtime_bounded(case_id: str, expected_reason: str) -> None:
    prompt = harness.build_minimax_h3_native_matrix_rejection_prompt(case_id)
    output = cast(dict[str, Any], prompt["5"])
    assert output["class_type"] == "SigmaxTest.MiniMaxH3NativeUnexpectedSuccessProbe"
    assert harness.minimax_h3_native_matrix_rejection_reason(case_id) == expected_reason
    assert "5" in prompt


def test_m7_15_cli_matrix_mode_is_explicit_and_requires_h3_only_lane() -> None:
    arguments = harness._parser().parse_args(["--minimax-h3-only", "--minimax-h3-scheduler-matrix"])
    assert arguments.minimax_h3_only is True
    assert arguments.minimax_h3_scheduler_matrix is True

    arguments = harness._parser().parse_args(["--minimax-h3-scheduler-matrix"])
    assert arguments.minimax_h3_only is False
    assert arguments.minimax_h3_scheduler_matrix is True
