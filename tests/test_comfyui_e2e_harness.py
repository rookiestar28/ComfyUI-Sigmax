"""Safety and evidence contracts for the real ComfyUI H1/H2 harness."""

from __future__ import annotations

import copy
import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from comfyui_sigmax.adapters.registration import builtin_node_registry
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.nodes.inspectors import build_schedule_inspection
from comfyui_sigmax.nodes.krea2_sigma_scheduler import (
    build_krea2_sigma_schedule,
    sigma_output_fingerprint,
)
from comfyui_sigmax.nodes.raw_workflow_output import build_raw_workflow_output
from comfyui_sigmax.nodes.turbo_workflow_output import build_turbo_workflow_output

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "run_comfyui_e2e.py"
H3_TEST_PACK = ROOT / "tests" / "fixtures" / "comfyui_h3_nodes" / "__init__.py"
NATIVE_EULER_FIXTURE = ROOT / "tests" / "parity" / "fixtures" / "krea2_native_euler_parity_v1.json"


def _harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sigmax_comfyui_e2e", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("ComfyUI E2E harness is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle_json() -> str:
    schedule = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    inspection = build_schedule_inspection(
        sigmas=schedule.sigmas,
        schedule_info=schedule.schedule_info_json,
    )
    return build_turbo_workflow_output(
        sigmas=schedule.sigmas,
        schedule_info=schedule.schedule_info_json,
        schedule_report=inspection.report_json,
    ).bundle_json


def _raw_bundle_json(
    *,
    steps: int,
    width: int,
    height: int,
    strict_official: bool,
) -> str:
    schedule = build_krea2_sigma_schedule(
        variant="RAW",
        steps=steps,
        width=width,
        height=height,
        strict_official=strict_official,
        start_step=0,
        end_step=-1,
    )
    inspection = build_schedule_inspection(
        sigmas=schedule.sigmas,
        schedule_info=schedule.schedule_info_json,
    )
    return build_raw_workflow_output(
        sigmas=schedule.sigmas,
        schedule_info=schedule.schedule_info_json,
        schedule_report=inspection.report_json,
    ).bundle_json


def _native_euler_case() -> dict[str, Any]:
    report = cast(
        dict[str, Any],
        json.loads(NATIVE_EULER_FIXTURE.read_text(encoding="utf-8")),
    )
    return cast(dict[str, Any], report["case"])


def test_harness_defaults_to_pinned_known_good_validation() -> None:
    harness = _harness()
    arguments = harness._parser().parse_args([])

    assert arguments.host_version == "0.29.0"
    assert arguments.validation_lane == "known_good"


def test_harness_latest_host_mode_requires_explicit_version_and_lane() -> None:
    harness = _harness()
    arguments = harness._parser().parse_args(
        [
            "--host-version",
            "0.29.2",
            "--validation-lane",
            "latest_host",
        ]
    )

    assert arguments.host_version == "0.29.2"
    assert arguments.validation_lane == "latest_host"


def test_api_prompt_executes_scheduler_inspector_and_output_chain() -> None:
    prompt = _harness().build_turbo_api_prompt()

    assert prompt == {
        "1": {
            "class_type": "Sigmax.Krea2SigmaScheduler",
            "inputs": {
                "variant": "Turbo",
                "steps": 8,
                "width": 1024,
                "height": 1024,
                "strict_official": True,
                "start_step": 0,
                "end_step": -1,
            },
        },
        "2": {
            "class_type": "Sigmax.ScheduleInspector",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
            },
        },
        "3": {
            "class_type": "Sigmax.TurboWorkflowOutput",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
                "schedule_report": ["2", 0],
            },
        },
    }


def test_schedule_algebra_h2_prompt_executes_every_operation_and_probe() -> None:
    prompt = _harness().build_schedule_algebra_h2_api_prompt()

    assert [prompt[str(index)]["class_type"] for index in range(1, 8)] == [
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.ScheduleSlice",
        "Sigmax.ScheduleSlice",
        "Sigmax.ScheduleConcatenate",
        "Sigmax.ScheduleResample",
        "Sigmax.ScheduleInspector",
        "SigmaxTest.ScheduleAlgebraProbe",
    ]
    assert prompt["4"]["inputs"]["sigmas_left"] == ["2", 0]
    assert prompt["4"]["inputs"]["sigmas_right"] == ["3", 0]
    assert prompt["7"]["inputs"]["schedule_report"] == ["6", 0]


def test_checkpoint_evidence_h2_fixture_and_prompt_are_bounded(
    tmp_path: Path,
) -> None:
    harness = _harness()
    run_path = tmp_path / "owned-run"
    (run_path / "base" / "models" / "checkpoints").mkdir(parents=True)

    staged = harness._stage_checkpoint_evidence_fixture(run_path)
    raw = staged.read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_size])

    assert staged.name == "sigmax-m6-08-fixture.safetensors"
    assert len(raw) == 8 + header_size + 8
    assert header["__metadata__"] == {"is_distilled": "true"}
    assert len(header) == 5
    assert harness.build_checkpoint_evidence_h2_api_prompt() == {
        "1": {
            "class_type": "Sigmax.CheckpointEvidenceInspector",
            "inputs": {"checkpoint": "checkpoints::sigmax-m6-08-fixture.safetensors"},
        },
        "2": {
            "class_type": "SigmaxTest.CheckpointEvidenceProbe",
            "inputs": {"checkpoint_evidence": ["1", 0]},
        },
    }


def test_checkpoint_evidence_h2_history_requires_path_free_suggestion_only_report() -> None:
    harness = _harness()
    report = {
        "model_identity": {
            "confidence": "corroborating",
            "confirmed_variant": None,
            "decisive_source": None,
            "family": "krea2",
            "reason_codes": [
                "header.is_distilled.turbo",
                "tensor.krea2_family",
                "non_authoritative_variant_suggestion",
            ],
            "resolution_status": "suggested",
            "suggested_variant": "turbo",
        },
        "reason_codes": [
            "header.is_distilled.turbo",
            "tensor.krea2_family",
            "non_authoritative_variant_suggestion",
        ],
        "schema": "sigmax.checkpoint-evidence-inspection/1",
        "source": {
            "display_name": "checkpoints::sigmax-m6-08-fixture.safetensors",
            "file_bytes": 512,
            "format": "safetensors",
            "header_bytes": 496,
            "payload_bytes_read": 0,
        },
        "status": "inspected",
        "structure": {
            "data_bytes": 8,
            "dtype_counts": {"F16": 4},
            "rank_counts": {"1": 4},
            "structure_fingerprint": "sha256:" + "a" * 64,
            "tensor_count": 4,
        },
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":"))
    prompt = harness.build_checkpoint_evidence_h2_api_prompt()
    history = {
        "prompt-checkpoint": {
            "prompt": [0, "prompt-checkpoint", prompt, {}, ["2"], {}],
            "outputs": {"2": {"sigmax_checkpoint_evidence": [encoded]}},
            "status": {"completed": True, "status_str": "success"},
        }
    }

    summary = harness.verify_checkpoint_evidence_h2_history(
        history,
        prompt_id="prompt-checkpoint",
    )

    assert summary == {
        "confidence": "corroborating",
        "confirmed_variant": None,
        "payload_bytes_read": 0,
        "reason_codes": report["reason_codes"],
        "status": "succeeded",
        "suggested_variant": "turbo",
        "tensor_count": 4,
    }

    tampered = cast(dict[str, Any], copy.deepcopy(history))
    decoded = json.loads(
        tampered["prompt-checkpoint"]["outputs"]["2"]["sigmax_checkpoint_evidence"][0]
    )
    decoded["model_identity"]["confirmed_variant"] = "turbo"
    tampered["prompt-checkpoint"]["outputs"]["2"]["sigmax_checkpoint_evidence"] = [
        json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    ]
    with pytest.raises(ScheduleContractError):
        harness.verify_checkpoint_evidence_h2_history(
            tampered,
            prompt_id="prompt-checkpoint",
        )


@pytest.mark.parametrize(("variant", "steps"), (("Base", 50), ("Turbo", 8)))
def test_z_image_h2_prompt_and_history_are_variant_bound(variant: str, steps: int) -> None:
    harness = _harness()
    prompt = harness.build_z_image_h2_api_prompt(variant)
    assert prompt["1"] == {
        "class_type": "Sigmax.ZImageSigmaScheduler",
        "inputs": {
            "end_step": -1,
            "start_step": 0,
            "steps": steps,
            "strict_official": True,
            "variant": variant,
        },
    }
    info = {
        "fingerprints": {"complete": "sha256:" + "a" * 64, "output": "sha256:" + "b" * 64},
        "profile": {
            "evidence": "official",
            "id": f"z_image.{variant.casefold()}.official",
            "recipe": f"z_image.{variant.casefold()}.official",
            "variant": variant.casefold(),
            "version": "1",
        },
        "schema": "sigmax.z-image-sigma-node/1",
        "shift": {
            "dynamic": False,
            "kind": "fixed_direct_ratio",
            "ratio": 6.0 if variant == "Base" else 3.0,
        },
        "slicing": {
            "available_steps": steps,
            "end_step": steps,
            "output_steps": steps,
            "start_step": 0,
        },
        "strict_official": True,
        "warnings": [],
    }
    sigmas = [1.0 - index / steps for index in range(steps)] + [0.0]
    trace = json.dumps(
        {"schedule_info": info, "sigmas": sigmas}, sort_keys=True, separators=(",", ":")
    )
    history = {
        "prompt-z": {
            "outputs": {"2": {"sigmax_z_image_schedule": [trace]}},
            "status": {"completed": True, "status_str": "success"},
        }
    }
    summary = harness.verify_z_image_h2_history(history, prompt_id="prompt-z", variant=variant)
    assert summary["profile_id"] == f"z_image.{variant.casefold()}.official"
    assert summary["ratio"] == (6.0 if variant == "Base" else 3.0)
    assert summary["requested_transitions"] == steps
    assert summary["status"] == "succeeded"


def test_flux1_schnell_h2_prompt_and_history_pin_unshifted_four_step_recipe() -> None:
    harness = _harness()
    prompt = harness.build_flux1_schnell_h2_api_prompt()
    assert prompt["1"] == {
        "class_type": "Sigmax.Flux1SchnellSigmaScheduler",
        "inputs": {
            "end_step": -1,
            "start_step": 0,
            "steps": 4,
            "strict_official": True,
        },
    }
    info = {
        "fingerprints": {"complete": "sha256:" + "a" * 64, "output": "sha256:" + "b" * 64},
        "guidance": {"host_cfg": 1.0, "model_guidance": 0.0},
        "profile": {
            "evidence": "official",
            "id": "flux1.schnell.official",
            "recipe": "flux1.schnell.official",
            "variant": "schnell",
            "version": "1",
        },
        "schema": "sigmax.flux1-schnell-sigma-node/1",
        "shift": {"dynamic": False, "kind": "none"},
        "slicing": {
            "available_steps": 4,
            "end_step": 4,
            "output_steps": 4,
            "start_step": 0,
        },
        "strict_official": True,
        "warnings": [],
    }
    trace = json.dumps(
        {"schedule_info": info, "sigmas": [1.0, 0.75, 0.5, 0.25, 0.0]},
        sort_keys=True,
        separators=(",", ":"),
    )
    history = {
        "prompt-flux": {
            "outputs": {"2": {"sigmax_flux1_schnell_schedule": [trace]}},
            "status": {"completed": True, "status_str": "success"},
        }
    }
    summary = harness.verify_flux1_schnell_h2_history(history, prompt_id="prompt-flux")
    assert summary == {
        "numerical_fingerprint": "sha256:" + "a" * 64,
        "profile_id": "flux1.schnell.official",
        "requested_transitions": 4,
        "status": "succeeded",
    }


def _schedule_algebra_history() -> dict[str, Any]:
    source = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    values = [
        struct.unpack(">f", struct.pack(">f", source.sigmas[index]))[0] for index in range(0, 9, 2)
    ]
    fingerprint = sigma_output_fingerprint(tuple(values), domain=SigmaDomain.UNIT_FLOW)
    trace = {
        "schedule_info": {
            "evidence": "modified",
            "fingerprints": {"complete": fingerprint, "output": fingerprint},
            "operation": "resample",
            "parameters": {
                "input_steps": 8,
                "method": "index_linear_v1",
                "output_steps": 4,
            },
            "schema": "sigmax.schedule-resample-node/1",
        },
        "schedule_report": {
            "fingerprints": {
                "computed_output": fingerprint,
                "verified": True,
            },
            "source_schema": "sigmax.schedule-resample-node/1",
        },
        "sigmas": values,
    }
    return {
        "prompt-algebra": {
            "outputs": {
                "7": {
                    "sigmax_schedule_algebra": [
                        json.dumps(trace, separators=(",", ":"), sort_keys=True)
                    ]
                }
            },
            "status": {"completed": True, "status_str": "success"},
        }
    }


def test_schedule_algebra_h2_history_requires_exact_host_values_and_verified_identity() -> None:
    harness = _harness()
    history = _schedule_algebra_history()

    summary = harness.verify_schedule_algebra_h2_history(
        history,
        prompt_id="prompt-algebra",
    )

    assert summary["status"] == "succeeded"
    assert summary["evidence"] == "modified"
    assert summary["operations"] == ["slice", "concatenate", "resample", "inspect"]

    tampered = copy.deepcopy(history)
    encoded = tampered["prompt-algebra"]["outputs"]["7"]["sigmax_schedule_algebra"][0]
    trace = json.loads(encoded)
    trace["sigmas"][1] -= 0.01
    tampered["prompt-algebra"]["outputs"]["7"]["sigmax_schedule_algebra"][0] = json.dumps(trace)
    with pytest.raises(ScheduleContractError, match="sigma values drifted"):
        harness.verify_schedule_algebra_h2_history(tampered, prompt_id="prompt-algebra")


def test_schedule_algebra_h2_noop_resample_is_a_terminal_runtime_rejection() -> None:
    harness = _harness()
    prompt = harness.build_schedule_algebra_h2_noop_rejection_prompt()
    assert prompt["5"]["inputs"]["output_steps"] == 8
    history = {
        "prompt-algebra-noop": {
            "outputs": {},
            "prompt": [0, "prompt-algebra-noop", prompt, {}, ["7"], {}],
            "status": {
                "completed": False,
                "messages": [
                    [
                        "execution_error",
                        {
                            "current_outputs": [],
                            "exception_message": "resampling must change the transition count\n",
                            "exception_type": (
                                "comfyui_sigmax.core.schedule_contracts.ScheduleContractError"
                            ),
                            "executed": ["1", "2", "3", "4"],
                            "node_id": "5",
                            "node_type": "Sigmax.ScheduleResample",
                            "prompt_id": "prompt-algebra-noop",
                        },
                    ]
                ],
                "status_str": "error",
            },
        }
    }

    summary = harness.verify_schedule_algebra_h2_noop_rejection(
        history,
        prompt_id="prompt-algebra-noop",
    )

    assert summary == {
        "boundary": "runtime_execution",
        "case_id": "algebra-noop-resample",
        "exception_type": "comfyui_sigmax.core.schedule_contracts.ScheduleContractError",
        "partial_output": False,
        "prompt_created": True,
        "reason_code": "input.algebra_noop",
        "status": "error",
    }


def test_h3_prompt_connects_one_schedule_to_native_probe_and_artifact_output() -> None:
    prompt = _harness().build_native_euler_h3_api_prompt()

    assert prompt == {
        "1": {
            "class_type": "Sigmax.Krea2SigmaScheduler",
            "inputs": {
                "variant": "Turbo",
                "steps": 8,
                "width": 1024,
                "height": 1024,
                "strict_official": True,
                "start_step": 0,
                "end_step": -1,
            },
        },
        "2": {
            "class_type": "Sigmax.ScheduleInspector",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
            },
        },
        "3": {
            "class_type": "Sigmax.TurboWorkflowOutput",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
                "schedule_report": ["2", 0],
            },
        },
        "4": {
            "class_type": "SigmaxTest.NativeEulerProbe",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
            },
        },
    }


def test_h3_partial_schedule_prompt_targets_explicit_runtime_rejection() -> None:
    prompt = _harness().build_native_euler_h3_partial_rejection_prompt()

    assert prompt == {
        "1": {
            "class_type": "Sigmax.Krea2SigmaScheduler",
            "inputs": {
                "variant": "Turbo",
                "steps": 8,
                "width": 1024,
                "height": 1024,
                "strict_official": True,
                "start_step": 1,
                "end_step": -1,
            },
        },
        "4": {
            "class_type": "SigmaxTest.NativeEulerProbe",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
            },
        },
    }


def test_h3_test_pack_is_namespaced_staged_only_and_not_public(
    tmp_path: Path,
) -> None:
    harness = _harness()
    run_path = tmp_path / "owned-run"
    (run_path / "base" / "custom_nodes").mkdir(parents=True)

    staged = harness._stage_h3_test_pack(run_path)

    assert H3_TEST_PACK.is_file()
    assert staged == (run_path / "base" / "custom_nodes" / "ComfyUI-Sigmax-H3" / "__init__.py")
    assert staged.read_bytes() == H3_TEST_PACK.read_bytes()
    assert "SigmaxTest.NativeEulerProbe" not in builtin_node_registry().class_mappings()


def _native_euler_h3_history(*, case: dict[str, Any] | None = None) -> dict[str, Any]:
    complete = case or _native_euler_case()
    trace = {
        field: complete[field]
        for field in (
            "counts",
            "deterministic_rerun",
            "initial_state",
            "native_final",
            "native_steps",
            "rerun_final",
            "sigmas",
            "steps",
        )
    }
    return {
        "prompt-h3": {
            "prompt": [
                0,
                "prompt-h3",
                _harness().build_native_euler_h3_api_prompt(),
                {},
                ["3", "4"],
                {},
            ],
            "outputs": {
                "3": {"sigmax_execution_bundle": [_bundle_json()]},
                "4": {
                    "sigmax_native_euler_trace": [
                        json.dumps(
                            trace,
                            allow_nan=False,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    ]
                },
            },
            "status": {
                "completed": True,
                "status_str": "success",
                "messages": [],
            },
        }
    }


def test_h3_history_builds_success_receipt_only_after_complete_native_trace() -> None:
    summary = _harness().verify_native_euler_h3_history(
        _native_euler_h3_history(),
        prompt_id="prompt-h3",
    )

    assert summary["status"] == "succeeded"
    assert summary["sampler_id"] == "comfy.euler"
    assert summary["counts"] == {
        "effective_model_evaluations": 8,
        "effective_transitions": 8,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }
    assert summary["native_step_count"] == 8
    assert summary["deterministic_rerun"] is True
    assert summary["final_state"] == _native_euler_case()["native_final"]
    assert summary["shift_count"] == 1
    assert summary["schedule_ownership"] == "external_sigmas"
    assert summary["noise_ownership"] == {
        "model": "none",
        "sampler": "none",
        "schedule": "none",
    }
    assert summary["unsupported_features"] == [
        "advanced_workflows",
        "partial_denoise_execution",
        "resume",
        "stochastic_euler",
    ]
    receipt = cast(dict[str, Any], summary["receipt"])
    assert receipt["execution"] == {"reason_code": None, "status": "succeeded"}
    assert receipt["counts"] == summary["counts"]
    assert receipt["artifact"]["numerical_fingerprint"] == (
        "sha256:24984ad4412a3c47103a52cfe3af16bb9df8789f98401d9fc281b3f6ca0892ac"
    )


def test_h3_partial_schedule_is_terminal_rejection_without_receipt() -> None:
    prompt = _harness().build_native_euler_h3_partial_rejection_prompt()
    history = {
        "prompt-h3-partial": {
            "prompt": [0, "prompt-h3-partial", prompt, {}, ["4"]],
            "outputs": {},
            "status": {
                "completed": False,
                "status_str": "error",
                "messages": [
                    [
                        "execution_start",
                        {"prompt_id": "prompt-h3-partial", "timestamp": 0},
                    ],
                    [
                        "execution_cached",
                        {
                            "nodes": [],
                            "prompt_id": "prompt-h3-partial",
                            "timestamp": 0,
                        },
                    ],
                    [
                        "execution_error",
                        {
                            "prompt_id": "prompt-h3-partial",
                            "node_id": "4",
                            "node_type": "SigmaxTest.NativeEulerProbe",
                            "executed": ["1"],
                            "exception_message": (
                                "H3 sigmas must be one float32 eight-transition schedule\n"
                            ),
                            "exception_type": "ValueError",
                            "traceback": ["bounded"],
                            "current_inputs": {},
                            "current_outputs": ["4"],
                            "timestamp": 1,
                        },
                    ],
                ],
            },
        }
    }

    expected = {
        "case_id": "partial_denoise_execution",
        "exception_type": "ValueError",
        "node_id": "4",
        "partial_output": False,
        "reason_code": "execution.partial_denoise_unsupported",
        "receipt_created": False,
        "status": "error",
    }
    assert (
        _harness().verify_native_euler_h3_partial_rejection(
            history,
            prompt_id="prompt-h3-partial",
        )
        == expected
    )

    cached_history = copy.deepcopy(history)
    cached_entry = cast(dict[str, Any], cached_history["prompt-h3-partial"])
    cached_status = cast(dict[str, Any], cached_entry["status"])
    cached_messages = cast(list[Any], cached_status["messages"])
    cast(dict[str, Any], cached_messages[1][1])["nodes"] = ["1"]
    cast(dict[str, Any], cached_messages[2][1])["executed"] = []
    assert (
        _harness().verify_native_euler_h3_partial_rejection(
            cached_history,
            prompt_id="prompt-h3-partial",
        )
        == expected
    )

    for cached_nodes, executed_nodes in (
        ([], []),
        (["1"], ["1"]),
        (["2"], []),
    ):
        invalid_history = copy.deepcopy(history)
        invalid_entry = cast(dict[str, Any], invalid_history["prompt-h3-partial"])
        invalid_status = cast(dict[str, Any], invalid_entry["status"])
        invalid_messages = cast(list[Any], invalid_status["messages"])
        cast(dict[str, Any], invalid_messages[1][1])["nodes"] = cached_nodes
        cast(dict[str, Any], invalid_messages[2][1])["executed"] = executed_nodes
        with pytest.raises(ScheduleContractError):
            _harness().verify_native_euler_h3_partial_rejection(
                invalid_history,
                prompt_id="prompt-h3-partial",
            )


def test_host_repeat_transition_retains_attempts_and_stable_rejection_reason() -> None:
    summary = {
        "case_id": "partial_denoise_execution",
        "partial_output": False,
        "reason_code": "execution.partial_denoise_unsupported",
        "receipt_created": False,
        "status": "error",
    }

    transition = _harness().build_verified_host_repeat_transition(
        lane="H3_EULER_M5_01",
        first_summary=summary,
        repeat_summary=copy.deepcopy(summary),
    )

    assert transition["accepted"] is True
    assert transition["transition"] == "pass_to_pass"
    assert transition["first"]["ordinal"] == 1
    assert transition["repeat"]["ordinal"] == 2
    assert transition["first"]["observed_status"] == "error"
    assert transition["first"]["reason_code"] == "execution.partial_denoise_unsupported"
    assert transition["first"]["result_fingerprint"] == (transition["repeat"]["result_fingerprint"])


def test_host_repeat_transition_types_finite_float_evidence() -> None:
    summary = {
        "mu": 1.15,
        "status": "not_executed",
    }

    transition = _harness().build_verified_host_repeat_transition(
        lane="H2_RAW_M3_06",
        first_summary=summary,
        repeat_summary=copy.deepcopy(summary),
    )

    assert transition["accepted"] is True
    assert transition["first"]["result_fingerprint"] == (transition["repeat"]["result_fingerprint"])

    summary["mu"] = float("nan")
    with pytest.raises(ScheduleContractError):
        _harness().build_verified_host_repeat_transition(
            lane="H2_RAW_M3_06",
            first_summary=summary,
            repeat_summary=copy.deepcopy(summary),
        )


@pytest.mark.parametrize(
    "mutation",
    ("changed_result", "missing_status", "missing_reason"),
)
def test_host_repeat_transition_fails_closed_on_drift_or_incomplete_summary(
    mutation: str,
) -> None:
    first = {
        "case_id": "partial_denoise_execution",
        "reason_code": "execution.partial_denoise_unsupported",
        "status": "error",
    }
    repeat = copy.deepcopy(first)
    if mutation == "changed_result":
        repeat["case_id"] = "changed"
    elif mutation == "missing_status":
        repeat.pop("status")
    else:
        repeat.pop("reason_code")

    with pytest.raises(ScheduleContractError):
        _harness().build_verified_host_repeat_transition(
            lane="H3_EULER_M5_01",
            first_summary=first,
            repeat_summary=repeat,
        )


def test_verified_host_lane_executes_two_explicit_nonretry_attempts() -> None:
    submissions: list[int] = []

    def submit(ordinal: int) -> tuple[str, dict[str, object]]:
        submissions.append(ordinal)
        return f"prompt-{ordinal}", {"ordinal": ordinal}

    def verify(history: object, prompt_id: str) -> dict[str, object]:
        assert history == {"ordinal": int(prompt_id[-1])}
        return {"fingerprint": "stable", "status": "succeeded"}

    first, transition = _harness().execute_verified_host_repeat(
        lane="H2_TURBO_M2_05",
        submit=submit,
        verify=verify,
    )

    assert submissions == [1, 2]
    assert first == {"fingerprint": "stable", "status": "succeeded"}
    assert transition["accepted"] is True
    assert transition["first"]["ordinal"] == 1
    assert transition["repeat"]["ordinal"] == 2


def test_verified_host_lane_does_not_retry_a_failed_first_attempt() -> None:
    submissions: list[int] = []

    def submit(ordinal: int) -> tuple[str, dict[str, object]]:
        submissions.append(ordinal)
        return f"prompt-{ordinal}", {}

    def verify(history: object, prompt_id: str) -> dict[str, object]:
        raise ScheduleContractError(f"first attempt failed: {prompt_id}")

    with pytest.raises(ScheduleContractError, match="first attempt"):
        _harness().execute_verified_host_repeat(
            lane="H3_EULER_M5_01",
            submit=submit,
            verify=verify,
        )
    assert submissions == [1]


@pytest.mark.parametrize(
    "mutation",
    (
        "incomplete",
        "stale_prompt",
        "missing_artifact",
        "missing_trace",
        "short_trace",
        "wrong_count",
        "wrong_output",
        "nondeterministic",
    ),
)
def test_h3_history_rejects_partial_stale_or_tampered_execution(
    mutation: str,
) -> None:
    history = _native_euler_h3_history(case=copy.deepcopy(_native_euler_case()))
    entry = cast(dict[str, Any], history["prompt-h3"])
    if mutation == "incomplete":
        cast(dict[str, Any], entry["status"])["completed"] = False
    elif mutation == "stale_prompt":
        cast(list[Any], entry["prompt"])[2] = {}
    elif mutation == "missing_artifact":
        cast(dict[str, Any], entry["outputs"]).pop("3")
    elif mutation == "missing_trace":
        cast(dict[str, Any], entry["outputs"]).pop("4")
    else:
        outputs = cast(dict[str, Any], entry["outputs"])
        trace_text = cast(
            list[str], cast(dict[str, Any], outputs["4"])["sigmax_native_euler_trace"]
        )[0]
        trace = cast(dict[str, Any], json.loads(trace_text))
        if mutation == "short_trace":
            cast(list[Any], trace["native_steps"]).pop()
        elif mutation == "wrong_count":
            cast(dict[str, Any], trace["counts"])["effective_transitions"] = 7
        elif mutation == "wrong_output":
            cast(list[Any], cast(list[Any], trace["native_steps"])[3]["output_state"])[0] = 99
        else:
            trace["deterministic_rerun"] = False
        cast(dict[str, Any], outputs["4"])["sigmax_native_euler_trace"] = [
            json.dumps(trace, separators=(",", ":"), sort_keys=True)
        ]

    with pytest.raises(ScheduleContractError):
        _harness().verify_native_euler_h3_history(
            history,
            prompt_id="prompt-h3",
        )


@pytest.mark.parametrize(
    ("case_id", "steps", "width", "height", "strict_official"),
    (
        ("krea2-raw-official-square-1024", 52, 1024, 1024, True),
        ("krea2-raw-official-landscape-1353x761", 52, 1353, 761, True),
        ("krea2-raw-diffusers-portrait-761x1353", 28, 761, 1353, False),
    ),
)
def test_raw_api_prompts_execute_each_published_graph(
    case_id: str,
    steps: int,
    width: int,
    height: int,
    strict_official: bool,
) -> None:
    prompt = _harness().build_raw_api_prompt(case_id)

    assert prompt["1"] == {
        "class_type": "Sigmax.Krea2SigmaScheduler",
        "inputs": {
            "variant": "RAW",
            "steps": steps,
            "width": width,
            "height": height,
            "strict_official": strict_official,
            "start_step": 0,
            "end_step": -1,
        },
    }
    assert prompt["2"] == {
        "class_type": "Sigmax.ScheduleInspector",
        "inputs": {
            "sigmas": ["1", 0],
            "schedule_info": ["1", 1],
        },
    }
    assert prompt["3"] == {
        "class_type": "Sigmax.RawWorkflowOutput",
        "inputs": {
            "sigmas": ["1", 0],
            "schedule_info": ["1", 1],
            "schedule_report": ["2", 0],
        },
    }


@pytest.mark.parametrize(
    (
        "case_id",
        "steps",
        "width",
        "height",
        "strict_official",
        "effective_width",
        "effective_height",
        "image_seq_len",
        "mu",
        "numerical_fingerprint",
    ),
    (
        (
            "krea2-raw-official-square-1024",
            52,
            1024,
            1024,
            True,
            1024,
            1024,
            4096,
            0.90625,
            "sha256:5ff69c30df41c7f37eae14502155b31f23724d32427180f69118cabcd6a3ac61",
        ),
        (
            "krea2-raw-official-landscape-1353x761",
            52,
            1353,
            761,
            True,
            1360,
            768,
            4080,
            0.9045572916666667,
            "sha256:01352f42660bd3b31bbaf7548a9891273899afd375adeb68c7f7c93fd2a4f0d4",
        ),
        (
            "krea2-raw-diffusers-portrait-761x1353",
            28,
            761,
            1353,
            False,
            768,
            1360,
            4080,
            0.9045572916666667,
            "sha256:52208c5fa3780c95cce399b1f842f3fea56503e76fdf5ef4abc3069cf3108f01",
        ),
    ),
)
def test_raw_history_verifier_requires_geometry_bundle_and_metadata_reload(
    case_id: str,
    steps: int,
    width: int,
    height: int,
    strict_official: bool,
    effective_width: int,
    effective_height: int,
    image_seq_len: int,
    mu: float,
    numerical_fingerprint: str,
) -> None:
    workflow = {"version": 0.4, "extra": {"case_id": case_id}}
    history = {
        "prompt-raw": {
            "prompt": [
                0,
                "prompt-raw",
                _harness().build_raw_api_prompt(case_id),
                {"extra_pnginfo": {"workflow": workflow}},
                ["3"],
                {},
            ],
            "outputs": {
                "3": {
                    "sigmax_execution_bundle": [
                        _raw_bundle_json(
                            steps=steps,
                            width=width,
                            height=height,
                            strict_official=strict_official,
                        )
                    ],
                }
            },
            "status": {
                "completed": True,
                "status_str": "success",
                "messages": [],
            },
        }
    }

    summary = _harness().verify_raw_history(
        history,
        prompt_id="prompt-raw",
        case_id=case_id,
        submitted_workflow=workflow,
    )

    assert summary["requested"] == {"width": width, "height": height}
    assert summary["effective"] == {
        "width": effective_width,
        "height": effective_height,
    }
    assert summary["image_seq_len"] == image_seq_len
    assert summary["mu"] == mu
    assert summary["requested_transitions"] == steps
    assert summary["numerical_fingerprint"] == numerical_fingerprint
    assert summary["metadata_reloaded"] is True
    assert summary["shift_count"] == 1
    assert summary["schedule_ownership"] == "external_sigmas"
    assert summary["status"] == "not_executed"


@pytest.mark.parametrize(
    "mutation",
    ("missing_prompt_tuple", "stale_workflow", "missing_output", "incomplete"),
)
def test_raw_history_verifier_rejects_incomplete_or_stale_evidence(mutation: str) -> None:
    harness = _harness()
    case_id = "krea2-raw-official-square-1024"
    workflow = {"version": 0.4, "extra": {"case_id": case_id}}
    entry: dict[str, Any] = {
        "prompt": [
            0,
            "prompt-raw",
            harness.build_raw_api_prompt(case_id),
            {"extra_pnginfo": {"workflow": workflow}},
            ["3"],
            {},
        ],
        "outputs": {
            "3": {
                "sigmax_execution_bundle": [
                    _raw_bundle_json(
                        steps=52,
                        width=1024,
                        height=1024,
                        strict_official=True,
                    )
                ]
            }
        },
        "status": {"completed": True, "status_str": "success", "messages": []},
    }
    if mutation == "missing_prompt_tuple":
        entry.pop("prompt")
    elif mutation == "stale_workflow":
        cast(dict[str, Any], cast(list[Any], entry["prompt"])[3])["extra_pnginfo"] = {
            "workflow": {"version": 0.4, "extra": {"case_id": "stale"}}
        }
    elif mutation == "missing_output":
        entry["outputs"] = {}
    else:
        cast(dict[str, Any], entry["status"])["completed"] = False

    with pytest.raises(ScheduleContractError):
        harness.verify_raw_history(
            {"prompt-raw": entry},
            prompt_id="prompt-raw",
            case_id=case_id,
            submitted_workflow=workflow,
        )


@pytest.mark.parametrize(
    ("case_id", "expected_message", "reason_code"),
    (
        (
            "raw-auto-variant",
            "variant must be Turbo or RAW",
            "input.variant_selection_required",
        ),
        (
            "raw-invalid-steps",
            "steps must be an integer between 1 and 10000",
            "input.steps_out_of_range",
        ),
    ),
)
def test_rejected_raw_prompts_require_terminal_error_without_partial_output(
    case_id: str,
    expected_message: str,
    reason_code: str,
) -> None:
    harness = _harness()
    prompt = harness.build_raw_api_prompt("krea2-raw-official-square-1024")
    scheduler = cast(dict[str, Any], prompt["1"])
    inputs = cast(dict[str, Any], scheduler["inputs"])
    inputs["variant" if case_id == "raw-auto-variant" else "steps"] = (
        "auto" if case_id == "raw-auto-variant" else 0
    )
    history = {
        "prompt-rejected": {
            "prompt": [0, "prompt-rejected", prompt, {}, ["3"]],
            "outputs": {},
            "status": {
                "completed": False,
                "status_str": "error",
                "messages": [
                    [
                        "execution_start",
                        {
                            "prompt_id": "prompt-rejected",
                            "timestamp": 0,
                        },
                    ],
                    [
                        "execution_cached",
                        {
                            "nodes": [],
                            "prompt_id": "prompt-rejected",
                            "timestamp": 0,
                        },
                    ],
                    [
                        "execution_error",
                        {
                            "prompt_id": "prompt-rejected",
                            "node_id": "1",
                            "node_type": "Sigmax.Krea2SigmaScheduler",
                            "executed": [],
                            "exception_message": f"{expected_message}\n",
                            "exception_type": (
                                "comfyui_sigmax.core.schedule_contracts.ScheduleContractError"
                            ),
                            "traceback": ["bounded"],
                            "current_inputs": {},
                            "current_outputs": ["1", "2", "3"],
                            "timestamp": 1,
                        },
                    ],
                ],
            },
        }
    }

    summary = harness.verify_rejected_history(
        history,
        prompt_id="prompt-rejected",
        case_id=case_id,
        expected_message=expected_message,
    )

    assert summary == {
        "boundary": "runtime_execution",
        "case_id": case_id,
        "exception_type": ("comfyui_sigmax.core.schedule_contracts.ScheduleContractError"),
        "partial_output": False,
        "prompt_created": True,
        "reason_code": reason_code,
        "status": "error",
    }


def _invalid_steps_prequeue_response() -> dict[str, Any]:
    return {
        "error": {
            "type": "prompt_outputs_failed_validation",
            "message": "Prompt outputs failed validation",
            "details": "",
            "extra_info": {},
        },
        "node_errors": {
            "1": {
                "class_type": "Sigmax.Krea2SigmaScheduler",
                "dependent_outputs": ["3"],
                "errors": [
                    {
                        "type": "value_smaller_than_min",
                        "message": "Value 0 smaller than min of 1",
                        "details": "steps",
                        "extra_info": {
                            "input_name": "steps",
                            "input_config": [
                                "INT",
                                {
                                    "default": 8,
                                    "min": 1,
                                    "max": 10000,
                                },
                            ],
                            "received_value": 0,
                        },
                    }
                ],
            }
        },
    }


def test_invalid_steps_prequeue_rejection_requires_structured_node_error() -> None:
    summary = _harness().verify_prequeue_rejection(
        _invalid_steps_prequeue_response(),
        case_id="raw-invalid-steps",
    )

    assert summary == {
        "boundary": "prequeue_validation",
        "case_id": "raw-invalid-steps",
        "http_status": 400,
        "node_id": "1",
        "node_type": "Sigmax.Krea2SigmaScheduler",
        "partial_output": False,
        "prompt_created": False,
        "reason_code": "input.steps_below_minimum",
        "reason_type": "value_smaller_than_min",
        "status": "rejected",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "prompt_id",
        "top_error",
        "missing_node_error",
        "wrong_node",
        "wrong_output",
        "wrong_reason",
        "wrong_input",
        "wrong_received",
        "wrong_min",
    ),
)
def test_invalid_steps_prequeue_rejection_fails_closed_on_drift(
    mutation: str,
) -> None:
    response = _invalid_steps_prequeue_response()
    root_error = cast(dict[str, Any], response["error"])
    node_errors = cast(dict[str, Any], response["node_errors"])
    node_error = cast(dict[str, Any], node_errors["1"])
    reason = cast(dict[str, Any], cast(list[Any], node_error["errors"])[0])
    extra_info = cast(dict[str, Any], reason["extra_info"])
    input_config = cast(list[Any], extra_info["input_config"])
    constraints = cast(dict[str, Any], input_config[1])
    if mutation == "prompt_id":
        response["prompt_id"] = "unexpected"
    elif mutation == "top_error":
        root_error["type"] = "stale"
    elif mutation == "missing_node_error":
        response["node_errors"] = {}
    elif mutation == "wrong_node":
        node_error["class_type"] = "stale"
    elif mutation == "wrong_output":
        node_error["dependent_outputs"] = ["2"]
    elif mutation == "wrong_reason":
        reason["type"] = "stale"
    elif mutation == "wrong_input":
        extra_info["input_name"] = "width"
    elif mutation == "wrong_received":
        extra_info["received_value"] = 1
    else:
        constraints["min"] = 0

    with pytest.raises(ScheduleContractError):
        _harness().verify_prequeue_rejection(
            response,
            case_id="raw-invalid-steps",
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "completed",
        "missing_cached_detail",
        "wrong_cached_prompt",
        "wrong_cached_nodes",
        "wrong_current_outputs",
        "wrong_node",
        "partial_output",
        "wrong_message",
        "missing_event",
    ),
)
def test_rejected_raw_history_fails_closed_on_stale_or_partial_evidence(
    mutation: str,
) -> None:
    harness = _harness()
    case_id = "raw-auto-variant"
    prompt = harness.build_raw_api_prompt("krea2-raw-official-square-1024")
    scheduler = cast(dict[str, Any], prompt["1"])
    cast(dict[str, Any], scheduler["inputs"])["variant"] = "auto"
    message: list[Any] = [
        "execution_error",
        {
            "prompt_id": "prompt-rejected",
            "node_id": "1",
            "node_type": "Sigmax.Krea2SigmaScheduler",
            "executed": [],
            "exception_message": "variant must be Turbo or RAW\n",
            "exception_type": ("comfyui_sigmax.core.schedule_contracts.ScheduleContractError"),
            "traceback": ["bounded"],
            "current_inputs": {},
            "current_outputs": ["1", "2", "3"],
            "timestamp": 1,
        },
    ]
    entry: dict[str, Any] = {
        "prompt": [0, "prompt-rejected", prompt, {}, ["3"]],
        "outputs": {},
        "status": {
            "completed": False,
            "status_str": "error",
            "messages": [
                [
                    "execution_start",
                    {
                        "prompt_id": "prompt-rejected",
                        "timestamp": 0,
                    },
                ],
                [
                    "execution_cached",
                    {
                        "nodes": [],
                        "prompt_id": "prompt-rejected",
                        "timestamp": 0,
                    },
                ],
                message,
            ],
        },
    }
    if mutation == "completed":
        cast(dict[str, Any], entry["status"])["completed"] = True
    elif mutation == "missing_cached_detail":
        messages = cast(dict[str, Any], entry["status"])["messages"]
        cast(list[Any], messages)[1] = ["execution_cached"]
    elif mutation == "wrong_cached_prompt":
        messages = cast(dict[str, Any], entry["status"])["messages"]
        cast(dict[str, Any], cast(list[Any], messages)[1][1])["prompt_id"] = "stale"
    elif mutation == "wrong_cached_nodes":
        messages = cast(dict[str, Any], entry["status"])["messages"]
        cast(dict[str, Any], cast(list[Any], messages)[1][1])["nodes"] = "stale"
    elif mutation == "wrong_current_outputs":
        cast(dict[str, Any], message[1])["current_outputs"] = "stale"
    elif mutation == "wrong_node":
        cast(dict[str, Any], message[1])["node_id"] = "2"
    elif mutation == "partial_output":
        entry["outputs"] = {
            "3": {
                "sigmax_execution_bundle": [
                    "unexpected",
                ]
            }
        }
    elif mutation == "wrong_message":
        cast(dict[str, Any], message[1])["exception_message"] = "stale"
    else:
        cast(dict[str, Any], entry["status"])["messages"] = [
            [
                "execution_start",
                {
                    "prompt_id": "prompt-rejected",
                    "timestamp": 0,
                },
            ],
            [
                "execution_cached",
                {
                    "nodes": [],
                    "prompt_id": "prompt-rejected",
                    "timestamp": 0,
                },
            ],
        ]

    with pytest.raises(ScheduleContractError):
        harness.verify_rejected_history(
            {"prompt-rejected": entry},
            prompt_id="prompt-rejected",
            case_id=case_id,
            expected_message="variant must be Turbo or RAW",
        )


def test_history_verifier_requires_completed_output_and_truthful_bundle() -> None:
    history = {
        "prompt-1": {
            "outputs": {
                "3": {
                    "sigmax_execution_bundle": [_bundle_json()],
                }
            },
            "status": {
                "completed": True,
                "status_str": "success",
                "messages": [],
            },
        }
    }

    summary = _harness().verify_turbo_history(history, prompt_id="prompt-1")

    assert summary["status"] == "not_executed"
    assert summary["requested_transitions"] == 8
    assert summary["effective_transitions"] == 0
    assert summary["effective_model_evaluations"] == 0
    assert summary["shift_count"] == 1
    assert summary["schedule_ownership"] == "external_sigmas"
    assert summary["numerical_fingerprint"] == (
        "sha256:24984ad4412a3c47103a52cfe3af16bb9df8789f98401d9fc281b3f6ca0892ac"
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing_prompt", "incomplete", "missing_output", "queue_only"),
)
def test_history_verifier_rejects_queue_only_or_incomplete_evidence(mutation: str) -> None:
    history: dict[str, Any] = {
        "prompt-1": {
            "outputs": {"3": {"sigmax_execution_bundle": [_bundle_json()]}},
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }
    if mutation == "missing_prompt":
        history = {}
    elif mutation == "incomplete":
        cast(dict[str, Any], history["prompt-1"]["status"])["completed"] = False
    elif mutation == "missing_output":
        history["prompt-1"]["outputs"] = {}
    else:
        history["prompt-1"] = {"prompt_id": "prompt-1"}

    with pytest.raises(ScheduleContractError):
        _harness().verify_turbo_history(history, prompt_id="prompt-1")


def test_owned_temp_guard_and_redaction_fail_closed(tmp_path: Path) -> None:
    harness = _harness()
    repository = tmp_path / "repo"
    owned_root = repository / ".tmp" / "e2e"
    run_path = owned_root / "run-1"
    run_path.mkdir(parents=True)

    assert (
        harness.require_owned_run_path(
            repository_root=repository,
            owned_root=owned_root,
            candidate=run_path,
        )
        == run_path.resolve()
    )
    with pytest.raises(ScheduleContractError):
        harness.require_owned_run_path(
            repository_root=repository,
            owned_root=owned_root,
            candidate=repository,
        )

    rendered = harness.redact_text(
        f"root={repository} token=secret-value Authorization: Bearer abc.def",
        sensitive_paths=(repository,),
    )
    assert str(repository) not in rendered
    assert "secret-value" not in rendered
    assert "abc.def" not in rendered
    assert "<redacted-path>" in rendered


@pytest.mark.parametrize(
    "url",
    (
        "file:///tmp/host",
        "https://127.0.0.1:8188/object_info",
        "http://localhost:8188/object_info",
        "http://user@127.0.0.1:8188/object_info",
        "http://127.0.0.1:8188/object_info#fragment",
        "http://127.0.0.1/object_info",
    ),
)
def test_http_guard_rejects_urls_outside_exact_loopback_boundary(url: str) -> None:
    with pytest.raises(ScheduleContractError):
        _harness()._require_loopback_http_url(url)

    assert (
        _harness()._require_loopback_http_url("http://127.0.0.1:8188/object_info")
        == "http://127.0.0.1:8188/object_info"
    )


def test_canonical_wrappers_require_explicit_host_configuration() -> None:
    windows = (ROOT / "scripts" / "run_comfyui_e2e_windows.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "scripts" / "run_comfyui_e2e_linux.sh").read_text(encoding="utf-8")

    for source in (windows, linux):
        assert "COMFYUI_ROOT" in source
        assert "SIGMAX_COMFYUI_PYTHON" in source
        assert "run_comfyui_e2e.py" in source
        assert "127.0.0.1" not in source or "--listen" not in source
