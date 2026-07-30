"""Safety and evidence contracts for the real ComfyUI H1/H2 harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.inspectors import build_schedule_inspection
from comfyui_sigmax.nodes.krea2_sigma_scheduler import build_krea2_sigma_schedule
from comfyui_sigmax.nodes.raw_workflow_output import build_raw_workflow_output
from comfyui_sigmax.nodes.turbo_workflow_output import build_turbo_workflow_output

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "run_comfyui_e2e.py"


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
    ("case_id", "expected_message"),
    (
        ("raw-auto-variant", "variant must be Turbo or RAW"),
        ("raw-invalid-steps", "steps must be an integer between 1 and 10000"),
    ),
)
def test_rejected_raw_prompts_require_terminal_error_without_partial_output(
    case_id: str,
    expected_message: str,
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
