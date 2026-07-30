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
