"""Executable publication contract for the strict Krea 2 Turbo host workflow."""

from __future__ import annotations

import json
import struct
from dataclasses import FrozenInstanceError
from typing import Any, cast

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import (
    ExecutionStatus,
    ScheduleContractError,
    ScheduleOwnership,
    deserialize_portable_execution_bundle,
)
from comfyui_sigmax.nodes.inspectors import build_schedule_inspection
from comfyui_sigmax.nodes.krea2_sigma_scheduler import (
    bind_krea2_sigma_output_info,
    build_krea2_sigma_schedule,
)
from comfyui_sigmax.nodes.turbo_workflow_output import (
    TURBO_WORKFLOW_OUTPUT_NODE_ID,
    TURBO_WORKFLOW_OUTPUT_SCHEMA_ID,
    TurboWorkflowOutput,
    TurboWorkflowOutputResult,
    build_turbo_workflow_output,
)


def _upstream() -> tuple[tuple[float, ...], str, str]:
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
    return schedule.sigmas, schedule.schedule_info_json, inspection.report_json


def _result() -> TurboWorkflowOutputResult:
    sigmas, schedule_info, schedule_report = _upstream()
    return build_turbo_workflow_output(
        sigmas=sigmas,
        schedule_info=schedule_info,
        schedule_report=schedule_report,
    )


def test_output_node_declares_stable_v1_schema() -> None:
    inputs = TurboWorkflowOutput.INPUT_TYPES()

    assert TURBO_WORKFLOW_OUTPUT_NODE_ID == "Sigmax.TurboWorkflowOutput"
    assert TURBO_WORKFLOW_OUTPUT_SCHEMA_ID == "sigmax.turbo-workflow-output/1"
    assert TurboWorkflowOutput.RETURN_TYPES == ()
    assert TurboWorkflowOutput.RETURN_NAMES == ()
    assert TurboWorkflowOutput.FUNCTION == "publish"
    assert TurboWorkflowOutput.CATEGORY == "Sigmax/workflows"
    assert TurboWorkflowOutput.OUTPUT_NODE is True
    assert inputs == {
        "required": {
            "sigmas": ("SIGMAS",),
            "schedule_info": ("STRING", {"default": "", "multiline": True}),
            "schedule_report": ("STRING", {"default": "", "multiline": True}),
        }
    }
    assert inputs == TurboWorkflowOutput.INPUT_TYPES()
    assert inputs is not TurboWorkflowOutput.INPUT_TYPES()


def test_builtin_mapping_registers_the_executable_output_node() -> None:
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS[TURBO_WORKFLOW_OUTPUT_NODE_ID] is TurboWorkflowOutput
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[TURBO_WORKFLOW_OUTPUT_NODE_ID]
        == "Turbo Workflow Output"
    )


def test_pure_output_builds_deterministic_truthful_portable_bundle() -> None:
    result = _result()
    repeated = _result()
    bundle = deserialize_portable_execution_bundle(result.bundle_json)
    construction = bundle.artifact.construction_projection()
    receipt = bundle.receipt.projection()

    assert result == repeated
    assert result.schema_id == TURBO_WORKFLOW_OUTPUT_SCHEMA_ID
    assert result.bundle_json == repeated.bundle_json
    assert result.bundle_fingerprint == repeated.bundle_fingerprint
    assert result.artifact_construction_fingerprint == bundle.artifact.construction_fingerprint
    assert result.artifact_numerical_fingerprint == bundle.artifact.numerical_fingerprint
    assert result.receipt_fingerprint == bundle.receipt.receipt_fingerprint

    assert construction["ownership"] == {
        "schedule": ScheduleOwnership.EXTERNAL_SIGMAS.value.casefold(),
        "shift": "construction_pipeline",
    }
    transforms = cast(list[dict[str, object]], construction["transforms"])
    assert [item["id"] for item in transforms] == [
        "krea.exponential_mu",
        "terminal.append_zero",
    ]
    assert sum(item["id"] == "krea.exponential_mu" for item in transforms) == 1
    assert cast(dict[str, object], transforms[0]["parameters"])["mu"] == {
        "bits": "3ff2666666666666",
        "precision": "float64",
    }

    assert receipt["execution"] == {
        "reason_code": None,
        "status": ExecutionStatus.NOT_EXECUTED.value,
    }
    assert receipt["counts"] == {
        "effective_model_evaluations": 0,
        "effective_transitions": 0,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }
    assert cast(dict[str, object], receipt["compatibility"])["level"] == "allow"
    assert receipt["artifact"] == {
        "construction_fingerprint": bundle.artifact.construction_fingerprint,
        "numerical_fingerprint": bundle.artifact.numerical_fingerprint,
    }
    assert receipt["host"] == {
        "api_version": "legacy_v1",
        "id": "comfyui",
        "revision": "e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
        "version": "0.29.0",
    }


def test_host_node_returns_bundle_only_through_bounded_v1_ui_payload() -> None:
    sigmas, schedule_info, schedule_report = _upstream()

    response = TurboWorkflowOutput().publish(
        sigmas=list(sigmas),
        schedule_info=schedule_info,
        schedule_report=schedule_report,
    )

    assert set(response) == {"result", "ui"}
    assert response["result"] == ()
    ui = cast(dict[str, object], response["ui"])
    assert set(ui) == {"sigmax_execution_bundle"}
    values = cast(list[object], ui["sigmax_execution_bundle"])
    assert len(values) == 1
    bundle = deserialize_portable_execution_bundle(cast(str, values[0]))
    execution = cast(dict[str, object], bundle.receipt.projection()["execution"])
    assert execution["status"] == "not_executed"


def test_output_accepts_only_exact_host_float32_quantization_and_keeps_float64_artifact() -> None:
    schedule = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    host_values = tuple(
        struct.unpack(">f", struct.pack(">f", value))[0] for value in schedule.sigmas
    )
    host_info = bind_krea2_sigma_output_info(schedule, output_sigmas=host_values)
    host_report = build_schedule_inspection(
        sigmas=host_values,
        schedule_info=host_info,
    ).report_json

    result = build_turbo_workflow_output(
        sigmas=host_values,
        schedule_info=host_info,
        schedule_report=host_report,
    )
    bundle = deserialize_portable_execution_bundle(result.bundle_json)

    assert host_values != schedule.sigmas
    assert bundle.artifact.numerical_fingerprint == (
        "sha256:24984ad4412a3c47103a52cfe3af16bb9df8789f98401d9fc281b3f6ca0892ac"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "sigma_value",
        "schedule_info",
        "schedule_report",
        "raw",
        "modified_steps",
        "partial_slice",
    ),
)
def test_output_rejects_mismatched_or_noncanonical_upstream_evidence(mutation: str) -> None:
    sigmas, schedule_info, schedule_report = _upstream()
    changed_sigmas = sigmas
    changed_info = schedule_info
    changed_report = schedule_report

    if mutation == "sigma_value":
        changed_sigmas = (sigmas[0] - 0.01, *sigmas[1:])
    elif mutation == "schedule_info":
        projection = cast(dict[str, Any], json.loads(schedule_info))
        projection["strict_official"] = False
        changed_info = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    elif mutation == "schedule_report":
        projection = cast(dict[str, Any], json.loads(schedule_report))
        cast(dict[str, object], projection["fingerprints"])["verified"] = False
        changed_report = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    else:
        schedule = build_krea2_sigma_schedule(
            variant="RAW" if mutation == "raw" else "Turbo",
            steps=52 if mutation == "raw" else (12 if mutation == "modified_steps" else 8),
            width=1024,
            height=1024,
            strict_official=mutation not in {"modified_steps"},
            start_step=2 if mutation == "partial_slice" else 0,
            end_step=6 if mutation == "partial_slice" else -1,
        )
        changed_sigmas = schedule.sigmas
        changed_info = schedule.schedule_info_json
        changed_report = build_schedule_inspection(
            sigmas=changed_sigmas,
            schedule_info=changed_info,
        ).report_json

    with pytest.raises(ScheduleContractError):
        build_turbo_workflow_output(
            sigmas=changed_sigmas,
            schedule_info=changed_info,
            schedule_report=changed_report,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"sigmas": object()},
        {"schedule_info": object()},
        {"schedule_report": object()},
        {"schedule_info": ""},
        {"schedule_report": ""},
    ),
)
def test_output_rejects_wrong_host_boundary_types(changes: dict[str, object]) -> None:
    sigmas, schedule_info, schedule_report = _upstream()
    arguments: dict[str, object] = {
        "sigmas": sigmas,
        "schedule_info": schedule_info,
        "schedule_report": schedule_report,
    }
    arguments.update(changes)

    with pytest.raises(ScheduleContractError):
        TurboWorkflowOutput().publish(**arguments)


def test_output_result_is_immutable() -> None:
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.bundle_json = "{}"  # type: ignore[misc]
