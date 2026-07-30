"""Executable publication contract for canonical Krea 2 RAW host workflows."""

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
from comfyui_sigmax.nodes.raw_workflow_output import (
    RAW_WORKFLOW_OUTPUT_NODE_ID,
    RAW_WORKFLOW_OUTPUT_SCHEMA_ID,
    RawWorkflowOutput,
    RawWorkflowOutputResult,
    build_raw_workflow_output,
)

CASES = (
    (
        52,
        1024,
        1024,
        True,
        1024,
        1024,
        4096,
        0.90625,
        "krea2.raw.official-full-52",
        "official",
        "sha256:5ff69c30df41c7f37eae14502155b31f23724d32427180f69118cabcd6a3ac61",
    ),
    (
        52,
        1353,
        761,
        True,
        1360,
        768,
        4080,
        0.9045572916666667,
        "krea2.raw.official-full-52",
        "official",
        "sha256:01352f42660bd3b31bbaf7548a9891273899afd375adeb68c7f7c93fd2a4f0d4",
    ),
    (
        28,
        761,
        1353,
        False,
        768,
        1360,
        4080,
        0.9045572916666667,
        "krea2.raw.diffusers-reference-28",
        "framework_reference",
        "sha256:52208c5fa3780c95cce399b1f842f3fea56503e76fdf5ef4abc3069cf3108f01",
    ),
)


def _upstream(
    *,
    steps: int = 52,
    width: int = 1024,
    height: int = 1024,
    strict_official: bool = True,
) -> tuple[tuple[float, ...], str, str]:
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
    return schedule.sigmas, schedule.schedule_info_json, inspection.report_json


def _result() -> RawWorkflowOutputResult:
    sigmas, schedule_info, schedule_report = _upstream()
    return build_raw_workflow_output(
        sigmas=sigmas,
        schedule_info=schedule_info,
        schedule_report=schedule_report,
    )


def test_output_node_declares_stable_v1_schema() -> None:
    inputs = RawWorkflowOutput.INPUT_TYPES()

    assert RAW_WORKFLOW_OUTPUT_NODE_ID == "Sigmax.RawWorkflowOutput"
    assert RAW_WORKFLOW_OUTPUT_SCHEMA_ID == "sigmax.raw-workflow-output/1"
    assert RawWorkflowOutput.RETURN_TYPES == ()
    assert RawWorkflowOutput.RETURN_NAMES == ()
    assert RawWorkflowOutput.FUNCTION == "publish"
    assert RawWorkflowOutput.CATEGORY == "Sigmax/workflows"
    assert RawWorkflowOutput.OUTPUT_NODE is True
    assert inputs == {
        "required": {
            "sigmas": ("SIGMAS",),
            "schedule_info": ("STRING", {"default": "", "multiline": True}),
            "schedule_report": ("STRING", {"default": "", "multiline": True}),
        }
    }
    assert inputs == RawWorkflowOutput.INPUT_TYPES()
    assert inputs is not RawWorkflowOutput.INPUT_TYPES()


def test_builtin_mapping_registers_the_raw_output_node() -> None:
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS[RAW_WORKFLOW_OUTPUT_NODE_ID] is RawWorkflowOutput
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[RAW_WORKFLOW_OUTPUT_NODE_ID]
        == "RAW Workflow Output"
    )


@pytest.mark.parametrize(
    (
        "steps",
        "width",
        "height",
        "strict_official",
        "effective_width",
        "effective_height",
        "image_seq_len",
        "mu",
        "recipe_id",
        "evidence",
        "numerical_fingerprint",
    ),
    CASES,
)
def test_output_builds_canonical_raw_bundle_for_every_published_case(
    steps: int,
    width: int,
    height: int,
    strict_official: bool,
    effective_width: int,
    effective_height: int,
    image_seq_len: int,
    mu: float,
    recipe_id: str,
    evidence: str,
    numerical_fingerprint: str,
) -> None:
    sigmas, schedule_info, schedule_report = _upstream(
        steps=steps,
        width=width,
        height=height,
        strict_official=strict_official,
    )
    result = build_raw_workflow_output(
        sigmas=sigmas,
        schedule_info=schedule_info,
        schedule_report=schedule_report,
    )
    bundle = deserialize_portable_execution_bundle(result.bundle_json)
    construction = bundle.artifact.construction_projection()
    receipt = bundle.receipt.projection()
    requested = cast(dict[str, Any], construction["requested"])
    effective = cast(dict[str, Any], construction["effective"])

    assert result.schema_id == RAW_WORKFLOW_OUTPUT_SCHEMA_ID
    assert result.artifact_numerical_fingerprint == numerical_fingerprint
    assert bundle.artifact.numerical_fingerprint == numerical_fingerprint
    assert {key: requested[key] for key in ("height", "steps", "width")} == {
        "height": height,
        "steps": steps,
        "width": width,
    }
    assert {key: effective[key] for key in ("height", "steps", "width")} == {
        "height": effective_height,
        "steps": steps,
        "width": effective_width,
    }
    assert requested["profile"] == effective["profile"] == "krea2.raw.official"
    assert cast(dict[str, object], construction["evidence"])["level"] == evidence
    assert cast(dict[str, object], construction["source"])["id"] == (
        "krea.krea2.official" if evidence == "official" else "diffusers.krea2.framework"
    )

    transforms = cast(list[dict[str, Any]], construction["transforms"])
    assert [item["id"] for item in transforms] == [
        "krea.exponential_mu",
        "terminal.append_zero",
    ]
    assert sum(item["id"] == "krea.exponential_mu" for item in transforms) == 1
    assert transforms[0]["parameters"]["mu"] == {
        "bits": struct.pack(">d", mu).hex(),
        "precision": "float64",
    }
    base_grid = cast(dict[str, Any], construction["base_grid"])
    assert base_grid["parameters"] == {
        "image_seq_len": image_seq_len,
        "recipe": recipe_id,
        "steps": steps,
    }
    assert construction["ownership"] == {
        "schedule": ScheduleOwnership.EXTERNAL_SIGMAS.value.casefold(),
        "shift": "construction_pipeline",
    }
    assert receipt["execution"] == {
        "reason_code": None,
        "status": ExecutionStatus.NOT_EXECUTED.value,
    }
    assert receipt["counts"] == {
        "effective_model_evaluations": 0,
        "effective_transitions": 0,
        "requested_model_evaluations": steps,
        "requested_transitions": steps,
    }


def test_output_is_deterministic_and_returns_only_bounded_history_payload() -> None:
    sigmas, schedule_info, schedule_report = _upstream()
    first = build_raw_workflow_output(
        sigmas=sigmas,
        schedule_info=schedule_info,
        schedule_report=schedule_report,
    )
    second = build_raw_workflow_output(
        sigmas=sigmas,
        schedule_info=schedule_info,
        schedule_report=schedule_report,
    )

    assert first == second
    response = RawWorkflowOutput().publish(
        sigmas=list(sigmas),
        schedule_info=schedule_info,
        schedule_report=schedule_report,
    )
    assert set(response) == {"result", "ui"}
    assert response["result"] == ()
    ui = cast(dict[str, object], response["ui"])
    assert set(ui) == {"sigmax_execution_bundle"}
    assert cast(list[object], ui["sigmax_execution_bundle"]) == [first.bundle_json]


@pytest.mark.parametrize("case", CASES)
def test_output_accepts_exact_host_float32_and_keeps_canonical_float64_artifact(
    case: tuple[object, ...],
) -> None:
    (
        steps,
        width,
        height,
        strict_official,
        _effective_width,
        _effective_height,
        _image_seq_len,
        _mu,
        _recipe_id,
        _evidence,
        numerical_fingerprint,
    ) = case
    schedule = build_krea2_sigma_schedule(
        variant="RAW",
        steps=steps,
        width=width,
        height=height,
        strict_official=strict_official,
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

    result = build_raw_workflow_output(
        sigmas=host_values,
        schedule_info=host_info,
        schedule_report=host_report,
    )
    bundle = deserialize_portable_execution_bundle(result.bundle_json)

    assert host_values != schedule.sigmas
    assert bundle.artifact.numerical_fingerprint == numerical_fingerprint


@pytest.mark.parametrize(
    "mutation",
    (
        "sigma_value",
        "schedule_info",
        "schedule_report",
        "turbo",
        "partial_slice",
        "wrong_mu",
        "swapped_effective_dimensions",
    ),
)
def test_output_rejects_noncanonical_or_mismatched_upstream_evidence(mutation: str) -> None:
    sigmas, schedule_info, schedule_report = _upstream(
        width=1353,
        height=761,
    )
    changed_sigmas = sigmas
    changed_info = schedule_info
    changed_report = schedule_report

    if mutation == "sigma_value":
        changed_sigmas = (sigmas[0] - 0.01, *sigmas[1:])
    elif mutation in {
        "schedule_info",
        "wrong_mu",
        "swapped_effective_dimensions",
    }:
        projection = cast(dict[str, Any], json.loads(schedule_info))
        if mutation == "schedule_info":
            projection["strict_official"] = False
        elif mutation == "wrong_mu":
            projection["shift"]["mu"] = cast(float, projection["shift"]["mu"]) + 0.01
        else:
            effective = projection["dimensions"]["effective"]
            effective["width"], effective["height"] = effective["height"], effective["width"]
        changed_info = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    elif mutation == "schedule_report":
        projection = cast(dict[str, Any], json.loads(schedule_report))
        projection["fingerprints"]["verified"] = False
        changed_report = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    else:
        alternate = build_krea2_sigma_schedule(
            variant="Turbo" if mutation == "turbo" else "RAW",
            steps=8 if mutation == "turbo" else 52,
            width=1353,
            height=761,
            strict_official=True,
            start_step=0 if mutation == "turbo" else 2,
            end_step=-1 if mutation == "turbo" else 20,
        )
        changed_sigmas = alternate.sigmas
        changed_info = alternate.schedule_info_json
        changed_report = build_schedule_inspection(
            sigmas=changed_sigmas,
            schedule_info=changed_info,
        ).report_json

    with pytest.raises(ScheduleContractError):
        build_raw_workflow_output(
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
        RawWorkflowOutput().publish(**arguments)


def test_output_result_is_immutable() -> None:
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.bundle_json = "{}"  # type: ignore[misc]
