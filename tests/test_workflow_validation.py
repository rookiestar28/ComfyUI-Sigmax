"""Deterministic static/live validation for canonical Sigmax workflow fixtures."""

from __future__ import annotations

import copy
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import comfyui_sigmax.workflows.validation as workflow_validation
import pytest
from comfyui_sigmax.core import ScheduleContractError, extract_workflow_metadata
from comfyui_sigmax.workflows import (
    CANONICAL_HOST_REVISION,
    CANONICAL_HOST_VERSION,
    WorkflowFixture,
    WorkflowIssue,
    WorkflowIssueKind,
    WorkflowIssueSeverity,
    WorkflowLiveLoadError,
    WorkflowLiveLoadReason,
    WorkflowNodeContract,
    WorkflowScanMode,
    WorkflowValidationLane,
    WorkflowWidgetSlot,
    deserialize_workflow_validation_report,
    fetch_live_object_info,
    load_canonical_workflow_fixtures,
    load_pinned_host_baseline,
    serialize_workflow_validation_report,
    validate_live_workflow_fixtures,
    validate_pinned_workflow_fixtures,
    validate_workflow_fixtures,
)


def _fixture(identifier: str = "krea2-turbo-1024") -> WorkflowFixture:
    return next(
        item for item in load_canonical_workflow_fixtures() if item.identifier == identifier
    )


def _workflow_copy(fixture: WorkflowFixture) -> dict[str, object]:
    return cast(dict[str, object], copy.deepcopy(fixture.workflow))


def _object_info_copy() -> dict[str, object]:
    return cast(dict[str, object], copy.deepcopy(load_pinned_host_baseline().object_info))


def _issue_kinds(report: object) -> tuple[WorkflowIssueKind, ...]:
    return tuple(item.kind for item in cast(Any, report).issues)


def _node(workflow: dict[str, object], node_id: int) -> dict[str, object]:
    nodes = cast(list[dict[str, object]], workflow["nodes"])
    return next(item for item in nodes if item["id"] == node_id)


def _host_node(object_info: dict[str, object], node_id: str) -> dict[str, object]:
    return cast(dict[str, object], object_info[node_id])


def _required_inputs(host_node: dict[str, object]) -> dict[str, object]:
    inputs = cast(dict[str, object], host_node["input"])
    return cast(dict[str, object], inputs["required"])


def test_packaged_canonical_workflows_are_complete_and_portable() -> None:
    fixtures = load_canonical_workflow_fixtures()

    assert tuple(item.identifier for item in fixtures) == (
        "anima-aesthetic-v1-framework-50",
        "anima-base-v1-framework-50",
        "anima-turbo-v1-framework-8",
        "auraflow-v0-2-official-50",
        "flux1-schnell-official-4",
        "hunyuan-image21-base-official-50",
        "hunyuan-image21-distilled-official-8",
        "krea2-raw-diffusers-portrait-761x1353",
        "krea2-raw-official-landscape-1353x761",
        "krea2-raw-official-square-1024",
        "krea2-turbo-1024",
        "ltx2-19b-dev-40",
        "ltx2-19b-distilled-stage1-8",
        "ltx2-3-22b-dev-30",
        "ltx2-3-22b-distilled-stage2-3",
        "ltxv-0-9-8-dev-20",
        "lumina2-v2-official-50",
        "qwen-image-comfy-fixed-official-50",
        "sd3-comfy-diffusers-fixed-framework-28",
        "sd3-publisher-reference-official-50",
        "wan-animate2-base-14b-official-40",
        "wan-animate2-distilled-14b-official-10",
        "wan21-flf2v-720p-official-50",
        "wan21-i2v-480p-official-40",
        "wan21-t2v-official-50",
        "wan21-vace-1-3b-official-50",
        "wan21-vace-14b-official-50",
        "wan22-animate-14b-official-20",
        "wan22-s2v-14b-official-40",
        "wan22-t2v-a14b-native-40",
        "wan22-ti2v-5b-native-50",
        "z-image-base-official-50",
        "z-image-turbo-official-8",
    )
    assert tuple(item.variant for item in fixtures) == (
        "Anima Aesthetic v1.x",
        "Anima Base v1.0",
        "Anima Turbo v1.0",
        "AuraFlow v0.2",
        "FLUX.1-schnell",
        "HunyuanImage 2.1 Base",
        "HunyuanImage 2.1 Distilled",
        "RAW",
        "RAW",
        "RAW",
        "Turbo",
        "LTX-2 19B Dev",
        "LTX-2 19B Distilled Stage 1",
        "LTX-2.3 22B Dev",
        "LTX-2.3 22B Distilled Stage 2",
        "LTXV 0.9.8 Dev",
        "Lumina-Image 2.0",
        "Qwen Image",
        "SD3",
        "SD3",
        "Wan Animate 2 Base 14B",
        "Wan Animate 2 Distilled 14B",
        "Wan 2.1 FLF2V 14B 720P",
        "Wan 2.1 I2V 480P",
        "Wan 2.1 T2V",
        "Wan 2.1 VACE 1.3B",
        "Wan 2.1 VACE 14B",
        "Wan 2.2 Animate 14B",
        "Wan 2.2 S2V 14B",
        "Wan 2.2 T2V A14B",
        "Wan 2.2 TI2V 5B",
        "Z-Image Base",
        "Z-Image Turbo",
    )
    assert fixtures[0].workflow is not fixtures[1].workflow

    expected_scheduler_widgets = {
        "anima-aesthetic-v1-framework-50": ["Aesthetic (3.0)", 50, True, 0, -1, False],
        "anima-base-v1-framework-50": ["Base (3.0)", 50, True, 0, -1, False],
        "anima-turbo-v1-framework-8": ["Turbo (3.0)", 8, True, 0, -1, False],
        "flux1-schnell-official-4": [4, True, 0, -1],
        "hunyuan-image21-base-official-50": ["Base (5.0)", 50, True, 0, -1, False],
        "hunyuan-image21-distilled-official-8": ["Distilled (4.0)", 8, True, 0, -1, False],
        "krea2-raw-diffusers-portrait-761x1353": [
            "RAW",
            28,
            761,
            1353,
            False,
            0,
            -1,
        ],
        "krea2-raw-official-landscape-1353x761": [
            "RAW",
            52,
            1353,
            761,
            True,
            0,
            -1,
        ],
        "krea2-raw-official-square-1024": [
            "RAW",
            52,
            1024,
            1024,
            True,
            0,
            -1,
        ],
        "krea2-turbo-1024": ["Turbo", 8, 1024, 1024, True, 0, -1],
        "ltx2-19b-dev-40": ["LTX-2 19B", "Dev", 40, 4096, True, 0.1, True, 0, -1],
        "ltx2-19b-distilled-stage1-8": [
            "LTX-2 19B",
            "Distilled Stage 1",
            8,
            4096,
            True,
            0.1,
            True,
            0,
            -1,
        ],
        "ltx2-3-22b-dev-30": ["LTX-2.3 22B", "Dev", 30, 4096, True, 0.1, True, 0, -1],
        "ltx2-3-22b-distilled-stage2-3": [
            "LTX-2.3 22B",
            "Distilled Stage 2",
            3,
            4096,
            True,
            0.1,
            True,
            0,
            -1,
        ],
        "ltxv-0-9-8-dev-20": ["LTXV 0.9.8", "Dev", 20, 4096, True, 0.1, True, 0, -1],
        "lumina2-v2-official-50": ["Official Fixed (6.0)", 50, True, 0, -1, False],
        "qwen-image-comfy-fixed-official-50": ["Comfy Fixed", 50, 0, True, 0, -1],
        "sd3-comfy-diffusers-fixed-framework-28": [
            "Comfy/Diffusers Fixed (3.0)",
            28,
            True,
            0,
            -1,
            False,
        ],
        "sd3-publisher-reference-official-50": [
            "Publisher Reference (1.0)",
            50,
            True,
            0,
            -1,
            False,
        ],
        "auraflow-v0-2-official-50": [
            "Official Fixed (1.73)",
            50,
            True,
            0,
            -1,
            False,
        ],
        "z-image-base-official-50": ["Base", 50, True, 0, -1],
        "z-image-turbo-official-8": ["Turbo", 8, True, 0, -1],
        "wan-animate2-base-14b-official-40": [
            "Wan Animate 2",
            "Animate Base",
            "Official native",
            "None",
            40,
            True,
            0,
            -1,
            False,
        ],
        "wan-animate2-distilled-14b-official-10": [
            "Wan Animate 2",
            "Animate Distilled",
            "Official native",
            "None",
            10,
            True,
            0,
            -1,
            False,
        ],
        "wan21-flf2v-720p-official-50": [
            "Wan 2.1",
            "FLF2V",
            "Official native",
            "720P",
            50,
            True,
            0,
            -1,
            False,
        ],
        "wan21-i2v-480p-official-40": [
            "Wan 2.1",
            "I2V",
            "Official native",
            "480P",
            40,
            True,
            0,
            -1,
            False,
        ],
        "wan21-vace-1-3b-official-50": [
            "Wan 2.1",
            "VACE 1.3B",
            "Official native",
            "None",
            50,
            True,
            0,
            -1,
            False,
        ],
        "wan21-vace-14b-official-50": [
            "Wan 2.1",
            "VACE 14B",
            "Official native",
            "None",
            50,
            True,
            0,
            -1,
            False,
        ],
        "wan22-animate-14b-official-20": [
            "Wan 2.2",
            "Animate",
            "Official native",
            "None",
            20,
            True,
            0,
            -1,
            False,
        ],
        "wan22-s2v-14b-official-40": [
            "Wan 2.2",
            "S2V",
            "Official native",
            "None",
            40,
            True,
            0,
            -1,
            False,
        ],
        "wan21-t2v-official-50": [
            "Wan 2.1",
            "T2V",
            "Official native",
            "None",
            50,
            True,
            0,
            -1,
            False,
        ],
        "wan22-t2v-a14b-native-40": [
            "Wan 2.2",
            "T2V A14B",
            "Official native",
            "None",
            40,
            True,
            0,
            -1,
            False,
        ],
        "wan22-ti2v-5b-native-50": [
            "Wan 2.2",
            "TI2V",
            "ComfyUI native",
            "None",
            50,
            True,
            0,
            -1,
            False,
        ],
    }
    for fixture in fixtures:
        workflow = fixture.workflow
        assert workflow["version"] == 0.4
        is_sigma_only = fixture.variant.startswith("Z-Image") or fixture.variant in {
            "Anima Aesthetic v1.x",
            "Anima Base v1.0",
            "Anima Turbo v1.0",
            "FLUX.1-schnell",
            "Qwen Image",
            "SD3",
            "AuraFlow v0.2",
            "Lumina-Image 2.0",
            "HunyuanImage 2.1 Base",
            "HunyuanImage 2.1 Distilled",
            "Wan Animate 2 Base 14B",
            "Wan Animate 2 Distilled 14B",
            "Wan 2.1 FLF2V 14B 720P",
            "Wan 2.1 I2V 480P",
            "Wan 2.1 T2V",
            "Wan 2.1 VACE 1.3B",
            "Wan 2.1 VACE 14B",
            "Wan 2.2 Animate 14B",
            "Wan 2.2 S2V 14B",
            "Wan 2.2 T2V A14B",
            "Wan 2.2 TI2V 5B",
            "LTXV 0.9.8 Dev",
            "LTX-2 19B Dev",
            "LTX-2 19B Distilled Stage 1",
            "LTX-2.3 22B Distilled Stage 2",
            "LTX-2.3 22B Dev",
        }
        assert workflow["last_node_id"] == (1 if is_sigma_only else 3)
        assert workflow["last_link_id"] == (0 if is_sigma_only else 5)
        assert len(cast(list[object], workflow["nodes"])) == (1 if is_sigma_only else 3)
        assert len(cast(list[object], workflow["links"])) == (0 if is_sigma_only else 5)
        metadata = extract_workflow_metadata(workflow)
        assert metadata is not None
        assert metadata.package.identifier == "comfyui-sigmax"
        assert metadata.package.version == "1.1.0"
        assert metadata.host.version == CANONICAL_HOST_VERSION
        expected_nodes = (
            (
                ["Sigmax.AnimaSigmaScheduler"]
                if fixture.variant.startswith("Anima")
                else (
                    ["Sigmax.Flux1SchnellSigmaScheduler"]
                    if fixture.variant == "FLUX.1-schnell"
                    else (
                        ["Sigmax.QwenImageSigmaScheduler"]
                        if fixture.variant == "Qwen Image"
                        else (
                            ["Sigmax.SD3SigmaScheduler"]
                            if fixture.variant == "SD3"
                            else (
                                ["Sigmax.AuraFlowSigmaScheduler"]
                                if fixture.variant == "AuraFlow v0.2"
                                else (
                                    ["Sigmax.Lumina2SigmaScheduler"]
                                    if fixture.variant == "Lumina-Image 2.0"
                                    else (
                                        ["Sigmax.LTXSigmaScheduler"]
                                        if fixture.variant.startswith("LTX")
                                        else (
                                            ["Sigmax.HunyuanImage21SigmaScheduler"]
                                            if fixture.variant.startswith("HunyuanImage 2.1")
                                            else (
                                                ["Sigmax.WanSigmaScheduler"]
                                                if fixture.variant.startswith("Wan")
                                                else ["Sigmax.ZImageSigmaScheduler"]
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
            if is_sigma_only
            else [
                "Sigmax.Krea2SigmaScheduler",
                "Sigmax.ScheduleInspector",
                (
                    "Sigmax.RawWorkflowOutput"
                    if fixture.variant == "RAW"
                    else "Sigmax.TurboWorkflowOutput"
                ),
            ]
        )
        assert tuple(item.identifier for item in metadata.nodes) == tuple(sorted(expected_nodes))
        expected_profile = {
            "Anima Aesthetic v1.x": "anima.aesthetic.framework-reference",
            "Anima Base v1.0": "anima.base.framework-reference",
            "Anima Turbo v1.0": "anima.turbo.framework-reference",
            "FLUX.1-schnell": "flux1.schnell.official",
            "RAW": "krea2.raw.official",
            "Turbo": "krea2.turbo.official",
            "Qwen Image": "qwen_image.comfy-fixed.official",
            "SD3": (
                "sd3.comfy-diffusers-fixed.framework-reference"
                if fixture.identifier.startswith("sd3-comfy")
                else "sd3.publisher-reference.official"
            ),
            "AuraFlow v0.2": "auraflow.v0-2.official",
            "Lumina-Image 2.0": "lumina2.v2.official",
            "HunyuanImage 2.1 Base": "hunyuan-image-2-1.base.official",
            "HunyuanImage 2.1 Distilled": "hunyuan-image-2-1.distilled.official",
            "Wan 2.1 I2V 480P": "wan2.1.i2v.480p.official-native",
            "Wan 2.1 T2V": "wan2.1.t2v.official-native",
            "Wan 2.1 FLF2V 14B 720P": "wan2.1.flf2v.14b.720p.official-native",
            "Wan 2.1 VACE 1.3B": "wan2.1.vace.1.3b.official-native",
            "Wan 2.1 VACE 14B": "wan2.1.vace.14b.official-native",
            "Wan 2.2 S2V 14B": "wan2.2.s2v.14b.official-native",
            "Wan 2.2 Animate 14B": "wan2.2.animate.14b.official-native",
            "Wan 2.2 T2V A14B": "wan2.2.t2v-a14b.official-native",
            "Wan 2.2 TI2V 5B": "wan2.2.ti2v.5b.comfy-native",
            "Wan Animate 2 Base 14B": "wan-animate2.14b.base.official-native",
            "Wan Animate 2 Distilled 14B": "wan-animate2.14b.distilled.official-native",
            "LTXV 0.9.8 Dev": "ltxv.0.9.8.dev",
            "LTX-2 19B Dev": "ltx2.19b.dev",
            "LTX-2 19B Distilled Stage 1": "ltx2.19b.distilled.stage1",
            "LTX-2.3 22B Dev": "ltx2.3.22b.dev",
            "LTX-2.3 22B Distilled Stage 2": "ltx2.3.22b.distilled.stage2",
            "Z-Image Base": "z_image.base.official",
            "Z-Image Turbo": "z_image.turbo.official",
        }
        assert metadata.profile.identifier == expected_profile[fixture.variant]
        scheduler = _node(cast(dict[str, object], workflow), 1)
        assert scheduler["widgets_values"] == expected_scheduler_widgets[fixture.identifier]
        assert cast(dict[str, object], scheduler["properties"])["cnr_id"] == "comfyui-sigmax"

    first = _workflow_copy(fixtures[0])
    cast(list[object], first["nodes"]).clear()
    assert len(cast(list[object], load_canonical_workflow_fixtures()[0].workflow["nodes"])) == 1


def test_pinned_static_baseline_is_explicit_and_known_good() -> None:
    baseline = load_pinned_host_baseline()
    report = validate_pinned_workflow_fixtures()

    assert baseline.host_version == CANONICAL_HOST_VERSION == "0.29.0"
    assert baseline.host_revision == CANONICAL_HOST_REVISION
    assert tuple(baseline.object_info) == (
        "Sigmax.AnimaSigmaScheduler",
        "Sigmax.AuraFlowSigmaScheduler",
        "Sigmax.Flux1SchnellSigmaScheduler",
        "Sigmax.HunyuanImage21SigmaScheduler",
        "Sigmax.Krea2ConditioningRebalance",
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.LTXSigmaScheduler",
        "Sigmax.Lumina2SigmaScheduler",
        "Sigmax.QwenImageSigmaScheduler",
        "Sigmax.RawWorkflowOutput",
        "Sigmax.SD3SigmaScheduler",
        "Sigmax.ScheduleInspector",
        "Sigmax.TurboWorkflowOutput",
        "Sigmax.WanSigmaScheduler",
        "Sigmax.ZImageSigmaScheduler",
    )
    assert report.scan_mode is WorkflowScanMode.PINNED_STATIC
    assert report.lane is WorkflowValidationLane.KNOWN_GOOD
    assert report.host_version == CANONICAL_HOST_VERSION
    assert report.host_revision == CANONICAL_HOST_REVISION
    assert report.workflow_count == 33
    assert report.compatible is True
    assert report.gate_passed is True
    assert report.observational is False
    assert report.issues == ()
    projection = report.projection()
    assert projection["package"] == {"id": "comfyui-sigmax", "version": "1.1.0"}
    assert projection["nodes"] == [
        {"id": "Sigmax.AnimaSigmaScheduler", "version": "1"},
        {"id": "Sigmax.AuraFlowSigmaScheduler", "version": "1"},
        {"id": "Sigmax.Flux1SchnellSigmaScheduler", "version": "1"},
        {"id": "Sigmax.HunyuanImage21SigmaScheduler", "version": "1"},
        {"id": "Sigmax.Krea2SigmaScheduler", "version": "1"},
        {"id": "Sigmax.LTXSigmaScheduler", "version": "1"},
        {"id": "Sigmax.Lumina2SigmaScheduler", "version": "1"},
        {"id": "Sigmax.QwenImageSigmaScheduler", "version": "1"},
        {"id": "Sigmax.RawWorkflowOutput", "version": "1"},
        {"id": "Sigmax.SD3SigmaScheduler", "version": "1"},
        {"id": "Sigmax.ScheduleInspector", "version": "1"},
        {"id": "Sigmax.TurboWorkflowOutput", "version": "1"},
        {"id": "Sigmax.WanSigmaScheduler", "version": "1"},
        {"id": "Sigmax.ZImageSigmaScheduler", "version": "1"},
    ]


def test_pinned_baseline_exposes_the_experimental_conditioning_schema() -> None:
    baseline = load_pinned_host_baseline()
    legacy = cast(dict[str, object], baseline.object_info["Sigmax.Krea2ConditioningRebalance"])
    legacy_input = cast(dict[str, object], legacy["input"])
    legacy_inputs = cast(dict[str, object], legacy_input["required"])
    assert tuple(legacy_inputs) == (
        "conditioning",
        "variant",
        "profile",
        "strength",
    )
    assert legacy_inputs["conditioning"] == ["CONDITIONING"]
    assert legacy_inputs["variant"] == [["RAW", "Turbo"]]
    assert legacy_inputs["profile"] == [["Disabled", "Subtle Experimental", "Classic Experimental"]]
    assert legacy["experimental"] is True

    v2 = baseline.node_definition_v2["Sigmax.Krea2ConditioningRebalance"]
    assert cast(dict[str, object], v2)["name"] == "Sigmax.Krea2ConditioningRebalance"
    assert cast(dict[str, object], v2)["outputs"] == [
        {"index": 0, "name": "conditioning", "type": "CONDITIONING", "is_list": False},
        {"index": 1, "name": "modifier_info", "type": "STRING", "is_list": False},
    ]


def test_report_serialization_is_canonical_deterministic_and_immutable() -> None:
    report = validate_pinned_workflow_fixtures()
    payload = serialize_workflow_validation_report(report)

    assert payload == serialize_workflow_validation_report(report)
    assert deserialize_workflow_validation_report(payload) == report
    assert (
        serialize_workflow_validation_report(deserialize_workflow_validation_report(payload))
        == payload
    )
    envelope = json.loads(payload)
    assert envelope["schema"] == "sigmax.workflow-validation-report-envelope/1"
    assert envelope["report"]["schema"] == "sigmax.workflow-validation-report/1"
    assert envelope["report_fingerprint"] == report.report_fingerprint
    with pytest.raises(FrozenInstanceError):
        report.workflow_count = 3  # type: ignore[misc]


def test_live_v2_schema_normalizes_to_the_same_known_good_result() -> None:
    baseline = load_pinned_host_baseline()
    v2 = baseline.node_definition_v2

    report = validate_live_workflow_fixtures(
        object_info=v2,
        host_version=baseline.host_version,
        host_revision=baseline.host_revision,
        lane=WorkflowValidationLane.KNOWN_GOOD,
    )

    assert report.scan_mode is WorkflowScanMode.LIVE_OBJECT_INFO
    assert report.lane is WorkflowValidationLane.KNOWN_GOOD
    assert report.compatible is True
    assert report.gate_passed is True
    assert report.issues == ()


@pytest.mark.parametrize(
    ("mutation", "expected_kind"),
    [
        ("missing_node", WorkflowIssueKind.MISSING_NODE),
        ("missing_input", WorkflowIssueKind.MISSING_INPUT),
        ("widget_slot", WorkflowIssueKind.WIDGET_SLOT_DRIFT),
        ("input_type", WorkflowIssueKind.INPUT_TYPE_DRIFT),
        ("fixed_combo", WorkflowIssueKind.INVALID_FIXED_COMBO_VALUE),
        ("deprecated", WorkflowIssueKind.DEPRECATED_NODE),
        ("experimental", WorkflowIssueKind.EXPERIMENTAL_NODE),
        ("directory", WorkflowIssueKind.NORMALIZED_DIRECTORY_FAILURE),
        ("metadata", WorkflowIssueKind.MALFORMED_METADATA),
    ],
)
def test_known_good_lane_fails_each_roadmap_issue(
    mutation: str,
    expected_kind: WorkflowIssueKind,
) -> None:
    fixture = _fixture()
    workflow = _workflow_copy(fixture)
    object_info = _object_info_copy()
    fixtures = (replace(fixture, workflow=workflow),)

    if mutation == "missing_node":
        object_info.pop("Sigmax.ScheduleInspector")
    elif mutation == "missing_input":
        _required_inputs(_host_node(object_info, "Sigmax.ScheduleInspector")).pop("schedule_info")
    elif mutation == "widget_slot":
        cast(list[object], _node(workflow, 1)["widgets_values"]).pop()
    elif mutation == "input_type":
        _required_inputs(_host_node(object_info, "Sigmax.ScheduleInspector"))["schedule_info"] = [
            "INT"
        ]
    elif mutation == "fixed_combo":
        cast(list[object], _node(workflow, 1)["widgets_values"])[0] = "RAW"
    elif mutation == "deprecated":
        _host_node(object_info, "Sigmax.Krea2SigmaScheduler")["deprecated"] = True
    elif mutation == "experimental":
        _host_node(object_info, "Sigmax.Krea2SigmaScheduler")["experimental"] = True
    elif mutation == "directory":
        _host_node(object_info, "Sigmax.Krea2SigmaScheduler")["python_module"] = (
            "custom_nodes.normalized_directory.nodes"
        )
    elif mutation == "metadata":
        extra = cast(dict[str, object], workflow["extra"])
        namespace = cast(dict[str, object], extra["comfyui_sigmax"])
        namespace["schema"] = "sigmax.workflow-metadata/999"

    report = validate_workflow_fixtures(
        fixtures=fixtures,
        object_info=object_info,
        scan_mode=WorkflowScanMode.LIVE_OBJECT_INFO,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )

    assert expected_kind in _issue_kinds(report)
    assert report.compatible is False
    assert report.gate_passed is False
    assert report.observational is False
    assert all(item.severity is WorkflowIssueSeverity.ERROR for item in report.issues)


def test_workflow_package_identity_detects_normalized_directory_drift() -> None:
    fixture = _fixture()
    workflow = _workflow_copy(fixture)
    cast(dict[str, object], _node(workflow, 1)["properties"])["cnr_id"] = "ComfyUI-Sigmax-0.1.0"

    report = validate_workflow_fixtures(
        fixtures=(replace(fixture, workflow=workflow),),
        object_info=_object_info_copy(),
        scan_mode=WorkflowScanMode.PINNED_STATIC,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )

    assert WorkflowIssueKind.NORMALIZED_DIRECTORY_FAILURE in _issue_kinds(report)
    assert report.gate_passed is False


def test_live_loader_accepts_only_the_canonical_comfyui_custom_node_module() -> None:
    object_info = _object_info_copy()
    for node in object_info.values():
        cast(dict[str, object], node)["python_module"] = "custom_nodes.ComfyUI-Sigmax"

    report = validate_live_workflow_fixtures(
        object_info=object_info,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
        lane=WorkflowValidationLane.KNOWN_GOOD,
    )

    assert report.gate_passed is True
    assert report.issues == ()


def test_latest_host_findings_remain_observational_and_separately_labeled() -> None:
    object_info = _object_info_copy()
    object_info.pop("Sigmax.ScheduleInspector")

    report = validate_live_workflow_fixtures(
        object_info=object_info,
        host_version="0.30.0",
        host_revision="latest-reviewed-revision",
        lane=WorkflowValidationLane.LATEST_HOST,
    )

    assert report.lane is WorkflowValidationLane.LATEST_HOST
    assert report.compatible is False
    assert report.gate_passed is True
    assert report.observational is True
    assert WorkflowIssueKind.MISSING_NODE in _issue_kinds(report)
    assert report.projection()["result"] == {
        "compatible": False,
        "gate_passed": True,
        "observational": True,
    }


def test_issue_order_is_stable_across_fixture_and_host_mapping_order() -> None:
    fixtures = tuple(reversed(load_canonical_workflow_fixtures()))
    object_info = _object_info_copy()
    scheduler = object_info.pop("Sigmax.Krea2SigmaScheduler")
    object_info = {"Sigmax.Krea2SigmaScheduler": scheduler, **object_info}
    _host_node(object_info, "Sigmax.Krea2SigmaScheduler")["deprecated"] = True
    _host_node(object_info, "Sigmax.ScheduleInspector")["experimental"] = True

    first = validate_workflow_fixtures(
        fixtures=fixtures,
        object_info=object_info,
        scan_mode=WorkflowScanMode.LIVE_OBJECT_INFO,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )
    second = validate_workflow_fixtures(
        fixtures=tuple(reversed(fixtures)),
        object_info=dict(reversed(tuple(object_info.items()))),
        scan_mode=WorkflowScanMode.LIVE_OBJECT_INFO,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )

    assert first.issues == second.issues
    assert serialize_workflow_validation_report(first) == serialize_workflow_validation_report(
        second
    )


def test_malformed_workflow_and_host_schema_become_stable_issues() -> None:
    fixture = _fixture()
    workflow = _workflow_copy(fixture)
    workflow["nodes"] = "not-an-array"
    malformed_workflow = validate_workflow_fixtures(
        fixtures=(replace(fixture, workflow=workflow),),
        object_info=_object_info_copy(),
        scan_mode=WorkflowScanMode.PINNED_STATIC,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )
    malformed_host = validate_workflow_fixtures(
        fixtures=(fixture,),
        object_info={"Sigmax.Krea2SigmaScheduler": {"input": {}, "inputs": {}}},
        scan_mode=WorkflowScanMode.LIVE_OBJECT_INFO,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )

    assert _issue_kinds(malformed_workflow) == (WorkflowIssueKind.WORKFLOW_SCHEMA_MALFORMED,)
    assert _issue_kinds(malformed_host) == (WorkflowIssueKind.HOST_SCHEMA_MALFORMED,)


def test_metadata_requirement_drift_is_malformed_metadata() -> None:
    fixture = _fixture()
    workflow = _workflow_copy(fixture)
    extra = cast(dict[str, object], workflow["extra"])
    namespace = cast(dict[str, object], extra["comfyui_sigmax"])
    metadata = cast(dict[str, object], namespace["metadata"])
    requirements = cast(dict[str, object], metadata["requirements"])
    package = cast(dict[str, object], requirements["package"])
    package["version"] = "9.9.9"

    report = validate_workflow_fixtures(
        fixtures=(replace(fixture, workflow=workflow),),
        object_info=_object_info_copy(),
        scan_mode=WorkflowScanMode.PINNED_STATIC,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )

    assert WorkflowIssueKind.MALFORMED_METADATA in _issue_kinds(report)


def test_primitive_widget_types_reject_bool_as_integer() -> None:
    fixture = _fixture()
    workflow = _workflow_copy(fixture)
    cast(list[object], _node(workflow, 1)["widgets_values"])[1] = True

    report = validate_workflow_fixtures(
        fixtures=(replace(fixture, workflow=workflow),),
        object_info=_object_info_copy(),
        scan_mode=WorkflowScanMode.PINNED_STATIC,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )

    assert WorkflowIssueKind.WIDGET_SLOT_DRIFT in _issue_kinds(report)


class _ObjectInfoHandler(BaseHTTPRequestHandler):
    body = b"{}"
    status = 200
    content_type = "application/json"
    location: str | None = None
    declared_length: int | None = None
    observed_path = ""

    def do_GET(self) -> None:
        type(self).observed_path = self.path
        self.send_response(type(self).status)
        self.send_header("Content-Type", type(self).content_type)
        location = type(self).location
        if location is not None:
            self.send_header("Location", location)
        length = (
            len(type(self).body)
            if type(self).declared_length is None
            else type(self).declared_length
        )
        self.send_header("Content-Length", str(length))
        self.end_headers()
        self.wfile.write(type(self).body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


@contextmanager
def _object_info_server(
    *,
    body: bytes,
    status: int = 200,
    content_type: str = "application/json",
    location: str | None = None,
    declared_length: int | None = None,
) -> Iterator[str]:
    handler = type(
        "ConfiguredObjectInfoHandler",
        (_ObjectInfoHandler,),
        {
            "body": body,
            "status": status,
            "content_type": content_type,
            "location": location,
            "declared_length": declared_length,
            "observed_path": "",
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/object_info"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


def test_live_loader_reads_bounded_loopback_object_info() -> None:
    payload = json.dumps(load_pinned_host_baseline().object_info).encode()
    with _object_info_server(body=payload) as url:
        loaded = fetch_live_object_info(url=url, timeout_seconds=2.0)

    assert loaded == load_pinned_host_baseline().object_info


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8188/object_info",
        "http://localhost:8188/object_info",
        "http://127.0.0.2:8188/object_info",
        "http://user@127.0.0.1:8188/object_info",
        "http://127.0.0.1:8188/system_stats",
        "http://127.0.0.1:8188/object_info?token=secret",
    ],
)
def test_live_loader_rejects_noncanonical_or_nonliteral_loopback_urls(url: str) -> None:
    with pytest.raises(WorkflowLiveLoadError) as captured:
        fetch_live_object_info(url=url)

    assert captured.value.reason is WorkflowLiveLoadReason.URL_NOT_LOOPBACK


@pytest.mark.parametrize(
    ("body", "status", "content_type", "location", "declared_length", "expected_reason"),
    [
        (
            b"{}",
            302,
            "application/json",
            "http://127.0.0.1:9/object_info",
            None,
            WorkflowLiveLoadReason.HTTP_ERROR,
        ),
        (
            b"not-json",
            200,
            "application/json",
            None,
            None,
            WorkflowLiveLoadReason.RESPONSE_NOT_JSON,
        ),
        (
            b"[]",
            200,
            "application/json",
            None,
            None,
            WorkflowLiveLoadReason.PAYLOAD_NOT_OBJECT,
        ),
        (
            b"{}",
            200,
            "application/json",
            None,
            2_000_001,
            WorkflowLiveLoadReason.RESPONSE_TOO_LARGE,
        ),
        (
            b"{}",
            200,
            "text/plain",
            None,
            None,
            WorkflowLiveLoadReason.RESPONSE_NOT_JSON,
        ),
    ],
)
def test_live_loader_fails_closed_with_stable_reasons(
    body: bytes,
    status: int,
    content_type: str,
    location: str | None,
    declared_length: int | None,
    expected_reason: WorkflowLiveLoadReason,
) -> None:
    with (
        _object_info_server(
            body=body,
            status=status,
            content_type=content_type,
            location=location,
            declared_length=declared_length,
        ) as url,
        pytest.raises(WorkflowLiveLoadError) as captured,
    ):
        fetch_live_object_info(url=url, timeout_seconds=2.0)

    assert captured.value.reason is expected_reason


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b'{"schema":"sigmax.workflow-validation-report-envelope/999"}',
        b"[]",
        "not-json",
    ],
)
def test_report_deserializer_rejects_malformed_payloads(payload: bytes | str) -> None:
    with pytest.raises(ScheduleContractError):
        deserialize_workflow_validation_report(payload)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "x" * 257,
        "line\nbreak",
    ],
)
def test_private_public_text_contract_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ScheduleContractError):
        workflow_validation._text(value, label="fixture")


def test_private_json_helpers_fail_closed() -> None:
    with pytest.raises(ScheduleContractError):
        workflow_validation._sequence({}, label="fixture")
    with pytest.raises(ScheduleContractError):
        workflow_validation._canonical_bytes({"invalid": object()})
    assert workflow_validation._display(object()) == "<unsupported>"
    assert len(workflow_validation._display("x" * 300)) == 256


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WorkflowWidgetSlot(name="value", type_name="MODEL"),
        lambda: WorkflowWidgetSlot(name="value", type_name="INT", fixed_value=1),
        lambda: WorkflowNodeContract(node_id=-1, node_type="Sigmax.Node", widget_slots=()),
        lambda: WorkflowNodeContract(
            node_id=1,
            node_type="Sigmax.Node",
            widget_slots=cast(Any, []),
        ),
        lambda: WorkflowNodeContract(
            node_id=1,
            node_type="Sigmax.Node",
            widget_slots=(
                WorkflowWidgetSlot(name="value", type_name="INT"),
                WorkflowWidgetSlot(name="value", type_name="INT"),
            ),
        ),
    ],
)
def test_widget_and_node_contracts_reject_invalid_shapes(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    "mutation",
    [
        {"variant": "Unknown"},
        {"package": "invalid"},
        {"nodes": []},
        {"host": "invalid"},
        {"profile": "invalid"},
        {"node_contracts": []},
        {"workflow": []},
    ],
)
def test_workflow_fixture_rejects_invalid_contract_members(
    mutation: dict[str, object],
) -> None:
    with pytest.raises(ScheduleContractError):
        replace(_fixture(), **cast(Any, mutation))


def test_host_baseline_and_issue_reject_invalid_contract_members() -> None:
    baseline = load_pinned_host_baseline()
    with pytest.raises(ScheduleContractError):
        replace(baseline, object_info=cast(Any, []))
    with pytest.raises(ScheduleContractError):
        replace(baseline, node_definition_v2=cast(Any, []))

    issue = WorkflowIssue(
        severity=WorkflowIssueSeverity.ERROR,
        kind=WorkflowIssueKind.MISSING_NODE,
        workflow_id="fixture",
    )
    for mutation in (
        {"severity": "error"},
        {"kind": "missing_node"},
        {"actual": cast(Any, object())},
        {"expected": "x" * 257},
    ):
        with pytest.raises(ScheduleContractError):
            replace(issue, **cast(Any, mutation))


def test_report_rejects_invalid_or_inconsistent_contract_members() -> None:
    report = validate_pinned_workflow_fixtures()
    first_issue = WorkflowIssue(
        severity=WorkflowIssueSeverity.ERROR,
        kind=WorkflowIssueKind.MISSING_NODE,
        workflow_id="a",
    )
    second_issue = WorkflowIssue(
        severity=WorkflowIssueSeverity.ERROR,
        kind=WorkflowIssueKind.MISSING_INPUT,
        workflow_id="b",
    )
    mutations: tuple[dict[str, object], ...] = (
        {"scan_mode": "pinned_static"},
        {"lane": "known_good"},
        {"package": "invalid"},
        {"nodes": []},
        {"workflow_count": 0},
        {"issues": []},
        {"issues": (second_issue, first_issue), "compatible": False, "gate_passed": False},
        {"compatible": cast(Any, 1)},
        {"compatible": False},
        {"observational": True},
        {"gate_passed": False},
    )
    for mutation in mutations:
        with pytest.raises(ScheduleContractError):
            replace(report, **cast(Any, mutation))


def test_packaged_resource_and_loader_contracts_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ScheduleContractError):
        workflow_validation._resource_json("missing.json")

    original = workflow_validation._resource_json
    bundle = original("fixtures.json")
    invalid_schema = copy.deepcopy(bundle)
    invalid_schema["schema"] = "unsupported"
    monkeypatch.setattr(workflow_validation, "_resource_json", lambda _name: invalid_schema)
    with pytest.raises(ScheduleContractError):
        load_canonical_workflow_fixtures()

    invalid_inventory = copy.deepcopy(bundle)
    cast(list[object], invalid_inventory["fixtures"]).pop()
    monkeypatch.setattr(workflow_validation, "_resource_json", lambda _name: invalid_inventory)
    with pytest.raises(ScheduleContractError):
        load_canonical_workflow_fixtures()

    invalid_node_id = copy.deepcopy(bundle)
    raw_fixture = cast(list[dict[str, object]], invalid_node_id["fixtures"])[0]
    node_contract = cast(list[dict[str, object]], raw_fixture["node_contracts"])[0]
    node_contract["node_id"] = True
    monkeypatch.setattr(workflow_validation, "_resource_json", lambda _name: invalid_node_id)
    with pytest.raises(ScheduleContractError):
        load_canonical_workflow_fixtures()

    baseline = original("host_baseline.json")
    invalid_baseline = copy.deepcopy(baseline)
    invalid_baseline["schema"] = "unsupported"
    monkeypatch.setattr(workflow_validation, "_resource_json", lambda _name: invalid_baseline)
    with pytest.raises(ScheduleContractError):
        load_pinned_host_baseline()


def test_private_host_and_value_helpers_cover_all_supported_paths() -> None:
    with pytest.raises(ScheduleContractError):
        workflow_validation._host_input_order({})
    with pytest.raises(ScheduleContractError):
        workflow_validation._host_input_order({"input": {}, "inputs": {}})
    assert workflow_validation._value_matches_type(1.25, "FLOAT")
    assert not workflow_validation._value_matches_type(float("nan"), "FLOAT")
    assert workflow_validation._value_matches_type(True, "BOOLEAN")
    assert not workflow_validation._value_matches_type("value", "MODEL")


def test_new_required_host_input_and_host_widget_order_drift_are_reported() -> None:
    object_info = _object_info_copy()
    required = _required_inputs(_host_node(object_info, "Sigmax.Krea2SigmaScheduler"))
    required["new_control"] = ["INT"]

    report = validate_live_workflow_fixtures(
        object_info=object_info,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
        lane=WorkflowValidationLane.KNOWN_GOOD,
    )

    assert WorkflowIssueKind.MISSING_INPUT in _issue_kinds(report)
    assert WorkflowIssueKind.WIDGET_SLOT_DRIFT in _issue_kinds(report)


def test_non_array_widgets_missing_saved_node_and_malformed_saved_node_are_reported() -> None:
    fixture = _fixture()
    non_array = _workflow_copy(fixture)
    _node(non_array, 1)["widgets_values"] = {}
    missing_node = _workflow_copy(fixture)
    cast(list[dict[str, object]], missing_node["nodes"]).pop()
    malformed = _workflow_copy(fixture)
    _node(malformed, 1)["properties"] = []

    reports = tuple(
        validate_workflow_fixtures(
            fixtures=(replace(fixture, workflow=workflow),),
            object_info=_object_info_copy(),
            scan_mode=WorkflowScanMode.PINNED_STATIC,
            lane=WorkflowValidationLane.KNOWN_GOOD,
            host_version=CANONICAL_HOST_VERSION,
            host_revision=CANONICAL_HOST_REVISION,
        )
        for workflow in (non_array, missing_node, malformed)
    )

    assert WorkflowIssueKind.WIDGET_SLOT_DRIFT in _issue_kinds(reports[0])
    assert WorkflowIssueKind.MISSING_NODE in _issue_kinds(reports[1])
    assert WorkflowIssueKind.WORKFLOW_SCHEMA_MALFORMED in _issue_kinds(reports[2])


def test_duplicate_saved_inputs_and_valid_metadata_requirement_drift_are_reported() -> None:
    fixture = _fixture()
    duplicate_inputs = _workflow_copy(fixture)
    inspector_inputs = cast(list[dict[str, object]], _node(duplicate_inputs, 2)["inputs"])
    inspector_inputs.append(copy.deepcopy(inspector_inputs[0]))
    duplicate_report = validate_workflow_fixtures(
        fixtures=(replace(fixture, workflow=duplicate_inputs),),
        object_info=_object_info_copy(),
        scan_mode=WorkflowScanMode.PINNED_STATIC,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )
    changed_profile = replace(fixture.profile, version="9")
    metadata_report = validate_workflow_fixtures(
        fixtures=(replace(fixture, profile=changed_profile),),
        object_info=_object_info_copy(),
        scan_mode=WorkflowScanMode.PINNED_STATIC,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
    )

    assert WorkflowIssueKind.WORKFLOW_SCHEMA_MALFORMED in _issue_kinds(duplicate_report)
    assert WorkflowIssueKind.MALFORMED_METADATA in _issue_kinds(metadata_report)


def test_missing_widget_schema_and_widget_type_drift_are_reported() -> None:
    without_variant = _object_info_copy()
    _required_inputs(_host_node(without_variant, "Sigmax.Krea2SigmaScheduler")).pop("variant")
    changed_type = _object_info_copy()
    _required_inputs(_host_node(changed_type, "Sigmax.Krea2SigmaScheduler"))["variant"] = ["STRING"]

    missing_report = validate_live_workflow_fixtures(
        object_info=without_variant,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
        lane=WorkflowValidationLane.KNOWN_GOOD,
    )
    drift_report = validate_live_workflow_fixtures(
        object_info=changed_type,
        host_version=CANONICAL_HOST_VERSION,
        host_revision=CANONICAL_HOST_REVISION,
        lane=WorkflowValidationLane.KNOWN_GOOD,
    )

    assert WorkflowIssueKind.WIDGET_SLOT_DRIFT in _issue_kinds(missing_report)
    assert WorkflowIssueKind.INPUT_TYPE_DRIFT in _issue_kinds(drift_report)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixtures": cast(Any, [])},
        {"fixtures": ()},
        {"fixtures": cast(Any, (object(),))},
        {"object_info": cast(Any, [])},
        {"scan_mode": cast(Any, "pinned_static")},
        {"lane": cast(Any, "known_good")},
    ],
)
def test_validator_rejects_invalid_entry_contracts(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "fixtures": load_canonical_workflow_fixtures(),
        "object_info": _object_info_copy(),
        "scan_mode": WorkflowScanMode.PINNED_STATIC,
        "lane": WorkflowValidationLane.KNOWN_GOOD,
        "host_version": CANONICAL_HOST_VERSION,
        "host_revision": CANONICAL_HOST_REVISION,
    }
    arguments.update(kwargs)
    with pytest.raises(ScheduleContractError):
        validate_workflow_fixtures(**cast(Any, arguments))


def test_validator_allows_per_fixture_nodes_but_rejects_requirement_disagreement() -> None:
    raw = _fixture("krea2-raw-official-square-1024")
    turbo = _fixture("krea2-turbo-1024")
    changed_package = replace(turbo.package, version="9.9.9")
    with pytest.raises(ScheduleContractError):
        validate_workflow_fixtures(
            fixtures=(raw, replace(turbo, package=changed_package)),
            object_info=_object_info_copy(),
            scan_mode=WorkflowScanMode.PINNED_STATIC,
            lane=WorkflowValidationLane.KNOWN_GOOD,
            host_version=CANONICAL_HOST_VERSION,
            host_revision=CANONICAL_HOST_REVISION,
        )

    conflicting_nodes = (
        replace(turbo.nodes[0], version="9"),
        *turbo.nodes[1:],
    )
    with pytest.raises(ScheduleContractError):
        validate_workflow_fixtures(
            fixtures=(raw, replace(turbo, nodes=conflicting_nodes)),
            object_info=_object_info_copy(),
            scan_mode=WorkflowScanMode.PINNED_STATIC,
            lane=WorkflowValidationLane.KNOWN_GOOD,
            host_version=CANONICAL_HOST_VERSION,
            host_revision=CANONICAL_HOST_REVISION,
        )


def test_report_serializer_and_verifier_reject_invalid_values() -> None:
    with pytest.raises(ScheduleContractError):
        serialize_workflow_validation_report(cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        deserialize_workflow_validation_report(cast(Any, object()))

    payload = json.loads(serialize_workflow_validation_report(validate_pinned_workflow_fixtures()))
    cast(dict[str, object], payload["report"])["schema"] = "unsupported"
    with pytest.raises(ScheduleContractError):
        deserialize_workflow_validation_report(json.dumps(payload))

    payload = json.loads(serialize_workflow_validation_report(validate_pinned_workflow_fixtures()))
    payload["report_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ScheduleContractError):
        deserialize_workflow_validation_report(json.dumps(payload))


def test_live_loader_rejects_invalid_url_type_and_timeout() -> None:
    with pytest.raises(WorkflowLiveLoadError) as invalid_url:
        fetch_live_object_info(url=cast(Any, object()))
    assert invalid_url.value.reason is WorkflowLiveLoadReason.URL_NOT_LOOPBACK

    for timeout in (True, 0, 31):
        with pytest.raises(WorkflowLiveLoadError) as invalid_timeout:
            fetch_live_object_info(timeout_seconds=cast(Any, timeout))
        assert invalid_timeout.value.reason is WorkflowLiveLoadReason.HTTP_ERROR


def test_live_loader_rejects_invalid_content_length_and_streamed_oversize() -> None:
    class InvalidLengthHandler(_ObjectInfoHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "invalid")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), InvalidLengthHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with pytest.raises(WorkflowLiveLoadError) as invalid_length:
            fetch_live_object_info(
                url=f"http://127.0.0.1:{server.server_port}/object_info",
                timeout_seconds=2,
            )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert invalid_length.value.reason is WorkflowLiveLoadReason.HTTP_ERROR

    class NoLengthOversizeHandler(_ObjectInfoHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b" " * 2_000_001)

    server = ThreadingHTTPServer(("127.0.0.1", 0), NoLengthOversizeHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with pytest.raises(WorkflowLiveLoadError) as oversized:
            fetch_live_object_info(
                url=f"http://127.0.0.1:{server.server_port}/object_info",
                timeout_seconds=2,
            )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert oversized.value.reason is WorkflowLiveLoadReason.RESPONSE_TOO_LARGE
