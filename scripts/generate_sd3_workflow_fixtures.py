"""Add deterministic original-SD3 scheduler workflows and refresh the host baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from comfyui_sigmax.adapters.registration import builtin_node_registry
from comfyui_sigmax.core import (
    CapabilityDimension,
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityReason,
    WorkflowArtifactReference,
    WorkflowHostRequirement,
    WorkflowMetadata,
    WorkflowRequirement,
    attach_workflow_metadata,
    construction_fingerprint,
    numerical_fingerprint,
)
from comfyui_sigmax.profiles.sd3 import SD3ShiftMode, build_sd3_schedule
from comfyui_sigmax.version import VERSION


def _json(value: object, *, indent: int | None = 2) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=indent) + "\n"


def _construction(result: Any, ratio: float) -> str:
    numerical = numerical_fingerprint(
        result.sigmas, domain=result.final_domain, precision="float64"
    )
    projection = {
        "base_grid": {"id": "flowmatch.reciprocal_step"},
        "effective": {"steps": result.effective_inputs.steps},
        "engine": {"version": result.request.provenance.engine_version},
        "evidence": {"level": result.request.provenance.evidence.value},
        "numerical_fingerprint": numerical,
        "overrides": [],
        "ownership": "external_sigmas",
        "requested": {"steps": result.request.requested_inputs.steps},
        "schema": "sigmax.schedule-artifact/1",
        "slicing": {"end_step": None, "start_step": 0},
        "source": {
            "profile_id": result.request.provenance.profile_id,
            "revision": result.request.provenance.source_revision,
        },
        "terminal": {"policy": "append_zero"},
        "transforms": [{"id": "direct_ratio.shift", "ratio": str(ratio)}],
        "warnings": list(result.warnings),
    }
    return construction_fingerprint(projection)


def _fixture(mode: SD3ShiftMode, steps: int, ratio: float) -> dict[str, object]:
    result = build_sd3_schedule(mode=mode, steps=steps, strict_source=True)
    numerical = numerical_fingerprint(
        result.sigmas, domain=result.final_domain, precision="float64"
    )
    construction = _construction(result, ratio)
    public_mode = (
        "Publisher Reference (1.0)"
        if mode is SD3ShiftMode.PUBLISHER_REFERENCE
        else "Comfy/Diffusers Fixed (3.0)"
    )
    node = WorkflowRequirement(identifier="Sigmax.SD3SigmaScheduler", version="1")
    profile = WorkflowRequirement(
        identifier=result.request.provenance.profile_id or "",
        version=result.request.provenance.profile_version or "",
    )
    metadata = WorkflowMetadata(
        package=WorkflowRequirement(identifier="comfyui-sigmax", version=VERSION),
        nodes=(node,),
        host=WorkflowHostRequirement(
            identifier="comfyui", version="0.29.0", api_version="legacy_v1"
        ),
        profile=profile,
        compatibility=CompatibilityDecision(
            level=CompatibilityLevel.ALLOW,
            considered=tuple(CapabilityDimension),
            reasons=(CompatibilityReason.COMPATIBLE,),
        ),
        artifact=WorkflowArtifactReference(
            construction_fingerprint=construction, numerical_fingerprint=numerical
        ),
    )
    identifier = (
        "sd3-publisher-reference-official-50"
        if mode is SD3ShiftMode.PUBLISHER_REFERENCE
        else "sd3-comfy-diffusers-fixed-framework-28"
    )
    workflow = attach_workflow_metadata(
        {
            "version": 0.4,
            "last_node_id": 1,
            "last_link_id": 0,
            "nodes": [
                {
                    "id": 1,
                    "type": "Sigmax.SD3SigmaScheduler",
                    "pos": [40, 80],
                    "size": [370, 230],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {"name": "sigmas", "type": "SIGMAS", "links": None, "slot_index": 0},
                        {"name": "schedule_info", "type": "STRING", "links": None, "slot_index": 1},
                    ],
                    "properties": {
                        "Node name for S&R": "Sigmax.SD3SigmaScheduler",
                        "cnr_id": "comfyui-sigmax",
                    },
                    "widgets_values": [public_mode, steps, True, 0, -1, False],
                }
            ],
            "links": [],
        },
        metadata,
    )
    return {
        "host": {"api_version": "legacy_v1", "id": "comfyui", "version": "0.29.0"},
        "id": identifier,
        "node_contracts": [
            {
                "node_id": 1,
                "node_type": "Sigmax.SD3SigmaScheduler",
                "widget_slots": [
                    {"fixed": public_mode, "name": "mode", "type": "COMBO"},
                    {"name": "steps", "type": "INT"},
                    {"name": "strict_source", "type": "BOOLEAN"},
                    {"name": "start_step", "type": "INT"},
                    {"name": "end_step", "type": "INT"},
                    {"name": "already_shifted", "type": "BOOLEAN"},
                ],
            }
        ],
        "nodes": [node.projection()],
        "package": {"id": "comfyui-sigmax", "version": VERSION},
        "profile": profile.projection(),
        "variant": "SD3",
        "workflow": workflow,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--host-baseline", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    bundle = cast(dict[str, Any], json.loads(parsed.fixtures.read_text(encoding="utf-8")))
    retained = [item for item in bundle["fixtures"] if not item["id"].startswith("sd3-")]
    retained.extend(
        (
            _fixture(SD3ShiftMode.PUBLISHER_REFERENCE, 50, 1.0),
            _fixture(SD3ShiftMode.COMFY_DIFFUSERS_FIXED, 28, 3.0),
        )
    )
    bundle["fixtures"] = sorted(retained, key=lambda item: item["id"])
    parsed.fixtures.write_text(_json(bundle), encoding="utf-8", newline="\n")

    baseline = cast(dict[str, Any], json.loads(parsed.host_baseline.read_text(encoding="utf-8")))
    registry = builtin_node_registry()
    required_ids = {
        "Sigmax.Krea2ConditioningRebalance",
        "Sigmax.Flux1SchnellSigmaScheduler",
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.RawWorkflowOutput",
        "Sigmax.ScheduleInspector",
        "Sigmax.SD3SigmaScheduler",
        "Sigmax.TurboWorkflowOutput",
        "Sigmax.QwenImageSigmaScheduler",
        "Sigmax.ZImageSigmaScheduler",
    }
    object_info = registry.object_info_projection()
    node_definition_v2 = registry.node_definition_v2_projection()
    classes = registry.class_mappings()
    for node_id in required_ids:
        node_class = cast(Any, classes[node_id])
        input_order = tuple(node_class.INPUT_TYPES().get("required", {}))
        legacy_input = object_info[node_id].get("input", {})
        if isinstance(legacy_input, dict) and isinstance(legacy_input.get("required"), dict):
            required = legacy_input["required"]
            legacy_input["required"] = {name: required[name] for name in input_order}
        v2_inputs = node_definition_v2[node_id].get("inputs", {})
        if isinstance(v2_inputs, dict):
            node_definition_v2[node_id]["inputs"] = {
                name: v2_inputs[name] for name in input_order if name in v2_inputs
            }
    baseline["object_info"] = {key: object_info[key] for key in sorted(required_ids)}
    baseline["node_definition_v2"] = {key: node_definition_v2[key] for key in sorted(required_ids)}
    parsed.host_baseline.write_text(_json(baseline), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
