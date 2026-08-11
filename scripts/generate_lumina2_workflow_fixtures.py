"""Add the deterministic Lumina-Image 2.0 workflow fixture and host baseline."""

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
from comfyui_sigmax.profiles.lumina2 import Lumina2ShiftMode, build_lumina2_schedule
from comfyui_sigmax.version import VERSION


def _json(value: object, *, indent: int | None = 2) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=indent) + "\n"


def _construction(result: Any) -> str:
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
        "transforms": [{"id": "direct_ratio.shift", "ratio": "6.0"}],
        "warnings": list(result.warnings),
    }
    return construction_fingerprint(projection)


def _fixture() -> dict[str, object]:
    result = build_lumina2_schedule(
        mode=Lumina2ShiftMode.OFFICIAL_FIXED,
        steps=50,
        strict_source=True,
    )
    numerical = numerical_fingerprint(
        result.sigmas, domain=result.final_domain, precision="float64"
    )
    node = WorkflowRequirement(identifier="Sigmax.Lumina2SigmaScheduler", version="1")
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
            construction_fingerprint=_construction(result),
            numerical_fingerprint=numerical,
        ),
    )
    mode = "Official Fixed (6.0)"
    workflow = attach_workflow_metadata(
        {
            "version": 0.4,
            "last_node_id": 1,
            "last_link_id": 0,
            "nodes": [
                {
                    "id": 1,
                    "type": "Sigmax.Lumina2SigmaScheduler",
                    "pos": [40, 80],
                    "size": [370, 230],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {"name": "sigmas", "type": "SIGMAS", "links": None, "slot_index": 0},
                        {
                            "name": "schedule_info",
                            "type": "STRING",
                            "links": None,
                            "slot_index": 1,
                        },
                    ],
                    "properties": {
                        "Node name for S&R": "Sigmax.Lumina2SigmaScheduler",
                        "cnr_id": "comfyui-sigmax",
                    },
                    "widgets_values": [mode, 50, True, 0, -1, False],
                }
            ],
            "links": [],
        },
        metadata,
    )
    return {
        "host": {"api_version": "legacy_v1", "id": "comfyui", "version": "0.29.0"},
        "id": "lumina2-v2-official-50",
        "node_contracts": [
            {
                "node_id": 1,
                "node_type": "Sigmax.Lumina2SigmaScheduler",
                "widget_slots": [
                    {"fixed": mode, "name": "mode", "type": "COMBO"},
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
        "variant": "Lumina-Image 2.0",
        "workflow": workflow,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--host-baseline", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    bundle = cast(dict[str, Any], json.loads(parsed.fixtures.read_text(encoding="utf-8")))
    retained = [item for item in bundle["fixtures"] if not item["id"].startswith("lumina2-")]
    retained.append(_fixture())
    bundle["fixtures"] = sorted(retained, key=lambda item: item["id"])
    parsed.fixtures.write_text(_json(bundle), encoding="utf-8", newline="\n")

    baseline = cast(dict[str, Any], json.loads(parsed.host_baseline.read_text(encoding="utf-8")))
    registry = builtin_node_registry()
    required_ids = {
        "Sigmax.AuraFlowSigmaScheduler",
        "Sigmax.Flux1SchnellSigmaScheduler",
        "Sigmax.Krea2ConditioningRebalance",
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.Lumina2SigmaScheduler",
        "Sigmax.QwenImageSigmaScheduler",
        "Sigmax.RawWorkflowOutput",
        "Sigmax.SD3SigmaScheduler",
        "Sigmax.ScheduleInspector",
        "Sigmax.TurboWorkflowOutput",
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
