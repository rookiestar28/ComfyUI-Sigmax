"""Add the deterministic FLUX.1-schnell workflow and refresh the host baseline."""

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
from comfyui_sigmax.profiles import build_flux1_schnell_schedule
from comfyui_sigmax.version import VERSION


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"


def _fixture() -> dict[str, object]:
    result = build_flux1_schnell_schedule(steps=4, strict_official=True)
    numerical = numerical_fingerprint(
        result.sigmas, domain=result.final_domain, precision="float64"
    )
    construction = construction_fingerprint(
        {
            "base_grid": {"id": "flowmatch.reciprocal_step"},
            "effective": {"steps": 4},
            "engine": {"version": result.request.provenance.engine_version},
            "evidence": {"level": result.request.provenance.evidence.value},
            "numerical_fingerprint": numerical,
            "overrides": [],
            "ownership": "external_sigmas",
            "requested": {"steps": 4},
            "schema": "sigmax.schedule-artifact/1",
            "slicing": {"end_step": None, "start_step": 0},
            "source": {
                "profile_id": result.request.provenance.profile_id,
                "revision": result.request.provenance.source_revision,
            },
            "terminal": {"policy": "append_zero"},
            "transforms": [],
            "warnings": [],
        }
    )
    node = WorkflowRequirement(identifier="Sigmax.Flux1SchnellSigmaScheduler", version="1")
    profile = WorkflowRequirement(identifier="flux1.schnell.official", version="1")
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
    workflow = attach_workflow_metadata(
        {
            "version": 0.4,
            "last_node_id": 1,
            "last_link_id": 0,
            "nodes": [
                {
                    "id": 1,
                    "type": "Sigmax.Flux1SchnellSigmaScheduler",
                    "pos": [40, 80],
                    "size": [340, 190],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {"name": "sigmas", "type": "SIGMAS", "links": None, "slot_index": 0},
                        {"name": "schedule_info", "type": "STRING", "links": None, "slot_index": 1},
                    ],
                    "properties": {
                        "Node name for S&R": "Sigmax.Flux1SchnellSigmaScheduler",
                        "cnr_id": "comfyui-sigmax",
                    },
                    "widgets_values": [4, True, 0, -1],
                }
            ],
            "links": [],
        },
        metadata,
    )
    return {
        "host": {"api_version": "legacy_v1", "id": "comfyui", "version": "0.29.0"},
        "id": "flux1-schnell-official-4",
        "node_contracts": [
            {
                "node_id": 1,
                "node_type": "Sigmax.Flux1SchnellSigmaScheduler",
                "widget_slots": [
                    {"name": "steps", "type": "INT"},
                    {"name": "strict_official", "type": "BOOLEAN"},
                    {"name": "start_step", "type": "INT"},
                    {"name": "end_step", "type": "INT"},
                ],
            }
        ],
        "nodes": [node.projection()],
        "package": {"id": "comfyui-sigmax", "version": VERSION},
        "profile": profile.projection(),
        "variant": "FLUX.1-schnell",
        "workflow": workflow,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--host-baseline", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    bundle = cast(dict[str, Any], json.loads(parsed.fixtures.read_text(encoding="utf-8")))
    retained = [item for item in bundle["fixtures"] if item["id"] != "flux1-schnell-official-4"]
    retained.append(_fixture())
    bundle["fixtures"] = sorted(retained, key=lambda item: item["id"])
    parsed.fixtures.write_text(_json(bundle), encoding="utf-8", newline="\n")

    baseline = cast(dict[str, Any], json.loads(parsed.host_baseline.read_text(encoding="utf-8")))
    registry = builtin_node_registry()
    required_ids = {
        "Sigmax.Flux1SchnellSigmaScheduler",
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.RawWorkflowOutput",
        "Sigmax.ScheduleInspector",
        "Sigmax.TurboWorkflowOutput",
        "Sigmax.ZImageSigmaScheduler",
    }
    object_info = registry.object_info_projection()
    node_definition_v2 = registry.node_definition_v2_projection()
    classes = registry.class_mappings()
    for node_id in required_ids:
        input_order = tuple(cast(Any, classes[node_id]).INPUT_TYPES().get("required", {}))
        required = object_info[node_id].get("input", {}).get("required", {})
        object_info[node_id]["input"]["required"] = {name: required[name] for name in input_order}
        v2_inputs = node_definition_v2[node_id].get("inputs", {})
        node_definition_v2[node_id]["inputs"] = {
            name: v2_inputs[name] for name in input_order if name in v2_inputs
        }
    baseline["object_info"] = {key: object_info[key] for key in sorted(required_ids)}
    baseline["node_definition_v2"] = {key: node_definition_v2[key] for key in sorted(required_ids)}
    parsed.host_baseline.write_text(_json(baseline), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
