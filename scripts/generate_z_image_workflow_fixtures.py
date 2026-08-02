"""Add deterministic Z-Image scheduler workflows and refresh the pinned node baseline."""

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
from comfyui_sigmax.profiles import ZImageVariant, build_z_image_schedule


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"


def _ordered_json(value: object) -> str:
    """Preserve nested host input declaration order while keeping stable compact JSON."""

    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"


def _construction(result: Any, variant: ZImageVariant, ratio: int) -> str:
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
        "transforms": [{"id": "flowmatch.direct_ratio", "ratio": str(ratio)}],
        "warnings": list(result.warnings),
    }
    return construction_fingerprint(projection)


def _fixture(variant: ZImageVariant, steps: int, ratio: int) -> dict[str, object]:
    result = build_z_image_schedule(variant=variant, steps=steps, strict_official=True)
    numerical = numerical_fingerprint(
        result.sigmas, domain=result.final_domain, precision="float64"
    )
    construction = _construction(result, variant, ratio)
    node = WorkflowRequirement(identifier="Sigmax.ZImageSigmaScheduler", version="1")
    profile = WorkflowRequirement(identifier=f"z_image.{variant.value}.official", version="1")
    metadata = WorkflowMetadata(
        package=WorkflowRequirement(identifier="comfyui-sigmax", version="1.0.0"),
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
    public_variant = "Base" if variant is ZImageVariant.BASE else "Turbo"
    identifier = f"z-image-{variant.value}-official-{steps}"
    workflow = attach_workflow_metadata(
        {
            "version": 0.4,
            "last_node_id": 1,
            "last_link_id": 0,
            "nodes": [
                {
                    "id": 1,
                    "type": "Sigmax.ZImageSigmaScheduler",
                    "pos": [40, 80],
                    "size": [340, 210],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {"name": "sigmas", "type": "SIGMAS", "links": None, "slot_index": 0},
                        {"name": "schedule_info", "type": "STRING", "links": None, "slot_index": 1},
                    ],
                    "properties": {
                        "Node name for S&R": "Sigmax.ZImageSigmaScheduler",
                        "cnr_id": "comfyui-sigmax",
                    },
                    "widgets_values": [public_variant, steps, True, 0, -1],
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
                "node_type": "Sigmax.ZImageSigmaScheduler",
                "widget_slots": [
                    {"fixed": public_variant, "name": "variant", "type": "COMBO"},
                    {"name": "steps", "type": "INT"},
                    {"name": "strict_official", "type": "BOOLEAN"},
                    {"name": "start_step", "type": "INT"},
                    {"name": "end_step", "type": "INT"},
                ],
            }
        ],
        "nodes": [node.projection()],
        "package": {"id": "comfyui-sigmax", "version": "1.0.0"},
        "profile": profile.projection(),
        "variant": f"Z-Image {public_variant}",
        "workflow": workflow,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--host-baseline", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    bundle = cast(dict[str, Any], json.loads(parsed.fixtures.read_text(encoding="utf-8")))
    retained = [item for item in bundle["fixtures"] if not item["id"].startswith("z-image-")]
    retained.extend((_fixture(ZImageVariant.BASE, 50, 6), _fixture(ZImageVariant.TURBO, 8, 3)))
    bundle["fixtures"] = sorted(retained, key=lambda item: item["id"])
    parsed.fixtures.write_text(_canonical(bundle), encoding="utf-8", newline="\n")

    baseline = cast(dict[str, Any], json.loads(parsed.host_baseline.read_text(encoding="utf-8")))
    registry = builtin_node_registry()
    required_ids = {
        "Sigmax.Krea2ConditioningRebalance",
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
    parsed.host_baseline.write_text(_ordered_json(baseline), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
