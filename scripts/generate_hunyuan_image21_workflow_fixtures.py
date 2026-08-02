"""Add deterministic HunyuanImage 2.1 Base/Distilled workflow fixtures."""

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
from comfyui_sigmax.profiles.hunyuan_image21 import (
    HunyuanImage21Variant,
    build_hunyuan_image21_schedule,
)


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"


def _fixture(*, variant: HunyuanImage21Variant, steps: int, label: str) -> dict[str, object]:
    result = build_hunyuan_image21_schedule(variant=variant, steps=steps, strict_source=True)
    ratio = 5.0 if variant is HunyuanImage21Variant.BASE else 4.0
    profile_id = result.request.provenance.profile_id or ""
    node_type = "Sigmax.HunyuanImage21SigmaScheduler"
    mode = "Base (5.0)" if variant is HunyuanImage21Variant.BASE else "Distilled (4.0)"
    variant_name = (
        "HunyuanImage 2.1 Base"
        if variant is HunyuanImage21Variant.BASE
        else "HunyuanImage 2.1 Distilled"
    )
    numerical = numerical_fingerprint(
        result.sigmas, domain=result.final_domain, precision="float64"
    )
    construction = {
        "base_grid": {"id": "flowmatch.reciprocal_step"},
        "effective": {"steps": steps},
        "engine": {"version": result.request.provenance.engine_version},
        "evidence": {"level": result.request.provenance.evidence.value},
        "numerical_fingerprint": numerical,
        "overrides": [],
        "ownership": "external_sigmas",
        "requested": {"steps": steps},
        "schema": "sigmax.schedule-artifact/1",
        "slicing": {"end_step": None, "start_step": 0},
        "source": {"profile_id": profile_id, "revision": result.request.provenance.source_revision},
        "terminal": {"policy": "append_zero"},
        "transforms": [{"id": "direct_ratio.shift", "ratio": str(ratio)}],
        "warnings": list(result.warnings),
    }
    node = WorkflowRequirement(identifier=node_type, version="1")
    profile = WorkflowRequirement(
        identifier=profile_id, version=result.request.provenance.profile_version or ""
    )
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
            construction_fingerprint=construction_fingerprint(construction),
            numerical_fingerprint=numerical,
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
                    "type": node_type,
                    "pos": [40, 80],
                    "size": [390, 250],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {"name": "sigmas", "type": "SIGMAS", "links": None, "slot_index": 0},
                        {"name": "schedule_info", "type": "STRING", "links": None, "slot_index": 1},
                    ],
                    "properties": {"Node name for S&R": node_type, "cnr_id": "comfyui-sigmax"},
                    "widgets_values": [mode, steps, True, 0, -1, False],
                }
            ],
            "links": [],
        },
        metadata,
    )
    return {
        "host": {"api_version": "legacy_v1", "id": "comfyui", "version": "0.29.0"},
        "id": f"hunyuan-image21-{label}-{steps}",
        "node_contracts": [
            {
                "node_id": 1,
                "node_type": node_type,
                "widget_slots": [
                    {"fixed": mode, "name": "variant", "type": "COMBO"},
                    {"name": "steps", "type": "INT"},
                    {"name": "strict_source", "type": "BOOLEAN"},
                    {"name": "start_step", "type": "INT"},
                    {"name": "end_step", "type": "INT"},
                    {"name": "already_shifted", "type": "BOOLEAN"},
                ],
            }
        ],
        "nodes": [node.projection()],
        "package": {"id": "comfyui-sigmax", "version": "1.0.0"},
        "profile": profile.projection(),
        "variant": variant_name,
        "workflow": workflow,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--host-baseline", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    bundle = cast(dict[str, Any], json.loads(parsed.fixtures.read_text(encoding="utf-8")))
    retained = [
        item for item in bundle["fixtures"] if not item["id"].startswith("hunyuan-image21-")
    ]
    retained.extend(
        (
            _fixture(variant=HunyuanImage21Variant.BASE, steps=50, label="base-official"),
            _fixture(variant=HunyuanImage21Variant.DISTILLED, steps=8, label="distilled-official"),
        )
    )
    bundle["fixtures"] = sorted(retained, key=lambda item: item["id"])
    parsed.fixtures.write_text(_json(bundle), encoding="utf-8", newline="\n")

    baseline = cast(dict[str, Any], json.loads(parsed.host_baseline.read_text(encoding="utf-8")))
    registry = builtin_node_registry()
    required_ids = {
        "Sigmax.AuraFlowSigmaScheduler",
        "Sigmax.Flux1SchnellSigmaScheduler",
        "Sigmax.HunyuanImage21SigmaScheduler",
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
