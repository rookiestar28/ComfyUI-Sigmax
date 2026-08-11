"""Add deterministic Wan family workflow fixtures and host schema evidence."""

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
from comfyui_sigmax.nodes.wan_sigma_scheduler import build_wan_sigma_schedule


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"


def _fixture(
    *,
    identifier: str,
    variant: str,
    generation: str,
    task: str,
    source: str,
    resolution: str,
    steps: int,
    profile_id: str,
) -> dict[str, object]:
    result = build_wan_sigma_schedule(
        generation=generation,
        task=task,
        source=source,
        resolution=resolution,
        steps=steps,
        strict_source=True,
        start_step=0,
        end_step=-1,
        already_shifted=False,
    )
    numerical = numerical_fingerprint(result.sigmas, domain=result.domain, precision="float64")
    info = json.loads(result.schedule_info_json)
    construction = {
        "base_grid": {
            "id": "flowmatch.reciprocal_step",
            "parameters": {"training_timesteps": 1000},
        },
        "effective": {"steps": steps},
        "engine": {"version": "1.0.0"},
        "evidence": {"level": info["profile"]["evidence"]},
        "numerical_fingerprint": numerical,
        "overrides": [],
        "ownership": "external_sigmas",
        "requested": {"steps": steps},
        "schema": "sigmax.schedule-artifact/1",
        "slicing": {"end_step": None, "start_step": 0},
        "source": {
            "profile_id": profile_id,
            "revision": info["profile"]["revision"],
        },
        "terminal": {"policy": "append_zero"},
        "transforms": [{"id": "direct_ratio.shift", "ratio": str(info["shift"]["ratio"])}],
        "warnings": list(info["warnings"]),
    }
    node_type = "Sigmax.WanSigmaScheduler"
    node = WorkflowRequirement(identifier=node_type, version="1")
    profile = WorkflowRequirement(identifier=profile_id, version="1")
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
                    "size": [440, 330],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {"name": "sigmas", "type": "SIGMAS", "links": None, "slot_index": 0},
                        {"name": "boundary_step", "type": "INT", "links": None, "slot_index": 1},
                        {"name": "schedule_info", "type": "STRING", "links": None, "slot_index": 2},
                    ],
                    "properties": {"Node name for S&R": node_type, "cnr_id": "comfyui-sigmax"},
                    "widgets_values": [
                        generation,
                        task,
                        source,
                        resolution,
                        steps,
                        True,
                        0,
                        -1,
                        False,
                    ],
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
                "node_type": node_type,
                "widget_slots": [
                    {"fixed": generation, "name": "generation", "type": "COMBO"},
                    {"fixed": task, "name": "task", "type": "COMBO"},
                    {"fixed": source, "name": "source", "type": "COMBO"},
                    {"fixed": resolution, "name": "resolution", "type": "COMBO"},
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
        "variant": variant,
        "workflow": workflow,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--host-baseline", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    bundle = cast(dict[str, Any], json.loads(parsed.fixtures.read_text(encoding="utf-8")))
    retained = [item for item in bundle["fixtures"] if not item["id"].startswith("wan")]
    retained.extend(
        (
            _fixture(
                identifier="wan21-t2v-official-50",
                variant="Wan 2.1 T2V",
                generation="Wan 2.1",
                task="T2V",
                source="Official native",
                resolution="None",
                steps=50,
                profile_id="wan2.1.t2v.official-native",
            ),
            _fixture(
                identifier="wan21-i2v-480p-official-40",
                variant="Wan 2.1 I2V 480P",
                generation="Wan 2.1",
                task="I2V",
                source="Official native",
                resolution="480P",
                steps=40,
                profile_id="wan2.1.i2v.480p.official-native",
            ),
            _fixture(
                identifier="wan22-ti2v-5b-native-50",
                variant="Wan 2.2 TI2V 5B",
                generation="Wan 2.2",
                task="TI2V",
                source="ComfyUI native",
                resolution="None",
                steps=50,
                profile_id="wan2.2.ti2v.5b.comfy-native",
            ),
            _fixture(
                identifier="wan22-t2v-a14b-native-40",
                variant="Wan 2.2 T2V A14B",
                generation="Wan 2.2",
                task="T2V A14B",
                source="Official native",
                resolution="None",
                steps=40,
                profile_id="wan2.2.t2v-a14b.official-native",
            ),
            _fixture(
                identifier="wan21-flf2v-720p-official-50",
                variant="Wan 2.1 FLF2V 14B 720P",
                generation="Wan 2.1",
                task="FLF2V",
                source="Official native",
                resolution="720P",
                steps=50,
                profile_id="wan2.1.flf2v.14b.720p.official-native",
            ),
            _fixture(
                identifier="wan21-vace-1-3b-official-50",
                variant="Wan 2.1 VACE 1.3B",
                generation="Wan 2.1",
                task="VACE 1.3B",
                source="Official native",
                resolution="None",
                steps=50,
                profile_id="wan2.1.vace.1.3b.official-native",
            ),
            _fixture(
                identifier="wan21-vace-14b-official-50",
                variant="Wan 2.1 VACE 14B",
                generation="Wan 2.1",
                task="VACE 14B",
                source="Official native",
                resolution="None",
                steps=50,
                profile_id="wan2.1.vace.14b.official-native",
            ),
            _fixture(
                identifier="wan22-s2v-14b-official-40",
                variant="Wan 2.2 S2V 14B",
                generation="Wan 2.2",
                task="S2V",
                source="Official native",
                resolution="None",
                steps=40,
                profile_id="wan2.2.s2v.14b.official-native",
            ),
            _fixture(
                identifier="wan22-animate-14b-official-20",
                variant="Wan 2.2 Animate 14B",
                generation="Wan 2.2",
                task="Animate",
                source="Official native",
                resolution="None",
                steps=20,
                profile_id="wan2.2.animate.14b.official-native",
            ),
            _fixture(
                identifier="wan-animate2-base-14b-official-40",
                variant="Wan Animate 2 Base 14B",
                generation="Wan Animate 2",
                task="Animate Base",
                source="Official native",
                resolution="None",
                steps=40,
                profile_id="wan-animate2.14b.base.official-native",
            ),
            _fixture(
                identifier="wan-animate2-distilled-14b-official-10",
                variant="Wan Animate 2 Distilled 14B",
                generation="Wan Animate 2",
                task="Animate Distilled",
                source="Official native",
                resolution="None",
                steps=10,
                profile_id="wan-animate2.14b.distilled.official-native",
            ),
        )
    )
    bundle["fixtures"] = sorted(retained, key=lambda item: item["id"])
    parsed.fixtures.write_text(_json(bundle), encoding="utf-8", newline="\n")

    baseline = cast(dict[str, Any], json.loads(parsed.host_baseline.read_text(encoding="utf-8")))
    registry = builtin_node_registry()
    required_ids = {
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
