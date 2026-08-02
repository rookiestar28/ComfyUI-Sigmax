"""Generate deterministic model-free workflow fixtures for the LTX family."""

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
from comfyui_sigmax.nodes.ltx_sigma_scheduler import build_ltx_sigma_schedule
from comfyui_sigmax.profiles.ltx import LTXProfileId, build_ltx_schedule


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2) + "\n"


def _fixture(
    *,
    identifier: str,
    variant: str,
    generation: str,
    stage: str,
    steps: int,
    token_count: int,
    profile_id: str,
) -> dict[str, object]:
    result = build_ltx_sigma_schedule(
        generation=generation,
        stage=stage,
        steps=steps,
        token_count=token_count,
        stretch=True,
        terminal=0.1,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    schedule = build_ltx_schedule(
        profile=LTXProfileId(profile_id),
        steps=steps,
        token_count=(token_count if stage == "Dev" else None),
        stretch=True,
        terminal=0.1,
        strict_official=True,
    )
    numerical = numerical_fingerprint(result.sigmas, domain=result.domain, precision="float64")
    construction = {
        "base_grid": {
            "id": "flowmatch.linear_endpoint",
            "parameters": {"steps": steps},
        },
        "effective": {"steps": steps},
        "engine": {"version": "1.0.0"},
        "evidence": {"level": schedule.request.provenance.evidence.value},
        "numerical_fingerprint": numerical,
        "overrides": [],
        "ownership": "external_sigmas",
        "requested": {"steps": steps, "token_count": token_count},
        "schema": "sigmax.schedule-artifact/1",
        "slicing": {"end_step": None, "start_step": 0},
        "source": {"profile_id": profile_id},
        "terminal": {"policy": "append_zero"},
        "transforms": [
            {
                "id": item.name,
                "stage": item.stage.value,
                "from_domain": item.input_domain.value,
                "to_domain": item.output_domain.value,
                "parameters": {},
            }
            for item in schedule.request.transforms
        ],
        "warnings": list(schedule.warnings),
    }
    node_type = "Sigmax.LTXSigmaScheduler"
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
                    "size": [460, 360],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "outputs": [
                        {"name": "sigmas", "type": "SIGMAS", "links": None, "slot_index": 0},
                        {"name": "schedule_info", "type": "STRING", "links": None, "slot_index": 1},
                    ],
                    "properties": {"Node name for S&R": node_type, "cnr_id": "comfyui-sigmax"},
                    "widgets_values": [
                        generation,
                        stage,
                        steps,
                        token_count,
                        True,
                        0.1,
                        True,
                        0,
                        -1,
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
                    {"fixed": stage, "name": "stage", "type": "COMBO"},
                    {"name": "steps", "type": "INT"},
                    {"name": "token_count", "type": "INT"},
                    {"name": "stretch", "type": "BOOLEAN"},
                    {"name": "terminal", "type": "FLOAT"},
                    {"name": "strict_official", "type": "BOOLEAN"},
                    {"name": "start_step", "type": "INT"},
                    {"name": "end_step", "type": "INT"},
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
    retained = [item for item in bundle["fixtures"] if not item["id"].startswith("ltx")]
    retained.extend(
        (
            _fixture(
                identifier="ltxv-0-9-8-dev-20",
                variant="LTXV 0.9.8 Dev",
                generation="LTXV 0.9.8",
                stage="Dev",
                steps=20,
                token_count=4096,
                profile_id=LTXProfileId.LTXV_098_DEV.value,
            ),
            _fixture(
                identifier="ltx2-19b-dev-40",
                variant="LTX-2 19B Dev",
                generation="LTX-2 19B",
                stage="Dev",
                steps=40,
                token_count=4096,
                profile_id=LTXProfileId.LTX2_19B_DEV.value,
            ),
            _fixture(
                identifier="ltx2-19b-distilled-stage1-8",
                variant="LTX-2 19B Distilled Stage 1",
                generation="LTX-2 19B",
                stage="Distilled Stage 1",
                steps=8,
                token_count=4096,
                profile_id=LTXProfileId.LTX2_19B_DISTILLED_STAGE1.value,
            ),
            _fixture(
                identifier="ltx2-3-22b-distilled-stage2-3",
                variant="LTX-2.3 22B Distilled Stage 2",
                generation="LTX-2.3 22B",
                stage="Distilled Stage 2",
                steps=3,
                token_count=4096,
                profile_id=LTXProfileId.LTX23_22B_DISTILLED_STAGE2.value,
            ),
            _fixture(
                identifier="ltx2-3-22b-dev-30",
                variant="LTX-2.3 22B Dev",
                generation="LTX-2.3 22B",
                stage="Dev",
                steps=30,
                token_count=4096,
                profile_id=LTXProfileId.LTX23_22B_DEV.value,
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
