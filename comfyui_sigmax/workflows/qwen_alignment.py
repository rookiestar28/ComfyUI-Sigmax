"""Packaged model-free migration workflows and exact saved-widget checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from comfyui_sigmax.core import ScheduleContractError

_ORIGINAL = "Sigmax.QwenImageSigmaScheduler"
_WIDGETS = ("mode", "steps", "image_seq_len", "strict_official", "start_step", "end_step")


def load_qwen_alignment_workflow() -> dict[str, Any]:
    value = json.loads(files(__package__).joinpath("qwen_native_original_v1.json").read_text())
    if not isinstance(value, dict):
        raise ScheduleContractError("Qwen workflow must be an object")
    return value


def validate_qwen_alignment_workflow(object_info: Mapping[str, Any]) -> dict[str, object]:
    workflow = load_qwen_alignment_workflow()
    try:
        node = workflow["nodes"][0]
        schema = object_info[_ORIGINAL]
        required = schema["input"]["required"]
        # IMPORTANT: canonical JSON may sort input objects. ComfyUI's explicit input_order
        # preserves widget slots; use it when supplied instead of inferring sorted-key order.
        order = schema.get("input_order", {}).get("required", list(required))
        values = node["widgets_values"]
        if (
            node["type"] != _ORIGINAL
            or tuple(order) != _WIDGETS
            or set(required) != set(_WIDGETS)
            or values != ["Comfy Native", 50, 0, True, 0, -1]
            or list(required["mode"][0]) != ["Comfy Fixed", "Diffusers Dynamic", "Comfy Native"]
            or required["steps"][1]["default"] != 50
            or list(schema["output"]) != ["SIGMAS", "STRING"]
        ):
            raise ScheduleContractError("Qwen native saved workflow/schema drift")
    except (KeyError, TypeError, IndexError) as exc:
        raise ScheduleContractError("Qwen native workflow/schema is incomplete") from exc
    return {"status": "succeeded", "node": _ORIGINAL, "widgets": list(_WIDGETS), "values": values}


def load_qwen_image21_workflows() -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(files(__package__).joinpath(f"qwen_image21_{lane}_v1.json").read_text())
        for lane in ("native", "dynamic")
    )


def validate_qwen_image21_workflows(object_info: Mapping[str, Any]) -> dict[str, object]:
    node_id = "Sigmax.QwenImage21SigmaScheduler"
    widgets = (
        "mode",
        "steps",
        "target_source",
        "width",
        "height",
        "image_seq_len",
        "strict_official",
        "start_step",
        "end_step",
    )
    expected = (
        ["Comfy Native", 25, "None (Native)", 0, 0, 0, True, 0, -1],
        ["Diffusers Dynamic", 40, "Dimensions", 1024, 1024, 0, True, 0, -1],
    )
    try:
        schema = object_info[node_id]
        required = schema["input"]["required"]
        order = schema.get("input_order", {}).get("required", list(required))
        if (
            tuple(order) != widgets
            or set(required) != set(widgets)
            or list(required["mode"][0]) != ["Comfy Native", "Diffusers Dynamic"]
            or list(required["target_source"][0])
            != ["None (Native)", "Dimensions", "Target Tokens"]
            or required["steps"][1]["default"] != 25
            or list(schema["output"]) != ["SIGMAS", "STRING"]
        ):
            raise ScheduleContractError("Qwen 2.1 workflow/schema drift")
        for workflow, values in zip(load_qwen_image21_workflows(), expected, strict=True):
            node = workflow["nodes"][0]
            if node["type"] != node_id or node["widgets_values"] != values:
                raise ScheduleContractError("Qwen 2.1 serialized values drift")
    except (KeyError, TypeError, IndexError) as exc:
        raise ScheduleContractError("Qwen 2.1 workflow/schema is incomplete") from exc
    return {
        "status": "succeeded",
        "node": node_id,
        "widgets": list(widgets),
        "values": list(expected),
    }


__all__ = [
    "load_qwen_alignment_workflow",
    "load_qwen_image21_workflows",
    "validate_qwen_alignment_workflow",
    "validate_qwen_image21_workflows",
]
