"""Model-free host-schema preflight for generated MiniMax H3 workflows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, cast

from comfyui_sigmax.adapters.comfyui import (
    ComfyAdapterCompatibilityError,
    ComfyAdapterReason,
    ComfyNodeDefinition,
    normalize_object_info,
)
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.workflows.minimax_h3 import (
    MiniMaxH3HostWorkflow,
    MiniMaxH3PublicVariant,
)

MINIMAX_H3_HOST_SCHEMA_REPORT: Final = "sigmax.minimax-h3-host-schema-report/1"
_MAX_ISSUE_TEXT: Final = 256
_DYNAMIC_PREFIX: Final = "ref_image_"
_DYNAMIC_INPUT: Final = "ref_images"
_LOAD_IMAGE: Final = "LoadImage"
_EXPECTED_INPUT_TYPES: Final = {
    "UNETLoader": {"unet_name": "COMBO", "weight_dtype": "COMBO"},
    "CLIPLoader": {"clip_name": "COMBO", "type": "COMBO"},
    "VAELoader": {"vae_name": "COMBO"},
    "MiniMaxH3SigmaShift": {
        "model": "MODEL",
        "shift_video": "FLOAT",
        "shift_audio": "FLOAT",
    },
    "MiniMaxH3ImageToVideo": {
        "clip": "CLIP",
        "vae": "VAE",
        "prompt": "STRING",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "first_frame": "IMAGE",
        "last_frame": "IMAGE",
    },
    "MiniMaxH3ReferenceToVideo": {
        "clip": "CLIP",
        "vae": "VAE",
        "audio_vae": "VAE",
        "prompt": "STRING",
        "width": "INT",
        "height": "INT",
        "length": "INT",
        "ref_image_size": "COMBO",
        _DYNAMIC_INPUT: "COMFY_AUTOGROW_V3",
    },
    "Sigmax.MiniMaxH3SigmaScheduler": {
        "variant": "COMBO",
        "grid_points": "INT",
        "start_step": "INT",
        "end_step": "INT",
        "already_shifted": "BOOLEAN",
    },
    "KSamplerSelect": {"sampler_name": "COMBO"},
    "BasicGuider": {"model": "MODEL", "conditioning": "CONDITIONING"},
    "RandomNoise": {"noise_seed": "INT"},
    "SamplerCustomAdvanced": {
        "noise": "NOISE",
        "guider": "GUIDER",
        "sampler": "SAMPLER",
        "sigmas": "SIGMAS",
        "latent_image": "LATENT",
    },
    "LTXVSeparateAVLatent": {"av_latent": "LATENT"},
    "VAEDecode": {"samples": "LATENT", "vae": "VAE"},
    "VAEDecodeAudio": {"samples": "LATENT", "vae": "VAE"},
    "PreviewImage": {"images": "IMAGE"},
    "PreviewAny": {"source": "*"},
    _LOAD_IMAGE: {"image": "COMBO"},
}


class MiniMaxH3SchemaIssueKind(str, Enum):
    """Stable failure categories for a generated H3 graph preflight."""

    MISSING_NODE = "missing_node"
    MISSING_INPUT = "missing_input"
    INPUT_TYPE_MISMATCH = "input_type_mismatch"
    COMBO_VALUE_UNAVAILABLE = "combo_value_unavailable"
    DYNAMIC_INPUT_MALFORMED = "dynamic_input_malformed"
    DYNAMIC_INPUT_LIMIT = "dynamic_input_limit"
    OUTPUT_MISMATCH = "output_mismatch"


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3SchemaIssue:
    """One bounded, deterministic H3 host-schema finding."""

    kind: MiniMaxH3SchemaIssueKind
    node_id: str
    input_name: str = ""
    expected: str = ""
    actual: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MiniMaxH3SchemaIssueKind):
            raise ScheduleContractError("MiniMax H3 schema issue kind is invalid")
        for label, value in (
            ("node ID", self.node_id),
            ("input name", self.input_name),
            ("expected value", self.expected),
            ("actual value", self.actual),
        ):
            if not isinstance(value, str) or len(value) > _MAX_ISSUE_TEXT:
                raise ScheduleContractError(f"MiniMax H3 schema issue {label} is invalid")

    def projection(self) -> dict[str, str]:
        return {
            "actual": self.actual,
            "expected": self.expected,
            "input": self.input_name,
            "kind": self.kind.value,
            "node": self.node_id,
        }


def _issue_sort_key(issue: MiniMaxH3SchemaIssue) -> tuple[str, str, str, str, str]:
    return (
        issue.node_id,
        issue.input_name,
        issue.kind.value,
        issue.expected,
        issue.actual,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3HostSchemaReport:
    """Fingerprintable result of validating one generated H3 graph against `/object_info`."""

    variant: MiniMaxH3PublicVariant
    checked_nodes: tuple[str, ...]
    issues: tuple[MiniMaxH3SchemaIssue, ...]
    compatible: bool
    report_bytes: bytes = field(init=False, repr=False)
    report_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.variant not in {"H3 Base FL2VA", "H3 Base Ref2VA"}:
            raise ScheduleContractError("MiniMax H3 schema report variant is invalid")
        if not isinstance(self.checked_nodes, tuple) or any(
            not isinstance(item, str) or not item for item in self.checked_nodes
        ):
            raise ScheduleContractError("MiniMax H3 checked node IDs are invalid")
        if self.checked_nodes != tuple(sorted(set(self.checked_nodes))):
            raise ScheduleContractError("MiniMax H3 checked node IDs are not canonical")
        if not isinstance(self.issues, tuple) or any(
            not isinstance(item, MiniMaxH3SchemaIssue) for item in self.issues
        ):
            raise ScheduleContractError("MiniMax H3 schema issues are invalid")
        if self.issues != tuple(sorted(self.issues, key=_issue_sort_key)):
            raise ScheduleContractError("MiniMax H3 schema issues are not canonical")
        if not isinstance(self.compatible, bool) or self.compatible != (not self.issues):
            raise ScheduleContractError("MiniMax H3 schema result disagrees with issues")
        projection = self.projection()
        report_bytes = json.dumps(
            projection,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        object.__setattr__(self, "report_bytes", report_bytes)
        object.__setattr__(
            self, "report_fingerprint", "sha256:" + hashlib.sha256(report_bytes).hexdigest()
        )

    def projection(self) -> dict[str, object]:
        return {
            "checked_nodes": list(self.checked_nodes),
            "compatible": self.compatible,
            "issues": [item.projection() for item in self.issues],
            "schema": MINIMAX_H3_HOST_SCHEMA_REPORT,
            "variant": self.variant,
        }


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ScheduleContractError(f"MiniMax H3 {label} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ScheduleContractError(f"MiniMax H3 {label} must be an array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_ISSUE_TEXT:
        raise ScheduleContractError(f"MiniMax H3 {label} must be bounded text")
    return value


def _raw_input_sections(raw_node: Mapping[str, object]) -> Mapping[str, object]:
    if "input" in raw_node and "inputs" not in raw_node:
        return _mapping(raw_node.get("input"), label="V1 input mapping")
    if "inputs" in raw_node and "input" not in raw_node:
        return {"required": raw_node.get("inputs")}
    raise ScheduleContractError("MiniMax H3 host node must use one input schema form")


def _raw_input_declaration(raw_node: Mapping[str, object], name: str) -> object:
    sections = _raw_input_sections(raw_node)
    for section in ("required", "optional", "hidden"):
        values = sections.get(section, {})
        if isinstance(values, Mapping) and name in values:
            return values[name]
    return None


def _raw_output_types(raw_node: Mapping[str, object]) -> tuple[str, ...]:
    value = raw_node.get("output", raw_node.get("outputs"))
    if value is None:
        return ()
    result: list[str] = []
    for item in _sequence(value, label="output types"):
        if isinstance(item, Mapping):
            item = item.get("type")
        result.append(_text(item, label="output type"))
    return tuple(result)


def _linked(value: object) -> tuple[str, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    node_id, output = value
    if (
        not isinstance(node_id, str)
        or not node_id
        or not isinstance(output, int)
        or isinstance(output, bool)
        or output < 0
    ):
        return None
    return node_id, output


def _declared_type(declaration: object) -> str | None:
    if isinstance(declaration, str):
        return declaration
    if isinstance(declaration, Mapping):
        value = declaration.get("type")
        return value if isinstance(value, str) else None
    if isinstance(declaration, Sequence) and not isinstance(declaration, (str, bytes, bytearray)):
        if not declaration:
            return None
        first = declaration[0]
        return first if isinstance(first, str) else "COMBO"
    return None


def _dynamic_contract(
    raw_node: Mapping[str, object],
) -> tuple[str, int, int, str] | None:
    declaration = _raw_input_declaration(raw_node, _DYNAMIC_INPUT)
    if not isinstance(declaration, Sequence) or isinstance(declaration, (str, bytes, bytearray)):
        return None
    if len(declaration) < 2 or declaration[0] != "COMFY_AUTOGROW_V3":
        return None
    metadata = declaration[1]
    if not isinstance(metadata, Mapping):
        return None
    template = metadata.get("template")
    if not isinstance(template, Mapping):
        return None
    prefix = metadata.get("prefix", template.get("prefix"))
    lower = metadata.get("min", template.get("min"))
    upper = metadata.get("max", template.get("max"))
    if (
        not isinstance(prefix, str)
        or not isinstance(lower, int)
        or isinstance(lower, bool)
        or not isinstance(upper, int)
        or isinstance(upper, bool)
        or not prefix
        or lower < 0
        or upper < lower
        or upper > 10_000
    ):
        return None
    template_input = _mapping(template.get("input"), label="autogrow template input")
    required = _mapping(template_input.get("required"), label="autogrow required input")
    nested = required.get("ref_image")
    nested_type = _declared_type(nested)
    if nested_type is None:
        return None
    return prefix, lower, upper, nested_type


def _normalized_choice(value: object) -> str:
    if not isinstance(value, str):
        return repr(value)
    return value.replace("\\", "/").casefold()


_MODEL_LOADER_COMBO_INPUTS: Final = {
    "UNETLoader": "unet_name",
    "CLIPLoader": "clip_name",
    "VAELoader": "vae_name",
}


def _resolve_host_combo_choice(
    *,
    class_type: str,
    input_name: str,
    value: object,
    definition: ComfyNodeDefinition,
) -> str:
    host_input = next((item for item in definition.inputs if item.name == input_name), None)
    if host_input is None or host_input.type_name != "COMBO" or not host_input.options:
        raise ComfyAdapterCompatibilityError(
            reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            action=f"provide a non-empty {class_type}.{input_name} combo for H3 host submission",
        )
    normalized_value = _normalized_choice(value)
    matches = tuple(
        option for option in host_input.options if _normalized_choice(option) == normalized_value
    )
    if len(matches) != 1:
        qualifier = "missing" if not matches else "ambiguous"
        raise ComfyAdapterCompatibilityError(
            reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            action=(
                f"resolve {qualifier} {class_type}.{input_name} host combo choice for {value!r}"
            ),
        )
    return matches[0]


def resolve_minimax_h3_host_workflow_prompt(
    workflow: MiniMaxH3HostWorkflow,
    object_info: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Return an API prompt with exact host spellings for H3 model-loader combos.

    The pure workflow contract uses slash-normalized, host-relative artifact names. ComfyUI's
    `/prompt` validator, however, compares combo values literally; Windows hosts commonly expose
    the same choices with backslashes. Resolve only the reviewed model-loader combos against the
    live `/object_info` choices and preserve every other input unchanged.
    """

    if not isinstance(workflow, MiniMaxH3HostWorkflow):
        raise ScheduleContractError("MiniMax H3 host prompt resolution requires a host workflow")
    if not isinstance(object_info, Mapping):
        raise ScheduleContractError("MiniMax H3 host object_info must be an object")

    required_classes = {
        str(node.get("class_type"))
        for node in workflow.prompt.values()
        if isinstance(node, Mapping) and isinstance(node.get("class_type"), str)
    }
    raw_for_normalization = {
        class_type: object_info[class_type]
        for class_type in required_classes & set(_MODEL_LOADER_COMBO_INPUTS)
        if class_type in object_info
    }
    normalized = {item.node_id: item for item in normalize_object_info(raw_for_normalization)}

    resolved: dict[str, dict[str, object]] = {
        node_id: dict(node)
        for node_id, node in workflow.prompt.items()
        if isinstance(node, Mapping)
    }
    for _node_id, node in resolved.items():
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or class_type not in _MODEL_LOADER_COMBO_INPUTS:
            continue
        input_name = _MODEL_LOADER_COMBO_INPUTS[class_type]
        inputs_value = node.get("inputs")
        if not isinstance(inputs_value, Mapping) or input_name not in inputs_value:
            raise ComfyAdapterCompatibilityError(
                reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                action=f"provide {class_type}.{input_name} for H3 host submission",
            )
        definition = normalized.get(class_type)
        if definition is None:
            raise ComfyAdapterCompatibilityError(
                reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                action=f"provide {class_type} object_info for H3 host submission",
            )
        resolved_inputs = dict(inputs_value)
        resolved_inputs[input_name] = _resolve_host_combo_choice(
            class_type=class_type,
            input_name=input_name,
            value=resolved_inputs[input_name],
            definition=definition,
        )
        node["inputs"] = resolved_inputs
    if len(resolved) != len(workflow.prompt):
        raise ScheduleContractError("MiniMax H3 host workflow contains a malformed node")
    return resolved


def _definition_output_types(
    node_id: str,
    raw_node: Mapping[str, object],
    normalized: Mapping[str, ComfyNodeDefinition],
) -> tuple[str, ...]:
    definition = normalized.get(node_id)
    if definition is not None:
        return tuple(item.type_name for item in definition.outputs)
    return _raw_output_types(raw_node)


def _append_input_issues(
    *,
    node_id: str,
    prompt_inputs: Mapping[str, object],
    raw_node: Mapping[str, object],
    definition: ComfyNodeDefinition,
    issues: list[MiniMaxH3SchemaIssue],
) -> None:
    host_inputs = {item.name: item for item in definition.inputs}
    dynamic_names = tuple(name for name in prompt_inputs if name.startswith(f"{_DYNAMIC_INPUT}."))
    dynamic_contract = _dynamic_contract(raw_node)
    if dynamic_names:
        dynamic_input = host_inputs.get(_DYNAMIC_INPUT)
        if dynamic_input is None:
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.MISSING_INPUT,
                    node_id=node_id,
                    input_name=_DYNAMIC_INPUT,
                    expected="COMFY_AUTOGROW_V3",
                    actual="missing",
                )
            )
        elif dynamic_input.type_name != "COMFY_AUTOGROW_V3":
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.INPUT_TYPE_MISMATCH,
                    node_id=node_id,
                    input_name=_DYNAMIC_INPUT,
                    expected="COMFY_AUTOGROW_V3",
                    actual=dynamic_input.type_name,
                )
            )
        if dynamic_contract is None:
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.DYNAMIC_INPUT_MALFORMED,
                    node_id=node_id,
                    input_name=_DYNAMIC_INPUT,
                    expected="COMFY_AUTOGROW_V3 ref_image_ IMAGE template",
                    actual="missing or malformed",
                )
            )
        else:
            prefix, lower, upper, nested_type = dynamic_contract
            if prefix != _DYNAMIC_PREFIX or nested_type != "IMAGE":
                issues.append(
                    MiniMaxH3SchemaIssue(
                        kind=MiniMaxH3SchemaIssueKind.DYNAMIC_INPUT_MALFORMED,
                        node_id=node_id,
                        input_name=_DYNAMIC_INPUT,
                        expected="prefix ref_image_ with IMAGE nested type",
                        actual=f"prefix {prefix!r}, type {nested_type!r}",
                    )
                )
            count = len(dynamic_names)
            if count < lower or count > upper:
                issues.append(
                    MiniMaxH3SchemaIssue(
                        kind=MiniMaxH3SchemaIssueKind.DYNAMIC_INPUT_LIMIT,
                        node_id=node_id,
                        input_name=_DYNAMIC_INPUT,
                        expected=f"count in [{lower}, {upper}]",
                        actual=str(count),
                    )
                )
            for name in dynamic_names:
                if not name.startswith(f"{_DYNAMIC_INPUT}.{prefix}"):
                    issues.append(
                        MiniMaxH3SchemaIssue(
                            kind=MiniMaxH3SchemaIssueKind.DYNAMIC_INPUT_MALFORMED,
                            node_id=node_id,
                            input_name=name,
                            expected=f"{_DYNAMIC_INPUT}.{prefix}<index>",
                            actual=name,
                        )
                    )
    for name, value in prompt_inputs.items():
        if name.startswith(f"{_DYNAMIC_INPUT}."):
            continue
        if name not in host_inputs:
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.MISSING_INPUT,
                    node_id=node_id,
                    input_name=name,
                    expected="host input",
                    actual="missing",
                )
            )
            continue
        host_input = host_inputs[name]
        expected_type = _EXPECTED_INPUT_TYPES.get(node_id, {}).get(name)
        if expected_type is not None and host_input.type_name != expected_type:
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.INPUT_TYPE_MISMATCH,
                    node_id=node_id,
                    input_name=name,
                    expected=expected_type,
                    actual=host_input.type_name,
                )
            )
        linked = _linked(value)
        if linked is not None:
            continue
        if host_input.type_name == "COMBO" and value not in host_input.options:
            normalized_value = _normalized_choice(value)
            if not any(_normalized_choice(item) == normalized_value for item in host_input.options):
                issues.append(
                    MiniMaxH3SchemaIssue(
                        kind=MiniMaxH3SchemaIssueKind.COMBO_VALUE_UNAVAILABLE,
                        node_id=node_id,
                        input_name=name,
                        expected="one of host combo choices",
                        actual=str(value),
                    )
                )
        elif host_input.type_name == "COMBO" and value in host_input.options:
            continue
        elif host_input.type_name == "COMBO":
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.COMBO_VALUE_UNAVAILABLE,
                    node_id=node_id,
                    input_name=name,
                    expected="one of host combo choices",
                    actual=str(value),
                )
            )

    for host_input in definition.inputs:
        if host_input.section == "required" and host_input.name not in prompt_inputs:
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.MISSING_INPUT,
                    node_id=node_id,
                    input_name=host_input.name,
                    expected=host_input.type_name,
                    actual="missing",
                )
            )


def _load_image_issue(
    *,
    raw_node: Mapping[str, object],
    node_id: str,
    prompt_inputs: Mapping[str, object],
) -> MiniMaxH3SchemaIssue | None:
    declaration = _raw_input_declaration(raw_node, "image")
    if declaration is None:
        return MiniMaxH3SchemaIssue(
            kind=MiniMaxH3SchemaIssueKind.MISSING_INPUT,
            node_id=node_id,
            input_name="image",
            expected="COMBO image input",
            actual="missing",
        )
    type_name = _declared_type(declaration)
    if type_name != "COMBO":
        return MiniMaxH3SchemaIssue(
            kind=MiniMaxH3SchemaIssueKind.INPUT_TYPE_MISMATCH,
            node_id=node_id,
            input_name="image",
            expected="COMBO",
            actual=str(type_name),
        )
    if "image" not in prompt_inputs:
        return MiniMaxH3SchemaIssue(
            kind=MiniMaxH3SchemaIssueKind.MISSING_INPUT,
            node_id=node_id,
            input_name="image",
            expected="COMFY image filename",
            actual="missing",
        )
    return None


def validate_minimax_h3_host_workflow_schema(
    workflow: MiniMaxH3HostWorkflow,
    object_info: Mapping[str, object],
) -> MiniMaxH3HostSchemaReport:
    """Validate a generated H3 graph against supplied host schemas without executing it."""

    if not isinstance(workflow, MiniMaxH3HostWorkflow):
        raise ScheduleContractError("MiniMax H3 schema preflight requires a host workflow")
    if not isinstance(object_info, Mapping):
        raise ScheduleContractError("MiniMax H3 host object_info must be an object")

    prompt = workflow.prompt
    class_types = {
        node_id: _text(node.get("class_type"), label="workflow class type")
        for node_id, node in prompt.items()
        if isinstance(node, Mapping)
    }
    if len(class_types) != len(prompt):
        raise ScheduleContractError("MiniMax H3 workflow prompt contains a malformed node")

    required_classes = tuple(sorted(set(class_types.values())))
    raw_for_normalization: dict[str, object] = {}
    for class_type in required_classes:
        raw_node = object_info.get(class_type)
        if raw_node is None:
            continue
        if class_type != _LOAD_IMAGE:
            raw_for_normalization[class_type] = raw_node
    normalized_tuple = normalize_object_info(raw_for_normalization)
    normalized = {item.node_id: item for item in normalized_tuple}
    issues: list[MiniMaxH3SchemaIssue] = []

    for node_id, class_type in class_types.items():
        raw_node_value = object_info.get(class_type)
        if raw_node_value is None:
            issues.append(
                MiniMaxH3SchemaIssue(
                    kind=MiniMaxH3SchemaIssueKind.MISSING_NODE,
                    node_id=node_id,
                    expected=class_type,
                    actual="missing",
                )
            )
            continue
        raw_node = _mapping(raw_node_value, label="host node schema")
        prompt_node = _mapping(prompt[node_id], label="workflow node")
        prompt_inputs = _mapping(prompt_node.get("inputs"), label="workflow node inputs")
        if class_type == _LOAD_IMAGE:
            issue = _load_image_issue(
                raw_node=raw_node,
                node_id=node_id,
                prompt_inputs=prompt_inputs,
            )
            if issue is not None:
                issues.append(issue)
        definition = normalized.get(class_type)
        if definition is None:
            if class_type != _LOAD_IMAGE:
                issues.append(
                    MiniMaxH3SchemaIssue(
                        kind=MiniMaxH3SchemaIssueKind.MISSING_NODE,
                        node_id=node_id,
                        expected=class_type,
                        actual="malformed",
                    )
                )
            continue
        _append_input_issues(
            node_id=node_id,
            prompt_inputs=prompt_inputs,
            raw_node=raw_node,
            definition=definition,
            issues=issues,
        )

    for node_id, _class_type in class_types.items():
        prompt_node = _mapping(prompt[node_id], label="workflow node")
        inputs = _mapping(prompt_node.get("inputs"), label="workflow node inputs")
        for input_name, value in inputs.items():
            link = _linked(value)
            if link is None:
                continue
            source_id, output_index = link
            source_class = class_types.get(source_id)
            if source_class is None:
                issues.append(
                    MiniMaxH3SchemaIssue(
                        kind=MiniMaxH3SchemaIssueKind.OUTPUT_MISMATCH,
                        node_id=node_id,
                        input_name=input_name,
                        expected="link to a generated node",
                        actual=source_id,
                    )
                )
                continue
            raw_source = _mapping(object_info.get(source_class), label="source node schema")
            outputs = _definition_output_types(source_class, raw_source, normalized)
            if output_index >= len(outputs):
                issues.append(
                    MiniMaxH3SchemaIssue(
                        kind=MiniMaxH3SchemaIssueKind.OUTPUT_MISMATCH,
                        node_id=node_id,
                        input_name=input_name,
                        expected=f"output index < {len(outputs)}",
                        actual=str(output_index),
                    )
                )

    checked_nodes = tuple(sorted(class_types))
    canonical_issues = tuple(sorted(issues, key=_issue_sort_key))
    return MiniMaxH3HostSchemaReport(
        variant=workflow.spec.variant,
        checked_nodes=checked_nodes,
        issues=canonical_issues,
        compatible=not canonical_issues,
    )


__all__ = [
    "MINIMAX_H3_HOST_SCHEMA_REPORT",
    "MiniMaxH3HostSchemaReport",
    "MiniMaxH3SchemaIssue",
    "MiniMaxH3SchemaIssueKind",
    "resolve_minimax_h3_host_workflow_prompt",
    "validate_minimax_h3_host_workflow_schema",
]
