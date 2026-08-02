"""Namespaced, collision-safe ComfyUI node registration and schema discovery."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NoReturn, cast

from comfyui_sigmax.adapters.comfyui import (
    ComfyAdapterCompatibilityError,
    ComfyApiProbe,
    ComfyNodeDefinition,
    ComfyNodeInput,
    normalize_node_definition_v2,
    normalize_object_info,
    require_stable_numbered_api,
)
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles import HostCapabilityLifecycle

NODE_REGISTRATION_SCHEMA_ID: Final = "sigmax.node-registration/1"
NODE_REGISTRATION_SCHEMA_VERSION: Final = "1"
SIGMAX_NODE_MODULE: Final = "comfyui_sigmax.nodes"

_NODE_ID_PATTERN: Final = re.compile(r"^Sigmax\.[A-Z][A-Za-z0-9]{0,63}$")
_PUBLIC_TEXT_LIMIT: Final = 512
_WIRE_SIZE_LIMIT: Final = 1_048_576


class NodeDefinitionKind(str, Enum):
    """Reviewed source form for one registered node."""

    LEGACY_V1 = "legacy_v1"
    COMFY_V3 = "comfy_v3"
    NODE_DEFINITION_V2 = "node_definition_v2"


class NodeRegistrationReason(str, Enum):
    """Stable fail-closed registration reason codes."""

    INVALID_NODE_ID = "registration.invalid_node_id"
    INVALID_NODE_CLASS = "registration.invalid_node_class"
    SCHEMA_MALFORMED = "registration.schema_malformed"
    SCHEMA_ID_MISMATCH = "registration.schema_id_mismatch"
    SCHEMA_LIFECYCLE_MISMATCH = "registration.schema_lifecycle_mismatch"
    NODE_ID_COLLISION = "registration.node_id_collision"
    V3_API_NOT_STABLE = "registration.v3_api_not_stable"


class NodeRegistrationError(ScheduleContractError):
    """A node declaration cannot enter the public Sigmax mapping."""

    def __init__(self, *, reason: NodeRegistrationReason, action: str) -> None:
        if not isinstance(reason, NodeRegistrationReason):
            raise ScheduleContractError("registration error reason is unsupported")
        if not isinstance(action, str) or not action:
            raise ScheduleContractError("registration error action must be non-empty")
        self.reason = reason
        self.action = action
        super().__init__(f"{reason.value}; action: {action}")


def _fail(reason: NodeRegistrationReason, action: str) -> NoReturn:
    raise NodeRegistrationError(reason=reason, action=action)


def _public_text(value: object, *, action: str, allow_empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not value and not allow_empty)
        or len(value) > _PUBLIC_TEXT_LIMIT
        or any(ord(character) < 32 for character in value)
    ):
        _fail(NodeRegistrationReason.SCHEMA_MALFORMED, action)
    return value


def _require_node_id(value: object) -> str:
    if not isinstance(value, str) or not _NODE_ID_PATTERN.fullmatch(value):
        _fail(
            NodeRegistrationReason.INVALID_NODE_ID,
            "use an explicit stable ID matching Sigmax.<PascalCaseName>",
        )
    return value


def _canonical_payload(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise NodeRegistrationError(
            reason=NodeRegistrationReason.SCHEMA_MALFORMED,
            action="use only bounded JSON-safe public schema values",
        ) from exc
    if len(encoded.encode("utf-8")) > _WIRE_SIZE_LIMIT:
        _fail(
            NodeRegistrationReason.SCHEMA_MALFORMED,
            "reduce the public node-schema payload size",
        )
    return encoded


def _payload_mapping(payload_json: str) -> dict[str, Any]:
    loaded = json.loads(payload_json)
    if not isinstance(loaded, dict):
        raise ScheduleContractError("canonical node payload is not an object")
    return cast(dict[str, Any], loaded)


def _lifecycle(definition: ComfyNodeDefinition) -> HostCapabilityLifecycle:
    if definition.deprecated:
        return HostCapabilityLifecycle.UNSUPPORTED
    if definition.experimental:
        return HostCapabilityLifecycle.EXPERIMENTAL
    return HostCapabilityLifecycle.LANDED


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeRegistration:
    """One immutable node class plus its reviewed public schema evidence."""

    schema_id: str
    schema_version: str
    node_id: str
    display_name: str
    description: str
    category: str
    python_module: str
    output_node: bool
    node_class: type
    definition_kind: NodeDefinitionKind
    definition: ComfyNodeDefinition
    lifecycle: HostCapabilityLifecycle
    source_payload_json: str

    def __post_init__(self) -> None:
        if self.schema_id != NODE_REGISTRATION_SCHEMA_ID:
            raise ScheduleContractError("registration schema_id is unsupported")
        if self.schema_version != NODE_REGISTRATION_SCHEMA_VERSION:
            raise ScheduleContractError("registration schema_version is unsupported")
        _require_node_id(self.node_id)
        _public_text(self.display_name, action="provide a bounded public display name")
        _public_text(
            self.description,
            action="provide a bounded public description",
            allow_empty=True,
        )
        category = _public_text(self.category, action="provide a bounded public category")
        if not category.startswith("Sigmax"):
            raise ScheduleContractError("registration category must be namespaced under Sigmax")
        if self.python_module != SIGMAX_NODE_MODULE:
            raise ScheduleContractError("registration python_module is not canonical")
        if not isinstance(self.output_node, bool):
            raise ScheduleContractError("registration output_node must be boolean")
        if not isinstance(self.node_class, type):
            raise ScheduleContractError("registration node_class must be a class")
        if not isinstance(self.definition_kind, NodeDefinitionKind):
            raise ScheduleContractError("registration definition_kind is unsupported")
        if not isinstance(self.definition, ComfyNodeDefinition):
            raise ScheduleContractError("registration definition is unsupported")
        if self.definition.node_id != self.node_id:
            raise ScheduleContractError("registration node ID disagrees with its definition")
        if not isinstance(self.lifecycle, HostCapabilityLifecycle):
            raise ScheduleContractError("registration lifecycle is unsupported")
        if self.lifecycle is not _lifecycle(self.definition):
            raise ScheduleContractError("registration lifecycle disagrees with its definition")
        if not isinstance(self.source_payload_json, str):
            raise ScheduleContractError("registration payload must be canonical JSON text")
        payload = _payload_mapping(self.source_payload_json)
        if _canonical_payload(payload) != self.source_payload_json:
            raise ScheduleContractError("registration payload is not canonical JSON")


@dataclass(frozen=True, slots=True, kw_only=True)
class NodeRegistry:
    """Copy-on-write canonical node registry with collision rejection."""

    entries: tuple[NodeRegistration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple) or any(
            not isinstance(item, NodeRegistration) for item in self.entries
        ):
            raise ScheduleContractError("registry entries must be immutable registrations")
        node_ids = tuple(item.node_id for item in self.entries)
        if len(node_ids) != len(set(node_ids)):
            raise ScheduleContractError("registry entries contain duplicate node IDs")
        if node_ids != tuple(sorted(node_ids)):
            raise ScheduleContractError("registry entries must use canonical node-ID order")

    @classmethod
    def empty(cls) -> NodeRegistry:
        return cls(entries=())

    def register(self, registration: NodeRegistration) -> NodeRegistry:
        if not isinstance(registration, NodeRegistration):
            raise ScheduleContractError("registration must be NodeRegistration")
        existing = next(
            (item for item in self.entries if item.node_id == registration.node_id),
            None,
        )
        if existing is not None:
            if existing == registration:
                return self
            _fail(
                NodeRegistrationReason.NODE_ID_COLLISION,
                "choose a new namespaced node ID; existing registrations are never overwritten",
            )
        return NodeRegistry(
            entries=tuple(sorted((*self.entries, registration), key=lambda item: item.node_id))
        )

    def register_many(self, registrations: tuple[NodeRegistration, ...]) -> NodeRegistry:
        if not isinstance(registrations, tuple):
            raise ScheduleContractError("registrations must be an immutable tuple")
        result = self
        for registration in registrations:
            result = result.register(registration)
        return result

    def class_mappings(self) -> dict[str, type]:
        return {item.node_id: item.node_class for item in self.entries}

    def display_name_mappings(self) -> dict[str, str]:
        return {item.node_id: item.display_name for item in self.entries}

    def object_info_projection(self) -> dict[str, dict[str, Any]]:
        return {item.node_id: _object_info_projection(item) for item in self.entries}

    def node_definition_v2_projection(self) -> dict[str, dict[str, Any]]:
        return {item.node_id: _node_definition_v2_projection(item) for item in self.entries}


def _registration(
    *,
    node_class: type,
    definition_kind: NodeDefinitionKind,
    definition: ComfyNodeDefinition,
    display_name: object,
    description: object,
    category: object,
    output_node: object,
    source_payload: Mapping[str, object],
) -> NodeRegistration:
    if not isinstance(output_node, bool):
        _fail(
            NodeRegistrationReason.SCHEMA_MALFORMED,
            "use a boolean public output-node flag",
        )
    return NodeRegistration(
        schema_id=NODE_REGISTRATION_SCHEMA_ID,
        schema_version=NODE_REGISTRATION_SCHEMA_VERSION,
        node_id=_require_node_id(definition.node_id),
        display_name=_public_text(
            display_name,
            action="provide a bounded public display name",
        ),
        description=_public_text(
            description,
            action="provide a bounded public description",
            allow_empty=True,
        ),
        category=_public_text(category, action="provide a bounded public category"),
        python_module=SIGMAX_NODE_MODULE,
        output_node=output_node,
        node_class=node_class,
        definition_kind=definition_kind,
        definition=definition,
        lifecycle=_lifecycle(definition),
        source_payload_json=_canonical_payload(source_payload),
    )


def _class_attribute(node_class: type, name: str, default: object) -> object:
    return getattr(node_class, name, default)


def discover_legacy_registration(
    *,
    node_id: str,
    display_name: str,
    node_class: type,
) -> NodeRegistration:
    """Discover one trusted V1 class through the reviewed public host fields."""

    stable_id = _require_node_id(node_id)
    if not isinstance(node_class, type):
        _fail(
            NodeRegistrationReason.INVALID_NODE_CLASS,
            "provide a V1 node class",
        )
    input_types = getattr(node_class, "INPUT_TYPES", None)
    return_types = getattr(node_class, "RETURN_TYPES", None)
    if (
        not callable(input_types)
        or not isinstance(return_types, Sequence)
        or isinstance(return_types, (str, bytes, bytearray))
    ):
        _fail(
            NodeRegistrationReason.INVALID_NODE_CLASS,
            "provide callable INPUT_TYPES and an immutable RETURN_TYPES sequence",
        )
    try:
        inputs = input_types()
    except Exception as exc:
        raise NodeRegistrationError(
            reason=NodeRegistrationReason.SCHEMA_MALFORMED,
            action="make the public V1 INPUT_TYPES declaration deterministic",
        ) from exc
    output_types = list(return_types)
    output_names = list(
        cast(Sequence[object], _class_attribute(node_class, "RETURN_NAMES", return_types))
    )
    output_is_list = list(
        cast(
            Sequence[object],
            _class_attribute(node_class, "OUTPUT_IS_LIST", [False] * len(output_types)),
        )
    )
    payload: dict[str, object] = {
        "input": inputs,
        "output": output_types,
        "output_is_list": output_is_list,
        "output_name": output_names,
        "name": stable_id,
        "display_name": display_name,
        "description": _class_attribute(node_class, "DESCRIPTION", ""),
        "python_module": SIGMAX_NODE_MODULE,
        "category": _class_attribute(node_class, "CATEGORY", "Sigmax"),
        "output_node": _class_attribute(node_class, "OUTPUT_NODE", False),
        "deprecated": _class_attribute(node_class, "DEPRECATED", False),
        "experimental": _class_attribute(node_class, "EXPERIMENTAL", False),
    }
    try:
        definition = normalize_object_info({stable_id: payload})[0]
    except (ComfyAdapterCompatibilityError, ScheduleContractError, TypeError) as exc:
        raise NodeRegistrationError(
            reason=NodeRegistrationReason.SCHEMA_MALFORMED,
            action="fix the public V1 node schema",
        ) from exc
    return _registration(
        node_class=node_class,
        definition_kind=NodeDefinitionKind.LEGACY_V1,
        definition=definition,
        display_name=display_name,
        description=payload["description"],
        category=payload["category"],
        output_node=payload["output_node"],
        source_payload=payload,
    )


def discover_v3_registration(node_class: type) -> NodeRegistration:
    """Discover one trusted V3 class through public numbered-API methods."""

    if not isinstance(node_class, type):
        _fail(
            NodeRegistrationReason.INVALID_NODE_CLASS,
            "provide a V3 node class",
        )
    get_schema = getattr(node_class, "GET_SCHEMA", None)
    get_node_info = getattr(node_class, "GET_NODE_INFO_V1", None)
    if not callable(get_schema) or not callable(get_node_info):
        _fail(
            NodeRegistrationReason.INVALID_NODE_CLASS,
            "provide public GET_SCHEMA and GET_NODE_INFO_V1 methods",
        )
    try:
        schema = get_schema()
        raw_info = get_node_info()
        schema_id = _require_node_id(cast(Any, schema).node_id)
        if not isinstance(raw_info, Mapping):
            _fail(
                NodeRegistrationReason.SCHEMA_MALFORMED,
                "return a public mapping from GET_NODE_INFO_V1",
            )
        info = dict(raw_info)
        if info.get("name") != schema_id:
            _fail(
                NodeRegistrationReason.SCHEMA_ID_MISMATCH,
                "make GET_SCHEMA and GET_NODE_INFO_V1 use the same namespaced node ID",
            )
        info["python_module"] = SIGMAX_NODE_MODULE
        definition = normalize_object_info({schema_id: info})[0]
    except NodeRegistrationError:
        raise
    except (AttributeError, TypeError, ValueError, ComfyAdapterCompatibilityError) as exc:
        raise NodeRegistrationError(
            reason=NodeRegistrationReason.SCHEMA_MALFORMED,
            action="fix the public V3 schema and compatibility projection",
        ) from exc
    schema_experimental = getattr(schema, "is_experimental", False)
    schema_deprecated = getattr(schema, "is_deprecated", False)
    if (
        not isinstance(schema_experimental, bool)
        or not isinstance(schema_deprecated, bool)
        or definition.experimental != schema_experimental
        or definition.deprecated != schema_deprecated
    ):
        _fail(
            NodeRegistrationReason.SCHEMA_LIFECYCLE_MISMATCH,
            "make V3 lifecycle flags agree with the /object_info compatibility projection",
        )
    display_name = getattr(schema, "display_name", None) or schema_id
    description = getattr(schema, "description", "")
    category = getattr(schema, "category", "Sigmax")
    output_node = getattr(schema, "is_output_node", False)
    info.update(
        {
            "display_name": display_name,
            "description": description,
            "category": category,
            "output_node": output_node,
            "deprecated": schema_deprecated,
            "experimental": schema_experimental,
        }
    )
    return _registration(
        node_class=node_class,
        definition_kind=NodeDefinitionKind.COMFY_V3,
        definition=definition,
        display_name=display_name,
        description=description,
        category=category,
        output_node=output_node,
        source_payload=info,
    )


def registration_from_node_definition_v2(
    *,
    node_class: type,
    payload: object,
) -> NodeRegistration:
    """Validate one documented Node Definition JSON v2 registration."""

    if not isinstance(node_class, type):
        _fail(
            NodeRegistrationReason.INVALID_NODE_CLASS,
            "provide a node class for the v2 schema",
        )
    try:
        definition = normalize_node_definition_v2(payload)
        data = cast(Mapping[object, object], payload)
        _require_node_id(definition.node_id)
        if data.get("python_module") != SIGMAX_NODE_MODULE:
            _fail(
                NodeRegistrationReason.SCHEMA_MALFORMED,
                "use the canonical public Sigmax node module",
            )
    except NodeRegistrationError:
        raise
    except (ComfyAdapterCompatibilityError, ScheduleContractError, TypeError) as exc:
        raise NodeRegistrationError(
            reason=NodeRegistrationReason.SCHEMA_MALFORMED,
            action="fix the documented Node Definition JSON v2 schema",
        ) from exc
    return _registration(
        node_class=node_class,
        definition_kind=NodeDefinitionKind.NODE_DEFINITION_V2,
        definition=definition,
        display_name=data.get("display_name"),
        description=data.get("description"),
        category=data.get("category"),
        output_node=data.get("output_node"),
        source_payload=cast(Mapping[str, object], data),
    )


def _input_declaration(input_value: ComfyNodeInput) -> list[object]:
    if input_value.options:
        return [list(input_value.options)]
    return [input_value.type_name]


def _object_info_projection(registration: NodeRegistration) -> dict[str, Any]:
    if registration.definition_kind in {
        NodeDefinitionKind.LEGACY_V1,
        NodeDefinitionKind.COMFY_V3,
    }:
        return _payload_mapping(registration.source_payload_json)
    sections: dict[str, dict[str, object]] = {}
    for input_value in registration.definition.inputs:
        sections.setdefault(input_value.section, {})[input_value.name] = _input_declaration(
            input_value
        )
    return {
        "input": sections,
        "output": [item.type_name for item in registration.definition.outputs],
        "output_is_list": [item.is_list for item in registration.definition.outputs],
        "output_name": [item.name for item in registration.definition.outputs],
        "name": registration.node_id,
        "display_name": registration.display_name,
        "description": registration.description,
        "python_module": registration.python_module,
        "category": registration.category,
        "output_node": registration.output_node,
        "deprecated": registration.definition.deprecated,
        "experimental": registration.definition.experimental,
    }


def _v2_input(input_value: ComfyNodeInput) -> dict[str, object]:
    result: dict[str, object] = {
        "name": input_value.name,
        "type": input_value.type_name,
        "isOptional": input_value.optional,
    }
    if input_value.options:
        result["options"] = list(input_value.options)
    return result


def _node_definition_v2_projection(registration: NodeRegistration) -> dict[str, Any]:
    if registration.definition_kind is NodeDefinitionKind.NODE_DEFINITION_V2:
        return _payload_mapping(registration.source_payload_json)
    return {
        "inputs": {
            item.name: _v2_input(item)
            for item in registration.definition.inputs
            if item.section != "hidden"
        },
        "outputs": [
            {
                "index": item.index,
                "name": item.name,
                "type": item.type_name,
                "is_list": item.is_list,
            }
            for item in registration.definition.outputs
        ],
        "name": registration.node_id,
        "display_name": registration.display_name,
        "description": registration.description,
        "category": registration.category,
        "output_node": registration.output_node,
        "python_module": registration.python_module,
        "deprecated": registration.definition.deprecated,
        "experimental": registration.definition.experimental,
    }


def require_registration_compatible(
    registration: NodeRegistration,
    api_probe: ComfyApiProbe,
) -> NodeRegistration:
    """Reject activation of V3 definitions on an unstable numbered API."""

    if not isinstance(registration, NodeRegistration):
        raise ScheduleContractError("registration must be NodeRegistration")
    if not isinstance(api_probe, ComfyApiProbe):
        raise ScheduleContractError("api_probe must be ComfyApiProbe")
    if registration.definition_kind is not NodeDefinitionKind.COMFY_V3:
        return registration
    try:
        require_stable_numbered_api(api_probe)
    except ComfyAdapterCompatibilityError as exc:
        raise NodeRegistrationError(
            reason=NodeRegistrationReason.V3_API_NOT_STABLE,
            action="select a stable numbered Comfy API before activating V3 nodes",
        ) from exc
    return registration


def builtin_node_registry() -> NodeRegistry:
    """Return the validated built-in product-node catalog."""

    # IMPORTANT: the reviewed loader ignores comfy_entrypoint when mappings are non-None.
    # Mixed V1/V3 nodes must therefore share this one validated mapping projection.
    from comfyui_sigmax.nodes import (
        AdvancedFlowMatchScheduler,
        AuraFlowSigmaScheduler,
        CheckpointEvidenceInspector,
        Flux1SchnellSigmaScheduler,
        Krea2ConditioningRebalance,
        Krea2SigmaScheduler,
        Lumina2SigmaScheduler,
        ModelAwareSigmaScheduler,
        ProfileInspector,
        QwenImageSigmaScheduler,
        RawWorkflowOutput,
        ScheduleComparison,
        ScheduleConcatenate,
        ScheduleInspector,
        ScheduleResample,
        ScheduleSlice,
        SD3SigmaScheduler,
        TurboWorkflowOutput,
        ZImageSigmaScheduler,
    )

    return NodeRegistry.empty().register_many(
        (
            discover_legacy_registration(
                node_id="Sigmax.AdvancedFlowMatchScheduler",
                display_name="Advanced FlowMatch Scheduler",
                node_class=AdvancedFlowMatchScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.AuraFlowSigmaScheduler",
                display_name="AuraFlow Sigma Scheduler",
                node_class=AuraFlowSigmaScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.CheckpointEvidenceInspector",
                display_name="Checkpoint Evidence Inspector",
                node_class=CheckpointEvidenceInspector,
            ),
            discover_legacy_registration(
                node_id="Sigmax.Flux1SchnellSigmaScheduler",
                display_name="FLUX.1-schnell Sigma Scheduler",
                node_class=Flux1SchnellSigmaScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.Krea2ConditioningRebalance",
                display_name="Krea 2 Conditioning Rebalance",
                node_class=Krea2ConditioningRebalance,
            ),
            discover_legacy_registration(
                node_id="Sigmax.Krea2SigmaScheduler",
                display_name="Krea 2 Sigma Scheduler",
                node_class=Krea2SigmaScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.Lumina2SigmaScheduler",
                display_name="Lumina-Image 2.0 Sigma Scheduler",
                node_class=Lumina2SigmaScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.QwenImageSigmaScheduler",
                display_name="Qwen Image Sigma Scheduler",
                node_class=QwenImageSigmaScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.SD3SigmaScheduler",
                display_name="Stable Diffusion 3 Sigma Scheduler",
                node_class=SD3SigmaScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ModelAwareSigmaScheduler",
                display_name="Model-Aware Sigma Scheduler",
                node_class=ModelAwareSigmaScheduler,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ProfileInspector",
                display_name="Profile Inspector",
                node_class=ProfileInspector,
            ),
            discover_legacy_registration(
                node_id="Sigmax.RawWorkflowOutput",
                display_name="RAW Workflow Output",
                node_class=RawWorkflowOutput,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ScheduleComparison",
                display_name="Schedule Comparison",
                node_class=ScheduleComparison,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ScheduleConcatenate",
                display_name="Schedule Concatenate",
                node_class=ScheduleConcatenate,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ScheduleInspector",
                display_name="Schedule Inspector",
                node_class=ScheduleInspector,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ScheduleResample",
                display_name="Schedule Resample",
                node_class=ScheduleResample,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ScheduleSlice",
                display_name="Schedule Slice",
                node_class=ScheduleSlice,
            ),
            discover_legacy_registration(
                node_id="Sigmax.TurboWorkflowOutput",
                display_name="Turbo Workflow Output",
                node_class=TurboWorkflowOutput,
            ),
            discover_legacy_registration(
                node_id="Sigmax.ZImageSigmaScheduler",
                display_name="Z-Image Sigma Scheduler",
                node_class=ZImageSigmaScheduler,
            ),
        )
    )
