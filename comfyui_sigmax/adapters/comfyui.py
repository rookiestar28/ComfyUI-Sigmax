"""Pure, capability-aware normalization of reviewed public ComfyUI surfaces."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, NoReturn, cast

from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles import (
    HostCapabilities,
    HostCapabilityEvidence,
    HostCapabilityLifecycle,
    Krea2Variant,
    ModelCapabilityEvidence,
    RegisteredProfile,
    model_identity_from_krea2_resolution,
    resolve_krea2_variant,
)

COMFYUI_ADAPTER_SCHEMA_ID: Final = "sigmax.comfyui-adapter/1"
COMFYUI_ADAPTER_SCHEMA_VERSION: Final = "1"

_NUMBERED_API_MODULE_PATTERN: Final = re.compile(r"^comfy_api\.v(\d+)_(\d+)_(\d+)$")
_API_VERSION_PATTERN: Final = re.compile(r"^\d+\.\d+\.\d+$")
_REVISION_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_PUBLIC_TEXT_LIMIT: Final = 256
_MAX_NODES: Final = 4096
_MAX_INPUTS: Final = 256
_MAX_OUTPUTS: Final = 64
_MAX_OPTIONS: Final = 512
_INPUT_SECTION_ORDER: Final = {"required": 0, "optional": 1, "hidden": 2}


class ComfyAdapterReason(str, Enum):
    """Stable adapter failure reason codes."""

    API_PUBLIC_SURFACE_MISSING = "api.public_surface_missing"
    API_MANIFEST_MALFORMED = "api.manifest_malformed"
    API_EXPERIMENTAL = "api.experimental"
    API_NOT_NUMBERED = "api.not_numbered"
    API_UNSUPPORTED = "api.unsupported"
    SYSTEM_STATS_MALFORMED = "host.system_stats_malformed"
    FEATURES_MALFORMED = "host.features_malformed"
    HOST_OUTSIDE_TESTED_WINDOW = "host.outside_tested_window"
    OBJECT_INFO_MALFORMED = "node.object_info_malformed"
    NODE_SCHEMA_MALFORMED = "node.schema_malformed"
    PAYLOAD_LIMIT_EXCEEDED = "payload.limit_exceeded"


class ComfyAdapterCompatibilityError(ScheduleContractError):
    """A public ComfyUI surface cannot satisfy the reviewed adapter contract."""

    def __init__(self, *, reason: ComfyAdapterReason, action: str) -> None:
        if not isinstance(reason, ComfyAdapterReason):
            raise ScheduleContractError("adapter error reason is unsupported")
        if not isinstance(action, str) or not action:
            raise ScheduleContractError("adapter error action must be non-empty")
        self.reason = reason
        self.action = action
        super().__init__(f"{reason.value}; action: {action}")


class ComfyApiLifecycle(str, Enum):
    """Stability of a reviewed numbered Comfy API surface."""

    LANDED = "landed"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class ComfyNodeSchemaForm(str, Enum):
    """Supported public node-definition representations."""

    OBJECT_INFO_V1 = "object_info_v1"
    NODE_DEFINITION_V2 = "node_definition_v2"


def _raise(reason: ComfyAdapterReason, action: str) -> NoReturn:
    raise ComfyAdapterCompatibilityError(reason=reason, action=action)


def _public_text(value: object, *, action: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _PUBLIC_TEXT_LIMIT
        or any(ord(character) < 32 for character in value)
    ):
        _raise(ComfyAdapterReason.NODE_SCHEMA_MALFORMED, action)
    return value


def _mapping(
    value: object,
    *,
    reason: ComfyAdapterReason,
    action: str,
) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        _raise(reason, action)
    return cast(Mapping[object, object], value)


def _plain_sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyApiProbe:
    """Public-symbol evidence from an already-loaded Comfy API module."""

    module_name: str
    api_version: str
    is_numbered: bool
    lifecycle: ComfyApiLifecycle
    public_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.module_name, str) or not self.module_name:
            raise ScheduleContractError("module_name must be non-empty")
        if not isinstance(self.api_version, str) or not self.api_version:
            raise ScheduleContractError("api_version must be non-empty")
        if not isinstance(self.is_numbered, bool):
            raise ScheduleContractError("is_numbered must be boolean")
        if not isinstance(self.lifecycle, ComfyApiLifecycle):
            raise ScheduleContractError("API lifecycle is unsupported")
        if self.public_symbols != ("ComfyExtension", "io", "ui"):
            raise ScheduleContractError("public API symbols are not canonical")
        if self.is_numbered != bool(_NUMBERED_API_MODULE_PATTERN.fullmatch(self.module_name)):
            raise ScheduleContractError("numbered API identity is inconsistent")


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyHostWindow:
    """Exact host range backed by reviewed source and contract fixtures."""

    minimum_version: str
    maximum_version: str
    tested_revisions: tuple[str, ...]
    validation_level: str

    def __post_init__(self) -> None:
        if not _API_VERSION_PATTERN.fullmatch(self.minimum_version):
            raise ScheduleContractError("minimum host version is malformed")
        if not _API_VERSION_PATTERN.fullmatch(self.maximum_version):
            raise ScheduleContractError("maximum host version is malformed")
        if not self.tested_revisions or any(
            not _REVISION_PATTERN.fullmatch(revision) for revision in self.tested_revisions
        ):
            raise ScheduleContractError("tested host revisions are malformed")
        if tuple(sorted(set(self.tested_revisions))) != self.tested_revisions:
            raise ScheduleContractError("tested host revisions must be canonical")
        if self.validation_level != "static_contract":
            raise ScheduleContractError("host validation level is unsupported")


COMFYUI_HOST_WINDOW: Final = ComfyHostWindow(
    minimum_version="0.29.0",
    maximum_version="0.29.0",
    tested_revisions=("e651b7bef55a5376343dcb1c0edb79f0142c985e",),  # pragma: allowlist secret
    validation_level="static_contract",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyNodeInput:
    """One canonical public node input."""

    name: str
    type_name: str
    section: str
    optional: bool
    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _public_text(self.name, action="provide a bounded public input name")
        _public_text(self.type_name, action="provide a bounded public input type")
        if self.section not in _INPUT_SECTION_ORDER:
            raise ScheduleContractError("input section is unsupported")
        if not isinstance(self.optional, bool):
            raise ScheduleContractError("input optional flag must be boolean")
        if not isinstance(self.options, tuple):
            raise ScheduleContractError("input options must be immutable")
        if len(self.options) > _MAX_OPTIONS:
            raise ScheduleContractError("input options exceed the limit")
        for option in self.options:
            _public_text(option, action="provide bounded public input options")
        if len(self.options) != len(set(self.options)):
            raise ScheduleContractError("input options contain duplicates")
        if bool(self.options) != (self.type_name == "COMBO"):
            raise ScheduleContractError("combo input options are inconsistent")


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyNodeOutput:
    """One canonical public node output."""

    index: int
    name: str
    type_name: str
    is_list: bool

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ScheduleContractError("output index is invalid")
        _public_text(self.name, action="provide a bounded public output name")
        _public_text(self.type_name, action="provide a bounded public output type")
        if not isinstance(self.is_list, bool):
            raise ScheduleContractError("output is_list must be boolean")


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyNodeDefinition:
    """Canonical node evidence shared by supported public representations."""

    node_id: str
    schema_form: ComfyNodeSchemaForm
    inputs: tuple[ComfyNodeInput, ...]
    outputs: tuple[ComfyNodeOutput, ...]
    deprecated: bool
    experimental: bool

    def __post_init__(self) -> None:
        _public_text(self.node_id, action="provide a bounded public node ID")
        if not isinstance(self.schema_form, ComfyNodeSchemaForm):
            raise ScheduleContractError("node schema form is unsupported")
        if not isinstance(self.inputs, tuple) or any(
            not isinstance(item, ComfyNodeInput) for item in self.inputs
        ):
            raise ScheduleContractError("node inputs must be canonical")
        if not isinstance(self.outputs, tuple) or any(
            not isinstance(item, ComfyNodeOutput) for item in self.outputs
        ):
            raise ScheduleContractError("node outputs must be canonical")
        if len(self.inputs) > _MAX_INPUTS or len(self.outputs) > _MAX_OUTPUTS:
            raise ScheduleContractError("node members exceed limits")
        input_names = tuple(item.name for item in self.inputs)
        if len(input_names) != len(set(input_names)):
            raise ScheduleContractError("node inputs contain duplicate names")
        expected_inputs = tuple(
            sorted(
                self.inputs,
                key=lambda item: (_INPUT_SECTION_ORDER[item.section], item.name),
            )
        )
        if self.inputs != expected_inputs:
            raise ScheduleContractError("node inputs must be canonical")
        if tuple(item.index for item in self.outputs) != tuple(range(len(self.outputs))):
            raise ScheduleContractError("node output indexes must be contiguous")
        if not isinstance(self.deprecated, bool) or not isinstance(self.experimental, bool):
            raise ScheduleContractError("node lifecycle flags must be boolean")


@dataclass(frozen=True, slots=True, kw_only=True)
class ComfyAdapterEvidence:
    """Complete immutable projection consumed by profile capability resolution."""

    schema_id: str
    schema_version: str
    api_probe: ComfyApiProbe
    host_window: ComfyHostWindow
    feature_ids: tuple[str, ...]
    nodes: tuple[ComfyNodeDefinition, ...]
    sampler_ids: tuple[str, ...]
    host_capabilities: HostCapabilities

    def __post_init__(self) -> None:
        if self.schema_id != COMFYUI_ADAPTER_SCHEMA_ID:
            raise ScheduleContractError("adapter schema_id is unsupported")
        if self.schema_version != COMFYUI_ADAPTER_SCHEMA_VERSION:
            raise ScheduleContractError("adapter schema_version is unsupported")
        if not isinstance(self.api_probe, ComfyApiProbe):
            raise ScheduleContractError("api_probe must be ComfyApiProbe")
        if not isinstance(self.host_window, ComfyHostWindow):
            raise ScheduleContractError("host_window must be ComfyHostWindow")
        if self.feature_ids != tuple(sorted(set(self.feature_ids))):
            raise ScheduleContractError("feature IDs must be canonical")
        if self.nodes != tuple(sorted(self.nodes, key=lambda item: item.node_id)):
            raise ScheduleContractError("nodes must be canonical")
        if self.sampler_ids != tuple(sorted(set(self.sampler_ids))):
            raise ScheduleContractError("sampler IDs must be canonical")
        if not isinstance(self.host_capabilities, HostCapabilities):
            raise ScheduleContractError("host_capabilities must be HostCapabilities")


def probe_comfy_api(module: object, *, module_name: str) -> ComfyApiProbe:
    """Probe reviewed public attributes without importing host-controlled module text."""

    if not isinstance(module_name, str):
        _raise(
            ComfyAdapterReason.API_MANIFEST_MALFORMED,
            "provide the public Comfy API module name",
        )
    numbered_match = _NUMBERED_API_MODULE_PATTERN.fullmatch(module_name)
    if numbered_match is None and module_name != "comfy_api.latest":
        _raise(
            ComfyAdapterReason.API_MANIFEST_MALFORMED,
            "use comfy_api.latest or a numbered comfy_api.vX_Y_Z module",
        )
    try:
        public_module = cast(Any, module)
        api = public_module.ComfyAPI
        version = api.VERSION
        stable = api.STABLE
        for symbol in ("ComfyExtension", "io", "ui"):
            getattr(module, symbol)
    except (AttributeError, TypeError) as exc:
        raise ComfyAdapterCompatibilityError(
            reason=ComfyAdapterReason.API_PUBLIC_SURFACE_MISSING,
            action="install a ComfyUI host exposing ComfyAPI, ComfyExtension, io, and ui",
        ) from exc
    if (
        not isinstance(version, str)
        or not isinstance(stable, bool)
        or (module_name == "comfy_api.latest" and version != "latest")
        or (
            numbered_match is not None
            and (
                not _API_VERSION_PATTERN.fullmatch(version)
                or version != ".".join(numbered_match.groups())
            )
        )
    ):
        _raise(
            ComfyAdapterReason.API_MANIFEST_MALFORMED,
            "use matching public ComfyAPI.VERSION and boolean ComfyAPI.STABLE values",
        )
    is_numbered = numbered_match is not None
    lifecycle = (
        ComfyApiLifecycle.LANDED if is_numbered and stable else ComfyApiLifecycle.EXPERIMENTAL
    )
    return ComfyApiProbe(
        module_name=module_name,
        api_version=version,
        is_numbered=is_numbered,
        lifecycle=lifecycle,
        public_symbols=("ComfyExtension", "io", "ui"),
    )


def require_stable_numbered_api(probe: ComfyApiProbe) -> ComfyApiProbe:
    """Require an explicitly stable numbered API before V3 registration."""

    if not isinstance(probe, ComfyApiProbe):
        raise ScheduleContractError("probe must be ComfyApiProbe")
    if not probe.is_numbered:
        _raise(
            ComfyAdapterReason.API_NOT_NUMBERED,
            "select an explicitly numbered Comfy API module",
        )
    if probe.lifecycle is ComfyApiLifecycle.EXPERIMENTAL:
        _raise(
            ComfyAdapterReason.API_EXPERIMENTAL,
            "wait for or select a stable numbered Comfy API before enabling V3 registration",
        )
    if probe.lifecycle is not ComfyApiLifecycle.LANDED:
        _raise(
            ComfyAdapterReason.API_UNSUPPORTED,
            "install a supported stable numbered Comfy API",
        )
    return probe


def _normalize_options(value: object) -> tuple[str, ...]:
    sequence = _plain_sequence(value)
    if sequence is None or not sequence or len(sequence) > _MAX_OPTIONS:
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "provide a bounded non-empty combo option list",
        )
    options = tuple(
        _public_text(option, action="provide bounded public combo options") for option in sequence
    )
    if len(options) != len(set(options)):
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "remove duplicate combo options",
        )
    return options


def _normalize_v1_input(name: object, value: object, *, section: str) -> ComfyNodeInput:
    input_name = _public_text(name, action="provide a bounded public input name")
    declaration = _plain_sequence(value)
    if declaration is None or not declaration:
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "provide a valid public object_info input declaration",
        )
    declared_type = declaration[0]
    if isinstance(declared_type, str):
        type_name = _public_text(
            declared_type,
            action="provide a bounded public input type",
        )
        options: tuple[str, ...] = ()
    else:
        type_name = "COMBO"
        options = _normalize_options(declared_type)
    return ComfyNodeInput(
        name=input_name,
        type_name=type_name,
        section=section,
        optional=section == "optional",
        options=options,
    )


def _normalize_v1_node(
    payload: object, *, expected_node_id: str | None = None
) -> ComfyNodeDefinition:
    data = _mapping(
        payload,
        reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
        action="provide a public object_info node mapping",
    )
    node_id = _public_text(data.get("name"), action="provide the public object_info node name")
    if expected_node_id is not None and node_id != expected_node_id:
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "make the object_info key match its public node name",
        )
    input_root = _mapping(
        data.get("input"),
        reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
        action="provide the public object_info input mapping",
    )
    inputs: list[ComfyNodeInput] = []
    for section in ("required", "optional", "hidden"):
        section_payload = input_root.get(section, {})
        section_mapping = _mapping(
            section_payload,
            reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            action="provide object_info input sections as mappings",
        )
        if len(section_mapping) > _MAX_INPUTS:
            _raise(
                ComfyAdapterReason.PAYLOAD_LIMIT_EXCEEDED,
                "reduce the node input count",
            )
        for name, declaration in section_mapping.items():
            inputs.append(_normalize_v1_input(name, declaration, section=section))
    if len(inputs) > _MAX_INPUTS:
        _raise(ComfyAdapterReason.PAYLOAD_LIMIT_EXCEEDED, "reduce the node input count")

    output_types = _plain_sequence(data.get("output"))
    if output_types is None or len(output_types) > _MAX_OUTPUTS:
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "provide a bounded public object_info output list",
        )
    output_names = _plain_sequence(data.get("output_name", output_types))
    output_lists = _plain_sequence(data.get("output_is_list", [False] * len(output_types)))
    if (
        output_names is None
        or output_lists is None
        or len(output_names) != len(output_types)
        or len(output_lists) != len(output_types)
    ):
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "align public object_info output fields",
        )
    outputs: list[ComfyNodeOutput] = []
    for index, (name, type_name, is_list) in enumerate(
        zip(output_names, output_types, output_lists, strict=True)
    ):
        if not isinstance(is_list, bool):
            _raise(
                ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                "use boolean object_info output list flags",
            )
        outputs.append(
            ComfyNodeOutput(
                index=index,
                name=_public_text(name, action="provide a bounded public output name"),
                type_name=_public_text(
                    type_name,
                    action="provide a bounded public output type",
                ),
                is_list=is_list,
            )
        )
    deprecated = data.get("deprecated", False)
    experimental = data.get("experimental", False)
    if not isinstance(deprecated, bool) or not isinstance(experimental, bool):
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "use boolean public node lifecycle flags",
        )
    return ComfyNodeDefinition(
        node_id=node_id,
        schema_form=ComfyNodeSchemaForm.OBJECT_INFO_V1,
        inputs=tuple(
            sorted(inputs, key=lambda item: (_INPUT_SECTION_ORDER[item.section], item.name))
        ),
        outputs=tuple(outputs),
        deprecated=deprecated,
        experimental=experimental,
    )


def normalize_node_definition_v2(payload: object) -> ComfyNodeDefinition:
    """Normalize one documented Node Definition JSON v2 object."""

    data = _mapping(
        payload,
        reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
        action="provide a Node Definition JSON v2 mapping",
    )
    required = {
        "inputs",
        "outputs",
        "name",
        "display_name",
        "description",
        "category",
        "output_node",
        "python_module",
    }
    if "input" in data or not required.issubset(data):
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "provide one complete Node Definition JSON v2 object",
        )
    node_id = _public_text(data.get("name"), action="provide the public v2 node name")
    input_mapping = _mapping(
        data.get("inputs"),
        reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
        action="provide the public v2 inputs mapping",
    )
    if len(input_mapping) > _MAX_INPUTS:
        _raise(ComfyAdapterReason.PAYLOAD_LIMIT_EXCEEDED, "reduce the v2 node input count")
    inputs: list[ComfyNodeInput] = []
    for key, raw_input in input_mapping.items():
        name = _public_text(key, action="provide a bounded public v2 input key")
        declaration = _mapping(
            raw_input,
            reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            action="provide public v2 input objects",
        )
        if declaration.get("name") != name:
            _raise(
                ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                "make each v2 input key match its public name",
            )
        type_name = _public_text(
            declaration.get("type"),
            action="provide a bounded public v2 input type",
        )
        optional = declaration.get("isOptional", False)
        if not isinstance(optional, bool):
            _raise(
                ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                "use boolean v2 input optional flags",
            )
        options: tuple[str, ...] = ()
        if type_name == "COMBO":
            options = _normalize_options(declaration.get("options"))
        inputs.append(
            ComfyNodeInput(
                name=name,
                type_name=type_name,
                section="optional" if optional else "required",
                optional=optional,
                options=options,
            )
        )
    output_sequence = _plain_sequence(data.get("outputs"))
    if output_sequence is None or len(output_sequence) > _MAX_OUTPUTS:
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "provide a bounded public v2 output list",
        )
    outputs: list[ComfyNodeOutput] = []
    for expected_index, raw_output in enumerate(output_sequence):
        output = _mapping(
            raw_output,
            reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            action="provide public v2 output objects",
        )
        if output.get("index") != expected_index or not isinstance(output.get("is_list"), bool):
            _raise(
                ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                "use contiguous v2 output indexes and boolean list flags",
            )
        outputs.append(
            ComfyNodeOutput(
                index=expected_index,
                name=_public_text(
                    output.get("name"),
                    action="provide a bounded public v2 output name",
                ),
                type_name=_public_text(
                    output.get("type"),
                    action="provide a bounded public v2 output type",
                ),
                is_list=output["is_list"],  # type: ignore[arg-type]
            )
        )
    deprecated = data.get("deprecated", False)
    experimental = data.get("experimental", False)
    if not isinstance(deprecated, bool) or not isinstance(experimental, bool):
        _raise(
            ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            "use boolean public v2 lifecycle flags",
        )
    return ComfyNodeDefinition(
        node_id=node_id,
        schema_form=ComfyNodeSchemaForm.NODE_DEFINITION_V2,
        inputs=tuple(
            sorted(inputs, key=lambda item: (_INPUT_SECTION_ORDER[item.section], item.name))
        ),
        outputs=tuple(outputs),
        deprecated=deprecated,
        experimental=experimental,
    )


def normalize_object_info(payload: object) -> tuple[ComfyNodeDefinition, ...]:
    """Normalize the public `/object_info` mapping for V1 and V3 host nodes."""

    data = _mapping(
        payload,
        reason=ComfyAdapterReason.OBJECT_INFO_MALFORMED,
        action="provide the public /object_info response mapping",
    )
    if len(data) > _MAX_NODES:
        _raise(ComfyAdapterReason.PAYLOAD_LIMIT_EXCEEDED, "reduce the object_info node count")
    nodes: list[ComfyNodeDefinition] = []
    for raw_node_id, raw_node in data.items():
        node_id = _public_text(
            raw_node_id,
            action="provide bounded public object_info node keys",
        )
        node_mapping = _mapping(
            raw_node,
            reason=ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
            action="provide public object_info node mappings",
        )
        if "input" in node_mapping and "inputs" not in node_mapping:
            node = _normalize_v1_node(node_mapping, expected_node_id=node_id)
        elif "inputs" in node_mapping and "input" not in node_mapping:
            node = normalize_node_definition_v2(node_mapping)
            if node.node_id != node_id:
                _raise(
                    ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                    "make the object_info key match its public node name",
                )
        else:
            _raise(
                ComfyAdapterReason.NODE_SCHEMA_MALFORMED,
                "provide one supported node schema form without mixing fields",
            )
        nodes.append(node)
    return tuple(sorted(nodes, key=lambda item: item.node_id))


def _node_input(node: ComfyNodeDefinition, name: str) -> ComfyNodeInput | None:
    return next((item for item in node.inputs if item.name == name), None)


def _node_lifecycle(node: ComfyNodeDefinition) -> HostCapabilityLifecycle:
    if node.deprecated:
        return HostCapabilityLifecycle.UNSUPPORTED
    if node.experimental:
        return HostCapabilityLifecycle.EXPERIMENTAL
    return HostCapabilityLifecycle.LANDED


def _best_lifecycle(nodes: Sequence[ComfyNodeDefinition]) -> HostCapabilityLifecycle:
    lifecycles = {_node_lifecycle(node) for node in nodes}
    if HostCapabilityLifecycle.LANDED in lifecycles:
        return HostCapabilityLifecycle.LANDED
    if HostCapabilityLifecycle.EXPERIMENTAL in lifecycles:
        return HostCapabilityLifecycle.EXPERIMENTAL
    return HostCapabilityLifecycle.UNSUPPORTED


def _feature_ids(features: object) -> tuple[str, ...]:
    data = _mapping(
        features,
        reason=ComfyAdapterReason.FEATURES_MALFORMED,
        action="provide the public /features response mapping",
    )
    if len(data) > _MAX_INPUTS:
        _raise(ComfyAdapterReason.PAYLOAD_LIMIT_EXCEEDED, "reduce the feature count")
    identifiers: list[str] = []
    for key in data:
        identifiers.append(_public_text(key, action="provide bounded public feature identifiers"))
    return tuple(sorted(set(identifiers)))


def _host_version(system_stats: object) -> str:
    root = _mapping(
        system_stats,
        reason=ComfyAdapterReason.SYSTEM_STATS_MALFORMED,
        action="provide the public /system_stats response mapping",
    )
    system = _mapping(
        root.get("system"),
        reason=ComfyAdapterReason.SYSTEM_STATS_MALFORMED,
        action="provide system.comfyui_version from /system_stats",
    )
    version = system.get("comfyui_version")
    if not isinstance(version, str) or not _API_VERSION_PATTERN.fullmatch(version):
        _raise(
            ComfyAdapterReason.SYSTEM_STATS_MALFORMED,
            "provide a semantic system.comfyui_version value",
        )
    return version


def adapt_comfyui_host(
    *,
    api_probe: ComfyApiProbe,
    system_stats: object,
    features: object,
    object_info: object,
    host_revision: str,
) -> ComfyAdapterEvidence:
    """Project reviewed public host evidence without network or framework imports."""

    if not isinstance(api_probe, ComfyApiProbe):
        raise ScheduleContractError("api_probe must be ComfyApiProbe")
    version = _host_version(system_stats)
    feature_ids = _feature_ids(features)
    if not isinstance(host_revision, str) or not _REVISION_PATTERN.fullmatch(host_revision):
        _raise(
            ComfyAdapterReason.HOST_OUTSIDE_TESTED_WINDOW,
            "provide a reviewed pinned lowercase host revision",
        )
    if (
        version < COMFYUI_HOST_WINDOW.minimum_version
        or version > COMFYUI_HOST_WINDOW.maximum_version
        or host_revision not in COMFYUI_HOST_WINDOW.tested_revisions
    ):
        _raise(
            ComfyAdapterReason.HOST_OUTSIDE_TESTED_WINDOW,
            "use a host inside the declared tested window or add reviewed compatibility evidence",
        )
    nodes = normalize_object_info(object_info)
    by_id = {node.node_id: node for node in nodes}

    external_candidates = tuple(
        node
        for node_id in ("SamplerCustom", "SamplerCustomAdvanced")
        if (node := by_id.get(node_id)) is not None
        and (input_value := _node_input(node, "sigmas")) is not None
        and input_value.type_name == "SIGMAS"
    )
    external_lifecycle = _best_lifecycle(external_candidates)

    partial_candidates: list[ComfyNodeDefinition] = []
    scheduler = by_id.get("BasicScheduler")
    if scheduler is not None and _node_input(scheduler, "denoise") is not None:
        partial_candidates.append(scheduler)
    for node_id, input_name in (
        ("SplitSigmas", "step"),
        ("SplitSigmasDenoise", "denoise"),
    ):
        candidate = by_id.get(node_id)
        if candidate is not None and _node_input(candidate, input_name) is not None:
            partial_candidates.append(candidate)
    partial_lifecycle = _best_lifecycle(partial_candidates)

    selector = by_id.get("KSamplerSelect")
    sampler_ids: tuple[str, ...] = ()
    sampler_lifecycle = HostCapabilityLifecycle.UNSUPPORTED
    if selector is not None:
        sampler_input = _node_input(selector, "sampler_name")
        if sampler_input is not None and "euler" in sampler_input.options:
            sampler_ids = ("comfy.euler",)
            sampler_lifecycle = _node_lifecycle(selector)

    api_lifecycle = {
        ComfyApiLifecycle.LANDED: HostCapabilityLifecycle.LANDED,
        ComfyApiLifecycle.EXPERIMENTAL: HostCapabilityLifecycle.EXPERIMENTAL,
        ComfyApiLifecycle.UNSUPPORTED: HostCapabilityLifecycle.UNSUPPORTED,
    }[api_probe.lifecycle]
    lifecycles = {
        "api.comfy.v3": api_lifecycle,
        "execution.partial_denoise": partial_lifecycle,
        "execution.per_token_timesteps": HostCapabilityLifecycle.UNSUPPORTED,
        "sampler.comfy.euler": sampler_lifecycle,
        "schedule.external_sigmas": external_lifecycle,
    }
    host_capabilities = HostCapabilities(
        evidence_version="1",
        host_id="comfyui",
        host_version=version,
        host_revision=host_revision,
        capabilities=tuple(
            HostCapabilityEvidence(capability_id=capability_id, lifecycle=lifecycle)
            for capability_id, lifecycle in sorted(lifecycles.items())
        ),
    )
    return ComfyAdapterEvidence(
        schema_id=COMFYUI_ADAPTER_SCHEMA_ID,
        schema_version=COMFYUI_ADAPTER_SCHEMA_VERSION,
        api_probe=api_probe,
        host_window=COMFYUI_HOST_WINDOW,
        feature_ids=feature_ids,
        nodes=nodes,
        sampler_ids=sampler_ids,
        host_capabilities=host_capabilities,
    )


def adapt_krea2_model_evidence(
    *,
    registered_profile: RegisteredProfile,
    explicit_variant: Krea2Variant | str | None = None,
    trusted_profile_id: str | None = None,
    trusted_framework_metadata: Mapping[str, object] | None = None,
    checkpoint_sha256: str | None = None,
    safetensors_metadata: Mapping[str, object] | None = None,
    tensor_keys: Sequence[str] = (),
    model_class: str | None = None,
    filename: str | None = None,
) -> ModelCapabilityEvidence:
    """Reuse the Krea 2 trust boundary while pairing identity with profile capabilities."""

    if not isinstance(registered_profile, RegisteredProfile):
        raise ScheduleContractError("registered_profile must be RegisteredProfile")
    resolution = resolve_krea2_variant(
        strict_official=False,
        explicit_variant=explicit_variant,
        trusted_profile_id=trusted_profile_id,
        trusted_framework_metadata=trusted_framework_metadata,
        checkpoint_sha256=checkpoint_sha256,
        safetensors_metadata=safetensors_metadata,
        tensor_keys=tensor_keys,
        model_class=model_class,
        filename=filename,
    )
    return ModelCapabilityEvidence(
        evidence_version="1",
        identity=model_identity_from_krea2_resolution(resolution),
        capabilities=registered_profile.schema.model_capabilities,
    )
