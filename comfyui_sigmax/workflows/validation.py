"""Deterministic compatibility validation for portable ComfyUI workflows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from typing import Any, Final, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from comfyui_sigmax.adapters.comfyui import (
    ComfyAdapterCompatibilityError,
    ComfyNodeDefinition,
    normalize_object_info,
)
from comfyui_sigmax.adapters.registration import SIGMAX_NODE_MODULE
from comfyui_sigmax.core.schedule_contracts import ScheduleContractError
from comfyui_sigmax.core.workflow_metadata import (
    WorkflowHostRequirement,
    WorkflowRequirement,
    extract_workflow_metadata,
)

WORKFLOW_FIXTURE_BUNDLE_SCHEMA_ID: Final = "sigmax.workflow-fixture-bundle/1"
HOST_BASELINE_SCHEMA_ID: Final = "sigmax.workflow-host-baseline/1"
WORKFLOW_VALIDATION_REPORT_SCHEMA_ID: Final = "sigmax.workflow-validation-report/1"
WORKFLOW_VALIDATION_REPORT_ENVELOPE_SCHEMA_ID: Final = (
    "sigmax.workflow-validation-report-envelope/1"
)
CANONICAL_HOST_VERSION: Final = "0.29.0"
CANONICAL_HOST_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"

_MAX_LIVE_BYTES: Final = 2_000_000
_MAX_TEXT: Final = 256
_WIDGET_TYPES: Final = frozenset({"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"})
_SUPPORTED_PYTHON_MODULES: Final = frozenset(
    {
        SIGMAX_NODE_MODULE,
        "custom_nodes.ComfyUI-Sigmax",
    }
)


class WorkflowScanMode(str, Enum):
    """Origin of the host node schema used by a scan."""

    PINNED_STATIC = "pinned_static"
    LIVE_OBJECT_INFO = "live_object_info"


class WorkflowValidationLane(str, Enum):
    """Blocking policy for one compatibility scan."""

    KNOWN_GOOD = "known_good"
    LATEST_HOST = "latest_host"


class WorkflowIssueSeverity(str, Enum):
    """Stable machine-readable validation severity."""

    ERROR = "error"
    WARNING = "warning"


class WorkflowIssueKind(str, Enum):
    """Stable compatibility issue taxonomy."""

    MISSING_NODE = "missing_node"
    MISSING_INPUT = "missing_input"
    WIDGET_SLOT_DRIFT = "widget_slot_drift"
    INPUT_TYPE_DRIFT = "input_type_drift"
    INVALID_FIXED_COMBO_VALUE = "invalid_fixed_combo_value"
    DEPRECATED_NODE = "deprecated_node"
    EXPERIMENTAL_NODE = "experimental_node"
    NORMALIZED_DIRECTORY_FAILURE = "normalized_directory_failure"
    MALFORMED_METADATA = "malformed_metadata"
    WORKFLOW_SCHEMA_MALFORMED = "workflow_schema_malformed"
    HOST_SCHEMA_MALFORMED = "host_schema_malformed"


class WorkflowLiveLoadReason(str, Enum):
    """Stable rejection reasons for loopback `/object_info` loading."""

    URL_NOT_LOOPBACK = "url_not_loopback"
    HTTP_ERROR = "http_error"
    RESPONSE_TOO_LARGE = "response_too_large"
    RESPONSE_NOT_JSON = "response_not_json"
    PAYLOAD_NOT_OBJECT = "payload_not_object"


class WorkflowLiveLoadError(ScheduleContractError):
    """Bounded live-loader failure without response-body or URL leakage."""

    reason: WorkflowLiveLoadReason

    def __init__(self, *, reason: WorkflowLiveLoadReason, action: str) -> None:
        self.reason = reason
        super().__init__(f"{reason.value}: {action}")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT:
        raise ScheduleContractError(f"{label} must be bounded public text")
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ScheduleContractError(f"{label} contains unsupported control characters")
    return value


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ScheduleContractError(f"{label} must be an object")
    return dict(cast(Mapping[str, object], value))


def _sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ScheduleContractError(f"{label} must be an array")
    return value


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleContractError("workflow validation data is not canonical JSON") from exc


def _display(value: object) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError):
        return "<unsupported>"
    return rendered[:_MAX_TEXT]


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowWidgetSlot:
    """One positional saved-widget contract."""

    name: str
    type_name: str
    fixed_value: object | None = None

    def __post_init__(self) -> None:
        _text(self.name, label="widget name")
        if self.type_name not in _WIDGET_TYPES:
            raise ScheduleContractError("widget type is unsupported")
        if self.fixed_value is not None and self.type_name != "COMBO":
            raise ScheduleContractError("only combo widgets may declare a fixed value")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowNodeContract:
    """One saved node and its ordered widget slots."""

    node_id: int
    node_type: str
    widget_slots: tuple[WorkflowWidgetSlot, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, int) or isinstance(self.node_id, bool) or self.node_id < 0:
            raise ScheduleContractError("workflow contract node ID is invalid")
        _text(self.node_type, label="workflow contract node type")
        if not isinstance(self.widget_slots, tuple) or any(
            not isinstance(item, WorkflowWidgetSlot) for item in self.widget_slots
        ):
            raise ScheduleContractError("workflow widget slots must be immutable")
        names = tuple(item.name for item in self.widget_slots)
        if len(names) != len(set(names)):
            raise ScheduleContractError("workflow widget slots contain duplicates")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowFixture:
    """One immutable canonical workflow plus its explicit compatibility contract."""

    identifier: str
    variant: str
    package: WorkflowRequirement
    nodes: tuple[WorkflowRequirement, ...]
    host: WorkflowHostRequirement
    profile: WorkflowRequirement
    node_contracts: tuple[WorkflowNodeContract, ...]
    workflow: Mapping[str, object]

    def __post_init__(self) -> None:
        _text(self.identifier, label="workflow fixture identifier")
        if self.variant not in {
            "Anima Aesthetic v1.x",
            "Anima Base v1.0",
            "Anima Turbo v1.0",
            "FLUX.1-schnell",
            "RAW",
            "Turbo",
            "Z-Image Base",
            "Z-Image Turbo",
            "Qwen Image",
            "SD3",
            "AuraFlow v0.2",
            "Lumina-Image 2.0",
            "HunyuanImage 2.1 Base",
            "HunyuanImage 2.1 Distilled",
        }:
            raise ScheduleContractError("workflow fixture variant is unsupported")
        if not isinstance(self.package, WorkflowRequirement):
            raise ScheduleContractError("workflow fixture package is invalid")
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(item, WorkflowRequirement) for item in self.nodes
        ):
            raise ScheduleContractError("workflow fixture node requirements are invalid")
        if not isinstance(self.host, WorkflowHostRequirement):
            raise ScheduleContractError("workflow fixture host requirement is invalid")
        if not isinstance(self.profile, WorkflowRequirement):
            raise ScheduleContractError("workflow fixture profile requirement is invalid")
        if not isinstance(self.node_contracts, tuple) or any(
            not isinstance(item, WorkflowNodeContract) for item in self.node_contracts
        ):
            raise ScheduleContractError("workflow node contracts are invalid")
        if not isinstance(self.workflow, Mapping):
            raise ScheduleContractError("workflow fixture must be an object")


@dataclass(frozen=True, slots=True, kw_only=True)
class HostSchemaBaseline:
    """Pinned static host evidence in both supported public schema forms."""

    host_version: str
    host_revision: str
    object_info: Mapping[str, object]
    node_definition_v2: Mapping[str, object]

    def __post_init__(self) -> None:
        _text(self.host_version, label="host version")
        _text(self.host_revision, label="host revision")
        if not isinstance(self.object_info, Mapping) or not isinstance(
            self.node_definition_v2, Mapping
        ):
            raise ScheduleContractError("host baseline schemas must be objects")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowIssue:
    """One stable, path-free compatibility finding."""

    severity: WorkflowIssueSeverity
    kind: WorkflowIssueKind
    workflow_id: str
    node_id: str = ""
    input_name: str = ""
    expected: str = ""
    actual: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.severity, WorkflowIssueSeverity):
            raise ScheduleContractError("workflow issue severity is invalid")
        if not isinstance(self.kind, WorkflowIssueKind):
            raise ScheduleContractError("workflow issue kind is invalid")
        _text(self.workflow_id, label="workflow issue fixture ID")
        for label, value in (
            ("node ID", self.node_id),
            ("input name", self.input_name),
            ("expected value", self.expected),
            ("actual value", self.actual),
        ):
            if not isinstance(value, str) or len(value) > _MAX_TEXT:
                raise ScheduleContractError(f"workflow issue {label} is invalid")

    def projection(self) -> dict[str, str]:
        return {
            "actual": self.actual,
            "expected": self.expected,
            "input": self.input_name,
            "kind": self.kind.value,
            "node": self.node_id,
            "severity": self.severity.value,
            "workflow": self.workflow_id,
        }


def _issue_sort_key(issue: WorkflowIssue) -> tuple[str, str, str, str, str, str]:
    return (
        issue.workflow_id,
        issue.node_id,
        issue.input_name,
        issue.kind.value,
        issue.expected,
        issue.actual,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowValidationReport:
    """Canonical machine-readable result for one fixture scan."""

    scan_mode: WorkflowScanMode
    lane: WorkflowValidationLane
    host_version: str
    host_revision: str
    package: WorkflowRequirement
    nodes: tuple[WorkflowRequirement, ...]
    workflow_count: int
    issues: tuple[WorkflowIssue, ...]
    compatible: bool
    gate_passed: bool
    observational: bool
    report_bytes: bytes = field(init=False, repr=False)
    report_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scan_mode, WorkflowScanMode):
            raise ScheduleContractError("workflow report scan mode is invalid")
        if not isinstance(self.lane, WorkflowValidationLane):
            raise ScheduleContractError("workflow report lane is invalid")
        _text(self.host_version, label="workflow report host version")
        _text(self.host_revision, label="workflow report host revision")
        if not isinstance(self.package, WorkflowRequirement):
            raise ScheduleContractError("workflow report package is invalid")
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(item, WorkflowRequirement) for item in self.nodes
        ):
            raise ScheduleContractError("workflow report nodes are invalid")
        if (
            not isinstance(self.workflow_count, int)
            or isinstance(self.workflow_count, bool)
            or self.workflow_count < 1
        ):
            raise ScheduleContractError("workflow report count is invalid")
        if not isinstance(self.issues, tuple) or any(
            not isinstance(item, WorkflowIssue) for item in self.issues
        ):
            raise ScheduleContractError("workflow report issues are invalid")
        if self.issues != tuple(sorted(self.issues, key=_issue_sort_key)):
            raise ScheduleContractError("workflow report issues are not canonical")
        if any(
            not isinstance(value, bool)
            for value in (self.compatible, self.gate_passed, self.observational)
        ):
            raise ScheduleContractError("workflow report result flags are invalid")
        if self.compatible != (not self.issues):
            raise ScheduleContractError("workflow report compatibility disagrees with issues")
        expected_observational = self.lane is WorkflowValidationLane.LATEST_HOST
        if self.observational is not expected_observational:
            raise ScheduleContractError("workflow report observational flag disagrees with lane")
        expected_gate = self.compatible or expected_observational
        if self.gate_passed is not expected_gate:
            raise ScheduleContractError("workflow report gate result disagrees with lane policy")
        report_bytes = _canonical_bytes(self.projection())
        object.__setattr__(self, "report_bytes", report_bytes)
        object.__setattr__(
            self,
            "report_fingerprint",
            "sha256:" + hashlib.sha256(report_bytes).hexdigest(),
        )

    def projection(self) -> dict[str, object]:
        return {
            "host": {
                "revision": self.host_revision,
                "version": self.host_version,
            },
            "issues": [item.projection() for item in self.issues],
            "lane": self.lane.value,
            "nodes": [item.projection() for item in self.nodes],
            "package": self.package.projection(),
            "result": {
                "compatible": self.compatible,
                "gate_passed": self.gate_passed,
                "observational": self.observational,
            },
            "scan_mode": self.scan_mode.value,
            "schema": WORKFLOW_VALIDATION_REPORT_SCHEMA_ID,
            "workflow_count": self.workflow_count,
        }


def _resource_json(name: str) -> dict[str, object]:
    try:
        raw = files("comfyui_sigmax.workflows").joinpath(name).read_bytes()
        loaded = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScheduleContractError(f"packaged workflow resource {name} is invalid") from exc
    return _mapping(loaded, label=f"packaged workflow resource {name}")


def _requirement(value: object, *, label: str) -> WorkflowRequirement:
    data = _mapping(value, label=label)
    return WorkflowRequirement(
        identifier=_text(data.get("id"), label=f"{label} ID"),
        version=_text(data.get("version"), label=f"{label} version"),
    )


def _host_requirement(value: object) -> WorkflowHostRequirement:
    data = _mapping(value, label="workflow fixture host")
    return WorkflowHostRequirement(
        identifier=_text(data.get("id"), label="workflow fixture host ID"),
        version=_text(data.get("version"), label="workflow fixture host version"),
        api_version=_text(data.get("api_version"), label="workflow fixture host API"),
    )


def _widget_slot(value: object) -> WorkflowWidgetSlot:
    data = _mapping(value, label="workflow widget slot")
    return WorkflowWidgetSlot(
        name=_text(data.get("name"), label="workflow widget slot name"),
        type_name=_text(data.get("type"), label="workflow widget slot type"),
        fixed_value=data.get("fixed"),
    )


def _node_contract(value: object) -> WorkflowNodeContract:
    data = _mapping(value, label="workflow node contract")
    node_id = data.get("node_id")
    if not isinstance(node_id, int) or isinstance(node_id, bool):
        raise ScheduleContractError("workflow node contract ID is invalid")
    return WorkflowNodeContract(
        node_id=node_id,
        node_type=_text(data.get("node_type"), label="workflow node contract type"),
        widget_slots=tuple(
            _widget_slot(item)
            for item in _sequence(data.get("widget_slots"), label="workflow widget slots")
        ),
    )


def load_canonical_workflow_fixtures() -> tuple[WorkflowFixture, ...]:
    """Load fresh canonical Turbo/RAW workflow fixtures from package data."""

    bundle = _resource_json("fixtures.json")
    if bundle.get("schema") != WORKFLOW_FIXTURE_BUNDLE_SCHEMA_ID:
        raise ScheduleContractError("workflow fixture bundle schema is unsupported")
    fixtures: list[WorkflowFixture] = []
    for raw_fixture in _sequence(bundle.get("fixtures"), label="workflow fixture bundle"):
        data = _mapping(raw_fixture, label="workflow fixture")
        fixtures.append(
            WorkflowFixture(
                identifier=_text(data.get("id"), label="workflow fixture ID"),
                variant=_text(data.get("variant"), label="workflow fixture variant"),
                package=_requirement(data.get("package"), label="workflow fixture package"),
                nodes=tuple(
                    _requirement(item, label="workflow fixture node")
                    for item in _sequence(data.get("nodes"), label="workflow fixture nodes")
                ),
                host=_host_requirement(data.get("host")),
                profile=_requirement(data.get("profile"), label="workflow fixture profile"),
                node_contracts=tuple(
                    _node_contract(item)
                    for item in _sequence(
                        data.get("node_contracts"),
                        label="workflow fixture node contracts",
                    )
                ),
                workflow=_mapping(data.get("workflow"), label="workflow fixture graph"),
            )
        )
    result = tuple(sorted(fixtures, key=lambda item: item.identifier))
    if tuple(item.identifier for item in result) != (
        "anima-aesthetic-v1-framework-50",
        "anima-base-v1-framework-50",
        "anima-turbo-v1-framework-8",
        "auraflow-v0-2-official-50",
        "flux1-schnell-official-4",
        "hunyuan-image21-base-official-50",
        "hunyuan-image21-distilled-official-8",
        "krea2-raw-diffusers-portrait-761x1353",
        "krea2-raw-official-landscape-1353x761",
        "krea2-raw-official-square-1024",
        "krea2-turbo-1024",
        "lumina2-v2-official-50",
        "qwen-image-comfy-fixed-official-50",
        "sd3-comfy-diffusers-fixed-framework-28",
        "sd3-publisher-reference-official-50",
        "z-image-base-official-50",
        "z-image-turbo-official-8",
    ):
        raise ScheduleContractError("workflow fixture inventory is not canonical")
    return result


def load_pinned_host_baseline() -> HostSchemaBaseline:
    """Load fresh pinned legacy/v2 host schemas from package data."""

    data = _resource_json("host_baseline.json")
    if data.get("schema") != HOST_BASELINE_SCHEMA_ID:
        raise ScheduleContractError("workflow host baseline schema is unsupported")
    host = _mapping(data.get("host"), label="workflow host baseline identity")
    return HostSchemaBaseline(
        host_version=_text(host.get("version"), label="workflow host baseline version"),
        host_revision=_text(host.get("revision"), label="workflow host baseline revision"),
        object_info=_mapping(data.get("object_info"), label="workflow host object_info"),
        node_definition_v2=_mapping(
            data.get("node_definition_v2"),
            label="workflow host node definitions",
        ),
    )


def _report(
    *,
    fixtures: Sequence[WorkflowFixture],
    scan_mode: WorkflowScanMode,
    lane: WorkflowValidationLane,
    host_version: str,
    host_revision: str,
    issues: Sequence[WorkflowIssue],
) -> WorkflowValidationReport:
    first = sorted(fixtures, key=lambda item: item.identifier)[0]
    requirements: dict[str, WorkflowRequirement] = {}
    for fixture in fixtures:
        for requirement in fixture.nodes:
            existing = requirements.get(requirement.identifier)
            if existing is not None and existing != requirement:
                raise ScheduleContractError(
                    "workflow fixtures disagree on one node requirement version"
                )
            requirements[requirement.identifier] = requirement
    ordered_issues = tuple(sorted(issues, key=_issue_sort_key))
    compatible = not ordered_issues
    observational = lane is WorkflowValidationLane.LATEST_HOST
    return WorkflowValidationReport(
        scan_mode=scan_mode,
        lane=lane,
        host_version=host_version,
        host_revision=host_revision,
        package=first.package,
        nodes=tuple(requirements[key] for key in sorted(requirements)),
        workflow_count=len(fixtures),
        issues=ordered_issues,
        compatible=compatible,
        gate_passed=compatible or observational,
        observational=observational,
    )


def _new_issue(
    *,
    kind: WorkflowIssueKind,
    workflow_id: str,
    node_id: object = "",
    input_name: str = "",
    expected: object = "",
    actual: object = "",
) -> WorkflowIssue:
    return WorkflowIssue(
        severity=WorkflowIssueSeverity.ERROR,
        kind=kind,
        workflow_id=workflow_id,
        node_id="" if node_id == "" else str(node_id),
        input_name=input_name,
        expected="" if expected == "" else _display(expected),
        actual="" if actual == "" else _display(actual),
    )


def _host_input_order(raw_node: Mapping[str, object]) -> tuple[str, ...]:
    if "input" in raw_node and "inputs" not in raw_node:
        input_root = _mapping(raw_node.get("input"), label="legacy host inputs")
        names: list[str] = []
        for section in ("required", "optional"):
            group = input_root.get(section, {})
            names.extend(_mapping(group, label=f"legacy host {section} inputs"))
        return tuple(names)
    if "inputs" in raw_node and "input" not in raw_node:
        return tuple(_mapping(raw_node.get("inputs"), label="v2 host inputs"))
    raise ScheduleContractError("host node mixes or omits supported input schemas")


def _saved_inputs(node: Mapping[str, object]) -> dict[str, str]:
    raw_inputs = node.get("inputs", [])
    result: dict[str, str] = {}
    for raw_input in _sequence(raw_inputs, label="saved workflow inputs"):
        input_data = _mapping(raw_input, label="saved workflow input")
        name = _text(input_data.get("name"), label="saved workflow input name")
        type_name = _text(input_data.get("type"), label="saved workflow input type")
        if name in result:
            raise ScheduleContractError("saved workflow inputs contain duplicates")
        result[name] = type_name
    return result


def _value_matches_type(value: object, type_name: str) -> bool:
    if type_name in {"STRING", "COMBO"}:
        return isinstance(value, str)
    if type_name == "INT":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "FLOAT":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if type_name == "BOOLEAN":
        return isinstance(value, bool)
    return False


def _metadata_issues(
    fixture: WorkflowFixture,
    workflow: Mapping[str, object],
) -> list[WorkflowIssue]:
    try:
        metadata = extract_workflow_metadata(workflow)
    except ScheduleContractError:
        metadata = None
    if metadata is None:
        return [
            _new_issue(
                kind=WorkflowIssueKind.MALFORMED_METADATA,
                workflow_id=fixture.identifier,
                expected="valid extra.comfyui_sigmax",
                actual="missing or malformed",
            )
        ]
    if (
        metadata.package != fixture.package
        or metadata.nodes != fixture.nodes
        or metadata.host != fixture.host
        or metadata.profile != fixture.profile
    ):
        return [
            _new_issue(
                kind=WorkflowIssueKind.MALFORMED_METADATA,
                workflow_id=fixture.identifier,
                expected="fixture identity requirements",
                actual="metadata requirement drift",
            )
        ]
    return []


def _validate_node(
    *,
    fixture: WorkflowFixture,
    contract: WorkflowNodeContract,
    saved_node: Mapping[str, object],
    raw_host_node: Mapping[str, object],
    host_node: ComfyNodeDefinition,
) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    if raw_host_node.get("python_module") not in _SUPPORTED_PYTHON_MODULES:
        issues.append(
            _new_issue(
                kind=WorkflowIssueKind.NORMALIZED_DIRECTORY_FAILURE,
                workflow_id=fixture.identifier,
                node_id=contract.node_id,
                expected=SIGMAX_NODE_MODULE,
                actual=raw_host_node.get("python_module"),
            )
        )
    if host_node.deprecated:
        issues.append(
            _new_issue(
                kind=WorkflowIssueKind.DEPRECATED_NODE,
                workflow_id=fixture.identifier,
                node_id=contract.node_id,
                actual=contract.node_type,
            )
        )
    if host_node.experimental:
        issues.append(
            _new_issue(
                kind=WorkflowIssueKind.EXPERIMENTAL_NODE,
                workflow_id=fixture.identifier,
                node_id=contract.node_id,
                actual=contract.node_type,
            )
        )
    properties = _mapping(saved_node.get("properties"), label="saved workflow node properties")
    if properties.get("cnr_id") != fixture.package.identifier:
        issues.append(
            _new_issue(
                kind=WorkflowIssueKind.NORMALIZED_DIRECTORY_FAILURE,
                workflow_id=fixture.identifier,
                node_id=contract.node_id,
                expected=fixture.package.identifier,
                actual=properties.get("cnr_id"),
            )
        )

    saved_inputs = _saved_inputs(saved_node)
    host_inputs = {item.name: item for item in host_node.inputs}
    for name, saved_type in saved_inputs.items():
        host_input = host_inputs.get(name)
        if host_input is None:
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.MISSING_INPUT,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    input_name=name,
                    expected=saved_type,
                    actual="missing",
                )
            )
        elif host_input.type_name != saved_type:
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.INPUT_TYPE_DRIFT,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    input_name=name,
                    expected=saved_type,
                    actual=host_input.type_name,
                )
            )

    slot_names = tuple(item.name for item in contract.widget_slots)
    represented = set(saved_inputs) | set(slot_names)
    for host_input in host_node.inputs:
        if host_input.section == "required" and host_input.name not in represented:
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.MISSING_INPUT,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    input_name=host_input.name,
                    expected=host_input.type_name,
                    actual="missing",
                )
            )

    raw_order = _host_input_order(raw_host_node)
    host_widget_order = tuple(
        name
        for name in raw_order
        if name not in saved_inputs
        and name in host_inputs
        and host_inputs[name].type_name in _WIDGET_TYPES
    )
    if host_widget_order != slot_names:
        issues.append(
            _new_issue(
                kind=WorkflowIssueKind.WIDGET_SLOT_DRIFT,
                workflow_id=fixture.identifier,
                node_id=contract.node_id,
                expected=slot_names,
                actual=host_widget_order,
            )
        )

    raw_values = saved_node.get("widgets_values", [])
    if not isinstance(raw_values, list):
        issues.append(
            _new_issue(
                kind=WorkflowIssueKind.WIDGET_SLOT_DRIFT,
                workflow_id=fixture.identifier,
                node_id=contract.node_id,
                expected=len(contract.widget_slots),
                actual="non-array",
            )
        )
        return issues
    if len(raw_values) != len(contract.widget_slots):
        issues.append(
            _new_issue(
                kind=WorkflowIssueKind.WIDGET_SLOT_DRIFT,
                workflow_id=fixture.identifier,
                node_id=contract.node_id,
                expected=len(contract.widget_slots),
                actual=len(raw_values),
            )
        )
    for slot, value in zip(contract.widget_slots, raw_values, strict=False):
        host_input = host_inputs.get(slot.name)
        if host_input is None:
            continue
        if host_input.type_name != slot.type_name:
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.INPUT_TYPE_DRIFT,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    input_name=slot.name,
                    expected=slot.type_name,
                    actual=host_input.type_name,
                )
            )
        if not _value_matches_type(value, slot.type_name):
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.WIDGET_SLOT_DRIFT,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    input_name=slot.name,
                    expected=slot.type_name,
                    actual=type(value).__name__,
                )
            )
        if slot.type_name == "COMBO" and (
            (slot.fixed_value is not None and value != slot.fixed_value)
            or value not in host_input.options
        ):
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.INVALID_FIXED_COMBO_VALUE,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    input_name=slot.name,
                    expected=slot.fixed_value
                    if slot.fixed_value is not None
                    else host_input.options,
                    actual=value,
                )
            )
    return issues


def _validate_fixture(
    *,
    fixture: WorkflowFixture,
    raw_host_nodes: Mapping[str, object],
    host_nodes: Mapping[str, ComfyNodeDefinition],
) -> list[WorkflowIssue]:
    workflow = fixture.workflow
    raw_nodes = workflow.get("nodes")
    if not isinstance(raw_nodes, list) or any(not isinstance(item, Mapping) for item in raw_nodes):
        return [
            _new_issue(
                kind=WorkflowIssueKind.WORKFLOW_SCHEMA_MALFORMED,
                workflow_id=fixture.identifier,
                expected="workflow nodes array",
                actual=type(raw_nodes).__name__,
            )
        ]
    issues = _metadata_issues(fixture, workflow)
    saved_by_id = {
        cast(Mapping[str, object], item).get("id"): cast(Mapping[str, object], item)
        for item in raw_nodes
    }
    for contract in fixture.node_contracts:
        saved_node = saved_by_id.get(contract.node_id)
        if saved_node is None or saved_node.get("type") != contract.node_type:
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.MISSING_NODE,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    expected=contract.node_type,
                    actual="missing or renamed",
                )
            )
            continue
        raw_host_node = raw_host_nodes.get(contract.node_type)
        host_node = host_nodes.get(contract.node_type)
        if not isinstance(raw_host_node, Mapping) or host_node is None:
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.MISSING_NODE,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    expected=contract.node_type,
                    actual="absent from host schema",
                )
            )
            continue
        try:
            issues.extend(
                _validate_node(
                    fixture=fixture,
                    contract=contract,
                    saved_node=saved_node,
                    raw_host_node=cast(Mapping[str, object], raw_host_node),
                    host_node=host_node,
                )
            )
        except ScheduleContractError:
            issues.append(
                _new_issue(
                    kind=WorkflowIssueKind.WORKFLOW_SCHEMA_MALFORMED,
                    workflow_id=fixture.identifier,
                    node_id=contract.node_id,
                    expected="valid saved node contract",
                    actual="malformed",
                )
            )
    return issues


def validate_workflow_fixtures(
    *,
    fixtures: tuple[WorkflowFixture, ...],
    object_info: Mapping[str, object],
    scan_mode: WorkflowScanMode,
    lane: WorkflowValidationLane,
    host_version: str,
    host_revision: str,
) -> WorkflowValidationReport:
    """Validate fixtures without importing or executing host node implementations."""

    if (
        not isinstance(fixtures, tuple)
        or not fixtures
        or any(not isinstance(item, WorkflowFixture) for item in fixtures)
    ):
        raise ScheduleContractError("workflow fixtures must be a non-empty immutable tuple")
    if not isinstance(object_info, Mapping):
        raise ScheduleContractError("object_info must be a mapping")
    if not isinstance(scan_mode, WorkflowScanMode) or not isinstance(lane, WorkflowValidationLane):
        raise ScheduleContractError("workflow validation mode or lane is invalid")
    host_version = _text(host_version, label="workflow validation host version")
    host_revision = _text(host_revision, label="workflow validation host revision")
    ordered = tuple(sorted(fixtures, key=lambda item: item.identifier))
    first = ordered[0]
    if any(item.package != first.package for item in ordered):
        raise ScheduleContractError("workflow fixtures disagree on package requirements")
    node_versions: dict[str, WorkflowRequirement] = {}
    for fixture in ordered:
        for requirement in fixture.nodes:
            existing = node_versions.get(requirement.identifier)
            if existing is not None and existing != requirement:
                raise ScheduleContractError(
                    "workflow fixtures disagree on one node requirement version"
                )
            node_versions[requirement.identifier] = requirement
    raw_host_nodes = dict(object_info)
    try:
        normalized = normalize_object_info(raw_host_nodes)
    except (ComfyAdapterCompatibilityError, ScheduleContractError, TypeError, ValueError):
        return _report(
            fixtures=ordered,
            scan_mode=scan_mode,
            lane=lane,
            host_version=host_version,
            host_revision=host_revision,
            issues=(
                _new_issue(
                    kind=WorkflowIssueKind.HOST_SCHEMA_MALFORMED,
                    workflow_id=ordered[0].identifier,
                    expected="supported /object_info or Node Definition v2",
                    actual="malformed",
                ),
            ),
        )
    host_nodes = {item.node_id: item for item in normalized}
    issues: list[WorkflowIssue] = []
    for fixture in ordered:
        issues.extend(
            _validate_fixture(
                fixture=fixture,
                raw_host_nodes=raw_host_nodes,
                host_nodes=host_nodes,
            )
        )
    return _report(
        fixtures=ordered,
        scan_mode=scan_mode,
        lane=lane,
        host_version=host_version,
        host_revision=host_revision,
        issues=issues,
    )


def validate_pinned_workflow_fixtures() -> WorkflowValidationReport:
    """Validate packaged fixtures against the pinned static known-good host baseline."""

    baseline = load_pinned_host_baseline()
    return validate_workflow_fixtures(
        fixtures=load_canonical_workflow_fixtures(),
        object_info=baseline.object_info,
        scan_mode=WorkflowScanMode.PINNED_STATIC,
        lane=WorkflowValidationLane.KNOWN_GOOD,
        host_version=baseline.host_version,
        host_revision=baseline.host_revision,
    )


def validate_live_workflow_fixtures(
    *,
    object_info: Mapping[str, object],
    host_version: str,
    host_revision: str,
    lane: WorkflowValidationLane,
) -> WorkflowValidationReport:
    """Validate packaged fixtures against caller-observed live `/object_info` data."""

    return validate_workflow_fixtures(
        fixtures=load_canonical_workflow_fixtures(),
        object_info=object_info,
        scan_mode=WorkflowScanMode.LIVE_OBJECT_INFO,
        lane=lane,
        host_version=host_version,
        host_revision=host_revision,
    )


def serialize_workflow_validation_report(report: WorkflowValidationReport) -> bytes:
    """Serialize one report as a fingerprinted canonical envelope."""

    if not isinstance(report, WorkflowValidationReport):
        raise ScheduleContractError("workflow validation report is invalid")
    return _canonical_bytes(
        {
            "report": report.projection(),
            "report_fingerprint": report.report_fingerprint,
            "schema": WORKFLOW_VALIDATION_REPORT_ENVELOPE_SCHEMA_ID,
        }
    )


def deserialize_workflow_validation_report(payload: bytes | str) -> WorkflowValidationReport:
    """Deserialize and verify one canonical validation report envelope."""

    if not isinstance(payload, (bytes, str)):
        raise ScheduleContractError("workflow validation report payload is invalid")
    try:
        loaded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ScheduleContractError("workflow validation report is not valid JSON") from exc
    envelope = _mapping(loaded, label="workflow validation report envelope")
    if (
        set(envelope) != {"report", "report_fingerprint", "schema"}
        or envelope.get("schema") != WORKFLOW_VALIDATION_REPORT_ENVELOPE_SCHEMA_ID
    ):
        raise ScheduleContractError("workflow validation report envelope is unsupported")
    projection = _mapping(envelope.get("report"), label="workflow validation report")
    if projection.get("schema") != WORKFLOW_VALIDATION_REPORT_SCHEMA_ID:
        raise ScheduleContractError("workflow validation report schema is unsupported")
    host = _mapping(projection.get("host"), label="workflow validation report host")
    result = _mapping(projection.get("result"), label="workflow validation report result")
    report = WorkflowValidationReport(
        scan_mode=WorkflowScanMode(cast(str, projection.get("scan_mode"))),
        lane=WorkflowValidationLane(cast(str, projection.get("lane"))),
        host_version=_text(host.get("version"), label="workflow validation report host version"),
        host_revision=_text(
            host.get("revision"),
            label="workflow validation report host revision",
        ),
        package=_requirement(projection.get("package"), label="workflow validation report package"),
        nodes=tuple(
            _requirement(item, label="workflow validation report node")
            for item in _sequence(projection.get("nodes"), label="workflow validation report nodes")
        ),
        workflow_count=cast(int, projection.get("workflow_count")),
        issues=tuple(
            WorkflowIssue(
                severity=WorkflowIssueSeverity(
                    cast(str, _mapping(item, label="workflow issue").get("severity"))
                ),
                kind=WorkflowIssueKind(
                    cast(str, _mapping(item, label="workflow issue").get("kind"))
                ),
                workflow_id=cast(
                    str,
                    _mapping(item, label="workflow issue").get("workflow"),
                ),
                node_id=cast(str, _mapping(item, label="workflow issue").get("node")),
                input_name=cast(str, _mapping(item, label="workflow issue").get("input")),
                expected=cast(str, _mapping(item, label="workflow issue").get("expected")),
                actual=cast(str, _mapping(item, label="workflow issue").get("actual")),
            )
            for item in _sequence(
                projection.get("issues"),
                label="workflow validation report issues",
            )
        ),
        compatible=cast(bool, result.get("compatible")),
        gate_passed=cast(bool, result.get("gate_passed")),
        observational=cast(bool, result.get("observational")),
    )
    if envelope.get(
        "report_fingerprint"
    ) != report.report_fingerprint or serialize_workflow_validation_report(
        report
    ) != _canonical_bytes(envelope):
        raise ScheduleContractError("workflow validation report fingerprint is invalid")
    return report


class _NoRedirect(HTTPRedirectHandler):
    """Reject redirects so a loopback URL cannot pivot to another target."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _live_error(reason: WorkflowLiveLoadReason, action: str) -> WorkflowLiveLoadError:
    return WorkflowLiveLoadError(reason=reason, action=action)


def fetch_live_object_info(
    *,
    url: str = "http://127.0.0.1:8188/object_info",
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Fetch a bounded JSON object from a literal-loopback ComfyUI `/object_info` endpoint."""

    if not isinstance(url, str):
        raise _live_error(WorkflowLiveLoadReason.URL_NOT_LOOPBACK, "use a literal loopback URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/object_info"
        or parsed.query
        or parsed.fragment
    ):
        raise _live_error(
            WorkflowLiveLoadReason.URL_NOT_LOOPBACK,
            "use http on a literal loopback address with the /object_info path",
        )
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 < float(timeout_seconds) <= 30
    ):
        raise _live_error(WorkflowLiveLoadReason.HTTP_ERROR, "use a timeout from 0 to 30 seconds")
    # CRITICAL: keep the literal-loopback validation above before this network request.
    request = Request(  # noqa: S310 - URL scheme and address are fail-closed above.
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ComfyUI-Sigmax-workflow-validator/1",
        },
        method="GET",
    )
    try:
        with build_opener(_NoRedirect()).open(
            request,
            timeout=float(timeout_seconds),
        ) as response:
            content_type = response.headers.get_content_type()
            if content_type != "application/json":
                raise _live_error(
                    WorkflowLiveLoadReason.RESPONSE_NOT_JSON,
                    "return application/json from /object_info",
                )
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None:
                try:
                    declared_length = int(raw_length)
                except ValueError as exc:
                    raise _live_error(
                        WorkflowLiveLoadReason.HTTP_ERROR,
                        "return a valid Content-Length",
                    ) from exc
                if declared_length < 0 or declared_length > _MAX_LIVE_BYTES:
                    raise _live_error(
                        WorkflowLiveLoadReason.RESPONSE_TOO_LARGE,
                        "reduce the /object_info response size",
                    )
            payload = response.read(_MAX_LIVE_BYTES + 1)
    except WorkflowLiveLoadError:
        raise
    except (HTTPError, URLError, OSError, TimeoutError) as exc:
        raise _live_error(
            WorkflowLiveLoadReason.HTTP_ERROR,
            "make the loopback ComfyUI endpoint reachable without redirects",
        ) from exc
    if len(payload) > _MAX_LIVE_BYTES:
        raise _live_error(
            WorkflowLiveLoadReason.RESPONSE_TOO_LARGE,
            "reduce the /object_info response size",
        )
    try:
        loaded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _live_error(
            WorkflowLiveLoadReason.RESPONSE_NOT_JSON,
            "return valid JSON from /object_info",
        ) from exc
    if not isinstance(loaded, dict):
        raise _live_error(
            WorkflowLiveLoadReason.PAYLOAD_NOT_OBJECT,
            "return a JSON object from /object_info",
        )
    return cast(dict[str, object], loaded)
