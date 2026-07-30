"""Versioned workflow metadata and non-destructive ComfyUI graph attachment."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from comfyui_sigmax.core.capabilities import (
    CapabilityDimension,
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityReason,
)
from comfyui_sigmax.core.execution_receipts import ExecutionStatus
from comfyui_sigmax.core.fingerprints import canonical_projection_bytes
from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

_METADATA_SCHEMA = "sigmax.workflow-metadata/1"
_ENVELOPE_SCHEMA = "sigmax.workflow-metadata-envelope/1"
_ARTIFACT_SCHEMA = "sigmax.schedule-artifact/1"
_RECEIPT_SCHEMA = "sigmax.execution-receipt-envelope/1"
_WORKFLOW_NAMESPACE = "comfyui_sigmax"
_MAX_TRANSPORT_BYTES = 1_048_576
_MAX_REQUIREMENTS = 128
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")
_PRIVATE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    r"(?:^|[_.-])(?:api_?key|access_key|private_key|secret|password|passwd|credential|"
    r"cookie|token|authorization|auth)(?:[_.-]|$)",
    re.IGNORECASE,
)


def _sha256_identity(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_public_identifier(field_name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not _IDENTIFIER_PATTERN.fullmatch(value)
        or _PRIVATE_PATH_PATTERN.search(value)
        or _SECRET_PATTERN.search(value)
    ):
        raise ScheduleContractError(f"{field_name} must be a bounded public identifier")
    return value


def _require_fingerprint(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ScheduleContractError(f"{field_name} must be a SHA-256 fingerprint")
    return value


def _object(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], fields: frozenset[str], *, field_name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{field_name} fields do not match its schema")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowRequirement:
    """Exact public identity/version requirement."""

    identifier: str
    version: str

    def __post_init__(self) -> None:
        _require_public_identifier("requirement identifier", self.identifier)
        _require_public_identifier("requirement version", self.version)

    def projection(self) -> dict[str, str]:
        return {"id": self.identifier, "version": self.version}


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowHostRequirement:
    """Exact public host and API requirement."""

    identifier: str
    version: str
    api_version: str

    def __post_init__(self) -> None:
        _require_public_identifier("host identifier", self.identifier)
        _require_public_identifier("host version", self.version)
        _require_public_identifier("host API version", self.api_version)

    def projection(self) -> dict[str, str]:
        return {
            "api_version": self.api_version,
            "id": self.identifier,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowArtifactReference:
    """Portable construction and numerical artifact identities."""

    construction_fingerprint: str
    numerical_fingerprint: str

    def __post_init__(self) -> None:
        _require_fingerprint("construction fingerprint", self.construction_fingerprint)
        _require_fingerprint("numerical fingerprint", self.numerical_fingerprint)

    def projection(self) -> dict[str, str]:
        return {
            "construction_fingerprint": self.construction_fingerprint,
            "construction_schema": _ARTIFACT_SCHEMA,
            "numerical_fingerprint": self.numerical_fingerprint,
            "receipt_schema": _RECEIPT_SCHEMA,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowReceiptReference:
    """Portable receipt identity tied to one declared artifact."""

    receipt_fingerprint: str
    construction_fingerprint: str
    numerical_fingerprint: str
    status: ExecutionStatus

    def __post_init__(self) -> None:
        _require_fingerprint("receipt fingerprint", self.receipt_fingerprint)
        _require_fingerprint("construction fingerprint", self.construction_fingerprint)
        _require_fingerprint("numerical fingerprint", self.numerical_fingerprint)
        if not isinstance(self.status, ExecutionStatus):
            raise ScheduleContractError("receipt status is unsupported")

    def projection(self) -> dict[str, str]:
        return {
            "construction_fingerprint": self.construction_fingerprint,
            "numerical_fingerprint": self.numerical_fingerprint,
            "receipt_fingerprint": self.receipt_fingerprint,
            "schema": _RECEIPT_SCHEMA,
            "status": self.status.value,
        }


def _compatibility_projection(decision: CompatibilityDecision) -> dict[str, object]:
    return {
        "considered": [item.value for item in decision.considered],
        "level": decision.level.value,
        "reasons": [item.value for item in decision.reasons],
    }


def _compatibility_from_projection(value: object) -> CompatibilityDecision:
    projection = _object(value, field_name="compatibility")
    _exact(
        projection,
        frozenset({"considered", "level", "reasons"}),
        field_name="compatibility",
    )
    considered = projection["considered"]
    reasons = projection["reasons"]
    if not isinstance(considered, list) or not isinstance(reasons, list):
        raise ScheduleContractError("compatibility considered and reasons must be arrays")
    try:
        return CompatibilityDecision(
            level=CompatibilityLevel(cast(str, projection["level"])),
            considered=tuple(CapabilityDimension(cast(str, item)) for item in considered),
            reasons=tuple(CompatibilityReason(cast(str, item)) for item in reasons),
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("compatibility projection is unsupported") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkflowMetadata:
    """Immutable canonical metadata embedded by reference in a ComfyUI workflow."""

    package: WorkflowRequirement
    nodes: tuple[WorkflowRequirement, ...]
    host: WorkflowHostRequirement
    profile: WorkflowRequirement
    compatibility: CompatibilityDecision
    artifact: WorkflowArtifactReference
    receipts: tuple[WorkflowReceiptReference, ...] = ()
    metadata_bytes: bytes = field(init=False, repr=False)
    metadata_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.package, WorkflowRequirement):
            raise ScheduleContractError("package must be a WorkflowRequirement")
        if (
            not isinstance(self.nodes, tuple)
            or not self.nodes
            or len(self.nodes) > _MAX_REQUIREMENTS
            or not all(isinstance(item, WorkflowRequirement) for item in self.nodes)
        ):
            raise ScheduleContractError("nodes must be a bounded non-empty requirement tuple")
        node_ids = [item.identifier for item in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ScheduleContractError("node requirements contain duplicate identifiers")
        if not isinstance(self.host, WorkflowHostRequirement):
            raise ScheduleContractError("host must be a WorkflowHostRequirement")
        if not isinstance(self.profile, WorkflowRequirement):
            raise ScheduleContractError("profile must be a WorkflowRequirement")
        if not isinstance(self.compatibility, CompatibilityDecision):
            raise ScheduleContractError("compatibility must be a CompatibilityDecision")
        if not isinstance(self.artifact, WorkflowArtifactReference):
            raise ScheduleContractError("artifact must be a WorkflowArtifactReference")
        if (
            not isinstance(self.receipts, tuple)
            or len(self.receipts) > _MAX_REQUIREMENTS
            or not all(isinstance(item, WorkflowReceiptReference) for item in self.receipts)
        ):
            raise ScheduleContractError("receipts must be a bounded reference tuple")
        receipt_ids = [item.receipt_fingerprint for item in self.receipts]
        if len(receipt_ids) != len(set(receipt_ids)):
            raise ScheduleContractError("receipt references contain duplicate fingerprints")
        for receipt in self.receipts:
            if (
                receipt.construction_fingerprint != self.artifact.construction_fingerprint
                or receipt.numerical_fingerprint != self.artifact.numerical_fingerprint
            ):
                raise ScheduleContractError(
                    "receipt reference disagrees with the workflow artifact"
                )
        metadata_bytes = canonical_projection_bytes(self._projection())
        object.__setattr__(self, "metadata_bytes", metadata_bytes)
        object.__setattr__(self, "metadata_fingerprint", _sha256_identity(metadata_bytes))

    def _projection(self) -> dict[str, object]:
        return {
            "artifact": self.artifact.projection(),
            "compatibility": _compatibility_projection(self.compatibility),
            "profile": self.profile.projection(),
            "receipts": [
                item.projection()
                for item in sorted(self.receipts, key=lambda item: item.receipt_fingerprint)
            ],
            "requirements": {
                "host": self.host.projection(),
                "nodes": [
                    item.projection()
                    for item in sorted(self.nodes, key=lambda item: item.identifier)
                ],
                "package": self.package.projection(),
            },
            "schema": _METADATA_SCHEMA,
        }

    def projection(self) -> dict[str, object]:
        value = json.loads(self.metadata_bytes)
        return cast(dict[str, object], value)


def serialize_workflow_metadata(metadata: WorkflowMetadata) -> bytes:
    """Serialize metadata inside its versioned canonical envelope."""

    if not isinstance(metadata, WorkflowMetadata):
        raise ScheduleContractError("metadata must be WorkflowMetadata")
    return canonical_projection_bytes(
        {
            "metadata": metadata.projection(),
            "metadata_fingerprint": metadata.metadata_fingerprint,
            "schema": _ENVELOPE_SCHEMA,
        }
    )


def _duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise ScheduleContractError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ScheduleContractError(f"non-finite JSON constant is forbidden: {value}")


def _decode_object(
    payload: bytes | str, *, maximum: int = _MAX_TRANSPORT_BYTES
) -> dict[str, object]:
    if isinstance(payload, str):
        if payload.startswith("\ufeff"):
            raise ScheduleContractError("transport must not contain a BOM")
        try:
            raw = payload.encode("utf-8")
        except UnicodeError as exc:
            raise ScheduleContractError("transport must be valid Unicode") from exc
    elif isinstance(payload, bytes):
        raw = payload
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ScheduleContractError("transport must not contain a BOM")
    else:
        raise ScheduleContractError("transport must be bytes or text")
    if not raw or len(raw) > maximum:
        raise ScheduleContractError("transport size is outside the allowed range")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_duplicate_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ScheduleContractError("transport is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ScheduleContractError("transport root must be an object")
    projection = cast(dict[str, object], decoded)
    if canonical_projection_bytes(projection) != raw:
        raise ScheduleContractError("transport must use canonical JSON")
    return projection


def _requirement_from_projection(value: object, *, field_name: str) -> WorkflowRequirement:
    projection = _object(value, field_name=field_name)
    _exact(projection, frozenset({"id", "version"}), field_name=field_name)
    return WorkflowRequirement(
        identifier=cast(str, projection["id"]),
        version=cast(str, projection["version"]),
    )


def _host_from_projection(value: object) -> WorkflowHostRequirement:
    projection = _object(value, field_name="requirements.host")
    _exact(
        projection,
        frozenset({"api_version", "id", "version"}),
        field_name="requirements.host",
    )
    return WorkflowHostRequirement(
        identifier=cast(str, projection["id"]),
        version=cast(str, projection["version"]),
        api_version=cast(str, projection["api_version"]),
    )


def _artifact_from_projection(value: object) -> WorkflowArtifactReference:
    projection = _object(value, field_name="artifact")
    _exact(
        projection,
        frozenset(
            {
                "construction_fingerprint",
                "construction_schema",
                "numerical_fingerprint",
                "receipt_schema",
            }
        ),
        field_name="artifact",
    )
    if (
        projection["construction_schema"] != _ARTIFACT_SCHEMA
        or projection["receipt_schema"] != _RECEIPT_SCHEMA
    ):
        raise ScheduleContractError("workflow artifact schema references are unsupported")
    return WorkflowArtifactReference(
        construction_fingerprint=cast(str, projection["construction_fingerprint"]),
        numerical_fingerprint=cast(str, projection["numerical_fingerprint"]),
    )


def _receipt_from_projection(value: object) -> WorkflowReceiptReference:
    projection = _object(value, field_name="receipt reference")
    _exact(
        projection,
        frozenset(
            {
                "construction_fingerprint",
                "numerical_fingerprint",
                "receipt_fingerprint",
                "schema",
                "status",
            }
        ),
        field_name="receipt reference",
    )
    if projection["schema"] != _RECEIPT_SCHEMA:
        raise ScheduleContractError("workflow receipt schema reference is unsupported")
    try:
        status = ExecutionStatus(cast(str, projection["status"]))
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("workflow receipt status is unsupported") from exc
    return WorkflowReceiptReference(
        receipt_fingerprint=cast(str, projection["receipt_fingerprint"]),
        construction_fingerprint=cast(str, projection["construction_fingerprint"]),
        numerical_fingerprint=cast(str, projection["numerical_fingerprint"]),
        status=status,
    )


def _metadata_from_projection(projection: dict[str, object]) -> WorkflowMetadata:
    _exact(
        projection,
        frozenset(
            {
                "artifact",
                "compatibility",
                "profile",
                "receipts",
                "requirements",
                "schema",
            }
        ),
        field_name="workflow metadata",
    )
    if projection["schema"] != _METADATA_SCHEMA:
        raise ScheduleContractError("workflow metadata schema is unsupported")
    requirements = _object(projection["requirements"], field_name="requirements")
    _exact(
        requirements,
        frozenset({"host", "nodes", "package"}),
        field_name="requirements",
    )
    nodes_value = requirements["nodes"]
    receipts_value = projection["receipts"]
    if not isinstance(nodes_value, list) or not isinstance(receipts_value, list):
        raise ScheduleContractError("workflow nodes and receipts must be arrays")
    metadata = WorkflowMetadata(
        package=_requirement_from_projection(
            requirements["package"],
            field_name="requirements.package",
        ),
        nodes=tuple(
            _requirement_from_projection(item, field_name="requirements.nodes")
            for item in nodes_value
        ),
        host=_host_from_projection(requirements["host"]),
        profile=_requirement_from_projection(projection["profile"], field_name="profile"),
        compatibility=_compatibility_from_projection(projection["compatibility"]),
        artifact=_artifact_from_projection(projection["artifact"]),
        receipts=tuple(_receipt_from_projection(item) for item in receipts_value),
    )
    if metadata.metadata_bytes != canonical_projection_bytes(projection):
        raise ScheduleContractError("workflow metadata ordering is not canonical")
    return metadata


def deserialize_workflow_metadata(payload: bytes | str) -> WorkflowMetadata:
    """Strictly parse and verify one workflow metadata envelope."""

    envelope = _decode_object(payload)
    _exact(
        envelope,
        frozenset({"metadata", "metadata_fingerprint", "schema"}),
        field_name="workflow metadata envelope",
    )
    if envelope["schema"] != _ENVELOPE_SCHEMA:
        raise ScheduleContractError("workflow metadata envelope schema is unsupported")
    metadata = _metadata_from_projection(_object(envelope["metadata"], field_name="metadata"))
    fingerprint = _require_fingerprint(
        "metadata fingerprint",
        envelope["metadata_fingerprint"],
    )
    if fingerprint != metadata.metadata_fingerprint:
        raise ScheduleContractError("workflow metadata fingerprint mismatch")
    return metadata


def _workflow_parts(
    workflow: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    # IMPORTANT: preserve the official 0.4/1 graph forms; only root `extra` is extended.
    # Source: https://docs.comfy.org/specs/workflow_json
    if not isinstance(workflow, Mapping):
        raise ScheduleContractError("workflow must be an object")
    version = workflow.get("version")
    numeric_version = type(version) in (int, float)
    legacy = numeric_version and version == 0.4
    current = numeric_version and version == 1
    if not legacy and not current:
        raise ScheduleContractError("workflow version must be legacy 0.4 or current 1")
    root = dict(workflow)
    extra_value = workflow.get("extra")
    if extra_value is None:
        return root, None
    if not isinstance(extra_value, Mapping):
        raise ScheduleContractError("workflow extra must be an object or null")
    return root, dict(cast(Mapping[str, object], extra_value))


def _metadata_from_namespace(value: object) -> WorkflowMetadata:
    if not isinstance(value, Mapping):
        raise ScheduleContractError("workflow Sigmax namespace must be an object")
    payload = canonical_projection_bytes(dict(cast(Mapping[str, object], value)))
    return deserialize_workflow_metadata(payload)


def attach_workflow_metadata(
    workflow: Mapping[str, object],
    metadata: WorkflowMetadata,
) -> dict[str, object]:
    """Copy-on-write attach metadata to a supported ComfyUI workflow."""

    if not isinstance(metadata, WorkflowMetadata):
        raise ScheduleContractError("metadata must be WorkflowMetadata")
    root, existing_extra = _workflow_parts(workflow)
    extra = {} if existing_extra is None else existing_extra
    existing = extra.get(_WORKFLOW_NAMESPACE)
    if existing is not None:
        if _metadata_from_namespace(existing) != metadata:
            raise ScheduleContractError("workflow contains conflicting Sigmax metadata")
        return root
    extra[_WORKFLOW_NAMESPACE] = json.loads(serialize_workflow_metadata(metadata))
    root["extra"] = extra
    return root


def extract_workflow_metadata(workflow: Mapping[str, object]) -> WorkflowMetadata | None:
    """Extract and verify metadata without changing the workflow."""

    _, extra = _workflow_parts(workflow)
    if extra is None or _WORKFLOW_NAMESPACE not in extra:
        return None
    return _metadata_from_namespace(extra[_WORKFLOW_NAMESPACE])


def detach_workflow_metadata(workflow: Mapping[str, object]) -> dict[str, object]:
    """Copy-on-write remove only the verified Sigmax namespace."""

    root, extra = _workflow_parts(workflow)
    if extra is None or _WORKFLOW_NAMESPACE not in extra:
        return root
    _metadata_from_namespace(extra[_WORKFLOW_NAMESPACE])
    del extra[_WORKFLOW_NAMESPACE]
    if extra:
        root["extra"] = extra
    else:
        root.pop("extra", None)
    return root
