"""Immutable execution receipts and portable artifact/receipt bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

from comfyui_sigmax.core.artifacts import (
    ArtifactField,
    ScheduleArtifact,
    deserialize_schedule_artifact,
    serialize_schedule_artifact,
)
from comfyui_sigmax.core.capabilities import (
    CapabilityDimension,
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityReason,
    NoiseOwnership,
)
from comfyui_sigmax.core.fingerprints import canonical_projection_bytes
from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

_RECEIPT_SCHEMA = "sigmax.execution-receipt/1"
_RECEIPT_ENVELOPE_SCHEMA = "sigmax.execution-receipt-envelope/1"
_BUNDLE_SCHEMA = "sigmax.portable-execution-bundle/1"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}$")
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PRIVATE_PATH_PATTERN = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    r"(?:^|[_.-])(?:api_?key|access_key|private_key|secret|password|passwd|credential|"
    r"cookie|token|authorization|auth)(?:[_.-]|$)",
    re.IGNORECASE,
)
_MAX_RECEIPT_BYTES = 1_048_576
_MAX_BUNDLE_BYTES = 2_097_152
_RECEIPT_FIELDS = frozenset(
    {
        "artifact",
        "compatibility",
        "counts",
        "effective_inputs",
        "execution",
        "host",
        "model",
        "profile",
        "rng_ownership",
        "sampler",
        "schema",
    }
)


class ExecutionStatus(str, Enum):
    """Truthful final state of one requested execution."""

    NOT_EXECUTED = "not_executed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


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


def _require_count(field_name: str, value: object, *, positive: bool) -> int:
    minimum = 1 if positive else 0
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or value > 1_000_000
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ScheduleContractError(f"{field_name} must be a bounded {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionComponent:
    """Portable identity for an executed model or sampler."""

    identifier: str
    version: str
    fingerprint: str

    def __post_init__(self) -> None:
        _require_public_identifier("component identifier", self.identifier)
        _require_public_identifier("component version", self.version)
        _require_fingerprint("component fingerprint", self.fingerprint)

    def projection(self) -> dict[str, str]:
        return {
            "fingerprint": self.fingerprint,
            "id": self.identifier,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionHost:
    """Portable bounded host identity without machine-local data."""

    identifier: str
    version: str
    revision: str
    api_version: str

    def __post_init__(self) -> None:
        _require_public_identifier("host identifier", self.identifier)
        _require_public_identifier("host version", self.version)
        _require_public_identifier("host revision", self.revision)
        _require_public_identifier("host API version", self.api_version)

    def projection(self) -> dict[str, str]:
        return {
            "api_version": self.api_version,
            "id": self.identifier,
            "revision": self.revision,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionRngOwnership:
    """Explicit random-source ownership for schedule, model, and sampler."""

    schedule: NoiseOwnership
    model: NoiseOwnership
    sampler: NoiseOwnership

    def __post_init__(self) -> None:
        for field_name in ("schedule", "model", "sampler"):
            if not isinstance(getattr(self, field_name), NoiseOwnership):
                raise ScheduleContractError(f"{field_name} RNG ownership is unsupported")

    def projection(self) -> dict[str, str]:
        return {
            "model": self.model.value,
            "sampler": self.sampler.value,
            "schedule": self.schedule.value,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionReceiptMetadata:
    """Explicit evidence required to build one execution receipt."""

    compatibility: CompatibilityDecision
    host: ExecutionHost
    model: ExecutionComponent
    sampler: ExecutionComponent
    rng_ownership: ExecutionRngOwnership
    requested_transitions: int
    effective_transitions: int
    requested_model_evaluations: int
    effective_model_evaluations: int
    status: ExecutionStatus
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.compatibility, CompatibilityDecision):
            raise ScheduleContractError("compatibility must be a CompatibilityDecision")
        if not isinstance(self.host, ExecutionHost):
            raise ScheduleContractError("host must be an ExecutionHost")
        if not isinstance(self.model, ExecutionComponent):
            raise ScheduleContractError("model must be an ExecutionComponent")
        if not isinstance(self.sampler, ExecutionComponent):
            raise ScheduleContractError("sampler must be an ExecutionComponent")
        if not isinstance(self.rng_ownership, ExecutionRngOwnership):
            raise ScheduleContractError("rng_ownership must be ExecutionRngOwnership")
        requested_transitions = _require_count(
            "requested_transitions",
            self.requested_transitions,
            positive=True,
        )
        effective_transitions = _require_count(
            "effective_transitions",
            self.effective_transitions,
            positive=False,
        )
        requested_evaluations = _require_count(
            "requested_model_evaluations",
            self.requested_model_evaluations,
            positive=True,
        )
        effective_evaluations = _require_count(
            "effective_model_evaluations",
            self.effective_model_evaluations,
            positive=False,
        )
        if effective_transitions > requested_transitions:
            raise ScheduleContractError("effective transitions exceed the request")
        if effective_evaluations > requested_evaluations:
            raise ScheduleContractError("effective model evaluations exceed the request")
        if not isinstance(self.status, ExecutionStatus):
            raise ScheduleContractError("execution status is unsupported")
        if self.reason_code is not None and (
            not isinstance(self.reason_code, str)
            or not _REASON_PATTERN.fullmatch(self.reason_code)
            or _SECRET_PATTERN.search(self.reason_code)
        ):
            raise ScheduleContractError("execution reason code is invalid")
        if self.status is ExecutionStatus.SUCCEEDED:
            if (
                effective_transitions != requested_transitions
                or effective_evaluations != requested_evaluations
                or self.reason_code is not None
            ):
                raise ScheduleContractError("succeeded execution requires complete counts")
        elif self.status is ExecutionStatus.NOT_EXECUTED:
            if effective_transitions != 0 or effective_evaluations != 0 or self.reason_code:
                raise ScheduleContractError("not-executed receipt cannot claim execution")
        elif self.reason_code is None:
            raise ScheduleContractError("failed or interrupted execution requires a reason code")


def _compatibility_projection(decision: CompatibilityDecision) -> dict[str, object]:
    return {
        "considered": [item.value for item in decision.considered],
        "level": decision.level.value,
        "reasons": [item.value for item in decision.reasons],
    }


def _object(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], fields: frozenset[str], *, field_name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{field_name} fields do not match its schema")


def _component_from_projection(value: object, *, field_name: str) -> ExecutionComponent:
    projection = _object(value, field_name=field_name)
    _exact(projection, frozenset({"fingerprint", "id", "version"}), field_name=field_name)
    return ExecutionComponent(
        identifier=cast(str, projection["id"]),
        version=cast(str, projection["version"]),
        fingerprint=cast(str, projection["fingerprint"]),
    )


def _host_from_projection(value: object) -> ExecutionHost:
    projection = _object(value, field_name="host")
    _exact(
        projection,
        frozenset({"api_version", "id", "revision", "version"}),
        field_name="host",
    )
    return ExecutionHost(
        identifier=cast(str, projection["id"]),
        version=cast(str, projection["version"]),
        revision=cast(str, projection["revision"]),
        api_version=cast(str, projection["api_version"]),
    )


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


def _validate_typed_compatibility_value(
    value: dict[str, object],
    *,
    field_name: str,
) -> None:
    _exact(value, frozenset({"bits", "precision"}), field_name=field_name)
    precision = value["precision"]
    bits = value["bits"]
    if precision not in {"float32", "float64"} or not isinstance(bits, str):
        raise ScheduleContractError(f"{field_name} is not a typed finite float")
    width = 8 if precision == "float32" else 16
    if len(bits) != width or bits != bits.casefold() or not re.fullmatch(r"[0-9a-f]+", bits):
        raise ScheduleContractError(f"{field_name} contains an invalid float token")
    decoded = struct.unpack(">f" if precision == "float32" else ">d", bytes.fromhex(bits))[0]
    if not math.isfinite(decoded) or (decoded == 0.0 and bits != "0" * width):
        raise ScheduleContractError(f"{field_name} must be finite and normalize negative zero")


def _validate_effective_compatibility(value: object) -> None:
    compatibility = _object(value, field_name="effective_inputs.compatibility")
    if len(compatibility) > 128:
        raise ScheduleContractError("effective_inputs.compatibility has too many fields")
    for name, child in compatibility.items():
        # SECURITY: reuse artifact field-name guards so receipts cannot carry secret/path keys.
        ArtifactField(name=name, value=None)
        field_name = f"effective_inputs.compatibility.{name}"
        if isinstance(child, dict):
            _validate_typed_compatibility_value(child, field_name=field_name)
        else:
            ArtifactField(name=name, value=cast(str | int | bool | None, child))


def _metadata_from_projection(
    projection: dict[str, object],
) -> ExecutionReceiptMetadata:
    counts = _object(projection["counts"], field_name="counts")
    _exact(
        counts,
        frozenset(
            {
                "effective_model_evaluations",
                "effective_transitions",
                "requested_model_evaluations",
                "requested_transitions",
            }
        ),
        field_name="counts",
    )
    execution = _object(projection["execution"], field_name="execution")
    _exact(execution, frozenset({"reason_code", "status"}), field_name="execution")
    rng = _object(projection["rng_ownership"], field_name="rng_ownership")
    _exact(rng, frozenset({"model", "sampler", "schedule"}), field_name="rng_ownership")
    try:
        ownership = ExecutionRngOwnership(
            schedule=NoiseOwnership(cast(str, rng["schedule"])),
            model=NoiseOwnership(cast(str, rng["model"])),
            sampler=NoiseOwnership(cast(str, rng["sampler"])),
        )
        status = ExecutionStatus(cast(str, execution["status"]))
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("receipt enum value is unsupported") from exc
    return ExecutionReceiptMetadata(
        compatibility=_compatibility_from_projection(projection["compatibility"]),
        host=_host_from_projection(projection["host"]),
        model=_component_from_projection(projection["model"], field_name="model"),
        sampler=_component_from_projection(projection["sampler"], field_name="sampler"),
        rng_ownership=ownership,
        requested_transitions=cast(int, counts["requested_transitions"]),
        effective_transitions=cast(int, counts["effective_transitions"]),
        requested_model_evaluations=cast(int, counts["requested_model_evaluations"]),
        effective_model_evaluations=cast(int, counts["effective_model_evaluations"]),
        status=status,
        reason_code=cast(str | None, execution["reason_code"]),
    )


def _validate_receipt_projection(projection: dict[str, object]) -> bytes:
    _exact(projection, _RECEIPT_FIELDS, field_name="receipt")
    if projection.get("schema") != _RECEIPT_SCHEMA:
        raise ScheduleContractError("execution receipt schema is unsupported")
    artifact = _object(projection["artifact"], field_name="artifact")
    _exact(
        artifact,
        frozenset({"construction_fingerprint", "numerical_fingerprint"}),
        field_name="artifact",
    )
    _require_fingerprint("construction fingerprint", artifact["construction_fingerprint"])
    _require_fingerprint("numerical fingerprint", artifact["numerical_fingerprint"])
    effective = _object(projection["effective_inputs"], field_name="effective_inputs")
    _exact(
        effective,
        frozenset(
            {
                "compatibility",
                "height",
                "precision",
                "profile",
                "profile_version",
                "steps",
                "width",
            }
        ),
        field_name="effective_inputs",
    )
    effective_steps = _require_count("effective_inputs.steps", effective["steps"], positive=True)
    for dimension_name in ("width", "height"):
        dimension = effective[dimension_name]
        if dimension is not None:
            _require_count(f"effective_inputs.{dimension_name}", dimension, positive=True)
    if (effective["width"] is None) != (effective["height"] is None):
        raise ScheduleContractError("effective_inputs dimensions must be supplied together")
    if effective["precision"] not in {"float32", "float64"}:
        raise ScheduleContractError("effective_inputs precision is unsupported")
    _validate_effective_compatibility(effective["compatibility"])
    profile = _object(projection["profile"], field_name="profile")
    _exact(profile, frozenset({"id", "version"}), field_name="profile")
    profile_id = _require_public_identifier("profile id", profile["id"])
    profile_version = _require_public_identifier("profile version", profile["version"])
    if (
        effective.get("profile") != profile_id
        or effective.get("profile_version") != profile_version
    ):
        raise ScheduleContractError("receipt profile disagrees with effective inputs")
    metadata = _metadata_from_projection(projection)
    if metadata.requested_transitions != effective_steps:
        raise ScheduleContractError("receipt transitions disagree with effective inputs")
    if (
        metadata.status is ExecutionStatus.SUCCEEDED
        and metadata.compatibility.level is CompatibilityLevel.REJECT
    ):
        raise ScheduleContractError("rejected compatibility cannot claim succeeded execution")
    return canonical_projection_bytes(projection)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionReceipt:
    """Immutable canonical record of one execution outcome."""

    receipt_bytes: bytes
    receipt_fingerprint: str
    construction_fingerprint: str
    numerical_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_bytes, bytes):
            raise ScheduleContractError("receipt projection must be canonical bytes")
        _require_fingerprint("receipt fingerprint", self.receipt_fingerprint)
        _require_fingerprint("construction fingerprint", self.construction_fingerprint)
        _require_fingerprint("numerical fingerprint", self.numerical_fingerprint)
        projection = _decode_object(self.receipt_bytes, maximum=_MAX_RECEIPT_BYTES)
        _validate_receipt_projection(projection)
        artifact = cast(dict[str, object], projection["artifact"])
        if artifact["construction_fingerprint"] != self.construction_fingerprint:
            raise ScheduleContractError("receipt construction fingerprint mismatch")
        if artifact["numerical_fingerprint"] != self.numerical_fingerprint:
            raise ScheduleContractError("receipt numerical fingerprint mismatch")
        if _sha256_identity(self.receipt_bytes) != self.receipt_fingerprint:
            raise ScheduleContractError("receipt fingerprint mismatch")

    def projection(self) -> dict[str, object]:
        return _decode_object(self.receipt_bytes, maximum=_MAX_RECEIPT_BYTES)


def build_execution_receipt(
    artifact: ScheduleArtifact,
    *,
    metadata: ExecutionReceiptMetadata,
) -> ExecutionReceipt:
    """Bind explicit execution evidence to one existing construction artifact."""

    if not isinstance(artifact, ScheduleArtifact):
        raise ScheduleContractError("artifact must be a ScheduleArtifact")
    if not isinstance(metadata, ExecutionReceiptMetadata):
        raise ScheduleContractError("metadata must be ExecutionReceiptMetadata")
    construction = artifact.construction_projection()
    effective = _object(construction["effective"], field_name="artifact effective inputs")
    artifact_steps = effective.get("steps")
    if metadata.requested_transitions != artifact_steps:
        raise ScheduleContractError("requested transitions disagree with the schedule artifact")
    if (
        metadata.status is ExecutionStatus.SUCCEEDED
        and metadata.compatibility.level is CompatibilityLevel.REJECT
    ):
        raise ScheduleContractError("rejected compatibility cannot claim succeeded execution")
    profile_id = effective.get("profile")
    profile_version = effective.get("profile_version")
    _require_public_identifier("profile id", profile_id)
    _require_public_identifier("profile version", profile_version)
    projection: dict[str, object] = {
        "artifact": {
            "construction_fingerprint": artifact.construction_fingerprint,
            "numerical_fingerprint": artifact.numerical_fingerprint,
        },
        "compatibility": _compatibility_projection(metadata.compatibility),
        "counts": {
            "effective_model_evaluations": metadata.effective_model_evaluations,
            "effective_transitions": metadata.effective_transitions,
            "requested_model_evaluations": metadata.requested_model_evaluations,
            "requested_transitions": metadata.requested_transitions,
        },
        "effective_inputs": effective,
        "execution": {
            "reason_code": metadata.reason_code,
            "status": metadata.status.value,
        },
        "host": metadata.host.projection(),
        "model": metadata.model.projection(),
        "profile": {"id": profile_id, "version": profile_version},
        "rng_ownership": metadata.rng_ownership.projection(),
        "sampler": metadata.sampler.projection(),
        "schema": _RECEIPT_SCHEMA,
    }
    receipt_bytes = _validate_receipt_projection(projection)
    return ExecutionReceipt(
        receipt_bytes=receipt_bytes,
        receipt_fingerprint=_sha256_identity(receipt_bytes),
        construction_fingerprint=artifact.construction_fingerprint,
        numerical_fingerprint=artifact.numerical_fingerprint,
    )


def serialize_execution_receipt(receipt: ExecutionReceipt) -> bytes:
    """Serialize one receipt inside its versioned canonical envelope."""

    if not isinstance(receipt, ExecutionReceipt):
        raise ScheduleContractError("receipt must be an ExecutionReceipt")
    return canonical_projection_bytes(
        {
            "receipt": receipt.projection(),
            "receipt_fingerprint": receipt.receipt_fingerprint,
            "schema": _RECEIPT_ENVELOPE_SCHEMA,
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


def _decode_object(payload: bytes | str, *, maximum: int) -> dict[str, object]:
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


def deserialize_execution_receipt(payload: bytes | str) -> ExecutionReceipt:
    """Strictly parse and verify one execution receipt envelope."""

    envelope = _decode_object(payload, maximum=_MAX_RECEIPT_BYTES)
    _exact(
        envelope,
        frozenset({"receipt", "receipt_fingerprint", "schema"}),
        field_name="receipt envelope",
    )
    if envelope.get("schema") != _RECEIPT_ENVELOPE_SCHEMA:
        raise ScheduleContractError("execution receipt envelope schema is unsupported")
    projection = _object(envelope["receipt"], field_name="receipt")
    receipt_bytes = _validate_receipt_projection(projection)
    artifact = cast(dict[str, object], projection["artifact"])
    return ExecutionReceipt(
        receipt_bytes=receipt_bytes,
        receipt_fingerprint=cast(str, envelope["receipt_fingerprint"]),
        construction_fingerprint=cast(str, artifact["construction_fingerprint"]),
        numerical_fingerprint=cast(str, artifact["numerical_fingerprint"]),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PortableExecutionBundle:
    """One portable construction artifact plus its matching execution receipt."""

    artifact: ScheduleArtifact
    receipt: ExecutionReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ScheduleArtifact):
            raise ScheduleContractError("bundle artifact must be ScheduleArtifact")
        if not isinstance(self.receipt, ExecutionReceipt):
            raise ScheduleContractError("bundle receipt must be ExecutionReceipt")
        if self.artifact.construction_fingerprint != self.receipt.construction_fingerprint:
            raise ScheduleContractError("bundle construction artifact fingerprint mismatch")
        if self.artifact.numerical_fingerprint != self.receipt.numerical_fingerprint:
            raise ScheduleContractError("bundle numerical artifact fingerprint mismatch")
        construction = self.artifact.construction_projection()
        receipt_projection = self.receipt.projection()
        if construction["effective"] != receipt_projection["effective_inputs"]:
            raise ScheduleContractError("bundle receipt effective inputs mismatch")


def serialize_portable_execution_bundle(bundle: PortableExecutionBundle) -> bytes:
    """Serialize an artifact and receipt without merging their schemas."""

    if not isinstance(bundle, PortableExecutionBundle):
        raise ScheduleContractError("bundle must be PortableExecutionBundle")
    artifact_envelope = json.loads(serialize_schedule_artifact(bundle.artifact))
    receipt_envelope = json.loads(serialize_execution_receipt(bundle.receipt))
    return canonical_projection_bytes(
        {
            "artifact": artifact_envelope,
            "receipt": receipt_envelope,
            "schema": _BUNDLE_SCHEMA,
        }
    )


def deserialize_portable_execution_bundle(payload: bytes | str) -> PortableExecutionBundle:
    """Strictly parse both transports and verify their cross-links."""

    projection = _decode_object(payload, maximum=_MAX_BUNDLE_BYTES)
    _exact(
        projection,
        frozenset({"artifact", "receipt", "schema"}),
        field_name="portable bundle",
    )
    if projection.get("schema") != _BUNDLE_SCHEMA:
        raise ScheduleContractError("portable bundle schema is unsupported")
    artifact_payload = canonical_projection_bytes(
        _object(projection["artifact"], field_name="artifact envelope")
    )
    receipt_payload = canonical_projection_bytes(
        _object(projection["receipt"], field_name="receipt envelope")
    )
    return PortableExecutionBundle(
        artifact=deserialize_schedule_artifact(artifact_payload),
        receipt=deserialize_execution_receipt(receipt_payload),
    )
