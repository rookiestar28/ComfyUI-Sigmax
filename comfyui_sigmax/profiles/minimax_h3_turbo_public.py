"""Readiness-only public MiniMax H3 Turbo bindings.

The M4-16 boundary deliberately exposes exact recipe metadata and pure sigma readiness while
keeping model-bound execution fail-closed until an M6-12 artifact is eligible.  This module never
loads, scans, hashes, or mutates a caller-owned model or LoRA path.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, cast

from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles.minimax_h3_acceleration import (
    MINIMAX_H3_ACCELERATION_ARTIFACTS,
    MiniMaxH3AccelerationArtifact,
    MiniMaxH3AccelerationDisposition,
    MiniMaxH3AccelerationError,
    MiniMaxH3AccelerationReasonCode,
    qualify_minimax_h3_candidate,
)
from comfyui_sigmax.profiles.minimax_h3_turbo import (
    MINIMAX_H3_TURBO_PROFILES,
    MiniMaxH3TurboError,
    MiniMaxH3TurboProfile,
    MiniMaxH3TurboReasonCode,
    get_minimax_h3_turbo_profile,
    validate_minimax_h3_turbo_artifact,
)

MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID: Final = "sigmax.minimax-h3-turbo-public/1"
MINIMAX_H3_TURBO_PUBLIC_SCHEMA_VERSION: Final = "1"
MINIMAX_H3_TURBO_PUBLIC_RECEIPT_LIMITATION: Final = (
    "model_execution_requires_eligible_exact_artifact_and_authorized_host"
)
MINIMAX_H3_TURBO_PUBLIC_TASKS: Final = ("fl2va", "t2va", "ref2va")
MINIMAX_H3_TURBO_RECIPE_IDS: Final = tuple(
    profile.recipe_id for profile in MINIMAX_H3_TURBO_PROFILES
)

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}$")
_CODE_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_PATH_PATTERN: Final = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)
_SECRET_PATTERN: Final = re.compile(
    r"(?:^|[_.-])(?:api_?key|access_key|private_key|secret|password|passwd|credential|"
    r"cookie|token|authorization|auth)(?:[_.-]|$)",
    re.IGNORECASE,
)
_MAX_RECEIPT_BYTES: Final = 1_048_576


class MiniMaxH3TurboPublicArtifactStatus(str, Enum):
    """Public artifact states; none of these imply model execution."""

    NOT_PROVIDED = "not_provided"
    ELIGIBLE = "eligible"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class MiniMaxH3TurboPublicValidationStatus(str, Enum):
    """Validation states retained in a portable receipt."""

    READINESS_ONLY = "readiness_only"
    ARTIFACT_ELIGIBLE = "artifact_eligible"
    ARTIFACT_BLOCKED = "artifact_blocked"
    ARTIFACT_REJECTED = "artifact_rejected"
    ARTIFACT_UNKNOWN = "artifact_unknown"
    ARTIFACT_HASH_UNVERIFIED = "artifact_hash_unverified"


def _public_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ScheduleContractError(f"{field_name} must be bounded public text")
    if any(ord(character) < 32 for character in value):
        raise ScheduleContractError(f"{field_name} contains a control character")
    if _PRIVATE_PATH_PATTERN.search(value) or _SECRET_PATTERN.search(value):
        raise ScheduleContractError(f"{field_name} must not contain private or secret-like text")
    return value


def _identifier(field_name: str, value: object) -> str:
    text = _public_text(field_name, value)
    if not text.isascii() or not _IDENTIFIER_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable public identifier")
    return text


def _code(field_name: str, value: object) -> str:
    text = _public_text(field_name, value)
    if not text.isascii() or not _CODE_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable public code")
    return text


def _fingerprint(field_name: str, value: object) -> str:
    text = _public_text(field_name, value)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a SHA-256 fingerprint")
    return text


def canonical_minimax_h3_turbo_task(value: object) -> str:
    """Map source/display task aliases to the frozen lowercase public vocabulary."""

    if not isinstance(value, str):
        raise ScheduleContractError("MiniMax H3 Turbo task must be text")
    aliases = {
        "fl2va": "fl2va",
        "FL2VA": "fl2va",
        "t2va": "t2va",
        "T2VA": "t2va",
        "ref2va": "ref2va",
        "Ref2VA": "ref2va",
    }
    task = aliases.get(value)
    if task is None:
        raise ScheduleContractError("MiniMax H3 Turbo task is unsupported; I2VA is not admitted")
    return task


def _recipe_task_matches(profile: MiniMaxH3TurboProfile, task: str) -> bool:
    # M6-13 source metadata calls the text-to-video family FL2VA.  T2VA is an explicitly mapped
    # public alias for that same source task, not a second recipe or a new execution claim.
    if profile.task == "fl2va":
        return task in {"fl2va", "t2va"}
    return task == profile.task


def _bounded_nfe(profile: MiniMaxH3TurboProfile, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in profile.allowed_nfe:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.UNSUPPORTED_RECIPE_NFE,
            "nfe is not independently proven for this recipe profile",
        )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3TurboPublicReceiptV1:
    """Allowlisted, deterministic receipt for one pure Turbo readiness projection."""

    profile_id: str
    recipe_id: str
    task: str
    artifact_status: MiniMaxH3TurboPublicArtifactStatus
    artifact_id: str | None
    artifact_sha256: str | None
    allowed_nfe: tuple[int, ...]
    nfe: int
    video_shift: float
    audio_shift: float
    resolution_policy: str
    reference_policy: str
    sampler: str
    validation_status: MiniMaxH3TurboPublicValidationStatus
    schedule_fingerprint: str
    limitation: str
    evidence: str
    receipt_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier("receipt profile_id", self.profile_id)
        _identifier("receipt recipe_id", self.recipe_id)
        task = canonical_minimax_h3_turbo_task(self.task)
        if task != self.task:
            raise ScheduleContractError("receipt task must use the canonical lowercase spelling")
        if not isinstance(self.artifact_status, MiniMaxH3TurboPublicArtifactStatus):
            raise ScheduleContractError("receipt artifact status is unsupported")
        if not isinstance(self.validation_status, MiniMaxH3TurboPublicValidationStatus):
            raise ScheduleContractError("receipt validation status is unsupported")
        if self.artifact_id is not None:
            _identifier("receipt artifact_id", self.artifact_id)
        if self.artifact_sha256 is not None:
            _fingerprint("receipt artifact_sha256", self.artifact_sha256)
        if not self.allowed_nfe or self.allowed_nfe != tuple(sorted(set(self.allowed_nfe))):
            raise ScheduleContractError("receipt allowed_nfe must be sorted and unique")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in self.allowed_nfe
        ):
            raise ScheduleContractError("receipt allowed_nfe must contain positive integers")
        if self.nfe not in self.allowed_nfe:
            raise ScheduleContractError("receipt nfe is not allowed by the recipe")
        for field_name, value in (
            ("video_shift", self.video_shift),
            ("audio_shift", self.audio_shift),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ScheduleContractError(f"receipt {field_name} must be finite")
        for field_name, text_value in (
            ("resolution_policy", self.resolution_policy),
            ("reference_policy", self.reference_policy),
            ("sampler", self.sampler),
            ("limitation", self.limitation),
            ("evidence", self.evidence),
        ):
            _code(f"receipt {field_name}", text_value)
        _fingerprint("receipt schedule_fingerprint", self.schedule_fingerprint)
        body = _canonical(self._body_projection())
        object.__setattr__(
            self,
            "receipt_fingerprint",
            "sha256:" + hashlib.sha256(body).hexdigest(),
        )

    def _body_projection(self) -> dict[str, object]:
        return {
            "allowed_nfe": list(self.allowed_nfe),
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "artifact_status": self.artifact_status.value,
            "audio_shift": self.audio_shift,
            "evidence": self.evidence,
            "limitation": self.limitation,
            "nfe": self.nfe,
            "profile_id": self.profile_id,
            "recipe_id": self.recipe_id,
            "reference_policy": self.reference_policy,
            "resolution_policy": self.resolution_policy,
            "sampler": self.sampler,
            "schedule_fingerprint": self.schedule_fingerprint,
            "schema": MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID,
            "task": self.task,
            "validation_status": self.validation_status.value,
            "version": MINIMAX_H3_TURBO_PUBLIC_SCHEMA_VERSION,
            "video_shift": self.video_shift,
        }

    def projection(self) -> dict[str, object]:
        return {**self._body_projection(), "receipt_fingerprint": self.receipt_fingerprint}


def _canonical(value: dict[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError(
            "MiniMax H3 Turbo public receipt is not canonical JSON"
        ) from exc
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise ScheduleContractError("MiniMax H3 Turbo public receipt is too large")
    return encoded


def _artifact(artifact_id: str) -> MiniMaxH3AccelerationArtifact | None:
    return next(
        (item for item in MINIMAX_H3_ACCELERATION_ARTIFACTS if item.artifact_id == artifact_id),
        None,
    )


def build_minimax_h3_turbo_public_receipt(
    *,
    recipe_id: str,
    nfe: int,
    schedule_fingerprint: str,
    task: str | None = None,
    artifact_id: str | None = None,
    artifact_sha256: str | None = None,
) -> MiniMaxH3TurboPublicReceiptV1:
    """Build a redacted readiness receipt without promoting an artifact or touching a host."""

    profile = get_minimax_h3_turbo_profile(recipe_id)
    selected_nfe = _bounded_nfe(profile, nfe)
    selected_task = canonical_minimax_h3_turbo_task(profile.task if task is None else task)
    if not _recipe_task_matches(profile, selected_task):
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.WRONG_TASK,
            "public task does not match the source recipe",
        )
    _fingerprint("schedule_fingerprint", schedule_fingerprint)

    selected_artifact_status = MiniMaxH3TurboPublicArtifactStatus.NOT_PROVIDED
    selected_validation_status = MiniMaxH3TurboPublicValidationStatus.READINESS_ONLY
    selected_artifact_sha256: str | None = None
    if artifact_id is not None:
        _identifier("artifact_id", artifact_id)
        if artifact_sha256 is not None:
            _fingerprint("artifact_sha256", artifact_sha256)
        artifact = _artifact(artifact_id)
        if artifact is None or artifact.recipe_id != profile.recipe_id:
            selected_artifact_status = MiniMaxH3TurboPublicArtifactStatus.REJECTED
            selected_validation_status = MiniMaxH3TurboPublicValidationStatus.ARTIFACT_UNKNOWN
        else:
            if artifact_sha256 is None:
                selected_artifact_status = MiniMaxH3TurboPublicArtifactStatus.REJECTED
                selected_validation_status = (
                    MiniMaxH3TurboPublicValidationStatus.ARTIFACT_HASH_UNVERIFIED
                )
            elif artifact_sha256 != artifact.sha256:
                raise MiniMaxH3AccelerationError(
                    MiniMaxH3AccelerationReasonCode.SIZE_HASH_MISMATCH,
                    "artifact SHA-256 does not match the exact source record",
                )
            else:
                selected_artifact_sha256 = artifact.sha256
            if (
                artifact_sha256 == artifact.sha256
                and artifact.disposition is MiniMaxH3AccelerationDisposition.QUALIFIED
            ):
                selected_artifact_status = MiniMaxH3TurboPublicArtifactStatus.ELIGIBLE
                selected_validation_status = MiniMaxH3TurboPublicValidationStatus.ARTIFACT_ELIGIBLE
            elif (
                artifact_sha256 == artifact.sha256
                and artifact.disposition is MiniMaxH3AccelerationDisposition.BLOCKED
            ):
                selected_artifact_status = MiniMaxH3TurboPublicArtifactStatus.BLOCKED
                selected_validation_status = MiniMaxH3TurboPublicValidationStatus.ARTIFACT_BLOCKED
            elif artifact_sha256 == artifact.sha256:
                selected_artifact_status = MiniMaxH3TurboPublicArtifactStatus.REJECTED
                selected_validation_status = MiniMaxH3TurboPublicValidationStatus.ARTIFACT_REJECTED

    return MiniMaxH3TurboPublicReceiptV1(
        profile_id=profile.profile_id,
        recipe_id=profile.recipe_id,
        task=selected_task,
        artifact_status=selected_artifact_status,
        artifact_id=artifact_id,
        artifact_sha256=selected_artifact_sha256,
        allowed_nfe=profile.allowed_nfe,
        nfe=selected_nfe,
        video_shift=profile.video_shift,
        audio_shift=profile.audio_shift,
        resolution_policy=profile.resolution_policy,
        reference_policy=profile.reference_policy,
        sampler="euler",
        validation_status=selected_validation_status,
        schedule_fingerprint=schedule_fingerprint,
        limitation=MINIMAX_H3_TURBO_PUBLIC_RECEIPT_LIMITATION,
        evidence=profile.evidence.value,
    )


def serialize_minimax_h3_turbo_public_receipt(
    receipt: MiniMaxH3TurboPublicReceiptV1,
) -> bytes:
    if not isinstance(receipt, MiniMaxH3TurboPublicReceiptV1):
        raise ScheduleContractError("MiniMax H3 Turbo receipt serialization needs a receipt")
    return _canonical(receipt.projection())


def _mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], fields: set[str], *, field_name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{field_name} fields do not match the public receipt schema")


def _enum_value(enum_type: type[Enum], value: object, *, field_name: str) -> Enum:
    if not isinstance(value, str):
        raise ScheduleContractError(f"{field_name} must be text")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ScheduleContractError(f"{field_name} is unsupported") from exc


def deserialize_minimax_h3_turbo_public_receipt(
    payload: bytes | str,
) -> MiniMaxH3TurboPublicReceiptV1:
    """Parse and fingerprint-check one strict public receipt projection."""

    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raise ScheduleContractError("MiniMax H3 Turbo receipt payload must be bytes or text")
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ScheduleContractError("MiniMax H3 Turbo receipt payload is too large")
    try:
        loaded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScheduleContractError("MiniMax H3 Turbo receipt payload is not JSON") from exc
    projection = _mapping(loaded, field_name="MiniMax H3 Turbo receipt")
    _exact(
        projection,
        {
            "allowed_nfe",
            "artifact_id",
            "artifact_sha256",
            "artifact_status",
            "audio_shift",
            "evidence",
            "limitation",
            "nfe",
            "profile_id",
            "recipe_id",
            "receipt_fingerprint",
            "reference_policy",
            "resolution_policy",
            "sampler",
            "schema",
            "schedule_fingerprint",
            "task",
            "validation_status",
            "version",
            "video_shift",
        },
        field_name="MiniMax H3 Turbo receipt",
    )
    if projection["schema"] != MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID:
        raise ScheduleContractError("MiniMax H3 Turbo public receipt schema is unsupported")
    if projection["version"] != MINIMAX_H3_TURBO_PUBLIC_SCHEMA_VERSION:
        raise ScheduleContractError("MiniMax H3 Turbo public receipt version is unsupported")
    allowed_nfe = projection["allowed_nfe"]
    if not isinstance(allowed_nfe, list):
        raise ScheduleContractError("receipt allowed_nfe must be an array")
    receipt = MiniMaxH3TurboPublicReceiptV1(
        profile_id=cast(str, projection["profile_id"]),
        recipe_id=cast(str, projection["recipe_id"]),
        task=cast(str, projection["task"]),
        artifact_status=cast(
            MiniMaxH3TurboPublicArtifactStatus,
            _enum_value(
                MiniMaxH3TurboPublicArtifactStatus,
                projection["artifact_status"],
                field_name="receipt artifact_status",
            ),
        ),
        artifact_id=cast(str | None, projection["artifact_id"]),
        artifact_sha256=cast(str | None, projection["artifact_sha256"]),
        allowed_nfe=tuple(cast(int, item) for item in allowed_nfe),
        nfe=cast(int, projection["nfe"]),
        video_shift=cast(float, projection["video_shift"]),
        audio_shift=cast(float, projection["audio_shift"]),
        resolution_policy=cast(str, projection["resolution_policy"]),
        reference_policy=cast(str, projection["reference_policy"]),
        sampler=cast(str, projection["sampler"]),
        validation_status=cast(
            MiniMaxH3TurboPublicValidationStatus,
            _enum_value(
                MiniMaxH3TurboPublicValidationStatus,
                projection["validation_status"],
                field_name="receipt validation_status",
            ),
        ),
        schedule_fingerprint=cast(str, projection["schedule_fingerprint"]),
        limitation=cast(str, projection["limitation"]),
        evidence=cast(str, projection["evidence"]),
    )
    if not isinstance(projection["receipt_fingerprint"], str):
        raise ScheduleContractError("receipt fingerprint must be text")
    if projection["receipt_fingerprint"] != receipt.receipt_fingerprint:
        raise ScheduleContractError("MiniMax H3 Turbo public receipt fingerprint does not match")
    return receipt


def require_minimax_h3_turbo_artifact(
    *, recipe_id: str, artifact_id: str, artifact_sha256: str, nfe: int
) -> MiniMaxH3AccelerationArtifact:
    """Require an exact eligible M6-12 artifact before any future host workflow execution."""

    profile = get_minimax_h3_turbo_profile(recipe_id)
    validate_minimax_h3_turbo_artifact(profile, artifact_id, nfe=nfe)
    return qualify_minimax_h3_candidate(
        candidate_id=artifact_id,
        task=profile.task,
        nfe=nfe,
        video_shift=profile.video_shift,
        audio_shift=profile.audio_shift,
        artifact_sha256=artifact_sha256,
        resolution_policy=profile.resolution_policy,
        require_eligible=True,
    )


__all__ = [
    "MINIMAX_H3_TURBO_PUBLIC_RECEIPT_LIMITATION",
    "MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID",
    "MINIMAX_H3_TURBO_PUBLIC_SCHEMA_VERSION",
    "MINIMAX_H3_TURBO_PUBLIC_TASKS",
    "MINIMAX_H3_TURBO_RECIPE_IDS",
    "MiniMaxH3TurboPublicArtifactStatus",
    "MiniMaxH3TurboPublicReceiptV1",
    "MiniMaxH3TurboPublicValidationStatus",
    "build_minimax_h3_turbo_public_receipt",
    "canonical_minimax_h3_turbo_task",
    "deserialize_minimax_h3_turbo_public_receipt",
    "require_minimax_h3_turbo_artifact",
    "serialize_minimax_h3_turbo_public_receipt",
]
