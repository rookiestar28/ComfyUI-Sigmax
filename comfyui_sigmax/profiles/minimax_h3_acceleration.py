"""Fail-closed MiniMax H3 acceleration qualification contracts.

This module is an evidence seam for the M6-12 successor lane.  It deliberately is not imported
by the public profile package, registry, node mappings, or workflow builders.  The records describe
what a later Turbo/profile or backend item may consume; they do not load LoRAs, select kernels, or
claim model, image, audio, or performance support.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError

MINIMAX_H3_ACCELERATION_SCHEMA_ID: Final = "sigmax.minimax-h3-acceleration/1"
MINIMAX_H3_ACCELERATION_SCHEMA_VERSION: Final = "1"
MINIMAX_H3_ACCELERATION_OBSERVED_ON: Final = "2026-08-18"

_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_PRIVATE_PATH_PATTERN: Final = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)


class MiniMaxH3AccelerationSourceRole(str, Enum):
    """Evidence owner for one source pin."""

    RECIPE = "recipe"
    ARTIFACT = "artifact"
    MODEL = "model"
    HOST = "host"
    BACKEND = "backend"
    ATTENTION = "attention"


class MiniMaxH3AccelerationLicenseScope(str, Enum):
    """License scope; software permission never implies weight permission."""

    SOFTWARE = "software"
    MODEL_WEIGHTS = "model_weights"
    CONVERSION = "conversion"


class MiniMaxH3AccelerationLicenseStatus(str, Enum):
    """Whether the observed license boundary is sufficient for downstream eligibility."""

    CONFIRMED = "confirmed"
    UNKNOWN = "unknown"


class MiniMaxH3AccelerationArtifactKind(str, Enum):
    """Artifact provenance class."""

    PUBLISHER_FULL = "publisher_full"
    REDUCED_EXACT = "reduced_exact"
    LOCAL_MODIFIED = "local_modified"


class MiniMaxH3AccelerationDisposition(str, Enum):
    """M6-12 outcome; QUALIFIED is readiness only, not runtime support."""

    QUALIFIED = "qualified"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class MiniMaxH3AccelerationBackendScope(str, Enum):
    """Independent host acceleration axes."""

    QUANTIZED_OPERATIONS = "quantized_operations"
    INT8_ATTENTION = "int8_attention"
    CORE_ATTENTION = "core_attention"
    MODEL_SPARSE_ATTENTION = "model_sparse_attention"


class MiniMaxH3AccelerationReasonCode(str, Enum):
    """Stable fail-closed reasons consumed by later planning and receipt layers."""

    UNPINNED_SOURCE = "UNPINNED_SOURCE"
    UNKNOWN_ARTIFACT_HASH = "UNKNOWN_ARTIFACT_HASH"
    SIZE_HASH_MISMATCH = "SIZE_HASH_MISMATCH"
    FILENAME_ONLY_IDENTITY = "FILENAME_ONLY_IDENTITY"
    TASK_METADATA_CONFLICT = "TASK_METADATA_CONFLICT"
    WRONG_TASK = "WRONG_TASK"
    UNAVAILABLE_EXACT_ARTIFACT = "UNAVAILABLE_EXACT_ARTIFACT"
    LOCAL_MODIFICATION = "LOCAL_MODIFICATION"
    DUPLICATE_SCALE_RISK = "DUPLICATE_SCALE_RISK"
    DUPLICATE_SHIFT_RISK = "DUPLICATE_SHIFT_RISK"
    UNVERIFIED_LICENSE = "UNVERIFIED_LICENSE"
    UNPROVEN_MSA_H3_LINK = "UNPROVEN_MSA_H3_LINK"
    UNSUPPORTED_RECIPE_NFE = "UNSUPPORTED_RECIPE_NFE"
    RESOLUTION_POLICY_MISMATCH = "RESOLUTION_POLICY_MISMATCH"
    MISSING_OWNERSHIP = "MISSING_OWNERSHIP"
    BACKEND_SCOPE_MISMATCH = "BACKEND_SCOPE_MISMATCH"


class MiniMaxH3AccelerationError(ScheduleContractError):
    """Fail-closed qualification error with a stable machine-readable reason."""

    def __init__(self, reason_code: MiniMaxH3AccelerationReasonCode, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code.value}: {message}")


def _require_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScheduleContractError(f"{field_name} must be a non-empty string")
    if _PRIVATE_PATH_PATTERN.search(value):
        raise ScheduleContractError(f"{field_name} must not contain a private path")
    return value


def _require_identifier(field_name: str, value: object) -> str:
    text = _require_text(field_name, value)
    if not text.isascii() or not _IDENTIFIER_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable lowercase identifier")
    return text


def _require_https(field_name: str, value: object) -> str:
    text = _require_text(field_name, value)
    if not text.startswith("https://"):
        raise ScheduleContractError(f"{field_name} must use HTTPS")
    return text


def _require_commit(field_name: str, value: object) -> str:
    text = _require_text(field_name, value)
    if not _COMMIT_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a pinned lowercase 40-hex revision")
    return text


def _require_sha256(field_name: str, value: object) -> str:
    text = _require_text(field_name, value)
    if not _SHA256_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a lowercase 64-hex SHA-256")
    return text


def _require_public_tuple(field_name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise ScheduleContractError(f"{field_name} must be a non-empty tuple")
    if not all(isinstance(item, str) for item in value):
        raise ScheduleContractError(f"{field_name} must contain strings")
    normalized = tuple(_require_text(field_name, item) for item in value)
    if normalized != tuple(sorted(set(normalized))):
        raise ScheduleContractError(f"{field_name} must be sorted and unique")
    return normalized


def _require_finite(field_name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScheduleContractError(f"{field_name} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0.0):
        raise ScheduleContractError(f"{field_name} must be finite and positive")
    return numeric


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3AccelerationSource:
    """One exact public source pin and its evidence/license role."""

    source_id: str
    role: MiniMaxH3AccelerationSourceRole
    url: str
    revision: str
    locators: tuple[str, ...]
    license_id: str
    license_status: MiniMaxH3AccelerationLicenseStatus = (
        MiniMaxH3AccelerationLicenseStatus.CONFIRMED
    )

    def __post_init__(self) -> None:
        _require_identifier("source_id", self.source_id)
        if not isinstance(self.role, MiniMaxH3AccelerationSourceRole):
            raise ScheduleContractError("source role is unsupported")
        _require_https("source URL", self.url)
        _require_commit("source revision", self.revision)
        _require_public_tuple("source locators", self.locators)
        _require_text("source license_id", self.license_id)
        if not isinstance(self.license_status, MiniMaxH3AccelerationLicenseStatus):
            raise ScheduleContractError("source license status is unsupported")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3AccelerationLicense:
    """A separately scoped software, model-weight, or conversion license."""

    scope: MiniMaxH3AccelerationLicenseScope
    identifier: str
    status: MiniMaxH3AccelerationLicenseStatus

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MiniMaxH3AccelerationLicenseScope):
            raise ScheduleContractError("license scope is unsupported")
        _require_text("license identifier", self.identifier)
        if not isinstance(self.status, MiniMaxH3AccelerationLicenseStatus):
            raise ScheduleContractError("license status is unsupported")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3AccelerationRecipe:
    """Source-qualified recipe identity, without a runtime profile registration."""

    recipe_id: str
    model_family: str
    model_variant: str
    task: str
    lora_task: str
    evidence: EvidenceLevel
    source_id: str
    allowed_nfe: tuple[int, ...]
    default_nfe: int
    video_shift: float
    audio_shift: float
    resolution_policy: str
    reference_policy: str
    sampler: str
    schedule_owner: str
    audio_owner: str
    lora_owner: str
    attention_owner: str
    transform_order: str
    terminal_policy: str
    disposition: MiniMaxH3AccelerationDisposition = MiniMaxH3AccelerationDisposition.QUALIFIED
    reason_codes: tuple[MiniMaxH3AccelerationReasonCode, ...] = ()
    runtime_registered: bool = False

    def __post_init__(self) -> None:
        _require_identifier("recipe_id", self.recipe_id)
        if self.model_family != "minimax_h3":
            raise ScheduleContractError("recipe model family is unsupported")
        if self.model_variant not in {"fl2va", "ref2va"} or self.task not in {"fl2va", "ref2va"}:
            raise ScheduleContractError("recipe task/variant is unsupported")
        if self.lora_task not in {"fl2va", "ref2va"}:
            raise ScheduleContractError("recipe LoRA task is unsupported")
        if self.task != self.lora_task or self.model_variant != self.task:
            raise ScheduleContractError("recipe task and model/LoRA task must agree")
        if not isinstance(self.evidence, EvidenceLevel):
            raise ScheduleContractError("recipe evidence is unsupported")
        _require_identifier("recipe source_id", self.source_id)
        if (
            not isinstance(self.allowed_nfe, tuple)
            or not self.allowed_nfe
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.allowed_nfe
            )
            or self.allowed_nfe != tuple(sorted(set(self.allowed_nfe)))
        ):
            raise ScheduleContractError("recipe allowed_nfe must be sorted positive integers")
        if self.default_nfe not in self.allowed_nfe:
            raise ScheduleContractError("recipe default_nfe must be allowed")
        _require_finite("recipe video_shift", self.video_shift, positive=True)
        _require_finite("recipe audio_shift", self.audio_shift, positive=True)
        for name, value in (
            ("resolution_policy", self.resolution_policy),
            ("reference_policy", self.reference_policy),
            ("sampler", self.sampler),
            ("schedule_owner", self.schedule_owner),
            ("audio_owner", self.audio_owner),
            ("lora_owner", self.lora_owner),
            ("attention_owner", self.attention_owner),
            ("transform_order", self.transform_order),
            ("terminal_policy", self.terminal_policy),
        ):
            _require_text(name, value)
        if not isinstance(self.disposition, MiniMaxH3AccelerationDisposition):
            raise ScheduleContractError("recipe disposition is unsupported")
        if not isinstance(self.reason_codes, tuple) or any(
            not isinstance(reason, MiniMaxH3AccelerationReasonCode) for reason in self.reason_codes
        ):
            raise ScheduleContractError("recipe reason codes are invalid")
        if self.reason_codes != tuple(
            sorted(set(self.reason_codes), key=lambda reason: reason.value)
        ):
            raise ScheduleContractError("recipe reason codes must be sorted and unique")
        if self.disposition is MiniMaxH3AccelerationDisposition.QUALIFIED and self.reason_codes:
            raise ScheduleContractError("qualified recipe cannot carry blocking reason codes")
        if (
            self.disposition is not MiniMaxH3AccelerationDisposition.QUALIFIED
            and not self.reason_codes
        ):
            raise ScheduleContractError("blocked/rejected recipe requires a reason code")
        if self.runtime_registered is not False:
            raise ScheduleContractError("M6-12 recipes cannot be runtime registered")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3AccelerationArtifact:
    """Exact artifact identity and disposition; payloads are never loaded here."""

    artifact_id: str
    recipe_id: str
    kind: MiniMaxH3AccelerationArtifactKind
    source_id: str
    filename: str
    sha256: str
    size_bytes: int | None
    declared_task: str
    metadata_task: str
    tensor_count: int
    tensor_layout: str
    rank_policy: str
    baked_scale: float
    loader_strength: float
    conversion_identity: str
    weight_license: MiniMaxH3AccelerationLicense
    conversion_license: MiniMaxH3AccelerationLicense
    disposition: MiniMaxH3AccelerationDisposition
    reason_codes: tuple[MiniMaxH3AccelerationReasonCode, ...]
    runtime_registered: bool = False

    def __post_init__(self) -> None:
        _require_identifier("artifact_id", self.artifact_id)
        _require_identifier("artifact recipe_id", self.recipe_id)
        if not isinstance(self.kind, MiniMaxH3AccelerationArtifactKind):
            raise ScheduleContractError("artifact kind is unsupported")
        _require_identifier("artifact source_id", self.source_id)
        filename = _require_text("artifact filename", self.filename)
        if "\\" in filename or filename.startswith("/") or ":" in filename:
            raise ScheduleContractError("artifact filename must be a source filename, not a path")
        _require_sha256("artifact sha256", self.sha256)
        if self.size_bytes is not None and (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise ScheduleContractError("artifact size_bytes must be positive when present")
        if self.declared_task not in {"fl2va", "ref2va"} or self.metadata_task not in {
            "fl2va",
            "ref2va",
        }:
            raise ScheduleContractError("artifact task is unsupported")
        if (
            isinstance(self.tensor_count, bool)
            or not isinstance(self.tensor_count, int)
            or self.tensor_count <= 0
        ):
            raise ScheduleContractError("artifact tensor_count must be positive")
        _require_text("artifact tensor_layout", self.tensor_layout)
        _require_text("artifact rank_policy", self.rank_policy)
        _require_finite("artifact baked_scale", self.baked_scale, positive=True)
        _require_finite("artifact loader_strength", self.loader_strength, positive=True)
        _require_text("artifact conversion_identity", self.conversion_identity)
        if not isinstance(self.weight_license, MiniMaxH3AccelerationLicense) or not isinstance(
            self.conversion_license, MiniMaxH3AccelerationLicense
        ):
            raise ScheduleContractError("artifact license boundaries are invalid")
        if not isinstance(self.disposition, MiniMaxH3AccelerationDisposition):
            raise ScheduleContractError("artifact disposition is unsupported")
        if not isinstance(self.reason_codes, tuple) or any(
            not isinstance(reason, MiniMaxH3AccelerationReasonCode) for reason in self.reason_codes
        ):
            raise ScheduleContractError("artifact reason codes are invalid")
        if self.reason_codes != tuple(
            sorted(set(self.reason_codes), key=lambda reason: reason.value)
        ):
            raise ScheduleContractError("artifact reason codes must be sorted and unique")
        if self.disposition is MiniMaxH3AccelerationDisposition.QUALIFIED and self.reason_codes:
            raise ScheduleContractError("qualified artifact cannot carry blocking reason codes")
        if (
            self.disposition is not MiniMaxH3AccelerationDisposition.QUALIFIED
            and not self.reason_codes
        ):
            raise ScheduleContractError("blocked/rejected artifact requires a reason code")
        if self.runtime_registered is not False:
            raise ScheduleContractError("M6-12 artifacts cannot be runtime registered")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3AccelerationBackend:
    """Backend/attention readiness observation without selecting or installing a backend."""

    backend_id: str
    scope: MiniMaxH3AccelerationBackendScope
    source_id: str
    disposition: MiniMaxH3AccelerationDisposition
    reason_codes: tuple[MiniMaxH3AccelerationReasonCode, ...]
    runtime_selected: bool = False

    def __post_init__(self) -> None:
        _require_identifier("backend_id", self.backend_id)
        if not isinstance(self.scope, MiniMaxH3AccelerationBackendScope):
            raise ScheduleContractError("backend scope is unsupported")
        _require_identifier("backend source_id", self.source_id)
        if not isinstance(self.disposition, MiniMaxH3AccelerationDisposition):
            raise ScheduleContractError("backend disposition is unsupported")
        if not isinstance(self.reason_codes, tuple) or any(
            not isinstance(reason, MiniMaxH3AccelerationReasonCode) for reason in self.reason_codes
        ):
            raise ScheduleContractError("backend reason codes are invalid")
        if self.runtime_selected is not False:
            raise ScheduleContractError("M6-12 cannot select a backend")


def _source(
    source_id: str,
    role: MiniMaxH3AccelerationSourceRole,
    url: str,
    revision: str,
    license_id: str,
    *locators: str,
    license_status: MiniMaxH3AccelerationLicenseStatus = MiniMaxH3AccelerationLicenseStatus.CONFIRMED,
) -> MiniMaxH3AccelerationSource:
    return MiniMaxH3AccelerationSource(
        source_id=source_id,
        role=role,
        url=url,
        revision=revision,
        locators=tuple(sorted(locators)),
        license_id=license_id,
        license_status=license_status,
    )


MINIMAX_H3_ACCELERATION_SOURCES: Final = tuple(
    sorted(
        (
            _source(
                "modeltc.minimax-h3-turbo",
                MiniMaxH3AccelerationSourceRole.RECIPE,
                "https://github.com/ModelTC/Minimax-H3-Turbo",
                "a7e148b8dc7db8ad976966060dcc022adf11fc8d",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
                "comfyui_workflows/",
            ),
            _source(
                "lightx2v.minimax-h3-turbo",
                MiniMaxH3AccelerationSourceRole.ARTIFACT,
                "https://huggingface.co/lightx2v/Minimax-h3-Turbo",
                "5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1",  # pragma: allowlist secret
                "Apache-2.0",
                "README.md",
                "*.safetensors",
            ),
            _source(
                "minimaxai.minimax-h3",
                MiniMaxH3AccelerationSourceRole.MODEL,
                "https://huggingface.co/MiniMaxAI/MiniMax-H3",
                "42ed227ee7df40d41602854ae760620d6eb651fe",  # pragma: allowlist secret
                "LicenseRef-MiniMax-H3-Community",
                "LICENSE",
                "README.md",
            ),
            _source(
                "kijai.minimax-h3-comfy",
                MiniMaxH3AccelerationSourceRole.ARTIFACT,
                "https://huggingface.co/Kijai/MiniMax-H3_comfy",
                "2dc3cedb9b58b0e448d9e950f794f25bf28dbbb5",  # pragma: allowlist secret
                "unknown",
                "README.md",
                "*.safetensors",
                license_status=MiniMaxH3AccelerationLicenseStatus.UNKNOWN,
            ),
            _source(
                "comfyui.repository",
                MiniMaxH3AccelerationSourceRole.HOST,
                "https://github.com/comfyanonymous/ComfyUI",
                "c1739380c6fab78e7e263cb665d04aafbfe24593",  # pragma: allowlist secret
                "GPL-3.0-only",
                "comfy/attention.py",
                "comfy/supported_models.py",
            ),
            _source(
                "comfy-kitchen.repository",
                MiniMaxH3AccelerationSourceRole.BACKEND,
                "https://github.com/Comfy-Org/comfy-kitchen",
                "cfcc843b6e8ec1e119b8fe8f7f8f6a46dad8599e",  # pragma: allowlist secret
                "Apache-2.0",
                "comfy_kitchen/attention.py",
                "pyproject.toml",
            ),
            _source(
                "minimaxai.msa",
                MiniMaxH3AccelerationSourceRole.ATTENTION,
                "https://github.com/MiniMax-AI/MSA",
                "80434d7f67877c6570ca19cac444b84bc9855dac",  # pragma: allowlist secret
                "MIT",
                "README.md",
            ),
            _source(
                "thu-ml.sageattention",
                MiniMaxH3AccelerationSourceRole.ATTENTION,
                "https://github.com/thu-ml/SageAttention",
                "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5",  # pragma: allowlist secret
                "unknown",
                "README.md",
                license_status=MiniMaxH3AccelerationLicenseStatus.UNKNOWN,
            ),
            _source(
                "dao-ai-lab.flash-attention",
                MiniMaxH3AccelerationSourceRole.ATTENTION,
                "https://github.com/Dao-AILab/flash-attention",
                "0251105a2fb19d2957484b7f023cd8c115286ced",  # pragma: allowlist secret
                "BSD-3-Clause",
                "README.md",
            ),
            _source(
                "kijai.comfyui-kjnodes",
                MiniMaxH3AccelerationSourceRole.ATTENTION,
                "https://github.com/kijai/ComfyUI-KJNodes",
                "3f20054214fec9f9234fd3841ae6f1e4287948f6",  # pragma: allowlist secret
                "unknown",
                "README.md",
                license_status=MiniMaxH3AccelerationLicenseStatus.UNKNOWN,
            ),
        ),
        key=lambda source: source.source_id,
    )
)


def _recipe(
    recipe_id: str,
    task: str,
    allowed_nfe: tuple[int, ...],
    default_nfe: int,
    video_shift: float,
    resolution_policy: str,
) -> MiniMaxH3AccelerationRecipe:
    return MiniMaxH3AccelerationRecipe(
        recipe_id=recipe_id,
        model_family="minimax_h3",
        model_variant=task,
        task=task,
        lora_task=task,
        evidence=EvidenceLevel.COMMUNITY_RECOMMENDED,
        source_id="modeltc.minimax-h3-turbo",
        allowed_nfe=allowed_nfe,
        default_nfe=default_nfe,
        video_shift=video_shift,
        audio_shift=3.0,
        resolution_policy=resolution_policy,
        reference_policy="explicit_reference_images" if task == "ref2va" else "no_reference_images",
        sampler="euler",
        schedule_owner="sigmax.external_video_sigma",
        audio_owner="minimax_h3.model_native",
        lora_owner="comfyui.LoraLoaderModelOnly",
        attention_owner="host.comfyui",
        transform_order="endpoint_grid_then_video_audio_direct_ratio_then_terminal_zero",
        terminal_policy="append_zero",
    )


MINIMAX_H3_ACCELERATION_RECIPES: Final = tuple(
    sorted(
        (
            _recipe(
                "h3.fl2va.lightx2v-turbo-4-v0.1-544p", "fl2va", (4,), 4, 12.0, "544p_mixed_aspect"
            ),
            _recipe(
                "h3.fl2va.lightx2v-turbo-8-v1.0-544p", "fl2va", (4, 8), 8, 12.0, "544p_mixed_aspect"
            ),
            _recipe("h3.fl2va.lightx2v-turbo-4-v1.0-768p", "fl2va", (4,), 4, 6.0, "1344x768"),
            _recipe(
                "h3.ref2va.lightx2v-turbo-4-v0.1-544p", "ref2va", (4,), 4, 12.0, "544p_mixed_aspect"
            ),
        ),
        key=lambda recipe: recipe.recipe_id,
    )
)

_TENSOR_LAYOUT: Final = "208_lora_A+208_lora_B;50_main+2_refiner;qkv,out,fc1,fc2"
_DYNAMIC_RANK: Final = "dynamic_per_projection"
_UNKNOWN_LICENSE = MiniMaxH3AccelerationLicense(
    scope=MiniMaxH3AccelerationLicenseScope.MODEL_WEIGHTS,
    identifier="unknown",
    status=MiniMaxH3AccelerationLicenseStatus.UNKNOWN,
)
_UNKNOWN_CONVERSION_LICENSE = MiniMaxH3AccelerationLicense(
    scope=MiniMaxH3AccelerationLicenseScope.CONVERSION,
    identifier="unknown",
    status=MiniMaxH3AccelerationLicenseStatus.UNKNOWN,
)


def _artifact(
    artifact_id: str,
    recipe_id: str,
    kind: MiniMaxH3AccelerationArtifactKind,
    source_id: str,
    filename: str,
    sha256: str,
    size_bytes: int | None,
    task: str,
    metadata_task: str,
    baked_scale: float,
    conversion_identity: str,
    disposition: MiniMaxH3AccelerationDisposition,
    reason_codes: tuple[MiniMaxH3AccelerationReasonCode, ...],
) -> MiniMaxH3AccelerationArtifact:
    return MiniMaxH3AccelerationArtifact(
        artifact_id=artifact_id,
        recipe_id=recipe_id,
        kind=kind,
        source_id=source_id,
        filename=filename,
        sha256=sha256,
        size_bytes=size_bytes,
        declared_task=task,
        metadata_task=metadata_task,
        tensor_count=416,
        tensor_layout=_TENSOR_LAYOUT,
        rank_policy=_DYNAMIC_RANK,
        baked_scale=baked_scale,
        loader_strength=1.0,
        conversion_identity=conversion_identity,
        weight_license=_UNKNOWN_LICENSE,
        conversion_license=_UNKNOWN_CONVERSION_LICENSE,
        disposition=disposition,
        reason_codes=reason_codes,
    )


_BLOCKED_LICENSE: Final = (MiniMaxH3AccelerationReasonCode.UNVERIFIED_LICENSE,)
_BLOCKED_UNAVAILABLE: Final = (MiniMaxH3AccelerationReasonCode.UNAVAILABLE_EXACT_ARTIFACT,)
_REJECTED_LOCAL: Final = (MiniMaxH3AccelerationReasonCode.LOCAL_MODIFICATION,)
_REJECTED_REF_CONFLICT: Final = (
    MiniMaxH3AccelerationReasonCode.LOCAL_MODIFICATION,
    MiniMaxH3AccelerationReasonCode.TASK_METADATA_CONFLICT,
)

MINIMAX_H3_ACCELERATION_ARTIFACTS: Final = tuple(
    sorted(
        (
            _artifact(
                "lightx2v.fl2v-4-768.full",
                "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
                MiniMaxH3AccelerationArtifactKind.PUBLISHER_FULL,
                "lightx2v.minimax-h3-turbo",
                "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
                "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",  # pragma: allowlist secret
                None,
                "fl2va",
                "fl2va",
                1.0,
                "publisher_full_source_lora_scale_unobserved",
                MiniMaxH3AccelerationDisposition.BLOCKED,
                _BLOCKED_UNAVAILABLE,
            ),
            _artifact(
                "lightx2v.fl2v-8.full",
                "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
                MiniMaxH3AccelerationArtifactKind.PUBLISHER_FULL,
                "lightx2v.minimax-h3-turbo",
                "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",  # pragma: allowlist secret
                None,
                "fl2va",
                "fl2va",
                1.0,
                "publisher_full_source_lora_scale_unobserved",
                MiniMaxH3AccelerationDisposition.BLOCKED,
                _BLOCKED_UNAVAILABLE,
            ),
            _artifact(
                "lightx2v.ref2v-4.full",
                "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
                MiniMaxH3AccelerationArtifactKind.PUBLISHER_FULL,
                "lightx2v.minimax-h3-turbo",
                "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
                "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",  # pragma: allowlist secret
                None,
                "ref2va",
                "ref2va",
                1.0,
                "publisher_full_source_lora_scale_unobserved",
                MiniMaxH3AccelerationDisposition.BLOCKED,
                _BLOCKED_UNAVAILABLE,
            ),
            _artifact(
                "kijai.fl2v-4-768.reduced",
                "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
                MiniMaxH3AccelerationArtifactKind.REDUCED_EXACT,
                "kijai.minimax-h3-comfy",
                "Minimax-KJ_h3_fl2v_lightx2v_turbo_4step_v1.0_768p_resized_avg_rank_31_bf16.safetensors",
                "9515eee9f642aa0e7fcc401f56d408ef2d6388f81881fe50bddded8220870a4d",  # pragma: allowlist secret
                440873704,
                "fl2va",
                "fl2va",
                1.0,
                "kijai_lfs_exact_reduced_rank31_alpha128",
                MiniMaxH3AccelerationDisposition.BLOCKED,
                _BLOCKED_LICENSE,
            ),
            _artifact(
                "kijai.fl2v-8.reduced",
                "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
                MiniMaxH3AccelerationArtifactKind.REDUCED_EXACT,
                "kijai.minimax-h3-comfy",
                "Minimax-KJ_h3_fl2v_lightx2v_turbo_8step_v1.0_resized_avg_rank_24_bf16.safetensors",
                "8e05b7b982c3aff7deb692a188c8a8d8acaeff8a12abfe1aeac822fb8ee3f0b7",  # pragma: allowlist secret
                364638304,
                "fl2va",
                "fl2va",
                0.0625,
                "kijai_lfs_exact_reduced_rank24_alpha8",
                MiniMaxH3AccelerationDisposition.BLOCKED,
                _BLOCKED_LICENSE,
            ),
            _artifact(
                "kijai.ref2v-4.reduced",
                "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
                MiniMaxH3AccelerationArtifactKind.REDUCED_EXACT,
                "kijai.minimax-h3-comfy",
                "Minimax-KJ_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors",
                "9ea3bd3a6aac22994153e294cf1ecab0a8766fc0f8d056ace645a01d1a6a4daf",  # pragma: allowlist secret
                306731560,
                "ref2va",
                "ref2va",
                0.0625,
                "kijai_lfs_exact_reduced_rank20_alpha8",
                MiniMaxH3AccelerationDisposition.BLOCKED,
                _BLOCKED_LICENSE,
            ),
            _artifact(
                "local.fl2v-4-768.modified",
                "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
                MiniMaxH3AccelerationArtifactKind.LOCAL_MODIFIED,
                "lightx2v.minimax-h3-turbo",
                "Minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
                "1b85da614014024a0c9507f12558917dcc69b6adb564e716324594f401723115",  # pragma: allowlist secret
                298177224,
                "fl2va",
                "fl2va",
                1.0,
                "local_dynamic_resize_qkv_fusion_rank21",
                MiniMaxH3AccelerationDisposition.REJECTED,
                _REJECTED_LOCAL,
            ),
            _artifact(
                "local.fl2v-8.modified",
                "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
                MiniMaxH3AccelerationArtifactKind.LOCAL_MODIFIED,
                "lightx2v.minimax-h3-turbo",
                "Minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors",
                "a3208be61329c27a6754c53db9a21a3c86e2a285381700adf2d97e279c062840",  # pragma: allowlist secret
                327035608,
                "fl2va",
                "fl2va",
                0.0625,
                "local_dynamic_resize_qkv_fusion_rank21",
                MiniMaxH3AccelerationDisposition.REJECTED,
                _REJECTED_LOCAL,
            ),
            _artifact(
                "local.ref2v-4.modified",
                "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
                MiniMaxH3AccelerationArtifactKind.LOCAL_MODIFIED,
                "lightx2v.minimax-h3-turbo",
                "Minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors",
                "2c6abb194cff3e26c2295c87892913adf0c92d8f784f305238246759f9b333d0",  # pragma: allowlist secret
                326935264,
                "ref2va",
                "fl2va",
                0.0625,
                "local_dynamic_resize_qkv_fusion_rank21",
                MiniMaxH3AccelerationDisposition.REJECTED,
                _REJECTED_REF_CONFLICT,
            ),
        ),
        key=lambda artifact: artifact.artifact_id,
    )
)


def _backend(
    backend_id: str,
    scope: MiniMaxH3AccelerationBackendScope,
    source_id: str,
    reason: MiniMaxH3AccelerationReasonCode,
) -> MiniMaxH3AccelerationBackend:
    return MiniMaxH3AccelerationBackend(
        backend_id=backend_id,
        scope=scope,
        source_id=source_id,
        disposition=MiniMaxH3AccelerationDisposition.BLOCKED,
        reason_codes=(reason,),
    )


MINIMAX_H3_ACCELERATION_BACKENDS: Final = tuple(
    sorted(
        (
            _backend(
                "comfy-kitchen.quantized-operations",
                MiniMaxH3AccelerationBackendScope.QUANTIZED_OPERATIONS,
                "comfy-kitchen.repository",
                MiniMaxH3AccelerationReasonCode.BACKEND_SCOPE_MISMATCH,
            ),
            _backend(
                "comfy-kitchen.int8-attention",
                MiniMaxH3AccelerationBackendScope.INT8_ATTENTION,
                "comfy-kitchen.repository",
                MiniMaxH3AccelerationReasonCode.BACKEND_SCOPE_MISMATCH,
            ),
            _backend(
                "comfyui.core-attention",
                MiniMaxH3AccelerationBackendScope.CORE_ATTENTION,
                "comfyui.repository",
                MiniMaxH3AccelerationReasonCode.BACKEND_SCOPE_MISMATCH,
            ),
            _backend(
                "minimaxai.msa",
                MiniMaxH3AccelerationBackendScope.MODEL_SPARSE_ATTENTION,
                "minimaxai.msa",
                MiniMaxH3AccelerationReasonCode.UNPROVEN_MSA_H3_LINK,
            ),
            _backend(
                "sageattention.upstream",
                MiniMaxH3AccelerationBackendScope.CORE_ATTENTION,
                "thu-ml.sageattention",
                MiniMaxH3AccelerationReasonCode.BACKEND_SCOPE_MISMATCH,
            ),
            _backend(
                "flashattention.upstream",
                MiniMaxH3AccelerationBackendScope.CORE_ATTENTION,
                "dao-ai-lab.flash-attention",
                MiniMaxH3AccelerationReasonCode.BACKEND_SCOPE_MISMATCH,
            ),
        ),
        key=lambda backend: backend.backend_id,
    )
)

MINIMAX_H3_ACCELERATION_CANDIDATES: Final = MINIMAX_H3_ACCELERATION_ARTIFACTS
MINIMAX_H3_ACCELERATION_REASON_CODES: Final = tuple(
    reason.value for reason in MiniMaxH3AccelerationReasonCode
)

_SOURCES_BY_ID: Final = {source.source_id: source for source in MINIMAX_H3_ACCELERATION_SOURCES}
_RECIPES_BY_ID: Final = {recipe.recipe_id: recipe for recipe in MINIMAX_H3_ACCELERATION_RECIPES}
_ARTIFACTS_BY_ID: Final = {
    artifact.artifact_id: artifact for artifact in MINIMAX_H3_ACCELERATION_ARTIFACTS
}


def _candidate_error(
    reason: MiniMaxH3AccelerationReasonCode, detail: str
) -> MiniMaxH3AccelerationError:
    return MiniMaxH3AccelerationError(reason, detail)


def qualify_minimax_h3_candidate(
    *,
    candidate_id: str,
    task: str | None = None,
    nfe: int | None = None,
    video_shift: float | None = None,
    audio_shift: float | None = None,
    artifact_sha256: str | None = None,
    artifact_size_bytes: int | None = None,
    artifact_filename: str | None = None,
    loader_strength: float | None = None,
    resolution_policy: str | None = None,
    backend_scope: MiniMaxH3AccelerationBackendScope | None = None,
    require_eligible: bool = False,
) -> MiniMaxH3AccelerationArtifact:
    """Resolve one exact candidate and reject mismatched caller claims.

    This function validates metadata only.  It never opens a filename, hashes a payload, loads a
    model, or promotes a blocked candidate into runtime support.
    """

    if not isinstance(candidate_id, str) or candidate_id not in _ARTIFACTS_BY_ID:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.UNKNOWN_ARTIFACT_HASH,
            "candidate identity is not an exact M6-12 artifact record",
        )
    candidate = _ARTIFACTS_BY_ID[candidate_id]
    recipe = _RECIPES_BY_ID.get(candidate.recipe_id)
    if recipe is None:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.MISSING_OWNERSHIP,
            "candidate recipe is not registered in the qualification table",
        )
    if task is not None and task != candidate.declared_task:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.WRONG_TASK,
            "candidate task does not match the exact artifact task",
        )
    if nfe is not None and nfe not in recipe.allowed_nfe:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.UNSUPPORTED_RECIPE_NFE,
            "requested NFE is outside the source-qualified recipe",
        )
    if resolution_policy is not None and resolution_policy != recipe.resolution_policy:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.RESOLUTION_POLICY_MISMATCH,
            "resolution policy does not match the source-qualified recipe",
        )
    if backend_scope is not None:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.BACKEND_SCOPE_MISMATCH,
            "backend scope is host-owned and cannot be bound by a recipe artifact",
        )
    if video_shift is not None and not math.isclose(
        video_shift, recipe.video_shift, rel_tol=0.0, abs_tol=1e-12
    ):
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.DUPLICATE_SHIFT_RISK,
            "video shift does not match the recipe-owned shift",
        )
    if audio_shift is not None and not math.isclose(
        audio_shift, recipe.audio_shift, rel_tol=0.0, abs_tol=1e-12
    ):
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.DUPLICATE_SHIFT_RISK,
            "audio shift does not match the recipe-owned shift",
        )
    if artifact_sha256 is not None and artifact_sha256 != candidate.sha256:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.SIZE_HASH_MISMATCH,
            "artifact SHA-256 does not match the exact source record",
        )
    if artifact_size_bytes is not None and artifact_size_bytes != candidate.size_bytes:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.SIZE_HASH_MISMATCH,
            "artifact byte size does not match the exact source record",
        )
    if artifact_filename is not None and artifact_filename != candidate.filename:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.FILENAME_ONLY_IDENTITY,
            "artifact filename is not an exact source identity",
        )
    if loader_strength is not None and not math.isclose(
        loader_strength, candidate.loader_strength, rel_tol=0.0, abs_tol=1e-12
    ):
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.DUPLICATE_SCALE_RISK,
            "loader strength would apply a second scale",
        )
    if candidate.metadata_task != candidate.declared_task:
        raise _candidate_error(
            MiniMaxH3AccelerationReasonCode.TASK_METADATA_CONFLICT,
            "artifact metadata task conflicts with declared task",
        )
    if require_eligible and candidate.disposition is not MiniMaxH3AccelerationDisposition.QUALIFIED:
        reason = candidate.reason_codes[0]
        raise _candidate_error(reason, "candidate is not eligible for downstream runtime planning")
    return candidate


def _license_projection(license_boundary: MiniMaxH3AccelerationLicense) -> dict[str, str]:
    return {
        "identifier": license_boundary.identifier,
        "scope": license_boundary.scope.value,
        "status": license_boundary.status.value,
    }


def serialize_minimax_h3_acceleration() -> dict[str, object]:
    """Return the canonical public-safe qualification projection."""

    return {
        "schema": MINIMAX_H3_ACCELERATION_SCHEMA_ID,
        "schema_version": MINIMAX_H3_ACCELERATION_SCHEMA_VERSION,
        "observed_on": MINIMAX_H3_ACCELERATION_OBSERVED_ON,
        "sources": [
            {
                "license_id": source.license_id,
                "license_status": source.license_status.value,
                "locators": list(source.locators),
                "revision": source.revision,
                "role": source.role.value,
                "source_id": source.source_id,
                "url": source.url,
            }
            for source in MINIMAX_H3_ACCELERATION_SOURCES
        ],
        "recipes": [
            {
                "allowed_nfe": list(recipe.allowed_nfe),
                "attention_owner": recipe.attention_owner,
                "audio_owner": recipe.audio_owner,
                "audio_shift": recipe.audio_shift,
                "default_nfe": recipe.default_nfe,
                "evidence": recipe.evidence.value,
                "lora_owner": recipe.lora_owner,
                "lora_task": recipe.lora_task,
                "model_family": recipe.model_family,
                "model_variant": recipe.model_variant,
                "recipe_id": recipe.recipe_id,
                "reference_policy": recipe.reference_policy,
                "resolution_policy": recipe.resolution_policy,
                "disposition": recipe.disposition.value,
                "reason_codes": [reason.value for reason in recipe.reason_codes],
                "runtime_registered": recipe.runtime_registered,
                "sampler": recipe.sampler,
                "schedule_owner": recipe.schedule_owner,
                "source_id": recipe.source_id,
                "task": recipe.task,
                "terminal_policy": recipe.terminal_policy,
                "transform_order": recipe.transform_order,
                "video_shift": recipe.video_shift,
            }
            for recipe in MINIMAX_H3_ACCELERATION_RECIPES
        ],
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "artifact_kind": artifact.kind.value,
                "baked_scale": artifact.baked_scale,
                "conversion_identity": artifact.conversion_identity,
                "conversion_license": _license_projection(artifact.conversion_license),
                "declared_task": artifact.declared_task,
                "disposition": artifact.disposition.value,
                "filename": artifact.filename,
                "loader_strength": artifact.loader_strength,
                "metadata_task": artifact.metadata_task,
                "rank_policy": artifact.rank_policy,
                "reason_codes": [reason.value for reason in artifact.reason_codes],
                "recipe_id": artifact.recipe_id,
                "runtime_registered": artifact.runtime_registered,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "source_id": artifact.source_id,
                "tensor_count": artifact.tensor_count,
                "tensor_layout": artifact.tensor_layout,
                "weight_license": _license_projection(artifact.weight_license),
            }
            for artifact in MINIMAX_H3_ACCELERATION_ARTIFACTS
        ],
        "backends": [
            {
                "backend_id": backend.backend_id,
                "disposition": backend.disposition.value,
                "reason_codes": [reason.value for reason in backend.reason_codes],
                "runtime_selected": backend.runtime_selected,
                "scope": backend.scope.value,
                "source_id": backend.source_id,
            }
            for backend in MINIMAX_H3_ACCELERATION_BACKENDS
        ],
        "reason_codes": list(MINIMAX_H3_ACCELERATION_REASON_CODES),
    }


def minimax_h3_acceleration_fingerprint() -> str:
    """Return the deterministic fingerprint of the qualification projection."""

    encoded = json.dumps(
        serialize_minimax_h3_acceleration(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MINIMAX_H3_ACCELERATION_ARTIFACTS",
    "MINIMAX_H3_ACCELERATION_BACKENDS",
    "MINIMAX_H3_ACCELERATION_CANDIDATES",
    "MINIMAX_H3_ACCELERATION_OBSERVED_ON",
    "MINIMAX_H3_ACCELERATION_REASON_CODES",
    "MINIMAX_H3_ACCELERATION_RECIPES",
    "MINIMAX_H3_ACCELERATION_SCHEMA_ID",
    "MINIMAX_H3_ACCELERATION_SCHEMA_VERSION",
    "MINIMAX_H3_ACCELERATION_SOURCES",
    "MiniMaxH3AccelerationArtifact",
    "MiniMaxH3AccelerationArtifactKind",
    "MiniMaxH3AccelerationBackend",
    "MiniMaxH3AccelerationBackendScope",
    "MiniMaxH3AccelerationDisposition",
    "MiniMaxH3AccelerationError",
    "MiniMaxH3AccelerationLicense",
    "MiniMaxH3AccelerationLicenseScope",
    "MiniMaxH3AccelerationLicenseStatus",
    "MiniMaxH3AccelerationReasonCode",
    "MiniMaxH3AccelerationRecipe",
    "MiniMaxH3AccelerationSource",
    "MiniMaxH3AccelerationSourceRole",
    "minimax_h3_acceleration_fingerprint",
    "qualify_minimax_h3_candidate",
    "serialize_minimax_h3_acceleration",
]
