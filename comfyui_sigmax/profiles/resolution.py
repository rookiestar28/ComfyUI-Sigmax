"""Pure profile/model/host/sampler capability resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityReason,
    ExecutionFeatureRequest,
    ModelCapabilities,
    SamplerCapabilities,
    ScheduleContractError,
    ScheduleOwnership,
    evaluate_compatibility,
)
from comfyui_sigmax.profiles.krea2_variant import (
    Krea2VariantResolution,
    Krea2VariantResolutionStatus,
)
from comfyui_sigmax.profiles.registry import RegisteredProfile

CAPABILITY_RESOLUTION_SCHEMA_ID: Final = "sigmax.capability-resolution/1"
CAPABILITY_RESOLUTION_SCHEMA_VERSION: Final = "1"

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REVISION_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_PROFILE_KEY_PATTERN: Final = re.compile(
    r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9][a-z0-9_-]*)+@[0-9]+(?:\.[0-9]+)*$"
)
_FINGERPRINT_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
CAPABILITY_RESOLUTION_CORE_REASON_CODES: Final = frozenset(
    f"core.{reason.value}" for reason in CompatibilityReason
)
CAPABILITY_RESOLUTION_ADDITIONAL_REASON_CODES: Final = frozenset(
    {
        "host.capability_experimental",
        "host.capability_missing",
        "host.capability_unsupported",
        "model.family_mismatch",
        "model.identity_ambiguous",
        "model.identity_conflict",
        "model.identity_suggested",
        "model.identity_unknown",
        "model.variant_mismatch",
    }
)


class ModelIdentityStatus(str, Enum):
    """Trust-preserving normalized model identity state."""

    CONFIRMED = "confirmed"
    SUGGESTED = "suggested"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class HostCapabilityLifecycle(str, Enum):
    """Stability state of one normalized host capability."""

    LANDED = "landed"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


def _require_identifier(field_name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not _IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise ScheduleContractError(f"{field_name} must be a stable lowercase identifier")
    return value


def _require_identifier_tuple(
    field_name: str,
    values: object,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ScheduleContractError(f"{field_name} must be an immutable tuple")
    if not allow_empty and not values:
        raise ScheduleContractError(f"{field_name} must not be empty")
    normalized = tuple(_require_identifier(field_name, value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ScheduleContractError(f"{field_name} contains duplicates")
    if normalized != tuple(sorted(normalized)):
        raise ScheduleContractError(f"{field_name} must use canonical order")
    return normalized


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelIdentityEvidence:
    """Normalized identity evidence without promoting suggestions to confirmations."""

    evidence_version: str
    model_family: str
    status: ModelIdentityStatus
    confirmed_variant: str | None
    suggested_variant: str | None
    confidence: str
    decisive_source: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evidence_version != "1":
            raise ScheduleContractError("model identity evidence_version must be 1")
        _require_identifier("model_family", self.model_family)
        if not isinstance(self.status, ModelIdentityStatus):
            raise ScheduleContractError("model identity status is unsupported")
        if self.confirmed_variant is not None:
            _require_identifier("confirmed_variant", self.confirmed_variant)
        if self.suggested_variant is not None:
            _require_identifier("suggested_variant", self.suggested_variant)
        _require_identifier("confidence", self.confidence)
        if self.decisive_source is not None:
            _require_identifier("decisive_source", self.decisive_source)
        _require_identifier_tuple("reason_codes", self.reason_codes)

        if self.status is ModelIdentityStatus.CONFIRMED:
            if (
                self.confirmed_variant is None
                or self.suggested_variant is not None
                or self.confidence == "none"
                or self.decisive_source is None
            ):
                raise ScheduleContractError("confirmed model identity is inconsistent")
            return
        if self.status is ModelIdentityStatus.SUGGESTED:
            if (
                self.confirmed_variant is not None
                or self.suggested_variant is None
                or self.confidence == "none"
                or self.decisive_source is None
            ):
                raise ScheduleContractError("suggested model identity is inconsistent")
            return
        if (
            self.confirmed_variant is not None
            or self.suggested_variant is not None
            or self.confidence != "none"
            or self.decisive_source is not None
        ):
            raise ScheduleContractError("unresolved model identity is inconsistent")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilityEvidence:
    """One identity result paired with independently declared model capabilities."""

    evidence_version: str
    identity: ModelIdentityEvidence
    capabilities: ModelCapabilities

    def __post_init__(self) -> None:
        if self.evidence_version != "1":
            raise ScheduleContractError("model capability evidence_version must be 1")
        if not isinstance(self.identity, ModelIdentityEvidence):
            raise ScheduleContractError("identity must be ModelIdentityEvidence")
        if not isinstance(self.capabilities, ModelCapabilities):
            raise ScheduleContractError("capabilities must be ModelCapabilities")


@dataclass(frozen=True, slots=True, kw_only=True)
class HostCapabilityEvidence:
    """One host capability and its explicit lifecycle."""

    capability_id: str
    lifecycle: HostCapabilityLifecycle

    def __post_init__(self) -> None:
        _require_identifier("capability_id", self.capability_id)
        if not isinstance(self.lifecycle, HostCapabilityLifecycle):
            raise ScheduleContractError("host capability lifecycle is unsupported")


@dataclass(frozen=True, slots=True, kw_only=True)
class HostCapabilities:
    """Versioned normalized host evidence; probing belongs to the adapter layer."""

    evidence_version: str
    host_id: str
    host_version: str
    host_revision: str
    capabilities: tuple[HostCapabilityEvidence, ...]

    def __post_init__(self) -> None:
        if self.evidence_version != "1":
            raise ScheduleContractError("host capability evidence_version must be 1")
        _require_identifier("host_id", self.host_id)
        if not isinstance(self.host_version, str) or not self.host_version.strip():
            raise ScheduleContractError("host_version must be a non-empty public string")
        if not isinstance(self.host_revision, str) or not _REVISION_PATTERN.fullmatch(
            self.host_revision
        ):
            raise ScheduleContractError("host_revision must be a pinned lowercase 40-hex revision")
        if not isinstance(self.capabilities, tuple) or not all(
            isinstance(item, HostCapabilityEvidence) for item in self.capabilities
        ):
            raise ScheduleContractError(
                "host capabilities must be an immutable HostCapabilityEvidence tuple"
            )
        identifiers = tuple(item.capability_id for item in self.capabilities)
        if len(identifiers) != len(set(identifiers)):
            raise ScheduleContractError("host capabilities contain duplicate identifiers")
        if identifiers != tuple(sorted(identifiers)):
            raise ScheduleContractError("host capabilities must use canonical order")


@dataclass(frozen=True, slots=True, kw_only=True)
class HostCapabilityRequirement:
    """Resolution of one capability required from the selected host."""

    capability_id: str
    lifecycle: HostCapabilityLifecycle | None
    satisfied: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        _require_identifier("capability_id", self.capability_id)
        if self.lifecycle is not None and not isinstance(
            self.lifecycle,
            HostCapabilityLifecycle,
        ):
            raise ScheduleContractError("requirement lifecycle is unsupported")
        if not isinstance(self.satisfied, bool):
            raise ScheduleContractError("requirement satisfied must be boolean")
        if self.reason_code is not None:
            _require_identifier("reason_code", self.reason_code)
        if self.satisfied != (self.lifecycle is HostCapabilityLifecycle.LANDED):
            raise ScheduleContractError("host requirement satisfaction is inconsistent")
        if self.satisfied != (self.reason_code is None):
            raise ScheduleContractError("host requirement reason is inconsistent")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileCapabilityDecision:
    """Versioned, auditable pre-execution resolution result."""

    schema_id: str
    schema_version: str
    profile_key: str
    profile_fingerprint: str
    level: CompatibilityLevel
    reason_codes: tuple[str, ...]
    model_identity: ModelIdentityEvidence
    host_id: str
    host_version: str
    host_revision: str
    host_requirements: tuple[HostCapabilityRequirement, ...]
    core_decision: CompatibilityDecision

    def __post_init__(self) -> None:
        if self.schema_id != CAPABILITY_RESOLUTION_SCHEMA_ID:
            raise ScheduleContractError("capability resolution schema_id is unsupported")
        if self.schema_version != CAPABILITY_RESOLUTION_SCHEMA_VERSION:
            raise ScheduleContractError("capability resolution schema_version is unsupported")
        if not isinstance(self.profile_key, str) or not _PROFILE_KEY_PATTERN.fullmatch(
            self.profile_key
        ):
            raise ScheduleContractError("profile_key must be an exact canonical profile key")
        if not isinstance(self.profile_fingerprint, str) or not _FINGERPRINT_PATTERN.fullmatch(
            self.profile_fingerprint
        ):
            raise ScheduleContractError("profile_fingerprint must be a sha256 fingerprint")
        if not isinstance(self.level, CompatibilityLevel):
            raise ScheduleContractError("resolution level is unsupported")
        _require_identifier_tuple("reason_codes", self.reason_codes)
        if not isinstance(self.model_identity, ModelIdentityEvidence):
            raise ScheduleContractError("model_identity must be ModelIdentityEvidence")
        _require_identifier("host_id", self.host_id)
        if not isinstance(self.host_version, str) or not self.host_version.strip():
            raise ScheduleContractError("host_version must be a non-empty public string")
        if not isinstance(self.host_revision, str) or not _REVISION_PATTERN.fullmatch(
            self.host_revision
        ):
            raise ScheduleContractError("host_revision must be a pinned lowercase 40-hex revision")
        if not isinstance(self.host_requirements, tuple) or not all(
            isinstance(item, HostCapabilityRequirement) for item in self.host_requirements
        ):
            raise ScheduleContractError("host_requirements must be an immutable requirement tuple")
        requirement_ids = tuple(item.capability_id for item in self.host_requirements)
        if requirement_ids != tuple(sorted(requirement_ids)) or len(requirement_ids) != len(
            set(requirement_ids)
        ):
            raise ScheduleContractError("host_requirements must be canonical and unique")
        if not isinstance(self.core_decision, CompatibilityDecision):
            raise ScheduleContractError("core_decision must be CompatibilityDecision")

        reason_set = set(self.reason_codes)
        if not reason_set.issubset(
            CAPABILITY_RESOLUTION_CORE_REASON_CODES | CAPABILITY_RESOLUTION_ADDITIONAL_REASON_CODES
        ):
            raise ScheduleContractError("resolution contains an unsupported reason code")
        additional_reasons = reason_set & CAPABILITY_RESOLUTION_ADDITIONAL_REASON_CODES
        actual_core_reasons = reason_set & CAPABILITY_RESOLUTION_CORE_REASON_CODES
        expected_core_reasons = {f"core.{reason.value}" for reason in self.core_decision.reasons}
        if additional_reasons and expected_core_reasons == {"core.compatible"}:
            expected_core_reasons = set()
        if actual_core_reasons != expected_core_reasons:
            raise ScheduleContractError("resolution reasons do not match core_decision")
        expected_level = (
            CompatibilityLevel.REJECT
            if additional_reasons or self.core_decision.level is CompatibilityLevel.REJECT
            else self.core_decision.level
        )
        if self.level is not expected_level:
            raise ScheduleContractError("resolution level does not match its reasons")


def model_identity_from_krea2_resolution(
    resolution: Krea2VariantResolution,
) -> ModelIdentityEvidence:
    """Map Krea 2 evidence without changing its trust level."""

    if not isinstance(resolution, Krea2VariantResolution):
        raise ScheduleContractError("resolution must be Krea2VariantResolution")
    status_map = {
        Krea2VariantResolutionStatus.RESOLVED: ModelIdentityStatus.CONFIRMED,
        Krea2VariantResolutionStatus.SUGGESTED: ModelIdentityStatus.SUGGESTED,
        Krea2VariantResolutionStatus.AMBIGUOUS: ModelIdentityStatus.AMBIGUOUS,
        Krea2VariantResolutionStatus.CONFLICT: ModelIdentityStatus.CONFLICT,
    }
    reasons = tuple(
        sorted(
            {
                *(item.reason_code for item in resolution.evidence),
                *resolution.warnings,
            }
        )
    )
    return ModelIdentityEvidence(
        evidence_version="1",
        model_family="krea2",
        status=status_map[resolution.status],
        confirmed_variant=(
            resolution.resolved_variant.value if resolution.resolved_variant is not None else None
        ),
        suggested_variant=(
            resolution.suggested_variant.value if resolution.suggested_variant is not None else None
        ),
        confidence=resolution.confidence.value,
        decisive_source=(
            resolution.decisive_source.value if resolution.decisive_source is not None else None
        ),
        reason_codes=reasons,
    )


def _required_host_capability_ids(
    *,
    registered_profile: RegisteredProfile,
    sampler: SamplerCapabilities,
    request: ExecutionFeatureRequest,
) -> tuple[str, ...]:
    required = {f"sampler.{sampler.sampler_id}"}
    if (
        registered_profile.schema.profile_capabilities.ownership
        is ScheduleOwnership.EXTERNAL_SIGMAS
    ):
        required.add("schedule.external_sigmas")
    if request.use_partial_denoise:
        required.add("execution.partial_denoise")
    if request.use_per_token_timesteps:
        required.add("execution.per_token_timesteps")
    return tuple(sorted(required))


def _resolve_host_requirements(
    required_ids: tuple[str, ...],
    host: HostCapabilities,
) -> tuple[HostCapabilityRequirement, ...]:
    available = {item.capability_id: item.lifecycle for item in host.capabilities}
    requirements: list[HostCapabilityRequirement] = []
    for capability_id in required_ids:
        lifecycle = available.get(capability_id)
        reason_code: str | None = None
        if lifecycle is None:
            reason_code = "host.capability_missing"
        # IMPORTANT: required experimental APIs are not stable host capabilities.
        # Source: https://github.com/Comfy-Org/rfcs#rfc-lifecycle
        elif lifecycle is HostCapabilityLifecycle.EXPERIMENTAL:
            reason_code = "host.capability_experimental"
        elif lifecycle is HostCapabilityLifecycle.UNSUPPORTED:
            reason_code = "host.capability_unsupported"
        requirements.append(
            HostCapabilityRequirement(
                capability_id=capability_id,
                lifecycle=lifecycle,
                satisfied=reason_code is None,
                reason_code=reason_code,
            )
        )
    return tuple(requirements)


def _identity_reasons(
    identity: ModelIdentityEvidence,
    registered_profile: RegisteredProfile,
) -> tuple[str, ...]:
    if identity.status is not ModelIdentityStatus.CONFIRMED:
        return (f"model.identity_{identity.status.value}",)
    reasons: list[str] = []
    if identity.model_family != registered_profile.schema.model_family:
        reasons.append("model.family_mismatch")
    if identity.confirmed_variant != registered_profile.schema.model_variant:
        reasons.append("model.variant_mismatch")
    return tuple(reasons)


def resolve_profile_capabilities(
    *,
    registered_profile: RegisteredProfile,
    model: ModelCapabilityEvidence,
    host: HostCapabilities,
    sampler: SamplerCapabilities,
    request: ExecutionFeatureRequest,
) -> ProfileCapabilityDecision:
    """Resolve normalized evidence without importing or executing a host or sampler."""

    for field_name, value, expected_type in (
        ("registered_profile", registered_profile, RegisteredProfile),
        ("model", model, ModelCapabilityEvidence),
        ("host", host, HostCapabilities),
        ("sampler", sampler, SamplerCapabilities),
        ("request", request, ExecutionFeatureRequest),
    ):
        if not isinstance(value, expected_type):
            raise ScheduleContractError(f"{field_name} must be {expected_type.__name__}")

    core_decision = evaluate_compatibility(
        model=model.capabilities,
        profile=registered_profile.schema.profile_capabilities,
        sampler=sampler,
        request=request,
    )
    host_requirements = _resolve_host_requirements(
        _required_host_capability_ids(
            registered_profile=registered_profile,
            sampler=sampler,
            request=request,
        ),
        host,
    )
    additional_reasons = (
        *_identity_reasons(model.identity, registered_profile),
        *(item.reason_code for item in host_requirements if item.reason_code is not None),
    )
    core_reasons = tuple(f"core.{reason.value}" for reason in core_decision.reasons)
    if additional_reasons and core_reasons == ("core.compatible",):
        core_reasons = ()
    reason_codes = tuple(sorted(set((*additional_reasons, *core_reasons))))
    if additional_reasons or core_decision.level is CompatibilityLevel.REJECT:
        level = CompatibilityLevel.REJECT
    else:
        level = core_decision.level

    return ProfileCapabilityDecision(
        schema_id=CAPABILITY_RESOLUTION_SCHEMA_ID,
        schema_version=CAPABILITY_RESOLUTION_SCHEMA_VERSION,
        profile_key=registered_profile.key.canonical,
        profile_fingerprint=registered_profile.fingerprint,
        level=level,
        reason_codes=reason_codes,
        model_identity=model.identity,
        host_id=host.host_id,
        host_version=host.host_version,
        host_revision=host.host_revision,
        host_requirements=host_requirements,
        core_decision=core_decision,
    )
