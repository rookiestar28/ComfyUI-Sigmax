"""Pure, fail-closed Krea 2 RAW/Turbo evidence resolution."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import ScheduleContractError

KREA2_RAW_OFFICIAL_SHA256: Final = (
    "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7"  # pragma: allowlist secret
)
KREA2_TURBO_OFFICIAL_SHA256: Final = (
    "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3"  # pragma: allowlist secret
)

_HASH_VARIANTS: Final = {
    KREA2_RAW_OFFICIAL_SHA256: "raw",
    KREA2_TURBO_OFFICIAL_SHA256: "turbo",
}
_PROFILE_VARIANTS: Final = {
    "krea2.raw.official": "raw",
    "krea2.turbo.official": "turbo",
}
_SHA256_PATTERN: Final = re.compile(r"[0-9a-fA-F]{64}\Z")
_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_RAW_FILENAME_PATTERN: Final = re.compile(r"(?:^|[._\-\s])raw(?:$|[._\-\s])")
_TURBO_FILENAME_PATTERN: Final = re.compile(r"(?:^|[._\-\s])turbo(?:$|[._\-\s])")


class Krea2Variant(str, Enum):
    """Schedule-relevant Krea 2 checkpoint variants."""

    RAW = "raw"
    TURBO = "turbo"


class Krea2VariantEvidenceSource(str, Enum):
    """Ordered evidence sources accepted by the Krea 2 resolver."""

    EXPLICIT_SELECTION = "explicit_selection"
    TRUSTED_PROFILE_METADATA = "trusted_profile_metadata"
    TRUSTED_FRAMEWORK_METADATA = "trusted_framework_metadata"
    VERIFIED_SHA256 = "verified_sha256"
    LOCAL_HEADER_SIGNAL = "local_header_signal"
    FILENAME_SIGNAL = "filename_signal"
    LOCAL_TENSOR_SIGNAL = "local_tensor_signal"
    MODEL_CLASS_SIGNAL = "model_class_signal"


class Krea2VariantConfidence(str, Enum):
    """Confidence vocabulary that cannot be supplied by untrusted inputs."""

    AUTHORITATIVE = "authoritative"
    VERIFIED = "verified"
    CORROBORATING = "corroborating"
    WEAK = "weak"
    FAMILY_ONLY = "family_only"
    NONE = "none"


class Krea2VariantResolutionStatus(str, Enum):
    """Possible variant-resolution outcomes."""

    RESOLVED = "resolved"
    SUGGESTED = "suggested"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"


_SOURCE_CONFIDENCE: Final = {
    Krea2VariantEvidenceSource.EXPLICIT_SELECTION: Krea2VariantConfidence.AUTHORITATIVE,
    Krea2VariantEvidenceSource.TRUSTED_PROFILE_METADATA: Krea2VariantConfidence.AUTHORITATIVE,
    Krea2VariantEvidenceSource.TRUSTED_FRAMEWORK_METADATA: Krea2VariantConfidence.AUTHORITATIVE,
    Krea2VariantEvidenceSource.VERIFIED_SHA256: Krea2VariantConfidence.VERIFIED,
    Krea2VariantEvidenceSource.LOCAL_HEADER_SIGNAL: Krea2VariantConfidence.CORROBORATING,
    Krea2VariantEvidenceSource.FILENAME_SIGNAL: Krea2VariantConfidence.WEAK,
    Krea2VariantEvidenceSource.LOCAL_TENSOR_SIGNAL: Krea2VariantConfidence.FAMILY_ONLY,
    Krea2VariantEvidenceSource.MODEL_CLASS_SIGNAL: Krea2VariantConfidence.FAMILY_ONLY,
}
_SOURCE_RANK: Final = {source: rank for rank, source in enumerate(Krea2VariantEvidenceSource)}
_RESOLVING_SOURCES: Final = frozenset(
    {
        Krea2VariantEvidenceSource.EXPLICIT_SELECTION,
        Krea2VariantEvidenceSource.TRUSTED_PROFILE_METADATA,
        Krea2VariantEvidenceSource.TRUSTED_FRAMEWORK_METADATA,
        Krea2VariantEvidenceSource.VERIFIED_SHA256,
    }
)
_SUGGESTION_SOURCES: Final = frozenset(
    {
        Krea2VariantEvidenceSource.LOCAL_HEADER_SIGNAL,
        Krea2VariantEvidenceSource.FILENAME_SIGNAL,
    }
)


def _require_token(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not _TOKEN_PATTERN.fullmatch(value):
        raise ScheduleContractError(f"{field_name} must be a bounded public token")


def _coerce_variant(value: Krea2Variant | str, *, field_name: str) -> Krea2Variant:
    if isinstance(value, Krea2Variant):
        return value
    if not isinstance(value, str):
        raise ScheduleContractError(f"{field_name} must identify RAW or Turbo")
    try:
        return Krea2Variant(value.casefold())
    except ValueError as exc:
        raise ScheduleContractError(f"{field_name} must identify RAW or Turbo") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2VariantEvidence:
    """One normalized, path-free variant evidence record."""

    source: Krea2VariantEvidenceSource
    confidence: Krea2VariantConfidence
    variant: Krea2Variant | None
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, Krea2VariantEvidenceSource):
            raise ScheduleContractError("variant evidence source is unsupported")
        if not isinstance(self.confidence, Krea2VariantConfidence):
            raise ScheduleContractError("variant evidence confidence is unsupported")
        if self.variant is not None and not isinstance(self.variant, Krea2Variant):
            raise ScheduleContractError("variant evidence value is unsupported")
        _require_token("reason_code", self.reason_code)

        expected_confidence = _SOURCE_CONFIDENCE[self.source]
        if self.source is Krea2VariantEvidenceSource.VERIFIED_SHA256 and self.variant is None:
            expected_confidence = Krea2VariantConfidence.NONE
        if self.confidence is not expected_confidence:
            raise ScheduleContractError("variant evidence confidence does not match its source")
        if (
            self.source in _RESOLVING_SOURCES | _SUGGESTION_SOURCES
            and self.source is not Krea2VariantEvidenceSource.VERIFIED_SHA256
            and self.variant is None
        ):
            raise ScheduleContractError("variant-bearing evidence requires a variant")
        if (
            self.source
            in {
                Krea2VariantEvidenceSource.LOCAL_TENSOR_SIGNAL,
                Krea2VariantEvidenceSource.MODEL_CLASS_SIGNAL,
            }
            and self.variant is not None
        ):
            raise ScheduleContractError("family-only evidence cannot identify a variant")


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2VariantResolution:
    """Auditable Krea 2 variant resolution or non-resolution."""

    status: Krea2VariantResolutionStatus
    resolved_variant: Krea2Variant | None
    suggested_variant: Krea2Variant | None
    confidence: Krea2VariantConfidence
    decisive_source: Krea2VariantEvidenceSource | None
    evidence: tuple[Krea2VariantEvidence, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, Krea2VariantResolutionStatus):
            raise ScheduleContractError("variant resolution status is unsupported")
        if self.resolved_variant is not None and not isinstance(
            self.resolved_variant, Krea2Variant
        ):
            raise ScheduleContractError("resolved variant is unsupported")
        if self.suggested_variant is not None and not isinstance(
            self.suggested_variant, Krea2Variant
        ):
            raise ScheduleContractError("suggested variant is unsupported")
        if not isinstance(self.confidence, Krea2VariantConfidence):
            raise ScheduleContractError("resolution confidence is unsupported")
        if self.decisive_source is not None and not isinstance(
            self.decisive_source, Krea2VariantEvidenceSource
        ):
            raise ScheduleContractError("decisive evidence source is unsupported")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, Krea2VariantEvidence) for item in self.evidence
        ):
            raise ScheduleContractError("resolution evidence must be an immutable tuple")
        if not isinstance(self.warnings, tuple):
            raise ScheduleContractError("resolution warnings must be an immutable tuple")
        for warning in self.warnings:
            _require_token("warning", warning)
        if len(self.warnings) != len(set(self.warnings)):
            raise ScheduleContractError("resolution warnings contain duplicates")

        if self.status is Krea2VariantResolutionStatus.RESOLVED:
            if (
                self.resolved_variant is None
                or self.suggested_variant is not None
                or self.confidence
                not in {
                    Krea2VariantConfidence.AUTHORITATIVE,
                    Krea2VariantConfidence.VERIFIED,
                }
                or self.decisive_source not in _RESOLVING_SOURCES
            ):
                raise ScheduleContractError("resolved variant result is inconsistent")
        elif self.status is Krea2VariantResolutionStatus.SUGGESTED:
            if (
                self.resolved_variant is not None
                or self.suggested_variant is None
                or self.confidence
                not in {
                    Krea2VariantConfidence.CORROBORATING,
                    Krea2VariantConfidence.WEAK,
                }
                or self.decisive_source not in _SUGGESTION_SOURCES
            ):
                raise ScheduleContractError("suggested variant result is inconsistent")
        elif (
            self.resolved_variant is not None
            or self.suggested_variant is not None
            or self.confidence is not Krea2VariantConfidence.NONE
            or self.decisive_source is not None
        ):
            raise ScheduleContractError("unresolved variant result is inconsistent")


class Krea2VariantResolutionError(ScheduleContractError):
    """Strict official resolution could not produce one trusted variant."""


def _evidence(
    source: Krea2VariantEvidenceSource,
    variant: Krea2Variant | None,
    reason_code: str,
) -> Krea2VariantEvidence:
    confidence = _SOURCE_CONFIDENCE[source]
    if source is Krea2VariantEvidenceSource.VERIFIED_SHA256 and variant is None:
        confidence = Krea2VariantConfidence.NONE
    return Krea2VariantEvidence(
        source=source,
        confidence=confidence,
        variant=variant,
        reason_code=reason_code,
    )


def _header_variant(value: object, *, field_name: str) -> Krea2Variant:
    if field_name == "krea2_variant":
        return _coerce_variant(value, field_name=field_name)  # type: ignore[arg-type]
    if isinstance(value, bool):
        return Krea2Variant.TURBO if value else Krea2Variant.RAW
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return Krea2Variant.TURBO if value.casefold() == "true" else Krea2Variant.RAW
    raise ScheduleContractError("local is_distilled metadata must be a boolean")


def collect_krea2_variant_evidence(
    *,
    explicit_variant: Krea2Variant | str | None = None,
    trusted_profile_id: str | None = None,
    trusted_framework_metadata: Mapping[str, object] | None = None,
    checkpoint_sha256: str | None = None,
    safetensors_metadata: Mapping[str, object] | None = None,
    tensor_keys: Iterable[str] = (),
    model_class: str | None = None,
    filename: str | None = None,
) -> tuple[Krea2VariantEvidence, ...]:
    """Normalize caller-labeled evidence without performing host or filesystem access."""

    collected: list[Krea2VariantEvidence] = []

    if explicit_variant is not None:
        collected.append(
            _evidence(
                Krea2VariantEvidenceSource.EXPLICIT_SELECTION,
                _coerce_variant(explicit_variant, field_name="explicit_variant"),
                "explicit.variant",
            )
        )

    if trusted_profile_id is not None:
        if not isinstance(trusted_profile_id, str) or trusted_profile_id not in _PROFILE_VARIANTS:
            raise ScheduleContractError("trusted_profile_id is not a known Krea 2 profile")
        collected.append(
            _evidence(
                Krea2VariantEvidenceSource.TRUSTED_PROFILE_METADATA,
                Krea2Variant(_PROFILE_VARIANTS[trusted_profile_id]),
                f"profile.{_PROFILE_VARIANTS[trusted_profile_id]}",
            )
        )

    if trusted_framework_metadata is not None:
        if not isinstance(trusted_framework_metadata, Mapping):
            raise ScheduleContractError("trusted framework metadata must be a mapping")
        if trusted_framework_metadata.get("_class_name") != "Krea2Pipeline":
            raise ScheduleContractError("trusted framework metadata is not Krea2Pipeline")
        is_distilled = trusted_framework_metadata.get("is_distilled")
        if not isinstance(is_distilled, bool):
            raise ScheduleContractError(
                "trusted Krea2Pipeline metadata requires boolean is_distilled"
            )
        framework_variant = Krea2Variant.TURBO if is_distilled else Krea2Variant.RAW
        collected.append(
            _evidence(
                Krea2VariantEvidenceSource.TRUSTED_FRAMEWORK_METADATA,
                framework_variant,
                f"framework.{framework_variant.value}",
            )
        )

    if checkpoint_sha256 is not None:
        if not isinstance(checkpoint_sha256, str) or not _SHA256_PATTERN.fullmatch(
            checkpoint_sha256
        ):
            raise ScheduleContractError("checkpoint_sha256 must be 64 hexadecimal characters")
        normalized_hash = checkpoint_sha256.casefold()
        hash_variant = _HASH_VARIANTS.get(normalized_hash)
        collected.append(
            _evidence(
                Krea2VariantEvidenceSource.VERIFIED_SHA256,
                Krea2Variant(hash_variant) if hash_variant is not None else None,
                f"sha256.{hash_variant}" if hash_variant is not None else "sha256.unverified",
            )
        )

    if safetensors_metadata is not None:
        if not isinstance(safetensors_metadata, Mapping):
            raise ScheduleContractError("safetensors metadata must be a mapping")
        for field_name in ("krea2_variant", "is_distilled"):
            if field_name in safetensors_metadata:
                header_variant = _header_variant(
                    safetensors_metadata[field_name],
                    field_name=field_name,
                )
                collected.append(
                    _evidence(
                        Krea2VariantEvidenceSource.LOCAL_HEADER_SIGNAL,
                        header_variant,
                        f"header.{field_name}.{header_variant.value}",
                    )
                )

    normalized_tensor_keys: list[str] = []
    try:
        for key in tensor_keys:
            if not isinstance(key, str) or not key:
                raise ScheduleContractError("tensor keys must be non-empty strings")
            normalized_tensor_keys.append(key)
    except TypeError as exc:
        raise ScheduleContractError("tensor_keys must be iterable") from exc
    if any(key.endswith("txtfusion.projector.weight") for key in normalized_tensor_keys):
        collected.append(
            _evidence(
                Krea2VariantEvidenceSource.LOCAL_TENSOR_SIGNAL,
                None,
                "tensor.krea2_family",
            )
        )

    if model_class is not None:
        if not isinstance(model_class, str) or not model_class:
            raise ScheduleContractError("model_class must be a non-empty string")
        if model_class.rsplit(".", maxsplit=1)[-1].casefold() == "krea2":
            collected.append(
                _evidence(
                    Krea2VariantEvidenceSource.MODEL_CLASS_SIGNAL,
                    None,
                    "model_class.krea2_family",
                )
            )

    if filename is not None:
        if not isinstance(filename, str) or not filename:
            raise ScheduleContractError("filename must be a non-empty string")
        basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
        for pattern, variant in (
            (_RAW_FILENAME_PATTERN, Krea2Variant.RAW),
            (_TURBO_FILENAME_PATTERN, Krea2Variant.TURBO),
        ):
            if pattern.search(basename):
                collected.append(
                    _evidence(
                        Krea2VariantEvidenceSource.FILENAME_SIGNAL,
                        variant,
                        f"filename.{variant.value}_token",
                    )
                )

    return tuple(sorted(collected, key=lambda item: _SOURCE_RANK[item.source]))


def _unresolved_result(
    status: Krea2VariantResolutionStatus,
    evidence: tuple[Krea2VariantEvidence, ...],
    *warnings: str,
) -> Krea2VariantResolution:
    return Krea2VariantResolution(
        status=status,
        resolved_variant=None,
        suggested_variant=None,
        confidence=Krea2VariantConfidence.NONE,
        decisive_source=None,
        evidence=evidence,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def resolve_krea2_variant(
    *,
    strict_official: bool = True,
    explicit_variant: Krea2Variant | str | None = None,
    trusted_profile_id: str | None = None,
    trusted_framework_metadata: Mapping[str, object] | None = None,
    checkpoint_sha256: str | None = None,
    safetensors_metadata: Mapping[str, object] | None = None,
    tensor_keys: Iterable[str] = (),
    model_class: str | None = None,
    filename: str | None = None,
) -> Krea2VariantResolution:
    """Resolve trusted Krea 2 evidence or expose a non-authoritative suggestion."""

    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be a boolean")
    evidence = collect_krea2_variant_evidence(
        explicit_variant=explicit_variant,
        trusted_profile_id=trusted_profile_id,
        trusted_framework_metadata=trusted_framework_metadata,
        checkpoint_sha256=checkpoint_sha256,
        safetensors_metadata=safetensors_metadata,
        tensor_keys=tensor_keys,
        model_class=model_class,
        filename=filename,
    )
    resolving = tuple(
        item for item in evidence if item.source in _RESOLVING_SOURCES and item.variant is not None
    )
    resolving_variants = {item.variant for item in resolving}

    if len(resolving_variants) > 1:
        result = _unresolved_result(
            Krea2VariantResolutionStatus.CONFLICT,
            evidence,
            "conflicting_resolving_evidence",
        )
    elif len(resolving_variants) == 1:
        resolved_variant = next(iter(resolving_variants))
        decisive = resolving[0]
        warnings: tuple[str, ...] = ()
        if any(
            item.variant is not None and item.variant is not resolved_variant
            for item in evidence
            if item.source not in _RESOLVING_SOURCES
        ):
            warnings = ("lower_confidence_evidence_disagrees",)
        if any(item.reason_code == "sha256.unverified" for item in evidence):
            warnings += ("checkpoint_hash_not_verified",)
        result = Krea2VariantResolution(
            status=Krea2VariantResolutionStatus.RESOLVED,
            resolved_variant=resolved_variant,
            suggested_variant=None,
            confidence=decisive.confidence,
            decisive_source=decisive.source,
            evidence=evidence,
            warnings=warnings,
        )
    else:
        suggestions = tuple(
            item
            for item in evidence
            if item.source in _SUGGESTION_SOURCES and item.variant is not None
        )
        suggestion_variants = {item.variant for item in suggestions}
        base_warnings: list[str] = []
        if any(item.reason_code == "sha256.unverified" for item in evidence):
            base_warnings.append("checkpoint_hash_not_verified")
        if any(item.confidence is Krea2VariantConfidence.FAMILY_ONLY for item in evidence):
            base_warnings.append("krea2_family_does_not_identify_variant")

        if len(suggestion_variants) > 1:
            result = _unresolved_result(
                Krea2VariantResolutionStatus.AMBIGUOUS,
                evidence,
                *base_warnings,
                "conflicting_suggestion_evidence",
            )
        elif len(suggestion_variants) == 1:
            suggested_variant = next(iter(suggestion_variants))
            decisive = suggestions[0]
            result = Krea2VariantResolution(
                status=Krea2VariantResolutionStatus.SUGGESTED,
                resolved_variant=None,
                suggested_variant=suggested_variant,
                confidence=decisive.confidence,
                decisive_source=decisive.source,
                evidence=evidence,
                warnings=(*base_warnings, "non_authoritative_variant_suggestion"),
            )
        else:
            result = _unresolved_result(
                Krea2VariantResolutionStatus.AMBIGUOUS,
                evidence,
                *base_warnings,
                "insufficient_variant_evidence",
            )

    if strict_official and result.status is not Krea2VariantResolutionStatus.RESOLVED:
        raise Krea2VariantResolutionError(
            f"Krea 2 official variant resolution is {result.status.value}; "
            "provide explicit, trusted, or verified evidence"
        )
    return result
