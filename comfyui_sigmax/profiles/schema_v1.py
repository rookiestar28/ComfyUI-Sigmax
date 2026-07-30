"""Frozen dependency-free model-profile schema v1."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Final, TypeAlias, cast

from comfyui_sigmax.core import (
    EvidenceLevel,
    ModelCapabilities,
    PredictionType,
    ProfileCapabilities,
    SamplerCapabilities,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    TerminalSigma,
    TransformContract,
    TransformStage,
    canonical_projection_bytes,
    float_to_ieee_hex,
    validate_transform_chain,
)

PROFILE_SCHEMA_ID: Final = "sigmax.model-profile/1"
PROFILE_SCHEMA_VERSION: Final = "1"

_NUMERICAL_SCHEMA: Final = "sigmax.numerical-schedule/1"
_CONSTRUCTION_SCHEMA: Final = "sigmax.schedule-artifact/1"
_ENVELOPE_SCHEMA: Final = "sigmax.schedule-artifact-envelope/1"
_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_FIELD_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_LICENSE_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SECRET_NAME_PATTERN: Final = re.compile(
    r"(?:^|_)(?:api_?key|access_key|private_key|secret|password|passwd|credential|cookie|"
    r"token|authorization|auth)(?:_|$)"
)
_PRIVATE_PATH_PATTERN: Final = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)
_MAX_PUBLIC_TEXT: Final = 512
_MAX_INTEROPERABLE_INTEGER: Final = (2**53) - 1

ProfileScalar: TypeAlias = str | int | bool | float | None


def _require_public_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScheduleContractError(f"{field_name} must be a non-empty string")
    if len(value) > _MAX_PUBLIC_TEXT:
        raise ScheduleContractError(f"{field_name} exceeds the public text limit")
    if _PRIVATE_PATH_PATTERN.search(value):
        raise ScheduleContractError(f"{field_name} must not contain a private local path")
    return value


def _require_identifier(field_name: str, value: object) -> str:
    text = _require_public_text(field_name, value)
    if not text.isascii() or not _IDENTIFIER_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable lowercase identifier")
    return text


def _require_version(field_name: str, value: object, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    return _require_public_text(field_name, value)


def _require_https(field_name: str, value: object) -> str:
    text = _require_public_text(field_name, value)
    if not text.startswith("https://"):
        raise ScheduleContractError(f"{field_name} must use HTTPS")
    return text


def _require_bool(field_name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ScheduleContractError(f"{field_name} must be boolean")


def _require_positive_integer(field_name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleContractError(f"{field_name} must be a positive integer")
    return value


def _require_finite(field_name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ScheduleContractError(f"{field_name} must be a finite number")
    return float(value)


def _require_tuple(field_name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ScheduleContractError(f"{field_name} must be a tuple")
    return value


def _require_public_text_tuple(
    field_name: str,
    value: object,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    values = _require_tuple(field_name, value)
    if not allow_empty and not values:
        raise ScheduleContractError(f"{field_name} must not be empty")
    normalized = tuple(_require_public_text(field_name, item) for item in values)
    if len(normalized) != len(set(normalized)):
        raise ScheduleContractError(f"{field_name} contains duplicate values")
    return normalized


def _require_identifier_tuple(
    field_name: str,
    value: object,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    values = _require_tuple(field_name, value)
    if not allow_empty and not values:
        raise ScheduleContractError(f"{field_name} must not be empty")
    normalized = tuple(_require_identifier(field_name, item) for item in values)
    if len(normalized) != len(set(normalized)):
        raise ScheduleContractError(f"{field_name} contains duplicate values")
    return normalized


def _require_commit(field_name: str, value: object) -> str:
    text = _require_public_text(field_name, value)
    if not _COMMIT_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a pinned lowercase 40-hex revision")
    return text


def _require_canonical_fields(field_name: str, fields: object) -> tuple[ProfileField, ...]:
    values = _require_tuple(field_name, fields)
    if not all(isinstance(item, ProfileField) for item in values):
        raise ScheduleContractError(f"{field_name} must contain ProfileField values")
    typed = cast(tuple[ProfileField, ...], values)
    names = tuple(item.name for item in typed)
    if len(names) != len(set(names)):
        raise ScheduleContractError(f"{field_name} contains duplicate names")
    if names != tuple(sorted(names)):
        raise ScheduleContractError(f"{field_name} must use canonical field order")
    return typed


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileField:
    """One controlled scalar field in a profile declaration."""

    name: str
    value: ProfileScalar

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.isascii()
            or not _FIELD_PATTERN.fullmatch(self.name)
        ):
            raise ScheduleContractError("profile field name must be an ASCII identifier")
        if _SECRET_NAME_PATTERN.search(self.name) or self.name.endswith("_path"):
            raise ScheduleContractError("profile field name is secret-like or path-like")

        if self.value is None or isinstance(self.value, bool):
            return
        if isinstance(self.value, int):
            if abs(self.value) > _MAX_INTEROPERABLE_INTEGER:
                raise ScheduleContractError("profile integer exceeds interoperable JSON range")
            return
        if isinstance(self.value, float):
            _require_finite("profile field value", self.value)
            return
        if isinstance(self.value, str):
            _require_public_text("profile field value", self.value)
            return
        raise ScheduleContractError("unsupported profile field value")


@dataclass(frozen=True, slots=True, kw_only=True)
class LicenseDeclaration:
    """A separately versioned license declaration for one resource."""

    declaration_version: str
    identifier: str
    name: str
    url: str

    def __post_init__(self) -> None:
        if self.declaration_version != "1":
            raise ScheduleContractError("license declaration_version must be 1")
        if (
            not isinstance(self.identifier, str)
            or not self.identifier.isascii()
            or not _LICENSE_PATTERN.fullmatch(self.identifier)
        ):
            raise ScheduleContractError("license identifier must be SPDX-like or LicenseRef-*")
        _require_public_text("license name", self.name)
        _require_https("license URL", self.url)


def _validate_resource_provenance(
    *,
    record_version: object,
    resource_id: object,
    resource_version: object,
    revision: object,
    url: object,
    license_declaration: object,
    locators: object | None,
) -> None:
    if record_version != "1":
        raise ScheduleContractError("provenance record_version must be 1")
    _require_identifier("resource identifier", resource_id)
    _require_version("resource_version", resource_version, allow_none=True)
    _require_commit("resource revision", revision)
    _require_https("resource URL", url)
    if not isinstance(license_declaration, LicenseDeclaration):
        raise ScheduleContractError("resource license must be a LicenseDeclaration")
    if locators is not None:
        values = _require_public_text_tuple("resource locators", locators)
        if values != tuple(sorted(values)):
            raise ScheduleContractError("resource locators must use canonical order")


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftwareSourceProvenance:
    """Versioned provenance and license for inference software."""

    record_version: str
    source_id: str
    resource_version: str | None
    revision: str
    url: str
    license: LicenseDeclaration
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_resource_provenance(
            record_version=self.record_version,
            resource_id=self.source_id,
            resource_version=self.resource_version,
            revision=self.revision,
            url=self.url,
            license_declaration=self.license,
            locators=self.locators,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameworkProvenance:
    """Versioned provenance and license for one supporting framework."""

    record_version: str
    framework_id: str
    resource_version: str | None
    revision: str
    url: str
    license: LicenseDeclaration
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_resource_provenance(
            record_version=self.record_version,
            resource_id=self.framework_id,
            resource_version=self.resource_version,
            revision=self.revision,
            url=self.url,
            license_declaration=self.license,
            locators=self.locators,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelWeightProvenance:
    """Versioned provenance and license for one model-weight resource."""

    record_version: str
    weight_id: str
    resource_version: str
    revision: str
    sha256: str
    url: str
    license: LicenseDeclaration

    def __post_init__(self) -> None:
        _validate_resource_provenance(
            record_version=self.record_version,
            resource_id=self.weight_id,
            resource_version=self.resource_version,
            revision=self.revision,
            url=self.url,
            license_declaration=self.license,
            locators=None,
        )
        if self.resource_version is None:
            raise ScheduleContractError("model-weight resource_version is required")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ScheduleContractError("model-weight sha256 must be lowercase 64-hex")


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseGridDeclaration:
    """Profile-owned base-grid identity and domain."""

    identifier: str
    output_domain: SigmaDomain
    terminal_included: bool
    parameters: tuple[ProfileField, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier("base-grid identifier", self.identifier)
        if (
            not isinstance(self.output_domain, SigmaDomain)
            or self.output_domain is SigmaDomain.MODEL_NATIVE
        ):
            raise ScheduleContractError("base-grid output_domain must be a non-opaque SigmaDomain")
        _require_bool("base-grid terminal_included", self.terminal_included)
        _require_canonical_fields("base-grid parameters", self.parameters)


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformDeclaration:
    """One ordered transform plus complete profile parameters."""

    identifier: str
    stage: TransformStage
    input_domain: SigmaDomain
    output_domain: SigmaDomain
    parameters: tuple[ProfileField, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier("transform identifier", self.identifier)
        if not isinstance(self.stage, TransformStage):
            raise ScheduleContractError("transform stage is unsupported")
        if not isinstance(self.input_domain, SigmaDomain) or not isinstance(
            self.output_domain, SigmaDomain
        ):
            raise ScheduleContractError("transform domains are unsupported")
        if SigmaDomain.MODEL_NATIVE in {self.input_domain, self.output_domain}:
            raise ScheduleContractError("profile transforms cannot use MODEL_NATIVE")
        _require_canonical_fields("transform parameters", self.parameters)

    def contract(self) -> TransformContract:
        """Return the existing core contract used for chain validation."""

        return TransformContract(
            name=self.identifier,
            stage=self.stage,
            input_domain=self.input_domain,
            output_domain=self.output_domain,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TerminalDeclaration:
    """Profile terminal policy and effective terminal value."""

    policy: TerminalPolicy
    sigma: TerminalSigma
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.policy, TerminalPolicy):
            raise ScheduleContractError("terminal policy is unsupported")
        if not isinstance(self.sigma, TerminalSigma):
            raise ScheduleContractError("terminal sigma is unsupported")
        value = _require_finite("terminal value", self.value)
        if self.sigma is TerminalSigma.ZERO and value != 0.0:
            raise ScheduleContractError("zero terminal sigma requires value 0")
        if self.sigma is TerminalSigma.NONZERO and value <= 0.0:
            raise ScheduleContractError("nonzero terminal sigma requires a positive value")
        if self.policy is TerminalPolicy.APPEND_ZERO and self.sigma is not TerminalSigma.ZERO:
            raise ScheduleContractError("APPEND_ZERO requires a zero terminal sigma")


@dataclass(frozen=True, slots=True, kw_only=True)
class SlicingDeclaration:
    """Profile support for terminal-inclusive slicing behavior."""

    supports_step_range: bool
    supports_denoise_tail: bool
    zero_denoise_is_empty: bool

    def __post_init__(self) -> None:
        _require_bool("supports_step_range", self.supports_step_range)
        _require_bool("supports_denoise_tail", self.supports_denoise_tail)
        _require_bool("zero_denoise_is_empty", self.zero_denoise_is_empty)
        if self.zero_denoise_is_empty and not self.supports_denoise_tail:
            raise ScheduleContractError(
                "zero_denoise_is_empty requires denoise-tail slicing support"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class GuidanceDeclaration:
    """One model-to-host guidance convention and exact values."""

    model_convention: str
    host_convention: str
    model_value: float
    host_value: float

    def __post_init__(self) -> None:
        _require_identifier("model guidance convention", self.model_convention)
        _require_identifier("host guidance convention", self.host_convention)
        _require_finite("model guidance value", self.model_value)
        _require_finite("host guidance value", self.host_value)


@dataclass(frozen=True, slots=True, kw_only=True)
class StepRangeDeclaration:
    """Reference and permitted step-count policy for one recipe."""

    minimum: int
    maximum: int | None
    default: int
    reference_steps: tuple[int, ...]
    allow_modified: bool

    def __post_init__(self) -> None:
        minimum = _require_positive_integer("minimum steps", self.minimum)
        default = _require_positive_integer("default steps", self.default)
        maximum = self.maximum
        if maximum is not None:
            maximum = _require_positive_integer("maximum steps", maximum)
            if maximum < minimum:
                raise ScheduleContractError("maximum steps must not be below minimum")
        if default < minimum or (maximum is not None and default > maximum):
            raise ScheduleContractError("default steps must be inside the declared range")

        values = _require_tuple("reference_steps", self.reference_steps)
        if not values:
            raise ScheduleContractError("reference_steps must not be empty")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values
        ):
            raise ScheduleContractError("reference_steps must contain positive integers")
        typed = cast(tuple[int, ...], values)
        if len(typed) != len(set(typed)) or typed != tuple(sorted(typed)):
            raise ScheduleContractError("reference_steps must be unique and sorted")
        if any(value < minimum or (maximum is not None and value > maximum) for value in typed):
            raise ScheduleContractError("reference_steps must be inside the declared range")
        _require_bool("allow_modified", self.allow_modified)
        if not self.allow_modified and (
            maximum != minimum or default != minimum or typed != (default,)
        ):
            raise ScheduleContractError(
                "non-modifiable recipes require one exact default/reference step count"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceRecipe:
    """One evidence-bearing guidance and step recipe."""

    recipe_id: str
    evidence: EvidenceLevel
    source_id: str
    steps: StepRangeDeclaration
    guidance: GuidanceDeclaration

    def __post_init__(self) -> None:
        _require_identifier("recipe_id", self.recipe_id)
        if not isinstance(self.evidence, EvidenceLevel):
            raise ScheduleContractError("recipe evidence is unsupported")
        _require_identifier("recipe source_id", self.source_id)
        if not isinstance(self.steps, StepRangeDeclaration):
            raise ScheduleContractError("recipe steps must be a StepRangeDeclaration")
        if not isinstance(self.guidance, GuidanceDeclaration):
            raise ScheduleContractError("recipe guidance must be a GuidanceDeclaration")


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectionDeclaration:
    """Fail-closed model/variant evidence policy."""

    strategy_id: str
    strict_default: bool
    ambiguity_requires_explicit: bool
    resolving_sources: tuple[str, ...]
    suggestion_sources: tuple[str, ...]
    family_only_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier("detection strategy_id", self.strategy_id)
        _require_bool("strict_default", self.strict_default)
        _require_bool("ambiguity_requires_explicit", self.ambiguity_requires_explicit)
        resolving = _require_identifier_tuple("resolving_sources", self.resolving_sources)
        suggestions = _require_identifier_tuple(
            "suggestion_sources",
            self.suggestion_sources,
            allow_empty=True,
        )
        family_only = _require_identifier_tuple(
            "family_only_sources",
            self.family_only_sources,
            allow_empty=True,
        )
        all_sources = resolving + suggestions + family_only
        if len(all_sources) != len(set(all_sources)):
            raise ScheduleContractError("detection evidence classes must be disjoint")
        if self.strict_default and not self.ambiguity_requires_explicit:
            raise ScheduleContractError("strict detection requires explicit selection on ambiguity")


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactVersionDeclaration:
    """Construction-artifact schema versions supported by this profile."""

    numerical_schema: str
    construction_schema: str
    envelope_schema: str

    def __post_init__(self) -> None:
        if (
            self.numerical_schema,
            self.construction_schema,
            self.envelope_schema,
        ) != (
            _NUMERICAL_SCHEMA,
            _CONSTRUCTION_SCHEMA,
            _ENVELOPE_SCHEMA,
        ):
            raise ScheduleContractError("profile schema v1 requires current artifact v1 formats")


def _require_typed_tuple(
    field_name: str,
    values: object,
    expected_type: type[object],
    *,
    allow_empty: bool = False,
) -> tuple[object, ...]:
    raw = _require_tuple(field_name, values)
    if not allow_empty and not raw:
        raise ScheduleContractError(f"{field_name} must not be empty")
    if not all(isinstance(item, expected_type) for item in raw):
        raise ScheduleContractError(f"{field_name} contains an unsupported value")
    return raw


def _require_sorted_unique_ids(field_name: str, identifiers: tuple[str, ...]) -> None:
    if len(identifiers) != len(set(identifiers)):
        raise ScheduleContractError(f"{field_name} contains duplicate identifiers")
    if identifiers != tuple(sorted(identifiers)):
        raise ScheduleContractError(f"{field_name} must use canonical identifier order")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileSchemaV1:
    """Complete immutable runtime contract for a Sigmax model profile."""

    schema_id: str
    schema_version: str
    profile_id: str
    profile_version: str
    display_name: str
    model_family: str
    model_variant: str
    evidence: EvidenceLevel
    primary_source_id: str
    prediction_type: PredictionType
    sigma_domain: SigmaDomain
    ownership: ScheduleOwnership
    base_grid: BaseGridDeclaration | None
    transforms: tuple[TransformDeclaration, ...]
    terminal: TerminalDeclaration
    slicing: SlicingDeclaration
    recipes: tuple[InferenceRecipe, ...]
    detection: DetectionDeclaration
    model_capabilities: ModelCapabilities
    profile_capabilities: ProfileCapabilities
    reference_sampler_capabilities: SamplerCapabilities
    artifact_versions: ArtifactVersionDeclaration
    software_sources: tuple[SoftwareSourceProvenance, ...]
    frameworks: tuple[FrameworkProvenance, ...]
    model_weights: tuple[ModelWeightProvenance, ...]
    parameters: tuple[ProfileField, ...]
    known_limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_id != PROFILE_SCHEMA_ID or self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ScheduleContractError("profile schema identifier/version is unsupported")
        _require_identifier("profile_id", self.profile_id)
        _require_version("profile_version", self.profile_version)
        _require_public_text("display_name", self.display_name)
        _require_identifier("model_family", self.model_family)
        _require_identifier("model_variant", self.model_variant)
        if not isinstance(self.evidence, EvidenceLevel):
            raise ScheduleContractError("profile evidence is unsupported")
        _require_identifier("primary_source_id", self.primary_source_id)
        if not isinstance(self.prediction_type, PredictionType):
            raise ScheduleContractError("profile prediction_type is unsupported")
        if not isinstance(self.sigma_domain, SigmaDomain):
            raise ScheduleContractError("profile sigma_domain is unsupported")
        if self.ownership is not ScheduleOwnership.EXTERNAL_SIGMAS:
            raise ScheduleContractError(
                "profile schema v1 currently requires EXTERNAL_SIGMAS ownership"
            )
        if self.sigma_domain is SigmaDomain.MODEL_NATIVE:
            raise ScheduleContractError("external profile cannot use MODEL_NATIVE sigma domain")
        if not isinstance(self.base_grid, BaseGridDeclaration):
            raise ScheduleContractError("external profile requires a BaseGridDeclaration")
        if self.base_grid.output_domain is not self.sigma_domain:
            raise ScheduleContractError("base-grid domain must match profile sigma_domain")
        if self.base_grid.terminal_included:
            raise ScheduleContractError("profile v1 base grids must exclude the terminal")

        transforms = _require_typed_tuple(
            "transforms",
            self.transforms,
            TransformDeclaration,
        )
        typed_transforms = cast(tuple[TransformDeclaration, ...], transforms)
        final_domain = validate_transform_chain(
            self.ownership,
            self.sigma_domain,
            tuple(transform.contract() for transform in typed_transforms),
        )
        if final_domain is not self.sigma_domain:
            raise ScheduleContractError("profile transforms must retain the declared sigma domain")
        terminal_transforms = tuple(
            transform
            for transform in typed_transforms
            if transform.stage is TransformStage.TERMINAL
        )
        if len(terminal_transforms) != 1:
            raise ScheduleContractError("profile requires exactly one terminal transform")
        if not isinstance(self.terminal, TerminalDeclaration):
            raise ScheduleContractError("terminal must be a TerminalDeclaration")
        if self.terminal.policy is TerminalPolicy.APPEND_ZERO and (
            terminal_transforms[0].identifier != "terminal.append_zero"
        ):
            raise ScheduleContractError("terminal declaration and transform identifier disagree")
        if not isinstance(self.slicing, SlicingDeclaration):
            raise ScheduleContractError("slicing must be a SlicingDeclaration")

        recipes = cast(
            tuple[InferenceRecipe, ...],
            _require_typed_tuple("recipes", self.recipes, InferenceRecipe),
        )
        recipe_ids = tuple(recipe.recipe_id for recipe in recipes)
        _require_sorted_unique_ids("recipes", recipe_ids)
        if not isinstance(self.detection, DetectionDeclaration):
            raise ScheduleContractError("detection must be a DetectionDeclaration")
        if not isinstance(self.artifact_versions, ArtifactVersionDeclaration):
            raise ScheduleContractError("artifact_versions must be an ArtifactVersionDeclaration")

        software = cast(
            tuple[SoftwareSourceProvenance, ...],
            _require_typed_tuple(
                "software_sources",
                self.software_sources,
                SoftwareSourceProvenance,
            ),
        )
        frameworks = cast(
            tuple[FrameworkProvenance, ...],
            _require_typed_tuple("frameworks", self.frameworks, FrameworkProvenance),
        )
        weights = cast(
            tuple[ModelWeightProvenance, ...],
            _require_typed_tuple("model_weights", self.model_weights, ModelWeightProvenance),
        )
        source_ids = tuple(item.source_id for item in software)
        framework_ids = tuple(item.framework_id for item in frameworks)
        weight_ids = tuple(item.weight_id for item in weights)
        _require_sorted_unique_ids("software_sources", source_ids)
        _require_sorted_unique_ids("frameworks", framework_ids)
        _require_sorted_unique_ids("model_weights", weight_ids)
        all_provenance_ids = source_ids + framework_ids + weight_ids
        if len(all_provenance_ids) != len(set(all_provenance_ids)):
            raise ScheduleContractError("provenance identifiers must be globally unique")
        evidence_source_ids = set(source_ids + framework_ids)
        if self.primary_source_id not in evidence_source_ids:
            raise ScheduleContractError("primary_source_id is missing from source provenance")
        if any(recipe.source_id not in evidence_source_ids for recipe in recipes):
            raise ScheduleContractError("recipe source_id is missing from source provenance")

        _require_canonical_fields("profile parameters", self.parameters)
        _require_public_text_tuple("known_limitations", self.known_limitations)
        self._validate_capabilities()

    def _validate_capabilities(self) -> None:
        if not isinstance(self.model_capabilities, ModelCapabilities):
            raise ScheduleContractError("model_capabilities must be ModelCapabilities")
        if not isinstance(self.profile_capabilities, ProfileCapabilities):
            raise ScheduleContractError("profile_capabilities must be ProfileCapabilities")
        if not isinstance(self.reference_sampler_capabilities, SamplerCapabilities):
            raise ScheduleContractError(
                "reference_sampler_capabilities must be SamplerCapabilities"
            )

        model = self.model_capabilities
        profile = self.profile_capabilities
        sampler = self.reference_sampler_capabilities
        if (model.model_family, model.model_variant) != (
            self.model_family,
            self.model_variant,
        ):
            raise ScheduleContractError("model capabilities do not match profile identity")
        if (
            self.prediction_type not in model.accepted_prediction_types
            or self.sigma_domain not in model.accepted_sigma_domains
            or self.ownership not in model.accepted_ownerships
        ):
            raise ScheduleContractError("model capabilities do not accept profile semantics")
        if (
            profile.profile_id,
            profile.profile_version,
            profile.model_family,
            profile.model_variant,
            profile.prediction_type,
            profile.sigma_domain,
            profile.ownership,
            profile.terminal_sigma,
        ) != (
            self.profile_id,
            self.profile_version,
            self.model_family,
            self.model_variant,
            self.prediction_type,
            self.sigma_domain,
            self.ownership,
            self.terminal.sigma,
        ):
            raise ScheduleContractError("profile capabilities do not match profile schema")
        if sampler.sampler_id not in profile.reference_sampler_ids:
            raise ScheduleContractError("reference sampler is not declared by profile capabilities")
        if (
            self.prediction_type not in sampler.accepted_prediction_types
            or self.sigma_domain not in sampler.accepted_sigma_domains
            or self.ownership not in sampler.accepted_ownerships
        ):
            raise ScheduleContractError("reference sampler does not accept profile semantics")


def _field_value_projection(value: ProfileScalar) -> object:
    if isinstance(value, float):
        return {
            "bits": float_to_ieee_hex(value, "float64"),
            "precision": "float64",
        }
    return value


def _field_projection(fields: tuple[ProfileField, ...]) -> dict[str, object]:
    return {field.name: _field_value_projection(field.value) for field in fields}


def _license_projection(license_declaration: LicenseDeclaration) -> dict[str, object]:
    return {
        "declaration_version": license_declaration.declaration_version,
        "identifier": license_declaration.identifier,
        "name": license_declaration.name,
        "url": license_declaration.url,
    }


def _source_projection(
    source: SoftwareSourceProvenance | FrameworkProvenance,
) -> dict[str, object]:
    if isinstance(source, SoftwareSourceProvenance):
        resource_id = source.source_id
    else:
        resource_id = source.framework_id
    return {
        "id": resource_id,
        "license": _license_projection(source.license),
        "locators": list(source.locators),
        "record_version": source.record_version,
        "resource_version": source.resource_version,
        "revision": source.revision,
        "url": source.url,
    }


def _weight_projection(weight: ModelWeightProvenance) -> dict[str, object]:
    return {
        "id": weight.weight_id,
        "license": _license_projection(weight.license),
        "record_version": weight.record_version,
        "resource_version": weight.resource_version,
        "revision": weight.revision,
        "sha256": weight.sha256,
        "url": weight.url,
    }


def _model_capabilities_projection(model: ModelCapabilities) -> dict[str, object]:
    return {
        "accepted_ownerships": [value.value.casefold() for value in model.accepted_ownerships],
        "accepted_prediction_types": [value.value for value in model.accepted_prediction_types],
        "accepted_sigma_domains": [
            value.value.casefold() for value in model.accepted_sigma_domains
        ],
        "model_family": model.model_family,
        "model_variant": model.model_variant,
        "supports_partial_denoise": model.supports_partial_denoise,
        "supports_per_token_timesteps": model.supports_per_token_timesteps,
    }


def _profile_capabilities_projection(profile: ProfileCapabilities) -> dict[str, object]:
    return {
        "allowed_execution_behaviors": [
            value.value for value in profile.allowed_execution_behaviors
        ],
        "allowed_noise_ownerships": [value.value for value in profile.allowed_noise_ownerships],
        "allowed_sampler_state": [value.value for value in profile.allowed_sampler_state],
        "model_family": profile.model_family,
        "model_variant": profile.model_variant,
        "ownership": profile.ownership.value.casefold(),
        "prediction_type": profile.prediction_type.value,
        "profile_id": profile.profile_id,
        "profile_version": profile.profile_version,
        "reference_sampler_ids": list(profile.reference_sampler_ids),
        "sigma_domain": profile.sigma_domain.value.casefold(),
        "supports_partial_denoise": profile.supports_partial_denoise,
        "supports_per_token_timesteps": profile.supports_per_token_timesteps,
        "terminal_sigma": profile.terminal_sigma.value,
    }


def _sampler_capabilities_projection(sampler: SamplerCapabilities) -> dict[str, object]:
    return {
        "accepted_ownerships": [value.value.casefold() for value in sampler.accepted_ownerships],
        "accepted_prediction_types": [value.value for value in sampler.accepted_prediction_types],
        "accepted_sigma_domains": [
            value.value.casefold() for value in sampler.accepted_sigma_domains
        ],
        "execution_behavior": sampler.execution_behavior.value,
        "noise_ownership": sampler.noise_ownership.value,
        "required_state": [value.value for value in sampler.required_state],
        "sampler_id": sampler.sampler_id,
        "sampler_version": sampler.sampler_version,
        "supports_partial_denoise": sampler.supports_partial_denoise,
        "supports_per_token_timesteps": sampler.supports_per_token_timesteps,
        "terminal_requirement": sampler.terminal_requirement.value,
    }


def profile_schema_projection(schema: ProfileSchemaV1) -> dict[str, object]:
    """Return the complete deterministic schema-v1 projection."""

    if not isinstance(schema, ProfileSchemaV1):
        raise ScheduleContractError("schema must be a ProfileSchemaV1")

    base_grid = cast(BaseGridDeclaration, schema.base_grid)
    projection: dict[str, object] = {
        "artifact_versions": {
            "construction": schema.artifact_versions.construction_schema,
            "envelope": schema.artifact_versions.envelope_schema,
            "numerical": schema.artifact_versions.numerical_schema,
        },
        "capabilities": {
            "model": _model_capabilities_projection(schema.model_capabilities),
            "profile": _profile_capabilities_projection(schema.profile_capabilities),
            "reference_sampler": _sampler_capabilities_projection(
                schema.reference_sampler_capabilities
            ),
        },
        "detection": {
            "ambiguity_requires_explicit": schema.detection.ambiguity_requires_explicit,
            "family_only_sources": list(schema.detection.family_only_sources),
            "resolving_sources": list(schema.detection.resolving_sources),
            "strategy_id": schema.detection.strategy_id,
            "strict_default": schema.detection.strict_default,
            "suggestion_sources": list(schema.detection.suggestion_sources),
        },
        "evidence": {
            "level": schema.evidence.value,
            "primary_source_id": schema.primary_source_id,
        },
        "known_limitations": list(schema.known_limitations),
        "parameters": _field_projection(schema.parameters),
        "profile": {
            "display_name": schema.display_name,
            "id": schema.profile_id,
            "model_family": schema.model_family,
            "model_variant": schema.model_variant,
            "version": schema.profile_version,
        },
        "provenance": {
            "frameworks": [_source_projection(item) for item in schema.frameworks],
            "model_weights": [_weight_projection(item) for item in schema.model_weights],
            "software_sources": [_source_projection(item) for item in schema.software_sources],
        },
        "recipes": [
            {
                "evidence": recipe.evidence.value,
                "guidance": {
                    "host_convention": recipe.guidance.host_convention,
                    "host_value": _field_value_projection(recipe.guidance.host_value),
                    "model_convention": recipe.guidance.model_convention,
                    "model_value": _field_value_projection(recipe.guidance.model_value),
                },
                "id": recipe.recipe_id,
                "source_id": recipe.source_id,
                "steps": {
                    "allow_modified": recipe.steps.allow_modified,
                    "default": recipe.steps.default,
                    "maximum": recipe.steps.maximum,
                    "minimum": recipe.steps.minimum,
                    "reference": list(recipe.steps.reference_steps),
                },
            }
            for recipe in schema.recipes
        ],
        "schedule": {
            "base_grid": {
                "id": base_grid.identifier,
                "output_domain": base_grid.output_domain.value.casefold(),
                "parameters": _field_projection(base_grid.parameters),
                "terminal_included": base_grid.terminal_included,
            },
            "ownership": schema.ownership.value.casefold(),
            "prediction_type": schema.prediction_type.value,
            "sigma_domain": schema.sigma_domain.value.casefold(),
            "slicing": {
                "supports_denoise_tail": schema.slicing.supports_denoise_tail,
                "supports_step_range": schema.slicing.supports_step_range,
                "zero_denoise_is_empty": schema.slicing.zero_denoise_is_empty,
            },
            "terminal": {
                "policy": schema.terminal.policy.value.casefold(),
                "sigma": schema.terminal.sigma.value,
                "value": _field_value_projection(schema.terminal.value),
            },
            "transforms": [
                {
                    "from_domain": transform.input_domain.value.casefold(),
                    "id": transform.identifier,
                    "parameters": _field_projection(transform.parameters),
                    "stage": index,
                    "stage_kind": transform.stage.value.casefold(),
                    "to_domain": transform.output_domain.value.casefold(),
                }
                for index, transform in enumerate(schema.transforms)
            ],
        },
        "schema": schema.schema_id,
        "schema_version": schema.schema_version,
    }
    canonical_projection_bytes(projection)
    return projection


def profile_schema_fingerprint(schema: ProfileSchemaV1) -> str:
    """Return the deterministic SHA-256 identity of one complete profile schema."""

    payload = canonical_projection_bytes(profile_schema_projection(schema))
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
