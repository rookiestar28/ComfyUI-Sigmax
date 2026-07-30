"""Typed model/profile/sampler capabilities and pre-execution compatibility."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
)

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MAX_PUBLIC_TEXT = 256


class PredictionType(str, Enum):
    """Model-output semantics consumed by a numerical sampler."""

    FLOW_VELOCITY = "flow_velocity"
    EPSILON = "epsilon"
    V_PREDICTION = "v_prediction"
    SAMPLE = "sample"
    MODEL_NATIVE_OPAQUE = "model_native_opaque"


class TerminalSigma(str, Enum):
    """Effective terminal condition supplied by a profile."""

    ZERO = "zero"
    NONZERO = "nonzero"


class TerminalRequirement(str, Enum):
    """Terminal condition accepted by a sampler."""

    REQUIRES_ZERO = "requires_zero"
    ACCEPTS_EITHER = "accepts_either"
    FORBIDS_ZERO = "forbids_zero"


class ExecutionBehavior(str, Enum):
    """Whether the selected sampler step owns random stochastic behavior."""

    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"


class NoiseOwnership(str, Enum):
    """Component responsible for supplying sampler-step randomness."""

    NONE = "none"
    CALLER = "caller"
    SAMPLER = "sampler"
    MODEL = "model"


class SamplerState(str, Enum):
    """State categories required by a sampler implementation."""

    BEGIN_INDEX = "begin_index"
    STEP_INDEX = "step_index"
    MULTISTEP_HISTORY = "multistep_history"
    RESUME = "resume"


class CompatibilityLevel(str, Enum):
    """Pre-execution compatibility outcome."""

    ALLOW = "allow"
    WARN = "warn"
    REJECT = "reject"


class CapabilityDimension(str, Enum):
    """Canonical dimensions considered by the compatibility evaluator."""

    MODEL_IDENTITY = "model_identity"
    PREDICTION_TYPE = "prediction_type"
    SIGMA_DOMAIN = "sigma_domain"
    SCHEDULE_OWNERSHIP = "schedule_ownership"
    TERMINAL_SIGMA = "terminal_sigma"
    EXECUTION_BEHAVIOR = "execution_behavior"
    NOISE_OWNERSHIP = "noise_ownership"
    SAMPLER_STATE = "sampler_state"
    PARTIAL_DENOISE = "partial_denoise"
    PER_TOKEN_TIMESTEPS = "per_token_timesteps"  # noqa: S105
    REFERENCE_SAMPLER = "reference_sampler"


class CompatibilityReason(str, Enum):
    """Stable ordered compatibility reason codes."""

    MODEL_FAMILY_MISMATCH = "model_family_mismatch"
    MODEL_VARIANT_MISMATCH = "model_variant_mismatch"
    MODEL_PREDICTION_UNSUPPORTED = "model_prediction_unsupported"
    MODEL_SIGMA_DOMAIN_UNSUPPORTED = "model_sigma_domain_unsupported"
    MODEL_OWNERSHIP_UNSUPPORTED = "model_ownership_unsupported"
    SAMPLER_PREDICTION_UNSUPPORTED = "sampler_prediction_unsupported"
    SAMPLER_SIGMA_DOMAIN_UNSUPPORTED = "sampler_sigma_domain_unsupported"
    SAMPLER_OWNERSHIP_UNSUPPORTED = "sampler_ownership_unsupported"
    TERMINAL_REQUIREMENT_MISMATCH = "terminal_requirement_mismatch"
    EXECUTION_BEHAVIOR_MISMATCH = "execution_behavior_mismatch"
    NOISE_OWNERSHIP_MISMATCH = "noise_ownership_mismatch"
    SAMPLER_STATE_UNSUPPORTED = "sampler_state_unsupported"
    PARTIAL_DENOISE_UNSUPPORTED_BY_MODEL = "partial_denoise_unsupported_by_model"
    PARTIAL_DENOISE_UNSUPPORTED_BY_PROFILE = "partial_denoise_unsupported_by_profile"
    PARTIAL_DENOISE_UNSUPPORTED_BY_SAMPLER = "partial_denoise_unsupported_by_sampler"
    PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_MODEL = "per_token_timesteps_unsupported_by_model"  # noqa: S105
    PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_PROFILE = "per_token_timesteps_unsupported_by_profile"  # noqa: S105
    PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_SAMPLER = "per_token_timesteps_unsupported_by_sampler"  # noqa: S105
    SAMPLER_NOT_PROFILE_REFERENCE = "sampler_not_profile_reference"
    COMPATIBLE = "compatible"


_WARNING_REASONS = frozenset({CompatibilityReason.SAMPLER_NOT_PROFILE_REFERENCE})
_REJECTION_REASONS = (
    frozenset(CompatibilityReason) - _WARNING_REASONS - {CompatibilityReason.COMPATIBLE}
)
_REASON_ORDER = {reason: index for index, reason in enumerate(CompatibilityReason)}

EnumValue = TypeVar("EnumValue", bound=Enum)


def _require_public_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScheduleContractError(f"{field_name} must be a non-empty string")
    if len(value) > _MAX_PUBLIC_TEXT:
        raise ScheduleContractError(f"{field_name} exceeds the public text limit")
    return value


def _require_identifier(field_name: str, value: object) -> str:
    text = _require_public_text(field_name, value)
    if not _IDENTIFIER_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable lowercase identifier")
    return text


def _require_bool(field_name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ScheduleContractError(f"{field_name} must be boolean")


def _require_enum_tuple(
    field_name: str,
    values: object,
    enum_type: type[EnumValue],
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(values, tuple):
        raise ScheduleContractError(f"{field_name} must be a tuple")
    if not allow_empty and not values:
        raise ScheduleContractError(f"{field_name} must not be empty")
    if not all(isinstance(value, enum_type) for value in values):
        raise ScheduleContractError(f"{field_name} contains an unsupported value")
    if len(values) != len(set(values)):
        raise ScheduleContractError(f"{field_name} contains duplicate values")
    order = {value: index for index, value in enumerate(enum_type)}
    if tuple(values) != tuple(sorted(values, key=order.__getitem__)):
        raise ScheduleContractError(f"{field_name} must use canonical enum order")


def _require_identifier_tuple(
    field_name: str,
    values: object,
) -> None:
    if not isinstance(values, tuple):
        raise ScheduleContractError(f"{field_name} must be a tuple")
    for value in values:
        _require_identifier(field_name, value)
    if len(values) != len(set(values)):
        raise ScheduleContractError(f"{field_name} contains duplicate values")
    if tuple(values) != tuple(sorted(values)):
        raise ScheduleContractError(f"{field_name} must use canonical identifier order")


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapabilities:
    """Resolved model-family capabilities independent of any host implementation."""

    model_family: str
    model_variant: str
    accepted_prediction_types: tuple[PredictionType, ...]
    accepted_sigma_domains: tuple[SigmaDomain, ...]
    accepted_ownerships: tuple[ScheduleOwnership, ...]
    supports_partial_denoise: bool
    supports_per_token_timesteps: bool

    def __post_init__(self) -> None:
        _require_identifier("model_family", self.model_family)
        _require_identifier("model_variant", self.model_variant)
        _require_enum_tuple(
            "accepted_prediction_types",
            self.accepted_prediction_types,
            PredictionType,
        )
        _require_enum_tuple(
            "accepted_sigma_domains",
            self.accepted_sigma_domains,
            SigmaDomain,
        )
        _require_enum_tuple(
            "accepted_ownerships",
            self.accepted_ownerships,
            ScheduleOwnership,
        )
        _require_bool("supports_partial_denoise", self.supports_partial_denoise)
        _require_bool(
            "supports_per_token_timesteps",
            self.supports_per_token_timesteps,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileCapabilities:
    """Resolved profile requirements and sampler compatibility policy."""

    profile_id: str
    profile_version: str
    model_family: str
    model_variant: str
    prediction_type: PredictionType
    sigma_domain: SigmaDomain
    ownership: ScheduleOwnership
    terminal_sigma: TerminalSigma
    allowed_execution_behaviors: tuple[ExecutionBehavior, ...]
    allowed_noise_ownerships: tuple[NoiseOwnership, ...]
    allowed_sampler_state: tuple[SamplerState, ...]
    supports_partial_denoise: bool
    supports_per_token_timesteps: bool
    reference_sampler_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier("profile_id", self.profile_id)
        _require_public_text("profile_version", self.profile_version)
        _require_identifier("model_family", self.model_family)
        _require_identifier("model_variant", self.model_variant)
        if not isinstance(self.prediction_type, PredictionType):
            raise ScheduleContractError("prediction_type is unsupported")
        if not isinstance(self.sigma_domain, SigmaDomain):
            raise ScheduleContractError("sigma_domain is unsupported")
        if not isinstance(self.ownership, ScheduleOwnership):
            raise ScheduleContractError("ownership is unsupported")
        if not isinstance(self.terminal_sigma, TerminalSigma):
            raise ScheduleContractError("terminal_sigma is unsupported")
        _require_enum_tuple(
            "allowed_execution_behaviors",
            self.allowed_execution_behaviors,
            ExecutionBehavior,
        )
        _require_enum_tuple(
            "allowed_noise_ownerships",
            self.allowed_noise_ownerships,
            NoiseOwnership,
        )
        _require_enum_tuple(
            "allowed_sampler_state",
            self.allowed_sampler_state,
            SamplerState,
            allow_empty=True,
        )
        _require_bool("supports_partial_denoise", self.supports_partial_denoise)
        _require_bool(
            "supports_per_token_timesteps",
            self.supports_per_token_timesteps,
        )
        _require_identifier_tuple(
            "reference_sampler_ids",
            self.reference_sampler_ids,
        )

        behaviors = set(self.allowed_execution_behaviors)
        noise_owners = set(self.allowed_noise_ownerships)
        if ExecutionBehavior.DETERMINISTIC in behaviors and NoiseOwnership.NONE not in noise_owners:
            raise ScheduleContractError(
                "deterministic profile behavior requires NONE noise ownership"
            )
        if ExecutionBehavior.DETERMINISTIC not in behaviors and NoiseOwnership.NONE in noise_owners:
            raise ScheduleContractError(
                "NONE noise ownership requires deterministic profile behavior"
            )
        stochastic_noise = noise_owners - {NoiseOwnership.NONE}
        if ExecutionBehavior.STOCHASTIC in behaviors and not stochastic_noise:
            raise ScheduleContractError(
                "stochastic profile behavior requires non-NONE noise ownership"
            )
        if ExecutionBehavior.STOCHASTIC not in behaviors and stochastic_noise:
            raise ScheduleContractError(
                "non-NONE noise ownership requires stochastic profile behavior"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplerCapabilities:
    """Declared numerical sampler behavior used for compatibility preflight."""

    sampler_id: str
    sampler_version: str
    accepted_prediction_types: tuple[PredictionType, ...]
    accepted_sigma_domains: tuple[SigmaDomain, ...]
    accepted_ownerships: tuple[ScheduleOwnership, ...]
    terminal_requirement: TerminalRequirement
    execution_behavior: ExecutionBehavior
    noise_ownership: NoiseOwnership
    required_state: tuple[SamplerState, ...]
    supports_partial_denoise: bool
    supports_per_token_timesteps: bool

    def __post_init__(self) -> None:
        _require_identifier("sampler_id", self.sampler_id)
        _require_public_text("sampler_version", self.sampler_version)
        _require_enum_tuple(
            "accepted_prediction_types",
            self.accepted_prediction_types,
            PredictionType,
        )
        _require_enum_tuple(
            "accepted_sigma_domains",
            self.accepted_sigma_domains,
            SigmaDomain,
        )
        _require_enum_tuple(
            "accepted_ownerships",
            self.accepted_ownerships,
            ScheduleOwnership,
        )
        if not isinstance(self.terminal_requirement, TerminalRequirement):
            raise ScheduleContractError("terminal_requirement is unsupported")
        if not isinstance(self.execution_behavior, ExecutionBehavior):
            raise ScheduleContractError("execution_behavior is unsupported")
        if not isinstance(self.noise_ownership, NoiseOwnership):
            raise ScheduleContractError("noise_ownership is unsupported")
        _require_enum_tuple(
            "required_state",
            self.required_state,
            SamplerState,
            allow_empty=True,
        )
        _require_bool("supports_partial_denoise", self.supports_partial_denoise)
        _require_bool(
            "supports_per_token_timesteps",
            self.supports_per_token_timesteps,
        )

        if (
            self.execution_behavior is ExecutionBehavior.DETERMINISTIC
            and self.noise_ownership is not NoiseOwnership.NONE
        ):
            raise ScheduleContractError("deterministic sampler must declare NONE noise ownership")
        if (
            self.execution_behavior is ExecutionBehavior.STOCHASTIC
            and self.noise_ownership is NoiseOwnership.NONE
        ):
            raise ScheduleContractError("stochastic sampler must declare non-NONE noise ownership")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionFeatureRequest:
    """Execution features that must be supported by all relevant components."""

    use_partial_denoise: bool = False
    use_per_token_timesteps: bool = False

    def __post_init__(self) -> None:
        _require_bool("use_partial_denoise", self.use_partial_denoise)
        _require_bool("use_per_token_timesteps", self.use_per_token_timesteps)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompatibilityDecision:
    """Immutable compatibility result with canonical considered dimensions and reasons."""

    level: CompatibilityLevel
    considered: tuple[CapabilityDimension, ...]
    reasons: tuple[CompatibilityReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.level, CompatibilityLevel):
            raise ScheduleContractError("compatibility level is unsupported")
        _require_enum_tuple(
            "considered",
            self.considered,
            CapabilityDimension,
        )
        if self.considered != tuple(CapabilityDimension):
            raise ScheduleContractError("considered dimensions must be complete")
        _require_enum_tuple("reasons", self.reasons, CompatibilityReason)

        if self.level is CompatibilityLevel.ALLOW:
            if self.reasons != (CompatibilityReason.COMPATIBLE,):
                raise ScheduleContractError("ALLOW requires only the compatible reason")
            return
        if self.level is CompatibilityLevel.WARN:
            if not set(self.reasons).issubset(_WARNING_REASONS):
                raise ScheduleContractError("WARN requires warning reason codes")
            return
        if not set(self.reasons).issubset(_REJECTION_REASONS):
            raise ScheduleContractError("REJECT requires rejection reason codes")


class CapabilityCompatibilityError(ScheduleContractError):
    """Raised when a rejected capability decision reaches the execution gate."""


def _terminal_is_compatible(
    terminal: TerminalSigma,
    requirement: TerminalRequirement,
) -> bool:
    if requirement is TerminalRequirement.ACCEPTS_EITHER:
        return True
    if requirement is TerminalRequirement.REQUIRES_ZERO:
        return terminal is TerminalSigma.ZERO
    return terminal is TerminalSigma.NONZERO


def evaluate_compatibility(
    *,
    model: ModelCapabilities,
    profile: ProfileCapabilities,
    sampler: SamplerCapabilities,
    request: ExecutionFeatureRequest,
) -> CompatibilityDecision:
    """Evaluate one resolved combination without executing any framework or sampler code."""

    for field_name, value, expected_type in (
        ("model", model, ModelCapabilities),
        ("profile", profile, ProfileCapabilities),
        ("sampler", sampler, SamplerCapabilities),
        ("request", request, ExecutionFeatureRequest),
    ):
        if not isinstance(value, expected_type):
            raise ScheduleContractError(f"{field_name} must be {expected_type.__name__}")

    reasons: list[CompatibilityReason] = []
    if model.model_family != profile.model_family:
        reasons.append(CompatibilityReason.MODEL_FAMILY_MISMATCH)
    if model.model_variant != profile.model_variant:
        reasons.append(CompatibilityReason.MODEL_VARIANT_MISMATCH)
    if profile.prediction_type not in model.accepted_prediction_types:
        reasons.append(CompatibilityReason.MODEL_PREDICTION_UNSUPPORTED)
    if profile.sigma_domain not in model.accepted_sigma_domains:
        reasons.append(CompatibilityReason.MODEL_SIGMA_DOMAIN_UNSUPPORTED)
    if profile.ownership not in model.accepted_ownerships:
        reasons.append(CompatibilityReason.MODEL_OWNERSHIP_UNSUPPORTED)
    if profile.prediction_type not in sampler.accepted_prediction_types:
        reasons.append(CompatibilityReason.SAMPLER_PREDICTION_UNSUPPORTED)
    if profile.sigma_domain not in sampler.accepted_sigma_domains:
        reasons.append(CompatibilityReason.SAMPLER_SIGMA_DOMAIN_UNSUPPORTED)
    if profile.ownership not in sampler.accepted_ownerships:
        reasons.append(CompatibilityReason.SAMPLER_OWNERSHIP_UNSUPPORTED)
    if not _terminal_is_compatible(
        profile.terminal_sigma,
        sampler.terminal_requirement,
    ):
        reasons.append(CompatibilityReason.TERMINAL_REQUIREMENT_MISMATCH)
    if sampler.execution_behavior not in profile.allowed_execution_behaviors:
        reasons.append(CompatibilityReason.EXECUTION_BEHAVIOR_MISMATCH)
    if sampler.noise_ownership not in profile.allowed_noise_ownerships:
        reasons.append(CompatibilityReason.NOISE_OWNERSHIP_MISMATCH)
    if not set(sampler.required_state).issubset(profile.allowed_sampler_state):
        reasons.append(CompatibilityReason.SAMPLER_STATE_UNSUPPORTED)

    if request.use_partial_denoise:
        if not model.supports_partial_denoise:
            reasons.append(CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_MODEL)
        if not profile.supports_partial_denoise:
            reasons.append(CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_PROFILE)
        if not sampler.supports_partial_denoise:
            reasons.append(CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_SAMPLER)

    if request.use_per_token_timesteps:
        if not model.supports_per_token_timesteps:
            reasons.append(CompatibilityReason.PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_MODEL)
        if not profile.supports_per_token_timesteps:
            reasons.append(CompatibilityReason.PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_PROFILE)
        if not sampler.supports_per_token_timesteps:
            reasons.append(CompatibilityReason.PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_SAMPLER)

    considered = tuple(CapabilityDimension)
    if reasons:
        return CompatibilityDecision(
            level=CompatibilityLevel.REJECT,
            considered=considered,
            reasons=tuple(reasons),
        )
    if profile.reference_sampler_ids and sampler.sampler_id not in profile.reference_sampler_ids:
        return CompatibilityDecision(
            level=CompatibilityLevel.WARN,
            considered=considered,
            reasons=(CompatibilityReason.SAMPLER_NOT_PROFILE_REFERENCE,),
        )
    return CompatibilityDecision(
        level=CompatibilityLevel.ALLOW,
        considered=considered,
        reasons=(CompatibilityReason.COMPATIBLE,),
    )


def require_compatible(decision: CompatibilityDecision) -> CompatibilityDecision:
    """Return an executable decision or raise before execution on rejection."""

    if not isinstance(decision, CompatibilityDecision):
        raise ScheduleContractError("decision must be a CompatibilityDecision")
    if decision.level is CompatibilityLevel.REJECT:
        codes = ",".join(reason.value for reason in decision.reasons)
        raise CapabilityCompatibilityError(f"capability compatibility rejected: {codes}")
    return decision
