"""Official Krea 2 Turbo profile and dependency-free schedule construction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    BaseGridSpec,
    EvidenceLevel,
    ExecutionBehavior,
    ModelCapabilities,
    NoiseOwnership,
    OverrideRecord,
    PredictionType,
    ProfileCapabilities,
    Provenance,
    SamplerCapabilities,
    ScheduleContractError,
    ScheduleInputs,
    ScheduleOwnership,
    ScheduleRequest,
    ScheduleResult,
    SigmaDomain,
    SliceSpec,
    TerminalPolicy,
    TerminalRequirement,
    TerminalSigma,
    TransformContract,
    TransformStage,
    apply_terminal_policy,
    exponential_mu_shift,
    krea_reciprocal_step_grid,
    validate_sigma_schedule,
)

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_ENGINE_VERSION: Final = "0.1.0.dev0"
_ALIGNMENT_REASON: Final = "Krea 2 pads image dimensions up to a multiple of 16"
_MODIFIED_STEPS_WARNING: Final = (
    "requested steps differ from the official Turbo 8-step recipe; evidence is modified"
)


class ShiftParameterization(str, Enum):
    """Named time-shift formulas exposed by evidence-pinned profiles."""

    EXPONENTIAL_MU = "exponential_mu"


class DimensionAlignmentMode(str, Enum):
    """Dimension-normalization modes declared by model profiles."""

    CEIL_MULTIPLE = "ceil_multiple"


def _require_finite_number(field_name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ScheduleContractError(f"{field_name} must be a finite number")
    return float(value)


def _require_positive_integer(field_name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleContractError(f"{field_name} must be a positive integer")
    return value


def _require_identifier(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ScheduleContractError(f"{field_name} must be a stable lowercase identifier")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class GuidanceConvention:
    """Krea guidance and equivalent standard ComfyUI CFG values."""

    krea_guidance: float
    comfy_cfg: float

    def __post_init__(self) -> None:
        krea_guidance = _require_finite_number("krea_guidance", self.krea_guidance)
        comfy_cfg = _require_finite_number("comfy_cfg", self.comfy_cfg)
        if krea_guidance != 0.0 or comfy_cfg != 1.0:
            raise ScheduleContractError(
                "the official Krea 2 Turbo guidance convention is Krea 0.0 / ComfyUI 1.0"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class DimensionPolicy:
    """Evidence-bearing image-dimension alignment policy."""

    mode: DimensionAlignmentMode
    multiple: int
    evidence_source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DimensionAlignmentMode):
            raise ScheduleContractError("dimension alignment mode is unsupported")
        multiple = _require_positive_integer("dimension multiple", self.multiple)
        source_id = _require_identifier("evidence_source_id", self.evidence_source_id)
        if (
            self.mode is not DimensionAlignmentMode.CEIL_MULTIPLE
            or multiple != 16
            or source_id != "krea.krea2.official"
        ):
            raise ScheduleContractError(
                "the official Krea 2 Turbo dimension policy is ceil-to-multiple-of-16"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceReference:
    """Pinned source revision and deterministic locators for one profile claim set."""

    source_id: str
    evidence: EvidenceLevel
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier("source_id", self.source_id)
        if not isinstance(self.evidence, EvidenceLevel):
            raise ScheduleContractError("evidence must be an EvidenceLevel")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("evidence URL must use HTTPS")
        if not isinstance(self.revision, str) or not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("evidence revision must be a pinned 40-character commit")
        if not isinstance(self.locators, tuple) or not self.locators:
            raise ScheduleContractError("evidence locators must be a non-empty tuple")
        if any(not isinstance(locator, str) or not locator.strip() for locator in self.locators):
            raise ScheduleContractError("evidence locators must contain non-empty strings")
        if len(self.locators) != len(set(self.locators)):
            raise ScheduleContractError("evidence locators must not contain duplicates")
        if self.locators != tuple(sorted(self.locators)):
            raise ScheduleContractError("evidence locators must use canonical order")


_KREA_REFERENCE: Final = EvidenceReference(
    source_id="krea.krea2.official",
    evidence=EvidenceLevel.OFFICIAL,
    url="https://github.com/krea-ai/krea-2",
    revision="db3984fbc6e13b34c0064990fc2d95ac64d00058",  # pragma: allowlist secret
    locators=("README.md", "inference.py", "sampling.py"),
)
_DIFFUSERS_REFERENCE: Final = EvidenceReference(
    source_id="diffusers.krea2.framework",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    url="https://github.com/huggingface/diffusers",
    revision="3c468926ffd12b69baa4316e27b09306b8da19a6",  # pragma: allowlist secret
    locators=("src/diffusers/pipelines/krea2/pipeline_krea2.py",),
)
_COMFYUI_REFERENCE: Final = EvidenceReference(
    source_id="comfyui.krea2.framework",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    url="https://github.com/Comfy-Org/ComfyUI",
    revision="e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
    locators=(
        "comfy/k_diffusion/sampling.py",
        "comfy/model_sampling.py",
        "comfy/supported_models.py",
    ),
)

_MODEL_CAPABILITIES: Final = ModelCapabilities(
    model_family="krea2",
    model_variant="turbo",
    accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
    accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
    accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
    supports_partial_denoise=True,
    supports_per_token_timesteps=False,
)
_PROFILE_CAPABILITIES: Final = ProfileCapabilities(
    profile_id="krea2.turbo.official",
    profile_version="1",
    model_family="krea2",
    model_variant="turbo",
    prediction_type=PredictionType.FLOW_VELOCITY,
    sigma_domain=SigmaDomain.UNIT_FLOW,
    ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
    terminal_sigma=TerminalSigma.ZERO,
    allowed_execution_behaviors=(ExecutionBehavior.DETERMINISTIC,),
    allowed_noise_ownerships=(NoiseOwnership.NONE,),
    allowed_sampler_state=(),
    supports_partial_denoise=True,
    supports_per_token_timesteps=False,
    reference_sampler_ids=("comfy.euler",),
)
_REFERENCE_SAMPLER_CAPABILITIES: Final = SamplerCapabilities(
    sampler_id="comfy.euler",
    sampler_version="e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
    accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
    accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
    accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
    terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
    execution_behavior=ExecutionBehavior.DETERMINISTIC,
    noise_ownership=NoiseOwnership.NONE,
    required_state=(),
    supports_partial_denoise=True,
    supports_per_token_timesteps=False,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2TurboProfile:
    """Immutable declaration of the official Krea 2 Turbo inference recipe."""

    profile_id: str = "krea2.turbo.official"
    profile_version: str = "1"
    evidence: EvidenceLevel = EvidenceLevel.OFFICIAL
    model_family: str = "krea2"
    model_variant: str = "turbo"
    prediction_type: PredictionType = PredictionType.FLOW_VELOCITY
    sigma_domain: SigmaDomain = SigmaDomain.UNIT_FLOW
    ownership: ScheduleOwnership = ScheduleOwnership.EXTERNAL_SIGMAS
    base_grid_identifier: str = "krea.reciprocal_step"
    shift_parameterization: ShiftParameterization = ShiftParameterization.EXPONENTIAL_MU
    fixed_mu: float = 1.15
    terminal_policy: TerminalPolicy = TerminalPolicy.APPEND_ZERO
    terminal_sigma: TerminalSigma = TerminalSigma.ZERO
    default_steps: int = 8
    reference_sampler_id: str = "comfy.euler"
    guidance: GuidanceConvention = field(
        default_factory=lambda: GuidanceConvention(krea_guidance=0.0, comfy_cfg=1.0)
    )
    dimensions: DimensionPolicy = field(
        default_factory=lambda: DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=16,
            evidence_source_id="krea.krea2.official",
        )
    )
    references: tuple[EvidenceReference, ...] = (
        _KREA_REFERENCE,
        _DIFFUSERS_REFERENCE,
        _COMFYUI_REFERENCE,
    )
    model_capabilities: ModelCapabilities = _MODEL_CAPABILITIES
    profile_capabilities: ProfileCapabilities = _PROFILE_CAPABILITIES
    reference_sampler_capabilities: SamplerCapabilities = _REFERENCE_SAMPLER_CAPABILITIES

    def __post_init__(self) -> None:
        expected_scalars = (
            self.profile_id == "krea2.turbo.official",
            self.profile_version == "1",
            self.evidence is EvidenceLevel.OFFICIAL,
            self.model_family == "krea2",
            self.model_variant == "turbo",
            self.prediction_type is PredictionType.FLOW_VELOCITY,
            self.sigma_domain is SigmaDomain.UNIT_FLOW,
            self.ownership is ScheduleOwnership.EXTERNAL_SIGMAS,
            self.base_grid_identifier == "krea.reciprocal_step",
            self.shift_parameterization is ShiftParameterization.EXPONENTIAL_MU,
            self.fixed_mu == 1.15,
            self.terminal_policy is TerminalPolicy.APPEND_ZERO,
            self.terminal_sigma is TerminalSigma.ZERO,
            self.default_steps == 8,
            self.reference_sampler_id == "comfy.euler",
        )
        if not all(expected_scalars):
            raise ScheduleContractError("Krea 2 Turbo official profile invariants were modified")
        if self.guidance != GuidanceConvention(krea_guidance=0.0, comfy_cfg=1.0):
            raise ScheduleContractError("Krea 2 Turbo guidance declaration is inconsistent")
        if self.dimensions != DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=16,
            evidence_source_id="krea.krea2.official",
        ):
            raise ScheduleContractError("Krea 2 Turbo dimension declaration is inconsistent")
        if self.references != (
            _KREA_REFERENCE,
            _DIFFUSERS_REFERENCE,
            _COMFYUI_REFERENCE,
        ):
            raise ScheduleContractError("Krea 2 Turbo evidence references are incomplete")
        if (
            self.model_capabilities != _MODEL_CAPABILITIES
            or self.profile_capabilities != _PROFILE_CAPABILITIES
            or self.reference_sampler_capabilities != _REFERENCE_SAMPLER_CAPABILITIES
        ):
            raise ScheduleContractError("Krea 2 Turbo capability declarations are inconsistent")

    @property
    def primary_reference(self) -> EvidenceReference:
        """Return the authoritative source used for schedule provenance."""

        return self.references[0]


KREA2_TURBO_PROFILE: Final = Krea2TurboProfile()


def _align_dimension(value: int, *, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _dimension_overrides(
    requested: ScheduleInputs,
    effective: ScheduleInputs,
) -> tuple[OverrideRecord, ...]:
    records: list[OverrideRecord] = []
    for field_name in ("width", "height"):
        requested_value = getattr(requested, field_name)
        effective_value = getattr(effective, field_name)
        if requested_value != effective_value:
            records.append(
                OverrideRecord(
                    field=field_name,
                    requested_value=str(requested_value),
                    effective_value=str(effective_value),
                    reason=_ALIGNMENT_REASON,
                )
            )
    return tuple(records)


def build_krea2_turbo_schedule(
    *,
    steps: int = 8,
    width: int = 1024,
    height: int = 1024,
    profile: Krea2TurboProfile = KREA2_TURBO_PROFILE,
) -> ScheduleResult:
    """Build the complete externally owned Krea 2 Turbo sigma schedule."""

    if not isinstance(profile, Krea2TurboProfile):
        raise ScheduleContractError("profile must be a Krea2TurboProfile")

    requested_inputs = ScheduleInputs(steps=steps, width=width, height=height)
    aligned_width = _align_dimension(width, multiple=profile.dimensions.multiple)
    aligned_height = _align_dimension(height, multiple=profile.dimensions.multiple)
    effective_inputs = ScheduleInputs(
        steps=steps,
        width=aligned_width,
        height=aligned_height,
    )
    overrides = _dimension_overrides(requested_inputs, effective_inputs)
    evidence = profile.evidence if steps == profile.default_steps else EvidenceLevel.MODIFIED
    warnings = () if evidence is profile.evidence else (_MODIFIED_STEPS_WARNING,)
    provenance = Provenance(
        engine_version=_ENGINE_VERSION,
        evidence=evidence,
        source=profile.primary_reference.url,
        source_revision=profile.primary_reference.revision,
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
    )
    transforms = (
        TransformContract(
            name="krea.exponential_mu",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=profile.sigma_domain,
            output_domain=profile.sigma_domain,
        ),
        TransformContract(
            name="terminal.append_zero",
            stage=TransformStage.TERMINAL,
            input_domain=profile.sigma_domain,
            output_domain=profile.sigma_domain,
        ),
    )
    request = ScheduleRequest(
        ownership=profile.ownership,
        requested_inputs=requested_inputs,
        sigma_domain=profile.sigma_domain,
        provenance=provenance,
        base_grid=BaseGridSpec(
            identifier=profile.base_grid_identifier,
            output_domain=profile.sigma_domain,
        ),
        transforms=transforms,
        terminal_policy=profile.terminal_policy,
        slicing=SliceSpec(),
        overrides=overrides,
    )
    sigmas = apply_terminal_policy(
        exponential_mu_shift(
            krea_reciprocal_step_grid(steps, domain=profile.sigma_domain),
            mu=profile.fixed_mu,
            domain=profile.sigma_domain,
        ),
        policy=profile.terminal_policy,
        domain=profile.sigma_domain,
    )
    validated_sigmas = validate_sigma_schedule(
        sigmas,
        domain=profile.sigma_domain,
        expected_steps=steps,
        require_terminal_zero=True,
    )
    return ScheduleResult(
        request=request,
        effective_inputs=effective_inputs,
        sigmas=validated_sigmas,
        final_domain=profile.sigma_domain,
        warnings=warnings,
    )
