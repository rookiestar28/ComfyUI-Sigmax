"""Official Krea 2 RAW structural profile declarations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    EvidenceLevel,
    ExecutionBehavior,
    ModelCapabilities,
    NoiseOwnership,
    PredictionType,
    ProfileCapabilities,
    SamplerCapabilities,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    TerminalRequirement,
    TerminalSigma,
)
from comfyui_sigmax.profiles.krea2_common import (
    KREA2_REFERENCES,
    DimensionAlignmentMode,
    DimensionPolicy,
    EvidenceReference,
    GuidanceConvention,
    Krea2ImageGeometry,
    ShiftParameterization,
    _require_finite_number,
    _require_identifier,
    _require_positive_integer,
    resolve_krea2_image_geometry,
)


class ResolutionShiftMode(str, Enum):
    """Supported resolution-to-shift derivation modes."""

    RESOLUTION_LINEAR = "resolution_linear"


class ExtrapolationPolicy(str, Enum):
    """Behavior outside the authoritative sequence-length endpoints."""

    UPSTREAM_UNCLAMPED = "upstream_unclamped"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolutionShiftPolicy:
    """Immutable declaration of Krea 2 RAW's dynamic exponential shift."""

    mode: ResolutionShiftMode
    base_image_seq_len: int
    max_image_seq_len: int
    base_mu: float
    max_mu: float
    extrapolation: ExtrapolationPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ResolutionShiftMode):
            raise ScheduleContractError("resolution shift mode is unsupported")
        base_length = _require_positive_integer(
            "base_image_seq_len",
            self.base_image_seq_len,
        )
        max_length = _require_positive_integer(
            "max_image_seq_len",
            self.max_image_seq_len,
        )
        base_mu = _require_finite_number("base_mu", self.base_mu)
        max_mu = _require_finite_number("max_mu", self.max_mu)
        if not isinstance(self.extrapolation, ExtrapolationPolicy):
            raise ScheduleContractError("resolution shift extrapolation policy is unsupported")
        if (
            self.mode is not ResolutionShiftMode.RESOLUTION_LINEAR
            or base_length != 256
            or max_length != 6400
            or base_mu != 0.5
            or max_mu != 1.15
            or self.extrapolation is not ExtrapolationPolicy.UPSTREAM_UNCLAMPED
        ):
            raise ScheduleContractError("Krea 2 RAW resolution shift invariants were modified")


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2RawRecipe:
    """One named authoritative or framework RAW inference recipe."""

    recipe_id: str
    evidence: EvidenceLevel
    steps: int
    guidance: GuidanceConvention
    evidence_source_id: str

    def __post_init__(self) -> None:
        recipe_id = _require_identifier("recipe_id", self.recipe_id)
        if not isinstance(self.evidence, EvidenceLevel):
            raise ScheduleContractError("recipe evidence must be an EvidenceLevel")
        steps = _require_positive_integer("recipe steps", self.steps)
        if not isinstance(self.guidance, GuidanceConvention):
            raise ScheduleContractError("recipe guidance must be a GuidanceConvention")
        source_id = _require_identifier("evidence_source_id", self.evidence_source_id)

        contracts = {
            "krea2.raw.diffusers-reference-28": (
                EvidenceLevel.FRAMEWORK_REFERENCE,
                28,
                GuidanceConvention(krea_guidance=4.5, comfy_cfg=5.5),
                "diffusers.krea2.framework",
            ),
            "krea2.raw.official-full-52": (
                EvidenceLevel.OFFICIAL,
                52,
                GuidanceConvention(krea_guidance=3.5, comfy_cfg=4.5),
                "krea.krea2.official",
            ),
        }
        expected = contracts.get(recipe_id)
        if expected is None or (self.evidence, steps, self.guidance, source_id) != expected:
            raise ScheduleContractError("Krea 2 RAW recipe invariants were modified")


KREA2_RAW_DIFFUSERS_REFERENCE_28: Final = Krea2RawRecipe(
    recipe_id="krea2.raw.diffusers-reference-28",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    steps=28,
    guidance=GuidanceConvention(krea_guidance=4.5, comfy_cfg=5.5),
    evidence_source_id="diffusers.krea2.framework",
)
KREA2_RAW_OFFICIAL_FULL_52: Final = Krea2RawRecipe(
    recipe_id="krea2.raw.official-full-52",
    evidence=EvidenceLevel.OFFICIAL,
    steps=52,
    guidance=GuidanceConvention(krea_guidance=3.5, comfy_cfg=4.5),
    evidence_source_id="krea.krea2.official",
)

_MODEL_CAPABILITIES: Final = ModelCapabilities(
    model_family="krea2",
    model_variant="raw",
    accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
    accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
    accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
    supports_partial_denoise=True,
    supports_per_token_timesteps=False,
)
_PROFILE_CAPABILITIES: Final = ProfileCapabilities(
    profile_id="krea2.raw.official",
    profile_version="1",
    model_family="krea2",
    model_variant="raw",
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
class Krea2RawProfile:
    """Immutable declaration of Krea 2 RAW inference structure."""

    profile_id: str = "krea2.raw.official"
    profile_version: str = "1"
    evidence: EvidenceLevel = EvidenceLevel.OFFICIAL
    model_family: str = "krea2"
    model_variant: str = "raw"
    prediction_type: PredictionType = PredictionType.FLOW_VELOCITY
    sigma_domain: SigmaDomain = SigmaDomain.UNIT_FLOW
    ownership: ScheduleOwnership = ScheduleOwnership.EXTERNAL_SIGMAS
    base_grid_identifier: str = "krea.reciprocal_step"
    shift_parameterization: ShiftParameterization = ShiftParameterization.EXPONENTIAL_MU
    shift_policy: ResolutionShiftPolicy = field(
        default_factory=lambda: ResolutionShiftPolicy(
            mode=ResolutionShiftMode.RESOLUTION_LINEAR,
            base_image_seq_len=256,
            max_image_seq_len=6400,
            base_mu=0.5,
            max_mu=1.15,
            extrapolation=ExtrapolationPolicy.UPSTREAM_UNCLAMPED,
        )
    )
    terminal_policy: TerminalPolicy = TerminalPolicy.APPEND_ZERO
    terminal_sigma: TerminalSigma = TerminalSigma.ZERO
    reference_sampler_id: str = "comfy.euler"
    dimensions: DimensionPolicy = field(
        default_factory=lambda: DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=16,
            evidence_source_id="krea.krea2.official",
        )
    )
    recipes: tuple[Krea2RawRecipe, ...] = (
        KREA2_RAW_DIFFUSERS_REFERENCE_28,
        KREA2_RAW_OFFICIAL_FULL_52,
    )
    references: tuple[EvidenceReference, ...] = KREA2_REFERENCES
    model_capabilities: ModelCapabilities = _MODEL_CAPABILITIES
    profile_capabilities: ProfileCapabilities = _PROFILE_CAPABILITIES
    reference_sampler_capabilities: SamplerCapabilities = _REFERENCE_SAMPLER_CAPABILITIES

    def __post_init__(self) -> None:
        expected_scalars = (
            self.profile_id == "krea2.raw.official",
            self.profile_version == "1",
            self.evidence is EvidenceLevel.OFFICIAL,
            self.model_family == "krea2",
            self.model_variant == "raw",
            self.prediction_type is PredictionType.FLOW_VELOCITY,
            self.sigma_domain is SigmaDomain.UNIT_FLOW,
            self.ownership is ScheduleOwnership.EXTERNAL_SIGMAS,
            self.base_grid_identifier == "krea.reciprocal_step",
            self.shift_parameterization is ShiftParameterization.EXPONENTIAL_MU,
            self.terminal_policy is TerminalPolicy.APPEND_ZERO,
            self.terminal_sigma is TerminalSigma.ZERO,
            self.reference_sampler_id == "comfy.euler",
        )
        if not all(expected_scalars):
            raise ScheduleContractError("Krea 2 RAW official profile invariants were modified")
        if self.shift_policy != ResolutionShiftPolicy(
            mode=ResolutionShiftMode.RESOLUTION_LINEAR,
            base_image_seq_len=256,
            max_image_seq_len=6400,
            base_mu=0.5,
            max_mu=1.15,
            extrapolation=ExtrapolationPolicy.UPSTREAM_UNCLAMPED,
        ):
            raise ScheduleContractError("Krea 2 RAW shift declaration is inconsistent")
        if self.dimensions != DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=16,
            evidence_source_id="krea.krea2.official",
        ):
            raise ScheduleContractError("Krea 2 RAW dimension declaration is inconsistent")
        if self.recipes != (
            KREA2_RAW_DIFFUSERS_REFERENCE_28,
            KREA2_RAW_OFFICIAL_FULL_52,
        ):
            raise ScheduleContractError("Krea 2 RAW recipes are incomplete or unordered")
        if self.references != KREA2_REFERENCES:
            raise ScheduleContractError("Krea 2 RAW evidence references are incomplete")
        if (
            self.model_capabilities != _MODEL_CAPABILITIES
            or self.profile_capabilities != _PROFILE_CAPABILITIES
            or self.reference_sampler_capabilities != _REFERENCE_SAMPLER_CAPABILITIES
        ):
            raise ScheduleContractError("Krea 2 RAW capability declarations are inconsistent")

    @property
    def primary_reference(self) -> EvidenceReference:
        """Return the authoritative source used for profile provenance."""

        return self.references[0]

    @property
    def framework_reference_recipe(self) -> Krea2RawRecipe:
        """Return the named 28-step Diffusers recipe."""

        return self.recipes[0]

    @property
    def official_full_recipe(self) -> Krea2RawRecipe:
        """Return the named 52-step official full recipe."""

        return self.recipes[1]


KREA2_RAW_PROFILE: Final = Krea2RawProfile()


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2RawShiftDerivation:
    """Traceable result of official RAW geometry and dynamic-mu derivation."""

    profile_id: str
    profile_version: str
    geometry: Krea2ImageGeometry
    mu: float
    extrapolated: bool

    def __post_init__(self) -> None:
        profile_id = _require_identifier("profile_id", self.profile_id)
        profile_version = _require_identifier("profile_version", self.profile_version)
        if (profile_id, profile_version) != (
            KREA2_RAW_PROFILE.profile_id,
            KREA2_RAW_PROFILE.profile_version,
        ):
            raise ScheduleContractError("RAW shift derivation must identify the official profile")
        if not isinstance(self.geometry, Krea2ImageGeometry):
            raise ScheduleContractError("RAW shift derivation requires Krea2ImageGeometry")
        mu = _require_finite_number("mu", self.mu)
        if type(self.extrapolated) is not bool:
            raise ScheduleContractError("extrapolated must be a Boolean")

        expected_mu = calculate_krea2_raw_mu(self.geometry.image_seq_len)
        policy = KREA2_RAW_PROFILE.shift_policy
        expected_extrapolated = not (
            policy.base_image_seq_len <= self.geometry.image_seq_len <= policy.max_image_seq_len
        )
        if mu != expected_mu:
            raise ScheduleContractError("RAW shift derivation mu is inconsistent with geometry")
        if self.extrapolated is not expected_extrapolated:
            raise ScheduleContractError("RAW shift derivation extrapolation status is inconsistent")


def calculate_krea2_raw_mu(
    image_seq_len: int,
    *,
    policy: ResolutionShiftPolicy = KREA2_RAW_PROFILE.shift_policy,
) -> float:
    """Calculate official Krea 2 RAW mu without clamping or hidden fallback."""

    if not isinstance(policy, ResolutionShiftPolicy):
        raise ScheduleContractError("RAW mu derivation requires a ResolutionShiftPolicy")
    sequence_length = _require_positive_integer("image_seq_len", image_seq_len)
    slope = (policy.max_mu - policy.base_mu) / (
        policy.max_image_seq_len - policy.base_image_seq_len
    )
    return slope * sequence_length + (policy.base_mu - slope * policy.base_image_seq_len)


def derive_krea2_raw_shift(
    width: int,
    height: int,
    *,
    profile: Krea2RawProfile = KREA2_RAW_PROFILE,
) -> Krea2RawShiftDerivation:
    """Resolve effective image geometry and official unclamped RAW mu."""

    if not isinstance(profile, Krea2RawProfile):
        raise ScheduleContractError("RAW shift derivation requires a Krea2RawProfile")
    geometry = resolve_krea2_image_geometry(
        width,
        height,
        policy=profile.dimensions,
    )
    mu = calculate_krea2_raw_mu(
        geometry.image_seq_len,
        policy=profile.shift_policy,
    )
    extrapolated = not (
        profile.shift_policy.base_image_seq_len
        <= geometry.image_seq_len
        <= profile.shift_policy.max_image_seq_len
    )
    return Krea2RawShiftDerivation(
        profile_id=profile.profile_id,
        profile_version=profile.profile_version,
        geometry=geometry,
        mu=mu,
        extrapolated=extrapolated,
    )
