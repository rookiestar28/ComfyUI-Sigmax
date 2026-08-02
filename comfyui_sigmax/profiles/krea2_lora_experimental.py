"""Experimental Krea 2 RAW-to-Turbo LoRA schedule profile."""

from __future__ import annotations

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
    validate_sigma_schedule,
)
from comfyui_sigmax.profiles.krea2_common import (
    DIFFUSERS_FRAMEWORK_PROVENANCE,
    GPL_3_ONLY_LICENSE,
    KREA2_ARTIFACT_VERSIONS,
    KREA2_SLICING,
    KREA_REFERENCE,
    KREA_SOFTWARE_PROVENANCE,
    DimensionAlignmentMode,
    DimensionPolicy,
    GuidanceConvention,
    Krea2ImageGeometry,
    ShiftParameterization,
    canonical_krea2_shifted_grid,
    resolve_krea2_image_geometry,
)
from comfyui_sigmax.profiles.krea2_raw import (
    KREA2_RAW_PROFILE,
    calculate_krea2_raw_mu,
)
from comfyui_sigmax.profiles.schema_v1 import (
    PROFILE_SCHEMA_ID,
    PROFILE_SCHEMA_VERSION,
    BaseGridDeclaration,
    DetectionDeclaration,
    FrameworkProvenance,
    GuidanceDeclaration,
    InferenceRecipe,
    LicenseDeclaration,
    ModelWeightProvenance,
    ProfileField,
    ProfileSchemaV1,
    StepRangeDeclaration,
    TerminalDeclaration,
    TransformDeclaration,
)
from comfyui_sigmax.version import VERSION

_ENGINE_VERSION: Final = VERSION
_PROFILE_ID: Final = "krea2.raw-turbo-lora.experimental"
_PROFILE_VERSION: Final = "1"
_RECIPE_ID: Final = "krea2.raw-turbo-lora.experimental"
_MAX_STEPS: Final = 10_000
_TURBO_MU: Final = 1.15
_ALIGNMENT_REASON: Final = "Krea 2 pads image dimensions up to a multiple of 16"
_EXPERIMENTAL_WARNINGS: Final = (
    "Experimental RAW-to-Turbo LoRA schedule; no official inference recipe exists.",
    (
        "Apply the RAW-to-Turbo LoRA to the Krea 2 RAW checkpoint only; do not stack it "
        "or apply it to Turbo."
    ),
    (
        "12 steps, CFG 1, Euler, and LoRA strength are community observations and are not "
        "enforced by this scheduler."
    ),
)


class Krea2ExperimentalMuSource(str, Enum):
    """Explicit shift source for the experimental RAW-plus-delta model state."""

    RAW = "raw"
    TURBO = "turbo"


_MODEL_CAPABILITIES: Final = ModelCapabilities(
    model_family="krea2",
    model_variant="raw_turbo_lora",
    accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
    accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
    accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
    supports_partial_denoise=True,
    supports_per_token_timesteps=False,
)
_PROFILE_CAPABILITIES: Final = ProfileCapabilities(
    profile_id=_PROFILE_ID,
    profile_version=_PROFILE_VERSION,
    model_family="krea2",
    model_variant="raw_turbo_lora",
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
    sampler_version="093d571b83e7a79833200e199b46b9f5a62217f9",  # pragma: allowlist secret
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
_CURRENT_COMFYUI_PROVENANCE: Final = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.krea2.framework",
    resource_version=None,
    revision="093d571b83e7a79833200e199b46b9f5a62217f9",  # pragma: allowlist secret
    url="https://github.com/Comfy-Org/ComfyUI",
    license=GPL_3_ONLY_LICENSE,
    locators=("comfy/lora.py", "comfy/model_sampling.py", "comfy/samplers.py"),
)
_ADAPTER_WEIGHT_PROVENANCE: Final = ModelWeightProvenance(
    record_version="1",
    weight_id="comfy_org.krea2.raw_to_turbo_lora.rank64",
    resource_version="1.0",
    revision="952f49d49653cb42e7d6cf7cbfad74738073ec7d",  # pragma: allowlist secret
    sha256=(
        "db8c5bae0a415d448da9d842111d6e51f7d32e47143a3118eb267e5c4773de87"  # pragma: allowlist secret
    ),
    url="https://huggingface.co/Comfy-Org/Krea-2",
    license=LicenseDeclaration(
        declaration_version="1",
        identifier="LicenseRef-Krea-2-Community",
        name="Krea 2 Community License",
        url="https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf",
    ),
)
_EXPLICIT_ONLY_DETECTION: Final = DetectionDeclaration(
    strategy_id="krea2.lora.experimental.explicit-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_selection",),
    suggestion_sources=(),
    family_only_sources=(),
)

KREA2_LORA_EXPERIMENTAL_SCHEMA: Final = ProfileSchemaV1(
    schema_id=PROFILE_SCHEMA_ID,
    schema_version=PROFILE_SCHEMA_VERSION,
    profile_id=_PROFILE_ID,
    profile_version=_PROFILE_VERSION,
    display_name="Krea 2 RAW-to-Turbo LoRA Experimental",
    model_family="krea2",
    model_variant="raw_turbo_lora",
    evidence=EvidenceLevel.EXPERIMENTAL,
    primary_source_id=KREA_REFERENCE.source_id,
    prediction_type=PredictionType.FLOW_VELOCITY,
    sigma_domain=SigmaDomain.UNIT_FLOW,
    ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
    base_grid=BaseGridDeclaration(
        identifier="krea.reciprocal_step",
        output_domain=SigmaDomain.UNIT_FLOW,
        terminal_included=False,
    ),
    transforms=(
        TransformDeclaration(
            identifier="krea.exponential_mu",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
            parameters=(
                ProfileField(name="base_image_seq_len", value=256),
                ProfileField(name="base_mu", value=0.5),
                ProfileField(name="max_image_seq_len", value=6400),
                ProfileField(name="max_mu", value=1.15),
                ProfileField(name="mu_sources", value="raw_resolution_or_turbo_fixed"),
                ProfileField(name="raw_extrapolation", value="upstream_unclamped"),
                ProfileField(name="turbo_mu", value=1.15),
            ),
        ),
        TransformDeclaration(
            identifier="terminal.append_zero",
            stage=TransformStage.TERMINAL,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
    ),
    terminal=TerminalDeclaration(
        policy=TerminalPolicy.APPEND_ZERO,
        sigma=TerminalSigma.ZERO,
        value=0.0,
    ),
    slicing=KREA2_SLICING,
    recipes=(
        InferenceRecipe(
            recipe_id=_RECIPE_ID,
            evidence=EvidenceLevel.EXPERIMENTAL,
            source_id=KREA_REFERENCE.source_id,
            steps=StepRangeDeclaration(
                minimum=1,
                maximum=_MAX_STEPS,
                default=12,
                reference_steps=(12,),
                allow_modified=True,
            ),
            guidance=GuidanceDeclaration(
                model_convention="krea.guidance",
                host_convention="comfy.cfg",
                model_value=0.0,
                host_value=1.0,
            ),
        ),
    ),
    detection=_EXPLICIT_ONLY_DETECTION,
    model_capabilities=_MODEL_CAPABILITIES,
    profile_capabilities=_PROFILE_CAPABILITIES,
    reference_sampler_capabilities=_REFERENCE_SAMPLER_CAPABILITIES,
    artifact_versions=KREA2_ARTIFACT_VERSIONS,
    software_sources=(KREA_SOFTWARE_PROVENANCE,),
    frameworks=(_CURRENT_COMFYUI_PROVENANCE, DIFFUSERS_FRAMEWORK_PROVENANCE),
    model_weights=(_ADAPTER_WEIGHT_PROVENANCE, KREA2_RAW_PROFILE.schema.model_weights[0]),
    parameters=(
        ProfileField(name="dimension_alignment_mode", value="ceil_multiple"),
        ProfileField(name="dimension_multiple", value=16),
        ProfileField(
            name="official_technical_report_recipe_finding",
            value="no_raw_to_turbo_lora_recipe_published",
        ),
        ProfileField(
            name="official_technical_report_url",
            value="https://www.krea.ai/blog/krea-2-technical-report",
        ),
        ProfileField(name="requires_external_lora", value=True),
    ),
    known_limitations=(
        "Automatic checkpoint or LoRA detection is intentionally unsupported.",
        "The scheduler does not load, scale, merge, or validate the external LoRA.",
        "The step, guidance, sampler, strength, and mu combinations are experimental.",
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2LoraExperimentalProfile:
    """Immutable declaration of the experimental RAW-plus-delta schedule boundary."""

    profile_id: str = _PROFILE_ID
    profile_version: str = _PROFILE_VERSION
    evidence: EvidenceLevel = EvidenceLevel.EXPERIMENTAL
    model_family: str = "krea2"
    model_variant: str = "raw_turbo_lora"
    prediction_type: PredictionType = PredictionType.FLOW_VELOCITY
    sigma_domain: SigmaDomain = SigmaDomain.UNIT_FLOW
    ownership: ScheduleOwnership = ScheduleOwnership.EXTERNAL_SIGMAS
    base_grid_identifier: str = "krea.reciprocal_step"
    shift_parameterization: ShiftParameterization = ShiftParameterization.EXPONENTIAL_MU
    terminal_policy: TerminalPolicy = TerminalPolicy.APPEND_ZERO
    terminal_sigma: TerminalSigma = TerminalSigma.ZERO
    default_steps: int = 12
    reference_sampler_id: str = "comfy.euler"
    guidance: GuidanceConvention = field(
        default_factory=lambda: GuidanceConvention(krea_guidance=0.0, comfy_cfg=1.0)
    )
    dimensions: DimensionPolicy = field(
        default_factory=lambda: DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=16,
            evidence_source_id=KREA_REFERENCE.source_id,
        )
    )
    schema: ProfileSchemaV1 = KREA2_LORA_EXPERIMENTAL_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.profile_id != _PROFILE_ID
            or self.profile_version != _PROFILE_VERSION
            or self.evidence is not EvidenceLevel.EXPERIMENTAL
            or self.model_family != "krea2"
            or self.model_variant != "raw_turbo_lora"
            or self.prediction_type is not PredictionType.FLOW_VELOCITY
            or self.sigma_domain is not SigmaDomain.UNIT_FLOW
            or self.ownership is not ScheduleOwnership.EXTERNAL_SIGMAS
            or self.base_grid_identifier != "krea.reciprocal_step"
            or self.shift_parameterization is not ShiftParameterization.EXPONENTIAL_MU
            or self.terminal_policy is not TerminalPolicy.APPEND_ZERO
            or self.terminal_sigma is not TerminalSigma.ZERO
            or self.default_steps != 12
            or self.reference_sampler_id != "comfy.euler"
            or self.guidance != GuidanceConvention(krea_guidance=0.0, comfy_cfg=1.0)
            or self.schema is not KREA2_LORA_EXPERIMENTAL_SCHEMA
        ):
            raise ScheduleContractError("experimental Krea 2 LoRA profile invariants changed")


KREA2_LORA_EXPERIMENTAL_PROFILE: Final = Krea2LoraExperimentalProfile()


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2ExperimentalShiftDerivation:
    """Resolved geometry and selected shift for one experimental request."""

    mu_source: Krea2ExperimentalMuSource
    geometry: Krea2ImageGeometry
    mu: float
    extrapolated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.mu_source, Krea2ExperimentalMuSource):
            raise ScheduleContractError("experimental mu source is unsupported")
        if not isinstance(self.geometry, Krea2ImageGeometry):
            raise ScheduleContractError("experimental shift requires Krea 2 geometry")
        expected_mu = (
            calculate_krea2_raw_mu(self.geometry.image_seq_len)
            if self.mu_source is Krea2ExperimentalMuSource.RAW
            else _TURBO_MU
        )
        policy = KREA2_RAW_PROFILE.shift_policy
        expected_extrapolated = self.mu_source is Krea2ExperimentalMuSource.RAW and not (
            policy.base_image_seq_len <= self.geometry.image_seq_len <= policy.max_image_seq_len
        )
        if self.mu != expected_mu or self.extrapolated is not expected_extrapolated:
            raise ScheduleContractError("experimental shift derivation is inconsistent")


def derive_krea2_lora_experimental_shift(
    *,
    width: int,
    height: int,
    mu_source: Krea2ExperimentalMuSource,
) -> Krea2ExperimentalShiftDerivation:
    """Resolve explicit RAW- or Turbo-mu behavior without automatic detection."""

    if not isinstance(mu_source, Krea2ExperimentalMuSource):
        raise ScheduleContractError("mu_source must be an experimental Krea 2 mu source")
    geometry = resolve_krea2_image_geometry(
        width,
        height,
        policy=KREA2_LORA_EXPERIMENTAL_PROFILE.dimensions,
    )
    mu = (
        calculate_krea2_raw_mu(geometry.image_seq_len)
        if mu_source is Krea2ExperimentalMuSource.RAW
        else _TURBO_MU
    )
    raw_policy = KREA2_RAW_PROFILE.shift_policy
    extrapolated = mu_source is Krea2ExperimentalMuSource.RAW and not (
        raw_policy.base_image_seq_len <= geometry.image_seq_len <= raw_policy.max_image_seq_len
    )
    return Krea2ExperimentalShiftDerivation(
        mu_source=mu_source,
        geometry=geometry,
        mu=mu,
        extrapolated=extrapolated,
    )


def build_krea2_lora_experimental_schedule(
    *,
    steps: int,
    width: int = 1024,
    height: int = 1024,
    mu_source: Krea2ExperimentalMuSource,
    profile: Krea2LoraExperimentalProfile = KREA2_LORA_EXPERIMENTAL_PROFILE,
) -> ScheduleResult:
    """Build an explicitly experimental external sigma schedule."""

    if not isinstance(profile, Krea2LoraExperimentalProfile):
        raise ScheduleContractError("profile must be a Krea2LoraExperimentalProfile")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError("steps must be an integer between 1 and 10000")
    derivation = derive_krea2_lora_experimental_shift(
        width=width,
        height=height,
        mu_source=mu_source,
    )
    requested_inputs = ScheduleInputs(steps=steps, width=width, height=height)
    effective_inputs = ScheduleInputs(
        steps=steps,
        width=derivation.geometry.effective_width,
        height=derivation.geometry.effective_height,
    )
    overrides = tuple(
        OverrideRecord(
            field=field_name,
            requested_value=str(getattr(requested_inputs, field_name)),
            effective_value=str(getattr(effective_inputs, field_name)),
            reason=_ALIGNMENT_REASON,
        )
        for field_name in ("width", "height")
        if getattr(requested_inputs, field_name) != getattr(effective_inputs, field_name)
    )
    request = ScheduleRequest(
        ownership=profile.ownership,
        requested_inputs=requested_inputs,
        sigma_domain=profile.sigma_domain,
        provenance=Provenance(
            engine_version=_ENGINE_VERSION,
            evidence=EvidenceLevel.EXPERIMENTAL,
            source=KREA_REFERENCE.url,
            source_revision=KREA_REFERENCE.revision,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
        ),
        base_grid=BaseGridSpec(
            identifier=profile.base_grid_identifier,
            output_domain=profile.sigma_domain,
        ),
        transforms=(
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
        ),
        terminal_policy=profile.terminal_policy,
        slicing=SliceSpec(),
        overrides=overrides,
    )
    sigmas = apply_terminal_policy(
        canonical_krea2_shifted_grid(steps=steps, mu=derivation.mu),
        policy=profile.terminal_policy,
        domain=profile.sigma_domain,
    )
    validated = validate_sigma_schedule(
        sigmas,
        domain=profile.sigma_domain,
        expected_steps=steps,
        require_terminal_zero=True,
    )
    return ScheduleResult(
        request=request,
        effective_inputs=effective_inputs,
        sigmas=validated,
        final_domain=profile.sigma_domain,
        warnings=_EXPERIMENTAL_WARNINGS,
    )
