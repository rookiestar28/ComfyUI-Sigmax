"""Official Krea 2 Turbo profile and dependency-free schedule construction."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from comfyui_sigmax.profiles.krea2_common import (
    COMFYUI_REFERENCE,
    DIFFUSERS_REFERENCE,
    KREA2_ARTIFACT_VERSIONS,
    KREA2_DETECTION,
    KREA2_FRAMEWORK_PROVENANCE,
    KREA2_SLICING,
    KREA_REFERENCE,
    KREA_SOFTWARE_PROVENANCE,
    DimensionAlignmentMode,
    DimensionPolicy,
    EvidenceReference,
    GuidanceConvention,
    ShiftParameterization,
)
from comfyui_sigmax.profiles.schema_v1 import (
    PROFILE_SCHEMA_ID,
    PROFILE_SCHEMA_VERSION,
    BaseGridDeclaration,
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
_ALIGNMENT_REASON: Final = "Krea 2 pads image dimensions up to a multiple of 16"
_MODIFIED_STEPS_WARNING: Final = (
    "requested steps differ from the official Turbo 8-step recipe; evidence is modified"
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

_TURBO_WEIGHT_PROVENANCE: Final = ModelWeightProvenance(
    record_version="1",
    weight_id="krea.krea2.turbo.weights",
    resource_version="1.0",
    revision="98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7",  # pragma: allowlist secret
    sha256=(
        "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3"  # pragma: allowlist secret
    ),
    url="https://huggingface.co/krea/Krea-2-Turbo",
    license=LicenseDeclaration(
        declaration_version="1",
        identifier="LicenseRef-Krea-2-Community",
        name="Krea 2 Community License",
        url="https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf",
    ),
)
KREA2_TURBO_SCHEMA: Final = ProfileSchemaV1(
    schema_id=PROFILE_SCHEMA_ID,
    schema_version=PROFILE_SCHEMA_VERSION,
    profile_id="krea2.turbo.official",
    profile_version="1",
    display_name="Krea 2 Turbo Official",
    model_family="krea2",
    model_variant="turbo",
    evidence=EvidenceLevel.OFFICIAL,
    primary_source_id="krea.krea2.official",
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
            parameters=(ProfileField(name="mu", value=1.15),),
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
            recipe_id="krea2.turbo.official-8",
            evidence=EvidenceLevel.OFFICIAL,
            source_id="krea.krea2.official",
            steps=StepRangeDeclaration(
                minimum=1,
                maximum=None,
                default=8,
                reference_steps=(8,),
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
    detection=KREA2_DETECTION,
    model_capabilities=_MODEL_CAPABILITIES,
    profile_capabilities=_PROFILE_CAPABILITIES,
    reference_sampler_capabilities=_REFERENCE_SAMPLER_CAPABILITIES,
    artifact_versions=KREA2_ARTIFACT_VERSIONS,
    software_sources=(KREA_SOFTWARE_PROVENANCE,),
    frameworks=KREA2_FRAMEWORK_PROVENANCE,
    model_weights=(_TURBO_WEIGHT_PROVENANCE,),
    parameters=(
        ProfileField(name="dimension_alignment_mode", value="ceil_multiple"),
        ProfileField(name="dimension_multiple", value=16),
    ),
    known_limitations=(
        "Automatic ComfyUI host evidence collection is not implemented.",
        "Only the eight-step recipe retains official Turbo evidence.",
        "Real-host and sampler-step execution remain unvalidated.",
    ),
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
        KREA_REFERENCE,
        DIFFUSERS_REFERENCE,
        COMFYUI_REFERENCE,
    )
    model_capabilities: ModelCapabilities = _MODEL_CAPABILITIES
    profile_capabilities: ProfileCapabilities = _PROFILE_CAPABILITIES
    reference_sampler_capabilities: SamplerCapabilities = _REFERENCE_SAMPLER_CAPABILITIES
    schema: ProfileSchemaV1 = KREA2_TURBO_SCHEMA

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
            KREA_REFERENCE,
            DIFFUSERS_REFERENCE,
            COMFYUI_REFERENCE,
        ):
            raise ScheduleContractError("Krea 2 Turbo evidence references are incomplete")
        if (
            self.model_capabilities != _MODEL_CAPABILITIES
            or self.profile_capabilities != _PROFILE_CAPABILITIES
            or self.reference_sampler_capabilities != _REFERENCE_SAMPLER_CAPABILITIES
            or self.schema is not KREA2_TURBO_SCHEMA
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
