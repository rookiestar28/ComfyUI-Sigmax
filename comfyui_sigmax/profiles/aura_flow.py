"""Four-source-pinned original AuraFlow v0.2 fixed-shift schedule profile."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import (
    BaseGridSpec,
    EvidenceLevel,
    ExecutionBehavior,
    ModelCapabilities,
    NoiseOwnership,
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
    direct_ratio_shift,
    flowmatch_reciprocal_step_grid,
    validate_sigma_schedule,
)
from comfyui_sigmax.profiles.schema_v1 import (
    PROFILE_SCHEMA_ID,
    PROFILE_SCHEMA_VERSION,
    ArtifactVersionDeclaration,
    BaseGridDeclaration,
    DetectionDeclaration,
    FrameworkProvenance,
    GuidanceDeclaration,
    InferenceRecipe,
    LicenseDeclaration,
    ModelWeightProvenance,
    ProfileField,
    ProfileSchemaV1,
    SlicingDeclaration,
    SoftwareSourceProvenance,
    StepRangeDeclaration,
    TerminalDeclaration,
    TransformDeclaration,
)
from comfyui_sigmax.version import VERSION

_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_DATE_PATTERN: Final = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_HF_REVISION: Final = "ea13150f559b7f85d2c5959297f7de10325584b4"  # pragma: allowlist secret
_COMFYUI_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_DIFFUSERS_REVISION: Final = "3c468926ffd12b69baa4316e27b09306b8da19a6"  # pragma: allowlist secret
_WEIGHT_SHA256: Final = (
    "e74e424b409ce5a9d7b5487f00610ec048930918fba875c6beeed88255844b7e"  # pragma: allowlist secret
)
_MAX_STEPS: Final = 10_000
_REFERENCE_STEPS: Final = 50
_SHIFT: Final = 1.73


class AuraFlowShiftMode(str, Enum):
    """The only qualified original AuraFlow v0.2 shift ownership mode."""

    OFFICIAL_FIXED = "official_fixed"


@dataclass(frozen=True, slots=True, kw_only=True)
class AuraFlowEvidenceReference:
    """One pinned evidence lane used to qualify AuraFlow v0.2."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lane not in {
            "comfyui_implementation",
            "diffusers_framework",
            "official_huggingface",
            "official_technical_document",
        }:
            raise ScheduleContractError("AuraFlow evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("AuraFlow evidence URL must use HTTPS")
        if self.lane == "official_technical_document":
            valid_revision = bool(_DATE_PATTERN.fullmatch(self.revision))
        else:
            valid_revision = bool(_COMMIT_PATTERN.fullmatch(self.revision))
        if not valid_revision:
            raise ScheduleContractError("AuraFlow evidence revision is invalid")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("AuraFlow evidence locators must be sorted and unique")


_APACHE_2 = LicenseDeclaration(
    declaration_version="1",
    identifier="Apache-2.0",
    name="Apache License 2.0",
    url="https://www.apache.org/licenses/LICENSE-2.0",
)
_GPL_3_ONLY = LicenseDeclaration(
    declaration_version="1",
    identifier="GPL-3.0-only",
    name="GNU General Public License v3.0 only",
    url="https://www.gnu.org/licenses/gpl-3.0.html",
)
_OFFICIAL_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="fal.auraflow.v0-2.official",
    resource_version="v0.2",
    revision=_HF_REVISION,
    url="https://huggingface.co/fal/AuraFlow-v0.2",
    license=_APACHE_2,
    locators=("LICENSE", "README.md", "scheduler/scheduler_config.json"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.auraflow.framework",
    resource_version="0.29.0",
    revision=_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=(
        "comfy/model_sampling.py",
        "comfy/supported_models.py",
        "comfy_extras/nodes_model_advanced.py",
    ),
)
_DIFFUSERS_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.auraflow.framework",
    resource_version="0.39.0",
    revision=_DIFFUSERS_REVISION,
    url="https://github.com/huggingface/diffusers",
    license=_APACHE_2,
    locators=(
        "src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py",
        "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    ),
)
_AURAFLOW_WEIGHT = ModelWeightProvenance(
    record_version="1",
    weight_id="fal.auraflow.v0-2.weights",
    resource_version="aura_flow_0.2.safetensors",
    revision=_HF_REVISION,
    sha256=_WEIGHT_SHA256,
    url="https://huggingface.co/fal/AuraFlow-v0.2",
    license=_APACHE_2,
)
_DETECTION = DetectionDeclaration(
    strategy_id="auraflow.v0-2.explicit-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_mode", "explicit_variant"),
    suggestion_sources=(),
    family_only_sources=(),
)
_ARTIFACT_VERSIONS = ArtifactVersionDeclaration(
    numerical_schema="sigmax.numerical-schedule/1",
    construction_schema="sigmax.schedule-artifact/1",
    envelope_schema="sigmax.schedule-artifact-envelope/1",
)
_BASE_GRID = BaseGridDeclaration(
    identifier="flowmatch.reciprocal_step",
    output_domain=SigmaDomain.UNIT_FLOW,
    terminal_included=False,
)
_TERMINAL = TerminalDeclaration(
    policy=TerminalPolicy.APPEND_ZERO,
    sigma=TerminalSigma.ZERO,
    value=0.0,
)
_SLICING = SlicingDeclaration(
    supports_step_range=True,
    supports_denoise_tail=True,
    zero_denoise_is_empty=True,
)
_SAMPLER = SamplerCapabilities(
    sampler_id="flowmatch.euler",
    sampler_version=_DIFFUSERS_REVISION,
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


def _capabilities() -> tuple[ModelCapabilities, ProfileCapabilities]:
    model = ModelCapabilities(
        model_family="auraflow",
        model_variant="v0.2",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id="auraflow.v0-2.official",
        profile_version="1",
        model_family="auraflow",
        model_variant="v0.2",
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        terminal_sigma=TerminalSigma.ZERO,
        allowed_execution_behaviors=(ExecutionBehavior.DETERMINISTIC,),
        allowed_noise_ownerships=(NoiseOwnership.NONE,),
        allowed_sampler_state=(),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
        reference_sampler_ids=("flowmatch.euler",),
    )
    return model, profile


_MODEL_CAPABILITIES, _PROFILE_CAPABILITIES = _capabilities()

AURAFLOW_V02_SCHEMA: Final = ProfileSchemaV1(
    schema_id=PROFILE_SCHEMA_ID,
    schema_version=PROFILE_SCHEMA_VERSION,
    profile_id="auraflow.v0-2.official",
    profile_version="1",
    display_name="AuraFlow v0.2 Official Fixed Shift",
    model_family="auraflow",
    model_variant="v0.2",
    evidence=EvidenceLevel.OFFICIAL,
    primary_source_id=_OFFICIAL_SOURCE.source_id,
    prediction_type=PredictionType.FLOW_VELOCITY,
    sigma_domain=SigmaDomain.UNIT_FLOW,
    ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
    base_grid=_BASE_GRID,
    transforms=(
        TransformDeclaration(
            identifier="direct_ratio.shift",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
            parameters=(ProfileField(name="ratio", value=_SHIFT),),
        ),
        TransformDeclaration(
            identifier="terminal.append_zero",
            stage=TransformStage.TERMINAL,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
    ),
    terminal=_TERMINAL,
    slicing=_SLICING,
    recipes=(
        InferenceRecipe(
            recipe_id="auraflow.v0-2.official",
            evidence=EvidenceLevel.OFFICIAL,
            source_id=_OFFICIAL_SOURCE.source_id,
            steps=StepRangeDeclaration(
                minimum=1,
                maximum=_MAX_STEPS,
                default=_REFERENCE_STEPS,
                reference_steps=(_REFERENCE_STEPS,),
                allow_modified=True,
            ),
            guidance=GuidanceDeclaration(
                model_convention="cfg_scale",
                host_convention="cfg_scale",
                model_value=3.5,
                host_value=3.5,
            ),
        ),
    ),
    detection=_DETECTION,
    model_capabilities=_MODEL_CAPABILITIES,
    profile_capabilities=_PROFILE_CAPABILITIES,
    reference_sampler_capabilities=_SAMPLER,
    artifact_versions=_ARTIFACT_VERSIONS,
    software_sources=(_OFFICIAL_SOURCE,),
    frameworks=(_COMFYUI_FRAMEWORK, _DIFFUSERS_FRAMEWORK),
    model_weights=(_AURAFLOW_WEIGHT,),
    parameters=(
        ProfileField(name="multiplier", value=1.0),
        ProfileField(name="shift", value=_SHIFT),
        ProfileField(name="source_mode", value=AuraFlowShiftMode.OFFICIAL_FIXED.value),
        ProfileField(name="variant", value="v0.2"),
    ),
    known_limitations=(
        "Only the original fal AuraFlow v0.2 text-to-image schedule is qualified; v0.1, v0.3, PonyFlow, and community finetunes are excluded.",
        "The fixed ratio shift owns the complete primary transform; it cannot be composed with another shift or already-shifted sigmas.",
        "Model weights, text encoders, conditioning, sampling execution, and image quality are not verified by this schedule profile.",
        "The official fixed 1.73 value is a model-native time-unit contract; resolution-dependent and dynamic-shift controls are intentionally absent.",
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AuraFlowProfile:
    """Immutable original AuraFlow v0.2 profile plus pinned evidence."""

    shift_mode: AuraFlowShiftMode
    schema: ProfileSchemaV1
    references: tuple[AuraFlowEvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        if not isinstance(self.shift_mode, AuraFlowShiftMode):
            raise ScheduleContractError("AuraFlow shift mode is unsupported")
        if self.schema is not AURAFLOW_V02_SCHEMA:
            raise ScheduleContractError("AuraFlow profile/schema mismatch")
        lanes = tuple(reference.lane for reference in self.references)
        if lanes != tuple(sorted(set(lanes))) or len(lanes) != 4:
            raise ScheduleContractError("AuraFlow requires four unique pinned evidence lanes")


def _references() -> tuple[AuraFlowEvidenceReference, ...]:
    return (
        AuraFlowEvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=_COMFYUI_REVISION,
            locators=(
                "comfy/model_sampling.py",
                "comfy/supported_models.py",
                "comfy_extras/nodes_model_advanced.py",
            ),
        ),
        AuraFlowEvidenceReference(
            lane="diffusers_framework",
            url="https://github.com/huggingface/diffusers",
            revision=_DIFFUSERS_REVISION,
            locators=(
                "src/diffusers/pipelines/aura_flow/pipeline_aura_flow.py",
                "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
            ),
        ),
        AuraFlowEvidenceReference(
            lane="official_huggingface",
            url="https://huggingface.co/fal/AuraFlow-v0.2",
            revision=_HF_REVISION,
            locators=("LICENSE", "README.md", "scheduler/scheduler_config.json"),
        ),
        AuraFlowEvidenceReference(
            lane="official_technical_document",
            url="https://blog.fal.ai/auraflow/",
            revision="2024-07-12",
            locators=("How do I use it?", "Technical Details"),
        ),
    )


AURAFLOW_V02_PROFILE: Final = AuraFlowProfile(
    shift_mode=AuraFlowShiftMode.OFFICIAL_FIXED,
    schema=AURAFLOW_V02_SCHEMA,
    references=_references(),
)


def build_aura_flow_schedule(
    *,
    mode: AuraFlowShiftMode,
    steps: int,
    strict_source: bool = False,
    already_shifted: bool = False,
) -> ScheduleResult:
    """Build one explicit original AuraFlow v0.2 unit-flow schedule."""

    if not isinstance(mode, AuraFlowShiftMode):
        raise ScheduleContractError("mode must be the explicit AuraFlow Official Fixed mode")
    if not isinstance(strict_source, bool):
        raise ScheduleContractError("strict_source must be boolean")
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError(
            "already shifted sigmas cannot be composed with the AuraFlow fixed shift"
        )
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    if strict_source and steps != _REFERENCE_STEPS:
        raise ScheduleContractError("steps are outside the official AuraFlow 50-step recipe")

    evidence = EvidenceLevel.OFFICIAL if steps == _REFERENCE_STEPS else EvidenceLevel.MODIFIED
    warnings = (
        ()
        if evidence is EvidenceLevel.OFFICIAL
        else ("steps differ from the official 50-step AuraFlow recipe; evidence is modified",)
    )
    shifted = direct_ratio_shift(flowmatch_reciprocal_step_grid(steps), ratio=_SHIFT)
    transforms = (
        TransformContract(
            name="direct_ratio.shift",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        TransformContract(
            name="terminal.append_zero",
            stage=TransformStage.TERMINAL,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
    )
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=steps),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=evidence,
            source=_OFFICIAL_SOURCE.url,
            source_revision=_HF_REVISION,
            profile_id=AURAFLOW_V02_PROFILE.profile_id,
            profile_version=AURAFLOW_V02_PROFILE.profile_version,
        ),
        base_grid=BaseGridSpec(
            identifier="flowmatch.reciprocal_step",
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        transforms=transforms,
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(),
    )
    sigmas = apply_terminal_policy(
        shifted,
        policy=TerminalPolicy.APPEND_ZERO,
        domain=SigmaDomain.UNIT_FLOW,
    )
    return ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=steps),
        sigmas=validate_sigma_schedule(
            sigmas,
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=steps,
            require_terminal_zero=True,
        ),
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=warnings,
    )


__all__ = [
    "AURAFLOW_V02_PROFILE",
    "AURAFLOW_V02_SCHEMA",
    "AuraFlowEvidenceReference",
    "AuraFlowProfile",
    "AuraFlowShiftMode",
    "build_aura_flow_schedule",
]
