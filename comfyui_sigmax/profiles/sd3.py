"""Source-qualified original Stable Diffusion 3 schedule profiles.

The original SD3 publisher recipe and the fixed-shift surface used by the pinned
ComfyUI/Diffusers implementations are intentionally separate modes.  They share
the clean-room FlowMatch construction, but callers must choose the source lane
explicitly; one mode never silently falls through to the other.
"""

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
_ARXIV_PATTERN: Final = re.compile(r"^arxiv:[0-9]{4}\.[0-9]{5}(?:v[0-9]+)?$")
_PUBLISHER_REVISION: Final = "8565799a3b41eb0c7ba976d18375f0f753f56402"  # pragma: allowlist secret
_HF_REVISION: Final = "19b7f516efea082d257947e057e6f419e26fd497"  # pragma: allowlist secret
_COMFYUI_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_DIFFUSERS_REVISION: Final = "cc92165331e1b20afc1a47e03f63e8f3a930f8cc"  # pragma: allowlist secret
_SD3_MEDIUM_SHA256: Final = (
    "cc236278d28c8c3eccb8e21ee0a67ebed7dd6e9ce40aa9de914fa34e8282f191"  # pragma: allowlist secret
)
_MAX_STEPS: Final = 10_000
_PUBLISHER_REFERENCE_STEPS: Final = 50
_FRAMEWORK_REFERENCE_STEPS: Final = 28
_PUBLISHER_SHIFT: Final = 1.0
_FRAMEWORK_SHIFT: Final = 3.0


class SD3ShiftMode(str, Enum):
    """Non-composable source lanes for original SD3 shift ownership."""

    PUBLISHER_REFERENCE = "publisher_reference"
    COMFY_DIFFUSERS_FIXED = "comfy_diffusers_fixed"


@dataclass(frozen=True, slots=True, kw_only=True)
class SD3EvidenceReference:
    """One pinned source lane used to qualify the original SD3 schedule."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lane not in {
            "comfyui_implementation",
            "diffusers_framework",
            "official_github",
            "official_huggingface",
            "official_technical_document",
        }:
            raise ScheduleContractError("SD3 evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("SD3 evidence URL must use HTTPS")
        if self.lane == "official_technical_document":
            if not _ARXIV_PATTERN.fullmatch(self.revision):
                raise ScheduleContractError("SD3 technical revision must be an arXiv ID")
        elif not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("SD3 evidence revision must be pinned")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("SD3 evidence locators must be sorted and unique")


_MIT = LicenseDeclaration(
    declaration_version="1",
    identifier="MIT",
    name="MIT License",
    url="https://opensource.org/license/mit/",
)
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
_STABILITY_COMMUNITY = LicenseDeclaration(
    declaration_version="1",
    identifier="LicenseRef-Stability-AI-Community",
    name="Stability AI Community License",
    url="https://stability.ai/license",
)

_OFFICIAL_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="stabilityai.sd3.official",
    resource_version=None,
    revision=_PUBLISHER_REVISION,
    url="https://github.com/Stability-AI/sd3.5",
    license=_MIT,
    locators=("LICENSE-CODE", "sd3_impls.py", "sd3_infer.py"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.sd3.framework",
    resource_version="0.29.0",
    revision=_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=("comfy/model_sampling.py", "comfy/samplers.py", "comfy/supported_models.py"),
)
_DIFFUSERS_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.sd3.framework",
    resource_version="0.39.0",
    revision=_DIFFUSERS_REVISION,
    url="https://github.com/huggingface/diffusers",
    license=_APACHE_2,
    locators=("src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",),
)
_SD3_MEDIUM_WEIGHT = ModelWeightProvenance(
    record_version="1",
    weight_id="stabilityai.sd3-medium.weights",
    resource_version="sd3_medium.safetensors",
    revision=_HF_REVISION,
    sha256=_SD3_MEDIUM_SHA256,
    url="https://huggingface.co/stabilityai/stable-diffusion-3-medium",
    license=_STABILITY_COMMUNITY,
)

_DETECTION = DetectionDeclaration(
    strategy_id="sd3.original.explicit-mode-v1",
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


def _capabilities(profile_id: str) -> tuple[ModelCapabilities, ProfileCapabilities]:
    model = ModelCapabilities(
        model_family="sd3",
        model_variant="original",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id=profile_id,
        profile_version="1",
        model_family="sd3",
        model_variant="original",
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


def _schema(
    *,
    profile_id: str,
    display_name: str,
    evidence: EvidenceLevel,
    primary_source_id: str,
    ratio: float,
    reference_steps: int,
    guidance: float,
    source_mode: str,
) -> ProfileSchemaV1:
    model, profile = _capabilities(profile_id)
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        profile_version="1",
        display_name=display_name,
        model_family="sd3",
        model_variant="original",
        evidence=evidence,
        primary_source_id=primary_source_id,
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
                parameters=(ProfileField(name="ratio", value=ratio),),
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
                recipe_id=profile_id,
                evidence=evidence,
                source_id=primary_source_id,
                steps=StepRangeDeclaration(
                    minimum=1,
                    maximum=_MAX_STEPS,
                    default=reference_steps,
                    reference_steps=(reference_steps,),
                    allow_modified=True,
                ),
                guidance=GuidanceDeclaration(
                    model_convention="cfg_scale",
                    host_convention="cfg_scale",
                    model_value=guidance,
                    host_value=guidance,
                ),
            ),
        ),
        detection=_DETECTION,
        model_capabilities=model,
        profile_capabilities=profile,
        reference_sampler_capabilities=_SAMPLER,
        artifact_versions=_ARTIFACT_VERSIONS,
        software_sources=(_OFFICIAL_SOURCE,),
        frameworks=(_COMFYUI_FRAMEWORK, _DIFFUSERS_FRAMEWORK),
        model_weights=(_SD3_MEDIUM_WEIGHT,),
        parameters=(
            ProfileField(name="shift", value=ratio),
            ProfileField(name="source_mode", value=source_mode),
            ProfileField(name="variant", value="original"),
        ),
        known_limitations=(
            "Only the original Stability AI SD3 Medium text-to-image schedule is qualified; SD3.5, Turbo, ControlNet, and other variants are excluded.",
            "The selected source mode owns the complete primary shift; fixed and publisher modes cannot be composed or applied to already-shifted sigmas.",
            "Model weights, text encoders, conditioning, sampling execution, and image quality are not verified by this schedule profile.",
            "The publisher 1.0 and ComfyUI/Diffusers 3.0 values are separate evidence lanes, not an unqualified SD3 default.",
        ),
    )


SD3_PUBLISHER_REFERENCE_SCHEMA: Final = _schema(
    profile_id="sd3.publisher-reference.official",
    display_name="Stable Diffusion 3 Publisher Reference Shift",
    evidence=EvidenceLevel.OFFICIAL,
    primary_source_id=_OFFICIAL_SOURCE.source_id,
    ratio=_PUBLISHER_SHIFT,
    reference_steps=_PUBLISHER_REFERENCE_STEPS,
    guidance=5.0,
    source_mode=SD3ShiftMode.PUBLISHER_REFERENCE.value,
)
SD3_COMFY_DIFFUSERS_SCHEMA: Final = _schema(
    profile_id="sd3.comfy-diffusers-fixed.framework-reference",
    display_name="Stable Diffusion 3 ComfyUI/Diffusers Fixed Shift",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    primary_source_id=_COMFYUI_FRAMEWORK.framework_id,
    ratio=_FRAMEWORK_SHIFT,
    reference_steps=_FRAMEWORK_REFERENCE_STEPS,
    guidance=7.0,
    source_mode=SD3ShiftMode.COMFY_DIFFUSERS_FIXED.value,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SD3Profile:
    """Immutable original SD3 profile plus its five pinned evidence lanes."""

    shift_mode: SD3ShiftMode
    schema: ProfileSchemaV1
    references: tuple[SD3EvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        if not isinstance(self.shift_mode, SD3ShiftMode):
            raise ScheduleContractError("SD3 shift mode is unsupported")
        if not isinstance(self.schema, ProfileSchemaV1):
            raise ScheduleContractError("SD3 schema is required")
        lanes = tuple(reference.lane for reference in self.references)
        if len(lanes) != 5 or len(set(lanes)) != 5 or lanes != tuple(sorted(lanes)):
            raise ScheduleContractError("SD3 requires five pinned evidence lanes")
        expected = {
            SD3ShiftMode.PUBLISHER_REFERENCE: SD3_PUBLISHER_REFERENCE_SCHEMA,
            SD3ShiftMode.COMFY_DIFFUSERS_FIXED: SD3_COMFY_DIFFUSERS_SCHEMA,
        }[self.shift_mode]
        if self.schema is not expected:
            raise ScheduleContractError("SD3 profile/schema mismatch")


def _references() -> tuple[SD3EvidenceReference, ...]:
    return (
        SD3EvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=_COMFYUI_REVISION,
            locators=("comfy/model_sampling.py", "comfy/samplers.py", "comfy/supported_models.py"),
        ),
        SD3EvidenceReference(
            lane="diffusers_framework",
            url="https://github.com/huggingface/diffusers",
            revision=_DIFFUSERS_REVISION,
            locators=("src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",),
        ),
        SD3EvidenceReference(
            lane="official_github",
            url="https://github.com/Stability-AI/sd3.5",
            revision=_PUBLISHER_REVISION,
            locators=("LICENSE-CODE", "sd3_impls.py", "sd3_infer.py"),
        ),
        SD3EvidenceReference(
            lane="official_huggingface",
            url="https://huggingface.co/stabilityai/stable-diffusion-3-medium",
            revision=_HF_REVISION,
            locators=("LICENSE", "README.md", "sd3_medium.safetensors"),
        ),
        SD3EvidenceReference(
            lane="official_technical_document",
            url="https://arxiv.org/abs/2403.03206",
            revision="arxiv:2403.03206",
            locators=("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis",),
        ),
    )


SD3_PUBLISHER_REFERENCE_PROFILE: Final = SD3Profile(
    shift_mode=SD3ShiftMode.PUBLISHER_REFERENCE,
    schema=SD3_PUBLISHER_REFERENCE_SCHEMA,
    references=_references(),
)
SD3_COMFY_DIFFUSERS_PROFILE: Final = SD3Profile(
    shift_mode=SD3ShiftMode.COMFY_DIFFUSERS_FIXED,
    schema=SD3_COMFY_DIFFUSERS_SCHEMA,
    references=_references(),
)


def _profile(mode: SD3ShiftMode) -> SD3Profile:
    if mode is SD3ShiftMode.PUBLISHER_REFERENCE:
        return SD3_PUBLISHER_REFERENCE_PROFILE
    if mode is SD3ShiftMode.COMFY_DIFFUSERS_FIXED:
        return SD3_COMFY_DIFFUSERS_PROFILE
    raise ScheduleContractError("mode must be an explicit SD3 source mode")


def build_sd3_schedule(
    *,
    mode: SD3ShiftMode,
    steps: int,
    strict_source: bool = False,
    already_shifted: bool = False,
) -> ScheduleResult:
    """Build an explicit original-SD3 unit-flow schedule without host imports."""

    if not isinstance(mode, SD3ShiftMode):
        raise ScheduleContractError("mode must be an explicit SD3 source mode")
    profile = _profile(mode)
    if not isinstance(strict_source, bool):
        raise ScheduleContractError("strict_source must be boolean")
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError(
            "already shifted sigmas cannot be composed with an SD3 source mode"
        )
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")

    reference_steps = (
        _PUBLISHER_REFERENCE_STEPS
        if mode is SD3ShiftMode.PUBLISHER_REFERENCE
        else _FRAMEWORK_REFERENCE_STEPS
    )
    if strict_source and steps != reference_steps:
        if mode is SD3ShiftMode.PUBLISHER_REFERENCE:
            raise ScheduleContractError("steps are outside the published 50-step SD3 recipe")
        raise ScheduleContractError("steps are outside the framework-reference 28-step SD3 recipe")

    evidence = profile.schema.evidence if steps == reference_steps else EvidenceLevel.MODIFIED
    warnings = (
        ()
        if steps == reference_steps
        else (
            f"steps differ from the source reference {reference_steps}-step recipe; evidence is modified",
        )
    )
    ratio = _PUBLISHER_SHIFT if mode is SD3ShiftMode.PUBLISHER_REFERENCE else _FRAMEWORK_SHIFT
    shifted = direct_ratio_shift(flowmatch_reciprocal_step_grid(steps), ratio=ratio)
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
    source_url = (
        _OFFICIAL_SOURCE.url if mode is SD3ShiftMode.PUBLISHER_REFERENCE else _COMFYUI_FRAMEWORK.url
    )
    source_revision = (
        _PUBLISHER_REVISION if mode is SD3ShiftMode.PUBLISHER_REFERENCE else _COMFYUI_REVISION
    )
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=steps),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=evidence,
            source=source_url,
            source_revision=source_revision,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
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
    "SD3_COMFY_DIFFUSERS_PROFILE",
    "SD3_COMFY_DIFFUSERS_SCHEMA",
    "SD3_PUBLISHER_REFERENCE_PROFILE",
    "SD3_PUBLISHER_REFERENCE_SCHEMA",
    "SD3EvidenceReference",
    "SD3Profile",
    "SD3ShiftMode",
    "build_sd3_schedule",
]
