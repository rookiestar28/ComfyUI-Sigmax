"""Source-qualified HunyuanImage 2.1 Base and Distilled schedules.

This module owns only the external unit-flow sigma construction.  It does not
patch a model, load weights, provide text conditioning, or claim that a
synthetic schedule proves native distilled-host support.
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
_PUBLISHER_REVISION: Final = "307df8801d176740dafb67b2872c831cb9362cf9"  # pragma: allowlist secret
_HF_REVISION: Final = "e435da11d9e8795a25e224c5ba27b099ed45c55b"  # pragma: allowlist secret
_COMFYUI_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_BASE_WEIGHT_SHA256: Final = (
    "049634539c631a6c097d989ecf2f29942637ddbb718a834a13d0c8cba22cf087"  # pragma: allowlist secret
)
_DISTILLED_WEIGHT_SHA256: Final = (
    "446ae6f68d0b4600e9c8cf8927fc27eed53953ef36299c3a444330d4c4ff92a4"  # pragma: allowlist secret
)
_MAX_STEPS: Final = 10_000
_BASE_REFERENCE_STEPS: Final = 50
_DISTILLED_REFERENCE_STEPS: Final = 8
_BASE_RATIO: Final = 5.0
_DISTILLED_RATIO: Final = 4.0


class HunyuanImage21Variant(str, Enum):
    """Explicit HunyuanImage 2.1 schedule variants."""

    BASE = "base"
    DISTILLED = "distilled"


@dataclass(frozen=True, slots=True, kw_only=True)
class HunyuanImage21EvidenceReference:
    """One pinned evidence lane for the HunyuanImage 2.1 family."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lane not in {
            "comfyui_implementation",
            "official_github",
            "official_huggingface",
            "official_technical_document",
        }:
            raise ScheduleContractError("HunyuanImage 2.1 evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("HunyuanImage 2.1 evidence URL must use HTTPS")
        valid = (
            bool(_ARXIV_PATTERN.fullmatch(self.revision))
            if self.lane == "official_technical_document"
            else bool(_COMMIT_PATTERN.fullmatch(self.revision))
        )
        if not valid:
            raise ScheduleContractError("HunyuanImage 2.1 evidence revision is invalid")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("HunyuanImage 2.1 locators must be sorted and unique")


_HUNYUAN_LICENSE = LicenseDeclaration(
    declaration_version="1",
    identifier="LicenseRef-Tencent-Hunyuan-Community",
    name="Tencent Hunyuan Community License Agreement",
    url="https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/blob/main/LICENSE",
)
_GPL_3_ONLY = LicenseDeclaration(
    declaration_version="1",
    identifier="GPL-3.0-only",
    name="GNU General Public License v3.0 only",
    url="https://www.gnu.org/licenses/gpl-3.0.html",
)
_OFFICIAL_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="tencent.hunyuanimage21.official",
    resource_version="2.1",
    revision=_PUBLISHER_REVISION,
    url="https://github.com/Tencent-Hunyuan/HunyuanImage-2.1",
    license=_HUNYUAN_LICENSE,
    locators=("LICENSE", "README.md", "hyimage/diffusion/pipelines/hunyuanimage_pipeline.py"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.hunyuanimage21.framework",
    resource_version="0.29.0",
    revision=_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
)
_BASE_WEIGHT = ModelWeightProvenance(
    record_version="1",
    weight_id="tencent.hunyuanimage21.base.dit",
    resource_version="dit/hunyuanimage2.1.safetensors",
    revision=_HF_REVISION,
    sha256=_BASE_WEIGHT_SHA256,
    url="https://huggingface.co/tencent/HunyuanImage-2.1",
    license=_HUNYUAN_LICENSE,
)
_DISTILLED_WEIGHT = ModelWeightProvenance(
    record_version="1",
    weight_id="tencent.hunyuanimage21.distilled.dit",
    resource_version="dit/hunyuanimage2.1-distilled.safetensors",
    revision=_HF_REVISION,
    sha256=_DISTILLED_WEIGHT_SHA256,
    url="https://huggingface.co/tencent/HunyuanImage-2.1",
    license=_HUNYUAN_LICENSE,
)
_DETECTION = DetectionDeclaration(
    strategy_id="hunyuanimage21.explicit-variant-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_variant",),
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
    sampler_version=_PUBLISHER_REVISION,
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


def _schema(
    *,
    variant: HunyuanImage21Variant,
    profile_id: str,
    display_name: str,
    ratio: float,
    reference_steps: int,
    guidance: float,
) -> ProfileSchemaV1:
    model = ModelCapabilities(
        model_family="hunyuanimage",
        model_variant="2.1" if variant is HunyuanImage21Variant.BASE else "2.1-distilled",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id=profile_id,
        profile_version="1",
        model_family="hunyuanimage",
        model_variant=model.model_variant,
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
    limitations = (
        "Only HunyuanImage 2.1 Base and Distilled external schedules are qualified; Refiner, text encoders, VAE, and 2K enforcement are excluded.",
        "The direct ratio owns the complete primary transform and cannot be composed with another shift or already-shifted sigmas.",
        "Model weights, conditioning, sampling execution, and image quality are not verified by this schedule profile.",
        "Native pinned ComfyUI host support is qualified for Base only; the Distilled variant is a publisher distilled schedule and its native host path is not qualified.",
    )
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        profile_version="1",
        display_name=display_name,
        model_family="hunyuanimage",
        model_variant=model.model_variant,
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
                evidence=EvidenceLevel.OFFICIAL,
                source_id=_OFFICIAL_SOURCE.source_id,
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
        frameworks=(_COMFYUI_FRAMEWORK,),
        model_weights=(_BASE_WEIGHT, _DISTILLED_WEIGHT),
        parameters=(
            ProfileField(name="multiplier", value=1.0),
            ProfileField(name="ratio", value=ratio),
            ProfileField(name="source_mode", value="official_timestep_ratio"),
            ProfileField(name="variant", value=variant.value),
        ),
        known_limitations=limitations,
    )


HUNYUAN_IMAGE21_BASE_SCHEMA: Final = _schema(
    variant=HunyuanImage21Variant.BASE,
    profile_id="hunyuan-image-2-1.base.official",
    display_name="HunyuanImage 2.1 Base Official Shift",
    ratio=_BASE_RATIO,
    reference_steps=_BASE_REFERENCE_STEPS,
    guidance=3.5,
)
HUNYUAN_IMAGE21_DISTILLED_SCHEMA: Final = _schema(
    variant=HunyuanImage21Variant.DISTILLED,
    profile_id="hunyuan-image-2-1.distilled.official",
    display_name="HunyuanImage 2.1 Distilled Official Shift",
    ratio=_DISTILLED_RATIO,
    reference_steps=_DISTILLED_REFERENCE_STEPS,
    guidance=3.25,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class HunyuanImage21Profile:
    """Immutable HunyuanImage 2.1 profile plus four pinned evidence lanes."""

    variant: HunyuanImage21Variant
    schema: ProfileSchemaV1
    references: tuple[HunyuanImage21EvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        expected = {
            HunyuanImage21Variant.BASE: HUNYUAN_IMAGE21_BASE_SCHEMA,
            HunyuanImage21Variant.DISTILLED: HUNYUAN_IMAGE21_DISTILLED_SCHEMA,
        }.get(self.variant)
        if expected is None or self.schema is not expected:
            raise ScheduleContractError("HunyuanImage 2.1 profile/schema mismatch")
        lanes = tuple(reference.lane for reference in self.references)
        if lanes != tuple(sorted(set(lanes))) or len(lanes) != 4:
            raise ScheduleContractError("HunyuanImage 2.1 requires four pinned evidence lanes")


def _references() -> tuple[HunyuanImage21EvidenceReference, ...]:
    return (
        HunyuanImage21EvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=_COMFYUI_REVISION,
            locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
        ),
        HunyuanImage21EvidenceReference(
            lane="official_github",
            url="https://github.com/Tencent-Hunyuan/HunyuanImage-2.1",
            revision=_PUBLISHER_REVISION,
            locators=(
                "LICENSE",
                "README.md",
                "hyimage/diffusion/pipelines/hunyuanimage_pipeline.py",
            ),
        ),
        HunyuanImage21EvidenceReference(
            lane="official_huggingface",
            url="https://huggingface.co/tencent/HunyuanImage-2.1",
            revision=_HF_REVISION,
            locators=(
                "README.md",
                "dit/hunyuanimage2.1-distilled.safetensors",
                "dit/hunyuanimage2.1.safetensors",
            ),
        ),
        HunyuanImage21EvidenceReference(
            lane="official_technical_document",
            url="https://arxiv.org/abs/2509.04545",
            revision="arxiv:2509.04545",
            locators=("HunyuanImage 2.1",),
        ),
    )


_REFERENCES = _references()
HUNYUAN_IMAGE21_BASE_PROFILE: Final = HunyuanImage21Profile(
    variant=HunyuanImage21Variant.BASE,
    schema=HUNYUAN_IMAGE21_BASE_SCHEMA,
    references=_REFERENCES,
)
HUNYUAN_IMAGE21_DISTILLED_PROFILE: Final = HunyuanImage21Profile(
    variant=HunyuanImage21Variant.DISTILLED,
    schema=HUNYUAN_IMAGE21_DISTILLED_SCHEMA,
    references=_REFERENCES,
)


def _profile_for_variant(
    variant: HunyuanImage21Variant,
) -> tuple[HunyuanImage21Profile, float, int]:
    if variant is HunyuanImage21Variant.BASE:
        return HUNYUAN_IMAGE21_BASE_PROFILE, _BASE_RATIO, _BASE_REFERENCE_STEPS
    if variant is HunyuanImage21Variant.DISTILLED:
        return HUNYUAN_IMAGE21_DISTILLED_PROFILE, _DISTILLED_RATIO, _DISTILLED_REFERENCE_STEPS
    raise ScheduleContractError("variant must be HunyuanImage21Variant.BASE or DISTILLED")


def build_hunyuan_image21_schedule(
    *,
    variant: HunyuanImage21Variant,
    steps: int,
    strict_source: bool = False,
    already_shifted: bool = False,
) -> ScheduleResult:
    """Build one explicit HunyuanImage 2.1 unit-flow schedule."""

    if not isinstance(variant, HunyuanImage21Variant):
        raise ScheduleContractError("variant must be an explicit HunyuanImage 2.1 variant")
    if not isinstance(strict_source, bool) or not isinstance(already_shifted, bool):
        raise ScheduleContractError("strict_source and already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError(
            "already shifted sigmas cannot be composed with HunyuanImage 2.1 shift"
        )
    profile, ratio, reference_steps = _profile_for_variant(variant)
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    if strict_source and steps != reference_steps:
        raise ScheduleContractError(
            f"steps must equal the official HunyuanImage 2.1 {reference_steps}-step recipe"
        )
    evidence = EvidenceLevel.OFFICIAL if steps == reference_steps else EvidenceLevel.MODIFIED
    warnings = (
        ()
        if evidence is EvidenceLevel.OFFICIAL
        else (
            f"steps differ from the official HunyuanImage 2.1 {reference_steps}-step recipe; evidence is modified",
        )
    )
    shifted = direct_ratio_shift(flowmatch_reciprocal_step_grid(steps), ratio=ratio)
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=steps),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=evidence,
            source=_OFFICIAL_SOURCE.url,
            source_revision=_PUBLISHER_REVISION,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
        ),
        base_grid=BaseGridSpec(
            identifier="flowmatch.reciprocal_step", output_domain=SigmaDomain.UNIT_FLOW
        ),
        transforms=(
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
        ),
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(),
    )
    sigmas = apply_terminal_policy(
        shifted, policy=TerminalPolicy.APPEND_ZERO, domain=SigmaDomain.UNIT_FLOW
    )
    return ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=steps),
        sigmas=validate_sigma_schedule(
            sigmas, domain=SigmaDomain.UNIT_FLOW, expected_steps=steps, require_terminal_zero=True
        ),
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=warnings,
    )


__all__ = [
    "HUNYUAN_IMAGE21_BASE_PROFILE",
    "HUNYUAN_IMAGE21_BASE_SCHEMA",
    "HUNYUAN_IMAGE21_DISTILLED_PROFILE",
    "HUNYUAN_IMAGE21_DISTILLED_SCHEMA",
    "HunyuanImage21EvidenceReference",
    "HunyuanImage21Profile",
    "HunyuanImage21Variant",
    "build_hunyuan_image21_schedule",
]
