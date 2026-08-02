"""Source-qualified Anima v1 schedule profiles.

This module owns only the external unit-flow sigma construction and immutable provenance.
It does not load Anima weights, run a text encoder, patch a model, or claim image quality.
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

ANIMA_REPOSITORY_REVISION: Final = (
    "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"  # pragma: allowlist secret
)
ANIMA_DIFFUSERS_REVISION: Final = (
    "073c3a9db359c31ad0e8aa268d15775473c2176c"  # pragma: allowlist secret
)
ANIMA_COMFYUI_REVISION: Final = (
    "5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25"  # pragma: allowlist secret
)

_ANIMA_URL: Final = "https://huggingface.co/circlestone-labs/Anima"
_ANIMA_DIFFUSERS_URL: Final = "https://huggingface.co/circlestone-labs/Anima-Base-v1.0-Diffusers"
_COMFYUI_URL: Final = "https://github.com/Comfy-Org/ComfyUI"
_MAX_STEPS: Final = 10_000
_SHIFT: Final = 3.0
_TRAINING_TIMESTEPS: Final = 1000
_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")


class AnimaVariant(str, Enum):
    """Released Anima v1 family variants supported by the schedule node."""

    BASE = "base"
    AESTHETIC = "aesthetic"
    TURBO = "turbo"


@dataclass(frozen=True, slots=True, kw_only=True)
class AnimaEvidenceReference:
    """One pinned evidence lane for Anima schedule qualification."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lane not in {
            "comfyui_implementation",
            "official_diffusers_conversion",
            "official_huggingface",
            "official_model_card",
        }:
            raise ScheduleContractError("Anima evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("Anima evidence URL must use HTTPS")
        if not isinstance(self.revision, str) or not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("Anima evidence revision is invalid")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("Anima evidence locators must be sorted and unique")


_ANIMA_LICENSE = LicenseDeclaration(
    declaration_version="1",
    identifier="LicenseRef-CircleStone-Labs-Non-Commercial",
    name="CircleStone Labs Non-Commercial License",
    url=_ANIMA_URL,
)
_GPL_3_ONLY = LicenseDeclaration(
    declaration_version="1",
    identifier="GPL-3.0-only",
    name="GNU General Public License v3.0 only",
    url="https://www.gnu.org/licenses/gpl-3.0.html",
)
_ANIMA_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="circlestone.anima.official",
    resource_version="v1",
    revision=ANIMA_REPOSITORY_REVISION,
    url=_ANIMA_URL,
    license=_ANIMA_LICENSE,
    locators=("LICENSE.md", "README.md"),
)
_ANIMA_DIFFUSERS_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="circlestone.anima.base.diffusers",
    resource_version="v1.0",
    revision=ANIMA_DIFFUSERS_REVISION,
    url=_ANIMA_DIFFUSERS_URL,
    license=_ANIMA_LICENSE,
    locators=("README.md", "scheduler/scheduler_config.json"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.anima.framework",
    resource_version="0.29.0",
    revision=ANIMA_COMFYUI_REVISION,
    url=_COMFYUI_URL,
    license=_GPL_3_ONLY,
    locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
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
    parameters=(ProfileField(name="training_timesteps", value=_TRAINING_TIMESTEPS),),
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
    sampler_version=ANIMA_COMFYUI_REVISION,
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
_DETECTION = DetectionDeclaration(
    strategy_id="anima.explicit-variant-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_variant",),
    suggestion_sources=(),
    family_only_sources=(),
)


def _weights_for_variant(variant: AnimaVariant) -> tuple[ModelWeightProvenance, ...]:
    if variant is AnimaVariant.BASE:
        return (
            ModelWeightProvenance(
                record_version="1",
                weight_id="circlestone.anima.base-v1-0.dit",
                resource_version="split_files/diffusion_models/anima-base-v1.0.safetensors",
                revision="457fbf842cb86e96af72c65bdd13e3f1c448de84",  # pragma: allowlist secret
                sha256="bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e",  # pragma: allowlist secret
                url=_ANIMA_URL,
                license=_ANIMA_LICENSE,
            ),
        )
    if variant is AnimaVariant.AESTHETIC:
        return (
            ModelWeightProvenance(
                record_version="1",
                weight_id="circlestone.anima.aesthetic-v1-0.dit",
                resource_version="split_files/diffusion_models/anima-aesthetic-v1.0.safetensors",
                revision="c4d6e15d2f4c2fd4ca5efc855fe066624cc250df",  # pragma: allowlist secret
                sha256="368aad0a7bdf33ed1571f4664466e0f25768ed8dc7d706697af46ab8c183bf41",  # pragma: allowlist secret
                url=_ANIMA_URL,
                license=_ANIMA_LICENSE,
            ),
            ModelWeightProvenance(
                record_version="1",
                weight_id="circlestone.anima.aesthetic-v1-0b.dit",
                resource_version="split_files/diffusion_models/anima-aesthetic-v1.0b.safetensors",
                revision="729b7ab2e1cde8101eafc54a478f8dcfab1ac3e6",  # pragma: allowlist secret
                sha256="49d1e23b9a7415ac56f4719017570a5f8d295184faba578e07fe76f41bee0fcc",  # pragma: allowlist secret
                url=_ANIMA_URL,
                license=_ANIMA_LICENSE,
            ),
            ModelWeightProvenance(
                record_version="1",
                weight_id="circlestone.anima.aesthetic-v1-1.dit",
                resource_version="split_files/diffusion_models/anima-aesthetic-v1.1.safetensors",
                revision="594c27fea35648b87c86a9b4d5436a6024c820b5",  # pragma: allowlist secret
                sha256="3c1868387a3a1ff504bbb87c33678321965ead381fcf87afbd0264daa600c082",  # pragma: allowlist secret
                url=_ANIMA_URL,
                license=_ANIMA_LICENSE,
            ),
        )
    if variant is AnimaVariant.TURBO:
        return (
            ModelWeightProvenance(
                record_version="1",
                weight_id="circlestone.anima.turbo-v1-0.dit",
                resource_version="split_files/diffusion_models/anima-turbo-v1.0.safetensors",
                revision="c4d6e15d2f4c2fd4ca5efc855fe066624cc250df",  # pragma: allowlist secret
                sha256="c0b905034510750a505d21aa96c81718f4ffcc500777318421f58a88636e2174",  # pragma: allowlist secret
                url=_ANIMA_URL,
                license=_ANIMA_LICENSE,
            ),
        )
    raise ScheduleContractError("variant must be an explicit AnimaVariant")


def _schema(
    *,
    variant: AnimaVariant,
    profile_id: str,
    display_name: str,
    model_variant: str,
    minimum_steps: int,
    maximum_steps: int,
    default_steps: int,
    reference_steps: tuple[int, ...],
    guidance: float,
) -> ProfileSchemaV1:
    model = ModelCapabilities(
        model_family="anima",
        model_variant=model_variant,
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id=profile_id,
        profile_version="1",
        model_family="anima",
        model_variant=model_variant,
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
    variant_limit = {
        AnimaVariant.BASE: "Anima Base v1.0",
        AnimaVariant.AESTHETIC: "Anima Aesthetic v1.x",
        AnimaVariant.TURBO: "Anima Turbo v1.0",
    }[variant]
    limitations = (
        f"Only released {variant_limit} checkpoints are in scope; previews and ambiguous automatic detection are excluded.",
        "The fixed rational shift 3.0 is applied exactly once; dynamic base/max shift controls are disabled and not exposed.",
        "The schedule node does not load weights, run conditioning, select a sampler, or prove image quality or prompt adherence.",
        "Anima weight files are subject to the CircleStone Labs non-commercial license and applicable NVIDIA derivative terms; those terms do not license Sigmax source code.",
    )
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        profile_version="1",
        display_name=display_name,
        model_family="anima",
        model_variant=model_variant,
        evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
        primary_source_id=_ANIMA_SOURCE.source_id,
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        base_grid=_BASE_GRID,
        transforms=(
            TransformDeclaration(
                identifier="rational_shift.fixed",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
                parameters=(ProfileField(name="shift", value=_SHIFT),),
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
                evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
                source_id=_ANIMA_SOURCE.source_id,
                steps=StepRangeDeclaration(
                    minimum=minimum_steps,
                    maximum=maximum_steps,
                    default=default_steps,
                    reference_steps=reference_steps,
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
        software_sources=(_ANIMA_DIFFUSERS_SOURCE, _ANIMA_SOURCE),
        frameworks=(_COMFYUI_FRAMEWORK,),
        model_weights=_weights_for_variant(variant),
        parameters=(
            ProfileField(name="dynamic_shift", value=False),
            ProfileField(name="multiplier", value=1.0),
            ProfileField(name="shift", value=_SHIFT),
            ProfileField(name="training_timesteps", value=_TRAINING_TIMESTEPS),
        ),
        known_limitations=limitations,
    )


ANIMA_BASE_SCHEMA: Final = _schema(
    variant=AnimaVariant.BASE,
    profile_id="anima.base.framework-reference",
    display_name="Anima Base v1.0 Fixed Shift",
    model_variant="base-v1.0",
    minimum_steps=30,
    maximum_steps=50,
    default_steps=50,
    reference_steps=(30, 50),
    guidance=4.5,
)
ANIMA_AESTHETIC_SCHEMA: Final = _schema(
    variant=AnimaVariant.AESTHETIC,
    profile_id="anima.aesthetic.framework-reference",
    display_name="Anima Aesthetic v1.x Fixed Shift",
    model_variant="aesthetic-v1",
    minimum_steps=30,
    maximum_steps=50,
    default_steps=50,
    reference_steps=(30, 50),
    guidance=4.5,
)
ANIMA_TURBO_SCHEMA: Final = _schema(
    variant=AnimaVariant.TURBO,
    profile_id="anima.turbo.framework-reference",
    display_name="Anima Turbo v1.0 Fixed Shift",
    model_variant="turbo-v1.0",
    minimum_steps=8,
    maximum_steps=12,
    default_steps=8,
    reference_steps=(8, 12),
    guidance=1.0,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnimaProfile:
    """Immutable Anima profile plus pinned source evidence."""

    variant: AnimaVariant
    schema: ProfileSchemaV1
    references: tuple[AnimaEvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        expected = {
            AnimaVariant.BASE: ANIMA_BASE_SCHEMA,
            AnimaVariant.AESTHETIC: ANIMA_AESTHETIC_SCHEMA,
            AnimaVariant.TURBO: ANIMA_TURBO_SCHEMA,
        }.get(self.variant)
        if expected is None or self.schema is not expected:
            raise ScheduleContractError("Anima profile/schema mismatch")
        lanes = tuple(reference.lane for reference in self.references)
        if lanes != tuple(sorted(set(lanes))) or len(lanes) != 4:
            raise ScheduleContractError("Anima requires four pinned evidence lanes")


def _references() -> tuple[AnimaEvidenceReference, ...]:
    return (
        AnimaEvidenceReference(
            lane="comfyui_implementation",
            url=_COMFYUI_URL,
            revision=ANIMA_COMFYUI_REVISION,
            locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
        ),
        AnimaEvidenceReference(
            lane="official_diffusers_conversion",
            url=_ANIMA_DIFFUSERS_URL,
            revision=ANIMA_DIFFUSERS_REVISION,
            locators=("README.md", "scheduler/scheduler_config.json"),
        ),
        AnimaEvidenceReference(
            lane="official_huggingface",
            url=_ANIMA_URL,
            revision=ANIMA_REPOSITORY_REVISION,
            locators=(
                "split_files/diffusion_models/anima-aesthetic-v1.1.safetensors",
                "split_files/diffusion_models/anima-base-v1.0.safetensors",
                "split_files/diffusion_models/anima-turbo-v1.0.safetensors",
            ),
        ),
        AnimaEvidenceReference(
            lane="official_model_card",
            url=_ANIMA_URL,
            revision=ANIMA_REPOSITORY_REVISION,
            locators=("LICENSE.md", "README.md"),
        ),
    )


_REFERENCES = _references()
ANIMA_BASE_PROFILE: Final = AnimaProfile(
    variant=AnimaVariant.BASE,
    schema=ANIMA_BASE_SCHEMA,
    references=_REFERENCES,
)
ANIMA_AESTHETIC_PROFILE: Final = AnimaProfile(
    variant=AnimaVariant.AESTHETIC,
    schema=ANIMA_AESTHETIC_SCHEMA,
    references=_REFERENCES,
)
ANIMA_TURBO_PROFILE: Final = AnimaProfile(
    variant=AnimaVariant.TURBO,
    schema=ANIMA_TURBO_SCHEMA,
    references=_REFERENCES,
)


def _profile_for_variant(
    variant: AnimaVariant,
) -> tuple[AnimaProfile, int, int]:
    if variant is AnimaVariant.BASE:
        return ANIMA_BASE_PROFILE, 30, 50
    if variant is AnimaVariant.AESTHETIC:
        return ANIMA_AESTHETIC_PROFILE, 30, 50
    if variant is AnimaVariant.TURBO:
        return ANIMA_TURBO_PROFILE, 8, 12
    raise ScheduleContractError("variant must be an explicit AnimaVariant")


def build_anima_schedule(
    *,
    variant: AnimaVariant,
    steps: int,
    strict_source: bool = False,
    already_shifted: bool = False,
) -> ScheduleResult:
    """Build one explicit Anima fixed-shift unit-flow schedule."""

    if not isinstance(variant, AnimaVariant):
        raise ScheduleContractError("variant must be an explicit AnimaVariant")
    if not isinstance(strict_source, bool) or not isinstance(already_shifted, bool):
        raise ScheduleContractError("strict_source and already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError("already shifted sigmas cannot be composed with Anima shift")
    profile, minimum_steps, maximum_steps = _profile_for_variant(variant)
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    in_recipe_range = minimum_steps <= steps <= maximum_steps
    if strict_source and not in_recipe_range:
        raise ScheduleContractError(
            f"steps must be inside the Anima {variant.value} recipe range "
            f"{minimum_steps}-{maximum_steps}"
        )
    evidence = EvidenceLevel.FRAMEWORK_REFERENCE if in_recipe_range else EvidenceLevel.MODIFIED
    warnings = (
        ()
        if in_recipe_range
        else (
            f"steps differ from the pinned Anima {variant.value} recipe range "
            f"{minimum_steps}-{maximum_steps}; evidence is modified",
        )
    )
    shifted = direct_ratio_shift(
        flowmatch_reciprocal_step_grid(steps), ratio=_SHIFT, domain=SigmaDomain.UNIT_FLOW
    )
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=steps),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=evidence,
            source=_ANIMA_SOURCE.url,
            source_revision=ANIMA_REPOSITORY_REVISION,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
        ),
        base_grid=BaseGridSpec(
            identifier="flowmatch.reciprocal_step", output_domain=SigmaDomain.UNIT_FLOW
        ),
        transforms=(
            TransformContract(
                name="rational_shift.fixed",
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
    "ANIMA_AESTHETIC_PROFILE",
    "ANIMA_AESTHETIC_SCHEMA",
    "ANIMA_BASE_PROFILE",
    "ANIMA_BASE_SCHEMA",
    "ANIMA_COMFYUI_REVISION",
    "ANIMA_DIFFUSERS_REVISION",
    "ANIMA_REPOSITORY_REVISION",
    "ANIMA_TURBO_PROFILE",
    "ANIMA_TURBO_SCHEMA",
    "AnimaEvidenceReference",
    "AnimaProfile",
    "AnimaVariant",
    "build_anima_schedule",
]
