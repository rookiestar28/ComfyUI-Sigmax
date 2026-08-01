"""Four-source-pinned Z-Image Base and Turbo schedule profiles."""

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
_GITHUB_REVISION: Final = "26f23eda626ffadda020b04ff79488e1d72004cd"  # pragma: allowlist secret
_SITE_REVISION: Final = "e67bafb673fa19d301f903ac62de26c48b4cc1c4"  # pragma: allowlist secret
_COMFYUI_REVISION: Final = "235b466a0cb26d47c24f2ab66d1a8c5e70b21070"  # pragma: allowlist secret
_BASE_HF_REVISION: Final = "04cc4abb7c5069926f75c9bfde9ef43d49423021"  # pragma: allowlist secret
_TURBO_HF_REVISION: Final = "f332072aa78be7aecdf3ee76d5c247082da564a6"  # pragma: allowlist secret


class ZImageVariant(str, Enum):
    """Explicit released Z-Image variants; automatic guessing is intentionally absent."""

    BASE = "base"
    TURBO = "turbo"


@dataclass(frozen=True, slots=True, kw_only=True)
class ZImageEvidenceReference:
    """One mandatory evidence lane for a Z-Image core claim."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lane not in {
            "official_github",
            "official_huggingface",
            "official_technical_document",
            "comfyui_implementation",
        }:
            raise ScheduleContractError("Z-Image evidence lane is unsupported")
        if not self.url.startswith("https://"):
            raise ScheduleContractError("Z-Image evidence URL must use HTTPS")
        if not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("Z-Image evidence revision must be pinned")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("Z-Image evidence locators must be sorted and unique")


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
    source_id="tongyi-mai.z-image.official",
    resource_version=None,
    revision=_GITHUB_REVISION,
    url="https://github.com/Tongyi-MAI/Z-Image",
    license=_APACHE_2,
    locators=("LICENSE", "README.md", "src/zimage/pipeline.py", "src/zimage/scheduler.py"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.z-image.framework",
    resource_version=None,
    revision=_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
)
_DETECTION = DetectionDeclaration(
    strategy_id="z_image.variant.explicit-v1",
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
_SLICING = SlicingDeclaration(
    supports_step_range=True,
    supports_denoise_tail=True,
    zero_denoise_is_empty=True,
)


def _capabilities(
    variant: ZImageVariant, profile_id: str
) -> tuple[ModelCapabilities, ProfileCapabilities, SamplerCapabilities]:
    model = ModelCapabilities(
        model_family="z_image",
        model_variant=variant.value,
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id=profile_id,
        profile_version="1",
        model_family="z_image",
        model_variant=variant.value,
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
    sampler = SamplerCapabilities(
        sampler_id="flowmatch.euler",
        sampler_version=_GITHUB_REVISION,
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
    return model, profile, sampler


def _weight(
    *, weight_id: str, revision: str, sha256: str, url: str, resource_version: str
) -> ModelWeightProvenance:
    return ModelWeightProvenance(
        record_version="1",
        weight_id=weight_id,
        resource_version=resource_version,
        revision=revision,
        sha256=sha256,
        url=url,
        license=_APACHE_2,
    )


_BASE_WEIGHTS = (
    _weight(
        weight_id="tongyi-mai.z-image.base.transformer-00001",
        revision=_BASE_HF_REVISION,
        sha256="ecd5df7768856215812af84d5cc785ba9d84c6d8c84e6939c7ccf30a2e0d2425",  # pragma: allowlist secret
        url="https://huggingface.co/Tongyi-MAI/Z-Image",
        resource_version="transformer-00001-of-00002",
    ),
    _weight(
        weight_id="tongyi-mai.z-image.base.transformer-00002",
        revision=_BASE_HF_REVISION,
        sha256="e2d5fb75ca504b2d669a33af9380b68e0c1632ecddbfca85f82631958a4b81b1",  # pragma: allowlist secret
        url="https://huggingface.co/Tongyi-MAI/Z-Image",
        resource_version="transformer-00002-of-00002",
    ),
    _weight(
        weight_id="tongyi-mai.z-image.base.vae",
        revision=_BASE_HF_REVISION,
        sha256="f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",  # pragma: allowlist secret
        url="https://huggingface.co/Tongyi-MAI/Z-Image",
        resource_version="vae-diffusion-pytorch-model",
    ),
)
_TURBO_WEIGHTS = (
    _weight(
        weight_id="tongyi-mai.z-image.turbo.transformer-00001",
        revision=_TURBO_HF_REVISION,
        sha256="95facd593e2549e8252acb571c653d57f7ddb7f1060d4e81712f152555a88804",  # pragma: allowlist secret
        url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        resource_version="transformer-00001-of-00003",
    ),
    _weight(
        weight_id="tongyi-mai.z-image.turbo.transformer-00002",
        revision=_TURBO_HF_REVISION,
        sha256="a4bbe43ee184a1fb5af4b412d27555f532893bdc3165b1149e304ed82b5d7015",  # pragma: allowlist secret
        url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        resource_version="transformer-00002-of-00003",
    ),
    _weight(
        weight_id="tongyi-mai.z-image.turbo.transformer-00003",
        revision=_TURBO_HF_REVISION,
        sha256="aba4e37a590e63210878160a718d916d80398f4e1f78ab6c9b2b2a00d92769fa",  # pragma: allowlist secret
        url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        resource_version="transformer-00003-of-00003",
    ),
    _weight(
        weight_id="tongyi-mai.z-image.turbo.vae",
        revision=_TURBO_HF_REVISION,
        sha256="f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3",  # pragma: allowlist secret
        url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
        resource_version="vae-diffusion-pytorch-model",
    ),
)


def _references(*, hf_url: str, hf_revision: str) -> tuple[ZImageEvidenceReference, ...]:
    return (
        ZImageEvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=_COMFYUI_REVISION,
            locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
        ),
        ZImageEvidenceReference(
            lane="official_github",
            url="https://github.com/Tongyi-MAI/Z-Image",
            revision=_GITHUB_REVISION,
            locators=("README.md", "src/zimage/pipeline.py", "src/zimage/scheduler.py"),
        ),
        ZImageEvidenceReference(
            lane="official_huggingface",
            url=hf_url,
            revision=hf_revision,
            locators=("README.md", "model_index.json", "scheduler/scheduler_config.json"),
        ),
        ZImageEvidenceReference(
            lane="official_technical_document",
            url="https://tongyi-mai.github.io/Z-Image-blog/",
            revision=_SITE_REVISION,
            locators=("Z-Image Technical Report arXiv:2511.22699 sections 4.3 and 5",),
        ),
    )


def _schema(
    *, variant: ZImageVariant, ratio: float, hf_url: str, weights: tuple[ModelWeightProvenance, ...]
) -> ProfileSchemaV1:
    profile_id = f"z_image.{variant.value}.official"
    model, profile, sampler = _capabilities(variant, profile_id)
    if variant is ZImageVariant.BASE:
        steps = StepRangeDeclaration(
            minimum=28, maximum=50, default=50, reference_steps=(28, 50), allow_modified=True
        )
        guidance = GuidanceDeclaration(
            model_convention="classifier_free_guidance",
            host_convention="comfy.cfg",
            model_value=4.0,
            host_value=4.0,
        )
    else:
        steps = StepRangeDeclaration(
            minimum=1, maximum=None, default=8, reference_steps=(8,), allow_modified=True
        )
        guidance = GuidanceDeclaration(
            model_convention="embedded_guidance",
            host_convention="comfy.cfg",
            model_value=0.0,
            host_value=1.0,
        )
    limitations: tuple[str, ...] = (
        "Automatic Base/Turbo detection is intentionally unsupported; select the variant explicitly.",
        "ComfyUI sampler execution is a separate host recipe and is not publisher Euler parity.",
    )
    if variant is ZImageVariant.BASE:
        limitations += (
            "Pinned ComfyUI uses ratio 3 for Base while the publisher scheduler config uses ratio 6.",
        )
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        profile_version="1",
        display_name=f"Z-Image {variant.value.title()} Official",
        model_family="z_image",
        model_variant=variant.value,
        evidence=EvidenceLevel.OFFICIAL,
        primary_source_id=_OFFICIAL_SOURCE.source_id,
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        base_grid=BaseGridDeclaration(
            identifier="flowmatch.reciprocal_step",
            output_domain=SigmaDomain.UNIT_FLOW,
            terminal_included=False,
        ),
        transforms=(
            TransformDeclaration(
                identifier="flowmatch.direct_ratio",
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
        terminal=TerminalDeclaration(
            policy=TerminalPolicy.APPEND_ZERO, sigma=TerminalSigma.ZERO, value=0.0
        ),
        slicing=_SLICING,
        recipes=(
            InferenceRecipe(
                recipe_id=f"z_image.{variant.value}.official",
                evidence=EvidenceLevel.OFFICIAL,
                source_id=_OFFICIAL_SOURCE.source_id,
                steps=steps,
                guidance=guidance,
            ),
        ),
        detection=_DETECTION,
        model_capabilities=model,
        profile_capabilities=profile,
        reference_sampler_capabilities=sampler,
        artifact_versions=_ARTIFACT_VERSIONS,
        software_sources=(_OFFICIAL_SOURCE,),
        frameworks=(_COMFYUI_FRAMEWORK,),
        model_weights=weights,
        parameters=(
            ProfileField(name="dynamic_shifting", value=False),
            ProfileField(name="scheduler_train_timesteps", value=1000),
        ),
        known_limitations=limitations,
    )


Z_IMAGE_BASE_SCHEMA: Final = _schema(
    variant=ZImageVariant.BASE,
    ratio=6.0,
    hf_url="https://huggingface.co/Tongyi-MAI/Z-Image",
    weights=_BASE_WEIGHTS,
)
Z_IMAGE_TURBO_SCHEMA: Final = _schema(
    variant=ZImageVariant.TURBO,
    ratio=3.0,
    hf_url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
    weights=_TURBO_WEIGHTS,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ZImageProfile:
    """Immutable released inference schedule for one explicit Z-Image variant."""

    variant: ZImageVariant
    fixed_shift_ratio: float
    minimum_official_steps: int
    maximum_official_steps: int
    default_steps: int
    huggingface_url: str
    huggingface_revision: str
    references: tuple[ZImageEvidenceReference, ...]
    schema: ProfileSchemaV1
    dynamic_shifting: bool = False

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    @property
    def evidence(self) -> EvidenceLevel:
        return self.schema.evidence

    def __post_init__(self) -> None:
        if self.fixed_shift_ratio <= 0.0 or self.dynamic_shifting:
            raise ScheduleContractError("Z-Image released profiles require a fixed positive ratio")
        if not (
            0 < self.minimum_official_steps <= self.default_steps <= self.maximum_official_steps
        ):
            raise ScheduleContractError("Z-Image official step bounds are inconsistent")
        lanes = tuple(reference.lane for reference in self.references)
        if lanes != tuple(sorted(lanes)) or len(lanes) != 4 or len(set(lanes)) != 4:
            raise ScheduleContractError(
                "Z-Image profiles require all four canonical evidence lanes"
            )
        if self.schema.model_variant != self.variant.value:
            raise ScheduleContractError("Z-Image profile and schema variants disagree")


Z_IMAGE_BASE_PROFILE: Final = ZImageProfile(
    variant=ZImageVariant.BASE,
    fixed_shift_ratio=6.0,
    minimum_official_steps=28,
    maximum_official_steps=50,
    default_steps=50,
    huggingface_url="https://huggingface.co/Tongyi-MAI/Z-Image",
    huggingface_revision=_BASE_HF_REVISION,
    references=_references(
        hf_url="https://huggingface.co/Tongyi-MAI/Z-Image", hf_revision=_BASE_HF_REVISION
    ),
    schema=Z_IMAGE_BASE_SCHEMA,
)
Z_IMAGE_TURBO_PROFILE: Final = ZImageProfile(
    variant=ZImageVariant.TURBO,
    fixed_shift_ratio=3.0,
    minimum_official_steps=8,
    maximum_official_steps=8,
    default_steps=8,
    huggingface_url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
    huggingface_revision=_TURBO_HF_REVISION,
    references=_references(
        hf_url="https://huggingface.co/Tongyi-MAI/Z-Image-Turbo", hf_revision=_TURBO_HF_REVISION
    ),
    schema=Z_IMAGE_TURBO_SCHEMA,
)


def _profile(variant: object) -> ZImageProfile:
    if not isinstance(variant, ZImageVariant):
        raise ScheduleContractError("variant must be a ZImageVariant")
    return Z_IMAGE_BASE_PROFILE if variant is ZImageVariant.BASE else Z_IMAGE_TURBO_PROFILE


def build_z_image_schedule(
    *, variant: ZImageVariant, steps: int, strict_official: bool = False
) -> ScheduleResult:
    """Build one complete fixed-ratio Z-Image schedule without framework imports."""

    profile = _profile(variant)
    inputs = ScheduleInputs(steps=steps)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")
    official = profile.minimum_official_steps <= steps <= profile.maximum_official_steps
    if strict_official and not official:
        raise ScheduleContractError("steps are outside the selected official Z-Image recipe")
    evidence = EvidenceLevel.OFFICIAL if official else EvidenceLevel.MODIFIED
    warning = () if official else ("steps are outside the published recipe; evidence is modified",)
    transforms = (
        TransformContract(
            name="flowmatch.direct_ratio",
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
        requested_inputs=inputs,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=evidence,
            source="https://github.com/Tongyi-MAI/Z-Image",
            source_revision=_GITHUB_REVISION,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
        ),
        base_grid=BaseGridSpec(
            identifier="flowmatch.reciprocal_step", output_domain=SigmaDomain.UNIT_FLOW
        ),
        transforms=transforms,
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(),
    )
    sigmas = apply_terminal_policy(
        direct_ratio_shift(flowmatch_reciprocal_step_grid(steps), ratio=profile.fixed_shift_ratio),
        policy=TerminalPolicy.APPEND_ZERO,
        domain=SigmaDomain.UNIT_FLOW,
    )
    return ScheduleResult(
        request=request,
        effective_inputs=inputs,
        sigmas=validate_sigma_schedule(
            sigmas,
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=steps,
            require_terminal_zero=True,
        ),
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=warning,
    )
