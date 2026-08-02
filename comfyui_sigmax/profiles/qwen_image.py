"""Source-qualified original Qwen Image schedule profiles.

The publisher/ Diffusers resolution-dependent shift and the pinned ComfyUI fixed-shift
surface are deliberately represented as separate profiles. They are not composable.
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
    exponential_mu_shift,
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
_PUBLISHER_REVISION: Final = "6b5e1f5cec987d404be5ac6657db3b9aacb56a89"  # pragma: allowlist secret
_HF_REVISION: Final = "75e0b4be04f60ec59a75f475837eced720f823b6"  # pragma: allowlist secret
_COMFYUI_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_DIFFUSERS_REVISION: Final = "cc92165331e1b20afc1a47e03f63e8f3a930f8cc"  # pragma: allowlist secret
_BASE_IMAGE_SEQ_LEN: Final = 256
_MAX_IMAGE_SEQ_LEN: Final = 4096
_BASE_SHIFT: Final = 0.5
_MAX_SHIFT: Final = 1.15
_FIXED_SHIFT: Final = 1.15
_MAX_STEPS: Final = 10_000
_MAX_IMAGE_SEQ_INPUT: Final = 1_000_000


class QwenImageShiftMode(str, Enum):
    """Non-composable Qwen Image shift ownership modes."""

    COMFY_FIXED = "comfy_fixed"
    DIFFUSERS_DYNAMIC = "diffusers_dynamic"


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenImageEvidenceReference:
    """One pinned source lane for Qwen Image qualification."""

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
            "diffusers_framework",
        }:
            raise ScheduleContractError("Qwen Image evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("Qwen Image evidence URL must use HTTPS")
        if self.lane == "official_technical_document":
            if not _ARXIV_PATTERN.fullmatch(self.revision):
                raise ScheduleContractError("Qwen Image technical revision must be an arXiv ID")
        elif not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("Qwen Image evidence revision must be pinned")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("Qwen Image evidence locators must be sorted and unique")


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
    source_id="qwenlm.qwen-image.official",
    resource_version=None,
    revision=_PUBLISHER_REVISION,
    url="https://github.com/QwenLM/Qwen-Image",
    license=_APACHE_2,
    locators=("LICENSE", "README.md", "src/examples/generate_w_image.py"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.qwen-image.framework",
    resource_version="0.29.0",
    revision=_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=("comfy/model_base.py", "comfy/model_sampling.py", "comfy/supported_models.py"),
)
_DIFFUSERS_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.qwen-image.framework",
    resource_version="0.39.0",
    revision=_DIFFUSERS_REVISION,
    url="https://github.com/huggingface/diffusers",
    license=_APACHE_2,
    locators=(
        "src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py",
        "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    ),
)
_QWEN_TRANSFORMER_SHARD = ModelWeightProvenance(
    record_version="1",
    weight_id="qwen.qwen-image.transformer-shard-01",
    resource_version="transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
    revision=_HF_REVISION,
    sha256="9f33a59093af3abcc2836d4cf4b7bd122c238ca70a26c70f34fdde64646b3bcd",  # pragma: allowlist secret
    url="https://huggingface.co/Qwen/Qwen-Image",
    license=_APACHE_2,
)
_DETECTION = DetectionDeclaration(
    strategy_id="qwen_image.original.explicit-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_mode",),
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
    policy=TerminalPolicy.APPEND_ZERO, sigma=TerminalSigma.ZERO, value=0.0
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
        model_family="qwen_image",
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
        model_family="qwen_image",
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
    mode: QwenImageShiftMode,
    profile_id: str,
    display_name: str,
    evidence: EvidenceLevel,
    primary_source_id: str,
    transform: TransformDeclaration,
    parameters: tuple[ProfileField, ...],
    frameworks: tuple[FrameworkProvenance, ...],
) -> ProfileSchemaV1:
    model, profile = _capabilities(profile_id)
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        profile_version="1",
        display_name=display_name,
        model_family="qwen_image",
        model_variant="original",
        evidence=evidence,
        primary_source_id=primary_source_id,
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        base_grid=_BASE_GRID,
        transforms=(
            transform,
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
                    default=50,
                    reference_steps=(50,),
                    allow_modified=True,
                ),
                guidance=GuidanceDeclaration(
                    model_convention="true_cfg_scale",
                    host_convention="true_cfg_scale",
                    model_value=4.0,
                    host_value=4.0,
                ),
            ),
        ),
        detection=_DETECTION,
        model_capabilities=model,
        profile_capabilities=profile,
        reference_sampler_capabilities=_SAMPLER,
        artifact_versions=_ARTIFACT_VERSIONS,
        software_sources=(_OFFICIAL_SOURCE,),
        frameworks=frameworks,
        model_weights=(_QWEN_TRANSFORMER_SHARD,),
        parameters=parameters,
        known_limitations=(
            "Only original Qwen/Qwen-Image text-to-image is qualified; later Qwen variants are excluded.",
            "Model weights and image quality are not verified by this schedule profile.",
            "The selected shift mode owns the complete primary transform; fixed and dynamic modes cannot be composed.",
        ),
    )


QWEN_IMAGE_COMFY_FIXED_SCHEMA: Final = _schema(
    mode=QwenImageShiftMode.COMFY_FIXED,
    profile_id="qwen_image.comfy-fixed.official",
    display_name="Qwen Image ComfyUI Fixed Shift",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    primary_source_id=_COMFYUI_FRAMEWORK.framework_id,
    transform=TransformDeclaration(
        identifier="comfy.direct_ratio",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=SigmaDomain.UNIT_FLOW,
        output_domain=SigmaDomain.UNIT_FLOW,
        parameters=(ProfileField(name="ratio", value=_FIXED_SHIFT),),
    ),
    parameters=(ProfileField(name="shift", value=_FIXED_SHIFT),),
    frameworks=(_COMFYUI_FRAMEWORK, _DIFFUSERS_FRAMEWORK),
)
QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA: Final = _schema(
    mode=QwenImageShiftMode.DIFFUSERS_DYNAMIC,
    profile_id="qwen_image.diffusers-dynamic.framework-reference",
    display_name="Qwen Image Diffusers Dynamic Shift",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    primary_source_id=_DIFFUSERS_FRAMEWORK.framework_id,
    transform=TransformDeclaration(
        identifier="diffusers.exponential_mu",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=SigmaDomain.UNIT_FLOW,
        output_domain=SigmaDomain.UNIT_FLOW,
        parameters=(
            ProfileField(name="exponent", value=1.0),
            ProfileField(name="mu", value=None),
            ProfileField(name="time_shift_type", value="exponential"),
        ),
    ),
    parameters=(
        ProfileField(name="base_image_seq_len", value=_BASE_IMAGE_SEQ_LEN),
        ProfileField(name="base_shift", value=_BASE_SHIFT),
        ProfileField(name="max_image_seq_len", value=_MAX_IMAGE_SEQ_LEN),
        ProfileField(name="max_shift", value=_MAX_SHIFT),
        ProfileField(name="use_dynamic_shifting", value=True),
    ),
    frameworks=(_COMFYUI_FRAMEWORK, _DIFFUSERS_FRAMEWORK),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenImageProfile:
    """Immutable Qwen Image schedule profile plus pinned evidence."""

    shift_mode: QwenImageShiftMode
    schema: ProfileSchemaV1
    references: tuple[QwenImageEvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        if not isinstance(self.shift_mode, QwenImageShiftMode):
            raise ScheduleContractError("Qwen Image shift mode is unsupported")
        if not isinstance(self.schema, ProfileSchemaV1):
            raise ScheduleContractError("Qwen Image schema is required")
        lanes = tuple(reference.lane for reference in self.references)
        if len(lanes) != 5 or len(set(lanes)) != 5 or lanes != tuple(sorted(lanes)):
            raise ScheduleContractError("Qwen Image requires five pinned evidence lanes")
        if (
            self.shift_mode is QwenImageShiftMode.COMFY_FIXED
            and self.schema is not QWEN_IMAGE_COMFY_FIXED_SCHEMA
        ):
            raise ScheduleContractError("fixed Qwen Image profile/schema mismatch")
        if (
            self.shift_mode is QwenImageShiftMode.DIFFUSERS_DYNAMIC
            and self.schema is not QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA
        ):
            raise ScheduleContractError("dynamic Qwen Image profile/schema mismatch")


def _references() -> tuple[QwenImageEvidenceReference, ...]:
    return (
        QwenImageEvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=_COMFYUI_REVISION,
            locators=(
                "comfy/model_base.py",
                "comfy/model_sampling.py",
                "comfy/supported_models.py",
            ),
        ),
        QwenImageEvidenceReference(
            lane="diffusers_framework",
            url="https://github.com/huggingface/diffusers",
            revision=_DIFFUSERS_REVISION,
            locators=(
                "src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py",
                "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
            ),
        ),
        QwenImageEvidenceReference(
            lane="official_github",
            url="https://github.com/QwenLM/Qwen-Image",
            revision=_PUBLISHER_REVISION,
            locators=("LICENSE", "README.md"),
        ),
        QwenImageEvidenceReference(
            lane="official_huggingface",
            url="https://huggingface.co/Qwen/Qwen-Image",
            revision=_HF_REVISION,
            locators=("README.md",),
        ),
        QwenImageEvidenceReference(
            lane="official_technical_document",
            url="https://arxiv.org/abs/2508.02324",
            revision="arxiv:2508.02324",
            locators=("Qwen-Image Technical Report",),
        ),
    )


QWEN_IMAGE_COMFY_FIXED_PROFILE: Final = QwenImageProfile(
    shift_mode=QwenImageShiftMode.COMFY_FIXED,
    schema=QWEN_IMAGE_COMFY_FIXED_SCHEMA,
    references=_references(),
)
QWEN_IMAGE_DIFFUSERS_DYNAMIC_PROFILE: Final = QwenImageProfile(
    shift_mode=QwenImageShiftMode.DIFFUSERS_DYNAMIC,
    schema=QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA,
    references=_references(),
)


def _profile(mode: QwenImageShiftMode) -> QwenImageProfile:
    if mode is QwenImageShiftMode.COMFY_FIXED:
        return QWEN_IMAGE_COMFY_FIXED_PROFILE
    if mode is QwenImageShiftMode.DIFFUSERS_DYNAMIC:
        return QWEN_IMAGE_DIFFUSERS_DYNAMIC_PROFILE
    raise ScheduleContractError("mode must be Comfy Fixed or Diffusers Dynamic")


def calculate_qwen_image_mu(image_seq_len: int) -> float:
    """Match Diffusers Qwen Image's linear resolution-to-mu calculation."""

    if (
        not isinstance(image_seq_len, int)
        or isinstance(image_seq_len, bool)
        or not 1 <= image_seq_len <= _MAX_IMAGE_SEQ_INPUT
    ):
        raise ScheduleContractError(
            f"image_seq_len must be an integer between 1 and {_MAX_IMAGE_SEQ_INPUT}"
        )
    slope = (_MAX_SHIFT - _BASE_SHIFT) / (_MAX_IMAGE_SEQ_LEN - _BASE_IMAGE_SEQ_LEN)
    return image_seq_len * slope + (_BASE_SHIFT - slope * _BASE_IMAGE_SEQ_LEN)


def build_qwen_image_schedule(
    *,
    mode: QwenImageShiftMode,
    steps: int,
    image_seq_len: int | None,
    strict_official: bool = False,
) -> ScheduleResult:
    """Build one explicit original-Qwen unit-flow schedule without framework imports."""

    profile = _profile(mode)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    if mode is QwenImageShiftMode.COMFY_FIXED:
        if image_seq_len not in (None, 0):
            raise ScheduleContractError("image_seq_len must be omitted for Comfy Fixed mode")
        mu = None
    else:
        if image_seq_len is None or image_seq_len == 0:
            raise ScheduleContractError("image_seq_len is required for Diffusers Dynamic mode")
        mu = calculate_qwen_image_mu(image_seq_len)
    if strict_official and steps != 50:
        raise ScheduleContractError("steps are outside the published 50-step Qwen Image recipe")

    evidence = profile.schema.evidence if steps == 50 else EvidenceLevel.MODIFIED
    warnings = (
        ()
        if steps == 50
        else ("steps differ from the published 50-step recipe; evidence is modified",)
    )
    base = flowmatch_reciprocal_step_grid(steps)
    if mode is QwenImageShiftMode.COMFY_FIXED:
        shifted = direct_ratio_shift(base, ratio=_FIXED_SHIFT)
        shift_name = "comfy.direct_ratio"
    else:
        if mu is None:
            raise ScheduleContractError("dynamic shift requires a calculated mu")
        shifted = exponential_mu_shift(base, mu=mu, exponent=1.0)
        shift_name = "diffusers.exponential_mu"
    transforms = (
        TransformContract(
            name=shift_name,
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
            source=(
                "https://github.com/Comfy-Org/ComfyUI"
                if mode is QwenImageShiftMode.COMFY_FIXED
                else "https://github.com/huggingface/diffusers"
            ),
            source_revision=(
                _COMFYUI_REVISION if mode is QwenImageShiftMode.COMFY_FIXED else _DIFFUSERS_REVISION
            ),
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
        shifted, policy=TerminalPolicy.APPEND_ZERO, domain=SigmaDomain.UNIT_FLOW
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
    "QWEN_IMAGE_COMFY_FIXED_PROFILE",
    "QWEN_IMAGE_COMFY_FIXED_SCHEMA",
    "QWEN_IMAGE_DIFFUSERS_DYNAMIC_PROFILE",
    "QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA",
    "QwenImageEvidenceReference",
    "QwenImageProfile",
    "QwenImageShiftMode",
    "build_qwen_image_schedule",
    "calculate_qwen_image_mu",
]
