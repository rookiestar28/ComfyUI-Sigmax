"""Five-source-pinned Lumina-Image 2.0 fixed-shift schedule profile."""

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
_PUBLISHER_REVISION: Final = "4a4d6f856115db07e8ae127280ebcc3d9f65004e"  # pragma: allowlist secret
_HF_REVISION: Final = "53504abd8178b30685b6c4c7a4cd181ff78b73e9"  # pragma: allowlist secret
_COMFYUI_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_DIFFUSERS_REVISION: Final = "3c468926ffd12b69baa4316e27b09306b8da19a6"  # pragma: allowlist secret
_WEIGHT_SHA256: Final = (
    "132b4d213fdd3cfc14333746fc3eb8bbe6358cd73c3bc95ac4ccec230b97dca3"  # pragma: allowlist secret
)
_MAX_STEPS: Final = 10_000
_REFERENCE_STEPS: Final = 50
_SHIFT: Final = 6.0


class Lumina2ShiftMode(str, Enum):
    """The only qualified Lumina-Image 2.0 shift ownership mode."""

    OFFICIAL_FIXED = "official_fixed"


@dataclass(frozen=True, slots=True, kw_only=True)
class Lumina2EvidenceReference:
    """One pinned evidence lane used to qualify Lumina-Image 2.0."""

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
            raise ScheduleContractError("Lumina2 evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("Lumina2 evidence URL must use HTTPS")
        if self.lane == "official_technical_document":
            valid_revision = bool(_ARXIV_PATTERN.fullmatch(self.revision))
        else:
            valid_revision = bool(_COMMIT_PATTERN.fullmatch(self.revision))
        if not valid_revision:
            raise ScheduleContractError("Lumina2 evidence revision is invalid")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("Lumina2 evidence locators must be sorted and unique")


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
    source_id="alphavllm.lumina2.official",
    resource_version="2.0",
    revision=_PUBLISHER_REVISION,
    url="https://github.com/Alpha-VLLM/Lumina-Image-2.0",
    license=_APACHE_2,
    locators=("LICENSE", "README.md", "demo.py"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.lumina2.framework",
    resource_version="0.29.0",
    revision=_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=("comfy/model_base.py", "comfy/model_sampling.py", "comfy/supported_models.py"),
)
_DIFFUSERS_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.lumina2.framework",
    resource_version="0.39.0",
    revision=_DIFFUSERS_REVISION,
    url="https://github.com/huggingface/diffusers",
    license=_APACHE_2,
    locators=("src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",),
)
_LUMINA2_WEIGHT = ModelWeightProvenance(
    record_version="1",
    weight_id="alphavllm.lumina2.transformer-shard-01",
    resource_version="transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    revision=_HF_REVISION,
    sha256=_WEIGHT_SHA256,
    url="https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0",
    license=_APACHE_2,
)
_DETECTION = DetectionDeclaration(
    strategy_id="lumina2.v2.explicit-v1",
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
        model_family="lumina2",
        model_variant="2.0",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id="lumina2.v2.official",
        profile_version="1",
        model_family="lumina2",
        model_variant="2.0",
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

LUMINA2_SCHEMA: Final = ProfileSchemaV1(
    schema_id=PROFILE_SCHEMA_ID,
    schema_version=PROFILE_SCHEMA_VERSION,
    profile_id="lumina2.v2.official",
    profile_version="1",
    display_name="Lumina-Image 2.0 Official Fixed Shift",
    model_family="lumina2",
    model_variant="2.0",
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
            recipe_id="lumina2.v2.official",
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
                model_value=4.0,
                host_value=4.0,
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
    model_weights=(_LUMINA2_WEIGHT,),
    parameters=(
        ProfileField(name="multiplier", value=1.0),
        ProfileField(name="ratio", value=_SHIFT),
        ProfileField(name="source_mode", value=Lumina2ShiftMode.OFFICIAL_FIXED.value),
        ProfileField(name="variant", value="2.0"),
    ),
    known_limitations=(
        "Only the original Alpha-VLLM Lumina-Image 2.0 text-to-image schedule is qualified; Lumina-Video, Lumina-mGPT, and accessory/editing paths are excluded.",
        "The fixed ratio shift owns the complete primary transform; it cannot be composed with another shift or already-shifted sigmas.",
        "Model weights, text encoders, conditioning, sampling execution, and image quality are not verified by this schedule profile.",
        "Dynamic resolution shifting, alternate solvers, and undocumented step/shift recommendations are intentionally absent.",
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Lumina2Profile:
    """Immutable Lumina-Image 2.0 profile plus pinned evidence."""

    shift_mode: Lumina2ShiftMode
    schema: ProfileSchemaV1
    references: tuple[Lumina2EvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        if not isinstance(self.shift_mode, Lumina2ShiftMode):
            raise ScheduleContractError("Lumina2 shift mode is unsupported")
        if self.schema is not LUMINA2_SCHEMA:
            raise ScheduleContractError("Lumina2 profile/schema mismatch")
        lanes = tuple(reference.lane for reference in self.references)
        if lanes != tuple(sorted(set(lanes))) or len(lanes) != 5:
            raise ScheduleContractError("Lumina2 requires five unique pinned evidence lanes")


def _references() -> tuple[Lumina2EvidenceReference, ...]:
    return (
        Lumina2EvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=_COMFYUI_REVISION,
            locators=(
                "comfy/model_base.py",
                "comfy/model_sampling.py",
                "comfy/supported_models.py",
            ),
        ),
        Lumina2EvidenceReference(
            lane="diffusers_framework",
            url="https://github.com/huggingface/diffusers",
            revision=_DIFFUSERS_REVISION,
            locators=("src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",),
        ),
        Lumina2EvidenceReference(
            lane="official_github",
            url="https://github.com/Alpha-VLLM/Lumina-Image-2.0",
            revision=_PUBLISHER_REVISION,
            locators=("LICENSE", "README.md", "demo.py"),
        ),
        Lumina2EvidenceReference(
            lane="official_huggingface",
            url="https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0",
            revision=_HF_REVISION,
            locators=("README.md", "scheduler/scheduler_config.json"),
        ),
        Lumina2EvidenceReference(
            lane="official_technical_document",
            url="https://arxiv.org/abs/2503.21758",
            revision="arxiv:2503.21758",
            locators=("Lumina-Image 2.0", "Unified Next-DiT"),
        ),
    )


LUMINA2_PROFILE: Final = Lumina2Profile(
    shift_mode=Lumina2ShiftMode.OFFICIAL_FIXED,
    schema=LUMINA2_SCHEMA,
    references=_references(),
)


def build_lumina2_schedule(
    *,
    mode: Lumina2ShiftMode,
    steps: int,
    strict_source: bool = False,
    already_shifted: bool = False,
) -> ScheduleResult:
    """Build one explicit Lumina-Image 2.0 unit-flow schedule."""

    if not isinstance(mode, Lumina2ShiftMode):
        raise ScheduleContractError("mode must be the explicit Lumina2 Official Fixed mode")
    if not isinstance(strict_source, bool):
        raise ScheduleContractError("strict_source must be boolean")
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError(
            "already shifted sigmas cannot be composed with the Lumina2 fixed shift"
        )
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    if strict_source and steps != _REFERENCE_STEPS:
        raise ScheduleContractError("steps are outside the official Lumina2 50-step recipe")

    evidence = EvidenceLevel.OFFICIAL if steps == _REFERENCE_STEPS else EvidenceLevel.MODIFIED
    warnings = (
        ()
        if evidence is EvidenceLevel.OFFICIAL
        else ("steps differ from the official Lumina2 50-step recipe; evidence is modified",)
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
            profile_id=LUMINA2_PROFILE.profile_id,
            profile_version=LUMINA2_PROFILE.profile_version,
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
    "LUMINA2_PROFILE",
    "LUMINA2_SCHEMA",
    "Lumina2EvidenceReference",
    "Lumina2Profile",
    "Lumina2ShiftMode",
    "build_lumina2_schedule",
]
