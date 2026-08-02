"""Source-qualified LTXV/LTX-2/LTX-2.3 schedule profiles.

The module intentionally exposes only schedule construction.  It does not load LTX weights,
execute audio/video models, copy LTX implementation code, or replace a host sampler.  Adaptive
profiles share the public token-derived FlowMatch equation; distilled profiles are immutable
publisher vectors and never enter the adaptive transform path.
"""

from __future__ import annotations

import hashlib
import math
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
    exponential_mu_shift,
    linear_endpoint_grid,
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
_MAX_STEPS: Final = 10_000
_MAX_TOKENS: Final = 16_777_216
_BASE_TOKENS: Final = 1024
_MAX_ANCHOR_TOKENS: Final = 4096
_BASE_SHIFT: Final = 0.95
_MAX_SHIFT: Final = 2.05
_DEFAULT_TOKENS: Final = 4096
_DEFAULT_TERMINAL: Final = 0.1

LTXV_REPOSITORY_REVISION: Final = (
    "4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"  # pragma: allowlist secret
)
LTXV_MODEL_REVISION: Final = "8984fa25007f376c1a299016d0957a37a2f797bb"  # pragma: allowlist secret
LTX2_REPOSITORY_REVISION: Final = (
    "9377758131b1ffde4b7f766804590a6617bf2ab9"  # pragma: allowlist secret
)
LTX2_MODEL_REVISION: Final = "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"  # pragma: allowlist secret
LTX23_MODEL_REVISION: Final = "7caa482d5cd10a2eae6b34cb48f093ebc45a263e"  # pragma: allowlist secret
LTX_COMFYUI_REVISION: Final = "5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25"  # pragma: allowlist secret
LTX_DIFFUSERS_REVISION: Final = (
    "3c468926ffd12b69baa4316e27b09306b8da19a6"  # pragma: allowlist secret
)


class LTXGeneration(str, Enum):
    """Explicit LTX generation identity."""

    LTXV_098 = "ltxv-0.9.8"
    LTX2_19B = "ltx2-19b"
    LTX23_22B = "ltx2.3-22b"


class LTXStage(str, Enum):
    """Schedule stage identity; stage vectors are not interchangeable."""

    DEV = "dev"
    DISTILLED_STAGE1 = "distilled.stage1"
    DISTILLED_STAGE2 = "distilled.stage2"


class LTXProfileId(str, Enum):
    """Stable public LTX profile keys."""

    LTXV_098_DEV = "ltxv.0.9.8.dev"
    LTX2_19B_DEV = "ltx2.19b.dev"
    LTX2_19B_DISTILLED_STAGE1 = "ltx2.19b.distilled.stage1"
    LTX2_19B_DISTILLED_STAGE2 = "ltx2.19b.distilled.stage2"
    LTX23_22B_DEV = "ltx2.3.22b.dev"
    LTX23_22B_DISTILLED_STAGE1 = "ltx2.3.22b.distilled.stage1"
    LTX23_22B_DISTILLED_STAGE2 = "ltx2.3.22b.distilled.stage2"


@dataclass(frozen=True, slots=True, kw_only=True)
class LTXEvidenceReference:
    """One pinned source lane for an LTX profile."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lane not in {"comfyui_implementation", "diffusers_framework", "official_publisher"}:
            raise ScheduleContractError("LTX evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("LTX evidence URL must use HTTPS")
        if not isinstance(self.revision, str) or not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("LTX evidence revision is invalid")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("LTX evidence locators must be sorted and unique")


_APACHE_2 = LicenseDeclaration(
    declaration_version="1",
    identifier="Apache-2.0",
    name="Apache License 2.0",
    url="https://www.apache.org/licenses/LICENSE-2.0",
)
_LTX2_COMMUNITY = LicenseDeclaration(
    declaration_version="1",
    identifier="LicenseRef-LTX-2-Community",
    name="LTX-2 Community License Agreement",
    url="https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
)
_GPL_3_ONLY = LicenseDeclaration(
    declaration_version="1",
    identifier="GPL-3.0-only",
    name="GNU General Public License v3.0 only",
    url="https://www.gnu.org/licenses/gpl-3.0.html",
)
_OFFICIAL_URLS: Final = {
    LTXGeneration.LTXV_098: "https://github.com/Lightricks/LTX-Video",
    LTXGeneration.LTX2_19B: "https://github.com/Lightricks/LTX-2",
    LTXGeneration.LTX23_22B: "https://huggingface.co/Lightricks/LTX-2.3",
}

_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.ltx.framework",
    resource_version="0.29.0",
    revision=LTX_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=("comfy/model_sampling.py", "comfy/supported_models.py", "comfy_extras/nodes_lt.py"),
)
_DIFFUSERS_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.ltx.framework",
    resource_version="0.39.0.dev0",
    revision=LTX_DIFFUSERS_REVISION,
    url="https://github.com/huggingface/diffusers",
    license=_APACHE_2,
    locators=("src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",),
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
_DETECTION = DetectionDeclaration(
    strategy_id="ltx.generation-stage.explicit-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_generation", "explicit_stage"),
    suggestion_sources=(),
    family_only_sources=(),
)


def _official_source(generation: LTXGeneration) -> SoftwareSourceProvenance:
    if generation is LTXGeneration.LTXV_098:
        source_id = "lightricks.ltx-video.official"
        revision = LTXV_REPOSITORY_REVISION
        version = "0.9.8"
        locators = ("LICENSE", "README.md", "ltx_video/schedulers/rf.py")
        license_declaration = _APACHE_2
    elif generation is LTXGeneration.LTX2_19B:
        source_id = "lightricks.ltx2.official"
        revision = LTX2_REPOSITORY_REVISION
        version = "2.0"
        locators = ("LICENSE", "README.md", "ltx_core/schedulers")
        license_declaration = _LTX2_COMMUNITY
    else:
        source_id = "lightricks.ltx2-3.official"
        revision = LTX23_MODEL_REVISION
        version = "2.3"
        locators = ("LICENSE", "README.md", "scheduler")
        license_declaration = _LTX2_COMMUNITY
    return SoftwareSourceProvenance(
        record_version="1",
        source_id=source_id,
        resource_version=version,
        revision=revision,
        url=_OFFICIAL_URLS[generation],
        license=license_declaration,
        locators=locators,
    )


def _references(generation: LTXGeneration) -> tuple[LTXEvidenceReference, ...]:
    official = _official_source(generation)
    official_reference = LTXEvidenceReference(
        lane="official_publisher",
        url=official.url,
        revision=official.revision,
        locators=official.locators,
    )
    comfy_reference = LTXEvidenceReference(
        lane="comfyui_implementation",
        url="https://github.com/Comfy-Org/ComfyUI",
        revision=LTX_COMFYUI_REVISION,
        locators=_COMFYUI_FRAMEWORK.locators,
    )
    diffusers_reference = LTXEvidenceReference(
        lane="diffusers_framework",
        url="https://github.com/huggingface/diffusers",
        revision=LTX_DIFFUSERS_REVISION,
        locators=_DIFFUSERS_FRAMEWORK.locators,
    )
    return tuple(
        sorted(
            (comfy_reference, diffusers_reference, official_reference), key=lambda item: item.lane
        )
    )


def _contract_digest(profile_id: str, revision: str) -> str:
    """Digest the public profile contract; no model payload is packaged or represented."""

    return hashlib.sha256(f"{profile_id}@{revision}".encode()).hexdigest()


def _weight(generation: LTXGeneration, profile_id: str) -> ModelWeightProvenance:
    official = _official_source(generation)
    license_declaration = _APACHE_2 if generation is LTXGeneration.LTXV_098 else _LTX2_COMMUNITY
    return ModelWeightProvenance(
        record_version="1",
        weight_id=f"{profile_id}.not-packaged-contract",
        resource_version="model-card-only-no-payload",
        revision=official.revision,
        sha256=_contract_digest(profile_id, official.revision),
        url=official.url,
        license=license_declaration,
    )


def _capabilities(
    profile_id: str, generation: LTXGeneration
) -> tuple[ModelCapabilities, ProfileCapabilities, SamplerCapabilities]:
    model_variant = generation.value.replace(".", "-")
    model = ModelCapabilities(
        model_family="ltx",
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
        model_family="ltx",
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
    sampler = SamplerCapabilities(
        sampler_id="flowmatch.euler",
        sampler_version=LTX_COMFYUI_REVISION,
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


def _schema(
    *,
    generation: LTXGeneration,
    stage: LTXStage,
    profile_id: LTXProfileId,
    default_steps: int,
    min_steps: int,
    max_steps: int,
) -> ProfileSchemaV1:
    profile_key = profile_id.value
    official = _official_source(generation)
    model, profile_caps, sampler = _capabilities(profile_key, generation)
    distilled = stage is not LTXStage.DEV
    transforms: tuple[TransformDeclaration, ...]
    if distilled:
        transforms = (
            TransformDeclaration(
                identifier="ltx.distilled.preset",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
            TransformDeclaration(
                identifier="terminal.append_zero",
                stage=TransformStage.TERMINAL,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
        )
    else:
        transforms = (
            TransformDeclaration(
                identifier="ltx.flow.exponential_token_shift",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
                parameters=(
                    ProfileField(name="base_shift", value=_BASE_SHIFT),
                    ProfileField(name="max_shift", value=_MAX_SHIFT),
                    ProfileField(name="sequence_max", value=_MAX_ANCHOR_TOKENS),
                    ProfileField(name="sequence_min", value=_BASE_TOKENS),
                ),
            ),
            TransformDeclaration(
                identifier="ltx.terminal.stretch",
                stage=TransformStage.OPTIONAL_SPACING,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
                parameters=(ProfileField(name="terminal", value=_DEFAULT_TERMINAL),),
            ),
            TransformDeclaration(
                identifier="terminal.append_zero",
                stage=TransformStage.TERMINAL,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
        )
    recipe_steps = StepRangeDeclaration(
        minimum=min_steps,
        maximum=max_steps,
        default=default_steps,
        reference_steps=(default_steps,),
        allow_modified=not distilled,
    )
    guidance = GuidanceDeclaration(
        model_convention="classifier_free_guidance",
        host_convention="comfy.cfg",
        model_value=1.0 if distilled else 3.0,
        host_value=1.0 if distilled else 3.0,
    )
    model_variant = generation.value.replace(".", "-")
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_key,
        profile_version="1",
        display_name=f"{generation.value} {stage.value}",
        model_family="ltx",
        model_variant=model_variant,
        evidence=EvidenceLevel.OFFICIAL,
        primary_source_id=official.source_id,
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        base_grid=BaseGridDeclaration(
            identifier="flowmatch.linear_endpoint",
            output_domain=SigmaDomain.UNIT_FLOW,
            terminal_included=False,
            parameters=(ProfileField(name="terminal_included", value=False),),
        ),
        transforms=transforms,
        terminal=TerminalDeclaration(
            policy=TerminalPolicy.APPEND_ZERO,
            sigma=TerminalSigma.ZERO,
            value=0.0,
        ),
        slicing=_SLICING,
        recipes=(
            InferenceRecipe(
                recipe_id=profile_key,
                evidence=EvidenceLevel.OFFICIAL,
                source_id=official.source_id,
                steps=recipe_steps,
                guidance=guidance,
            ),
        ),
        detection=_DETECTION,
        model_capabilities=model,
        profile_capabilities=profile_caps,
        reference_sampler_capabilities=sampler,
        artifact_versions=_ARTIFACT_VERSIONS,
        software_sources=(official,),
        frameworks=(_COMFYUI_FRAMEWORK, _DIFFUSERS_FRAMEWORK),
        model_weights=(_weight(generation, profile_key),),
        parameters=(
            ProfileField(name="dynamic_shifting", value=not distilled),
            ProfileField(name="sequence_anchor_max", value=_MAX_ANCHOR_TOKENS),
            ProfileField(name="sequence_anchor_min", value=_BASE_TOKENS),
            ProfileField(name="terminal", value=_DEFAULT_TERMINAL),
        ),
        known_limitations=(
            "Generation and stage must be selected explicitly; automatic checkpoint detection is unsupported.",
            "This node constructs SIGMAS only and does not load weights or execute audio/video models.",
            "Distilled vectors are immutable presets and are not adaptive-shift or sampler parity claims.",
        ),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class LTXProfile:
    """Immutable schedule metadata for one LTX generation/stage."""

    generation: LTXGeneration
    stage: LTXStage
    profile_id: LTXProfileId
    default_steps: int
    minimum_steps: int
    maximum_steps: int
    references: tuple[LTXEvidenceReference, ...]
    schema: ProfileSchemaV1
    distilled_vector: tuple[float, ...] | None = None

    @property
    def profile_key(self) -> str:
        return self.profile_id.value

    @property
    def evidence(self) -> EvidenceLevel:
        return self.schema.evidence

    def __post_init__(self) -> None:
        if self.schema.profile_id != self.profile_key:
            raise ScheduleContractError("LTX profile/schema IDs disagree")
        if self.schema.model_family != "ltx":
            raise ScheduleContractError("LTX profile schema family is invalid")
        if len(self.references) != 3 or tuple(
            reference.lane for reference in self.references
        ) != tuple(sorted(reference.lane for reference in self.references)):
            raise ScheduleContractError("LTX profiles require three canonical evidence lanes")
        if not 0 < self.minimum_steps <= self.default_steps <= self.maximum_steps <= _MAX_STEPS:
            raise ScheduleContractError("LTX profile step bounds are inconsistent")
        if self.stage is LTXStage.DEV and self.distilled_vector is not None:
            raise ScheduleContractError("adaptive LTX profiles cannot carry a distilled vector")
        if self.stage is not LTXStage.DEV:
            if self.distilled_vector is None or len(self.distilled_vector) < 2:
                raise ScheduleContractError("distilled LTX profiles require an immutable vector")
            validate_sigma_schedule(
                self.distilled_vector,
                domain=SigmaDomain.UNIT_FLOW,
                expected_steps=len(self.distilled_vector) - 1,
                require_terminal_zero=True,
            )


_STAGE1_VECTOR: Final = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
_STAGE2_VECTOR: Final = (0.909375, 0.725, 0.421875, 0.0)


def _profile(
    *,
    generation: LTXGeneration,
    stage: LTXStage,
    profile_id: LTXProfileId,
    default_steps: int,
    minimum_steps: int,
    maximum_steps: int,
    vector: tuple[float, ...] | None = None,
) -> LTXProfile:
    return LTXProfile(
        generation=generation,
        stage=stage,
        profile_id=profile_id,
        default_steps=default_steps,
        minimum_steps=minimum_steps,
        maximum_steps=maximum_steps,
        references=_references(generation),
        schema=_schema(
            generation=generation,
            stage=stage,
            profile_id=profile_id,
            default_steps=default_steps,
            min_steps=minimum_steps,
            max_steps=maximum_steps,
        ),
        distilled_vector=vector,
    )


LTXV_098_PROFILE: Final = _profile(
    generation=LTXGeneration.LTXV_098,
    stage=LTXStage.DEV,
    profile_id=LTXProfileId.LTXV_098_DEV,
    default_steps=20,
    minimum_steps=1,
    maximum_steps=_MAX_STEPS,
)
LTX2_19B_PROFILE: Final = _profile(
    generation=LTXGeneration.LTX2_19B,
    stage=LTXStage.DEV,
    profile_id=LTXProfileId.LTX2_19B_DEV,
    default_steps=40,
    minimum_steps=1,
    maximum_steps=_MAX_STEPS,
)
LTX2_19B_DISTILLED_STAGE1_PROFILE: Final = _profile(
    generation=LTXGeneration.LTX2_19B,
    stage=LTXStage.DISTILLED_STAGE1,
    profile_id=LTXProfileId.LTX2_19B_DISTILLED_STAGE1,
    default_steps=8,
    minimum_steps=8,
    maximum_steps=8,
    vector=_STAGE1_VECTOR,
)
LTX2_19B_DISTILLED_STAGE2_PROFILE: Final = _profile(
    generation=LTXGeneration.LTX2_19B,
    stage=LTXStage.DISTILLED_STAGE2,
    profile_id=LTXProfileId.LTX2_19B_DISTILLED_STAGE2,
    default_steps=3,
    minimum_steps=3,
    maximum_steps=3,
    vector=_STAGE2_VECTOR,
)
LTX23_22B_PROFILE: Final = _profile(
    generation=LTXGeneration.LTX23_22B,
    stage=LTXStage.DEV,
    profile_id=LTXProfileId.LTX23_22B_DEV,
    default_steps=30,
    minimum_steps=1,
    maximum_steps=_MAX_STEPS,
)
LTX23_22B_DISTILLED_STAGE1_PROFILE: Final = _profile(
    generation=LTXGeneration.LTX23_22B,
    stage=LTXStage.DISTILLED_STAGE1,
    profile_id=LTXProfileId.LTX23_22B_DISTILLED_STAGE1,
    default_steps=8,
    minimum_steps=8,
    maximum_steps=8,
    vector=_STAGE1_VECTOR,
)
LTX23_22B_DISTILLED_STAGE2_PROFILE: Final = _profile(
    generation=LTXGeneration.LTX23_22B,
    stage=LTXStage.DISTILLED_STAGE2,
    profile_id=LTXProfileId.LTX23_22B_DISTILLED_STAGE2,
    default_steps=3,
    minimum_steps=3,
    maximum_steps=3,
    vector=_STAGE2_VECTOR,
)

LTX_PROFILES: Final = (
    LTXV_098_PROFILE,
    LTX2_19B_PROFILE,
    LTX2_19B_DISTILLED_STAGE1_PROFILE,
    LTX2_19B_DISTILLED_STAGE2_PROFILE,
    LTX23_22B_PROFILE,
    LTX23_22B_DISTILLED_STAGE1_PROFILE,
    LTX23_22B_DISTILLED_STAGE2_PROFILE,
)
_PROFILE_MAP: Final = {profile.profile_id: profile for profile in LTX_PROFILES}


def derive_ltx_shift(
    token_count: int,
    *,
    base_tokens: int = _BASE_TOKENS,
    max_tokens: int = _MAX_ANCHOR_TOKENS,
    base_shift: float = _BASE_SHIFT,
    max_shift: float = _MAX_SHIFT,
) -> float:
    """Derive the explicit linear token-to-shift value used by the adaptive LTX path."""

    values = (token_count, base_tokens, max_tokens)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ScheduleContractError("LTX token anchors must be integers")
    if not 1 <= token_count <= _MAX_TOKENS:
        raise ScheduleContractError("token_count must be between 1 and 16777216")
    if not 1 <= base_tokens < max_tokens <= _MAX_TOKENS:
        raise ScheduleContractError("LTX token anchors must be positive and strictly ordered")
    if (
        isinstance(base_shift, bool)
        or isinstance(max_shift, bool)
        or not isinstance(base_shift, (int, float))
        or not isinstance(max_shift, (int, float))
        or not math.isfinite(float(base_shift))
        or not math.isfinite(float(max_shift))
        or float(base_shift) <= 0.0
        or float(max_shift) <= 0.0
    ):
        raise ScheduleContractError("LTX shifts must be finite positive numbers")
    shift = float(base_shift) + (token_count - base_tokens) * (
        float(max_shift) - float(base_shift)
    ) / (max_tokens - base_tokens)
    if not math.isfinite(shift) or shift <= 0.0:
        raise ScheduleContractError("derived LTX shift must be finite and positive")
    return shift


def _profile_for(profile: object) -> LTXProfile:
    if not isinstance(profile, LTXProfileId):
        raise ScheduleContractError("profile must be an LTXProfileId")
    return _PROFILE_MAP[profile]


def _stretch_nonzero(values: tuple[float, ...], *, terminal: float) -> tuple[float, ...]:
    if not 0.0 < terminal < 1.0:
        raise ScheduleContractError("terminal must be in (0, 1) when LTX stretching is enabled")
    if len(values) < 2 or values[-1] != 0.0:
        raise ScheduleContractError(
            "LTX adaptive schedule must have a final zero before stretching"
        )
    nonzero = values[:-1]
    if not nonzero or nonzero[-1] <= 0.0:
        raise ScheduleContractError("LTX schedule has no positive nonzero terminal")
    scale = (1.0 - nonzero[-1]) / (1.0 - terminal)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ScheduleContractError("LTX terminal stretch scale is invalid")
    stretched = tuple(1.0 - (1.0 - value) / scale for value in nonzero)
    return (*stretched, 0.0)


def _adaptive_schedule(
    *,
    profile: LTXProfile,
    steps: int,
    token_count: int,
    stretch: bool,
    terminal: float,
) -> tuple[tuple[float, ...], float]:
    base = linear_endpoint_grid(
        points=steps + 1,
        start=1.0,
        end=0.0,
        domain=SigmaDomain.UNIT_FLOW,
    )
    shift = derive_ltx_shift(token_count)
    shifted = exponential_mu_shift(base, mu=shift, domain=SigmaDomain.UNIT_FLOW)
    if stretch:
        return _stretch_nonzero(shifted, terminal=terminal), shift
    if terminal != _DEFAULT_TERMINAL:
        raise ScheduleContractError("terminal is inert when LTX stretching is disabled")
    return shifted, shift


def build_ltx_schedule(
    *,
    profile: LTXProfileId,
    steps: int,
    token_count: int | None = None,
    stretch: bool = True,
    terminal: float = _DEFAULT_TERMINAL,
    strict_official: bool = False,
) -> ScheduleResult:
    """Build one explicit adaptive or distilled LTX schedule without framework imports."""

    selected = _profile_for(profile)
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    if not isinstance(stretch, bool) or not isinstance(strict_official, bool):
        raise ScheduleContractError("stretch and strict_official must be boolean")
    if (
        isinstance(terminal, bool)
        or not isinstance(terminal, (int, float))
        or not math.isfinite(float(terminal))
    ):
        raise ScheduleContractError("terminal must be a finite number")
    transforms: tuple[TransformContract, ...]
    if selected.stage is not LTXStage.DEV:
        if token_count is not None:
            raise ScheduleContractError("distilled LTX profiles do not accept token_count")
        if selected.distilled_vector is None or steps != len(selected.distilled_vector) - 1:
            raise ScheduleContractError("distilled LTX steps must match the immutable vector")
        values = selected.distilled_vector
        transforms = (
            TransformContract(
                name="ltx.distilled.preset",
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
        warning: tuple[str, ...] = ()
        shift = None
    else:
        effective_tokens = _DEFAULT_TOKENS if token_count is None else token_count
        if not isinstance(effective_tokens, int) or isinstance(effective_tokens, bool):
            raise ScheduleContractError("token_count must be an integer when supplied")
        # Strict mode is the reproducibility lane: only the pinned recipe step count
        # is official.  The bounded range remains available only as an explicit modified lane.
        official = steps == selected.default_steps and stretch and terminal == _DEFAULT_TERMINAL
        if strict_official and not official:
            raise ScheduleContractError("inputs are outside the selected official LTX recipe")
        values, shift = _adaptive_schedule(
            profile=selected,
            steps=steps,
            token_count=effective_tokens,
            stretch=stretch,
            terminal=float(terminal),
        )
        transforms = (
            TransformContract(
                name="ltx.flow.exponential_token_shift",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
            TransformContract(
                name="ltx.terminal.stretch",
                stage=TransformStage.OPTIONAL_SPACING,
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
        warning = (
            ()
            if official
            else ("inputs are outside the published LTX recipe; evidence is modified",)
        )
    inputs = ScheduleInputs(steps=steps)
    evidence = EvidenceLevel.OFFICIAL if not warning else EvidenceLevel.MODIFIED
    official_source = _official_source(selected.generation)
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=inputs,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=evidence,
            source=official_source.url,
            source_revision=official_source.revision,
            profile_id=selected.profile_key,
            profile_version="1",
        ),
        base_grid=BaseGridSpec(
            identifier="flowmatch.linear_endpoint",
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        transforms=transforms,
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(),
    )
    validated = validate_sigma_schedule(
        values,
        domain=SigmaDomain.UNIT_FLOW,
        expected_steps=steps,
        require_terminal_zero=True,
    )
    del profile, shift  # values and provenance are the complete immutable result
    return ScheduleResult(
        request=request,
        effective_inputs=inputs,
        sigmas=validated,
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=warning,
    )
