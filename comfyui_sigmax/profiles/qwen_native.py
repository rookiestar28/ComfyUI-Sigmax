"""Explicit Qwen native and 2.1 source lanes; pure math, no host/framework imports.

Legacy qwen_image profiles remain frozen. These identities independently express the reviewed
ComfyUI Flux table and Diffusers 2.1 scheduler specifications through existing Sigmax primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Final, cast

from comfyui_sigmax.core import (
    BaseGridSpec,
    EvidenceLevel,
    Provenance,
    ScheduleContractError,
    ScheduleInputs,
    ScheduleOwnership,
    ScheduleRequest,
    ScheduleResult,
    SigmaDomain,
    SliceSpec,
    TerminalPolicy,
    TransformStage,
    apply_terminal_policy,
    comfyui_simple_discrete_flow_grid,
    exponential_mu_shift,
    flowmatch_reciprocal_step_grid,
    slice_step_range,
    validate_sigma_schedule,
)
from comfyui_sigmax.profiles.qwen_image import QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA
from comfyui_sigmax.profiles.schema_v1 import (
    BaseGridDeclaration,
    DetectionDeclaration,
    FrameworkProvenance,
    GuidanceDeclaration,
    InferenceRecipe,
    LicenseDeclaration,
    ModelWeightProvenance,
    ProfileField,
    ProfileSchemaV1,
    SoftwareSourceProvenance,
    StepRangeDeclaration,
    TransformDeclaration,
)
from comfyui_sigmax.version import VERSION

QWEN_NATIVE_COMFYUI_REVISION: Final = (
    "87c32827017c50c6a629da941439015b4ad656e6"  # pragma: allowlist secret
)
QWEN21_DIFFUSERS_REVISION: Final = (
    "122b1e11fd497c3eeef14b3b98ca26a60166e48b"  # pragma: allowlist secret
)
QWEN21_PUBLISHER_REVISION: Final = (
    "6627d87c6433151463ec4b48b8945a24fcf16a35"  # pragma: allowlist secret
)
QWEN21_MODEL_REVISION: Final = (
    "d26bb61231c349cf6b7896fa83353113880e1ba3"  # pragma: allowlist secret
)
QWEN21_TEMPLATE_REVISION: Final = (
    "9d60ee018d52e461a425e5b9d7fee8d40ca19c1b"  # pragma: allowlist secret
)
_MAX_STEPS: Final = 10_000
_MAX_TOKENS: Final = 1_000_000


class QwenNativeLane(str, Enum):
    """Non-composable, version-specific source identities."""

    ORIGINAL_COMFY = "qwen_image.comfy-native.framework-reference"
    QWEN21_COMFY = "qwen_image21.comfy-native.framework-reference"
    QWEN21_DYNAMIC = "qwen_image21.diffusers-dynamic.framework-reference"


_BASE: Final = QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA
_RESEARCH_LICENSE: Final = LicenseDeclaration(
    declaration_version="1",
    identifier="LicenseRef-Qwen-Research",
    name="Qwen Research License Agreement (noncommercial research or evaluation)",
    url=f"https://github.com/QwenLM/Qwen-Image-2.1/blob/{QWEN21_PUBLISHER_REVISION}/LICENSE",
)
_QWEN21_SOURCE: Final = SoftwareSourceProvenance(
    record_version="1",
    source_id="qwenlm.qwen-image21.official",
    resource_version=None,
    revision=QWEN21_PUBLISHER_REVISION,
    url="https://github.com/QwenLM/Qwen-Image-2.1",
    license=_RESEARCH_LICENSE,
    locators=("LICENSE", "README.md"),
)
_QWEN21_WEIGHT: Final = ModelWeightProvenance(
    record_version="1",
    weight_id="qwen.qwen-image21.transformer-shard-01",
    resource_version="transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    revision=QWEN21_MODEL_REVISION,
    sha256="9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b",  # pragma: allowlist secret
    url="https://huggingface.co/Qwen/Qwen-Image-2.1",
    license=_RESEARCH_LICENSE,
)
_COMFY: Final = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.qwen-native.framework",
    resource_version="0.39.0",
    revision=QWEN_NATIVE_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_BASE.frameworks[0].license,
    locators=("comfy/model_base.py", "comfy/model_sampling.py", "comfy/samplers.py"),
)
_DIFFUSERS: Final = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.qwen-image21.framework",
    resource_version=None,
    revision=QWEN21_DIFFUSERS_REVISION,
    url="https://github.com/huggingface/diffusers",
    license=_BASE.frameworks[1].license,
    locators=(
        "src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py",
        "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenNativeProfile:
    lane: QwenNativeLane
    schema: ProfileSchemaV1

    def __post_init__(self) -> None:
        if not isinstance(self.lane, QwenNativeLane) or not isinstance(
            self.schema, ProfileSchemaV1
        ):
            raise ScheduleContractError("Qwen native profile requires a typed lane and schema")
        if self.schema.profile_id != self.lane.value:
            raise ScheduleContractError("Qwen native lane/profile identity mismatch")


def _profile(lane: QwenNativeLane) -> QwenNativeProfile:
    original = lane is QwenNativeLane.ORIGINAL_COMFY
    dynamic = lane is QwenNativeLane.QWEN21_DYNAMIC
    family, variant = ("qwen_image", "original") if original else ("qwen_image21", "v2_1")
    source = _DIFFUSERS if dynamic else _COMFY
    default_steps = 50 if original else 40 if dynamic else 25
    parameters = (
        (
            ProfileField(name="base_image_seq_len", value=256),
            ProfileField(name="base_shift", value=0.5),
            ProfileField(name="extrapolation", value="unclamped_affine"),
            ProfileField(name="max_image_seq_len", value=8192),
            ProfileField(name="max_shift", value=0.9),
            ProfileField(name="shift_terminal", value=0.02),
            ProfileField(name="target_tokens", value="unpatched_target_only_vae16_aligned32"),
        )
        if dynamic
        else (ProfileField(name="mu", value=1.15 if original else 0.69),)
    )
    shift = TransformDeclaration(
        identifier="qwen.exponential_mu",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=SigmaDomain.UNIT_FLOW,
        output_domain=SigmaDomain.UNIT_FLOW,
        parameters=(ProfileField(name="exponent", value=1.0),),
    )
    spacing = TransformDeclaration(
        identifier="qwen21.stretch_to_terminal",
        stage=TransformStage.OPTIONAL_SPACING,
        input_domain=SigmaDomain.UNIT_FLOW,
        output_domain=SigmaDomain.UNIT_FLOW,
        parameters=(ProfileField(name="shift_terminal", value=0.02),),
    )
    schema = replace(
        _BASE,
        profile_id=lane.value,
        display_name=f"{'Original Qwen Image' if original else 'Qwen Image 2.1'} {'Dynamic' if dynamic else 'ComfyUI Native'}",
        model_family=family,
        model_variant=variant,
        primary_source_id=source.framework_id,
        base_grid=BaseGridDeclaration(
            identifier="flowmatch.reciprocal_step" if dynamic else "comfy.simple.flux_table",
            output_domain=SigmaDomain.UNIT_FLOW,
            terminal_included=False,
            parameters=() if dynamic else (ProfileField(name="training_timesteps", value=10_000),),
        ),
        transforms=(shift, spacing, _BASE.transforms[-1])
        if dynamic
        else (shift, _BASE.transforms[-1]),
        recipes=(
            InferenceRecipe(
                recipe_id=lane.value,
                evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
                source_id=source.framework_id,
                steps=StepRangeDeclaration(
                    minimum=2 if dynamic else 1,
                    maximum=_MAX_STEPS,
                    default=default_steps,
                    reference_steps=(default_steps,),
                    allow_modified=True,
                ),
                guidance=GuidanceDeclaration(
                    model_convention="true_cfg_scale",
                    host_convention="true_cfg_scale",
                    model_value=4.0 if original else 1.0,
                    host_value=4.0 if original else 1.0,
                ),
            ),
        ),
        detection=DetectionDeclaration(
            strategy_id=f"{family}.{variant}.explicit-v1",
            strict_default=True,
            ambiguity_requires_explicit=True,
            resolving_sources=("explicit_source_lane",),
            suggestion_sources=(),
            family_only_sources=(),
        ),
        model_capabilities=replace(
            _BASE.model_capabilities, model_family=family, model_variant=variant
        ),
        profile_capabilities=replace(
            _BASE.profile_capabilities,
            profile_id=lane.value,
            model_family=family,
            model_variant=variant,
        ),
        reference_sampler_capabilities=replace(
            _BASE.reference_sampler_capabilities, sampler_version=source.revision
        ),
        software_sources=_BASE.software_sources if original else (_QWEN21_SOURCE,),
        frameworks=(_COMFY, _DIFFUSERS),
        model_weights=_BASE.model_weights if original else (_QWEN21_WEIGHT,),
        parameters=parameters,
        known_limitations=(
            "Sigma construction only; no weights, image quality or inference execution verified.",
            "Use exactly one external-sigma source; do not shift these sigmas or patch sampling again.",
            "Original native 50 steps is a compatibility default, not a universal quality recommendation."
            if original
            else "Qwen 2.1 model materials use the Qwen Research License; commercial use needs separate authorization.",
            "Dynamic shift uses target-only unpatched tokens and affine extrapolation without clamping."
            if dynamic
            else "Fixed native mu is independent of resolution; select dynamic explicitly when required.",
        ),
    )
    return QwenNativeProfile(lane=lane, schema=schema)


QWEN_ORIGINAL_NATIVE_PROFILE: Final = _profile(QwenNativeLane.ORIGINAL_COMFY)
QWEN21_COMFY_NATIVE_PROFILE: Final = _profile(QwenNativeLane.QWEN21_COMFY)
QWEN21_DYNAMIC_PROFILE: Final = _profile(QwenNativeLane.QWEN21_DYNAMIC)
QWEN_NATIVE_PROFILES: Final = (
    QWEN_ORIGINAL_NATIVE_PROFILE,
    QWEN21_COMFY_NATIVE_PROFILE,
    QWEN21_DYNAMIC_PROFILE,
)
_PROFILES: Final = {profile.lane: profile for profile in QWEN_NATIVE_PROFILES}


def qwen21_target_tokens(width: int, height: int) -> int:
    for dimension in (width, height):
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            or dimension % 32
        ):
            raise ScheduleContractError(
                "Qwen 2.1 dimensions must be positive integers aligned to 32"
            )
    # CRITICAL: 2.1 flattens unpatched VAE-16 target latents. Original Qwen's /2 packing or
    # adding reference-image prefix tokens computes the wrong dynamic shift.
    tokens = (width // 16) * (height // 16)
    if tokens > _MAX_TOKENS:
        raise ScheduleContractError("Qwen 2.1 target token count exceeds 1000000")
    return tokens


def calculate_qwen21_mu(image_seq_len: int) -> float:
    if (
        not isinstance(image_seq_len, int)
        or isinstance(image_seq_len, bool)
        or not 1 <= image_seq_len <= _MAX_TOKENS
    ):
        raise ScheduleContractError("Qwen 2.1 target token count must be an integer in 1..1000000")
    return 0.5 + (image_seq_len - 256) * (0.9 - 0.5) / (8192 - 256)


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenNativeRequest:
    lane: QwenNativeLane
    steps: int
    width: int | None = None
    height: int | None = None
    image_seq_len: int | None = None
    strict_official: bool = False
    start_step: int = 0
    end_step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lane, QwenNativeLane):
            raise ScheduleContractError("Qwen source lane must be explicit and typed")
        dynamic = self.lane is QwenNativeLane.QWEN21_DYNAMIC
        minimum = 2 if dynamic else 1
        if (
            not isinstance(self.steps, int)
            or isinstance(self.steps, bool)
            or not minimum <= self.steps <= _MAX_STEPS
        ):
            raise ScheduleContractError(f"Qwen steps must be an integer in {minimum}..{_MAX_STEPS}")
        if not isinstance(self.strict_official, bool):
            raise ScheduleContractError("strict_official must be boolean")
        default = _PROFILES[self.lane].schema.recipes[0].steps.default
        if self.strict_official and self.steps != default:
            raise ScheduleContractError(f"strict source recipe requires {default} steps")
        if dynamic:
            has_dimensions = self.width is not None or self.height is not None
            if has_dimensions == (self.image_seq_len is not None):
                raise ScheduleContractError(
                    "choose exactly one target authority: dimensions or image_seq_len"
                )
            if has_dimensions:
                if self.width is None or self.height is None:
                    raise ScheduleContractError("width and height must be supplied together")
                qwen21_target_tokens(self.width, self.height)
            else:
                if self.image_seq_len is None:
                    raise ScheduleContractError("dynamic lane requires target tokens")
                calculate_qwen21_mu(self.image_seq_len)
        elif any(value is not None for value in (self.width, self.height, self.image_seq_len)):
            raise ScheduleContractError(
                "native fixed lane does not accept dynamic geometry or tokens"
            )
        end = self.steps if self.end_step is None else self.end_step
        if (
            not isinstance(self.start_step, int)
            or isinstance(self.start_step, bool)
            or not 0 <= self.start_step < self.steps
        ):
            raise ScheduleContractError("start_step must be an integer below steps")
        if (
            not isinstance(end, int)
            or isinstance(end, bool)
            or not self.start_step < end <= self.steps
        ):
            raise ScheduleContractError(
                "end_step must be greater than start_step and at most steps"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenNativeSchedule:
    request: QwenNativeRequest
    profile: QwenNativeProfile
    schedule: ScheduleResult
    output_sigmas: tuple[float, ...]
    mu: float
    image_seq_len: int | None


def build_qwen_native_schedule(request: QwenNativeRequest) -> QwenNativeSchedule:
    if not isinstance(request, QwenNativeRequest):
        raise ScheduleContractError("Qwen schedule requires a typed request")
    profile = _PROFILES[request.lane]
    dynamic = request.lane is QwenNativeLane.QWEN21_DYNAMIC
    tokens = request.image_seq_len
    if request.width is not None and request.height is not None:
        tokens = qwen21_target_tokens(request.width, request.height)
    if dynamic:
        if tokens is None:
            raise ScheduleContractError("dynamic lane requires target tokens")
        mu = calculate_qwen21_mu(tokens)
        base = flowmatch_reciprocal_step_grid(request.steps)
    else:
        mu = 1.15 if request.lane is QwenNativeLane.ORIGINAL_COMFY else 0.69
        base = comfyui_simple_discrete_flow_grid(request.steps, training_timesteps=10_000)
    # CRITICAL: ModelSamplingFlux shift is exponential mu, not the legacy direct ratio with
    # the same number. Transform exactly once before stretching, terminal append and slicing.
    shifted = exponential_mu_shift(base, mu=mu)
    if dynamic:
        tail_distance = 1.0 - shifted[-1]
        if tail_distance <= 0.0:
            raise ScheduleContractError("dynamic terminal stretch requires a non-degenerate tail")
        shifted = tuple(1.0 - (1.0 - value) * 0.98 / tail_distance for value in shifted)
    complete = apply_terminal_policy(
        shifted, policy=TerminalPolicy.APPEND_ZERO, domain=SigmaDomain.UNIT_FLOW
    )
    complete = validate_sigma_schedule(
        complete,
        domain=SigmaDomain.UNIT_FLOW,
        expected_steps=request.steps,
        require_terminal_zero=True,
    )
    default = profile.schema.recipes[0].steps.default
    inputs = ScheduleInputs(steps=request.steps, width=request.width, height=request.height)
    source = _DIFFUSERS if dynamic else _COMFY
    schedule = ScheduleResult(
        request=ScheduleRequest(
            ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
            requested_inputs=inputs,
            sigma_domain=SigmaDomain.UNIT_FLOW,
            provenance=Provenance(
                engine_version=VERSION,
                evidence=EvidenceLevel.FRAMEWORK_REFERENCE
                if request.steps == default
                else EvidenceLevel.MODIFIED,
                source=source.url,
                source_revision=source.revision,
                profile_id=profile.schema.profile_id,
                profile_version=profile.schema.profile_version,
            ),
            base_grid=BaseGridSpec(
                identifier=cast(BaseGridDeclaration, profile.schema.base_grid).identifier,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
            transforms=tuple(transform.contract() for transform in profile.schema.transforms),
            terminal_policy=TerminalPolicy.APPEND_ZERO,
            slicing=SliceSpec(),
        ),
        effective_inputs=inputs,
        sigmas=complete,
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=()
        if request.steps == default
        else (
            "steps differ from the declared source/compatibility baseline; evidence is modified",
        ),
    )
    return QwenNativeSchedule(
        request=request,
        profile=profile,
        schedule=schedule,
        mu=mu,
        image_seq_len=tokens,
        output_sigmas=slice_step_range(
            complete, start_step=request.start_step, end_step=request.end_step
        ),
    )


__all__ = [
    "QWEN21_COMFY_NATIVE_PROFILE",
    "QWEN21_DIFFUSERS_REVISION",
    "QWEN21_DYNAMIC_PROFILE",
    "QWEN21_MODEL_REVISION",
    "QWEN21_PUBLISHER_REVISION",
    "QWEN21_TEMPLATE_REVISION",
    "QWEN_NATIVE_COMFYUI_REVISION",
    "QWEN_NATIVE_PROFILES",
    "QWEN_ORIGINAL_NATIVE_PROFILE",
    "QwenNativeLane",
    "QwenNativeProfile",
    "QwenNativeRequest",
    "QwenNativeSchedule",
    "build_qwen_native_schedule",
    "calculate_qwen21_mu",
    "qwen21_target_tokens",
]
