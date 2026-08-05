"""Evidence-pinned MiniMax H3 Base schedule and paired-coordinate contracts.

The public schedule in this module is deliberately narrow: it owns only the external
video sigma lane.  MiniMax H3's audio coordinate remap and velocity correction remain
model-owned diagnostics, so this module never returns a second audio schedule or patches
ComfyUI internals.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, cast

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

FloatPrecision = Literal["float32", "float64"]

MINIMAX_H3_HF_REVISION: Final = (
    "5d9b308a59ab12e67147f191e184baf704185bd1"  # pragma: allowlist secret
)
MINIMAX_H3_COMFY_ARTIFACT_REVISION: Final = (
    "0543966fbdce5ba05709a8f2031c94bdba629b4a"  # pragma: allowlist secret
)
MINIMAX_H3_COMFYUI_REVISION: Final = (
    "14b05228cef127ce529bc0c08660770d4af3e9a8"  # pragma: allowlist secret
)
MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT: Final = (
    "57500fc5bc92566a63f2046824f522cd55c335ca"  # pragma: allowlist secret
)
MINIMAX_H3_DIFFUSERS_REVISION: Final = (
    "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"  # pragma: allowlist secret
)

MINIMAX_H3_VIDEO_SHIFT: Final = 12.0
MINIMAX_H3_AUDIO_SHIFT: Final = 3.0
# Native ComfyUI's MiniMaxH3Model returns the data-ward model velocity with a
# leading minus sign so CONST.calculate_denoised/to_d uses the host's outer
# sigma direction.  Keep this as an explicit contract; never infer it from a
# generic flow profile.
MINIMAX_H3_NATIVE_MODEL_OUTPUT_SIGN: Final = -1.0
MINIMAX_H3_DEFAULT_GRID_POINTS: Final = 20
MINIMAX_H3_MAX_GRID_POINTS: Final = 10_000
MINIMAX_H3_COMFY_TIMESTEPS: Final = 1_000

_H3_FLOAT32 = ">f"


class MiniMaxH3Variant(str, Enum):
    """Explicit H3 Base checkpoint variants; automatic variant guessing is forbidden."""

    BASE_FL2VA = "base_fl2va"
    BASE_REF2VA = "base_ref2va"


class MiniMaxH3ScheduleLane(str, Enum):
    """Independent numerical lanes whose interior grids must not be conflated."""

    DIFFUSERS_ENDPOINT = "diffusers_endpoint_inclusive"
    COMFYUI_SIMPLE = "comfyui_simple"


class MiniMaxH3VelocityDirection(str, Enum):
    """Direction convention for a model velocity and an integration adapter."""

    DATA_WARD = "data_ward"
    NOISE_WARD = "noise_ward"


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3EvidenceReference:
    """One pinned source lane supporting the H3 schedule claim."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        allowed = {
            "official_huggingface",
            "comfy_org_artifact",
            "comfyui_implementation",
            "diffusers_scheduler",
        }
        if self.lane not in allowed:
            raise ScheduleContractError("MiniMax H3 evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("MiniMax H3 evidence URL must use HTTPS")
        if not isinstance(self.revision, str) or len(self.revision) != 40:
            raise ScheduleContractError("MiniMax H3 evidence revision must be pinned")
        try:
            int(self.revision, 16)
        except ValueError as exc:
            raise ScheduleContractError("MiniMax H3 evidence revision must be hexadecimal") from exc
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("MiniMax H3 evidence locators must be sorted and unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3AudioMapping:
    """Paired video/audio coordinates and the analytic velocity derivative."""

    video_sigma: float
    base_coordinate: float
    audio_sigma: float
    derivative: float
    precision: FloatPrecision

    def __post_init__(self) -> None:
        if self.precision not in {"float32", "float64"}:
            raise ScheduleContractError("MiniMax H3 audio mapping precision is unsupported")
        for name in ("video_sigma", "base_coordinate", "audio_sigma", "derivative"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ScheduleContractError(f"MiniMax H3 {name} must be finite")
        if not 0.0 <= self.video_sigma <= 1.0:
            raise ScheduleContractError("MiniMax H3 video_sigma must be in [0, 1]")
        if not 0.0 <= self.base_coordinate <= 1.0 or not 0.0 <= self.audio_sigma <= 1.0:
            raise ScheduleContractError("MiniMax H3 paired coordinates must be in [0, 1]")
        if self.derivative <= 0.0:
            raise ScheduleContractError("MiniMax H3 audio derivative must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3Profile:
    """Immutable profile plus the four independently pinned evidence lanes."""

    variant: MiniMaxH3Variant
    schema: ProfileSchemaV1
    references: tuple[MiniMaxH3EvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        expected = {
            MiniMaxH3Variant.BASE_FL2VA: MINIMAX_H3_BASE_FL2VA_SCHEMA,
            MiniMaxH3Variant.BASE_REF2VA: MINIMAX_H3_BASE_REF2VA_SCHEMA,
        }.get(self.variant)
        if expected is None or self.schema is not expected:
            raise ScheduleContractError("MiniMax H3 profile/schema mismatch")
        lanes = tuple(reference.lane for reference in self.references)
        if lanes != tuple(sorted(set(lanes))) or len(lanes) != 4:
            raise ScheduleContractError("MiniMax H3 requires four pinned evidence lanes")


def _f32(value: float) -> float:
    """Round one scalar exactly as a host float32 tensor would."""

    try:
        return cast(float, struct.unpack(_H3_FLOAT32, struct.pack(_H3_FLOAT32, float(value)))[0])
    except (OverflowError, struct.error) as exc:
        raise ScheduleContractError("MiniMax H3 float32 conversion failed") from exc


def _precision(value: float, precision: FloatPrecision) -> float:
    if precision == "float64":
        return float(value)
    if precision == "float32":
        return _f32(value)
    raise ScheduleContractError("precision must be float32 or float64")


def _require_variant(value: object) -> MiniMaxH3Variant:
    if not isinstance(value, MiniMaxH3Variant):
        raise ScheduleContractError("MiniMax H3 variant must be selected explicitly")
    return value


def _require_grid_points(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 2 <= value <= MINIMAX_H3_MAX_GRID_POINTS
    ):
        raise ScheduleContractError(
            f"MiniMax H3 grid_points must be an integer between 2 and {MINIMAX_H3_MAX_GRID_POINTS}"
        )
    return value


def _endpoint_grid(points: int, precision: FloatPrecision) -> tuple[float, ...]:
    denominator = points - 1
    values: list[float] = []
    for index in range(points):
        if index == 0:
            value = 1.0
        elif index == denominator:
            value = 0.0
        else:
            value = 1.0 - (index / denominator)
        values.append(_precision(value, precision))
    return tuple(values)


def _direct_ratio(value: float, ratio: float, precision: FloatPrecision) -> float:
    if value == 0.0 or value == 1.0:
        return _precision(value, precision)
    if precision == "float64":
        return ratio * value / (1.0 + (ratio - 1.0) * value)
    # Preserve the operation order used by the pinned Diffusers/ComfyUI expression.
    numerator = _f32(ratio * value)
    denominator = _f32(1.0 + _f32((ratio - 1.0) * value))
    return _f32(numerator / denominator)


def _unique_consecutive(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        raise ScheduleContractError("MiniMax H3 schedule cannot be empty")
    result = [values[0]]
    for value in values[1:]:
        if value != result[-1]:
            result.append(value)
    return tuple(result)


def _profile_for_variant(variant: MiniMaxH3Variant) -> MiniMaxH3Profile:
    if variant is MiniMaxH3Variant.BASE_FL2VA:
        return MINIMAX_H3_BASE_FL2VA_PROFILE
    if variant is MiniMaxH3Variant.BASE_REF2VA:
        return MINIMAX_H3_BASE_REF2VA_PROFILE
    raise ScheduleContractError("MiniMax H3 variant is unsupported")


def _schedule_request(
    *,
    profile: MiniMaxH3Profile,
    lane: MiniMaxH3ScheduleLane,
    requested_steps: int,
    source: str,
    source_revision: str,
    base_grid: str,
) -> ScheduleRequest:
    return ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=requested_steps),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
            source=source,
            source_revision=source_revision,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
        ),
        base_grid=BaseGridSpec(identifier=base_grid, output_domain=SigmaDomain.UNIT_FLOW),
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


def build_minimax_h3_schedule(
    *,
    variant: object,
    grid_points: object,
    precision: FloatPrecision = "float32",
    already_shifted: object = False,
) -> ScheduleResult:
    """Build the endpoint-inclusive Diffusers H3 video schedule.

    ``grid_points`` is the requested number of points before float32 consecutive
    deduplication; the returned ``ScheduleInputs.steps`` values are transitions.
    """

    selected = _require_variant(variant)
    points = _require_grid_points(grid_points)
    if precision not in {"float32", "float64"}:
        raise ScheduleContractError("precision must be float32 or float64")
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError("already shifted sigmas cannot receive a second H3 shift")
    profile = _profile_for_variant(selected)
    base = _endpoint_grid(points, precision)
    shifted = _unique_consecutive(
        tuple(_direct_ratio(value, MINIMAX_H3_VIDEO_SHIFT, precision) for value in base)
    )
    if shifted[-1] != 0.0:
        raise ScheduleContractError("MiniMax H3 shifted endpoint must be exactly zero")
    effective_steps = len(shifted) - 1
    requested_steps = points - 1
    overrides: tuple[OverrideRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    if effective_steps != requested_steps:
        override = OverrideRecord(
            field="steps",
            requested_value=str(requested_steps),
            effective_value=str(effective_steps),
            reason="float32 endpoint grid collapsed consecutive values",
        )
        overrides = (override,)
        warnings = ("float32 endpoint grid collapsed consecutive values",)
    request = _schedule_request(
        profile=profile,
        lane=MiniMaxH3ScheduleLane.DIFFUSERS_ENDPOINT,
        requested_steps=requested_steps,
        source="https://github.com/huggingface/diffusers",
        source_revision=MINIMAX_H3_DIFFUSERS_REVISION,
        base_grid="minimax_h3.endpoint_grid",
    )
    return ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=effective_steps),
        sigmas=validate_sigma_schedule(
            shifted,
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=effective_steps,
            require_terminal_zero=True,
        ),
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=warnings,
        overrides=overrides,
    )


def build_minimax_h3_comfyui_simple_schedule(
    *,
    variant: object,
    transitions: object,
    already_shifted: object = False,
) -> ScheduleResult:
    """Build the separately tracked native ComfyUI ``simple`` H3 lane."""

    selected = _require_variant(variant)
    if (
        not isinstance(transitions, int)
        or isinstance(transitions, bool)
        or not 1 <= transitions <= MINIMAX_H3_COMFY_TIMESTEPS
    ):
        raise ScheduleContractError(
            f"MiniMax H3 simple transitions must be between 1 and {MINIMAX_H3_COMFY_TIMESTEPS}"
        )
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError("already shifted sigmas cannot receive a second H3 shift")
    profile = _profile_for_variant(selected)
    table = tuple(
        _direct_ratio(
            _precision(index / MINIMAX_H3_COMFY_TIMESTEPS, "float32"),
            MINIMAX_H3_VIDEO_SHIFT,
            "float32",
        )
        for index in range(1, MINIMAX_H3_COMFY_TIMESTEPS + 1)
    )
    spacing = len(table) / transitions
    values = (*tuple(table[-(1 + int(index * spacing))] for index in range(transitions)), 0.0)
    request = _schedule_request(
        profile=profile,
        lane=MiniMaxH3ScheduleLane.COMFYUI_SIMPLE,
        requested_steps=transitions,
        source="https://github.com/Comfy-Org/ComfyUI",
        source_revision=MINIMAX_H3_COMFYUI_REVISION,
        base_grid="comfyui.discrete_flow_1000",
    )
    return ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=transitions),
        sigmas=validate_sigma_schedule(
            values,
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=transitions,
            require_terminal_zero=True,
        ),
        final_domain=SigmaDomain.UNIT_FLOW,
    )


def map_minimax_h3_audio_coordinate(
    video_sigma: object,
    *,
    precision: FloatPrecision = "float64",
    video_shift: float = MINIMAX_H3_VIDEO_SHIFT,
    audio_shift: float = MINIMAX_H3_AUDIO_SHIFT,
) -> MiniMaxH3AudioMapping:
    """Map one video sigma to H3's model-owned audio coordinate and derivative."""

    if precision not in {"float32", "float64"}:
        raise ScheduleContractError("precision must be float32 or float64")
    if (
        isinstance(video_sigma, bool)
        or not isinstance(video_sigma, (int, float))
        or not math.isfinite(float(video_sigma))
        or not 0.0 <= float(video_sigma) <= 1.0
    ):
        raise ScheduleContractError("video_sigma must be finite and in [0, 1]")
    if video_shift <= 0.0 or audio_shift <= 0.0:
        raise ScheduleContractError("H3 shifts must be positive")
    video = _precision(float(video_sigma), precision)
    if video == 0.0:
        return MiniMaxH3AudioMapping(
            video_sigma=video,
            base_coordinate=_precision(0.0, precision),
            audio_sigma=_precision(0.0, precision),
            derivative=_precision(audio_shift / video_shift, precision),
            precision=precision,
        )
    if video == 1.0:
        return MiniMaxH3AudioMapping(
            video_sigma=video,
            base_coordinate=_precision(1.0, precision),
            audio_sigma=_precision(1.0, precision),
            derivative=_precision(video_shift / audio_shift, precision),
            precision=precision,
        )
    base = _precision(video / (video_shift + video * (1.0 - video_shift)), precision)
    audio = _precision(audio_shift * base / (1.0 + (audio_shift - 1.0) * base), precision)
    derivative = _precision(
        audio_shift
        * (1.0 + (video_shift - 1.0) * base) ** 2
        / (video_shift * (1.0 + (audio_shift - 1.0) * base) ** 2),
        precision,
    )
    return MiniMaxH3AudioMapping(
        video_sigma=video,
        base_coordinate=base,
        audio_sigma=audio,
        derivative=derivative,
        precision=precision,
    )


def minimax_h3_velocity_conversion_sign(
    source: MiniMaxH3VelocityDirection,
    target: MiniMaxH3VelocityDirection,
) -> float:
    """Return the explicit sign required when crossing H3 velocity conventions."""

    if not isinstance(source, MiniMaxH3VelocityDirection) or not isinstance(
        target, MiniMaxH3VelocityDirection
    ):
        raise ScheduleContractError("H3 velocity directions must be explicit enum values")
    return 1.0 if source is target else -1.0


def minimax_h3_native_model_output_sign() -> float:
    """Return the pinned native ComfyUI model-output sign adapter."""

    return MINIMAX_H3_NATIVE_MODEL_OUTPUT_SIGN


_H3_LICENSE = LicenseDeclaration(
    declaration_version="1",
    identifier="LicenseRef-MiniMax-H3-Community",
    name="MiniMax H3 Community License",
    url="https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE",
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
_OFFICIAL_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="minimax.h3.official",
    resource_version="MiniMax-H3",
    revision=MINIMAX_H3_HF_REVISION,
    url="https://huggingface.co/MiniMaxAI/MiniMax-H3",
    license=_H3_LICENSE,
    locators=("LICENSE", "README.md", "config.json"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.minimax-h3.framework",
    resource_version="v0.30.0",
    revision=MINIMAX_H3_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=("comfy/model_sampling.py", "comfy/samplers.py", "comfy/supported_models.py"),
)
_DIFFUSERS_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.minimax-h3.framework",
    resource_version="pre-release-h3",
    revision=MINIMAX_H3_DIFFUSERS_REVISION,
    url="https://github.com/huggingface/diffusers",
    license=_APACHE_2,
    locators=("src/diffusers/schedulers/scheduling_minimax_h3.py",),
)


def _weight(
    *, variant: MiniMaxH3Variant, quantization: str, resource_version: str, sha256: str
) -> ModelWeightProvenance:
    return ModelWeightProvenance(
        record_version="1",
        weight_id=f"comfy-org.minimax-h3.{variant.value}.{quantization}",
        resource_version=resource_version,
        revision=MINIMAX_H3_COMFY_ARTIFACT_REVISION,
        sha256=sha256,
        url="https://huggingface.co/Comfy-Org/MiniMax-H3",
        license=_H3_LICENSE,
    )


_FL2VA_WEIGHTS = (
    _weight(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        quantization="bf16",
        resource_version="diffusion_models/minimax_h3_fl2va_bf16.safetensors",
        sha256="907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182",  # pragma: allowlist secret
    ),
    _weight(
        variant=MiniMaxH3Variant.BASE_FL2VA,
        quantization="int8",
        resource_version="diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors",
        sha256="7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5",  # pragma: allowlist secret
    ),
)
_REF2VA_WEIGHTS = (
    _weight(
        variant=MiniMaxH3Variant.BASE_REF2VA,
        quantization="bf16",
        resource_version="diffusion_models/minimax_h3_ref2va_bf16.safetensors",
        sha256="e32c54c1a7b4f5f397f195cea267ccb18806303bb665678c4bee60953bdf3026",  # pragma: allowlist secret
    ),
    _weight(
        variant=MiniMaxH3Variant.BASE_REF2VA,
        quantization="int8",
        resource_version="diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors",
        sha256="9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9",  # pragma: allowlist secret
    ),
)
_DETECTION = DetectionDeclaration(
    strategy_id="minimax_h3.explicit-variant-v1",
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
_BASE_GRID = BaseGridDeclaration(
    identifier="minimax_h3.endpoint_grid",
    output_domain=SigmaDomain.UNIT_FLOW,
    terminal_included=False,
    parameters=(
        ProfileField(name="grid_semantics", value="endpoint_inclusive"),
        ProfileField(name="precision_deduplication", value="float32_unique_consecutive"),
    ),
)
_SAMPLER = SamplerCapabilities(
    sampler_id="flowmatch.euler",
    sampler_version=MINIMAX_H3_DIFFUSERS_REVISION,
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


def _capabilities(
    *, variant: MiniMaxH3Variant, profile_id: str
) -> tuple[ModelCapabilities, ProfileCapabilities]:
    model = ModelCapabilities(
        model_family="minimax_h3",
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
        model_family="minimax_h3",
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
    return model, profile


def _schema(
    *, variant: MiniMaxH3Variant, display_name: str, weights: tuple[ModelWeightProvenance, ...]
) -> ProfileSchemaV1:
    profile_id = f"minimax-h3.{variant.value}"
    model, profile = _capabilities(variant=variant, profile_id=profile_id)
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=profile_id,
        profile_version="1",
        display_name=display_name,
        model_family="minimax_h3",
        model_variant=variant.value,
        evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
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
                parameters=(
                    ProfileField(name="ratio", value=MINIMAX_H3_VIDEO_SHIFT),
                    ProfileField(name="transform_order", value="base_then_shift"),
                ),
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
                recipe_id="minimax-h3.comfy-simple",
                evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
                source_id=_COMFYUI_FRAMEWORK.framework_id,
                steps=StepRangeDeclaration(
                    minimum=1,
                    maximum=MINIMAX_H3_COMFY_TIMESTEPS,
                    default=20,
                    reference_steps=(20,),
                    allow_modified=True,
                ),
                guidance=GuidanceDeclaration(
                    model_convention="embedded_cfg_distilled",
                    host_convention="cfg_scale",
                    model_value=1.0,
                    host_value=1.0,
                ),
            ),
            InferenceRecipe(
                recipe_id="minimax-h3.diffusers",
                evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
                source_id=_DIFFUSERS_FRAMEWORK.framework_id,
                steps=StepRangeDeclaration(
                    minimum=1,
                    maximum=MINIMAX_H3_MAX_GRID_POINTS - 1,
                    default=MINIMAX_H3_DEFAULT_GRID_POINTS - 1,
                    reference_steps=(MINIMAX_H3_DEFAULT_GRID_POINTS - 1,),
                    allow_modified=True,
                ),
                guidance=GuidanceDeclaration(
                    model_convention="embedded_cfg_distilled",
                    host_convention="cfg_scale",
                    model_value=1.0,
                    host_value=1.0,
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
        model_weights=weights,
        parameters=(
            ProfileField(name="audio_derivative_ownership", value="model_native"),
            ProfileField(name="audio_shift", value=MINIMAX_H3_AUDIO_SHIFT),
            ProfileField(name="grid_semantics", value="endpoint_inclusive_diffusers"),
            ProfileField(name="license_boundary", value="code_only_no_weight_redistribution"),
            ProfileField(name="model_time", value="t_equals_one_minus_sigma"),
            ProfileField(name="variant", value=variant.value),
            ProfileField(
                name="velocity_direction", value=MiniMaxH3VelocityDirection.DATA_WARD.value
            ),
            ProfileField(name="video_shift", value=MINIMAX_H3_VIDEO_SHIFT),
            ProfileField(name="weight_distribution", value="local_or_official_only"),
        ),
        known_limitations=(
            "FL2VA and Ref2VA checkpoint structures are indistinguishable; explicit variant selection is required.",
            "Only the external video sigma lane is exposed; audio remapping and its derivative remain model-owned.",
            "Diffusers endpoint-inclusive and native ComfyUI simple grids are separate evidence lanes.",
            "This slice does not implement Context-IR, Regenerate-2K, sparse attention, hosted/API behavior, a sampler, or image/audio quality claims.",
            "MiniMax H3 weights are not redistributed; weight redistribution is outside this code-only slice. Recorded hashes are official artifact metadata, not local full-payload verification.",
        ),
    )


MINIMAX_H3_BASE_FL2VA_SCHEMA: Final = _schema(
    variant=MiniMaxH3Variant.BASE_FL2VA,
    display_name="MiniMax H3 Base FL2VA Sigma Scheduler",
    weights=_FL2VA_WEIGHTS,
)
MINIMAX_H3_BASE_REF2VA_SCHEMA: Final = _schema(
    variant=MiniMaxH3Variant.BASE_REF2VA,
    display_name="MiniMax H3 Base Ref2VA Sigma Scheduler",
    weights=_REF2VA_WEIGHTS,
)


def _references(variant: MiniMaxH3Variant) -> tuple[MiniMaxH3EvidenceReference, ...]:
    artifact_prefix = (
        "minimax_h3_fl2va" if variant is MiniMaxH3Variant.BASE_FL2VA else "minimax_h3_ref2va"
    )
    return (
        MiniMaxH3EvidenceReference(
            lane="comfy_org_artifact",
            url="https://huggingface.co/Comfy-Org/MiniMax-H3",
            revision=MINIMAX_H3_COMFY_ARTIFACT_REVISION,
            locators=(
                "README.md",
                f"diffusion_models/{artifact_prefix}_bf16.safetensors",
                f"diffusion_models/{artifact_prefix}_int8_convrot.safetensors",
            ),
        ),
        MiniMaxH3EvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT,
            locators=("comfy/model_sampling.py", "comfy/samplers.py", "comfy/supported_models.py"),
        ),
        MiniMaxH3EvidenceReference(
            lane="diffusers_scheduler",
            url="https://github.com/huggingface/diffusers",
            revision=MINIMAX_H3_DIFFUSERS_REVISION,
            locators=("src/diffusers/schedulers/scheduling_minimax_h3.py",),
        ),
        MiniMaxH3EvidenceReference(
            lane="official_huggingface",
            url="https://huggingface.co/MiniMaxAI/MiniMax-H3",
            revision=MINIMAX_H3_HF_REVISION,
            locators=("LICENSE", "README.md", "config.json"),
        ),
    )


MINIMAX_H3_BASE_FL2VA_PROFILE: Final = MiniMaxH3Profile(
    variant=MiniMaxH3Variant.BASE_FL2VA,
    schema=MINIMAX_H3_BASE_FL2VA_SCHEMA,
    references=_references(MiniMaxH3Variant.BASE_FL2VA),
)
MINIMAX_H3_BASE_REF2VA_PROFILE: Final = MiniMaxH3Profile(
    variant=MiniMaxH3Variant.BASE_REF2VA,
    schema=MINIMAX_H3_BASE_REF2VA_SCHEMA,
    references=_references(MiniMaxH3Variant.BASE_REF2VA),
)


__all__ = [
    "MINIMAX_H3_AUDIO_SHIFT",
    "MINIMAX_H3_BASE_FL2VA_PROFILE",
    "MINIMAX_H3_BASE_FL2VA_SCHEMA",
    "MINIMAX_H3_BASE_REF2VA_PROFILE",
    "MINIMAX_H3_BASE_REF2VA_SCHEMA",
    "MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT",
    "MINIMAX_H3_COMFYUI_REVISION",
    "MINIMAX_H3_COMFY_ARTIFACT_REVISION",
    "MINIMAX_H3_DEFAULT_GRID_POINTS",
    "MINIMAX_H3_DIFFUSERS_REVISION",
    "MINIMAX_H3_HF_REVISION",
    "MINIMAX_H3_NATIVE_MODEL_OUTPUT_SIGN",
    "MINIMAX_H3_VIDEO_SHIFT",
    "MiniMaxH3AudioMapping",
    "MiniMaxH3EvidenceReference",
    "MiniMaxH3Profile",
    "MiniMaxH3ScheduleLane",
    "MiniMaxH3Variant",
    "MiniMaxH3VelocityDirection",
    "build_minimax_h3_comfyui_simple_schedule",
    "build_minimax_h3_schedule",
    "map_minimax_h3_audio_coordinate",
    "minimax_h3_native_model_output_sign",
    "minimax_h3_velocity_conversion_sign",
]
