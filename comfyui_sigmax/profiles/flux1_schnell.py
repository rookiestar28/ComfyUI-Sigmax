"""Four-source-pinned FLUX.1-schnell schedule profile."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
_GITHUB_REVISION: Final = "802fb4713906133fcbd0d8dc5351620ca4773036"  # pragma: allowlist secret
_HF_REVISION: Final = "741f7c3ce8b383c54771c7003378a50191e9efe9"  # pragma: allowlist secret
_HF_README_REVISION: Final = "adb67b7ac923e832bfb7284be9ae3d00bcdad000"  # pragma: allowlist secret
_COMFYUI_REVISION: Final = "2881e6161081439b1c3fb3b6c1f51b3d272da710"  # pragma: allowlist secret
_COMFY_WEIGHT_REVISION: Final = (
    "7d679837b018bfeb28eca55734b335efcd0e7100"  # pragma: allowlist secret
)
_COMFY_WEIGHT_SHA256: Final = (
    "ead426278b49030e9da5df862994f25ce94ab2ee4df38b556ddddb3db093bf72"  # pragma: allowlist secret
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Flux1SchnellEvidenceReference:
    """One mandatory source lane for the FLUX.1-schnell profile."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        lanes = {
            "comfyui_implementation",
            "official_github",
            "official_huggingface",
            "official_technical_document",
        }
        if self.lane not in lanes:
            raise ScheduleContractError("FLUX.1-schnell evidence lane is unsupported")
        if not self.url.startswith("https://"):
            raise ScheduleContractError("FLUX.1-schnell evidence URL must use HTTPS")
        valid_revision = bool(_COMMIT_PATTERN.fullmatch(self.revision))
        if self.lane == "official_technical_document":
            valid_revision = bool(_DATE_PATTERN.fullmatch(self.revision))
        if not valid_revision:
            raise ScheduleContractError("FLUX.1-schnell evidence revision is invalid")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError(
                "FLUX.1-schnell evidence locators must be sorted and unique"
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
    source_id="black-forest-labs.flux1.official",
    resource_version=None,
    revision=_GITHUB_REVISION,
    url="https://github.com/black-forest-labs/flux",
    license=_APACHE_2,
    locators=(
        "model_cards/FLUX.1-schnell.md",
        "src/flux/cli.py",
        "src/flux/sampling.py",
        "src/flux/util.py",
    ),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.flux1-schnell.framework",
    resource_version=None,
    revision=_COMFYUI_REVISION,
    url="https://github.com/Comfy-Org/ComfyUI",
    license=_GPL_3_ONLY,
    locators=(
        "comfy/model_base.py",
        "comfy/model_sampling.py",
        "comfy/samplers.py",
        "comfy/supported_models.py",
    ),
)
_HOST_WEIGHT = ModelWeightProvenance(
    record_version="1",
    weight_id="comfy-org.flux1-schnell.fp8",
    resource_version="flux1-schnell-fp8.safetensors",
    revision=_COMFY_WEIGHT_REVISION,
    sha256=_COMFY_WEIGHT_SHA256,
    url="https://huggingface.co/Comfy-Org/flux1-schnell",
    license=_APACHE_2,
)
_DETECTION = DetectionDeclaration(
    strategy_id="flux1.schnell.explicit-v1",
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
_TERMINAL_TRANSFORM = TransformDeclaration(
    identifier="terminal.append_zero",
    stage=TransformStage.TERMINAL,
    input_domain=SigmaDomain.UNIT_FLOW,
    output_domain=SigmaDomain.UNIT_FLOW,
)


def _capabilities() -> tuple[ModelCapabilities, ProfileCapabilities, SamplerCapabilities]:
    model = ModelCapabilities(
        model_family="flux1",
        model_variant="schnell",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id="flux1.schnell.official",
        profile_version="1",
        model_family="flux1",
        model_variant="schnell",
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


_MODEL_CAPABILITIES, _PROFILE_CAPABILITIES, _SAMPLER_CAPABILITIES = _capabilities()

FLUX1_SCHNELL_SCHEMA: Final = ProfileSchemaV1(
    schema_id=PROFILE_SCHEMA_ID,
    schema_version=PROFILE_SCHEMA_VERSION,
    profile_id="flux1.schnell.official",
    profile_version="1",
    display_name="FLUX.1-schnell Official",
    model_family="flux1",
    model_variant="schnell",
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
    transforms=(_TERMINAL_TRANSFORM,),
    terminal=TerminalDeclaration(
        policy=TerminalPolicy.APPEND_ZERO,
        sigma=TerminalSigma.ZERO,
        value=0.0,
    ),
    slicing=_SLICING,
    recipes=(
        InferenceRecipe(
            recipe_id="flux1.schnell.official",
            evidence=EvidenceLevel.OFFICIAL,
            source_id=_OFFICIAL_SOURCE.source_id,
            steps=StepRangeDeclaration(
                minimum=1,
                maximum=4,
                default=4,
                reference_steps=(1, 4),
                allow_modified=True,
            ),
            guidance=GuidanceDeclaration(
                model_convention="no_guidance_embedding",
                host_convention="comfy.basic_guider_or_cfg",
                model_value=0.0,
                host_value=1.0,
            ),
        ),
    ),
    detection=_DETECTION,
    model_capabilities=_MODEL_CAPABILITIES,
    profile_capabilities=_PROFILE_CAPABILITIES,
    reference_sampler_capabilities=_SAMPLER_CAPABILITIES,
    artifact_versions=_ARTIFACT_VERSIONS,
    software_sources=(_OFFICIAL_SOURCE,),
    frameworks=(_COMFYUI_FRAMEWORK,),
    model_weights=(_HOST_WEIGHT,),
    parameters=(
        ProfileField(name="dynamic_shifting", value=False),
        ProfileField(name="guidance_embedding", value=False),
        ProfileField(name="max_sequence_length", value=256),
        ProfileField(name="publisher_grid_terminal_included", value=True),
    ),
    known_limitations=(
        "Automatic FLUX variant detection is intentionally unsupported; select Schnell explicitly.",
        "The gated publisher Hub config bodies and BF16 LFS digest were not anonymously retrievable.",
        "The pinned exact weight digest is the official ComfyUI FP8 checkpoint and is not byte-identical to the publisher BF16 file.",
        "Schedule parity does not claim model-weight, GPU, image-quality, or full sampler parity.",
    ),
)


def _references() -> tuple[Flux1SchnellEvidenceReference, ...]:
    return (
        Flux1SchnellEvidenceReference(
            lane="comfyui_implementation",
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=_COMFYUI_REVISION,
            locators=(
                "comfy/model_sampling.py",
                "comfy/samplers.py",
                "comfy/supported_models.py",
            ),
        ),
        Flux1SchnellEvidenceReference(
            lane="official_github",
            url="https://github.com/black-forest-labs/flux",
            revision=_GITHUB_REVISION,
            locators=("src/flux/cli.py", "src/flux/sampling.py", "src/flux/util.py"),
        ),
        Flux1SchnellEvidenceReference(
            lane="official_huggingface",
            url="https://huggingface.co/black-forest-labs/FLUX.1-schnell",
            revision=_HF_REVISION,
            locators=(
                f"README.md@{_HF_README_REVISION}",
                "flux1-schnell.safetensors",
                "scheduler/scheduler_config.json",
            ),
        ),
        Flux1SchnellEvidenceReference(
            lane="official_technical_document",
            url="https://bfl.ai/blog/24-08-01-bfl",
            revision="2024-08-01",
            locators=("Flux.1 Model Family", "Transformer-powered Flow Models at Scale"),
        ),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Flux1SchnellProfile:
    """Immutable publisher schedule and four-source evidence contract."""

    minimum_official_steps: int
    maximum_official_steps: int
    default_steps: int
    references: tuple[Flux1SchnellEvidenceReference, ...]
    schema: ProfileSchemaV1
    dynamic_shifting: bool = False

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        if self.dynamic_shifting:
            raise ScheduleContractError("FLUX.1-schnell must remain unshifted")
        if not (
            self.minimum_official_steps == 1
            and self.maximum_official_steps == self.default_steps == 4
        ):
            raise ScheduleContractError("FLUX.1-schnell official step bounds are inconsistent")
        lanes = tuple(reference.lane for reference in self.references)
        if lanes != tuple(sorted(lanes)) or len(lanes) != 4 or len(set(lanes)) != 4:
            raise ScheduleContractError("FLUX.1-schnell requires all four canonical evidence lanes")


FLUX1_SCHNELL_PROFILE: Final = Flux1SchnellProfile(
    minimum_official_steps=1,
    maximum_official_steps=4,
    default_steps=4,
    references=_references(),
    schema=FLUX1_SCHNELL_SCHEMA,
)


def build_flux1_schnell_schedule(*, steps: int, strict_official: bool = False) -> ScheduleResult:
    """Build the exact unshifted FLUX.1-schnell grid without framework imports."""

    inputs = ScheduleInputs(steps=steps)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")
    official = 1 <= steps <= 4
    if strict_official and not official:
        raise ScheduleContractError("steps are outside the official FLUX.1-schnell recipe")
    evidence = EvidenceLevel.OFFICIAL if official else EvidenceLevel.MODIFIED
    warnings = () if official else ("steps exceed the published 1-4 range; evidence is modified",)
    transforms = (
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
            source="https://github.com/black-forest-labs/flux",
            source_revision=_GITHUB_REVISION,
            profile_id=FLUX1_SCHNELL_PROFILE.profile_id,
            profile_version=FLUX1_SCHNELL_PROFILE.profile_version,
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
        flowmatch_reciprocal_step_grid(steps),
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
        warnings=warnings,
    )
