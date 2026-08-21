"""Source-qualified Wan 2.1 and Wan 2.2 schedule profiles.

The Wan family has several documented shift owners.  This module keeps the ComfyUI-native,
official-native, and Diffusers-reference paths separate and exposes A14B boundaries as metadata
only.  It never loads weights, dispatches experts, implements UniPC, or patches a model.
"""

from __future__ import annotations

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
    apply_terminal_policy,
    comfyui_simple_discrete_flow_grid,
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

WAN21_REPOSITORY_REVISION: Final = (
    "9737cba9c1c3c4d04b33fcad41c111989865d315"  # pragma: allowlist secret
)
WAN22_REPOSITORY_REVISION: Final = (
    "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"  # pragma: allowlist secret
)
WAN_ANIMATE2_REPOSITORY_REVISION: Final = (
    "3ad2fef7d61d6200c9c653e0fe47be7616b323f3"  # pragma: allowlist secret
)
WAN_COMFYUI_REVISION: Final = "5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25"  # pragma: allowlist secret
WAN_ANIMATE2_COMFYUI_REVISION: Final = (
    "76135e557da1ec7dcb270160f01e597565e3e003"  # pragma: allowlist secret
)
WAN_ANIMATE2_COMFY_WORKFLOW_REVISION: Final = (
    "e95e3b20567bea8df16510c8390b7f897b7e6d4b"  # pragma: allowlist secret
)
WAN_ANIMATE2_COMFY_MODEL_REVISION: Final = (
    "ed158470869ff31fa51cf56012dac33fb00f494b"  # pragma: allowlist secret
)
WAN_DIFFUSERS_REVISION: Final = (
    "3c468926ffd12b69baa4316e27b09306b8da19a6"  # pragma: allowlist secret
)
WAN21_T2V_MODEL_REVISION: Final = (
    "37ec512624d61f7aa208f7ea8140a131f93afc9a"  # pragma: allowlist secret
)
WAN21_I2V_480_MODEL_REVISION: Final = (
    "6b73f84e66371cdfe870c72acd6826e1d61cf279"  # pragma: allowlist secret
)
WAN21_I2V_720_MODEL_REVISION: Final = (
    "8823af45fcc58a8aa999a54b04be9abc7d2aac98"  # pragma: allowlist secret
)
WAN22_T2V_MODEL_REVISION: Final = (
    "5be7df9619b54f4e2667b2755bc6a756675b5cd7"  # pragma: allowlist secret
)
WAN22_I2V_MODEL_REVISION: Final = (
    "596658fd9ca6b7b71d5057529bbf319ecbc61d74"  # pragma: allowlist secret
)
WAN22_TI2V_MODEL_REVISION: Final = (
    "b8fff7315c768468a5333511427288870b2e9635"  # pragma: allowlist secret
)
WAN21_FLF2V_MODEL_REVISION: Final = (
    "c8db168d95d3ebeb63430b3b6d264885cb8a0df3"  # pragma: allowlist secret
)
WAN21_VACE_1_3B_MODEL_REVISION: Final = (
    "574e6a744642ce3bee319afc31496b88bde8aac4"  # pragma: allowlist secret
)
WAN21_VACE_14B_MODEL_REVISION: Final = (
    "539c162b1387eac9dc4c20bd3f74671309e76a4c"  # pragma: allowlist secret
)
WAN22_S2V_MODEL_REVISION: Final = (
    "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b"  # pragma: allowlist secret
)
WAN22_ANIMATE_MODEL_REVISION: Final = (
    "cb93a225fbaf1ca100f54e79da8f994995b689b3"  # pragma: allowlist secret
)
WAN_ANIMATE2_MODEL_REVISION: Final = (
    "6e8f1973bf0abc2aafd517992e8b6d88c3c46e69"  # pragma: allowlist secret
)

_WAN21_T2V_SHA256: Final = (
    "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"  # pragma: allowlist secret
)
_WAN21_I2V_480_SHA256: Final = (
    "09c0170242cfe9598208724585196ca18f294928fe25971149e1d7b37b3b51d6"  # pragma: allowlist secret
)
_WAN21_I2V_720_SHA256: Final = (
    "3c36c371b3060931770f693f22253a7de7c76fc79cffb0ab08032fb5a04784e4"  # pragma: allowlist secret
)
_WAN22_T2V_SHA256: Final = (
    "299e6304544f2783896372fa919e755a8bb9ab8caf898ce08a678dae391e1179"  # pragma: allowlist secret
)
_WAN22_I2V_SHA256: Final = (
    "0400c403bd7cfe6c1c29b47ff9cb575495dc590c9be2511a3b44ae5795add106"  # pragma: allowlist secret
)
_WAN22_TI2V_SHA256: Final = (
    "511bec832a201caa410d09c5ce7dbbf8ad2708c345d82038f684fc74cce982be"  # pragma: allowlist secret
)
_WAN21_FLF2V_SHA256: Final = (
    "c8644162efd3f6f7407daeff84f2e54f285cd3b2553e4c7282c0c7299c896df6"  # pragma: allowlist secret
)
_WAN21_VACE_1_3B_SHA256: Final = (
    "c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2"  # pragma: allowlist secret
)
_WAN21_VACE_14B_SHA256: Final = (
    "569d54a07279b89f8281421fccf27ee2459ea853ce6845d3536b8664b0070078"  # pragma: allowlist secret
)
_WAN22_S2V_SHA256: Final = (
    "5fb54febf10b729a6da7da222625d6ecaedde78becda01efbc13f6bebaeb6d43"  # pragma: allowlist secret
)
_WAN22_ANIMATE_SHA256: Final = (
    "575c2dba750c3b40240fb742a4224453aa97dfbd3c5f5a0086be431cdefdd69c"  # pragma: allowlist secret
)
_WAN_ANIMATE2_BASE_SHA256: Final = (
    "48abc389b8d9bba17a7f54a1cd7f1286fd3e3e0e292ddf756721aee324aede09"  # pragma: allowlist secret
)
_WAN_ANIMATE2_DISTILLED_SHA256: Final = (
    "66161359fc58a1d6c46e14fe2d81c881ccaca440757b1905e01a91902bff29d2"  # pragma: allowlist secret
)
_WAN_ANIMATE2_COMFY_MODEL_SHA256: Final = (
    "0580ecdd65e47e97c30df9670d13a6c4a131d26de5a1faf2ccc78392d5167584"  # pragma: allowlist secret
)
_WAN_ANIMATE2_COMFY_LORA_SHA256: Final = (
    "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3"  # pragma: allowlist secret
)

_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
_MAX_STEPS: Final = 10_000
_TRAINING_TIMESTEPS: Final = 1000
_BOUNDARY_TOLERANCE: Final = 1e-15


class WanGeneration(str, Enum):
    """Explicit Wan generation axis."""

    WAN21 = "wan2.1"
    WAN22 = "wan2.2"
    WAN_ANIMATE2 = "wan-animate2"


class WanTask(str, Enum):
    """Supported task families; derivative wrappers are intentionally absent."""

    T2V = "t2v"
    I2V = "i2v"
    TI2V = "ti2v"
    FLF2V = "flf2v"
    VACE = "vace"
    S2V = "s2v"
    ANIMATE = "animate"


class WanSource(str, Enum):
    """The complete owner of the selected Wan shift."""

    COMFY_NATIVE = "comfy_native"
    OFFICIAL_NATIVE = "official_native"
    DIFFUSERS_REFERENCE = "diffusers_reference"


class WanResolution(str, Enum):
    """Resolution classes required by the official Wan I2V profiles."""

    NONE = "none"
    P480 = "480p"
    P720 = "720p"


class WanProfileId(str, Enum):
    """Exact public profile identities in the first Wan support slice."""

    WAN21_COMFY_NATIVE = "wan2.1.t2v.comfy-native"
    WAN21_T2V_OFFICIAL = "wan2.1.t2v.official-native"
    WAN21_I2V_480P_OFFICIAL = "wan2.1.i2v.480p.official-native"
    WAN21_I2V_720P_OFFICIAL = "wan2.1.i2v.720p.official-native"
    WAN21_T2V_DIFFUSERS = "wan2.1.t2v.diffusers-reference"
    WAN21_I2V_480P_DIFFUSERS = "wan2.1.i2v.480p.diffusers-reference"
    WAN21_I2V_720P_DIFFUSERS = "wan2.1.i2v.720p.diffusers-reference"
    WAN22_TI2V_5B_NATIVE = "wan2.2.ti2v.5b.comfy-native"
    WAN22_T2V_A14B_NATIVE = "wan2.2.t2v-a14b.official-native"
    WAN22_I2V_A14B_NATIVE = "wan2.2.i2v-a14b.official-native"
    WAN22_TI2V_5B_DIFFUSERS = "wan2.2.ti2v.5b.diffusers-reference"
    WAN22_T2V_A14B_DIFFUSERS = "wan2.2.t2v-a14b.diffusers-reference"
    WAN22_I2V_A14B_DIFFUSERS = "wan2.2.i2v-a14b.diffusers-reference"
    WAN21_FLF2V_14B_720P_OFFICIAL = "wan2.1.flf2v.14b.720p.official-native"
    WAN21_VACE_1_3B_OFFICIAL = "wan2.1.vace.1.3b.official-native"
    WAN21_VACE_14B_OFFICIAL = "wan2.1.vace.14b.official-native"
    WAN22_S2V_14B_OFFICIAL = "wan2.2.s2v.14b.official-native"
    WAN22_ANIMATE_14B_OFFICIAL = "wan2.2.animate.14b.official-native"
    WAN_ANIMATE2_BASE_14B_OFFICIAL = "wan-animate2.14b.base.official-native"
    WAN_ANIMATE2_DISTILLED_14B_OFFICIAL = "wan-animate2.14b.distilled.official-native"
    WAN_ANIMATE2_COMFY_OPTIMIZED_6 = "wan-animate2.14b.comfy-optimized-6.framework-reference"


_M6_10_PROFILE_IDS: Final = frozenset(
    {
        WanProfileId.WAN21_FLF2V_14B_720P_OFFICIAL,
        WanProfileId.WAN21_VACE_1_3B_OFFICIAL,
        WanProfileId.WAN21_VACE_14B_OFFICIAL,
        WanProfileId.WAN22_S2V_14B_OFFICIAL,
    }
)

_M6_11_PROFILE_IDS: Final = frozenset(
    {
        WanProfileId.WAN22_ANIMATE_14B_OFFICIAL,
        WanProfileId.WAN_ANIMATE2_BASE_14B_OFFICIAL,
        WanProfileId.WAN_ANIMATE2_DISTILLED_14B_OFFICIAL,
    }
)

_M4_18_PROFILE_IDS: Final = frozenset({WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6})


@dataclass(frozen=True, slots=True, kw_only=True)
class WanEvidenceReference:
    """One pinned source lane for a Wan schedule profile."""

    lane: str
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.lane not in {
            "comfyui_implementation",
            "diffusers_framework",
            "official_wan21",
            "official_wan22",
            "official_wan_animate2",
            "comfyui_workflow",
            "comfyui_weights",
        }:
            raise ScheduleContractError("Wan evidence lane is unsupported")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("Wan evidence URL must use HTTPS")
        if not isinstance(self.revision, str) or not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("Wan evidence revision is invalid")
        if not self.locators or self.locators != tuple(sorted(set(self.locators))):
            raise ScheduleContractError("Wan evidence locators must be sorted and unique")


@dataclass(frozen=True, slots=True, kw_only=True)
class WanBoundary:
    """A descriptive A14B split index; it never performs model dispatch."""

    normalized: float
    transition_index: int
    crossing: str
    routing_owner: str = "caller"
    model_dispatch: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.normalized, bool)
            or not isinstance(self.normalized, int | float)
            or not math.isfinite(float(self.normalized))
            or not 0.0 < float(self.normalized) < 1.0
        ):
            raise ScheduleContractError("Wan boundary must be a finite normalized value in (0, 1)")
        if (
            not isinstance(self.transition_index, int)
            or isinstance(self.transition_index, bool)
            or self.transition_index < 0
        ):
            raise ScheduleContractError("Wan boundary transition index must be non-negative")
        if self.crossing not in {"at_or_above", "crossed_below"}:
            raise ScheduleContractError("Wan boundary crossing is unsupported")
        if self.routing_owner != "caller" or self.model_dispatch is not False:
            raise ScheduleContractError("Wan boundary cannot own model dispatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class WanScheduleResult:
    """Schedule result plus optional caller-owned A14B boundary metadata."""

    schedule: ScheduleResult
    boundary: WanBoundary | None

    @property
    def request(self) -> ScheduleRequest:
        return self.schedule.request

    @property
    def effective_inputs(self) -> ScheduleInputs:
        return self.schedule.effective_inputs

    @property
    def sigmas(self) -> tuple[float, ...]:
        return self.schedule.sigmas

    @property
    def final_domain(self) -> SigmaDomain:
        return self.schedule.final_domain

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.schedule.warnings


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
_MIT = LicenseDeclaration(
    declaration_version="1",
    identifier="MIT",
    name="MIT License",
    url="https://opensource.org/license/mit",
)
_WAN21_URL: Final = "https://github.com/Wan-Video/Wan2.1"
_WAN22_URL: Final = "https://github.com/Wan-Video/Wan2.2"
_WAN_ANIMATE2_URL: Final = "https://github.com/Wan-Video/Wan-Animate-2"
_COMFYUI_URL: Final = "https://github.com/Comfy-Org/ComfyUI"
_COMFYUI_WORKFLOWS_URL: Final = "https://github.com/Comfy-Org/workflow_templates"
_DIFFUSERS_URL: Final = "https://github.com/huggingface/diffusers"
_WAN21_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B"
_WAN21_I2V_480_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P"
_WAN21_I2V_720_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P"
_WAN22_T2V_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
_WAN22_I2V_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers"
_WAN22_TI2V_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
_WAN21_FLF2V_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.1-FLF2V-14B-720P"
_WAN21_VACE_1_3B_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B"
_WAN21_VACE_14B_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.1-VACE-14B"
_WAN22_S2V_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.2-S2V-14B"
_WAN22_ANIMATE_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B"
_WAN_ANIMATE2_HF_URL: Final = "https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B"
_WAN_ANIMATE2_COMFY_HF_URL: Final = (
    "https://huggingface.co/Comfy-Org/Wan_Animate_ComfyUI_repackaged"
)

_WAN21_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="wan.video.wan2-1.official",
    resource_version="2.1",
    revision=WAN21_REPOSITORY_REVISION,
    url=_WAN21_URL,
    license=_APACHE_2,
    locators=("LICENSE", "README.md", "wan/configs/wan_t2v_1.3B.py"),
)
_WAN22_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="wan.video.wan2-2.official",
    resource_version="2.2",
    revision=WAN22_REPOSITORY_REVISION,
    url=_WAN22_URL,
    license=_APACHE_2,
    locators=("LICENSE", "README.md", "wan/configs/wan_t2v_A14B.py"),
)
_WAN21_M6_10_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="wan.video.wan2-1.task-profiles.m6-10",
    resource_version="2.1",
    revision=WAN21_REPOSITORY_REVISION,
    url=_WAN21_URL,
    license=_APACHE_2,
    locators=("LICENSE", "generate.py"),
)
_WAN22_M6_10_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="wan.video.wan2-2.task-profiles.m6-10",
    resource_version="2.2",
    revision=WAN22_REPOSITORY_REVISION,
    url=_WAN22_URL,
    license=_APACHE_2,
    locators=("LICENSE", "generate.py", "wan/configs/wan_s2v_14B.py"),
)
_WAN22_M6_11_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="wan.video.wan2-2.animate.task-profiles.m6-11",
    resource_version="2.2-Animate",
    revision=WAN22_REPOSITORY_REVISION,
    url=_WAN22_URL,
    license=_APACHE_2,
    locators=("LICENSE", "generate.py", "wan/animate.py", "wan/configs/wan_animate_14B.py"),
)
_WAN_ANIMATE2_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="wan.video.wan-animate-2.task-profiles.m6-11",
    resource_version="2.0",
    revision=WAN_ANIMATE2_REPOSITORY_REVISION,
    url=_WAN_ANIMATE2_URL,
    license=_APACHE_2,
    locators=(
        "LICENSE",
        "README.md",
        "infer/wan_animate_2.yaml",
        "infer/wan_animate_2_demo.py",
        "infer/wan_animate_2_distillation.yaml",
        "pipelines/wan_animate_2_pipeline.py",
    ),
)
_WAN_ANIMATE2_COMFY_WORKFLOW_SOURCE = SoftwareSourceProvenance(
    record_version="1",
    source_id="comfyui.workflow.wan-animate-2.optimized.m4-18",
    resource_version="2026-08-22",
    revision=WAN_ANIMATE2_COMFY_WORKFLOW_REVISION,
    url=_COMFYUI_WORKFLOWS_URL,
    license=_MIT,
    locators=("LICENSE", "templates/video_wan_animate2.json"),
)
_COMFYUI_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.wan.framework",
    resource_version="0.29.0",
    revision=WAN_COMFYUI_REVISION,
    url=_COMFYUI_URL,
    license=_GPL_3_ONLY,
    locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
)
_DIFFUSERS_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="diffusers.wan.framework",
    resource_version="0.39.0",
    revision=WAN_DIFFUSERS_REVISION,
    url=_DIFFUSERS_URL,
    license=_APACHE_2,
    locators=(
        "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
        "src/diffusers/schedulers/scheduling_unipc_multistep.py",
    ),
)
_WAN21_NATIVE_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="wan.video.wan2-1.native-inference.m6-10",
    resource_version="2.1",
    revision=WAN21_REPOSITORY_REVISION,
    url=_WAN21_URL,
    license=_APACHE_2,
    locators=("generate.py", "wan/utils/fm_solvers.py"),
)
_WAN22_NATIVE_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="wan.video.wan2-2.native-inference.m6-10",
    resource_version="2.2",
    revision=WAN22_REPOSITORY_REVISION,
    url=_WAN22_URL,
    license=_APACHE_2,
    locators=("generate.py", "wan/configs/wan_s2v_14B.py", "wan/utils/fm_solvers.py"),
)
_WAN22_ANIMATE_NATIVE_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="wan.video.wan2-2.animate.native-inference.m6-11",
    resource_version="2.2-Animate",
    revision=WAN22_REPOSITORY_REVISION,
    url=_WAN22_URL,
    license=_APACHE_2,
    locators=("generate.py", "wan/animate.py", "wan/configs/wan_animate_14B.py"),
)
_WAN_ANIMATE2_NATIVE_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="wan.video.wan-animate-2.native-inference.m6-11",
    resource_version="2.0",
    revision=WAN_ANIMATE2_REPOSITORY_REVISION,
    url=_WAN_ANIMATE2_URL,
    license=_APACHE_2,
    locators=(
        "README.md",
        "infer/wan_animate_2.yaml",
        "infer/wan_animate_2_demo.py",
        "infer/wan_animate_2_distillation.yaml",
        "pipelines/wan_animate_2_pipeline.py",
    ),
)
_WAN_ANIMATE2_COMFY_FRAMEWORK = FrameworkProvenance(
    record_version="1",
    framework_id="comfyui.wan-animate-2.optimized.m4-18",
    resource_version=None,
    revision=WAN_ANIMATE2_COMFYUI_REVISION,
    url=_COMFYUI_URL,
    license=_GPL_3_ONLY,
    locators=(
        "comfy/model_sampling.py",
        "comfy/samplers.py",
        "comfy_extras/nodes_model_advanced.py",
        "comfy_extras/nodes_wan.py",
    ),
)


def _weight(
    *,
    weight_id: str,
    resource_version: str,
    revision: str,
    sha256: str,
    url: str,
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


_WAN21_T2V_WEIGHT = _weight(
    weight_id="wan-ai.wan2-1.t2v-1-3b.vae",
    resource_version="Wan2.1_VAE.pth",
    revision=WAN21_T2V_MODEL_REVISION,
    sha256=_WAN21_T2V_SHA256,
    url=_WAN21_HF_URL,
)
_WAN21_I2V_480_WEIGHT = _weight(
    weight_id="wan-ai.wan2-1.i2v-480p.transformer-01",
    resource_version="diffusion_pytorch_model-00001-of-00007.safetensors",
    revision=WAN21_I2V_480_MODEL_REVISION,
    sha256=_WAN21_I2V_480_SHA256,
    url=_WAN21_I2V_480_HF_URL,
)
_WAN21_I2V_720_WEIGHT = _weight(
    weight_id="wan-ai.wan2-1.i2v-720p.transformer-01",
    resource_version="diffusion_pytorch_model-00001-of-00007.safetensors",
    revision=WAN21_I2V_720_MODEL_REVISION,
    sha256=_WAN21_I2V_720_SHA256,
    url=_WAN21_I2V_720_HF_URL,
)
_WAN22_T2V_WEIGHT = _weight(
    weight_id="wan-ai.wan2-2.t2v-a14b.transformer-01",
    resource_version="transformer/diffusion_pytorch_model-00001-of-00012.safetensors",
    revision=WAN22_T2V_MODEL_REVISION,
    sha256=_WAN22_T2V_SHA256,
    url=_WAN22_T2V_HF_URL,
)
_WAN22_I2V_WEIGHT = _weight(
    weight_id="wan-ai.wan2-2.i2v-a14b.transformer-01",
    resource_version="transformer/diffusion_pytorch_model-00001-of-00012.safetensors",
    revision=WAN22_I2V_MODEL_REVISION,
    sha256=_WAN22_I2V_SHA256,
    url=_WAN22_I2V_HF_URL,
)
_WAN22_TI2V_WEIGHT = _weight(
    weight_id="wan-ai.wan2-2.ti2v-5b.transformer-01",
    resource_version="transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    revision=WAN22_TI2V_MODEL_REVISION,
    sha256=_WAN22_TI2V_SHA256,
    url=_WAN22_TI2V_HF_URL,
)
_WAN21_FLF2V_WEIGHT = _weight(
    weight_id="wan-ai.wan2-1.flf2v-14b-720p.transformer-01",
    resource_version="diffusion_pytorch_model-00001-of-00007.safetensors",
    revision=WAN21_FLF2V_MODEL_REVISION,
    sha256=_WAN21_FLF2V_SHA256,
    url=_WAN21_FLF2V_HF_URL,
)
_WAN21_VACE_1_3B_WEIGHT = _weight(
    weight_id="wan-ai.wan2-1.vace-1-3b.transformer",
    resource_version="diffusion_pytorch_model.safetensors",
    revision=WAN21_VACE_1_3B_MODEL_REVISION,
    sha256=_WAN21_VACE_1_3B_SHA256,
    url=_WAN21_VACE_1_3B_HF_URL,
)
_WAN21_VACE_14B_WEIGHT = _weight(
    weight_id="wan-ai.wan2-1.vace-14b.transformer-01",
    resource_version="diffusion_pytorch_model-00001-of-00007.safetensors",
    revision=WAN21_VACE_14B_MODEL_REVISION,
    sha256=_WAN21_VACE_14B_SHA256,
    url=_WAN21_VACE_14B_HF_URL,
)
_WAN22_S2V_WEIGHT = _weight(
    weight_id="wan-ai.wan2-2.s2v-14b.transformer-01",
    resource_version="diffusion_pytorch_model-00001-of-00004.safetensors",
    revision=WAN22_S2V_MODEL_REVISION,
    sha256=_WAN22_S2V_SHA256,
    url=_WAN22_S2V_HF_URL,
)
_WAN22_ANIMATE_WEIGHT = _weight(
    weight_id="wan-ai.wan2-2.animate-14b.transformer-01",
    resource_version="diffusion_pytorch_model-00001-of-00004.safetensors",
    revision=WAN22_ANIMATE_MODEL_REVISION,
    sha256=_WAN22_ANIMATE_SHA256,
    url=_WAN22_ANIMATE_HF_URL,
)
_WAN_ANIMATE2_BASE_WEIGHT = _weight(
    weight_id="wan-ai.wan-animate-2.14b.base",
    resource_version="wan_animate_2/wan_animate_2_bf16.safetensors",
    revision=WAN_ANIMATE2_MODEL_REVISION,
    sha256=_WAN_ANIMATE2_BASE_SHA256,
    url=_WAN_ANIMATE2_HF_URL,
)
_WAN_ANIMATE2_DISTILLED_WEIGHT = _weight(
    weight_id="wan-ai.wan-animate-2.14b.distilled",
    resource_version="wan_animate_2/wan_animate_2_bf16_distillation.safetensors",
    revision=WAN_ANIMATE2_MODEL_REVISION,
    sha256=_WAN_ANIMATE2_DISTILLED_SHA256,
    url=_WAN_ANIMATE2_HF_URL,
)
_WAN_ANIMATE2_COMFY_LORA_WEIGHT = _weight(
    weight_id="comfy-org.wan-animate-2.optimized.lightx2v-lora",
    resource_version="loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
    revision=WAN_ANIMATE2_COMFY_MODEL_REVISION,
    sha256=_WAN_ANIMATE2_COMFY_LORA_SHA256,
    url=_WAN_ANIMATE2_COMFY_HF_URL,
)
_WAN_ANIMATE2_COMFY_MODEL_WEIGHT = _weight(
    weight_id="comfy-org.wan-animate-2.optimized.model",
    resource_version="diffusion_models/wan_animate_2_int8_convrot.safetensors",
    revision=WAN_ANIMATE2_COMFY_MODEL_REVISION,
    sha256=_WAN_ANIMATE2_COMFY_MODEL_SHA256,
    url=_WAN_ANIMATE2_COMFY_HF_URL,
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
_COMFY_SIMPLE_BASE_GRID = BaseGridDeclaration(
    identifier="comfyui.simple_discrete_flow",
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
_DETECTION = DetectionDeclaration(
    strategy_id="wan.explicit-profile-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=("explicit_profile",),
    suggestion_sources=(),
    family_only_sources=(),
)


def _sampler(sampler_id: str, revision: str) -> SamplerCapabilities:
    return SamplerCapabilities(
        sampler_id=sampler_id,
        sampler_version=revision,
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


_COMFY_SAMPLER_ID: Final = "flowmatch.euler"
_UNIPC_SAMPLER_ID: Final = "unipc.multistep"
_FLOW_DPM_SAMPLER_ID: Final = "flow_dpm.multistep"
_LCM_SAMPLER_ID: Final = "lcm"


@dataclass(frozen=True, slots=True, kw_only=True)
class _WanDefinition:
    profile: WanProfileId
    generation: WanGeneration
    task: WanTask
    source: WanSource
    resolution: WanResolution
    ratio: float
    steps: int
    model_variant: str
    display_name: str
    evidence: EvidenceLevel
    primary_source_id: str
    sampler_id: str
    guidance: float
    guidance_mode: str = "cfg"
    cfg_low: float | None = None
    cfg_high: float | None = None
    boundary: float | None = None
    weight: ModelWeightProvenance = _WAN21_T2V_WEIGHT
    additional_weights: tuple[ModelWeightProvenance, ...] = ()
    exact_steps: bool = False


def _definition(
    profile: WanProfileId,
    generation: WanGeneration,
    task: WanTask,
    source: WanSource,
    resolution: WanResolution,
    ratio: float,
    steps: int,
    model_variant: str,
    display_name: str,
    evidence: EvidenceLevel,
    primary_source_id: str,
    sampler_id: str,
    guidance: float,
    *,
    guidance_mode: str = "cfg",
    cfg_low: float | None = None,
    cfg_high: float | None = None,
    boundary: float | None = None,
    weight: ModelWeightProvenance,
    additional_weights: tuple[ModelWeightProvenance, ...] = (),
    exact_steps: bool = False,
) -> _WanDefinition:
    return _WanDefinition(
        profile=profile,
        generation=generation,
        task=task,
        source=source,
        resolution=resolution,
        ratio=ratio,
        steps=steps,
        model_variant=model_variant,
        display_name=display_name,
        evidence=evidence,
        primary_source_id=primary_source_id,
        sampler_id=sampler_id,
        guidance=guidance,
        guidance_mode=guidance_mode,
        cfg_low=cfg_low,
        cfg_high=cfg_high,
        boundary=boundary,
        weight=weight,
        additional_weights=additional_weights,
        exact_steps=exact_steps,
    )


_DEFINITIONS: tuple[_WanDefinition, ...] = (
    _definition(
        WanProfileId.WAN21_COMFY_NATIVE,
        WanGeneration.WAN21,
        WanTask.T2V,
        WanSource.COMFY_NATIVE,
        WanResolution.NONE,
        8.0,
        50,
        "2.1-t2v-comfy-native",
        "Wan 2.1 T2V ComfyUI Native Shift",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _COMFYUI_FRAMEWORK.framework_id,
        _COMFY_SAMPLER_ID,
        5.0,
        weight=_WAN21_T2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_T2V_OFFICIAL,
        WanGeneration.WAN21,
        WanTask.T2V,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        5.0,
        50,
        "2.1-t2v",
        "Wan 2.1 T2V Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN21_SOURCE.source_id,
        _COMFY_SAMPLER_ID,
        5.0,
        weight=_WAN21_T2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_I2V_480P_OFFICIAL,
        WanGeneration.WAN21,
        WanTask.I2V,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.P480,
        3.0,
        40,
        "2.1-i2v-480p",
        "Wan 2.1 I2V 480P Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN21_SOURCE.source_id,
        _COMFY_SAMPLER_ID,
        5.0,
        weight=_WAN21_I2V_480_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_I2V_720P_OFFICIAL,
        WanGeneration.WAN21,
        WanTask.I2V,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.P720,
        5.0,
        40,
        "2.1-i2v-720p",
        "Wan 2.1 I2V 720P Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN21_SOURCE.source_id,
        _COMFY_SAMPLER_ID,
        5.0,
        weight=_WAN21_I2V_720_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_T2V_DIFFUSERS,
        WanGeneration.WAN21,
        WanTask.T2V,
        WanSource.DIFFUSERS_REFERENCE,
        WanResolution.NONE,
        3.0,
        50,
        "2.1-t2v-diffusers",
        "Wan 2.1 T2V Diffusers Schedule Reference",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _DIFFUSERS_FRAMEWORK.framework_id,
        _UNIPC_SAMPLER_ID,
        5.0,
        weight=_WAN21_T2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_I2V_480P_DIFFUSERS,
        WanGeneration.WAN21,
        WanTask.I2V,
        WanSource.DIFFUSERS_REFERENCE,
        WanResolution.P480,
        3.0,
        40,
        "2.1-i2v-480p-diffusers",
        "Wan 2.1 I2V 480P Diffusers Schedule Reference",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _DIFFUSERS_FRAMEWORK.framework_id,
        _UNIPC_SAMPLER_ID,
        5.0,
        weight=_WAN21_I2V_480_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_I2V_720P_DIFFUSERS,
        WanGeneration.WAN21,
        WanTask.I2V,
        WanSource.DIFFUSERS_REFERENCE,
        WanResolution.P720,
        5.0,
        40,
        "2.1-i2v-720p-diffusers",
        "Wan 2.1 I2V 720P Diffusers Schedule Reference",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _DIFFUSERS_FRAMEWORK.framework_id,
        _UNIPC_SAMPLER_ID,
        5.0,
        weight=_WAN21_I2V_720_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_TI2V_5B_NATIVE,
        WanGeneration.WAN22,
        WanTask.TI2V,
        WanSource.COMFY_NATIVE,
        WanResolution.NONE,
        5.0,
        50,
        "2.2-ti2v-5b",
        "Wan 2.2 TI2V 5B Native Shift",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _COMFYUI_FRAMEWORK.framework_id,
        _COMFY_SAMPLER_ID,
        5.0,
        weight=_WAN22_TI2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_T2V_A14B_NATIVE,
        WanGeneration.WAN22,
        WanTask.T2V,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        12.0,
        40,
        "2.2-t2v-a14b",
        "Wan 2.2 T2V A14B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN22_SOURCE.source_id,
        _COMFY_SAMPLER_ID,
        3.0,
        cfg_low=3.0,
        cfg_high=4.0,
        boundary=0.875,
        weight=_WAN22_T2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_I2V_A14B_NATIVE,
        WanGeneration.WAN22,
        WanTask.I2V,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        5.0,
        40,
        "2.2-i2v-a14b",
        "Wan 2.2 I2V A14B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN22_SOURCE.source_id,
        _COMFY_SAMPLER_ID,
        3.5,
        cfg_low=3.5,
        cfg_high=3.5,
        boundary=0.9,
        weight=_WAN22_I2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_TI2V_5B_DIFFUSERS,
        WanGeneration.WAN22,
        WanTask.TI2V,
        WanSource.DIFFUSERS_REFERENCE,
        WanResolution.NONE,
        5.0,
        50,
        "2.2-ti2v-5b-diffusers",
        "Wan 2.2 TI2V 5B Diffusers Schedule Reference",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _DIFFUSERS_FRAMEWORK.framework_id,
        _UNIPC_SAMPLER_ID,
        5.0,
        weight=_WAN22_TI2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_T2V_A14B_DIFFUSERS,
        WanGeneration.WAN22,
        WanTask.T2V,
        WanSource.DIFFUSERS_REFERENCE,
        WanResolution.NONE,
        3.0,
        40,
        "2.2-t2v-a14b-diffusers",
        "Wan 2.2 T2V A14B Diffusers Schedule Reference",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _DIFFUSERS_FRAMEWORK.framework_id,
        _UNIPC_SAMPLER_ID,
        3.0,
        boundary=0.875,
        weight=_WAN22_T2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_I2V_A14B_DIFFUSERS,
        WanGeneration.WAN22,
        WanTask.I2V,
        WanSource.DIFFUSERS_REFERENCE,
        WanResolution.NONE,
        3.0,
        40,
        "2.2-i2v-a14b-diffusers",
        "Wan 2.2 I2V A14B Diffusers Schedule Reference",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _DIFFUSERS_FRAMEWORK.framework_id,
        _UNIPC_SAMPLER_ID,
        3.5,
        boundary=0.9,
        weight=_WAN22_I2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_FLF2V_14B_720P_OFFICIAL,
        WanGeneration.WAN21,
        WanTask.FLF2V,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.P720,
        16.0,
        50,
        "14b",
        "Wan 2.1 FLF2V 14B 720P Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN21_M6_10_SOURCE.source_id,
        _UNIPC_SAMPLER_ID,
        5.0,
        weight=_WAN21_FLF2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_VACE_1_3B_OFFICIAL,
        WanGeneration.WAN21,
        WanTask.VACE,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        16.0,
        50,
        "1.3b",
        "Wan 2.1 VACE 1.3B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN21_M6_10_SOURCE.source_id,
        _UNIPC_SAMPLER_ID,
        5.0,
        weight=_WAN21_VACE_1_3B_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN21_VACE_14B_OFFICIAL,
        WanGeneration.WAN21,
        WanTask.VACE,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        16.0,
        50,
        "14b",
        "Wan 2.1 VACE 14B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN21_M6_10_SOURCE.source_id,
        _UNIPC_SAMPLER_ID,
        5.0,
        weight=_WAN21_VACE_14B_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_S2V_14B_OFFICIAL,
        WanGeneration.WAN22,
        WanTask.S2V,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        3.0,
        40,
        "14b",
        "Wan 2.2 S2V 14B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN22_M6_10_SOURCE.source_id,
        _UNIPC_SAMPLER_ID,
        4.5,
        weight=_WAN22_S2V_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN22_ANIMATE_14B_OFFICIAL,
        WanGeneration.WAN22,
        WanTask.ANIMATE,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        5.0,
        20,
        "14b",
        "Wan 2.2 Animate 14B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN22_M6_11_SOURCE.source_id,
        _UNIPC_SAMPLER_ID,
        1.0,
        guidance_mode="no_cfg",
        weight=_WAN22_ANIMATE_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN_ANIMATE2_BASE_14B_OFFICIAL,
        WanGeneration.WAN_ANIMATE2,
        WanTask.ANIMATE,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        5.0,
        40,
        "base-14b",
        "Wan Animate 2 Base 14B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN_ANIMATE2_SOURCE.source_id,
        _FLOW_DPM_SAMPLER_ID,
        3.0,
        weight=_WAN_ANIMATE2_BASE_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN_ANIMATE2_DISTILLED_14B_OFFICIAL,
        WanGeneration.WAN_ANIMATE2,
        WanTask.ANIMATE,
        WanSource.OFFICIAL_NATIVE,
        WanResolution.NONE,
        5.0,
        10,
        "distilled-14b",
        "Wan Animate 2 Distilled 14B Official Native Shift",
        EvidenceLevel.OFFICIAL,
        _WAN_ANIMATE2_SOURCE.source_id,
        _FLOW_DPM_SAMPLER_ID,
        1.0,
        guidance_mode="no_cfg",
        weight=_WAN_ANIMATE2_DISTILLED_WEIGHT,
    ),
    _definition(
        WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6,
        WanGeneration.WAN_ANIMATE2,
        WanTask.ANIMATE,
        WanSource.COMFY_NATIVE,
        WanResolution.P480,
        5.0,
        6,
        "comfy-optimized-14b-480p",
        "Wan Animate 2 Comfy Optimized 14B 480P Six-Step",
        EvidenceLevel.FRAMEWORK_REFERENCE,
        _WAN_ANIMATE2_COMFY_WORKFLOW_SOURCE.source_id,
        _LCM_SAMPLER_ID,
        1.0,
        guidance_mode="no_cfg",
        weight=_WAN_ANIMATE2_COMFY_LORA_WEIGHT,
        additional_weights=(_WAN_ANIMATE2_COMFY_MODEL_WEIGHT,),
        exact_steps=True,
    ),
)


def _references(*, include_animate2: bool = False) -> tuple[WanEvidenceReference, ...]:
    references = [
        WanEvidenceReference(
            lane="comfyui_implementation",
            url=_COMFYUI_URL,
            revision=WAN_COMFYUI_REVISION,
            locators=("comfy/model_sampling.py", "comfy/supported_models.py"),
        ),
        WanEvidenceReference(
            lane="diffusers_framework",
            url=_DIFFUSERS_URL,
            revision=WAN_DIFFUSERS_REVISION,
            locators=(
                "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
                "src/diffusers/schedulers/scheduling_unipc_multistep.py",
            ),
        ),
        WanEvidenceReference(
            lane="official_wan21",
            url=_WAN21_URL,
            revision=WAN21_REPOSITORY_REVISION,
            locators=("LICENSE", "README.md", "wan/configs/wan_t2v_1.3B.py"),
        ),
        WanEvidenceReference(
            lane="official_wan22",
            url=_WAN22_URL,
            revision=WAN22_REPOSITORY_REVISION,
            locators=("LICENSE", "README.md", "wan/configs/wan_t2v_A14B.py"),
        ),
    ]
    if include_animate2:
        references.append(
            WanEvidenceReference(
                lane="official_wan_animate2",
                url=_WAN_ANIMATE2_URL,
                revision=WAN_ANIMATE2_REPOSITORY_REVISION,
                locators=(
                    "LICENSE",
                    "README.md",
                    "infer/wan_animate_2.yaml",
                    "infer/wan_animate_2_demo.py",
                    "infer/wan_animate_2_distillation.yaml",
                    "pipelines/wan_animate_2_pipeline.py",
                ),
            )
        )
    return tuple(references)


_REFERENCES = _references()
_ANIMATE2_REFERENCES = _references(include_animate2=True)
_COMFY_OPTIMIZED_REFERENCES = (
    WanEvidenceReference(
        lane="comfyui_implementation",
        url=_COMFYUI_URL,
        revision=WAN_ANIMATE2_COMFYUI_REVISION,
        locators=(
            "comfy/model_sampling.py",
            "comfy/samplers.py",
            "comfy_extras/nodes_model_advanced.py",
            "comfy_extras/nodes_wan.py",
        ),
    ),
    WanEvidenceReference(
        lane="comfyui_weights",
        url=_WAN_ANIMATE2_COMFY_HF_URL,
        revision=WAN_ANIMATE2_COMFY_MODEL_REVISION,
        locators=("README.md",),
    ),
    WanEvidenceReference(
        lane="comfyui_workflow",
        url=_COMFYUI_WORKFLOWS_URL,
        revision=WAN_ANIMATE2_COMFY_WORKFLOW_REVISION,
        locators=("LICENSE", "templates/video_wan_animate2.json"),
    ),
    WanEvidenceReference(
        lane="official_wan_animate2",
        url=_WAN_ANIMATE2_URL,
        revision=WAN_ANIMATE2_REPOSITORY_REVISION,
        locators=(
            "LICENSE",
            "README.md",
            "infer/wan_animate_2_demo.py",
            "pipelines/wan_animate_2_pipeline.py",
        ),
    ),
)


def _source_for(definition: _WanDefinition) -> str:
    if definition.profile in _M4_18_PROFILE_IDS:
        return _COMFYUI_WORKFLOWS_URL
    if definition.source is WanSource.COMFY_NATIVE:
        return _COMFYUI_URL
    if definition.source is WanSource.DIFFUSERS_REFERENCE:
        return _DIFFUSERS_URL
    if definition.generation is WanGeneration.WAN21:
        return _WAN21_URL
    if definition.generation is WanGeneration.WAN22:
        return _WAN22_URL
    return _WAN_ANIMATE2_URL


def _revision_for(definition: _WanDefinition) -> str:
    if definition.profile in _M4_18_PROFILE_IDS:
        return WAN_ANIMATE2_COMFY_WORKFLOW_REVISION
    if definition.source is WanSource.COMFY_NATIVE:
        return WAN_COMFYUI_REVISION
    if definition.source is WanSource.DIFFUSERS_REFERENCE:
        return WAN_DIFFUSERS_REVISION
    if definition.generation is WanGeneration.WAN21:
        return WAN21_REPOSITORY_REVISION
    if definition.generation is WanGeneration.WAN22:
        return WAN22_REPOSITORY_REVISION
    return WAN_ANIMATE2_REPOSITORY_REVISION


def _schema(definition: _WanDefinition) -> ProfileSchemaV1:
    model = ModelCapabilities(
        model_family="wan",
        model_variant=definition.model_variant,
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id=definition.profile.value,
        profile_version="1",
        model_family="wan",
        model_variant=definition.model_variant,
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        terminal_sigma=TerminalSigma.ZERO,
        allowed_execution_behaviors=(ExecutionBehavior.DETERMINISTIC,),
        allowed_noise_ownerships=(NoiseOwnership.NONE,),
        allowed_sampler_state=(),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
        reference_sampler_ids=(definition.sampler_id,),
    )
    frameworks: tuple[FrameworkProvenance, ...]
    if definition.profile in _M4_18_PROFILE_IDS:
        source_ids = (_WAN_ANIMATE2_COMFY_WORKFLOW_SOURCE,)
        frameworks = (_WAN_ANIMATE2_COMFY_FRAMEWORK,)
    elif definition.profile in _M6_10_PROFILE_IDS:
        source_ids = (
            (_WAN21_M6_10_SOURCE,)
            if definition.generation is WanGeneration.WAN21
            else (_WAN22_M6_10_SOURCE,)
        )
        frameworks = (
            (_WAN21_NATIVE_FRAMEWORK,)
            if definition.generation is WanGeneration.WAN21
            else (_WAN22_NATIVE_FRAMEWORK,)
        )
    elif definition.profile in _M6_11_PROFILE_IDS:
        if definition.profile is WanProfileId.WAN22_ANIMATE_14B_OFFICIAL:
            source_ids = (_WAN22_M6_11_SOURCE,)
            frameworks = (_WAN22_ANIMATE_NATIVE_FRAMEWORK,)
        else:
            source_ids = (_WAN_ANIMATE2_SOURCE,)
            frameworks = (_WAN_ANIMATE2_NATIVE_FRAMEWORK,)
    else:
        source_ids = (
            (_WAN21_SOURCE,) if definition.generation is WanGeneration.WAN21 else (_WAN22_SOURCE,)
        )
        frameworks = (_COMFYUI_FRAMEWORK, _DIFFUSERS_FRAMEWORK)
    parameters: list[ProfileField] = [
        ProfileField(name="generation", value=definition.generation.value),
        ProfileField(name="resolution_class", value=definition.resolution.value),
        ProfileField(name="shift", value=definition.ratio),
        ProfileField(name="solver", value=definition.sampler_id),
        ProfileField(name="source_mode", value=definition.source.value),
        ProfileField(name="task", value=definition.task.value),
        ProfileField(name="training_timesteps", value=_TRAINING_TIMESTEPS),
    ]
    if definition.boundary is not None:
        parameters.append(ProfileField(name="boundary", value=definition.boundary))
    if definition.cfg_high is not None:
        parameters.append(ProfileField(name="cfg_high", value=definition.cfg_high))
    if definition.cfg_low is not None:
        parameters.append(ProfileField(name="cfg_low", value=definition.cfg_low))
    if definition.profile in _M6_10_PROFILE_IDS:
        parameters.append(ProfileField(name="solver_options", value="dpm++,unipc"))
    if definition.profile in _M6_11_PROFILE_IDS:
        parameters.append(ProfileField(name="guidance_mode", value=definition.guidance_mode))
        parameters.append(
            ProfileField(
                name="solver_options",
                value=(
                    "dpm++,unipc"
                    if definition.profile is WanProfileId.WAN22_ANIMATE_14B_OFFICIAL
                    else "flow_dpm"
                ),
            )
        )
    if definition.profile in _M4_18_PROFILE_IDS:
        parameters.extend(
            (
                ProfileField(name="fps_max", value=24),
                ProfileField(name="fps_min", value=16),
                ProfileField(name="frame_overlap", value=1),
                ProfileField(name="frame_step", value=4),
                ProfileField(name="guidance_mode", value=definition.guidance_mode),
                ProfileField(name="recommended_frames", value=81),
                ProfileField(name="scheduler", value="simple"),
                ProfileField(name="solver_options", value="lcm"),
            )
        )
    parameters = sorted(parameters, key=lambda field: field.name)
    limitations = [
        "Only the exact released Wan generation/task/source matrix is qualified; derivative wrappers and weak-name aliases fail closed.",
        "The direct-ratio shift owns the complete primary transform and cannot be composed with another shift or already-shifted sigmas.",
        "Model weights, text conditioning, video execution, and visual quality are not verified by this schedule profile.",
    ]
    if definition.source is WanSource.DIFFUSERS_REFERENCE:
        limitations.append(
            "The Diffusers profile describes the UniPC scheduler sigma/timestep contract only; it does not establish UniPC solver parity under an Euler sampler."
        )
    if definition.boundary is not None:
        limitations.append(
            "The A14B boundary is caller-owned metadata; Sigmax never selects a high/low expert or dispatches a model."
        )
    if (
        definition.resolution is not WanResolution.NONE
        and definition.profile not in _M4_18_PROFILE_IDS
    ):
        limitations.append(
            "The resolution class is required because the official Wan I2V shift is resolution-sensitive."
        )
    if definition.profile in _M4_18_PROFILE_IDS:
        limitations.extend(
            (
                "The exact ComfyUI model and step-distilled LoRA are recipe requirements; substituting either artifact is outside this framework-reference profile.",
                "Sigmax constructs the source-exact simple sigmas but does not implement or execute LCM, load weights, or apply the LoRA.",
                "Eighty-one frames is tested chunk guidance with one-frame continuation overlap, not a hard ComfyUI node maximum or an automatic segmentation feature.",
            )
        )
    elif definition.profile in _M6_10_PROFILE_IDS:
        limitations.append(
            "The official CLI defaults to UniPC and also permits DPM++; this profile constructs sigmas but implements neither solver."
        )
    elif definition.profile in _M6_11_PROFILE_IDS:
        if definition.profile is WanProfileId.WAN22_ANIMATE_14B_OFFICIAL:
            limitations.append(
                "Wan 2.2 Animate's native CLI defaults to UniPC and permits DPM++; Sigmax constructs sigmas but implements neither solver."
            )
        else:
            limitations.append(
                "The native Animate-2 pipeline owns FlowDPM and constructs its reciprocal shifted sigmas; Sigmax records that ownership but does not implement the solver."
            )
    else:
        limitations.append(
            "Shift-16 FLF2V/VACE and all other derivatives remain a named Phase 1 defer condition pending independent evidence."
        )
    guidance_convention = "cfg_scale" if definition.guidance_mode == "cfg" else "none"
    guidance = GuidanceDeclaration(
        model_convention=guidance_convention,
        host_convention=guidance_convention,
        model_value=definition.guidance,
        host_value=definition.guidance,
    )
    recipe = InferenceRecipe(
        recipe_id=definition.profile.value,
        evidence=definition.evidence,
        source_id=definition.primary_source_id,
        steps=StepRangeDeclaration(
            minimum=definition.steps if definition.exact_steps else 1,
            maximum=definition.steps if definition.exact_steps else _MAX_STEPS,
            default=definition.steps,
            reference_steps=(definition.steps,),
            allow_modified=not definition.exact_steps,
        ),
        guidance=guidance,
    )
    return ProfileSchemaV1(
        schema_id=PROFILE_SCHEMA_ID,
        schema_version=PROFILE_SCHEMA_VERSION,
        profile_id=definition.profile.value,
        profile_version="1",
        display_name=definition.display_name,
        model_family="wan",
        model_variant=definition.model_variant,
        evidence=definition.evidence,
        primary_source_id=definition.primary_source_id,
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        base_grid=(
            _COMFY_SIMPLE_BASE_GRID if definition.profile in _M4_18_PROFILE_IDS else _BASE_GRID
        ),
        transforms=(
            TransformDeclaration(
                identifier="direct_ratio.shift",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
                parameters=(ProfileField(name="ratio", value=definition.ratio),),
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
        recipes=(recipe,),
        detection=_DETECTION,
        model_capabilities=model,
        profile_capabilities=profile,
        reference_sampler_capabilities=_sampler(definition.sampler_id, _revision_for(definition)),
        artifact_versions=_ARTIFACT_VERSIONS,
        software_sources=tuple(sorted(source_ids, key=lambda item: item.source_id)),
        frameworks=tuple(sorted(frameworks, key=lambda item: item.framework_id)),
        model_weights=tuple(
            sorted(
                (definition.weight, *definition.additional_weights),
                key=lambda item: item.weight_id,
            )
        ),
        parameters=tuple(parameters),
        known_limitations=tuple(limitations),
    )


_SCHEMAS_BY_PROFILE = {definition.profile: _schema(definition) for definition in _DEFINITIONS}

WAN21_COMFY_NATIVE_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_COMFY_NATIVE]
WAN21_T2V_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_T2V_OFFICIAL]
WAN21_I2V_480P_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_I2V_480P_OFFICIAL]
WAN21_I2V_720P_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_I2V_720P_OFFICIAL]
WAN21_T2V_DIFFUSERS_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_T2V_DIFFUSERS]
WAN21_I2V_480P_DIFFUSERS_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_I2V_480P_DIFFUSERS]
WAN21_I2V_720P_DIFFUSERS_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_I2V_720P_DIFFUSERS]
WAN22_TI2V_5B_NATIVE_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN22_TI2V_5B_NATIVE]
WAN22_T2V_A14B_NATIVE_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN22_T2V_A14B_NATIVE]
WAN22_I2V_A14B_NATIVE_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN22_I2V_A14B_NATIVE]
WAN22_TI2V_5B_DIFFUSERS_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN22_TI2V_5B_DIFFUSERS]
WAN22_T2V_A14B_DIFFUSERS_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN22_T2V_A14B_DIFFUSERS]
WAN22_I2V_A14B_DIFFUSERS_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN22_I2V_A14B_DIFFUSERS]
WAN21_FLF2V_14B_720P_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[
    WanProfileId.WAN21_FLF2V_14B_720P_OFFICIAL
]
WAN21_VACE_1_3B_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_VACE_1_3B_OFFICIAL]
WAN21_VACE_14B_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN21_VACE_14B_OFFICIAL]
WAN22_S2V_14B_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[WanProfileId.WAN22_S2V_14B_OFFICIAL]
WAN22_ANIMATE_14B_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[
    WanProfileId.WAN22_ANIMATE_14B_OFFICIAL
]
WAN_ANIMATE2_BASE_14B_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[
    WanProfileId.WAN_ANIMATE2_BASE_14B_OFFICIAL
]
WAN_ANIMATE2_DISTILLED_14B_OFFICIAL_SCHEMA: Final = _SCHEMAS_BY_PROFILE[
    WanProfileId.WAN_ANIMATE2_DISTILLED_14B_OFFICIAL
]
WAN_ANIMATE2_COMFY_OPTIMIZED_6_SCHEMA: Final = _SCHEMAS_BY_PROFILE[
    WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6
]


@dataclass(frozen=True, slots=True, kw_only=True)
class WanProfile:
    """Immutable Wan profile plus its pinned evidence lanes."""

    profile: WanProfileId
    schema: ProfileSchemaV1
    references: tuple[WanEvidenceReference, ...]

    @property
    def profile_id(self) -> str:
        return self.schema.profile_id

    @property
    def profile_version(self) -> str:
        return self.schema.profile_version

    def __post_init__(self) -> None:
        if _SCHEMAS_BY_PROFILE.get(self.profile) is not self.schema:
            raise ScheduleContractError("Wan profile/schema mismatch")
        lanes = tuple(reference.lane for reference in self.references)
        required_lanes = (
            5
            if self.profile
            in {
                WanProfileId.WAN_ANIMATE2_BASE_14B_OFFICIAL,
                WanProfileId.WAN_ANIMATE2_DISTILLED_14B_OFFICIAL,
            }
            else 4
        )
        if lanes != tuple(sorted(set(lanes))) or len(lanes) != required_lanes:
            raise ScheduleContractError(f"Wan requires {required_lanes} pinned evidence lanes")


_PROFILES_BY_ID = {
    profile: WanProfile(
        profile=profile,
        schema=schema,
        references=(
            _COMFY_OPTIMIZED_REFERENCES
            if profile in _M4_18_PROFILE_IDS
            else _ANIMATE2_REFERENCES
            if profile
            in {
                WanProfileId.WAN_ANIMATE2_BASE_14B_OFFICIAL,
                WanProfileId.WAN_ANIMATE2_DISTILLED_14B_OFFICIAL,
            }
            else _REFERENCES
        ),
    )
    for profile, schema in _SCHEMAS_BY_PROFILE.items()
}

WAN21_COMFY_NATIVE_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_COMFY_NATIVE]
WAN21_T2V_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_T2V_OFFICIAL]
WAN21_I2V_480P_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_I2V_480P_OFFICIAL]
WAN21_I2V_720P_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_I2V_720P_OFFICIAL]
WAN21_T2V_DIFFUSERS_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_T2V_DIFFUSERS]
WAN21_I2V_480P_DIFFUSERS_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_I2V_480P_DIFFUSERS]
WAN21_I2V_720P_DIFFUSERS_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_I2V_720P_DIFFUSERS]
WAN22_TI2V_5B_NATIVE_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_TI2V_5B_NATIVE]
WAN22_T2V_A14B_NATIVE_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_T2V_A14B_NATIVE]
WAN22_I2V_A14B_NATIVE_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_I2V_A14B_NATIVE]
WAN22_TI2V_5B_DIFFUSERS_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_TI2V_5B_DIFFUSERS]
WAN22_T2V_A14B_DIFFUSERS_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_T2V_A14B_DIFFUSERS]
WAN22_I2V_A14B_DIFFUSERS_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_I2V_A14B_DIFFUSERS]
WAN21_FLF2V_14B_720P_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[
    WanProfileId.WAN21_FLF2V_14B_720P_OFFICIAL
]
WAN21_VACE_1_3B_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_VACE_1_3B_OFFICIAL]
WAN21_VACE_14B_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN21_VACE_14B_OFFICIAL]
WAN22_S2V_14B_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_S2V_14B_OFFICIAL]
WAN22_ANIMATE_14B_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[WanProfileId.WAN22_ANIMATE_14B_OFFICIAL]
WAN_ANIMATE2_BASE_14B_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[
    WanProfileId.WAN_ANIMATE2_BASE_14B_OFFICIAL
]
WAN_ANIMATE2_DISTILLED_14B_OFFICIAL_PROFILE: Final = _PROFILES_BY_ID[
    WanProfileId.WAN_ANIMATE2_DISTILLED_14B_OFFICIAL
]
WAN_ANIMATE2_COMFY_OPTIMIZED_6_PROFILE: Final = _PROFILES_BY_ID[
    WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6
]


def _coerce_profile(value: object) -> WanProfileId:
    if isinstance(value, WanProfileId):
        return value
    raise ScheduleContractError("profile must be an explicit WanProfileId")


def _coerce_resolution(value: object, *, default: WanResolution) -> WanResolution:
    if value is None:
        return default
    if isinstance(value, WanResolution):
        return value
    if isinstance(value, str):
        try:
            return WanResolution(value)
        except ValueError as exc:
            raise ScheduleContractError("resolution is unsupported") from exc
    raise ScheduleContractError("resolution is unsupported")


def derive_wan_boundary(*, sigmas: tuple[float, ...], normalized_boundary: float) -> WanBoundary:
    """Return the first low-noise transition at or below a normalized boundary."""

    if not isinstance(sigmas, tuple) or len(sigmas) < 2:
        raise ScheduleContractError("Wan boundary requires a terminal-inclusive sigma tuple")
    if (
        isinstance(normalized_boundary, bool)
        or not isinstance(normalized_boundary, int | float)
        or not math.isfinite(float(normalized_boundary))
        or not 0.0 < float(normalized_boundary) < 1.0
    ):
        raise ScheduleContractError("Wan boundary must be a finite normalized value in (0, 1)")
    previous = float("inf")
    boundary = float(normalized_boundary)
    normalized_sigmas: list[float] = []
    for value in sigmas:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) > previous
        ):
            raise ScheduleContractError("Wan boundary sigmas must be finite and descending")
        current = float(value)
        normalized_sigmas.append(current)
        previous = current
    for index, current in enumerate(normalized_sigmas):
        if current <= boundary:
            crossing = (
                "at_or_above"
                if math.isclose(current, boundary, rel_tol=0.0, abs_tol=_BOUNDARY_TOLERANCE)
                else "crossed_below"
            )
            return WanBoundary(
                normalized=boundary,
                transition_index=index,
                crossing=crossing,
            )
    raise ScheduleContractError("Wan boundary is outside the constructed schedule")


def _definition_for(profile: WanProfileId) -> _WanDefinition:
    for definition in _DEFINITIONS:
        if definition.profile is profile:
            return definition
    raise ScheduleContractError("profile is unsupported")


def build_wan_schedule(
    *,
    profile: WanProfileId,
    steps: int,
    resolution: WanResolution | str | None = None,
    strict_source: bool = False,
    already_shifted: bool = False,
) -> WanScheduleResult:
    """Build one explicit Wan schedule and caller-owned A14B boundary metadata."""

    selected = _coerce_profile(profile)
    definition = _definition_for(selected)
    if not isinstance(strict_source, bool) or not isinstance(already_shifted, bool):
        raise ScheduleContractError("strict_source and already_shifted must be boolean")
    if already_shifted:
        raise ScheduleContractError("already shifted sigmas cannot be composed with Wan shift")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    selected_resolution = _coerce_resolution(resolution, default=WanResolution.NONE)
    if selected_resolution is not definition.resolution:
        if definition.resolution is not WanResolution.NONE:
            raise ScheduleContractError(
                f"resolution {definition.resolution.value} is required for {selected.value}"
            )
        raise ScheduleContractError("resolution must be none for this Wan profile")
    if definition.exact_steps and steps != definition.steps:
        raise ScheduleContractError(
            f"{definition.profile.value} requires exactly {definition.steps} steps"
        )
    if strict_source and steps != definition.steps:
        raise ScheduleContractError(
            f"steps must equal the pinned {definition.profile.value} {definition.steps}-step recipe"
        )
    evidence = definition.evidence if steps == definition.steps else EvidenceLevel.MODIFIED
    warnings = (
        ()
        if evidence is not EvidenceLevel.MODIFIED
        else (
            f"steps differ from the pinned {definition.profile.value} {definition.steps}-step recipe; evidence is modified",
        )
    )
    base_grid = (
        comfyui_simple_discrete_flow_grid(
            steps,
            training_timesteps=_TRAINING_TIMESTEPS,
            domain=SigmaDomain.UNIT_FLOW,
        )
        if definition.profile in _M4_18_PROFILE_IDS
        else flowmatch_reciprocal_step_grid(steps)
    )
    shifted = direct_ratio_shift(
        base_grid,
        ratio=definition.ratio,
        domain=SigmaDomain.UNIT_FLOW,
    )
    sigmas = apply_terminal_policy(
        shifted, policy=TerminalPolicy.APPEND_ZERO, domain=SigmaDomain.UNIT_FLOW
    )
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=steps),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version=VERSION,
            evidence=evidence,
            source=_source_for(definition),
            source_revision=_revision_for(definition),
            profile_id=definition.profile.value,
            profile_version="1",
        ),
        base_grid=BaseGridSpec(
            identifier=(
                "comfyui.simple_discrete_flow"
                if definition.profile in _M4_18_PROFILE_IDS
                else "flowmatch.reciprocal_step"
            ),
            output_domain=SigmaDomain.UNIT_FLOW,
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
    schedule = ScheduleResult(
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
    boundary = (
        derive_wan_boundary(sigmas=schedule.sigmas, normalized_boundary=definition.boundary)
        if definition.boundary is not None
        else None
    )
    return WanScheduleResult(schedule=schedule, boundary=boundary)


__all__ = [
    "WAN21_COMFY_NATIVE_PROFILE",
    "WAN21_COMFY_NATIVE_SCHEMA",
    "WAN21_FLF2V_14B_720P_OFFICIAL_PROFILE",
    "WAN21_FLF2V_14B_720P_OFFICIAL_SCHEMA",
    "WAN21_FLF2V_MODEL_REVISION",
    "WAN21_I2V_480P_DIFFUSERS_PROFILE",
    "WAN21_I2V_480P_DIFFUSERS_SCHEMA",
    "WAN21_I2V_480P_OFFICIAL_PROFILE",
    "WAN21_I2V_480P_OFFICIAL_SCHEMA",
    "WAN21_I2V_720P_DIFFUSERS_PROFILE",
    "WAN21_I2V_720P_DIFFUSERS_SCHEMA",
    "WAN21_I2V_720P_OFFICIAL_PROFILE",
    "WAN21_I2V_720P_OFFICIAL_SCHEMA",
    "WAN21_REPOSITORY_REVISION",
    "WAN21_T2V_DIFFUSERS_PROFILE",
    "WAN21_T2V_DIFFUSERS_SCHEMA",
    "WAN21_T2V_OFFICIAL_PROFILE",
    "WAN21_T2V_OFFICIAL_SCHEMA",
    "WAN21_VACE_1_3B_MODEL_REVISION",
    "WAN21_VACE_1_3B_OFFICIAL_PROFILE",
    "WAN21_VACE_1_3B_OFFICIAL_SCHEMA",
    "WAN21_VACE_14B_MODEL_REVISION",
    "WAN21_VACE_14B_OFFICIAL_PROFILE",
    "WAN21_VACE_14B_OFFICIAL_SCHEMA",
    "WAN22_ANIMATE_14B_OFFICIAL_PROFILE",
    "WAN22_ANIMATE_14B_OFFICIAL_SCHEMA",
    "WAN22_ANIMATE_MODEL_REVISION",
    "WAN22_I2V_A14B_DIFFUSERS_PROFILE",
    "WAN22_I2V_A14B_DIFFUSERS_SCHEMA",
    "WAN22_I2V_A14B_NATIVE_PROFILE",
    "WAN22_I2V_A14B_NATIVE_SCHEMA",
    "WAN22_REPOSITORY_REVISION",
    "WAN22_S2V_14B_OFFICIAL_PROFILE",
    "WAN22_S2V_14B_OFFICIAL_SCHEMA",
    "WAN22_S2V_MODEL_REVISION",
    "WAN22_T2V_A14B_DIFFUSERS_PROFILE",
    "WAN22_T2V_A14B_DIFFUSERS_SCHEMA",
    "WAN22_T2V_A14B_NATIVE_PROFILE",
    "WAN22_T2V_A14B_NATIVE_SCHEMA",
    "WAN22_TI2V_5B_DIFFUSERS_PROFILE",
    "WAN22_TI2V_5B_DIFFUSERS_SCHEMA",
    "WAN22_TI2V_5B_NATIVE_PROFILE",
    "WAN22_TI2V_5B_NATIVE_SCHEMA",
    "WAN_ANIMATE2_BASE_14B_OFFICIAL_PROFILE",
    "WAN_ANIMATE2_BASE_14B_OFFICIAL_SCHEMA",
    "WAN_ANIMATE2_COMFYUI_REVISION",
    "WAN_ANIMATE2_COMFY_MODEL_REVISION",
    "WAN_ANIMATE2_COMFY_OPTIMIZED_6_PROFILE",
    "WAN_ANIMATE2_COMFY_OPTIMIZED_6_SCHEMA",
    "WAN_ANIMATE2_COMFY_WORKFLOW_REVISION",
    "WAN_ANIMATE2_DISTILLED_14B_OFFICIAL_PROFILE",
    "WAN_ANIMATE2_DISTILLED_14B_OFFICIAL_SCHEMA",
    "WAN_ANIMATE2_MODEL_REVISION",
    "WAN_ANIMATE2_REPOSITORY_REVISION",
    "WAN_COMFYUI_REVISION",
    "WAN_DIFFUSERS_REVISION",
    "WanBoundary",
    "WanEvidenceReference",
    "WanGeneration",
    "WanProfile",
    "WanProfileId",
    "WanResolution",
    "WanScheduleResult",
    "WanSource",
    "WanTask",
    "build_wan_schedule",
    "derive_wan_boundary",
]
