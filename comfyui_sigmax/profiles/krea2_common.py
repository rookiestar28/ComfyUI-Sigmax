"""Shared immutable declarations for evidence-pinned Krea 2 profiles."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError
from comfyui_sigmax.profiles.schema_v1 import (
    ArtifactVersionDeclaration,
    DetectionDeclaration,
    FrameworkProvenance,
    LicenseDeclaration,
    SlicingDeclaration,
    SoftwareSourceProvenance,
)

_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")


class ShiftParameterization(str, Enum):
    """Named time-shift formulas exposed by evidence-pinned profiles."""

    EXPONENTIAL_MU = "exponential_mu"


class DimensionAlignmentMode(str, Enum):
    """Dimension-normalization modes declared by model profiles."""

    CEIL_MULTIPLE = "ceil_multiple"


def _require_finite_number(field_name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ScheduleContractError(f"{field_name} must be a finite number")
    return float(value)


def _require_positive_integer(field_name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleContractError(f"{field_name} must be a positive integer")
    return value


def _require_identifier(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ScheduleContractError(f"{field_name} must be a stable lowercase identifier")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class GuidanceConvention:
    """Krea guidance and equivalent standard ComfyUI CFG values."""

    krea_guidance: float
    comfy_cfg: float

    def __post_init__(self) -> None:
        krea_guidance = _require_finite_number("krea_guidance", self.krea_guidance)
        comfy_cfg = _require_finite_number("comfy_cfg", self.comfy_cfg)
        if krea_guidance < 0.0 or comfy_cfg != krea_guidance + 1.0:
            raise ScheduleContractError(
                "Krea guidance must be nonnegative and ComfyUI CFG must equal guidance + 1"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class DimensionPolicy:
    """Evidence-bearing Krea 2 image-dimension alignment policy."""

    mode: DimensionAlignmentMode
    multiple: int
    evidence_source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.mode, DimensionAlignmentMode):
            raise ScheduleContractError("dimension alignment mode is unsupported")
        multiple = _require_positive_integer("dimension multiple", self.multiple)
        source_id = _require_identifier("evidence_source_id", self.evidence_source_id)
        if (
            self.mode is not DimensionAlignmentMode.CEIL_MULTIPLE
            or multiple != 16
            or source_id != "krea.krea2.official"
        ):
            raise ScheduleContractError(
                "the official Krea 2 dimension policy is ceil-to-multiple-of-16"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2ImageGeometry:
    """Requested and effective Krea 2 packed-image geometry."""

    requested_width: int
    requested_height: int
    effective_width: int
    effective_height: int
    alignment_multiple: int
    grid_width: int
    grid_height: int
    image_seq_len: int

    def __post_init__(self) -> None:
        requested_width = _require_positive_integer(
            "requested_width",
            self.requested_width,
        )
        requested_height = _require_positive_integer(
            "requested_height",
            self.requested_height,
        )
        effective_width = _require_positive_integer(
            "effective_width",
            self.effective_width,
        )
        effective_height = _require_positive_integer(
            "effective_height",
            self.effective_height,
        )
        alignment = _require_positive_integer(
            "alignment_multiple",
            self.alignment_multiple,
        )
        grid_width = _require_positive_integer("grid_width", self.grid_width)
        grid_height = _require_positive_integer("grid_height", self.grid_height)
        image_seq_len = _require_positive_integer("image_seq_len", self.image_seq_len)

        expected_width = ((requested_width + alignment - 1) // alignment) * alignment
        expected_height = ((requested_height + alignment - 1) // alignment) * alignment
        if alignment != 16:
            raise ScheduleContractError("Krea 2 image geometry requires 16-pixel alignment")
        if (effective_width, effective_height) != (expected_width, expected_height):
            raise ScheduleContractError(
                "effective dimensions must be requested dimensions rounded up to alignment"
            )
        if (grid_width, grid_height) != (
            effective_width // alignment,
            effective_height // alignment,
        ):
            raise ScheduleContractError("Krea 2 packed-image grid dimensions are inconsistent")
        if image_seq_len != grid_width * grid_height:
            raise ScheduleContractError("image_seq_len must equal grid_width * grid_height")


def resolve_krea2_image_geometry(
    width: int,
    height: int,
    *,
    policy: DimensionPolicy,
) -> Krea2ImageGeometry:
    """Resolve positive requested pixels to the authoritative Krea 2 packed grid."""

    if not isinstance(policy, DimensionPolicy):
        raise ScheduleContractError("Krea 2 geometry requires a DimensionPolicy")
    requested_width = _require_positive_integer("width", width)
    requested_height = _require_positive_integer("height", height)
    multiple = policy.multiple
    effective_width = ((requested_width + multiple - 1) // multiple) * multiple
    effective_height = ((requested_height + multiple - 1) // multiple) * multiple
    grid_width = effective_width // multiple
    grid_height = effective_height // multiple
    return Krea2ImageGeometry(
        requested_width=requested_width,
        requested_height=requested_height,
        effective_width=effective_width,
        effective_height=effective_height,
        alignment_multiple=multiple,
        grid_width=grid_width,
        grid_height=grid_height,
        image_seq_len=grid_width * grid_height,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceReference:
    """Pinned source revision and deterministic locators for one profile claim set."""

    source_id: str
    evidence: EvidenceLevel
    url: str
    revision: str
    locators: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier("source_id", self.source_id)
        if not isinstance(self.evidence, EvidenceLevel):
            raise ScheduleContractError("evidence must be an EvidenceLevel")
        if not isinstance(self.url, str) or not self.url.startswith("https://"):
            raise ScheduleContractError("evidence URL must use HTTPS")
        if not isinstance(self.revision, str) or not _COMMIT_PATTERN.fullmatch(self.revision):
            raise ScheduleContractError("evidence revision must be a pinned 40-character commit")
        if not isinstance(self.locators, tuple) or not self.locators:
            raise ScheduleContractError("evidence locators must be a non-empty tuple")
        if any(not isinstance(locator, str) or not locator.strip() for locator in self.locators):
            raise ScheduleContractError("evidence locators must contain non-empty strings")
        if len(self.locators) != len(set(self.locators)):
            raise ScheduleContractError("evidence locators must not contain duplicates")
        if self.locators != tuple(sorted(self.locators)):
            raise ScheduleContractError("evidence locators must use canonical order")


KREA_REFERENCE: Final = EvidenceReference(
    source_id="krea.krea2.official",
    evidence=EvidenceLevel.OFFICIAL,
    url="https://github.com/krea-ai/krea-2",
    revision="db3984fbc6e13b34c0064990fc2d95ac64d00058",  # pragma: allowlist secret
    locators=("README.md", "inference.py", "sampling.py"),
)
DIFFUSERS_REFERENCE: Final = EvidenceReference(
    source_id="diffusers.krea2.framework",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    url="https://github.com/huggingface/diffusers",
    revision="a3608b512ed7248499a44c61d954965ed9bdae4d",  # pragma: allowlist secret
    locators=(
        "src/diffusers/pipelines/krea2/pipeline_krea2.py",
        "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    ),
)
COMFYUI_REFERENCE: Final = EvidenceReference(
    source_id="comfyui.krea2.framework",
    evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
    url="https://github.com/Comfy-Org/ComfyUI",
    revision="e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
    locators=(
        "comfy/k_diffusion/sampling.py",
        "comfy/model_sampling.py",
        "comfy/supported_models.py",
    ),
)

KREA2_REFERENCES: Final = (
    KREA_REFERENCE,
    DIFFUSERS_REFERENCE,
    COMFYUI_REFERENCE,
)

APACHE_2_LICENSE: Final = LicenseDeclaration(
    declaration_version="1",
    identifier="Apache-2.0",
    name="Apache License 2.0",
    url="https://www.apache.org/licenses/LICENSE-2.0",
)
GPL_3_ONLY_LICENSE: Final = LicenseDeclaration(
    declaration_version="1",
    identifier="GPL-3.0-only",
    name="GNU General Public License v3.0 only",
    url="https://www.gnu.org/licenses/gpl-3.0.html",
)
KREA_SOFTWARE_PROVENANCE: Final = SoftwareSourceProvenance(
    record_version="1",
    source_id=KREA_REFERENCE.source_id,
    resource_version=None,
    revision=KREA_REFERENCE.revision,
    url=KREA_REFERENCE.url,
    license=APACHE_2_LICENSE,
    locators=KREA_REFERENCE.locators,
)
COMFYUI_FRAMEWORK_PROVENANCE: Final = FrameworkProvenance(
    record_version="1",
    framework_id=COMFYUI_REFERENCE.source_id,
    resource_version=None,
    revision=COMFYUI_REFERENCE.revision,
    url=COMFYUI_REFERENCE.url,
    license=GPL_3_ONLY_LICENSE,
    locators=COMFYUI_REFERENCE.locators,
)
DIFFUSERS_FRAMEWORK_PROVENANCE: Final = FrameworkProvenance(
    record_version="1",
    framework_id=DIFFUSERS_REFERENCE.source_id,
    resource_version="0.39.0",
    revision=DIFFUSERS_REFERENCE.revision,
    url=DIFFUSERS_REFERENCE.url,
    license=APACHE_2_LICENSE,
    locators=DIFFUSERS_REFERENCE.locators,
)
KREA2_FRAMEWORK_PROVENANCE: Final = (
    COMFYUI_FRAMEWORK_PROVENANCE,
    DIFFUSERS_FRAMEWORK_PROVENANCE,
)
KREA2_DETECTION: Final = DetectionDeclaration(
    strategy_id="krea2.variant.evidence-v1",
    strict_default=True,
    ambiguity_requires_explicit=True,
    resolving_sources=(
        "explicit_selection",
        "trusted_profile_metadata",
        "trusted_framework_metadata",
        "verified_sha256",
    ),
    suggestion_sources=("local_header_signal", "filename_signal"),
    family_only_sources=("local_tensor_signal", "model_class_signal"),
)
KREA2_ARTIFACT_VERSIONS: Final = ArtifactVersionDeclaration(
    numerical_schema="sigmax.numerical-schedule/1",
    construction_schema="sigmax.schedule-artifact/1",
    envelope_schema="sigmax.schedule-artifact-envelope/1",
)
KREA2_SLICING: Final = SlicingDeclaration(
    supports_step_range=True,
    supports_denoise_tail=True,
    zero_denoise_is_empty=True,
)
