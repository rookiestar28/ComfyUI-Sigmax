"""Private, pure MiniMax H3 Turbo recipe profiles.

This module is intentionally not imported by the public profile package or built-in registry.
It records source-qualified recipe identity and owns only the endpoint-inclusive direct-ratio
sigma construction needed by the M6-13 evidence lane.  It does not load weights, apply LoRA
scales, select a sampler/backend, or claim model/runtime support.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Final, Literal

from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError
from comfyui_sigmax.profiles.minimax_h3_acceleration import (
    MINIMAX_H3_ACCELERATION_ARTIFACTS,
    MINIMAX_H3_ACCELERATION_RECIPES,
    MiniMaxH3AccelerationArtifact,
    MiniMaxH3AccelerationDisposition,
    MiniMaxH3AccelerationError,
    qualify_minimax_h3_candidate,
)

MINIMAX_H3_TURBO_SCHEMA_ID: Final = "sigmax.minimax-h3-turbo-pure/1"
MINIMAX_H3_TURBO_SCHEMA_VERSION: Final = "1"
MINIMAX_H3_TURBO_MODELTECH_REVISION: Final = (
    "a7e148b8dc7db8ad976966060dcc022adf11fc8d"  # pragma: allowlist secret
)
MINIMAX_H3_TURBO_SOURCE_ID: Final = "modeltc.minimax-h3-turbo"
MINIMAX_H3_TURBO_PROFILE_VERSION: Final = "1"
FloatPrecision = Literal["float32", "float64"]
_F32: Final = ">f"


class MiniMaxH3TurboReasonCode(str, Enum):
    """Stable fail-closed reasons for the private Turbo contract."""

    UNKNOWN_RECIPE = "UNKNOWN_RECIPE"
    WRONG_TASK = "WRONG_TASK"
    WRONG_RESOLUTION = "WRONG_RESOLUTION"
    UNSUPPORTED_RECIPE_NFE = "UNSUPPORTED_RECIPE_NFE"
    DUPLICATE_SHIFT_RISK = "DUPLICATE_SHIFT_RISK"
    DUPLICATE_SCALE_RISK = "DUPLICATE_SCALE_RISK"
    RECIPE_ARTIFACT_MISMATCH = "RECIPE_ARTIFACT_MISMATCH"
    ARTIFACT_NOT_ELIGIBLE = "ARTIFACT_NOT_ELIGIBLE"
    INVALID_PRECISION = "INVALID_PRECISION"
    INVALID_VALUE = "INVALID_VALUE"


class MiniMaxH3TurboError(ScheduleContractError):
    """Contract error carrying a stable private-profile reason."""

    def __init__(self, reason_code: MiniMaxH3TurboReasonCode, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code.value}: {message}")


def _f32(value: float) -> float:
    try:
        return float(struct.unpack(_F32, struct.pack(_F32, float(value)))[0])
    except (OverflowError, struct.error) as exc:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.INVALID_VALUE, "value cannot be represented as float32"
        ) from exc


def _require_shift(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MiniMaxH3TurboError(MiniMaxH3TurboReasonCode.INVALID_VALUE, f"{name} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.INVALID_VALUE, f"{name} must be finite and positive"
        )
    return numeric


def _direct_ratio(value: float, shift: float, precision: FloatPrecision) -> float:
    if value in (0.0, 1.0):
        return _f32(value) if precision == "float32" else float(value)
    if precision == "float64":
        return shift * value / (1.0 + (shift - 1.0) * value)
    numerator = _f32(shift * value)
    denominator = _f32(1.0 + _f32((shift - 1.0) * value))
    return _f32(numerator / denominator)


def _direct_ratio_vector(nfe: int, shift: float, precision: FloatPrecision) -> tuple[float, ...]:
    values: list[float] = []
    for index in range(nfe):
        base = (nfe - index) / nfe
        if precision == "float32":
            base = _f32(base)
        values.append(_direct_ratio(base, shift, precision))
    values.append(_f32(0.0) if precision == "float32" else 0.0)
    return tuple(values)


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3TurboProfile:
    """One recipe-exact pure profile; never a public/runtime registration."""

    recipe_id: str
    task: str
    evidence: EvidenceLevel
    source_id: str
    source_revision: str
    video_shift: float
    audio_shift: float
    allowed_nfe: tuple[int, ...]
    default_nfe: int
    resolution_policy: str
    reference_policy: str
    artifact_ids: tuple[str, ...]
    eligible_artifact_ids: tuple[str, ...] = ()
    runtime_registered: bool = False
    runtime_supported: bool = False

    def __post_init__(self) -> None:
        recipe = _RECIPES_BY_ID.get(self.recipe_id)
        if recipe is None:
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.UNKNOWN_RECIPE, "recipe is not M6-12-qualified"
            )
        if self.task != recipe.task:
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.WRONG_TASK, "profile task differs from source recipe"
            )
        if self.source_id != MINIMAX_H3_TURBO_SOURCE_ID or self.source_revision != (
            MINIMAX_H3_TURBO_MODELTECH_REVISION
        ):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.INVALID_VALUE, "profile provenance is not pinned"
            )
        if not isinstance(self.evidence, EvidenceLevel):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.INVALID_VALUE, "profile evidence is unsupported"
            )
        if self.allowed_nfe != tuple(sorted(set(self.allowed_nfe))) or not self.allowed_nfe:
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.INVALID_VALUE, "allowed_nfe must be sorted and non-empty"
            )
        if any(nfe not in recipe.allowed_nfe for nfe in self.allowed_nfe):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.UNSUPPORTED_RECIPE_NFE,
                "profile step count is outside the source recipe",
            )
        if self.default_nfe not in self.allowed_nfe:
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.UNSUPPORTED_RECIPE_NFE, "default_nfe is not allowed"
            )
        if self.video_shift != recipe.video_shift or self.audio_shift != recipe.audio_shift:
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.DUPLICATE_SHIFT_RISK,
                "profile shifts differ from recipe-owned shifts",
            )
        if (
            self.resolution_policy != recipe.resolution_policy
            or self.reference_policy != recipe.reference_policy
        ):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.WRONG_RESOLUTION,
                "profile resolution/reference policy differs from source recipe",
            )
        if self.artifact_ids != tuple(sorted(set(self.artifact_ids))):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.INVALID_VALUE, "artifact_ids must be sorted and unique"
            )
        for artifact_id in self.artifact_ids:
            artifact = _ARTIFACTS_BY_ID.get(artifact_id)
            if artifact is None or artifact.recipe_id != self.recipe_id:
                raise MiniMaxH3TurboError(
                    MiniMaxH3TurboReasonCode.RECIPE_ARTIFACT_MISMATCH,
                    "artifact identity is not owned by this recipe",
                )
        if self.eligible_artifact_ids != tuple(sorted(set(self.eligible_artifact_ids))):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.INVALID_VALUE,
                "eligible_artifact_ids must be sorted and unique",
            )
        if any(artifact_id not in self.artifact_ids for artifact_id in self.eligible_artifact_ids):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.ARTIFACT_NOT_ELIGIBLE,
                "eligible artifact is not in the exact artifact set",
            )
        if self.eligible_artifact_ids and not all(
            _ARTIFACTS_BY_ID[artifact_id].disposition is MiniMaxH3AccelerationDisposition.QUALIFIED
            for artifact_id in self.eligible_artifact_ids
        ):
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.ARTIFACT_NOT_ELIGIBLE,
                "blocked or rejected artifact cannot be eligible",
            )
        if self.runtime_registered or self.runtime_supported:
            raise MiniMaxH3TurboError(
                MiniMaxH3TurboReasonCode.INVALID_VALUE,
                "M6-13 profiles cannot be runtime or public registrations",
            )

    @property
    def profile_id(self) -> str:
        return f"minimax-h3.turbo.{self.recipe_id.removeprefix('h3.')}"

    @property
    def profile_version(self) -> str:
        return MINIMAX_H3_TURBO_PROFILE_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3TurboSchedule:
    """Paired pure video/audio vectors with endpoint-inclusive transition semantics."""

    recipe_id: str
    nfe: int
    precision: FloatPrecision
    video_sigmas: tuple[float, ...]
    audio_sigmas: tuple[float, ...]

    @property
    def fingerprint(self) -> str:
        return minimax_h3_turbo_schedule_fingerprint(self)


_RECIPES_BY_ID = {recipe.recipe_id: recipe for recipe in MINIMAX_H3_ACCELERATION_RECIPES}
_ARTIFACTS_BY_ID = {
    artifact.artifact_id: artifact for artifact in MINIMAX_H3_ACCELERATION_ARTIFACTS
}


def _profile(
    recipe_id: str,
    *,
    allowed_nfe: tuple[int, ...],
    artifact_ids: tuple[str, ...],
) -> MiniMaxH3TurboProfile:
    recipe = _RECIPES_BY_ID[recipe_id]
    return MiniMaxH3TurboProfile(
        recipe_id=recipe.recipe_id,
        task=recipe.task,
        evidence=recipe.evidence,
        source_id=recipe.source_id,
        source_revision=MINIMAX_H3_TURBO_MODELTECH_REVISION,
        video_shift=recipe.video_shift,
        audio_shift=recipe.audio_shift,
        allowed_nfe=allowed_nfe,
        default_nfe=recipe.default_nfe if recipe.default_nfe in allowed_nfe else allowed_nfe[-1],
        resolution_policy=recipe.resolution_policy,
        reference_policy=recipe.reference_policy,
        artifact_ids=artifact_ids,
    )


MINIMAX_H3_TURBO_PROFILES: Final[tuple[MiniMaxH3TurboProfile, ...]] = tuple(
    sorted(
        (
            _profile(
                "h3.fl2va.lightx2v-turbo-4-v0.1-544p",
                allowed_nfe=(4,),
                artifact_ids=(),
            ),
            _profile(
                "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
                # The source allows 4 as a recommendation, but M6-13 rejects it until independently
                # proven by a complete vector/parity/fingerprint fixture.
                allowed_nfe=(8,),
                artifact_ids=(
                    "kijai.fl2v-8.reduced",
                    "lightx2v.fl2v-8.full",
                    "local.fl2v-8.modified",
                ),
            ),
            _profile(
                "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
                allowed_nfe=(4,),
                artifact_ids=(
                    "kijai.fl2v-4-768.reduced",
                    "lightx2v.fl2v-4-768.full",
                    "local.fl2v-4-768.modified",
                ),
            ),
            _profile(
                "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
                allowed_nfe=(4,),
                artifact_ids=(
                    "kijai.ref2v-4.reduced",
                    "lightx2v.ref2v-4.full",
                    "local.ref2v-4.modified",
                ),
            ),
        ),
        key=lambda profile: profile.recipe_id,
    )
)
_PROFILES_BY_ID: Final = {profile.recipe_id: profile for profile in MINIMAX_H3_TURBO_PROFILES}


def get_minimax_h3_turbo_profile(recipe_id: str) -> MiniMaxH3TurboProfile:
    profile = _PROFILES_BY_ID.get(recipe_id)
    if profile is None:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.UNKNOWN_RECIPE, "recipe is not a private Turbo profile"
        )
    return profile


def _require_precision(value: object) -> FloatPrecision:
    if value not in {"float32", "float64"}:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.INVALID_PRECISION, "precision must be float32 or float64"
        )
    return value


def build_minimax_h3_turbo_schedule(
    recipe_id: str,
    *,
    nfe: int,
    precision: FloatPrecision = "float64",
    task: str | None = None,
    resolution_policy: str | None = None,
    video_shift: float | None = None,
    audio_shift: float | None = None,
    input_already_shifted: bool = False,
    loader_strength: float | None = None,
) -> MiniMaxH3TurboSchedule:
    """Build a recipe-owned direct-ratio pair without touching runtime/model state."""

    profile = get_minimax_h3_turbo_profile(recipe_id)
    if task is not None and task != profile.task:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.WRONG_TASK, "requested task differs from recipe task"
        )
    if resolution_policy is not None and resolution_policy != profile.resolution_policy:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.WRONG_RESOLUTION,
            "requested resolution policy differs from recipe policy",
        )
    if video_shift is not None and not math.isclose(
        _require_shift("video_shift", video_shift), profile.video_shift, rel_tol=0.0, abs_tol=1e-12
    ):
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.DUPLICATE_SHIFT_RISK,
            "video shift differs from recipe-owned shift",
        )
    if audio_shift is not None and not math.isclose(
        _require_shift("audio_shift", audio_shift), profile.audio_shift, rel_tol=0.0, abs_tol=1e-12
    ):
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.DUPLICATE_SHIFT_RISK,
            "audio shift differs from recipe-owned shift",
        )
    if input_already_shifted:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.DUPLICATE_SHIFT_RISK,
            "already-shifted input cannot receive recipe-owned shifts again",
        )
    if loader_strength is not None and not math.isclose(
        _require_shift("loader_strength", loader_strength), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.DUPLICATE_SCALE_RISK,
            "pure schedule cannot apply a second LoRA scale",
        )
    if not isinstance(nfe, int) or isinstance(nfe, bool) or nfe <= 0:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.UNSUPPORTED_RECIPE_NFE, "nfe must be a positive integer"
        )
    if nfe not in profile.allowed_nfe:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.UNSUPPORTED_RECIPE_NFE,
            "nfe is not independently proven for this recipe profile",
        )
    selected_precision = _require_precision(precision)
    video = _direct_ratio_vector(nfe, profile.video_shift, selected_precision)
    audio = _direct_ratio_vector(nfe, profile.audio_shift, selected_precision)
    if not all(math.isfinite(value) for value in (*video, *audio)):
        raise MiniMaxH3TurboError(MiniMaxH3TurboReasonCode.INVALID_VALUE, "schedule is not finite")
    if not all(left >= right for left, right in pairwise(video)) or not all(
        left >= right for left, right in pairwise(audio)
    ):
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.INVALID_VALUE, "schedule is not monotone"
        )
    return MiniMaxH3TurboSchedule(
        recipe_id=recipe_id,
        nfe=nfe,
        precision=selected_precision,
        video_sigmas=video,
        audio_sigmas=audio,
    )


def validate_minimax_h3_turbo_artifact(
    profile: MiniMaxH3TurboProfile | str,
    artifact_id: str,
    *,
    nfe: int | None = None,
    loader_strength: float | None = None,
) -> MiniMaxH3AccelerationArtifact:
    """Validate exact artifact metadata and require an eligible M6-12 disposition."""

    selected = get_minimax_h3_turbo_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(selected, MiniMaxH3TurboProfile):
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.UNKNOWN_RECIPE, "artifact validation needs a Turbo profile"
        )
    artifact = _ARTIFACTS_BY_ID.get(artifact_id)
    if artifact is None:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.ARTIFACT_NOT_ELIGIBLE,
            "artifact identity is not an exact M6-12 record",
        )
    if artifact.recipe_id != selected.recipe_id:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.RECIPE_ARTIFACT_MISMATCH,
            "artifact belongs to a different recipe",
        )
    if nfe is not None and nfe not in selected.allowed_nfe:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.UNSUPPORTED_RECIPE_NFE,
            "artifact request uses an unproven profile step count",
        )
    if artifact_id not in selected.artifact_ids:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.ARTIFACT_NOT_ELIGIBLE,
            "artifact is not in this profile's exact identity set",
        )
    try:
        return qualify_minimax_h3_candidate(
            candidate_id=artifact_id,
            task=selected.task,
            nfe=selected.default_nfe if nfe is None else nfe,
            video_shift=selected.video_shift,
            audio_shift=selected.audio_shift,
            loader_strength=loader_strength,
            require_eligible=True,
        )
    except MiniMaxH3AccelerationError:
        raise


def _schedule_projection(schedule: MiniMaxH3TurboSchedule) -> dict[str, object]:
    return {
        "audio_sigmas": list(schedule.audio_sigmas),
        "nfe": schedule.nfe,
        "precision": schedule.precision,
        "recipe_id": schedule.recipe_id,
        "schema": MINIMAX_H3_TURBO_SCHEMA_ID,
        "video_sigmas": list(schedule.video_sigmas),
    }


def minimax_h3_turbo_schedule_fingerprint(schedule: MiniMaxH3TurboSchedule) -> str:
    if not isinstance(schedule, MiniMaxH3TurboSchedule):
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.INVALID_VALUE, "fingerprint requires a Turbo schedule"
        )
    encoded = json.dumps(
        _schedule_projection(schedule),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MINIMAX_H3_TURBO_MODELTECH_REVISION",
    "MINIMAX_H3_TURBO_PROFILES",
    "MINIMAX_H3_TURBO_PROFILE_VERSION",
    "MINIMAX_H3_TURBO_SCHEMA_ID",
    "MINIMAX_H3_TURBO_SCHEMA_VERSION",
    "MiniMaxH3TurboError",
    "MiniMaxH3TurboProfile",
    "MiniMaxH3TurboReasonCode",
    "MiniMaxH3TurboSchedule",
    "build_minimax_h3_turbo_schedule",
    "get_minimax_h3_turbo_profile",
    "minimax_h3_turbo_schedule_fingerprint",
    "validate_minimax_h3_turbo_artifact",
]
