"""Framework-independent contracts for the experimental Krea 2 modifier."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, cast

from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError

CONDITIONING_MODIFIER_ALGORITHM_ID = "sigmax.krea2-tap-rms-rebalance/1"
CONDITIONING_MODIFIER_REPORT_SCHEMA_ID = "sigmax.conditioning-modifier/1"
KREA2_CONDITIONING_PROFILE_SCHEMA_ID = "sigmax.krea2-conditioning-profile/1"
KREA2_TAP_COUNT = 12
KREA2_TAP_DIM = 2560
KREA2_FEATURE_DIM = KREA2_TAP_COUNT * KREA2_TAP_DIM
_MAX_PUBLIC_TEXT = 256
_MAX_WARNINGS = 16
_MAX_REPORT_BYTES = 65_536
_PUBLIC_TOKEN_PATTERN: Final = re.compile(r"[A-Za-z0-9_.:+/-]+\Z")


class Krea2ConditioningProfileId(str, Enum):
    DISABLED = "Disabled"
    SUBTLE_EXPERIMENTAL = "Subtle Experimental"
    CLASSIC_EXPERIMENTAL = "Classic Experimental"


class Krea2ConditioningVariant(str, Enum):
    TURBO = "Turbo"
    RAW = "RAW"


@dataclass(frozen=True, slots=True, kw_only=True)
class Krea2ConditioningProfile:
    profile_id: str
    profile_version: str
    evidence: EvidenceLevel
    source: str
    gains: tuple[float, ...]
    source_revision: str | None = None

    @property
    def schema_id(self) -> str:
        return KREA2_CONDITIONING_PROFILE_SCHEMA_ID

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ScheduleContractError("conditioning profile id must be non-empty")
        if len(self.profile_id) > _MAX_PUBLIC_TEXT or any(
            ord(char) < 0x20 for char in self.profile_id
        ):
            raise ScheduleContractError("conditioning profile id is invalid")
        if not isinstance(self.profile_version, str) or not self.profile_version.strip():
            raise ScheduleContractError("conditioning profile version must be non-empty")
        if len(self.profile_version) > 32 or not _PUBLIC_TOKEN_PATTERN.fullmatch(
            self.profile_version
        ):
            raise ScheduleContractError("conditioning profile version is invalid")
        if self.evidence not in {
            EvidenceLevel.COMMUNITY_RECOMMENDED,
            EvidenceLevel.EXPERIMENTAL,
            EvidenceLevel.MODIFIED,
        }:
            raise ScheduleContractError(
                "conditioning profile evidence must be community_recommended, experimental, or modified"
            )
        _public_token("conditioning profile source", self.source)
        if self.source_revision is not None:
            _public_token("conditioning profile source revision", self.source_revision)
        if not isinstance(self.gains, tuple) or len(self.gains) != KREA2_TAP_COUNT:
            raise ScheduleContractError("conditioning profile must contain exactly twelve gains")
        for gain in self.gains:
            if (
                isinstance(gain, bool)
                or not isinstance(gain, (int, float))
                or not math.isfinite(float(gain))
                or not 0.0 < float(gain) <= 8.0
            ):
                raise ScheduleContractError(
                    "conditioning profile gains must be finite, positive, and at most 8"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditioningModifierRequest:
    variant: Krea2ConditioningVariant
    profile: Krea2ConditioningProfile
    strength: float

    @property
    def variant_evidence(self) -> str:
        return "user_selected"

    def __post_init__(self) -> None:
        if not isinstance(self.variant, Krea2ConditioningVariant):
            raise ScheduleContractError(
                "conditioning modifier variant must be explicit RAW or Turbo"
            )
        if not isinstance(self.profile, Krea2ConditioningProfile):
            raise ScheduleContractError("conditioning modifier profile is unsupported")
        if (
            isinstance(self.strength, bool)
            or not isinstance(self.strength, (int, float))
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
        ):
            raise ScheduleContractError(
                "conditioning modifier strength must be finite and between 0 and 1"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditioningModifierReport:
    schema_id: str
    algorithm_id: str
    variant_evidence: str
    schedule_affected: bool
    fingerprint: str
    json_text: str

    def __post_init__(self) -> None:
        if self.schema_id != CONDITIONING_MODIFIER_REPORT_SCHEMA_ID:
            raise ScheduleContractError("conditioning modifier report schema is unsupported")
        if self.algorithm_id != CONDITIONING_MODIFIER_ALGORITHM_ID:
            raise ScheduleContractError("conditioning modifier algorithm is unsupported")
        if self.variant_evidence != "user_selected":
            raise ScheduleContractError("conditioning modifier variant evidence is unsupported")
        if self.schedule_affected is not False:
            raise ScheduleContractError("conditioning modifier must not affect schedules")
        if not isinstance(self.fingerprint, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.fingerprint
        ):
            raise ScheduleContractError("conditioning modifier fingerprint is invalid")
        if (
            not isinstance(self.json_text, str)
            or len(self.json_text.encode("utf-8")) > _MAX_REPORT_BYTES
        ):
            raise ScheduleContractError("conditioning modifier report is oversized")


def _public_token(label: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_PUBLIC_TEXT
        or not _PUBLIC_TOKEN_PATTERN.fullmatch(value)
    ):
        raise ScheduleContractError(f"{label} must be a bounded public token")
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("conditioning modifier report is not canonical JSON") from exc


def validate_krea2_conditioning_shape(shape: object) -> tuple[int, int, int]:
    if isinstance(shape, (str, bytes, bytearray)):
        raise ScheduleContractError("conditioning tensor shape must be rank-3")
    try:
        values = tuple(cast(Sequence[object], shape))
    except TypeError as exc:
        raise ScheduleContractError("conditioning tensor shape must be rank-3") from exc
    if len(values) != 3:
        raise ScheduleContractError("conditioning tensor shape must be rank-3")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ScheduleContractError("conditioning tensor dimensions must be positive integers")
    normalized: tuple[int, int, int] = (
        cast(int, values[0]),
        cast(int, values[1]),
        cast(int, values[2]),
    )
    if normalized[2] != KREA2_FEATURE_DIM:
        raise ScheduleContractError(
            "Krea 2 conditioning tensor must have exactly 30720 features (12x2560)"
        )
    return normalized


def get_krea2_profile(profile_id: Krea2ConditioningProfileId) -> Krea2ConditioningProfile:
    from comfyui_sigmax.conditioning.profiles import KREA2_CONDITIONING_PROFILES

    if not isinstance(profile_id, Krea2ConditioningProfileId):
        raise ScheduleContractError("conditioning profile selection is unsupported")
    return KREA2_CONDITIONING_PROFILES[profile_id]


def effective_gains(request: ConditioningModifierRequest) -> tuple[float, ...]:
    if not isinstance(request, ConditioningModifierRequest):
        raise ScheduleContractError("effective gains require a conditioning modifier request")
    if request.strength == 0.0:
        return (1.0,) * KREA2_TAP_COUNT
    gains = tuple(
        1.0 + float(request.strength) * (float(gain) - 1.0) for gain in request.profile.gains
    )
    if any(not math.isfinite(gain) or gain <= 0.0 for gain in gains):
        raise ScheduleContractError("effective conditioning gains are invalid")
    return gains


def build_modifier_report(
    *,
    request: ConditioningModifierRequest,
    input_shape: object,
    input_shapes: object | None = None,
    dtype: object,
    device: object,
    conditioning_entries: object,
    transformed_entries: object,
    warnings: object = (),
) -> ConditioningModifierReport:
    if not isinstance(request, ConditioningModifierRequest):
        raise ScheduleContractError("modifier report requires a conditioning modifier request")
    shape = validate_krea2_conditioning_shape(input_shape)
    if input_shapes is None:
        shape_facts: tuple[tuple[int, int, int], ...] = (shape,)
    elif not isinstance(input_shapes, tuple) or not input_shapes:
        raise ScheduleContractError("conditioning input shapes must be a non-empty tuple")
    else:
        shape_facts = tuple(validate_krea2_conditioning_shape(item) for item in input_shapes)
        if shape_facts[0] != shape:
            raise ScheduleContractError(
                "conditioning input shape must match the first input-shapes fact"
            )
    dtype_text = _public_token("conditioning dtype", dtype)
    device_text = _public_token("conditioning device", device)
    counts: dict[str, int] = {}
    for label, value in (
        ("conditioning entry count", conditioning_entries),
        ("transformed entry count", transformed_entries),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ScheduleContractError(f"{label} must be a non-negative integer")
        counts[label] = value
    if counts["transformed entry count"] > counts["conditioning entry count"]:
        raise ScheduleContractError("transformed entries cannot exceed conditioning entries")
    if not isinstance(warnings, tuple) or len(warnings) > _MAX_WARNINGS:
        raise ScheduleContractError("conditioning modifier warnings must be a bounded tuple")
    if any(not isinstance(warning, str) or not warning for warning in warnings):
        raise ScheduleContractError("conditioning modifier warnings must be non-empty strings")
    if warnings != tuple(sorted(set(warnings))):
        raise ScheduleContractError("conditioning modifier warnings must be canonical")
    gains = effective_gains(request)
    projection: dict[str, object] = {
        "algorithm": CONDITIONING_MODIFIER_ALGORITHM_ID,
        "evidence": EvidenceLevel.EXPERIMENTAL.value,
        "input": {
            "device": device_text,
            "dtype": dtype_text,
            "shape": list(shape),
            "shapes": [list(item) for item in shape_facts],
        },
        "modifier": {
            "effective_gains": list(gains),
            "profile": {
                "evidence": request.profile.evidence.value,
                "id": request.profile.profile_id,
                "source": request.profile.source,
                "source_revision": request.profile.source_revision,
                "version": request.profile.profile_version,
            },
            "requested_strength": float(request.strength),
            "rms_preservation": True,
        },
        "output": {
            "conditioning_entries": counts["conditioning entry count"],
            "transformed_entries": counts["transformed entry count"],
        },
        "schedule_affected": False,
        "schema": CONDITIONING_MODIFIER_REPORT_SCHEMA_ID,
        "variant": {
            "evidence": "user_selected",
            "value": request.variant.value,
        },
        "warnings": list(warnings),
    }
    canonical_projection = _canonical_json(projection)
    fingerprint = "sha256:" + hashlib.sha256(canonical_projection.encode("utf-8")).hexdigest()
    projection["fingerprint"] = fingerprint
    json_text = _canonical_json(projection)
    if len(json_text.encode("utf-8")) > _MAX_REPORT_BYTES:
        raise ScheduleContractError("conditioning modifier report is oversized")
    return ConditioningModifierReport(
        schema_id=CONDITIONING_MODIFIER_REPORT_SCHEMA_ID,
        algorithm_id=CONDITIONING_MODIFIER_ALGORITHM_ID,
        variant_evidence="user_selected",
        schedule_affected=False,
        fingerprint=fingerprint,
        json_text=json_text,
    )
