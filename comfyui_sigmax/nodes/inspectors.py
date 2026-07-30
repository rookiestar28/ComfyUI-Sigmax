"""Read-only bounded profile and schedule inspectors."""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from comfyui_sigmax.core import ScheduleContractError, SigmaDomain, validate_sigma_schedule
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint
from comfyui_sigmax.nodes.model_aware_sigma_scheduler import (
    MODEL_AWARE_SIGMA_NODE_SCHEMA_ID,
    build_model_aware_sigma_schedule,
)
from comfyui_sigmax.profiles import ProfileKey, builtin_profile_registry

PROFILE_INSPECTOR_NODE_ID: Final = "Sigmax.ProfileInspector"
PROFILE_INSPECTOR_SCHEMA_ID: Final = "sigmax.profile-inspector/1"
SCHEDULE_INSPECTOR_NODE_ID: Final = "Sigmax.ScheduleInspector"
SCHEDULE_INSPECTOR_SCHEMA_ID: Final = "sigmax.schedule-inspector/1"
_KREA2_SCHEMA_ID: Final = "sigmax.krea2-sigma-node/1"
_ADVANCED_SCHEMA_ID: Final = "sigmax.advanced-flowmatch-node/1"
_MAX_STEPS: Final = 10_000
_MAX_DIMENSION: Final = 65_536
_MAX_JSON_BYTES: Final = 1_048_576
_MAX_DEPTH: Final = 32
_MAX_COLLECTION: Final = 1024
_MAX_STRING: Final = 4096
_SAMPLING_CLASS_PATTERN: Final = re.compile(r"ModelSampling[A-Za-z0-9_]{0,64}")
_FINGERPRINT_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")

_KREA2_FIELDS: Final = {
    "dimensions",
    "fingerprints",
    "profile",
    "schema",
    "shift",
    "slicing",
    "strict_official",
    "warnings",
}
_MODEL_AWARE_FIELDS: Final = {
    "capability_decision",
    "host_evidence",
    "model_probe",
    "profile",
    "schedule",
    "schema",
    "variant_resolution",
}
_ADVANCED_FIELDS: Final = {
    "base_grid",
    "domain",
    "fingerprints",
    "ownership",
    "provenance",
    "schema",
    "shift",
    "slicing",
    "terminal",
    "transform_order",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileInspectorResult:
    """Pure profile report."""

    schema_id: str
    report_json: str

    def __post_init__(self) -> None:
        if self.schema_id != PROFILE_INSPECTOR_SCHEMA_ID:
            raise ScheduleContractError("profile inspector schema is unsupported")
        if not isinstance(self.report_json, str) or not self.report_json:
            raise ScheduleContractError("profile inspector report is required")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleInspectorResult:
    """Pure schedule report."""

    schema_id: str
    report_json: str

    def __post_init__(self) -> None:
        if self.schema_id != SCHEDULE_INSPECTOR_SCHEMA_ID:
            raise ScheduleContractError("schedule inspector schema is unsupported")
        if not isinstance(self.report_json, str) or not self.report_json:
            raise ScheduleContractError("schedule inspector report is required")


def _canonical_json(value: dict[str, object]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("inspector report is not canonical JSON") from exc


def _static_attribute(value: object, name: str) -> object:
    try:
        candidate = inspect.getattr_static(value, name)
    except AttributeError as exc:
        raise ScheduleContractError(f"MODEL lacks bounded public {name} evidence") from exc
    if isinstance(candidate, property) or inspect.isroutine(candidate):
        raise ScheduleContractError(f"MODEL {name} evidence must not be executable")
    return candidate


def _native_sampling_class(model: object) -> str:
    inner = _static_attribute(model, "model")
    sampling = _static_attribute(inner, "model_sampling")
    class_name = type(sampling).__name__
    if not _SAMPLING_CLASS_PATTERN.fullmatch(class_name):
        raise ScheduleContractError("native model_sampling class is unsupported")
    return class_name


def build_profile_inspection(
    *,
    model: object,
    variant: object,
    steps: object,
    width: object,
    height: object,
    strict_official: object,
) -> ProfileInspectorResult:
    """Build a bounded exact-profile report without Torch or host mutation."""

    native_class = _native_sampling_class(model)
    result = build_model_aware_sigma_schedule(
        model=model,
        variant=variant,
        steps=steps,
        width=width,
        height=height,
        strict_official=strict_official,
        start_step=0,
        end_step=-1,
    )
    source = cast(dict[str, object], json.loads(result.schedule_info_json))
    capability = cast(dict[str, object], source["capability_decision"])
    schedule = cast(dict[str, object], source["schedule"])
    profile = cast(dict[str, object], source["profile"])
    selected = result.selected_variant
    registered = builtin_profile_registry().resolve(
        ProfileKey(
            profile_id=f"krea2.{selected}.official",
            profile_version="1",
        )
    )
    report: dict[str, object] = {
        "compatibility": capability,
        "dimensions": schedule["dimensions"],
        "fingerprints": schedule["fingerprints"],
        "model_identity": capability["model_identity"],
        "native_sampling": {
            "class": native_class,
            "reference_sampler_id": (registered.schema.reference_sampler_capabilities.sampler_id),
        },
        "profile": profile,
        "provenance": {
            "host": source["host_evidence"],
            "profile": profile,
            "schedule": schedule["profile"],
        },
        "schema": PROFILE_INSPECTOR_SCHEMA_ID,
        "shift": schedule["shift"],
        "source_schema": MODEL_AWARE_SIGMA_NODE_SCHEMA_ID,
        "warnings": schedule["warnings"],
    }
    return ProfileInspectorResult(
        schema_id=PROFILE_INSPECTOR_SCHEMA_ID,
        report_json=_canonical_json(report),
    )


def _duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ScheduleContractError(f"non-finite JSON constant is forbidden: {value}")


def _bound_json(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ScheduleContractError("schedule information exceeds maximum depth")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScheduleContractError("schedule information contains a non-finite number")
        return
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise ScheduleContractError("schedule information string exceeds limit")
        return
    if isinstance(value, list):
        if len(value) > _MAX_COLLECTION:
            raise ScheduleContractError("schedule information collection exceeds limit")
        for child in value:
            _bound_json(child, depth=depth + 1)
        return
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION:
            raise ScheduleContractError("schedule information collection exceeds limit")
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > _MAX_STRING:
                raise ScheduleContractError("schedule information key is invalid")
            _bound_json(child, depth=depth + 1)
        return
    raise ScheduleContractError("schedule information contains an unsupported JSON value")


def _decode_schedule_info(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value:
        raise ScheduleContractError("schedule_info must be a non-empty string")
    try:
        encoded_size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ScheduleContractError("schedule_info must be valid Unicode") from exc
    if encoded_size > _MAX_JSON_BYTES:
        raise ScheduleContractError("schedule_info exceeds maximum byte size")
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ScheduleContractError("schedule_info is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ScheduleContractError("schedule_info root must be an object")
    _bound_json(decoded)
    return cast(dict[str, object], decoded)


def _require_exact_fields(
    value: dict[str, object],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise ScheduleContractError(f"{label} fields do not match its schema")


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ScheduleContractError(f"{label} must be an array")
    return value


def _fingerprints(value: object) -> dict[str, object]:
    fingerprints = _object(value, label="fingerprints")
    if set(fingerprints) != {"complete", "output"}:
        raise ScheduleContractError("fingerprints fields do not match schema")
    if not all(
        isinstance(fingerprints[key], str)
        and _FINGERPRINT_PATTERN.fullmatch(cast(str, fingerprints[key]))
        for key in ("complete", "output")
    ):
        raise ScheduleContractError("advertised fingerprint is invalid")
    return fingerprints


def _normalized_source(
    source: dict[str, object],
) -> tuple[str, dict[str, object], dict[str, object] | None]:
    schema = source.get("schema")
    if schema == MODEL_AWARE_SIGMA_NODE_SCHEMA_ID:
        _require_exact_fields(source, _MODEL_AWARE_FIELDS, label="model-aware information")
        schedule = _object(source["schedule"], label="model-aware schedule")
        _require_exact_fields(schedule, _KREA2_FIELDS, label="nested Krea schedule")
        return schema, schedule, source
    if schema == _KREA2_SCHEMA_ID:
        _require_exact_fields(source, _KREA2_FIELDS, label="Krea schedule")
        return schema, source, None
    if schema == _ADVANCED_SCHEMA_ID:
        _require_exact_fields(source, _ADVANCED_FIELDS, label="advanced schedule")
        return schema, source, None
    raise ScheduleContractError("schedule_info schema is unsupported")


def build_schedule_inspection(
    *,
    sigmas: object,
    schedule_info: object,
) -> ScheduleInspectorResult:
    """Verify connected sigmas against one controlled scheduler projection."""

    if not isinstance(sigmas, tuple):
        raise ScheduleContractError("pure inspector sigmas must be a tuple")
    if not 2 <= len(sigmas) <= _MAX_STEPS + 1:
        raise ScheduleContractError("sigmas length is outside the inspector limit")
    values = validate_sigma_schedule(
        sigmas,
        domain=SigmaDomain.UNIT_FLOW,
        expected_steps=len(sigmas) - 1,
        require_terminal_zero=False,
    )
    source = _decode_schedule_info(schedule_info)
    source_schema, schedule, model_aware = _normalized_source(source)
    fingerprints = _fingerprints(schedule.get("fingerprints"))
    advertised = cast(str, fingerprints["output"])
    computed = sigma_output_fingerprint(values, domain=SigmaDomain.UNIT_FLOW)
    if advertised != computed:
        raise ScheduleContractError("connected SIGMAS fingerprint does not match schedule_info")

    if source_schema == _ADVANCED_SCHEMA_ID:
        domain = _object(schedule["domain"], label="domain").get("sigma")
        dimensions: object = None
        profile: object = schedule["provenance"]
    else:
        domain = SigmaDomain.UNIT_FLOW.value
        dimensions = schedule["dimensions"]
        profile = model_aware["profile"] if model_aware is not None else schedule["profile"]
    if domain != SigmaDomain.UNIT_FLOW.value:
        raise ScheduleContractError("inspector supports only UNIT_FLOW schedules")

    warnings = _list(schedule.get("warnings", []), label="warnings")
    if not all(isinstance(item, str) and item for item in warnings):
        raise ScheduleContractError("warnings must contain non-empty strings")
    compatibility = model_aware["capability_decision"] if model_aware is not None else None
    model_identity = (
        _object(compatibility, label="compatibility").get("model_identity")
        if compatibility is not None
        else None
    )
    report: dict[str, object] = {
        "compatibility": compatibility,
        "dimensions": dimensions,
        "domain": domain,
        "fingerprints": {
            "advertised_complete": fingerprints["complete"],
            "advertised_output": advertised,
            "computed_output": computed,
            "verified": True,
        },
        "model_identity": model_identity,
        "profile": profile,
        "schema": SCHEDULE_INSPECTOR_SCHEMA_ID,
        "shift": schedule["shift"],
        "slicing": schedule["slicing"],
        "source_schema": source_schema,
        "warnings": warnings,
    }
    return ScheduleInspectorResult(
        schema_id=SCHEDULE_INSPECTOR_SCHEMA_ID,
        report_json=_canonical_json(report),
    )


class ProfileInspector:
    """Display an exact bounded Krea profile and capability report."""

    DESCRIPTION = "Inspects a supported Krea 2 MODEL without serializing or mutating it."
    CATEGORY = "Sigmax/inspection"
    FUNCTION = "inspect"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("profile_report",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "model": ("MODEL",),
                "variant": (("Turbo", "RAW"),),
                "steps": (
                    "INT",
                    {"default": 8, "min": 1, "max": _MAX_STEPS, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 16, "max": _MAX_DIMENSION, "step": 16},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 16, "max": _MAX_DIMENSION, "step": 16},
                ),
                "strict_official": ("BOOLEAN", {"default": True}),
            }
        }

    def inspect(
        self,
        model: object,
        variant: object,
        steps: object,
        width: object,
        height: object,
        strict_official: object,
    ) -> tuple[str]:
        result = build_profile_inspection(
            model=model,
            variant=variant,
            steps=steps,
            width=width,
            height=height,
            strict_official=strict_official,
        )
        return (result.report_json,)


class ScheduleInspector:
    """Verify and normalize one connected Sigmax schedule report."""

    DESCRIPTION = "Verifies connected SIGMAS against bounded versioned Sigmax schedule information."
    CATEGORY = "Sigmax/inspection"
    FUNCTION = "inspect"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("schedule_report",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": (
                    "STRING",
                    {"default": "", "multiline": True},
                ),
            }
        }

    def inspect(self, sigmas: object, schedule_info: object) -> tuple[str]:
        try:
            length = len(cast(Sequence[object], sigmas))
        except (TypeError, AttributeError) as exc:
            raise ScheduleContractError("host SIGMAS must be a bounded sequence") from exc
        if not 2 <= length <= _MAX_STEPS + 1:
            raise ScheduleContractError("host SIGMAS length is outside the inspector limit")
        try:
            values = tuple(float(cast(Any, value)) for value in cast(Sequence[object], sigmas))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScheduleContractError("host SIGMAS must contain numeric values") from exc
        result = build_schedule_inspection(
            sigmas=values,
            schedule_info=schedule_info,
        )
        return (result.report_json,)
