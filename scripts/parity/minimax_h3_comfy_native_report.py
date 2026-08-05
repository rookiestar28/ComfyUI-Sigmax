"""Strict report contract for pinned native ComfyUI MiniMax H3 parity."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final, cast

from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_BASE_FL2VA_PROFILE,
    MINIMAX_H3_BASE_REF2VA_PROFILE,
    MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT,
    MINIMAX_H3_NATIVE_MODEL_OUTPUT_SIGN,
    MINIMAX_H3_VIDEO_SHIFT,
    build_minimax_h3_comfyui_simple_schedule,
    map_minimax_h3_audio_coordinate,
)

REPORT_SCHEMA: Final = "sigmax.minimax-h3-comfy-native-parity/1"
REQUIRED_TRANSITIONS: Final = (4, 8, 12, 16, 20)
REQUIRED_AUDIO_PROBES: Final = (0.0, 0.125, 0.5, 0.875, 1.0)
SCHEDULE_TOLERANCE: Final = 1e-6
MAPPING_TOLERANCE: Final = 1e-6
VELOCITY_PROBE_TOLERANCE: Final = 1e-6
VELOCITY_PROBE_SIGMA: Final = 0.5
VELOCITY_PROBE_INPUT: Final = (0.125, -0.25)

COMFYUI_URL: Final = "https://github.com/Comfy-Org/ComfyUI"
COMFYUI_H3_REVISION: Final = MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT
SOURCE_BLOBS: Final = {
    "comfy/ldm/minimax/model.py": "494350d40b2678812af92c0cba75b5c564b02810",  # pragma: allowlist secret
    "comfy/model_base.py": "6631c9eb0e58f7328a11f2d042818c7faaa1bfee",  # pragma: allowlist secret
    "comfy/model_sampling.py": "5af336e76fd480a50425dd924f2ac9752083c09f",  # pragma: allowlist secret
    "comfy/samplers.py": "a280f3bb69fc6302802a6504f282965642a92a48",  # pragma: allowlist secret
    "comfy/supported_models.py": "51b58ed1e2a3c84d0e5132ccb0db214cccfd70a2",  # pragma: allowlist secret
}
DEPENDENCY_VERSIONS: Final = {
    "comfy-aimdo": "0.4.10",
    "comfy-kitchen": "0.2.23",
    "einops": "0.8.2",
    "numpy": "2.5.1",
    "packaging": "26.2",
    "pillow": "12.3.0",
    "psutil": "7.2.2",
    "safetensors": "0.8.0",
    "scipy": "1.18.0",
    "sentencepiece": "0.2.2",
    "torch": "2.13.0",
    "torchsde": "0.2.6",
    "torchvision": "0.28.0",
    "tqdm": "4.70.0",
    "transformers": "5.14.1",
}

_ROOT_FIELDS: Final = frozenset(
    {
        "adapter",
        "cases",
        "configuration",
        "environment",
        "profile",
        "schema",
        "source",
        "status",
        "tolerances",
    }
)
_CASE_FIELDS: Final = frozenset(
    {
        "difference_reason",
        "max_abs_error",
        "mean_abs_error",
        "native",
        "native_fingerprint",
        "sigmax",
        "sigmax_fingerprint",
        "status",
        "tolerance",
        "transitions",
    }
)
_MAPPING_FIELDS: Final = frozenset(
    {
        "audio_max_abs_error",
        "derivative_max_abs_error",
        "native_audio_sigma",
        "native_derivative",
        "sigmax_audio_sigma",
        "sigmax_derivative",
        "status",
        "tolerance",
        "video_sigma",
    }
)
_ADAPTER_FIELDS: Final = frozenset(
    {
        "coordinate_mapping",
        "direction",
        "host_conversion",
        "model_output_sign",
        "sign_adapter",
        "source_locator",
        "status",
        "velocity_probe",
    }
)
_VELOCITY_FIELDS: Final = frozenset(
    {
        "adapted_velocity",
        "data_ward_velocity",
        "max_abs_error",
        "native_model_output",
        "sigma",
        "slope",
        "status",
        "tolerance",
    }
)


def _git_chunks(value: str) -> list[str]:
    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _error_statistics(actual: Sequence[float], expected: Sequence[float]) -> tuple[float, float]:
    if len(actual) != len(expected) or not actual:
        raise ValueError("native comparison vectors must have equal non-zero length")
    errors = [abs(float(left) - float(right)) for left, right in zip(actual, expected, strict=True)]
    return max(errors), math.fsum(errors) / len(errors)


def _fixed_metadata(environment: Mapping[str, str]) -> dict[str, object]:
    return {
        "configuration": {
            "model_sampling": "ModelSamplingDiscreteFlow",
            "scheduler": "simple",
            "table_length": 1000,
            "terminal": "zero",
            "audio_shift": MINIMAX_H3_AUDIO_SHIFT,
            "video_shift": MINIMAX_H3_VIDEO_SHIFT,
        },
        "environment": dict(environment),
        "profile": {
            "family": "minimax_h3",
            "variants": [
                MINIMAX_H3_BASE_FL2VA_PROFILE.schema.model_variant,
                MINIMAX_H3_BASE_REF2VA_PROFILE.schema.model_variant,
            ],
        },
        "schema": REPORT_SCHEMA,
        "source": {
            "blobs": {path: _git_chunks(blob) for path, blob in SOURCE_BLOBS.items()},
            "evidence": "framework_reference",
            "locators": {
                "adapter": "comfy/ldm/minimax/model.py:35-45,642-646",
                "model_type": "comfy/model_base.py:115-118,2067-2069",
                "sampling": "comfy/model_sampling.py:284-322",
                "scheduler": "comfy/samplers.py:645-652,1363-1384",
                "settings": "comfy/supported_models.py:959-980",
            },
            "revision_chunks": _git_chunks(COMFYUI_H3_REVISION),
            "url": COMFYUI_URL,
        },
        "tolerances": {
            "mapping_float32_max_abs": repr(MAPPING_TOLERANCE),
            "schedule_float32_max_abs": repr(SCHEDULE_TOLERANCE),
        },
    }


def _finite_vector(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _sigma_vector(value: object, *, transitions: int, name: str) -> tuple[float, ...]:
    vector = _finite_vector(value, name=name)
    if len(vector) != transitions + 1 or vector[0] != 1.0 or vector[-1] != 0.0:
        raise ValueError(f"{name} must contain transitions + 1 values and zero terminal")
    if any(left <= right for left, right in pairwise(vector)):
        raise ValueError(f"{name} must be strictly decreasing")
    if any(value < 0.0 or value > 1.0 for value in vector):
        raise ValueError(f"{name} must stay in [0, 1]")
    return vector


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _exact_fields(value: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are incomplete or unknown")


def _error_string(value: object, *, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} error must be a string")
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{name} error is invalid") from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} error is invalid")
    return number


def _validate_case(value: object, *, transitions: int) -> None:
    case = _mapping(value, name="case")
    _exact_fields(case, _CASE_FIELDS, name="case")
    native = _sigma_vector(case.get("native"), transitions=transitions, name="case.native")
    sigmax = _sigma_vector(case.get("sigmax"), transitions=transitions, name="case.sigmax")
    maximum, mean = _error_statistics(native, sigmax)
    if case.get("transitions") != transitions:
        raise ValueError("case transitions are invalid")
    if case.get("status") != "PASS" or case.get("tolerance") != repr(SCHEDULE_TOLERANCE):
        raise ValueError("case status or tolerance is invalid")
    if _error_string(case.get("max_abs_error"), name="case") != maximum:
        raise ValueError("case max error does not match vectors")
    if _error_string(case.get("mean_abs_error"), name="case") != mean:
        raise ValueError("case mean error does not match vectors")
    if maximum > SCHEDULE_TOLERANCE:
        raise ValueError("case error exceeds tolerance")
    expected_fingerprint = numerical_fingerprint(
        sigmax, domain=SigmaDomain.UNIT_FLOW, precision="float32"
    )
    if case.get("sigmax_fingerprint") != expected_fingerprint:
        raise ValueError("case Sigmax fingerprint does not match")
    if case.get("native_fingerprint") != numerical_fingerprint(
        native, domain=SigmaDomain.UNIT_FLOW, precision="float32"
    ):
        raise ValueError("case native fingerprint does not match")


def _validate_mapping_case(value: object, *, video_sigma: float) -> None:
    case = _mapping(value, name="mapping case")
    _exact_fields(case, _MAPPING_FIELDS, name="mapping case")
    if case.get("video_sigma") != video_sigma:
        raise ValueError("mapping video sigma is invalid")
    native_audio = _finite_vector(case.get("native_audio_sigma"), name="native audio")
    sigmax_audio = _finite_vector(case.get("sigmax_audio_sigma"), name="Sigmax audio")
    native_derivative = _finite_vector(case.get("native_derivative"), name="native derivative")
    sigmax_derivative = _finite_vector(case.get("sigmax_derivative"), name="Sigmax derivative")
    if len(native_audio) != len(sigmax_audio) or len(native_derivative) != len(sigmax_derivative):
        raise ValueError("mapping vector dimensions do not match")
    if len(native_audio) != 1 or len(native_derivative) != 1:
        raise ValueError("mapping vectors must contain one scalar")
    audio_error = abs(native_audio[0] - sigmax_audio[0])
    derivative_error = abs(native_derivative[0] - sigmax_derivative[0])
    if case.get("status") != "PASS" or case.get("tolerance") != repr(MAPPING_TOLERANCE):
        raise ValueError("mapping status or tolerance is invalid")
    if _error_string(case.get("audio_max_abs_error"), name="mapping audio") != audio_error:
        raise ValueError("mapping audio error does not match")
    if (
        _error_string(case.get("derivative_max_abs_error"), name="mapping derivative")
        != derivative_error
    ):
        raise ValueError("mapping derivative error does not match")
    if audio_error > MAPPING_TOLERANCE or derivative_error > MAPPING_TOLERANCE:
        raise ValueError("mapping error exceeds tolerance")


def _validate_adapter(value: object) -> None:
    adapter = _mapping(value, name="adapter")
    _exact_fields(adapter, _ADAPTER_FIELDS, name="adapter")
    if adapter.get("direction") != "data_ward":
        raise ValueError("native adapter direction is invalid")
    if adapter.get("model_output_sign") != MINIMAX_H3_NATIVE_MODEL_OUTPUT_SIGN:
        raise ValueError("native adapter model-output sign is invalid")
    if adapter.get("sign_adapter") != "explicit_negate_to_data_ward":
        raise ValueError("native adapter sign policy is invalid")
    if adapter.get("host_conversion") != "CONST.calculate_denoised_then_to_d":
        raise ValueError("native adapter host conversion is invalid")
    if adapter.get("source_locator") != "comfy/ldm/minimax/model.py:642-646":
        raise ValueError("native adapter source locator is invalid")
    if adapter.get("status") != "PASS":
        raise ValueError("native adapter status is invalid")
    mapping_cases = adapter.get("coordinate_mapping")
    if not isinstance(mapping_cases, list) or len(mapping_cases) != len(REQUIRED_AUDIO_PROBES):
        raise ValueError("native coordinate mapping cases are incomplete")
    probe = _mapping(adapter.get("velocity_probe"), name="velocity probe")
    _exact_fields(probe, _VELOCITY_FIELDS, name="velocity probe")
    if probe.get("sigma") != VELOCITY_PROBE_SIGMA:
        raise ValueError("velocity probe sigma is invalid")
    if probe.get("tolerance") != repr(VELOCITY_PROBE_TOLERANCE):
        raise ValueError("velocity probe tolerance is invalid")
    slope = probe.get("slope")
    if (
        isinstance(slope, bool)
        or not isinstance(slope, (int, float))
        or not math.isfinite(float(slope))
    ):
        raise ValueError("velocity probe slope is invalid")
    if float(slope) <= 0.0:
        raise ValueError("velocity probe slope must be positive")
    dataward = _finite_vector(probe.get("data_ward_velocity"), name="data-ward velocity")
    native_output = _finite_vector(probe.get("native_model_output"), name="native model output")
    adapted = _finite_vector(probe.get("adapted_velocity"), name="adapted velocity")
    if len(dataward) != len(native_output) or len(dataward) != len(adapted):
        raise ValueError("velocity probe dimensions do not match")
    expected_native = tuple(-float(slope) * value for value in dataward)
    expected_adapted = tuple(-value / float(slope) for value in native_output)
    if native_output != expected_native or adapted != expected_adapted:
        raise ValueError("velocity probe sign adapter arithmetic is invalid")
    maximum = max(abs(left - right) for left, right in zip(adapted, dataward, strict=True))
    if (
        probe.get("status") != "PASS"
        or _error_string(probe.get("max_abs_error"), name="velocity probe") != maximum
    ):
        raise ValueError("velocity probe status or error is invalid")
    if maximum > VELOCITY_PROBE_TOLERANCE:
        raise ValueError("velocity probe error exceeds tolerance")


def build_native_report(
    native_vectors: Mapping[int, Sequence[float]],
    *,
    native_mappings: Mapping[float, tuple[float, float]],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Build complete report from vectors emitted by pinned native modules."""

    if tuple(sorted(native_vectors)) != REQUIRED_TRANSITIONS:
        raise ValueError("native schedule cases must be exactly 4, 8, 12, 16, and 20 transitions")
    if tuple(sorted(native_mappings)) != REQUIRED_AUDIO_PROBES:
        raise ValueError("native mapping probes are incomplete")

    cases: list[dict[str, object]] = []
    for transitions in REQUIRED_TRANSITIONS:
        native = tuple(float(value) for value in native_vectors[transitions])
        sigmax = tuple(
            float(value)
            for value in build_minimax_h3_comfyui_simple_schedule(
                variant=MINIMAX_H3_BASE_FL2VA_PROFILE.variant,
                transitions=transitions,
            ).sigmas
        )
        maximum, mean = _error_statistics(native, sigmax)
        cases.append(
            {
                "difference_reason": "float32_native_table_and_simple_indexing",
                "max_abs_error": repr(maximum),
                "mean_abs_error": repr(mean),
                "native": list(native),
                "native_fingerprint": numerical_fingerprint(
                    native, domain=SigmaDomain.UNIT_FLOW, precision="float32"
                ),
                "sigmax": list(sigmax),
                "sigmax_fingerprint": numerical_fingerprint(
                    sigmax, domain=SigmaDomain.UNIT_FLOW, precision="float32"
                ),
                "status": "PASS" if maximum <= SCHEDULE_TOLERANCE else "FAIL",
                "tolerance": repr(SCHEDULE_TOLERANCE),
                "transitions": transitions,
            }
        )

    mapping_cases: list[dict[str, object]] = []
    for video_sigma in REQUIRED_AUDIO_PROBES:
        native_audio, native_derivative = native_mappings[video_sigma]
        expected = map_minimax_h3_audio_coordinate(video_sigma, precision="float32")
        audio_error = abs(float(native_audio) - expected.audio_sigma)
        derivative_error = abs(float(native_derivative) - expected.derivative)
        mapping_cases.append(
            {
                "audio_max_abs_error": repr(audio_error),
                "derivative_max_abs_error": repr(derivative_error),
                "native_audio_sigma": [float(native_audio)],
                "native_derivative": [float(native_derivative)],
                "sigmax_audio_sigma": [expected.audio_sigma],
                "sigmax_derivative": [expected.derivative],
                "status": "PASS"
                if max(audio_error, derivative_error) <= MAPPING_TOLERANCE
                else "FAIL",
                "tolerance": repr(MAPPING_TOLERANCE),
                "video_sigma": video_sigma,
            }
        )

    probe_slope = float(native_mappings[VELOCITY_PROBE_SIGMA][1])
    dataward_velocity = tuple(float(value) for value in VELOCITY_PROBE_INPUT)
    native_model_output = tuple(-probe_slope * value for value in dataward_velocity)
    adapted_velocity = tuple(-value / probe_slope for value in native_model_output)
    velocity_error = max(
        abs(left - right) for left, right in zip(adapted_velocity, dataward_velocity, strict=True)
    )

    report: dict[str, Any] = {
        **_fixed_metadata(environment),
        "adapter": {
            "direction": "data_ward",
            "host_conversion": "CONST.calculate_denoised_then_to_d",
            "model_output_sign": MINIMAX_H3_NATIVE_MODEL_OUTPUT_SIGN,
            "sign_adapter": "explicit_negate_to_data_ward",
            "source_locator": "comfy/ldm/minimax/model.py:642-646",
            "status": "PASS",
            "velocity_probe": {
                "adapted_velocity": list(adapted_velocity),
                "data_ward_velocity": list(dataward_velocity),
                "max_abs_error": repr(velocity_error),
                "native_model_output": list(native_model_output),
                "sigma": VELOCITY_PROBE_SIGMA,
                "slope": probe_slope,
                "status": "PASS" if velocity_error <= VELOCITY_PROBE_TOLERANCE else "FAIL",
                "tolerance": repr(VELOCITY_PROBE_TOLERANCE),
            },
        },
        "cases": cases,
        "status": "PASS",
    }
    report["adapter"]["coordinate_mapping"] = mapping_cases
    return validate_native_report(report)


def _exact_metadata(environment: Mapping[str, str]) -> dict[str, object]:
    return {**_fixed_metadata(environment)}


def validate_native_report(report: object) -> dict[str, Any]:
    """Fail closed unless native schedule, paired mapping, and sign metadata are complete."""

    root = _mapping(report, name="report")
    _exact_fields(root, _ROOT_FIELDS, name="report")
    if root.get("schema") != REPORT_SCHEMA or root.get("status") != "PASS":
        raise ValueError("native report schema or status is invalid")
    environment = _mapping(root.get("environment"), name="environment")
    if any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
    ):
        raise ValueError("environment metadata is invalid")
    expected = _exact_metadata(cast(Mapping[str, str], environment))
    for field in ("configuration", "environment", "profile", "schema", "source", "tolerances"):
        if root.get(field) != expected[field]:
            raise ValueError(f"{field} metadata does not match the pinned contract")
    _validate_adapter(root.get("adapter"))
    mapping_cases = cast(Mapping[str, object], root["adapter"]).get("coordinate_mapping")
    if not isinstance(mapping_cases, list):
        raise ValueError("native coordinate mapping cases are incomplete")
    for raw_case, video_sigma in zip(mapping_cases, REQUIRED_AUDIO_PROBES, strict=True):
        _validate_mapping_case(raw_case, video_sigma=video_sigma)

    cases = root.get("cases")
    if not isinstance(cases, list) or len(cases) != len(REQUIRED_TRANSITIONS):
        raise ValueError("native schedule cases are incomplete")
    observed: list[int] = []
    for raw_case, transitions in zip(cases, REQUIRED_TRANSITIONS, strict=True):
        _validate_case(raw_case, transitions=transitions)
        observed.append(transitions)
    if tuple(observed) != REQUIRED_TRANSITIONS:
        raise ValueError("native schedule case order is invalid")
    return cast(dict[str, Any], report)


def canonical_json(report: Mapping[str, object]) -> str:
    """Encode one report without nondeterministic whitespace or key order."""

    return (
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    )


__all__ = [
    "COMFYUI_H3_REVISION",
    "COMFYUI_URL",
    "DEPENDENCY_VERSIONS",
    "MAPPING_TOLERANCE",
    "REPORT_SCHEMA",
    "REQUIRED_AUDIO_PROBES",
    "REQUIRED_TRANSITIONS",
    "SCHEDULE_TOLERANCE",
    "SOURCE_BLOBS",
    "VELOCITY_PROBE_INPUT",
    "VELOCITY_PROBE_SIGMA",
    "VELOCITY_PROBE_TOLERANCE",
    "build_native_report",
    "canonical_json",
    "validate_native_report",
]
