"""Strict report contract for native ComfyUI Krea 2 schedule parity."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final, cast

from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles import build_krea2_turbo_schedule

REPORT_SCHEMA: Final = "sigmax.krea2-comfy-native-parity/1"
REQUIRED_STEPS: Final = (4, 8, 12, 16)
COMFYUI_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
COMFYUI_URL: Final = "https://github.com/Comfy-Org/ComfyUI"
SOURCE_BLOBS: Final = {
    "comfy/model_base.py": (
        "ee6dc57a25b5d071623a81b5f82703a2e5d5c6b6"  # pragma: allowlist secret
    ),
    "comfy/model_sampling.py": (
        "5af336e76fd480a50425dd924f2ac9752083c09f"  # pragma: allowlist secret
    ),
    "comfy/samplers.py": (
        "9f571ece9ab7d79e35a8c3437a92fd93ccce0c09"  # pragma: allowlist secret
    ),
    "comfy/supported_models.py": (
        "ca89850a594fba87ff669dbd94b027efc1ad79dc"  # pragma: allowlist secret
    ),
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
EXACT_TOLERANCE: Final = 1e-6
QUANTIZED_TOLERANCE: Final = 2e-4

_ROOT_FIELDS: Final = frozenset(
    {
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
        "evidence",
        "exact_table_positions",
        "max_abs_error",
        "mean_abs_error",
        "native",
        "native_fingerprint",
        "sigmax",
        "sigmax_fingerprint",
        "status",
        "steps",
        "tolerance",
    }
)


def _float32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _git_chunks(value: str) -> list[str]:
    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _error_statistics(
    actual: Sequence[float],
    expected: Sequence[float],
) -> tuple[float, float]:
    if len(actual) != len(expected):
        raise ValueError("comparison vector length mismatch")
    errors = [abs(float(left) - float(right)) for left, right in zip(actual, expected, strict=True)]
    return max(errors), math.fsum(errors) / len(errors)


def _fixed_metadata() -> dict[str, object]:
    return {
        "configuration": {
            "model_sampling": "ModelSamplingFlux",
            "mu": "1.15",
            "scheduler": "simple",
            "table_length": 10000,
            "terminal": "zero",
        },
        "environment": {
            "device": "cpu",
            "dtype": "float32",
            "numpy": DEPENDENCY_VERSIONS["numpy"],
            "python": "3.13",
            "torch": DEPENDENCY_VERSIONS["torch"],
        },
        "profile": {"id": "krea2.turbo.official", "version": "1"},
        "schema": REPORT_SCHEMA,
        "source": {
            "blobs": {path: _git_chunks(blob) for path, blob in SOURCE_BLOBS.items()},
            "evidence": "framework_reference",
            "locators": {
                "model_type": "comfy/model_base.py:2330-2332",
                "sampling": "comfy/model_sampling.py:382-412",
                "scheduler": "comfy/samplers.py:645-652,1348-1376",
                "settings": "comfy/supported_models.py:1859-1867",
            },
            "revision_chunks": _git_chunks(COMFYUI_REVISION),
            "url": COMFYUI_URL,
        },
        "tolerances": {
            "exact_table_positions_max_abs": repr(EXACT_TOLERANCE),
            "integer_index_quantization_max_abs": repr(QUANTIZED_TOLERANCE),
        },
    }


def build_native_report(
    native_vectors: Mapping[int, Sequence[float]],
) -> dict[str, Any]:
    """Build one complete report from vectors produced by pinned native modules."""

    if tuple(sorted(native_vectors)) != REQUIRED_STEPS:
        raise ValueError("native case set must be exactly 4, 8, 12, and 16 steps")

    cases: list[dict[str, object]] = []
    for steps in REQUIRED_STEPS:
        native = tuple(float(value) for value in native_vectors[steps])
        sigmax = tuple(_float32(value) for value in build_krea2_turbo_schedule(steps=steps).sigmas)
        maximum, mean = _error_statistics(native, sigmax)
        exact_positions = 10000 % steps == 0
        tolerance = EXACT_TOLERANCE if exact_positions else QUANTIZED_TOLERANCE
        cases.append(
            {
                "difference_reason": (
                    "float32_evaluation"
                    if exact_positions
                    else "simple_scheduler_integer_index_quantization"
                ),
                "evidence": "official" if steps == 8 else "modified",
                "exact_table_positions": exact_positions,
                "max_abs_error": repr(maximum),
                "mean_abs_error": repr(mean),
                "native": list(native),
                "native_fingerprint": numerical_fingerprint(
                    native,
                    domain=SigmaDomain.UNIT_FLOW,
                    precision="float32",
                ),
                "sigmax": list(sigmax),
                "sigmax_fingerprint": numerical_fingerprint(
                    sigmax,
                    domain=SigmaDomain.UNIT_FLOW,
                    precision="float32",
                ),
                "status": "PASS" if maximum <= tolerance else "FAIL",
                "steps": steps,
                "tolerance": repr(tolerance),
            }
        )

    report: dict[str, Any] = {
        **_fixed_metadata(),
        "cases": cases,
        "status": "PASS",
    }
    return validate_native_report(report)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _require_exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are incomplete or unknown")


def _require_vector(value: object, *, steps: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != steps + 1:
        raise ValueError(f"{name} vector length must equal steps + 1")
    vector: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{name} vector must contain finite numbers")
        vector.append(float(item))
    if vector[0] != 1.0 or vector[-1] != 0.0:
        raise ValueError(f"{name} vector endpoints are invalid")
    if any(left <= right for left, right in pairwise(vector)):
        raise ValueError(f"{name} vector must be strictly decreasing")
    return tuple(vector)


def _require_nonnegative_float_string(value: object, *, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} error value must be a string")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} error value is invalid") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} error value is invalid")
    return parsed


def validate_native_report(report: object) -> dict[str, Any]:
    """Fail closed unless a report proves the complete M2-04 contract."""

    root = _require_mapping(report, name="report")
    _require_exact_fields(root, _ROOT_FIELDS, name="report")
    if root.get("schema") != REPORT_SCHEMA:
        raise ValueError("report schema is unsupported")
    if root.get("status") != "PASS":
        raise ValueError("report status must be PASS")

    expected = _fixed_metadata()
    for field in (
        "configuration",
        "environment",
        "profile",
        "schema",
        "source",
        "tolerances",
    ):
        if root.get(field) != expected[field]:
            raise ValueError(f"{field} metadata does not match the pinned contract")

    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(REQUIRED_STEPS):
        raise ValueError("report case set is incomplete")

    observed_steps: list[int] = []
    for raw_case in raw_cases:
        case = _require_mapping(raw_case, name="case")
        _require_exact_fields(case, _CASE_FIELDS, name="case")
        steps = case.get("steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps not in REQUIRED_STEPS:
            raise ValueError("case steps are invalid")
        observed_steps.append(steps)

        exact_positions = 10000 % steps == 0
        tolerance = EXACT_TOLERANCE if exact_positions else QUANTIZED_TOLERANCE
        expected_reason = (
            "float32_evaluation"
            if exact_positions
            else "simple_scheduler_integer_index_quantization"
        )
        if case.get("evidence") != ("official" if steps == 8 else "modified"):
            raise ValueError("case evidence is invalid")
        if case.get("exact_table_positions") is not exact_positions:
            raise ValueError("case table-position policy is invalid")
        if case.get("difference_reason") != expected_reason:
            raise ValueError("case difference reason is invalid")
        if case.get("status") != "PASS" or case.get("tolerance") != repr(tolerance):
            raise ValueError("case status or tolerance is invalid")

        native = _require_vector(case.get("native"), steps=steps, name="native")
        sigmax = _require_vector(case.get("sigmax"), steps=steps, name="sigmax")
        expected_sigmax = tuple(
            _float32(value) for value in build_krea2_turbo_schedule(steps=steps).sigmas
        )
        if sigmax != expected_sigmax:
            raise ValueError("sigmax vector does not match the current profile")

        maximum, mean = _error_statistics(native, sigmax)
        if (
            _require_nonnegative_float_string(case.get("max_abs_error"), name="maximum") != maximum
            or _require_nonnegative_float_string(case.get("mean_abs_error"), name="mean") != mean
        ):
            raise ValueError("case error statistics do not match complete vectors")
        if maximum > tolerance:
            raise ValueError("case error exceeds tolerance")

        expected_native_fingerprint = numerical_fingerprint(
            native,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float32",
        )
        expected_sigmax_fingerprint = numerical_fingerprint(
            sigmax,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float32",
        )
        if case.get("native_fingerprint") != expected_native_fingerprint:
            raise ValueError("native fingerprint is invalid")
        if case.get("sigmax_fingerprint") != expected_sigmax_fingerprint:
            raise ValueError("sigmax fingerprint is invalid")

    if tuple(observed_steps) != REQUIRED_STEPS:
        raise ValueError("report case order or set is invalid")
    return cast(dict[str, Any], report)


def canonical_json(report: object) -> str:
    """Serialize one validated report as deterministic JSON with a terminal newline."""

    validated = validate_native_report(report)
    return (
        json.dumps(
            validated,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
