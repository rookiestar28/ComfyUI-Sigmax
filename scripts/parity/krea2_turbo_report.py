"""Canonical report construction for Krea 2 Turbo schedule parity."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final, Literal, cast

from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles import build_krea2_turbo_schedule
from scripts.parity.krea2_official import (
    KREA_LOCATOR,
    KREA_REVISION,
    KREA_SOURCE_URL,
    official_krea2_turbo_sigmas,
)

REPORT_SCHEMA: Final = "sigmax.krea2-turbo-parity/1"
REQUIRED_STEPS: Final = (4, 8, 12, 16)
DIFFUSERS_VERSION: Final = "0.39.0"
NUMPY_VERSION: Final = "2.3.4"
TORCH_VERSION: Final = "2.9.0"
DIFFUSERS_SOURCE_URL: Final = "https://github.com/huggingface/diffusers"
DIFFUSERS_TAG: Final = "v0.39.0"
DIFFUSERS_REVISION: Final = "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret
DIFFUSERS_PIPELINE_BLOB: Final = (
    "51d33cb4861903ee1ac682f8da3b7256013656aa"  # pragma: allowlist secret
)
DIFFUSERS_SCHEDULER_BLOB: Final = (
    "7b207f7820797c53b093452ca2bc52938a8d84e7"  # pragma: allowlist secret
)
KREA_TOLERANCE: Final = 1e-8
DIFFUSERS_TOLERANCE: Final = 1e-6

_ROOT_FIELDS: Final = frozenset(
    {
        "cases",
        "configuration",
        "environment",
        "profile",
        "schema",
        "sources",
        "status",
        "tolerances",
    }
)
_CASE_FIELDS: Final = frozenset({"comparisons", "evidence", "steps"})
_COMPARISON_FIELDS: Final = frozenset(
    {
        "device",
        "dtype",
        "fingerprint",
        "max_abs_error",
        "mean_abs_error",
        "reference",
        "sigmax",
        "status",
        "tolerance",
    }
)


def _float32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _float32_vector(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(_float32(float(value)) for value in values)


def _git_object_chunks(value: str) -> list[str]:
    """Keep exact public Git identities without secret-scanner false positives in JSON."""

    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _error_statistics(
    actual: Sequence[float],
    expected: Sequence[float],
) -> tuple[float, float]:
    if len(actual) != len(expected):
        raise ValueError("comparison vector length mismatch")
    errors = [abs(float(left) - float(right)) for left, right in zip(actual, expected, strict=True)]
    return max(errors), math.fsum(errors) / len(errors)


def _comparison(
    *,
    sigmax: tuple[float, ...],
    reference: tuple[float, ...],
    dtype: str,
    tolerance: float,
) -> dict[str, object]:
    maximum, mean = _error_statistics(sigmax, reference)
    precision: Literal["float32", "float64"] = "float64" if dtype == "float64" else "float32"
    return {
        "device": "cpu",
        "dtype": dtype,
        "fingerprint": numerical_fingerprint(
            sigmax,
            domain=SigmaDomain.UNIT_FLOW,
            precision=precision,
        ),
        "max_abs_error": repr(maximum),
        "mean_abs_error": repr(mean),
        "reference": list(reference),
        "sigmax": list(sigmax),
        "status": "PASS" if maximum <= tolerance else "FAIL",
        "tolerance": repr(tolerance),
    }


def _fixed_metadata(environment: Mapping[str, str]) -> dict[str, object]:
    return {
        "configuration": {
            "base_grid": "krea.reciprocal_step",
            "mu": "1.15",
            "terminal": "zero",
        },
        "environment": dict(environment),
        "profile": {"id": "krea2.turbo.official", "version": "1"},
        "schema": REPORT_SCHEMA,
        "sources": {
            "diffusers": {
                "evidence": "framework_reference",
                "pipeline_blob_chunks": _git_object_chunks(DIFFUSERS_PIPELINE_BLOB),
                "pipeline_locator": "src/diffusers/pipelines/krea2/pipeline_krea2.py:613-630",
                "revision_chunks": _git_object_chunks(DIFFUSERS_REVISION),
                "scheduler_blob_chunks": _git_object_chunks(DIFFUSERS_SCHEDULER_BLOB),
                "scheduler_locator": (
                    "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:283-378"
                ),
                "tag": DIFFUSERS_TAG,
                "url": DIFFUSERS_SOURCE_URL,
            },
            "krea": {
                "evidence": "official",
                "locator": KREA_LOCATOR,
                "revision_chunks": _git_object_chunks(KREA_REVISION),
                "url": KREA_SOURCE_URL,
            },
        },
        "tolerances": {
            "diffusers_float32_max_abs": "1e-6",
            "krea_float64_max_abs": "1e-8",
        },
    }


def build_parity_report(
    diffusers_vectors: Mapping[int, Sequence[float]],
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Build and validate one complete authoritative/framework parity report."""

    if tuple(sorted(diffusers_vectors)) != REQUIRED_STEPS:
        raise ValueError("diffusers case set must be exactly 4, 8, 12, and 16 steps")

    cases: list[dict[str, object]] = []
    for steps in REQUIRED_STEPS:
        result = build_krea2_turbo_schedule(steps=steps)
        sigmax_float64 = tuple(result.sigmas)
        sigmax_float32 = _float32_vector(sigmax_float64)
        diffusers_float32 = tuple(float(value) for value in diffusers_vectors[steps])
        cases.append(
            {
                "comparisons": {
                    "diffusers_float32": _comparison(
                        sigmax=sigmax_float32,
                        reference=diffusers_float32,
                        dtype="float32",
                        tolerance=DIFFUSERS_TOLERANCE,
                    ),
                    "krea_float64": _comparison(
                        sigmax=sigmax_float64,
                        reference=official_krea2_turbo_sigmas(steps),
                        dtype="float64",
                        tolerance=KREA_TOLERANCE,
                    ),
                },
                "evidence": "official" if steps == 8 else "modified",
                "steps": steps,
            }
        )

    report: dict[str, Any] = {
        **_fixed_metadata(environment),
        "cases": cases,
        "status": "PASS",
    }
    return validate_parity_report(report)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
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


def _require_error_string(value: object, *, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} error value must be a string")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} error value is invalid") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} error value is invalid")
    return parsed


def _validate_comparison(
    raw: object,
    *,
    steps: int,
    name: str,
    dtype: str,
    tolerance: float,
) -> None:
    comparison = _require_mapping(raw, name=name)
    _require_exact_fields(comparison, _COMPARISON_FIELDS, name=name)
    if comparison.get("status") != "PASS":
        raise ValueError(f"{name} status must be PASS")
    if comparison.get("device") != "cpu" or comparison.get("dtype") != dtype:
        raise ValueError(f"{name} execution metadata is invalid")
    if comparison.get("tolerance") != repr(tolerance):
        raise ValueError(f"{name} tolerance is invalid")

    sigmax = _require_vector(comparison.get("sigmax"), steps=steps, name=f"{name} sigmax")
    reference = _require_vector(
        comparison.get("reference"),
        steps=steps,
        name=f"{name} reference",
    )
    maximum, mean = _error_statistics(sigmax, reference)
    stored_maximum = _require_error_string(comparison.get("max_abs_error"), name=name)
    stored_mean = _require_error_string(comparison.get("mean_abs_error"), name=name)
    if stored_maximum != maximum or stored_mean != mean:
        raise ValueError(f"{name} error statistics do not match complete vectors")
    if maximum > tolerance:
        raise ValueError(f"{name} error exceeds tolerance")

    precision: Literal["float32", "float64"] = "float64" if dtype == "float64" else "float32"
    expected_fingerprint = numerical_fingerprint(
        sigmax,
        domain=SigmaDomain.UNIT_FLOW,
        precision=precision,
    )
    if comparison.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"{name} fingerprint does not match the Sigmax vector")


def validate_parity_report(report: object) -> dict[str, Any]:
    """Fail closed unless a report proves the exact M2-03 evidence contract."""

    root = _require_mapping(report, name="report")
    _require_exact_fields(root, _ROOT_FIELDS, name="report")
    if root.get("schema") != REPORT_SCHEMA:
        raise ValueError("report schema is unsupported")
    if root.get("status") != "PASS":
        raise ValueError("report status must be PASS")

    expected_environment = {
        "device": "cpu",
        "diffusers": DIFFUSERS_VERSION,
        "numpy": NUMPY_VERSION,
        "torch": TORCH_VERSION,
    }
    expected_metadata = _fixed_metadata(expected_environment)
    for field in (
        "configuration",
        "environment",
        "profile",
        "schema",
        "sources",
        "tolerances",
    ):
        if root.get(field) != expected_metadata[field]:
            category = "source" if field == "sources" else field
            raise ValueError(f"{category} metadata does not match the pinned contract")

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
        expected_evidence = "official" if steps == 8 else "modified"
        if case.get("evidence") != expected_evidence:
            raise ValueError("case evidence is invalid")
        comparisons = _require_mapping(case.get("comparisons"), name="case comparisons")
        if set(comparisons) != {"diffusers_float32", "krea_float64"}:
            raise ValueError("case comparisons are incomplete")
        _validate_comparison(
            comparisons["krea_float64"],
            steps=steps,
            name="krea_float64",
            dtype="float64",
            tolerance=KREA_TOLERANCE,
        )
        _validate_comparison(
            comparisons["diffusers_float32"],
            steps=steps,
            name="diffusers_float32",
            dtype="float32",
            tolerance=DIFFUSERS_TOLERANCE,
        )
    if tuple(observed_steps) != REQUIRED_STEPS:
        raise ValueError("report case order or set is invalid")

    return cast(dict[str, Any], report)


def canonical_json(report: object) -> str:
    """Serialize one validated report as deterministic JSON with a terminal newline."""

    validated = validate_parity_report(report)
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
