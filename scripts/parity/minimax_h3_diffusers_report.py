"""Canonical report construction for isolated MiniMax H3 Diffusers parity."""

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
    MINIMAX_H3_DIFFUSERS_REVISION,
    MINIMAX_H3_VIDEO_SHIFT,
    build_minimax_h3_schedule,
    map_minimax_h3_audio_coordinate,
)
from scripts.parity.minimax_h3_official import (
    BRANCH_URL,
    SCHEDULER_LOCATOR,
    SOURCE_URL,
    clean_room_dataward_step,
)

REPORT_SCHEMA: Final = "sigmax.minimax-h3-diffusers-parity/1"
REQUIRED_GRID_POINTS: Final = (4, 8, 12, 16, 20)
TORCH_VERSION: Final = "2.9.0"
NUMPY_VERSION: Final = "2.3.4"
VIDEO_TOLERANCE: Final = 1e-6
AUDIO_TOLERANCE: Final = 1e-6
VELOCITY_TOLERANCE: Final = 1e-6

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
        "velocity_probe",
    }
)
_CASE_FIELDS: Final = frozenset({"audio", "grid_points", "video"})
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
_PROBE_FIELDS: Final = frozenset(
    {
        "device",
        "dtype",
        "expected",
        "max_abs_error",
        "mean_abs_error",
        "reference",
        "sample",
        "sigma",
        "sigma_next",
        "status",
        "timestep",
        "tolerance",
        "velocity",
    }
)


def _git_object_chunks(value: str) -> list[str]:
    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _error_statistics(actual: Sequence[float], expected: Sequence[float]) -> tuple[float, float]:
    if len(actual) != len(expected) or not actual:
        raise ValueError("comparison vectors must have equal non-zero length")
    errors = [abs(float(left) - float(right)) for left, right in zip(actual, expected, strict=True)]
    return max(errors), math.fsum(errors) / len(errors)


def _comparison(
    *,
    sigmax: tuple[float, ...],
    reference: tuple[float, ...],
    tolerance: float,
) -> dict[str, object]:
    maximum, mean = _error_statistics(sigmax, reference)
    return {
        "device": "cpu",
        "dtype": "float32",
        "fingerprint": numerical_fingerprint(
            sigmax,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float32",
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
            "audio_shift": MINIMAX_H3_AUDIO_SHIFT,
            "base_grid": "endpoint_inclusive",
            "terminal": "zero_in_grid",
            "velocity_direction": "data_ward",
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
        "sources": {
            "diffusers": {
                "branch_url": BRANCH_URL,
                "revision_chunks": _git_object_chunks(MINIMAX_H3_DIFFUSERS_REVISION),
                "scheduler_locator": SCHEDULER_LOCATOR,
                "url": SOURCE_URL,
            }
        },
        "tolerances": {
            "audio_float32_max_abs": repr(AUDIO_TOLERANCE),
            "velocity_float32_max_abs": repr(VELOCITY_TOLERANCE),
            "video_float32_max_abs": repr(VIDEO_TOLERANCE),
        },
    }


def _tuple_finite(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{name} must be a non-empty vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _strict_sigma_vector(value: object, *, points: int, name: str) -> tuple[float, ...]:
    vector = _tuple_finite(value, name=name)
    if len(vector) != points or vector[0] != 1.0 or vector[-1] != 0.0:
        raise ValueError(f"{name} must contain exactly {points} points with 1.0 and 0.0 endpoints")
    if any(left <= right for left, right in pairwise(vector)):
        raise ValueError(f"{name} must be strictly decreasing")
    if any(value < 0.0 or value > 1.0 for value in vector):
        raise ValueError(f"{name} must stay inside [0, 1]")
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
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} error is invalid") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} error is invalid")
    return parsed


def _validate_comparison(
    value: object,
    *,
    points: int,
    name: str,
    tolerance: float,
) -> None:
    comparison = _mapping(value, name=name)
    _exact_fields(comparison, _COMPARISON_FIELDS, name=name)
    if comparison.get("device") != "cpu" or comparison.get("dtype") != "float32":
        raise ValueError(f"{name} execution metadata is invalid")
    if comparison.get("status") != "PASS" or comparison.get("tolerance") != repr(tolerance):
        raise ValueError(f"{name} status or tolerance is invalid")
    sigmax = _strict_sigma_vector(comparison.get("sigmax"), points=points, name=f"{name}.sigmax")
    reference = _strict_sigma_vector(
        comparison.get("reference"), points=points, name=f"{name}.reference"
    )
    maximum, mean = _error_statistics(sigmax, reference)
    if _error_string(comparison.get("max_abs_error"), name=name) != maximum:
        raise ValueError(f"{name} max error does not match vectors")
    if _error_string(comparison.get("mean_abs_error"), name=name) != mean:
        raise ValueError(f"{name} mean error does not match vectors")
    if maximum > tolerance:
        raise ValueError(f"{name} error exceeds tolerance")
    expected_fingerprint = numerical_fingerprint(
        sigmax,
        domain=SigmaDomain.UNIT_FLOW,
        precision="float32",
    )
    if comparison.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"{name} fingerprint does not match Sigmax values")


def _probe(
    value: Mapping[str, object],
    *,
    require_reference: bool = True,
) -> None:
    _exact_fields(value, _PROBE_FIELDS, name="velocity_probe")
    for field in ("sample", "velocity", "reference", "expected"):
        _tuple_finite(value.get(field), name=f"velocity_probe.{field}")
    sample = _tuple_finite(value.get("sample"), name="velocity_probe.sample")
    velocity = _tuple_finite(value.get("velocity"), name="velocity_probe.velocity")
    reference = _tuple_finite(value.get("reference"), name="velocity_probe.reference")
    expected = _tuple_finite(value.get("expected"), name="velocity_probe.expected")
    if (
        len(sample) != len(velocity)
        or len(reference) != len(expected)
        or len(sample) != len(expected)
    ):
        raise ValueError("velocity_probe vector dimensions do not match")
    if value.get("device") != "cpu" or value.get("dtype") != "float32":
        raise ValueError("velocity_probe execution metadata is invalid")
    if value.get("status") != "PASS" or value.get("tolerance") != repr(VELOCITY_TOLERANCE):
        raise ValueError("velocity_probe status or tolerance is invalid")
    timestep = _finite_scalar(value.get("timestep"), name="velocity_probe.timestep")
    sigma = _finite_scalar(value.get("sigma"), name="velocity_probe.sigma")
    sigma_next = _finite_scalar(value.get("sigma_next"), name="velocity_probe.sigma_next")
    if sigma <= 0.0 or sigma_next >= sigma:
        raise ValueError("velocity_probe sigma transition is invalid")
    recomputed_expected = clean_room_dataward_step(
        sample=sample,
        velocity=velocity,
        timestep=timestep,
        sigma=sigma,
        sigma_next=sigma_next,
    )
    if expected != recomputed_expected:
        raise ValueError("velocity_probe expected data-ward step does not match source formula")
    maximum, mean = _error_statistics(reference, expected)
    if _error_string(value.get("max_abs_error"), name="velocity_probe") != maximum:
        raise ValueError("velocity_probe max error does not match vectors")
    if _error_string(value.get("mean_abs_error"), name="velocity_probe") != mean:
        raise ValueError("velocity_probe mean error does not match vectors")
    if require_reference and maximum > VELOCITY_TOLERANCE:
        raise ValueError("velocity_probe error exceeds tolerance")


def build_parity_report(
    diffusers_cases: Mapping[int, Mapping[str, Sequence[float]]],
    *,
    velocity_reference: Mapping[str, object],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Build one complete report from vectors produced by the pinned Diffusers runtime."""

    if tuple(sorted(diffusers_cases)) != REQUIRED_GRID_POINTS:
        raise ValueError("Diffusers case set must be exactly 4, 8, 12, 16, and 20 grid points")
    cases: list[dict[str, object]] = []
    for points in REQUIRED_GRID_POINTS:
        raw_case = diffusers_cases[points]
        video_reference = tuple(float(value) for value in raw_case["video"])
        audio_reference = tuple(float(value) for value in raw_case["audio"])
        video_sigmas = tuple(
            build_minimax_h3_schedule(
                variant=MINIMAX_H3_BASE_FL2VA_PROFILE.variant,
                grid_points=points,
                precision="float32",
            ).sigmas
        )
        audio_sigmas = tuple(
            map_minimax_h3_audio_coordinate(value, precision="float32").audio_sigma
            for value in video_sigmas
        )
        cases.append(
            {
                "audio": _comparison(
                    sigmax=audio_sigmas,
                    reference=audio_reference,
                    tolerance=AUDIO_TOLERANCE,
                ),
                "grid_points": points,
                "video": _comparison(
                    sigmax=video_sigmas,
                    reference=video_reference,
                    tolerance=VIDEO_TOLERANCE,
                ),
            }
        )

    probe = dict(velocity_reference)
    sample = _tuple_finite(probe.get("sample"), name="velocity_reference.sample")
    velocity = _tuple_finite(probe.get("velocity"), name="velocity_reference.velocity")
    reference = _tuple_finite(probe.get("reference"), name="velocity_reference.reference")
    expected = clean_room_dataward_step(
        sample=sample,
        velocity=velocity,
        timestep=_finite_scalar(probe.get("timestep"), name="velocity_reference.timestep"),
        sigma=_finite_scalar(probe.get("sigma"), name="velocity_reference.sigma"),
        sigma_next=_finite_scalar(probe.get("sigma_next"), name="velocity_reference.sigma_next"),
    )
    maximum, mean = _error_statistics(reference, expected)
    velocity_probe = {
        "device": "cpu",
        "dtype": "float32",
        "expected": list(expected),
        "max_abs_error": repr(maximum),
        "mean_abs_error": repr(mean),
        "reference": list(reference),
        "sample": list(sample),
        "sigma": _finite_scalar(probe.get("sigma"), name="velocity_reference.sigma"),
        "sigma_next": _finite_scalar(probe.get("sigma_next"), name="velocity_reference.sigma_next"),
        "status": "PASS" if maximum <= VELOCITY_TOLERANCE else "FAIL",
        "timestep": _finite_scalar(probe.get("timestep"), name="velocity_reference.timestep"),
        "tolerance": repr(VELOCITY_TOLERANCE),
        "velocity": list(velocity),
    }
    metadata = _fixed_metadata(environment)
    report: dict[str, Any] = {
        **metadata,
        "cases": cases,
        "status": "PASS",
        "velocity_probe": velocity_probe,
    }
    return validate_parity_report(report)


def validate_parity_report(report: object) -> dict[str, Any]:
    """Fail closed unless vectors and source/runtime metadata satisfy the H3 contract."""

    root = _mapping(report, name="report")
    _exact_fields(root, _ROOT_FIELDS, name="report")
    if root.get("schema") != REPORT_SCHEMA or root.get("status") != "PASS":
        raise ValueError("report schema or status is invalid")
    configuration = root.get("configuration")
    if configuration != {
        "audio_shift": MINIMAX_H3_AUDIO_SHIFT,
        "base_grid": "endpoint_inclusive",
        "terminal": "zero_in_grid",
        "velocity_direction": "data_ward",
        "video_shift": MINIMAX_H3_VIDEO_SHIFT,
    }:
        raise ValueError("configuration does not match H3 contract")
    if root.get("profile") != {
        "family": "minimax_h3",
        "variants": ["base_fl2va", "base_ref2va"],
    }:
        raise ValueError("profile metadata does not match H3 contract")
    if root.get("sources") != _fixed_metadata({}).get("sources"):
        raise ValueError("source metadata does not match the pinned Diffusers revision")
    environment = _mapping(root.get("environment"), name="environment")
    if set(environment) != {
        "device",
        "diffusers",
        "diffusers_revision",
        "numpy",
        "python",
        "torch",
    }:
        raise ValueError("environment fields are incomplete or unknown")
    if environment.get("device") != "cpu":
        raise ValueError("only CPU parity reports are accepted")
    if environment.get("diffusers_revision") != MINIMAX_H3_DIFFUSERS_REVISION:
        raise ValueError("Diffusers revision is not pinned")
    if environment.get("numpy") != NUMPY_VERSION or environment.get("torch") != TORCH_VERSION:
        raise ValueError("parity dependency versions are not pinned")
    for field in ("diffusers", "python"):
        if not isinstance(environment.get(field), str) or not environment[field]:
            raise ValueError(f"environment.{field} is invalid")
    if root.get("tolerances") != _fixed_metadata({})["tolerances"]:
        raise ValueError("tolerances do not match the H3 contract")

    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(REQUIRED_GRID_POINTS):
        raise ValueError("report case set is incomplete")
    observed_points: list[int] = []
    for raw_case in raw_cases:
        case = _mapping(raw_case, name="case")
        _exact_fields(case, _CASE_FIELDS, name="case")
        points = case.get("grid_points")
        if not isinstance(points, int) or points not in REQUIRED_GRID_POINTS:
            raise ValueError("case grid_points are invalid")
        observed_points.append(points)
        _validate_comparison(
            case.get("video"), points=points, name=f"video[{points}]", tolerance=VIDEO_TOLERANCE
        )
        _validate_comparison(
            case.get("audio"), points=points, name=f"audio[{points}]", tolerance=AUDIO_TOLERANCE
        )
    if tuple(observed_points) != REQUIRED_GRID_POINTS:
        raise ValueError("case order or set is invalid")
    probe = _mapping(root.get("velocity_probe"), name="velocity_probe")
    _probe(probe)
    return cast(dict[str, Any], dict(root))


def canonical_json(value: Mapping[str, object]) -> str:
    """Encode a report without platform-dependent whitespace or key order."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "AUDIO_TOLERANCE",
    "NUMPY_VERSION",
    "REPORT_SCHEMA",
    "REQUIRED_GRID_POINTS",
    "TORCH_VERSION",
    "VELOCITY_TOLERANCE",
    "VIDEO_TOLERANCE",
    "build_parity_report",
    "canonical_json",
    "validate_parity_report",
]
