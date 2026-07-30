"""Strict evidence contract for pinned native ComfyUI Euler execution."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final, cast

from comfyui_sigmax.profiles import build_krea2_turbo_schedule

REPORT_SCHEMA: Final = "sigmax.krea2-native-euler-parity/1"
COMFYUI_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
COMFYUI_URL: Final = "https://github.com/Comfy-Org/ComfyUI"
SOURCE_BLOBS: Final = {
    "comfy/k_diffusion/sampling.py": (
        "11db46d94c9b7aef4bcb67e62e6e3eab94e0e36f"  # pragma: allowlist secret
    ),
    "comfy/model_sampling.py": (
        "5af336e76fd480a50425dd924f2ac9752083c09f"  # pragma: allowlist secret
    ),
}
DEPENDENCY_VERSIONS: Final = {
    "numpy": "2.5.1",
    "torch": "2.13.0",
}
CONTROL_INITIAL_STATE: Final = (0.75, -0.5, 1.25, -1.0)
CONTROL_BIASES: Final = (0.0625, -0.125, 0.1875, -0.25)
TOLERANCE: Final = 2e-6

_ROOT_FIELDS: Final = frozenset(
    {"case", "environment", "profile", "schema", "semantics", "source", "status"}
)
_CASE_FIELDS: Final = frozenset(
    {
        "counts",
        "deterministic_rerun",
        "initial_state",
        "max_abs_error",
        "mean_abs_error",
        "native_final",
        "native_steps",
        "oracle_states",
        "rerun_final",
        "sigmas",
        "status",
        "steps",
        "tolerance",
        "trace_fingerprint",
    }
)
_RAW_CASE_FIELDS: Final = frozenset(
    {
        "counts",
        "deterministic_rerun",
        "initial_state",
        "native_final",
        "native_steps",
        "rerun_final",
        "sigmas",
        "steps",
    }
)
_STEP_FIELDS: Final = frozenset(
    {
        "denoised",
        "index",
        "input_state",
        "output_state",
        "sigma",
        "sigma_next",
        "velocity",
    }
)


def _git_chunks(value: str) -> list[str]:
    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _fixed_metadata() -> dict[str, object]:
    return {
        "environment": {
            "device": "cpu",
            "dtype": "float32",
            "numpy": DEPENDENCY_VERSIONS["numpy"],
            "python": "3.13",
            "torch": DEPENDENCY_VERSIONS["torch"],
        },
        "profile": {"id": "krea2.turbo.official", "version": "1"},
        "schema": REPORT_SCHEMA,
        "semantics": {
            "equation": "x_next=x+(sigma_next-sigma)*flow_velocity",
            "execution": "deterministic",
            "model_output_conversion": "denoised=x-flow_velocity*sigma",
            "noise_ownership": "none",
            "prediction_type": "flow_velocity",
            "sampler": "comfy.euler",
            "sampler_state": [],
            "schedule_ownership": "external_sigmas",
            "terminal": "zero_target_without_model_evaluation",
        },
        "source": {
            "blobs": {path: _git_chunks(blob) for path, blob in SOURCE_BLOBS.items()},
            "evidence": "framework_reference",
            "license": "GPL-3.0-only",
            "locators": {
                "euler": "comfy/k_diffusion/sampling.py:190-214",
                "flow_conversion": "comfy/model_sampling.py:86-96",
            },
            "revision_chunks": _git_chunks(COMFYUI_REVISION),
            "url": COMFYUI_URL,
        },
    }


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _exact_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields are incomplete or unknown")


def _finite_vector(
    value: object,
    *,
    name: str,
    length: int = 4,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} vector length is invalid")
    result: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{name} vector must contain finite numbers")
        result.append(float(item))
    return tuple(result)


def _close_vectors(
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    tolerance: float = TOLERANCE,
) -> bool:
    return len(actual) == len(expected) and all(
        abs(float(left) - float(right)) <= tolerance
        for left, right in zip(actual, expected, strict=True)
    )


def controlled_velocity(
    state: Sequence[float],
    sigma: float,
    index: int,
) -> tuple[float, ...]:
    """Return the nontrivial deterministic flow-velocity fixture."""

    if len(state) != len(CONTROL_BIASES):
        raise ValueError("controlled state width is invalid")
    if not math.isfinite(sigma) or not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("controlled velocity inputs are invalid")
    step_bias = (index + 1) * 0.03125
    return tuple(
        float(value) * 0.125 + sigma * 0.25 + bias + step_bias
        for value, bias in zip(state, CONTROL_BIASES, strict=True)
    )


def independent_euler_states(
    sigmas: Sequence[float],
    *,
    initial_state: Sequence[float] = CONTROL_INITIAL_STATE,
) -> tuple[tuple[float, ...], ...]:
    """Calculate the official flow Euler equation without tensor/ComfyUI imports."""

    if len(sigmas) != 9 or len(initial_state) != len(CONTROL_INITIAL_STATE):
        raise ValueError("independent Euler fixture shape is invalid")
    state = tuple(float(value) for value in initial_state)
    states = [state]
    for index, (sigma, sigma_next) in enumerate(pairwise(sigmas)):
        if not math.isfinite(float(sigma)) or not math.isfinite(float(sigma_next)):
            raise ValueError("independent Euler sigmas must be finite")
        velocity = controlled_velocity(state, float(sigma), index)
        dt = float(sigma_next) - float(sigma)
        state = tuple(
            value + dt * derivative for value, derivative in zip(state, velocity, strict=True)
        )
        if not all(math.isfinite(value) for value in state):
            raise ValueError("independent Euler state became non-finite")
        states.append(state)
    return tuple(states)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _trace_fingerprint(steps: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(steps)).hexdigest()


def _validate_raw_case(raw_case: object) -> dict[str, Any]:
    case = _mapping(raw_case, name="native case")
    _exact_fields(case, _RAW_CASE_FIELDS, name="native case")
    if case.get("steps") != 8:
        raise ValueError("native case steps must be eight")
    sigmas = _finite_vector(case.get("sigmas"), name="sigma", length=9)
    expected_sigmas = tuple(float(value) for value in build_krea2_turbo_schedule(steps=8).sigmas)
    if not _close_vectors(sigmas, expected_sigmas, tolerance=8e-8):
        raise ValueError("native case sigma vector is not the official Turbo schedule")
    if sigmas[0] != 1.0 or sigmas[-1] != 0.0:
        raise ValueError("native case sigma terminal is invalid")
    if any(left <= right for left, right in pairwise(sigmas)):
        raise ValueError("native case sigmas must be strictly decreasing")
    initial = _finite_vector(case.get("initial_state"), name="initial state")
    if initial != CONTROL_INITIAL_STATE:
        raise ValueError("native case initial state is invalid")

    steps = case.get("native_steps")
    if not isinstance(steps, list) or len(steps) != 8:
        raise ValueError("native step trace must contain eight steps")
    previous_output: tuple[float, ...] = initial
    normalized_steps: list[dict[str, object]] = []
    for index, raw_step in enumerate(steps):
        step = _mapping(raw_step, name="native step")
        _exact_fields(step, _STEP_FIELDS, name="native step")
        if step.get("index") != index:
            raise ValueError("native step index is invalid")
        sigma = step.get("sigma")
        sigma_next = step.get("sigma_next")
        if (
            isinstance(sigma, bool)
            or not isinstance(sigma, (int, float))
            or isinstance(sigma_next, bool)
            or not isinstance(sigma_next, (int, float))
            or abs(float(sigma) - sigmas[index]) > 8e-8
            or abs(float(sigma_next) - sigmas[index + 1]) > 8e-8
        ):
            raise ValueError("native step sigma pair is invalid")
        input_state = _finite_vector(step.get("input_state"), name="input state")
        if not _close_vectors(input_state, previous_output):
            raise ValueError("native step input is not the previous output")
        velocity = _finite_vector(step.get("velocity"), name="velocity")
        expected_velocity = controlled_velocity(input_state, float(sigma), index)
        if not _close_vectors(velocity, expected_velocity):
            raise ValueError("native step velocity is invalid")
        denoised = _finite_vector(step.get("denoised"), name="denoised")
        expected_denoised = tuple(
            value - derivative * float(sigma)
            for value, derivative in zip(input_state, velocity, strict=True)
        )
        if not _close_vectors(denoised, expected_denoised):
            raise ValueError("native step denoised conversion is invalid")
        output = _finite_vector(step.get("output_state"), name="output state")
        dt = float(sigma_next) - float(sigma)
        expected_output = tuple(
            value + derivative * dt for value, derivative in zip(input_state, velocity, strict=True)
        )
        if not _close_vectors(output, expected_output):
            raise ValueError("native step output is invalid")
        previous_output = output
        normalized_steps.append(dict(step))

    native_final = _finite_vector(case.get("native_final"), name="native final")
    rerun_final = _finite_vector(case.get("rerun_final"), name="rerun final")
    if not _close_vectors(native_final, previous_output):
        raise ValueError("native final does not match the last output")
    if case.get("deterministic_rerun") is not True or native_final != rerun_final:
        raise ValueError("deterministic rerun evidence is invalid")
    counts = _mapping(case.get("counts"), name="count evidence")
    expected_counts = {
        "effective_model_evaluations": 8,
        "effective_transitions": 8,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }
    if dict(counts) != expected_counts:
        raise ValueError("count evidence is invalid")
    return {
        "counts": expected_counts,
        "deterministic_rerun": True,
        "initial_state": list(initial),
        "native_final": list(native_final),
        "native_steps": normalized_steps,
        "rerun_final": list(rerun_final),
        "sigmas": list(sigmas),
        "steps": 8,
    }


def build_native_euler_report(native_case: object) -> dict[str, Any]:
    """Build complete immutable evidence from an actual native execution trace."""

    raw = _validate_raw_case(native_case)
    oracle_states = independent_euler_states(
        cast(Sequence[float], raw["sigmas"]),
        initial_state=cast(Sequence[float], raw["initial_state"]),
    )
    errors: list[float] = []
    for step, oracle in zip(
        cast(list[dict[str, object]], raw["native_steps"]),
        oracle_states[1:],
        strict=True,
    ):
        output = cast(Sequence[float], step["output_state"])
        errors.extend(
            abs(float(actual) - expected) for actual, expected in zip(output, oracle, strict=True)
        )
    maximum = max(errors)
    mean = math.fsum(errors) / len(errors)
    if maximum > TOLERANCE:
        raise ValueError("native output exceeds the independent Euler tolerance")
    case = {
        **raw,
        "max_abs_error": repr(maximum),
        "mean_abs_error": repr(mean),
        "oracle_states": [list(state) for state in oracle_states],
        "status": "PASS",
        "tolerance": repr(TOLERANCE),
        "trace_fingerprint": _trace_fingerprint(raw["native_steps"]),
    }
    report: dict[str, Any] = {
        **_fixed_metadata(),
        "case": case,
        "status": "PASS",
    }
    return validate_native_euler_report(report)


def validate_native_euler_case(case: object) -> dict[str, Any]:
    """Validate a complete case and recompute every derived field."""

    complete = _mapping(case, name="case")
    _exact_fields(complete, _CASE_FIELDS, name="case")
    raw = {field: complete[field] for field in _RAW_CASE_FIELDS}
    rebuilt = cast(dict[str, Any], build_native_euler_report_unchecked(raw)["case"])
    if dict(complete) != rebuilt:
        for field in _CASE_FIELDS:
            if complete.get(field) != rebuilt.get(field):
                raise ValueError(f"case {field.replace('_', ' ')} evidence is invalid")
        raise ValueError("case evidence is invalid")
    return cast(dict[str, Any], case)


def build_native_euler_report_unchecked(native_case: object) -> dict[str, Any]:
    """Internal non-recursive builder used by the complete-case validator."""

    raw = _validate_raw_case(native_case)
    oracle_states = independent_euler_states(
        cast(Sequence[float], raw["sigmas"]),
        initial_state=cast(Sequence[float], raw["initial_state"]),
    )
    errors = [
        abs(float(actual) - expected)
        for step, oracle in zip(
            cast(list[dict[str, object]], raw["native_steps"]),
            oracle_states[1:],
            strict=True,
        )
        for actual, expected in zip(
            cast(Sequence[float], step["output_state"]),
            oracle,
            strict=True,
        )
    ]
    maximum = max(errors)
    if maximum > TOLERANCE:
        raise ValueError("native output exceeds the independent Euler tolerance")
    case = {
        **raw,
        "max_abs_error": repr(maximum),
        "mean_abs_error": repr(math.fsum(errors) / len(errors)),
        "oracle_states": [list(state) for state in oracle_states],
        "status": "PASS",
        "tolerance": repr(TOLERANCE),
        "trace_fingerprint": _trace_fingerprint(raw["native_steps"]),
    }
    return {**_fixed_metadata(), "case": case, "status": "PASS"}


def validate_native_euler_report(report: object) -> dict[str, Any]:
    """Fail closed unless a report proves the complete M5-01 execution contract."""

    root = _mapping(report, name="report")
    _exact_fields(root, _ROOT_FIELDS, name="report")
    if root.get("status") != "PASS":
        raise ValueError("report status must be PASS")
    expected = _fixed_metadata()
    for field in ("environment", "profile", "schema", "semantics", "source"):
        if root.get(field) != expected[field]:
            raise ValueError(f"{field} metadata does not match the pinned contract")
    validate_native_euler_case(root.get("case"))
    return cast(dict[str, Any], report)


def canonical_json(report: object) -> str:
    """Serialize validated evidence deterministically with a terminal newline."""

    return _canonical_bytes(validate_native_euler_report(report)).decode("utf-8") + "\n"
