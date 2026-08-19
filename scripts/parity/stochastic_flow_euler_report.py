"""Canonical report contract for M5-04 Diffusers stochastic Flow Euler parity."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any, Final, cast

REPORT_SCHEMA: Final = "sigmax.stochastic-flow-euler-diffusers-parity/1"
DIFFUSERS_VERSION: Final = "0.39.0"
TORCH_VERSION: Final = "2.9.0"
DIFFUSERS_REVISION: Final = "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret
DIFFUSERS_SCHEDULER_BLOB: Final = (
    "7b207f7820797c53b093452ca2bc52938a8d84e7"  # pragma: allowlist secret
)
SIGMAS: Final = (1.0, 0.75, 0.25, 0.0)
_FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")


def _chunks(value: str) -> list[str]:
    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _metadata(environment: Mapping[str, str]) -> dict[str, object]:
    return {
        "configuration": {
            "algorithm": "diffusers.flow-match-euler.stochastic",
            "expression": "(1-next_sigma)*x0+next_sigma*noise",
            "noise_ownership": "caller",
            "terminal_noise_draw": True,
        },
        "environment": dict(environment),
        "schema": REPORT_SCHEMA,
        "source": {
            "blob_chunks": _chunks(DIFFUSERS_SCHEDULER_BLOB),
            "evidence": "framework_reference",
            "locator": ("src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:509-513"),
            "revision_chunks": _chunks(DIFFUSERS_REVISION),
            "tag": "v0.39.0",
            "url": "https://github.com/huggingface/diffusers",
        },
    }


def build_parity_report(
    case: Mapping[str, object],
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Bind one runtime-produced case to exact public source metadata."""

    report: dict[str, Any] = {
        **_metadata(environment),
        "case": dict(case),
        "status": "PASS",
    }
    return validate_parity_report(report)


def _require_fingerprint(value: object, name: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical fingerprint")
    return value


def validate_parity_report(value: object) -> dict[str, Any]:
    """Reject incomplete, drifted, or non-passing parity evidence."""

    if not isinstance(value, dict):
        raise ValueError("report must be an object")
    report = cast(dict[str, Any], value)
    if set(report) != {"case", "configuration", "environment", "schema", "source", "status"}:
        raise ValueError("report fields are incomplete or unknown")
    if report["schema"] != REPORT_SCHEMA or report["status"] != "PASS":
        raise ValueError("report schema or status is invalid")
    if report["environment"] != {
        "device": "cpu",
        "diffusers": DIFFUSERS_VERSION,
        "torch": TORCH_VERSION,
    }:
        raise ValueError("environment does not match the pinned parity runtime")
    fixed = _metadata(cast(Mapping[str, str], report["environment"]))
    if report["configuration"] != fixed["configuration"] or report["source"] != fixed["source"]:
        raise ValueError("configuration or source metadata drifted")

    case = report["case"]
    if not isinstance(case, dict) or set(case) != {
        "different_seed_diverges",
        "final_result_fingerprint",
        "global_rng_unchanged",
        "local_generator_state_fingerprint",
        "local_generator_state_matches",
        "model_evaluation_count",
        "noise_draw_count",
        "noise_fingerprints",
        "same_seed_repeat",
        "schedule_fingerprint",
        "sigmas",
        "steps",
        "terminal_noise_draw",
        "transition_count",
    }:
        raise ValueError("case fields are incomplete or unknown")
    if case["sigmas"] != list(SIGMAS):
        raise ValueError("case sigmas drifted")
    for name in ("transition_count", "model_evaluation_count", "noise_draw_count"):
        if case[name] != 3:
            raise ValueError(f"{name} must equal three")
    for name in (
        "different_seed_diverges",
        "global_rng_unchanged",
        "local_generator_state_matches",
        "same_seed_repeat",
        "terminal_noise_draw",
    ):
        if case[name] is not True:
            raise ValueError(f"{name} must be true")
    for name in (
        "final_result_fingerprint",
        "local_generator_state_fingerprint",
        "schedule_fingerprint",
    ):
        _require_fingerprint(case[name], name)
    noise_fingerprints = case["noise_fingerprints"]
    if not isinstance(noise_fingerprints, list) or len(noise_fingerprints) != 3:
        raise ValueError("noise fingerprints must contain three draws")
    for index, fingerprint in enumerate(noise_fingerprints):
        _require_fingerprint(fingerprint, f"noise_fingerprints[{index}]")

    steps = case["steps"]
    if not isinstance(steps, list) or len(steps) != 3:
        raise ValueError("steps must contain three transitions")
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) != {
            "max_abs_error",
            "mean_abs_error",
            "scheduler_index",
            "sigmax_state_fingerprint",
            "reference_state_fingerprint",
        }:
            raise ValueError("step fields are incomplete or unknown")
        if step["scheduler_index"] != index:
            raise ValueError("step indexes are not contiguous")
        for error_name in ("max_abs_error", "mean_abs_error"):
            error = step[error_name]
            if not isinstance(error, str) or not math.isfinite(float(error)) or float(error) != 0:
                raise ValueError("step error must be exact zero")
        sigmax_fingerprint = _require_fingerprint(
            step["sigmax_state_fingerprint"], "Sigmax state fingerprint"
        )
        reference_fingerprint = _require_fingerprint(
            step["reference_state_fingerprint"], "reference state fingerprint"
        )
        if sigmax_fingerprint != reference_fingerprint:
            raise ValueError("step fingerprints differ despite exact parity claim")
    return report


def canonical_json(report: Mapping[str, object]) -> str:
    """Serialize validated evidence deterministically with a final newline."""

    validated = validate_parity_report(dict(report))
    return (
        json.dumps(
            validated,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
