"""Sanitize two approved pinned-host runs into integer startup evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, cast

ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA: Final = "sigmax.performance-host-startup-evidence/1"
HOST_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
MAXIMUM_NS: Final = 30_000_000_000


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load(path: Path, *, attempt: str) -> tuple[dict[str, object], str]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)
    if not isinstance(value, dict):
        raise RuntimeError("host evidence must be an object")
    evidence = cast(dict[str, Any], value)
    if evidence.get("schema") != "sigmax.comfyui-host-e2e-evidence/3":
        raise RuntimeError("host evidence schema is unsupported")
    if evidence.get("host") != {
        "id": "comfyui",
        "revision": HOST_REVISION,
        "version": "0.29.0",
    }:
        raise RuntimeError("host evidence identity drifted")
    if evidence.get("platform") != "windows":
        raise RuntimeError("host startup evidence must be Windows")
    transitions = evidence.get("attempt_transitions")
    if (
        not isinstance(transitions, dict)
        or len(transitions) != 12
        or not all(
            isinstance(item, dict)
            and item.get("accepted") is True
            and item.get("transition") == "pass_to_pass"
            for item in transitions.values()
        )
    ):
        raise RuntimeError("host evidence transitions are incomplete")
    probe = evidence.get("import_probe")
    if not isinstance(probe, dict) or (
        probe.get("diffusers_loaded") is not False
        or probe.get("scheduler_registry_unchanged") is not True
        or probe.get("torch_call_unchanged") is not True
    ):
        raise RuntimeError("host import-safety evidence failed")
    shutdown = evidence.get("shutdown")
    if shutdown != {"interrupt_requested": True, "return_code": 1}:
        raise RuntimeError("host shutdown evidence is incomplete")
    seconds = evidence.get("readiness_seconds")
    if not isinstance(seconds, Decimal) or not seconds.is_finite() or seconds <= 0:
        raise RuntimeError("host readiness duration is invalid")
    nanoseconds = int(seconds * Decimal(1_000_000_000))
    if nanoseconds > MAXIMUM_NS:
        raise RuntimeError("host readiness duration exceeds budget")
    observation = {
        "attempt": attempt,
        "metric_id": "host.comfyui0290.readiness",
        "platform_lane": "windows.comfyui0290",
        "schema": "sigmax.performance-observation/1",
        "unit": "nanoseconds",
        "value": nanoseconds,
        "workload_fingerprint": _identity(
            {
                "host_revision": HOST_REVISION,
                "harness_sha256": "sha256:"
                + hashlib.sha256((ROOT / "scripts/run_comfyui_e2e.py").read_bytes()).hexdigest(),
                "lanes": evidence["lanes"],
            }
        ),
    }
    sigmax_revision = evidence.get("sigmax_revision")
    if not isinstance(sigmax_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", sigmax_revision):
        raise RuntimeError("Sigmax host revision is invalid")
    return observation, sigmax_revision


def build_evidence(first_path: Path, repeat_path: Path) -> dict[str, object]:
    first, first_revision = _load(first_path, attempt="first")
    repeat, repeat_revision = _load(repeat_path, attempt="repeat")
    if first["workload_fingerprint"] != repeat["workload_fingerprint"]:
        raise RuntimeError("host startup workloads differ")
    if first_revision != repeat_revision:
        raise RuntimeError("Sigmax host revisions differ")
    budget = {
        "maximum": MAXIMUM_NS,
        "metric_id": "host.comfyui0290.readiness",
        "minimum": 1,
        "unit": "nanoseconds",
        "workload_fingerprint": first["workload_fingerprint"],
    }
    evaluation = {
        "budget": budget,
        "observations": [first, repeat],
        "schema": "sigmax.performance-budget-evaluation/1",
        "verdict": "pass",
    }
    result = {
        "cleanup": {"attempts": 2, "completed": True},
        "evaluation": evaluation,
        "evaluation_fingerprint": _identity(evaluation),
        "host": {"id": "comfyui", "revision": HOST_REVISION, "version": "0.29.0"},
        "model_weights_present": False,
        "schema": SCHEMA,
        "sigmax_revision": first_revision,
        "status": "PASS",
    }
    result["evidence_fingerprint"] = _identity(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    payload = _canonical(build_evidence(arguments.first, arguments.repeat)) + b"\n"
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_bytes() != payload:
            raise RuntimeError("host startup evidence drifted")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(payload)
    print("PERFORMANCE_HOST_STARTUP=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
