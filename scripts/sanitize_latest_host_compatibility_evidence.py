"""Sanitize approved latest-host H1/H2/H3 evidence for the compatibility matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Final, cast

from comfyui_sigmax.core import ScheduleContractError

SCHEMA: Final = "sigmax.latest-host-compatibility-evidence/1"
HOST_SCHEMA: Final = "sigmax.comfyui-host-e2e-evidence/3"
TRANSITION_SCHEMA: Final = "sigmax.host-attempt-transition/1"
_REVISION: Final = re.compile(r"[0-9a-f]{40}")
_FINGERPRINT: Final = re.compile(r"sha256:[0-9a-f]{64}")
_EXPECTED_LANES: Final = [
    "H1",
    "H2_TURBO_M2_05",
    "H2_RAW_M3_06",
    "H2_ALGEBRA_M4_09",
    "H2_CHECKPOINT_EVIDENCE_M6_08",
    "H3_EULER_M5_01",
]
_EXPECTED_TRANSITIONS: Final = [
    "h1",
    "h2_checkpoint_evidence",
    "h2_raw.krea2-raw-diffusers-portrait-761x1353",
    "h2_raw.krea2-raw-official-landscape-1353x761",
    "h2_raw.krea2-raw-official-square-1024",
    "h2_raw.raw-auto-variant",
    "h2_raw.raw-invalid-steps",
    "h2_schedule_algebra",
    "h2_schedule_algebra.noop_resample",
    "h2_turbo",
    "h3_native_euler",
    "h3_native_euler.partial_denoise",
]
_EXPECTED_NODE_IDS: Final = [
    "Sigmax.AdvancedFlowMatchScheduler",
    "Sigmax.CheckpointEvidenceInspector",
    "Sigmax.Krea2SigmaScheduler",
    "Sigmax.ModelAwareSigmaScheduler",
    "Sigmax.ProfileInspector",
    "Sigmax.RawWorkflowOutput",
    "Sigmax.ScheduleComparison",
    "Sigmax.ScheduleConcatenate",
    "Sigmax.ScheduleInspector",
    "Sigmax.ScheduleResample",
    "Sigmax.ScheduleSlice",
    "Sigmax.TurboWorkflowOutput",
]
_EVIDENCE_FIELDS: Final = {
    "first_attempt",
    "host",
    "invariants",
    "lane_id",
    "platform",
    "repeat",
    "result_fingerprint",
    "runtime",
    "schema",
    "status",
}


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


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _validate_transition(value: object, *, transition_id: str) -> dict[str, object]:
    transition = _object(value, label=f"{transition_id} transition")
    if (
        transition.get("schema") != TRANSITION_SCHEMA
        or transition.get("accepted") is not True
        or transition.get("transition") != "pass_to_pass"
    ):
        raise ScheduleContractError(f"{transition_id} transition is not accepted")
    first = _object(transition.get("first"), label=f"{transition_id} first attempt")
    repeat = _object(transition.get("repeat"), label=f"{transition_id} repeat")
    if (
        first.get("ordinal") != 1
        or repeat.get("ordinal") != 2
        or first.get("verdict") != "pass"
        or repeat.get("verdict") != "pass"
    ):
        raise ScheduleContractError(f"{transition_id} transition attempts are not PASS")
    first_fingerprint = first.get("result_fingerprint")
    repeat_fingerprint = repeat.get("result_fingerprint")
    if (
        not isinstance(first_fingerprint, str)
        or not _FINGERPRINT.fullmatch(first_fingerprint)
        or repeat_fingerprint != first_fingerprint
    ):
        raise ScheduleContractError(f"{transition_id} first/repeat fingerprints differ")
    return cast(dict[str, object], json.loads(_canonical(transition)))


def build_latest_host_evidence(
    raw: dict[str, object],
    *,
    lane_id: str,
    expected_revision: str,
    expected_version: str,
    python_version: str,
    torch_version: str,
) -> dict[str, object]:
    """Validate raw host evidence and return a stable, path-free public projection."""

    if raw.get("schema") != HOST_SCHEMA or raw.get("cleanup") != "removed":
        raise ScheduleContractError("latest-host evidence schema or cleanup is invalid")
    if raw.get("platform") != "windows" or raw.get("lanes") != _EXPECTED_LANES:
        raise ScheduleContractError("latest-host platform or executed lanes are invalid")
    if not _REVISION.fullmatch(expected_revision):
        raise ScheduleContractError("latest-host expected revision is invalid")
    host = _object(raw.get("host"), label="latest-host identity")
    expected_host = {
        "id": "comfyui",
        "revision": expected_revision,
        "version": expected_version,
    }
    if host != expected_host:
        raise ScheduleContractError("latest-host identity does not match the expected revision")
    import_probe = _object(raw.get("import_probe"), label="latest-host import probe")
    expected_probe = {
        "diffusers_loaded": False,
        "node_ids": _EXPECTED_NODE_IDS,
        "scheduler_registry_unchanged": True,
        "torch_call_unchanged": True,
    }
    if import_probe != expected_probe:
        raise ScheduleContractError("latest-host import-safety probe failed")
    transitions = _object(raw.get("attempt_transitions"), label="latest-host transitions")
    if sorted(transitions) != _EXPECTED_TRANSITIONS:
        raise ScheduleContractError("latest-host transition inventory is incomplete")
    stable_transitions = {
        transition_id: _validate_transition(
            transitions[transition_id],
            transition_id=transition_id,
        )
        for transition_id in _EXPECTED_TRANSITIONS
    }
    stable_result = {
        "host": expected_host,
        "import_probe": expected_probe,
        "lanes": _EXPECTED_LANES,
        "transitions": stable_transitions,
    }
    return {
        "first_attempt": "passed",
        "host": expected_host,
        "invariants": stable_result,
        "lane_id": lane_id,
        "platform": "windows",
        "repeat": "passed",
        "result_fingerprint": _identity(stable_result),
        "runtime": {
            "device": "cpu",
            "model_weights": "not_loaded",
            "python": python_version,
            "torch": torch_version,
        },
        "schema": SCHEMA,
        "status": "passed",
    }


def validate_latest_host_evidence(evidence: dict[str, object]) -> None:
    """Reject drift or false PASS in one sanitized latest-host evidence record."""

    if set(evidence) != _EVIDENCE_FIELDS or evidence.get("schema") != SCHEMA:
        raise ScheduleContractError("sanitized latest-host evidence schema is invalid")
    if (
        evidence.get("status") != "passed"
        or evidence.get("first_attempt") != "passed"
        or evidence.get("repeat") != "passed"
        or evidence.get("platform") != "windows"
    ):
        raise ScheduleContractError("sanitized latest-host PASS state is invalid")
    host = _object(evidence.get("host"), label="sanitized latest-host identity")
    if (
        host.get("id") != "comfyui"
        or not isinstance(host.get("version"), str)
        or not isinstance(host.get("revision"), str)
        or not _REVISION.fullmatch(cast(str, host["revision"]))
    ):
        raise ScheduleContractError("sanitized latest-host identity is invalid")
    runtime = _object(evidence.get("runtime"), label="sanitized latest-host runtime")
    if (
        set(runtime) != {"device", "model_weights", "python", "torch"}
        or runtime.get("device") != "cpu"
        or runtime.get("model_weights") != "not_loaded"
        or not isinstance(runtime.get("python"), str)
        or not isinstance(runtime.get("torch"), str)
    ):
        raise ScheduleContractError("sanitized latest-host runtime is invalid")
    invariants = _object(evidence.get("invariants"), label="sanitized latest-host invariants")
    if evidence.get("result_fingerprint") != _identity(invariants):
        raise ScheduleContractError("sanitized latest-host result fingerprint drifted")
    if invariants.get("host") != host or invariants.get("lanes") != _EXPECTED_LANES:
        raise ScheduleContractError("sanitized latest-host invariant identity drifted")
    if invariants.get("import_probe") != {
        "diffusers_loaded": False,
        "node_ids": _EXPECTED_NODE_IDS,
        "scheduler_registry_unchanged": True,
        "torch_call_unchanged": True,
    }:
        raise ScheduleContractError("sanitized latest-host import-safety evidence drifted")
    transitions = _object(
        invariants.get("transitions"),
        label="sanitized latest-host transitions",
    )
    if sorted(transitions) != _EXPECTED_TRANSITIONS:
        raise ScheduleContractError("sanitized latest-host transition inventory drifted")
    for transition_id in _EXPECTED_TRANSITIONS:
        _validate_transition(transitions[transition_id], transition_id=transition_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--torch-version", required=True)
    arguments = parser.parse_args()
    raw = json.loads(arguments.input.read_text(encoding="utf-8"))
    evidence = build_latest_host_evidence(
        _object(raw, label="raw latest-host evidence"),
        lane_id=arguments.lane_id,
        expected_revision=arguments.expected_revision,
        expected_version=arguments.expected_version,
        python_version=arguments.python_version,
        torch_version=arguments.torch_version,
    )
    validate_latest_host_evidence(evidence)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(_canonical(evidence) + b"\n")
    print("LATEST_HOST_COMPATIBILITY_EVIDENCE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
