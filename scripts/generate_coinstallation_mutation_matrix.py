"""Generate the canonical M7-08 co-installation and host-mutation matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

_RUNNER = importlib.import_module(
    "scripts.run_synthetic_coinstallation_matrix"
    if __package__
    else "run_synthetic_coinstallation_matrix"
)
build_evidence = cast(Callable[[], dict[str, object]], _RUNNER.build_evidence)

ROOT: Final = Path(__file__).resolve().parents[1]
TARGET: Final = ROOT / "comfyui_sigmax" / "coinstallation" / "matrix_v1.json"
EVIDENCE: Final = "tests/coinstallation/fixtures/synthetic_mutation_evidence_v1.json"
SOURCES: Final = (
    "comfyui_sigmax/adapters/registration.py",
    "comfyui_sigmax/compatibility/matrix_v1.json",
    "comfyui_sigmax/host_mutation.py",
    "tests/coinstallation/fixtures/synthetic_mutation_evidence_v1.json",
    "tests/coinstallation/fixtures/synthetic_mutations_v1.json",
)


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


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _source(path: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
    }


def build_envelope() -> dict[str, object]:
    evidence = _read_json(EVIDENCE)
    executed = build_evidence()
    if evidence != executed:
        raise RuntimeError("synthetic mutation evidence drifted from current execution")
    if (
        evidence.get("schema") != "sigmax.synthetic-host-mutation-evidence/1"
        or evidence.get("status") != "PASS"
    ):
        raise RuntimeError("synthetic mutation evidence schema or status is invalid")
    rows: list[dict[str, object]] = []
    for value in cast(list[dict[str, Any]], evidence["rows"]):
        rows.append(
            {
                "evidence_source": EVIDENCE,
                "expected_findings": value["expected_findings"],
                "expected_verdict": value["expected_verdict"],
                "first_attempt": value["first_attempt"],
                "first_report_fingerprint": value["first_report_fingerprint"],
                "id": value["id"],
                "observed_findings": value["observed_findings"],
                "observed_verdict": value["observed_verdict"],
                "operation": value["operation"],
                "pack_id": value["pack_id"],
                "repeat": value["repeat"],
                "repeat_report_fingerprint": value["repeat_report_fingerprint"],
                "result_fingerprint": _identity(value),
                "status": value["status"],
            }
        )
    matrix = {
        "context": evidence["context"],
        "policy": {
            "external_reference_code_executed": False,
            "protected_existing_identities": True,
            "third_party_claims": False,
        },
        "rows": rows,
        "schema": "sigmax.co-installation-mutation-matrix/1",
        "sources": [_source(path) for path in sorted(SOURCES)],
    }
    return {
        "matrix": matrix,
        "matrix_fingerprint": _identity(matrix),
        "schema": "sigmax.co-installation-mutation-matrix-envelope/1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    payload = _canonical(build_envelope()) + b"\n"
    if arguments.check:
        if not TARGET.is_file() or TARGET.read_bytes() != payload:
            raise RuntimeError("co-installation mutation matrix drifted")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
    print("COINSTALLATION_MUTATION_MATRIX=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
