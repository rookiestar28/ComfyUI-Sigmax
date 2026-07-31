"""Generate the canonical M7-05 performance budget matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Final, cast

ROOT: Final = Path(__file__).resolve().parents[1]
TARGET: Final = ROOT / "comfyui_sigmax" / "performance" / "matrix_v1.json"
EVIDENCE: Final = (
    "tests/performance/fixtures/comfyui0290_startup_v1.json",
    "tests/performance/fixtures/windows_py313_v1.json",
    "tests/performance/fixtures/wsl_py310_v1.json",
)
SOURCES: Final = (
    "comfyui_sigmax/performance_budgets.py",
    "scripts/generate_performance_host_startup_evidence.py",
    "scripts/run_comfyui_e2e.py",
    "scripts/run_performance_budget_lane.py",
    *EVIDENCE,
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


def _read(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return cast(dict[str, Any], value)


def _source(path: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
    }


def build_envelope() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in EVIDENCE:
        evidence = _read(path)
        if evidence.get("status") != "PASS":
            raise RuntimeError(f"{path} is not accepted performance evidence")
        if evidence.get("schema") == "sigmax.performance-lane-evidence/1":
            lane_id = evidence["context"]["lane_id"]
            expected_evidence = _identity(
                {"context": evidence["context"], "rows": evidence["rows"]}
            )
            if evidence.get("evidence_fingerprint") != expected_evidence:
                raise RuntimeError(f"{path} evidence fingerprint drifted")
            source_evidence_fingerprint = expected_evidence
            for value in evidence["rows"]:
                rows.append(
                    {
                        "evaluation": value["evaluation"],
                        "evaluation_fingerprint": value["evaluation_fingerprint"],
                        "evidence_source": path,
                        "id": f"{lane_id}.{value['id']}",
                        "source_evidence_fingerprint": source_evidence_fingerprint,
                        "status": value["status"],
                    }
                )
        elif evidence.get("schema") == "sigmax.performance-host-startup-evidence/1":
            stored_fingerprint = evidence.pop("evidence_fingerprint", None)
            expected_evidence = _identity(evidence)
            if stored_fingerprint != expected_evidence:
                raise RuntimeError(f"{path} evidence fingerprint drifted")
            rows.append(
                {
                    "evaluation": evidence["evaluation"],
                    "evaluation_fingerprint": evidence["evaluation_fingerprint"],
                    "evidence_source": path,
                    "id": "windows.comfyui0290.host.comfyui0290.readiness",
                    "source_evidence_fingerprint": expected_evidence,
                    "status": "passed",
                }
            )
        else:
            raise RuntimeError(f"{path} schema is unsupported")
    rows.sort(key=lambda row: cast(str, row["id"]))
    matrix = {
        "exclusions": [
            {"id": "gpu", "reason": "gpu_not_approved", "status": "not_evaluated"},
            {
                "id": "latest_host",
                "reason": "non_blocking_observation_only",
                "status": "not_evaluated",
            },
            {
                "id": "model_weights",
                "reason": "model_weights_not_approved",
                "status": "not_evaluated",
            },
            {
                "id": "official_container",
                "reason": "immutable_digest_unavailable",
                "status": "not_evaluated",
            },
        ],
        "policy": {
            "observations_machine_specific": True,
            "thresholds_are_regression_limits": True,
            "wall_clock_portable_guarantee": False,
        },
        "results": rows,
        "schema": "sigmax.performance-budget-matrix/1",
        "sources": [_source(path) for path in sorted(SOURCES)],
    }
    return {
        "matrix": matrix,
        "matrix_fingerprint": _identity(matrix),
        "schema": "sigmax.performance-budget-matrix-envelope/1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    payload = _canonical(build_envelope()) + b"\n"
    if arguments.check:
        if not TARGET.is_file() or TARGET.read_bytes() != payload:
            raise RuntimeError("performance budget matrix drifted")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
    print("PERFORMANCE_BUDGET_MATRIX=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
