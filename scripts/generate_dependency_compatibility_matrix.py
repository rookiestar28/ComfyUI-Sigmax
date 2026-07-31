"""Generate the canonical M7-04 dependency compatibility matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

from comfyui_sigmax.benchmark_matrix import load_numerical_benchmark_matrix
from comfyui_sigmax.core import ScheduleContractError

_CONTRACT_RUNNER = importlib.import_module(
    "scripts.run_dependency_compatibility_lane"
    if __package__
    else "run_dependency_compatibility_lane"
)
_LATEST_HOST_SANITIZER = importlib.import_module(
    "scripts.sanitize_latest_host_compatibility_evidence"
    if __package__
    else "sanitize_latest_host_compatibility_evidence"
)
CONTRACT_SOURCE_PATHS = cast(tuple[str, ...], _CONTRACT_RUNNER.SOURCE_PATHS)
build_invariant_contract = cast(
    Callable[[], dict[str, object]],
    _CONTRACT_RUNNER.build_invariant_contract,
)
validate_latest_host_evidence = cast(
    Callable[[dict[str, object]], None],
    _LATEST_HOST_SANITIZER.validate_latest_host_evidence,
)

ROOT: Final = Path(__file__).resolve().parents[1]
TARGET: Final = ROOT / "comfyui_sigmax" / "compatibility" / "matrix_v1.json"
EVIDENCE: Final = "tests/compatibility/fixtures/dependency_compatibility_evidence_v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _source(path: str) -> dict[str, str]:
    return {"path": path, "sha256": _identity((ROOT / path).read_bytes())}


def _lane(value: dict[str, Any]) -> dict[str, object]:
    status = value["status"]
    source = value["evidence_source"]
    if status == "passed":
        evidence_path = ROOT / source
        if not evidence_path.is_file():
            raise RuntimeError(f"passed compatibility evidence is missing: {source}")
        if value["evidence_kind"] == "executed_invariant_contract":
            executed = _read_json(source)
            expected = {
                "first_attempt": "passed",
                "lane_id": value["id"],
                "mandatory_dependencies": 0,
                "platform": value["platform"],
                "python": value["components"]["python"],
                "repeat": "passed",
                "schema": "sigmax.compatibility-lane-evidence/1",
                "status": "passed",
            }
            for key, expected_value in expected.items():
                if executed.get(key) != expected_value:
                    raise RuntimeError(f"executed compatibility evidence {key} does not match lane")
            for key in ("contract_fingerprint", "test_selection_fingerprint"):
                fingerprint = executed.get(key)
                if (
                    not isinstance(fingerprint, str)
                    or not fingerprint.startswith("sha256:")
                    or len(fingerprint) != 71
                ):
                    raise RuntimeError(f"executed compatibility evidence {key} is invalid")
            expected_contract = _identity(_canonical(build_invariant_contract()))
            if executed["contract_fingerprint"] != expected_contract:
                raise RuntimeError("executed compatibility evidence contract fingerprint drifted")
        elif value["evidence_kind"] == "executed_latest_host":
            executed = _read_json(source)
            try:
                validate_latest_host_evidence(cast(dict[str, object], executed))
            except ScheduleContractError as exc:
                raise RuntimeError("latest-host compatibility evidence is invalid") from exc
            if (
                executed.get("lane_id") != value["id"]
                or executed.get("platform") != value["platform"]
                or executed.get("host")
                != {
                    "id": "comfyui",
                    "revision": value["host_revision"],
                    "version": value["host_version"],
                }
                or executed.get("runtime")
                != {
                    "device": "cpu",
                    "model_weights": "not_loaded",
                    "python": value["components"]["python"],
                    "torch": value["components"]["torch"],
                }
            ):
                raise RuntimeError("latest-host compatibility evidence does not match lane")
        result_fingerprint: str | None = _identity(evidence_path.read_bytes())
        first = "passed"
        repeat = "passed"
    else:
        result_fingerprint = None
        first = "not_evaluated"
        repeat = "not_evaluated"
    return {
        "blocking": value["blocking"],
        "components": value["components"],
        "evidence": {
            "first_attempt": first,
            "kind": value["evidence_kind"],
            "repeat": repeat,
            "result_fingerprint": result_fingerprint,
            "source": source,
        },
        "id": value["id"],
        "platform": value["platform"],
        "reason": value["reason"],
        "role": value["role"],
        "status": status,
    }


def build_envelope() -> dict[str, object]:
    evidence = _read_json(EVIDENCE)
    if evidence.get("schema") != "sigmax.dependency-compatibility-evidence/1":
        raise RuntimeError("dependency compatibility evidence schema drifted")
    lanes = [_lane(cast(dict[str, Any], item)) for item in evidence["lanes"]]
    lanes.sort(key=lambda lane: cast(str, lane["id"]))
    source_paths = sorted(
        {
            EVIDENCE,
            *CONTRACT_SOURCE_PATHS,
            *(
                cast(str, lane["evidence_source"])
                for lane in evidence["lanes"]
                if lane["status"] == "passed"
            ),
        }
    )
    benchmark = load_numerical_benchmark_matrix()
    invariant = build_invariant_contract()
    windows = _read_json("tests/compatibility/fixtures/windows_py313_v1.json")
    matrix = {
        "contract": {
            "expected": {
                "benchmark_matrix_fingerprint": benchmark.matrix_fingerprint,
                "lane_contract_fingerprint": _identity(_canonical(invariant)),
                "mandatory_dependencies": 0,
                "test_selection_fingerprint": windows["test_selection_fingerprint"],
            },
            "id": "sigmax.tier1-compatibility-invariants/1",
            "schema": "sigmax.compatibility-invariant-contract/1",
            "source_fingerprints": [_source(path) for path in sorted(CONTRACT_SOURCE_PATHS)],
        },
        "lanes": lanes,
        "policy": {
            "api_stability": {"v0_0_2": "experimental"},
            "known_good_is_blocking": True,
            "latest_can_expand_support": False,
            "official_container_requires_resolvable_digest": True,
            "official_container_unavailable_is_blocking": False,
            "reference_diffusers": "0.39.0",
            "supported_comfyui": "0.29.0",
            "supported_python": ["3.10", "3.13"],
            "third_party_container_substitution": False,
            "unavailable_is_pass": False,
        },
        "schema": "sigmax.dependency-compatibility-matrix/1",
        "sources": [_source(path) for path in source_paths],
    }
    return {
        "matrix": matrix,
        "matrix_fingerprint": _identity(_canonical(matrix)),
        "schema": "sigmax.dependency-compatibility-matrix-envelope/1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    payload = _canonical(build_envelope()) + b"\n"
    if arguments.check:
        if not TARGET.is_file() or TARGET.read_bytes() != payload:
            raise RuntimeError("dependency compatibility matrix drifted")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(payload)
    print("DEPENDENCY_COMPATIBILITY_MATRIX=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
