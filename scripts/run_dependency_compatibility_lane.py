"""Execute one fixed local M7-04 compatibility invariant lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import tomli
from comfyui_sigmax.benchmark_matrix import load_numerical_benchmark_matrix

ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA: Final = "sigmax.compatibility-lane-evidence/1"
SOURCE_PATHS: Final = (
    "comfyui_sigmax/benchmarks/numerical_matrix_v1.json",
    "comfyui_sigmax/nodes/anima_sigma_scheduler.py",
    "comfyui_sigmax/nodes/wan_sigma_scheduler.py",
    "comfyui_sigmax/nodes/ltx_sigma_scheduler.py",
    "comfyui_sigmax/profiles/anima.py",
    "comfyui_sigmax/profiles/wan.py",
    "comfyui_sigmax/profiles/ltx.py",
    "comfyui_sigmax/workflows/fixtures.json",
    "tests/conformance/fixtures/capability_receipt_conformance_v1.json",
    "tests/fixtures/artifacts/execution_receipt_hashes_v1.json",
    "tests/fixtures/artifacts/golden_hashes_v1.json",
    "tests/golden/krea2_raw_v1.json",
    "tests/golden/sd3_v1.json",
    "tests/golden/aura_flow_v0_2.json",
    "tests/golden/anima_v1.json",
    "tests/golden/wan_v1.json",
    "tests/golden/wan_m4_18_v1.json",
    "tests/golden/ltx_v1.json",
    "tests/golden/hunyuan_image21_v1.json",
    "tests/golden/krea2_turbo_v1.json",
    "tests/parity/fixtures/krea2_native_euler_parity_v1.json",
    "tests/parity/fixtures/krea2_raw_parity_v1.json",
    "tests/parity/fixtures/krea2_turbo_comfy_native_parity_v1.json",
    "tests/parity/fixtures/krea2_turbo_parity_v1.json",
    "tests/conformance/test_capability_receipt_conformance.py",
    "tests/golden/test_krea2_raw_goldens.py",
    "tests/golden/test_sd3_goldens.py",
    "tests/golden/test_aura_flow_goldens.py",
    "tests/golden/test_anima_phase0_goldens.py",
    "tests/golden/test_wan_goldens.py",
    "tests/golden/test_wan_m4_18_golden.py",
    "tests/golden/test_ltx_goldens.py",
    "tests/golden/test_hunyuan_image21_goldens.py",
    "tests/golden/test_krea2_turbo_goldens.py",
    "tests/parity/test_krea2_comfy_native_parity.py",
    "tests/parity/test_krea2_native_euler_parity.py",
    "tests/parity/test_krea2_raw_parity.py",
    "tests/parity/test_sd3_parity.py",
    "tests/parity/test_aura_flow_parity.py",
    "tests/parity/test_anima_phase0_parity.py",
    "tests/parity/test_wan_phase0_parity.py",
    "tests/parity/test_wan_m4_18_comfy_optimized_parity.py",
    "tests/parity/test_ltx_phase0_parity.py",
    "tests/parity/test_hunyuan_image21_parity.py",
    "tests/parity/test_krea2_turbo_parity.py",
    "tests/test_artifact_serialization.py",
    "tests/test_benchmark_matrix.py",
    "tests/test_capabilities.py",
    "tests/test_compatibility_matrix.py",
    "tests/test_execution_receipts.py",
    "tests/test_package_contract.py",
    "tests/test_workflow_validation.py",
    "tests/test_hunyuan_image21_profile.py",
    "tests/test_hunyuan_image21_sigma_scheduler_node.py",
    "tests/test_anima_phase0_node_contract.py",
    "tests/test_anima_phase0_profile_contract.py",
    "tests/test_anima_phase0_registry_workflow_host_contract.py",
    "tests/test_anima_phase3_host_contract.py",
    "tests/test_wan_phase2_boundary.py",
    "tests/test_wan_phase3_host_contract.py",
    "tests/test_wan_m4_18_comfy_optimized_contract.py",
    "tests/test_ltx_phase0_node_contract.py",
    "tests/test_ltx_phase0_profile_contract.py",
    "tests/test_ltx_phase0_registry_workflow_host_contract.py",
    "tests/test_ltx_phase3_host_contract.py",
)
TEST_SELECTION: Final = (
    "tests/conformance",
    "tests/golden",
    "tests/parity",
    "tests/test_anima_phase0_node_contract.py",
    "tests/test_anima_phase0_profile_contract.py",
    "tests/test_anima_phase0_registry_workflow_host_contract.py",
    "tests/test_anima_phase3_host_contract.py",
    "tests/test_wan_phase2_boundary.py",
    "tests/test_wan_phase3_host_contract.py",
    "tests/test_artifact_serialization.py",
    "tests/test_benchmark_matrix.py",
    "tests/test_capabilities.py",
    "tests/test_execution_receipts.py",
    "tests/test_package_contract.py",
    "tests/test_workflow_validation.py",
)
LANES: Final = {
    "core-windows-py313": ("windows", (3, 13)),
    "core-wsl-py310": ("wsl", (3, 10)),
}


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


def _platform() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        release = Path("/proc/sys/kernel/osrelease")
        if release.is_file() and "microsoft" in release.read_text(encoding="utf-8").lower():
            return "wsl"
    return "unsupported"


def build_invariant_contract() -> dict[str, object]:
    sources = [
        {"path": path, "sha256": _identity((ROOT / path).read_bytes())}
        for path in sorted(SOURCE_PATHS)
    ]
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomli.load(stream)["project"]
    benchmark = load_numerical_benchmark_matrix()
    return {
        "benchmark_matrix_fingerprint": benchmark.matrix_fingerprint,
        "mandatory_dependencies": len(project["dependencies"]),
        "source_fingerprints": sources,
        "test_selection": list(TEST_SELECTION),
    }


def _run_suite() -> str:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and fixed local test selection
        [sys.executable, "-m", "pytest", "-q", *TEST_SELECTION],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return "passed" if completed.returncode == 0 else "failed"


def build_evidence(lane_id: str) -> dict[str, object]:
    if lane_id not in LANES:
        raise RuntimeError("unknown fixed compatibility lane")
    expected_platform, expected_python = LANES[lane_id]
    observed_platform = _platform()
    observed_python = sys.version_info[:2]
    if observed_platform != expected_platform:
        raise RuntimeError("compatibility lane platform does not match this interpreter")
    if observed_python != expected_python:
        raise RuntimeError("compatibility lane Python does not match this interpreter")
    contract = build_invariant_contract()
    if contract["mandatory_dependencies"] != 0:
        raise RuntimeError("mandatory runtime dependencies must remain zero")
    first = _run_suite()
    repeat = _run_suite() if first == "passed" else "not_evaluated"
    status = "passed" if first == repeat == "passed" else "failed"
    return {
        "contract_fingerprint": _identity(_canonical(contract)),
        "first_attempt": first,
        "lane_id": lane_id,
        "mandatory_dependencies": contract["mandatory_dependencies"],
        "platform": observed_platform,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "repeat": repeat,
        "schema": SCHEMA,
        "status": status,
        "test_selection_fingerprint": _identity(_canonical(list(TEST_SELECTION))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-id", required=True, choices=sorted(LANES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    evidence = build_evidence(arguments.lane_id)
    payload = _canonical(evidence) + b"\n"
    target = arguments.output.resolve()
    if arguments.check:
        if not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("compatibility lane evidence drifted")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    if evidence["status"] != "passed":
        print("COMPATIBILITY_LANE=FAIL")
        return 1
    print("COMPATIBILITY_LANE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
