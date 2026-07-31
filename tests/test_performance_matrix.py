from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.performance_matrix import (
    PerformanceMatrixError,
    load_performance_budget_matrix,
)
from scripts import generate_performance_budget_matrix as generator

ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _envelope() -> dict[str, Any]:
    payload = (
        importlib.resources.files("comfyui_sigmax.performance")
        .joinpath("matrix_v1.json")
        .read_bytes()
    )
    return cast(dict[str, Any], json.loads(payload))


def _rehashed(envelope: dict[str, Any]) -> bytes:
    envelope["matrix_fingerprint"] = (
        "sha256:" + hashlib.sha256(_canonical(envelope["matrix"])).hexdigest()
    )
    return _canonical(envelope) + b"\n"


def test_packaged_performance_matrix_is_complete() -> None:
    matrix = load_performance_budget_matrix()
    projection = matrix.projection()
    results = cast(list[dict[str, Any]], projection["results"])
    exclusions = cast(list[dict[str, Any]], projection["exclusions"])

    assert len(results) == 21
    assert all(row["status"] == "passed" for row in results)
    assert [item["id"] for item in exclusions] == [
        "gpu",
        "latest_host",
        "model_weights",
        "official_container",
    ]
    host = cast(
        dict[str, Any],
        matrix.require_result("windows.comfyui0290.host.comfyui0290.readiness"),
    )
    assert host["evaluation"]["verdict"] == "pass"


def test_exact_device_boundary_counts_are_published_for_both_platforms() -> None:
    matrix = load_performance_budget_matrix()
    for lane in ("performance-windows-py313", "performance-wsl-py310"):
        result = cast(
            dict[str, Any],
            matrix.require_result(f"{lane}.tensor.explicit_device_transfers"),
        )
        budget = result["evaluation"]["budget"]
        observations = result["evaluation"]["observations"]
        assert budget["minimum"] == budget["maximum"] == 0
        assert [item["value"] for item in observations] == [0, 0]


def test_generator_matches_packaged_resource() -> None:
    expected = generator._canonical(generator.build_envelope()) + b"\n"
    actual = (
        importlib.resources.files("comfyui_sigmax.performance")
        .joinpath("matrix_v1.json")
        .read_bytes()
    )
    assert actual == expected


def test_current_platform_performance_lane_remains_within_budget() -> None:
    lane = "performance-windows-py313" if os.name == "nt" else "performance-wsl-py310"
    fixture = (
        "tests/performance/fixtures/windows_py313_v1.json"
        if os.name == "nt"
        else "tests/performance/fixtures/wsl_py310_v1.json"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_performance_budget_lane.py",
            "--lane-id",
            lane,
            "--output",
            fixture,
            "--check",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PERFORMANCE_BUDGET_LANE=PASS" in completed.stdout


def test_rehashed_false_pass_is_rejected() -> None:
    envelope = _envelope()
    row = envelope["matrix"]["results"][0]
    row["evaluation"]["observations"][0]["value"] = row["evaluation"]["budget"]["maximum"] + 1
    row["evaluation_fingerprint"] = (
        "sha256:" + hashlib.sha256(_canonical(row["evaluation"])).hexdigest()
    )

    with pytest.raises(PerformanceMatrixError, match=r"within-budget|semantically invalid"):
        load_performance_budget_matrix(_rehashed(envelope))


@pytest.mark.parametrize("target", ["budget", "observation"])
def test_semantically_invalid_units_use_the_public_loader_error(target: str) -> None:
    envelope = _envelope()
    evaluation = envelope["matrix"]["results"][0]["evaluation"]
    if target == "budget":
        evaluation["budget"]["unit"] = "seconds"
    else:
        evaluation["observations"][0]["unit"] = "seconds"

    with pytest.raises(PerformanceMatrixError, match="semantically invalid"):
        load_performance_budget_matrix(_rehashed(envelope))


def test_noncanonical_transport_and_unknown_result_are_rejected() -> None:
    with pytest.raises(PerformanceMatrixError, match="canonical JSON"):
        load_performance_budget_matrix(json.dumps(_envelope()).encode())
    with pytest.raises(PerformanceMatrixError, match="unknown performance result"):
        load_performance_budget_matrix().require_result("missing.result")
