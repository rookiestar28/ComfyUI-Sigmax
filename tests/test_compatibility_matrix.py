from __future__ import annotations

import hashlib
import importlib.resources
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.compatibility_matrix import (
    CompatibilityMatrixError,
    load_dependency_compatibility_matrix,
)
from scripts import generate_dependency_compatibility_matrix as generator
from scripts import run_dependency_compatibility_lane as runner

ROOT = Path(__file__).resolve().parents[1]


def _envelope() -> dict[str, Any]:
    payload = (
        importlib.resources.files("comfyui_sigmax.compatibility")
        .joinpath("matrix_v1.json")
        .read_bytes()
    )
    return cast(dict[str, Any], json.loads(payload))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _rehashed(envelope: dict[str, Any]) -> bytes:
    envelope["matrix_fingerprint"] = (
        "sha256:" + hashlib.sha256(_canonical(envelope["matrix"])).hexdigest()
    )
    return _canonical(envelope) + b"\n"


def test_packaged_dependency_compatibility_matrix_is_strict_and_versioned() -> None:
    matrix = load_dependency_compatibility_matrix()

    assert matrix.schema == "sigmax.dependency-compatibility-matrix/1"
    assert matrix.matrix_fingerprint.startswith("sha256:")
    policy = cast(dict[str, object], matrix.projection()["policy"])
    assert policy["latest_can_expand_support"] is False
    assert policy["official_container_requires_resolvable_digest"] is True
    assert policy["official_container_unavailable_is_blocking"] is False
    assert policy["third_party_container_substitution"] is False


def test_unknown_lane_is_rejected() -> None:
    matrix = load_dependency_compatibility_matrix()

    with pytest.raises(CompatibilityMatrixError, match="unknown compatibility lane"):
        matrix.require_lane("not-a-lane")


def test_lane_roles_and_non_pass_states_are_explicit() -> None:
    matrix = load_dependency_compatibility_matrix()
    projected_lanes = cast(list[dict[str, Any]], matrix.projection()["lanes"])
    lanes = {lane["id"]: lane for lane in projected_lanes}

    assert lanes["core-windows-py313"]["status"] == "passed"
    assert lanes["core-wsl-py310"]["status"] == "passed"
    assert lanes["pinned-diffusers039-linux-py313-torch290"]["role"] == "known_good"
    assert lanes["pinned-comfyui029-windows-py313-torch213"]["blocking"] is True
    assert lanes["latest-comfyui-head"]["status"] == "passed"
    assert lanes["latest-comfyui-head"]["reason"] == "compatible"
    assert lanes["latest-comfyui-head"]["role"] == "latest_informational"
    assert lanes["latest-comfyui-head"]["components"]["comfy_api"] is None
    assert lanes["latest-comfyui-release-v029"]["status"] == "passed"
    assert lanes["official-comfyui-ci-container"]["status"] == "unavailable"
    assert lanes["official-comfyui-ci-container"]["reason"] == "registry_access_denied"
    assert lanes["official-comfyui-ci-container"]["evidence"]["result_fingerprint"] is None


def test_windows_and_wsl_executed_the_same_invariant_contract() -> None:
    windows = json.loads(
        (ROOT / "tests/compatibility/fixtures/windows_py313_v1.json").read_text(encoding="utf-8")
    )
    wsl = json.loads(
        (ROOT / "tests/compatibility/fixtures/wsl_py310_v1.json").read_text(encoding="utf-8")
    )

    assert windows["status"] == wsl["status"] == "passed"
    assert windows["first_attempt"] == windows["repeat"] == "passed"
    assert wsl["first_attempt"] == wsl["repeat"] == "passed"
    assert windows["contract_fingerprint"] == wsl["contract_fingerprint"]
    assert windows["test_selection_fingerprint"] == wsl["test_selection_fingerprint"]
    assert windows["mandatory_dependencies"] == wsl["mandatory_dependencies"] == 0


def test_lane_selection_excludes_post_generation_matrix_validation() -> None:
    """Lane receipts must be refreshable independently after contract changes."""

    assert "tests/test_compatibility_matrix.py" not in runner.TEST_SELECTION
    assert "tests/test_compatibility_matrix.py" in runner.SOURCE_PATHS


def test_generator_matches_the_packaged_resource() -> None:
    expected = generator._canonical(generator.build_envelope()) + b"\n"
    actual = (
        importlib.resources.files("comfyui_sigmax.compatibility")
        .joinpath("matrix_v1.json")
        .read_bytes()
    )

    assert actual == expected


def test_generator_rejects_failed_evidence_for_a_passed_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = {
        "contract_fingerprint": "sha256:" + "0" * 64,
        "first_attempt": "failed",
        "lane_id": "core-windows-py313",
        "mandatory_dependencies": 0,
        "platform": "windows",
        "python": "3.13.9",
        "repeat": "not_evaluated",
        "schema": "sigmax.compatibility-lane-evidence/1",
        "status": "failed",
        "test_selection_fingerprint": "sha256:" + "1" * 64,
    }
    evidence_path = tmp_path / "failed.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    lane = {
        "blocking": True,
        "components": {
            "comfy_api": None,
            "comfyui": None,
            "container": None,
            "diffusers": None,
            "python": "3.13.9",
            "torch": None,
        },
        "evidence_kind": "executed_invariant_contract",
        "evidence_source": "failed.json",
        "id": "core-windows-py313",
        "platform": "windows",
        "reason": "compatible",
        "role": "supported",
        "status": "passed",
    }

    with pytest.raises(RuntimeError, match="first_attempt"):
        generator._lane(lane)


def test_runner_returns_failure_when_the_executed_contract_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evidence.json"
    failed = {
        "contract_fingerprint": "sha256:" + "0" * 64,
        "first_attempt": "failed",
        "lane_id": "core-windows-py313",
        "mandatory_dependencies": 0,
        "platform": "windows",
        "python": "3.13.9",
        "repeat": "not_evaluated",
        "schema": "sigmax.compatibility-lane-evidence/1",
        "status": "failed",
        "test_selection_fingerprint": "sha256:" + "1" * 64,
    }
    monkeypatch.setattr(runner, "build_evidence", lambda _lane_id: failed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dependency_compatibility_lane.py",
            "--lane-id",
            "core-windows-py313",
            "--output",
            str(output),
        ],
    )

    assert runner.main() == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda envelope: envelope["matrix"]["lanes"][5].update(blocking=False),
            "known-good compatibility lanes must be blocking",
        ),
        (
            lambda envelope: envelope["matrix"]["lanes"][0]["evidence"].update(
                result_fingerprint=None
            ),
            "first/repeat evidence",
        ),
        (
            lambda envelope: envelope["matrix"]["lanes"][2].update(blocking=True),
            "latest compatibility lanes cannot be blocking",
        ),
        (
            lambda envelope: envelope["matrix"]["lanes"][0]["evidence"].update(
                source="C:\\private\\evidence.json"
            ),
            "private or absolute path",
        ),
        (
            lambda envelope: envelope["matrix"]["lanes"][1].update(
                id=envelope["matrix"]["lanes"][0]["id"]
            ),
            "unique and sorted",
        ),
    ],
)
def test_semantically_invalid_rehashed_matrix_is_rejected(mutate: Any, message: str) -> None:
    envelope = _envelope()
    mutate(envelope)

    with pytest.raises(CompatibilityMatrixError, match=message):
        load_dependency_compatibility_matrix(_rehashed(envelope))


def test_mutable_container_tag_cannot_be_rehashed_as_passed() -> None:
    envelope = _envelope()
    lane = next(
        item
        for item in envelope["matrix"]["lanes"]
        if item["id"] == "official-comfyui-ci-container"
    )
    lane.update(status="passed", reason="compatible")
    lane["evidence"].update(
        first_attempt="passed",
        repeat="passed",
        result_fingerprint="sha256:" + "0" * 64,
    )

    with pytest.raises(CompatibilityMatrixError, match="immutable digest"):
        load_dependency_compatibility_matrix(_rehashed(envelope))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "official_container_requires_resolvable_digest",
            False,
            "requires a resolvable digest",
        ),
        (
            "official_container_unavailable_is_blocking",
            True,
            "cannot block acceptance",
        ),
        (
            "third_party_container_substitution",
            True,
            "substitution is forbidden",
        ),
    ],
)
def test_official_container_policy_cannot_be_weakened(
    field: str, value: bool, message: str
) -> None:
    envelope = _envelope()
    envelope["matrix"]["policy"][field] = value

    with pytest.raises(CompatibilityMatrixError, match=message):
        load_dependency_compatibility_matrix(_rehashed(envelope))


def test_noncanonical_transport_is_rejected() -> None:
    payload = json.dumps(_envelope(), ensure_ascii=False).encode("utf-8")

    with pytest.raises(CompatibilityMatrixError, match="canonical JSON"):
        load_dependency_compatibility_matrix(payload)
