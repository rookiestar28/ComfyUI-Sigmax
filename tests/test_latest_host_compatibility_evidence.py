from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from comfyui_sigmax.core import ScheduleContractError
from scripts.sanitize_latest_host_compatibility_evidence import (
    build_latest_host_evidence,
    validate_latest_host_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _transition(result: str = "sha256:" + "1" * 64) -> dict[str, object]:
    return {
        "accepted": True,
        "first": {
            "observed_status": "succeeded",
            "ordinal": 1,
            "reason_code": None,
            "result_fingerprint": result,
            "verdict": "pass",
        },
        "lane": "H1",
        "repeat": {
            "observed_status": "succeeded",
            "ordinal": 2,
            "reason_code": None,
            "result_fingerprint": result,
            "verdict": "pass",
        },
        "schema": "sigmax.host-attempt-transition/1",
        "transition": "pass_to_pass",
    }


def _raw() -> dict[str, object]:
    transition = _transition()
    return {
        "attempt_transitions": {
            "h1": transition,
            "h2_raw.krea2-raw-diffusers-portrait-761x1353": transition,
            "h2_raw.krea2-raw-official-landscape-1353x761": transition,
            "h2_raw.krea2-raw-official-square-1024": transition,
            "h2_raw.raw-auto-variant": transition,
            "h2_raw.raw-invalid-steps": transition,
            "h2_turbo": transition,
            "h3_native_euler": transition,
            "h3_native_euler.partial_denoise": transition,
        },
        "cleanup": "removed",
        "host": {
            "id": "comfyui",
            "revision": "a" * 40,
            "version": "0.29.2",
        },
        "import_probe": {
            "diffusers_loaded": False,
            "node_ids": [
                "Sigmax.AdvancedFlowMatchScheduler",
                "Sigmax.Krea2SigmaScheduler",
                "Sigmax.ModelAwareSigmaScheduler",
                "Sigmax.ProfileInspector",
                "Sigmax.RawWorkflowOutput",
                "Sigmax.ScheduleComparison",
                "Sigmax.ScheduleInspector",
                "Sigmax.TurboWorkflowOutput",
            ],
            "scheduler_registry_unchanged": True,
            "torch_call_unchanged": True,
        },
        "lanes": [
            "H1",
            "H2_TURBO_M2_05",
            "H2_RAW_M3_06",
            "H3_EULER_M5_01",
        ],
        "platform": "windows",
        "schema": "sigmax.comfyui-host-e2e-evidence/3",
    }


def _build(raw: dict[str, object]) -> dict[str, object]:
    return build_latest_host_evidence(
        raw,
        lane_id="latest-comfyui-release-v029",
        expected_revision="a" * 40,
        expected_version="0.29.2",
        python_version="3.13.9",
        torch_version="2.13.0+cpu",
    )


def test_latest_host_evidence_is_sanitized_and_fingerprinted() -> None:
    evidence = _build(_raw())

    assert evidence["status"] == "passed"
    assert evidence["first_attempt"] == evidence["repeat"] == "passed"
    assert evidence["runtime"] == {
        "device": "cpu",
        "model_weights": "not_loaded",
        "python": "3.13.9",
        "torch": "2.13.0+cpu",
    }
    assert str(evidence["result_fingerprint"]).startswith("sha256:")
    assert "host_log_tail" not in evidence


@pytest.mark.parametrize(
    "relative_path",
    [
        "tests/compatibility/fixtures/comfyui_release_v0292_v1.json",
        "tests/compatibility/fixtures/comfyui_head_v1.json",
    ],
)
def test_public_latest_host_evidence_is_valid(relative_path: str) -> None:
    evidence = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))

    validate_latest_host_evidence(evidence)


def test_sanitized_latest_host_fingerprint_drift_is_rejected() -> None:
    evidence = _build(_raw())
    evidence["result_fingerprint"] = "sha256:" + "0" * 64

    with pytest.raises(ScheduleContractError, match="fingerprint drifted"):
        validate_latest_host_evidence(evidence)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda raw: raw.update(cleanup="retained_failure_artifacts"),
            "cleanup",
        ),
        (
            lambda raw: raw["import_probe"].update(diffusers_loaded=True),
            "import-safety",
        ),
        (
            lambda raw: raw["attempt_transitions"]["h1"].update(accepted=False),
            "transition",
        ),
        (
            lambda raw: raw["attempt_transitions"]["h1"]["repeat"].update(
                result_fingerprint="sha256:" + "2" * 64
            ),
            "first/repeat",
        ),
    ],
)
def test_latest_host_evidence_rejects_false_pass(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    raw = copy.deepcopy(_raw())
    mutation(raw)

    with pytest.raises(ScheduleContractError, match=message):
        _build(raw)
