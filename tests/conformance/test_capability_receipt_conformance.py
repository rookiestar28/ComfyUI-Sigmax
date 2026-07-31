"""Canonical capability, receipt, and host-attempt conformance regressions."""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

REPORT = importlib.import_module("scripts.conformance.capability_receipt_report")
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "capability_receipt_conformance_v1.json"


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in cast(list[dict[str, Any]], report["cases"])}


def test_conformance_assets_and_canonical_fixture_exist() -> None:
    assert REPORT.REPORT_SCHEMA == "sigmax.capability-receipt-conformance/1"
    assert FIXTURE.is_file()
    assert _fixture()["schema"] == REPORT.REPORT_SCHEMA


def test_report_rebuilds_the_committed_fixture_from_real_contracts() -> None:
    rebuilt = REPORT.build_conformance_report()

    assert rebuilt == _fixture()
    assert REPORT.validate_conformance_report(rebuilt) is rebuilt
    assert REPORT.canonical_json(rebuilt) == FIXTURE.read_text(encoding="utf-8")


def test_fixture_covers_allow_warn_reject_and_stable_reason_codes() -> None:
    cases = _case_map(_fixture())

    assert {
        identifier: (case["decision"]["level"], case["decision"]["reasons"])
        for identifier, case in cases.items()
    } == {
        "model_owned_sigmas_reject": (
            "reject",
            ["model_ownership_unsupported"],
        ),
        "native_euler_allow": ("allow", ["compatible"]),
        "nonreference_euler_warn": (
            "warn",
            ["sampler_not_profile_reference"],
        ),
        "partial_denoise_reject": (
            "reject",
            ["partial_denoise_unsupported_by_sampler"],
        ),
        "per_token_timesteps_reject": (
            "reject",
            [
                "per_token_timesteps_unsupported_by_model",
                "per_token_timesteps_unsupported_by_profile",
                "per_token_timesteps_unsupported_by_sampler",
            ],
        ),
        "resume_state_reject": ("reject", ["sampler_state_unsupported"]),
        "stochastic_noise_reject": (
            "reject",
            [
                "execution_behavior_mismatch",
                "noise_ownership_mismatch",
            ],
        ),
        "terminal_zero_reject": (
            "reject",
            ["terminal_requirement_mismatch"],
        ),
    }
    assert all(
        case["decision"]["considered"] == [item.value for item in REPORT.CapabilityDimension]
        for case in cases.values()
    )


def test_native_euler_case_pins_effective_semantics_without_double_shift() -> None:
    allow = _case_map(_fixture())["native_euler_allow"]

    assert allow["semantics"] == {
        "execution_behavior": "deterministic",
        "noise_ownership": "none",
        "partial_denoise_requested": False,
        "per_token_timesteps_requested": False,
        "required_state": [],
        "sampler_id": "comfy.euler",
        "schedule_ownership": "external_sigmas",
        "terminal_requirement": "requires_zero",
    }
    portable = _fixture()["portable_execution"]
    assert portable["double_shift"] is False
    assert portable["shift_count"] == 1
    assert portable["status"] == "succeeded"
    assert portable["counts"] == {
        "effective_model_evaluations": 8,
        "effective_transitions": 8,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }
    assert portable["rng_ownership"] == {
        "model": "none",
        "sampler": "none",
        "schedule": "none",
    }


def test_unsupported_capability_seams_are_explicit_and_not_skipped() -> None:
    coverage = _fixture()["coverage"]

    assert coverage["levels"] == ["allow", "warn", "reject"]
    assert coverage["capabilities"] == [
        "deterministic_native_euler",
        "effective_model_evaluations",
        "noise_ownership",
        "partial_denoise",
        "per_token_timesteps",
        "resume",
        "rng_separation",
        "sampler_state",
        "schedule_ownership",
        "stochastic_execution",
        "terminal_sigma",
    ]
    assert coverage["unsupported"] == [
        "advanced_workflows",
        "partial_denoise_execution",
        "resume_execution",
        "stochastic_euler_execution",
    ]
    assert coverage["skipped"] == []


def test_artifact_receipt_and_bundle_round_trips_are_exact() -> None:
    portable = _fixture()["portable_execution"]

    assert portable["artifact_round_trip"] is True
    assert portable["receipt_round_trip"] is True
    assert portable["bundle_round_trip"] is True
    assert portable["artifact"]["construction_fingerprint"].startswith("sha256:")
    assert portable["artifact"]["numerical_fingerprint"].startswith("sha256:")
    assert portable["receipt_fingerprint"].startswith("sha256:")
    assert portable["bundle_fingerprint"].startswith("sha256:")
    assert portable["effective_inputs"] == {
        "height": 1024,
        "precision": "float64",
        "profile": "krea2.turbo.official",
        "profile_version": "1",
        "steps": 8,
        "width": 1024,
    }


def test_host_attempt_regressions_retain_first_failure_and_repeat_transitions() -> None:
    regressions = {
        item["id"]: item
        for item in cast(list[dict[str, Any]], _fixture()["host_attempt_regressions"])
    }

    assert regressions["stable_expected_rejection"]["transition"] == "pass_to_pass"
    assert regressions["stable_expected_rejection"]["accepted"] is True
    assert regressions["first_failure_then_pass"]["transition"] == "fail_to_pass"
    assert regressions["first_failure_then_pass"]["accepted"] is False
    assert regressions["first_failure_then_pass"]["first"]["reason_code"] == (
        "host.first_attempt_failure"
    )
    assert regressions["repeat_result_drift"]["transition"] == "pass_to_pass_changed"
    assert regressions["repeat_result_drift"]["accepted"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_root",
        "case_reason",
        "case_level",
        "case_ownership",
        "case_resume",
        "count",
        "rng_owner",
        "artifact_fingerprint",
        "receipt_fingerprint",
        "round_trip",
        "first_attempt",
        "repeat_transition",
        "unsupported_skip",
    ),
)
def test_report_validator_rejects_tampered_conformance_evidence(mutation: str) -> None:
    report = copy.deepcopy(_fixture())
    cases = _case_map(report)
    if mutation == "unknown_root":
        report["unknown"] = True
    elif mutation == "case_reason":
        cases["native_euler_allow"]["decision"]["reasons"] = ["compatible", "extra"]
    elif mutation == "case_level":
        cases["nonreference_euler_warn"]["decision"]["level"] = "allow"
    elif mutation == "case_ownership":
        cases["native_euler_allow"]["semantics"]["schedule_ownership"] = "model_native"
    elif mutation == "case_resume":
        cases["resume_state_reject"]["semantics"]["required_state"] = []
    elif mutation == "count":
        report["portable_execution"]["counts"]["effective_model_evaluations"] = 7
    elif mutation == "rng_owner":
        report["portable_execution"]["rng_ownership"]["sampler"] = "sampler"
    elif mutation == "artifact_fingerprint":
        report["portable_execution"]["artifact"]["construction_fingerprint"] = "sha256:" + (
            "0" * 64
        )
    elif mutation == "receipt_fingerprint":
        report["portable_execution"]["receipt_fingerprint"] = "sha256:" + ("0" * 64)
    elif mutation == "round_trip":
        report["portable_execution"]["bundle_round_trip"] = False
    elif mutation == "first_attempt":
        report["host_attempt_regressions"][0]["first"]["verdict"] = "pass"
    elif mutation == "repeat_transition":
        report["host_attempt_regressions"][1]["transition"] = "pass_to_pass"
    else:
        report["coverage"]["skipped"] = ["resume"]

    with pytest.raises((TypeError, ValueError)):
        REPORT.validate_conformance_report(report)


def test_attempt_builder_never_allows_repeat_to_erase_first_failure() -> None:
    transition = REPORT.build_host_attempt_transition(
        lane="H3_EULER_M5_01",
        first=REPORT.HostAttempt(
            ordinal=1,
            verdict="fail",
            observed_status="error",
            reason_code="host.first_attempt_failure",
            result_fingerprint="sha256:" + ("1" * 64),
        ),
        repeat=REPORT.HostAttempt(
            ordinal=2,
            verdict="pass",
            observed_status="succeeded",
            reason_code=None,
            result_fingerprint="sha256:" + ("2" * 64),
        ),
    )

    assert transition["transition"] == "fail_to_pass"
    assert transition["accepted"] is False
    assert REPORT.validate_host_attempt_transition(transition) is transition


def test_host_attempt_rejects_noninteger_ordinal() -> None:
    with pytest.raises((TypeError, ValueError)):
        REPORT.HostAttempt(
            ordinal=cast(Any, 1.0),
            verdict="pass",
            observed_status="succeeded",
            reason_code=None,
            result_fingerprint="sha256:" + ("1" * 64),
        )


def test_canonical_report_is_stable_across_hash_seeds() -> None:
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from scripts.conformance.capability_receipt_report import "
        "build_conformance_report, canonical_json; "
        "print(canonical_json(build_conformance_report()), end='')"
    )
    expected = FIXTURE.read_text(encoding="utf-8")
    for seed in ("1", "917"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script, str(ROOT)],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=environment,
        )
        assert completed.stdout == expected
