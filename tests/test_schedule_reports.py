"""Immutable artifact/receipt schedule reports and comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    ArtifactBuildMetadata,
    ArtifactField,
    BaseGridSpec,
    CapabilityDimension,
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityReason,
    EvidenceLevel,
    ExecutionComponent,
    ExecutionHost,
    ExecutionReceipt,
    ExecutionReceiptMetadata,
    ExecutionRngOwnership,
    ExecutionStatus,
    NoiseOwnership,
    PortableExecutionBundle,
    Provenance,
    ScheduleArtifact,
    ScheduleContractError,
    ScheduleInputs,
    ScheduleOwnership,
    ScheduleRequest,
    ScheduleResult,
    SigmaDomain,
    SliceSpec,
    TerminalPolicy,
    TransformContract,
    TransformStage,
    TypedArtifactValue,
    build_execution_receipt,
    build_schedule_artifact,
    build_schedule_comparison_report,
    build_schedule_report,
    build_schedule_report_from_bundle,
    deserialize_schedule_comparison_report,
    deserialize_schedule_report,
    float_to_ieee_hex,
    serialize_schedule_comparison_report,
    serialize_schedule_report,
)
from comfyui_sigmax.nodes.advanced_flowmatch_scheduler import (
    build_advanced_flowmatch_schedule,
)
from comfyui_sigmax.nodes.inspectors import build_schedule_comparison

ROOT = Path(__file__).resolve().parents[1]


def _artifact(
    sigmas: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.0),
    *,
    domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
    precision: str = "float64",
) -> ScheduleArtifact:
    steps = len(sigmas) - 1
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=steps, width=1024, height=768),
        sigma_domain=domain,
        provenance=Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.EXPERIMENTAL,
            source="fixture",
            source_revision="abc123",
            profile_id="fixture.report",
            profile_version="1",
        ),
        base_grid=BaseGridSpec(identifier="fixture.grid", output_domain=domain),
        transforms=(
            TransformContract(
                name="fixture.shift",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=domain,
                output_domain=domain,
            ),
        ),
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(start_step=0, end_step=steps, denoise=1.0),
    )
    result = ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=steps, width=1024, height=768),
        sigmas=sigmas,
        final_domain=domain,
    )
    return build_schedule_artifact(
        result,
        metadata=ArtifactBuildMetadata(
            source_id="fixture.schedule-report",
            source_label="Fixture schedule report",
            base_grid_parameters=(ArtifactField(name="points", value=steps),),
            transform_parameters=(
                (
                    ArtifactField(
                        name="mu",
                        value=TypedArtifactValue(value=1.15, precision=cast(Any, precision)),
                    ),
                ),
            ),
            compatibility=(ArtifactField(name="decision", value="allow"),),
        ),
        precision=cast(Any, precision),
    )


def _fingerprint(character: str) -> str:
    return "sha256:" + character * 64


def _receipt(
    artifact: ScheduleArtifact,
    *,
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
) -> ExecutionReceipt:
    effective_inputs = cast(dict[str, object], artifact.construction_projection()["effective"])
    steps = cast(int, effective_inputs["steps"])
    effective = steps if status is ExecutionStatus.SUCCEEDED else 0
    return build_execution_receipt(
        artifact,
        metadata=ExecutionReceiptMetadata(
            compatibility=CompatibilityDecision(
                level=CompatibilityLevel.ALLOW,
                considered=tuple(CapabilityDimension),
                reasons=(CompatibilityReason.COMPATIBLE,),
            ),
            host=ExecutionHost(
                identifier="comfyui",
                version="0.29.0",
                revision="host-revision",
                api_version="legacy_v1",
            ),
            model=ExecutionComponent(
                identifier="fixture.model",
                version="1",
                fingerprint=_fingerprint("1"),
            ),
            sampler=ExecutionComponent(
                identifier="comfy.euler",
                version="0.29.0",
                fingerprint=_fingerprint("2"),
            ),
            rng_ownership=ExecutionRngOwnership(
                schedule=NoiseOwnership.NONE,
                model=NoiseOwnership.NONE,
                sampler=NoiseOwnership.NONE,
            ),
            requested_transitions=steps,
            effective_transitions=effective,
            requested_model_evaluations=steps,
            effective_model_evaluations=effective,
            status=status,
        ),
    )


def _decoded(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _token(value: float, precision: str = "float64") -> dict[str, str]:
    return {
        "bits": float_to_ieee_hex(value, cast(Any, precision)),
        "precision": precision,
    }


def test_schedule_report_exposes_artifact_samples_deltas_and_construction() -> None:
    artifact = _artifact()
    report = build_schedule_report(artifact)
    projection = report.projection()

    assert projection["schema"] == "sigmax.schedule-report/1"
    assert projection["artifact"] == {
        "construction_fingerprint": artifact.construction_fingerprint,
        "numerical_fingerprint": artifact.numerical_fingerprint,
    }
    assert projection["domain"] == "unit_flow"
    assert projection["precision"] == "float64"
    assert projection["receipt_present"] is False
    assert projection["execution"] is None
    assert projection["effective_inputs"] == artifact.construction_projection()["effective"]
    assert projection["construction"] == {
        "base_grid": artifact.construction_projection()["base_grid"],
        "slicing": artifact.construction_projection()["slicing"],
        "terminal": artifact.construction_projection()["terminal"],
        "transforms": artifact.construction_projection()["transforms"],
    }
    assert projection["samples"] == [
        {"delta_to_next": _token(-0.25), "index": 0, "sigma": _token(1.0)},
        {"delta_to_next": _token(-0.25), "index": 1, "sigma": _token(0.75)},
        {"delta_to_next": _token(-0.25), "index": 2, "sigma": _token(0.5)},
        {"delta_to_next": _token(-0.25), "index": 3, "sigma": _token(0.25)},
        {"delta_to_next": None, "index": 4, "sigma": _token(0.0)},
    ]


def test_report_with_receipt_exposes_truthful_execution_and_bundle_path() -> None:
    artifact = _artifact()
    receipt = _receipt(artifact)
    direct = build_schedule_report(artifact, receipt=receipt)
    bundled = build_schedule_report_from_bundle(
        PortableExecutionBundle(artifact=artifact, receipt=receipt)
    )
    projection = direct.projection()

    assert bundled == direct
    assert projection["receipt_present"] is True
    assert projection["execution"] == {
        "compatibility": receipt.projection()["compatibility"],
        "counts": {
            "effective_model_evaluations": 4,
            "effective_transitions": 4,
            "requested_model_evaluations": 4,
            "requested_transitions": 4,
        },
        "host": receipt.projection()["host"],
        "model": receipt.projection()["model"],
        "reason_code": None,
        "receipt_fingerprint": receipt.receipt_fingerprint,
        "rng_ownership": receipt.projection()["rng_ownership"],
        "sampler": receipt.projection()["sampler"],
        "status": "succeeded",
    }


def test_schedule_report_rejects_receipt_from_another_artifact() -> None:
    first = _artifact()
    second = _artifact((1.0, 0.5, 0.25, 0.125, 0.0))

    with pytest.raises(ScheduleContractError, match=r"receipt|artifact|fingerprint"):
        build_schedule_report(first, receipt=_receipt(second))


def test_schedule_report_transport_is_canonical_and_tamper_evident() -> None:
    report = build_schedule_report(_artifact(), receipt=_receipt(_artifact()))
    payload = serialize_schedule_report(report)
    envelope = _decoded(payload)

    assert envelope["schema"] == "sigmax.schedule-report-envelope/1"
    assert deserialize_schedule_report(payload) == report
    assert serialize_schedule_report(deserialize_schedule_report(payload)) == payload
    envelope["report"]["samples"][1]["sigma"]["bits"] = "0000000000000000"
    tampered = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ScheduleContractError, match=r"delta|fingerprint"):
        deserialize_schedule_report(tampered)

    envelope = _decoded(payload)
    envelope["report"]["unknown"] = True
    report_bytes = json.dumps(envelope["report"], separators=(",", ":"), sort_keys=True).encode()
    envelope["report_fingerprint"] = "sha256:" + hashlib.sha256(report_bytes).hexdigest()
    rehashed = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ScheduleContractError, match=r"fields|schema"):
        deserialize_schedule_report(rehashed)


def test_schedule_report_fingerprint_is_pinned() -> None:
    artifact = _artifact()
    report = build_schedule_report(artifact, receipt=_receipt(artifact))

    assert (
        report.report_fingerprint
        == "sha256:85247c91aac8ed3ca775e67aa59b5db9b6433770ab0ca7d11aa51a1ceb5b4948"
    )


@pytest.mark.parametrize("location", ("source", "construction", "execution"))
def test_rehashed_unknown_nested_report_fields_reject(location: str) -> None:
    artifact = _artifact()
    envelope = _decoded(
        serialize_schedule_report(build_schedule_report(artifact, receipt=_receipt(artifact)))
    )
    if location == "source":
        envelope["report"]["source"]["unknown"] = True
    elif location == "construction":
        envelope["report"]["construction"]["base_grid"]["unknown"] = True
    else:
        envelope["report"]["execution"]["unknown"] = True
    report_bytes = json.dumps(
        envelope["report"],
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    envelope["report_fingerprint"] = "sha256:" + hashlib.sha256(report_bytes).hexdigest()

    with pytest.raises(ScheduleContractError, match="fields"):
        deserialize_schedule_report(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode()
        )


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"\xef\xbb\xbf{}",
        b'{"schema":"sigmax.schedule-report-envelope/1","schema":"duplicate"}',
        b'{ "schema":"sigmax.schedule-report-envelope/1" }',
        b"[]",
    ),
)
def test_schedule_report_transport_rejects_malformed_input(payload: bytes) -> None:
    with pytest.raises(ScheduleContractError):
        deserialize_schedule_report(payload)


def test_comparison_report_exposes_exact_rows_and_statistics() -> None:
    first = build_schedule_report(_artifact())
    second = build_schedule_report(_artifact((1.0, 0.5, 0.375, 0.125, 0.0)))
    comparison = build_schedule_comparison_report(first, second)
    projection = comparison.projection()

    assert projection["schema"] == "sigmax.schedule-comparison-report/1"
    assert projection["comparable"] is True
    assert projection["reason"] is None
    assert projection["alignment"] == {
        "kind": "sigma_index",
        "length": 5,
        "terminal_inclusive": True,
    }
    rows = cast(list[dict[str, Any]], projection["samples"])
    assert [row["absolute_difference"] for row in rows] == [
        _token(0.0),
        _token(0.25),
        _token(0.125),
        _token(0.125),
        _token(0.0),
    ]
    assert [row["relative_difference"] for row in rows] == [
        _token(0.0),
        _token(1.0 / 3.0),
        _token(0.25),
        _token(0.5),
        _token(0.0),
    ]
    assert projection["summary"] == {
        "exact_match_count": 2,
        "maximum_absolute_difference": _token(0.25),
        "maximum_absolute_index": 1,
        "maximum_relative_difference": _token(0.5),
        "maximum_relative_index": 3,
        "mean_absolute_difference": _token(0.1),
        "mean_relative_difference": _token(13.0 / 60.0),
    }


@pytest.mark.parametrize(
    ("second", "reason"),
    (
        (_artifact((1.0, 0.5, 0.0)), "comparison.length_mismatch"),
        (
            _artifact((10.0, 1.0, 0.0), domain=SigmaDomain.CONTINUOUS_EDM),
            "comparison.domain_mismatch",
        ),
    ),
)
def test_comparison_mismatch_never_aligns_or_truncates(
    second: ScheduleArtifact,
    reason: str,
) -> None:
    comparison = build_schedule_comparison_report(
        build_schedule_report(_artifact()),
        build_schedule_report(second),
    ).projection()

    assert comparison["comparable"] is False
    assert comparison["reason"] == reason
    alignment = cast(dict[str, object], comparison["alignment"])
    assert alignment["kind"] == "none"
    assert comparison["samples"] == []
    assert comparison["summary"] is None


def test_comparison_transport_round_trips_and_detects_tampering() -> None:
    comparison = build_schedule_comparison_report(
        build_schedule_report(_artifact()),
        build_schedule_report(_artifact((1.0, 0.5, 0.375, 0.125, 0.0))),
    )
    payload = serialize_schedule_comparison_report(comparison)
    envelope = _decoded(payload)

    assert envelope["schema"] == "sigmax.schedule-comparison-report-envelope/1"
    assert deserialize_schedule_comparison_report(payload) == comparison
    assert (
        serialize_schedule_comparison_report(deserialize_schedule_comparison_report(payload))
        == payload
    )
    envelope["report"]["summary"]["exact_match_count"] = 4
    tampered = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ScheduleContractError, match=r"statistics|fingerprint"):
        deserialize_schedule_comparison_report(tampered)


def test_comparison_report_fingerprint_is_pinned() -> None:
    comparison = build_schedule_comparison_report(
        build_schedule_report(_artifact()),
        build_schedule_report(_artifact((1.0, 0.5, 0.375, 0.125, 0.0))),
    )

    assert (
        comparison.report_fingerprint
        == "sha256:c037b77c693309a080e46ca49c738ad32961e61103f2bfd9826f223fc08759e3"
    )


def test_comparison_statistics_match_existing_verified_schedule_comparator() -> None:
    schedules = tuple(
        build_advanced_flowmatch_schedule(
            domain="UNIT_FLOW",
            steps=4,
            sigma_start=1.0,
            sigma_end=0.1,
            shift_mode="direct_ratio",
            shift_value=shift,
            terminal_policy="append_zero",
            start_step=0,
            end_step=-1,
        )
        for shift in (2.0, 3.0)
    )
    existing = _decoded(
        build_schedule_comparison(
            sigmas_a=schedules[0].sigmas,
            schedule_info_a=schedules[0].schedule_info_json,
            sigmas_b=schedules[1].sigmas,
            schedule_info_b=schedules[1].schedule_info_json,
        ).report_json.encode()
    )
    report = build_schedule_comparison_report(
        build_schedule_report(_artifact(schedules[0].sigmas)),
        build_schedule_report(_artifact(schedules[1].sigmas)),
    ).projection()

    assert report["comparable"] is True
    assert report["alignment"] == existing["alignment"]
    report_rows = cast(list[dict[str, Any]], report["samples"])
    for report_row, existing_row in zip(
        report_rows,
        existing["steps"],
        strict=True,
    ):
        for report_name, existing_name in (
            ("sigma_a", "sigma_a"),
            ("sigma_b", "sigma_b"),
            ("absolute_difference", "absolute_difference"),
            ("relative_difference", "relative_difference"),
        ):
            assert report_row[report_name] == _token(existing_row[existing_name])
    report_summary = cast(dict[str, Any], report["summary"])
    existing_summary = existing["summary"]
    for field_name in (
        "maximum_absolute_difference",
        "maximum_relative_difference",
        "mean_absolute_difference",
        "mean_relative_difference",
    ):
        assert report_summary[field_name] == _token(existing_summary[field_name])
    for field_name in (
        "exact_match_count",
        "maximum_absolute_index",
        "maximum_relative_index",
    ):
        assert report_summary[field_name] == existing_summary[field_name]


def test_rehashed_unknown_comparison_alignment_field_rejects() -> None:
    comparison = build_schedule_comparison_report(
        build_schedule_report(_artifact()),
        build_schedule_report(_artifact((1.0, 0.5, 0.375, 0.125, 0.0))),
    )
    envelope = _decoded(serialize_schedule_comparison_report(comparison))
    envelope["report"]["alignment"]["unknown"] = True
    report_bytes = json.dumps(
        envelope["report"],
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    envelope["report_fingerprint"] = "sha256:" + hashlib.sha256(report_bytes).hexdigest()

    with pytest.raises(ScheduleContractError, match="fields"):
        deserialize_schedule_comparison_report(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode()
        )


def test_report_serialization_is_stable_across_hash_seeds() -> None:
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tests.test_schedule_reports import _artifact, _receipt; "
        "from comfyui_sigmax.core import build_schedule_report, serialize_schedule_report; "
        "a=_artifact(); sys.stdout.buffer.write(serialize_schedule_report("
        "build_schedule_report(a, receipt=_receipt(a))))"
    )
    expected_artifact = _artifact()
    expected = serialize_schedule_report(
        build_schedule_report(expected_artifact, receipt=_receipt(expected_artifact))
    )
    for seed in ("1", "917"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script, str(ROOT)],
            check=True,
            capture_output=True,
            cwd=ROOT,
            env=environment,
        )
        assert completed.stdout == expected


def test_report_import_does_not_load_optional_or_host_frameworks() -> None:
    script = (
        "import builtins,sys; real=builtins.__import__; "
        "blocked={'matplotlib','numpy','torch','comfy','diffusers'}; "
        "builtins.__import__=lambda n,*a,**k: "
        "(_ for _ in ()).throw(ImportError(n)) if n.split('.')[0] in blocked "
        "else real(n,*a,**k); "
        "import comfyui_sigmax.core.reports; "
        "assert not blocked.intersection(sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True, cwd=ROOT)
