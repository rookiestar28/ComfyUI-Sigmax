"""Portable execution receipt and artifact bundle contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
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
    OverrideRecord,
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
    deserialize_execution_receipt,
    deserialize_portable_execution_bundle,
    serialize_execution_receipt,
    serialize_portable_execution_bundle,
)
from comfyui_sigmax.core import execution_receipts as receipts_module

ROOT = Path(__file__).resolve().parents[1]


def _artifact() -> ScheduleArtifact:
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=5, width=1024, height=768),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.EXPERIMENTAL,
            source="fixture",
            source_revision="abc123",
            profile_id="fixture.profile",
            profile_version="1",
        ),
        base_grid=BaseGridSpec(
            identifier="fixture.grid",
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        transforms=(
            TransformContract(
                name="fixture.shift",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
        ),
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=SliceSpec(start_step=0, end_step=4, denoise=1.0),
        overrides=(
            OverrideRecord(
                field="steps",
                requested_value="5",
                effective_value="4",
                reason="fixture override",
            ),
        ),
    )
    result = ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=4, width=1024, height=768),
        sigmas=(1.0, 0.75, 0.5, 0.25, 0.0),
        final_domain=SigmaDomain.UNIT_FLOW,
    )
    return build_schedule_artifact(
        result,
        metadata=ArtifactBuildMetadata(
            source_id="fixture.receipt",
            source_label="Fixture receipt",
            base_grid_parameters=(ArtifactField(name="points", value=4),),
            transform_parameters=(
                (
                    ArtifactField(
                        name="mu",
                        value=TypedArtifactValue(value=1.15, precision="float64"),
                    ),
                ),
            ),
            compatibility=(ArtifactField(name="decision", value="allow"),),
        ),
        precision="float64",
    )


def _fingerprint(character: str) -> str:
    return "sha256:" + character * 64


def _decision(
    level: CompatibilityLevel = CompatibilityLevel.ALLOW,
) -> CompatibilityDecision:
    reasons = (
        (CompatibilityReason.COMPATIBLE,)
        if level is CompatibilityLevel.ALLOW
        else (CompatibilityReason.MODEL_FAMILY_MISMATCH,)
    )
    return CompatibilityDecision(
        level=level,
        considered=tuple(CapabilityDimension),
        reasons=reasons,
    )


def _metadata(
    *,
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
    effective_transitions: int = 4,
    effective_model_evaluations: int = 4,
    reason_code: str | None = None,
    compatibility: CompatibilityDecision | None = None,
) -> ExecutionReceiptMetadata:
    return ExecutionReceiptMetadata(
        compatibility=compatibility or _decision(),
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
        requested_transitions=4,
        effective_transitions=effective_transitions,
        requested_model_evaluations=4,
        effective_model_evaluations=effective_model_evaluations,
        status=status,
        reason_code=reason_code,
    )


def _receipt(**changes: object) -> ExecutionReceipt:
    metadata = replace(_metadata(), **cast(Any, changes)) if changes else _metadata()
    return build_execution_receipt(_artifact(), metadata=metadata)


def _decoded(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload)
    assert isinstance(value, dict)
    return value


def _mapping(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _receipt_from_projection(projection: dict[str, object]) -> ExecutionReceipt:
    payload = _canonical(projection)
    artifact = cast(dict[str, object], projection["artifact"])
    return ExecutionReceipt(
        receipt_bytes=payload,
        receipt_fingerprint="sha256:" + hashlib.sha256(payload).hexdigest(),
        construction_fingerprint=cast(str, artifact["construction_fingerprint"]),
        numerical_fingerprint=cast(str, artifact["numerical_fingerprint"]),
    )


def test_receipt_records_required_execution_evidence_and_artifact_identities() -> None:
    artifact = _artifact()
    receipt = build_execution_receipt(artifact, metadata=_metadata())
    projection = receipt.projection()

    assert projection["schema"] == "sigmax.execution-receipt/1"
    assert projection["artifact"] == {
        "construction_fingerprint": artifact.construction_fingerprint,
        "numerical_fingerprint": artifact.numerical_fingerprint,
    }
    assert projection["effective_inputs"] == artifact.construction_projection()["effective"]
    assert projection["profile"] == {"id": "fixture.profile", "version": "1"}
    assert _mapping(projection["compatibility"])["level"] == "allow"
    assert _mapping(projection["compatibility"])["reasons"] == ["compatible"]
    assert projection["host"] == {
        "api_version": "legacy_v1",
        "id": "comfyui",
        "revision": "host-revision",
        "version": "0.29.0",
    }
    assert _mapping(projection["model"])["fingerprint"] == _fingerprint("1")
    assert _mapping(projection["sampler"])["fingerprint"] == _fingerprint("2")
    assert projection["rng_ownership"] == {
        "model": "none",
        "sampler": "none",
        "schedule": "none",
    }
    assert projection["counts"] == {
        "effective_model_evaluations": 4,
        "effective_transitions": 4,
        "requested_model_evaluations": 4,
        "requested_transitions": 4,
    }
    assert projection["execution"] == {"reason_code": None, "status": "succeeded"}


def test_receipt_transport_is_canonical_deterministic_and_immutable() -> None:
    first = _receipt()
    second = _receipt()
    payload = serialize_execution_receipt(first)

    assert first == second
    assert deserialize_execution_receipt(payload) == first
    assert serialize_execution_receipt(deserialize_execution_receipt(payload)) == payload
    assert _decoded(payload)["schema"] == "sigmax.execution-receipt-envelope/1"
    assert _decoded(payload)["receipt_fingerprint"] == first.receipt_fingerprint
    assert first.receipt_fingerprint == (
        "sha256:" + hashlib.sha256(first.receipt_bytes).hexdigest()
    )
    with pytest.raises(FrozenInstanceError):
        first.receipt_bytes = b"{}"  # type: ignore[misc]
    projection = first.projection()
    projection.clear()
    assert first.projection()["schema"] == "sigmax.execution-receipt/1"


def test_receipt_and_bundle_match_committed_golden_identities() -> None:
    fixture_root = ROOT / "tests" / "fixtures" / "artifacts"
    expected_projection = json.loads(
        (fixture_root / "execution_receipt_projection_v1.json").read_text(encoding="utf-8")
    )
    expected_hashes = json.loads(
        (fixture_root / "execution_receipt_hashes_v1.json").read_text(encoding="utf-8")
    )
    artifact = _artifact()
    receipt = _receipt()
    bundle = PortableExecutionBundle(artifact=artifact, receipt=receipt)
    bundle_payload = serialize_portable_execution_bundle(bundle)

    assert receipt.projection() == expected_projection
    assert receipt.receipt_fingerprint == expected_hashes["receipt_fingerprint"]
    assert (
        "sha256:" + hashlib.sha256(bundle_payload).hexdigest() == expected_hashes["bundle_sha256"]
    )


@pytest.mark.parametrize(
    ("status", "effective_transitions", "effective_evaluations", "reason"),
    (
        (ExecutionStatus.NOT_EXECUTED, 0, 0, None),
        (ExecutionStatus.SUCCEEDED, 4, 4, None),
        (ExecutionStatus.FAILED, 1, 2, "execution.model_error"),
        (ExecutionStatus.INTERRUPTED, 2, 2, "execution.user_interrupt"),
    ),
)
def test_receipt_statuses_preserve_truthful_counts(
    status: ExecutionStatus,
    effective_transitions: int,
    effective_evaluations: int,
    reason: str | None,
) -> None:
    receipt = build_execution_receipt(
        _artifact(),
        metadata=_metadata(
            status=status,
            effective_transitions=effective_transitions,
            effective_model_evaluations=effective_evaluations,
            reason_code=reason,
        ),
    )
    projection = receipt.projection()

    assert projection["execution"] == {
        "reason_code": reason,
        "status": status.value,
    }
    assert _mapping(projection["counts"])["effective_transitions"] == effective_transitions
    assert _mapping(projection["counts"])["effective_model_evaluations"] == effective_evaluations


@pytest.mark.parametrize(
    "changes",
    (
        {"requested_transitions": 3},
        {"effective_transitions": 5},
        {"requested_model_evaluations": 0},
        {"effective_model_evaluations": 5},
        {"status": ExecutionStatus.SUCCEEDED, "effective_transitions": 3},
        {"status": ExecutionStatus.SUCCEEDED, "effective_model_evaluations": 3},
        {"status": ExecutionStatus.SUCCEEDED, "reason_code": "execution.bad"},
        {"status": ExecutionStatus.NOT_EXECUTED, "effective_transitions": 1},
        {"status": ExecutionStatus.NOT_EXECUTED, "effective_model_evaluations": 1},
        {"status": ExecutionStatus.NOT_EXECUTED, "reason_code": "execution.bad"},
        {"status": ExecutionStatus.FAILED, "reason_code": None},
        {"status": ExecutionStatus.INTERRUPTED, "reason_code": None},
    ),
)
def test_receipt_rejects_inconsistent_counts_and_status(changes: dict[str, object]) -> None:
    with pytest.raises(ScheduleContractError):
        build_execution_receipt(
            _artifact(),
            metadata=replace(_metadata(), **cast(Any, changes)),
        )


def test_rejected_compatibility_cannot_claim_success() -> None:
    rejected = _decision(CompatibilityLevel.REJECT)

    with pytest.raises(ScheduleContractError, match=r"REJECT|rejected"):
        build_execution_receipt(
            _artifact(),
            metadata=_metadata(compatibility=rejected),
        )
    receipt = build_execution_receipt(
        _artifact(),
        metadata=_metadata(
            status=ExecutionStatus.NOT_EXECUTED,
            effective_transitions=0,
            effective_model_evaluations=0,
            compatibility=rejected,
        ),
    )
    assert _mapping(receipt.projection()["compatibility"])["level"] == "reject"


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ExecutionComponent(
            identifier="model",
            version="1",
            fingerprint="bad",
        ),
        lambda: ExecutionComponent(
            identifier="C:\\Users\\private\\model",
            version="1",
            fingerprint=_fingerprint("1"),
        ),
        lambda: ExecutionComponent(
            identifier="api_token",
            version="1",
            fingerprint=_fingerprint("1"),
        ),
        lambda: ExecutionHost(
            identifier="host",
            version="/home/private/host",
            revision="rev",
            api_version="v1",
        ),
        lambda: ExecutionRngOwnership(
            schedule=cast(Any, "none"),
            model=NoiseOwnership.NONE,
            sampler=NoiseOwnership.NONE,
        ),
    ),
)
def test_execution_metadata_rejects_invalid_secret_or_private_values(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("compatibility", object()),
        ("host", object()),
        ("model", object()),
        ("sampler", object()),
        ("rng_ownership", object()),
        ("requested_transitions", True),
        ("requested_transitions", 0),
        ("requested_transitions", 1_000_001),
        ("effective_transitions", -1),
        ("requested_model_evaluations", 0),
        ("effective_model_evaluations", -1),
        ("status", "succeeded"),
        ("reason_code", 1),
        ("reason_code", "Bad reason"),
        ("reason_code", "execution.api_token.failure"),
    ),
)
def test_receipt_metadata_contract_rejects_wrong_types_and_bounds(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ScheduleContractError):
        replace(_metadata(), **cast(Any, {field_name: value}))


def test_receipt_public_identity_contracts_reject_wrong_types_and_shapes() -> None:
    with pytest.raises(ScheduleContractError):
        ExecutionComponent(
            identifier=cast(Any, 1),
            version="1",
            fingerprint=_fingerprint("1"),
        )
    with pytest.raises(ScheduleContractError):
        ExecutionComponent(
            identifier="bad value",
            version="1",
            fingerprint=_fingerprint("1"),
        )
    with pytest.raises(ScheduleContractError):
        ExecutionComponent(
            identifier="model",
            version=cast(Any, 1),
            fingerprint=_fingerprint("1"),
        )
    with pytest.raises(ScheduleContractError):
        ExecutionComponent(
            identifier="model",
            version="1",
            fingerprint=cast(Any, 1),
        )


def test_portable_bundle_round_trips_and_keeps_artifact_and_receipt_distinct() -> None:
    artifact = _artifact()
    receipt = build_execution_receipt(artifact, metadata=_metadata())
    bundle = PortableExecutionBundle(artifact=artifact, receipt=receipt)
    payload = serialize_portable_execution_bundle(bundle)
    projection = _decoded(payload)

    assert projection["schema"] == "sigmax.portable-execution-bundle/1"
    assert projection["artifact"]["schema"] == "sigmax.schedule-artifact-envelope/1"
    assert projection["receipt"]["schema"] == "sigmax.execution-receipt-envelope/1"
    assert deserialize_portable_execution_bundle(payload) == bundle
    assert (
        serialize_portable_execution_bundle(deserialize_portable_execution_bundle(payload))
        == payload
    )


def test_bundle_rejects_cross_artifact_receipt_identity_mismatch() -> None:
    first = _artifact()
    construction = first.construction_projection()
    _mapping(construction["source"])["revision"] = "different"
    second = type(first)(
        construction_bytes=json.dumps(
            construction,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
        numerical_bytes=first.numerical_bytes,
        construction_fingerprint=(
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    construction,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        ),
        numerical_fingerprint=first.numerical_fingerprint,
    )
    receipt = build_execution_receipt(first, metadata=_metadata())

    with pytest.raises(ScheduleContractError, match=r"artifact|fingerprint"):
        PortableExecutionBundle(artifact=second, receipt=receipt)


def test_bundle_rejects_numerical_and_effective_input_cross_link_mismatches() -> None:
    artifact = _artifact()
    receipt = _receipt()
    fake_numerical = object.__new__(ExecutionReceipt)
    for field_name, value in (
        ("receipt_bytes", receipt.receipt_bytes),
        ("receipt_fingerprint", receipt.receipt_fingerprint),
        ("construction_fingerprint", receipt.construction_fingerprint),
        ("numerical_fingerprint", _fingerprint("9")),
    ):
        object.__setattr__(fake_numerical, field_name, value)
    with pytest.raises(ScheduleContractError, match="numerical"):
        PortableExecutionBundle(artifact=artifact, receipt=fake_numerical)

    changed = receipt.projection()
    _mapping(changed["effective_inputs"])["width"] = 2048
    changed_receipt = _receipt_from_projection(changed)
    with pytest.raises(ScheduleContractError, match="effective inputs"):
        PortableExecutionBundle(artifact=artifact, receipt=changed_receipt)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(extra=True),
        lambda value: value.update(receipt_fingerprint=_fingerprint("0")),
        lambda value: value["receipt"].update(extra=True),
        lambda value: value["receipt"]["counts"].update(effective_transitions=3),
    ),
)
def test_receipt_parser_rejects_unknown_tampered_or_stale_payloads(mutation: Any) -> None:
    decoded = _decoded(serialize_execution_receipt(_receipt()))
    mutation(decoded)
    payload = json.dumps(decoded, separators=(",", ":"), sort_keys=True).encode()

    with pytest.raises(ScheduleContractError):
        deserialize_execution_receipt(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["receipt"].update(schema="unknown"),
        lambda value: value["receipt"].update(artifact=[]),
        lambda value: value["receipt"]["artifact"].update(extra=True),
        lambda value: value["receipt"].update(effective_inputs=[]),
        lambda value: value["receipt"]["effective_inputs"].update(extra=True),
        lambda value: value["receipt"]["effective_inputs"].update(steps=0),
        lambda value: value["receipt"]["effective_inputs"].update(width=0),
        lambda value: value["receipt"]["effective_inputs"].update(width=None),
        lambda value: value["receipt"]["effective_inputs"].update(precision="float16"),
        lambda value: value["receipt"]["effective_inputs"].update(compatibility=[]),
        lambda value: value["receipt"].update(profile=[]),
        lambda value: value["receipt"]["profile"].update(extra=True),
        lambda value: value["receipt"]["profile"].update(id="other.profile"),
        lambda value: value["receipt"]["profile"].update(version="2"),
        lambda value: value["receipt"].update(counts=[]),
        lambda value: value["receipt"]["counts"].update(extra=True),
        lambda value: value["receipt"]["counts"].update(
            requested_transitions=3,
            effective_transitions=3,
        ),
        lambda value: value["receipt"].update(execution=[]),
        lambda value: value["receipt"]["execution"].update(extra=True),
        lambda value: value["receipt"]["execution"].update(status="unknown"),
        lambda value: value["receipt"].update(rng_ownership=[]),
        lambda value: value["receipt"]["rng_ownership"].update(extra=True),
        lambda value: value["receipt"]["rng_ownership"].update(schedule="unknown"),
        lambda value: value["receipt"].update(compatibility=[]),
        lambda value: value["receipt"]["compatibility"].update(considered={}),
        lambda value: value["receipt"]["compatibility"].update(level="unknown"),
        lambda value: value["receipt"].update(host=[]),
        lambda value: value["receipt"]["host"].update(extra=True),
        lambda value: value["receipt"].update(model=[]),
        lambda value: value["receipt"]["model"].update(extra=True),
    ),
)
def test_receipt_parser_rejects_structurally_invalid_projections(mutation: Any) -> None:
    decoded = _decoded(serialize_execution_receipt(_receipt()))
    mutation(decoded)
    receipt = cast(dict[str, object], decoded["receipt"])
    decoded["receipt_fingerprint"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()

    with pytest.raises(ScheduleContractError):
        deserialize_execution_receipt(_canonical(decoded))


def test_receipt_parser_accepts_optional_effective_dimensions() -> None:
    decoded = _decoded(serialize_execution_receipt(_receipt()))
    receipt = cast(dict[str, object], decoded["receipt"])
    cast(dict[str, object], receipt["effective_inputs"])["width"] = None
    cast(dict[str, object], receipt["effective_inputs"])["height"] = None
    decoded["receipt_fingerprint"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()

    restored = deserialize_execution_receipt(_canonical(decoded)).projection()
    assert _mapping(restored["effective_inputs"])["width"] is None


@pytest.mark.parametrize(
    "compatibility",
    (
        {"api_token": "redacted"},
        {"model_path": "relative"},
        {"decision": "C:\\Users\\private\\model"},
        {"decision": "/home/private/model"},
        {"decision": "x" * 4097},
        {"decision": {"bits": 0, "precision": "float64"}},
        {"decision": {"bits": "00000000", "precision": "float16"}},
        {"decision": {"bits": "00000000", "precision": "float64"}},
        {"decision": {"bits": "8000000000000000", "precision": "float64"}},
        {"decision": {"bits": "7ff0000000000000", "precision": "float64"}},
        {f"field_{index}": index for index in range(129)},
    ),
)
def test_receipt_parser_rejects_unsafe_effective_compatibility(
    compatibility: dict[str, object],
) -> None:
    decoded = _decoded(serialize_execution_receipt(_receipt()))
    receipt = cast(dict[str, object], decoded["receipt"])
    _mapping(receipt["effective_inputs"])["compatibility"] = compatibility
    decoded["receipt_fingerprint"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()

    with pytest.raises(ScheduleContractError):
        deserialize_execution_receipt(_canonical(decoded))


@pytest.mark.parametrize(
    ("bits", "precision"),
    (
        ("3f933333", "float32"),
        ("3ff2666666666666", "float64"),
    ),
)
def test_receipt_parser_accepts_typed_effective_compatibility(
    bits: str,
    precision: str,
) -> None:
    decoded = _decoded(serialize_execution_receipt(_receipt()))
    receipt = cast(dict[str, object], decoded["receipt"])
    _mapping(receipt["effective_inputs"])["compatibility"] = {
        "mu": {"bits": bits, "precision": precision}
    }
    decoded["receipt_fingerprint"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()

    restored = deserialize_execution_receipt(_canonical(decoded)).projection()
    assert _mapping(_mapping(restored["effective_inputs"])["compatibility"])["mu"] == {
        "bits": bits,
        "precision": precision,
    }


def test_receipt_parser_rejects_recomputed_success_with_rejected_compatibility() -> None:
    decoded = _decoded(serialize_execution_receipt(_receipt()))
    receipt = cast(dict[str, object], decoded["receipt"])
    _mapping(receipt["compatibility"])["level"] = "reject"
    _mapping(receipt["compatibility"])["reasons"] = ["model_family_mismatch"]
    decoded["receipt_fingerprint"] = "sha256:" + hashlib.sha256(_canonical(receipt)).hexdigest()

    with pytest.raises(ScheduleContractError, match="rejected"):
        deserialize_execution_receipt(_canonical(decoded))


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"[]",
        b'{"schema":"x","schema":"y"}',
        b'{"value":NaN}',
        b'{ "schema": "noncanonical" }',
        b"x" * 1_048_577,
        "\ufeff{}".encode(),
    ),
    ids=(
        "empty",
        "array-root",
        "duplicate",
        "non-finite",
        "noncanonical",
        "oversized",
        "bom",
    ),
)
def test_receipt_parser_rejects_untrusted_transport(payload: bytes) -> None:
    with pytest.raises(ScheduleContractError):
        deserialize_execution_receipt(payload)


def test_receipt_parser_accepts_canonical_text_and_rejects_other_input_forms() -> None:
    payload = serialize_execution_receipt(_receipt())
    assert deserialize_execution_receipt(payload.decode()) == _receipt()
    with pytest.raises(ScheduleContractError, match="BOM"):
        deserialize_execution_receipt("\ufeff{}")
    with pytest.raises(ScheduleContractError, match="Unicode"):
        deserialize_execution_receipt("\ud800")
    with pytest.raises(ScheduleContractError, match="bytes or text"):
        deserialize_execution_receipt(cast(Any, 1))
    with pytest.raises(ScheduleContractError, match="valid JSON"):
        deserialize_execution_receipt(b"{")
    with pytest.raises(ScheduleContractError, match="float"):
        deserialize_execution_receipt(b'{"value":1.5}')


def test_receipt_direct_contract_and_public_functions_reject_invalid_types() -> None:
    receipt = _receipt()
    with pytest.raises(ScheduleContractError):
        replace(receipt, receipt_bytes=cast(Any, "{}"))
    with pytest.raises(ScheduleContractError, match="construction"):
        replace(receipt, construction_fingerprint=_fingerprint("0"))
    with pytest.raises(ScheduleContractError, match="numerical"):
        replace(receipt, numerical_fingerprint=_fingerprint("0"))
    with pytest.raises(ScheduleContractError):
        replace(receipt, receipt_fingerprint="bad")
    with pytest.raises(ScheduleContractError):
        build_execution_receipt(cast(Any, object()), metadata=_metadata())
    with pytest.raises(ScheduleContractError):
        build_execution_receipt(_artifact(), metadata=cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        build_execution_receipt(
            _artifact(),
            metadata=replace(
                _metadata(),
                requested_transitions=3,
                effective_transitions=3,
            ),
        )
    with pytest.raises(ScheduleContractError):
        serialize_execution_receipt(cast(Any, object()))


def test_receipt_and_bundle_envelope_guards_fail_closed() -> None:
    receipt_envelope = _decoded(serialize_execution_receipt(_receipt()))
    receipt_envelope["schema"] = "unknown"
    with pytest.raises(ScheduleContractError, match="schema"):
        deserialize_execution_receipt(_canonical(receipt_envelope))

    bundle = PortableExecutionBundle(artifact=_artifact(), receipt=_receipt())
    projection = _decoded(serialize_portable_execution_bundle(bundle))
    projection["schema"] = "unknown"
    with pytest.raises(ScheduleContractError, match="schema"):
        deserialize_portable_execution_bundle(_canonical(projection))
    projection = _decoded(serialize_portable_execution_bundle(bundle))
    projection["artifact"] = []
    with pytest.raises(ScheduleContractError):
        deserialize_portable_execution_bundle(_canonical(projection))
    with pytest.raises(ScheduleContractError):
        PortableExecutionBundle(artifact=cast(Any, object()), receipt=_receipt())
    with pytest.raises(ScheduleContractError):
        PortableExecutionBundle(artifact=_artifact(), receipt=cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        serialize_portable_execution_bundle(cast(Any, object()))


def test_private_projection_helpers_reject_unsupported_values() -> None:
    with pytest.raises(ScheduleContractError):
        receipts_module._object([], field_name="fixture")


def test_receipt_round_trip_is_stable_across_subprocess_hash_seeds() -> None:
    payload = serialize_execution_receipt(_receipt())
    code = """
import sys
from comfyui_sigmax.core import deserialize_execution_receipt, serialize_execution_receipt
payload = bytes.fromhex(sys.argv[1])
sys.stdout.write(serialize_execution_receipt(deserialize_execution_receipt(payload)).hex())
"""
    for seed in ("1", "917"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code, payload.hex()],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert bytes.fromhex(completed.stdout) == payload
