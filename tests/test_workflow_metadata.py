"""Versioned workflow metadata and graph-preservation contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    CapabilityDimension,
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityReason,
    ExecutionStatus,
    ScheduleContractError,
    WorkflowArtifactReference,
    WorkflowHostRequirement,
    WorkflowMetadata,
    WorkflowReceiptReference,
    WorkflowRequirement,
    attach_workflow_metadata,
    deserialize_workflow_metadata,
    detach_workflow_metadata,
    extract_workflow_metadata,
    serialize_workflow_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def _fingerprint(character: str) -> str:
    return "sha256:" + character * 64


def _decision() -> CompatibilityDecision:
    return CompatibilityDecision(
        level=CompatibilityLevel.ALLOW,
        considered=tuple(CapabilityDimension),
        reasons=(CompatibilityReason.COMPATIBLE,),
    )


def _metadata(*, receipt_status: ExecutionStatus = ExecutionStatus.SUCCEEDED) -> WorkflowMetadata:
    artifact = WorkflowArtifactReference(
        construction_fingerprint=_fingerprint("1"),
        numerical_fingerprint=_fingerprint("2"),
    )
    return WorkflowMetadata(
        package=WorkflowRequirement(identifier="comfyui-sigmax", version="0.1.0.dev0"),
        nodes=(
            WorkflowRequirement(identifier="Sigmax.Krea2SigmaScheduler", version="1"),
            WorkflowRequirement(identifier="Sigmax.ScheduleInspector", version="1"),
        ),
        host=WorkflowHostRequirement(
            identifier="comfyui",
            version="0.29.0",
            api_version="legacy_v1",
        ),
        profile=WorkflowRequirement(identifier="krea2.turbo.official", version="1"),
        compatibility=_decision(),
        artifact=artifact,
        receipts=(
            WorkflowReceiptReference(
                receipt_fingerprint=_fingerprint("3"),
                construction_fingerprint=artifact.construction_fingerprint,
                numerical_fingerprint=artifact.numerical_fingerprint,
                status=receipt_status,
            ),
        ),
    )


def _legacy_workflow(*, extra: object = ...) -> dict[str, object]:
    workflow: dict[str, object] = {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": "Sigmax.Krea2SigmaScheduler",
                "pos": [100.25, 200.5],
                "properties": {"cnr_id": "comfyui-sigmax", "ver": "0.1.0.dev0"},
                "widgets_values": ["Turbo", 8],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    if extra is not ...:
        workflow["extra"] = extra
    return workflow


def _current_workflow(*, extra: object = ...) -> dict[str, object]:
    workflow: dict[str, object] = {
        "version": 1,
        "state": {
            "lastGroupId": 0,
            "lastNodeId": 1,
            "lastLinkId": 0,
            "lastRerouteId": 0,
        },
        "nodes": [
            {
                "id": 1,
                "type": "Sigmax.Krea2SigmaScheduler",
                "pos": [100.25, 200.5],
                "properties": {"cnr_id": "comfyui-sigmax", "ver": "0.1.0.dev0"},
                "widgets_values": ["Turbo", 8],
            }
        ],
        "links": [],
        "definitions": {"subgraphs": []},
    }
    if extra is not ...:
        workflow["extra"] = extra
    return workflow


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _decoded(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_metadata_projection_records_all_portable_requirements_and_references() -> None:
    metadata = _metadata()
    projection = metadata.projection()

    assert projection["schema"] == "sigmax.workflow-metadata/1"
    assert projection["requirements"] == {
        "host": {
            "api_version": "legacy_v1",
            "id": "comfyui",
            "version": "0.29.0",
        },
        "nodes": [
            {"id": "Sigmax.Krea2SigmaScheduler", "version": "1"},
            {"id": "Sigmax.ScheduleInspector", "version": "1"},
        ],
        "package": {"id": "comfyui-sigmax", "version": "0.1.0.dev0"},
    }
    assert projection["profile"] == {"id": "krea2.turbo.official", "version": "1"}
    assert cast(dict[str, object], projection["compatibility"])["level"] == "allow"
    assert projection["artifact"] == {
        "construction_fingerprint": _fingerprint("1"),
        "construction_schema": "sigmax.schedule-artifact/1",
        "numerical_fingerprint": _fingerprint("2"),
        "receipt_schema": "sigmax.execution-receipt-envelope/1",
    }
    assert projection["receipts"] == [
        {
            "construction_fingerprint": _fingerprint("1"),
            "numerical_fingerprint": _fingerprint("2"),
            "receipt_fingerprint": _fingerprint("3"),
            "schema": "sigmax.execution-receipt-envelope/1",
            "status": "succeeded",
        }
    ]


def test_metadata_transport_is_canonical_deterministic_and_immutable() -> None:
    first = _metadata()
    second = _metadata()
    payload = serialize_workflow_metadata(first)

    assert first == second
    assert deserialize_workflow_metadata(payload) == first
    assert serialize_workflow_metadata(deserialize_workflow_metadata(payload)) == payload
    assert _decoded(payload)["schema"] == "sigmax.workflow-metadata-envelope/1"
    assert _decoded(payload)["metadata_fingerprint"] == first.metadata_fingerprint
    assert first.metadata_fingerprint == (
        "sha256:" + hashlib.sha256(first.metadata_bytes).hexdigest()
    )
    with pytest.raises(FrozenInstanceError):
        first.metadata_bytes = b"{}"  # type: ignore[misc]
    projection = first.projection()
    projection.clear()
    assert first.projection()["schema"] == "sigmax.workflow-metadata/1"


@pytest.mark.parametrize("factory", (_legacy_workflow, _current_workflow))
@pytest.mark.parametrize(
    "extra",
    (
        ...,
        None,
        {},
        {"frontendVersion": "1.46.3", "ds": {"scale": 1.0, "offset": [0.0, 0.0]}},
    ),
)
def test_attach_extract_and_detach_preserve_supported_workflow_forms(
    factory: Any,
    extra: object,
) -> None:
    original = factory(extra=extra)
    snapshot = json.loads(json.dumps(original))
    attached = attach_workflow_metadata(original, _metadata())

    assert original == snapshot
    assert attached["version"] == original["version"]
    assert attached["nodes"] == original["nodes"]
    assert attached["links"] == original["links"]
    assert extract_workflow_metadata(attached) == _metadata()
    assert attach_workflow_metadata(attached, _metadata()) == attached

    detached = detach_workflow_metadata(attached)
    assert "comfyui_sigmax" not in cast(dict[str, object], detached.get("extra", {}))
    for key, value in original.items():
        if key != "extra":
            assert detached[key] == value
    if isinstance(extra, dict) and extra:
        assert detached["extra"] == extra


def test_extract_missing_metadata_returns_none_and_detach_is_idempotent() -> None:
    workflow = _legacy_workflow(extra={"frontendVersion": "1.46.3"})

    assert extract_workflow_metadata(workflow) is None
    assert detach_workflow_metadata(workflow) == workflow


@pytest.mark.parametrize("factory", (_legacy_workflow, _current_workflow))
def test_saved_workflow_json_round_trip_preserves_graph_and_metadata(factory: Any) -> None:
    original = factory(extra={"frontendVersion": "1.46.3"})
    attached = attach_workflow_metadata(original, _metadata())

    restored = json.loads(json.dumps(attached, ensure_ascii=False))
    assert extract_workflow_metadata(restored) == _metadata()
    assert detach_workflow_metadata(restored) == original


def test_current_workflow_accepts_json_numeric_version_one() -> None:
    workflow = _current_workflow()
    workflow["version"] = 1.0

    attached = attach_workflow_metadata(workflow, _metadata())
    assert attached["version"] == 1.0
    assert extract_workflow_metadata(attached) == _metadata()


def test_attach_rejects_conflicting_existing_namespace() -> None:
    attached = attach_workflow_metadata(_current_workflow(), _metadata())
    changed = _metadata(receipt_status=ExecutionStatus.FAILED)

    with pytest.raises(ScheduleContractError, match="conflict"):
        attach_workflow_metadata(attached, changed)


@pytest.mark.parametrize(
    "workflow",
    (
        [],
        {},
        {"version": True},
        {"version": 0.3},
        {"version": 2},
        {"version": "1"},
        {"version": 1, "extra": []},
        {"version": 0.4, "extra": "bad"},
    ),
)
def test_workflow_helpers_reject_unsupported_roots_versions_and_extra(workflow: object) -> None:
    with pytest.raises(ScheduleContractError):
        attach_workflow_metadata(cast(Any, workflow), _metadata())


def test_extract_and_detach_reject_malformed_namespace() -> None:
    workflow = _legacy_workflow(extra={"comfyui_sigmax": []})
    with pytest.raises(ScheduleContractError):
        extract_workflow_metadata(workflow)
    with pytest.raises(ScheduleContractError):
        detach_workflow_metadata(workflow)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: WorkflowRequirement(identifier="api_token", version="1"),
        lambda: WorkflowRequirement(identifier="C:\\Users\\private\\node", version="1"),
        lambda: WorkflowRequirement(identifier="node", version="/home/private/version"),
        lambda: WorkflowRequirement(identifier=cast(Any, 1), version="1"),
        lambda: WorkflowHostRequirement(
            identifier="comfyui",
            version="1",
            api_version="bad value",
        ),
        lambda: WorkflowArtifactReference(
            construction_fingerprint="bad",
            numerical_fingerprint=_fingerprint("2"),
        ),
        lambda: WorkflowReceiptReference(
            receipt_fingerprint=_fingerprint("3"),
            construction_fingerprint=_fingerprint("1"),
            numerical_fingerprint=_fingerprint("2"),
            status=cast(Any, "succeeded"),
        ),
    ),
)
def test_metadata_components_reject_invalid_secret_private_or_typed_values(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


def test_metadata_rejects_duplicate_requirements_and_receipts() -> None:
    metadata = _metadata()
    with pytest.raises(ScheduleContractError, match="duplicate"):
        replace(metadata, nodes=(metadata.nodes[0], metadata.nodes[0]))
    with pytest.raises(ScheduleContractError, match="duplicate"):
        replace(metadata, receipts=(metadata.receipts[0], metadata.receipts[0]))


def test_metadata_rejects_receipt_artifact_cross_link_mismatch() -> None:
    metadata = _metadata()
    stale = replace(
        metadata.receipts[0],
        construction_fingerprint=_fingerprint("9"),
    )
    with pytest.raises(ScheduleContractError, match="artifact"):
        replace(metadata, receipts=(stale,))


def test_metadata_rejects_wrong_component_types_and_unsorted_input_is_canonicalized() -> None:
    metadata = _metadata()
    reversed_nodes = tuple(reversed(metadata.nodes))
    rebuilt = replace(metadata, nodes=reversed_nodes)
    requirements = cast(dict[str, object], rebuilt.projection()["requirements"])
    assert [item["id"] for item in cast(list[dict[str, str]], requirements["nodes"])] == [
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.ScheduleInspector",
    ]
    for field_name in ("package", "host", "profile", "compatibility", "artifact"):
        with pytest.raises(ScheduleContractError):
            replace(metadata, **cast(Any, {field_name: object()}))
    with pytest.raises(ScheduleContractError):
        replace(metadata, nodes=cast(Any, []))
    with pytest.raises(ScheduleContractError):
        replace(metadata, receipts=cast(Any, []))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(extra=True),
        lambda value: value["metadata"].update(extra=True),
        lambda value: value["metadata"].update(schema="unknown"),
        lambda value: value["metadata"].update(requirements=[]),
        lambda value: value["metadata"]["requirements"].update(extra=True),
        lambda value: value["metadata"]["requirements"].update(nodes=[]),
        lambda value: value["metadata"].update(profile=[]),
        lambda value: value["metadata"].update(compatibility=[]),
        lambda value: value["metadata"]["compatibility"].update(considered={}),
        lambda value: value["metadata"]["compatibility"].update(level="unknown"),
        lambda value: value["metadata"].update(artifact=[]),
        lambda value: value["metadata"]["artifact"].update(extra=True),
        lambda value: value["metadata"]["artifact"].update(construction_schema="unknown"),
        lambda value: value["metadata"].update(receipts={}),
        lambda value: value["metadata"]["receipts"][0].update(schema="unknown"),
        lambda value: value["metadata"]["receipts"][0].update(status="unknown"),
        lambda value: value["metadata"]["receipts"][0].update(
            construction_fingerprint=_fingerprint("9")
        ),
    ),
)
def test_metadata_parser_rejects_unknown_tampered_or_stale_payloads(mutation: Any) -> None:
    decoded = _decoded(serialize_workflow_metadata(_metadata()))
    mutation(decoded)
    metadata = cast(dict[str, object], decoded["metadata"])
    decoded["metadata_fingerprint"] = "sha256:" + hashlib.sha256(_canonical(metadata)).hexdigest()

    with pytest.raises(ScheduleContractError):
        deserialize_workflow_metadata(_canonical(decoded))


def test_metadata_parser_rejects_stale_fingerprint() -> None:
    decoded = _decoded(serialize_workflow_metadata(_metadata()))
    decoded["metadata_fingerprint"] = _fingerprint("0")

    with pytest.raises(ScheduleContractError, match="fingerprint"):
        deserialize_workflow_metadata(_canonical(decoded))


def test_metadata_parser_rejects_noncanonical_requirement_order() -> None:
    decoded = _decoded(serialize_workflow_metadata(_metadata()))
    metadata = cast(dict[str, Any], decoded["metadata"])
    metadata["requirements"]["nodes"].reverse()
    decoded["metadata_fingerprint"] = "sha256:" + hashlib.sha256(_canonical(metadata)).hexdigest()

    with pytest.raises(ScheduleContractError, match="ordering"):
        deserialize_workflow_metadata(_canonical(decoded))


def test_metadata_parser_rejects_unsupported_envelope_schema() -> None:
    decoded = _decoded(serialize_workflow_metadata(_metadata()))
    decoded["schema"] = "unknown"
    with pytest.raises(ScheduleContractError, match="envelope schema"):
        deserialize_workflow_metadata(_canonical(decoded))


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"[]",
        b'{"schema":"x","schema":"y"}',
        b'{"value":NaN}',
        b'{"value":1.5}',
        b'{ "schema": "noncanonical" }',
        b"x" * 1_048_577,
        "\ufeff{}".encode(),
    ),
    ids=(
        "empty",
        "array-root",
        "duplicate",
        "non-finite",
        "untyped-float",
        "noncanonical",
        "oversized",
        "bom",
    ),
)
def test_metadata_parser_rejects_untrusted_transport(payload: bytes) -> None:
    with pytest.raises(ScheduleContractError):
        deserialize_workflow_metadata(payload)


def test_metadata_parser_accepts_canonical_text_and_rejects_other_input_types() -> None:
    payload = serialize_workflow_metadata(_metadata())
    assert deserialize_workflow_metadata(payload.decode()) == _metadata()
    with pytest.raises(ScheduleContractError, match="BOM"):
        deserialize_workflow_metadata("\ufeff{}")
    with pytest.raises(ScheduleContractError):
        deserialize_workflow_metadata(cast(Any, 1))
    with pytest.raises(ScheduleContractError):
        deserialize_workflow_metadata("\ud800")
    with pytest.raises(ScheduleContractError, match="valid JSON"):
        deserialize_workflow_metadata(b"{")


def test_public_functions_reject_invalid_types() -> None:
    with pytest.raises(ScheduleContractError):
        serialize_workflow_metadata(cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        attach_workflow_metadata(_legacy_workflow(), cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        extract_workflow_metadata(cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        detach_workflow_metadata(cast(Any, object()))


def test_metadata_round_trip_is_stable_across_subprocess_hash_seeds() -> None:
    payload = serialize_workflow_metadata(_metadata())
    code = """
import sys
from comfyui_sigmax.core import deserialize_workflow_metadata, serialize_workflow_metadata
payload = bytes.fromhex(sys.argv[1])
sys.stdout.write(serialize_workflow_metadata(deserialize_workflow_metadata(payload)).hex())
"""
    for seed in ("1", "917"):
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code, payload.hex()],
            check=True,
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed},
        )
        assert bytes.fromhex(completed.stdout) == payload


def test_metadata_matches_committed_golden_projection_and_fingerprint() -> None:
    fixture_root = ROOT / "tests" / "fixtures" / "workflows"
    expected_projection = json.loads(
        (fixture_root / "workflow_metadata_projection_v1.json").read_text(encoding="utf-8")
    )
    expected_hash = json.loads(
        (fixture_root / "workflow_metadata_hashes_v1.json").read_text(encoding="utf-8")
    )

    assert _metadata().projection() == expected_projection
    assert _metadata().metadata_fingerprint == expected_hash["metadata_fingerprint"]
