from __future__ import annotations

import hashlib
import importlib.resources
import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from comfyui_sigmax.coinstallation_matrix import (
    CoInstallationMatrixError,
    load_coinstallation_mutation_matrix,
)
from scripts import generate_coinstallation_mutation_matrix as generator


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
        importlib.resources.files("comfyui_sigmax.coinstallation")
        .joinpath("matrix_v1.json")
        .read_bytes()
    )
    return cast(dict[str, Any], json.loads(payload))


def _rehashed(envelope: dict[str, Any]) -> bytes:
    envelope["matrix_fingerprint"] = (
        "sha256:" + hashlib.sha256(_canonical(envelope["matrix"])).hexdigest()
    )
    return _canonical(envelope) + b"\n"


def _duplicate_row_id(envelope: dict[str, Any]) -> None:
    row = envelope["matrix"]["rows"][1]
    row["id"] = envelope["matrix"]["rows"][0]["id"]
    evidence_projection = {
        key: value
        for key, value in row.items()
        if key not in {"evidence_source", "result_fingerprint"}
    }
    row["result_fingerprint"] = (
        "sha256:" + hashlib.sha256(_canonical(evidence_projection)).hexdigest()
    )


def test_packaged_coinstallation_matrix_is_strict_and_complete() -> None:
    matrix = load_coinstallation_mutation_matrix()
    projection = matrix.projection()
    rows = cast(list[dict[str, Any]], projection["rows"])

    assert matrix.schema == "sigmax.co-installation-mutation-matrix/1"
    assert matrix.matrix_fingerprint.startswith("sha256:")
    assert len(rows) == 10
    assert all(row["status"] == "passed" for row in rows)
    assert all(row["first_attempt"] == row["repeat"] == "passed" for row in rows)
    assert projection["policy"] == {
        "external_reference_code_executed": False,
        "protected_existing_identities": True,
        "third_party_claims": False,
    }


def test_matrix_covers_every_required_mutation_seam() -> None:
    rows = {
        row["id"]: row
        for row in cast(
            list[dict[str, Any]],
            load_coinstallation_mutation_matrix().projection()["rows"],
        )
    }

    assert rows["clean-install"]["observed_verdict"] == "allow"
    assert rows["idempotent-reload"]["observed_verdict"] == "allow"
    assert rows["unrelated-node-addition"]["observed_verdict"] == "allow"
    assert rows["unrelated-scheduler-addition"]["observed_verdict"] == "allow"
    assert rows["node-id-collision"]["observed_findings"] == ["node_registry_collision"]
    assert rows["sigmax-namespace-hijack"]["observed_findings"] == ["sigmax_namespace_hijack"]
    assert rows["scheduler-overwrite"]["observed_findings"] == ["scheduler_registry_overwrite"]
    assert rows["torch-call-replacement"]["observed_findings"] == ["torch_call_path_changed"]
    assert rows["model-patch-mutation"]["observed_findings"] == ["model_patch_state_changed"]
    assert rows["double-shift"]["observed_findings"] == [
        "construction_shift_repeated",
        "model_native_external_double_shift",
    ]


def test_generator_matches_packaged_resource() -> None:
    expected = generator._canonical(generator.build_envelope()) + b"\n"
    actual = (
        importlib.resources.files("comfyui_sigmax.coinstallation")
        .joinpath("matrix_v1.json")
        .read_bytes()
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda envelope: envelope["matrix"]["policy"].update(
                external_reference_code_executed=True
            ),
            "external reference code",
        ),
        (
            lambda envelope: envelope["matrix"]["rows"][0].update(status="failed"),
            "passed rows",
        ),
        (
            lambda envelope: envelope["matrix"]["rows"][0].update(repeat="failed"),
            "first/repeat",
        ),
        (
            lambda envelope: envelope["matrix"]["rows"][0].update(
                evidence_source="C:\\private\\evidence.json"
            ),
            "private or absolute path",
        ),
        (
            _duplicate_row_id,
            "unique and sorted",
        ),
        (
            lambda envelope: envelope["matrix"]["rows"][4].update(observed_findings=[]),
            "expectation",
        ),
        (
            lambda envelope: envelope["matrix"]["rows"][0].update(pack_id="synthetic.changed"),
            "result fingerprint drifted",
        ),
        (
            lambda envelope: envelope["matrix"]["rows"][0].update(
                evidence_source="tests/coinstallation/fixtures/undeclared.json"
            ),
            "evidence source is undeclared",
        ),
        (
            lambda envelope: envelope["matrix"]["context"].update(
                dependency_compatibility_matrix_fingerprint="sha256:" + "0" * 64
            ),
            "dependency compatibility matrix fingerprint drifted",
        ),
    ],
)
def test_semantically_invalid_rehashed_matrix_is_rejected(
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    envelope = _envelope()
    mutation(envelope)

    with pytest.raises(CoInstallationMatrixError, match=message):
        load_coinstallation_mutation_matrix(_rehashed(envelope))


def test_noncanonical_transport_is_rejected() -> None:
    payload = json.dumps(_envelope(), ensure_ascii=False).encode("utf-8")

    with pytest.raises(CoInstallationMatrixError, match="canonical JSON"):
        load_coinstallation_mutation_matrix(payload)
