"""Versioned schedule artifact build, transport, and strict parsing tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    ArtifactBuildMetadata,
    ArtifactField,
    BaseGridSpec,
    EvidenceLevel,
    OverrideRecord,
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
    build_schedule_artifact,
    canonical_projection_bytes,
    construction_fingerprint,
    deserialize_schedule_artifact,
    serialize_schedule_artifact,
)


def _result(*, slicing: SliceSpec | None = None) -> ScheduleResult:
    override = OverrideRecord(
        field="steps",
        requested_value="5",
        effective_value="4",
        reason="fixture execution budget",
    )
    request = ScheduleRequest(
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        requested_inputs=ScheduleInputs(steps=5, width=1024, height=1024),
        sigma_domain=SigmaDomain.UNIT_FLOW,
        provenance=Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.EXPERIMENTAL,
            source="Kréa 臺灣 fixture",
            source_revision="abc123",
            profile_id="fixture.power-of-two",
            profile_version="1",
        ),
        base_grid=BaseGridSpec(
            identifier="fixture.power-of-two",
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        transforms=(
            TransformContract(
                name="exponential_mu",
                stage=TransformStage.PRIMARY_TIME_SHIFT,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
            ),
        ),
        terminal_policy=TerminalPolicy.APPEND_ZERO,
        slicing=slicing or SliceSpec(start_step=0, end_step=4, denoise=1.0),
        overrides=(override,),
    )
    return ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=4, width=1024, height=1024),
        sigmas=(1.0, 0.75, 0.5, 0.25, 0.0),
        final_domain=SigmaDomain.UNIT_FLOW,
        warnings=("requested steps were overridden",),
    )


def _metadata() -> ArtifactBuildMetadata:
    return ArtifactBuildMetadata(
        source_id="fixture.schedule-artifact-v1",
        source_label="Kre\u0301a 臺灣 fixture",
        base_grid_parameters=(
            ArtifactField(name="count", value=4),
            ArtifactField(
                name="start",
                value=TypedArtifactValue(value=1.0, precision="float64"),
            ),
        ),
        transform_parameters=(
            (
                ArtifactField(
                    name="mu",
                    value=TypedArtifactValue(value=1.15, precision="float64"),
                ),
            ),
        ),
        compatibility=(
            ArtifactField(name="decision", value="allow"),
            ArtifactField(name="known_good", value=True),
            ArtifactField(name="reason_code", value="fixture_compatible"),
        ),
    )


def _artifact() -> ScheduleArtifact:
    return build_schedule_artifact(
        _result(),
        metadata=_metadata(),
        precision="float64",
    )


def _sha256_identity(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _artifact_from_projections(
    construction: dict[str, object],
    numerical: dict[str, object],
) -> ScheduleArtifact:
    numerical_bytes = canonical_projection_bytes(numerical)
    numerical_identity = _sha256_identity(numerical_bytes)
    construction["numerical_fingerprint"] = numerical_identity
    construction_bytes = canonical_projection_bytes(construction)
    return ScheduleArtifact(
        construction_bytes=construction_bytes,
        numerical_bytes=numerical_bytes,
        construction_fingerprint=construction_fingerprint(construction),
        numerical_fingerprint=numerical_identity,
    )


def _artifact_from_untrusted_construction(
    construction: dict[str, object],
    numerical: dict[str, object] | None = None,
) -> ScheduleArtifact:
    numerical = numerical or _artifact().numerical_projection()
    numerical_bytes = canonical_projection_bytes(numerical)
    numerical_identity = _sha256_identity(numerical_bytes)
    construction["numerical_fingerprint"] = numerical_identity
    construction_bytes = canonical_projection_bytes(construction)
    return ScheduleArtifact(
        construction_bytes=construction_bytes,
        numerical_bytes=numerical_bytes,
        construction_fingerprint=_sha256_identity(construction_bytes),
        numerical_fingerprint=numerical_identity,
    )


def _set_nested(
    projection: dict[str, object],
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    cursor: object = projection
    for part in path[:-1]:
        if isinstance(part, int):
            cursor = cast(list[object], cursor)[part]
        else:
            cursor = cast(dict[str, object], cursor)[part]
    final = path[-1]
    if isinstance(final, int):
        cast(list[object], cursor)[final] = replacement
    else:
        cast(dict[str, object], cursor)[final] = replacement


def test_build_records_complete_versioned_construction_metadata() -> None:
    artifact = _artifact()
    construction = artifact.construction_projection()
    numerical = artifact.numerical_projection()

    assert construction["schema"] == "sigmax.schedule-artifact/1"
    assert numerical["schema"] == "sigmax.numerical-schedule/1"
    assert construction["numerical_fingerprint"] == artifact.numerical_fingerprint
    assert construction["engine"] == {
        "name": "comfyui-sigmax",
        "version": "0.1.0.dev0",
    }
    assert construction["source"] == {
        "id": "fixture.schedule-artifact-v1",
        "label": "Kréa 臺灣 fixture",
        "revision": "abc123",
    }
    assert construction["evidence"] == {
        "level": "experimental",
        "reference": "abc123",
    }
    assert construction["requested"] == {
        "height": 1024,
        "precision": "float64",
        "profile": "fixture.power-of-two",
        "profile_version": "1",
        "steps": 5,
        "width": 1024,
    }
    assert construction["effective"] == {
        "compatibility": {
            "decision": "allow",
            "known_good": True,
            "reason_code": "fixture_compatible",
        },
        "height": 1024,
        "precision": "float64",
        "profile": "fixture.power-of-two",
        "profile_version": "1",
        "steps": 4,
        "width": 1024,
    }
    assert construction["overrides"] == [
        {
            "effective": "4",
            "path": "steps",
            "reason": "fixture execution budget",
            "requested": "5",
        }
    ]
    assert construction["ownership"] == {
        "schedule": "external_sigmas",
        "shift": "construction_pipeline",
    }
    assert construction["base_grid"] == {
        "id": "fixture.power-of-two",
        "parameters": {
            "count": 4,
            "start": {"bits": "3ff0000000000000", "precision": "float64"},
        },
    }
    assert construction["transforms"] == [
        {
            "from_domain": "unit_flow",
            "id": "exponential_mu",
            "parameters": {
                "mu": {"bits": "3ff2666666666666", "precision": "float64"},
            },
            "stage": 0,
            "stage_kind": "primary_time_shift",
            "to_domain": "unit_flow",
        }
    ]
    assert construction["terminal"] == {
        "policy": "append_zero",
        "value": {"bits": "0000000000000000", "precision": "float64"},
    }
    assert construction["slicing"] == {
        "denoise": {"bits": "3ff0000000000000", "precision": "float64"},
        "end_step": 4,
        "policy": "manual_range",
        "start_step": 0,
    }
    assert construction["warnings"] == ["requested steps were overridden"]


def test_artifact_is_immutable_and_projection_access_returns_fresh_copies() -> None:
    artifact = _artifact()
    first = artifact.construction_projection()
    first["warnings"] = []

    assert artifact.construction_projection()["warnings"] == ["requested steps were overridden"]
    with pytest.raises(FrozenInstanceError):
        artifact.numerical_fingerprint = "changed"  # type: ignore[misc]


def test_transport_round_trip_is_exact_canonical_utf8() -> None:
    artifact = _artifact()
    encoded = serialize_schedule_artifact(artifact)
    restored = deserialize_schedule_artifact(encoded)

    assert restored == artifact
    assert serialize_schedule_artifact(restored) == encoded
    assert not encoded.startswith(b"\xef\xbb\xbf")
    assert not encoded.endswith(b"\n")
    assert (
        json.dumps(
            json.loads(encoded),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        == encoded
    )


def test_transport_envelope_contains_both_verified_identities() -> None:
    artifact = _artifact()
    envelope = json.loads(serialize_schedule_artifact(artifact))

    assert envelope["schema"] == "sigmax.schedule-artifact-envelope/1"
    assert envelope["construction_fingerprint"] == artifact.construction_fingerprint
    assert envelope["numerical_fingerprint"] == artifact.numerical_fingerprint
    assert envelope["construction"]["numerical_fingerprint"] == artifact.numerical_fingerprint


def test_float32_artifact_round_trip_uses_float32_tokens() -> None:
    artifact = build_schedule_artifact(
        _result(),
        metadata=_metadata(),
        precision="float32",
    )
    numerical = artifact.numerical_projection()

    assert numerical["precision"] == "float32"
    assert all(len(token) == 8 for token in cast(list[str], numerical["sigmas"]))
    assert deserialize_schedule_artifact(serialize_schedule_artifact(artifact)) == artifact


@pytest.mark.parametrize(
    ("slicing", "expected_policy"),
    [
        (SliceSpec(), "full"),
        (SliceSpec(start_step=0, end_step=4), "manual_range"),
        (SliceSpec(denoise=0.5), "denoise_tail"),
    ],
)
def test_build_records_each_slice_policy(
    slicing: SliceSpec,
    expected_policy: str,
) -> None:
    construction = build_schedule_artifact(
        _result(slicing=slicing),
        metadata=_metadata(),
        precision="float64",
    ).construction_projection()

    assert cast(dict[str, object], construction["slicing"])["policy"] == expected_policy


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xef\xbb\xbf{}", "BOM"),
        (b"\xff", "UTF-8"),
        (b'{"schema":1.5}', "floating"),
        (b'{"schema":NaN}', "non-standard"),
        (b'{"schema":"a","schema":"b"}', "duplicate"),
        (b"[]", "object"),
    ],
)
def test_strict_parser_rejects_invalid_json_transport(payload: bytes, message: str) -> None:
    with pytest.raises(ScheduleContractError, match=message):
        deserialize_schedule_artifact(payload)


def test_strict_parser_accepts_canonical_text_transport() -> None:
    payload = serialize_schedule_artifact(_artifact()).decode("utf-8")

    assert deserialize_schedule_artifact(payload) == _artifact()


def test_strict_parser_rejects_text_bom_invalid_type_and_invalid_json() -> None:
    with pytest.raises(ScheduleContractError, match="BOM"):
        deserialize_schedule_artifact("\ufeff{}")
    with pytest.raises(ScheduleContractError, match="bytes or str"):
        deserialize_schedule_artifact(cast(Any, object()))
    with pytest.raises(ScheduleContractError, match="valid JSON"):
        deserialize_schedule_artifact(b"{")


def test_parser_rejects_noncanonical_but_parseable_transport() -> None:
    envelope = json.loads(serialize_schedule_artifact(_artifact()))
    pretty = json.dumps(envelope, ensure_ascii=False, indent=2).encode()

    with pytest.raises(ScheduleContractError, match="canonical"):
        deserialize_schedule_artifact(pretty)


@pytest.mark.parametrize("kind", ["numerical", "construction", "embedded"])
def test_parser_rejects_tampered_payloads_and_stale_fingerprints(kind: str) -> None:
    envelope = json.loads(serialize_schedule_artifact(_artifact()))
    if kind == "numerical":
        envelope["numerical"]["sigmas"][1] = "3fe4000000000000"
    elif kind == "construction":
        envelope["construction"]["warnings"].append("tampered")
    else:
        envelope["construction"]["numerical_fingerprint"] = "sha256:" + ("0" * 64)
    payload = canonical_projection_bytes(envelope)

    with pytest.raises(ScheduleContractError, match="fingerprint"):
        deserialize_schedule_artifact(payload)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(schema="unknown/1"),
        lambda value: value.update(unexpected=None),
    ],
)
def test_parser_rejects_unknown_envelope_schema_or_member(mutator: Any) -> None:
    envelope = json.loads(serialize_schedule_artifact(_artifact()))
    mutator(envelope)
    payload = canonical_projection_bytes(envelope)

    with pytest.raises(ScheduleContractError, match="envelope"):
        deserialize_schedule_artifact(payload)


@pytest.mark.parametrize("field", ["construction", "numerical"])
def test_parser_rejects_nonobject_projection(field: str) -> None:
    envelope = json.loads(serialize_schedule_artifact(_artifact()))
    envelope[field] = []

    with pytest.raises(ScheduleContractError, match="projections must be objects"):
        deserialize_schedule_artifact(canonical_projection_bytes(envelope))


@pytest.mark.parametrize("field", ["construction_fingerprint", "numerical_fingerprint"])
def test_parser_rejects_nonstring_fingerprint(field: str) -> None:
    envelope = json.loads(serialize_schedule_artifact(_artifact()))
    envelope[field] = None

    with pytest.raises(ScheduleContractError, match="fingerprints must be strings"):
        deserialize_schedule_artifact(canonical_projection_bytes(envelope))


def test_parser_rejects_oversized_input_before_json_decode() -> None:
    with pytest.raises(ScheduleContractError, match="size"):
        deserialize_schedule_artifact(b" " * 1_048_577)


def test_build_rejects_non_external_schedule_ownership() -> None:
    request = ScheduleRequest(
        ownership=ScheduleOwnership.MODEL_NATIVE,
        requested_inputs=ScheduleInputs(steps=1),
        sigma_domain=SigmaDomain.MODEL_NATIVE,
        provenance=Provenance(
            engine_version="0.1.0.dev0",
            evidence=EvidenceLevel.EXPERIMENTAL,
            source="fixture",
        ),
    )
    result = ScheduleResult(
        request=request,
        effective_inputs=ScheduleInputs(steps=1),
        sigmas=(1.0, 0.0),
        final_domain=SigmaDomain.MODEL_NATIVE,
    )

    with pytest.raises(ScheduleContractError, match="EXTERNAL_SIGMAS"):
        build_schedule_artifact(result, metadata=_metadata(), precision="float64")


def test_build_rejects_transform_parameter_count_mismatch() -> None:
    metadata = ArtifactBuildMetadata(
        source_id="fixture.schedule-artifact-v1",
        source_label="fixture",
        transform_parameters=(),
    )

    with pytest.raises(ScheduleContractError, match="transform parameter"):
        build_schedule_artifact(_result(), metadata=metadata, precision="float64")


def test_serialize_rejects_nonartifact_input() -> None:
    with pytest.raises(ScheduleContractError, match="ScheduleArtifact"):
        serialize_schedule_artifact(cast(Any, object()))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ArtifactField(name="UpperCase", value="value"),
        lambda: ArtifactField(name="value", value=""),
        lambda: ArtifactField(name="value", value=cast(Any, object())),
        lambda: ArtifactBuildMetadata(source_id="", source_label="fixture"),
        lambda: ArtifactBuildMetadata(source_id="fixture", source_label=""),
        lambda: ArtifactBuildMetadata(
            source_id="fixture",
            source_label="fixture",
            engine_name="",
        ),
        lambda: ArtifactBuildMetadata(
            source_id="fixture",
            source_label="fixture",
            base_grid_parameters=cast(Any, []),
        ),
        lambda: ArtifactBuildMetadata(
            source_id="fixture",
            source_label="fixture",
            base_grid_parameters=cast(Any, ("not-a-field",)),
        ),
        lambda: ArtifactBuildMetadata(
            source_id="fixture",
            source_label="fixture",
            transform_parameters=cast(Any, []),
        ),
        lambda: ArtifactBuildMetadata(
            source_id="fixture",
            source_label="fixture",
            transform_parameters=(cast(Any, ("not-a-field",)),),
        ),
    ],
)
def test_metadata_contract_rejects_invalid_values(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ArtifactField(name="api_token", value="redacted"),
        lambda: ArtifactField(name="model_path", value="redacted"),
        lambda: ArtifactField(name="source", value=r"B:\private\model.safetensors"),
        lambda: ArtifactField(name="source", value="/home/user/private/model"),
    ],
)
def test_metadata_fields_reject_secret_names_and_private_paths(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


def test_metadata_groups_reject_duplicate_field_names() -> None:
    with pytest.raises(ScheduleContractError, match="duplicate"):
        ArtifactBuildMetadata(
            source_id="fixture.schedule-artifact-v1",
            source_label="fixture",
            compatibility=(
                ArtifactField(name="decision", value="allow"),
                ArtifactField(name="decision", value="reject"),
            ),
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.pop("domain"), "fields"),
        (lambda value: value.update(schema="unknown/1"), "schema"),
        (lambda value: value.update(precision="float16"), "precision"),
        (lambda value: value.update(domain="model_native"), "domain"),
        (lambda value: value.update(sigmas=["3ff0000000000000"]), "array"),
        (lambda value: value.update(sigmas="not-an-array"), "array"),
        (
            lambda value: cast(list[object], value["sigmas"]).__setitem__(0, 1),
            "invalid sigma token",
        ),
        (
            lambda value: cast(list[object], value["sigmas"]).__setitem__(0, "3FF0000000000000"),
            "invalid sigma token",
        ),
        (
            lambda value: cast(list[object], value["sigmas"]).__setitem__(0, "3ff000000000000"),
            "invalid sigma token",
        ),
        (
            lambda value: cast(list[object], value["sigmas"]).__setitem__(0, "zzzzzzzzzzzzzzzz"),
            "invalid sigma token",
        ),
    ],
)
def test_artifact_rejects_malformed_numerical_projection(
    mutator: Any,
    message: str,
) -> None:
    construction = _artifact().construction_projection()
    numerical = _artifact().numerical_projection()
    mutator(numerical)

    with pytest.raises(ScheduleContractError, match=message):
        _artifact_from_projections(construction, numerical)


def test_artifact_rejects_nonobject_and_noncanonical_numerical_projection() -> None:
    artifact = _artifact()
    with pytest.raises(ScheduleContractError, match="projection must be an object"):
        ScheduleArtifact(
            construction_bytes=artifact.construction_bytes,
            numerical_bytes=b"[]",
            construction_fingerprint=artifact.construction_fingerprint,
            numerical_fingerprint=artifact.numerical_fingerprint,
        )

    pretty = json.dumps(artifact.numerical_projection(), indent=2).encode()
    with pytest.raises(ScheduleContractError, match="not canonical"):
        ScheduleArtifact(
            construction_bytes=artifact.construction_bytes,
            numerical_bytes=pretty,
            construction_fingerprint=artifact.construction_fingerprint,
            numerical_fingerprint=_sha256_identity(pretty),
        )


def test_artifact_rejects_invalid_storage_and_fingerprint_shapes() -> None:
    artifact = _artifact()
    with pytest.raises(ScheduleContractError, match="canonical bytes"):
        ScheduleArtifact(
            construction_bytes=cast(Any, "not-bytes"),
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint=artifact.construction_fingerprint,
            numerical_fingerprint=artifact.numerical_fingerprint,
        )
    with pytest.raises(ScheduleContractError, match="construction fingerprint"):
        ScheduleArtifact(
            construction_bytes=artifact.construction_bytes,
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint="invalid",
            numerical_fingerprint=artifact.numerical_fingerprint,
        )
    with pytest.raises(ScheduleContractError, match="numerical fingerprint"):
        ScheduleArtifact(
            construction_bytes=artifact.construction_bytes,
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint=artifact.construction_fingerprint,
            numerical_fingerprint="invalid",
        )


def test_artifact_rejects_stale_numerical_identity() -> None:
    artifact = _artifact()
    with pytest.raises(ScheduleContractError, match="numerical fingerprint mismatch"):
        ScheduleArtifact(
            construction_bytes=artifact.construction_bytes,
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint=artifact.construction_fingerprint,
            numerical_fingerprint="sha256:" + ("0" * 64),
        )


def test_artifact_rejects_invalid_construction_projection_variants() -> None:
    artifact = _artifact()
    numerical_identity = artifact.numerical_fingerprint
    valid_identity = "sha256:" + ("0" * 64)

    construction = artifact.construction_projection()
    construction["schema"] = "unknown/1"
    with pytest.raises(ScheduleContractError, match="construction projection schema"):
        ScheduleArtifact(
            construction_bytes=canonical_projection_bytes(construction),
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint=valid_identity,
            numerical_fingerprint=numerical_identity,
        )

    construction = artifact.construction_projection()
    construction["numerical_fingerprint"] = valid_identity
    with pytest.raises(ScheduleContractError, match="embedded numerical fingerprint"):
        ScheduleArtifact(
            construction_bytes=canonical_projection_bytes(construction),
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint=valid_identity,
            numerical_fingerprint=numerical_identity,
        )

    pretty = json.dumps(artifact.construction_projection(), indent=2).encode()
    with pytest.raises(ScheduleContractError, match="construction projection bytes"):
        ScheduleArtifact(
            construction_bytes=pretty,
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint=_sha256_identity(pretty),
            numerical_fingerprint=numerical_identity,
        )

    with pytest.raises(ScheduleContractError, match="construction fingerprint mismatch"):
        ScheduleArtifact(
            construction_bytes=artifact.construction_bytes,
            numerical_bytes=artifact.numerical_bytes,
            construction_fingerprint=valid_identity,
            numerical_fingerprint=numerical_identity,
        )


def test_artifact_rejects_rehashed_but_semantically_invalid_construction() -> None:
    construction = _artifact().construction_projection()
    numerical = _artifact().numerical_projection()
    transforms = cast(list[dict[str, object]], construction["transforms"])
    transforms[0]["stage"] = 7

    with pytest.raises(ScheduleContractError, match="transform stage"):
        _artifact_from_projections(construction, numerical)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("engine",), [], "engine fields"),
        (("engine", "unexpected"), "value", "engine fields"),
        (("engine", "name"), 7, "must be a string"),
        (("source", "id"), r"copied from B:\private\model", "private local path"),
        (("evidence", "level"), "unknown", "evidence.level"),
        (("ownership", "shift"), "model_native", "ownership"),
        (("requested", "steps"), 0, "positive integer"),
        (("requested", "width"), None, "dimensions"),
        (("requested", "precision"), "float16", "precision"),
        (("requested", "profile_version"), None, "profile identity"),
        (("effective", "precision"), "float32", "precision does not match"),
        (("effective", "steps"), 3, "effective steps"),
        (("effective", "compatibility"), [], "must be an object"),
        (("effective", "compatibility", "unexpected"), [], "unsupported metadata"),
        (("overrides",), {}, "overrides must be an array"),
        (("overrides", 0), {}, r"overrides\[0\] fields"),
        (("overrides", 0, "reason"), None, "must be a string"),
        (("overrides", 0, "path"), "api_token", "secret-like"),
        (("overrides", 0, "path"), "model_path", "path-like"),
        (("base_grid", "parameters"), [], "must be an object"),
        (("transforms",), {}, "transforms must be an array"),
        (("transforms", 0, "stage_kind"), "unknown", "kind or domain"),
        (("transforms", 0, "to_domain"), "continuous_edm", "final domain"),
        (("terminal", "policy"), "unknown", "terminal policy"),
        (("terminal", "value", "precision"), "float32", "precision does not match"),
        (("terminal", "value", "bits"), "0000000000000001", "terminal value"),
        (("slicing", "denoise", "bits"), "3ff8000000000000", "between zero and one"),
        (("slicing", "start_step"), -1, "bounds are invalid"),
        (("slicing", "end_step"), 3, "bounds do not match"),
        (("slicing", "policy"), "unknown", "slicing policy"),
        (("warnings",), {}, "warnings must be an array"),
        (("warnings", 0), 7, "must be a string"),
    ],
)
def test_artifact_rejects_invalid_nested_construction_fields(
    path: tuple[str | int, ...],
    replacement: object,
    message: str,
) -> None:
    construction = _artifact().construction_projection()
    if len(path) == 2 and path[1] == "unexpected":
        cast(dict[str, object], construction[cast(str, path[0])])["unexpected"] = replacement
    elif len(path) == 3 and path[2] == "unexpected":
        parent = cast(dict[str, object], construction[cast(str, path[0])])
        cast(dict[str, object], parent[cast(str, path[1])])["unexpected"] = replacement
    else:
        _set_nested(construction, path, replacement)

    with pytest.raises(ScheduleContractError, match=message):
        _artifact_from_untrusted_construction(construction)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"bits": "0000000000000000"}, "fields"),
        ({"bits": "0000000000000000", "precision": "float16"}, "precision"),
        ({"bits": 0, "precision": "float64"}, "invalid float token"),
        ({"bits": "000000000000000", "precision": "float64"}, "invalid float token"),
        ({"bits": "AAAAAAAAAAAAAAAA", "precision": "float64"}, "invalid float token"),
        ({"bits": "zzzzzzzzzzzzzzzz", "precision": "float64"}, "invalid float token"),
        ({"bits": "7ff0000000000000", "precision": "float64"}, "finite"),
        ({"bits": "8000000000000000", "precision": "float64"}, "negative zero"),
    ],
)
def test_artifact_rejects_invalid_typed_metadata_value(
    replacement: object,
    message: str,
) -> None:
    construction = _artifact().construction_projection()
    _set_nested(construction, ("base_grid", "parameters", "start"), replacement)

    with pytest.raises(ScheduleContractError, match=message):
        _artifact_from_untrusted_construction(construction)


def test_artifact_rejects_duplicate_or_missing_override_explanations() -> None:
    construction = _artifact().construction_projection()
    overrides = cast(list[object], construction["overrides"])
    overrides.append(dict(cast(dict[str, object], overrides[0])))
    with pytest.raises(ScheduleContractError, match="duplicate field paths"):
        _artifact_from_untrusted_construction(construction)

    construction = _artifact().construction_projection()
    construction["overrides"] = []
    with pytest.raises(ScheduleContractError, match="lack override records"):
        _artifact_from_untrusted_construction(construction)


@pytest.mark.parametrize(
    ("policy", "denoise_bits", "start_step", "message"),
    [
        ("full", "3fe0000000000000", 0, "full slicing"),
        ("full", "3ff0000000000000", 1, "full slicing"),
        ("manual_range", "3fe0000000000000", 0, "manual slicing"),
        ("denoise_tail", "3ff0000000000000", 0, "denoise-tail"),
    ],
)
def test_artifact_rejects_inconsistent_slice_policy(
    policy: str,
    denoise_bits: str,
    start_step: int,
    message: str,
) -> None:
    construction = _artifact().construction_projection()
    slicing = cast(dict[str, object], construction["slicing"])
    slicing["policy"] = policy
    slicing["start_step"] = start_step
    slicing["end_step"] = start_step + 4
    cast(dict[str, object], slicing["denoise"])["bits"] = denoise_bits

    with pytest.raises(ScheduleContractError, match=message):
        _artifact_from_untrusted_construction(construction)


def test_artifact_accepts_optional_and_empty_construction_variants() -> None:
    construction = _artifact().construction_projection()
    cast(dict[str, object], construction["source"])["revision"] = None
    for field_name in ("requested", "effective"):
        inputs = cast(dict[str, object], construction[field_name])
        inputs["width"] = None
        inputs["height"] = None
        inputs["profile"] = None
        inputs["profile_version"] = None
    compatibility = cast(
        dict[str, object],
        cast(dict[str, object], construction["effective"])["compatibility"],
    )
    compatibility["optional"] = None
    construction["transforms"] = []

    assert _artifact_from_untrusted_construction(construction).construction_projection() == (
        construction
    )


def test_artifact_rejects_missing_top_level_construction_field() -> None:
    construction = _artifact().construction_projection()
    construction.pop("warnings")

    with pytest.raises(ScheduleContractError, match="construction projection fields"):
        _artifact_from_untrusted_construction(construction)


def test_transport_is_stable_across_subprocess_hash_seeds(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_bytes(serialize_schedule_artifact(_artifact()))
    script = """
import pathlib
import sys
from comfyui_sigmax.core import deserialize_schedule_artifact, serialize_schedule_artifact

payload = pathlib.Path(sys.argv[1]).read_bytes()
artifact = deserialize_schedule_artifact(payload)
sys.stdout.buffer.write(serialize_schedule_artifact(artifact))
"""

    for seed in ("1", "777", "random"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script, str(artifact_path)],
            check=True,
            capture_output=True,
            env=environment,
        )
        assert result.stdout == artifact_path.read_bytes()
