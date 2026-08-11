"""Optional image benchmark protocol and blind-review contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.image_benchmark import (
    BLIND_BALLOT_SCHEMA,
    BLIND_REVEAL_SCHEMA,
    IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA,
    IMAGE_BENCHMARK_PROTOCOL_SCHEMA,
    BlindReviewBallot,
    CandidateImageEvidence,
    build_blind_ballot,
    build_blind_reveal,
    load_image_benchmark_protocol,
    serialize_blind_ballot,
    serialize_blind_reveal,
    serialize_image_benchmark_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
SECRET_SEED = "42" * 32


def _decoded(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _rehashed_protocol(envelope: dict[str, Any]) -> bytes:
    protocol_bytes = json.dumps(
        envelope["protocol"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    envelope["protocol_fingerprint"] = "sha256:" + hashlib.sha256(protocol_bytes).hexdigest()
    return (
        json.dumps(
            envelope,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _candidate_evidence() -> tuple[CandidateImageEvidence, ...]:
    protocol = load_image_benchmark_protocol().projection()
    evidence: list[CandidateImageEvidence] = []
    for case_index, case in enumerate(cast(list[dict[str, Any]], protocol["cases"])):
        schedule = cast(dict[str, str], case["schedule_evidence"])
        for candidate_index, candidate in enumerate(cast(list[dict[str, str]], case["candidates"])):
            suffix = case_index * 2 + candidate_index + 1
            identity = f"{suffix:064x}"
            evidence.append(
                CandidateImageEvidence(
                    case_id=cast(str, case["id"]),
                    candidate_id=candidate["id"],
                    execution_status="succeeded",
                    settings_fingerprint=cast(str, case["settings_fingerprint"]),
                    construction_fingerprint=schedule["construction_fingerprint"],
                    numerical_fingerprint=schedule["numerical_fingerprint"],
                    receipt_fingerprint=f"sha256:{suffix + 100:064x}",
                    checkpoint_sha256=f"sha256:{suffix + 200:064x}",
                    text_encoder_sha256=f"sha256:{suffix + 300:064x}",
                    vae_sha256=f"sha256:{suffix + 400:064x}",
                    image_sha256=f"sha256:{identity}",
                    precision="bf16",
                    comfyui_version="0.29.0",
                    python_version="3.13.9",
                    torch_version="2.9.1",
                    gpu="approved-test-device",
                )
            )
    return tuple(evidence)


def test_packaged_protocol_has_fixed_cases_settings_hashes_and_evidence() -> None:
    protocol = load_image_benchmark_protocol()
    projection = protocol.projection()
    cases = cast(list[dict[str, Any]], projection["cases"])

    assert projection["schema"] == IMAGE_BENCHMARK_PROTOCOL_SCHEMA
    assert len(cases) == 4
    assert [case["id"] for case in cases] == [
        "krea2.raw.framework_portrait",
        "krea2.raw.official_landscape",
        "krea2.raw.official_square",
        "krea2.turbo.official_square",
    ]
    assert all(case["prompt"] and isinstance(case["negative_prompt"], str) for case in cases)
    assert len({case["prompt_fingerprint"] for case in cases}) == 4
    assert all(case["settings_fingerprint"].startswith("sha256:") for case in cases)
    assert all(case["seed"] >= 0 for case in cases)
    assert all(case["settings"]["sampler"] == "euler" for case in cases)
    assert {case["settings"]["cfg"] for case in cases} == {"1", "4.5", "5.5"}
    assert all(
        case["schedule_evidence"]["construction_fingerprint"].startswith("sha256:")
        for case in cases
    )
    assert all(
        case["schedule_evidence"]["numerical_fingerprint"].startswith("sha256:") for case in cases
    )
    assert all(
        case["schedule_evidence"]["receipt_fingerprint"].startswith("sha256:") for case in cases
    )
    assert all(case["schedule_evidence"]["receipt_status"] == "not_executed" for case in cases)
    assert all(
        [candidate["id"] for candidate in case["candidates"]]
        == [
            "reference_control",
            "sigmax_candidate",
        ]
        for case in cases
    )


def test_protocol_binds_numerical_matrix_and_declares_unapproved_heavy_state() -> None:
    projection = load_image_benchmark_protocol().projection()

    assert projection["numerical_prerequisite"] == {
        "matrix_fingerprint": "sha256:79a6fdb5fd2cd8bb9f3abc8ed7a0099134193c107628b4bad52c2989fc81ed02",
        "required_status": "PASS",
        "schema": "sigmax.numerical-benchmark-matrix/1",
    }
    assert projection["execution_state"] == {
        "component_hashes": None,
        "image_hashes": None,
        "reason": "gpu_model_weights_not_approved",
        "status": "not_executed",
    }
    assert projection["metrics"] == {
        "authority": "supplemental_only",
        "observations": [],
        "required_fields": [
            "implementation",
            "metric_id",
            "value",
            "version",
        ],
        "value_encoding": "canonical_decimal_string",
    }
    authority = cast(dict[str, Any], projection["authority"])
    assert authority["level"] == "supplemental_only"
    assert authority["cannot_establish"] == [
        "mathematical_parity",
        "official_profile_status",
        "schedule_correctness",
    ]


def test_protocol_transport_round_trips_and_rejects_semantic_tampering() -> None:
    protocol = load_image_benchmark_protocol()
    payload = serialize_image_benchmark_protocol(protocol)
    envelope = _decoded(payload)

    assert envelope["schema"] == IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA
    assert load_image_benchmark_protocol(payload) == protocol
    assert serialize_image_benchmark_protocol(load_image_benchmark_protocol(payload)) == payload

    envelope["protocol"]["authority"]["level"] = "official"
    with pytest.raises(ScheduleContractError, match="authority"):
        load_image_benchmark_protocol(_rehashed_protocol(envelope))


def test_candidate_evidence_is_immutable_and_requires_complete_execution_truth() -> None:
    evidence = _candidate_evidence()[0]

    assert dataclasses.is_dataclass(evidence)
    with pytest.raises(dataclasses.FrozenInstanceError):
        evidence.execution_status = "not_executed"  # type: ignore[misc]

    values = dataclasses.asdict(evidence)
    values["execution_status"] = "not_executed"
    with pytest.raises(ScheduleContractError, match="succeeded"):
        CandidateImageEvidence(**values)

    values = dataclasses.asdict(evidence)
    values["checkpoint_sha256"] = None
    with pytest.raises(ScheduleContractError, match="checkpoint"):
        CandidateImageEvidence(**values)


def test_blind_ballot_is_deterministic_balanced_and_hides_assignment() -> None:
    protocol = load_image_benchmark_protocol()
    evidence = _candidate_evidence()

    first = build_blind_ballot(protocol, evidence, secret_seed=SECRET_SEED)
    repeat = build_blind_ballot(protocol, tuple(reversed(evidence)), secret_seed=SECRET_SEED)
    payload = serialize_blind_ballot(first)
    projection = first.projection()

    assert isinstance(first, BlindReviewBallot)
    assert repeat == first
    assert projection["schema"] == BLIND_BALLOT_SCHEMA
    assert projection["authority"] == "supplemental_only"
    assert cast(str, projection["seed_commitment"]).startswith("sha256:")
    assert len(cast(list[object], projection["trials"])) == 4
    assert b"reference_control" not in payload
    assert b"sigmax_candidate" not in payload
    assert SECRET_SEED.encode() not in payload

    reveal = build_blind_reveal(protocol, first, evidence, secret_seed=SECRET_SEED)
    assignments = cast(list[dict[str, str]], reveal.projection()["assignments"])
    assert sum(item["image_a_candidate_id"] == "reference_control" for item in assignments) == 2
    assert sum(item["image_b_candidate_id"] == "reference_control" for item in assignments) == 2


def test_blind_reveal_verifies_commitment_ballot_and_exact_mapping() -> None:
    protocol = load_image_benchmark_protocol()
    evidence = _candidate_evidence()
    ballot = build_blind_ballot(protocol, evidence, secret_seed=SECRET_SEED)
    reveal = build_blind_reveal(protocol, ballot, evidence, secret_seed=SECRET_SEED)
    projection = reveal.projection()

    assert projection["schema"] == BLIND_REVEAL_SCHEMA
    assert projection["authority"] == "supplemental_only"
    assert projection["ballot_fingerprint"] == ballot.ballot_fingerprint
    assert projection["secret_seed"] == SECRET_SEED
    assert len(cast(list[object], projection["assignments"])) == 4
    assert serialize_blind_reveal(reveal).endswith(b"\n")

    with pytest.raises(ScheduleContractError, match=r"commitment|seed|ballot"):
        build_blind_reveal(protocol, ballot, evidence, secret_seed="24" * 32)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda values: values.update(case_id="unknown.case"),
            "case",
        ),
        (
            lambda values: values.update(candidate_id="unknown_candidate"),
            "candidate",
        ),
        (
            lambda values: values.update(construction_fingerprint="sha256:" + "f" * 64),
            "construction",
        ),
        (
            lambda values: values.update(settings_fingerprint="sha256:" + "f" * 64),
            "settings",
        ),
        (
            lambda values: values.update(image_sha256=None),
            "image",
        ),
    ),
)
def test_blind_ballot_rejects_incomplete_or_cross_linked_evidence(
    mutation: Any,
    match: str,
) -> None:
    protocol = load_image_benchmark_protocol()
    evidence = list(_candidate_evidence())
    values = dataclasses.asdict(evidence[0])
    mutation(values)
    if values.get("image_sha256") is None:
        with pytest.raises(ScheduleContractError, match=match):
            CandidateImageEvidence(**values)
        return
    evidence[0] = CandidateImageEvidence(**values)

    with pytest.raises(ScheduleContractError, match=match):
        build_blind_ballot(protocol, evidence, secret_seed=SECRET_SEED)


def test_blind_ballot_rejects_duplicate_images_missing_pairs_and_invalid_seed() -> None:
    protocol = load_image_benchmark_protocol()
    evidence = list(_candidate_evidence())

    duplicate = dataclasses.asdict(evidence[1])
    duplicate["image_sha256"] = evidence[0].image_sha256
    evidence[1] = CandidateImageEvidence(**duplicate)
    with pytest.raises(ScheduleContractError, match="image"):
        build_blind_ballot(protocol, evidence, secret_seed=SECRET_SEED)

    with pytest.raises(ScheduleContractError, match="coverage"):
        build_blind_ballot(protocol, _candidate_evidence()[:-1], secret_seed=SECRET_SEED)
    with pytest.raises(ScheduleContractError, match="seed"):
        build_blind_ballot(protocol, _candidate_evidence(), secret_seed=cast(Any, 7))


def test_candidate_evidence_rejects_private_runtime_paths_and_reused_receipts() -> None:
    protocol = load_image_benchmark_protocol()
    evidence = list(_candidate_evidence())
    private_runtime = dataclasses.asdict(evidence[0])
    private_runtime["gpu"] = "C:\\private\\gpu.txt"
    with pytest.raises(ScheduleContractError, match="path"):
        CandidateImageEvidence(**private_runtime)

    reused_receipt = dataclasses.asdict(evidence[1])
    reused_receipt["receipt_fingerprint"] = evidence[0].receipt_fingerprint
    evidence[1] = CandidateImageEvidence(**reused_receipt)
    with pytest.raises(ScheduleContractError, match="receipt"):
        build_blind_ballot(protocol, evidence, secret_seed=SECRET_SEED)


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"\xef\xbb\xbf{}",
        b'{"schema":"sigmax.image-benchmark-protocol-envelope/1","schema":"duplicate"}',
        b'{ "schema":"sigmax.image-benchmark-protocol-envelope/1" }',
        b'{"value":1.5}',
        b"[]",
    ),
)
def test_protocol_transport_rejects_malformed_input(payload: bytes) -> None:
    with pytest.raises(ScheduleContractError):
        load_image_benchmark_protocol(payload)


def test_protocol_rejects_paths_secret_fields_and_oversized_input() -> None:
    envelope = _decoded(serialize_image_benchmark_protocol(load_image_benchmark_protocol()))
    envelope["protocol"]["cases"][0]["prompt"] = "C:\\private\\prompt.txt"
    with pytest.raises(ScheduleContractError, match="path"):
        load_image_benchmark_protocol(_rehashed_protocol(envelope))

    envelope = _decoded(serialize_image_benchmark_protocol(load_image_benchmark_protocol()))
    envelope["protocol"]["token"] = None
    with pytest.raises(ScheduleContractError, match=r"forbidden|field"):
        load_image_benchmark_protocol(_rehashed_protocol(envelope))

    with pytest.raises(ScheduleContractError, match="size"):
        load_image_benchmark_protocol(b"{" + b"x" * 1_048_576)


def test_image_benchmark_import_does_not_load_optional_or_host_frameworks() -> None:
    script = (
        "import builtins,sys; real=builtins.__import__; "
        "blocked={'PIL','matplotlib','numpy','torch','comfy','diffusers'}; "
        "builtins.__import__=lambda n,*a,**k: "
        "(_ for _ in ()).throw(ImportError(n)) if n.split('.')[0] in blocked "
        "else real(n,*a,**k); "
        "from comfyui_sigmax.image_benchmark import load_image_benchmark_protocol; "
        "p=load_image_benchmark_protocol(); assert len(p.projection()['cases'])==4; "
        "assert not blocked.intersection(sys.modules)"
    )

    subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True)


def test_image_benchmark_generator_check_matches_packaged_protocol() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_image_benchmark_protocol.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "IMAGE_BENCHMARK_PROTOCOL=PASS" in completed.stdout
