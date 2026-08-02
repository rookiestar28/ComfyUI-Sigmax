"""Canonical capability-filtered numerical benchmark matrix."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.benchmark_matrix import (
    NUMERICAL_BENCHMARK_MATRIX_ENVELOPE_SCHEMA,
    NUMERICAL_BENCHMARK_MATRIX_SCHEMA,
    load_numerical_benchmark_matrix,
    serialize_numerical_benchmark_matrix,
)
from comfyui_sigmax.core import ScheduleContractError

ROOT = Path(__file__).resolve().parents[1]


def _decoded(payload: bytes) -> dict[str, Any]:
    value = json.loads(payload)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _rehashed(envelope: dict[str, Any]) -> bytes:
    matrix_bytes = json.dumps(
        envelope["matrix"],
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    envelope["matrix_fingerprint"] = "sha256:" + hashlib.sha256(matrix_bytes).hexdigest()
    return json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def test_packaged_matrix_has_exact_schema_identity_and_coverage() -> None:
    matrix = load_numerical_benchmark_matrix()
    projection = matrix.projection()
    results = cast(list[dict[str, Any]], projection["results"])

    assert projection["schema"] == NUMERICAL_BENCHMARK_MATRIX_SCHEMA
    assert (
        matrix.matrix_fingerprint
        == "sha256:e4f291bfe5762ee69ab9019818e0f20c75eb2de5e3692a2afcf91b3a172874e3"
    )
    assert len(results) == 23
    assert projection["coverage"] == {
        "h2_workflow": 4,
        "h3_native_euler": 1,
        "raw_schedule_parity": 14,
        "total_verified_results": 23,
        "turbo_schedule_parity": 4,
    }
    assert [row["id"] for row in results] == sorted(row["id"] for row in results)
    assert all(
        row["capability"] == {"level": "allow", "reasons": ["compatible"]} for row in results
    )


def test_matrix_records_parity_geometry_runtime_and_exact_metrics() -> None:
    results = {
        row["id"]: row
        for row in cast(
            list[dict[str, Any]],
            load_numerical_benchmark_matrix().projection()["results"],
        )
    }
    turbo = results["parity.turbo.8"]
    raw = results["parity.raw.framework_28_1360x768"]

    assert turbo["profile"] == {
        "evidence": "official",
        "id": "krea2.turbo.official",
        "recipe": "krea2.turbo.official-8",
        "variant": "Turbo",
        "version": "1",
    }
    assert turbo["workload"]["requested"] == {
        "height": 1024,
        "transitions": 8,
        "width": 1024,
    }
    assert turbo["runtime"]["diffusers"] == "0.39.0"
    assert turbo["baselines"]["krea_float64"]["max_abs_error"] == "1.1102230246251565e-16"
    assert turbo["baselines"]["diffusers_float32"]["tolerance"] == "1e-06"
    assert raw["workload"]["requested"] == {
        "height": 768,
        "transitions": 28,
        "width": 1360,
    }
    assert raw["workload"]["effective"] == {
        "height": 768,
        "transitions": 28,
        "width": 1360,
    }
    assert raw["schedule"]["image_seq_len"] == 4080
    assert raw["schedule"]["mu"] == "0.904557291666667"


def test_matrix_preserves_h2_artifact_receipt_and_first_repeat_truth() -> None:
    results = {
        row["id"]: row
        for row in cast(
            list[dict[str, Any]],
            load_numerical_benchmark_matrix().projection()["results"],
        )
    }
    turbo = results["host.h2.krea2-turbo-1024"]
    portrait = results["host.h2.krea2-raw-diffusers-portrait-761x1353"]

    assert turbo["execution"]["status"] == "not_executed"
    assert turbo["execution"]["counts"] == {
        "effective_model_evaluations": 0,
        "effective_transitions": 0,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }
    assert turbo["evidence"]["artifact"]["construction_fingerprint"].startswith("sha256:")
    assert turbo["evidence"]["receipt_fingerprint"].startswith("sha256:")
    assert turbo["repeat"] == {
        "accepted": True,
        "first_status": "not_executed",
        "repeat_status": "not_executed",
        "stable": True,
        "transition": "pass_to_pass",
    }
    assert portrait["workload"]["requested"] == {
        "height": 1353,
        "transitions": 28,
        "width": 761,
    }
    assert portrait["workload"]["effective"] == {
        "height": 1360,
        "transitions": 28,
        "width": 768,
    }


def test_matrix_preserves_h3_counts_rng_determinism_and_error() -> None:
    results = cast(
        list[dict[str, Any]],
        load_numerical_benchmark_matrix().projection()["results"],
    )
    h3 = next(row for row in results if row["id"] == "host.h3.native-euler")

    assert h3["execution"]["status"] == "succeeded"
    assert h3["execution"]["counts"] == {
        "effective_model_evaluations": 8,
        "effective_transitions": 8,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }
    assert h3["execution"]["rng_ownership"] == {
        "model": "none",
        "sampler": "none",
        "schedule": "none",
    }
    assert h3["determinism"] == {
        "deterministic_rerun": True,
        "max_abs_error": "8.144636609586087e-08",
        "mean_abs_error": "2.3719542649458525e-08",
        "status": "PASS",
        "tolerance": "2e-06",
        "trace_fingerprint": "sha256:f8c6213978199c6c312140428e1225818e50469f54eb5fc64e616af8dba63105",
    }
    assert h3["repeat"]["first_status"] == h3["repeat"]["repeat_status"] == "succeeded"
    assert h3["repeat"]["stable"] is True
    assert "".join(h3["runtime"]["sigmax_revision_chunks"]) == "".join(
        ("ce822edb", "7addfdf4", "befdb7c0", "f37595f6", "8ab9e902")
    )


def test_weight_variants_are_separate_and_never_claim_pass() -> None:
    projection = load_numerical_benchmark_matrix().projection()

    assert projection["weight_variants"] == [
        {
            "kind": "bf16",
            "reason": "gpu_model_weights_not_approved",
            "result": None,
            "status": "not_evaluated",
        },
        {
            "kind": "quantized",
            "reason": "gpu_model_weights_not_approved",
            "result": None,
            "status": "not_evaluated",
        },
    ]
    exclusions = cast(list[str], projection["exclusions"])
    assert "partial_denoise_execution" in exclusions
    assert "stochastic_euler_execution" in exclusions


def test_matrix_source_identities_match_every_allowlisted_public_fixture() -> None:
    sources = cast(
        list[dict[str, Any]],
        load_numerical_benchmark_matrix().projection()["sources"],
    )

    for source in sources:
        payload = (ROOT / source["path"]).read_bytes()
        assert source["sha256"] == "sha256:" + hashlib.sha256(payload).hexdigest()


def test_matrix_import_does_not_load_optional_or_host_frameworks() -> None:
    script = (
        "import builtins,sys; real=builtins.__import__; "
        "blocked={'matplotlib','numpy','torch','comfy','diffusers'}; "
        "builtins.__import__=lambda n,*a,**k: "
        "(_ for _ in ()).throw(ImportError(n)) if n.split('.')[0] in blocked "
        "else real(n,*a,**k); "
        "from comfyui_sigmax.benchmark_matrix import load_numerical_benchmark_matrix; "
        "m=load_numerical_benchmark_matrix(); assert len(m.projection()['results'])==23; "
        "assert not blocked.intersection(sys.modules)"
    )

    subprocess.run([sys.executable, "-I", "-c", script], cwd=ROOT, check=True)


def test_matrix_transport_round_trips_and_rejects_tampering() -> None:
    matrix = load_numerical_benchmark_matrix()
    payload = serialize_numerical_benchmark_matrix(matrix)
    envelope = _decoded(payload)

    assert envelope["schema"] == NUMERICAL_BENCHMARK_MATRIX_ENVELOPE_SCHEMA
    assert load_numerical_benchmark_matrix(payload) == matrix
    assert serialize_numerical_benchmark_matrix(load_numerical_benchmark_matrix(payload)) == payload

    envelope["matrix"]["coverage"]["total_verified_results"] = 22
    tampered = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    with pytest.raises(ScheduleContractError, match=r"fingerprint|coverage"):
        load_numerical_benchmark_matrix(tampered)

    envelope = _decoded(payload)
    envelope["matrix"]["unknown"] = True
    with pytest.raises(ScheduleContractError, match=r"fields|schema"):
        load_numerical_benchmark_matrix(_rehashed(envelope))


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda matrix: matrix["results"][0]["capability"].update(level="reject"),
            "capability",
        ),
        (
            lambda matrix: matrix["results"][0]["execution"]["counts"].update(
                requested_transitions=99
            ),
            "transitions",
        ),
        (
            lambda matrix: matrix["results"][0]["repeat"].update(stable=False),
            "repeat",
        ),
        (
            lambda matrix: matrix["results"][0].update(unknown=True),
            "fields",
        ),
        (
            lambda matrix: matrix["sources"][0].update(path="C:\\private\\evidence.json"),
            "path",
        ),
        (
            lambda matrix: matrix["weight_variants"][0].update(
                result={"status": "PASS"},
                status="PASS",
            ),
            "weight variants",
        ),
    ),
)
def test_rehashed_semantic_drift_fails_closed(
    mutation: Any,
    match: str,
) -> None:
    envelope = _decoded(serialize_numerical_benchmark_matrix(load_numerical_benchmark_matrix()))
    mutation(envelope["matrix"])

    with pytest.raises(ScheduleContractError, match=match):
        load_numerical_benchmark_matrix(_rehashed(envelope))


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"\xef\xbb\xbf{}",
        b'{"schema":"sigmax.numerical-benchmark-matrix-envelope/1","schema":"duplicate"}',
        b'{ "schema":"sigmax.numerical-benchmark-matrix-envelope/1" }',
        b'{"value":1.5}',
        b"[]",
    ),
)
def test_matrix_transport_rejects_malformed_input(payload: bytes) -> None:
    with pytest.raises(ScheduleContractError):
        load_numerical_benchmark_matrix(payload)


def test_matrix_transport_rejects_oversized_input() -> None:
    with pytest.raises(ScheduleContractError, match="size"):
        load_numerical_benchmark_matrix(b"{" + b"x" * 1_048_576)


def test_generator_check_matches_packaged_matrix() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_numerical_benchmark_matrix.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "BENCHMARK_MATRIX=PASS" in completed.stdout
