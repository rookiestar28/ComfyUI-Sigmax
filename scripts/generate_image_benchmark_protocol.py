"""Generate the canonical M7-03 optional image benchmark protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from comfyui_sigmax.benchmark_matrix import load_numerical_benchmark_matrix
from comfyui_sigmax.image_benchmark import (
    BLIND_BALLOT_SCHEMA,
    BLIND_REVEAL_SCHEMA,
    IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA,
    IMAGE_BENCHMARK_PROTOCOL_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = Path("comfyui_sigmax/benchmarks/numerical_matrix_v1.json")
OUTPUT_PATH = ROOT / "comfyui_sigmax/benchmarks/image_protocol_v1.json"

_CASES = (
    {
        "cfg": "5.5",
        "id": "krea2.raw.framework_portrait",
        "negative_prompt": "",
        "prompt": (
            "A ceramic astronomer in a quiet observatory, soft window light, finely layered "
            "blue glazes."
        ),
        "seed": 73031,
        "source_row": "host.h2.krea2-raw-diffusers-portrait-761x1353",
    },
    {
        "cfg": "4.5",
        "id": "krea2.raw.official_landscape",
        "negative_prompt": "",
        "prompt": (
            "A wind farm crossing amber hills beneath layered storm clouds, wide cinematic "
            "composition."
        ),
        "seed": 73032,
        "source_row": "host.h2.krea2-raw-official-landscape-1353x761",
    },
    {
        "cfg": "4.5",
        "id": "krea2.raw.official_square",
        "negative_prompt": "",
        "prompt": (
            "An intricate brass automaton tending a glass greenhouse at dawn, centered composition."
        ),
        "seed": 73033,
        "source_row": "host.h2.krea2-raw-official-square-1024",
    },
    {
        "cfg": "1",
        "id": "krea2.turbo.official_square",
        "negative_prompt": "",
        "prompt": (
            "A red paper kite above a white coastal village, crisp midday light, clean "
            "geometric shadows."
        ),
        "seed": 73034,
        "source_row": "host.h2.krea2-turbo-1024",
    },
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _dimensions(value: dict[str, Any]) -> dict[str, int]:
    return {"height": cast(int, value["height"]), "width": cast(int, value["width"])}


def build_protocol_envelope() -> dict[str, object]:
    matrix = load_numerical_benchmark_matrix()
    matrix_projection = matrix.projection()
    rows = {
        cast(str, row["id"]): row
        for row in cast(list[dict[str, Any]], matrix_projection["results"])
    }
    cases: list[dict[str, object]] = []
    for declaration in _CASES:
        row = rows[cast(str, declaration["source_row"])]
        prompt = cast(str, declaration["prompt"])
        negative_prompt = cast(str, declaration["negative_prompt"])
        evidence = cast(dict[str, Any], row["evidence"])
        artifact = cast(dict[str, str], evidence["artifact"])
        workload = cast(dict[str, dict[str, Any]], row["workload"])
        requested = workload["requested"]
        profile = cast(dict[str, object], row["profile"])
        settings = {
            "cfg": declaration["cfg"],
            "sampler": "euler",
            "steps": requested["transitions"],
        }
        workload_projection = {
            "effective": _dimensions(workload["effective"]),
            "requested": _dimensions(requested),
        }
        prompt_fingerprint = _identity(
            _canonical({"negative_prompt": negative_prompt, "prompt": prompt})
        )
        cases.append(
            {
                "candidates": [
                    {
                        "id": "reference_control",
                        "schedule_provider": "reference_replay",
                    },
                    {
                        "id": "sigmax_candidate",
                        "schedule_provider": "sigmax_profile",
                    },
                ],
                "id": declaration["id"],
                "negative_prompt": negative_prompt,
                "profile": profile,
                "prompt": prompt,
                "prompt_fingerprint": prompt_fingerprint,
                "schedule_evidence": {
                    "construction_fingerprint": artifact["construction_fingerprint"],
                    "numerical_fingerprint": artifact["numerical_fingerprint"],
                    "receipt_fingerprint": evidence["receipt_fingerprint"],
                    "receipt_status": row["execution"]["status"],
                },
                "seed": declaration["seed"],
                "settings": settings,
                "settings_fingerprint": _identity(
                    _canonical(
                        {
                            "profile": profile,
                            "prompt_fingerprint": prompt_fingerprint,
                            "seed": declaration["seed"],
                            "settings": settings,
                            "workload": workload_projection,
                        }
                    )
                ),
                "workload": workload_projection,
            }
        )
    source_bytes = (ROOT / SOURCE_PATH).read_bytes()
    protocol: dict[str, object] = {
        "authority": {
            "cannot_establish": [
                "mathematical_parity",
                "official_profile_status",
                "schedule_correctness",
            ],
            "level": "supplemental_only",
        },
        "blind_protocol": {
            "assignment_algorithm": "sha256-ranked-balanced-ab/1",
            "ballot_schema": BLIND_BALLOT_SCHEMA,
            "commitment_algorithm": "sha256-secret-seed/1",
            "labels": ["A", "B"],
            "reveal_schema": BLIND_REVEAL_SCHEMA,
            "review_sequence": ["ballot_frozen", "votes_frozen", "seed_revealed"],
            "seed_encoding": "lowercase_hex_256bit",
        },
        "cases": cases,
        "execution_state": {
            "component_hashes": None,
            "image_hashes": None,
            "reason": "gpu_model_weights_not_approved",
            "status": "not_executed",
        },
        "metrics": {
            "authority": "supplemental_only",
            "observations": [],
            "required_fields": ["implementation", "metric_id", "value", "version"],
            "value_encoding": "canonical_decimal_string",
        },
        "numerical_prerequisite": {
            "matrix_fingerprint": matrix.matrix_fingerprint,
            "required_status": "PASS",
            "schema": matrix_projection["schema"],
        },
        "schema": IMAGE_BENCHMARK_PROTOCOL_SCHEMA,
        "sources": [
            {
                "matrix_fingerprint": matrix.matrix_fingerprint,
                "path": SOURCE_PATH.as_posix(),
                "schema": "sigmax.numerical-benchmark-matrix-envelope/1",
                "sha256": _identity(source_bytes),
                "status": "PASS",
            }
        ],
    }
    protocol_bytes = _canonical(protocol)
    return {
        "protocol": protocol,
        "protocol_fingerprint": _identity(protocol_bytes),
        "schema": IMAGE_BENCHMARK_PROTOCOL_ENVELOPE_SCHEMA,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = _canonical(build_protocol_envelope()) + b"\n"
    if args.write:
        OUTPUT_PATH.write_bytes(payload)
        print("IMAGE_BENCHMARK_PROTOCOL=WRITTEN")
        return 0
    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_bytes() != payload:
        print("IMAGE_BENCHMARK_PROTOCOL=DRIFT")
        return 1
    print("IMAGE_BENCHMARK_PROTOCOL=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
