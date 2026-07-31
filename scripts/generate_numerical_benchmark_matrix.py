"""Generate the canonical M7-02 numerical benchmark matrix from public evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "comfyui_sigmax" / "benchmarks" / "numerical_matrix_v1.json"
SOURCE_PATHS = (
    "comfyui_sigmax/workflows/fixtures.json",
    "tests/benchmarks/fixtures/known_good_host_attempts_v1.json",
    "tests/conformance/fixtures/capability_receipt_conformance_v1.json",
    "tests/parity/fixtures/krea2_native_euler_parity_v1.json",
    "tests/parity/fixtures/krea2_raw_parity_v1.json",
    "tests/parity/fixtures/krea2_turbo_parity_v1.json",
)


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


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


def _source_records(sources: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "schema": sources[path]["schema"],
            "sha256": _identity((ROOT / path).read_bytes()),
            "status": sources[path].get("status"),
        }
        for path in SOURCE_PATHS
    ]


def _runtime(
    *,
    device: str,
    dtype: str,
    python: str | None = None,
    torch: str | None = None,
    numpy: str | None = None,
    diffusers: str | None = None,
    comfyui: str | None = None,
    comfyui_revision_chunks: list[str] | None = None,
    sigmax_revision_chunks: list[str] | None = None,
) -> dict[str, object]:
    return {
        "comfyui": comfyui,
        "comfyui_revision_chunks": comfyui_revision_chunks,
        "device": device,
        "diffusers": diffusers,
        "dtype": dtype,
        "numpy": numpy,
        "python": python,
        "sigmax_revision_chunks": sigmax_revision_chunks,
        "torch": torch,
    }


def _counts(
    *,
    requested_transitions: int,
    effective_transitions: int,
    requested_model_evaluations: int,
    effective_model_evaluations: int,
) -> dict[str, int]:
    return {
        "effective_model_evaluations": effective_model_evaluations,
        "effective_transitions": effective_transitions,
        "requested_model_evaluations": requested_model_evaluations,
        "requested_transitions": requested_transitions,
    }


def _workload(
    *,
    requested_width: int,
    requested_height: int,
    effective_width: int,
    effective_height: int,
    transitions: int,
) -> dict[str, object]:
    return {
        "effective": {
            "height": effective_height,
            "transitions": transitions,
            "width": effective_width,
        },
        "requested": {
            "height": requested_height,
            "transitions": transitions,
            "width": requested_width,
        },
    }


def _baseline_summary(comparisons: dict[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("diffusers_float32", "krea_float64"):
        comparison = comparisons[name]
        result[name] = {
            "device": comparison["device"],
            "dtype": comparison["dtype"],
            "fingerprint": comparison["fingerprint"],
            "max_abs_error": comparison["max_abs_error"],
            "mean_abs_error": comparison["mean_abs_error"],
            "status": comparison["status"],
            "tolerance": comparison["tolerance"],
        }
    return result


def _profile(
    *,
    identifier: str,
    variant: str,
    evidence: str,
    recipe: str,
) -> dict[str, str]:
    return {
        "evidence": evidence,
        "id": identifier,
        "recipe": recipe,
        "variant": variant,
        "version": "1",
    }


def _parity_rows(
    turbo: dict[str, Any],
    raw: dict[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    turbo_runtime = turbo["environment"]
    for case in turbo["cases"]:
        steps = cast(int, case["steps"])
        comparisons = cast(dict[str, Any], case["comparisons"])
        rows.append(
            {
                "baselines": _baseline_summary(comparisons),
                "capability": {"level": "allow", "reasons": ["compatible"]},
                "determinism": None,
                "evidence": {
                    "artifact": {
                        "construction_fingerprint": None,
                        "numerical_fingerprint": comparisons["krea_float64"]["fingerprint"],
                    },
                    "receipt_fingerprint": None,
                    "source_ids": ["parity.turbo"],
                },
                "execution": {
                    "counts": _counts(
                        requested_transitions=steps,
                        effective_transitions=0,
                        requested_model_evaluations=0,
                        effective_model_evaluations=0,
                    ),
                    "rng_ownership": {
                        "model": "none",
                        "sampler": "none",
                        "schedule": "none",
                    },
                    "status": "not_executed",
                },
                "id": f"parity.turbo.{steps}",
                "lane": "turbo_schedule_parity",
                "model_weights_present": False,
                "profile": _profile(
                    identifier=turbo["profile"]["id"],
                    variant="Turbo",
                    evidence=case["evidence"],
                    recipe=f"krea2.turbo.official-{steps}",
                ),
                "repeat": None,
                "runtime": _runtime(
                    device=turbo_runtime["device"],
                    dtype="float64+float32",
                    torch=turbo_runtime["torch"],
                    numpy=turbo_runtime["numpy"],
                    diffusers=turbo_runtime["diffusers"],
                ),
                "schedule": {"image_seq_len": None, "mu": "1.15"},
                "weight_variant": "none",
                "workload": _workload(
                    requested_width=1024,
                    requested_height=1024,
                    effective_width=1024,
                    effective_height=1024,
                    transitions=steps,
                ),
            }
        )
    raw_runtime = raw["environment"]
    for case in raw["cases"]:
        comparisons = cast(dict[str, Any], case["comparisons"])
        rows.append(
            {
                "baselines": _baseline_summary(comparisons),
                "capability": {"level": "allow", "reasons": ["compatible"]},
                "determinism": None,
                "evidence": {
                    "artifact": {
                        "construction_fingerprint": None,
                        "numerical_fingerprint": comparisons["krea_float64"]["fingerprint"],
                    },
                    "receipt_fingerprint": None,
                    "source_ids": ["parity.raw"],
                },
                "execution": {
                    "counts": _counts(
                        requested_transitions=case["steps"],
                        effective_transitions=0,
                        requested_model_evaluations=0,
                        effective_model_evaluations=0,
                    ),
                    "rng_ownership": {
                        "model": "none",
                        "sampler": "none",
                        "schedule": "none",
                    },
                    "status": "not_executed",
                },
                "id": f"parity.raw.{case['case_id']}",
                "lane": "raw_schedule_parity",
                "model_weights_present": False,
                "profile": _profile(
                    identifier=raw["profile"]["id"],
                    variant="RAW",
                    evidence=case["evidence"],
                    recipe=case["recipe_id"],
                ),
                "repeat": None,
                "runtime": _runtime(
                    device=raw_runtime["device"],
                    dtype="float64+float32",
                    torch=raw_runtime["torch"],
                    numpy=raw_runtime["numpy"],
                    diffusers=raw_runtime["diffusers"],
                ),
                "schedule": {
                    "image_seq_len": case["image_seq_len"],
                    "mu": case["mu"],
                },
                "weight_variant": "none",
                "workload": _workload(
                    requested_width=case["requested_width"],
                    requested_height=case["requested_height"],
                    effective_width=case["effective_width"],
                    effective_height=case["effective_height"],
                    transitions=case["steps"],
                ),
            }
        )
    return rows


def _repeat(lane: dict[str, Any]) -> dict[str, object]:
    return {
        "accepted": lane["accepted"],
        "first_status": lane["first_status"],
        "repeat_status": lane["repeat_status"],
        "stable": lane["accepted"] and lane["first_status"] == lane["repeat_status"],
        "transition": lane["transition"],
    }


def _workflow_rows(
    workflows: dict[str, Any],
    host_attempts: dict[str, Any],
    raw_parity: dict[str, Any],
) -> list[dict[str, object]]:
    host = host_attempts["host"]
    lanes = {lane["id"]: lane for lane in host_attempts["lanes"]}
    raw_cases = cast(list[dict[str, Any]], raw_parity["cases"])
    rows: list[dict[str, object]] = []
    for fixture in workflows["fixtures"]:
        scheduler = next(
            node
            for node in fixture["workflow"]["nodes"]
            if node["type"] == "Sigmax.Krea2SigmaScheduler"
        )
        variant, steps, width, height, strict_official, _, _ = scheduler["widgets_values"]
        effective_width = (width + 15) // 16 * 16
        effective_height = (height + 15) // 16 * 16
        metadata = fixture["workflow"]["extra"]["comfyui_sigmax"]["metadata"]
        artifact = metadata["artifact"]
        receipt = metadata["receipts"][0]
        lane = lanes[f"h2.{fixture['id']}"]
        if variant == "RAW":
            parity_case = next(
                case
                for case in raw_cases
                if case["steps"] == steps
                and case["effective_width"] == effective_width
                and case["effective_height"] == effective_height
            )
            evidence = "official" if strict_official else "framework_reference"
            recipe = parity_case["recipe_id"]
            schedule = {
                "image_seq_len": parity_case["image_seq_len"],
                "mu": parity_case["mu"],
            }
        else:
            evidence = "official"
            recipe = "krea2.turbo.official-8"
            schedule = {"image_seq_len": None, "mu": "1.15"}
        rows.append(
            {
                "baselines": None,
                "capability": {"level": "allow", "reasons": ["compatible"]},
                "determinism": None,
                "evidence": {
                    "artifact": {
                        "construction_fingerprint": artifact["construction_fingerprint"],
                        "numerical_fingerprint": artifact["numerical_fingerprint"],
                    },
                    "receipt_fingerprint": receipt["receipt_fingerprint"],
                    "source_ids": ["workflow.fixtures", "host.known_good"],
                },
                "execution": {
                    "counts": _counts(
                        requested_transitions=steps,
                        effective_transitions=0,
                        requested_model_evaluations=steps,
                        effective_model_evaluations=0,
                    ),
                    "rng_ownership": {
                        "model": "none",
                        "sampler": "none",
                        "schedule": "none",
                    },
                    "status": "not_executed",
                },
                "id": f"host.h2.{fixture['id']}",
                "lane": "h2_workflow",
                "model_weights_present": False,
                "profile": _profile(
                    identifier=metadata["profile"]["id"],
                    variant=variant,
                    evidence=evidence,
                    recipe=recipe,
                ),
                "repeat": _repeat(lane),
                "runtime": _runtime(
                    device="cpu",
                    dtype="float64",
                    comfyui=host["version"],
                    comfyui_revision_chunks=host["revision_chunks"],
                    sigmax_revision_chunks=host_attempts["sigmax_revision_chunks"],
                ),
                "schedule": schedule,
                "weight_variant": "none",
                "workload": _workload(
                    requested_width=width,
                    requested_height=height,
                    effective_width=effective_width,
                    effective_height=effective_height,
                    transitions=steps,
                ),
            }
        )
    return rows


def _native_euler_row(
    native: dict[str, Any],
    workflows: dict[str, Any],
    host_attempts: dict[str, Any],
) -> dict[str, object]:
    fixture = next(
        fixture for fixture in workflows["fixtures"] if fixture["id"] == "krea2-turbo-1024"
    )
    metadata = fixture["workflow"]["extra"]["comfyui_sigmax"]["metadata"]
    lane = next(lane for lane in host_attempts["lanes"] if lane["id"] == "h3.native-euler")
    host = host_attempts["host"]
    case = native["case"]
    environment = native["environment"]
    return {
        "baselines": {
            "native_euler": {
                "device": environment["device"],
                "dtype": environment["dtype"],
                "fingerprint": case["trace_fingerprint"],
                "max_abs_error": case["max_abs_error"],
                "mean_abs_error": case["mean_abs_error"],
                "status": case["status"],
                "tolerance": case["tolerance"],
            }
        },
        "capability": {"level": "allow", "reasons": ["compatible"]},
        "determinism": {
            "deterministic_rerun": case["deterministic_rerun"],
            "max_abs_error": case["max_abs_error"],
            "mean_abs_error": case["mean_abs_error"],
            "status": case["status"],
            "tolerance": case["tolerance"],
            "trace_fingerprint": case["trace_fingerprint"],
        },
        "evidence": {
            "artifact": {
                "construction_fingerprint": metadata["artifact"]["construction_fingerprint"],
                "numerical_fingerprint": metadata["artifact"]["numerical_fingerprint"],
            },
            "receipt_fingerprint": lane["receipt_fingerprint"],
            "source_ids": ["parity.native_euler", "host.known_good"],
        },
        "execution": {
            "counts": case["counts"],
            "rng_ownership": {
                "model": "none",
                "sampler": "none",
                "schedule": "none",
            },
            "status": "succeeded",
        },
        "id": "host.h3.native-euler",
        "lane": "h3_native_euler",
        "model_weights_present": False,
        "profile": _profile(
            identifier=native["profile"]["id"],
            variant="Turbo",
            evidence="official",
            recipe="krea2.turbo.official-8",
        ),
        "repeat": _repeat(lane),
        "runtime": _runtime(
            device=environment["device"],
            dtype=environment["dtype"],
            python=environment["python"],
            torch=environment["torch"],
            numpy=environment["numpy"],
            comfyui=host["version"],
            comfyui_revision_chunks=host["revision_chunks"],
            sigmax_revision_chunks=host_attempts["sigmax_revision_chunks"],
        ),
        "schedule": {"image_seq_len": None, "mu": "1.15"},
        "weight_variant": "none",
        "workload": _workload(
            requested_width=1024,
            requested_height=1024,
            effective_width=1024,
            effective_height=1024,
            transitions=8,
        ),
    }


def build_matrix_envelope() -> dict[str, object]:
    sources = {path: _read_json(path) for path in SOURCE_PATHS}
    for path, source in sources.items():
        if source.get("status") not in {None, "PASS"}:
            raise RuntimeError(f"{path} is not accepted evidence")
    results = _parity_rows(
        sources["tests/parity/fixtures/krea2_turbo_parity_v1.json"],
        sources["tests/parity/fixtures/krea2_raw_parity_v1.json"],
    )
    results.extend(
        _workflow_rows(
            sources["comfyui_sigmax/workflows/fixtures.json"],
            sources["tests/benchmarks/fixtures/known_good_host_attempts_v1.json"],
            sources["tests/parity/fixtures/krea2_raw_parity_v1.json"],
        )
    )
    results.append(
        _native_euler_row(
            sources["tests/parity/fixtures/krea2_native_euler_parity_v1.json"],
            sources["comfyui_sigmax/workflows/fixtures.json"],
            sources["tests/benchmarks/fixtures/known_good_host_attempts_v1.json"],
        )
    )
    results.sort(key=lambda row: cast(str, row["id"]))
    coverage = {
        lane: sum(row["lane"] == lane for row in results)
        for lane in (
            "h2_workflow",
            "h3_native_euler",
            "raw_schedule_parity",
            "turbo_schedule_parity",
        )
    }
    coverage["total_verified_results"] = len(results)
    matrix = {
        "coverage": coverage,
        "exclusions": [
            "advanced_workflows",
            "gpu_model_weights",
            "partial_denoise_execution",
            "resume_execution",
            "stochastic_euler_execution",
        ],
        "results": results,
        "schema": "sigmax.numerical-benchmark-matrix/1",
        "sources": _source_records(sources),
        "weight_variants": [
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
        ],
    }
    matrix_bytes = _canonical(matrix)
    return {
        "matrix": matrix,
        "matrix_fingerprint": _identity(matrix_bytes),
        "schema": "sigmax.numerical-benchmark-matrix-envelope/1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = _canonical(build_matrix_envelope()) + b"\n"
    if args.check:
        if not TARGET.is_file() or TARGET.read_bytes() != payload:
            print("BENCHMARK_MATRIX=FAIL")
            return 1
        print("BENCHMARK_MATRIX=PASS")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(payload)
    print(f"WROTE={TARGET.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
