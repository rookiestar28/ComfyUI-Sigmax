"""Measure the experimental Krea 2 CONDITIONING transform on a CPU host.

This is a benchmark/evidence path only.  It intentionally imports Torch lazily and
does not make prompt-adherence, image-quality, or cross-machine performance claims.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfyui_sigmax.adapters.krea2_conditioning import transform_krea2_conditioning  # noqa: E402
from comfyui_sigmax.conditioning import (  # noqa: E402
    ConditioningModifierRequest,
    Krea2ConditioningVariant,
)
from comfyui_sigmax.conditioning.diagnostics import compute_conditioning_diagnostics  # noqa: E402
from comfyui_sigmax.conditioning.profiles import SUBTLE_EXPERIMENTAL_PROFILE  # noqa: E402
from comfyui_sigmax.performance_budgets import (  # noqa: E402
    PerformanceBudget,
    PerformanceObservation,
    PerformanceUnit,
    PerformanceVerdict,
    evaluate_performance_budget,
)

SCHEMA: Final = "sigmax.krea2-conditioning-performance/1"
WORKLOAD_ID: Final = "krea2-conditioning-turbo-subtle-1x97x30720-cpu-f32"
SHAPE: Final = (1, 97, 30_720)
STRENGTH: Final = 0.5
REPEAT_COUNT: Final = 9
WARMUP_COUNT: Final = 3

# These are deliberately conservative, reviewed CPU-only ceilings for this fixed
# workload.  They are evidence gates for the named host, not product SLAs.
LATENCY_MAX_NS: Final = 500_000_000
PEAK_PYTHON_MAX_BYTES: Final = 16 * 1024 * 1024


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _platform_id() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        release = Path("/proc/sys/kernel/osrelease")
        if release.is_file() and "microsoft" in release.read_text(encoding="utf-8").lower():
            return "wsl"
    return "unsupported"


def _lane_id() -> str:
    platform_id = _platform_id()
    if platform_id == "unsupported":
        raise RuntimeError("conditioning benchmark requires Windows or WSL")
    return (
        f"conditioning-{platform_id}-py{sys.version_info.major}{sys.version_info.minor}-torch-cpu"
    )


def _build_request() -> ConditioningModifierRequest:
    return ConditioningModifierRequest(
        variant=Krea2ConditioningVariant.TURBO,
        profile=SUBTLE_EXPERIMENTAL_PROFILE,
        strength=STRENGTH,
    )


def _build_input(torch: Any) -> Any:
    return torch.linspace(
        -1.0,
        1.0,
        steps=SHAPE[0] * SHAPE[1] * SHAPE[2],
        dtype=torch.float32,
        device="cpu",
    ).reshape(SHAPE)


def _execute(torch: Any, source: Any, request: ConditioningModifierRequest) -> Any:
    metadata = {"source_marker": "sigmax-krea2-conditioning-benchmark"}
    with torch.inference_mode():
        transformed, stats = transform_krea2_conditioning([[source, metadata]], request)
    if stats.input_shape != SHAPE or stats.transformed_entries != 1:
        raise RuntimeError("conditioning benchmark returned unexpected transform statistics")
    if transformed[0][1] != metadata:
        raise RuntimeError("conditioning benchmark did not preserve metadata")
    output: Any = transformed[0][0]
    if not torch.is_tensor(output) or tuple(output.shape) != SHAPE:
        raise RuntimeError("conditioning benchmark returned an unexpected tensor")
    return output


def _storage_bytes(tensor: Any) -> int:
    storage = tensor.untyped_storage()
    return int(storage.nbytes())


def _timed(function: Callable[[], Any]) -> tuple[Any, int]:
    started = time.perf_counter_ns()
    value = function()
    return value, time.perf_counter_ns() - started


def _traced_peak(function: Callable[[], Any]) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        function()
        return int(tracemalloc.get_traced_memory()[1])
    finally:
        tracemalloc.stop()


def _diagnostics_projection(diagnostics: Any) -> dict[str, object]:
    projection = dataclasses.asdict(diagnostics)
    projection["input_tap_rms"] = list(projection["input_tap_rms"])
    projection["output_tap_rms"] = list(projection["output_tap_rms"])
    return projection


def _budget_evaluation(
    *,
    metric_id: str,
    unit: PerformanceUnit,
    minimum: int,
    maximum: int,
    workload_fingerprint: str,
    lane_id: str,
    first: int,
    repeat: int,
) -> dict[str, object]:
    budget = PerformanceBudget(
        metric_id=metric_id,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        workload_fingerprint=workload_fingerprint,
    )
    evaluation = evaluate_performance_budget(
        budget=budget,
        first=PerformanceObservation(
            metric_id=metric_id,
            unit=unit,
            value=first,
            workload_fingerprint=workload_fingerprint,
            attempt="first",
            platform_lane=lane_id,
        ),
        repeat=PerformanceObservation(
            metric_id=metric_id,
            unit=unit,
            value=repeat,
            workload_fingerprint=workload_fingerprint,
            attempt="repeat",
            platform_lane=lane_id,
        ),
    )
    return {
        "evaluation": evaluation.projection(),
        "evaluation_fingerprint": evaluation.evaluation_fingerprint,
    }


def build_evidence() -> dict[str, object]:
    """Run the fixed CPU workload and return canonical benchmark evidence."""

    torch = importlib.import_module("torch")
    if not bool(torch.__version__):
        raise RuntimeError("Torch version is unavailable")
    # The workload is intentionally CPU-only; CUDA availability does not change that.
    device_note = "cpu_selected_cuda_available" if torch.cuda.is_available() else "cpu_only"

    source = _build_input(torch)
    request = _build_request()
    workload = {
        "id": WORKLOAD_ID,
        "input": {
            "dtype": str(source.dtype),
            "pattern": "torch.linspace(-1,1,inclusive,reshape)",
            "shape": list(SHAPE),
        },
        "profile": request.profile.profile_id,
        "strength": STRENGTH,
        "variant": request.variant.value,
    }
    workload_fingerprint = _fingerprint(workload)

    def execute() -> Any:
        return _execute(torch, source, request)

    first_output, first_latency_ns = _timed(execute)
    first_peak_python_bytes = _traced_peak(execute)

    for _ in range(WARMUP_COUNT):
        execute()
    repeat_samples = [_timed(execute)[1] for _ in range(REPEAT_COUNT)]
    repeat_output = execute()
    repeat_peak_python_bytes = _traced_peak(execute)
    repeat_latency_ns = int(statistics.median(repeat_samples))

    diagnostics = compute_conditioning_diagnostics(source, repeat_output)
    output_bytes = _storage_bytes(repeat_output)
    if output_bytes != source.numel() * source.element_size():
        raise RuntimeError("conditioning benchmark output storage is not dense and contiguous")

    evaluations = {
        "latency_ns": _budget_evaluation(
            metric_id="conditioning.krea2.latency",
            unit=PerformanceUnit.NANOSECONDS,
            minimum=1,
            maximum=LATENCY_MAX_NS,
            workload_fingerprint=workload_fingerprint,
            lane_id=_lane_id(),
            first=first_latency_ns,
            repeat=repeat_latency_ns,
        ),
        "output_storage_bytes": _budget_evaluation(
            metric_id="conditioning.krea2.output_storage",
            unit=PerformanceUnit.BYTES,
            minimum=output_bytes,
            maximum=output_bytes,
            workload_fingerprint=workload_fingerprint,
            lane_id=_lane_id(),
            first=output_bytes,
            repeat=_storage_bytes(first_output),
        ),
        "peak_python_bytes": _budget_evaluation(
            metric_id="conditioning.krea2.peak_python_allocation",
            unit=PerformanceUnit.BYTES,
            minimum=0,
            maximum=PEAK_PYTHON_MAX_BYTES,
            workload_fingerprint=workload_fingerprint,
            lane_id=_lane_id(),
            first=first_peak_python_bytes,
            repeat=repeat_peak_python_bytes,
        ),
    }
    verdicts: list[str] = []
    for item in evaluations.values():
        evaluation = cast(dict[str, object], item["evaluation"])
        verdicts.append(str(evaluation["verdict"]))
    overall_verdict = (
        PerformanceVerdict.PASS.value
        if verdicts and all(verdict == PerformanceVerdict.PASS.value for verdict in verdicts)
        else PerformanceVerdict.FAIL.value
    )
    evidence: dict[str, object] = {
        "schema": SCHEMA,
        "workload": workload,
        "workload_fingerprint": workload_fingerprint,
        "environment": {
            "device": "cpu",
            "device_note": device_note,
            "machine": platform.machine(),
            "platform": _platform_id(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_threads": int(torch.get_num_threads()),
        },
        "measurements": {
            "first": {
                "latency_ns": first_latency_ns,
                "peak_python_bytes": first_peak_python_bytes,
                "output_storage_bytes": _storage_bytes(first_output),
            },
            "repeat": {
                "latency_ns": repeat_latency_ns,
                "latency_samples_ns": repeat_samples,
                "peak_python_bytes": repeat_peak_python_bytes,
                "output_storage_bytes": output_bytes,
            },
        },
        "diagnostics": _diagnostics_projection(diagnostics),
        "budgets": evaluations,
        "verdict": overall_verdict,
        "limitations": [
            "CPU-only fixed synthetic tensor workload; no model weights or image-quality claim.",
            "Python tracemalloc excludes Torch native allocator internals; output storage is exact.",
            "Timing is evidence for the recorded host/interpreter, not a cross-machine SLA.",
        ],
    }
    evidence["evidence_fingerprint"] = _fingerprint(evidence)
    return evidence


def _output_path(value: str) -> Path:
    candidate = Path(value)
    resolved = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("benchmark output must stay inside the repository")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=".planning/260802-M4-12_KREA2_CONDITIONING_PERFORMANCE_EVIDENCE.json",
        help="repository-local JSON evidence path",
    )
    args = parser.parse_args(argv)
    evidence = build_evidence()
    destination = _output_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical(evidence) + b"\n")
    verdict = cast(str, evidence["verdict"])
    print(f"KREA2_CONDITIONING_BENCHMARK={verdict.upper()}")
    print(f"EVIDENCE={destination}")
    return 0 if verdict == PerformanceVerdict.PASS.value else 1


if __name__ == "__main__":
    raise SystemExit(main())
