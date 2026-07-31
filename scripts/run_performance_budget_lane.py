"""Execute fixed dependency-free M7-05 performance workloads."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

from comfyui_sigmax.core.fingerprints import numerical_fingerprint
from comfyui_sigmax.core.request_result import ScheduleResult
from comfyui_sigmax.nodes.krea2_sigma_scheduler import Krea2SigmaScheduler
from comfyui_sigmax.performance_budgets import (
    PerformanceBudget,
    PerformanceObservation,
    PerformanceUnit,
    PerformanceVerdict,
    evaluate_performance_budget,
)
from comfyui_sigmax.profiles import (
    KREA2_RAW_OFFICIAL_FULL_52,
    build_krea2_raw_schedule,
    build_krea2_turbo_schedule,
)

ROOT: Final = Path(__file__).resolve().parents[1]
SCHEMA: Final = "sigmax.performance-lane-evidence/1"
LANES: Final = {
    "performance-windows-py313": ("windows", (3, 13)),
    "performance-wsl-py310": ("wsl", (3, 10)),
}
SOURCE_PATHS: Final = (
    "comfyui_sigmax/__init__.py",
    "comfyui_sigmax/nodes/krea2_sigma_scheduler.py",
    "comfyui_sigmax/profiles/krea2_raw.py",
    "comfyui_sigmax/profiles/krea2_turbo.py",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _platform() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        release = Path("/proc/sys/kernel/osrelease")
        if release.is_file() and "microsoft" in release.read_text(encoding="utf-8").lower():
            return "wsl"
    return "unsupported"


def _source_fingerprints() -> list[dict[str, str]]:
    return [
        {
            "path": path,
            "sha256": "sha256:" + hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
        }
        for path in SOURCE_PATHS
    ]


def _latency_ns(function: Callable[[], object], *, iterations: int) -> int:
    for _ in range(10):
        function()
    samples: list[int] = []
    for _ in range(9):
        started = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter_ns() - started) // iterations)
    return int(statistics.median(samples))


def _peak_bytes(function: Callable[[], object]) -> int:
    gc.collect()
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    retained = function()
    _ = retained
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak - before


class _InstrumentedTensor:
    def __init__(self, values: Iterable[object], counters: dict[str, int]) -> None:
        self._values: tuple[object, ...] = tuple(values)
        self._counters = counters

    def tolist(self) -> list[object]:
        self._counters["round_trips"] += 1
        return list(self._values)

    def to(self, *_args: object, **_kwargs: object) -> _InstrumentedTensor:
        self._counters["transfers"] += 1
        return self


def _tensor_call(counters: dict[str, int]) -> object:
    def float_tensor(values: object) -> _InstrumentedTensor:
        counters["constructions"] += 1
        return _InstrumentedTensor(cast(Iterable[object], values), counters)

    previous = sys.modules.get("torch")
    sys.modules["torch"] = SimpleNamespace(FloatTensor=float_tensor)  # type: ignore[assignment]
    try:
        return Krea2SigmaScheduler().build("Turbo", 8, 1024, 1024, True, 0, -1)
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


def _package_startup() -> tuple[int, int]:
    command = [
        sys.executable,
        "-I",
        "-c",
        (
            "import json,sys; import comfyui_sigmax; "
            "print(json.dumps({'optional_imports':sum(name in sys.modules "
            "for name in ('torch','comfy','diffusers'))}))"
        ),
    ]
    started = time.perf_counter_ns()
    completed = subprocess.run(  # noqa: S603 - fixed current interpreter and fixed script
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    elapsed = time.perf_counter_ns() - started
    result = json.loads(completed.stdout)
    return elapsed, int(result["optional_imports"])


def _budget(
    metric_id: str,
    unit: PerformanceUnit,
    minimum: int,
    maximum: int,
    workload_fingerprint: str,
) -> PerformanceBudget:
    return PerformanceBudget(
        metric_id=metric_id,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        workload_fingerprint=workload_fingerprint,
    )


def build_evidence(lane_id: str) -> dict[str, object]:
    """Execute every fixed workload twice and evaluate exact budgets."""

    if lane_id not in LANES:
        raise RuntimeError("unknown performance lane")
    expected_platform, expected_python = LANES[lane_id]
    if _platform() != expected_platform or sys.version_info[:2] != expected_python:
        raise RuntimeError("performance lane does not match this platform/interpreter")

    def turbo() -> ScheduleResult:
        return build_krea2_turbo_schedule(steps=8, width=1024, height=1024)

    def raw() -> ScheduleResult:
        return build_krea2_raw_schedule(
            width=1360,
            height=768,
            recipe=KREA2_RAW_OFFICIAL_FULL_52,
        )

    turbo_result = turbo()
    raw_result = raw()
    sources = _source_fingerprints()
    turbo_workload = _identity(
        {
            "expected_output": numerical_fingerprint(
                turbo_result.sigmas,
                domain=turbo_result.final_domain,
                precision="float64",
            ),
            "id": "turbo-8-1024x1024",
            "sources": sources,
        }
    )
    raw_workload = _identity(
        {
            "expected_output": numerical_fingerprint(
                raw_result.sigmas,
                domain=raw_result.final_domain,
                precision="float64",
            ),
            "id": "raw-52-1360x768",
            "sources": sources,
        }
    )
    tensor_workload = _identity({"id": "tensor-turbo-8-cpu-boundary", "sources": sources})
    import_workload = _identity({"id": "isolated-package-startup", "sources": sources})

    counters_first = {"constructions": 0, "round_trips": 0, "transfers": 0}
    counters_repeat = {"constructions": 0, "round_trips": 0, "transfers": 0}
    _tensor_call(counters_first)
    _tensor_call(counters_repeat)
    startup_first = _package_startup()
    startup_repeat = _package_startup()

    measurements: list[tuple[PerformanceBudget, int, int]] = [
        (
            _budget(
                "schedule.turbo8.latency", PerformanceUnit.NANOSECONDS, 1, 1_000_000, turbo_workload
            ),
            _latency_ns(turbo, iterations=100),
            _latency_ns(turbo, iterations=100),
        ),
        (
            _budget(
                "schedule.raw52.latency", PerformanceUnit.NANOSECONDS, 1, 2_000_000, raw_workload
            ),
            _latency_ns(raw, iterations=50),
            _latency_ns(raw, iterations=50),
        ),
        (
            _budget(
                "schedule.turbo8.peak_allocation", PerformanceUnit.BYTES, 1, 131_072, turbo_workload
            ),
            _peak_bytes(turbo),
            _peak_bytes(turbo),
        ),
        (
            _budget(
                "schedule.raw52.peak_allocation", PerformanceUnit.BYTES, 1, 131_072, raw_workload
            ),
            _peak_bytes(raw),
            _peak_bytes(raw),
        ),
        (
            _budget(
                "tensor.cpu_boundary.latency",
                PerformanceUnit.NANOSECONDS,
                1,
                3_000_000,
                tensor_workload,
            ),
            _latency_ns(
                lambda: _tensor_call({"constructions": 0, "round_trips": 0, "transfers": 0}),
                iterations=25,
            ),
            _latency_ns(
                lambda: _tensor_call({"constructions": 0, "round_trips": 0, "transfers": 0}),
                iterations=25,
            ),
        ),
        (
            _budget("tensor.constructions", PerformanceUnit.COUNT, 1, 1, tensor_workload),
            counters_first["constructions"],
            counters_repeat["constructions"],
        ),
        (
            _budget("tensor.host_round_trips", PerformanceUnit.COUNT, 1, 1, tensor_workload),
            counters_first["round_trips"],
            counters_repeat["round_trips"],
        ),
        (
            _budget(
                "tensor.explicit_device_transfers", PerformanceUnit.COUNT, 0, 0, tensor_workload
            ),
            counters_first["transfers"],
            counters_repeat["transfers"],
        ),
        (
            _budget(
                "package.isolated_startup",
                PerformanceUnit.NANOSECONDS,
                1,
                1_000_000_000,
                import_workload,
            ),
            startup_first[0],
            startup_repeat[0],
        ),
        (
            _budget(
                "package.optional_framework_imports", PerformanceUnit.COUNT, 0, 0, import_workload
            ),
            startup_first[1],
            startup_repeat[1],
        ),
    ]
    rows: list[dict[str, object]] = []
    for budget, first_value, repeat_value in measurements:
        first = PerformanceObservation(
            metric_id=budget.metric_id,
            unit=budget.unit,
            value=first_value,
            workload_fingerprint=budget.workload_fingerprint,
            attempt="first",
            platform_lane=lane_id,
        )
        repeat = PerformanceObservation(
            metric_id=budget.metric_id,
            unit=budget.unit,
            value=repeat_value,
            workload_fingerprint=budget.workload_fingerprint,
            attempt="repeat",
            platform_lane=lane_id,
        )
        evaluation = evaluate_performance_budget(
            budget=budget,
            first=first,
            repeat=repeat,
        )
        rows.append(
            {
                "evaluation": evaluation.projection(),
                "evaluation_fingerprint": evaluation.evaluation_fingerprint,
                "id": budget.metric_id,
                "status": "passed" if evaluation.verdict is PerformanceVerdict.PASS else "failed",
            }
        )
    rows.sort(key=lambda row: cast(str, row["id"]))
    status = "PASS" if all(row["status"] == "passed" for row in rows) else "FAIL"
    context = {
        "lane_id": lane_id,
        "platform": expected_platform,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "sources": sources,
    }
    return {
        "context": context,
        "evidence_fingerprint": _identity({"context": context, "rows": rows}),
        "rows": rows,
        "schema": SCHEMA,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-id", required=True, choices=sorted(LANES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    evidence = build_evidence(arguments.lane_id)
    if evidence["status"] != "PASS":
        print("PERFORMANCE_BUDGET_LANE=FAIL")
        return 1
    target = arguments.output.resolve()
    if arguments.check:
        if not target.is_file():
            raise RuntimeError("performance evidence is missing")
        stored = json.loads(target.read_text(encoding="utf-8"))
        stored_rows = cast(list[dict[str, object]], stored.get("rows", []))
        evidence_rows = cast(list[dict[str, object]], evidence["rows"])
        if (
            stored.get("schema") != SCHEMA
            or stored.get("status") != "PASS"
            or stored.get("context", {}).get("lane_id") != arguments.lane_id
            or [row["id"] for row in stored_rows] != [row["id"] for row in evidence_rows]
        ):
            raise RuntimeError("stored performance evidence contract drifted")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_canonical(evidence) + b"\n")
    print("PERFORMANCE_BUDGET_LANE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
