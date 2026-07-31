from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.performance_budgets import (
    PerformanceBudget,
    PerformanceObservation,
    PerformanceUnit,
    PerformanceVerdict,
    evaluate_performance_budget,
)

WORKLOAD = "sha256:" + "a" * 64


def _budget(maximum: int = 1_000) -> PerformanceBudget:
    return PerformanceBudget(
        metric_id="schedule.turbo8.latency",
        unit=PerformanceUnit.NANOSECONDS,
        minimum=1,
        maximum=maximum,
        workload_fingerprint=WORKLOAD,
    )


def _observation(*, attempt: str, value: int) -> PerformanceObservation:
    return PerformanceObservation(
        metric_id="schedule.turbo8.latency",
        unit=PerformanceUnit.NANOSECONDS,
        value=value,
        workload_fingerprint=WORKLOAD,
        attempt=attempt,
        platform_lane="windows.py313",
    )


def test_repeated_observations_within_budget_pass_deterministically() -> None:
    evaluation = evaluate_performance_budget(
        budget=_budget(),
        first=_observation(attempt="first", value=800),
        repeat=_observation(attempt="repeat", value=900),
    )

    assert evaluation.verdict is PerformanceVerdict.PASS
    assert evaluation.projection()["schema"] == "sigmax.performance-budget-evaluation/1"
    assert evaluation.evaluation_fingerprint.startswith("sha256:")
    assert evaluation.evaluation_fingerprint == evaluation.evaluation_fingerprint


def test_any_over_budget_attempt_fails() -> None:
    evaluation = evaluate_performance_budget(
        budget=_budget(),
        first=_observation(attempt="first", value=1_001),
        repeat=_observation(attempt="repeat", value=900),
    )

    assert evaluation.verdict is PerformanceVerdict.FAIL


def test_exact_zero_count_budget_is_supported() -> None:
    budget = PerformanceBudget(
        metric_id="tensor.explicit_device_transfers",
        unit=PerformanceUnit.COUNT,
        minimum=0,
        maximum=0,
        workload_fingerprint=WORKLOAD,
    )
    first = replace(
        _observation(attempt="first", value=0),
        metric_id=budget.metric_id,
        unit=budget.unit,
    )
    repeat = replace(first, attempt="repeat")

    assert (
        evaluate_performance_budget(budget=budget, first=first, repeat=repeat).verdict
        is PerformanceVerdict.PASS
    )


def test_contracts_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        _budget().maximum = 2_000  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_budget(), metric_id="bad"),
        lambda: replace(_budget(), maximum=True),
        lambda: replace(_budget(), minimum=2_000),
        lambda: replace(_budget(), workload_fingerprint="sha256:bad"),
        lambda: replace(_observation(attempt="first", value=1), attempt="warmup"),
        lambda: replace(_observation(attempt="first", value=1), value=-1),
    ],
)
def test_invalid_budget_and_observation_contracts_fail_closed(factory: object) -> None:
    with pytest.raises(ScheduleContractError):
        factory()  # type: ignore[operator]


def test_mismatched_or_misordered_observations_fail_closed() -> None:
    first = _observation(attempt="first", value=1)
    repeat = _observation(attempt="repeat", value=1)

    with pytest.raises(ScheduleContractError, match="ordered first/repeat"):
        evaluate_performance_budget(budget=_budget(), first=repeat, repeat=first)
    with pytest.raises(ScheduleContractError, match="disagrees with budget"):
        evaluate_performance_budget(
            budget=_budget(),
            first=replace(first, workload_fingerprint="sha256:" + "b" * 64),
            repeat=repeat,
        )
