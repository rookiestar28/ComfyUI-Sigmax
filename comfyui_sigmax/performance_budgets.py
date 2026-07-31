"""Immutable integer-unit performance budgets and repeated observations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

PERFORMANCE_OBSERVATION_SCHEMA: Final = "sigmax.performance-observation/1"
PERFORMANCE_EVALUATION_SCHEMA: Final = "sigmax.performance-budget-evaluation/1"
_IDENTIFIER: Final = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+){1,15}")
_FINGERPRINT: Final = re.compile(r"sha256:[0-9a-f]{64}")


class PerformanceUnit(str, Enum):
    """Stable integer measurement units."""

    BYTES = "bytes"
    COUNT = "count"
    NANOSECONDS = "nanoseconds"


class PerformanceVerdict(str, Enum):
    """Budget evaluation result."""

    PASS = "pass"  # noqa: S105 - public verdict token, not a credential
    FAIL = "fail"


def _identity(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ScheduleContractError(f"{label} must be a stable namespaced identifier")
    return value


def _fingerprint(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT.fullmatch(value):
        raise ScheduleContractError(f"{label} must be a lowercase SHA-256 identity")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ScheduleContractError(f"{label} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceBudget:
    """One immutable upper bound for a fixed workload."""

    metric_id: str
    unit: PerformanceUnit
    minimum: int
    maximum: int
    workload_fingerprint: str

    def __post_init__(self) -> None:
        _identifier(self.metric_id, label="performance metric ID")
        if not isinstance(self.unit, PerformanceUnit):
            raise ScheduleContractError("performance unit is invalid")
        _integer(self.minimum, label="performance minimum")
        _integer(self.maximum, label="performance maximum")
        if self.minimum > self.maximum:
            raise ScheduleContractError("performance minimum exceeds maximum")
        _fingerprint(self.workload_fingerprint, label="performance workload fingerprint")

    def projection(self) -> dict[str, object]:
        return {
            "maximum": self.maximum,
            "metric_id": self.metric_id,
            "minimum": self.minimum,
            "unit": self.unit.value,
            "workload_fingerprint": self.workload_fingerprint,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceObservation:
    """One measured integer value for a fixed attempt and platform lane."""

    metric_id: str
    unit: PerformanceUnit
    value: int
    workload_fingerprint: str
    attempt: str
    platform_lane: str

    def __post_init__(self) -> None:
        _identifier(self.metric_id, label="performance metric ID")
        if not isinstance(self.unit, PerformanceUnit):
            raise ScheduleContractError("performance unit is invalid")
        _integer(self.value, label="performance observation")
        _fingerprint(self.workload_fingerprint, label="performance workload fingerprint")
        if self.attempt not in {"first", "repeat"}:
            raise ScheduleContractError("performance attempt must be first or repeat")
        _identifier(self.platform_lane, label="performance platform lane")

    def projection(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "metric_id": self.metric_id,
            "platform_lane": self.platform_lane,
            "schema": PERFORMANCE_OBSERVATION_SCHEMA,
            "unit": self.unit.value,
            "value": self.value,
            "workload_fingerprint": self.workload_fingerprint,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceBudgetEvaluation:
    """First/repeat evaluation against one exact budget."""

    budget: PerformanceBudget
    observations: tuple[PerformanceObservation, PerformanceObservation]
    verdict: PerformanceVerdict

    def __post_init__(self) -> None:
        if not isinstance(self.budget, PerformanceBudget):
            raise ScheduleContractError("performance evaluation budget is invalid")
        if not isinstance(self.observations, tuple) or len(self.observations) != 2:
            raise ScheduleContractError("performance evaluation requires first/repeat observations")
        first, repeat = self.observations
        if not all(isinstance(item, PerformanceObservation) for item in self.observations):
            raise ScheduleContractError("performance observations are invalid")
        if (first.attempt, repeat.attempt) != ("first", "repeat"):
            raise ScheduleContractError("performance observations must be ordered first/repeat")
        for observation in self.observations:
            if (
                observation.metric_id != self.budget.metric_id
                or observation.unit is not self.budget.unit
                or observation.workload_fingerprint != self.budget.workload_fingerprint
            ):
                raise ScheduleContractError("performance observation disagrees with budget")
        if first.platform_lane != repeat.platform_lane:
            raise ScheduleContractError("performance observations must use one platform lane")
        expected = (
            PerformanceVerdict.PASS
            if all(
                self.budget.minimum <= item.value <= self.budget.maximum
                for item in self.observations
            )
            else PerformanceVerdict.FAIL
        )
        if self.verdict is not expected:
            raise ScheduleContractError("performance verdict disagrees with observations")

    @property
    def evaluation_fingerprint(self) -> str:
        return _identity(self.projection())

    def projection(self) -> dict[str, object]:
        return {
            "budget": self.budget.projection(),
            "observations": [item.projection() for item in self.observations],
            "schema": PERFORMANCE_EVALUATION_SCHEMA,
            "verdict": self.verdict.value,
        }


def evaluate_performance_budget(
    *,
    budget: PerformanceBudget,
    first: PerformanceObservation,
    repeat: PerformanceObservation,
) -> PerformanceBudgetEvaluation:
    """Evaluate two ordered observations without timing or host side effects."""

    if not isinstance(budget, PerformanceBudget):
        raise ScheduleContractError("performance budget is invalid")
    observations = (first, repeat)
    values_are_within_budget = all(
        isinstance(item, PerformanceObservation) and budget.minimum <= item.value <= budget.maximum
        for item in observations
    )
    return PerformanceBudgetEvaluation(
        budget=budget,
        observations=observations,
        verdict=(PerformanceVerdict.PASS if values_are_within_budget else PerformanceVerdict.FAIL),
    )


__all__ = [
    "PERFORMANCE_EVALUATION_SCHEMA",
    "PERFORMANCE_OBSERVATION_SCHEMA",
    "PerformanceBudget",
    "PerformanceBudgetEvaluation",
    "PerformanceObservation",
    "PerformanceUnit",
    "PerformanceVerdict",
    "evaluate_performance_budget",
]
