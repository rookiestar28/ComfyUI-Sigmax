"""Strict loader for the packaged performance budget matrix."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Final, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError
from comfyui_sigmax.performance_budgets import (
    PerformanceBudget,
    PerformanceObservation,
    PerformanceUnit,
    PerformanceVerdict,
    evaluate_performance_budget,
)

PERFORMANCE_MATRIX_SCHEMA: Final = "sigmax.performance-budget-matrix/1"
PERFORMANCE_MATRIX_ENVELOPE_SCHEMA: Final = "sigmax.performance-budget-matrix-envelope/1"
_MAX_BYTES: Final = 262_144
_FINGERPRINT: Final = re.compile(r"sha256:[0-9a-f]{64}")
_ROW_FIELDS: Final = frozenset(
    {
        "evaluation",
        "evaluation_fingerprint",
        "evidence_source",
        "id",
        "source_evidence_fingerprint",
        "status",
    }
)
_EXPECTED_EXCLUSIONS: Final = ["gpu", "latest_host", "model_weights", "official_container"]


class PerformanceMatrixError(ScheduleContractError):
    """Raised when performance evidence is malformed or semantically invalid."""


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


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PerformanceMatrixError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise PerformanceMatrixError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise PerformanceMatrixError(f"non-finite JSON value is forbidden: {value}")


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PerformanceMatrixError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise PerformanceMatrixError(f"{label} must be an array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise PerformanceMatrixError(f"{label} must be bounded non-empty text")
    if re.match(r"(?:[A-Za-z]:[\\/]|/|\\\\)", value):
        raise PerformanceMatrixError(f"{label} contains a private or absolute path")
    return value


def _fingerprint(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT.fullmatch(value):
        raise PerformanceMatrixError(f"{label} must be a lowercase SHA-256 identity")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PerformanceMatrixError(f"{label} must be a non-negative integer")
    return value


def _decode(payload: bytes | str) -> dict[str, object]:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_BYTES:
        raise PerformanceMatrixError("performance matrix transport size is invalid")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PerformanceMatrixError("performance matrix must not contain a BOM")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PerformanceMatrixError("performance matrix is not valid JSON") from exc
    root = _object(decoded, label="performance matrix envelope")
    if _canonical(root) + b"\n" != raw:
        raise PerformanceMatrixError("performance matrix must use canonical JSON")
    return root


def _evaluation(value: object, *, row_id: str) -> tuple[dict[str, object], str]:
    evaluation = _object(value, label=f"performance result {row_id} evaluation")
    if set(evaluation) != {"budget", "observations", "schema", "verdict"}:
        raise PerformanceMatrixError("performance evaluation fields do not match schema")
    if evaluation["schema"] != "sigmax.performance-budget-evaluation/1":
        raise PerformanceMatrixError("performance evaluation schema is unsupported")
    budget_value = _object(evaluation["budget"], label="performance budget")
    if set(budget_value) != {
        "maximum",
        "metric_id",
        "minimum",
        "unit",
        "workload_fingerprint",
    }:
        raise PerformanceMatrixError("performance budget fields do not match schema")
    try:
        unit = PerformanceUnit(_text(budget_value["unit"], label="performance unit"))
        budget = PerformanceBudget(
            metric_id=_text(budget_value["metric_id"], label="performance metric ID"),
            unit=unit,
            minimum=_integer(budget_value["minimum"], label="performance minimum"),
            maximum=_integer(budget_value["maximum"], label="performance maximum"),
            workload_fingerprint=_fingerprint(
                budget_value["workload_fingerprint"], label="performance workload"
            ),
        )
    except (ValueError, ScheduleContractError) as exc:
        raise PerformanceMatrixError("performance budget is semantically invalid") from exc
    observations_value = _array(evaluation["observations"], label="performance observations")
    if len(observations_value) != 2:
        raise PerformanceMatrixError("performance evidence requires first/repeat observations")
    observations: list[PerformanceObservation] = []
    expected_fields = {
        "attempt",
        "metric_id",
        "platform_lane",
        "schema",
        "unit",
        "value",
        "workload_fingerprint",
    }
    for observation_value in observations_value:
        item = _object(observation_value, label="performance observation")
        if set(item) != expected_fields or item["schema"] != "sigmax.performance-observation/1":
            raise PerformanceMatrixError("performance observation fields or schema drifted")
        try:
            observations.append(
                PerformanceObservation(
                    metric_id=_text(item["metric_id"], label="performance metric ID"),
                    unit=PerformanceUnit(_text(item["unit"], label="performance unit")),
                    value=_integer(item["value"], label="performance value"),
                    workload_fingerprint=_fingerprint(
                        item["workload_fingerprint"], label="performance workload"
                    ),
                    attempt=_text(item["attempt"], label="performance attempt"),
                    platform_lane=_text(item["platform_lane"], label="performance platform lane"),
                )
            )
        except (ValueError, ScheduleContractError) as exc:
            raise PerformanceMatrixError("performance observation is semantically invalid") from exc
    try:
        verdict = PerformanceVerdict(_text(evaluation["verdict"], label="performance verdict"))
        validated = evaluate_performance_budget(
            budget=budget,
            first=observations[0],
            repeat=observations[1],
        )
    except (ValueError, ScheduleContractError) as exc:
        raise PerformanceMatrixError("performance evaluation is semantically invalid") from exc
    if verdict is not validated.verdict or validated.verdict is not PerformanceVerdict.PASS:
        raise PerformanceMatrixError("only within-budget performance evidence may pass")
    if evaluation != validated.projection():
        raise PerformanceMatrixError("performance evaluation projection drifted")
    return evaluation, validated.evaluation_fingerprint


def _validate_matrix(value: object) -> dict[str, object]:
    matrix = _object(value, label="performance matrix")
    if set(matrix) != {"exclusions", "policy", "results", "schema", "sources"}:
        raise PerformanceMatrixError("performance matrix fields do not match schema")
    if matrix["schema"] != PERFORMANCE_MATRIX_SCHEMA:
        raise PerformanceMatrixError("performance matrix schema is unsupported")
    if matrix["policy"] != {
        "observations_machine_specific": True,
        "thresholds_are_regression_limits": True,
        "wall_clock_portable_guarantee": False,
    }:
        raise PerformanceMatrixError("performance policy drifted")
    exclusions = _array(matrix["exclusions"], label="performance exclusions")
    exclusion_ids: list[str] = []
    for value in exclusions:
        exclusion = _object(value, label="performance exclusion")
        if set(exclusion) != {"id", "reason", "status"} or exclusion["status"] != "not_evaluated":
            raise PerformanceMatrixError("performance exclusion is invalid")
        exclusion_ids.append(_text(exclusion["id"], label="performance exclusion ID"))
        _text(exclusion["reason"], label="performance exclusion reason")
    if exclusion_ids != _EXPECTED_EXCLUSIONS:
        raise PerformanceMatrixError("performance exclusion inventory drifted")
    sources = _array(matrix["sources"], label="performance sources")
    source_paths: list[str] = []
    for value in sources:
        source = _object(value, label="performance source")
        if set(source) != {"path", "sha256"}:
            raise PerformanceMatrixError("performance source fields do not match schema")
        source_paths.append(_text(source["path"], label="performance source path"))
        _fingerprint(source["sha256"], label="performance source fingerprint")
    if source_paths != sorted(set(source_paths)):
        raise PerformanceMatrixError("performance sources must be unique and sorted")
    results = _array(matrix["results"], label="performance results")
    if len(results) != 21:
        raise PerformanceMatrixError("performance result inventory is incomplete")
    result_ids: list[str] = []
    for value in results:
        row = _object(value, label="performance result")
        if set(row) != _ROW_FIELDS or row["status"] != "passed":
            raise PerformanceMatrixError("only exact passed performance rows may be published")
        row_id = _text(row["id"], label="performance result ID")
        result_ids.append(row_id)
        evidence_source = _text(row["evidence_source"], label="performance evidence source")
        if evidence_source not in source_paths:
            raise PerformanceMatrixError("performance evidence source is undeclared")
        _fingerprint(row["source_evidence_fingerprint"], label="source evidence fingerprint")
        _, expected_fingerprint = _evaluation(row["evaluation"], row_id=row_id)
        if (
            _fingerprint(row["evaluation_fingerprint"], label="performance evaluation fingerprint")
            != expected_fingerprint
        ):
            raise PerformanceMatrixError("performance evaluation fingerprint drifted")
    if result_ids != sorted(set(result_ids)):
        raise PerformanceMatrixError("performance result IDs must be unique and sorted")
    return matrix


@dataclass(frozen=True, slots=True)
class PerformanceBudgetMatrix:
    """Validated immutable performance matrix."""

    _matrix: dict[str, object]
    matrix_fingerprint: str

    def projection(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(_canonical(self._matrix)))

    def require_result(self, result_id: str) -> dict[str, object]:
        for row in cast(list[dict[str, object]], self._matrix["results"]):
            if row["id"] == result_id:
                return cast(dict[str, object], json.loads(_canonical(row)))
        raise PerformanceMatrixError("unknown performance result")


def load_performance_budget_matrix(
    payload: bytes | str | None = None,
) -> PerformanceBudgetMatrix:
    """Load the packaged matrix or caller-supplied canonical transport."""

    if payload is None:
        payload = (
            importlib.resources.files("comfyui_sigmax.performance")
            .joinpath("matrix_v1.json")
            .read_bytes()
        )
    envelope = _decode(payload)
    if set(envelope) != {"matrix", "matrix_fingerprint", "schema"}:
        raise PerformanceMatrixError("performance envelope fields do not match schema")
    if envelope["schema"] != PERFORMANCE_MATRIX_ENVELOPE_SCHEMA:
        raise PerformanceMatrixError("performance envelope schema is unsupported")
    matrix = _validate_matrix(envelope["matrix"])
    observed = _fingerprint(envelope["matrix_fingerprint"], label="performance matrix fingerprint")
    if observed != _identity(matrix):
        raise PerformanceMatrixError("performance matrix fingerprint drifted")
    return PerformanceBudgetMatrix(matrix, observed)


__all__ = [
    "PERFORMANCE_MATRIX_ENVELOPE_SCHEMA",
    "PERFORMANCE_MATRIX_SCHEMA",
    "PerformanceBudgetMatrix",
    "PerformanceMatrixError",
    "load_performance_budget_matrix",
]
