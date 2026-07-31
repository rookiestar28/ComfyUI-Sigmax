"""Canonical human-facing views over immutable schedule and execution evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Final, cast

from comfyui_sigmax.core.artifacts import ScheduleArtifact
from comfyui_sigmax.core.execution_receipts import (
    ExecutionReceipt,
    PortableExecutionBundle,
)
from comfyui_sigmax.core.fingerprints import (
    FloatPrecision,
    canonical_projection_bytes,
    float_to_ieee_hex,
)
from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

SCHEDULE_REPORT_SCHEMA: Final = "sigmax.schedule-report/1"
SCHEDULE_REPORT_ENVELOPE_SCHEMA: Final = "sigmax.schedule-report-envelope/1"
SCHEDULE_COMPARISON_REPORT_SCHEMA: Final = "sigmax.schedule-comparison-report/1"
SCHEDULE_COMPARISON_REPORT_ENVELOPE_SCHEMA: Final = "sigmax.schedule-comparison-report-envelope/1"

_SHA256_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_HEX32_PATTERN: Final = re.compile(r"[0-9a-f]{8}")
_HEX64_PATTERN: Final = re.compile(r"[0-9a-f]{16}")
_MAX_REPORT_BYTES: Final = 1_048_576
_MAX_COMPARISON_BYTES: Final = 2_097_152
_MAX_SAMPLES: Final = 10_001
_MAX_DEPTH: Final = 32
_MAX_COLLECTION: Final = 10_001
_MAX_STRING: Final = 4096
_REPORT_FIELDS: Final = frozenset(
    {
        "artifact",
        "construction",
        "domain",
        "effective_inputs",
        "evidence",
        "execution",
        "precision",
        "receipt_present",
        "samples",
        "schema",
        "source",
    }
)
_COMPARISON_FIELDS: Final = frozenset(
    {
        "alignment",
        "comparable",
        "reason",
        "samples",
        "schema",
        "sources",
        "summary",
    }
)
_EXECUTION_FIELDS: Final = frozenset(
    {
        "compatibility",
        "counts",
        "host",
        "model",
        "reason_code",
        "receipt_fingerprint",
        "rng_ownership",
        "sampler",
        "status",
    }
)
_STATUSES: Final = frozenset({"not_executed", "succeeded", "failed", "interrupted"})


def _sha256_identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _object(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ScheduleContractError(f"{field_name} must be an array")
    return value


def _exact(value: dict[str, object], fields: frozenset[str], *, field_name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{field_name} fields do not match schema")


def _fingerprint(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ScheduleContractError(f"{field_name} must be a lowercase SHA-256 identity")
    return value


def _precision(value: object) -> FloatPrecision:
    if value not in {"float32", "float64"}:
        raise ScheduleContractError("report precision must be float32 or float64")
    return value


def _token(value: float, *, precision: FloatPrecision) -> dict[str, object]:
    return {
        "bits": float_to_ieee_hex(value, precision),
        "precision": precision,
    }


def _token_value(value: object, *, field_name: str) -> tuple[float, FloatPrecision]:
    token = _object(value, field_name=field_name)
    _exact(token, frozenset({"bits", "precision"}), field_name=field_name)
    precision = _precision(token["precision"])
    bits = token["bits"]
    pattern = _HEX32_PATTERN if precision == "float32" else _HEX64_PATTERN
    if not isinstance(bits, str) or not pattern.fullmatch(bits):
        raise ScheduleContractError(f"{field_name} bits do not match precision")
    format_code = ">f" if precision == "float32" else ">d"
    number = struct.unpack(format_code, bytes.fromhex(bits))[0]
    if not math.isfinite(number):
        raise ScheduleContractError(f"{field_name} must be finite")
    return (0.0 if number == 0.0 else number), precision


def _bound_json(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ScheduleContractError("report exceeds maximum depth")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if len(value) > _MAX_STRING:
            raise ScheduleContractError("report string exceeds limit")
        return
    if isinstance(value, list):
        if len(value) > _MAX_COLLECTION:
            raise ScheduleContractError("report collection exceeds limit")
        for child in value:
            _bound_json(child, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_COLLECTION:
            raise ScheduleContractError("report collection exceeds limit")
        for key, child in value.items():
            if not isinstance(key, str) or len(key) > _MAX_STRING:
                raise ScheduleContractError("report object key is invalid")
            _bound_json(child, depth=depth + 1)
        return
    raise ScheduleContractError("report contains an unsupported JSON value")


def _duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise ScheduleContractError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ScheduleContractError(f"non-finite JSON constant is forbidden: {value}")


def _decode_object(
    payload: bytes | str,
    *,
    maximum: int,
    canonical: bool = True,
) -> dict[str, object]:
    if isinstance(payload, str):
        if payload.startswith("\ufeff"):
            raise ScheduleContractError("report transport must not contain a BOM")
        try:
            raw = payload.encode("utf-8")
        except UnicodeError as exc:
            raise ScheduleContractError("report transport must be valid Unicode") from exc
    elif isinstance(payload, bytes):
        raw = payload
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ScheduleContractError("report transport must not contain a BOM")
    else:
        raise ScheduleContractError("report transport must be bytes or text")
    if not raw or len(raw) > maximum:
        raise ScheduleContractError("report transport size is outside the allowed range")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_duplicate_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ScheduleContractError("report transport is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ScheduleContractError("report transport root must be an object")
    projection = cast(dict[str, object], decoded)
    _bound_json(projection)
    if canonical and canonical_projection_bytes(projection) != raw:
        raise ScheduleContractError("report transport must use canonical JSON")
    return projection


def _validate_artifact_reference(value: object) -> dict[str, object]:
    artifact = _object(value, field_name="report artifact")
    _exact(
        artifact,
        frozenset({"construction_fingerprint", "numerical_fingerprint"}),
        field_name="report artifact",
    )
    _fingerprint(
        artifact["construction_fingerprint"],
        field_name="construction fingerprint",
    )
    _fingerprint(
        artifact["numerical_fingerprint"],
        field_name="numerical fingerprint",
    )
    return artifact


def _validate_counts(value: object, *, effective_steps: int) -> dict[str, object]:
    counts = _object(value, field_name="report execution counts")
    _exact(
        counts,
        frozenset(
            {
                "effective_model_evaluations",
                "effective_transitions",
                "requested_model_evaluations",
                "requested_transitions",
            }
        ),
        field_name="report execution counts",
    )
    for field_name, child in counts.items():
        if type(child) is not int or child < 0:
            raise ScheduleContractError(f"{field_name} must be a non-negative integer")
    if counts["requested_transitions"] != effective_steps:
        raise ScheduleContractError("report execution count disagrees with effective steps")
    effective_transitions = cast(int, counts["effective_transitions"])
    requested_transitions = counts["requested_transitions"]
    effective_evaluations = cast(int, counts["effective_model_evaluations"])
    requested_evaluations = cast(int, counts["requested_model_evaluations"])
    if effective_transitions > requested_transitions:
        raise ScheduleContractError("effective transitions exceed requested transitions")
    if effective_evaluations > requested_evaluations:
        raise ScheduleContractError("effective evaluations exceed requested evaluations")
    return counts


def _validate_execution(value: object, *, effective_steps: int) -> dict[str, object]:
    execution = _object(value, field_name="report execution")
    _exact(execution, _EXECUTION_FIELDS, field_name="report execution")
    _fingerprint(execution["receipt_fingerprint"], field_name="receipt fingerprint")
    status = execution["status"]
    reason = execution["reason_code"]
    if status not in _STATUSES:
        raise ScheduleContractError("report execution status is unsupported")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ScheduleContractError("report execution reason code is invalid")
    if status in {"failed", "interrupted"} and reason is None:
        raise ScheduleContractError("failed or interrupted report requires a reason code")
    if status in {"not_executed", "succeeded"} and reason is not None:
        raise ScheduleContractError("successful or unexecuted report forbids a reason code")
    counts = _validate_counts(execution["counts"], effective_steps=effective_steps)
    if status == "not_executed" and (
        counts["effective_transitions"] != 0 or counts["effective_model_evaluations"] != 0
    ):
        raise ScheduleContractError("not-executed report must have zero effective counts")
    if status == "succeeded" and (
        counts["effective_transitions"] != counts["requested_transitions"]
        or counts["effective_model_evaluations"] != counts["requested_model_evaluations"]
    ):
        raise ScheduleContractError("succeeded report must have complete effective counts")
    compatibility = _object(execution["compatibility"], field_name="report compatibility")
    _exact(
        compatibility,
        frozenset({"considered", "level", "reasons"}),
        field_name="report compatibility",
    )
    _array(compatibility["considered"], field_name="report considered capabilities")
    _array(compatibility["reasons"], field_name="report compatibility reasons")
    host = _object(execution["host"], field_name="report host")
    _exact(
        host,
        frozenset({"api_version", "id", "revision", "version"}),
        field_name="report host",
    )
    for field_name in ("model", "sampler"):
        component = _object(execution[field_name], field_name=f"report {field_name}")
        _exact(
            component,
            frozenset({"fingerprint", "id", "version"}),
            field_name=f"report {field_name}",
        )
        _fingerprint(
            component["fingerprint"],
            field_name=f"report {field_name} fingerprint",
        )
    rng = _object(execution["rng_ownership"], field_name="report RNG ownership")
    _exact(
        rng,
        frozenset({"model", "sampler", "schedule"}),
        field_name="report RNG ownership",
    )
    return execution


def _validate_schedule_projection(projection: dict[str, object]) -> bytes:
    _exact(projection, _REPORT_FIELDS, field_name="schedule report")
    if projection.get("schema") != SCHEDULE_REPORT_SCHEMA:
        raise ScheduleContractError("schedule report schema is unsupported")
    _validate_artifact_reference(projection["artifact"])
    precision = _precision(projection["precision"])
    domain = projection["domain"]
    if not isinstance(domain, str) or not domain or domain != domain.casefold():
        raise ScheduleContractError("schedule report domain is invalid")
    effective = _object(projection["effective_inputs"], field_name="effective inputs")
    _exact(
        effective,
        frozenset(
            {
                "compatibility",
                "height",
                "precision",
                "profile",
                "profile_version",
                "steps",
                "width",
            }
        ),
        field_name="effective inputs",
    )
    steps = effective.get("steps")
    if type(steps) is not int or steps < 1 or steps > _MAX_SAMPLES - 1:
        raise ScheduleContractError("effective report steps are invalid")
    if effective["precision"] != precision:
        raise ScheduleContractError("effective input precision drifted")
    source = _object(projection["source"], field_name="report source")
    _exact(
        source,
        frozenset({"id", "label", "revision"}),
        field_name="report source",
    )
    evidence = _object(projection["evidence"], field_name="report evidence")
    _exact(
        evidence,
        frozenset({"level", "reference"}),
        field_name="report evidence",
    )
    construction = _object(projection["construction"], field_name="report construction")
    _exact(
        construction,
        frozenset({"base_grid", "slicing", "terminal", "transforms"}),
        field_name="report construction",
    )
    base_grid = _object(construction["base_grid"], field_name="report base grid")
    _exact(
        base_grid,
        frozenset({"id", "parameters"}),
        field_name="report base grid",
    )
    _object(base_grid["parameters"], field_name="report base grid parameters")
    slicing = _object(construction["slicing"], field_name="report slicing")
    _exact(
        slicing,
        frozenset({"denoise", "end_step", "policy", "start_step"}),
        field_name="report slicing",
    )
    _, slicing_precision = _token_value(
        slicing["denoise"],
        field_name="report denoise",
    )
    if slicing_precision != precision:
        raise ScheduleContractError("report denoise precision drifted")
    terminal = _object(construction["terminal"], field_name="report terminal")
    _exact(
        terminal,
        frozenset({"policy", "value"}),
        field_name="report terminal",
    )
    _, terminal_precision = _token_value(
        terminal["value"],
        field_name="report terminal value",
    )
    if terminal_precision != precision:
        raise ScheduleContractError("report terminal precision drifted")
    transforms = _array(construction["transforms"], field_name="report transforms")
    for index, item in enumerate(transforms):
        transform = _object(item, field_name=f"report transform {index}")
        _exact(
            transform,
            frozenset(
                {
                    "from_domain",
                    "id",
                    "parameters",
                    "stage",
                    "stage_kind",
                    "to_domain",
                }
            ),
            field_name=f"report transform {index}",
        )
        if transform["stage"] != index:
            raise ScheduleContractError("report transform stages are not contiguous")
        _object(
            transform["parameters"],
            field_name=f"report transform parameters {index}",
        )

    samples = _array(projection["samples"], field_name="report samples")
    if len(samples) != steps + 1:
        raise ScheduleContractError("report sample count disagrees with effective steps")
    decoded: list[float] = []
    for index, item in enumerate(samples):
        sample = _object(item, field_name=f"report sample {index}")
        _exact(
            sample,
            frozenset({"delta_to_next", "index", "sigma"}),
            field_name=f"report sample {index}",
        )
        if sample["index"] != index:
            raise ScheduleContractError("report sample indices are not contiguous")
        sigma, sigma_precision = _token_value(
            sample["sigma"],
            field_name=f"report sigma {index}",
        )
        if sigma_precision != precision:
            raise ScheduleContractError("report sigma precision drifted")
        decoded.append(sigma)
    for index, sample_value in enumerate(samples):
        sample = cast(dict[str, object], sample_value)
        delta = sample["delta_to_next"]
        if index == len(samples) - 1:
            if delta is not None:
                raise ScheduleContractError("terminal report sample must not have a delta")
            continue
        delta_value, delta_precision = _token_value(
            delta,
            field_name=f"report delta {index}",
        )
        if delta_precision != precision:
            raise ScheduleContractError("report delta precision drifted")
        expected = _token(decoded[index + 1] - decoded[index], precision=precision)
        if delta != expected or not math.isfinite(delta_value):
            raise ScheduleContractError("report delta does not match adjacent sigmas")

    receipt_present = projection["receipt_present"]
    if type(receipt_present) is not bool:
        raise ScheduleContractError("receipt_present must be boolean")
    execution = projection["execution"]
    if receipt_present:
        _validate_execution(execution, effective_steps=steps)
    elif execution is not None:
        raise ScheduleContractError("construction-only report cannot contain execution evidence")
    return canonical_projection_bytes(projection)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleReport:
    """Immutable canonical schedule report."""

    report_bytes: bytes
    report_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.report_bytes, bytes):
            raise ScheduleContractError("schedule report projection must be canonical bytes")
        _fingerprint(self.report_fingerprint, field_name="schedule report fingerprint")
        projection = _decode_object(self.report_bytes, maximum=_MAX_REPORT_BYTES)
        validated = _validate_schedule_projection(projection)
        if validated != self.report_bytes:
            raise ScheduleContractError("schedule report projection is not canonical")
        if _sha256_identity(validated) != self.report_fingerprint:
            raise ScheduleContractError("schedule report fingerprint mismatch")

    def projection(self) -> dict[str, object]:
        return _decode_object(self.report_bytes, maximum=_MAX_REPORT_BYTES)


def _make_schedule_report(projection: dict[str, object]) -> ScheduleReport:
    report_bytes = _validate_schedule_projection(projection)
    return ScheduleReport(
        report_bytes=report_bytes,
        report_fingerprint=_sha256_identity(report_bytes),
    )


def build_schedule_report(
    artifact: ScheduleArtifact,
    *,
    receipt: ExecutionReceipt | None = None,
) -> ScheduleReport:
    """Build a canonical report from immutable construction and optional execution evidence."""

    if not isinstance(artifact, ScheduleArtifact):
        raise ScheduleContractError("schedule report requires a ScheduleArtifact")
    if receipt is not None and not isinstance(receipt, ExecutionReceipt):
        raise ScheduleContractError("schedule report receipt must be an ExecutionReceipt")
    if receipt is not None and (
        receipt.construction_fingerprint != artifact.construction_fingerprint
        or receipt.numerical_fingerprint != artifact.numerical_fingerprint
    ):
        raise ScheduleContractError("schedule report receipt artifact fingerprints mismatch")

    construction = artifact.construction_projection()
    numerical = artifact.numerical_projection()
    effective = _object(construction["effective"], field_name="artifact effective inputs")
    if receipt is not None:
        receipt_projection = receipt.projection()
        if receipt_projection["effective_inputs"] != effective:
            raise ScheduleContractError("schedule report receipt effective inputs mismatch")
        receipt_execution = _object(
            receipt_projection["execution"],
            field_name="receipt execution",
        )
        execution: dict[str, object] | None = {
            "compatibility": receipt_projection["compatibility"],
            "counts": receipt_projection["counts"],
            "host": receipt_projection["host"],
            "model": receipt_projection["model"],
            "reason_code": receipt_execution["reason_code"],
            "receipt_fingerprint": receipt.receipt_fingerprint,
            "rng_ownership": receipt_projection["rng_ownership"],
            "sampler": receipt_projection["sampler"],
            "status": receipt_execution["status"],
        }
    else:
        execution = None

    precision = _precision(numerical["precision"])
    sigma_bits = _array(numerical["sigmas"], field_name="artifact sigmas")
    sigmas: list[float] = []
    for index, bits in enumerate(sigma_bits):
        value, token_precision = _token_value(
            {"bits": bits, "precision": precision},
            field_name=f"artifact sigma {index}",
        )
        if token_precision != precision:  # pragma: no cover - defensive type narrowing
            raise ScheduleContractError("artifact sigma precision drifted")
        sigmas.append(value)
    samples: list[dict[str, object]] = []
    for index, sigma in enumerate(sigmas):
        delta = (
            _token(sigmas[index + 1] - sigma, precision=precision)
            if index + 1 < len(sigmas)
            else None
        )
        samples.append(
            {
                "delta_to_next": delta,
                "index": index,
                "sigma": {"bits": sigma_bits[index], "precision": precision},
            }
        )
    projection: dict[str, object] = {
        "artifact": {
            "construction_fingerprint": artifact.construction_fingerprint,
            "numerical_fingerprint": artifact.numerical_fingerprint,
        },
        "construction": {
            "base_grid": construction["base_grid"],
            "slicing": construction["slicing"],
            "terminal": construction["terminal"],
            "transforms": construction["transforms"],
        },
        "domain": numerical["domain"],
        "effective_inputs": effective,
        "evidence": construction["evidence"],
        "execution": execution,
        "precision": precision,
        "receipt_present": receipt is not None,
        "samples": samples,
        "schema": SCHEDULE_REPORT_SCHEMA,
        "source": construction["source"],
    }
    return _make_schedule_report(projection)


def build_schedule_report_from_bundle(bundle: PortableExecutionBundle) -> ScheduleReport:
    """Build a report from an already cross-linked portable bundle."""

    if not isinstance(bundle, PortableExecutionBundle):
        raise ScheduleContractError("schedule report bundle must be a PortableExecutionBundle")
    return build_schedule_report(bundle.artifact, receipt=bundle.receipt)


def serialize_schedule_report(report: ScheduleReport) -> bytes:
    if not isinstance(report, ScheduleReport):
        raise ScheduleContractError("report must be a ScheduleReport")
    return canonical_projection_bytes(
        {
            "report": report.projection(),
            "report_fingerprint": report.report_fingerprint,
            "schema": SCHEDULE_REPORT_ENVELOPE_SCHEMA,
        }
    )


def deserialize_schedule_report(payload: bytes | str) -> ScheduleReport:
    envelope = _decode_object(payload, maximum=_MAX_REPORT_BYTES)
    _exact(
        envelope,
        frozenset({"report", "report_fingerprint", "schema"}),
        field_name="schedule report envelope",
    )
    if envelope.get("schema") != SCHEDULE_REPORT_ENVELOPE_SCHEMA:
        raise ScheduleContractError("schedule report envelope schema is unsupported")
    report = _object(envelope["report"], field_name="schedule report")
    return ScheduleReport(
        report_bytes=_validate_schedule_projection(report),
        report_fingerprint=cast(str, envelope["report_fingerprint"]),
    )


def _comparison_source(report: ScheduleReport) -> dict[str, object]:
    projection = report.projection()
    execution = projection["execution"]
    receipt_fingerprint = (
        cast(dict[str, object], execution)["receipt_fingerprint"]
        if isinstance(execution, dict)
        else None
    )
    return {
        "artifact": projection["artifact"],
        "domain": projection["domain"],
        "length": len(cast(list[object], projection["samples"])),
        "precision": projection["precision"],
        "receipt_fingerprint": receipt_fingerprint,
        "report_fingerprint": report.report_fingerprint,
    }


def _report_sigmas(report: ScheduleReport) -> list[tuple[float, dict[str, object]]]:
    samples = cast(list[dict[str, object]], report.projection()["samples"])
    result: list[tuple[float, dict[str, object]]] = []
    for index, sample in enumerate(samples):
        token = cast(dict[str, object], sample["sigma"])
        value, _ = _token_value(token, field_name=f"comparison source sigma {index}")
        result.append((value, token))
    return result


def _metric(value: float) -> dict[str, object]:
    return _token(value, precision="float64")


def _comparison_rows_and_summary(
    first: list[tuple[float, dict[str, object]]],
    second: list[tuple[float, dict[str, object]]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    absolute_values: list[float] = []
    relative_values: list[float] = []
    for index, ((sigma_a, token_a), (sigma_b, token_b)) in enumerate(
        zip(first, second, strict=True)
    ):
        absolute = abs(sigma_a - sigma_b)
        denominator = max(abs(sigma_a), abs(sigma_b))
        relative = 0.0 if denominator == 0.0 else absolute / denominator
        rows.append(
            {
                "absolute_difference": _metric(absolute),
                "index": index,
                "relative_difference": _metric(relative),
                "sigma_a": token_a,
                "sigma_b": token_b,
            }
        )
        absolute_values.append(absolute)
        relative_values.append(relative)
    length = len(rows)
    maximum_absolute_index = max(range(length), key=absolute_values.__getitem__)
    maximum_relative_index = max(range(length), key=relative_values.__getitem__)
    return rows, {
        "exact_match_count": sum(value == 0.0 for value in absolute_values),
        "maximum_absolute_difference": _metric(absolute_values[maximum_absolute_index]),
        "maximum_absolute_index": maximum_absolute_index,
        "maximum_relative_difference": _metric(relative_values[maximum_relative_index]),
        "maximum_relative_index": maximum_relative_index,
        "mean_absolute_difference": _metric(math.fsum(value / length for value in absolute_values)),
        "mean_relative_difference": _metric(math.fsum(value / length for value in relative_values)),
    }


def _validate_comparison_source(value: object, *, field_name: str) -> dict[str, object]:
    source = _object(value, field_name=field_name)
    _exact(
        source,
        frozenset(
            {
                "artifact",
                "domain",
                "length",
                "precision",
                "receipt_fingerprint",
                "report_fingerprint",
            }
        ),
        field_name=field_name,
    )
    _validate_artifact_reference(source["artifact"])
    _precision(source["precision"])
    if not isinstance(source["domain"], str) or not source["domain"]:
        raise ScheduleContractError(f"{field_name} domain is invalid")
    if type(source["length"]) is not int or source["length"] < 2 or source["length"] > _MAX_SAMPLES:
        raise ScheduleContractError(f"{field_name} length is invalid")
    _fingerprint(source["report_fingerprint"], field_name=f"{field_name} report fingerprint")
    receipt = source["receipt_fingerprint"]
    if receipt is not None:
        _fingerprint(receipt, field_name=f"{field_name} receipt fingerprint")
    return source


def _validate_comparison_projection(projection: dict[str, object]) -> bytes:
    _exact(projection, _COMPARISON_FIELDS, field_name="schedule comparison report")
    if projection.get("schema") != SCHEDULE_COMPARISON_REPORT_SCHEMA:
        raise ScheduleContractError("schedule comparison report schema is unsupported")
    if type(projection["comparable"]) is not bool:
        raise ScheduleContractError("comparison comparable must be boolean")
    sources = _object(projection["sources"], field_name="comparison sources")
    _exact(sources, frozenset({"a", "b"}), field_name="comparison sources")
    source_a = _validate_comparison_source(sources["a"], field_name="comparison source a")
    source_b = _validate_comparison_source(sources["b"], field_name="comparison source b")
    alignment = _object(projection["alignment"], field_name="comparison alignment")
    rows = _array(projection["samples"], field_name="comparison samples")

    if projection["comparable"] is False:
        reason = projection["reason"]
        if reason not in {"comparison.domain_mismatch", "comparison.length_mismatch"}:
            raise ScheduleContractError("non-comparable report reason is unsupported")
        if alignment.get("kind") != "none" or rows or projection["summary"] is not None:
            raise ScheduleContractError("non-comparable report must not contain aligned results")
        if alignment.get("terminal_inclusive") is not True:
            raise ScheduleContractError("comparison alignment must remain terminal-inclusive")
        if reason == "comparison.domain_mismatch":
            _exact(
                alignment,
                frozenset({"domain_a", "domain_b", "kind", "terminal_inclusive"}),
                field_name="domain mismatch alignment",
            )
            if (
                source_a["domain"] == source_b["domain"]
                or alignment["domain_a"] != source_a["domain"]
                or alignment["domain_b"] != source_b["domain"]
            ):
                raise ScheduleContractError("domain mismatch evidence is inconsistent")
        else:
            _exact(
                alignment,
                frozenset({"kind", "length_a", "length_b", "terminal_inclusive"}),
                field_name="length mismatch alignment",
            )
            if (
                source_a["domain"] != source_b["domain"]
                or source_a["length"] == source_b["length"]
                or alignment["length_a"] != source_a["length"]
                or alignment["length_b"] != source_b["length"]
            ):
                raise ScheduleContractError("length mismatch evidence is inconsistent")
        return canonical_projection_bytes(projection)

    if projection["reason"] is not None:
        raise ScheduleContractError("comparable report cannot contain a reason")
    _exact(
        alignment,
        frozenset({"kind", "length", "terminal_inclusive"}),
        field_name="comparison alignment",
    )
    length = alignment["length"]
    if (
        alignment["kind"] != "sigma_index"
        or alignment["terminal_inclusive"] is not True
        or type(length) is not int
        or length < 2
        or length > _MAX_SAMPLES
        or len(rows) != length
    ):
        raise ScheduleContractError("comparable report alignment is invalid")
    if source_a["length"] != length or source_b["length"] != length:
        raise ScheduleContractError("comparison source lengths drifted")
    if source_a["domain"] != source_b["domain"]:
        raise ScheduleContractError("comparable report domains differ")

    first: list[tuple[float, dict[str, object]]] = []
    second: list[tuple[float, dict[str, object]]] = []
    for index, item in enumerate(rows):
        row = _object(item, field_name=f"comparison sample {index}")
        _exact(
            row,
            frozenset(
                {
                    "absolute_difference",
                    "index",
                    "relative_difference",
                    "sigma_a",
                    "sigma_b",
                }
            ),
            field_name=f"comparison sample {index}",
        )
        if row["index"] != index:
            raise ScheduleContractError("comparison sample indices are not contiguous")
        sigma_a, precision_a = _token_value(
            row["sigma_a"],
            field_name=f"comparison sigma a {index}",
        )
        sigma_b, precision_b = _token_value(
            row["sigma_b"],
            field_name=f"comparison sigma b {index}",
        )
        if precision_a != source_a["precision"] or precision_b != source_b["precision"]:
            raise ScheduleContractError("comparison source precision drifted")
        absolute, absolute_precision = _token_value(
            row["absolute_difference"],
            field_name=f"comparison absolute difference {index}",
        )
        relative, relative_precision = _token_value(
            row["relative_difference"],
            field_name=f"comparison relative difference {index}",
        )
        if absolute_precision != "float64" or relative_precision != "float64":
            raise ScheduleContractError("comparison metrics must use float64")
        denominator = max(abs(sigma_a), abs(sigma_b))
        expected_absolute = abs(sigma_a - sigma_b)
        expected_relative = 0.0 if denominator == 0.0 else expected_absolute / denominator
        if row["absolute_difference"] != _metric(expected_absolute) or absolute < 0.0:
            raise ScheduleContractError("comparison absolute difference drifted")
        if row["relative_difference"] != _metric(expected_relative) or relative < 0.0:
            raise ScheduleContractError("comparison relative difference drifted")
        first.append((sigma_a, cast(dict[str, object], row["sigma_a"])))
        second.append((sigma_b, cast(dict[str, object], row["sigma_b"])))
    expected_rows, expected_summary = _comparison_rows_and_summary(first, second)
    if rows != expected_rows or projection["summary"] != expected_summary:
        raise ScheduleContractError("comparison statistics drifted")
    return canonical_projection_bytes(projection)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleComparisonReport:
    """Immutable canonical comparison of two schedule reports."""

    report_bytes: bytes
    report_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.report_bytes, bytes):
            raise ScheduleContractError("comparison report projection must be canonical bytes")
        _fingerprint(self.report_fingerprint, field_name="comparison report fingerprint")
        projection = _decode_object(self.report_bytes, maximum=_MAX_COMPARISON_BYTES)
        validated = _validate_comparison_projection(projection)
        if validated != self.report_bytes:
            raise ScheduleContractError("comparison report projection is not canonical")
        if _sha256_identity(validated) != self.report_fingerprint:
            raise ScheduleContractError("comparison report fingerprint mismatch")

    def projection(self) -> dict[str, object]:
        return _decode_object(self.report_bytes, maximum=_MAX_COMPARISON_BYTES)


def _make_comparison_report(projection: dict[str, object]) -> ScheduleComparisonReport:
    report_bytes = _validate_comparison_projection(projection)
    return ScheduleComparisonReport(
        report_bytes=report_bytes,
        report_fingerprint=_sha256_identity(report_bytes),
    )


def build_schedule_comparison_report(
    report_a: ScheduleReport,
    report_b: ScheduleReport,
) -> ScheduleComparisonReport:
    """Compare two immutable reports without conversion, truncation, or interpolation."""

    if not isinstance(report_a, ScheduleReport) or not isinstance(report_b, ScheduleReport):
        raise ScheduleContractError("comparison requires two ScheduleReport values")
    projection_a = report_a.projection()
    projection_b = report_b.projection()
    sources = {
        "a": _comparison_source(report_a),
        "b": _comparison_source(report_b),
    }
    domain_a = projection_a["domain"]
    domain_b = projection_b["domain"]
    first = _report_sigmas(report_a)
    second = _report_sigmas(report_b)
    if domain_a != domain_b:
        projection: dict[str, object] = {
            "alignment": {
                "domain_a": domain_a,
                "domain_b": domain_b,
                "kind": "none",
                "terminal_inclusive": True,
            },
            "comparable": False,
            "reason": "comparison.domain_mismatch",
            "samples": [],
            "schema": SCHEDULE_COMPARISON_REPORT_SCHEMA,
            "sources": sources,
            "summary": None,
        }
    elif len(first) != len(second):
        projection = {
            "alignment": {
                "kind": "none",
                "length_a": len(first),
                "length_b": len(second),
                "terminal_inclusive": True,
            },
            "comparable": False,
            "reason": "comparison.length_mismatch",
            "samples": [],
            "schema": SCHEDULE_COMPARISON_REPORT_SCHEMA,
            "sources": sources,
            "summary": None,
        }
    else:
        rows, summary = _comparison_rows_and_summary(first, second)
        projection = {
            "alignment": {
                "kind": "sigma_index",
                "length": len(first),
                "terminal_inclusive": True,
            },
            "comparable": True,
            "reason": None,
            "samples": rows,
            "schema": SCHEDULE_COMPARISON_REPORT_SCHEMA,
            "sources": sources,
            "summary": summary,
        }
    return _make_comparison_report(projection)


def serialize_schedule_comparison_report(report: ScheduleComparisonReport) -> bytes:
    if not isinstance(report, ScheduleComparisonReport):
        raise ScheduleContractError("report must be a ScheduleComparisonReport")
    return canonical_projection_bytes(
        {
            "report": report.projection(),
            "report_fingerprint": report.report_fingerprint,
            "schema": SCHEDULE_COMPARISON_REPORT_ENVELOPE_SCHEMA,
        }
    )


def deserialize_schedule_comparison_report(
    payload: bytes | str,
) -> ScheduleComparisonReport:
    envelope = _decode_object(payload, maximum=_MAX_COMPARISON_BYTES)
    _exact(
        envelope,
        frozenset({"report", "report_fingerprint", "schema"}),
        field_name="schedule comparison report envelope",
    )
    if envelope.get("schema") != SCHEDULE_COMPARISON_REPORT_ENVELOPE_SCHEMA:
        raise ScheduleContractError("schedule comparison report envelope schema is unsupported")
    report = _object(envelope["report"], field_name="schedule comparison report")
    return ScheduleComparisonReport(
        report_bytes=_validate_comparison_projection(report),
        report_fingerprint=cast(str, envelope["report_fingerprint"]),
    )
