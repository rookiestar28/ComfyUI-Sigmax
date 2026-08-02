"""Strict loader for the packaged capability-filtered numerical benchmark matrix."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Final, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

NUMERICAL_BENCHMARK_MATRIX_SCHEMA: Final = "sigmax.numerical-benchmark-matrix/1"
NUMERICAL_BENCHMARK_MATRIX_ENVELOPE_SCHEMA: Final = "sigmax.numerical-benchmark-matrix-envelope/1"
_MAX_BYTES: Final = 1_048_576
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}")
_DECIMAL: Final = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e-[0-9]+)?")
_REVISION_CHUNK: Final = re.compile(r"[0-9a-f]{8}")
_PRIVATE_PATH: Final = re.compile(r"(?:[A-Za-z]:[\\/]|/|\\\\)")
_SECRET_WORDS: Final = ("authorization", "cookie", "credential", "password", "secret", "token")
_MATRIX_FIELDS: Final = frozenset(
    {"coverage", "exclusions", "results", "schema", "sources", "weight_variants"}
)
_ROW_FIELDS: Final = frozenset(
    {
        "baselines",
        "capability",
        "determinism",
        "evidence",
        "execution",
        "id",
        "lane",
        "model_weights_present",
        "profile",
        "repeat",
        "runtime",
        "schedule",
        "weight_variant",
        "workload",
    }
)
_SOURCE_PATHS: Final = (
    "comfyui_sigmax/workflows/fixtures.json",
    "tests/benchmarks/fixtures/known_good_host_attempts_v1.json",
    "tests/conformance/fixtures/capability_receipt_conformance_v1.json",
    "tests/parity/fixtures/krea2_native_euler_parity_v1.json",
    "tests/parity/fixtures/krea2_raw_parity_v1.json",
    "tests/parity/fixtures/krea2_turbo_parity_v1.json",
    "tests/golden/aura_flow_v0_2.json",
    "tests/golden/anima_v1.json",
    "tests/golden/wan_v1.json",
)
_SOURCE_SCHEMAS: Final = {
    "comfyui_sigmax/workflows/fixtures.json": ("sigmax.workflow-fixture-bundle/1", None),
    "tests/benchmarks/fixtures/known_good_host_attempts_v1.json": (
        "sigmax.known-good-host-attempts/1",
        "PASS",
    ),
    "tests/conformance/fixtures/capability_receipt_conformance_v1.json": (
        "sigmax.capability-receipt-conformance/1",
        "PASS",
    ),
    "tests/parity/fixtures/krea2_native_euler_parity_v1.json": (
        "sigmax.krea2-native-euler-parity/1",
        "PASS",
    ),
    "tests/parity/fixtures/krea2_raw_parity_v1.json": (
        "sigmax.krea2-raw-parity/1",
        "PASS",
    ),
    "tests/parity/fixtures/krea2_turbo_parity_v1.json": (
        "sigmax.krea2-turbo-parity/1",
        "PASS",
    ),
    "tests/golden/aura_flow_v0_2.json": ("sigmax.aura-flow-golden/1", None),
    "tests/golden/anima_v1.json": ("sigmax.anima-golden/1", None),
    "tests/golden/wan_v1.json": ("sigmax.wan-golden/1", None),
}
_SOURCE_IDS: Final = frozenset(
    {
        "host.known_good",
        "parity.anima",
        "parity.native_euler",
        "parity.raw",
        "parity.turbo",
        "parity.wan",
        "workflow.fixtures",
    }
)
_LANES: Final = (
    "anima_schedule_parity",
    "h2_workflow",
    "h3_native_euler",
    "raw_schedule_parity",
    "turbo_schedule_parity",
    "wan_schedule_parity",
)
_EXCLUSIONS: Final = [
    "advanced_workflows",
    "gpu_model_weights",
    "partial_denoise_execution",
    "resume_execution",
    "stochastic_euler_execution",
]


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


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise ScheduleContractError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ScheduleContractError(f"non-finite JSON value is forbidden: {value}")


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ScheduleContractError(f"{name} must be an array")
    return value


def _exact(value: dict[str, object], fields: frozenset[str], *, name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{name} fields do not match schema")


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ScheduleContractError(f"{name} must be bounded non-empty text")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ScheduleContractError(f"{name} must be an integer >= {minimum}")
    return value


def _fingerprint(value: object, *, name: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ScheduleContractError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _metric(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if not _DECIMAL.fullmatch(text):
        raise ScheduleContractError(f"{name} must be a canonical non-negative decimal string")
    return text


def _scan_safe(value: object, *, depth: int = 0) -> None:
    if depth > 24:
        raise ScheduleContractError("benchmark matrix exceeds maximum depth")
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, str):
        if len(value) > 4096:
            raise ScheduleContractError("benchmark matrix string exceeds limit")
        if _PRIVATE_PATH.match(value):
            raise ScheduleContractError("benchmark matrix contains a private or absolute path")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise ScheduleContractError("benchmark matrix collection exceeds limit")
        for child in value:
            _scan_safe(child, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 256:
            raise ScheduleContractError("benchmark matrix collection exceeds limit")
        for key, child in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in _SECRET_WORDS):
                raise ScheduleContractError("benchmark matrix contains a forbidden field name")
            _scan_safe(child, depth=depth + 1)
        return
    raise ScheduleContractError("benchmark matrix contains an unsupported JSON value")


def _decode(payload: bytes | str) -> dict[str, object]:
    if isinstance(payload, str):
        if payload.startswith("\ufeff"):
            raise ScheduleContractError("benchmark matrix must not contain a BOM")
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
        if raw.startswith(b"\xef\xbb\xbf"):
            raise ScheduleContractError("benchmark matrix must not contain a BOM")
    else:
        raise ScheduleContractError("benchmark matrix transport must be bytes or text")
    if not raw or len(raw) > _MAX_BYTES:
        raise ScheduleContractError("benchmark matrix transport size is outside the allowed range")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ScheduleContractError("benchmark matrix transport is not valid JSON") from exc
    root = _object(decoded, name="benchmark matrix envelope")
    _scan_safe(root)
    if _canonical(root) + b"\n" != raw:
        raise ScheduleContractError("benchmark matrix transport must use canonical JSON")
    return root


def _validate_dimensions(value: object, *, name: str) -> dict[str, object]:
    dimensions = _object(value, name=name)
    _exact(dimensions, frozenset({"height", "transitions", "width"}), name=name)
    _integer(dimensions["height"], name=f"{name} height", minimum=16)
    _integer(dimensions["width"], name=f"{name} width", minimum=16)
    _integer(dimensions["transitions"], name=f"{name} transitions", minimum=1)
    return dimensions


def _validate_metric_set(value: object, *, name: str) -> None:
    metrics = _object(value, name=name)
    _exact(
        metrics,
        frozenset(
            {
                "device",
                "dtype",
                "fingerprint",
                "max_abs_error",
                "mean_abs_error",
                "status",
                "tolerance",
            }
        ),
        name=name,
    )
    if metrics["status"] != "PASS":
        raise ScheduleContractError(f"{name} must retain PASS source evidence")
    _text(metrics["device"], name=f"{name} device")
    _text(metrics["dtype"], name=f"{name} dtype")
    _fingerprint(metrics["fingerprint"], name=f"{name} fingerprint")
    for field in ("max_abs_error", "mean_abs_error", "tolerance"):
        _metric(metrics[field], name=f"{name} {field}")


def _validate_row(value: object) -> tuple[str, str]:
    row = _object(value, name="benchmark result")
    _exact(row, _ROW_FIELDS, name="benchmark result")
    identifier = _text(row["id"], name="benchmark result id")
    lane = _text(row["lane"], name="benchmark result lane")
    if lane not in _LANES:
        raise ScheduleContractError("benchmark result lane is unsupported")
    capability = _object(row["capability"], name="benchmark capability")
    _exact(capability, frozenset({"level", "reasons"}), name="benchmark capability")
    if capability != {"level": "allow", "reasons": ["compatible"]}:
        raise ScheduleContractError("verified benchmark results must be capability allow")
    if row["model_weights_present"] is not False or row["weight_variant"] != "none":
        raise ScheduleContractError("verified benchmark results must not imply model weights")

    profile = _object(row["profile"], name="benchmark profile")
    _exact(
        profile,
        frozenset({"evidence", "id", "recipe", "variant", "version"}),
        name="benchmark profile",
    )
    if profile["id"] not in {
        "anima.aesthetic.framework-reference",
        "anima.base.framework-reference",
        "anima.turbo.framework-reference",
        "wan2.1.t2v.comfy-native",
        "wan2.1.t2v.official-native",
        "wan2.1.i2v.480p.official-native",
        "wan2.1.i2v.720p.official-native",
        "wan2.1.t2v.diffusers-reference",
        "wan2.1.i2v.480p.diffusers-reference",
        "wan2.1.i2v.720p.diffusers-reference",
        "wan2.2.ti2v.5b.comfy-native",
        "wan2.2.t2v-a14b.official-native",
        "wan2.2.i2v-a14b.official-native",
        "wan2.2.ti2v.5b.diffusers-reference",
        "wan2.2.t2v-a14b.diffusers-reference",
        "wan2.2.i2v-a14b.diffusers-reference",
        "krea2.raw.official",
        "krea2.turbo.official",
    }:
        raise ScheduleContractError("benchmark profile is unsupported")
    if (
        profile["variant"] not in {"Aesthetic", "Base", "RAW", "Turbo", "Wan"}
        or profile["version"] != "1"
    ):
        raise ScheduleContractError("benchmark profile variant/version is invalid")
    if profile["evidence"] not in {"official", "framework_reference", "modified"}:
        raise ScheduleContractError("benchmark profile evidence is invalid")
    _text(profile["recipe"], name="benchmark recipe")

    workload = _object(row["workload"], name="benchmark workload")
    _exact(workload, frozenset({"effective", "requested"}), name="benchmark workload")
    requested = _validate_dimensions(workload["requested"], name="requested workload")
    effective = _validate_dimensions(workload["effective"], name="effective workload")
    if requested["transitions"] != effective["transitions"]:
        raise ScheduleContractError("benchmark transition count drifted")
    if cast(int, effective["width"]) % 16 or cast(int, effective["height"]) % 16:
        raise ScheduleContractError("effective benchmark geometry must be divisible by 16")

    execution = _object(row["execution"], name="benchmark execution")
    _exact(
        execution,
        frozenset({"counts", "rng_ownership", "status"}),
        name="benchmark execution",
    )
    status = execution["status"]
    if status not in {"not_executed", "succeeded"}:
        raise ScheduleContractError("benchmark execution status is invalid")
    counts = _object(execution["counts"], name="benchmark execution counts")
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
        name="benchmark execution counts",
    )
    for field in counts:
        _integer(counts[field], name=f"benchmark {field}")
    if counts["requested_transitions"] != requested["transitions"]:
        raise ScheduleContractError("benchmark requested transitions do not match workload")
    if status == "not_executed" and (
        counts["effective_transitions"] != 0 or counts["effective_model_evaluations"] != 0
    ):
        raise ScheduleContractError("not-executed benchmark has effective execution counts")
    if status == "succeeded" and (
        counts["effective_transitions"] != requested["transitions"]
        or counts["effective_model_evaluations"] != requested["transitions"]
    ):
        raise ScheduleContractError("succeeded benchmark counts are incomplete")
    rng = _object(execution["rng_ownership"], name="benchmark RNG ownership")
    _exact(rng, frozenset({"model", "sampler", "schedule"}), name="benchmark RNG ownership")
    if set(rng.values()) != {"none"}:
        raise ScheduleContractError("verified deterministic benchmark must have no RNG owner")

    runtime = _object(row["runtime"], name="benchmark runtime")
    _exact(
        runtime,
        frozenset(
            {
                "comfyui",
                "comfyui_revision_chunks",
                "device",
                "diffusers",
                "dtype",
                "numpy",
                "python",
                "sigmax_revision_chunks",
                "torch",
            }
        ),
        name="benchmark runtime",
    )
    for field, item in runtime.items():
        if field.endswith("_revision_chunks"):
            if item is not None:
                chunks = _array(item, name=f"benchmark runtime {field}")
                if len(chunks) != 5 or any(
                    not isinstance(chunk, str) or not _REVISION_CHUNK.fullmatch(chunk)
                    for chunk in chunks
                ):
                    raise ScheduleContractError(f"benchmark runtime {field} is invalid")
            continue
        if item is not None:
            _text(item, name=f"benchmark runtime {field}")

    schedule = _object(row["schedule"], name="benchmark schedule")
    _exact(schedule, frozenset({"image_seq_len", "mu"}), name="benchmark schedule")
    if schedule["image_seq_len"] is not None:
        _integer(schedule["image_seq_len"], name="benchmark image sequence length", minimum=1)
    _metric(schedule["mu"], name="benchmark mu")

    evidence = _object(row["evidence"], name="benchmark evidence")
    _exact(
        evidence,
        frozenset({"artifact", "receipt_fingerprint", "source_ids"}),
        name="benchmark evidence",
    )
    artifact = _object(evidence["artifact"], name="benchmark artifact")
    _exact(
        artifact,
        frozenset({"construction_fingerprint", "numerical_fingerprint"}),
        name="benchmark artifact",
    )
    _fingerprint(
        artifact["construction_fingerprint"],
        name="construction fingerprint",
        optional=True,
    )
    _fingerprint(artifact["numerical_fingerprint"], name="numerical fingerprint")
    _fingerprint(evidence["receipt_fingerprint"], name="receipt fingerprint", optional=True)
    source_ids = _array(evidence["source_ids"], name="benchmark source ids")
    if (
        not source_ids
        or any(not isinstance(item, str) for item in source_ids)
        or not set(cast(list[str], source_ids)).issubset(_SOURCE_IDS)
    ):
        raise ScheduleContractError("benchmark source ids are invalid")

    baselines = row["baselines"]
    if lane in {"raw_schedule_parity", "turbo_schedule_parity"}:
        baseline_map = _object(baselines, name="schedule parity baselines")
        _exact(
            baseline_map,
            frozenset({"diffusers_float32", "krea_float64"}),
            name="schedule parity baselines",
        )
        for name, metric_set in baseline_map.items():
            _validate_metric_set(metric_set, name=f"baseline {name}")
    elif lane == "anima_schedule_parity":
        baseline_map = _object(baselines, name="Anima schedule parity baselines")
        _exact(
            baseline_map,
            frozenset({"anima_float32", "anima_float64"}),
            name="Anima schedule parity baselines",
        )
        for name, metric_set in baseline_map.items():
            _validate_metric_set(metric_set, name=f"baseline {name}")
    elif lane == "wan_schedule_parity":
        baseline_map = _object(baselines, name="Wan schedule parity baselines")
        _exact(
            baseline_map,
            frozenset({"wan_float32", "wan_float64"}),
            name="Wan schedule parity baselines",
        )
        for name, metric_set in baseline_map.items():
            _validate_metric_set(metric_set, name=f"baseline {name}")
    elif lane == "h3_native_euler":
        baseline_map = _object(baselines, name="native Euler baseline")
        _exact(baseline_map, frozenset({"native_euler"}), name="native Euler baseline")
        _validate_metric_set(baseline_map["native_euler"], name="native Euler baseline")
    elif baselines is not None:
        raise ScheduleContractError("H2 workflow benchmark cannot claim a numerical baseline")

    repeat = row["repeat"]
    if lane.startswith("h"):
        repeat_map = _object(repeat, name="benchmark repeat")
        _exact(
            repeat_map,
            frozenset(
                {
                    "accepted",
                    "first_status",
                    "repeat_status",
                    "stable",
                    "transition",
                }
            ),
            name="benchmark repeat",
        )
        if (
            repeat_map["accepted"] is not True
            or repeat_map["stable"] is not True
            or repeat_map["transition"] != "pass_to_pass"
            or repeat_map["first_status"] != status
            or repeat_map["repeat_status"] != status
        ):
            raise ScheduleContractError("benchmark first/repeat evidence is not stable")
    elif repeat is not None:
        raise ScheduleContractError("schedule parity row cannot claim host repeat evidence")

    determinism = row["determinism"]
    if lane == "h3_native_euler":
        deterministic = _object(determinism, name="benchmark determinism")
        _exact(
            deterministic,
            frozenset(
                {
                    "deterministic_rerun",
                    "max_abs_error",
                    "mean_abs_error",
                    "status",
                    "tolerance",
                    "trace_fingerprint",
                }
            ),
            name="benchmark determinism",
        )
        if deterministic["deterministic_rerun"] is not True or deterministic["status"] != "PASS":
            raise ScheduleContractError("native Euler determinism evidence did not pass")
        for field in ("max_abs_error", "mean_abs_error", "tolerance"):
            _metric(deterministic[field], name=f"benchmark determinism {field}")
        _fingerprint(deterministic["trace_fingerprint"], name="trace fingerprint")
    elif determinism is not None:
        raise ScheduleContractError("non-H3 benchmark cannot claim H3 determinism")
    return identifier, lane


def _validate_matrix(value: object) -> dict[str, object]:
    matrix = _object(value, name="benchmark matrix")
    _exact(matrix, _MATRIX_FIELDS, name="benchmark matrix")
    if matrix["schema"] != NUMERICAL_BENCHMARK_MATRIX_SCHEMA:
        raise ScheduleContractError("benchmark matrix schema is unsupported")
    results = _array(matrix["results"], name="benchmark results")
    identities_and_lanes = [_validate_row(row) for row in results]
    identities = [item[0] for item in identities_and_lanes]
    if len(results) != 39 or identities != sorted(identities) or len(set(identities)) != 39:
        raise ScheduleContractError("benchmark result identity/order coverage is invalid")
    observed = {lane: sum(item[1] == lane for item in identities_and_lanes) for lane in _LANES}
    observed["total_verified_results"] = len(results)
    coverage = _object(matrix["coverage"], name="benchmark coverage")
    if coverage != observed:
        raise ScheduleContractError("benchmark coverage does not match verified results")
    if matrix["exclusions"] != _EXCLUSIONS:
        raise ScheduleContractError("benchmark exclusions are invalid")
    if matrix["weight_variants"] != [
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
    ]:
        raise ScheduleContractError("benchmark weight variants are invalid")
    sources = _array(matrix["sources"], name="benchmark sources")
    if len(sources) != len(_SOURCE_PATHS):
        raise ScheduleContractError("benchmark source coverage is invalid")
    observed_paths: list[str] = []
    for source_value in sources:
        source = _object(source_value, name="benchmark source")
        _exact(
            source,
            frozenset({"path", "schema", "sha256", "status"}),
            name="benchmark source",
        )
        path = _text(source["path"], name="benchmark source path")
        if path not in _SOURCE_SCHEMAS:
            raise ScheduleContractError("benchmark source path is not allowlisted")
        observed_paths.append(path)
        schema = _text(source["schema"], name="benchmark source schema")
        _fingerprint(source["sha256"], name="benchmark source identity")
        expected_schema, expected_status = _SOURCE_SCHEMAS[observed_paths[-1]]
        if schema != expected_schema or source["status"] != expected_status:
            raise ScheduleContractError("benchmark source schema/status is not accepted")
    if tuple(observed_paths) != _SOURCE_PATHS:
        raise ScheduleContractError("benchmark source allowlist/order drifted")
    return matrix


@dataclass(frozen=True, slots=True)
class NumericalBenchmarkMatrix:
    """Canonical immutable numerical benchmark matrix."""

    _matrix_bytes: bytes
    matrix_fingerprint: str

    def projection(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._matrix_bytes))


def serialize_numerical_benchmark_matrix(matrix: NumericalBenchmarkMatrix) -> bytes:
    if not isinstance(matrix, NumericalBenchmarkMatrix):
        raise ScheduleContractError("matrix must be a NumericalBenchmarkMatrix")
    projection = matrix.projection()
    _validate_matrix(projection)
    matrix_bytes = _canonical(projection)
    if _identity(matrix_bytes) != matrix.matrix_fingerprint:
        raise ScheduleContractError("benchmark matrix fingerprint drifted")
    return (
        _canonical(
            {
                "matrix": projection,
                "matrix_fingerprint": matrix.matrix_fingerprint,
                "schema": NUMERICAL_BENCHMARK_MATRIX_ENVELOPE_SCHEMA,
            }
        )
        + b"\n"
    )


def load_numerical_benchmark_matrix(
    payload: bytes | str | None = None,
) -> NumericalBenchmarkMatrix:
    if payload is None:
        payload = (
            importlib.resources.files("comfyui_sigmax.benchmarks")
            .joinpath("numerical_matrix_v1.json")
            .read_bytes()
        )
    envelope = _decode(payload)
    _exact(
        envelope,
        frozenset({"matrix", "matrix_fingerprint", "schema"}),
        name="benchmark matrix envelope",
    )
    if envelope["schema"] != NUMERICAL_BENCHMARK_MATRIX_ENVELOPE_SCHEMA:
        raise ScheduleContractError("benchmark matrix envelope schema is unsupported")
    matrix = _validate_matrix(envelope["matrix"])
    matrix_bytes = _canonical(matrix)
    fingerprint = _fingerprint(
        envelope["matrix_fingerprint"],
        name="benchmark matrix fingerprint",
    )
    if fingerprint != _identity(matrix_bytes):
        raise ScheduleContractError("benchmark matrix fingerprint does not match content")
    if fingerprint is None:
        raise ScheduleContractError("benchmark matrix fingerprint is required")
    return NumericalBenchmarkMatrix(
        _matrix_bytes=matrix_bytes,
        matrix_fingerprint=fingerprint,
    )
