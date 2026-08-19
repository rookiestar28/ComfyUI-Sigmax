"""Pure, redacted MiniMax H3 acceleration readiness receipts.

This module deliberately observes no host, model, GPU, or Comfy Kitchen state.  It defines a
versioned boundary for a future authorized dispatcher observation and permits only synthetic
readiness records in the current M7-12 lane.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

H3_BACKEND_RECEIPT_SCHEMA: Final = "sigmax.h3-backend-execution-receipt/1"
H3_BACKEND_RECEIPT_VERSION: Final = "1"
_MAX_TEXT: Final = 128
_MAX_ARRAY: Final = 32
_MAX_RECEIPT_BYTES: Final = 1_048_576
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._+/-]{0,127}$")
_REASON_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PRIVATE_PATH_PATTERN: Final = re.compile(
    r"(?:^|[\s\"'=(])(?:[a-z]:[\\/]|\\\\[^\\]|/(?:home|users|mnt)/)",
    re.IGNORECASE,
)
_SECRET_PATTERN: Final = re.compile(
    r"(?:^|[_.-])(?:api_?key|access_key|private_key|secret|password|passwd|credential|"
    r"cookie|token|authorization|auth)(?:[_.-]|$)",
    re.IGNORECASE,
)
_UNAVAILABLE: Final = "unavailable"


class OperationBackend(str, Enum):
    """Actual or requested backend for quantized model operations."""

    EAGER = "eager"
    CUDA = "cuda"
    TRITON = "triton"
    HIP = "hip"
    NOT_OBSERVED = "not_observed"
    UNAVAILABLE = "unavailable"


class AttentionBackend(str, Enum):
    """Actual or requested attention implementation."""

    PYTORCH = "pytorch"
    CK_INT8 = "ck_int8"
    SAGE = "sage"
    FLASH = "flash"
    NOT_OBSERVED = "not_observed"
    UNAVAILABLE = "unavailable"


class ObservationSource(str, Enum):
    """How actual-backend evidence was obtained."""

    SYNTHETIC = "synthetic"
    AUTHORIZED_HOST_DISPATCH = "authorized_host_dispatch"
    NOT_OBSERVED = "not_observed"


class BackendResultStatus(str, Enum):
    """Truthful result states retained by the receipt contract."""

    SUCCEEDED = "succeeded"
    FALLBACK = "fallback"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    NOT_EXECUTED = "not_executed"


def _public_text(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT:
        raise ScheduleContractError(f"{field_name} must be bounded public text")
    if any(ord(character) < 32 for character in value):
        raise ScheduleContractError(f"{field_name} contains a control character")
    if _PRIVATE_PATH_PATTERN.search(value):
        raise ScheduleContractError(f"{field_name} must not contain a private path")
    if _SECRET_PATTERN.search(value):
        raise ScheduleContractError(f"{field_name} must not contain secret-like text")
    return value


def _identifier(field_name: str, value: object) -> str:
    text = _public_text(field_name, value)
    if not text.isascii() or not _IDENTIFIER_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable public identifier")
    return text


def _reason(field_name: str, value: object) -> str:
    text = _public_text(field_name, value)
    if not text.isascii() or not _REASON_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a stable reason code")
    return text


def _fingerprint(field_name: str, value: object, *, allow_unavailable: bool = False) -> str:
    text = _public_text(field_name, value)
    if allow_unavailable and text == _UNAVAILABLE:
        return text
    if not _SHA256_PATTERN.fullmatch(text):
        raise ScheduleContractError(f"{field_name} must be a SHA-256 fingerprint")
    return text


def _bounded_non_negative(field_name: str, value: object, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
        raise ScheduleContractError(f"{field_name} must be a bounded non-negative integer")
    return value


def _enum(field_name: str, value: object, enum_type: type[Enum]) -> Enum:
    if not isinstance(value, enum_type):
        raise ScheduleContractError(f"{field_name} has an unsupported enum value")
    return value


def _tuple_reasons(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) > _MAX_ARRAY:
        raise ScheduleContractError("reason_codes must be a bounded tuple")
    reasons = tuple(_reason("reason_code", item) for item in value)
    if reasons != tuple(sorted(set(reasons))):
        raise ScheduleContractError("reason_codes must be sorted and unique")
    return reasons


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleIdentity:
    """The exact pure schedule identity linked by a backend receipt."""

    profile_id: str
    recipe_id: str
    construction_fingerprint: str
    numerical_fingerprint: str

    def __post_init__(self) -> None:
        _identifier("schedule profile_id", self.profile_id)
        _identifier("schedule recipe_id", self.recipe_id)
        _fingerprint("schedule construction_fingerprint", self.construction_fingerprint)
        _fingerprint("schedule numerical_fingerprint", self.numerical_fingerprint)

    def projection(self) -> dict[str, str]:
        return {
            "construction_fingerprint": self.construction_fingerprint,
            "numerical_fingerprint": self.numerical_fingerprint,
            "profile_id": self.profile_id,
            "recipe_id": self.recipe_id,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointIdentity:
    """Redacted caller-owned checkpoint/LoRA/GPU identity."""

    identifier: str
    version: str
    fingerprint: str

    def __post_init__(self) -> None:
        _identifier("identity identifier", self.identifier)
        _identifier("identity version", self.version)
        _fingerprint("identity fingerprint", self.fingerprint, allow_unavailable=True)

    @classmethod
    def unavailable(cls) -> CheckpointIdentity:
        return cls(identifier=_UNAVAILABLE, version=_UNAVAILABLE, fingerprint=_UNAVAILABLE)

    def projection(self) -> dict[str, str]:
        return {
            "fingerprint": self.fingerprint,
            "id": self.identifier,
            "version": self.version,
        }


def _identity_projection(value: str | CheckpointIdentity, *, field_name: str) -> dict[str, str]:
    if isinstance(value, CheckpointIdentity):
        return value.projection()
    return CheckpointIdentity(
        identifier=_identifier(field_name, value),
        version=_UNAVAILABLE,
        fingerprint=_UNAVAILABLE,
    ).projection()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionEnvironment:
    """Bounded public runtime identities; no machine-local paths are accepted."""

    comfyui: str
    comfy_kitchen: str
    torch: str
    accelerator: str
    driver: str
    gpu: str
    os: str
    dtype: str
    checkpoint: str | CheckpointIdentity
    lora: str | CheckpointIdentity
    launch_config: str

    def __post_init__(self) -> None:
        for field_name in (
            "comfyui",
            "comfy_kitchen",
            "torch",
            "accelerator",
            "driver",
            "gpu",
            "os",
            "dtype",
            "launch_config",
        ):
            _identifier(f"environment {field_name}", getattr(self, field_name))
        for field_name in ("checkpoint", "lora"):
            value = getattr(self, field_name)
            if not isinstance(value, str | CheckpointIdentity):
                raise ScheduleContractError(f"environment {field_name} identity is malformed")
            if isinstance(value, str):
                _identifier(f"environment {field_name}", value)

    def projection(self) -> dict[str, object]:
        return {
            "accelerator": self.accelerator,
            "checkpoint": _identity_projection(self.checkpoint, field_name="checkpoint"),
            "comfy_kitchen": self.comfy_kitchen,
            "comfyui": self.comfyui,
            "driver": self.driver,
            "dtype": self.dtype,
            "gpu": self.gpu,
            "launch_config": self.launch_config,
            "lora": _identity_projection(self.lora, field_name="lora"),
            "os": self.os,
            "torch": self.torch,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendRequest:
    """Requested axes and launch controls; never actual-dispatch evidence."""

    operation_backend: OperationBackend
    attention_backend: AttentionBackend
    enable_triton_backend: bool = False
    use_ck_attention: bool = False
    override_requested: bool = False

    def __post_init__(self) -> None:
        _enum("requested operation_backend", self.operation_backend, OperationBackend)
        _enum("requested attention_backend", self.attention_backend, AttentionBackend)
        if not isinstance(self.enable_triton_backend, bool) or not isinstance(
            self.use_ck_attention, bool
        ):
            raise ScheduleContractError("backend launch controls must be boolean")
        if not isinstance(self.override_requested, bool):
            raise ScheduleContractError("override_requested must be boolean")
        if self.override_requested:
            raise ScheduleContractError("override evidence is not qualified in M7-12")

    def projection(self) -> dict[str, object]:
        return {
            "attention_backend": self.attention_backend.value,
            "enable_triton_backend": self.enable_triton_backend,
            "operation_backend": self.operation_backend.value,
            "use_ck_attention": self.use_ck_attention,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendObservation:
    """Observed axes with explicit provenance and no flag-to-backend inference."""

    source: ObservationSource
    status: BackendResultStatus
    actual_operation_backend: OperationBackend
    actual_attention_backend: AttentionBackend
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _enum("observation source", self.source, ObservationSource)
        _enum("observation status", self.status, BackendResultStatus)
        _enum("actual operation_backend", self.actual_operation_backend, OperationBackend)
        _enum("actual attention_backend", self.actual_attention_backend, AttentionBackend)
        reasons = _tuple_reasons(self.reason_codes)
        if self.status is not BackendResultStatus.SUCCEEDED and not reasons:
            raise ScheduleContractError("non-success observation requires a reason code")
        if self.source is not ObservationSource.AUTHORIZED_HOST_DISPATCH and (
            self.actual_operation_backend is not OperationBackend.NOT_OBSERVED
            or self.actual_attention_backend is not AttentionBackend.NOT_OBSERVED
        ):
            raise ScheduleContractError(
                "synthetic or unobserved evidence cannot claim an actual backend"
            )
        if self.status in {
            BackendResultStatus.REJECTED,
            BackendResultStatus.UNAVAILABLE,
            BackendResultStatus.NOT_EXECUTED,
        } and (
            self.actual_operation_backend is not OperationBackend.NOT_OBSERVED
            or self.actual_attention_backend is not AttentionBackend.NOT_OBSERVED
        ):
            raise ScheduleContractError("rejected or unavailable evidence has no actual backend")
        if self.status in {BackendResultStatus.SUCCEEDED, BackendResultStatus.FALLBACK} and (
            self.source is not ObservationSource.AUTHORIZED_HOST_DISPATCH
        ):
            raise ScheduleContractError(
                "successful actual dispatch requires authorized host evidence"
            )
        if self.status is BackendResultStatus.FALLBACK and not reasons:
            raise ScheduleContractError("fallback observation requires a reason code")

    def projection(self) -> dict[str, object]:
        return {
            "actual_attention_backend": self.actual_attention_backend.value,
            "actual_operation_backend": self.actual_operation_backend.value,
            "reason_codes": list(self.reason_codes),
            "source": self.source.value,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RunEvidence:
    """Optional timing/output evidence, valid only for authorized successful dispatch."""

    warmup_runs: int
    first_latency_us: int | None
    repeat_latency_us: int | None
    peak_memory_bytes: int | None
    stable_repeat: bool | None
    output_fingerprint: str | None
    cleanup_status: str
    mutation_status: str

    def __post_init__(self) -> None:
        _bounded_non_negative("warmup_runs", self.warmup_runs, maximum=1_000)
        for field_name, value, maximum in (
            ("first_latency_us", self.first_latency_us, 86_400_000_000),
            ("repeat_latency_us", self.repeat_latency_us, 86_400_000_000),
            ("peak_memory_bytes", self.peak_memory_bytes, 2**63 - 1),
        ):
            if value is not None:
                _bounded_non_negative(field_name, value, maximum=maximum)
        if self.stable_repeat is not None and not isinstance(self.stable_repeat, bool):
            raise ScheduleContractError("stable_repeat must be boolean or None")
        if self.output_fingerprint is not None:
            _fingerprint("output_fingerprint", self.output_fingerprint)
        for status_name, status_value in (
            ("cleanup_status", self.cleanup_status),
            ("mutation_status", self.mutation_status),
        ):
            if status_value not in {"not_applicable", "pass", "fail", "rejected"}:
                raise ScheduleContractError(f"{status_name} is unsupported")

    @property
    def has_runtime_metrics(self) -> bool:
        return any(
            value is not None
            for value in (
                self.first_latency_us,
                self.repeat_latency_us,
                self.peak_memory_bytes,
                self.stable_repeat,
                self.output_fingerprint,
            )
        )

    def projection(self) -> dict[str, object]:
        return {
            "cleanup_status": self.cleanup_status,
            "first_latency_us": self.first_latency_us,
            "mutation_status": self.mutation_status,
            "output_fingerprint": self.output_fingerprint,
            "peak_memory_bytes": self.peak_memory_bytes,
            "repeat_latency_us": self.repeat_latency_us,
            "stable_repeat": self.stable_repeat,
            "warmup_runs": self.warmup_runs,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendResult:
    """Final result, retaining negative and not-executed outcomes."""

    status: BackendResultStatus
    reason_code: str | None

    def __post_init__(self) -> None:
        _enum("result status", self.status, BackendResultStatus)
        if self.status is BackendResultStatus.SUCCEEDED:
            if self.reason_code is not None:
                raise ScheduleContractError("succeeded result cannot carry a reason code")
        elif self.reason_code is None:
            raise ScheduleContractError("non-success result requires a reason code")
        else:
            _reason("result reason_code", self.reason_code)

    def projection(self) -> dict[str, object]:
        return {"reason_code": self.reason_code, "status": self.status.value}


def _canonical(value: dict[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("backend receipt projection is not canonical JSON") from exc
    if len(encoded) > _MAX_RECEIPT_BYTES:
        raise ScheduleContractError("backend receipt exceeds the bounded size")
    return encoded


def _body_projection(
    *,
    schedule: ScheduleIdentity,
    request: BackendRequest,
    observation: BackendObservation,
    environment: ExecutionEnvironment,
    evidence: RunEvidence,
    result: BackendResult,
) -> dict[str, object]:
    return {
        "environment": environment.projection(),
        "evidence": evidence.projection(),
        "observation": observation.projection(),
        "request": request.projection(),
        "result": result.projection(),
        "schedule": schedule.projection(),
        "schema": H3_BACKEND_RECEIPT_SCHEMA,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class H3BackendExecutionReceipt:
    """Canonical H3 backend receipt with a self-checking body fingerprint."""

    schedule: ScheduleIdentity
    request: BackendRequest
    observation: BackendObservation
    environment: ExecutionEnvironment
    evidence: RunEvidence
    result: BackendResult
    receipt_fingerprint: str

    def __post_init__(self) -> None:
        for field_name, value, expected in (
            ("schedule", self.schedule, ScheduleIdentity),
            ("request", self.request, BackendRequest),
            ("observation", self.observation, BackendObservation),
            ("environment", self.environment, ExecutionEnvironment),
            ("evidence", self.evidence, RunEvidence),
            ("result", self.result, BackendResult),
        ):
            if not isinstance(value, expected):
                raise ScheduleContractError(f"backend receipt {field_name} is malformed")
        if self.observation.status is not self.result.status:
            raise ScheduleContractError("observation and result status must agree")
        if (
            self.result.status is not BackendResultStatus.SUCCEEDED
            and self.evidence.has_runtime_metrics
        ):
            raise ScheduleContractError(
                "runtime metrics and output are only valid for successful execution"
            )
        if self.result.status is BackendResultStatus.SUCCEEDED:
            if self.observation.source is not ObservationSource.AUTHORIZED_HOST_DISPATCH:
                raise ScheduleContractError("successful result needs authorized host observation")
            if not self.evidence.has_runtime_metrics:
                raise ScheduleContractError("successful result needs runtime evidence")
        _fingerprint("receipt_fingerprint", self.receipt_fingerprint)
        body = _canonical(
            _body_projection(
                schedule=self.schedule,
                request=self.request,
                observation=self.observation,
                environment=self.environment,
                evidence=self.evidence,
                result=self.result,
            )
        )
        expected_fingerprint = "sha256:" + hashlib.sha256(body).hexdigest()
        if self.receipt_fingerprint != expected_fingerprint:
            raise ScheduleContractError("backend receipt fingerprint does not match its body")

    @property
    def receipt_bytes(self) -> bytes:
        return _canonical(self.projection())

    def projection(self) -> dict[str, object]:
        body = _body_projection(
            schedule=self.schedule,
            request=self.request,
            observation=self.observation,
            environment=self.environment,
            evidence=self.evidence,
            result=self.result,
        )
        return {**body, "receipt_fingerprint": self.receipt_fingerprint}


def build_h3_backend_execution_receipt(
    *,
    schedule: ScheduleIdentity,
    request: BackendRequest,
    observation: BackendObservation,
    environment: ExecutionEnvironment,
    evidence: RunEvidence,
    result: BackendResult,
) -> H3BackendExecutionReceipt:
    """Build one canonical readiness receipt without observing or mutating a host."""

    body = _canonical(
        _body_projection(
            schedule=schedule,
            request=request,
            observation=observation,
            environment=environment,
            evidence=evidence,
            result=result,
        )
    )
    return H3BackendExecutionReceipt(
        schedule=schedule,
        request=request,
        observation=observation,
        environment=environment,
        evidence=evidence,
        result=result,
        receipt_fingerprint="sha256:" + hashlib.sha256(body).hexdigest(),
    )


def serialize_h3_backend_execution_receipt(receipt: H3BackendExecutionReceipt) -> bytes:
    if not isinstance(receipt, H3BackendExecutionReceipt):
        raise ScheduleContractError("backend receipt serialization needs a receipt")
    return receipt.receipt_bytes


def _mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{field_name} must be an object")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], fields: set[str], *, field_name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{field_name} fields do not match the receipt schema")


def _enum_value(enum_type: type[Enum], value: object, *, field_name: str) -> Enum:
    if not isinstance(value, str):
        raise ScheduleContractError(f"{field_name} enum value is malformed")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ScheduleContractError(f"{field_name} enum value is unsupported") from exc


def _parse_identity(value: object, *, field_name: str) -> CheckpointIdentity:
    if isinstance(value, str):
        _identifier(field_name, value)
        return CheckpointIdentity(identifier=value, version=_UNAVAILABLE, fingerprint=_UNAVAILABLE)
    projection = _mapping(value, field_name=field_name)
    _exact(projection, {"fingerprint", "id", "version"}, field_name=field_name)
    return CheckpointIdentity(
        identifier=cast(str, projection["id"]),
        version=cast(str, projection["version"]),
        fingerprint=cast(str, projection["fingerprint"]),
    )


def deserialize_h3_backend_execution_receipt(payload: bytes | str) -> H3BackendExecutionReceipt:
    """Strictly parse a canonical receipt and verify its body fingerprint."""

    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raise ScheduleContractError("backend receipt payload must be bytes or text")
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise ScheduleContractError("backend receipt payload is too large")
    try:
        loaded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScheduleContractError("backend receipt payload is not JSON") from exc
    projection = _mapping(loaded, field_name="backend receipt")
    _exact(
        projection,
        {
            "environment",
            "evidence",
            "observation",
            "receipt_fingerprint",
            "request",
            "result",
            "schedule",
            "schema",
        },
        field_name="backend receipt",
    )
    if projection["schema"] != H3_BACKEND_RECEIPT_SCHEMA:
        raise ScheduleContractError("backend receipt schema is unsupported")

    schedule_projection = _mapping(projection["schedule"], field_name="schedule")
    _exact(
        schedule_projection,
        {"construction_fingerprint", "numerical_fingerprint", "profile_id", "recipe_id"},
        field_name="schedule",
    )
    schedule = ScheduleIdentity(
        profile_id=cast(str, schedule_projection["profile_id"]),
        recipe_id=cast(str, schedule_projection["recipe_id"]),
        construction_fingerprint=cast(str, schedule_projection["construction_fingerprint"]),
        numerical_fingerprint=cast(str, schedule_projection["numerical_fingerprint"]),
    )

    request_projection = _mapping(projection["request"], field_name="request")
    _exact(
        request_projection,
        {"attention_backend", "enable_triton_backend", "operation_backend", "use_ck_attention"},
        field_name="request",
    )
    request = BackendRequest(
        operation_backend=cast(
            OperationBackend,
            _enum_value(
                OperationBackend,
                request_projection["operation_backend"],
                field_name="request operation_backend",
            ),
        ),
        attention_backend=cast(
            AttentionBackend,
            _enum_value(
                AttentionBackend,
                request_projection["attention_backend"],
                field_name="request attention_backend",
            ),
        ),
        enable_triton_backend=request_projection["enable_triton_backend"],  # type: ignore[arg-type]
        use_ck_attention=request_projection["use_ck_attention"],  # type: ignore[arg-type]
    )

    observation_projection = _mapping(projection["observation"], field_name="observation")
    _exact(
        observation_projection,
        {
            "actual_attention_backend",
            "actual_operation_backend",
            "reason_codes",
            "source",
            "status",
        },
        field_name="observation",
    )
    reason_codes = observation_projection["reason_codes"]
    if not isinstance(reason_codes, list):
        raise ScheduleContractError("observation reason_codes must be an array")
    observation = BackendObservation(
        source=cast(
            ObservationSource,
            _enum_value(
                ObservationSource, observation_projection["source"], field_name="observation source"
            ),
        ),
        status=cast(
            BackendResultStatus,
            _enum_value(
                BackendResultStatus,
                observation_projection["status"],
                field_name="observation status",
            ),
        ),
        actual_operation_backend=cast(
            OperationBackend,
            _enum_value(
                OperationBackend,
                observation_projection["actual_operation_backend"],
                field_name="actual operation_backend",
            ),
        ),
        actual_attention_backend=cast(
            AttentionBackend,
            _enum_value(
                AttentionBackend,
                observation_projection["actual_attention_backend"],
                field_name="actual attention_backend",
            ),
        ),
        reason_codes=tuple(cast(str, item) for item in reason_codes),
    )

    environment_projection = _mapping(projection["environment"], field_name="environment")
    _exact(
        environment_projection,
        {
            "accelerator",
            "checkpoint",
            "comfy_kitchen",
            "comfyui",
            "driver",
            "dtype",
            "gpu",
            "launch_config",
            "lora",
            "os",
            "torch",
        },
        field_name="environment",
    )
    environment = ExecutionEnvironment(
        comfyui=cast(str, environment_projection["comfyui"]),
        comfy_kitchen=cast(str, environment_projection["comfy_kitchen"]),
        torch=cast(str, environment_projection["torch"]),
        accelerator=cast(str, environment_projection["accelerator"]),
        driver=cast(str, environment_projection["driver"]),
        gpu=cast(str, environment_projection["gpu"]),
        os=cast(str, environment_projection["os"]),
        dtype=cast(str, environment_projection["dtype"]),
        checkpoint=_parse_identity(environment_projection["checkpoint"], field_name="checkpoint"),
        lora=_parse_identity(environment_projection["lora"], field_name="lora"),
        launch_config=cast(str, environment_projection["launch_config"]),
    )

    evidence_projection = _mapping(projection["evidence"], field_name="evidence")
    _exact(
        evidence_projection,
        {
            "cleanup_status",
            "first_latency_us",
            "mutation_status",
            "output_fingerprint",
            "peak_memory_bytes",
            "repeat_latency_us",
            "stable_repeat",
            "warmup_runs",
        },
        field_name="evidence",
    )
    evidence = RunEvidence(
        cleanup_status=cast(str, evidence_projection["cleanup_status"]),
        first_latency_us=cast(int | None, evidence_projection["first_latency_us"]),
        mutation_status=cast(str, evidence_projection["mutation_status"]),
        output_fingerprint=cast(str | None, evidence_projection["output_fingerprint"]),
        peak_memory_bytes=cast(int | None, evidence_projection["peak_memory_bytes"]),
        repeat_latency_us=cast(int | None, evidence_projection["repeat_latency_us"]),
        stable_repeat=cast(bool | None, evidence_projection["stable_repeat"]),
        warmup_runs=cast(int, evidence_projection["warmup_runs"]),
    )

    result_projection = _mapping(projection["result"], field_name="result")
    _exact(result_projection, {"reason_code", "status"}, field_name="result")
    reason_code = result_projection["reason_code"]
    if reason_code is not None and not isinstance(reason_code, str):
        raise ScheduleContractError("result reason_code must be text or null")
    result = BackendResult(
        reason_code=reason_code,
        status=cast(
            BackendResultStatus,
            _enum_value(
                BackendResultStatus, result_projection["status"], field_name="result status"
            ),
        ),
    )
    return H3BackendExecutionReceipt(
        schedule=schedule,
        request=request,
        observation=observation,
        environment=environment,
        evidence=evidence,
        result=result,
        receipt_fingerprint=cast(str, projection["receipt_fingerprint"]),
    )


__all__ = [
    "H3_BACKEND_RECEIPT_SCHEMA",
    "H3_BACKEND_RECEIPT_VERSION",
    "AttentionBackend",
    "BackendObservation",
    "BackendRequest",
    "BackendResult",
    "BackendResultStatus",
    "CheckpointIdentity",
    "ExecutionEnvironment",
    "H3BackendExecutionReceipt",
    "ObservationSource",
    "OperationBackend",
    "RunEvidence",
    "ScheduleIdentity",
    "build_h3_backend_execution_receipt",
    "deserialize_h3_backend_execution_receipt",
    "serialize_h3_backend_execution_receipt",
]
