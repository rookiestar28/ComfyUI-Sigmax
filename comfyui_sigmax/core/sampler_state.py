"""Immutable state contracts for resumable sampler executions.

This module deliberately describes sampler state; it does not implement a numerical
integrator or call ComfyUI.  A snapshot can be serialized and resumed only when its
execution specification fingerprint matches exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, replace
from enum import Enum
from itertools import pairwise
from typing import Any, cast

from comfyui_sigmax.core.capabilities import (
    ExecutionBehavior,
    NoiseOwnership,
    PredictionType,
    SamplerCapabilities,
    SamplerState,
    TerminalRequirement,
)
from comfyui_sigmax.core.fingerprints import canonical_projection_bytes, float_to_ieee_hex
from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
)

SAMPLER_EXECUTION_SPEC_SCHEMA = "sigmax.sampler-execution-spec/1"
SAMPLER_STATE_SNAPSHOT_SCHEMA = "sigmax.sampler-state-snapshot/1"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_HISTORY = 10_000
_MAX_PER_TOKEN_TIME = 16_384


class SamplerStateStatus(str, Enum):
    """Lifecycle state of one immutable sampler snapshot."""

    READY = "ready"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


def _sha256_identity(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_non_negative_int(field_name: str, value: object, *, maximum: int = 1_000_000) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > maximum:
        raise ScheduleContractError(f"{field_name} must be a bounded non-negative integer")
    return value


def _require_positive_int(field_name: str, value: object, *, maximum: int = 1_000_000) -> int:
    normalized = _require_non_negative_int(field_name, value, maximum=maximum)
    if normalized == 0:
        raise ScheduleContractError(f"{field_name} must be positive")
    return normalized


def _require_finite_float(field_name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScheduleContractError(f"{field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ScheduleContractError(f"{field_name} must be finite")
    return normalized


def _require_fingerprint(field_name: str, value: object, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ScheduleContractError(f"{field_name} must be a lowercase SHA-256 identity")
    return value


def _require_public_text(field_name: str, value: object, *, maximum: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ScheduleContractError(f"{field_name} must be bounded public text")
    if any(ord(character) < 0x20 for character in value):
        raise ScheduleContractError(f"{field_name} contains control characters")
    return value


def _float_token(value: float) -> str:
    return float_to_ieee_hex(value, "float64")


def _decode_float_token(field_name: str, value: object) -> float:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        raise ScheduleContractError(f"{field_name} must be a float64 IEEE token")
    decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    return _require_finite_float(field_name, decoded)


def _capabilities_projection(capabilities: SamplerCapabilities) -> dict[str, object]:
    return {
        "accepted_ownerships": [item.value for item in capabilities.accepted_ownerships],
        "accepted_prediction_types": [
            item.value for item in capabilities.accepted_prediction_types
        ],
        "accepted_sigma_domains": [item.value for item in capabilities.accepted_sigma_domains],
        "execution_behavior": capabilities.execution_behavior.value,
        "noise_ownership": capabilities.noise_ownership.value,
        "required_state": [item.value for item in capabilities.required_state],
        "sampler_id": capabilities.sampler_id,
        "sampler_version": capabilities.sampler_version,
        "supports_partial_denoise": capabilities.supports_partial_denoise,
        "supports_per_token_timesteps": capabilities.supports_per_token_timesteps,
        "terminal_requirement": capabilities.terminal_requirement.value,
    }


def _capabilities_from_projection(value: object) -> SamplerCapabilities:
    if not isinstance(value, dict):
        raise ScheduleContractError("capabilities must be an object")
    expected = {
        "accepted_ownerships",
        "accepted_prediction_types",
        "accepted_sigma_domains",
        "execution_behavior",
        "noise_ownership",
        "required_state",
        "sampler_id",
        "sampler_version",
        "supports_partial_denoise",
        "supports_per_token_timesteps",
        "terminal_requirement",
    }
    if set(value) != expected:
        raise ScheduleContractError("capabilities fields do not match schema")
    try:
        ownerships = tuple(ScheduleOwnership(item) for item in value["accepted_ownerships"])
        prediction_types = tuple(
            PredictionType(item) for item in value["accepted_prediction_types"]
        )
        sigma_domains = tuple(SigmaDomain(item) for item in value["accepted_sigma_domains"])
        required_state = tuple(SamplerState(item) for item in value["required_state"])
        execution_behavior = ExecutionBehavior(value["execution_behavior"])
        noise_ownership = NoiseOwnership(value["noise_ownership"])
        terminal_requirement = TerminalRequirement(value["terminal_requirement"])
    except (TypeError, ValueError) as error:
        raise ScheduleContractError("capabilities contain an unknown enum value") from error
    return SamplerCapabilities(
        sampler_id=cast(str, value["sampler_id"]),
        sampler_version=cast(str, value["sampler_version"]),
        accepted_prediction_types=prediction_types,
        accepted_sigma_domains=sigma_domains,
        accepted_ownerships=ownerships,
        terminal_requirement=terminal_requirement,
        execution_behavior=execution_behavior,
        noise_ownership=noise_ownership,
        required_state=required_state,
        supports_partial_denoise=cast(bool, value["supports_partial_denoise"]),
        supports_per_token_timesteps=cast(bool, value["supports_per_token_timesteps"]),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplerExecutionSpec:
    """Fixed configuration that binds a resumable sampler state."""

    capabilities: SamplerCapabilities
    scheduler_index: int
    begin_index: int
    solver_order: int
    timestep_spacing: str
    random_source_ownership: NoiseOwnership
    per_token_time: tuple[float, ...] | None
    requested_transitions: int
    requested_model_evaluations: int

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, SamplerCapabilities):
            raise ScheduleContractError("capabilities must be SamplerCapabilities")
        _require_non_negative_int("scheduler_index", self.scheduler_index)
        _require_non_negative_int("begin_index", self.begin_index)
        _require_positive_int("solver_order", self.solver_order, maximum=64)
        _require_public_text("timestep_spacing", self.timestep_spacing)
        if not isinstance(self.random_source_ownership, NoiseOwnership):
            raise ScheduleContractError("random_source_ownership is unsupported")
        if self.per_token_time is not None:
            if not isinstance(self.per_token_time, tuple):
                raise ScheduleContractError("per_token_time must be a tuple or None")
            if len(self.per_token_time) > _MAX_PER_TOKEN_TIME:
                raise ScheduleContractError("per_token_time is too large")
            for index, value in enumerate(self.per_token_time):
                _require_finite_float(f"per_token_time[{index}]", value)
        _require_non_negative_int("requested_transitions", self.requested_transitions)
        _require_non_negative_int("requested_model_evaluations", self.requested_model_evaluations)
        if self.requested_model_evaluations < self.requested_transitions:
            raise ScheduleContractError("requested model evaluations cannot be below transitions")

    def projection(self) -> dict[str, object]:
        return {
            "begin_index": self.begin_index,
            "capabilities": _capabilities_projection(self.capabilities),
            "per_token_time": (
                None
                if self.per_token_time is None
                else [_float_token(value) for value in self.per_token_time]
            ),
            "random_source_ownership": self.random_source_ownership.value,
            "requested_model_evaluations": self.requested_model_evaluations,
            "requested_transitions": self.requested_transitions,
            "scheduler_index": self.scheduler_index,
            "schema": SAMPLER_EXECUTION_SPEC_SCHEMA,
            "solver_order": self.solver_order,
            "timestep_spacing": self.timestep_spacing,
        }


def sampler_execution_spec_fingerprint(spec: SamplerExecutionSpec) -> str:
    """Return the deterministic identity of one execution specification."""

    payload = canonical_projection_bytes(spec.projection())
    return _sha256_identity(payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplerStep:
    """Evidence for one completed transition in a sampler history."""

    step_index: int
    scheduler_index: int
    sigma: float
    next_sigma: float
    model_evaluations: int
    input_state_fingerprint: str
    output_state_fingerprint: str

    def __post_init__(self) -> None:
        _require_non_negative_int("step_index", self.step_index, maximum=_MAX_HISTORY - 1)
        _require_non_negative_int("scheduler_index", self.scheduler_index)
        _require_finite_float("sigma", self.sigma)
        _require_finite_float("next_sigma", self.next_sigma)
        _require_positive_int("model_evaluations", self.model_evaluations, maximum=1_000_000)
        _require_fingerprint("input_state_fingerprint", self.input_state_fingerprint)
        _require_fingerprint("output_state_fingerprint", self.output_state_fingerprint)

    def projection(self) -> dict[str, object]:
        return {
            "input_state_fingerprint": self.input_state_fingerprint,
            "model_evaluations": self.model_evaluations,
            "next_sigma": _float_token(self.next_sigma),
            "output_state_fingerprint": self.output_state_fingerprint,
            "scheduler_index": self.scheduler_index,
            "sigma": _float_token(self.sigma),
            "step_index": self.step_index,
        }


def _require_history_continuity(history: tuple[SamplerStep, ...]) -> None:
    for previous, current in pairwise(history):
        if previous.output_state_fingerprint != current.input_state_fingerprint:
            raise ScheduleContractError(
                "history input state fingerprint is not continuous with the previous output"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SamplerStateSnapshot:
    """Immutable cursor/history snapshot that can be resumed by matching spec."""

    spec_fingerprint: str
    next_step_index: int
    history: tuple[SamplerStep, ...]
    status: SamplerStateStatus
    execution_receipt_fingerprint: str | None = None

    def __post_init__(self) -> None:
        _require_fingerprint("spec_fingerprint", self.spec_fingerprint)
        _require_non_negative_int("next_step_index", self.next_step_index, maximum=_MAX_HISTORY)
        if not isinstance(self.history, tuple) or len(self.history) > _MAX_HISTORY:
            raise ScheduleContractError("history must be a bounded tuple")
        if self.next_step_index != len(self.history):
            raise ScheduleContractError("next_step_index must equal history length")
        if not isinstance(self.status, SamplerStateStatus):
            raise ScheduleContractError("status is unsupported")
        _require_fingerprint(
            "execution_receipt_fingerprint",
            self.execution_receipt_fingerprint,
            allow_none=True,
        )
        for index, step in enumerate(self.history):
            if not isinstance(step, SamplerStep) or step.step_index != index:
                raise ScheduleContractError("history must contain contiguous SamplerStep values")
        _require_history_continuity(self.history)

    @classmethod
    def initial(cls, spec: SamplerExecutionSpec) -> SamplerStateSnapshot:
        return cls(
            spec_fingerprint=sampler_execution_spec_fingerprint(spec),
            next_step_index=0,
            history=(),
            status=SamplerStateStatus.READY,
        )

    @property
    def effective_transitions(self) -> int:
        return len(self.history)

    @property
    def effective_model_evaluations(self) -> int:
        return sum(step.model_evaluations for step in self.history)

    def _require_spec(self, spec: SamplerExecutionSpec) -> None:
        expected = sampler_execution_spec_fingerprint(spec)
        if self.spec_fingerprint != expected:
            raise ScheduleContractError("sampler state does not match execution specification")

    def append_step(self, spec: SamplerExecutionSpec, step: SamplerStep) -> SamplerStateSnapshot:
        """Return a new running snapshot with one contiguous step appended."""

        self._require_spec(spec)
        if self.status in (SamplerStateStatus.INTERRUPTED, SamplerStateStatus.COMPLETED):
            raise ScheduleContractError("resume or reset the snapshot before appending a step")
        if step.step_index != self.next_step_index:
            raise ScheduleContractError("step_index is not contiguous with the snapshot cursor")
        expected_scheduler_index = spec.scheduler_index + step.step_index
        if step.scheduler_index != expected_scheduler_index:
            raise ScheduleContractError("scheduler_index is not contiguous with the snapshot")
        if (
            self.history
            and self.history[-1].output_state_fingerprint != step.input_state_fingerprint
        ):
            raise ScheduleContractError(
                "input state fingerprint is not continuous with the previous output"
            )
        if self.effective_transitions + 1 > spec.requested_transitions:
            raise ScheduleContractError("step count exceeds requested transitions")
        if (
            self.effective_model_evaluations + step.model_evaluations
            > spec.requested_model_evaluations
        ):
            raise ScheduleContractError("model-evaluation count exceeds requested count")
        return replace(
            self,
            next_step_index=self.next_step_index + 1,
            history=(*self.history, step),
            status=SamplerStateStatus.RUNNING,
        )

    def interrupt(self, spec: SamplerExecutionSpec) -> SamplerStateSnapshot:
        self._require_spec(spec)
        if self.status is SamplerStateStatus.COMPLETED:
            raise ScheduleContractError("completed sampler state cannot be interrupted")
        return replace(self, status=SamplerStateStatus.INTERRUPTED)

    def resume(self, spec: SamplerExecutionSpec) -> SamplerStateSnapshot:
        self._require_spec(spec)
        if self.status is not SamplerStateStatus.INTERRUPTED:
            raise ScheduleContractError("only interrupted sampler state can be resumed")
        return replace(self, status=SamplerStateStatus.RUNNING)

    def complete(self, spec: SamplerExecutionSpec) -> SamplerStateSnapshot:
        self._require_spec(spec)
        if self.effective_transitions != spec.requested_transitions:
            raise ScheduleContractError("cannot complete before all transitions execute")
        if self.effective_model_evaluations != spec.requested_model_evaluations:
            raise ScheduleContractError("cannot complete before all model evaluations execute")
        return replace(self, status=SamplerStateStatus.COMPLETED)

    def attach_execution_receipt(
        self,
        spec: SamplerExecutionSpec,
        receipt_fingerprint: str,
    ) -> SamplerStateSnapshot:
        self._require_spec(spec)
        _require_fingerprint("receipt_fingerprint", receipt_fingerprint)
        return replace(self, execution_receipt_fingerprint=receipt_fingerprint)

    def projection(self, spec: SamplerExecutionSpec) -> dict[str, object]:
        self._require_spec(spec)
        return {
            "effective_model_evaluations": self.effective_model_evaluations,
            "effective_transitions": self.effective_transitions,
            "execution_receipt_fingerprint": self.execution_receipt_fingerprint,
            "history": [step.projection() for step in self.history],
            "next_step_index": self.next_step_index,
            "schema": SAMPLER_STATE_SNAPSHOT_SCHEMA,
            "spec_fingerprint": self.spec_fingerprint,
            "status": self.status.value,
        }


def sampler_state_snapshot_fingerprint(
    snapshot: SamplerStateSnapshot,
    spec: SamplerExecutionSpec,
) -> str:
    """Return the deterministic identity of one spec-bound state snapshot."""

    if not isinstance(snapshot, SamplerStateSnapshot):
        raise ScheduleContractError("snapshot must be a SamplerStateSnapshot")
    if not isinstance(spec, SamplerExecutionSpec):
        raise ScheduleContractError("spec must be a SamplerExecutionSpec")
    return _sha256_identity(canonical_projection_bytes(snapshot.projection(spec)))


def serialize_sampler_execution_spec(spec: SamplerExecutionSpec) -> bytes:
    """Serialize one execution specification to canonical JSON bytes."""

    return canonical_projection_bytes(spec.projection())


def deserialize_sampler_execution_spec(payload: bytes | str) -> SamplerExecutionSpec:
    """Deserialize and validate one canonical execution specification."""

    raw = _decode_json(payload)
    expected = {
        "begin_index",
        "capabilities",
        "per_token_time",
        "random_source_ownership",
        "requested_model_evaluations",
        "requested_transitions",
        "scheduler_index",
        "schema",
        "solver_order",
        "timestep_spacing",
    }
    if set(raw) != expected or raw["schema"] != SAMPLER_EXECUTION_SPEC_SCHEMA:
        raise ScheduleContractError("sampler execution spec fields do not match schema")
    per_token = raw["per_token_time"]
    if per_token is not None:
        if not isinstance(per_token, list):
            raise ScheduleContractError("per_token_time must be a token list or null")
        per_token_time = tuple(
            _decode_float_token(f"per_token_time[{index}]", token)
            for index, token in enumerate(per_token)
        )
    else:
        per_token_time = None
    try:
        ownership = NoiseOwnership(raw["random_source_ownership"])
    except (TypeError, ValueError) as error:
        raise ScheduleContractError("random_source_ownership is unsupported") from error
    return SamplerExecutionSpec(
        capabilities=_capabilities_from_projection(raw["capabilities"]),
        scheduler_index=cast(int, raw["scheduler_index"]),
        begin_index=cast(int, raw["begin_index"]),
        solver_order=cast(int, raw["solver_order"]),
        timestep_spacing=cast(str, raw["timestep_spacing"]),
        random_source_ownership=ownership,
        per_token_time=per_token_time,
        requested_transitions=cast(int, raw["requested_transitions"]),
        requested_model_evaluations=cast(int, raw["requested_model_evaluations"]),
    )


def serialize_sampler_state_snapshot(
    snapshot: SamplerStateSnapshot,
    spec: SamplerExecutionSpec,
) -> bytes:
    """Serialize one spec-bound sampler snapshot to canonical JSON bytes."""

    return canonical_projection_bytes(snapshot.projection(spec))


def deserialize_sampler_state_snapshot(
    payload: bytes | str,
    spec: SamplerExecutionSpec,
) -> SamplerStateSnapshot:
    """Deserialize and validate one spec-bound sampler snapshot."""

    raw = _decode_json(payload)
    expected = {
        "effective_model_evaluations",
        "effective_transitions",
        "execution_receipt_fingerprint",
        "history",
        "next_step_index",
        "schema",
        "spec_fingerprint",
        "status",
    }
    if set(raw) != expected or raw["schema"] != SAMPLER_STATE_SNAPSHOT_SCHEMA:
        raise ScheduleContractError("sampler state fields do not match schema")
    history_raw = raw["history"]
    if not isinstance(history_raw, list):
        raise ScheduleContractError("history must be a list")
    history: list[SamplerStep] = []
    for index, value in enumerate(history_raw):
        if not isinstance(value, dict):
            raise ScheduleContractError("history entry must be an object")
        step_fields = {
            "input_state_fingerprint",
            "model_evaluations",
            "next_sigma",
            "output_state_fingerprint",
            "scheduler_index",
            "sigma",
            "step_index",
        }
        if set(value) != step_fields:
            raise ScheduleContractError("history entry fields do not match schema")
        step = SamplerStep(
            step_index=cast(int, value["step_index"]),
            scheduler_index=cast(int, value["scheduler_index"]),
            sigma=_decode_float_token("sigma", value["sigma"]),
            next_sigma=_decode_float_token("next_sigma", value["next_sigma"]),
            model_evaluations=cast(int, value["model_evaluations"]),
            input_state_fingerprint=cast(str, value["input_state_fingerprint"]),
            output_state_fingerprint=cast(str, value["output_state_fingerprint"]),
        )
        if step.step_index != index:
            raise ScheduleContractError("history step indexes must be contiguous")
        history.append(step)
    try:
        status = SamplerStateStatus(raw["status"])
    except (TypeError, ValueError) as error:
        raise ScheduleContractError("sampler state status is unsupported") from error
    snapshot = SamplerStateSnapshot(
        spec_fingerprint=cast(str, raw["spec_fingerprint"]),
        next_step_index=cast(int, raw["next_step_index"]),
        history=tuple(history),
        status=status,
        execution_receipt_fingerprint=cast(str | None, raw["execution_receipt_fingerprint"]),
    )
    snapshot._require_spec(spec)
    if snapshot.effective_transitions != raw["effective_transitions"]:
        raise ScheduleContractError("effective transition count is inconsistent")
    if snapshot.effective_model_evaluations != raw["effective_model_evaluations"]:
        raise ScheduleContractError("effective model-evaluation count is inconsistent")
    if snapshot.effective_transitions > spec.requested_transitions:
        raise ScheduleContractError("state exceeds requested transitions")
    if snapshot.effective_model_evaluations > spec.requested_model_evaluations:
        raise ScheduleContractError("state exceeds requested model evaluations")
    return snapshot


def _decode_json(payload: bytes | str) -> dict[str, Any]:
    if isinstance(payload, bytes):
        if len(payload) > 1_048_576:
            raise ScheduleContractError("sampler state payload exceeds byte limit")
        source = payload.decode("utf-8")
    elif isinstance(payload, str):
        if len(payload.encode("utf-8")) > 1_048_576:
            raise ScheduleContractError("sampler state payload exceeds byte limit")
        source = payload
    else:
        raise ScheduleContractError("sampler state payload must be bytes or text")
    try:
        raw = json.loads(source)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScheduleContractError("sampler state payload is not valid JSON") from error
    if not isinstance(raw, dict):
        raise ScheduleContractError("sampler state payload root must be an object")
    return cast(dict[str, Any], raw)
