"""Framework-independent deterministic Flow Euler execution controller."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Generic, Protocol, TypeVar

from comfyui_sigmax.core.capabilities import (
    ExecutionBehavior,
    NoiseOwnership,
    PredictionType,
    SamplerState,
    TerminalRequirement,
)
from comfyui_sigmax.core.fingerprints import canonical_projection_bytes, float_to_ieee_hex
from comfyui_sigmax.core.sampler_state import (
    SamplerExecutionSpec,
    SamplerStateSnapshot,
    SamplerStateStatus,
    SamplerStep,
    sampler_execution_spec_fingerprint,
    sampler_state_snapshot_fingerprint,
)
from comfyui_sigmax.core.schedule_contracts import (
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
)

FLOW_EULER_EXECUTION_RESULT_SCHEMA = "sigmax.flow-euler-execution-result/1"
FLOW_EULER_SAMPLER_ID = "sigmax.flow_euler"
FLOW_EULER_SAMPLER_VERSION = "1"
_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_SIGMAS = 10_001

StateT = TypeVar("StateT")


class FlowEulerStateOperations(Protocol[StateT]):
    """State-specific operations supplied without coupling the controller to a framework."""

    def validate(self, state: StateT) -> None: ...

    def fingerprint(self, state: StateT) -> str: ...

    def add_scaled(self, state: StateT, velocity: StateT, scale: float) -> StateT: ...


class FlowEulerVelocityEvaluator(Protocol[StateT]):
    """Return one direct flow velocity for the current state and schedule coordinate."""

    def __call__(self, state: StateT, sigma: float, scheduler_index: int) -> StateT: ...


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_fingerprint(field_name: str, value: object) -> str:
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ScheduleContractError(f"{field_name} must be a canonical SHA-256 fingerprint")
    return value


def _validate_sigmas(sigmas: object) -> tuple[float, ...]:
    if not isinstance(sigmas, tuple) or not 2 <= len(sigmas) <= _MAX_SIGMAS:
        raise ScheduleContractError("Flow Euler requires a bounded tuple with at least two sigmas")
    for index, sigma in enumerate(sigmas):
        if isinstance(sigma, bool) or not isinstance(sigma, float) or not math.isfinite(sigma):
            raise ScheduleContractError(f"sigmas[{index}] must be a finite float")
        if not 0.0 <= sigma <= 1.0:
            raise ScheduleContractError(f"sigmas[{index}] is outside the unit-flow domain")
    if any(current <= following for current, following in pairwise(sigmas)):
        raise ScheduleContractError("Flow Euler sigmas must be strictly descending")
    if sigmas[-1] != 0.0:
        raise ScheduleContractError("Flow Euler schedule must have terminal zero")
    return sigmas


def _schedule_fingerprint(sigmas: tuple[float, ...]) -> str:
    payload = canonical_projection_bytes(
        {
            "schema": "sigmax.flow-euler-schedule/1",
            "sigmas": [float_to_ieee_hex(value, "float64") for value in sigmas],
        }
    )
    return _sha256(payload)


def _validate_spec(spec: object, sigmas: tuple[float, ...]) -> SamplerExecutionSpec:
    if not isinstance(spec, SamplerExecutionSpec):
        raise ScheduleContractError("spec must be a SamplerExecutionSpec")
    capabilities = spec.capabilities
    if (
        capabilities.sampler_id != FLOW_EULER_SAMPLER_ID
        or capabilities.sampler_version != FLOW_EULER_SAMPLER_VERSION
    ):
        raise ScheduleContractError(
            "execution spec does not identify Sigmax deterministic Flow Euler"
        )
    if PredictionType.FLOW_VELOCITY not in capabilities.accepted_prediction_types:
        raise ScheduleContractError("Flow Euler requires flow-velocity prediction support")
    if SigmaDomain.UNIT_FLOW not in capabilities.accepted_sigma_domains:
        raise ScheduleContractError("Flow Euler requires the unit-flow sigma domain")
    if ScheduleOwnership.EXTERNAL_SIGMAS not in capabilities.accepted_ownerships:
        raise ScheduleContractError("Flow Euler requires external sigma ownership")
    if (
        capabilities.execution_behavior is not ExecutionBehavior.DETERMINISTIC
        or capabilities.noise_ownership is not NoiseOwnership.NONE
        or spec.random_source_ownership is not NoiseOwnership.NONE
    ):
        raise ScheduleContractError("deterministic Flow Euler cannot own or use random noise")
    required_state = {
        SamplerState.BEGIN_INDEX,
        SamplerState.STEP_INDEX,
        SamplerState.MULTISTEP_HISTORY,
        SamplerState.RESUME,
    }
    if not required_state <= set(capabilities.required_state):
        raise ScheduleContractError("Flow Euler capabilities omit required state dimensions")
    if (
        capabilities.terminal_requirement is not TerminalRequirement.REQUIRES_ZERO
        or not capabilities.supports_partial_denoise
        or capabilities.supports_per_token_timesteps
        or spec.per_token_time is not None
    ):
        raise ScheduleContractError(
            "Flow Euler terminal, partial, or per-token capabilities drifted"
        )
    if spec.solver_order != 1:
        raise ScheduleContractError("deterministic Flow Euler requires solver order one")
    if spec.scheduler_index != spec.begin_index:
        raise ScheduleContractError("scheduler_index must equal the explicit begin_index")
    if spec.scheduler_index >= len(sigmas) - 1:
        raise ScheduleContractError("begin index is outside the executable schedule")
    expected_transitions = len(sigmas) - 1 - spec.scheduler_index
    if (
        spec.requested_transitions != expected_transitions
        or spec.requested_model_evaluations != expected_transitions
    ):
        raise ScheduleContractError("requested counts do not match the selected schedule slice")
    return spec


@dataclass(frozen=True, slots=True, kw_only=True)
class FlowEulerExecutionResult(Generic[StateT]):
    """One immutable controller result without embedding raw state in its projection."""

    spec: SamplerExecutionSpec
    state: StateT
    state_fingerprint: str
    schedule_fingerprint: str
    snapshot: SamplerStateSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.spec, SamplerExecutionSpec):
            raise ScheduleContractError("result spec must be a SamplerExecutionSpec")
        _require_fingerprint("state_fingerprint", self.state_fingerprint)
        _require_fingerprint("schedule_fingerprint", self.schedule_fingerprint)
        if not isinstance(self.snapshot, SamplerStateSnapshot):
            raise ScheduleContractError("result snapshot must be a SamplerStateSnapshot")
        if self.snapshot.spec_fingerprint != sampler_execution_spec_fingerprint(self.spec):
            raise ScheduleContractError("result snapshot does not match its execution spec")
        if (
            self.snapshot.status
            not in {SamplerStateStatus.INTERRUPTED, SamplerStateStatus.COMPLETED}
            or not self.snapshot.history
            or self.snapshot.history[-1].output_state_fingerprint != self.state_fingerprint
        ):
            raise ScheduleContractError("result state does not match terminal execution history")

    def projection(self) -> dict[str, object]:
        indexes = [step.scheduler_index for step in self.snapshot.history]
        return {
            "counts": {
                "effective_model_evaluations": self.snapshot.effective_model_evaluations,
                "effective_transitions": self.snapshot.effective_transitions,
                "requested_model_evaluations": self.spec.requested_model_evaluations,
                "requested_transitions": self.spec.requested_transitions,
            },
            "executed_scheduler_indexes": indexes,
            "schedule_fingerprint": self.schedule_fingerprint,
            "schema": FLOW_EULER_EXECUTION_RESULT_SCHEMA,
            "snapshot_fingerprint": sampler_state_snapshot_fingerprint(
                self.snapshot,
                self.spec,
            ),
            "spec_fingerprint": sampler_execution_spec_fingerprint(self.spec),
            "state_fingerprint": self.state_fingerprint,
            "status": self.snapshot.status.value,
        }

    @property
    def result_fingerprint(self) -> str:
        return flow_euler_execution_result_fingerprint(self)


def flow_euler_execution_result_fingerprint(result: FlowEulerExecutionResult[Any]) -> str:
    """Return the canonical identity of a Flow Euler execution result."""

    if not isinstance(result, FlowEulerExecutionResult):
        raise ScheduleContractError("result must be a FlowEulerExecutionResult")
    return _sha256(canonical_projection_bytes(result.projection()))


def execute_deterministic_flow_euler(
    *,
    spec: SamplerExecutionSpec,
    sigmas: tuple[float, ...],
    state: StateT,
    evaluator: FlowEulerVelocityEvaluator[StateT],
    operations: FlowEulerStateOperations[StateT],
    snapshot: SamplerStateSnapshot | None = None,
    transition_limit: int | None = None,
) -> FlowEulerExecutionResult[StateT]:
    """Execute a deterministic Flow Euler slice and bind every transition to M5-02 state."""

    validated_sigmas = _validate_sigmas(sigmas)
    validated_spec = _validate_spec(spec, validated_sigmas)
    if transition_limit is not None and (
        isinstance(transition_limit, bool)
        or not isinstance(transition_limit, int)
        or transition_limit <= 0
    ):
        raise ScheduleContractError("transition_limit must be a positive integer or None")
    if not callable(evaluator):
        raise ScheduleContractError("evaluator must be callable")
    for name in ("validate", "fingerprint", "add_scaled"):
        if not callable(getattr(operations, name, None)):
            raise ScheduleContractError("operations do not implement the Flow Euler state contract")

    operations.validate(state)
    current_fingerprint = _require_fingerprint(
        "current state fingerprint",
        operations.fingerprint(state),
    )
    current_snapshot = (
        SamplerStateSnapshot.initial(validated_spec) if snapshot is None else snapshot
    )
    if not isinstance(current_snapshot, SamplerStateSnapshot):
        raise ScheduleContractError("snapshot must be a SamplerStateSnapshot or None")
    if current_snapshot.spec_fingerprint != sampler_execution_spec_fingerprint(validated_spec):
        raise ScheduleContractError("snapshot does not match the execution spec")
    if current_snapshot.execution_receipt_fingerprint is not None:
        raise ScheduleContractError("receipt-bound snapshot cannot execute")
    if current_snapshot.status is SamplerStateStatus.INTERRUPTED:
        if (
            not current_snapshot.history
            or current_snapshot.history[-1].output_state_fingerprint != current_fingerprint
        ):
            raise ScheduleContractError(
                "resume state fingerprint does not match interrupted history"
            )
        current_snapshot = current_snapshot.resume(validated_spec)
    elif current_snapshot.status is not SamplerStateStatus.READY:
        raise ScheduleContractError("snapshot status is not executable")
    elif current_snapshot.history:
        raise ScheduleContractError("ready snapshot cannot contain history")

    remaining = validated_spec.requested_transitions - current_snapshot.next_step_index
    if remaining <= 0:
        raise ScheduleContractError("snapshot has no remaining transitions")
    invocation_count = remaining if transition_limit is None else min(remaining, transition_limit)

    current_state = state
    for _ in range(invocation_count):
        step_index = current_snapshot.next_step_index
        scheduler_index = validated_spec.scheduler_index + step_index
        sigma = validated_sigmas[scheduler_index]
        next_sigma = validated_sigmas[scheduler_index + 1]
        input_fingerprint = _require_fingerprint(
            "input state fingerprint",
            operations.fingerprint(current_state),
        )
        velocity = evaluator(current_state, sigma, scheduler_index)
        # CRITICAL: a model callback must not mutate caller-owned state before history is committed.
        if operations.fingerprint(current_state) != input_fingerprint:
            raise ScheduleContractError("velocity evaluator mutated the input state")
        operations.validate(velocity)
        next_state = operations.add_scaled(current_state, velocity, next_sigma - sigma)
        operations.validate(next_state)
        output_fingerprint = _require_fingerprint(
            "output state fingerprint",
            operations.fingerprint(next_state),
        )
        current_snapshot = current_snapshot.append_step(
            validated_spec,
            SamplerStep(
                step_index=step_index,
                scheduler_index=scheduler_index,
                sigma=sigma,
                next_sigma=next_sigma,
                model_evaluations=1,
                input_state_fingerprint=input_fingerprint,
                output_state_fingerprint=output_fingerprint,
            ),
        )
        current_state = next_state

    if current_snapshot.effective_transitions == validated_spec.requested_transitions:
        current_snapshot = current_snapshot.complete(validated_spec)
    else:
        current_snapshot = current_snapshot.interrupt(validated_spec)
    final_fingerprint = _require_fingerprint(
        "final state fingerprint",
        operations.fingerprint(current_state),
    )
    return FlowEulerExecutionResult(
        spec=validated_spec,
        state=current_state,
        state_fingerprint=final_fingerprint,
        schedule_fingerprint=_schedule_fingerprint(validated_sigmas),
        snapshot=current_snapshot,
    )
