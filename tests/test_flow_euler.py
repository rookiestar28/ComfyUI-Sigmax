"""M5-03 deterministic Flow Euler controller contract tests."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Callable
from dataclasses import replace

import pytest
from comfyui_sigmax.core import (
    ExecutionBehavior,
    FlowEulerExecutionResult,
    NoiseOwnership,
    PredictionType,
    SamplerCapabilities,
    SamplerExecutionSpec,
    SamplerState,
    SamplerStateSnapshot,
    SamplerStateStatus,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalRequirement,
    execute_deterministic_flow_euler,
    flow_euler_execution_result_fingerprint,
)

State = tuple[float, ...]


class TupleStateOperations:
    def validate(self, state: State) -> None:
        if (
            not isinstance(state, tuple)
            or not state
            or any(not isinstance(value, float) or not math.isfinite(value) for value in state)
        ):
            raise ScheduleContractError("state must be a finite float tuple")

    def fingerprint(self, state: State) -> str:
        self.validate(state)
        payload = b"".join(struct.pack(">d", value) for value in state)
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def add_scaled(self, state: State, velocity: State, scale: float) -> State:
        self.validate(state)
        self.validate(velocity)
        if len(state) != len(velocity):
            raise ScheduleContractError("state and velocity shapes differ")
        return tuple(value + scale * delta for value, delta in zip(state, velocity, strict=True))


OPS = TupleStateOperations()
SIGMAS = (1.0, 0.5, 0.0)
INITIAL = (1.0, -0.5)


def _capabilities() -> SamplerCapabilities:
    return SamplerCapabilities(
        sampler_id="sigmax.flow_euler",
        sampler_version="1",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
        execution_behavior=ExecutionBehavior.DETERMINISTIC,
        noise_ownership=NoiseOwnership.NONE,
        required_state=(
            SamplerState.BEGIN_INDEX,
            SamplerState.STEP_INDEX,
            SamplerState.MULTISTEP_HISTORY,
            SamplerState.RESUME,
        ),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )


def _spec(*, begin: int = 0, transitions: int | None = None) -> SamplerExecutionSpec:
    actual = len(SIGMAS) - 1 - begin if transitions is None else transitions
    return SamplerExecutionSpec(
        capabilities=_capabilities(),
        scheduler_index=begin,
        begin_index=begin,
        solver_order=1,
        timestep_spacing="explicit_unit_flow",
        random_source_ownership=NoiseOwnership.NONE,
        per_token_time=None,
        requested_transitions=actual,
        requested_model_evaluations=actual,
    )


def _velocity(state: State, sigma: float, scheduler_index: int) -> State:
    return tuple(0.25 * value + sigma + scheduler_index * 0.125 for value in state)


def _manual(
    state: State,
    *,
    begin: int = 0,
    end: int | None = None,
    evaluator: Callable[[State, float, int], State] = _velocity,
) -> State:
    stop = len(SIGMAS) - 1 if end is None else end
    for index in range(begin, stop):
        velocity = evaluator(state, SIGMAS[index], index)
        state = OPS.add_scaled(state, velocity, SIGMAS[index + 1] - SIGMAS[index])
    return state


def test_full_flow_euler_matches_direct_official_equation_and_counts() -> None:
    calls: list[tuple[float, int]] = []

    def evaluator(state: State, sigma: float, scheduler_index: int) -> State:
        calls.append((sigma, scheduler_index))
        return _velocity(state, sigma, scheduler_index)

    result = execute_deterministic_flow_euler(
        spec=_spec(),
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=evaluator,
        operations=OPS,
    )

    assert isinstance(result, FlowEulerExecutionResult)
    assert result.state == pytest.approx(_manual(INITIAL))
    assert result.snapshot.status is SamplerStateStatus.COMPLETED
    assert result.snapshot.effective_transitions == 2
    assert result.snapshot.effective_model_evaluations == 2
    assert [step.scheduler_index for step in result.snapshot.history] == [0, 1]
    assert calls == [(1.0, 0), (0.5, 1)]
    assert all(sigma != 0.0 for sigma, _ in calls)


def test_explicit_begin_partial_run_uses_absolute_scheduler_indexes() -> None:
    partial_initial = _manual(INITIAL, end=1)
    result = execute_deterministic_flow_euler(
        spec=_spec(begin=1),
        sigmas=SIGMAS,
        state=partial_initial,
        evaluator=_velocity,
        operations=OPS,
    )

    assert result.state == pytest.approx(_manual(partial_initial, begin=1))
    assert result.snapshot.status is SamplerStateStatus.COMPLETED
    assert [step.scheduler_index for step in result.snapshot.history] == [1]
    assert result.snapshot.history[0].sigma == 0.5
    assert result.snapshot.history[0].next_sigma == 0.0


def test_interrupt_then_resume_matches_uninterrupted_result() -> None:
    spec = _spec()
    first = execute_deterministic_flow_euler(
        spec=spec,
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        operations=OPS,
        transition_limit=1,
    )
    assert first.snapshot.status is SamplerStateStatus.INTERRUPTED
    assert first.snapshot.next_step_index == 1

    resumed = execute_deterministic_flow_euler(
        spec=spec,
        sigmas=SIGMAS,
        state=first.state,
        evaluator=_velocity,
        operations=OPS,
        snapshot=first.snapshot,
    )
    uninterrupted = execute_deterministic_flow_euler(
        spec=spec,
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        operations=OPS,
    )

    assert resumed.state == pytest.approx(uninterrupted.state)
    assert resumed.snapshot == uninterrupted.snapshot
    assert resumed.result_fingerprint == uninterrupted.result_fingerprint


def test_resume_requires_the_exact_current_state_fingerprint() -> None:
    spec = _spec()
    first = execute_deterministic_flow_euler(
        spec=spec,
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        operations=OPS,
        transition_limit=1,
    )
    calls = 0

    def evaluator(state: State, sigma: float, scheduler_index: int) -> State:
        nonlocal calls
        calls += 1
        return _velocity(state, sigma, scheduler_index)

    with pytest.raises(ScheduleContractError, match="state fingerprint"):
        execute_deterministic_flow_euler(
            spec=spec,
            sigmas=SIGMAS,
            state=(999.0, 999.0),
            evaluator=evaluator,
            operations=OPS,
            snapshot=first.snapshot,
        )
    assert calls == 0


@pytest.mark.parametrize(
    ("sigmas", "message"),
    [
        ((1.0,), "two"),
        ((1.0, 0.5), "terminal zero"),
        ((1.0, 0.5, 0.5, 0.0), "strictly descending"),
        ((0.5, 0.75, 0.0), "strictly descending"),
        ((1.1, 0.0), "unit-flow"),
        ((1.0, math.nan, 0.0), "finite"),
    ],
)
def test_invalid_schedule_rejects_before_model_evaluation(
    sigmas: tuple[float, ...],
    message: str,
) -> None:
    calls = 0

    def evaluator(state: State, sigma: float, scheduler_index: int) -> State:
        nonlocal calls
        calls += 1
        return _velocity(state, sigma, scheduler_index)

    with pytest.raises(ScheduleContractError, match=message):
        execute_deterministic_flow_euler(
            spec=_spec(transitions=max(0, len(sigmas) - 1)),
            sigmas=sigmas,
            state=INITIAL,
            evaluator=evaluator,
            operations=OPS,
        )
    assert calls == 0


@pytest.mark.parametrize(
    "spec",
    [
        replace(_spec(), solver_order=2),
        replace(_spec(), scheduler_index=1),
        replace(_spec(), requested_model_evaluations=3),
        replace(_spec(), capabilities=replace(_capabilities(), supports_partial_denoise=False)),
        replace(
            _spec(),
            capabilities=replace(
                _capabilities(),
                accepted_prediction_types=(PredictionType.EPSILON,),
            ),
        ),
    ],
)
def test_incompatible_execution_spec_rejects_before_model_evaluation(
    spec: SamplerExecutionSpec,
) -> None:
    calls = 0

    def evaluator(state: State, sigma: float, scheduler_index: int) -> State:
        nonlocal calls
        calls += 1
        return _velocity(state, sigma, scheduler_index)

    with pytest.raises(ScheduleContractError):
        execute_deterministic_flow_euler(
            spec=spec,
            sigmas=SIGMAS,
            state=INITIAL,
            evaluator=evaluator,
            operations=OPS,
        )
    assert calls == 0


def test_receipt_bound_or_completed_snapshot_cannot_execute_again() -> None:
    spec = _spec()
    completed = execute_deterministic_flow_euler(
        spec=spec,
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        operations=OPS,
    ).snapshot
    receipt_bound = replace(completed, execution_receipt_fingerprint="sha256:" + "a" * 64)

    for snapshot in (completed, receipt_bound):
        with pytest.raises(ScheduleContractError, match="snapshot"):
            execute_deterministic_flow_euler(
                spec=spec,
                sigmas=SIGMAS,
                state=_manual(INITIAL),
                evaluator=_velocity,
                operations=OPS,
                snapshot=snapshot,
            )


@pytest.mark.parametrize("transition_limit", [0, -1, True])
def test_transition_limit_must_be_a_positive_integer(transition_limit: int) -> None:
    with pytest.raises(ScheduleContractError, match="transition_limit"):
        execute_deterministic_flow_euler(
            spec=_spec(),
            sigmas=SIGMAS,
            state=INITIAL,
            evaluator=_velocity,
            operations=OPS,
            transition_limit=transition_limit,
        )


def test_evaluator_failure_does_not_create_phantom_history() -> None:
    spec = _spec()
    initial_snapshot = SamplerStateSnapshot.initial(spec)

    def evaluator(state: State, sigma: float, scheduler_index: int) -> State:
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        execute_deterministic_flow_euler(
            spec=spec,
            sigmas=SIGMAS,
            state=INITIAL,
            evaluator=evaluator,
            operations=OPS,
            snapshot=initial_snapshot,
        )
    assert initial_snapshot.status is SamplerStateStatus.READY
    assert initial_snapshot.history == ()


def test_evaluator_cannot_mutate_the_input_state() -> None:
    mutable = [1.0, -0.5]

    class MutableOperations:
        def validate(self, state: list[float]) -> None:
            if any(not math.isfinite(item) for item in state):
                raise ScheduleContractError("invalid")

        def fingerprint(self, state: list[float]) -> str:
            payload = repr(state).encode("ascii")
            return "sha256:" + hashlib.sha256(payload).hexdigest()

        def add_scaled(
            self,
            state: list[float],
            velocity: list[float],
            scale: float,
        ) -> list[float]:
            return [value + scale * delta for value, delta in zip(state, velocity, strict=True)]

    def evaluator(state: list[float], sigma: float, scheduler_index: int) -> list[float]:
        state[0] = 999.0
        return [0.0, 0.0]

    with pytest.raises(ScheduleContractError, match="mutated"):
        execute_deterministic_flow_euler(
            spec=_spec(),
            sigmas=SIGMAS,
            state=mutable,
            evaluator=evaluator,
            operations=MutableOperations(),
        )


def test_result_projection_and_fingerprint_are_repeat_stable_and_state_bound() -> None:
    first = execute_deterministic_flow_euler(
        spec=_spec(),
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        operations=OPS,
    )
    repeat = execute_deterministic_flow_euler(
        spec=_spec(),
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        operations=OPS,
    )

    assert first == repeat
    assert first.result_fingerprint == flow_euler_execution_result_fingerprint(first)
    assert first.projection()["schema"] == "sigmax.flow-euler-execution-result/1"
    assert first.projection()["state_fingerprint"] == OPS.fingerprint(first.state)
    assert first.result_fingerprint.startswith("sha256:")
    with pytest.raises(ScheduleContractError, match="terminal execution history"):
        replace(first, state_fingerprint="sha256:" + "f" * 64)


def test_first_order_convergence_improves_under_uniform_refinement() -> None:
    def solve(transitions: int) -> float:
        sigmas = tuple(1.0 - index / transitions for index in range(transitions + 1))
        spec = replace(
            _spec(),
            requested_transitions=transitions,
            requested_model_evaluations=transitions,
        )
        result = execute_deterministic_flow_euler(
            spec=spec,
            sigmas=sigmas,
            state=(1.0,),
            evaluator=lambda state, sigma, scheduler_index: state,
            operations=OPS,
        )
        return float(abs(result.state[0] - math.exp(-1.0)))

    error_8 = solve(8)
    error_16 = solve(16)
    error_32 = solve(32)

    assert error_32 < error_16 < error_8
    assert 1.8 < error_8 / error_16 < 2.2
    assert 1.8 < error_16 / error_32 < 2.2
