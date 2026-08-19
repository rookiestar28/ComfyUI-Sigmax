"""M5-04 stochastic FlowMatch Euler controller contract tests."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import replace
from itertools import pairwise
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    ExecutionBehavior,
    NoiseOwnership,
    PredictionType,
    SamplerCapabilities,
    SamplerExecutionSpec,
    SamplerState,
    SamplerStateStatus,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    StochasticFlowEulerExecutionResult,
    TerminalRequirement,
    execute_stochastic_flow_euler,
    stochastic_flow_euler_execution_result_fingerprint,
)

State = tuple[float, ...]
SIGMAS = (1.0, 0.75, 0.25, 0.0)
INITIAL = (0.75, -0.5, 1.25, -1.0)


class TupleStochasticOperations:
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

    def interpolate(self, x0: State, noise: State, weight: float) -> State:
        self.validate(x0)
        self.validate(noise)
        if len(x0) != len(noise):
            raise ScheduleContractError("state and noise shapes differ")
        return tuple(
            (1.0 - weight) * value + weight * random_value
            for value, random_value in zip(x0, noise, strict=True)
        )


OPS = TupleStochasticOperations()


def _capabilities() -> SamplerCapabilities:
    return SamplerCapabilities(
        sampler_id="sigmax.flow_euler_stochastic",
        sampler_version="1",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
        execution_behavior=ExecutionBehavior.STOCHASTIC,
        noise_ownership=NoiseOwnership.CALLER,
        required_state=(
            SamplerState.BEGIN_INDEX,
            SamplerState.STEP_INDEX,
            SamplerState.MULTISTEP_HISTORY,
        ),
        supports_partial_denoise=False,
        supports_per_token_timesteps=False,
    )


def _spec() -> SamplerExecutionSpec:
    transitions = len(SIGMAS) - 1
    return SamplerExecutionSpec(
        capabilities=_capabilities(),
        scheduler_index=0,
        begin_index=0,
        solver_order=1,
        timestep_spacing="explicit_unit_flow",
        random_source_ownership=NoiseOwnership.CALLER,
        per_token_time=None,
        requested_transitions=transitions,
        requested_model_evaluations=transitions,
    )


def _velocity(state: State, sigma: float, scheduler_index: int) -> State:
    return tuple(0.25 * value + sigma + scheduler_index * 0.125 for value in state)


class SeededNoise:
    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._draw_index = 0
        self.calls: list[tuple[float, float, int]] = []

    def __call__(
        self,
        reference: State,
        sigma: float,
        next_sigma: float,
        scheduler_index: int,
    ) -> State:
        self.calls.append((sigma, next_sigma, scheduler_index))
        self._draw_index += 1
        return tuple(
            math.sin(self._seed + self._draw_index * 0.5 + index * 0.25)
            for index, _ in enumerate(reference)
        )


def _execute(seed: int) -> StochasticFlowEulerExecutionResult[State]:
    return execute_stochastic_flow_euler(
        spec=_spec(),
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        noise_provider=SeededNoise(seed),
        operations=OPS,
    )


def test_direct_interpolation_matches_manual_formula_and_exact_counts() -> None:
    provider = SeededNoise(123)
    manual_provider = SeededNoise(123)
    expected: State = INITIAL
    for index, (sigma, next_sigma) in enumerate(pairwise(SIGMAS)):
        velocity = _velocity(expected, sigma, index)
        x0 = OPS.add_scaled(expected, velocity, -sigma)
        expected = OPS.interpolate(
            x0,
            manual_provider(x0, sigma, next_sigma, index),
            next_sigma,
        )

    result = execute_stochastic_flow_euler(
        spec=_spec(),
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        noise_provider=provider,
        operations=OPS,
    )

    assert isinstance(result, StochasticFlowEulerExecutionResult)
    assert result.state == expected
    assert result.snapshot.status is SamplerStateStatus.COMPLETED
    assert result.snapshot.effective_transitions == 3
    assert result.snapshot.effective_model_evaluations == 3
    assert [step.scheduler_index for step in result.snapshot.history] == [0, 1, 2]
    assert provider.calls == [(1.0, 0.75, 0), (0.75, 0.25, 1), (0.25, 0.0, 2)]
    assert len(result.noise_fingerprints) == 3


def test_terminal_transition_consumes_noise_but_does_not_change_x0() -> None:
    provider = SeededNoise(321)
    result = execute_stochastic_flow_euler(
        spec=_spec(),
        sigmas=SIGMAS,
        state=INITIAL,
        evaluator=_velocity,
        noise_provider=provider,
        operations=OPS,
    )
    before_terminal = result.snapshot.history[-1].input_state_fingerprint
    assert len(provider.calls) == 3
    assert provider.calls[-1][1] == 0.0
    assert result.state == OPS.add_scaled(
        # Recover the terminal input with the same first two draws.
        _execute_first_two(321),
        _velocity(_execute_first_two(321), 0.25, 2),
        -0.25,
    )
    assert before_terminal == OPS.fingerprint(_execute_first_two(321))


def _execute_first_two(seed: int) -> State:
    provider = SeededNoise(seed)
    state: State = INITIAL
    for index in range(2):
        sigma, next_sigma = SIGMAS[index], SIGMAS[index + 1]
        x0 = OPS.add_scaled(state, _velocity(state, sigma, index), -sigma)
        state = OPS.interpolate(x0, provider(x0, sigma, next_sigma, index), next_sigma)
    return state


def test_fixed_seed_repeats_and_different_seed_diverges() -> None:
    first = _execute(123)
    repeat = _execute(123)
    alternate = _execute(124)

    assert first == repeat
    assert first.result_fingerprint == repeat.result_fingerprint
    assert first.noise_fingerprints == repeat.noise_fingerprints
    assert alternate.state != first.state
    assert alternate.result_fingerprint != first.result_fingerprint


def test_result_projection_is_noise_and_state_bound() -> None:
    result = _execute(123)
    projection = result.projection()

    assert projection["schema"] == "sigmax.stochastic-flow-euler-execution-result/1"
    assert projection["noise_ownership"] == "caller"
    counts = projection["counts"]
    assert isinstance(counts, dict)
    assert counts["effective_noise_draws"] == 3
    assert result.result_fingerprint == stochastic_flow_euler_execution_result_fingerprint(result)
    with pytest.raises(ScheduleContractError, match="noise fingerprint count"):
        replace(result, noise_fingerprints=result.noise_fingerprints[:-1])


@pytest.mark.parametrize(
    "spec",
    [
        replace(_spec(), solver_order=2),
        replace(_spec(), scheduler_index=1, begin_index=1),
        replace(_spec(), requested_model_evaluations=4),
        replace(_spec(), capabilities=replace(_capabilities(), supports_partial_denoise=True)),
        replace(
            _spec(),
            capabilities=replace(
                _capabilities(),
                execution_behavior=ExecutionBehavior.DETERMINISTIC,
                noise_ownership=NoiseOwnership.NONE,
            ),
            random_source_ownership=NoiseOwnership.NONE,
        ),
    ],
)
def test_incompatible_spec_rejects_before_evaluator_or_noise(spec: SamplerExecutionSpec) -> None:
    evaluator_calls = 0
    provider = SeededNoise(123)

    def evaluator(state: State, sigma: float, scheduler_index: int) -> State:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return _velocity(state, sigma, scheduler_index)

    with pytest.raises(ScheduleContractError):
        execute_stochastic_flow_euler(
            spec=spec,
            sigmas=SIGMAS,
            state=INITIAL,
            evaluator=evaluator,
            noise_provider=provider,
            operations=OPS,
        )
    assert evaluator_calls == 0
    assert provider.calls == []


def test_invalid_noise_rejects_without_returning_phantom_result() -> None:
    calls = 0

    def invalid_noise(
        reference: State,
        sigma: float,
        next_sigma: float,
        scheduler_index: int,
    ) -> State:
        nonlocal calls
        calls += 1
        return (math.nan,) * len(reference)

    with pytest.raises(ScheduleContractError, match="finite"):
        execute_stochastic_flow_euler(
            spec=_spec(),
            sigmas=SIGMAS,
            state=INITIAL,
            evaluator=_velocity,
            noise_provider=invalid_noise,
            operations=OPS,
        )
    assert calls == 1


def test_missing_direct_interpolation_operation_fails_before_calls() -> None:
    class DeterministicOnlyOperations:
        validate = OPS.validate
        fingerprint = OPS.fingerprint
        add_scaled = OPS.add_scaled

    provider = SeededNoise(123)
    evaluator_calls = 0

    def evaluator(state: State, sigma: float, scheduler_index: int) -> State:
        nonlocal evaluator_calls
        evaluator_calls += 1
        return _velocity(state, sigma, scheduler_index)

    with pytest.raises(ScheduleContractError, match="state contract"):
        execute_stochastic_flow_euler(
            spec=_spec(),
            sigmas=SIGMAS,
            state=INITIAL,
            evaluator=evaluator,
            noise_provider=provider,
            operations=cast(Any, DeterministicOnlyOperations()),
        )
    assert evaluator_calls == 0
    assert provider.calls == []
