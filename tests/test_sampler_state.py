"""M5-02 immutable sampler state contract tests."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from comfyui_sigmax.core import (
    ExecutionBehavior,
    NoiseOwnership,
    PredictionType,
    SamplerCapabilities,
    SamplerExecutionSpec,
    SamplerState,
    SamplerStateSnapshot,
    SamplerStateStatus,
    SamplerStep,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalRequirement,
    deserialize_sampler_execution_spec,
    deserialize_sampler_state_snapshot,
    sampler_execution_spec_fingerprint,
    serialize_sampler_execution_spec,
    serialize_sampler_state_snapshot,
)


def _spec() -> SamplerExecutionSpec:
    return SamplerExecutionSpec(
        capabilities=SamplerCapabilities(
            sampler_id="comfy.euler",
            sampler_version="native",
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
            supports_per_token_timesteps=True,
        ),
        scheduler_index=4,
        begin_index=2,
        solver_order=1,
        timestep_spacing="native",
        random_source_ownership=NoiseOwnership.NONE,
        per_token_time=(0.25, 0.5),
        requested_transitions=2,
        requested_model_evaluations=2,
    )


def _fingerprint(character: str) -> str:
    return "sha256:" + character * 64


def _step(index: int, *, evaluations: int = 1) -> SamplerStep:
    input_character = "1" if index == 0 else str(index + 2)
    return SamplerStep(
        step_index=index,
        scheduler_index=4 + index,
        sigma=1.0 - index * 0.25,
        next_sigma=0.75 - index * 0.25,
        model_evaluations=evaluations,
        input_state_fingerprint=_fingerprint(input_character),
        output_state_fingerprint=_fingerprint(str(index + 3)),
    )


def test_state_is_immutable_and_tracks_effective_counts() -> None:
    spec = _spec()
    initial = SamplerStateSnapshot.initial(spec)
    running = initial.append_step(spec, _step(0))
    completed = running.append_step(spec, _step(1)).complete(spec)

    assert initial.status is SamplerStateStatus.READY
    assert initial.effective_transitions == 0
    assert running.status is SamplerStateStatus.RUNNING
    assert running.effective_model_evaluations == 1
    assert completed.status is SamplerStateStatus.COMPLETED
    assert completed.effective_transitions == 2
    assert completed.effective_model_evaluations == 2
    with pytest.raises(FrozenInstanceError):
        initial.next_step_index = 1  # type: ignore[misc]


def test_interrupted_state_resumes_without_mutating_history() -> None:
    spec = _spec()
    running = SamplerStateSnapshot.initial(spec).append_step(spec, _step(0))
    interrupted = running.interrupt(spec)
    resumed = interrupted.resume(spec)

    assert interrupted.status is SamplerStateStatus.INTERRUPTED
    assert resumed.status is SamplerStateStatus.RUNNING
    assert resumed.history == running.history
    assert running.status is SamplerStateStatus.RUNNING


def test_round_trip_and_fingerprint_are_byte_deterministic() -> None:
    spec = _spec()
    spec_bytes = serialize_sampler_execution_spec(spec)
    assert spec_bytes == serialize_sampler_execution_spec(
        deserialize_sampler_execution_spec(spec_bytes)
    )
    assert sampler_execution_spec_fingerprint(spec) == sampler_execution_spec_fingerprint(
        deserialize_sampler_execution_spec(spec_bytes)
    )

    state = SamplerStateSnapshot.initial(spec).append_step(spec, _step(0))
    state_bytes = serialize_sampler_state_snapshot(state, spec)
    restored = deserialize_sampler_state_snapshot(state_bytes, spec)
    assert state_bytes == serialize_sampler_state_snapshot(restored, spec)
    assert restored == state


def test_state_requires_matching_spec_and_contiguous_steps() -> None:
    spec = _spec()
    different = SamplerExecutionSpec(
        capabilities=spec.capabilities,
        scheduler_index=5,
        begin_index=spec.begin_index,
        solver_order=spec.solver_order,
        timestep_spacing=spec.timestep_spacing,
        random_source_ownership=spec.random_source_ownership,
        per_token_time=spec.per_token_time,
        requested_transitions=spec.requested_transitions,
        requested_model_evaluations=spec.requested_model_evaluations,
    )
    state = SamplerStateSnapshot.initial(spec)
    with pytest.raises(ScheduleContractError, match="does not match"):
        state.append_step(different, _step(0))
    with pytest.raises(ScheduleContractError, match="contiguous"):
        state.append_step(spec, _step(1))


def test_state_requires_fingerprint_continuity_between_steps() -> None:
    spec = _spec()
    running = SamplerStateSnapshot.initial(spec).append_step(spec, _step(0))
    discontinuous = replace(_step(1), input_state_fingerprint=_fingerprint("9"))

    with pytest.raises(ScheduleContractError, match="input state fingerprint"):
        running.append_step(spec, discontinuous)


def test_state_rejects_inconsistent_counts_and_unknown_fields() -> None:
    spec = _spec()
    state = SamplerStateSnapshot.initial(spec).append_step(spec, _step(0))
    payload = json.loads(serialize_sampler_state_snapshot(state, spec))
    payload["effective_transitions"] = 0
    with pytest.raises(ScheduleContractError, match="transition count"):
        deserialize_sampler_state_snapshot(json.dumps(payload), spec)

    payload = json.loads(serialize_sampler_state_snapshot(state, spec))
    payload["unexpected"] = True
    with pytest.raises(ScheduleContractError, match="fields"):
        deserialize_sampler_state_snapshot(json.dumps(payload), spec)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solver_order", 0),
        ("requested_transitions", -1),
        ("per_token_time", (float("nan"),)),
    ],
)
def test_spec_rejects_invalid_state_dimensions(field: str, value: object) -> None:
    values: dict[str, Any] = {
        "capabilities": _spec().capabilities,
        "scheduler_index": _spec().scheduler_index,
        "begin_index": _spec().begin_index,
        "solver_order": _spec().solver_order,
        "timestep_spacing": _spec().timestep_spacing,
        "random_source_ownership": _spec().random_source_ownership,
        "per_token_time": _spec().per_token_time,
        "requested_transitions": _spec().requested_transitions,
        "requested_model_evaluations": _spec().requested_model_evaluations,
    }
    values[field] = value
    with pytest.raises(ScheduleContractError):
        SamplerExecutionSpec(**values)
