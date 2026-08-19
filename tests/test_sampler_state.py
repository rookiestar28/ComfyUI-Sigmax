"""M5-02 immutable sampler state contract tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    CapabilityDimension,
    ExecutionBehavior,
    ExecutionReceipt,
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
    canonical_projection_bytes,
    deserialize_sampler_execution_spec,
    deserialize_sampler_state_snapshot,
    sampler_execution_spec_fingerprint,
    serialize_sampler_execution_spec,
    serialize_sampler_state_snapshot,
)
from comfyui_sigmax.core import sampler_state as sampler_state_module


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


def _receipt(
    spec: SamplerExecutionSpec,
    *,
    sampler_rng: str = "none",
    sampler_id: str = "comfy.euler",
    sampler_version: str = "native",
    execution_status: str = "succeeded",
    execution_reason_code: str | None = None,
    effective_transitions: int | None = None,
    effective_model_evaluations: int | None = None,
    host_revision: str = "test-revision",
) -> ExecutionReceipt:
    actual_transitions = (
        spec.requested_transitions if effective_transitions is None else effective_transitions
    )
    actual_model_evaluations = (
        spec.requested_model_evaluations
        if effective_model_evaluations is None
        else effective_model_evaluations
    )
    projection: dict[str, object] = {
        "artifact": {
            "construction_fingerprint": _fingerprint("a"),
            "numerical_fingerprint": _fingerprint("b"),
        },
        "compatibility": {
            "considered": [item.value for item in CapabilityDimension],
            "level": "allow",
            "reasons": ["compatible"],
        },
        "counts": {
            "effective_model_evaluations": actual_model_evaluations,
            "effective_transitions": actual_transitions,
            "requested_model_evaluations": spec.requested_model_evaluations,
            "requested_transitions": spec.requested_transitions,
        },
        "effective_inputs": {
            "compatibility": {},
            "height": None,
            "precision": "float64",
            "profile": "fixture.profile",
            "profile_version": "1",
            "steps": spec.requested_transitions,
            "width": None,
        },
        "execution": {
            "reason_code": execution_reason_code,
            "status": execution_status,
        },
        "host": {
            "api_version": "test",
            "id": "comfyui",
            "revision": host_revision,
            "version": "0.30.0",
        },
        "model": {
            "fingerprint": _fingerprint("c"),
            "id": "fixture.model",
            "version": "1",
        },
        "profile": {"id": "fixture.profile", "version": "1"},
        "rng_ownership": {"model": "none", "sampler": sampler_rng, "schedule": "none"},
        "sampler": {
            "fingerprint": _fingerprint("d"),
            "id": sampler_id,
            "version": sampler_version,
        },
        "schema": "sigmax.execution-receipt/1",
    }
    payload = canonical_projection_bytes(projection)
    return ExecutionReceipt(
        receipt_bytes=payload,
        receipt_fingerprint=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        construction_fingerprint=_fingerprint("a"),
        numerical_fingerprint=_fingerprint("b"),
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


def test_schema_v1_spec_and_snapshot_goldens_are_frozen() -> None:
    spec = _spec()
    state = SamplerStateSnapshot.initial(spec).append_step(spec, _step(0))
    expected_spec = (
        b'{"begin_index":2,"capabilities":{"accepted_ownerships":["EXTERNAL_SIGMAS"],'
        b'"accepted_prediction_types":["flow_velocity"],"accepted_sigma_domains":'
        b'["UNIT_FLOW"],"execution_behavior":"deterministic","noise_ownership":"none",'
        b'"required_state":["begin_index","step_index","multistep_history","resume"],'
        b'"sampler_id":"comfy.euler","sampler_version":"native",'
        b'"supports_partial_denoise":true,"supports_per_token_timesteps":true,'
        b'"terminal_requirement":"requires_zero"},"per_token_time":'
        b'["3fd0000000000000","3fe0000000000000"],"random_source_ownership":"none",'
        b'"requested_model_evaluations":2,"requested_transitions":2,'
        b'"scheduler_index":4,"schema":"sigmax.sampler-execution-spec/1",'
        b'"solver_order":1,"timestep_spacing":"native"}'
    )
    expected_state = (
        b'{"effective_model_evaluations":1,"effective_transitions":1,'
        b'"execution_receipt_fingerprint":null,"history":'
        b'[{"input_state_fingerprint":"sha256:'
        b'1111111111111111111111111111111111111111111111111111111111111111",'
        b'"model_evaluations":1,"next_sigma":"3fe8000000000000",'
        b'"output_state_fingerprint":"sha256:'
        b'3333333333333333333333333333333333333333333333333333333333333333",'
        b'"scheduler_index":4,"sigma":"3ff0000000000000","step_index":0}],'
        b'"next_step_index":1,"schema":"sigmax.sampler-state-snapshot/1",'
        b'"spec_fingerprint":"sha256:'
        b'b0433c362287832b9e92868894ea03d4cb78520a90ef3054aee824da14c86887",'
        b'"status":"running"}'
    )

    assert serialize_sampler_execution_spec(spec) == expected_spec
    assert sampler_execution_spec_fingerprint(spec) == (
        "sha256:b0433c362287832b9e92868894ea03d4cb78520a90ef3054aee824da14c86887"
    )
    assert serialize_sampler_state_snapshot(state, spec) == expected_state
    assert sampler_state_module.sampler_state_snapshot_fingerprint(state, spec) == (
        "sha256:d5a8c1c2b657feedd53fdbbb495f24dca06d0aebdf0815323e88afcbc7abe416"
    )


def test_snapshot_fingerprint_is_state_bound_and_round_trip_stable() -> None:
    spec = _spec()
    initial = SamplerStateSnapshot.initial(spec)
    running = initial.append_step(spec, _step(0))
    fingerprint = getattr(sampler_state_module, "sampler_state_snapshot_fingerprint", None)

    assert callable(fingerprint)
    initial_fingerprint = fingerprint(initial, spec)
    running_fingerprint = fingerprint(running, spec)
    restored = deserialize_sampler_state_snapshot(
        serialize_sampler_state_snapshot(running, spec),
        spec,
    )

    assert initial_fingerprint.startswith("sha256:")
    assert running_fingerprint.startswith("sha256:")
    assert initial_fingerprint != running_fingerprint
    assert running_fingerprint == fingerprint(restored, spec)


def test_execution_receipt_binding_is_spec_and_state_consistent() -> None:
    spec = _spec()
    running = (
        SamplerStateSnapshot.initial(spec).append_step(spec, _step(0)).append_step(spec, _step(1))
    )
    completed = running.complete(spec)
    receipt = _receipt(spec)
    binding = getattr(SamplerStateSnapshot, "attach_execution_receipt_evidence", None)

    assert callable(binding)
    with pytest.raises(ScheduleContractError, match="running sampler state"):
        binding(running, spec, receipt)
    attached = binding(completed, spec, receipt)
    assert attached.execution_receipt_fingerprint == receipt.receipt_fingerprint

    with pytest.raises(ScheduleContractError, match="ExecutionReceipt"):
        completed.attach_execution_receipt(spec, cast(ExecutionReceipt, _fingerprint("e")))
    legacy_named_binding = completed.attach_execution_receipt(spec, receipt)
    assert legacy_named_binding.execution_receipt_fingerprint == receipt.receipt_fingerprint

    with pytest.raises(ScheduleContractError, match="sampler RNG ownership"):
        binding(completed, spec, _receipt(spec, sampler_rng="caller"))


@pytest.mark.parametrize(
    ("field", "value"),
    [("sampler_id", "other.euler"), ("sampler_version", "other-version")],
)
def test_execution_receipt_binding_requires_exact_sampler_identity(
    field: str,
    value: str,
) -> None:
    spec = _spec()
    completed = (
        SamplerStateSnapshot.initial(spec)
        .append_step(spec, _step(0))
        .append_step(spec, _step(1))
        .complete(spec)
    )

    receipt = (
        _receipt(spec, sampler_id=value)
        if field == "sampler_id"
        else _receipt(spec, sampler_version=value)
    )
    with pytest.raises(ScheduleContractError, match="sampler identity"):
        completed.attach_execution_receipt_evidence(spec, receipt)


def test_receipt_binding_is_terminal_and_same_receipt_is_idempotent() -> None:
    spec = _spec()
    ready_receipt = _receipt(
        spec,
        execution_status="not_executed",
        effective_transitions=0,
        effective_model_evaluations=0,
    )
    ready = SamplerStateSnapshot.initial(spec).attach_execution_receipt_evidence(
        spec,
        ready_receipt,
    )
    with pytest.raises(ScheduleContractError, match="receipt-bound"):
        ready.append_step(spec, _step(0))
    with pytest.raises(ScheduleContractError, match="receipt-bound"):
        ready.interrupt(spec)

    interrupted = SamplerStateSnapshot.initial(spec).append_step(spec, _step(0)).interrupt(spec)
    interrupted_receipt = _receipt(
        spec,
        execution_status="interrupted",
        execution_reason_code="interrupted",
        effective_transitions=1,
        effective_model_evaluations=1,
    )
    bound_interrupted = interrupted.attach_execution_receipt_evidence(
        spec,
        interrupted_receipt,
    )
    with pytest.raises(ScheduleContractError, match="receipt-bound"):
        bound_interrupted.resume(spec)

    completed = interrupted.resume(spec).append_step(spec, _step(1)).complete(spec)
    receipt = _receipt(spec)
    attached = completed.attach_execution_receipt_evidence(spec, receipt)
    assert attached.attach_execution_receipt_evidence(spec, receipt) == attached
    with pytest.raises(ScheduleContractError, match="cannot be replaced"):
        attached.attach_execution_receipt_evidence(
            spec,
            _receipt(spec, host_revision="other-revision"),
        )


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
        deserialize_sampler_state_snapshot(canonical_projection_bytes(payload), spec)

    payload = json.loads(serialize_sampler_state_snapshot(state, spec))
    payload["unexpected"] = True
    with pytest.raises(ScheduleContractError, match="fields"):
        deserialize_sampler_state_snapshot(canonical_projection_bytes(payload), spec)


def test_state_rejects_inconsistent_lifecycle_status_on_restore() -> None:
    spec = _spec()
    running = SamplerStateSnapshot.initial(spec).append_step(spec, _step(0))
    payload = json.loads(serialize_sampler_state_snapshot(running, spec))

    payload["status"] = SamplerStateStatus.READY.value
    with pytest.raises(ScheduleContractError, match="ready state cannot contain executed steps"):
        deserialize_sampler_state_snapshot(canonical_projection_bytes(payload), spec)

    payload["status"] = SamplerStateStatus.COMPLETED.value

    with pytest.raises(ScheduleContractError, match="completed state counts"):
        deserialize_sampler_state_snapshot(canonical_projection_bytes(payload), spec)

    completed = (
        running.append_step(spec, _step(1))
        .complete(spec)
        .attach_execution_receipt_evidence(spec, _receipt(spec))
    )
    payload = json.loads(serialize_sampler_state_snapshot(completed, spec))
    payload["status"] = SamplerStateStatus.RUNNING.value

    with pytest.raises(ScheduleContractError, match="running state cannot carry receipt"):
        deserialize_sampler_state_snapshot(canonical_projection_bytes(payload), spec)


def test_spec_requires_capability_consistent_rng_and_per_token_time() -> None:
    spec = _spec()
    caller_owned = replace(
        spec.capabilities,
        execution_behavior=ExecutionBehavior.STOCHASTIC,
        noise_ownership=NoiseOwnership.CALLER,
    )
    with pytest.raises(ScheduleContractError, match="noise ownership"):
        replace(spec, capabilities=caller_owned)

    no_per_token = replace(spec.capabilities, supports_per_token_timesteps=False)
    with pytest.raises(ScheduleContractError, match="per-token"):
        replace(spec, capabilities=no_per_token)
    assert replace(spec, capabilities=no_per_token, per_token_time=None).per_token_time is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xef\xbb\xbf{}", "BOM"),
        ("\ufeff{}", "BOM"),
        (b'{"schema":"x","schema":"x"}', "duplicate JSON object name"),
        (b'{"value":0.25}', "untyped JSON float"),
        (b'{"value":NaN}', "non-finite JSON constant"),
        (b"{\xff}", "valid JSON"),
        ('{"schema":"x"} ', "canonical JSON"),
        ("\ud800", "valid Unicode"),
    ],
)
def test_sampler_transport_rejects_ambiguous_or_noncanonical_json(
    payload: bytes | str,
    message: str,
) -> None:
    with pytest.raises(ScheduleContractError, match=message):
        deserialize_sampler_execution_spec(payload)


def test_sampler_transport_rejects_oversized_payload() -> None:
    with pytest.raises(ScheduleContractError, match="size"):
        deserialize_sampler_execution_spec(b"{" + b" " * 1_048_576)


@pytest.mark.parametrize(
    "timestep_spacing",
    [r"A:\\Users\\Ray\\private-grid", "/home/ray/private-grid", "api_token"],
)
def test_spec_rejects_private_or_secret_like_timestep_spacing(
    timestep_spacing: str,
) -> None:
    with pytest.raises(ScheduleContractError, match="public text"):
        replace(_spec(), timestep_spacing=timestep_spacing)


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
