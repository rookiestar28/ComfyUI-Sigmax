"""M7-12 pure MiniMax H3 backend/readiness receipt contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.core.h3_backend_receipts import (
    H3_BACKEND_RECEIPT_SCHEMA,
    AttentionBackend,
    BackendObservation,
    BackendRequest,
    BackendResult,
    BackendResultStatus,
    CheckpointIdentity,
    ExecutionEnvironment,
    H3BackendExecutionReceipt,
    ObservationSource,
    OperationBackend,
    RunEvidence,
    ScheduleIdentity,
    build_h3_backend_execution_receipt,
    deserialize_h3_backend_execution_receipt,
    serialize_h3_backend_execution_receipt,
)


def _fingerprint(character: str) -> str:
    return "sha256:" + character * 64


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _schedule() -> ScheduleIdentity:
    return ScheduleIdentity(
        profile_id="minimax-h3.turbo.fl2va",
        recipe_id="h3.fl2va.lightx2v-turbo-8-v1.0-544p",
        construction_fingerprint=_fingerprint("1"),
        numerical_fingerprint=_fingerprint("2"),
    )


def _request(
    *,
    operation: OperationBackend = OperationBackend.EAGER,
    attention: AttentionBackend = AttentionBackend.PYTORCH,
    enable_triton: bool = False,
    use_ck: bool = False,
    override_requested: bool = False,
) -> BackendRequest:
    return BackendRequest(
        operation_backend=operation,
        attention_backend=attention,
        enable_triton_backend=enable_triton,
        use_ck_attention=use_ck,
        override_requested=override_requested,
    )


def _observation(
    *,
    status: BackendResultStatus = BackendResultStatus.NOT_EXECUTED,
    source: ObservationSource = ObservationSource.SYNTHETIC,
    operation: OperationBackend = OperationBackend.NOT_OBSERVED,
    attention: AttentionBackend = AttentionBackend.NOT_OBSERVED,
    reason: str = "backend.not_authorized",
) -> BackendObservation:
    return BackendObservation(
        source=source,
        status=status,
        actual_operation_backend=operation,
        actual_attention_backend=attention,
        reason_codes=(reason,),
    )


def _environment() -> ExecutionEnvironment:
    unavailable = CheckpointIdentity.unavailable()
    return ExecutionEnvironment(
        comfyui="comfyui/0.30.0",
        comfy_kitchen="unavailable",
        torch="torch/cpu",
        accelerator="unavailable",
        driver="unavailable",
        gpu="unavailable",
        os="windows",
        dtype="float32",
        checkpoint=unavailable,
        lora=unavailable,
        launch_config="sigmax.synthetic.readiness",
    )


def _evidence() -> RunEvidence:
    return RunEvidence(
        warmup_runs=0,
        first_latency_us=None,
        repeat_latency_us=None,
        peak_memory_bytes=None,
        stable_repeat=None,
        output_fingerprint=None,
        cleanup_status="not_applicable",
        mutation_status="not_applicable",
    )


def _receipt(**changes: object) -> H3BackendExecutionReceipt:
    values: dict[str, object] = {
        "schedule": _schedule(),
        "request": _request(),
        "observation": _observation(),
        "environment": _environment(),
        "evidence": _evidence(),
        "result": BackendResult(
            status=BackendResultStatus.NOT_EXECUTED,
            reason_code="backend.not_authorized",
        ),
    }
    values.update(changes)
    return build_h3_backend_execution_receipt(**values)  # type: ignore[arg-type]


def test_synthetic_receipt_is_redacted_deterministic_and_round_trips() -> None:
    first = _receipt()
    second = _receipt()

    assert isinstance(first, H3BackendExecutionReceipt)
    assert first.projection() == second.projection()
    assert first.projection()["schema"] == H3_BACKEND_RECEIPT_SCHEMA
    assert set(first.projection()) == {
        "schema",
        "receipt_fingerprint",
        "schedule",
        "request",
        "observation",
        "environment",
        "evidence",
        "result",
    }
    payload = serialize_h3_backend_execution_receipt(first)
    assert deserialize_h3_backend_execution_receipt(payload) == first
    assert payload == serialize_h3_backend_execution_receipt(second)
    encoded = payload.decode("utf-8")
    assert "not_authorized" in encoded
    assert '"path"' not in encoded
    assert "C:\\" not in encoded


def test_flags_and_actual_axes_are_independent() -> None:
    receipt = _receipt(
        request=_request(
            operation=OperationBackend.TRITON,
            attention=AttentionBackend.PYTORCH,
            enable_triton=True,
            use_ck=False,
        )
    )
    projection = receipt.projection()
    assert projection["request"] == {
        "attention_backend": "pytorch",
        "enable_triton_backend": True,
        "operation_backend": "triton",
        "use_ck_attention": False,
    }
    observation = _object(projection["observation"])
    assert observation["actual_operation_backend"] == "not_observed"
    assert observation["actual_attention_backend"] == "not_observed"

    ck_flag_only = _receipt(
        request=_request(
            operation=OperationBackend.EAGER,
            attention=AttentionBackend.PYTORCH,
            enable_triton=False,
            use_ck=True,
        )
    )
    request = _object(ck_flag_only.projection()["request"])
    assert request["use_ck_attention"] is True
    assert request["attention_backend"] == "pytorch"


def test_synthetic_observation_cannot_claim_actual_success() -> None:
    with pytest.raises(ScheduleContractError, match=r"synthetic|actual"):
        _receipt(
            observation=_observation(
                status=BackendResultStatus.SUCCEEDED,
                operation=OperationBackend.EAGER,
                attention=AttentionBackend.PYTORCH,
                reason="backend.synthetic_success",
            ),
            result=BackendResult(status=BackendResultStatus.SUCCEEDED, reason_code=None),
        )


def test_non_success_cannot_fabricate_metrics_or_output() -> None:
    with pytest.raises(ScheduleContractError, match=r"evidence|not-executed|output"):
        _receipt(
            evidence=RunEvidence(
                warmup_runs=1,
                first_latency_us=10,
                repeat_latency_us=None,
                peak_memory_bytes=None,
                stable_repeat=None,
                output_fingerprint=_fingerprint("3"),
                cleanup_status="not_applicable",
                mutation_status="not_applicable",
            )
        )


def test_override_requests_are_rejected_without_host_mutation() -> None:
    with pytest.raises(ScheduleContractError, match="override"):
        _receipt(request=_request(override_requested=True))


def test_private_paths_and_secret_like_values_are_rejected() -> None:
    with pytest.raises(ScheduleContractError, match=r"public|path"):
        ExecutionEnvironment(
            comfyui="C:\\private\\comfyui",
            comfy_kitchen="unavailable",
            torch="torch/cpu",
            accelerator="unavailable",
            driver="unavailable",
            gpu="unavailable",
            os="windows",
            dtype="float32",
            checkpoint=CheckpointIdentity.unavailable(),
            lora=CheckpointIdentity.unavailable(),
            launch_config="sigmax.synthetic.readiness",
        )


def test_receipt_tampering_is_detected_and_types_are_immutable() -> None:
    receipt = _receipt()
    projection = receipt.projection()
    _object(projection["result"])["reason_code"] = "changed"
    tampered = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ScheduleContractError, match="fingerprint"):
        deserialize_h3_backend_execution_receipt(tampered)
    with pytest.raises(FrozenInstanceError):
        receipt.receipt_fingerprint = _fingerprint("9")  # type: ignore[misc]
