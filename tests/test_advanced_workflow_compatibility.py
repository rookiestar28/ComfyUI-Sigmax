"""Pure advanced-workflow compatibility contract tests for M5-05."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax.core.advanced_workflow_compatibility import (
    ADVANCED_WORKFLOW_COMPATIBILITY_SCHEMA,
    AdvancedDecisionLevel,
    AdvancedExecutionMode,
    AdvancedOwnership,
    AdvancedReasonCode,
    AdvancedReceiptStatus,
    AdvancedWorkflowCompatibilityReceipt,
    AdvancedWorkflowFeature,
    AdvancedWorkflowRequest,
    build_advanced_workflow_receipt,
    deserialize_advanced_workflow_receipt,
    resolve_advanced_workflow,
    serialize_advanced_workflow_receipt,
)
from comfyui_sigmax.core.schedule_contracts import ScheduleContractError


def _request(
    *features: AdvancedWorkflowFeature,
    mode: AdvancedExecutionMode,
    host: tuple[AdvancedWorkflowFeature, ...] = (),
    state: tuple[str, ...] = (),
    snapshot: str | None = None,
    spec: str | None = None,
    snapshot_spec: str | None = None,
    model_object: bool = False,
    patch_object: bool = False,
) -> AdvancedWorkflowRequest:
    return AdvancedWorkflowRequest(
        features=features,
        execution_mode=mode,
        host_capabilities=host,
        required_state=state,
        snapshot_fingerprint=snapshot,
        spec_fingerprint=spec,
        snapshot_spec_fingerprint=snapshot_spec,
        model_object_supplied=model_object,
        patch_object_supplied=patch_object,
    )


def test_contract_declares_all_seven_features_in_canonical_order() -> None:
    assert tuple(AdvancedWorkflowFeature) == (
        AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
        AdvancedWorkflowFeature.INPAINTING,
        AdvancedWorkflowFeature.PARTIAL_DENOISE,
        AdvancedWorkflowFeature.CONTROLNET,
        AdvancedWorkflowFeature.MODEL_PATCHES,
        AdvancedWorkflowFeature.INTERRUPTION,
        AdvancedWorkflowFeature.RESUME,
    )


def test_native_host_requires_explicit_capability_for_each_feature() -> None:
    request = _request(
        AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
        AdvancedWorkflowFeature.INPAINTING,
        mode=AdvancedExecutionMode.NATIVE_HOST,
        host=(AdvancedWorkflowFeature.IMAGE_TO_IMAGE,),
    )

    decision = resolve_advanced_workflow(request)

    assert decision.level is AdvancedDecisionLevel.REJECT
    assert decision.ownership is AdvancedOwnership.HOST
    assert decision.reasons == (AdvancedReasonCode.HOST_CAPABILITY_MISSING,)


def test_deterministic_pure_supports_bounded_latent_and_partial_controller_decisions() -> None:
    request = _request(
        AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
        AdvancedWorkflowFeature.PARTIAL_DENOISE,
        mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
        state=("latent", "sigma_cursor"),
    )

    decision = resolve_advanced_workflow(request)

    assert decision.level is AdvancedDecisionLevel.ALLOW
    assert decision.ownership is AdvancedOwnership.CONTROLLER
    assert decision.reasons == (AdvancedReasonCode.COMPATIBLE,)


@pytest.mark.parametrize(
    "feature",
    (
        AdvancedWorkflowFeature.INPAINTING,
        AdvancedWorkflowFeature.CONTROLNET,
        AdvancedWorkflowFeature.MODEL_PATCHES,
    ),
)
def test_pure_controller_rejects_host_owned_features(feature: AdvancedWorkflowFeature) -> None:
    decision = resolve_advanced_workflow(
        _request(feature, mode=AdvancedExecutionMode.DETERMINISTIC_PURE)
    )

    assert decision.level is AdvancedDecisionLevel.REJECT
    assert decision.ownership is AdvancedOwnership.UNSUPPORTED
    assert AdvancedReasonCode.PURE_HOST_FEATURE_UNSUPPORTED in decision.reasons


def test_pure_boundary_rejects_model_or_patch_objects_without_serializing_them() -> None:
    decision = resolve_advanced_workflow(
        _request(
            AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
            mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
            model_object=True,
            patch_object=True,
        )
    )

    assert decision.level is AdvancedDecisionLevel.REJECT
    assert decision.reasons == (AdvancedReasonCode.PURE_MODEL_OWNERSHIP_UNSUPPORTED,)
    assert "model_object" not in json.dumps(decision.projection(), sort_keys=True)


def test_stochastic_pure_rejects_partial_and_resume_without_claiming_controller_support() -> None:
    decision = resolve_advanced_workflow(
        _request(
            AdvancedWorkflowFeature.PARTIAL_DENOISE,
            AdvancedWorkflowFeature.RESUME,
            mode=AdvancedExecutionMode.STOCHASTIC_PURE,
        )
    )

    assert decision.level is AdvancedDecisionLevel.REJECT
    assert decision.ownership is AdvancedOwnership.UNSUPPORTED
    assert decision.reasons == (
        AdvancedReasonCode.STOCHASTIC_PARTIAL_UNSUPPORTED,
        AdvancedReasonCode.STOCHASTIC_RESUME_UNSUPPORTED,
    )


def test_deterministic_resume_requires_matching_snapshot_and_spec_identity() -> None:
    valid = _request(
        AdvancedWorkflowFeature.RESUME,
        mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
        snapshot="sha256:" + "1" * 64,
        spec="sha256:" + "2" * 64,
        snapshot_spec="sha256:" + "2" * 64,
        state=("execution_cursor", "snapshot"),
    )
    decision = resolve_advanced_workflow(valid)
    assert decision.level is AdvancedDecisionLevel.ALLOW
    assert decision.reasons == (AdvancedReasonCode.COMPATIBLE,)

    missing = resolve_advanced_workflow(
        _request(AdvancedWorkflowFeature.RESUME, mode=AdvancedExecutionMode.DETERMINISTIC_PURE)
    )
    assert missing.reasons == (AdvancedReasonCode.RESUME_SNAPSHOT_REQUIRED,)

    mismatch = resolve_advanced_workflow(
        _request(
            AdvancedWorkflowFeature.RESUME,
            mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
            snapshot="sha256:" + "1" * 64,
            spec="sha256:" + "2" * 64,
            snapshot_spec="sha256:" + "3" * 64,
        )
    )
    assert mismatch.reasons == (AdvancedReasonCode.RESUME_SNAPSHOT_MISMATCH,)


def test_native_interruption_without_snapshot_is_warned_non_resumable() -> None:
    decision = resolve_advanced_workflow(
        _request(
            AdvancedWorkflowFeature.INTERRUPTION,
            mode=AdvancedExecutionMode.NATIVE_HOST,
            host=(AdvancedWorkflowFeature.INTERRUPTION,),
        )
    )

    assert decision.level is AdvancedDecisionLevel.WARN
    assert decision.ownership is AdvancedOwnership.HOST
    assert decision.reasons == (AdvancedReasonCode.HOST_INTERRUPT_NON_RESUMABLE,)


def test_receipt_round_trip_is_bounded_and_fingerprint_stable() -> None:
    request = _request(
        AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
        AdvancedWorkflowFeature.PARTIAL_DENOISE,
        mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
        state=("latent", "sigma_cursor"),
    )
    decision = resolve_advanced_workflow(request)
    receipt = build_advanced_workflow_receipt(
        request,
        decision,
        execution_status=AdvancedReceiptStatus.NOT_EXECUTED,
    )

    payload = serialize_advanced_workflow_receipt(receipt)
    rebuilt = deserialize_advanced_workflow_receipt(payload)

    assert receipt.receipt_fingerprint == rebuilt.receipt_fingerprint
    assert rebuilt.projection()["schema"] == ADVANCED_WORKFLOW_COMPATIBILITY_SCHEMA
    assert rebuilt.resumable is False
    assert len(payload) < 16_384


def test_receipt_rejects_noncanonical_transport() -> None:
    request = _request(
        AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
        mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
    )
    decision = resolve_advanced_workflow(request)
    receipt = build_advanced_workflow_receipt(request, decision)

    with pytest.raises(ScheduleContractError, match="canonical JSON"):
        deserialize_advanced_workflow_receipt(b" " + serialize_advanced_workflow_receipt(receipt))


def test_receipt_rejects_mismatched_decision_and_private_result_identity() -> None:
    request = _request(
        AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
        mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
    )
    decision = resolve_advanced_workflow(request)
    other = _request(
        AdvancedWorkflowFeature.PARTIAL_DENOISE,
        mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
    )
    other_decision = resolve_advanced_workflow(other)

    with pytest.raises(ScheduleContractError, match="decision does not match"):
        build_advanced_workflow_receipt(request, other_decision)

    with pytest.raises(ScheduleContractError, match="fingerprint"):
        build_advanced_workflow_receipt(
            request,
            decision,
            result_fingerprint="C:\\private\\latent.bin",
        )

    assert decision.request_fingerprint != other_decision.request_fingerprint
    assert decision.fingerprint != other_decision.fingerprint


def test_request_rejects_noncanonical_or_private_state() -> None:
    with pytest.raises(ScheduleContractError, match="canonical enum order"):
        _request(
            AdvancedWorkflowFeature.PARTIAL_DENOISE,
            AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
            mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
        )

    with pytest.raises(ScheduleContractError, match="bounded public identifier"):
        _request(
            AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
            mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
            state=("C:\\private\\prompt.txt",),
        )


def test_receipt_constructor_is_immutable() -> None:
    request = _request(
        AdvancedWorkflowFeature.IMAGE_TO_IMAGE,
        mode=AdvancedExecutionMode.DETERMINISTIC_PURE,
    )
    decision = resolve_advanced_workflow(request)
    receipt = AdvancedWorkflowCompatibilityReceipt(
        request=request,
        decision=decision,
        execution_status=AdvancedReceiptStatus.NOT_EXECUTED,
        resumable=False,
    )

    with pytest.raises(AttributeError):
        receipt.resumable = True  # type: ignore[misc]
