"""Contracts for schedule ownership, domains, and transform ordering."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from comfyui_sigmax.core import (
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TransformContract,
    TransformStage,
    require_single_ownership,
    validate_transform_chain,
)


def _transform(
    name: str,
    stage: TransformStage,
    input_domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
    output_domain: SigmaDomain = SigmaDomain.UNIT_FLOW,
) -> TransformContract:
    return TransformContract(
        name=name,
        stage=stage,
        input_domain=input_domain,
        output_domain=output_domain,
    )


def test_contract_wire_values_are_explicit_and_complete() -> None:
    assert {mode.value for mode in ScheduleOwnership} == {
        "MODEL_NATIVE",
        "EXTERNAL_SIGMAS",
        "MODEL_PATCH",
    }
    assert {domain.value for domain in SigmaDomain} == {
        "UNIT_FLOW",
        "MODEL_NATIVE",
        "CONTINUOUS_EDM",
        "DISCRETE_TRAINING_INDEX",
    }
    assert [stage.value for stage in TransformStage] == [
        "PRIMARY_TIME_SHIFT",
        "OPTIONAL_SPACING",
        "TERMINAL",
        "SLICE",
    ]


@pytest.mark.parametrize("mode", list(ScheduleOwnership))
def test_each_single_ownership_mode_resolves(mode: ScheduleOwnership) -> None:
    assert require_single_ownership(mode) is mode


@pytest.mark.parametrize(
    "modes",
    [
        (),
        (ScheduleOwnership.MODEL_NATIVE, ScheduleOwnership.EXTERNAL_SIGMAS),
        (ScheduleOwnership.MODEL_PATCH, ScheduleOwnership.MODEL_PATCH),
    ],
)
def test_ownership_requires_exactly_one_mode(modes: tuple[ScheduleOwnership, ...]) -> None:
    with pytest.raises(ScheduleContractError, match="exactly one"):
        require_single_ownership(*modes)


def test_ownership_rejects_untyped_wire_value() -> None:
    with pytest.raises(ScheduleContractError, match="unsupported"):
        require_single_ownership(cast(ScheduleOwnership, "MODEL_NATIVE"))


@pytest.mark.parametrize(
    "ownership",
    [ScheduleOwnership.MODEL_NATIVE, ScheduleOwnership.MODEL_PATCH],
)
def test_native_or_patched_ownership_rejects_external_shift_before_execution(
    ownership: ScheduleOwnership,
) -> None:
    execution_reached = False

    with pytest.raises(ScheduleContractError, match="double shifting"):
        validate_transform_chain(
            ownership,
            SigmaDomain.MODEL_NATIVE,
            [_transform("krea exponential mu", TransformStage.PRIMARY_TIME_SHIFT)],
        )
        execution_reached = True

    assert execution_reached is False


def test_native_and_patch_ownership_accept_only_opaque_native_domain() -> None:
    for ownership in (ScheduleOwnership.MODEL_NATIVE, ScheduleOwnership.MODEL_PATCH):
        assert (
            validate_transform_chain(ownership, SigmaDomain.MODEL_NATIVE, [])
            is SigmaDomain.MODEL_NATIVE
        )
        with pytest.raises(ScheduleContractError, match="MODEL_NATIVE sigma domain"):
            validate_transform_chain(ownership, SigmaDomain.UNIT_FLOW, [])


def test_external_ownership_rejects_opaque_model_native_domain() -> None:
    with pytest.raises(ScheduleContractError, match="opaque MODEL_NATIVE"):
        validate_transform_chain(
            ScheduleOwnership.EXTERNAL_SIGMAS,
            SigmaDomain.MODEL_NATIVE,
            [],
        )


def test_transform_validation_rejects_untyped_ownership_value() -> None:
    with pytest.raises(ScheduleContractError, match="unsupported"):
        validate_transform_chain(
            cast(ScheduleOwnership, "EXTERNAL_SIGMAS"),
            SigmaDomain.UNIT_FLOW,
            [],
        )


def test_valid_external_chain_returns_explicit_final_domain() -> None:
    final_domain = validate_transform_chain(
        ScheduleOwnership.EXTERNAL_SIGMAS,
        SigmaDomain.DISCRETE_TRAINING_INDEX,
        [
            _transform(
                "normalize flow time",
                TransformStage.PRIMARY_TIME_SHIFT,
                SigmaDomain.DISCRETE_TRAINING_INDEX,
                SigmaDomain.UNIT_FLOW,
            ),
            _transform("profile spacing", TransformStage.OPTIONAL_SPACING),
            _transform("append zero", TransformStage.TERMINAL),
            _transform("partial denoise", TransformStage.SLICE),
        ],
    )

    assert final_domain is SigmaDomain.UNIT_FLOW


def test_external_chain_rejects_domain_discontinuity() -> None:
    with pytest.raises(ScheduleContractError, match="expects CONTINUOUS_EDM"):
        validate_transform_chain(
            ScheduleOwnership.EXTERNAL_SIGMAS,
            SigmaDomain.UNIT_FLOW,
            [
                _transform(
                    "wrong-domain shift",
                    TransformStage.PRIMARY_TIME_SHIFT,
                    SigmaDomain.CONTINUOUS_EDM,
                    SigmaDomain.UNIT_FLOW,
                )
            ],
        )


def test_external_chain_rejects_stage_regression() -> None:
    with pytest.raises(ScheduleContractError, match="stage order"):
        validate_transform_chain(
            ScheduleOwnership.EXTERNAL_SIGMAS,
            SigmaDomain.UNIT_FLOW,
            [
                _transform("append zero", TransformStage.TERMINAL),
                _transform("late shift", TransformStage.PRIMARY_TIME_SHIFT),
            ],
        )


@pytest.mark.parametrize(
    "stage",
    [TransformStage.PRIMARY_TIME_SHIFT, TransformStage.OPTIONAL_SPACING],
)
def test_external_chain_rejects_duplicate_singleton_stage(stage: TransformStage) -> None:
    with pytest.raises(ScheduleContractError, match="at most one"):
        validate_transform_chain(
            ScheduleOwnership.EXTERNAL_SIGMAS,
            SigmaDomain.UNIT_FLOW,
            [_transform("first", stage), _transform("second", stage)],
        )


def test_transform_contract_is_immutable() -> None:
    contract = _transform("fixed shift", TransformStage.PRIMARY_TIME_SHIFT)

    with pytest.raises(FrozenInstanceError):
        contract.name = "changed"  # type: ignore[misc]


def test_transform_contract_rejects_empty_name() -> None:
    with pytest.raises(ScheduleContractError, match="name must not be empty"):
        _transform("   ", TransformStage.TERMINAL)
