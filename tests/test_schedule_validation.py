"""Numerical validation tests for complete sigma schedules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pytest
from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    validate_sigma_schedule,
)


def test_valid_unit_flow_schedule_returns_immutable_float_tuple() -> None:
    result = validate_sigma_schedule(
        [1, 0.75, 0.5, 0.25, 0],
        domain=SigmaDomain.UNIT_FLOW,
        expected_steps=4,
        require_terminal_zero=True,
    )

    assert result == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert isinstance(result, tuple)


@pytest.mark.parametrize(
    "sigmas",
    [
        (1.0, float("nan"), 0.0),
        (1.0, float("inf"), 0.0),
        (1.0, float("-inf"), 0.0),
    ],
)
def test_nonfinite_sigmas_fail(sigmas: tuple[float, ...]) -> None:
    with pytest.raises(ScheduleContractError, match="finite"):
        validate_sigma_schedule(
            sigmas,
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=2,
            require_terminal_zero=True,
        )


@pytest.mark.parametrize(
    "sigmas",
    [
        (1.0, True, 0.0),
        (1.0, cast(float, "0.5"), 0.0),
    ],
)
def test_non_numeric_or_boolean_sigmas_fail(sigmas: tuple[object, ...]) -> None:
    with pytest.raises(ScheduleContractError, match="numeric"):
        validate_sigma_schedule(
            cast(Iterable[float], sigmas),
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=2,
            require_terminal_zero=True,
        )


@pytest.mark.parametrize(
    ("domain", "sigmas"),
    [
        (SigmaDomain.UNIT_FLOW, (1.1, 0.5, 0.0)),
        (SigmaDomain.UNIT_FLOW, (1.0, 0.5, -0.1)),
        (SigmaDomain.CONTINUOUS_EDM, (4.0, 1.0, -0.1)),
        (SigmaDomain.DISCRETE_TRAINING_INDEX, (999.0, 10.0, -1.0)),
    ],
)
def test_wrong_domain_values_fail(
    domain: SigmaDomain,
    sigmas: tuple[float, ...],
) -> None:
    with pytest.raises(ScheduleContractError, match="domain"):
        validate_sigma_schedule(
            sigmas,
            domain=domain,
            expected_steps=2,
            require_terminal_zero=False,
        )


def test_opaque_model_native_domain_fails_closed() -> None:
    with pytest.raises(ScheduleContractError, match="MODEL_NATIVE"):
        validate_sigma_schedule(
            (1.0, 0.0),
            domain=SigmaDomain.MODEL_NATIVE,
            expected_steps=1,
            require_terminal_zero=True,
        )


def test_untyped_domain_fails_closed() -> None:
    with pytest.raises(ScheduleContractError, match="unsupported sigma domain"):
        validate_sigma_schedule(
            (1.0, 0.0),
            domain=cast(SigmaDomain, "UNIT_FLOW"),
            expected_steps=1,
            require_terminal_zero=True,
        )


def test_terminal_flag_must_be_boolean() -> None:
    with pytest.raises(ScheduleContractError, match="must be boolean"):
        validate_sigma_schedule(
            (1.0, 0.0),
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=1,
            require_terminal_zero=cast(bool, 1),
        )


@pytest.mark.parametrize(
    "sigmas",
    [
        (1.0, 0.5, 0.75, 0.0),
        (1.0, 0.5, 0.5, 0.0),
    ],
)
def test_non_monotonic_or_duplicate_sigmas_fail(sigmas: tuple[float, ...]) -> None:
    with pytest.raises(ScheduleContractError, match="strictly decreasing"):
        validate_sigma_schedule(
            sigmas,
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=3,
            require_terminal_zero=True,
        )


@pytest.mark.parametrize("expected_steps", [0, -1, True])
def test_expected_steps_must_be_positive_integer(expected_steps: int) -> None:
    with pytest.raises(ScheduleContractError, match="expected_steps"):
        validate_sigma_schedule(
            (1.0, 0.0),
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=expected_steps,
            require_terminal_zero=True,
        )


def test_wrong_transition_count_fails() -> None:
    with pytest.raises(ScheduleContractError, match="length"):
        validate_sigma_schedule(
            (1.0, 0.5, 0.0),
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=3,
            require_terminal_zero=True,
        )


def test_required_terminal_zero_fails_when_schedule_ends_nonzero() -> None:
    with pytest.raises(ScheduleContractError, match="terminal zero"):
        validate_sigma_schedule(
            (1.0, 0.5, 0.25),
            domain=SigmaDomain.UNIT_FLOW,
            expected_steps=2,
            require_terminal_zero=True,
        )


def test_nonzero_terminal_is_allowed_only_when_explicit() -> None:
    assert validate_sigma_schedule(
        (4.0, 2.0, 1.0),
        domain=SigmaDomain.CONTINUOUS_EDM,
        expected_steps=2,
        require_terminal_zero=False,
    ) == (4.0, 2.0, 1.0)
