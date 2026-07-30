"""Closed-form tests for dependency-free unit-flow shift transforms."""

from __future__ import annotations

import math

import pytest
from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    direct_ratio_shift,
    exponential_mu_shift,
    no_shift,
)

KREA_TURBO_BASE = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0)
KREA_TURBO_SHIFTED_FLOAT64 = (
    1.0,
    0.9567237271222464,
    0.9045307667386396,
    0.8403488020923521,
    0.7595109169491111,
    0.6545668033876179,
    0.5128441015091338,
    0.3109010566906254,
    0.0,
)
KREA_TURBO_SHIFTED_FLOAT32 = (
    1.0,
    0.9567237,
    0.90453076,
    0.8403488,
    0.75951093,
    0.6545668,
    0.5128441,
    0.31090105,
    0.0,
)


def test_exponential_mu_matches_krea_turbo_float64_reference() -> None:
    actual = exponential_mu_shift(KREA_TURBO_BASE, mu=1.15)

    assert actual == pytest.approx(KREA_TURBO_SHIFTED_FLOAT64, rel=0.0, abs=1e-15)


def test_exponential_mu_matches_krea_turbo_float32_tolerance() -> None:
    actual = exponential_mu_shift(KREA_TURBO_BASE, mu=1.15)

    assert actual == pytest.approx(KREA_TURBO_SHIFTED_FLOAT32, rel=0.0, abs=1e-6)


def test_exponential_mu_supports_declared_positive_exponent() -> None:
    actual = exponential_mu_shift((1.0, 0.5, 0.25, 0.0), mu=0.0, exponent=2.0)

    assert actual == pytest.approx((1.0, 0.5, 0.1, 0.0), rel=0.0, abs=1e-15)


def test_direct_ratio_matches_closed_form_reference() -> None:
    actual = direct_ratio_shift((1.0, 0.75, 0.5, 0.25, 0.0), ratio=3.0)

    assert actual == pytest.approx((1.0, 0.9, 0.75, 0.5, 0.0), rel=0.0, abs=1e-15)


@pytest.mark.parametrize("mu", [-5.0, -1.15, 0.0, 1.15, 5.0])
def test_direct_ratio_exp_mu_is_equivalent_to_exponential_mu(mu: float) -> None:
    values = (1.0, 0.875, 0.5, 0.125, 0.0)

    exponential = exponential_mu_shift(values, mu=mu)
    direct = direct_ratio_shift(values, ratio=math.exp(mu))

    assert direct == pytest.approx(exponential, rel=0.0, abs=1e-15)


def test_identity_controls_and_explicit_no_shift_preserve_tuple() -> None:
    values = (1.0, 0.5, 0.0)

    assert exponential_mu_shift(values, mu=0.0) == values
    assert direct_ratio_shift(values, ratio=1.0) == values
    assert no_shift(values) is values


def test_extreme_finite_mu_values_do_not_overflow() -> None:
    values = (1.0, 0.5, 0.0)

    assert exponential_mu_shift(values, mu=1000.0) == (1.0, 1.0, 0.0)
    assert exponential_mu_shift(values, mu=-1000.0) == (1.0, 0.0, 0.0)


def test_extreme_finite_direct_ratios_do_not_overflow() -> None:
    values = (1.0, 0.5, 0.0)

    high = direct_ratio_shift(values, ratio=float.fromhex("0x1.fffffffffffffp+1023"))
    low = direct_ratio_shift(values, ratio=float.fromhex("0x0.0000000000001p-1022"))

    assert high == (1.0, 1.0, 0.0)
    assert low == (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "values",
    [
        (),
        [1.0, 0.0],
        (True, 0.0),
        ("1", 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (1.1, 0.0),
        (1.0, -0.1),
    ],
)
@pytest.mark.parametrize("transform", ["exponential", "direct", "none"])
def test_shift_transforms_reject_invalid_unit_flow_values(
    values: object,
    transform: str,
) -> None:
    with pytest.raises(ScheduleContractError):
        if transform == "exponential":
            exponential_mu_shift(values, mu=1.15)  # type: ignore[arg-type]
        elif transform == "direct":
            direct_ratio_shift(values, ratio=3.0)  # type: ignore[arg-type]
        else:
            no_shift(values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "domain",
    [
        SigmaDomain.MODEL_NATIVE,
        SigmaDomain.CONTINUOUS_EDM,
        SigmaDomain.DISCRETE_TRAINING_INDEX,
        "unit_flow",
    ],
)
@pytest.mark.parametrize("transform", ["exponential", "direct", "none"])
def test_shift_transforms_reject_non_unit_flow_domains(
    domain: object,
    transform: str,
) -> None:
    with pytest.raises(ScheduleContractError, match="UNIT_FLOW"):
        if transform == "exponential":
            exponential_mu_shift(
                (1.0, 0.0),
                mu=1.15,
                domain=domain,  # type: ignore[arg-type]
            )
        elif transform == "direct":
            direct_ratio_shift(
                (1.0, 0.0),
                ratio=3.0,
                domain=domain,  # type: ignore[arg-type]
            )
        else:
            no_shift((1.0, 0.0), domain=domain)  # type: ignore[arg-type]


@pytest.mark.parametrize("mu", [True, "1.15", float("nan"), float("inf")])
def test_exponential_mu_rejects_invalid_mu(mu: object) -> None:
    with pytest.raises(ScheduleContractError, match="mu"):
        exponential_mu_shift((1.0, 0.0), mu=mu)  # type: ignore[arg-type]


@pytest.mark.parametrize("exponent", [True, "1", 0.0, -1.0, float("nan"), float("inf")])
def test_exponential_mu_rejects_invalid_exponent(exponent: object) -> None:
    with pytest.raises(ScheduleContractError, match="exponent"):
        exponential_mu_shift(
            (1.0, 0.0),
            mu=1.15,
            exponent=exponent,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "ratio",
    [True, "3", 0.0, -1.0, float("nan"), float("inf")],
)
def test_direct_ratio_rejects_invalid_ratio(ratio: object) -> None:
    with pytest.raises(ScheduleContractError, match="ratio"):
        direct_ratio_shift((1.0, 0.0), ratio=ratio)  # type: ignore[arg-type]
