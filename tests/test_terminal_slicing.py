"""Boundary tests for terminal and terminal-inclusive slicing policies."""

from __future__ import annotations

import pytest
from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    TerminalPolicy,
    apply_terminal_policy,
    denoise_construction_steps,
    direct_ratio_shift,
    krea_reciprocal_step_grid,
    slice_denoise_tail,
    slice_step_range,
)


@pytest.mark.parametrize(
    "domain",
    [
        SigmaDomain.UNIT_FLOW,
        SigmaDomain.CONTINUOUS_EDM,
        SigmaDomain.DISCRETE_TRAINING_INDEX,
    ],
)
def test_append_zero_is_explicit_for_constructed_domains(domain: SigmaDomain) -> None:
    assert apply_terminal_policy(
        (1.0, 0.5),
        policy=TerminalPolicy.APPEND_ZERO,
        domain=domain,
    ) == (1.0, 0.5, 0.0)


def test_preserve_terminal_returns_original_tuple() -> None:
    values = (1.0, 0.5, 0.0)

    assert (
        apply_terminal_policy(
            values,
            policy=TerminalPolicy.PRESERVE,
            domain=SigmaDomain.UNIT_FLOW,
        )
        is values
    )


@pytest.mark.parametrize("terminal", [0.0, -0.0])
def test_append_zero_rejects_duplicate_terminal(terminal: float) -> None:
    with pytest.raises(ScheduleContractError, match="already"):
        apply_terminal_policy(
            (1.0, terminal),
            policy=TerminalPolicy.APPEND_ZERO,
            domain=SigmaDomain.UNIT_FLOW,
        )


@pytest.mark.parametrize(
    "values",
    [
        (),
        [1.0, 0.5],
        (True, 0.5),
        ("1", 0.5),
    ],
)
def test_terminal_policy_rejects_invalid_structural_values(values: object) -> None:
    with pytest.raises(ScheduleContractError):
        apply_terminal_policy(
            values,  # type: ignore[arg-type]
            policy=TerminalPolicy.APPEND_ZERO,
            domain=SigmaDomain.UNIT_FLOW,
        )


@pytest.mark.parametrize("policy", ["APPEND_ZERO", None])
def test_terminal_policy_rejects_invalid_policy(policy: object) -> None:
    with pytest.raises(ScheduleContractError, match="TerminalPolicy"):
        apply_terminal_policy(
            (1.0, 0.5),
            policy=policy,  # type: ignore[arg-type]
            domain=SigmaDomain.UNIT_FLOW,
        )


@pytest.mark.parametrize("domain", [SigmaDomain.MODEL_NATIVE, "UNIT_FLOW"])
def test_terminal_policy_rejects_opaque_or_invalid_domain(domain: object) -> None:
    with pytest.raises(ScheduleContractError, match="MODEL_NATIVE"):
        apply_terminal_policy(
            (1.0, 0.5),
            policy=TerminalPolicy.APPEND_ZERO,
            domain=domain,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("start_step", "end_step", "expected"),
    [
        (0, None, (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)),
        (0, 2, (1.0, 0.8, 0.6)),
        (1, 4, (0.8, 0.6, 0.4, 0.2)),
        (3, 5, (0.4, 0.2, 0.0)),
        (4, 5, (0.2, 0.0)),
    ],
)
def test_step_range_uses_terminal_inclusive_boundaries(
    start_step: int,
    end_step: int | None,
    expected: tuple[float, ...],
) -> None:
    values = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)

    actual = slice_step_range(values, start_step=start_step, end_step=end_step)

    assert actual == expected
    assert len(actual) - 1 == (5 if end_step is None else end_step) - start_step


@pytest.mark.parametrize(
    "values",
    [
        (),
        (1.0,),
        [1.0, 0.0],
        (True, 0.0),
        ("1", 0.0),
    ],
)
def test_step_range_requires_terminal_inclusive_numeric_tuple(values: object) -> None:
    with pytest.raises(ScheduleContractError):
        slice_step_range(values)  # type: ignore[arg-type]


@pytest.mark.parametrize("start_step", [-1, True, 1.5, 5, 6])
def test_step_range_rejects_invalid_start(start_step: object) -> None:
    with pytest.raises(ScheduleContractError, match="start_step"):
        slice_step_range(
            (1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
            start_step=start_step,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("end_step", [-1, 0, True, 1.5, 6])
def test_step_range_rejects_invalid_or_empty_end(end_step: object) -> None:
    with pytest.raises(ScheduleContractError, match="end_step"):
        slice_step_range(
            (1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
            start_step=0,
            end_step=end_step,  # type: ignore[arg-type]
        )


def test_step_range_rejects_reversed_or_empty_middle_range() -> None:
    values = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)

    with pytest.raises(ScheduleContractError, match="greater than start_step"):
        slice_step_range(values, start_step=3, end_step=3)
    with pytest.raises(ScheduleContractError, match="greater than start_step"):
        slice_step_range(values, start_step=3, end_step=2)


@pytest.mark.parametrize(
    ("requested_steps", "denoise", "expected"),
    [
        (8, 1.0, 8),
        (8, 0.9999, 8),
        (8, 0.8, 10),
        (8, 0.5, 16),
        (1, 0.1, 10),
        (8, 0.0, 0),
    ],
)
def test_denoise_construction_steps_matches_comfyui(
    requested_steps: int,
    denoise: float,
    expected: int,
) -> None:
    assert denoise_construction_steps(requested_steps, denoise) == expected


@pytest.mark.parametrize("requested_steps", [0, -1, True, 1.5])
def test_denoise_construction_rejects_invalid_requested_steps(
    requested_steps: object,
) -> None:
    with pytest.raises(ScheduleContractError, match="positive integer"):
        denoise_construction_steps(
            requested_steps,  # type: ignore[arg-type]
            1.0,
        )


@pytest.mark.parametrize(
    "denoise",
    [-0.1, 1.1, True, "1", float("nan"), float("inf")],
)
def test_denoise_construction_rejects_invalid_denoise(denoise: object) -> None:
    with pytest.raises(ScheduleContractError, match="denoise"):
        denoise_construction_steps(8, denoise)  # type: ignore[arg-type]


def test_partial_denoise_keeps_requested_terminal_inclusive_tail() -> None:
    construction = tuple(float(value) for value in range(8, -1, -1))

    assert slice_denoise_tail(
        construction,
        requested_steps=4,
        denoise=0.5,
    ) == (4.0, 3.0, 2.0, 1.0, 0.0)


def test_full_denoise_preserves_exact_requested_vector() -> None:
    values = (1.0, 0.75, 0.5, 0.25, 0.0)

    assert slice_denoise_tail(values, requested_steps=4, denoise=1.0) is values


def test_zero_denoise_requires_and_returns_explicit_empty_vector() -> None:
    assert slice_denoise_tail((), requested_steps=4, denoise=0.0) == ()

    with pytest.raises(ScheduleContractError, match="empty"):
        slice_denoise_tail((1.0, 0.0), requested_steps=4, denoise=0.0)


@pytest.mark.parametrize(
    ("values", "requested_steps", "denoise"),
    [
        ((4.0, 3.0, 2.0, 1.0, 0.0), 4, 0.5),
        ((1.0, 0.5, 0.0), 4, 1.0),
        ([1.0, 0.0], 1, 1.0),
        ((True, 0.0), 1, 1.0),
    ],
)
def test_denoise_tail_rejects_wrong_length_or_structure(
    values: object,
    requested_steps: int,
    denoise: float,
) -> None:
    with pytest.raises(ScheduleContractError):
        slice_denoise_tail(
            values,  # type: ignore[arg-type]
            requested_steps=requested_steps,
            denoise=denoise,
        )


def test_terminal_denoise_and_manual_range_compose_in_declared_order() -> None:
    base = krea_reciprocal_step_grid(8)
    shifted = direct_ratio_shift(base, ratio=3.0)
    terminal = apply_terminal_policy(
        shifted,
        policy=TerminalPolicy.APPEND_ZERO,
        domain=SigmaDomain.UNIT_FLOW,
    )
    denoised = slice_denoise_tail(terminal, requested_steps=4, denoise=0.5)

    assert slice_step_range(denoised, start_step=1, end_step=4) == (
        shifted[5],
        shifted[6],
        shifted[7],
        0.0,
    )
