"""Exact tests for dependency-free base-grid builders."""

from __future__ import annotations

import pytest
from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    krea_reciprocal_step_grid,
    linear_endpoint_grid,
)


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        (1, (1.0,)),
        (2, (1.0, 0.5)),
        (4, (1.0, 0.75, 0.5, 0.25)),
        (8, (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125)),
    ],
)
def test_krea_reciprocal_step_grid_is_exact(
    steps: int,
    expected: tuple[float, ...],
) -> None:
    assert krea_reciprocal_step_grid(steps) == expected


def test_krea_grid_plus_terminal_zero_matches_official_unshifted_vector() -> None:
    non_terminal = krea_reciprocal_step_grid(8)
    full_vector = (*non_terminal, 0.0)

    assert full_vector == tuple(1.0 - index / 8 for index in range(9))


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_krea_grid_rejects_invalid_step_count(steps: object) -> None:
    with pytest.raises(ScheduleContractError, match="positive integer"):
        krea_reciprocal_step_grid(steps)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "domain",
    [
        SigmaDomain.MODEL_NATIVE,
        SigmaDomain.CONTINUOUS_EDM,
        SigmaDomain.DISCRETE_TRAINING_INDEX,
    ],
)
def test_krea_grid_accepts_only_unit_flow(domain: SigmaDomain) -> None:
    with pytest.raises(ScheduleContractError, match="UNIT_FLOW"):
        krea_reciprocal_step_grid(8, domain=domain)


@pytest.mark.parametrize(
    ("points", "start", "end", "expected"),
    [
        (2, 1.0, 0.0, (1.0, 0.0)),
        (3, 1.0, 0.0, (1.0, 0.5, 0.0)),
        (5, 4.0, 0.0, (4.0, 3.0, 2.0, 1.0, 0.0)),
    ],
)
def test_linear_endpoint_grid_is_exact(
    points: int,
    start: float,
    end: float,
    expected: tuple[float, ...],
) -> None:
    assert (
        linear_endpoint_grid(
            points=points,
            start=start,
            end=end,
            domain=SigmaDomain.CONTINUOUS_EDM,
        )
        == expected
    )


@pytest.mark.parametrize("points", [0, 1, -1, True, 2.5])
def test_linear_grid_requires_at_least_two_integer_points(points: object) -> None:
    with pytest.raises(ScheduleContractError, match="at least two"):
        linear_endpoint_grid(
            points=points,  # type: ignore[arg-type]
            start=1.0,
            end=0.0,
            domain=SigmaDomain.UNIT_FLOW,
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (float("nan"), 0.0),
        (1.0, float("inf")),
        (0.0, 1.0),
        (1.0, 1.0),
        (True, 0.0),
    ],
)
def test_linear_grid_rejects_invalid_endpoints(start: object, end: object) -> None:
    with pytest.raises(ScheduleContractError):
        linear_endpoint_grid(
            points=4,
            start=start,  # type: ignore[arg-type]
            end=end,  # type: ignore[arg-type]
            domain=SigmaDomain.UNIT_FLOW,
        )


def test_linear_grid_rejects_opaque_model_native_domain() -> None:
    with pytest.raises(ScheduleContractError, match="MODEL_NATIVE"):
        linear_endpoint_grid(
            points=4,
            start=1.0,
            end=0.0,
            domain=SigmaDomain.MODEL_NATIVE,
        )
