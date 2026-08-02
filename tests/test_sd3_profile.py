"""RED coverage for the source-qualified original SD3 schedule profiles."""

from __future__ import annotations

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.sd3 import (
    SD3_COMFY_DIFFUSERS_PROFILE,
    SD3_COMFY_DIFFUSERS_SCHEMA,
    SD3_PUBLISHER_REFERENCE_PROFILE,
    SD3_PUBLISHER_REFERENCE_SCHEMA,
    SD3ShiftMode,
    build_sd3_schedule,
)


def test_sd3_profiles_pin_the_publisher_conflict_as_two_explicit_modes() -> None:
    assert SD3_PUBLISHER_REFERENCE_SCHEMA.profile_id == "sd3.publisher-reference.official"
    assert SD3_COMFY_DIFFUSERS_SCHEMA.profile_id == "sd3.comfy-diffusers-fixed.framework-reference"
    assert SD3_PUBLISHER_REFERENCE_PROFILE.shift_mode is SD3ShiftMode.PUBLISHER_REFERENCE
    assert SD3_COMFY_DIFFUSERS_PROFILE.shift_mode is SD3ShiftMode.COMFY_DIFFUSERS_FIXED
    assert SD3_PUBLISHER_REFERENCE_SCHEMA.profile_version == "1"
    assert SD3_COMFY_DIFFUSERS_SCHEMA.profile_version == "1"


@pytest.mark.parametrize(
    ("mode", "shift"),
    [
        (SD3ShiftMode.PUBLISHER_REFERENCE, 1.0),
        (SD3ShiftMode.COMFY_DIFFUSERS_FIXED, 3.0),
    ],
)
def test_sd3_modes_have_explicit_direct_ratio_shift_and_terminal_zero(
    mode: SD3ShiftMode, shift: float
) -> None:
    result = build_sd3_schedule(mode=mode, steps=4, strict_source=False)
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas[-1] == 0.0
    expected = tuple(
        shift * value / (1.0 + (shift - 1.0) * value) for value in (1.0, 0.75, 0.5, 0.25)
    )
    assert result.sigmas[:-1] == pytest.approx(expected)
    assert all(
        left > right for left, right in zip(result.sigmas[:-1], result.sigmas[1:], strict=True)
    )


def test_sd3_mode_is_required_to_resolve_the_publisher_shift_conflict() -> None:
    with pytest.raises(ScheduleContractError, match="mode"):
        build_sd3_schedule(mode="auto", steps=4, strict_source=False)  # type: ignore[arg-type]


def test_sd3_strict_publisher_recipe_is_fifty_steps() -> None:
    with pytest.raises(ScheduleContractError, match="published 50-step"):
        build_sd3_schedule(
            mode=SD3ShiftMode.PUBLISHER_REFERENCE,
            steps=49,
            strict_source=True,
        )


def test_sd3_modes_are_not_composable() -> None:
    with pytest.raises(ScheduleContractError, match=r"already shifted|composition|mode"):
        build_sd3_schedule(
            mode=SD3ShiftMode.COMFY_DIFFUSERS_FIXED,
            steps=4,
            strict_source=False,
            already_shifted=True,
        )
