"""RED contract coverage for the source-qualified Lumina-Image 2.0 schedule."""

from __future__ import annotations

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.lumina2 import (
    LUMINA2_PROFILE,
    LUMINA2_SCHEMA,
    Lumina2ShiftMode,
    build_lumina2_schedule,
)


def test_lumina2_profile_is_explicit_v20_fixed_shift() -> None:
    assert LUMINA2_SCHEMA.profile_id == "lumina2.v2.official"
    assert LUMINA2_SCHEMA.profile_version == "1"
    assert LUMINA2_PROFILE.shift_mode is Lumina2ShiftMode.OFFICIAL_FIXED
    assert LUMINA2_SCHEMA.model_family == "lumina2"
    assert LUMINA2_SCHEMA.model_variant == "2.0"


def test_lumina2_uses_one_ratio_shift_unit_flow_and_terminal_zero() -> None:
    result = build_lumina2_schedule(
        mode=Lumina2ShiftMode.OFFICIAL_FIXED,
        steps=4,
        strict_source=False,
    )
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas[-1] == 0.0
    expected = tuple(6.0 * value / (1.0 + (6.0 - 1.0) * value) for value in (1.0, 0.75, 0.5, 0.25))
    assert result.sigmas[:-1] == pytest.approx(expected, abs=1e-15)
    assert all(
        left > right for left, right in zip(result.sigmas[:-1], result.sigmas[1:], strict=True)
    )


def test_lumina2_requires_explicit_mode_and_rejects_composition() -> None:
    with pytest.raises(ScheduleContractError, match="mode"):
        build_lumina2_schedule(mode="auto", steps=4, strict_source=False)  # type: ignore[arg-type]
    with pytest.raises(ScheduleContractError, match=r"already shifted|composition"):
        build_lumina2_schedule(
            mode=Lumina2ShiftMode.OFFICIAL_FIXED,
            steps=4,
            strict_source=False,
            already_shifted=True,
        )


def test_lumina2_strict_source_requires_the_fifty_step_reference_recipe() -> None:
    with pytest.raises(ScheduleContractError, match="50-step"):
        build_lumina2_schedule(
            mode=Lumina2ShiftMode.OFFICIAL_FIXED,
            steps=49,
            strict_source=True,
        )


def test_lumina2_non_reference_steps_are_modified_evidence() -> None:
    result = build_lumina2_schedule(
        mode=Lumina2ShiftMode.OFFICIAL_FIXED,
        steps=28,
        strict_source=False,
    )
    assert result.request.provenance.evidence.value == "modified"
    assert result.warnings
