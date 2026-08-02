"""RED contract coverage for the pinned original AuraFlow v0.2 schedule."""

from __future__ import annotations

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.aura_flow import (
    AURAFLOW_V02_PROFILE,
    AURAFLOW_V02_SCHEMA,
    AuraFlowShiftMode,
    build_aura_flow_schedule,
)


def test_auraflow_profile_is_explicit_original_v02_fixed_shift() -> None:
    assert AURAFLOW_V02_SCHEMA.profile_id == "auraflow.v0-2.official"
    assert AURAFLOW_V02_SCHEMA.profile_version == "1"
    assert AURAFLOW_V02_PROFILE.shift_mode is AuraFlowShiftMode.OFFICIAL_FIXED
    assert AURAFLOW_V02_SCHEMA.model_family == "auraflow"
    assert AURAFLOW_V02_SCHEMA.model_variant == "v0.2"


def test_auraflow_uses_one_ratio_shift_and_unit_flow_terminal_zero() -> None:
    result = build_aura_flow_schedule(
        mode=AuraFlowShiftMode.OFFICIAL_FIXED,
        steps=4,
        strict_source=False,
    )
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas[-1] == 0.0
    expected = tuple(
        1.73 * value / (1.0 + (1.73 - 1.0) * value) for value in (1.0, 0.75, 0.5, 0.25)
    )
    assert result.sigmas[:-1] == pytest.approx(expected, abs=1e-15)
    assert all(
        left > right for left, right in zip(result.sigmas[:-1], result.sigmas[1:], strict=True)
    )


def test_auraflow_requires_explicit_mode_and_rejects_composition() -> None:
    with pytest.raises(ScheduleContractError, match="mode"):
        build_aura_flow_schedule(mode="auto", steps=4, strict_source=False)  # type: ignore[arg-type]
    with pytest.raises(ScheduleContractError, match=r"already shifted|composition"):
        build_aura_flow_schedule(
            mode=AuraFlowShiftMode.OFFICIAL_FIXED,
            steps=4,
            strict_source=False,
            already_shifted=True,
        )


def test_auraflow_strict_source_requires_the_fifty_step_reference_recipe() -> None:
    with pytest.raises(ScheduleContractError, match="50-step"):
        build_aura_flow_schedule(
            mode=AuraFlowShiftMode.OFFICIAL_FIXED,
            steps=49,
            strict_source=True,
        )


def test_auraflow_non_reference_steps_are_modified_evidence() -> None:
    result = build_aura_flow_schedule(
        mode=AuraFlowShiftMode.OFFICIAL_FIXED,
        steps=28,
        strict_source=False,
    )
    assert result.request.provenance.evidence.value == "modified"
    assert result.warnings
