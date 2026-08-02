"""Phase 0 RED contracts for the explicit LTX family slice."""

from __future__ import annotations

from itertools import pairwise

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.ltx import (
    LTX2_19B_DISTILLED_STAGE1_PROFILE,
    LTX2_19B_DISTILLED_STAGE2_PROFILE,
    LTX2_19B_PROFILE,
    LTX23_22B_DISTILLED_STAGE1_PROFILE,
    LTX23_22B_DISTILLED_STAGE2_PROFILE,
    LTX23_22B_PROFILE,
    LTXV_098_PROFILE,
    LTXGeneration,
    LTXProfileId,
    build_ltx_schedule,
    derive_ltx_shift,
)


def test_phase0_pins_three_explicit_generations_and_source_lanes() -> None:
    profiles = (
        LTXV_098_PROFILE,
        LTX2_19B_PROFILE,
        LTX23_22B_PROFILE,
        LTX2_19B_DISTILLED_STAGE1_PROFILE,
        LTX2_19B_DISTILLED_STAGE2_PROFILE,
        LTX23_22B_DISTILLED_STAGE1_PROFILE,
        LTX23_22B_DISTILLED_STAGE2_PROFILE,
    )
    assert {profile.generation for profile in profiles} == {
        LTXGeneration.LTXV_098,
        LTXGeneration.LTX2_19B,
        LTXGeneration.LTX23_22B,
    }
    for profile in profiles:
        lanes = {reference.lane for reference in profile.references}
        assert lanes == {"comfyui_implementation", "diffusers_framework", "official_publisher"}
        assert profile.schema.model_family == "ltx"
        assert profile.schema.sigma_domain is SigmaDomain.UNIT_FLOW


def test_adaptive_contract_uses_explicit_token_shift_and_terminal_stretch() -> None:
    assert derive_ltx_shift(1024) == pytest.approx(0.95)
    assert derive_ltx_shift(4096) == pytest.approx(2.05)
    assert derive_ltx_shift(2560) == pytest.approx(1.5)
    result = build_ltx_schedule(
        profile=LTXProfileId.LTXV_098_DEV,
        steps=8,
        token_count=2560,
        stretch=True,
        terminal=0.1,
    )
    assert len(result.sigmas) == 9
    assert result.sigmas[-1] == 0.0
    assert result.sigmas[-2] == pytest.approx(0.1, abs=1e-15)
    assert all(current > following for current, following in pairwise(result.sigmas))
    assert result.request.provenance.evidence.value == "modified"


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        (
            LTXProfileId.LTX2_19B_DISTILLED_STAGE1,
            (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0),
        ),
        (LTXProfileId.LTX2_19B_DISTILLED_STAGE2, (0.909375, 0.725, 0.421875, 0.0)),
        (
            LTXProfileId.LTX23_22B_DISTILLED_STAGE1,
            (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0),
        ),
        (LTXProfileId.LTX23_22B_DISTILLED_STAGE2, (0.909375, 0.725, 0.421875, 0.0)),
    ),
)
def test_distilled_vectors_are_immutable_presets(
    profile: LTXProfileId, expected: tuple[float, ...]
) -> None:
    result = build_ltx_schedule(profile=profile, steps=len(expected) - 1)
    assert result.sigmas == expected
    with pytest.raises(ScheduleContractError, match="distilled"):
        build_ltx_schedule(profile=profile, steps=8, token_count=4096)


def test_invalid_tokens_and_terminal_fail_closed() -> None:
    with pytest.raises(ScheduleContractError):
        derive_ltx_shift(0)
    with pytest.raises(ScheduleContractError):
        build_ltx_schedule(
            profile=LTXProfileId.LTXV_098_DEV, steps=8, token_count=4096, terminal=1.0
        )


def test_strict_official_locks_the_published_recipe_step_count() -> None:
    with pytest.raises(ScheduleContractError, match="official LTX recipe"):
        build_ltx_schedule(
            profile=LTXProfileId.LTX2_19B_DEV,
            steps=20,
            token_count=4096,
            strict_official=True,
        )
    modified = build_ltx_schedule(
        profile=LTXProfileId.LTX2_19B_DEV,
        steps=20,
        token_count=4096,
        strict_official=False,
    )
    assert modified.request.provenance.evidence.value == "modified"
