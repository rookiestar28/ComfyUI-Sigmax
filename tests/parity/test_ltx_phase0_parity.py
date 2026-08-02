"""Phase 0 numerical parity contracts for the LTX schedule slice."""

from __future__ import annotations

import pytest
from comfyui_sigmax.profiles.ltx import LTXProfileId, build_ltx_schedule


@pytest.mark.parametrize(
    ("profile", "steps", "tokens"),
    (
        (LTXProfileId.LTXV_098_DEV, 20, 4096),
        (LTXProfileId.LTX2_19B_DEV, 40, 4096),
        (LTXProfileId.LTX23_22B_DEV, 30, 4096),
    ),
)
def test_adaptive_vectors_are_deterministic(profile: LTXProfileId, steps: int, tokens: int) -> None:
    first = build_ltx_schedule(profile=profile, steps=steps, token_count=tokens)
    second = build_ltx_schedule(profile=profile, steps=steps, token_count=tokens)
    assert first.sigmas == second.sigmas
    assert first.sigmas[-1] == 0.0


@pytest.mark.parametrize(
    "profile", (LTXProfileId.LTX2_19B_DISTILLED_STAGE1, LTXProfileId.LTX23_22B_DISTILLED_STAGE2)
)
def test_distilled_vectors_do_not_depend_on_token_count(profile: LTXProfileId) -> None:
    steps = 8 if "stage1" in profile.value else 3
    first = build_ltx_schedule(profile=profile, steps=steps)
    second = build_ltx_schedule(profile=profile, steps=steps)
    assert first.sigmas == second.sigmas
