"""RED coverage for the source-qualified Qwen Image schedule profiles."""

from __future__ import annotations

import math

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.qwen_image import (
    QWEN_IMAGE_COMFY_FIXED_PROFILE,
    QWEN_IMAGE_COMFY_FIXED_SCHEMA,
    QWEN_IMAGE_DIFFUSERS_DYNAMIC_PROFILE,
    QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA,
    QwenImageShiftMode,
    build_qwen_image_schedule,
    calculate_qwen_image_mu,
)


def test_qwen_profiles_pin_two_non_composable_shift_modes() -> None:
    assert QWEN_IMAGE_COMFY_FIXED_SCHEMA.profile_id == "qwen_image.comfy-fixed.official"
    assert QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA.profile_id == (
        "qwen_image.diffusers-dynamic.framework-reference"
    )
    assert QWEN_IMAGE_COMFY_FIXED_PROFILE.shift_mode is QwenImageShiftMode.COMFY_FIXED
    assert QWEN_IMAGE_DIFFUSERS_DYNAMIC_PROFILE.shift_mode is QwenImageShiftMode.DIFFUSERS_DYNAMIC
    assert QWEN_IMAGE_COMFY_FIXED_SCHEMA.profile_version == "1"
    assert QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA.profile_version == "1"


def test_qwen_dynamic_mu_is_explicit_linear_extrapolation() -> None:
    assert calculate_qwen_image_mu(256) == pytest.approx(0.5)
    assert calculate_qwen_image_mu(4096) == pytest.approx(1.15)
    assert calculate_qwen_image_mu(6032) == pytest.approx(1.4777083333333332)


def test_qwen_fixed_schedule_matches_diffusers_fixed_shift_and_terminal_zero() -> None:
    result = build_qwen_image_schedule(
        mode=QwenImageShiftMode.COMFY_FIXED,
        steps=4,
        image_seq_len=None,
        strict_official=False,
    )
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas[-1] == 0.0
    expected = tuple(1.15 * value / (1.0 + 0.15 * value) for value in (1.0, 0.75, 0.5, 0.25))
    assert result.sigmas[:-1] == pytest.approx(expected)
    assert all(
        left > right for left, right in zip(result.sigmas[:-1], result.sigmas[1:], strict=True)
    )


def test_qwen_dynamic_schedule_requires_sequence_length_and_uses_mu() -> None:
    with pytest.raises(ScheduleContractError, match="image_seq_len"):
        build_qwen_image_schedule(
            mode=QwenImageShiftMode.DIFFUSERS_DYNAMIC,
            steps=50,
            image_seq_len=None,
            strict_official=True,
        )

    result = build_qwen_image_schedule(
        mode=QwenImageShiftMode.DIFFUSERS_DYNAMIC,
        steps=4,
        image_seq_len=256,
        strict_official=False,
    )
    assert result.sigmas[0] == pytest.approx(1.0)
    assert result.sigmas[-1] == 0.0
    assert result.sigmas[1] == pytest.approx(math.exp(0.5) / (math.exp(0.5) + (1.0 / 0.75 - 1.0)))


@pytest.mark.parametrize("mode", list(QwenImageShiftMode))
def test_qwen_strict_official_baseline_is_fifty_steps(mode: QwenImageShiftMode) -> None:
    image_seq_len = 1024 if mode is QwenImageShiftMode.DIFFUSERS_DYNAMIC else None
    with pytest.raises(ScheduleContractError, match="published 50-step"):
        build_qwen_image_schedule(
            mode=mode,
            steps=49,
            image_seq_len=image_seq_len,
            strict_official=True,
        )
