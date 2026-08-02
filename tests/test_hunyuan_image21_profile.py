"""RED contract coverage for HunyuanImage 2.1 Base/Distilled schedules."""

from __future__ import annotations

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.hunyuan_image21 import (
    HUNYUAN_IMAGE21_BASE_PROFILE,
    HUNYUAN_IMAGE21_BASE_SCHEMA,
    HUNYUAN_IMAGE21_DISTILLED_PROFILE,
    HUNYUAN_IMAGE21_DISTILLED_SCHEMA,
    HunyuanImage21Variant,
    build_hunyuan_image21_schedule,
)


def test_hunyuan_image21_profiles_are_explicit_and_distinct() -> None:
    assert HUNYUAN_IMAGE21_BASE_SCHEMA.profile_id == "hunyuan-image-2-1.base.official"
    assert HUNYUAN_IMAGE21_DISTILLED_SCHEMA.profile_id == "hunyuan-image-2-1.distilled.official"
    assert HUNYUAN_IMAGE21_BASE_SCHEMA.profile_version == "1"
    assert HUNYUAN_IMAGE21_DISTILLED_SCHEMA.profile_version == "1"
    assert HUNYUAN_IMAGE21_BASE_SCHEMA.model_family == "hunyuanimage"
    assert HUNYUAN_IMAGE21_DISTILLED_SCHEMA.model_variant == "2.1-distilled"
    assert HUNYUAN_IMAGE21_BASE_PROFILE.variant is HunyuanImage21Variant.BASE
    assert HUNYUAN_IMAGE21_DISTILLED_PROFILE.variant is HunyuanImage21Variant.DISTILLED


@pytest.mark.parametrize(
    ("variant", "steps", "ratio"),
    [
        (HunyuanImage21Variant.BASE, 4, 5.0),
        (HunyuanImage21Variant.DISTILLED, 4, 4.0),
    ],
)
def test_hunyuan_image21_uses_one_ratio_shift_unit_flow_and_zero_terminal(
    variant: HunyuanImage21Variant,
    steps: int,
    ratio: float,
) -> None:
    result = build_hunyuan_image21_schedule(variant=variant, steps=steps)
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas[-1] == 0.0
    expected = tuple(
        ratio * value / (1.0 + (ratio - 1.0) * value)
        for value in ((steps - index) / steps for index in range(steps))
    )
    assert result.sigmas[:-1] == pytest.approx(expected, abs=1e-15)
    assert all(
        left > right for left, right in zip(result.sigmas[:-1], result.sigmas[1:], strict=True)
    )


def test_hunyuan_image21_requires_explicit_variant_and_rejects_composition() -> None:
    with pytest.raises(ScheduleContractError, match="variant"):
        build_hunyuan_image21_schedule(variant="auto", steps=4)  # type: ignore[arg-type]
    with pytest.raises(ScheduleContractError, match=r"already shifted|composition"):
        build_hunyuan_image21_schedule(
            variant=HunyuanImage21Variant.BASE,
            steps=4,
            already_shifted=True,
        )


@pytest.mark.parametrize(
    ("variant", "reference_steps", "rejected_steps"),
    [
        (HunyuanImage21Variant.BASE, 50, 49),
        (HunyuanImage21Variant.DISTILLED, 8, 7),
    ],
)
def test_hunyuan_image21_strict_source_uses_variant_recipe(
    variant: HunyuanImage21Variant,
    reference_steps: int,
    rejected_steps: int,
) -> None:
    result = build_hunyuan_image21_schedule(
        variant=variant,
        steps=reference_steps,
        strict_source=True,
    )
    assert result.request.provenance.evidence.value == "official"
    with pytest.raises(ScheduleContractError, match="step"):
        build_hunyuan_image21_schedule(
            variant=variant,
            steps=rejected_steps,
            strict_source=True,
        )


def test_hunyuan_image21_non_reference_steps_are_modified() -> None:
    result = build_hunyuan_image21_schedule(
        variant=HunyuanImage21Variant.DISTILLED,
        steps=12,
        strict_source=False,
    )
    assert result.request.provenance.evidence.value == "modified"
    assert result.warnings
