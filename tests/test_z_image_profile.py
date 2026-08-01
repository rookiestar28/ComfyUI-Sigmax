"""Evidence and numerical contracts for the Z-Image Base/Turbo profiles."""

from __future__ import annotations

from itertools import pairwise

import pytest
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles.z_image import (
    Z_IMAGE_BASE_PROFILE,
    Z_IMAGE_BASE_SCHEMA,
    Z_IMAGE_TURBO_PROFILE,
    Z_IMAGE_TURBO_SCHEMA,
    ZImageVariant,
    build_z_image_schedule,
)


def test_each_variant_pins_all_four_required_source_lanes() -> None:
    expected = {
        "official_github": (
            "https://github.com/Tongyi-MAI/Z-Image",
            "26f23eda626ffadda020b04ff79488e1d72004cd",  # pragma: allowlist secret
        ),
        "official_huggingface": (
            "https://huggingface.co/Tongyi-MAI/Z-Image",
            "04cc4abb7c5069926f75c9bfde9ef43d49423021",  # pragma: allowlist secret
        ),
        "official_technical_document": (
            "https://tongyi-mai.github.io/Z-Image-blog/",
            "e67bafb673fa19d301f903ac62de26c48b4cc1c4",  # pragma: allowlist secret
        ),
        "comfyui_implementation": (
            "https://github.com/Comfy-Org/ComfyUI",
            "235b466a0cb26d47c24f2ab66d1a8c5e70b21070",  # pragma: allowlist secret
        ),
    }

    for profile in (Z_IMAGE_BASE_PROFILE, Z_IMAGE_TURBO_PROFILE):
        lanes = {
            reference.lane: (reference.url, reference.revision) for reference in profile.references
        }
        assert lanes == expected | {
            "official_huggingface": (
                profile.huggingface_url,
                profile.huggingface_revision,
            )
        }


def test_profile_schema_declares_exact_released_schedule_contracts() -> None:
    assert Z_IMAGE_BASE_SCHEMA.profile_id == "z_image.base.official"
    assert Z_IMAGE_TURBO_SCHEMA.profile_id == "z_image.turbo.official"
    assert Z_IMAGE_BASE_SCHEMA.evidence is EvidenceLevel.OFFICIAL
    assert Z_IMAGE_TURBO_SCHEMA.evidence is EvidenceLevel.OFFICIAL
    assert Z_IMAGE_BASE_PROFILE.fixed_shift_ratio == 6.0
    assert Z_IMAGE_TURBO_PROFILE.fixed_shift_ratio == 3.0
    assert Z_IMAGE_BASE_PROFILE.dynamic_shifting is False
    assert Z_IMAGE_TURBO_PROFILE.dynamic_shifting is False
    assert Z_IMAGE_BASE_SCHEMA.detection.resolving_sources == ("explicit_variant",)
    assert Z_IMAGE_TURBO_SCHEMA.detection.resolving_sources == ("explicit_variant",)

    for schema in (Z_IMAGE_BASE_SCHEMA, Z_IMAGE_TURBO_SCHEMA):
        assert schema.model_family == "z_image"
        assert schema.sigma_domain is SigmaDomain.UNIT_FLOW
        assert schema.base_grid is not None
        assert schema.base_grid.identifier == "flowmatch.reciprocal_step"
        assert tuple(transform.identifier for transform in schema.transforms) == (
            "flowmatch.direct_ratio",
            "terminal.append_zero",
        )
        projection = repr(schema).casefold()
        assert "krea" not in projection
        assert "1.15" not in projection


@pytest.mark.parametrize(
    ("variant", "steps", "ratio", "evidence"),
    (
        (ZImageVariant.BASE, 28, 6.0, EvidenceLevel.OFFICIAL),
        (ZImageVariant.BASE, 50, 6.0, EvidenceLevel.OFFICIAL),
        (ZImageVariant.TURBO, 8, 3.0, EvidenceLevel.OFFICIAL),
        (ZImageVariant.TURBO, 12, 3.0, EvidenceLevel.MODIFIED),
    ),
)
def test_schedule_is_exact_fixed_ratio_flowmatch(
    variant: ZImageVariant,
    steps: int,
    ratio: float,
    evidence: EvidenceLevel,
) -> None:
    result = build_z_image_schedule(variant=variant, steps=steps)

    expected = (
        *(
            1.0 if index == 0 else (ratio * (steps - index)) / (ratio * (steps - index) + index)
            for index in range(steps)
        ),
        0.0,
    )
    assert result.sigmas == pytest.approx(expected, abs=1e-15, rel=1e-15)
    assert len(result.sigmas) == steps + 1
    assert all(current > following for current, following in pairwise(result.sigmas))
    assert result.request.provenance.evidence is evidence


@pytest.mark.parametrize(
    ("variant", "steps"),
    ((ZImageVariant.BASE, 27), (ZImageVariant.BASE, 51), (ZImageVariant.TURBO, 7)),
)
def test_strict_official_recipes_fail_closed(variant: ZImageVariant, steps: int) -> None:
    with pytest.raises(ScheduleContractError):
        build_z_image_schedule(variant=variant, steps=steps, strict_official=True)
