"""Four-source and numerical contracts for FLUX.1-schnell."""

from __future__ import annotations

from itertools import pairwise

import pytest
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError, SigmaDomain, TerminalPolicy
from comfyui_sigmax.profiles.flux1_schnell import (
    FLUX1_SCHNELL_PROFILE,
    FLUX1_SCHNELL_SCHEMA,
    build_flux1_schnell_schedule,
)


def test_profile_pins_all_four_required_source_lanes() -> None:
    assert {
        reference.lane: (reference.url, reference.revision)
        for reference in FLUX1_SCHNELL_PROFILE.references
    } == {
        "comfyui_implementation": (
            "https://github.com/Comfy-Org/ComfyUI",
            "2881e6161081439b1c3fb3b6c1f51b3d272da710",  # pragma: allowlist secret
        ),
        "official_github": (
            "https://github.com/black-forest-labs/flux",
            "802fb4713906133fcbd0d8dc5351620ca4773036",  # pragma: allowlist secret
        ),
        "official_huggingface": (
            "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
            "741f7c3ce8b383c54771c7003378a50191e9efe9",  # pragma: allowlist secret
        ),
        "official_technical_document": (
            "https://bfl.ai/blog/24-08-01-bfl",
            "2024-08-01",
        ),
    }


def test_schema_is_explicit_unshifted_schnell_only() -> None:
    schema = FLUX1_SCHNELL_SCHEMA

    assert schema.profile_id == "flux1.schnell.official"
    assert schema.model_family == "flux1"
    assert schema.model_variant == "schnell"
    assert schema.evidence is EvidenceLevel.OFFICIAL
    assert schema.sigma_domain is SigmaDomain.UNIT_FLOW
    assert schema.base_grid is not None
    assert schema.base_grid.identifier == "flowmatch.reciprocal_step"
    assert schema.base_grid.terminal_included is False
    assert tuple(transform.identifier for transform in schema.transforms) == (
        "terminal.append_zero",
    )
    assert schema.terminal.policy is TerminalPolicy.APPEND_ZERO
    assert schema.detection.resolving_sources == ("explicit_variant",)
    projection = repr(schema).casefold()
    assert "krea" not in projection
    assert "z_image" not in projection
    assert "dynamic_shifting', value=false" in projection


@pytest.mark.parametrize("steps", (1, 2, 3, 4))
def test_schedule_is_exact_endpoint_inclusive_unshifted_grid(steps: int) -> None:
    result = build_flux1_schnell_schedule(steps=steps)
    expected = tuple((steps - index) / steps for index in range(steps + 1))

    assert result.sigmas == expected
    assert len(result.sigmas) == steps + 1
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(current > following for current, following in pairwise(result.sigmas))
    assert result.request.provenance.evidence is EvidenceLevel.OFFICIAL


def test_nonpublisher_step_count_is_modified_or_strictly_rejected() -> None:
    modified = build_flux1_schnell_schedule(steps=5)

    assert modified.request.provenance.evidence is EvidenceLevel.MODIFIED
    assert modified.warnings
    with pytest.raises(ScheduleContractError):
        build_flux1_schnell_schedule(steps=5, strict_official=True)
