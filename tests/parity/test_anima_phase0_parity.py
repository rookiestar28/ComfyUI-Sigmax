"""Phase 0 RED clean-room parity checks for the pinned Anima Base equation."""

from __future__ import annotations

from itertools import pairwise

import pytest

pytestmark = pytest.mark.parity


def _expected(steps: int) -> tuple[float, ...]:
    values = tuple((steps - index) / steps for index in range(steps))
    shifted = tuple(3.0 * value / (1.0 + 2.0 * value) for value in values)
    return (*shifted, 0.0)


def test_anima_base_matches_pinned_fixed_shift_equation() -> None:
    import importlib

    module = importlib.import_module("comfyui_sigmax.profiles.anima")
    variant_type = getattr(module, "AnimaVariant", None)
    builder = getattr(module, "build_anima_schedule", None)
    assert variant_type is not None
    assert callable(builder)
    result = builder(variant=variant_type.BASE, steps=50, strict_source=True)
    assert result.sigmas == pytest.approx(_expected(50), abs=1e-15)
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(left > right for left, right in pairwise(result.sigmas))


def test_anima_revisions_and_static_scheduler_contract_are_pinned() -> None:
    import importlib

    module = importlib.import_module("comfyui_sigmax.profiles.anima")
    repository_revision = getattr(module, "ANIMA_REPOSITORY_REVISION", None)
    diffusers_revision = getattr(module, "ANIMA_DIFFUSERS_REVISION", None)
    comfyui_revision = getattr(module, "ANIMA_COMFYUI_REVISION", None)
    schema = getattr(module, "ANIMA_BASE_SCHEMA", None)
    profile = getattr(module, "ANIMA_BASE_PROFILE", None)
    assert repository_revision == "f7382c4bf9d7ffe4ceea593a0adbb470c56dd79b"
    assert diffusers_revision == "073c3a9db359c31ad0e8aa268d15775473c2176c"
    assert comfyui_revision == "5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25"
    assert schema is not None
    assert profile is not None
    assert schema.parameters == tuple(sorted(schema.parameters, key=lambda item: item.name))
    assert any("dynamic" in limitation.lower() for limitation in profile.schema.known_limitations)
