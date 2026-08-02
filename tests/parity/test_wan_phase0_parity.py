"""Phase 0 RED parity contracts for clean-room Wan schedule equations."""

from __future__ import annotations

from itertools import pairwise

import pytest

pytestmark = pytest.mark.parity


def _expected(steps: int, ratio: float) -> tuple[float, ...]:
    values = tuple((steps - index) / steps for index in range(steps))
    shifted = tuple(ratio * value / (1.0 + (ratio - 1.0) * value) for value in values)
    return (*shifted, 0.0)


@pytest.mark.parametrize("steps,ratio", [(50, 5.0), (40, 3.0), (40, 5.0), (50, 8.0)])
def test_wan_profiles_match_pinned_direct_ratio_equation(steps: int, ratio: float) -> None:
    import importlib

    module = importlib.import_module("comfyui_sigmax.profiles.wan")
    profile = {
        (50, 5.0): module.WanProfileId.WAN21_T2V_OFFICIAL,
        (40, 3.0): module.WanProfileId.WAN21_I2V_480P_OFFICIAL,
        (40, 5.0): module.WanProfileId.WAN21_I2V_720P_OFFICIAL,
        (50, 8.0): module.WanProfileId.WAN21_COMFY_NATIVE,
    }[(steps, ratio)]
    resolution = {
        module.WanProfileId.WAN21_I2V_480P_OFFICIAL: module.WanResolution.P480,
        module.WanProfileId.WAN21_I2V_720P_OFFICIAL: module.WanResolution.P720,
    }.get(profile, module.WanResolution.NONE)
    result = module.build_wan_schedule(profile=profile, steps=steps, resolution=resolution)
    assert result.sigmas == pytest.approx(_expected(steps, ratio), abs=1e-15)
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(left > right for left, right in pairwise(result.sigmas))


def test_wan_pins_and_solver_ownership_are_explicit() -> None:
    import importlib

    module = importlib.import_module("comfyui_sigmax.profiles.wan")
    assert (
        module.WAN21_REPOSITORY_REVISION
        == "9737cba9c1c3c4d04b33fcad41c111989865d315"  # pragma: allowlist secret
    )  # pragma: allowlist secret
    assert (
        module.WAN22_REPOSITORY_REVISION
        == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"  # pragma: allowlist secret
    )  # pragma: allowlist secret
    assert (
        module.WAN_COMFYUI_REVISION
        == "5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25"  # pragma: allowlist secret
    )  # pragma: allowlist secret
    assert (
        module.WAN_DIFFUSERS_REVISION
        == "3c468926ffd12b69baa4316e27b09306b8da19a6"  # pragma: allowlist secret
    )  # pragma: allowlist secret
    schema = module.WAN21_T2V_DIFFUSERS_SCHEMA
    assert schema.parameters == tuple(sorted(schema.parameters, key=lambda item: item.name))
    assert any("unipc" in limitation.lower() for limitation in schema.known_limitations)


def test_wan_a14b_boundary_crossing_is_deterministic_and_exact_at_equality() -> None:
    import importlib

    module = importlib.import_module("comfyui_sigmax.profiles.wan")
    helper = getattr(module, "derive_wan_boundary", None)
    assert callable(helper)
    sigmas = (1.0, 0.95, 0.875, 0.7, 0.0)
    exact = helper(sigmas=sigmas, normalized_boundary=0.875)
    assert exact.transition_index == 2
    assert exact.crossing == "at_or_above"
    assert exact.routing_owner == "caller"
    assert exact.model_dispatch is False
