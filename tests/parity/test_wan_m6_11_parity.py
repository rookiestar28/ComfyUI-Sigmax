"""Source-pinned equation parity for M6-11 official-native Wan profiles."""

from __future__ import annotations

import importlib
import math
from decimal import Decimal, localcontext
from itertools import pairwise

import pytest

pytestmark = pytest.mark.parity

_CASES = (
    ("wan2.2.animate.14b.official-native", 20, 5.0),
    ("wan-animate2.14b.base.official-native", 40, 5.0),
    ("wan-animate2.14b.distilled.official-native", 10, 5.0),
)


def _official_native_equation(*, steps: int, shift: float) -> tuple[float, ...]:
    with localcontext() as context:
        context.prec = 80
        ratio = Decimal(str(shift))
        values = []
        for index in range(steps):
            base = Decimal(steps - index) / Decimal(steps)
            values.append(float(ratio * base / (Decimal(1) + (ratio - Decimal(1)) * base)))
    return (*values, 0.0)


@pytest.mark.parametrize(("profile_id", "steps", "shift"), _CASES)
def test_m6_11_profiles_match_independent_native_reciprocal_equation(
    profile_id: str, steps: int, shift: float
) -> None:
    module = importlib.import_module("comfyui_sigmax.profiles.wan")
    result = module.build_wan_schedule(
        profile=module.WanProfileId(profile_id),
        steps=steps,
        resolution=module.WanResolution.NONE,
        strict_source=True,
    )
    expected = _official_native_equation(steps=steps, shift=shift)
    errors = tuple(
        abs(actual - reference) for actual, reference in zip(result.sigmas, expected, strict=True)
    )
    assert len(result.sigmas) == steps + 1
    assert max(errors) <= 1e-15
    assert sum(errors) / len(errors) <= 1e-15
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result.sigmas)
    assert all(left > right for left, right in pairwise(result.sigmas))
    assert result.request.base_grid.identifier == "flowmatch.reciprocal_step"
    assert tuple(transform.name for transform in result.request.transforms) == (
        "direct_ratio.shift",
        "terminal.append_zero",
    )
    assert result.sigmas[0] == 1.0 and result.sigmas[-1] == 0.0
