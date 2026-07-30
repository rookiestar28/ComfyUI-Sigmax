"""Reviewed standard-library extraction of the official Krea 2 timestep formula."""

from __future__ import annotations

import math
from typing import Any, Final

KREA_SOURCE_URL: Final = "https://github.com/krea-ai/krea-2"
KREA_REVISION: Final = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret
KREA_LOCATOR: Final = "sampling.py:40-53"
KREA_TURBO_MU: Final = 1.15


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def official_krea2_turbo_sigmas(steps: object) -> tuple[float, ...]:
    """Return the fixed-mu Turbo schedule from Krea's pinned ``timesteps`` formula."""

    steps = _positive_int(steps, name="steps")

    exponential = math.exp(KREA_TURBO_MU)
    values = [1.0]
    for index in range(1, steps):
        timestep = (steps - index) / steps
        values.append(exponential / (exponential + (1.0 / timestep - 1.0)))
    values.append(0.0)
    return tuple(values)


def official_krea2_raw_case(
    *,
    width: object,
    height: object,
    steps: object,
) -> dict[str, Any]:
    """Extract RAW alignment, dynamic ``mu``, and sigmas from pinned Krea formulas."""

    width_int = _positive_int(width, name="width")
    height_int = _positive_int(height, name="height")
    steps_int = _positive_int(steps, name="steps")
    effective_width = ((width_int + 15) // 16) * 16
    effective_height = ((height_int + 15) // 16) * 16
    image_seq_len = (effective_width // 16) * (effective_height // 16)
    slope = (1.15 - 0.5) / (6400 - 256)
    mu = slope * image_seq_len + (0.5 - slope * 256)
    exponential = math.exp(mu)
    values = [1.0]
    for index in range(1, steps_int):
        timestep = (steps_int - index) / steps_int
        values.append(exponential / (exponential + (1.0 / timestep - 1.0)))
    values.append(0.0)
    return {
        "effective_height": effective_height,
        "effective_width": effective_width,
        "image_seq_len": image_seq_len,
        "mu": mu,
        "sigmas": tuple(values),
    }
