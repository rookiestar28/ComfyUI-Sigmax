"""Reviewed standard-library extraction of the official Krea 2 timestep formula."""

from __future__ import annotations

import math
from typing import Final

KREA_SOURCE_URL: Final = "https://github.com/krea-ai/krea-2"
KREA_REVISION: Final = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret
KREA_LOCATOR: Final = "sampling.py:40-53"
KREA_TURBO_MU: Final = 1.15


def official_krea2_turbo_sigmas(steps: object) -> tuple[float, ...]:
    """Return the fixed-mu Turbo schedule from Krea's pinned ``timesteps`` formula."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("steps must be a positive integer")

    exponential = math.exp(KREA_TURBO_MU)
    values = [1.0]
    for index in range(1, steps):
        timestep = (steps - index) / steps
        values.append(exponential / (exponential + (1.0 / timestep - 1.0)))
    values.append(0.0)
    return tuple(values)
