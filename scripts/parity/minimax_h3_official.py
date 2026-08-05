"""Clean-room standard-library transcription of pinned MiniMax H3 scheduler formulas.

This module is a source-level oracle for tests only. It deliberately imports no framework and does
not execute publisher or Diffusers code. The isolated parity runner separately invokes the pinned
Diffusers class when that optional environment is available.
"""

from __future__ import annotations

import math
import struct
from typing import Final

SOURCE_URL: Final = "https://github.com/huggingface/diffusers"
BRANCH_URL: Final = "https://github.com/huggingface/diffusers/tree/minimax-h3"
DIFFUSERS_REVISION: Final = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"  # pragma: allowlist secret
SCHEDULER_LOCATOR: Final = "src/diffusers/schedulers/scheduling_minimax_h3.py"
_F32: Final = ">f"


def _f32(value: float) -> float:
    try:
        return float(struct.unpack(_F32, struct.pack(_F32, float(value)))[0])
    except (OverflowError, struct.error) as exc:
        raise ValueError("value cannot be represented as float32") from exc


def _positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise ValueError(f"{name} must be an integer >= 2")
    return value


def _positive_shift(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("shift must be a finite positive number")
    return float(value)


def _direct_ratio(value: float, shift: float) -> float:
    numerator = _f32(shift * value)
    denominator = _f32(1.0 + _f32((shift - 1.0) * value))
    return _f32(numerator / denominator)


def clean_room_sigma_grid(grid_points: object, shift: object) -> tuple[float, ...]:
    """Transcribe H3's float32 endpoint grid and direct-ratio shift."""

    points = _positive_int(grid_points, name="grid_points")
    ratio = _positive_shift(shift)
    denominator = points - 1
    base = tuple(_f32(1.0 - (index / denominator)) for index in range(points))
    shifted = tuple(_direct_ratio(value, ratio) for value in base)
    result = [shifted[0]]
    for value in shifted[1:]:
        if value != result[-1]:
            result.append(value)
    if result[-1] != 0.0:
        raise ValueError("H3 schedule must terminate at zero")
    return tuple(result)


def clean_room_dataward_step(
    *,
    sample: tuple[float, ...],
    velocity: tuple[float, ...],
    timestep: float,
    sigma: float,
    sigma_next: float,
) -> tuple[float, ...]:
    """Transcribe the pinned H3 data-ward Euler step in float32 arithmetic."""

    if len(sample) != len(velocity) or not sample:
        raise ValueError("sample and velocity dimensions must match")
    if not all(math.isfinite(float(value)) for value in (*sample, *velocity)):
        raise ValueError("sample and velocity values must be finite")
    current_sigma = _f32(1.0 - _f32(timestep))
    denoised = tuple(
        _f32(_f32(x) + _f32(current_sigma * _f32(v))) for x, v in zip(sample, velocity, strict=True)
    )
    ratio = _f32(_f32(sigma_next) / _f32(sigma))
    return tuple(
        _f32(_f32(ratio * _f32(x)) + _f32(_f32(1.0 - ratio) * _f32(x0)))
        for x, x0 in zip(sample, denoised, strict=True)
    )


__all__ = [
    "BRANCH_URL",
    "DIFFUSERS_REVISION",
    "SCHEDULER_LOCATOR",
    "SOURCE_URL",
    "clean_room_dataward_step",
    "clean_room_sigma_grid",
]
