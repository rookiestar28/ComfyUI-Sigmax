"""Optional benchmark-only diagnostics for Krea 2 conditioning transforms."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

from comfyui_sigmax.conditioning.contracts import (
    KREA2_TAP_COUNT,
    KREA2_TAP_DIM,
    validate_krea2_conditioning_shape,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditioningDiagnostics:
    """Bounded scalar diagnostics kept out of the default node report."""

    input_rms: float
    output_rms: float
    cosine: float
    relative_change: float
    input_tap_rms: tuple[float, ...]
    output_tap_rms: tuple[float, ...]

    def __post_init__(self) -> None:
        scalars = (
            self.input_rms,
            self.output_rms,
            self.cosine,
            self.relative_change,
        )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in scalars
        ):
            raise ValueError("conditioning diagnostics contain non-finite scalar values")
        if self.input_rms < 0.0 or self.output_rms < 0.0 or self.relative_change < 0.0:
            raise ValueError(
                "conditioning diagnostics RMS and relative change must be non-negative"
            )
        if not -1.0 <= self.cosine <= 1.0:
            raise ValueError("conditioning diagnostics cosine must be between -1 and 1")
        for label, values in (
            ("input", self.input_tap_rms),
            ("output", self.output_tap_rms),
        ):
            if len(values) != KREA2_TAP_COUNT or any(
                not math.isfinite(float(value)) or float(value) < 0.0 for value in values
            ):
                raise ValueError(f"conditioning diagnostics {label} tap RMS is invalid")


def compute_conditioning_diagnostics(
    input_tensor: object, output_tensor: object
) -> ConditioningDiagnostics:
    """Compute bounded tensor metrics for an explicit benchmark or test lane only."""

    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as exc:
        raise ValueError("conditioning diagnostics require the optional Torch runtime") from exc
    if not torch.is_tensor(input_tensor) or not torch.is_tensor(output_tensor):
        raise ValueError("conditioning diagnostics require two tensors")
    input_value: Any = input_tensor
    output_value: Any = output_tensor
    input_shape = validate_krea2_conditioning_shape(tuple(input_value.shape))
    output_shape = validate_krea2_conditioning_shape(tuple(output_value.shape))
    if input_shape != output_shape:
        raise ValueError("conditioning diagnostics tensor shapes must match")
    if (
        not torch.is_floating_point(input_value)
        or not torch.is_floating_point(output_value)
        or torch.is_complex(input_value)
        or torch.is_complex(output_value)
    ):
        raise ValueError("conditioning diagnostics require real floating tensors")
    if not bool(torch.isfinite(input_value).all().item()) or not bool(
        torch.isfinite(output_value).all().item()
    ):
        raise ValueError("conditioning diagnostics require finite tensors")

    source = input_value.float()
    result = output_value.float()
    source_flat = source.reshape(-1)
    result_flat = result.reshape(-1)
    source_norm = torch.linalg.vector_norm(source_flat)
    result_norm = torch.linalg.vector_norm(result_flat)
    denominator = (source_norm * result_norm).clamp_min(1e-8)
    cosine = torch.dot(source_flat, result_flat) / denominator
    relative_change = torch.linalg.vector_norm(result_flat - source_flat) / source_norm.clamp_min(
        1e-8
    )
    source_shaped = source.reshape(input_shape[0], input_shape[1], KREA2_TAP_COUNT, KREA2_TAP_DIM)
    result_shaped = result.reshape(input_shape[0], input_shape[1], KREA2_TAP_COUNT, KREA2_TAP_DIM)
    source_tap_rms = torch.sqrt(torch.mean(source_shaped * source_shaped, dim=(0, 1, 3)))
    result_tap_rms = torch.sqrt(torch.mean(result_shaped * result_shaped, dim=(0, 1, 3)))
    return ConditioningDiagnostics(
        input_rms=float(torch.sqrt(torch.mean(source * source)).item()),
        output_rms=float(torch.sqrt(torch.mean(result * result)).item()),
        cosine=float(cosine.item()),
        relative_change=float(relative_change.item()),
        input_tap_rms=tuple(float(item) for item in source_tap_rms.tolist()),
        output_tap_rms=tuple(float(item) for item in result_tap_rms.tolist()),
    )
