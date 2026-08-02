"""Torch execution boundary for the graph-local Krea 2 conditioning modifier."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from comfyui_sigmax.conditioning import (
    KREA2_TAP_COUNT,
    KREA2_TAP_DIM,
    ConditioningModifierRequest,
    Krea2ConditioningVariant,
    effective_gains,
    validate_krea2_conditioning_shape,
)
from comfyui_sigmax.core import ScheduleContractError


class ConditioningAdapterError(ScheduleContractError):
    """A standard ComfyUI CONDITIONING value cannot be transformed safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditioningTransformStats:
    input_shape: tuple[int, int, int]
    input_shapes: tuple[tuple[int, int, int], ...]
    dtype: str
    device: str
    variant: Krea2ConditioningVariant
    conditioning_entries: int
    transformed_entries: int


def transform_krea2_conditioning(
    conditioning: object,
    request: ConditioningModifierRequest,
) -> tuple[list[list[object]], ConditioningTransformStats]:
    if not isinstance(request, ConditioningModifierRequest):
        raise ConditioningAdapterError("conditioning adapter requires a validated modifier request")
    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as exc:
        raise ConditioningAdapterError(
            "ComfyUI host execution requires the optional Torch runtime"
        ) from exc
    if not isinstance(conditioning, list) or not conditioning:
        raise ConditioningAdapterError("CONDITIONING must be a non-empty list")

    output: list[list[object]] = []
    shapes: list[tuple[int, int, int]] = []
    first_dtype: str | None = None
    first_device: str | None = None
    transformed_entries = 0
    gains = effective_gains(request)
    identity = request.strength == 0.0 or all(gain == 1.0 for gain in gains)
    for index, entry in enumerate(conditioning):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ConditioningAdapterError(
                f"CONDITIONING entry {index} must be a two-item tensor/metadata pair"
            )
        tensor, metadata = entry
        if not torch.is_tensor(tensor):
            raise ConditioningAdapterError(f"CONDITIONING entry {index} tensor is invalid")
        if not isinstance(metadata, Mapping):
            raise ConditioningAdapterError(f"CONDITIONING entry {index} metadata must be a mapping")
        if not torch.is_floating_point(tensor) or torch.is_complex(tensor):
            raise ConditioningAdapterError(
                f"CONDITIONING entry {index} tensor must use a real floating dtype"
            )
        shape = validate_krea2_conditioning_shape(tuple(tensor.shape))
        if not bool(torch.isfinite(tensor).all().item()):
            raise ConditioningAdapterError(f"CONDITIONING entry {index} contains non-finite values")
        if first_dtype is None:
            first_dtype = str(tensor.dtype)
            first_device = str(tensor.device)
        if identity:
            result_tensor = tensor
        else:
            result_tensor = _rebalance_tensor(
                torch,
                tensor,
                shape=shape,
                gains=gains,
            )
            transformed_entries += 1
        output.append([result_tensor, dict(metadata)])
        shapes.append(shape)

    if first_dtype is None or first_device is None:
        raise ConditioningAdapterError("CONDITIONING contains no valid entries")
    return output, ConditioningTransformStats(
        input_shape=shapes[0],
        input_shapes=tuple(shapes),
        dtype=first_dtype,
        device=first_device,
        variant=request.variant,
        conditioning_entries=len(output),
        transformed_entries=transformed_entries,
    )


def _rebalance_tensor(
    torch: Any,
    tensor: Any,
    *,
    shape: tuple[int, int, int],
    gains: tuple[float, ...],
) -> Any:
    """Apply tap gains and per-sample RMS restoration at the Torch boundary."""

    # Keep the pure package Torch-free; all tensor operations stay behind this lazy host edge.
    work = tensor.float()
    batch, sequence, _ = shape
    shaped = work.reshape(batch, sequence, KREA2_TAP_COUNT, KREA2_TAP_DIM)
    gain_tensor = torch.tensor(gains, dtype=torch.float32, device=tensor.device).reshape(
        1, 1, KREA2_TAP_COUNT, 1
    )
    weighted = shaped * gain_tensor
    input_rms = torch.sqrt(torch.mean(work * work, dim=(1, 2)))
    weighted_rms = torch.sqrt(torch.mean(weighted * weighted, dim=(1, 2, 3)))
    scale = input_rms / weighted_rms.clamp_min(1e-8)
    scaled = weighted * scale.reshape(-1, 1, 1, 1)
    zero_input = input_rms == 0
    result = torch.where(zero_input.reshape(-1, 1, 1, 1), shaped, scaled)
    result = result.reshape_as(work).to(dtype=tensor.dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise ConditioningAdapterError("conditioning rebalance produced non-finite output")
    return result
