"""Optional Torch/ComfyUI state operations for deterministic Flow Euler."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

_MAX_TENSOR_ELEMENTS = 67_108_864


class ComfyDenoisedFlowVelocityEvaluator:
    """Convert one Comfy denoised model result back to direct flow velocity."""

    def __init__(
        self,
        *,
        model: Callable[..., Any],
        extra_args: Mapping[str, object] | None = None,
    ) -> None:
        if not callable(model):
            raise ScheduleContractError("model must be callable")
        self._model = model
        self._extra_args = dict(extra_args or {})

    def __call__(self, state: Any, sigma: float, scheduler_index: int) -> Any:
        del scheduler_index
        if (
            isinstance(sigma, bool)
            or not isinstance(sigma, float)
            or not math.isfinite(sigma)
            or sigma <= 0.0
        ):
            raise ScheduleContractError("Comfy flow velocity requires a positive finite sigma")
        shape = getattr(state, "shape", None)
        if (
            not isinstance(shape, tuple)
            or not shape
            or not isinstance(shape[0], int)
            or shape[0] <= 0
        ):
            raise ScheduleContractError("state must expose a non-empty batch shape")
        new_ones = getattr(state, "new_ones", None)
        if not callable(new_ones):
            raise ScheduleContractError("state cannot construct a Comfy sigma batch")
        sigma_batch = new_ones([shape[0]]) * sigma
        denoised = self._model(state, sigma_batch, **self._extra_args)
        return (state - denoised) / sigma


class TorchFlowEulerStateOperations:
    """Non-mutating tensor operations loaded only inside an explicit host boundary."""

    def __init__(self, *, torch_module: ModuleType | object | None = None) -> None:
        self._torch: Any = (
            importlib.import_module("torch") if torch_module is None else torch_module
        )
        if not isinstance(getattr(self._torch, "Tensor", None), type):
            raise ScheduleContractError("torch module does not expose Tensor")

    def _validated_tensor(self, state: object) -> Any:
        tensor_type = self._torch.Tensor
        if not isinstance(state, tensor_type):
            raise ScheduleContractError("state must be a torch tensor")
        numel = state.numel()
        if (
            isinstance(numel, bool)
            or not isinstance(numel, int)
            or not 0 < numel <= _MAX_TENSOR_ELEMENTS
        ):
            raise ScheduleContractError("tensor element count is outside the allowed range")
        if not state.is_floating_point():
            raise ScheduleContractError("Flow Euler tensors must use a floating dtype")
        finite = self._torch.isfinite(state).all().item()
        if finite is not True:
            raise ScheduleContractError("Flow Euler tensor contains non-finite values")
        return state

    def validate(self, state: object) -> None:
        self._validated_tensor(state)

    def fingerprint(self, state: object) -> str:
        tensor = self._validated_tensor(state)
        cpu = tensor.detach().to(device="cpu").contiguous()
        payload = cpu.view(self._torch.uint8).numpy().tobytes(order="C")
        metadata = (
            f"dtype={tensor.dtype};device={tensor.device};shape="
            + ",".join(str(value) for value in tensor.shape)
            + ";"
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(metadata + payload).hexdigest()

    def add_scaled(self, state: object, velocity: object, scale: float) -> object:
        state_tensor = self._validated_tensor(state)
        velocity_tensor = self._validated_tensor(velocity)
        if (
            state_tensor.shape != velocity_tensor.shape
            or state_tensor.dtype != velocity_tensor.dtype
            or state_tensor.device != velocity_tensor.device
        ):
            raise ScheduleContractError("state and velocity tensor metadata differ")
        if isinstance(scale, bool) or not isinstance(scale, float) or not math.isfinite(scale):
            raise ScheduleContractError("Flow Euler scale must be a finite float")
        result = state_tensor + velocity_tensor * scale
        if result is state_tensor:
            raise ScheduleContractError("Flow Euler tensor operation mutated state in place")
        self.validate(result)
        return cast(object, result)
