"""M5-03 optional ComfyUI/Torch adapter isolation tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest
from comfyui_sigmax.core import ScheduleContractError

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "comfyui_sigmax" / "adapters" / "comfyui_flow_euler.py"


class FakeTensor:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def new_ones(self, shape: list[int]) -> FakeTensor:
        return FakeTensor(tuple(1.0 for _ in range(shape[0])))

    def __mul__(self, scalar: float) -> FakeTensor:
        return FakeTensor(tuple(value * scalar for value in self.values))

    def __sub__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(tuple(a - b for a, b in zip(self.values, other.values, strict=True)))

    def __truediv__(self, scalar: float) -> FakeTensor:
        return FakeTensor(tuple(value / scalar for value in self.values))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeTensor) and self.values == pytest.approx(other.values)


def test_adapter_module_has_no_top_level_optional_framework_imports() -> None:
    tree = ast.parse(ADAPTER_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert "torch" not in imports
    assert "comfy" not in imports
    module = importlib.import_module("comfyui_sigmax.adapters.comfyui_flow_euler")
    assert module.ComfyDenoisedFlowVelocityEvaluator is not None


def test_denoised_adapter_recovers_direct_flow_velocity_once() -> None:
    module = importlib.import_module("comfyui_sigmax.adapters.comfyui_flow_euler")
    calls: list[tuple[FakeTensor, FakeTensor, dict[str, Any]]] = []
    expected_velocity = FakeTensor((0.25, -0.75))

    def model(state: FakeTensor, sigma: FakeTensor, **extra_args: Any) -> FakeTensor:
        calls.append((state, sigma, extra_args))
        return state - expected_velocity * sigma.values[0]

    evaluator = module.ComfyDenoisedFlowVelocityEvaluator(
        model=model,
        extra_args={"public_case": "m5-03"},
    )
    state = FakeTensor((1.0, -0.5))

    assert evaluator(state, 0.5, 7) == expected_velocity
    assert len(calls) == 1
    assert calls[0][1].values == (0.5,)
    assert calls[0][2] == {"public_case": "m5-03"}


def test_denoised_adapter_rejects_terminal_or_invalid_sigma_before_model_call() -> None:
    module = importlib.import_module("comfyui_sigmax.adapters.comfyui_flow_euler")
    calls = 0

    def model(state: FakeTensor, sigma: FakeTensor, **extra_args: Any) -> FakeTensor:
        nonlocal calls
        calls += 1
        return state

    evaluator = module.ComfyDenoisedFlowVelocityEvaluator(model=model)
    for sigma in (0.0, -0.1, float("nan")):
        with pytest.raises(ScheduleContractError, match="sigma"):
            evaluator(FakeTensor((1.0,)), sigma, 0)
    assert calls == 0


def test_torch_operations_require_an_explicit_or_available_torch_module() -> None:
    module = importlib.import_module("comfyui_sigmax.adapters.comfyui_flow_euler")

    class FakeTorch:
        class Tensor:
            pass

    operations = module.TorchFlowEulerStateOperations(torch_module=FakeTorch())
    with pytest.raises(ScheduleContractError, match="tensor"):
        operations.validate(object())
