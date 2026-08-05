"""Authorization-gated MiniMax H3 model-lane planning contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles.minimax_h3 import MINIMAX_H3_COMFYUI_REVISION

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "scripts" / "run_comfyui_e2e.py"


def _harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sigmax_comfyui_e2e_model_lane", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("ComfyUI E2E harness is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_exposes_explicit_model_plan_controls() -> None:
    harness = _harness()
    arguments = harness._parser().parse_args(
        [
            "--minimax-h3-model-plan",
            "--minimax-h3-model-variant",
            "H3 Base FL2VA",
            "--minimax-h3-model-prompt",
            "a controlled local test prompt",
            "--minimax-h3-model-artifact",
            "H3/minimax_h3_fl2va_bf16.safetensors",
            "--minimax-h3-reference-image",
            "input/ref.png",
            "--minimax-h3-allow-model-weights",
            "--minimax-h3-license-ack",
        ]
    )

    assert arguments.minimax_h3_model_plan is True
    assert arguments.minimax_h3_model_variant == "H3 Base FL2VA"
    assert arguments.minimax_h3_reference_image == ["input/ref.png"]
    assert arguments.minimax_h3_allow_model_weights is True
    assert arguments.minimax_h3_license_ack is True


def test_model_plan_requires_both_weight_and_license_acknowledgements() -> None:
    harness = _harness()

    with pytest.raises(ScheduleContractError, match="license"):
        harness.build_minimax_h3_model_lane_plan(
            variant="H3 Base FL2VA",
            prompt="controlled local test prompt",
            model_artifact="H3/minimax_h3_fl2va_bf16.safetensors",
            allow_model_weights=True,
            license_ack=False,
        )

    with pytest.raises(ScheduleContractError, match="weight"):
        harness.build_minimax_h3_model_lane_plan(
            variant="H3 Base FL2VA",
            prompt="controlled local test prompt",
            model_artifact="H3/minimax_h3_fl2va_bf16.safetensors",
            allow_model_weights=False,
            license_ack=True,
        )


def test_model_plan_is_explicit_and_does_not_execute_weights() -> None:
    harness = _harness()
    plan = harness.build_minimax_h3_model_lane_plan(
        variant="H3 Base Ref2VA",
        prompt="controlled local reference test prompt",
        model_artifact="H3/minimax_h3_ref2va_bf16.safetensors",
        allow_model_weights=True,
        license_ack=True,
        reference_images=("input/ref.png",),
    )

    assert plan["schema"] == "sigmax.minimax-h3-model-lane/1"
    assert plan["status"] == "authorized_not_executed"
    assert plan["host"] == {
        "version": "0.30.0",
        "revision": MINIMAX_H3_COMFYUI_REVISION,
    }
    assert plan["authorization"] == {
        "allow_model_weights": True,
        "license_ack": True,
        "network": False,
    }
    assert plan["execution"] == {"performed": False, "weights_loaded": False}
    assert plan["model_files"]["diffusion_model"] == ("H3/minimax_h3_ref2va_bf16.safetensors")
    prompt = plan["workflow"]
    assert isinstance(prompt, dict)
    assert prompt["1"]["class_type"] == "UNETLoader"
    assert prompt["6"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert prompt["8"]["class_type"] == "KSamplerSelect"


def test_model_plan_rejects_wrong_host_and_unsafe_artifact() -> None:
    harness = _harness()
    common = {
        "variant": "H3 Base FL2VA",
        "prompt": "controlled local test prompt",
        "allow_model_weights": True,
        "license_ack": True,
    }

    with pytest.raises(ScheduleContractError, match="host revision"):
        harness.build_minimax_h3_model_lane_plan(
            **common,
            model_artifact="H3/minimax_h3_fl2va_bf16.safetensors",
            host_revision="unreviewed",
        )

    with pytest.raises(ScheduleContractError, match="host-relative"):
        harness.build_minimax_h3_model_lane_plan(
            **common,
            model_artifact="C:/private/minimax_h3_fl2va_bf16.safetensors",
        )
