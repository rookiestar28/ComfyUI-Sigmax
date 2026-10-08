"""Explicit migration preserves saved original-Qwen controls and values."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax.adapters.registration import builtin_node_registry
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.qwen_image_sigma_scheduler import (
    QwenImageSigmaScheduler,
    build_qwen_image_sigma_schedule,
)
from comfyui_sigmax.profiles.qwen_native import QWEN_ORIGINAL_NATIVE_PROFILE
from comfyui_sigmax.profiles.registry import ProfileKey, builtin_profile_registry
from comfyui_sigmax.workflows.qwen_alignment import validate_qwen_alignment_workflow


def test_native_choice_is_appended_and_saved_widget_positions_remain() -> None:
    inputs = QwenImageSigmaScheduler.INPUT_TYPES()["required"]
    assert inputs["mode"][0] == ("Comfy Fixed", "Diffusers Dynamic", "Comfy Native")
    assert tuple(inputs) == (
        "mode",
        "steps",
        "image_seq_len",
        "strict_official",
        "start_step",
        "end_step",
    )
    step_options = inputs["steps"][1]
    assert isinstance(step_options, dict)
    assert step_options["default"] == 50


def test_native_output_is_explicit_and_differs_from_legacy_ratio() -> None:
    result = build_qwen_image_sigma_schedule(
        mode="Comfy Native",
        steps=50,
        image_seq_len=0,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert result.sigmas[25] == pytest.approx(0.759510916949111)
    assert info["profile"]["id"] == QWEN_ORIGINAL_NATIVE_PROFILE.schema.profile_id
    assert info["shift"]["mu"] == 1.15
    assert info["shift"]["training_timesteps"] == 10000
    assert info["shift"]["kind"] == "exponential_mu"
    assert (
        builtin_profile_registry()
        .resolve(ProfileKey.from_schema(QWEN_ORIGINAL_NATIVE_PROFILE.schema))
        .schema
        == QWEN_ORIGINAL_NATIVE_PROFILE.schema
    )


def test_native_rejects_unexecuted_token_control() -> None:
    with pytest.raises(ScheduleContractError, match=r"native.*image_seq_len"):
        build_qwen_image_sigma_schedule(
            mode="Comfy Native",
            steps=50,
            image_seq_len=1024,
            strict_official=True,
            start_step=0,
            end_step=-1,
        )


def test_serialized_native_workflow_validates_and_rejects_choice_drift() -> None:
    schema = builtin_node_registry().object_info_projection()
    schema["Sigmax.QwenImageSigmaScheduler"]["input_order"] = {
        "required": list(QwenImageSigmaScheduler.INPUT_TYPES()["required"])
    }
    assert validate_qwen_alignment_workflow(schema)["status"] == "succeeded"
    schema["Sigmax.QwenImageSigmaScheduler"]["input"]["required"]["mode"][0].pop()
    with pytest.raises(ScheduleContractError, match="drift"):
        validate_qwen_alignment_workflow(schema)
