"""Model-free native ComfyUI workflow contracts for MiniMax H3 Base."""

from __future__ import annotations

import json
from typing import cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.workflows.minimax_h3 import (
    MiniMaxH3HostWorkflow,
    MiniMaxH3ModelFiles,
    MiniMaxH3WorkflowSpec,
    build_minimax_h3_host_workflow,
    build_minimax_h3_host_workflow_prompt,
    default_minimax_h3_model_files,
)


def _class_types(prompt: dict[str, dict[str, object]]) -> list[str]:
    return [str(node["class_type"]) for node in prompt.values()]


def _inputs(prompt: dict[str, dict[str, object]], node_id: str) -> dict[str, object]:
    return cast(dict[str, object], prompt[node_id]["inputs"])


def test_fl2va_graph_is_explicit_and_uses_sigmax_sigmas_once() -> None:
    spec = MiniMaxH3WorkflowSpec(
        variant="H3 Base FL2VA",
        prompt="A slow camera move through a quiet glass greenhouse with soft stereo ambience.",
        first_frame="input/start.png",
        last_frame="input/end.png",
    )
    workflow = build_minimax_h3_host_workflow(spec)
    prompt = workflow.prompt

    assert isinstance(workflow, MiniMaxH3HostWorkflow)
    assert prompt["1"] == {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "H3/minimax_h3_fl2va_bf16.safetensors",
            "weight_dtype": "default",
        },
    }
    assert prompt["2"] == {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "type": "minimax",
        },
    }
    assert prompt["6"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert _inputs(prompt, "6")["first_frame"] == ["17", 0]
    assert _inputs(prompt, "6")["last_frame"] == ["18", 0]
    assert _inputs(prompt, "11")["sigmas"] == ["7", 0]
    assert _inputs(prompt, "11")["latent_image"] == ["6", 1]
    assert _inputs(prompt, "9")["model"] == ["5", 0]
    assert _inputs(prompt, "9")["conditioning"] == ["6", 0]
    assert _inputs(prompt, "5")["shift_video"] == 12.0
    assert _inputs(prompt, "5")["shift_audio"] == 3.0
    assert "already_shifted" not in _inputs(prompt, "7")
    assert workflow.contract.schedule_ownership == "external_video_only"
    assert workflow.contract.audio_ownership == "model_native"
    assert workflow.contract.external_video_shift_applied_once is True
    assert workflow.contract.external_audio_schedule is False
    assert _class_types(prompt).count("Sigmax.MiniMaxH3SigmaScheduler") == 1
    assert _class_types(prompt).count("MiniMaxH3SigmaShift") == 1
    assert "BasicScheduler" not in _class_types(prompt)
    assert "SamplerCustom" not in _class_types(prompt)
    assert "SamplerCustomAdvanced" in _class_types(prompt)
    assert json.dumps(prompt, ensure_ascii=False, allow_nan=False)


def test_ref2va_graph_uses_v3_dynamic_reference_paths() -> None:
    spec = MiniMaxH3WorkflowSpec(
        variant="H3 Base Ref2VA",
        prompt="A portrait subject walks through a sunlit courtyard; preserve the reference identity.",
        reference_images=("refs/subject-a.png", "refs/subject-b.jpg"),
        ref_image_size="max",
    )
    prompt = build_minimax_h3_host_workflow_prompt(spec)

    assert prompt["6"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    condition_inputs = _inputs(prompt, "6")
    assert condition_inputs["ref_image_size"] == "max"
    assert condition_inputs["ref_images.ref_image_0"] == ["17", 0]
    assert condition_inputs["ref_images.ref_image_1"] == ["18", 0]
    assert prompt["17"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "refs/subject-a.png"},
    }
    assert prompt["18"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "refs/subject-b.jpg"},
    }
    assert _inputs(prompt, "11")["guider"] == ["9", 0]
    assert _inputs(prompt, "12")["av_latent"] == ["11", 0]
    assert _inputs(prompt, "13")["samples"] == ["12", 0]
    assert _inputs(prompt, "14")["samples"] == ["12", 1]


def test_default_model_files_are_variant_specific_and_overrides_are_explicit() -> None:
    fl2va = default_minimax_h3_model_files("H3 Base FL2VA")
    ref2va = default_minimax_h3_model_files("H3 Base Ref2VA")
    assert fl2va.diffusion_model.endswith("minimax_h3_fl2va_bf16.safetensors")
    assert ref2va.diffusion_model.endswith("minimax_h3_ref2va_bf16.safetensors")

    custom = MiniMaxH3ModelFiles(
        diffusion_model="H3/minimax_h3_fl2va_int8_convrot.safetensors",
        text_encoder="text_encoders/custom_h3.safetensors",
    )
    spec = MiniMaxH3WorkflowSpec(
        variant="H3 Base FL2VA",
        prompt="custom artifact contract",
        model_files=custom,
    )
    prompt = build_minimax_h3_host_workflow_prompt(spec)
    assert _inputs(prompt, "1")["unet_name"] == custom.diffusion_model
    assert _inputs(prompt, "2")["clip_name"] == custom.text_encoder


def test_workflow_builder_is_deterministic() -> None:
    spec = MiniMaxH3WorkflowSpec(
        variant="H3 Base Ref2VA",
        prompt="deterministic graph",
        reference_images=("ref.png",),
        grid_points=20,
        seed=42,
    )
    first = build_minimax_h3_host_workflow_prompt(spec)
    second = build_minimax_h3_host_workflow_prompt(spec)
    assert first == second
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variant": "auto", "prompt": "bad"},
        {"variant": "H3 Base Ref2VA", "prompt": "bad"},
        {"variant": "H3 Base FL2VA", "prompt": "bad", "width": 1000},
        {"variant": "H3 Base FL2VA", "prompt": "bad", "length": 6},
        {"variant": "H3 Base FL2VA", "prompt": "bad", "sampler_name": "res_multistep"},
    ],
)
def test_workflow_spec_rejects_ambiguous_or_unsafe_inputs(kwargs: dict[str, object]) -> None:
    if kwargs.get("variant") == "H3 Base Ref2VA" and "reference_images" not in kwargs:
        with pytest.raises(ScheduleContractError, match="reference"):
            MiniMaxH3WorkflowSpec(**kwargs)  # type: ignore[arg-type]
        return
    if kwargs.get("variant") == "auto":
        with pytest.raises(ScheduleContractError, match="explicit"):
            MiniMaxH3WorkflowSpec(**kwargs)  # type: ignore[arg-type]
        return
    with pytest.raises(ScheduleContractError):
        MiniMaxH3WorkflowSpec(**kwargs)  # type: ignore[arg-type]


def test_model_file_bundle_rejects_absolute_paths() -> None:
    with pytest.raises(ScheduleContractError, match="host-relative"):
        MiniMaxH3ModelFiles(diffusion_model="C:/outside/minimax_h3_fl2va_bf16.safetensors")


def test_workflow_rejects_a_contradictory_explicit_diffusion_variant() -> None:
    with pytest.raises(ScheduleContractError, match="contradictory"):
        MiniMaxH3WorkflowSpec(
            variant="H3 Base Ref2VA",
            prompt="mismatched artifact",
            reference_images=("ref.png",),
            model_files=MiniMaxH3ModelFiles(diffusion_model="H3/minimax_h3_fl2va_bf16.safetensors"),
        )


def test_variants_cannot_mix_ref2va_images_and_fl2va_keyframes() -> None:
    with pytest.raises(ScheduleContractError, match="cannot receive"):
        MiniMaxH3WorkflowSpec(
            variant="H3 Base FL2VA",
            prompt="mixed inputs",
            reference_images=("ref.png",),
        )
    with pytest.raises(ScheduleContractError, match="cannot combine"):
        MiniMaxH3WorkflowSpec(
            variant="H3 Base Ref2VA",
            prompt="mixed inputs",
            reference_images=("ref.png",),
            first_frame="first.png",
        )
