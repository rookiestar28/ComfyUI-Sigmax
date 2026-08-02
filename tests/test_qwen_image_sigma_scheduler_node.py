"""RED contract coverage for the explicit Qwen Image SIGMAS node."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.qwen_image_sigma_scheduler import (
    QWEN_IMAGE_SIGMA_NODE_ID,
    QWEN_IMAGE_SIGMA_NODE_SCHEMA_ID,
    QwenImageSigmaScheduler,
    build_qwen_image_sigma_schedule,
)


def test_qwen_node_is_registered_with_explicit_modes() -> None:
    assert QWEN_IMAGE_SIGMA_NODE_ID == "Sigmax.QwenImageSigmaScheduler"
    assert NODE_CLASS_MAPPINGS[QWEN_IMAGE_SIGMA_NODE_ID] is QwenImageSigmaScheduler
    inputs = QwenImageSigmaScheduler.INPUT_TYPES()["required"]
    assert inputs["mode"][0] == ("Comfy Fixed", "Diffusers Dynamic")
    assert QwenImageSigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")


def test_qwen_node_fixed_metadata_is_canonical_and_explicit() -> None:
    result = build_qwen_image_sigma_schedule(
        mode="Comfy Fixed",
        steps=50,
        image_seq_len=0,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == QWEN_IMAGE_SIGMA_NODE_SCHEMA_ID
    assert info["profile"]["id"] == "qwen_image.comfy-fixed.official"
    assert info["shift"] == {"dynamic": False, "kind": "fixed_direct_ratio", "ratio": 1.15}
    assert info["slicing"]["output_steps"] == 50
    assert info["guidance"] == {"host_true_cfg": 4.0, "model_guidance": 0.0}


def test_qwen_node_dynamic_mode_rejects_missing_sequence_length() -> None:
    with pytest.raises(ScheduleContractError, match="image_seq_len"):
        build_qwen_image_sigma_schedule(
            mode="Diffusers Dynamic",
            steps=50,
            image_seq_len=0,
            strict_official=True,
            start_step=0,
            end_step=-1,
        )
