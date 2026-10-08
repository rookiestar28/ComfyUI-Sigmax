"""Separate 2.1 identity and source-authority contracts."""

from __future__ import annotations

import importlib
import json
import struct
from typing import Any

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.adapters.registration import builtin_node_registry
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.workflows.qwen_alignment import validate_qwen_image21_workflows


def _module() -> Any:
    return importlib.import_module("comfyui_sigmax.nodes.qwen_image21_sigma_scheduler")


def _inputs(**changes: object) -> dict[str, Any]:
    return {
        "mode": "Comfy Native",
        "steps": 25,
        "target_source": "None (Native)",
        "width": 0,
        "height": 0,
        "image_seq_len": 0,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
        **changes,
    }


def test_new_node_has_distinct_identity_and_native_profile() -> None:
    module = _module()
    assert (
        NODE_CLASS_MAPPINGS["Sigmax.QwenImage21SigmaScheduler"] is module.QwenImage21SigmaScheduler
    )
    result = module.build_qwen_image21_sigma_schedule(**_inputs())
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == "sigmax.qwen-image21-sigma-node/1"
    assert info["profile"]["id"] == "qwen_image21.comfy-native.framework-reference"
    assert info["profile"]["variant"] == "v2_1"
    assert info["shift"]["mu"] == 0.69
    assert info["license"]["identifier"] == "LicenseRef-Qwen-Research"
    assert len(result.sigmas) == 26 and result.sigmas[-1] == 0.0


def test_dynamic_dimensions_and_tokens_are_equivalent_and_explicit() -> None:
    module = _module()
    dimensions = module.build_qwen_image21_sigma_schedule(
        **_inputs(
            mode="Diffusers Dynamic",
            steps=40,
            target_source="Dimensions",
            width=1024,
            height=1536,
        )
    )
    tokens = module.build_qwen_image21_sigma_schedule(
        **_inputs(
            mode="Diffusers Dynamic",
            steps=40,
            target_source="Target Tokens",
            image_seq_len=6144,
        )
    )
    assert dimensions.sigmas == tokens.sigmas
    info = json.loads(dimensions.schedule_info_json)
    assert info["target"]["image_seq_len"] == 6144
    assert info["shift"]["shift_terminal"] == 0.02
    assert dimensions.sigmas[-2] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 32},
        {"image_seq_len": 256},
        {"target_source": "Dimensions"},
        {"mode": "Diffusers Dynamic", "steps": 40},
        {
            "mode": "Diffusers Dynamic",
            "steps": 40,
            "target_source": "Dimensions",
            "width": 1024,
            "height": 1024,
            "image_seq_len": 4096,
        },
        {
            "mode": "Diffusers Dynamic",
            "steps": 40,
            "target_source": "Dimensions",
            "width": 1025,
            "height": 1024,
        },
        {
            "mode": "Diffusers Dynamic",
            "steps": 1,
            "strict_official": False,
            "target_source": "Target Tokens",
            "image_seq_len": 4096,
        },
        {
            "mode": "Diffusers Dynamic",
            "steps": 40,
            "target_source": "Target Tokens",
            "image_seq_len": 4096,
            "height": 32,
        },
        {"width": False},
        {"steps": True},
        {"mode": "automatic"},
        {"target_source": "automatic"},
        {"start_step": 25},
        {"end_step": 0},
        {"steps": 24},
    ],
)
def test_ambiguous_unused_or_invalid_controls_fail(changes: dict[str, object]) -> None:
    with pytest.raises(ScheduleContractError):
        _module().build_qwen_image21_sigma_schedule(**_inputs(**changes))


def test_modified_steps_and_slice_follow_complete_schedule() -> None:
    module = _module()
    complete = module.build_qwen_image21_sigma_schedule(**_inputs(steps=7, strict_official=False))
    sliced = module.build_qwen_image21_sigma_schedule(
        **_inputs(steps=7, strict_official=False, start_step=2, end_step=5)
    )
    info = json.loads(sliced.schedule_info_json)
    assert sliced.sigmas == complete.sigmas[2:6]
    assert info["profile"]["evidence"] == "modified"
    assert info["slicing"]["output_steps"] == 3
    assert (
        info["fingerprints"]["complete"]
        == json.loads(complete.schedule_info_json)["fingerprints"]["complete"]
    )
    assert (
        info["fingerprints"]["output"]
        != json.loads(complete.schedule_info_json)["fingerprints"]["output"]
    )


def test_actual_tensor_quantization_is_bound_to_output_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()

    class Tensor:
        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = [struct.unpack("!f", struct.pack("!f", value))[0] for value in values]

        def tolist(self) -> list[float]:
            return self.values

    class Torch:
        FloatTensor = Tensor

    monkeypatch.setattr(
        module.importlib, "import_module", lambda name: Torch if name == "torch" else None
    )
    tensor, text = module.QwenImage21SigmaScheduler().build(**_inputs())
    from comfyui_sigmax.nodes.qwen_image_sigma_scheduler import bind_qwen_image_sigma_output_info

    built = module.build_qwen_image21_sigma_schedule(**_inputs())
    assert text == bind_qwen_image_sigma_output_info(built, output_sigmas=tuple(tensor.tolist()))


@pytest.mark.parametrize("drift", ["none", "order", "choice", "default"])
def test_two_saved_workflows_validate_exact_widget_contract(drift: str) -> None:
    schema = builtin_node_registry().object_info_projection()
    node = schema["Sigmax.QwenImage21SigmaScheduler"]
    node["input_order"] = {
        "required": list(_module().QwenImage21SigmaScheduler.INPUT_TYPES()["required"])
    }
    if drift == "order":
        node["input_order"]["required"].reverse()
    elif drift == "choice":
        node["input"]["required"]["target_source"][0].pop()
    elif drift == "default":
        node["input"]["required"]["steps"][1]["default"] = 50
    if drift == "none":
        assert validate_qwen_image21_workflows(schema)["status"] == "succeeded"
    else:
        with pytest.raises(ScheduleContractError):
            validate_qwen_image21_workflows(schema)
