"""RM6 profile migration regressions; pure source math, no host/model imports."""

from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest


def _module() -> Any:
    return importlib.import_module("comfyui_sigmax.profiles.qwen_native")


def test_original_native_uses_exponential_mu_not_legacy_ratio() -> None:
    module = _module()
    request = module.QwenNativeRequest(lane=module.QwenNativeLane.ORIGINAL_COMFY, steps=50)
    result = module.build_qwen_native_schedule(request)
    from comfyui_sigmax.profiles.qwen_image import QwenImageShiftMode, build_qwen_image_schedule

    legacy = build_qwen_image_schedule(
        mode=QwenImageShiftMode.COMFY_FIXED, steps=50, image_seq_len=None
    )
    assert result.schedule.sigmas[25] == pytest.approx(0.759510916949111, abs=1e-14)
    assert result.schedule.sigmas[25] != pytest.approx(legacy.sigmas[25], abs=0.1)
    assert result.mu == 1.15
    assert result.profile.schema.profile_id != legacy.request.provenance.profile_id


def test_qwen21_dynamic_shift_has_distinct_endpoints_and_terminal_order() -> None:
    module = _module()
    request = module.QwenNativeRequest(
        lane=module.QwenNativeLane.QWEN21_DYNAMIC, steps=40, width=1024, height=1024
    )
    result = module.build_qwen_native_schedule(request)
    assert result.image_seq_len == 4096
    assert result.mu == pytest.approx(0.6935483870967742)
    assert result.schedule.sigmas[-2] == pytest.approx(0.02, abs=1e-14)
    assert result.schedule.sigmas[-1] == 0.0
    assert len(result.schedule.sigmas) == 41


@pytest.mark.parametrize(
    "width,height,tokens",
    [(1024, 1024, 4096), (1536, 1024, 6144), (256, 256, 256), (2048, 1024, 8192)],
)
def test_target_only_unpatched_geometry_and_explicit_tokens_are_equivalent(
    width: int, height: int, tokens: int
) -> None:
    module = _module()
    lane = module.QwenNativeLane.QWEN21_DYNAMIC
    geometry = module.build_qwen_native_schedule(
        module.QwenNativeRequest(lane=lane, steps=40, width=width, height=height)
    )
    explicit = module.build_qwen_native_schedule(
        module.QwenNativeRequest(lane=lane, steps=40, image_seq_len=tokens)
    )
    assert geometry.image_seq_len == tokens
    assert geometry.schedule.sigmas == explicit.schedule.sigmas
    assert module.calculate_qwen21_mu(256) == 0.5
    assert module.calculate_qwen21_mu(8192) == 0.9
    assert module.calculate_qwen21_mu(9216) > 0.9


@pytest.mark.parametrize(
    "fields",
    [
        {"steps": 1},
        {"steps": True},
        {"steps": 10001},
        {"width": 1024},
        {"width": 1025, "height": 1024},
        {"width": False, "height": 1024},
        {"width": 16032, "height": 16032},
        {"image_seq_len": 0},
        {"image_seq_len": True},
        {"image_seq_len": 1000001},
        {"width": 1024, "height": 1024, "image_seq_len": 4096},
        {"strict_official": "true"},
        {"steps": 25, "strict_official": True},
        {"start_step": True},
        {"start_step": 40},
        {"end_step": 0},
        {"end_step": 41},
    ],
)
def test_invalid_dynamic_requests_fail_closed(fields: dict[str, object]) -> None:
    module = _module()
    arguments: dict[str, object] = {"lane": module.QwenNativeLane.QWEN21_DYNAMIC, "steps": 40}
    if "width" not in fields and "height" not in fields:
        arguments["image_seq_len"] = 4096
    arguments.update(fields)
    with pytest.raises(ValueError):
        module.QwenNativeRequest(**arguments)


def test_native_lane_rejects_inactive_controls_and_ambiguous_lanes() -> None:
    module = _module()
    with pytest.raises(ValueError, match="typed"):
        module.QwenNativeRequest(lane="qwen_image21", steps=25)
    with pytest.raises(ValueError, match="geometry"):
        module.QwenNativeRequest(
            lane=module.QwenNativeLane.QWEN21_COMFY, steps=25, image_seq_len=4096
        )
    with pytest.raises(ValueError, match="authority"):
        module.QwenNativeRequest(lane=module.QwenNativeLane.QWEN21_DYNAMIC, steps=40)
    with pytest.raises(ValueError, match="typed request"):
        module.build_qwen_native_schedule(None)


def test_profile_license_shift_order_and_immutable_boundaries() -> None:
    module = _module()
    native = module.QWEN21_COMFY_NATIVE_PROFILE
    dynamic = module.QWEN21_DYNAMIC_PROFILE
    assert native.schema.model_family == dynamic.schema.model_family == "qwen_image21"
    assert native.schema.model_weights[0].license.identifier == "LicenseRef-Qwen-Research"
    assert [transform.identifier for transform in dynamic.schema.transforms] == [
        "qwen.exponential_mu",
        "qwen21.stretch_to_terminal",
        "terminal.append_zero",
    ]
    assert dynamic.schema.recipes[0].guidance.host_value == 1.0
    assert native.schema.recipes[0].steps.default == 25
    assert dynamic.schema.recipes[0].steps.default == 40
    with pytest.raises(FrozenInstanceError):
        native.lane = module.QwenNativeLane.ORIGINAL_COMFY
    with pytest.raises(ValueError, match="identity"):
        replace(native, lane=module.QwenNativeLane.ORIGINAL_COMFY)


def test_slicing_follows_complete_schedule_and_modified_recipe_is_truthful() -> None:
    module = _module()
    request = module.QwenNativeRequest(
        lane=module.QwenNativeLane.QWEN21_DYNAMIC,
        steps=27,
        image_seq_len=4096,
        start_step=2,
        end_step=20,
    )
    result = module.build_qwen_native_schedule(request)
    assert result.output_sigmas == result.schedule.sigmas[2:21]
    assert result.schedule.sigmas[-1] == 0.0
    assert result.output_sigmas[-1] > 0.02
    assert result.schedule.request.provenance.evidence.value == "modified"
    assert result.schedule.warnings
