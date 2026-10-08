"""Thin explicit Qwen Image 2.1 external-sigma adapter; no model patching."""

from __future__ import annotations

import importlib
import json
from typing import Final

from comfyui_sigmax.core import ScheduleContractError, numerical_fingerprint
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint
from comfyui_sigmax.nodes.qwen_image_sigma_scheduler import (
    QwenImageSigmaNodeResult,
    _positive_steps,
    _slice_bounds,
    bind_qwen_image_sigma_output_info,
)
from comfyui_sigmax.profiles.qwen_native import (
    QwenNativeLane,
    QwenNativeRequest,
    build_qwen_native_schedule,
)
from comfyui_sigmax.profiles.schema_v1 import profile_schema_fingerprint

QWEN_IMAGE21_SIGMA_NODE_ID: Final = "Sigmax.QwenImage21SigmaScheduler"
QWEN_IMAGE21_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.qwen-image21-sigma-node/1"


def _control(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 1_000_000:
        raise ScheduleContractError(f"{name} must be an integer in 0..1000000")
    return value


def build_qwen_image21_sigma_schedule(
    *,
    mode: object,
    steps: object,
    target_source: object,
    width: object,
    height: object,
    image_seq_len: object,
    strict_official: object,
    start_step: object,
    end_step: object,
) -> QwenImageSigmaNodeResult:
    if mode not in ("Comfy Native", "Diffusers Dynamic"):
        raise ScheduleContractError("Qwen 2.1 mode must be explicitly native or dynamic")
    count = _positive_steps(steps)
    w, h, tokens = (
        _control(width, "width"),
        _control(height, "height"),
        _control(image_seq_len, "image_seq_len"),
    )
    start, end = _slice_bounds(start_step=start_step, end_step=end_step, available_steps=count)
    if not isinstance(strict_official, bool):
        raise ScheduleContractError("strict_official must be boolean")
    dynamic = mode == "Diffusers Dynamic"
    lane = QwenNativeLane.QWEN21_DYNAMIC if dynamic else QwenNativeLane.QWEN21_COMFY
    actual_width: int | None = None
    actual_height: int | None = None
    actual_tokens: int | None = None
    # CRITICAL: zero means inactive, not an inferred target. Reject competing/nonzero inactive
    # controls; original packed-token geometry or reference-prefix tokens changes the 2.1 shift.
    if not dynamic:
        if target_source != "None (Native)" or any((w, h, tokens)):
            raise ScheduleContractError(
                "native 2.1 requires None (Native) and zero geometry/tokens"
            )
    elif target_source == "Dimensions":
        if w == 0 or h == 0 or tokens != 0:
            raise ScheduleContractError("Dimensions requires positive width/height and zero tokens")
        actual_width, actual_height = w, h
    elif target_source == "Target Tokens":
        if tokens == 0 or w != 0 or h != 0:
            raise ScheduleContractError(
                "Target Tokens requires positive tokens and zero dimensions"
            )
        actual_tokens = tokens
    else:
        raise ScheduleContractError("dynamic 2.1 requires explicit Dimensions or Target Tokens")
    built = build_qwen_native_schedule(
        QwenNativeRequest(
            lane=lane,
            steps=count,
            width=actual_width,
            height=actual_height,
            image_seq_len=actual_tokens,
            strict_official=strict_official,
            start_step=start,
            end_step=end,
        )
    )
    complete, profile = built.schedule, built.profile.schema
    shift: dict[str, object] = {"dynamic": dynamic, "kind": "exponential_mu", "mu": built.mu}
    if dynamic:
        shift.update(
            {
                "base_image_seq_len": 256,
                "base_shift": 0.5,
                "max_image_seq_len": 8192,
                "max_shift": 0.9,
                "extrapolation": "unclamped_affine",
                "shift_terminal": 0.02,
            }
        )
    else:
        shift["training_timesteps"] = 10000
    info = {
        "schema": QWEN_IMAGE21_SIGMA_NODE_SCHEMA_ID,
        "profile": {
            "id": profile.profile_id,
            "version": profile.profile_version,
            "variant": "v2_1",
            "evidence": complete.request.provenance.evidence.value,
            "fingerprint": profile_schema_fingerprint(profile),
        },
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(built.output_sigmas, domain=complete.final_domain),
        },
        "shift": shift,
        "target": {
            "source": target_source,
            "width": actual_width,
            "height": actual_height,
            "image_seq_len": built.image_seq_len,
            "packing": "unpatched_target_only_vae16_aligned32" if dynamic else None,
        },
        "terminal": {"append_zero": True, "stretch_to": 0.02 if dynamic else None},
        "transform_order": [stage.identifier for stage in profile.transforms] + ["slice_last"],
        "slicing": {
            "available_steps": count,
            "output_steps": len(built.output_sigmas) - 1,
            "start_step": start,
            "end_step": count if end is None else end,
        },
        "guidance": {"host_true_cfg": 1.0},
        "ownership": "external_sigmas",
        "source": {
            "url": complete.request.provenance.source,
            "revision": complete.request.provenance.source_revision,
        },
        "license": {
            "identifier": profile.model_weights[0].license.identifier,
            "url": profile.model_weights[0].license.url,
        },
        "strict_official": strict_official,
        "warnings": list(complete.warnings),
    }
    return QwenImageSigmaNodeResult(
        mode=str(mode),
        domain=complete.final_domain,
        sigmas=built.output_sigmas,
        schedule_info_json=json.dumps(info, allow_nan=False, sort_keys=True, separators=(",", ":")),
    )


class QwenImage21SigmaScheduler:
    DESCRIPTION = (
        "Qwen Image 2.1 external sigmas: native 25-step or explicit dynamic 40-step reference."
    )
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False
    EXPERIMENTAL = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "mode": (("Comfy Native", "Diffusers Dynamic"),),
                "steps": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Native baseline 25; dynamic baseline 40; strict rejects other counts.",
                    },
                ),
                "target_source": (("None (Native)", "Dimensions", "Target Tokens"),),
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Dimensions only: positive multiple of 32. Otherwise keep zero.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Dimensions only: positive multiple of 32. Otherwise keep zero.",
                    },
                ),
                "image_seq_len": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1000000,
                        "tooltip": "Target Tokens only: target latent tokens without packing or reference images. Otherwise zero.",
                    },
                ),
                "strict_official": ("BOOLEAN", {"default": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000}),
            }
        }

    def build(
        self,
        mode: object,
        steps: object,
        target_source: object,
        width: object,
        height: object,
        image_seq_len: object,
        strict_official: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        result = build_qwen_image21_sigma_schedule(
            mode=mode,
            steps=steps,
            target_source=target_source,
            width=width,
            height=height,
            image_seq_len=image_seq_len,
            strict_official=strict_official,
            start_step=start_step,
            end_step=end_step,
        )
        try:
            # CRITICAL: keep Torch execution-only so package/core imports stay dependency-free.
            tensor = importlib.import_module("torch").__dict__["FloatTensor"](result.sigmas)
            values = tuple(float(value) for value in tensor.tolist())
        except (ImportError, KeyError, AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "ComfyUI host execution requires numeric Torch FloatTensor output"
            ) from exc
        return tensor, bind_qwen_image_sigma_output_info(result, output_sigmas=values)


__all__ = [
    "QWEN_IMAGE21_SIGMA_NODE_ID",
    "QWEN_IMAGE21_SIGMA_NODE_SCHEMA_ID",
    "QwenImage21SigmaScheduler",
    "build_qwen_image21_sigma_schedule",
]
