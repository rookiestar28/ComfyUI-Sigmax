"""Thin MiniMax H3 Base video-SIGMAS node."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, replace
from typing import Final

from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    numerical_fingerprint,
    slice_step_range,
    validate_sigma_schedule,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_BASE_FL2VA_PROFILE,
    MINIMAX_H3_BASE_REF2VA_PROFILE,
    MINIMAX_H3_DEFAULT_STEPS,
    MINIMAX_H3_MAX_STEPS,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3Profile,
    MiniMaxH3Variant,
    build_minimax_h3_schedule,
)
from comfyui_sigmax.profiles.minimax_h3_turbo import (
    MiniMaxH3TurboError,
    MiniMaxH3TurboProfile,
    MiniMaxH3TurboReasonCode,
    build_minimax_h3_turbo_schedule,
    get_minimax_h3_turbo_profile,
)
from comfyui_sigmax.profiles.minimax_h3_turbo_public import (
    MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID,
    MINIMAX_H3_TURBO_RECIPE_IDS,
    build_minimax_h3_turbo_public_receipt,
    deserialize_minimax_h3_turbo_public_receipt,
)

MINIMAX_H3_SIGMA_NODE_ID: Final = "Sigmax.MiniMaxH3SigmaScheduler"
MINIMAX_H3_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.minimax-h3-sigma-node/1"
_FL2VA_MODE: Final = "H3 Base FL2VA"
_REF2VA_MODE: Final = "H3 Base Ref2VA"
_MODES: Final = (_FL2VA_MODE, _REF2VA_MODE)
MINIMAX_H3_TURBO_RECIPE_CHOICES: Final = ("disabled", *MINIMAX_H3_TURBO_RECIPE_IDS)


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3SigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    variant: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str
    recipe_id: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in _MODES:
            raise ScheduleContractError("MiniMax H3 node variant is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("MiniMax H3 node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("MiniMax H3 node result is incomplete")


def _variant(value: object) -> tuple[str, MiniMaxH3Variant, MiniMaxH3Profile]:
    if value == _FL2VA_MODE:
        return _FL2VA_MODE, MiniMaxH3Variant.BASE_FL2VA, MINIMAX_H3_BASE_FL2VA_PROFILE
    if value == _REF2VA_MODE:
        return _REF2VA_MODE, MiniMaxH3Variant.BASE_REF2VA, MINIMAX_H3_BASE_REF2VA_PROFILE
    raise ScheduleContractError("MiniMax H3 variant must be selected explicitly")


def _steps(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= MINIMAX_H3_MAX_STEPS
    ):
        raise ScheduleContractError(
            f"steps must be an integer between 1 and {MINIMAX_H3_MAX_STEPS}"
        )
    return value


def _slice_bounds(
    *, start_step: object, end_step: object, available_steps: int
) -> tuple[int, int | None]:
    if not isinstance(start_step, int) or isinstance(start_step, bool) or start_step < 0:
        raise ScheduleContractError("start_step must be a non-negative integer")
    if not isinstance(end_step, int) or isinstance(end_step, bool) or end_step < -1:
        raise ScheduleContractError("end_step must be -1 or a non-negative integer")
    end = None if end_step == -1 else end_step
    if start_step >= available_steps:
        raise ScheduleContractError("start_step must be below the constructed step count")
    if end is not None and (end <= start_step or end > available_steps):
        raise ScheduleContractError(
            "end_step must exceed start_step and not exceed the constructed step count"
        )
    return start_step, end


def _canonical_info(value: dict[str, object]) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError(
            "MiniMax H3 schedule information must be canonical JSON"
        ) from exc


def _turbo_profile(
    *, public_variant: str, recipe_id: object
) -> tuple[str, MiniMaxH3TurboProfile] | None:
    if recipe_id is None:
        return None
    if not isinstance(recipe_id, str):
        raise ScheduleContractError("MiniMax H3 Turbo recipe_id must be selected explicitly")
    if recipe_id in {"", "disabled"}:
        return None
    try:
        profile = get_minimax_h3_turbo_profile(recipe_id)
    except MiniMaxH3TurboError:
        raise
    expected_task = "fl2va" if public_variant == _FL2VA_MODE else "ref2va"
    if profile.task != expected_task:
        raise MiniMaxH3TurboError(
            MiniMaxH3TurboReasonCode.WRONG_TASK,
            "recipe task does not match the selected H3 variant",
        )
    return recipe_id, profile


def _build_base_schedule(
    *,
    public_variant: str,
    selected_variant: MiniMaxH3Variant,
    profile: MiniMaxH3Profile,
    requested_steps: int,
    start_step: object,
    end_step: object,
) -> MiniMaxH3SigmaNodeResult:
    requested_points = requested_steps + 1
    complete = build_minimax_h3_schedule(
        variant=selected_variant,
        grid_points=requested_points,
        precision="float32",
    )
    available_steps = complete.effective_inputs.steps
    start, end = _slice_bounds(
        start_step=start_step,
        end_step=end_step,
        available_steps=available_steps,
    )
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    effective_end = available_steps if end is None else end
    provenance = complete.request.provenance
    info: dict[str, object] = {
        "audio": {
            "derivative": "model_native",
            "ownership": "model_native",
            "shift": MINIMAX_H3_AUDIO_SHIFT,
            "video_mapping": "paired_coordinate_inversion",
        },
        "counts": {
            "effective_grid_points": len(complete.sigmas),
            "effective_model_evaluations": available_steps,
            "effective_steps": available_steps,
            "effective_transitions": available_steps,
            "requested_grid_points": requested_points,
            "requested_model_evaluations": requested_steps,
            "requested_steps": requested_steps,
            "requested_transitions": requested_steps,
        },
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "lane": "diffusers_endpoint_inclusive",
        "license_boundary": "code_only_no_weight_redistribution",
        "profile": {
            "evidence": provenance.evidence.value,
            "id": profile.profile_id,
            "variant": profile.schema.model_variant,
            "version": profile.profile_version,
        },
        "schema": MINIMAX_H3_SIGMA_NODE_SCHEMA_ID,
        "shift": {
            "audio": MINIMAX_H3_AUDIO_SHIFT,
            "kind": "direct_ratio",
            "transform_order": "endpoint_grid_then_video_shift_then_terminal",
            "video": MINIMAX_H3_VIDEO_SHIFT,
        },
        "slicing": {
            "available_steps": available_steps,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "timestep": {
            "clean": 1.0,
            "convention": "t_equals_one_minus_sigma",
            "terminal_sigma": 0.0,
        },
        "velocity": {
            "direction": "data_ward",
            "sign_adapter": "explicit_only",
        },
        "warnings": list(complete.warnings),
    }
    return MiniMaxH3SigmaNodeResult(
        variant=public_variant,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def _build_turbo_schedule(
    *,
    public_variant: str,
    profile: MiniMaxH3TurboProfile,
    recipe_id: str,
    requested_steps: int,
    start_step: object,
    end_step: object,
) -> MiniMaxH3SigmaNodeResult:
    complete = build_minimax_h3_turbo_schedule(
        recipe_id,
        nfe=requested_steps,
        precision="float64",
        task=profile.task,
    )
    available_steps = complete.nfe
    start, end = _slice_bounds(
        start_step=start_step,
        end_step=end_step,
        available_steps=available_steps,
    )
    output = slice_step_range(complete.video_sigmas, start_step=start, end_step=end)
    effective_end = available_steps if end is None else end
    receipt = build_minimax_h3_turbo_public_receipt(
        recipe_id=recipe_id,
        task=profile.task,
        nfe=requested_steps,
        schedule_fingerprint=sigma_output_fingerprint(output, domain=SigmaDomain.UNIT_FLOW),
    )
    info: dict[str, object] = {
        "audio": {
            "derivative": "model_native",
            "ownership": "model_native",
            "shift": profile.audio_shift,
            "video_mapping": "paired_coordinate_inversion",
        },
        "counts": {
            "effective_grid_points": len(complete.video_sigmas),
            "effective_model_evaluations": available_steps,
            "effective_steps": available_steps,
            "effective_transitions": available_steps,
            "requested_grid_points": requested_steps + 1,
            "requested_model_evaluations": requested_steps,
            "requested_steps": requested_steps,
            "requested_transitions": requested_steps,
        },
        "fingerprints": {
            "complete": complete.fingerprint,
            "output": sigma_output_fingerprint(output, domain=SigmaDomain.UNIT_FLOW),
        },
        "lane": "m6_13_recipe_owned_endpoint_inclusive_readiness",
        "license_boundary": "code_only_no_weight_or_lora_redistribution",
        "mode": "turbo_readiness_only",
        "profile": {
            "id": profile.profile_id,
            "recipe_id": recipe_id,
            "version": profile.profile_version,
        },
        "schema": MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID,
        "shift": {
            "audio": profile.audio_shift,
            "kind": "direct_ratio",
            "transform_order": "endpoint_grid_then_video_audio_direct_ratio_then_terminal_zero",
            "video": profile.video_shift,
        },
        "slicing": {
            "available_steps": available_steps,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "timestep": {
            "clean": 1.0,
            "convention": "t_equals_one_minus_sigma",
            "terminal_sigma": 0.0,
        },
        "turbo_receipt": receipt.projection(),
        "warnings": [
            "readiness_only_no_eligible_artifact",
            "model_bound_workflow_requires_caller_verified_artifact",
        ],
    }
    return MiniMaxH3SigmaNodeResult(
        variant=public_variant,
        domain=SigmaDomain.UNIT_FLOW,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
        recipe_id=recipe_id,
    )


def build_minimax_h3_sigma_schedule(
    *,
    variant: object,
    steps: object,
    start_step: object,
    end_step: object,
    recipe_id: object = None,
) -> MiniMaxH3SigmaNodeResult:
    """Build and slice the explicit Diffusers endpoint lane without host imports.

    ``steps`` is the number of sigma transitions.  The source-facing parity builder retains its
    ``grid_points`` vocabulary, so this adapter deliberately maps public ``N`` to ``N + 1``.
    """

    public_variant, selected_variant, base_profile = _variant(variant)
    requested_steps = _steps(steps)
    selected_turbo = _turbo_profile(public_variant=public_variant, recipe_id=recipe_id)
    if selected_turbo is not None:
        selected_recipe, turbo_profile = selected_turbo
        return _build_turbo_schedule(
            public_variant=public_variant,
            profile=turbo_profile,
            recipe_id=selected_recipe,
            requested_steps=requested_steps,
            start_step=start_step,
            end_step=end_step,
        )
    return _build_base_schedule(
        public_variant=public_variant,
        selected_variant=selected_variant,
        profile=base_profile,
        requested_steps=requested_steps,
        start_step=start_step,
        end_step=end_step,
    )


def bind_minimax_h3_sigma_output_info(
    result: MiniMaxH3SigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to the exact values crossing the host tensor boundary."""

    if not isinstance(result, MiniMaxH3SigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("MiniMax H3 output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("MiniMax H3 schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    if result.recipe_id is not None:
        receipt_projection = info.get("turbo_receipt")
        receipt = deserialize_minimax_h3_turbo_public_receipt(
            json.dumps(
                receipt_projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        info["turbo_receipt"] = replace(
            receipt,
            schedule_fingerprint=sigma_output_fingerprint(values, domain=result.domain),
        ).projection()
    return _canonical_info(info)


class MiniMaxH3SigmaScheduler:
    """Construct explicit MiniMax H3 Base video sigmas without model patching."""

    DESCRIPTION = (
        "Builds an explicit MiniMax H3 Base FL2VA or Ref2VA video sigma schedule; "
        "an optional exact Turbo recipe is readiness-only and requires eligible artifact evidence "
        "before any model workflow can execute."
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
                "variant": (_MODES,),
                "steps": (
                    "INT",
                    {
                        "default": MINIMAX_H3_DEFAULT_STEPS,
                        "min": 1,
                        "max": MINIMAX_H3_MAX_STEPS,
                        "step": 1,
                    },
                ),
                "start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": MINIMAX_H3_MAX_STEPS - 1},
                ),
                "end_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": MINIMAX_H3_MAX_STEPS},
                ),
            },
            "optional": {
                "recipe_id": (
                    MINIMAX_H3_TURBO_RECIPE_CHOICES,
                    {
                        "default": "disabled",
                        "tooltip": "Experimental exact M6-13 recipe; readiness-only until artifact evidence is eligible.",
                    },
                )
            },
        }

    def build(
        self,
        variant: object,
        steps: object,
        start_step: object,
        end_step: object,
        recipe_id: object = None,
    ) -> tuple[object, str]:
        result = build_minimax_h3_sigma_schedule(
            variant=variant,
            steps=steps,
            start_step=start_step,
            end_step=end_step,
            recipe_id=recipe_id,
        )
        try:
            torch = importlib.import_module("torch")
            tensor = torch.__dict__["tensor"]
            float32 = torch.__dict__["float32"]
        except (ImportError, KeyError) as exc:
            # CRITICAL: keep Torch execution-only; package imports remain dependency-free.
            raise RuntimeError(
                "ComfyUI host execution requires Torch float32 tensor support"
            ) from exc
        tensor = tensor(result.sigmas, dtype=float32)
        try:
            host_values = tuple(float(value) for value in tensor.tolist())
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("ComfyUI host tensor must expose numeric tolist output") from exc
        return tensor, bind_minimax_h3_sigma_output_info(result, output_sigmas=host_values)


__all__ = [
    "MINIMAX_H3_SIGMA_NODE_ID",
    "MINIMAX_H3_SIGMA_NODE_SCHEMA_ID",
    "MINIMAX_H3_TURBO_RECIPE_CHOICES",
    "MiniMaxH3SigmaNodeResult",
    "MiniMaxH3SigmaScheduler",
    "bind_minimax_h3_sigma_output_info",
    "build_minimax_h3_sigma_schedule",
]
