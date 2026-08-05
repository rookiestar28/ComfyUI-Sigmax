"""Thin MiniMax H3 Base video-SIGMAS node."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
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
    MINIMAX_H3_DEFAULT_GRID_POINTS,
    MINIMAX_H3_MAX_GRID_POINTS,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3Profile,
    MiniMaxH3Variant,
    build_minimax_h3_schedule,
)

MINIMAX_H3_SIGMA_NODE_ID: Final = "Sigmax.MiniMaxH3SigmaScheduler"
MINIMAX_H3_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.minimax-h3-sigma-node/1"
_FL2VA_MODE: Final = "H3 Base FL2VA"
_REF2VA_MODE: Final = "H3 Base Ref2VA"
_MODES: Final = (_FL2VA_MODE, _REF2VA_MODE)


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3SigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    variant: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

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


def _grid_points(value: object) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 2 <= value <= MINIMAX_H3_MAX_GRID_POINTS
    ):
        raise ScheduleContractError(
            f"grid_points must be an integer between 2 and {MINIMAX_H3_MAX_GRID_POINTS}"
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


def build_minimax_h3_sigma_schedule(
    *,
    variant: object,
    grid_points: object,
    start_step: object,
    end_step: object,
    already_shifted: object = False,
) -> MiniMaxH3SigmaNodeResult:
    """Build and slice the explicit Diffusers endpoint lane without host imports."""

    public_variant, selected_variant, profile = _variant(variant)
    requested_points = _grid_points(grid_points)
    if not isinstance(already_shifted, bool):
        raise ScheduleContractError("already_shifted must be boolean")
    complete = build_minimax_h3_schedule(
        variant=selected_variant,
        grid_points=requested_points,
        precision="float32",
        already_shifted=already_shifted,
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
            "effective_transitions": available_steps,
            "requested_grid_points": requested_points,
            "requested_transitions": requested_points - 1,
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
    return _canonical_info(info)


class MiniMaxH3SigmaScheduler:
    """Construct explicit MiniMax H3 Base video sigmas without model patching."""

    DESCRIPTION = (
        "Builds an explicit MiniMax H3 Base FL2VA or Ref2VA video sigma schedule; "
        "audio mapping remains model-owned."
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
                "grid_points": (
                    "INT",
                    {
                        "default": MINIMAX_H3_DEFAULT_GRID_POINTS,
                        "min": 2,
                        "max": MINIMAX_H3_MAX_GRID_POINTS,
                        "step": 1,
                    },
                ),
                "start_step": (
                    "INT",
                    {"default": 0, "min": 0, "max": MINIMAX_H3_MAX_GRID_POINTS - 2},
                ),
                "end_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": MINIMAX_H3_MAX_GRID_POINTS - 1},
                ),
            }
        }

    def build(
        self,
        variant: object,
        grid_points: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        result = build_minimax_h3_sigma_schedule(
            variant=variant,
            grid_points=grid_points,
            start_step=start_step,
            end_step=end_step,
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
    "MiniMaxH3SigmaNodeResult",
    "MiniMaxH3SigmaScheduler",
    "bind_minimax_h3_sigma_output_info",
    "build_minimax_h3_sigma_schedule",
]
