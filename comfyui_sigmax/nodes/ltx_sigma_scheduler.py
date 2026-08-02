"""Thin explicit LTX SIGMAS node over the dependency-free profile builder."""

from __future__ import annotations

import importlib
import json
import math
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
from comfyui_sigmax.profiles.ltx import (
    LTXProfileId,
    build_ltx_schedule,
    derive_ltx_shift,
)

LTX_SIGMA_NODE_ID: Final = "Sigmax.LTXSigmaScheduler"
LTX_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.ltx-sigma-node/1"
_MAX_STEPS: Final = 10_000
_MAX_TOKENS: Final = 16_777_216
_GENERATIONS: Final = ("LTXV 0.9.8", "LTX-2 19B", "LTX-2.3 22B")
_STAGES: Final = ("Dev", "Distilled Stage 1", "Distilled Stage 2")
_PROFILE_MAP: Final = {
    ("LTXV 0.9.8", "Dev"): LTXProfileId.LTXV_098_DEV,
    ("LTX-2 19B", "Dev"): LTXProfileId.LTX2_19B_DEV,
    ("LTX-2 19B", "Distilled Stage 1"): LTXProfileId.LTX2_19B_DISTILLED_STAGE1,
    ("LTX-2 19B", "Distilled Stage 2"): LTXProfileId.LTX2_19B_DISTILLED_STAGE2,
    ("LTX-2.3 22B", "Dev"): LTXProfileId.LTX23_22B_DEV,
    ("LTX-2.3 22B", "Distilled Stage 1"): LTXProfileId.LTX23_22B_DISTILLED_STAGE1,
    ("LTX-2.3 22B", "Distilled Stage 2"): LTXProfileId.LTX23_22B_DISTILLED_STAGE2,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class LTXSigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    generation: str
    stage: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.generation not in _GENERATIONS or self.stage not in _STAGES:
            raise ScheduleContractError("LTX node generation/stage is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("LTX node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("LTX node result is incomplete")


def _selection(generation: object, stage: object) -> tuple[str, str, LTXProfileId]:
    if not isinstance(generation, str) or not isinstance(stage, str):
        raise ScheduleContractError("generation and stage must be strings")
    profile = _PROFILE_MAP.get((generation, stage))
    if profile is None:
        raise ScheduleContractError("LTX generation/stage selection is unsupported")
    return generation, stage, profile


def _positive_steps(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
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
        raise ScheduleContractError("LTX schedule information must be canonical JSON") from exc


def build_ltx_sigma_schedule(
    *,
    generation: object,
    stage: object,
    steps: object,
    token_count: object,
    stretch: object,
    terminal: object,
    strict_official: object,
    start_step: object,
    end_step: object,
) -> LTXSigmaNodeResult:
    public_generation, public_stage, profile = _selection(generation, stage)
    count = _positive_steps(steps)
    if not isinstance(stretch, bool) or not isinstance(strict_official, bool):
        raise ScheduleContractError("stretch and strict_official must be boolean")
    if isinstance(terminal, bool) or not isinstance(terminal, (int, float)):
        raise ScheduleContractError("terminal must be a finite number")
    terminal_value = float(terminal)
    if not math.isfinite(terminal_value):
        raise ScheduleContractError("terminal must be a finite number")
    selected_stage = (
        "Dev"
        if profile
        in {
            LTXProfileId.LTXV_098_DEV,
            LTXProfileId.LTX2_19B_DEV,
            LTXProfileId.LTX23_22B_DEV,
        }
        else "Distilled"
    )
    if selected_stage == "Dev":
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or not 1 <= token_count <= _MAX_TOKENS
        ):
            raise ScheduleContractError("token_count must be between 1 and 16777216")
        effective_tokens: int | None = token_count
    else:
        if token_count not in (None, 4096):
            raise ScheduleContractError("distilled LTX stages do not accept token_count")
        if not stretch or terminal_value != 0.1 or not strict_official:
            raise ScheduleContractError(
                "distilled LTX stages reject adaptive or non-strict controls"
            )
        effective_tokens = None
    complete = build_ltx_schedule(
        profile=profile,
        steps=count,
        token_count=effective_tokens,
        stretch=stretch,
        terminal=terminal_value,
        strict_official=strict_official,
    )
    start, end = _slice_bounds(start_step=start_step, end_step=end_step, available_steps=count)
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    effective_end = count if end is None else end
    info: dict[str, object] = {
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "generation": public_generation,
        "profile": profile.value,
        "schema": LTX_SIGMA_NODE_SCHEMA_ID,
        "stage": public_stage,
        "stretch": stretch,
        "terminal": terminal_value,
        "slicing": {
            "available_steps": count,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "strict_official": strict_official,
        "token_count": effective_tokens,
        "token_source": "explicit" if selected_stage == "Dev" else "publisher_vector",
        "shift": None if selected_stage != "Dev" else derive_ltx_shift(effective_tokens or 4096),
        "warnings": list(complete.warnings),
    }
    return LTXSigmaNodeResult(
        generation=public_generation,
        stage=public_stage,
        domain=complete.final_domain,
        sigmas=output,
        schedule_info_json=_canonical_info(info),
    )


def bind_ltx_sigma_output_info(
    result: LTXSigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind node metadata to the actual host tensor values."""

    if not isinstance(result, LTXSigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("LTX output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("LTX schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class LTXSigmaScheduler:
    """Construct explicit LTXV/LTX-2/LTX-2.3 external sigma schedules."""

    DESCRIPTION = "Builds explicit LTX adaptive or distilled sigma schedules without model loading."
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "generation": (_GENERATIONS,),
                "stage": (_STAGES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": _MAX_STEPS, "step": 1}),
                "token_count": ("INT", {"default": 4096, "min": 1, "max": _MAX_TOKENS, "step": 1}),
                "stretch": ("BOOLEAN", {"default": True}),
                "terminal": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 0.99, "step": 0.001}),
                "strict_official": ("BOOLEAN", {"default": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS - 1, "step": 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": _MAX_STEPS, "step": 1}),
            }
        }

    def build(
        self,
        generation: object,
        stage: object,
        steps: object,
        token_count: object,
        stretch: object,
        terminal: object,
        strict_official: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        result = build_ltx_sigma_schedule(
            generation=generation,
            stage=stage,
            steps=steps,
            token_count=token_count,
            stretch=stretch,
            terminal=terminal,
            strict_official=strict_official,
            start_step=start_step,
            end_step=end_step,
        )
        torch = importlib.import_module("torch")
        output = torch.tensor(result.sigmas, dtype=torch.float32)
        bound_info = bind_ltx_sigma_output_info(
            result, output_sigmas=tuple(float(item) for item in output.tolist())
        )
        return output, bound_info
