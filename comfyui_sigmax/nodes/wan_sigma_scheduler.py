"""Thin explicit Wan SIGMAS node with caller-owned A14B boundary metadata."""

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
from comfyui_sigmax.profiles.wan import (
    WAN21_COMFY_NATIVE_PROFILE,
    WAN21_I2V_480P_DIFFUSERS_PROFILE,
    WAN21_I2V_480P_OFFICIAL_PROFILE,
    WAN21_I2V_720P_DIFFUSERS_PROFILE,
    WAN21_I2V_720P_OFFICIAL_PROFILE,
    WAN21_T2V_DIFFUSERS_PROFILE,
    WAN21_T2V_OFFICIAL_PROFILE,
    WAN22_I2V_A14B_DIFFUSERS_PROFILE,
    WAN22_I2V_A14B_NATIVE_PROFILE,
    WAN22_T2V_A14B_DIFFUSERS_PROFILE,
    WAN22_T2V_A14B_NATIVE_PROFILE,
    WAN22_TI2V_5B_DIFFUSERS_PROFILE,
    WAN22_TI2V_5B_NATIVE_PROFILE,
    WanProfile,
    WanProfileId,
    WanResolution,
    WanScheduleResult,
    build_wan_schedule,
)

WAN_SIGMA_NODE_ID: Final = "Sigmax.WanSigmaScheduler"
WAN_SIGMA_NODE_SCHEMA_ID: Final = "sigmax.wan-sigma-node/1"
_MAX_STEPS: Final = 10_000
_GENERATIONS: Final = ("Wan 2.1", "Wan 2.2")
_TASKS: Final = ("T2V", "I2V", "TI2V", "T2V A14B", "I2V A14B")
_SOURCES: Final = ("ComfyUI native", "Official native", "Diffusers reference")
_RESOLUTIONS: Final = ("None", "480P", "720P")

_PROFILES: dict[tuple[str, str, str, str], tuple[WanProfileId, WanResolution]] = {
    ("Wan 2.1", "T2V", "ComfyUI native", "None"): (
        WanProfileId.WAN21_COMFY_NATIVE,
        WanResolution.NONE,
    ),
    ("Wan 2.1", "T2V", "Official native", "None"): (
        WanProfileId.WAN21_T2V_OFFICIAL,
        WanResolution.NONE,
    ),
    ("Wan 2.1", "T2V", "Diffusers reference", "None"): (
        WanProfileId.WAN21_T2V_DIFFUSERS,
        WanResolution.NONE,
    ),
    ("Wan 2.1", "I2V", "Official native", "480P"): (
        WanProfileId.WAN21_I2V_480P_OFFICIAL,
        WanResolution.P480,
    ),
    ("Wan 2.1", "I2V", "Official native", "720P"): (
        WanProfileId.WAN21_I2V_720P_OFFICIAL,
        WanResolution.P720,
    ),
    ("Wan 2.1", "I2V", "Diffusers reference", "480P"): (
        WanProfileId.WAN21_I2V_480P_DIFFUSERS,
        WanResolution.P480,
    ),
    ("Wan 2.1", "I2V", "Diffusers reference", "720P"): (
        WanProfileId.WAN21_I2V_720P_DIFFUSERS,
        WanResolution.P720,
    ),
    ("Wan 2.2", "TI2V", "ComfyUI native", "None"): (
        WanProfileId.WAN22_TI2V_5B_NATIVE,
        WanResolution.NONE,
    ),
    ("Wan 2.2", "TI2V", "Diffusers reference", "None"): (
        WanProfileId.WAN22_TI2V_5B_DIFFUSERS,
        WanResolution.NONE,
    ),
    ("Wan 2.2", "T2V A14B", "Official native", "None"): (
        WanProfileId.WAN22_T2V_A14B_NATIVE,
        WanResolution.NONE,
    ),
    ("Wan 2.2", "T2V A14B", "Diffusers reference", "None"): (
        WanProfileId.WAN22_T2V_A14B_DIFFUSERS,
        WanResolution.NONE,
    ),
    ("Wan 2.2", "I2V A14B", "Official native", "None"): (
        WanProfileId.WAN22_I2V_A14B_NATIVE,
        WanResolution.NONE,
    ),
    ("Wan 2.2", "I2V A14B", "Diffusers reference", "None"): (
        WanProfileId.WAN22_I2V_A14B_DIFFUSERS,
        WanResolution.NONE,
    ),
}

_PROFILE_OBJECTS: dict[WanProfileId, WanProfile] = {
    WanProfileId.WAN21_COMFY_NATIVE: WAN21_COMFY_NATIVE_PROFILE,
    WanProfileId.WAN21_T2V_OFFICIAL: WAN21_T2V_OFFICIAL_PROFILE,
    WanProfileId.WAN21_I2V_480P_OFFICIAL: WAN21_I2V_480P_OFFICIAL_PROFILE,
    WanProfileId.WAN21_I2V_720P_OFFICIAL: WAN21_I2V_720P_OFFICIAL_PROFILE,
    WanProfileId.WAN21_T2V_DIFFUSERS: WAN21_T2V_DIFFUSERS_PROFILE,
    WanProfileId.WAN21_I2V_480P_DIFFUSERS: WAN21_I2V_480P_DIFFUSERS_PROFILE,
    WanProfileId.WAN21_I2V_720P_DIFFUSERS: WAN21_I2V_720P_DIFFUSERS_PROFILE,
    WanProfileId.WAN22_TI2V_5B_NATIVE: WAN22_TI2V_5B_NATIVE_PROFILE,
    WanProfileId.WAN22_T2V_A14B_NATIVE: WAN22_T2V_A14B_NATIVE_PROFILE,
    WanProfileId.WAN22_I2V_A14B_NATIVE: WAN22_I2V_A14B_NATIVE_PROFILE,
    WanProfileId.WAN22_TI2V_5B_DIFFUSERS: WAN22_TI2V_5B_DIFFUSERS_PROFILE,
    WanProfileId.WAN22_T2V_A14B_DIFFUSERS: WAN22_T2V_A14B_DIFFUSERS_PROFILE,
    WanProfileId.WAN22_I2V_A14B_DIFFUSERS: WAN22_I2V_A14B_DIFFUSERS_PROFILE,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class WanSigmaNodeResult:
    """Pure node output before conversion to the host tensor type."""

    generation: str
    task: str
    source: str
    resolution: str
    domain: SigmaDomain
    sigmas: tuple[float, ...]
    boundary_step: int
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.generation not in _GENERATIONS or self.task not in _TASKS:
            raise ScheduleContractError("Wan node generation/task is unsupported")
        if self.source not in _SOURCES or self.resolution not in _RESOLUTIONS:
            raise ScheduleContractError("Wan node source/resolution is unsupported")
        if self.domain is not SigmaDomain.UNIT_FLOW:
            raise ScheduleContractError("Wan node requires UNIT_FLOW")
        if len(self.sigmas) < 2 or not self.schedule_info_json:
            raise ScheduleContractError("Wan node result is incomplete")
        if not isinstance(self.boundary_step, int) or self.boundary_step < -1:
            raise ScheduleContractError("Wan node boundary_step is invalid")


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
        raise ScheduleContractError("Wan schedule information must be canonical JSON") from exc


def _resolve_selection(
    *, generation: object, task: object, source: object, resolution: object
) -> tuple[str, str, str, str, WanProfileId, WanResolution]:
    values = (generation, task, source, resolution)
    if not all(isinstance(value, str) for value in values):
        raise ScheduleContractError("Wan generation, task, source, and resolution must be strings")
    key = (generation, task, source, resolution)
    selection = _PROFILES.get(key)
    if selection is None:
        raise ScheduleContractError(
            "Wan profile selection is unsupported; choose an explicit released generation/task/source/resolution"
        )
    profile, profile_resolution = selection
    return generation, task, source, resolution, profile, profile_resolution


def build_wan_sigma_schedule(
    *,
    generation: object,
    task: object,
    source: object,
    resolution: object,
    steps: object,
    strict_source: object,
    start_step: object,
    end_step: object,
    already_shifted: object = False,
) -> WanSigmaNodeResult:
    """Build and slice a Wan schedule without importing ComfyUI or dispatching experts."""

    (
        public_generation,
        public_task,
        public_source,
        public_resolution,
        profile_id,
        profile_resolution,
    ) = _resolve_selection(generation=generation, task=task, source=source, resolution=resolution)
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= _MAX_STEPS:
        raise ScheduleContractError(f"steps must be an integer between 1 and {_MAX_STEPS}")
    if not isinstance(strict_source, bool) or not isinstance(already_shifted, bool):
        raise ScheduleContractError("strict_source and already_shifted must be boolean")
    if not isinstance(profile_resolution, WanResolution):
        raise ScheduleContractError("Wan profile resolution is malformed")
    complete: WanScheduleResult = build_wan_schedule(
        profile=profile_id,
        steps=steps,
        resolution=profile_resolution,
        strict_source=strict_source,
        already_shifted=already_shifted,
    )
    start, end = _slice_bounds(start_step=start_step, end_step=end_step, available_steps=steps)
    output = slice_step_range(complete.sigmas, start_step=start, end_step=end)
    profile = _PROFILE_OBJECTS[profile_id]
    evidence = complete.request.provenance.evidence
    boundary_step = -1 if complete.boundary is None else complete.boundary.transition_index
    effective_end = steps if end is None else end
    info_boundary: dict[str, object] = {
        "model_dispatch": False,
        "routing_owner": "caller",
        "step": boundary_step,
    }
    if complete.boundary is not None:
        info_boundary.update(
            {
                "crossing": complete.boundary.crossing,
                "normalized": complete.boundary.normalized,
            }
        )
    guidance_info: dict[str, object] = {
        "host_cfg_scale": profile.schema.recipes[0].guidance.host_value,
        "model_cfg_scale": profile.schema.recipes[0].guidance.model_value,
    }
    for guidance_name in ("cfg_low", "cfg_high"):
        guidance_field = next(
            (field for field in profile.schema.parameters if field.name == guidance_name),
            None,
        )
        if guidance_field is not None:
            guidance_info[guidance_name] = guidance_field.value
    info: dict[str, object] = {
        "boundary": info_boundary,
        "fingerprints": {
            "complete": numerical_fingerprint(
                complete.sigmas, domain=complete.final_domain, precision="float64"
            ),
            "output": sigma_output_fingerprint(output, domain=complete.final_domain),
        },
        "guidance": guidance_info,
        "profile": {
            "evidence": evidence.value,
            "id": profile.profile_id,
            "recipe": profile.profile_id
            if evidence is not None and evidence.value != "modified"
            else f"{profile.profile_id}.modified-{steps}",
            "revision": complete.request.provenance.source_revision,
            "version": profile.profile_version,
        },
        "schema": WAN_SIGMA_NODE_SCHEMA_ID,
        "shift": {
            "kind": "direct_ratio",
            "multiplier": 1.0,
            "ratio": next(
                field.value for field in profile.schema.parameters if field.name == "shift"
            ),
        },
        "slicing": {
            "available_steps": steps,
            "end_step": effective_end,
            "output_steps": len(output) - 1,
            "start_step": start,
        },
        "solver_ownership": profile.schema.parameters[
            next(
                index
                for index, field in enumerate(profile.schema.parameters)
                if field.name == "solver"
            )
        ].value,
        "strict_source": strict_source,
        "warnings": list(complete.warnings),
    }
    return WanSigmaNodeResult(
        generation=public_generation,
        task=public_task,
        source=public_source,
        resolution=public_resolution,
        domain=complete.final_domain,
        sigmas=output,
        boundary_step=boundary_step,
        schedule_info_json=_canonical_info(info),
    )


def bind_wan_sigma_output_info(
    result: WanSigmaNodeResult, *, output_sigmas: tuple[float, ...]
) -> str:
    """Bind metadata to actual host tensor values."""

    if not isinstance(result, WanSigmaNodeResult) or len(output_sigmas) != len(result.sigmas):
        raise ScheduleContractError("Wan output binding is inconsistent")
    values = validate_sigma_schedule(
        output_sigmas,
        domain=result.domain,
        expected_steps=len(output_sigmas) - 1,
        require_terminal_zero=False,
    )
    info = json.loads(result.schedule_info_json)
    if not isinstance(info, dict) or not isinstance(info.get("fingerprints"), dict):
        raise ScheduleContractError("Wan schedule information is malformed")
    info["fingerprints"]["output"] = sigma_output_fingerprint(values, domain=result.domain)
    return _canonical_info(info)


class WanSigmaScheduler:
    """Construct explicit Wan external sigmas; boundary output never routes models."""

    DESCRIPTION = (
        "Builds an explicit Wan 2.1/2.2 sigma schedule and caller-owned boundary metadata."
    )
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "build"
    RETURN_TYPES = ("SIGMAS", "INT", "STRING")
    RETURN_NAMES = ("sigmas", "boundary_step", "schedule_info")
    OUTPUT_NODE = False
    EXPERIMENTAL = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "generation": (_GENERATIONS,),
                "task": (_TASKS,),
                "source": (_SOURCES,),
                "resolution": (_RESOLUTIONS,),
                "steps": ("INT", {"default": 50, "min": 1, "max": _MAX_STEPS, "step": 1}),
                "strict_source": ("BOOLEAN", {"default": False}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS - 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": _MAX_STEPS}),
                "already_shifted": ("BOOLEAN", {"default": False}),
            }
        }

    def build(
        self,
        generation: object,
        task: object,
        source: object,
        resolution: object,
        steps: object,
        strict_source: object,
        start_step: object,
        end_step: object,
        already_shifted: object = False,
    ) -> tuple[object, int, str]:
        result = build_wan_sigma_schedule(
            generation=generation,
            task=task,
            source=source,
            resolution=resolution,
            steps=steps,
            strict_source=strict_source,
            start_step=start_step,
            end_step=end_step,
            already_shifted=already_shifted,
        )
        try:
            torch = importlib.import_module("torch")
            float_tensor = torch.__dict__["FloatTensor"]
        except (ImportError, KeyError) as exc:
            # CRITICAL: keep Torch execution-only; package imports remain dependency-free.
            raise RuntimeError("ComfyUI host execution requires Torch FloatTensor support") from exc
        tensor = float_tensor(result.sigmas)
        try:
            host_values = tuple(float(value) for value in tensor.tolist())
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("ComfyUI host tensor must expose numeric tolist output") from exc
        return (
            tensor,
            result.boundary_step,
            bind_wan_sigma_output_info(result, output_sigmas=host_values),
        )


__all__ = [
    "WAN_SIGMA_NODE_ID",
    "WAN_SIGMA_NODE_SCHEMA_ID",
    "WanSigmaNodeResult",
    "WanSigmaScheduler",
    "bind_wan_sigma_output_info",
    "build_wan_sigma_schedule",
]
