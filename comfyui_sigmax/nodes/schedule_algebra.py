"""Validated schedule algebra over fingerprint-bound Sigmax SIGMAS."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Final

from comfyui_sigmax.core import (
    EvidenceLevel,
    ScheduleContractError,
    SigmaDomain,
    validate_sigma_schedule,
)
from comfyui_sigmax.nodes.inspectors import _host_sigma_tuple, _verified_schedule, _VerifiedSchedule
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint

SCHEDULE_SLICE_NODE_ID: Final = "Sigmax.ScheduleSlice"
SCHEDULE_SLICE_SCHEMA_ID: Final = "sigmax.schedule-slice-node/1"
SCHEDULE_CONCATENATE_NODE_ID: Final = "Sigmax.ScheduleConcatenate"
SCHEDULE_CONCATENATE_SCHEMA_ID: Final = "sigmax.schedule-concatenate-node/1"
SCHEDULE_RESAMPLE_NODE_ID: Final = "Sigmax.ScheduleResample"
SCHEDULE_RESAMPLE_SCHEMA_ID: Final = "sigmax.schedule-resample-node/1"

_MAX_STEPS: Final = 10_000
_SOURCE_ID: Final = "sigmax.schedule-algebra"
_SUPPORTED_DOMAINS: Final = frozenset({SigmaDomain.UNIT_FLOW})


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleAlgebraNodeResult:
    """Pure schedule-algebra output before host tensor conversion."""

    domain: SigmaDomain
    sigmas: tuple[float, ...]
    schedule_info_json: str

    def __post_init__(self) -> None:
        if self.domain not in _SUPPORTED_DOMAINS:
            raise ScheduleContractError("schedule algebra domain is unsupported")
        if not isinstance(self.sigmas, tuple) or len(self.sigmas) < 2:
            raise ScheduleContractError("schedule algebra requires at least one transition")
        validate_sigma_schedule(
            self.sigmas,
            domain=self.domain,
            expected_steps=len(self.sigmas) - 1,
            require_terminal_zero=False,
        )
        if not isinstance(self.schedule_info_json, str) or not self.schedule_info_json:
            raise ScheduleContractError("schedule algebra information is required")


def _canonical_json(value: dict[str, object]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("schedule algebra information is not canonical JSON") from exc


def _step(value: object, *, label: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise ScheduleContractError(f"{label} must be an integer between 0 and {maximum}")
    return value


def _positive_steps(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAX_STEPS:
        raise ScheduleContractError(f"output_steps must be an integer between 1 and {_MAX_STEPS}")
    return value


def _source_identity(verified: _VerifiedSchedule) -> dict[str, object]:
    return {
        "domain": verified.domain.value,
        "fingerprint": verified.fingerprints["computed_output"],
        "schema": verified.source_schema,
    }


def _projection(
    *,
    schema: str,
    operation: str,
    domain: SigmaDomain,
    output: tuple[float, ...],
    sources: list[dict[str, object]],
    parameters: dict[str, object],
) -> str:
    fingerprint = sigma_output_fingerprint(output, domain=domain)
    projection: dict[str, object] = {
        "base_grid": None,
        "domain": {"sigma": domain.value, "time": domain.value},
        "evidence": EvidenceLevel.MODIFIED.value,
        "fingerprints": {"complete": fingerprint, "output": fingerprint},
        "operation": operation,
        "parameters": parameters,
        "provenance": {
            "evidence": EvidenceLevel.MODIFIED.value,
            "profile_id": None,
            "source": _SOURCE_ID,
        },
        "schema": schema,
        "shift": None,
        "slicing": {
            "available_steps": len(output) - 1,
            "end_step": len(output) - 1,
            "output_steps": len(output) - 1,
            "start_step": 0,
        },
        "sources": sources,
        "terminal": {"is_zero": output[-1] == 0.0, "value": output[-1]},
        "transform_order": [f"algebra.{operation}"],
        "warnings": [],
    }
    return _canonical_json(projection)


def build_schedule_slice(
    *,
    sigmas: object,
    schedule_info: object,
    start_step: object,
    end_step: object,
) -> ScheduleAlgebraNodeResult:
    """Return one strict terminal-inclusive subrange of a verified schedule."""

    verified = _verified_schedule(
        sigmas=sigmas,
        schedule_info=schedule_info,
        allowed_domains=_SUPPORTED_DOMAINS,
    )
    maximum = len(verified.values) - 1
    start = _step(start_step, label="start_step", maximum=maximum)
    end = _step(end_step, label="end_step", maximum=maximum)
    if start >= end:
        raise ScheduleContractError("end_step must be greater than start_step")
    if start == 0 and end == maximum:
        raise ScheduleContractError("schedule slice must alter the source range")
    output = verified.values[start : end + 1]
    return ScheduleAlgebraNodeResult(
        domain=verified.domain,
        sigmas=output,
        schedule_info_json=_projection(
            schema=SCHEDULE_SLICE_SCHEMA_ID,
            operation="slice",
            domain=verified.domain,
            output=output,
            sources=[_source_identity(verified)],
            parameters={"end_step": end, "start_step": start},
        ),
    )


def build_schedule_concatenation(
    *,
    sigmas_left: object,
    schedule_info_left: object,
    sigmas_right: object,
    schedule_info_right: object,
) -> ScheduleAlgebraNodeResult:
    """Join verified schedules at one exactly equal shared boundary."""

    left = _verified_schedule(
        sigmas=sigmas_left,
        schedule_info=schedule_info_left,
        allowed_domains=_SUPPORTED_DOMAINS,
    )
    right = _verified_schedule(
        sigmas=sigmas_right,
        schedule_info=schedule_info_right,
        allowed_domains=_SUPPORTED_DOMAINS,
    )
    if left.domain is not right.domain:
        raise ScheduleContractError("concatenation domains must match")
    if left.values[-1] != right.values[0]:
        raise ScheduleContractError("concatenation requires one exact shared boundary")
    output = left.values + right.values[1:]
    if len(output) > _MAX_STEPS + 1:
        raise ScheduleContractError("concatenated schedule exceeds the step limit")
    return ScheduleAlgebraNodeResult(
        domain=left.domain,
        sigmas=output,
        schedule_info_json=_projection(
            schema=SCHEDULE_CONCATENATE_SCHEMA_ID,
            operation="concatenate",
            domain=left.domain,
            output=output,
            sources=[_source_identity(left), _source_identity(right)],
            parameters={"boundary": left.values[-1], "shared_boundary_count": 1},
        ),
    )


def build_schedule_resample(
    *,
    sigmas: object,
    schedule_info: object,
    output_steps: object,
) -> ScheduleAlgebraNodeResult:
    """Linearly resample a verified schedule in normalized index space."""

    verified = _verified_schedule(
        sigmas=sigmas,
        schedule_info=schedule_info,
        allowed_domains=_SUPPORTED_DOMAINS,
    )
    requested_steps = _positive_steps(output_steps)
    input_steps = len(verified.values) - 1
    if requested_steps == input_steps:
        raise ScheduleContractError("resampling must change the transition count")

    values: list[float] = []
    for index in range(requested_steps + 1):
        position = index * input_steps / requested_steps
        lower = min(int(position), input_steps - 1)
        fraction = position - lower
        if index == requested_steps:
            values.append(verified.values[-1])
        else:
            start = verified.values[lower]
            values.append(start + (verified.values[lower + 1] - start) * fraction)
    output = tuple(values)
    return ScheduleAlgebraNodeResult(
        domain=verified.domain,
        sigmas=output,
        schedule_info_json=_projection(
            schema=SCHEDULE_RESAMPLE_SCHEMA_ID,
            operation="resample",
            domain=verified.domain,
            output=output,
            sources=[_source_identity(verified)],
            parameters={
                "input_steps": input_steps,
                "method": "index_linear_v1",
                "output_steps": requested_steps,
            },
        ),
    )


def _host_result(result: ScheduleAlgebraNodeResult) -> tuple[object, str]:
    try:
        torch = importlib.import_module("torch")
        float_tensor = torch.__dict__["FloatTensor"]
    except (ImportError, KeyError) as exc:
        # CRITICAL: keep Torch execution-only; package imports must remain dependency-free.
        raise RuntimeError("ComfyUI host execution requires Torch FloatTensor support") from exc
    tensor = float_tensor(result.sigmas)
    try:
        host_values = tuple(float(value) for value in tensor.tolist())
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("ComfyUI host tensor must expose numeric tolist output") from exc
    values = validate_sigma_schedule(
        host_values,
        domain=result.domain,
        expected_steps=len(host_values) - 1,
        require_terminal_zero=False,
    )
    projection = json.loads(result.schedule_info_json)
    if not isinstance(projection, dict) or not isinstance(projection.get("fingerprints"), dict):
        raise ScheduleContractError("schedule algebra fingerprints are missing")
    fingerprint = sigma_output_fingerprint(values, domain=result.domain)
    projection["fingerprints"] = {"complete": fingerprint, "output": fingerprint}
    return tensor, _canonical_json(projection)


class ScheduleSlice:
    """Slice a fingerprint-verified schedule without implicit terminal changes."""

    DESCRIPTION = "Slices a verified schedule by terminal-inclusive sigma indices."
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "slice"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": _MAX_STEPS, "step": 1}),
                "end_step": ("INT", {"default": 1, "min": 0, "max": _MAX_STEPS, "step": 1}),
            }
        }

    def slice(
        self,
        sigmas: object,
        schedule_info: object,
        start_step: object,
        end_step: object,
    ) -> tuple[object, str]:
        return _host_result(
            build_schedule_slice(
                sigmas=_host_sigma_tuple(sigmas),
                schedule_info=schedule_info,
                start_step=start_step,
                end_step=end_step,
            )
        )


class ScheduleConcatenate:
    """Concatenate schedules only at an exact fingerprint-verified boundary."""

    DESCRIPTION = "Concatenates verified schedules sharing one exactly equal boundary sigma."
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "concatenate"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        schedule_info = ("STRING", {"default": "", "multiline": True})
        return {
            "required": {
                "sigmas_left": ("SIGMAS",),
                "schedule_info_left": schedule_info,
                "sigmas_right": ("SIGMAS",),
                "schedule_info_right": schedule_info,
            }
        }

    def concatenate(
        self,
        sigmas_left: object,
        schedule_info_left: object,
        sigmas_right: object,
        schedule_info_right: object,
    ) -> tuple[object, str]:
        return _host_result(
            build_schedule_concatenation(
                sigmas_left=_host_sigma_tuple(sigmas_left),
                schedule_info_left=schedule_info_left,
                sigmas_right=_host_sigma_tuple(sigmas_right),
                schedule_info_right=schedule_info_right,
            )
        )


class ScheduleResample:
    """Explicitly resample a fingerprint-verified schedule."""

    DESCRIPTION = "Resamples a verified schedule by explicit normalized-index interpolation."
    CATEGORY = "Sigmax/scheduling"
    FUNCTION = "resample"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "schedule_info")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
                "output_steps": ("INT", {"default": 20, "min": 1, "max": _MAX_STEPS, "step": 1}),
            }
        }

    def resample(
        self,
        sigmas: object,
        schedule_info: object,
        output_steps: object,
    ) -> tuple[object, str]:
        return _host_result(
            build_schedule_resample(
                sigmas=_host_sigma_tuple(sigmas),
                schedule_info=schedule_info,
                output_steps=output_steps,
            )
        )
