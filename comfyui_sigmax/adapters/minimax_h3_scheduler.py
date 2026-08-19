"""Lazy ComfyUI delegation for MiniMax H3 host-native scheduler choices.

The module itself remains dependency-free. ComfyUI modules are imported only while executing a
native scheduler, and the installed host retains ownership of every scheduler formula.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles.minimax_h3_scheduler_contract import (
    MINIMAX_H3_NATIVE_SCHEDULERS,
    MINIMAX_H3_SCHEDULER_HOSTS,
    MiniMaxH3ModelSamplingEvidence,
    MiniMaxH3SamplingAPI,
    MiniMaxH3SchedulerContractError,
    MiniMaxH3SchedulerReasonCode,
    MiniMaxH3SchedulerResultValidation,
    qualify_minimax_h3_scheduler_request,
    validate_minimax_h3_scheduler_result,
)

_FL2VA_MODE: Final = "H3 Base FL2VA"
_REF2VA_MODE: Final = "H3 Base Ref2VA"
_VIDEO_SHIFT_MARKER: Final = "minimax_h3_sigma_shift_video"
_AUDIO_SHIFT_MARKER: Final = "minimax_h3_sigma_shift_audio"
MINIMAX_H3_HOST_REVISION_BY_VERSION: Final = {
    host.version: host.revision for host in MINIMAX_H3_SCHEDULER_HOSTS
}
_MINIMAX_H3_HOST_BY_VERSION: Final = {host.version: host for host in MINIMAX_H3_SCHEDULER_HOSTS}


def _error(reason: MiniMaxH3SchedulerReasonCode, message: str) -> MiniMaxH3SchedulerContractError:
    return MiniMaxH3SchedulerContractError(reason, message)


def _task(variant: object) -> str:
    if variant == _FL2VA_MODE:
        return "fl2va"
    if variant == _REF2VA_MODE:
        return "ref2va"
    raise ScheduleContractError("MiniMax H3 variant must be selected explicitly")


def _host_version(module: object) -> str:
    value = getattr(module, "__version__", None)
    if not isinstance(value, str) or not value:
        raise _error(
            MiniMaxH3SchedulerReasonCode.UNSUPPORTED_HOST,
            "installed ComfyUI does not expose a supported version",
        )
    return value


def _finite_shift(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise _error(
            MiniMaxH3SchedulerReasonCode.SHIFT_MISMATCH,
            f"MODEL {label} shift must be finite and positive",
        )
    return float(value)


def _model_sampling(
    model: object,
    *,
    model_sampling_module: object,
    supported_models_module: object,
    task: str,
    sampling_api: MiniMaxH3SamplingAPI,
) -> tuple[object, MiniMaxH3ModelSamplingEvidence]:
    getter = getattr(model, "get_model_object", None)
    if not callable(getter):
        raise _error(
            MiniMaxH3SchedulerReasonCode.MODEL_FAMILY_MISMATCH,
            "MODEL does not expose the reviewed model-sampling interface",
        )
    try:
        sampling = getter("model_sampling")
    except Exception as exc:
        raise _error(
            MiniMaxH3SchedulerReasonCode.MODEL_FAMILY_MISMATCH,
            "MODEL sampling object is unavailable",
        ) from exc

    if sampling_api is MiniMaxH3SamplingAPI.DISCRETE_FLOW_H3_V030:
        sampling_type = getattr(model_sampling_module, "ModelSamplingDiscreteFlow", None)
        is_model_sampling_av = False
    else:
        sampling_type = getattr(model_sampling_module, "ModelSamplingAV", None)
        is_model_sampling_av = True
    if not isinstance(sampling_type, type) or not isinstance(sampling, sampling_type):
        raise _error(
            MiniMaxH3SchedulerReasonCode.MODEL_SAMPLING_NOT_AV,
            "MODEL sampling object is incompatible with the exact host H3 sampling API",
        )

    model_core = getattr(model, "model", None)
    model_config = getattr(model_core, "model_config", None)
    h3_config_type = getattr(supported_models_module, "MiniMaxH3", None)
    unet_config = getattr(model_config, "unet_config", None)
    image_model = unet_config.get("image_model") if isinstance(unet_config, Mapping) else None
    if (
        not isinstance(h3_config_type, type)
        or not isinstance(model_config, h3_config_type)
        or image_model != "minimax_h3"
    ):
        raise _error(
            MiniMaxH3SchedulerReasonCode.MODEL_FAMILY_MISMATCH,
            "MODEL config does not identify MiniMax H3",
        )

    video_shift = _finite_shift(getattr(sampling, "shift", None), label="video")
    model_options = getattr(model, "model_options", None)
    transformer_options: Mapping[object, object] = {}
    if isinstance(model_options, Mapping):
        candidate = model_options.get("transformer_options")
        if isinstance(candidate, Mapping):
            transformer_options = candidate
    video_marker = transformer_options.get(_VIDEO_SHIFT_MARKER)
    audio_marker = transformer_options.get(_AUDIO_SHIFT_MARKER)
    if (video_marker is None) != (audio_marker is None):
        raise _error(
            MiniMaxH3SchedulerReasonCode.SHIFT_MISMATCH,
            "MODEL contains incomplete MiniMax H3 shift markers",
        )
    if sampling_api is MiniMaxH3SamplingAPI.DISCRETE_FLOW_H3_V030:
        if video_marker is None:
            raise _error(
                MiniMaxH3SchedulerReasonCode.SHIFT_MISMATCH,
                "ComfyUI 0.30.0 H3 MODEL requires complete video/audio shift markers",
            )
        marked_video = _finite_shift(video_marker, label="marked video")
        marked_audio = _finite_shift(audio_marker, label="marked audio")
        if not math.isclose(marked_video, video_shift, abs_tol=1e-9):
            raise _error(
                MiniMaxH3SchedulerReasonCode.SHIFT_MISMATCH,
                "MODEL shift markers conflict with its sampling object",
            )
        audio_shift = marked_audio
    else:
        audio_shift = _finite_shift(getattr(sampling, "audio_shift", None), label="audio")
        if video_marker is not None:
            marked_video = _finite_shift(video_marker, label="marked video")
            marked_audio = _finite_shift(audio_marker, label="marked audio")
            if not math.isclose(marked_video, video_shift, abs_tol=1e-9) or not math.isclose(
                marked_audio, audio_shift, abs_tol=1e-9
            ):
                raise _error(
                    MiniMaxH3SchedulerReasonCode.SHIFT_MISMATCH,
                    "MODEL shift markers conflict with its sampling object",
                )

    return sampling, MiniMaxH3ModelSamplingEvidence(
        family_id="minimax_h3",
        task=task,
        is_model_sampling_av=is_model_sampling_av,
        sampling_api=sampling_api,
        video_shift=video_shift,
        audio_shift=audio_shift,
        already_shifted=True,
    )


def _handler_names(samplers_module: object) -> tuple[str, ...]:
    names = getattr(samplers_module, "SCHEDULER_NAMES", None)
    handlers = getattr(samplers_module, "SCHEDULER_HANDLERS", None)
    if not isinstance(names, (list, tuple)) or not isinstance(handlers, Mapping):
        return ()
    return tuple(
        name
        for name in names
        if isinstance(name, str) and name in handlers and name in MINIMAX_H3_NATIVE_SCHEDULERS
    )


def _raw_values(result: object) -> tuple[tuple[float, ...], str]:
    dtype_value = str(getattr(result, "dtype", ""))
    dtype = dtype_value.rsplit(".", maxsplit=1)[-1]
    detached = getattr(result, "detach", None)
    selected = detached() if callable(detached) else result
    cpu = getattr(selected, "cpu", None)
    selected = cpu() if callable(cpu) else selected
    tolist = getattr(selected, "tolist", None)
    if not callable(tolist):
        raise _error(
            MiniMaxH3SchedulerReasonCode.RESULT_COUNT_INVALID,
            "host scheduler result does not expose a tensor-like tolist boundary",
        )
    try:
        values = tolist()
    except Exception as exc:
        raise _error(
            MiniMaxH3SchedulerReasonCode.RESULT_COUNT_INVALID,
            "host scheduler result could not cross the tensor boundary",
        ) from exc
    if not isinstance(values, (list, tuple)):
        raise _error(
            MiniMaxH3SchedulerReasonCode.RESULT_COUNT_INVALID,
            "host scheduler result must be a one-dimensional vector",
        )
    return tuple(values), dtype


@dataclass(frozen=True, slots=True, kw_only=True)
class MiniMaxH3NativeScheduleResult:
    """Allowlisted native adapter result used by the thin public node."""

    host_version: str
    qualified_host_revision: str
    expected_video_shift: float
    expected_audio_shift: float
    sampling_api: str
    validation: MiniMaxH3SchedulerResultValidation

    def projection(self) -> dict[str, object]:
        selected = self.validation
        return {
            "schema": "sigmax.minimax-h3-native-scheduler-adapter/1",
            "owner": "comfyui_native",
            "scheduler": selected.scheduler,
            "model_task": selected.model_task,
            "recipe_id": selected.recipe_id,
            "sampling_api": self.sampling_api,
            "dtype": selected.dtype,
            "host": {
                "observed_version": self.host_version,
                "qualified_revision": self.qualified_host_revision,
            },
            "shift": {
                "already_applied": True,
                "audio": self.expected_audio_shift,
                "video": self.expected_video_shift,
            },
            "counts": {
                "requested_steps": selected.requested_steps,
                "raw_sigmas": selected.raw_count,
                "actual_sigmas": len(selected.output_sigmas),
                "actual_transitions": selected.output_transitions,
            },
            "terminal": {
                "included": selected.output_sigmas[-1] == 0.0,
                "value": selected.output_sigmas[-1],
            },
            "slicing": {"start_step": selected.start_step, "end_step": selected.end_step},
            "fingerprints": {
                "contract": selected.contract_fingerprint,
                "output": selected.output_fingerprint,
            },
        }


def build_minimax_h3_native_schedule(
    *,
    model: object,
    scheduler: object,
    variant: object,
    steps: object,
    start_step: object,
    end_step: object,
    recipe_id: str | None,
) -> MiniMaxH3NativeScheduleResult:
    """Validate and delegate one native schedule without copying host scheduler math."""

    if scheduler not in MINIMAX_H3_NATIVE_SCHEDULERS:
        raise _error(
            MiniMaxH3SchedulerReasonCode.UNSUPPORTED_SCHEDULER,
            "native adapter received a non-native scheduler",
        )
    if model is None:
        raise _error(
            MiniMaxH3SchedulerReasonCode.MODEL_REQUIRED,
            "MODEL is required for a ComfyUI-native scheduler",
        )
    task = _task(variant)

    # CRITICAL: keep all ComfyUI imports execution-only so package imports remain host-independent.
    try:
        version_module = importlib.import_module("comfyui_version")
        samplers_module = importlib.import_module("comfy.samplers")
        model_sampling_module = importlib.import_module("comfy.model_sampling")
        supported_models_module = importlib.import_module("comfy.supported_models")
    except ImportError as exc:
        raise _error(
            MiniMaxH3SchedulerReasonCode.UNSUPPORTED_HOST,
            "native scheduler execution requires a supported ComfyUI host",
        ) from exc

    host_version = _host_version(version_module)
    host = _MINIMAX_H3_HOST_BY_VERSION.get(host_version)
    if host is None:
        raise _error(
            MiniMaxH3SchedulerReasonCode.UNSUPPORTED_HOST,
            "installed ComfyUI version is outside the qualified host matrix",
        )
    handlers = _handler_names(samplers_module)
    sampling, evidence = _model_sampling(
        model,
        model_sampling_module=model_sampling_module,
        supported_models_module=supported_models_module,
        task=task,
        sampling_api=host.sampling_api,
    )
    qualification = qualify_minimax_h3_scheduler_request(
        scheduler=scheduler,
        steps=steps,
        model_sampling=evidence,
        recipe_id=recipe_id,
        host_revision=host.revision,
        available_handlers=handlers,
    )
    calculate_sigmas = getattr(samplers_module, "calculate_sigmas", None)
    if not callable(calculate_sigmas):
        raise _error(
            MiniMaxH3SchedulerReasonCode.MISSING_HANDLER,
            "installed ComfyUI lacks the scheduler dispatcher",
        )
    try:
        raw_result = calculate_sigmas(sampling, scheduler, qualification.steps)
    except Exception as exc:
        raise ScheduleContractError(f"ComfyUI scheduler {scheduler!r} execution failed") from exc
    raw, dtype = _raw_values(raw_result)
    validation = validate_minimax_h3_scheduler_result(
        qualification=qualification,
        raw_sigmas=raw,
        dtype=dtype,
        start_step=start_step,
        end_step=None if end_step == -1 else end_step,
    )
    return MiniMaxH3NativeScheduleResult(
        host_version=host_version,
        qualified_host_revision=host.revision,
        expected_video_shift=qualification.expected_video_shift,
        expected_audio_shift=qualification.expected_audio_shift,
        sampling_api=host.sampling_api.value,
        validation=validation,
    )


__all__ = [
    "MINIMAX_H3_HOST_REVISION_BY_VERSION",
    "MiniMaxH3NativeScheduleResult",
    "build_minimax_h3_native_schedule",
]
