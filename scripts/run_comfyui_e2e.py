"""Run isolated real-host ComfyUI H1 plus Turbo and RAW H2 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from comfyui_sigmax.adapters.registration import builtin_node_registry  # noqa: E402
from comfyui_sigmax.core import (  # noqa: E402
    ExecutionComponent,
    ExecutionFeatureRequest,
    ExecutionHost,
    ExecutionReceiptMetadata,
    ExecutionRngOwnership,
    ExecutionStatus,
    NoiseOwnership,
    PortableExecutionBundle,
    ScheduleContractError,
    SigmaDomain,
    build_execution_receipt,
    canonical_projection_bytes,
    deserialize_portable_execution_bundle,
    evaluate_compatibility,
)
from comfyui_sigmax.nodes import (  # noqa: E402
    build_krea2_sigma_schedule,
    sigma_output_fingerprint,
)
from comfyui_sigmax.nodes.minimax_h3_sigma_scheduler import (  # noqa: E402
    build_minimax_h3_sigma_schedule,
)
from comfyui_sigmax.profiles import KREA2_TURBO_SCHEMA  # noqa: E402
from comfyui_sigmax.profiles.minimax_h3 import (  # noqa: E402
    MINIMAX_H3_COMFYUI_REVISION,
)
from comfyui_sigmax.workflows import (  # noqa: E402
    WorkflowValidationLane,
    load_canonical_workflow_fixtures,
    validate_live_workflow_fixtures,
)
from comfyui_sigmax.workflows.minimax_h3 import (  # noqa: E402
    MiniMaxH3ModelFiles,
    MiniMaxH3PublicVariant,
    MiniMaxH3WorkflowSpec,
    build_minimax_h3_host_workflow,
)
from comfyui_sigmax.workflows.validation import (  # noqa: E402
    CANONICAL_HOST_REVISION,
    CANONICAL_HOST_VERSION,
)
from scripts.conformance.capability_receipt_report import (  # noqa: E402
    HostAttempt,
    build_host_attempt_transition,
    validate_host_attempt_transition,
)
from scripts.parity.krea2_native_euler_report import (  # noqa: E402
    SOURCE_BLOBS as NATIVE_EULER_SOURCE_BLOBS,
)
from scripts.parity.krea2_native_euler_report import (  # noqa: E402
    build_native_euler_report,
)

_LOOPBACK: Final = "127.0.0.1"
_OUTPUT_NODE_ID: Final = "3"
_H3_OUTPUT_NODE_ID: Final = "4"
_BUNDLE_KEY: Final = "sigmax_execution_bundle"
_H3_TRACE_KEY: Final = "sigmax_native_euler_trace"
_MINIMAX_H3_OUTPUT_NODE_ID: Final = "5"
_MINIMAX_H3_TRACE_KEY: Final = "sigmax_minimax_h3_h2"
_MINIMAX_H3_HOST_VERSION: Final = "0.30.0"
_MINIMAX_H3_MIN_LATEST_HOST_VERSION: Final = (0, 31, 0)
_MINIMAX_H3_LATEST_LANE: Final = "latest"
_MINIMAX_H3_MODEL_LANE_SCHEMA: Final = "sigmax.minimax-h3-model-lane/1"
_MINIMAX_H3_PUBLIC_VARIANTS: Final = ("H3 Base FL2VA", "H3 Base Ref2VA")
_ALGEBRA_OUTPUT_NODE_ID: Final = "7"
_ALGEBRA_TRACE_KEY: Final = "sigmax_schedule_algebra"
_CHECKPOINT_OUTPUT_NODE_ID: Final = "2"
_CHECKPOINT_TRACE_KEY: Final = "sigmax_checkpoint_evidence"
_Z_IMAGE_OUTPUT_NODE_ID: Final = "2"
_Z_IMAGE_TRACE_KEY: Final = "sigmax_z_image_schedule"
_FLUX1_SCHNELL_OUTPUT_NODE_ID: Final = "2"
_FLUX1_SCHNELL_TRACE_KEY: Final = "sigmax_flux1_schnell_schedule"
_QWEN_IMAGE_OUTPUT_NODE_ID: Final = "2"
_QWEN_IMAGE_TRACE_KEY: Final = "sigmax_qwen_image_schedule"
_SD3_OUTPUT_NODE_ID: Final = "2"
_SD3_TRACE_KEY: Final = "sigmax_sd3_schedule"
_AURAFLOW_OUTPUT_NODE_ID: Final = "2"
_AURAFLOW_TRACE_KEY: Final = "sigmax_auraflow_schedule"
_LUMINA2_OUTPUT_NODE_ID: Final = "2"
_LUMINA2_TRACE_KEY: Final = "sigmax_lumina2_schedule"
_HUNYUAN_IMAGE21_OUTPUT_NODE_ID: Final = "2"
_HUNYUAN_IMAGE21_TRACE_KEY: Final = "sigmax_hunyuan_image21_schedule"
_ANIMA_OUTPUT_NODE_ID: Final = "2"
_ANIMA_TRACE_KEY: Final = "sigmax_anima_schedule"
_WAN_OUTPUT_NODE_ID: Final = "2"
_WAN_TRACE_KEY: Final = "sigmax_wan_schedule"
_LTX_OUTPUT_NODE_ID: Final = "2"
_LTX_TRACE_KEY: Final = "sigmax_ltx_schedule"
_KREA2_LORA_OUTPUT_NODE_ID: Final = "2"
_KREA2_LORA_TRACE_KEY: Final = "sigmax_krea2_lora_experimental"
_KREA2_CONDITIONING_OUTPUT_NODE_ID: Final = "3"
_KREA2_CONDITIONING_TRACE_KEY: Final = "sigmax_krea2_conditioning"
_CHECKPOINT_FIXTURE_NAME: Final = "sigmax-m6-08-fixture.safetensors"
_H3_TEST_PACK_NAME: Final = "ComfyUI-Sigmax-H3"
_H3_TEST_PACK_SOURCE: Final = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "comfyui_h3_nodes" / "__init__.py"
)
_EXPECTED_NUMERICAL_FINGERPRINT: Final = (
    "sha256:24984ad4412a3c47103a52cfe3af16bb9df8789f98401d9fc281b3f6ca0892ac"
)
_RAW_CASES: Final = {
    "krea2-raw-official-square-1024": {
        "steps": 52,
        "width": 1024,
        "height": 1024,
        "strict_official": True,
        "effective_width": 1024,
        "effective_height": 1024,
        "image_seq_len": 4096,
        "mu": 0.90625,
        "recipe": "krea2.raw.official-full-52",
        "evidence": "official",
        "numerical_fingerprint": (
            "sha256:5ff69c30df41c7f37eae14502155b31f23724d32427180f69118cabcd6a3ac61"
        ),
    },
    "krea2-raw-official-landscape-1353x761": {
        "steps": 52,
        "width": 1353,
        "height": 761,
        "strict_official": True,
        "effective_width": 1360,
        "effective_height": 768,
        "image_seq_len": 4080,
        "mu": 0.9045572916666667,
        "recipe": "krea2.raw.official-full-52",
        "evidence": "official",
        "numerical_fingerprint": (
            "sha256:01352f42660bd3b31bbaf7548a9891273899afd375adeb68c7f7c93fd2a4f0d4"
        ),
    },
    "krea2-raw-diffusers-portrait-761x1353": {
        "steps": 28,
        "width": 761,
        "height": 1353,
        "strict_official": False,
        "effective_width": 768,
        "effective_height": 1360,
        "image_seq_len": 4080,
        "mu": 0.9045572916666667,
        "recipe": "krea2.raw.diffusers-reference-28",
        "evidence": "framework_reference",
        "numerical_fingerprint": (
            "sha256:52208c5fa3780c95cce399b1f842f3fea56503e76fdf5ef4abc3069cf3108f01"
        ),
    },
}
_MAX_HTTP_BYTES: Final = 4_000_000
_MAX_LOG_BYTES: Final = 1_000_000
_SECRET_PATTERN: Final = re.compile(
    r"(?i)(authorization\s*:\s*bearer|token|secret|password|api[_-]?key)"
    r"(\s*[:=]\s*|\s+)[^\s,;]+"
)


def build_turbo_api_prompt() -> dict[str, object]:
    """Return the exact model-free scheduler -> inspector -> output API graph."""

    return {
        "1": {
            "class_type": "Sigmax.Krea2SigmaScheduler",
            "inputs": {
                "variant": "Turbo",
                "steps": 8,
                "width": 1024,
                "height": 1024,
                "strict_official": True,
                "start_step": 0,
                "end_step": -1,
            },
        },
        "2": {
            "class_type": "Sigmax.ScheduleInspector",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
            },
        },
        "3": {
            "class_type": "Sigmax.TurboWorkflowOutput",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
                "schedule_report": ["2", 0],
            },
        },
    }


def build_minimax_h3_h2_api_prompt(variant: str) -> dict[str, object]:
    """Return a model-free MiniMax H3 scheduler -> probe graph for one explicit variant."""

    if variant not in {"H3 Base FL2VA", "H3 Base Ref2VA"}:
        raise ScheduleContractError("MiniMax H3 H2 variant must be selected explicitly")
    return {
        "1": {
            "class_type": "Sigmax.MiniMaxH3SigmaScheduler",
            "inputs": {
                "end_step": -1,
                "steps": 20,
                "start_step": 0,
                "variant": variant,
            },
        },
        _MINIMAX_H3_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.MiniMaxH3ScheduleProbe",
            "inputs": {
                "schedule_info": ["1", 1],
                "sigmas": ["1", 0],
            },
        },
    }


def build_krea2_lora_experimental_h2_api_prompt(variant: str) -> dict[str, object]:
    """Return a model-free 12-step experimental scheduler -> probe graph."""

    allowed = {
        "LoRA Experimental (RAW mu)",
        "LoRA Experimental (Turbo mu)",
    }
    if variant not in allowed:
        raise ScheduleContractError("experimental Krea 2 H2 variant is unsupported")
    return {
        "1": {
            "class_type": "Sigmax.Krea2SigmaScheduler",
            "inputs": {
                "end_step": -1,
                "height": 1024,
                "start_step": 0,
                "steps": 12,
                "strict_official": True,
                "variant": variant,
                "width": 1024,
            },
        },
        _KREA2_LORA_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.Krea2LoraExperimentalProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_krea2_lora_experimental_h2_history(
    history: object,
    *,
    prompt_id: str,
    variant: str,
) -> dict[str, object]:
    """Verify one real-host experimental schedule and the selected mu source."""

    expected = {
        "LoRA Experimental (RAW mu)": ("raw", 0.90625),
        "LoRA Experimental (Turbo mu)": ("turbo", 1.15),
    }
    selection = expected.get(variant)
    if selection is None:
        raise ScheduleContractError("experimental Krea 2 H2 variant is unsupported")
    expected_source, expected_mu = selection
    root = _object(history, label="experimental Krea 2 history")
    entry = _object(root.get(prompt_id), label="experimental Krea 2 history entry")
    status = _object(entry.get("status"), label="experimental Krea 2 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("experimental Krea 2 prompt did not prove success")
    outputs = _object(entry.get("outputs"), label="experimental Krea 2 outputs")
    output = _object(
        outputs.get(_KREA2_LORA_OUTPUT_NODE_ID),
        label="experimental Krea 2 probe output",
    )
    traces = _array(output.get(_KREA2_LORA_TRACE_KEY), label="experimental Krea 2 trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("experimental Krea 2 probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="experimental Krea 2 decoded trace")
    info = _object(trace.get("schedule_info"), label="experimental Krea 2 information")
    profile = _object(info.get("profile"), label="experimental Krea 2 profile")
    shift = _object(info.get("shift"), label="experimental Krea 2 shift")
    slicing = _object(info.get("slicing"), label="experimental Krea 2 slicing")
    fingerprints = _object(info.get("fingerprints"), label="experimental Krea 2 fingerprints")
    sigmas = _array(trace.get("sigmas"), label="experimental Krea 2 sigmas")
    float_sigmas = tuple(float(value) for value in sigmas)
    if (
        info.get("schema") != "sigmax.krea2-sigma-node/1"
        or profile
        != {
            "evidence": "experimental",
            "id": "krea2.raw-turbo-lora.experimental",
            "recipe": "krea2.raw-turbo-lora.experimental",
            "variant": "raw_turbo_lora",
            "version": "1",
        }
        or info.get("strict_official") is not False
        or shift.get("mu_source") != expected_source
        or shift.get("mu") != expected_mu
        or slicing.get("output_steps") != 12
        or len(float_sigmas) != 13
        or float_sigmas[0] != 1.0
        or float_sigmas[-1] != 0.0
        or any(left <= right for left, right in pairwise(float_sigmas))
        or fingerprints.get("output")
        != sigma_output_fingerprint(float_sigmas, domain=SigmaDomain.UNIT_FLOW)
    ):
        raise ScheduleContractError("experimental Krea 2 H2 execution evidence drifted")
    return {
        "mu": expected_mu,
        "mu_source": expected_source,
        "numerical_fingerprint": fingerprints.get("complete"),
        "output_fingerprint": fingerprints.get("output"),
        "profile_id": "krea2.raw-turbo-lora.experimental",
        "requested_transitions": 12,
        "status": "succeeded",
    }


def build_krea2_conditioning_h2_api_prompt(variant: str) -> dict[str, object]:
    """Return a model-free long-sequence source -> rebalance -> probe graph."""

    if variant not in {"RAW", "Turbo"}:
        raise ScheduleContractError("conditioning H2 variant must be RAW or Turbo")
    return {
        "1": {
            "class_type": "SigmaxTest.Krea2ConditioningSource",
            "inputs": {"sequence_length": 97, "variant": variant},
        },
        "2": {
            "class_type": "Sigmax.Krea2ConditioningRebalance",
            "inputs": {
                "conditioning": ["1", 0],
                "profile": "Subtle Experimental",
                "strength": 0.5,
                "variant": variant,
            },
        },
        _KREA2_CONDITIONING_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.Krea2ConditioningProbe",
            "inputs": {
                "conditioning": ["2", 0],
                "modifier_info": ["2", 1],
                "variant": variant,
            },
        },
    }


def verify_krea2_conditioning_h2_history(
    history: object,
    *,
    prompt_id: str,
    variant: str,
) -> dict[str, object]:
    """Verify model-free RAW/Turbo conditioning execution and metadata preservation."""

    if variant not in {"RAW", "Turbo"}:
        raise ScheduleContractError("conditioning H2 variant must be RAW or Turbo")
    root = _object(history, label="conditioning H2 history")
    entry = _object(root.get(prompt_id), label="conditioning H2 history entry")
    status = _object(entry.get("status"), label="conditioning H2 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("conditioning H2 prompt did not prove success")
    outputs = _object(entry.get("outputs"), label="conditioning H2 outputs")
    output = _object(
        outputs.get(_KREA2_CONDITIONING_OUTPUT_NODE_ID),
        label="conditioning H2 probe output",
    )
    traces = _array(
        output.get(_KREA2_CONDITIONING_TRACE_KEY),
        label="conditioning H2 trace",
    )
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("conditioning H2 trace is malformed")
    trace = _object(json.loads(traces[0]), label="conditioning H2 decoded trace")
    shape = trace.get("shape")
    metadata_keys = trace.get("metadata_keys")
    rms = trace.get("rms")
    if (
        trace.get("variant") != variant
        or shape != [1, 97, 30720]
        or metadata_keys
        != ["area", "attention_mask", "pooled_output", "reference_latents", "source_marker"]
        or not isinstance(rms, int | float)
        or isinstance(rms, bool)
        or not math.isfinite(float(rms))
        or float(rms) <= 0.0
        or not isinstance(trace.get("report_fingerprint"), str)
    ):
        raise ScheduleContractError("conditioning H2 execution evidence drifted")
    return {
        "metadata_keys": metadata_keys,
        "report_fingerprint": trace["report_fingerprint"],
        "rms": float(rms),
        "shape": shape,
        "status": "succeeded",
        "variant": variant,
    }


def build_z_image_h2_api_prompt(variant: str) -> dict[str, object]:
    """Return a model-free Z-Image scheduler -> test probe graph."""

    if variant not in {"Base", "Turbo"}:
        raise ScheduleContractError("Z-Image H2 variant must be Base or Turbo")
    steps = 50 if variant == "Base" else 8
    return {
        "1": {
            "class_type": "Sigmax.ZImageSigmaScheduler",
            "inputs": {
                "end_step": -1,
                "start_step": 0,
                "steps": steps,
                "strict_official": True,
                "variant": variant,
            },
        },
        _Z_IMAGE_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.ZImageScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_z_image_h2_history(
    history: object, *, prompt_id: str, variant: str
) -> dict[str, object]:
    """Verify completed Z-Image schedule execution and exact variant semantics."""

    if variant not in {"Base", "Turbo"}:
        raise ScheduleContractError("Z-Image H2 variant must be Base or Turbo")
    root = _object(history, label="Z-Image history")
    entry = _object(root.get(prompt_id), label="Z-Image history entry")
    status = _object(entry.get("status"), label="Z-Image prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("Z-Image prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="Z-Image prompt outputs")
    output = _object(outputs.get(_Z_IMAGE_OUTPUT_NODE_ID), label="Z-Image probe output")
    traces = _array(output.get(_Z_IMAGE_TRACE_KEY), label="Z-Image probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("Z-Image probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="Z-Image decoded trace")
    info = _object(trace.get("schedule_info"), label="Z-Image schedule information")
    sigmas = _array(trace.get("sigmas"), label="Z-Image sigma vector")
    expected_steps = 50 if variant == "Base" else 8
    expected_ratio = 6.0 if variant == "Base" else 3.0
    expected_profile = f"z_image.{variant.casefold()}.official"
    if (
        info.get("schema") != "sigmax.z-image-sigma-node/1"
        or _object(info.get("profile"), label="Z-Image profile").get("id") != expected_profile
        or _object(info.get("profile"), label="Z-Image profile").get("evidence") != "official"
        or _object(info.get("shift"), label="Z-Image shift")
        != {"dynamic": False, "kind": "fixed_direct_ratio", "ratio": expected_ratio}
        or _object(info.get("slicing"), label="Z-Image slicing").get("output_steps")
        != expected_steps
        or len(sigmas) != expected_steps + 1
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("Z-Image H2 execution evidence drifted")
    return {
        "numerical_fingerprint": _object(
            info.get("fingerprints"), label="Z-Image fingerprints"
        ).get("complete"),
        "profile_id": expected_profile,
        "ratio": expected_ratio,
        "requested_transitions": expected_steps,
        "status": "succeeded",
    }


def build_flux1_schnell_h2_api_prompt() -> dict[str, object]:
    """Return the official four-step FLUX.1-schnell scheduler -> probe graph."""

    return {
        "1": {
            "class_type": "Sigmax.Flux1SchnellSigmaScheduler",
            "inputs": {
                "end_step": -1,
                "start_step": 0,
                "steps": 4,
                "strict_official": True,
            },
        },
        _FLUX1_SCHNELL_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.Flux1SchnellScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_flux1_schnell_h2_history(history: object, *, prompt_id: str) -> dict[str, object]:
    """Verify the host returned the exact unshifted four-step schedule."""

    root = _object(history, label="FLUX.1-schnell history")
    entry = _object(root.get(prompt_id), label="FLUX.1-schnell history entry")
    status = _object(entry.get("status"), label="FLUX.1-schnell prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("FLUX.1-schnell prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="FLUX.1-schnell prompt outputs")
    output = _object(
        outputs.get(_FLUX1_SCHNELL_OUTPUT_NODE_ID), label="FLUX.1-schnell probe output"
    )
    traces = _array(output.get(_FLUX1_SCHNELL_TRACE_KEY), label="FLUX.1-schnell probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("FLUX.1-schnell probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="FLUX.1-schnell decoded trace")
    info = _object(trace.get("schedule_info"), label="FLUX.1-schnell schedule information")
    sigmas = _array(trace.get("sigmas"), label="FLUX.1-schnell sigma vector")
    if (
        info.get("schema") != "sigmax.flux1-schnell-sigma-node/1"
        or _object(info.get("profile"), label="FLUX.1-schnell profile").get("id")
        != "flux1.schnell.official"
        or _object(info.get("profile"), label="FLUX.1-schnell profile").get("evidence")
        != "official"
        or _object(info.get("shift"), label="FLUX.1-schnell shift")
        != {"dynamic": False, "kind": "none"}
        or _object(info.get("guidance"), label="FLUX.1-schnell guidance")
        != {"host_cfg": 1.0, "model_guidance": 0.0}
        or sigmas != [1.0, 0.75, 0.5, 0.25, 0.0]
    ):
        raise ScheduleContractError("FLUX.1-schnell H2 execution evidence drifted")
    return {
        "numerical_fingerprint": _object(
            info.get("fingerprints"), label="FLUX.1-schnell fingerprints"
        ).get("complete"),
        "profile_id": "flux1.schnell.official",
        "requested_transitions": 4,
        "status": "succeeded",
    }


def build_qwen_image_h2_api_prompt(mode: str) -> dict[str, object]:
    """Return one model-free original-Qwen scheduler -> probe graph."""

    if mode not in {"Comfy Fixed", "Diffusers Dynamic"}:
        raise ScheduleContractError("Qwen Image H2 mode is unsupported")
    return {
        "1": {
            "class_type": "Sigmax.QwenImageSigmaScheduler",
            "inputs": {
                "end_step": -1,
                "image_seq_len": 0 if mode == "Comfy Fixed" else 1024,
                "mode": mode,
                "start_step": 0,
                "steps": 50,
                "strict_official": True,
            },
        },
        _QWEN_IMAGE_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.QwenImageScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_qwen_image_h2_history(
    history: object, *, prompt_id: str, mode: str
) -> dict[str, object]:
    """Verify one original-Qwen fixed or dynamic schedule execution."""

    if mode not in {"Comfy Fixed", "Diffusers Dynamic"}:
        raise ScheduleContractError("Qwen Image H2 mode is unsupported")
    root = _object(history, label="Qwen Image history")
    entry = _object(root.get(prompt_id), label="Qwen Image history entry")
    status = _object(entry.get("status"), label="Qwen Image prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("Qwen Image prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="Qwen Image prompt outputs")
    output = _object(outputs.get(_QWEN_IMAGE_OUTPUT_NODE_ID), label="Qwen Image probe output")
    traces = _array(output.get(_QWEN_IMAGE_TRACE_KEY), label="Qwen Image probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("Qwen Image probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="Qwen Image decoded trace")
    info = _object(trace.get("schedule_info"), label="Qwen Image schedule information")
    sigmas = _array(trace.get("sigmas"), label="Qwen Image sigma vector")
    profile_id = (
        "qwen_image.comfy-fixed.official"
        if mode == "Comfy Fixed"
        else "qwen_image.diffusers-dynamic.framework-reference"
    )
    expected_shift: dict[str, object]
    if mode == "Comfy Fixed":
        expected_shift = {"dynamic": False, "kind": "fixed_direct_ratio", "ratio": 1.15}
    else:
        expected_shift = {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "dynamic": True,
            "image_seq_len": 1024,
            "kind": "exponential_mu",
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "mu": 0.63,
        }
    if (
        info.get("schema") != "sigmax.qwen-image-sigma-node/1"
        or _object(info.get("profile"), label="Qwen Image profile").get("id") != profile_id
        or _object(info.get("profile"), label="Qwen Image profile").get("evidence")
        != "framework_reference"
        or _object(info.get("shift"), label="Qwen Image shift") != expected_shift
        or _object(info.get("slicing"), label="Qwen Image slicing").get("output_steps") != 50
        or len(sigmas) != 51
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("Qwen Image H2 execution evidence drifted")
    return {
        "image_seq_len": 0 if mode == "Comfy Fixed" else 1024,
        "mode": mode,
        "numerical_fingerprint": _object(
            info.get("fingerprints"), label="Qwen Image fingerprints"
        ).get("complete"),
        "profile_id": profile_id,
        "requested_transitions": 50,
        "status": "succeeded",
    }


def build_sd3_h2_api_prompt(mode: str) -> dict[str, object]:
    """Return one model-free original-SD3 scheduler -> probe graph."""

    if mode not in {"Publisher Reference (1.0)", "Comfy/Diffusers Fixed (3.0)"}:
        raise ScheduleContractError("SD3 H2 mode is unsupported")
    steps = 50 if mode == "Publisher Reference (1.0)" else 28
    return {
        "1": {
            "class_type": "Sigmax.SD3SigmaScheduler",
            "inputs": {
                "already_shifted": False,
                "end_step": -1,
                "mode": mode,
                "start_step": 0,
                "steps": steps,
                "strict_source": True,
            },
        },
        _SD3_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.SD3ScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_sd3_h2_history(history: object, *, prompt_id: str, mode: str) -> dict[str, object]:
    """Verify one original-SD3 source mode on the model-free host lane."""

    if mode not in {"Publisher Reference (1.0)", "Comfy/Diffusers Fixed (3.0)"}:
        raise ScheduleContractError("SD3 H2 mode is unsupported")
    root = _object(history, label="SD3 history")
    entry = _object(root.get(prompt_id), label="SD3 history entry")
    status = _object(entry.get("status"), label="SD3 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("SD3 prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="SD3 prompt outputs")
    output = _object(outputs.get(_SD3_OUTPUT_NODE_ID), label="SD3 probe output")
    traces = _array(output.get(_SD3_TRACE_KEY), label="SD3 probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("SD3 probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="SD3 decoded trace")
    info = _object(trace.get("schedule_info"), label="SD3 schedule information")
    sigmas = _array(trace.get("sigmas"), label="SD3 sigma vector")
    if mode == "Publisher Reference (1.0)":
        profile_id, evidence, ratio, steps = "sd3.publisher-reference.official", "official", 1.0, 50
    else:
        profile_id, evidence, ratio, steps = (
            "sd3.comfy-diffusers-fixed.framework-reference",
            "framework_reference",
            3.0,
            28,
        )
    if (
        info.get("schema") != "sigmax.sd3-sigma-node/1"
        or _object(info.get("profile"), label="SD3 profile").get("id") != profile_id
        or _object(info.get("profile"), label="SD3 profile").get("evidence") != evidence
        or _object(info.get("shift"), label="SD3 shift") != {"kind": "direct_ratio", "ratio": ratio}
        or _object(info.get("slicing"), label="SD3 slicing").get("output_steps") != steps
        or len(sigmas) != steps + 1
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("SD3 H2 execution evidence drifted")
    return {
        "mode": mode,
        "numerical_fingerprint": _object(info.get("fingerprints"), label="SD3 fingerprints").get(
            "complete"
        ),
        "profile_id": profile_id,
        "ratio": ratio,
        "requested_transitions": steps,
        "status": "succeeded",
    }


def build_aura_flow_h2_api_prompt() -> dict[str, object]:
    """Return one model-free original AuraFlow v0.2 scheduler -> probe graph."""

    return {
        "1": {
            "class_type": "Sigmax.AuraFlowSigmaScheduler",
            "inputs": {
                "already_shifted": False,
                "end_step": -1,
                "mode": "Official Fixed (1.73)",
                "start_step": 0,
                "steps": 50,
                "strict_source": True,
            },
        },
        _AURAFLOW_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.AuraFlowScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_aura_flow_h2_history(history: object, *, prompt_id: str) -> dict[str, object]:
    """Verify original AuraFlow v0.2 on the model-free host lane."""

    root = _object(history, label="AuraFlow history")
    entry = _object(root.get(prompt_id), label="AuraFlow history entry")
    status = _object(entry.get("status"), label="AuraFlow prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("AuraFlow prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="AuraFlow prompt outputs")
    output = _object(outputs.get(_AURAFLOW_OUTPUT_NODE_ID), label="AuraFlow probe output")
    traces = _array(output.get(_AURAFLOW_TRACE_KEY), label="AuraFlow probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("AuraFlow probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="AuraFlow decoded trace")
    info = _object(trace.get("schedule_info"), label="AuraFlow schedule information")
    sigmas = _array(trace.get("sigmas"), label="AuraFlow sigma vector")
    if (
        info.get("schema") != "sigmax.aura-flow-sigma-node/1"
        or _object(info.get("profile"), label="AuraFlow profile").get("id")
        != "auraflow.v0-2.official"
        or _object(info.get("profile"), label="AuraFlow profile").get("evidence") != "official"
        or _object(info.get("shift"), label="AuraFlow shift")
        != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": 1.73}
        or _object(info.get("slicing"), label="AuraFlow slicing").get("output_steps") != 50
        or len(sigmas) != 51
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("AuraFlow H2 execution evidence drifted")
    return {
        "mode": "Official Fixed (1.73)",
        "numerical_fingerprint": _object(
            info.get("fingerprints"), label="AuraFlow fingerprints"
        ).get("complete"),
        "profile_id": "auraflow.v0-2.official",
        "ratio": 1.73,
        "requested_transitions": 50,
        "status": "succeeded",
    }


def build_lumina2_h2_api_prompt() -> dict[str, object]:
    """Return one model-free Lumina-Image 2.0 scheduler -> probe graph."""

    return {
        "1": {
            "class_type": "Sigmax.Lumina2SigmaScheduler",
            "inputs": {
                "already_shifted": False,
                "end_step": -1,
                "mode": "Official Fixed (6.0)",
                "start_step": 0,
                "steps": 50,
                "strict_source": True,
            },
        },
        _LUMINA2_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.Lumina2ScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_lumina2_h2_history(history: object, *, prompt_id: str) -> dict[str, object]:
    """Verify Lumina-Image 2.0 on the model-free host lane."""

    root = _object(history, label="Lumina2 history")
    entry = _object(root.get(prompt_id), label="Lumina2 history entry")
    status = _object(entry.get("status"), label="Lumina2 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("Lumina2 prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="Lumina2 prompt outputs")
    output = _object(outputs.get(_LUMINA2_OUTPUT_NODE_ID), label="Lumina2 probe output")
    traces = _array(output.get(_LUMINA2_TRACE_KEY), label="Lumina2 probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("Lumina2 probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="Lumina2 decoded trace")
    info = _object(trace.get("schedule_info"), label="Lumina2 schedule information")
    sigmas = _array(trace.get("sigmas"), label="Lumina2 sigma vector")
    if (
        info.get("schema") != "sigmax.lumina2-sigma-node/1"
        or _object(info.get("profile"), label="Lumina2 profile").get("id") != "lumina2.v2.official"
        or _object(info.get("profile"), label="Lumina2 profile").get("evidence") != "official"
        or _object(info.get("shift"), label="Lumina2 shift")
        != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": 6.0}
        or _object(info.get("slicing"), label="Lumina2 slicing").get("output_steps") != 50
        or len(sigmas) != 51
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("Lumina2 H2 execution evidence drifted")
    return {
        "mode": "Official Fixed (6.0)",
        "numerical_fingerprint": _object(
            info.get("fingerprints"), label="Lumina2 fingerprints"
        ).get("complete"),
        "profile_id": "lumina2.v2.official",
        "ratio": 6.0,
        "requested_transitions": 50,
        "status": "succeeded",
    }


def build_hunyuan_image21_h2_api_prompt(variant: str) -> dict[str, object]:
    """Return one model-free HunyuanImage 2.1 scheduler -> probe graph."""

    if variant not in {"Base (5.0)", "Distilled (4.0)"}:
        raise ScheduleContractError("HunyuanImage 2.1 H2 variant must be explicit")
    steps = 50 if variant == "Base (5.0)" else 8
    return {
        "1": {
            "class_type": "Sigmax.HunyuanImage21SigmaScheduler",
            "inputs": {
                "already_shifted": False,
                "end_step": -1,
                "start_step": 0,
                "steps": steps,
                "strict_source": True,
                "variant": variant,
            },
        },
        _HUNYUAN_IMAGE21_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.HunyuanImage21ScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_hunyuan_image21_h2_history(
    history: object, *, prompt_id: str, variant: str
) -> dict[str, object]:
    """Verify one HunyuanImage 2.1 variant on the model-free host lane."""

    if variant not in {"Base (5.0)", "Distilled (4.0)"}:
        raise ScheduleContractError("HunyuanImage 2.1 H2 variant must be explicit")
    expected_profile = (
        "hunyuan-image-2-1.base.official"
        if variant == "Base (5.0)"
        else "hunyuan-image-2-1.distilled.official"
    )
    expected_ratio = 5.0 if variant == "Base (5.0)" else 4.0
    expected_steps = 50 if variant == "Base (5.0)" else 8
    root = _object(history, label="HunyuanImage 2.1 history")
    entry = _object(root.get(prompt_id), label="HunyuanImage 2.1 history entry")
    status = _object(entry.get("status"), label="HunyuanImage 2.1 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("HunyuanImage 2.1 prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="HunyuanImage 2.1 prompt outputs")
    output = _object(
        outputs.get(_HUNYUAN_IMAGE21_OUTPUT_NODE_ID), label="HunyuanImage 2.1 probe output"
    )
    traces = _array(output.get(_HUNYUAN_IMAGE21_TRACE_KEY), label="HunyuanImage 2.1 probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("HunyuanImage 2.1 probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="HunyuanImage 2.1 decoded trace")
    info = _object(trace.get("schedule_info"), label="HunyuanImage 2.1 schedule information")
    sigmas = _array(trace.get("sigmas"), label="HunyuanImage 2.1 sigma vector")
    profile = _object(info.get("profile"), label="HunyuanImage 2.1 profile")
    if (
        info.get("schema") != "sigmax.hunyuan-image-2-1-sigma-node/1"
        or profile.get("id") != expected_profile
        or profile.get("evidence") != "official"
        or profile.get("variant") != ("2.1" if variant == "Base (5.0)" else "2.1-distilled")
        or _object(info.get("shift"), label="HunyuanImage 2.1 shift")
        != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": expected_ratio}
        or _object(info.get("slicing"), label="HunyuanImage 2.1 slicing").get("output_steps")
        != expected_steps
        or len(sigmas) != expected_steps + 1
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("HunyuanImage 2.1 H2 execution evidence drifted")
    return {
        "variant": variant,
        "numerical_fingerprint": _object(
            info.get("fingerprints"), label="HunyuanImage 2.1 fingerprints"
        ).get("complete"),
        "profile_id": expected_profile,
        "ratio": expected_ratio,
        "requested_transitions": expected_steps,
        "status": "succeeded",
    }


def build_anima_h2_api_prompt(variant: str) -> dict[str, object]:
    """Return one model-free Anima v1 scheduler -> probe graph."""

    if variant not in {"Base (3.0)", "Aesthetic (3.0)", "Turbo (3.0)"}:
        raise ScheduleContractError("Anima H2 variant must be explicit")
    steps = 8 if variant == "Turbo (3.0)" else 50
    return {
        "1": {
            "class_type": "Sigmax.AnimaSigmaScheduler",
            "inputs": {
                "already_shifted": False,
                "end_step": -1,
                "start_step": 0,
                "steps": steps,
                "strict_source": True,
                "variant": variant,
            },
        },
        _ANIMA_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.AnimaScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_anima_h2_history(history: object, *, prompt_id: str, variant: str) -> dict[str, object]:
    """Verify one Anima v1 variant on the model-free host lane."""

    if variant not in {"Base (3.0)", "Aesthetic (3.0)", "Turbo (3.0)"}:
        raise ScheduleContractError("Anima H2 variant must be explicit")
    expected: tuple[str, str, int] = {
        "Base (3.0)": ("anima.base.framework-reference", "base-v1.0", 50),
        "Aesthetic (3.0)": ("anima.aesthetic.framework-reference", "aesthetic-v1", 50),
        "Turbo (3.0)": ("anima.turbo.framework-reference", "turbo-v1.0", 8),
    }[variant]
    root = _object(history, label="Anima history")
    entry = _object(root.get(prompt_id), label="Anima history entry")
    status = _object(entry.get("status"), label="Anima prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("Anima prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="Anima prompt outputs")
    output = _object(outputs.get(_ANIMA_OUTPUT_NODE_ID), label="Anima probe output")
    traces = _array(output.get(_ANIMA_TRACE_KEY), label="Anima probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("Anima probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="Anima decoded trace")
    info = _object(trace.get("schedule_info"), label="Anima schedule information")
    sigmas = _array(trace.get("sigmas"), label="Anima sigma vector")
    profile = _object(info.get("profile"), label="Anima profile")
    steps = expected[2]
    if (
        info.get("schema") != "sigmax.anima-sigma-node/1"
        or profile.get("id") != expected[0]
        or profile.get("variant") != expected[1]
        or profile.get("evidence") != "framework_reference"
        or _object(info.get("shift"), label="Anima shift")
        != {"kind": "rational", "multiplier": 1.0, "shift": 3.0}
        or _object(info.get("slicing"), label="Anima slicing").get("output_steps") != steps
        or len(sigmas) != steps + 1
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("Anima H2 execution evidence drifted")
    return {
        "variant": variant,
        "numerical_fingerprint": _object(info.get("fingerprints"), label="Anima fingerprints").get(
            "complete"
        ),
        "profile_id": expected[0],
        "shift": 3.0,
        "requested_transitions": steps,
        "status": "succeeded",
    }


def build_ltx_h2_api_prompt(
    *, generation: str, stage: str, steps: int, token_count: int = 4096
) -> dict[str, object]:
    """Return one model-free explicit LTX scheduler -> probe graph."""

    allowed = {
        ("LTXV 0.9.8", "Dev", 20),
        ("LTX-2 19B", "Distilled Stage 1", 8),
        ("LTX-2.3 22B", "Dev", 30),
        ("LTX-2.3 22B", "Distilled Stage 2", 3),
    }
    if (generation, stage, steps) not in allowed:
        raise ScheduleContractError("LTX H2 selection must be one of the pinned dense cases")
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 1:
        raise ScheduleContractError("LTX H2 token_count must be a positive integer")
    return {
        "1": {
            "class_type": "Sigmax.LTXSigmaScheduler",
            "inputs": {
                "end_step": -1,
                "generation": generation,
                "stage": stage,
                "steps": steps,
                "start_step": 0,
                "stretch": True,
                "strict_official": True,
                "terminal": 0.1,
                "token_count": token_count,
            },
        },
        _LTX_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.LTXScheduleProbe",
            "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
        },
    }


def verify_ltx_h2_history(
    history: object,
    *,
    prompt_id: str,
    generation: str,
    stage: str,
    steps: int,
) -> dict[str, object]:
    """Verify one model-free LTX trace and explicit generation/stage identity."""

    expected_map: dict[tuple[str, str], tuple[str, int, float | None]] = {
        ("LTXV 0.9.8", "Dev"): ("ltxv.0.9.8.dev", 20, 2.05),
        ("LTX-2 19B", "Distilled Stage 1"): (
            "ltx2.19b.distilled.stage1",
            8,
            None,
        ),
        ("LTX-2.3 22B", "Dev"): ("ltx2.3.22b.dev", 30, 2.05),
        ("LTX-2.3 22B", "Distilled Stage 2"): (
            "ltx2.3.22b.distilled.stage2",
            3,
            None,
        ),
    }
    expected = expected_map.get((generation, stage))
    if expected is None or steps != expected[1]:
        raise ScheduleContractError("LTX H2 selection is unsupported")
    root = _object(history, label="LTX H2 history")
    entry = _object(root.get(prompt_id), label="LTX H2 history entry")
    status = _object(entry.get("status"), label="LTX H2 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("LTX H2 prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="LTX H2 prompt outputs")
    output = _object(outputs.get(_LTX_OUTPUT_NODE_ID), label="LTX H2 probe output")
    traces = _array(output.get(_LTX_TRACE_KEY), label="LTX H2 probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("LTX H2 probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="LTX H2 decoded trace")
    info = _object(trace.get("schedule_info"), label="LTX H2 schedule information")
    sigmas = _array(trace.get("sigmas"), label="LTX H2 sigma vector")
    shift = info.get("shift")
    expected_start = 0.909375 if stage == "Distilled Stage 2" else 1.0
    valid_shift = (
        shift is None
        if expected[2] is None
        else isinstance(shift, int | float) and abs(float(shift) - expected[2]) <= 1e-12
    )
    if (
        info.get("schema") != "sigmax.ltx-sigma-node/1"
        or info.get("generation") != generation
        or info.get("stage") != stage
        or info.get("profile") != expected[0]
        or info.get("slicing", {}).get("output_steps") != expected[1]
        or not valid_shift
        or len(sigmas) != expected[1] + 1
        or abs(sigmas[0] - expected_start) > 1e-6
        or sigmas[-1] != 0.0
        or any(not isinstance(value, int | float) or isinstance(value, bool) for value in sigmas)
        or any(float(left) <= float(right) for left, right in pairwise(sigmas))
    ):
        raise ScheduleContractError("LTX H2 execution evidence drifted")
    return {
        "generation": generation,
        "stage": stage,
        "numerical_fingerprint": _object(info.get("fingerprints"), label="LTX H2 fingerprints").get(
            "complete"
        ),
        "profile_id": expected[0],
        "requested_transitions": expected[1],
        "status": "succeeded",
    }


def build_wan_h2_api_prompt(
    *,
    generation: str,
    task: str,
    source: str,
    resolution: str,
    steps: int,
    strict_source: bool = True,
    start_step: int = 0,
    end_step: int = -1,
) -> dict[str, object]:
    """Return one model-free explicit Wan scheduler -> probe graph."""

    allowed = {
        ("Wan 2.1", "T2V", "Official native", "None", 50),
        ("Wan 2.1", "I2V", "Official native", "480P", 40),
        ("Wan 2.2", "TI2V", "ComfyUI native", "None", 50),
        ("Wan 2.2", "T2V A14B", "Official native", "None", 40),
        ("Wan 2.1", "FLF2V", "Official native", "720P", 50),
        ("Wan 2.1", "VACE 1.3B", "Official native", "None", 50),
        ("Wan 2.1", "VACE 14B", "Official native", "None", 50),
        ("Wan 2.2", "S2V", "Official native", "None", 40),
        ("Wan 2.2", "Animate", "Official native", "None", 20),
        ("Wan Animate 2", "Animate Base", "Official native", "None", 40),
        ("Wan Animate 2", "Animate Distilled", "Official native", "None", 10),
    }
    if (generation, task, source, resolution, steps) not in allowed:
        raise ScheduleContractError("Wan H2 selection must be one of the pinned dense cases")
    if not isinstance(strict_source, bool):
        raise ScheduleContractError("Wan H2 strict_source must be boolean")
    return {
        "1": {
            "class_type": "Sigmax.WanSigmaScheduler",
            "inputs": {
                "already_shifted": False,
                "end_step": end_step,
                "generation": generation,
                "resolution": resolution,
                "source": source,
                "start_step": start_step,
                "steps": steps,
                "strict_source": strict_source,
                "task": task,
            },
        },
        _WAN_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.WanScheduleProbe",
            "inputs": {"schedule_info": ["1", 2], "sigmas": ["1", 0]},
        },
    }


def verify_wan_h2_history(
    history: object,
    *,
    prompt_id: str,
    generation: str,
    task: str,
    source: str,
    resolution: str,
    steps: int,
) -> dict[str, object]:
    """Verify one model-free Wan H2 trace and caller-owned A14B boundary."""

    expected = {
        ("Wan 2.1", "T2V", "Official native", "None", 50): (
            "wan2.1.t2v.official-native",
            "official",
            5.0,
            None,
        ),
        ("Wan 2.1", "I2V", "Official native", "480P", 40): (
            "wan2.1.i2v.480p.official-native",
            "official",
            3.0,
            None,
        ),
        ("Wan 2.2", "TI2V", "ComfyUI native", "None", 50): (
            "wan2.2.ti2v.5b.comfy-native",
            "framework_reference",
            5.0,
            None,
        ),
        ("Wan 2.2", "T2V A14B", "Official native", "None", 40): (
            "wan2.2.t2v-a14b.official-native",
            "official",
            12.0,
            0.875,
        ),
        ("Wan 2.1", "FLF2V", "Official native", "720P", 50): (
            "wan2.1.flf2v.14b.720p.official-native",
            "official",
            16.0,
            None,
        ),
        ("Wan 2.1", "VACE 1.3B", "Official native", "None", 50): (
            "wan2.1.vace.1.3b.official-native",
            "official",
            16.0,
            None,
        ),
        ("Wan 2.1", "VACE 14B", "Official native", "None", 50): (
            "wan2.1.vace.14b.official-native",
            "official",
            16.0,
            None,
        ),
        ("Wan 2.2", "S2V", "Official native", "None", 40): (
            "wan2.2.s2v.14b.official-native",
            "official",
            3.0,
            None,
        ),
        ("Wan 2.2", "Animate", "Official native", "None", 20): (
            "wan2.2.animate.14b.official-native",
            "official",
            5.0,
            None,
        ),
        ("Wan Animate 2", "Animate Base", "Official native", "None", 40): (
            "wan-animate2.14b.base.official-native",
            "official",
            5.0,
            None,
        ),
        ("Wan Animate 2", "Animate Distilled", "Official native", "None", 10): (
            "wan-animate2.14b.distilled.official-native",
            "official",
            5.0,
            None,
        ),
    }
    selection = expected.get((generation, task, source, resolution, steps))
    if selection is None:
        raise ScheduleContractError("Wan H2 selection is unsupported")
    expected_profile, expected_evidence, expected_ratio, expected_boundary = selection
    root = _object(history, label="Wan H2 history")
    entry = _object(root.get(prompt_id), label="Wan H2 history entry")
    status = _object(entry.get("status"), label="Wan H2 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("Wan H2 prompt history does not prove success")
    outputs = _object(entry.get("outputs"), label="Wan H2 prompt outputs")
    output = _object(outputs.get(_WAN_OUTPUT_NODE_ID), label="Wan H2 probe output")
    traces = _array(output.get(_WAN_TRACE_KEY), label="Wan H2 probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("Wan H2 probe trace is malformed")
    trace = _object(json.loads(traces[0]), label="Wan H2 decoded trace")
    info = _object(trace.get("schedule_info"), label="Wan H2 schedule information")
    profile = _object(info.get("profile"), label="Wan H2 profile")
    shift = _object(info.get("shift"), label="Wan H2 shift")
    slicing = _object(info.get("slicing"), label="Wan H2 slicing")
    boundary = _object(info.get("boundary"), label="Wan H2 boundary")
    fingerprints = _object(info.get("fingerprints"), label="Wan H2 fingerprints")
    sigmas = _array(trace.get("sigmas"), label="Wan H2 sigma vector")
    float_sigmas = tuple(float(value) for value in sigmas)
    if (
        info.get("schema") != "sigmax.wan-sigma-node/1"
        or profile.get("id") != expected_profile
        or profile.get("evidence") != expected_evidence
        or shift != {"kind": "direct_ratio", "multiplier": 1.0, "ratio": expected_ratio}
        or info.get("strict_source") is not True
        or slicing.get("output_steps") != steps
        or len(float_sigmas) != steps + 1
        or float_sigmas[0] != 1.0
        or float_sigmas[-1] != 0.0
        or any(left <= right for left, right in pairwise(float_sigmas))
        or boundary.get("model_dispatch") is not False
        or boundary.get("routing_owner") != "caller"
        or (expected_boundary is None and boundary.get("step") != -1)
        or (expected_boundary is not None and boundary.get("normalized") != expected_boundary)
        or fingerprints.get("output")
        != sigma_output_fingerprint(float_sigmas, domain=SigmaDomain.UNIT_FLOW)
    ):
        raise ScheduleContractError("Wan H2 execution evidence drifted")
    return {
        "boundary": expected_boundary,
        "numerical_fingerprint": fingerprints.get("complete"),
        "output_fingerprint": fingerprints.get("output"),
        "profile_id": expected_profile,
        "requested_transitions": steps,
        "status": "succeeded",
    }


def build_schedule_algebra_h2_api_prompt() -> dict[str, object]:
    """Execute slice, exact-boundary concat, resample, and fingerprint inspection."""

    return {
        "1": build_turbo_api_prompt()["1"],
        "2": {
            "class_type": "Sigmax.ScheduleSlice",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
                "start_step": 0,
                "end_step": 4,
            },
        },
        "3": {
            "class_type": "Sigmax.ScheduleSlice",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
                "start_step": 4,
                "end_step": 8,
            },
        },
        "4": {
            "class_type": "Sigmax.ScheduleConcatenate",
            "inputs": {
                "sigmas_left": ["2", 0],
                "schedule_info_left": ["2", 1],
                "sigmas_right": ["3", 0],
                "schedule_info_right": ["3", 1],
            },
        },
        "5": {
            "class_type": "Sigmax.ScheduleResample",
            "inputs": {
                "sigmas": ["4", 0],
                "schedule_info": ["4", 1],
                "output_steps": 4,
            },
        },
        "6": {
            "class_type": "Sigmax.ScheduleInspector",
            "inputs": {"sigmas": ["5", 0], "schedule_info": ["5", 1]},
        },
        _ALGEBRA_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.ScheduleAlgebraProbe",
            "inputs": {
                "sigmas": ["5", 0],
                "schedule_info": ["5", 1],
                "schedule_report": ["6", 0],
            },
        },
    }


def build_schedule_algebra_h2_noop_rejection_prompt() -> dict[str, object]:
    """Return an algebra graph whose explicit resample is a forbidden no-op."""

    prompt = build_schedule_algebra_h2_api_prompt()
    resample = _object(prompt["5"], label="algebra resample prompt node")
    inputs = _object(resample["inputs"], label="algebra resample prompt inputs")
    inputs["output_steps"] = 8
    resample["inputs"] = inputs
    prompt["5"] = resample
    return prompt


def build_checkpoint_evidence_h2_api_prompt() -> dict[str, object]:
    """Return the header-only checkpoint inspector -> test output graph."""

    return {
        "1": {
            "class_type": "Sigmax.CheckpointEvidenceInspector",
            "inputs": {"checkpoint": f"checkpoints::{_CHECKPOINT_FIXTURE_NAME}"},
        },
        _CHECKPOINT_OUTPUT_NODE_ID: {
            "class_type": "SigmaxTest.CheckpointEvidenceProbe",
            "inputs": {"checkpoint_evidence": ["1", 0]},
        },
    }


def build_native_euler_h3_api_prompt() -> dict[str, object]:
    """Return the controlled scheduler -> artifact output + native Euler H3 graph."""

    prompt = build_turbo_api_prompt()
    prompt[_H3_OUTPUT_NODE_ID] = {
        "class_type": "SigmaxTest.NativeEulerProbe",
        "inputs": {
            "sigmas": ["1", 0],
            "schedule_info": ["1", 1],
        },
    }
    return prompt


def build_native_euler_h3_partial_rejection_prompt() -> dict[str, object]:
    """Return a partial schedule that M5-01 must reject instead of misclaiming."""

    prompt = build_native_euler_h3_api_prompt()
    scheduler = cast(dict[str, object], prompt["1"])
    inputs = cast(dict[str, object], scheduler["inputs"])
    inputs["start_step"] = 1
    return {
        "1": scheduler,
        _H3_OUTPUT_NODE_ID: prompt[_H3_OUTPUT_NODE_ID],
    }


def _raw_case(case_id: str) -> dict[str, object]:
    case = _RAW_CASES.get(case_id)
    if case is None:
        raise ScheduleContractError("RAW host case ID is unsupported")
    return dict(case)


def build_raw_api_prompt(case_id: str) -> dict[str, object]:
    """Return one exact model-free RAW scheduler -> inspector -> output API graph."""

    case = _raw_case(case_id)
    return {
        "1": {
            "class_type": "Sigmax.Krea2SigmaScheduler",
            "inputs": {
                "variant": "RAW",
                "steps": case["steps"],
                "width": case["width"],
                "height": case["height"],
                "strict_official": case["strict_official"],
                "start_step": 0,
                "end_step": -1,
            },
        },
        "2": {
            "class_type": "Sigmax.ScheduleInspector",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
            },
        },
        "3": {
            "class_type": "Sigmax.RawWorkflowOutput",
            "inputs": {
                "sigmas": ["1", 0],
                "schedule_info": ["1", 1],
                "schedule_report": ["2", 0],
            },
        },
    }


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScheduleContractError(f"{label} must be an object")
    return dict(cast(Mapping[str, Any], value))


def _array(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScheduleContractError(f"{label} must be an array")
    return value


def verify_turbo_history(
    history: object,
    *,
    prompt_id: str,
) -> dict[str, object]:
    """Require completed history and verify the final canonical Turbo bundle."""

    root = _object(history, label="history")
    entry = _object(root.get(prompt_id), label="prompt history entry")
    status = _object(entry.get("status"), label="prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("prompt history does not prove completed success")
    outputs = _object(entry.get("outputs"), label="prompt outputs")
    output = _object(outputs.get(_OUTPUT_NODE_ID), label="Turbo output-node history")
    bundle_values = _array(output.get(_BUNDLE_KEY), label="Turbo execution bundle")
    if len(bundle_values) != 1 or not isinstance(bundle_values[0], str):
        raise ScheduleContractError("Turbo execution bundle history is malformed")

    bundle = deserialize_portable_execution_bundle(bundle_values[0])
    construction = bundle.artifact.construction_projection()
    receipt = bundle.receipt.projection()
    execution = _object(receipt.get("execution"), label="receipt execution")
    counts = _object(receipt.get("counts"), label="receipt counts")
    ownership = _object(construction.get("ownership"), label="artifact ownership")
    transforms = _array(construction.get("transforms"), label="artifact transforms")
    shift_count = sum(
        _object(item, label="artifact transform").get("id") == "krea.exponential_mu"
        for item in transforms
    )

    if execution != {
        "reason_code": None,
        "status": ExecutionStatus.NOT_EXECUTED.value,
    }:
        raise ScheduleContractError("model-free receipt status is not truthful")
    if counts != {
        "effective_model_evaluations": 0,
        "effective_transitions": 0,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }:
        raise ScheduleContractError("model-free receipt counts are not canonical")
    if ownership != {
        "schedule": "external_sigmas",
        "shift": "construction_pipeline",
    }:
        raise ScheduleContractError("artifact ownership permits an implicit second shift")
    if shift_count != 1:
        raise ScheduleContractError("artifact does not contain exactly one time shift")
    if bundle.artifact.numerical_fingerprint != _EXPECTED_NUMERICAL_FINGERPRINT:
        raise ScheduleContractError("executed Turbo schedule fingerprint drifted")

    return {
        "artifact_construction_fingerprint": bundle.artifact.construction_fingerprint,
        "effective_model_evaluations": counts["effective_model_evaluations"],
        "effective_transitions": counts["effective_transitions"],
        "numerical_fingerprint": bundle.artifact.numerical_fingerprint,
        "receipt_fingerprint": bundle.receipt.receipt_fingerprint,
        "requested_transitions": counts["requested_transitions"],
        "schedule_ownership": ownership["schedule"],
        "shift_count": shift_count,
        "status": execution["status"],
    }


def verify_schedule_algebra_h2_history(
    history: object,
    *,
    prompt_id: str,
) -> dict[str, object]:
    """Verify host-executed algebra values, modified evidence, and inspector identity."""

    root = _object(history, label="algebra history")
    entry = _object(root.get(prompt_id), label="algebra prompt history entry")
    status = _object(entry.get("status"), label="algebra prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("algebra prompt history does not prove completed success")
    outputs = _object(entry.get("outputs"), label="algebra prompt outputs")
    output = _object(
        outputs.get(_ALGEBRA_OUTPUT_NODE_ID),
        label="algebra output-node history",
    )
    traces = _array(output.get(_ALGEBRA_TRACE_KEY), label="algebra execution trace")
    if len(traces) != 1 or not isinstance(traces[0], str):
        raise ScheduleContractError("algebra execution trace is malformed")
    trace = _object(_decode_json(traces[0].encode(), label="algebra trace"), label="algebra trace")
    info = _object(trace.get("schedule_info"), label="algebra schedule information")
    report = _object(trace.get("schedule_report"), label="algebra schedule report")
    values = _array(trace.get("sigmas"), label="algebra sigmas")

    source = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    expected = [
        struct.unpack(">f", struct.pack(">f", source.sigmas[index]))[0] for index in range(0, 9, 2)
    ]
    if values != expected:
        raise ScheduleContractError(
            "host algebra sigma values drifted from explicit index resample"
        )
    expected_fingerprint = sigma_output_fingerprint(
        tuple(expected),
        domain=SigmaDomain.UNIT_FLOW,
    )
    fingerprints = _object(info.get("fingerprints"), label="algebra fingerprints")
    report_fingerprints = _object(
        report.get("fingerprints"),
        label="algebra report fingerprints",
    )
    if (
        info.get("schema") != "sigmax.schedule-resample-node/1"
        or info.get("operation") != "resample"
        or info.get("evidence") != "modified"
        or info.get("parameters")
        != {"input_steps": 8, "method": "index_linear_v1", "output_steps": 4}
        or fingerprints.get("output") != expected_fingerprint
        or report.get("source_schema") != "sigmax.schedule-resample-node/1"
        or report_fingerprints.get("computed_output") != expected_fingerprint
        or report_fingerprints.get("verified") is not True
    ):
        raise ScheduleContractError("host algebra contract or fingerprint verification drifted")
    return {
        "evidence": "modified",
        "method": "index_linear_v1",
        "numerical_fingerprint": expected_fingerprint,
        "operations": ["slice", "concatenate", "resample", "inspect"],
        "status": "succeeded",
        "transitions": 4,
    }


def verify_checkpoint_evidence_h2_history(
    history: object,
    *,
    prompt_id: str,
) -> dict[str, object]:
    """Require one completed, path-free, suggestion-only checkpoint inspection."""

    root = _object(history, label="checkpoint prompt history")
    entry = _object(root.get(prompt_id), label="checkpoint prompt history entry")
    status = _object(entry.get("status"), label="checkpoint prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("checkpoint prompt history does not prove completed success")
    prompt_tuple = _array(entry.get("prompt"), label="checkpoint retained prompt tuple")
    if len(prompt_tuple) < 3 or prompt_tuple[2] != build_checkpoint_evidence_h2_api_prompt():
        raise ScheduleContractError("checkpoint prompt history retained a stale API graph")
    outputs = _object(entry.get("outputs"), label="checkpoint prompt outputs")
    output = _object(
        outputs.get(_CHECKPOINT_OUTPUT_NODE_ID),
        label="checkpoint output-node history",
    )
    traces = _array(output.get(_CHECKPOINT_TRACE_KEY), label="checkpoint execution trace")
    if len(traces) != 1 or not isinstance(traces[0], str) or len(traces[0]) > 100_000:
        raise ScheduleContractError("checkpoint execution trace is invalid")
    report = _object(_decode_json(traces[0].encode(), label="checkpoint report"), label="report")
    canonical = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    source = _object(report.get("source"), label="checkpoint report source")
    structure = _object(report.get("structure"), label="checkpoint report structure")
    identity = _object(report.get("model_identity"), label="checkpoint report identity")
    reason_codes = _array(report.get("reason_codes"), label="checkpoint reason codes")
    if (
        traces[0] != canonical
        or report.get("schema") != "sigmax.checkpoint-evidence-inspection/1"
        or report.get("status") != "inspected"
        or source.get("display_name") != f"checkpoints::{_CHECKPOINT_FIXTURE_NAME}"
        or source.get("format") != "safetensors"
        or source.get("payload_bytes_read") != 0
        or structure.get("data_bytes") != 8
        or structure.get("dtype_counts") != {"F16": 4}
        or structure.get("tensor_count") != 4
        or identity.get("confidence") != "corroborating"
        or identity.get("confirmed_variant") is not None
        or identity.get("resolution_status") != "suggested"
        or identity.get("suggested_variant") != "turbo"
        or reason_codes != identity.get("reason_codes")
        or not all(isinstance(item, str) for item in reason_codes)
    ):
        raise ScheduleContractError("checkpoint execution evidence drifted")
    return {
        "confidence": identity["confidence"],
        "confirmed_variant": None,
        "payload_bytes_read": 0,
        "reason_codes": reason_codes,
        "status": "succeeded",
        "suggested_variant": identity["suggested_variant"],
        "tensor_count": structure["tensor_count"],
    }


def verify_schedule_algebra_h2_noop_rejection(
    history: object,
    *,
    prompt_id: str,
) -> dict[str, object]:
    """Require terminal runtime rejection for an explicit no-op resample."""

    root = _object(history, label="rejected algebra history")
    entry = _object(root.get(prompt_id), label="rejected algebra history entry")
    status = _object(entry.get("status"), label="rejected algebra status")
    if status.get("completed") is not False or status.get("status_str") != "error":
        raise ScheduleContractError("rejected algebra history is not a terminal error")
    prompt_tuple = _array(entry.get("prompt"), label="rejected algebra prompt tuple")
    if (
        len(prompt_tuple) < 3
        or prompt_tuple[2] != build_schedule_algebra_h2_noop_rejection_prompt()
    ):
        raise ScheduleContractError("rejected algebra history retained a stale API graph")
    if _object(entry.get("outputs"), label="rejected algebra outputs"):
        raise ScheduleContractError("rejected algebra produced partial output")
    messages = _array(status.get("messages"), label="rejected algebra messages")
    events = [_array(item, label="rejected algebra event") for item in messages]
    if not events or events[-1][0] != "execution_error":
        raise ScheduleContractError("rejected algebra has no terminal execution error")
    detail = _object(events[-1][1], label="rejected algebra error detail")
    if (
        detail.get("prompt_id") != prompt_id
        or detail.get("node_id") != "5"
        or detail.get("node_type") != "Sigmax.ScheduleResample"
        or detail.get("exception_type")
        != "comfyui_sigmax.core.schedule_contracts.ScheduleContractError"
        or detail.get("exception_message") != "resampling must change the transition count\n"
        or not isinstance(detail.get("executed"), list)
        or not isinstance(detail.get("current_outputs"), list)
    ):
        raise ScheduleContractError("rejected algebra error evidence drifted")
    return {
        "boundary": "runtime_execution",
        "case_id": "algebra-noop-resample",
        "exception_type": detail["exception_type"],
        "partial_output": False,
        "prompt_created": True,
        "reason_code": "input.algebra_noop",
        "status": status["status_str"],
    }


def _component_fingerprint(projection: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(canonical_projection_bytes(dict(projection))).hexdigest()


def _typed_host_summary(value: object, *, depth: int = 0) -> object:
    if depth > 32:
        raise ScheduleContractError("verified host summary exceeds maximum depth")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScheduleContractError("verified host summary float must be finite")
        normalized = 0.0 if value == 0.0 else value
        return {
            "bits": struct.pack(">d", normalized).hex(),
            "precision": "float64",
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ScheduleContractError("verified host summary keys must be strings")
        return {
            key: _typed_host_summary(child, depth=depth + 1)
            for key, child in cast(Mapping[str, object], value).items()
        }
    if isinstance(value, list | tuple):
        return [_typed_host_summary(child, depth=depth + 1) for child in value]
    return value


def _verified_host_attempt(
    summary: object,
    *,
    ordinal: int,
) -> HostAttempt:
    projection = _object(summary, label="verified host attempt summary")
    status = projection.get("status")
    if not isinstance(status, str) or not status:
        raise ScheduleContractError("verified host attempt status is missing")
    reason_code = projection.get("reason_code")
    if reason_code is not None and not isinstance(reason_code, str):
        raise ScheduleContractError("verified host attempt reason code is invalid")
    if status in {"error", "failed", "rejected"} and not reason_code:
        raise ScheduleContractError("verified rejected host attempt lacks a stable reason code")
    try:
        return HostAttempt(
            ordinal=ordinal,
            verdict="pass",
            observed_status=status,
            reason_code=reason_code,
            result_fingerprint=_component_fingerprint(
                cast(Mapping[str, object], _typed_host_summary(projection))
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError(f"verified host attempt evidence is invalid: {exc}") from exc


def build_verified_host_repeat_transition(
    *,
    lane: str,
    first_summary: object,
    repeat_summary: object,
) -> dict[str, object]:
    """Bind two explicit verified submissions; never mask first-attempt drift."""

    try:
        transition = build_host_attempt_transition(
            lane=lane,
            first=_verified_host_attempt(first_summary, ordinal=1),
            repeat=_verified_host_attempt(repeat_summary, ordinal=2),
        )
        validate_host_attempt_transition(transition)
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError(f"host repeat transition evidence is invalid: {exc}") from exc
    if transition["accepted"] is not True:
        raise ScheduleContractError("host repeat changed or masked the first-attempt result")
    return transition


def execute_verified_host_repeat(
    *,
    lane: str,
    submit: Callable[[int], tuple[str, dict[str, object]]],
    verify: Callable[[object, str], dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Execute two explicit submissions; a failed first attempt is never retried."""

    first_prompt_id, first_history = submit(1)
    first_summary = verify(first_history, first_prompt_id)
    repeat_prompt_id, repeat_history = submit(2)
    repeat_summary = verify(repeat_history, repeat_prompt_id)
    transition = build_verified_host_repeat_transition(
        lane=lane,
        first_summary=first_summary,
        repeat_summary=repeat_summary,
    )
    return first_summary, transition


def verify_minimax_h3_h2_history(
    history: object,
    *,
    prompt_id: str,
    variant: str,
) -> dict[str, object]:
    """Verify one model-free host execution of the explicit MiniMax H3 sigma node."""

    if variant not in {"H3 Base FL2VA", "H3 Base Ref2VA"}:
        raise ScheduleContractError("MiniMax H3 H2 variant must be selected explicitly")
    expected = build_minimax_h3_sigma_schedule(
        variant=variant,
        steps=20,
        start_step=0,
        end_step=-1,
    )
    root = _object(history, label="MiniMax H3 H2 history")
    entry = _object(root.get(prompt_id), label="MiniMax H3 H2 history entry")
    status = _object(entry.get("status"), label="MiniMax H3 H2 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("MiniMax H3 H2 history does not prove completed success")
    prompt_tuple = _array(entry.get("prompt"), label="MiniMax H3 H2 retained prompt tuple")
    if len(prompt_tuple) < 3 or prompt_tuple[2] != build_minimax_h3_h2_api_prompt(variant):
        raise ScheduleContractError("MiniMax H3 H2 retained API graph or explicit variant is stale")
    outputs = _object(entry.get("outputs"), label="MiniMax H3 H2 prompt outputs")
    output = _object(
        outputs.get(_MINIMAX_H3_OUTPUT_NODE_ID),
        label="MiniMax H3 H2 probe output",
    )
    traces = _array(output.get(_MINIMAX_H3_TRACE_KEY), label="MiniMax H3 H2 probe trace")
    if len(traces) != 1 or not isinstance(traces[0], str) or len(traces[0]) > 100_000:
        raise ScheduleContractError("MiniMax H3 H2 probe trace is malformed")
    trace = _object(
        _decode_json(traces[0].encode(), label="MiniMax H3 H2 trace"),
        label="MiniMax H3 H2 decoded trace",
    )
    canonical = json.dumps(
        trace,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    info = _object(trace.get("schedule_info"), label="MiniMax H3 H2 schedule information")
    sigmas = _array(trace.get("sigmas"), label="MiniMax H3 H2 sigma vector")
    expected_info = _object(
        json.loads(expected.schedule_info_json),
        label="MiniMax H3 H2 expected schedule information",
    )
    if (
        traces[0] != canonical
        or info != expected_info
        or len(sigmas) != len(expected.sigmas)
        or tuple(float(value) for value in sigmas) != expected.sigmas
        or any(
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in sigmas
        )
    ):
        raise ScheduleContractError("MiniMax H3 H2 execution evidence drifted")
    profile = _object(info.get("profile"), label="MiniMax H3 H2 profile")
    audio = _object(info.get("audio"), label="MiniMax H3 H2 audio ownership")
    counts = _object(info.get("counts"), label="MiniMax H3 H2 count metadata")
    return {
        "audio_ownership": audio.get("ownership"),
        "effective_steps": counts.get("effective_steps"),
        "effective_transitions": counts.get("effective_transitions"),
        "lane": info.get("lane"),
        "numerical_fingerprint": _object(
            info.get("fingerprints"), label="MiniMax H3 H2 fingerprints"
        ).get("complete"),
        "profile_id": profile.get("id"),
        "requested_grid_points": counts.get("requested_grid_points"),
        "requested_steps": counts.get("requested_steps"),
        "requested_transitions": counts.get("requested_transitions"),
        "status": "succeeded",
        "variant": variant,
    }


def verify_native_euler_h3_history(
    history: object,
    *,
    prompt_id: str,
) -> dict[str, object]:
    """Validate actual native Euler trace, then and only then build success evidence."""

    root = _object(history, label="H3 history")
    entry = _object(root.get(prompt_id), label="H3 prompt history entry")
    status = _object(entry.get("status"), label="H3 prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("H3 history does not prove completed success")
    prompt_tuple = _array(entry.get("prompt"), label="H3 retained prompt tuple")
    if len(prompt_tuple) < 3 or prompt_tuple[2] != build_native_euler_h3_api_prompt():
        raise ScheduleContractError("H3 retained API graph is missing or stale")
    outputs = _object(entry.get("outputs"), label="H3 prompt outputs")

    artifact_output = _object(
        outputs.get(_OUTPUT_NODE_ID),
        label="H3 artifact output",
    )
    bundle_values = _array(
        artifact_output.get(_BUNDLE_KEY),
        label="H3 construction bundle",
    )
    if len(bundle_values) != 1 or not isinstance(bundle_values[0], str):
        raise ScheduleContractError("H3 construction bundle is malformed")
    construction_bundle = deserialize_portable_execution_bundle(bundle_values[0])

    trace_output = _object(
        outputs.get(_H3_OUTPUT_NODE_ID),
        label="H3 native Euler output",
    )
    trace_values = _array(trace_output.get(_H3_TRACE_KEY), label="H3 native Euler trace")
    if len(trace_values) != 1 or not isinstance(trace_values[0], str):
        raise ScheduleContractError("H3 native Euler trace is malformed")
    try:
        trace_bytes = trace_values[0].encode("utf-8")
    except UnicodeError as exc:
        raise ScheduleContractError("H3 native Euler trace is not valid Unicode") from exc
    trace = _decode_json(trace_bytes, label="H3 native Euler trace")
    try:
        report = build_native_euler_report(trace)
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("H3 native Euler trace failed parity validation") from exc
    case = _object(report["case"], label="H3 validated Euler case")
    counts = _object(case["counts"], label="H3 validated execution counts")

    artifact = construction_bundle.artifact
    construction = artifact.construction_projection()
    ownership = _object(construction.get("ownership"), label="H3 artifact ownership")
    transforms = _array(construction.get("transforms"), label="H3 artifact transforms")
    shift_count = sum(
        _object(item, label="H3 artifact transform").get("id") == "krea.exponential_mu"
        for item in transforms
    )
    if ownership != {
        "schedule": "external_sigmas",
        "shift": "construction_pipeline",
    }:
        raise ScheduleContractError("H3 artifact ownership permits an implicit second shift")
    if shift_count != 1:
        raise ScheduleContractError("H3 artifact must contain exactly one time shift")
    if artifact.numerical_fingerprint != _EXPECTED_NUMERICAL_FINGERPRINT:
        raise ScheduleContractError("H3 artifact numerical fingerprint drifted")

    compatibility = evaluate_compatibility(
        model=KREA2_TURBO_SCHEMA.model_capabilities,
        profile=KREA2_TURBO_SCHEMA.profile_capabilities,
        sampler=KREA2_TURBO_SCHEMA.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )
    receipt = build_execution_receipt(
        artifact,
        metadata=ExecutionReceiptMetadata(
            compatibility=compatibility,
            host=ExecutionHost(
                identifier="comfyui",
                version=CANONICAL_HOST_VERSION,
                revision=CANONICAL_HOST_REVISION,
                api_version="legacy_v1",
            ),
            model=ExecutionComponent(
                identifier="sigmax.controlled-flow-model",
                version="1",
                fingerprint=_component_fingerprint(
                    {
                        "biases": ["0.0625", "-0.125", "0.1875", "-0.25"],
                        "fixture": "m5-01",
                        "formula": "x*0.125+sigma*0.25+bias+(index+1)*0.03125",
                    }
                ),
            ),
            sampler=ExecutionComponent(
                identifier="comfy.euler",
                version=CANONICAL_HOST_VERSION,
                fingerprint=_component_fingerprint(
                    {
                        "blobs": [
                            {"blob": blob, "path": path}
                            for path, blob in sorted(NATIVE_EULER_SOURCE_BLOBS.items())
                        ],
                        "host_revision": CANONICAL_HOST_REVISION,
                        "function": "comfy.k_diffusion.sampling.sample_euler",
                    }
                ),
            ),
            rng_ownership=ExecutionRngOwnership(
                schedule=NoiseOwnership.NONE,
                model=NoiseOwnership.NONE,
                sampler=NoiseOwnership.NONE,
            ),
            requested_transitions=cast(int, counts["requested_transitions"]),
            effective_transitions=cast(int, counts["effective_transitions"]),
            requested_model_evaluations=cast(int, counts["requested_model_evaluations"]),
            effective_model_evaluations=cast(int, counts["effective_model_evaluations"]),
            status=ExecutionStatus.SUCCEEDED,
        ),
    )
    # Cross-link construction and success evidence without replacing the truthful H2 receipt.
    PortableExecutionBundle(artifact=artifact, receipt=receipt)
    return {
        "artifact_construction_fingerprint": artifact.construction_fingerprint,
        "artifact_numerical_fingerprint": artifact.numerical_fingerprint,
        "counts": dict(counts),
        "deterministic_rerun": case["deterministic_rerun"],
        "final_state": case["native_final"],
        "native_step_count": len(_array(case["native_steps"], label="H3 native steps")),
        "noise_ownership": {
            "model": "none",
            "sampler": "none",
            "schedule": "none",
        },
        "receipt": receipt.projection(),
        "receipt_fingerprint": receipt.receipt_fingerprint,
        "sampler_id": "comfy.euler",
        "schedule_ownership": ownership["schedule"],
        "shift_count": shift_count,
        "status": "succeeded",
        "trace_fingerprint": case["trace_fingerprint"],
        "unsupported_features": [
            "advanced_workflows",
            "partial_denoise_execution",
            "resume",
            "stochastic_euler",
        ],
    }


def verify_native_euler_h3_partial_rejection(
    history: object,
    *,
    prompt_id: str,
) -> dict[str, object]:
    """Require partial-denoise execution to fail at the test-only H3 boundary."""

    root = _object(history, label="H3 partial rejection history")
    entry = _object(root.get(prompt_id), label="H3 partial rejection entry")
    status = _object(entry.get("status"), label="H3 partial rejection status")
    if status.get("completed") is not False or status.get("status_str") != "error":
        raise ScheduleContractError("H3 partial schedule is not a terminal rejection")
    prompt_tuple = _array(entry.get("prompt"), label="H3 partial retained prompt tuple")
    if len(prompt_tuple) < 3 or prompt_tuple[2] != build_native_euler_h3_partial_rejection_prompt():
        raise ScheduleContractError("H3 partial rejection retained a stale graph")
    if _object(entry.get("outputs"), label="H3 partial outputs"):
        raise ScheduleContractError("H3 partial rejection produced a partial output")
    messages = _array(status.get("messages"), label="H3 partial rejection messages")
    events = [_array(item, label="H3 partial rejection event") for item in messages]
    if [item[0] if item else None for item in events] != [
        "execution_start",
        "execution_cached",
        "execution_error",
    ]:
        raise ScheduleContractError("H3 partial rejection event sequence drifted")
    cached = _object(events[1][1], label="H3 partial cached event")
    cached_nodes = cached.get("nodes")
    if cached.get("prompt_id") != prompt_id or not isinstance(cached_nodes, list):
        raise ScheduleContractError("H3 partial cached evidence drifted")
    detail = _object(events[2][1], label="H3 partial error detail")
    if detail.get("prompt_id") != prompt_id:
        raise ScheduleContractError("H3 partial rejection prompt identity drifted")
    if (
        detail.get("node_id") != _H3_OUTPUT_NODE_ID
        or detail.get("node_type") != "SigmaxTest.NativeEulerProbe"
    ):
        raise ScheduleContractError("H3 partial rejection node identity drifted")
    if (
        detail.get("exception_type") != "ValueError"
        or detail.get("exception_message")
        != "H3 sigmas must be one float32 eight-transition schedule\n"
    ):
        raise ScheduleContractError("H3 partial rejection exception contract drifted")
    executed_nodes = detail.get("executed")
    if (cached_nodes, executed_nodes) not in (([], ["1"]), (["1"], [])):
        raise ScheduleContractError("H3 partial rejection scheduler evidence drifted")
    current_outputs = detail.get("current_outputs")
    if (
        not isinstance(current_outputs, list)
        or _H3_OUTPUT_NODE_ID not in current_outputs
        or _OUTPUT_NODE_ID in current_outputs
    ):
        raise ScheduleContractError("H3 partial rejection output-node evidence drifted")
    return {
        "case_id": "partial_denoise_execution",
        "exception_type": detail["exception_type"],
        "node_id": detail["node_id"],
        "partial_output": False,
        "reason_code": "execution.partial_denoise_unsupported",
        "receipt_created": False,
        "status": status["status_str"],
    }


def verify_raw_history(
    history: object,
    *,
    prompt_id: str,
    case_id: str,
    submitted_workflow: object,
) -> dict[str, object]:
    """Verify one completed RAW bundle and the exact submitted workflow metadata reload."""

    case = _raw_case(case_id)
    root = _object(history, label="history")
    entry = _object(root.get(prompt_id), label="RAW prompt history entry")
    status = _object(entry.get("status"), label="RAW prompt status")
    if status.get("completed") is not True or status.get("status_str") != "success":
        raise ScheduleContractError("RAW prompt history does not prove completed success")

    prompt_tuple = _array(entry.get("prompt"), label="RAW retained prompt tuple")
    if len(prompt_tuple) < 4 or prompt_tuple[2] != build_raw_api_prompt(case_id):
        raise ScheduleContractError("RAW retained API graph is missing or stale")
    extra_data = _object(prompt_tuple[3], label="RAW retained extra_data")
    extra_pnginfo = _object(extra_data.get("extra_pnginfo"), label="RAW retained extra_pnginfo")
    if extra_pnginfo.get("workflow") != submitted_workflow:
        raise ScheduleContractError("RAW workflow metadata did not survive history reload")

    outputs = _object(entry.get("outputs"), label="RAW prompt outputs")
    output = _object(outputs.get(_OUTPUT_NODE_ID), label="RAW output-node history")
    bundle_values = _array(output.get(_BUNDLE_KEY), label="RAW execution bundle")
    if len(bundle_values) != 1 or not isinstance(bundle_values[0], str):
        raise ScheduleContractError("RAW execution bundle history is malformed")

    bundle = deserialize_portable_execution_bundle(bundle_values[0])
    construction = bundle.artifact.construction_projection()
    receipt = bundle.receipt.projection()
    execution = _object(receipt.get("execution"), label="RAW receipt execution")
    counts = _object(receipt.get("counts"), label="RAW receipt counts")
    requested = _object(construction.get("requested"), label="RAW requested geometry")
    effective = _object(construction.get("effective"), label="RAW effective geometry")
    base_grid = _object(construction.get("base_grid"), label="RAW base grid")
    parameters = _object(base_grid.get("parameters"), label="RAW base-grid parameters")
    evidence = _object(construction.get("evidence"), label="RAW evidence")
    ownership = _object(construction.get("ownership"), label="RAW artifact ownership")
    transforms = _array(construction.get("transforms"), label="RAW artifact transforms")
    shift_count = sum(
        _object(item, label="RAW artifact transform").get("id") == "krea.exponential_mu"
        for item in transforms
    )
    shift = next(
        (
            _object(item, label="RAW shift transform")
            for item in transforms
            if _object(item, label="RAW artifact transform").get("id") == "krea.exponential_mu"
        ),
        None,
    )
    if shift is None:
        raise ScheduleContractError("RAW artifact has no dynamic shift")
    shift_parameters = _object(shift.get("parameters"), label="RAW shift parameters")
    expected_mu = cast(float, case["mu"])
    expected_mu_value = {
        "bits": struct.pack(">d", expected_mu).hex(),
        "precision": "float64",
    }

    steps = cast(int, case["steps"])
    expected_counts = {
        "effective_model_evaluations": 0,
        "effective_transitions": 0,
        "requested_model_evaluations": steps,
        "requested_transitions": steps,
    }
    if execution != {
        "reason_code": None,
        "status": ExecutionStatus.NOT_EXECUTED.value,
    }:
        raise ScheduleContractError("RAW model-free receipt status is not truthful")
    if counts != expected_counts:
        raise ScheduleContractError("RAW model-free receipt counts are not canonical")
    if {
        "width": requested.get("width"),
        "height": requested.get("height"),
    } != {"width": case["width"], "height": case["height"]}:
        raise ScheduleContractError("RAW requested geometry drifted")
    if {
        "width": effective.get("width"),
        "height": effective.get("height"),
    } != {
        "width": case["effective_width"],
        "height": case["effective_height"],
    }:
        raise ScheduleContractError("RAW effective geometry drifted")
    if parameters != {
        "image_seq_len": case["image_seq_len"],
        "recipe": case["recipe"],
        "steps": steps,
    }:
        raise ScheduleContractError("RAW recipe or image sequence evidence drifted")
    if evidence.get("level") != case["evidence"]:
        raise ScheduleContractError("RAW evidence level drifted")
    if shift_parameters.get("mu") != expected_mu_value:
        raise ScheduleContractError("RAW dynamic mu drifted")
    if ownership != {
        "schedule": "external_sigmas",
        "shift": "construction_pipeline",
    }:
        raise ScheduleContractError("RAW artifact ownership permits an implicit second shift")
    if shift_count != 1:
        raise ScheduleContractError("RAW artifact does not contain exactly one time shift")
    if bundle.artifact.numerical_fingerprint != case["numerical_fingerprint"]:
        raise ScheduleContractError("executed RAW schedule fingerprint drifted")

    return {
        "artifact_construction_fingerprint": bundle.artifact.construction_fingerprint,
        "case_id": case_id,
        "effective": {
            "height": effective["height"],
            "width": effective["width"],
        },
        "image_seq_len": parameters["image_seq_len"],
        "metadata_reloaded": True,
        "mu": expected_mu,
        "numerical_fingerprint": bundle.artifact.numerical_fingerprint,
        "receipt_fingerprint": bundle.receipt.receipt_fingerprint,
        "requested": {
            "height": requested["height"],
            "width": requested["width"],
        },
        "requested_transitions": counts["requested_transitions"],
        "schedule_ownership": ownership["schedule"],
        "shift_count": shift_count,
        "status": execution["status"],
    }


def verify_rejected_history(
    history: object,
    *,
    prompt_id: str,
    case_id: str,
    expected_message: str,
) -> dict[str, object]:
    """Require a terminal scheduler error with no partial output bundle."""

    root = _object(history, label="rejected prompt history")
    entry = _object(root.get(prompt_id), label="rejected prompt history entry")
    status = _object(entry.get("status"), label="rejected prompt status")
    if status.get("completed") is not False or status.get("status_str") != "error":
        raise ScheduleContractError("rejected prompt history is not a terminal error")
    prompt_tuple = _array(entry.get("prompt"), label="rejected retained prompt tuple")
    if len(prompt_tuple) < 3 or prompt_tuple[2] != _rejected_raw_api_prompt(case_id):
        raise ScheduleContractError("rejected prompt history retained a stale API graph")
    outputs = _object(entry.get("outputs"), label="rejected prompt outputs")
    if outputs:
        raise ScheduleContractError("rejected prompt produced partial output")
    messages = _array(status.get("messages"), label="rejected prompt messages")
    events = [_array(item, label="rejected prompt event") for item in messages]
    if [item[0] if item else None for item in events] != [
        "execution_start",
        "execution_cached",
        "execution_error",
    ]:
        raise ScheduleContractError("rejected prompt has an unexpected event sequence")
    cached_event = events[1]
    if len(cached_event) != 2:
        raise ScheduleContractError("rejected prompt cached evidence drifted")
    cached_detail = _object(cached_event[1], label="rejected prompt cached detail")
    if cached_detail.get("prompt_id") != prompt_id or not isinstance(
        cached_detail.get("nodes"), list
    ):
        raise ScheduleContractError("rejected prompt cached evidence drifted")
    event = events[2]
    if len(event) != 2 or event[0] != "execution_error":
        raise ScheduleContractError("rejected prompt has no execution_error event")
    detail = _object(event[1], label="rejected prompt error detail")
    if detail.get("prompt_id") != prompt_id:
        raise ScheduleContractError("rejected prompt error references a different prompt")
    if (
        detail.get("node_id") != "1"
        or detail.get("node_type") != "Sigmax.Krea2SigmaScheduler"
        or detail.get("exception_type")
        != "comfyui_sigmax.core.schedule_contracts.ScheduleContractError"
        or detail.get("exception_message") != f"{expected_message}\n"
        or detail.get("executed") != []
        or not isinstance(detail.get("current_outputs"), list)
    ):
        raise ScheduleContractError("rejected prompt error evidence drifted")
    reason_codes = {
        "raw-auto-variant": "input.variant_selection_required",
        "raw-invalid-steps": "input.steps_out_of_range",
    }
    reason_code = reason_codes.get(case_id)
    if reason_code is None:
        raise ScheduleContractError("rejected prompt case has no stable reason code")
    return {
        "boundary": "runtime_execution",
        "case_id": case_id,
        "exception_type": detail["exception_type"],
        "partial_output": False,
        "prompt_created": True,
        "reason_code": reason_code,
        "status": status["status_str"],
    }


def verify_prequeue_rejection(
    response: object,
    *,
    case_id: str,
) -> dict[str, object]:
    """Require the pinned structured HTTP 400 contract for invalid RAW steps."""

    if case_id != "raw-invalid-steps":
        raise ScheduleContractError("RAW prequeue rejection case ID is unsupported")
    root = _object(response, label="RAW prequeue rejection response")
    if "prompt_id" in root:
        raise ScheduleContractError("prequeue rejection unexpectedly created a prompt ID")
    error = _object(root.get("error"), label="RAW prequeue top-level error")
    if (
        error.get("type") != "prompt_outputs_failed_validation"
        or error.get("message") != "Prompt outputs failed validation"
        or error.get("details") != ""
        or error.get("extra_info") != {}
    ):
        raise ScheduleContractError("RAW prequeue top-level error drifted")
    node_errors = _object(root.get("node_errors"), label="RAW prequeue node errors")
    if set(node_errors) != {"1"}:
        raise ScheduleContractError("RAW prequeue scheduler error is missing or ambiguous")
    node_error = _object(node_errors["1"], label="RAW prequeue scheduler error")
    if node_error.get("class_type") != "Sigmax.Krea2SigmaScheduler" or node_error.get(
        "dependent_outputs"
    ) != ["3"]:
        raise ScheduleContractError("RAW prequeue scheduler identity drifted")
    reasons = _array(node_error.get("errors"), label="RAW prequeue validation reasons")
    if len(reasons) != 1:
        raise ScheduleContractError("RAW prequeue validation reasons are ambiguous")
    reason = _object(reasons[0], label="RAW prequeue validation reason")
    if (
        reason.get("type") != "value_smaller_than_min"
        or reason.get("message") != "Value 0 smaller than min of 1"
        or reason.get("details") != "steps"
    ):
        raise ScheduleContractError("RAW prequeue validation reason drifted")
    extra_info = _object(
        reason.get("extra_info"),
        label="RAW prequeue validation reason details",
    )
    input_config = _array(
        extra_info.get("input_config"),
        label="RAW prequeue input configuration",
    )
    if len(input_config) != 2 or input_config[0] != "INT":
        raise ScheduleContractError("RAW prequeue steps configuration drifted")
    constraints = _object(
        input_config[1],
        label="RAW prequeue steps constraints",
    )
    if (
        extra_info.get("input_name") != "steps"
        or type(extra_info.get("received_value")) is not int
        or extra_info.get("received_value") != 0
        or type(constraints.get("min")) is not int
        or constraints.get("min") != 1
    ):
        raise ScheduleContractError("RAW prequeue steps evidence drifted")
    return {
        "boundary": "prequeue_validation",
        "case_id": case_id,
        "http_status": 400,
        "node_id": "1",
        "node_type": node_error["class_type"],
        "partial_output": False,
        "prompt_created": False,
        "reason_code": "input.steps_below_minimum",
        "reason_type": reason["type"],
        "status": "rejected",
    }


def require_owned_run_path(
    *,
    repository_root: Path,
    owned_root: Path,
    candidate: Path,
) -> Path:
    """Resolve one strict descendant of an in-repository E2E temp root."""

    repository = repository_root.resolve()
    root = owned_root.resolve()
    resolved = candidate.resolve()
    if root == repository or repository not in root.parents:
        raise ScheduleContractError("E2E temp root must be a strict repository descendant")
    if resolved == root or root not in resolved.parents:
        raise ScheduleContractError("E2E run path must be a strict owned-root descendant")
    return resolved


def redact_text(text: object, *, sensitive_paths: Sequence[Path] = ()) -> str:
    """Remove owned paths and common credential forms from bounded diagnostics."""

    rendered = str(text)
    for path in sorted((str(item.resolve()) for item in sensitive_paths), key=len, reverse=True):
        if path:
            rendered = rendered.replace(path, "<redacted-path>")
            rendered = rendered.replace(path.replace("\\", "/"), "<redacted-path>")
    rendered = _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=<redacted>", rendered)
    return rendered[-_MAX_LOG_BYTES:]


def _json_unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError("host JSON contains duplicate object names")
        result[key] = value
    return result


def _decode_json(payload: bytes, *, label: str) -> object:
    if not payload or len(payload) > _MAX_HTTP_BYTES:
        raise ScheduleContractError(f"{label} size is outside the allowed range")
    try:
        return json.loads(payload, object_pairs_hook=_json_unique_pairs)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ScheduleContractError(f"{label} is not valid JSON") from exc


def _require_loopback_http_url(url: str) -> str:
    """Reject any URL that could escape the owned local ComfyUI process."""

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ScheduleContractError("host URL is malformed") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != _LOOPBACK
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ScheduleContractError("host URL must be credential-free loopback HTTP")
    return url


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: Mapping[str, object] | None = None,
    timeout: float = 5.0,
) -> object:
    payload = None
    headers = {"Accept": "application/json"}
    if body is not None:
        payload = json.dumps(
            dict(body),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
    # SECURITY: reject file/custom schemes and non-loopback destinations before urllib.
    request = Request(  # noqa: S310
        _require_loopback_http_url(url),
        data=payload,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(_MAX_HTTP_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ScheduleContractError("loopback host request failed") from exc
    return _decode_json(raw, label="loopback host response")


def _http_json_error(
    url: str,
    *,
    method: str,
    body: Mapping[str, object],
    expected_status: int,
    timeout: float = 5.0,
) -> object:
    payload = json.dumps(
        dict(body),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(  # noqa: S310
        _require_loopback_http_url(url),
        data=payload,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout):  # noqa: S310
            raise ScheduleContractError("loopback host unexpectedly accepted an invalid prompt")
    except HTTPError as exc:
        if exc.code != expected_status:
            raise ScheduleContractError(
                "loopback host returned an unexpected error status"
            ) from exc
        raw = exc.read(_MAX_HTTP_BYTES + 1)
    except (URLError, TimeoutError, OSError) as exc:
        raise ScheduleContractError("loopback host rejection request failed") from exc
    return _decode_json(raw, label="loopback host rejection response")


def _http_no_content(url: str, *, method: str, timeout: float) -> None:
    # SECURITY: reject file/custom schemes and non-loopback destinations before urllib.
    request = Request(  # noqa: S310
        _require_loopback_http_url(url),
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(_MAX_HTTP_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ScheduleContractError("loopback host request failed") from exc
    if len(raw) > _MAX_HTTP_BYTES:
        raise ScheduleContractError("loopback host response exceeds the allowed size")


def _select_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.bind((_LOOPBACK, 0))
        return cast(int, candidate.getsockname()[1])


def _git_revision(root: Path) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise ScheduleContractError("git executable is unavailable")
    # SECURITY: the executable is resolved and every argument is a fixed token or resolved path.
    result = subprocess.run(  # noqa: S603
        [git_executable, "-C", str(root.resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ScheduleContractError("selected ComfyUI root is not a readable Git checkout")
    return revision


def _minimax_h3_validation_lane(value: object) -> WorkflowValidationLane:
    """Resolve the H3-specific latest alias without changing other lane semantics."""

    if value == _MINIMAX_H3_LATEST_LANE:
        return WorkflowValidationLane.LATEST_HOST
    try:
        return WorkflowValidationLane(value)
    except (TypeError, ValueError) as exc:
        raise ScheduleContractError("MiniMax H3 validation lane is unsupported") from exc


def _minimax_h3_host_version(value: object) -> tuple[int, int, int]:
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ScheduleContractError("MiniMax H3 host version must be semantic X.Y.Z text")
    parts = tuple(int(item) for item in value.split("."))
    if len(parts) != 3:
        raise ScheduleContractError("MiniMax H3 host version must contain three components")
    return parts


def _validate_minimax_h3_host_identity(
    *,
    lane: WorkflowValidationLane,
    host_version: object,
    expected_revision: object,
    actual_revision: object,
) -> None:
    """Fail closed on caller/checkout identity before an H3 host process starts."""

    if not isinstance(expected_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise ScheduleContractError("MiniMax H3 expected host revision must be a 40-digit SHA")
    if not isinstance(actual_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", actual_revision):
        raise ScheduleContractError("MiniMax H3 selected host revision is invalid")
    if expected_revision != actual_revision:
        raise ScheduleContractError(
            "selected ComfyUI revision does not match the exact MiniMax H3 host revision"
        )
    version = _minimax_h3_host_version(host_version)
    if lane is WorkflowValidationLane.KNOWN_GOOD:
        if version != _minimax_h3_host_version(_MINIMAX_H3_HOST_VERSION):
            raise ScheduleContractError("MiniMax H3 known-good host version is not pinned")
        if expected_revision != MINIMAX_H3_COMFYUI_REVISION:
            raise ScheduleContractError("MiniMax H3 known-good host revision is not pinned")
        return
    if lane is not WorkflowValidationLane.LATEST_HOST:
        raise ScheduleContractError("MiniMax H3 host lane must be known_good or latest")
    if version < _MINIMAX_H3_MIN_LATEST_HOST_VERSION:
        raise ScheduleContractError(
            "MiniMax H3 latest host lane requires ComfyUI 0.31.0 or newer"
        )


def _verify_minimax_h3_live_host_version(
    system_stats: object, *, expected_version: object
) -> str:
    """Verify the version reported by the running host without retaining private stats."""

    expected = _minimax_h3_host_version(expected_version)
    root = _object(system_stats, label="MiniMax H3 live system stats")
    system = _object(root.get("system"), label="MiniMax H3 live system identity")
    reported = system.get("comfyui_version")
    if _minimax_h3_host_version(reported) != expected:
        raise ScheduleContractError("running ComfyUI version does not match the exact H3 host version")
    return cast(str, reported)


def _stage_extension(run_path: Path) -> Path:
    custom_node = run_path / "base" / "custom_nodes" / "ComfyUI-Sigmax"
    custom_node.mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / "__init__.py", custom_node / "__init__.py")
    shutil.copytree(
        REPOSITORY_ROOT / "comfyui_sigmax",
        custom_node / "comfyui_sigmax",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(REPOSITORY_ROOT / "web", custom_node / "web")
    return custom_node


def _stage_checkpoint_evidence_fixture(run_path: Path) -> Path:
    """Create one tiny deterministic safetensors file inside the owned host model folder."""

    header: dict[str, object] = {"__metadata__": {"is_distilled": "true"}}
    names = (
        "diffusion_model.first.weight",
        "diffusion_model.blocks.0.attn.wq.weight",
        "diffusion_model.blocks.0.attn.wk.weight",
        "diffusion_model.txtfusion.projector.weight",
    )
    for index, name in enumerate(names):
        header[name] = {
            "data_offsets": [index * 2, (index + 1) * 2],
            "dtype": "F16",
            "shape": [1],
        }
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    target = run_path / "base" / "models" / "checkpoints" / _CHECKPOINT_FIXTURE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(8))
    return target


def _stage_h3_test_pack(run_path: Path) -> Path:
    """Stage the repository-owned release-excluded H3 fixture pack."""

    if not _H3_TEST_PACK_SOURCE.is_file():
        raise ScheduleContractError("H3 test-pack source is missing")
    target_root = run_path / "base" / "custom_nodes" / _H3_TEST_PACK_NAME
    target_root.mkdir(parents=True)
    target = target_root / "__init__.py"
    shutil.copy2(_H3_TEST_PACK_SOURCE, target)
    return target


def _run_import_probe(
    *, host_python: Path, comfyui_root: Path, staged_node: Path
) -> dict[str, Any]:
    probe = """
import json
import sys
import torch
staged_node = sys.argv[1]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.samplers
before_call = torch.nn.Module.__call__
before_schedulers = tuple(comfy.samplers.SCHEDULER_NAMES)
sys.path.insert(0, staged_node)
import comfyui_sigmax
after_schedulers = tuple(comfy.samplers.SCHEDULER_NAMES)
print(json.dumps({
    "torch_call_unchanged": torch.nn.Module.__call__ is before_call,
    "scheduler_registry_unchanged": after_schedulers == before_schedulers,
    "node_ids": sorted(comfyui_sigmax.NODE_CLASS_MAPPINGS),
    "diffusers_loaded": "diffusers" in sys.modules,
}))
"""
    # SECURITY: host_python is an explicit existing file and the probe is a fixed local program.
    result = subprocess.run(  # noqa: S603
        [str(host_python), "-c", probe, str(staged_node)],
        cwd=comfyui_root,
        check=False,
        capture_output=True,
        timeout=90,
    )
    if result.returncode != 0:
        diagnostic = redact_text(
            result.stderr.decode("utf-8", errors="replace"),
            sensitive_paths=(REPOSITORY_ROOT, comfyui_root, staged_node, host_python),
        )[-2_000:]
        raise ScheduleContractError(f"host interpreter import-safety probe failed: {diagnostic}")
    data = _object(
        _decode_json(result.stdout, label="host import probe"), label="host import probe"
    )
    expected_ids = sorted(builtin_node_registry().class_mappings())
    if data != {
        "torch_call_unchanged": True,
        "scheduler_registry_unchanged": True,
        "node_ids": expected_ids,
        "diffusers_loaded": False,
    }:
        raise ScheduleContractError("host interpreter import-safety assertions failed")
    return data


def _readiness(
    *,
    base_url: str,
    process: subprocess.Popen[bytes],
    deadline: float,
) -> dict[str, object]:
    last_error = "not attempted"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise ScheduleContractError(f"ComfyUI exited before readiness ({return_code})")
        try:
            return _object(_http_json(f"{base_url}/object_info"), label="live object_info")
        except ScheduleContractError as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise ScheduleContractError(f"ComfyUI readiness deadline expired: {last_error}")


def _wait_for_history(*, base_url: str, prompt_id: str, deadline: float) -> dict[str, object]:
    last_error = "history not available"
    while time.monotonic() < deadline:
        try:
            value = _http_json(f"{base_url}/history/{prompt_id}")
            history = _object(value, label="prompt history")
            entry = history.get(prompt_id)
            if isinstance(entry, Mapping):
                status = cast(Mapping[str, object], entry).get("status")
                if isinstance(status, Mapping) and status.get("completed") is True:
                    return history
        except ScheduleContractError as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise ScheduleContractError(f"prompt history deadline expired: {last_error}")


def _wait_for_error_history(
    *,
    base_url: str,
    prompt_id: str,
    deadline: float,
) -> dict[str, object]:
    last_error = "error history not available"
    while time.monotonic() < deadline:
        try:
            value = _http_json(f"{base_url}/history/{prompt_id}")
            history = _object(value, label="rejected prompt history")
            entry = history.get(prompt_id)
            if isinstance(entry, Mapping):
                status = cast(Mapping[str, object], entry).get("status")
                if isinstance(status, Mapping) and status.get("status_str") == "error":
                    return history
        except ScheduleContractError as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise ScheduleContractError(f"prompt error-history deadline expired: {last_error}")


def _submit_successful_prompt(
    *,
    base_url: str,
    client_id: str,
    prompt: Mapping[str, object],
    execution_timeout: float,
    extra_data: Mapping[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    body: dict[str, object] = {
        "client_id": client_id,
        "prompt": dict(prompt),
    }
    if extra_data is not None:
        body["extra_data"] = dict(extra_data)
    response = _object(
        _http_json(
            f"{base_url}/prompt",
            method="POST",
            body=body,
            timeout=10,
        ),
        label="prompt response",
    )
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ScheduleContractError("prompt response does not contain a prompt ID")
    if response.get("node_errors", {}) not in ({}, None):
        raise ScheduleContractError("prompt validation returned node errors")
    history = _wait_for_history(
        base_url=base_url,
        prompt_id=prompt_id,
        deadline=time.monotonic() + execution_timeout,
    )
    return prompt_id, history


def _submit_rejected_runtime_prompt(
    *,
    base_url: str,
    client_id: str,
    prompt: Mapping[str, object],
    execution_timeout: float,
) -> tuple[str, dict[str, object]]:
    response = _object(
        _http_json(
            f"{base_url}/prompt",
            method="POST",
            body={"client_id": client_id, "prompt": dict(prompt)},
            timeout=10,
        ),
        label="rejected prompt submission",
    )
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ScheduleContractError("runtime rejection did not create a prompt ID")
    if response.get("node_errors", {}) not in ({}, None):
        raise ScheduleContractError("runtime rejection returned unexpected preflight errors")
    history = _wait_for_error_history(
        base_url=base_url,
        prompt_id=prompt_id,
        deadline=time.monotonic() + execution_timeout,
    )
    return prompt_id, history


def _submit_rejected_prequeue_prompt(
    *,
    base_url: str,
    client_id: str,
    prompt: Mapping[str, object],
) -> object:
    return _http_json_error(
        f"{base_url}/prompt",
        method="POST",
        body={"client_id": client_id, "prompt": dict(prompt)},
        expected_status=400,
        timeout=10,
    )


def _rejected_raw_api_prompt(case_id: str) -> dict[str, object]:
    prompt = build_raw_api_prompt("krea2-raw-official-square-1024")
    scheduler = cast(dict[str, object], prompt["1"])
    inputs = cast(dict[str, object], scheduler["inputs"])
    if case_id == "raw-auto-variant":
        inputs["variant"] = "auto"
    elif case_id == "raw-invalid-steps":
        inputs["steps"] = 0
    else:
        raise ScheduleContractError("RAW rejection case ID is unsupported")
    return prompt


def _signal_posix_process_group(pid: int, signal_name: str) -> None:
    """Send a named signal without importing POSIX-only attributes on Windows."""

    # IMPORTANT: resolve POSIX-only APIs dynamically so Windows type/import checks stay valid.
    killpg = getattr(os, "killpg", None)
    signum = getattr(signal, signal_name, None)
    if not callable(killpg) or not isinstance(signum, int):
        raise ScheduleContractError("POSIX process-group signaling is unavailable")
    cast(Callable[[int, int], None], killpg)(pid, signum)


def _terminate_owned_process(
    process: subprocess.Popen[bytes],
    *,
    base_url: str,
) -> dict[str, object]:
    interrupted = False
    try:
        _http_no_content(f"{base_url}/interrupt", method="POST", timeout=2)
        interrupted = True
    except ScheduleContractError:
        pass

    if process.poll() is None:
        try:
            if os.name == "posix":
                _signal_posix_process_group(process.pid, "SIGINT")
            else:
                process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            if os.name == "nt":
                taskkill_executable = shutil.which("taskkill")
                if taskkill_executable is None:
                    raise ScheduleContractError("taskkill executable is unavailable") from None
                # SECURITY: taskkill is resolved and targets only the owned child-process PID.
                subprocess.run(  # noqa: S603
                    [taskkill_executable, "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    timeout=15,
                )
            else:
                with suppress(ProcessLookupError):
                    _signal_posix_process_group(process.pid, "SIGKILL")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                raise ScheduleContractError("owned ComfyUI process did not terminate") from exc
    return {"interrupt_requested": interrupted, "return_code": process.returncode}


def _wait_for_port_release(port: int, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((_LOOPBACK, port))
            except OSError:
                time.sleep(0.1)
                continue
            return
    raise ScheduleContractError("owned loopback port was not released")


def _host_command(
    *,
    host_python: Path,
    comfyui_root: Path,
    run_path: Path,
    port: int,
) -> list[str]:
    return [
        str(host_python),
        str(comfyui_root / "main.py"),
        "--cpu",
        "--listen",
        _LOOPBACK,
        "--port",
        str(port),
        "--base-directory",
        str(run_path / "base"),
        "--output-directory",
        str(run_path / "output"),
        "--input-directory",
        str(run_path / "input"),
        "--temp-directory",
        str(run_path / "temp"),
        "--user-directory",
        str(run_path / "user"),
        "--database-url",
        "sqlite:///:memory:",
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        "ComfyUI-Sigmax",
        _H3_TEST_PACK_NAME,
    ]


def _write_evidence(path: Path | None, evidence: Mapping[str, object]) -> None:
    if path is None:
        return
    resolved = path.resolve()
    if REPOSITORY_ROOT.resolve() not in resolved.parents:
        raise ScheduleContractError("evidence file must stay inside the repository")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(dict(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_conditioning(args: argparse.Namespace) -> dict[str, object]:
    """Execute the isolated M4-12 H1 plus model-free RAW/Turbo H2 lane."""

    started = time.time()
    comfyui_root = Path(args.comfyui_root).resolve()
    host_python = Path(args.host_python).resolve()
    if not (comfyui_root / "main.py").is_file():
        raise ScheduleContractError("COMFYUI_ROOT does not contain main.py")
    if not host_python.is_file():
        raise ScheduleContractError("SIGMAX_COMFYUI_PYTHON is not a file")
    host_revision = _git_revision(comfyui_root)
    if host_revision != args.expected_revision:
        raise ScheduleContractError(
            "selected ComfyUI revision does not match the exact expected revision"
        )
    validation_lane = WorkflowValidationLane(args.validation_lane)
    if validation_lane is WorkflowValidationLane.KNOWN_GOOD and (
        args.host_version != CANONICAL_HOST_VERSION or host_revision != CANONICAL_HOST_REVISION
    ):
        raise ScheduleContractError(
            "known-good validation requires the canonical pinned host identity"
        )

    owned_root = Path(args.temp_root).resolve()
    run_path = require_owned_run_path(
        repository_root=REPOSITORY_ROOT,
        owned_root=owned_root,
        candidate=owned_root / f"conditioning-run-{uuid.uuid4().hex}",
    )
    run_path.mkdir(parents=True)
    for name in ("base", "input", "output", "temp", "user"):
        (run_path / name).mkdir()
    staged_node = _stage_extension(run_path)
    _stage_h3_test_pack(run_path)
    import_probe = _run_import_probe(
        host_python=host_python,
        comfyui_root=comfyui_root,
        staged_node=staged_node,
    )

    port = _select_free_port()
    base_url = f"http://{_LOOPBACK}:{port}"
    log_path = run_path / "comfyui.log"
    process: subprocess.Popen[bytes] | None = None
    shutdown: dict[str, object] = {}
    succeeded = False
    evidence: dict[str, object] = {
        "schema": "sigmax.krea2-conditioning-host-e2e/1",
        "lanes": ["H1", "H2_KREA2_CONDITIONING_M4_12"],
        "host": {
            "id": "comfyui",
            "version": args.host_version,
            "revision": host_revision,
        },
        "sigmax_revision": _git_revision(REPOSITORY_ROOT),
        "platform": platform.system().casefold(),
        "listen": _LOOPBACK,
        "port": port,
        "import_probe": import_probe,
        "attempt_transitions": {},
    }
    try:
        creationflags = (
            cast(int, getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        )
        with log_path.open("wb") as log:
            process = subprocess.Popen(  # noqa: S603
                _host_command(
                    host_python=host_python,
                    comfyui_root=comfyui_root,
                    run_path=run_path,
                    port=port,
                ),
                cwd=run_path,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                start_new_session=os.name == "posix",
            )
            object_info = _readiness(
                base_url=base_url,
                process=process,
                deadline=time.monotonic() + args.readiness_timeout,
            )
            public_ids = tuple(builtin_node_registry().class_mappings())
            filtered = {
                node_id: object_info[node_id] for node_id in public_ids if node_id in object_info
            }
            test_ids = {
                "SigmaxTest.Krea2ConditioningSource",
                "SigmaxTest.Krea2ConditioningProbe",
            }
            if tuple(sorted(filtered)) != public_ids or not test_ids <= set(object_info):
                raise ScheduleContractError("conditioning H1 is missing a required node ID")
            live_report = validate_live_workflow_fixtures(
                object_info=filtered,
                host_version=args.host_version,
                host_revision=host_revision,
                lane=validation_lane,
            )
            if not live_report.gate_passed or live_report.issues:
                raise ScheduleContractError("conditioning H1 live schema validation failed")
            h1_summary = {
                "expected_node_ids": list(public_ids),
                "live_schema_fingerprint": live_report.report_fingerprint,
                "registered": True,
                "status": "succeeded",
            }
            repeat_info = _object(
                _http_json(f"{base_url}/object_info"),
                label="repeat conditioning live object_info",
            )
            repeat_filtered = {
                node_id: repeat_info[node_id] for node_id in public_ids if node_id in repeat_info
            }
            repeat_report = validate_live_workflow_fixtures(
                object_info=repeat_filtered,
                host_version=args.host_version,
                host_revision=host_revision,
                lane=validation_lane,
            )
            repeat_h1_summary = {
                "expected_node_ids": list(public_ids),
                "live_schema_fingerprint": repeat_report.report_fingerprint,
                "registered": (
                    tuple(sorted(repeat_filtered)) == public_ids
                    and repeat_report.gate_passed
                    and not repeat_report.issues
                ),
                "status": "succeeded",
            }
            attempts = cast(dict[str, object], evidence["attempt_transitions"])
            attempts["h1"] = build_verified_host_repeat_transition(
                lane="H1",
                first_summary=h1_summary,
                repeat_summary=repeat_h1_summary,
            )
            conditioning_results: list[dict[str, object]] = []
            for variant in ("RAW", "Turbo"):

                def submit_conditioning(
                    ordinal: int,
                    *,
                    selected_variant: str = variant,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-m4-12-conditioning-{selected_variant.casefold()}-{ordinal}",
                        prompt=build_krea2_conditioning_h2_api_prompt(selected_variant),
                        execution_timeout=args.execution_timeout,
                    )

                def verify_conditioning(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_variant: str = variant,
                ) -> dict[str, object]:
                    return verify_krea2_conditioning_h2_history(
                        history,
                        prompt_id=prompt_id,
                        variant=selected_variant,
                    )

                summary, transition = execute_verified_host_repeat(
                    lane="H2_KREA2_CONDITIONING_M4_12",
                    submit=submit_conditioning,
                    verify=verify_conditioning,
                )
                conditioning_results.append(summary)
                attempts[f"h2_krea2_conditioning.{variant.casefold()}"] = transition
            evidence["h2_krea2_conditioning"] = conditioning_results
            succeeded = True
    finally:
        if process is not None:
            shutdown = _terminate_owned_process(process, base_url=base_url)
        _wait_for_port_release(port)
        evidence["shutdown"] = shutdown
        evidence["duration_seconds"] = round(time.time() - started, 3)
        if log_path.exists():
            evidence["host_log_tail"] = redact_text(
                log_path.read_text(encoding="utf-8", errors="replace"),
                sensitive_paths=(REPOSITORY_ROOT, comfyui_root, run_path, host_python),
            )[-8_000:]
        evidence["cleanup"] = "removed" if succeeded else "retained_failure_artifacts"
        _write_evidence(Path(args.evidence_file) if args.evidence_file else None, evidence)
        if succeeded:
            shutil.rmtree(run_path)
    return evidence


def build_minimax_h3_model_lane_plan(
    *,
    variant: str,
    prompt: str,
    model_artifact: str,
    allow_model_weights: bool,
    license_ack: bool,
    host_revision: str = MINIMAX_H3_COMFYUI_REVISION,
    host_version: str = _MINIMAX_H3_HOST_VERSION,
    width: int = 1344,
    height: int = 768,
    length: int = 124,
    steps: int = 20,
    seed: int = 0,
    first_frame: str | None = None,
    last_frame: str | None = None,
    reference_images: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build an authorization-gated H3 model workflow plan without executing a host.

    This is the bridge between the accepted model-free H1/H2 contract and a future authorized
    weight-backed host lane. It deliberately produces a JSON-compatible plan only: no model file
    is opened, no host process starts, and no network call is made.
    """

    if license_ack is not True:
        raise ScheduleContractError(
            "MiniMax H3 model lane requires an explicit license acknowledgement"
        )
    if allow_model_weights is not True:
        raise ScheduleContractError(
            "MiniMax H3 model lane requires explicit weight execution authorization"
        )
    if host_revision != MINIMAX_H3_COMFYUI_REVISION:
        raise ScheduleContractError("MiniMax H3 model lane host revision is not pinned")
    if host_version != _MINIMAX_H3_HOST_VERSION:
        raise ScheduleContractError("MiniMax H3 model lane host version is not pinned")
    if variant not in _MINIMAX_H3_PUBLIC_VARIANTS:
        raise ScheduleContractError("MiniMax H3 model lane variant must be selected explicitly")
    if not isinstance(reference_images, tuple):
        raise ScheduleContractError("MiniMax H3 model lane reference_images must be a tuple")

    selected_variant = cast(MiniMaxH3PublicVariant, variant)
    model_files = MiniMaxH3ModelFiles(diffusion_model=model_artifact)
    spec = MiniMaxH3WorkflowSpec(
        variant=selected_variant,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        steps=steps,
        seed=seed,
        first_frame=first_frame,
        last_frame=last_frame,
        reference_images=reference_images,
        model_files=model_files,
    )
    workflow = build_minimax_h3_host_workflow(spec)
    # ComfyUI input names may contain dots (for example ``ref_images.ref_image_0``), while the
    # pure-core projection intentionally accepts only controlled ASCII identifier keys. The
    # model-lane receipt therefore fingerprints the JSON-compatible API graph directly and keeps
    # the native host key unchanged.
    workflow_projection = json.dumps(
        workflow.prompt,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": _MINIMAX_H3_MODEL_LANE_SCHEMA,
        "lane": "H4_MINIMAX_H3_MODEL_M6_05",
        "status": "authorized_not_executed",
        "host": {"version": host_version, "revision": host_revision},
        "variant": variant,
        "authorization": {
            "allow_model_weights": True,
            "license_ack": True,
            "network": False,
        },
        "execution": {"performed": False, "weights_loaded": False},
        "model_files": {
            "diffusion_model": workflow.model_files.diffusion_model,
            "text_encoder": workflow.model_files.text_encoder,
            "video_vae": workflow.model_files.video_vae,
            "audio_vae": workflow.model_files.audio_vae,
        },
        "workflow": workflow.prompt,
        "contract": {
            "schema": workflow.contract.schema,
            "host_min_version": workflow.contract.host_min_version,
            "schedule_node_id": workflow.contract.schedule_node_id,
            "native_shift_node_id": workflow.contract.native_shift_node_id,
            "sampler_node_id": workflow.contract.sampler_node_id,
            "schedule_ownership": workflow.contract.schedule_ownership,
            "audio_ownership": workflow.contract.audio_ownership,
            "external_video_shift_applied_once": workflow.contract.external_video_shift_applied_once,
            "external_audio_schedule": workflow.contract.external_audio_schedule,
        },
        "workflow_fingerprint": "sha256:" + hashlib.sha256(workflow_projection).hexdigest(),
    }


def run_minimax_h3(args: argparse.Namespace) -> dict[str, object]:
    """Execute the model-free MiniMax H3 H1/H2 contract on an exact reviewed host."""

    started = time.time()
    comfyui_root = Path(args.comfyui_root).resolve()
    host_python = Path(args.host_python).resolve()
    if not (comfyui_root / "main.py").is_file():
        raise ScheduleContractError("COMFYUI_ROOT does not contain main.py")
    if not host_python.is_file():
        raise ScheduleContractError("SIGMAX_COMFYUI_PYTHON is not a file")
    host_revision = _git_revision(comfyui_root)
    validation_lane = _minimax_h3_validation_lane(args.validation_lane)
    _validate_minimax_h3_host_identity(
        lane=validation_lane,
        host_version=args.minimax_h3_host_version,
        expected_revision=args.minimax_h3_expected_revision,
        actual_revision=host_revision,
    )

    owned_root = Path(args.temp_root).resolve()
    run_path = require_owned_run_path(
        repository_root=REPOSITORY_ROOT,
        owned_root=owned_root,
        candidate=owned_root / f"minimax-h3-run-{uuid.uuid4().hex}",
    )
    run_path.mkdir(parents=True)
    for name in ("base", "input", "output", "temp", "user"):
        (run_path / name).mkdir()
    staged_node = _stage_extension(run_path)
    _stage_h3_test_pack(run_path)
    import_probe = _run_import_probe(
        host_python=host_python,
        comfyui_root=comfyui_root,
        staged_node=staged_node,
    )

    port = _select_free_port()
    base_url = f"http://{_LOOPBACK}:{port}"
    log_path = run_path / "comfyui.log"
    process: subprocess.Popen[bytes] | None = None
    shutdown: dict[str, object] = {}
    succeeded = False
    evidence: dict[str, object] = {
        "schema": "sigmax.minimax-h3-host-e2e/1",
        "lanes": ["H1", "H2_MINIMAX_H3_M6_05"],
        "validation_lane": validation_lane.value,
        "host": {
            "id": "comfyui",
            "version": args.minimax_h3_host_version,
            "revision": host_revision,
        },
        "sigmax_revision": _git_revision(REPOSITORY_ROOT),
        "platform": platform.system().casefold(),
        "listen": _LOOPBACK,
        "port": port,
        "import_probe": import_probe,
        "attempt_transitions": {},
        "model_execution": "not_loaded",
    }
    try:
        creationflags = (
            cast(int, getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        )
        with log_path.open("wb") as log:
            process = subprocess.Popen(  # noqa: S603
                _host_command(
                    host_python=host_python,
                    comfyui_root=comfyui_root,
                    run_path=run_path,
                    port=port,
                ),
                cwd=run_path,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                start_new_session=os.name == "posix",
            )
            object_info = _readiness(
                base_url=base_url,
                process=process,
                deadline=time.monotonic() + args.readiness_timeout,
            )
            reported_host_version = _verify_minimax_h3_live_host_version(
                _http_json(f"{base_url}/system_stats"),
                expected_version=args.minimax_h3_host_version,
            )
            cast(dict[str, object], evidence["host"])["reported_version"] = reported_host_version
            registry = builtin_node_registry()
            expected_ids = tuple(registry.class_mappings())
            filtered = {
                node_id: object_info[node_id] for node_id in expected_ids if node_id in object_info
            }
            if tuple(sorted(filtered)) != expected_ids:
                raise ScheduleContractError(
                    "MiniMax H3 host is missing one or more Sigmax node IDs"
                )
            test_ids = {"SigmaxTest.MiniMaxH3ScheduleProbe"}
            if not test_ids <= set(object_info):
                raise ScheduleContractError("MiniMax H3 H2 test probe is not registered")
            live_report = validate_live_workflow_fixtures(
                object_info=filtered,
                host_version=args.minimax_h3_host_version,
                host_revision=host_revision,
                lane=validation_lane,
            )
            if not live_report.gate_passed or live_report.issues:
                raise ScheduleContractError("MiniMax H3 host H1 live schema validation failed")
            h1_summary = {
                "expected_node_ids": list(expected_ids),
                "live_schema_fingerprint": live_report.report_fingerprint,
                "registered": True,
                "status": "succeeded",
            }
            repeat_info = _object(
                _http_json(f"{base_url}/object_info"),
                label="MiniMax H3 repeat live object_info",
            )
            repeat_filtered = {
                node_id: repeat_info[node_id] for node_id in expected_ids if node_id in repeat_info
            }
            repeat_report = validate_live_workflow_fixtures(
                object_info=repeat_filtered,
                host_version=args.minimax_h3_host_version,
                host_revision=host_revision,
                lane=validation_lane,
            )
            repeat_h1_summary = {
                "expected_node_ids": list(expected_ids),
                "live_schema_fingerprint": repeat_report.report_fingerprint,
                "registered": (
                    tuple(sorted(repeat_filtered)) == expected_ids
                    and repeat_report.gate_passed
                    and not repeat_report.issues
                ),
                "status": "succeeded",
            }
            attempts = cast(dict[str, object], evidence["attempt_transitions"])
            attempts["h1"] = build_verified_host_repeat_transition(
                lane="H1",
                first_summary=h1_summary,
                repeat_summary=repeat_h1_summary,
            )
            h2_results: list[dict[str, object]] = []
            for variant, variant_id in (
                ("H3 Base FL2VA", "fl2va"),
                ("H3 Base Ref2VA", "ref2va"),
            ):

                def submit_h2(
                    ordinal: int,
                    *,
                    selected_variant: str = variant,
                    selected_id: str = variant_id,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-m6-05-minimax-h3-{selected_id}-attempt-{ordinal}",
                        prompt=build_minimax_h3_h2_api_prompt(selected_variant),
                        execution_timeout=args.execution_timeout,
                    )

                def verify_h2(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_variant: str = variant,
                ) -> dict[str, object]:
                    return verify_minimax_h3_h2_history(
                        history,
                        prompt_id=prompt_id,
                        variant=selected_variant,
                    )

                summary, transition = execute_verified_host_repeat(
                    lane="H2_MINIMAX_H3_M6_05",
                    submit=submit_h2,
                    verify=verify_h2,
                )
                h2_results.append(summary)
                attempts[f"h2_minimax_h3.{variant_id}"] = transition
            evidence["h1"] = h1_summary
            evidence["h2_minimax_h3"] = h2_results
            succeeded = True
    finally:
        if process is not None:
            shutdown = _terminate_owned_process(process, base_url=base_url)
        _wait_for_port_release(port)
        evidence["shutdown"] = shutdown
        evidence["duration_seconds"] = round(time.time() - started, 3)
        if log_path.exists():
            evidence["host_log_tail"] = redact_text(
                log_path.read_text(encoding="utf-8", errors="replace"),
                sensitive_paths=(REPOSITORY_ROOT, comfyui_root, run_path, host_python),
            )[-8_000:]
        evidence["cleanup"] = "removed" if succeeded else "retained_failure_artifacts"
        _write_evidence(Path(args.evidence_file) if args.evidence_file else None, evidence)
        if succeeded:
            shutil.rmtree(run_path)
    return evidence


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute isolated H1, activated H2 workflows, and M5-01 H3."""

    started = time.time()
    comfyui_root = Path(args.comfyui_root).resolve()
    host_python = Path(args.host_python).resolve()
    if not (comfyui_root / "main.py").is_file():
        raise ScheduleContractError("COMFYUI_ROOT does not contain main.py")
    if not host_python.is_file():
        raise ScheduleContractError("SIGMAX_COMFYUI_PYTHON is not a file")
    host_revision = _git_revision(comfyui_root)
    if host_revision != args.expected_revision:
        raise ScheduleContractError(
            "selected ComfyUI revision does not match the exact expected revision"
        )
    validation_lane = WorkflowValidationLane(args.validation_lane)
    if validation_lane is WorkflowValidationLane.KNOWN_GOOD and (
        args.host_version != CANONICAL_HOST_VERSION or host_revision != CANONICAL_HOST_REVISION
    ):
        raise ScheduleContractError(
            "known-good validation requires the canonical pinned host identity"
        )

    owned_root = Path(args.temp_root).resolve()
    run_path = require_owned_run_path(
        repository_root=REPOSITORY_ROOT,
        owned_root=owned_root,
        candidate=owned_root / f"run-{uuid.uuid4().hex}",
    )
    run_path.mkdir(parents=True)
    for name in ("base", "input", "output", "temp", "user"):
        (run_path / name).mkdir()
    staged_node = _stage_extension(run_path)
    _stage_checkpoint_evidence_fixture(run_path)
    _stage_h3_test_pack(run_path)
    import_probe = _run_import_probe(
        host_python=host_python,
        comfyui_root=comfyui_root,
        staged_node=staged_node,
    )

    port = _select_free_port()
    base_url = f"http://{_LOOPBACK}:{port}"
    log_path = run_path / "comfyui.log"
    process: subprocess.Popen[bytes] | None = None
    shutdown: dict[str, object] = {}
    succeeded = False
    evidence: dict[str, object] = {
        "schema": "sigmax.comfyui-host-e2e-evidence/3",
        "lanes": [
            "H1",
            "H2_TURBO_M2_05",
            "H2_KREA2_LORA_EXPERIMENTAL_M4_11",
            "H2_RAW_M3_06",
            "H2_ALGEBRA_M4_09",
            "H2_CHECKPOINT_EVIDENCE_M6_08",
            "H2_Z_IMAGE_M6_04",
            "H2_FLUX1_SCHNELL_M6_05",
            "H2_QWEN_IMAGE_M6_05",
            "H2_AURAFLOW_M6_05",
            "H2_LUMINA2_M6_05",
            "H2_HUNYUAN_IMAGE21_M6_05",
            "H2_ANIMA_M6_05",
            "H2_WAN_M4_14",
            "H2_LTX_M6_05",
            "H3_EULER_M5_01",
        ],
        "host": {
            "id": "comfyui",
            "version": args.host_version,
            "revision": host_revision,
        },
        "sigmax_revision": _git_revision(REPOSITORY_ROOT),
        "platform": platform.system().casefold(),
        "listen": _LOOPBACK,
        "port": port,
        "import_probe": import_probe,
        "attempt_transitions": {},
    }
    try:
        # IMPORTANT: Windows-only subprocess constants are absent from Linux type stubs.
        creationflags = (
            cast(int, getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0
        )
        with log_path.open("wb") as log:
            # SECURITY: the executable is an explicit existing host-venv Python path and all
            # arguments are constructed from validated/pinned paths and an owned free port.
            process = subprocess.Popen(  # noqa: S603
                _host_command(
                    host_python=host_python,
                    comfyui_root=comfyui_root,
                    run_path=run_path,
                    port=port,
                ),
                # CRITICAL: host-local logs must stay in the owned run directory; the
                # pinned reference checkout is an untrusted, read-only source input.
                cwd=run_path,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                start_new_session=os.name == "posix",
            )
            ready_started = time.monotonic()
            object_info = _readiness(
                base_url=base_url,
                process=process,
                deadline=ready_started + args.readiness_timeout,
            )
            evidence["readiness_seconds"] = round(time.monotonic() - ready_started, 3)

            registry = builtin_node_registry()
            expected_ids = tuple(registry.class_mappings())
            filtered = {
                node_id: object_info[node_id] for node_id in expected_ids if node_id in object_info
            }
            if tuple(sorted(filtered)) != expected_ids:
                raise ScheduleContractError("live host is missing one or more Sigmax node IDs")
            live_report = validate_live_workflow_fixtures(
                object_info=filtered,
                host_version=args.host_version,
                host_revision=host_revision,
                lane=validation_lane,
            )
            if not live_report.gate_passed or live_report.issues:
                issue_payload = [issue.projection() for issue in live_report.issues]
                raise ScheduleContractError(
                    "live Sigmax node schema validation failed: "
                    + json.dumps(issue_payload, sort_keys=True)
                )

            h1_summary = {
                "expected_node_ids": list(expected_ids),
                "live_schema_fingerprint": live_report.report_fingerprint,
                "registered": True,
                "status": "succeeded",
            }
            evidence["h1"] = h1_summary
            repeat_object_info = _object(
                _http_json(f"{base_url}/object_info"),
                label="repeat live object_info",
            )
            repeat_filtered = {
                node_id: repeat_object_info[node_id]
                for node_id in expected_ids
                if node_id in repeat_object_info
            }
            repeat_report = validate_live_workflow_fixtures(
                object_info=repeat_filtered,
                host_version=args.host_version,
                host_revision=host_revision,
                lane=validation_lane,
            )
            repeat_h1_summary = {
                "expected_node_ids": list(expected_ids),
                "live_schema_fingerprint": repeat_report.report_fingerprint,
                "registered": (
                    tuple(sorted(repeat_filtered)) == expected_ids
                    and repeat_report.gate_passed
                    and not repeat_report.issues
                ),
                "status": "succeeded",
            }
            attempts = cast(dict[str, object], evidence["attempt_transitions"])
            attempts["h1"] = build_verified_host_repeat_transition(
                lane="H1",
                first_summary=h1_summary,
                repeat_summary=repeat_h1_summary,
            )

            def submit_turbo(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_successful_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m2-05-e2e-attempt-{ordinal}",
                    prompt=build_turbo_api_prompt(),
                    execution_timeout=args.execution_timeout,
                )

            def verify_turbo(history: object, prompt_id: str) -> dict[str, object]:
                return verify_turbo_history(history, prompt_id=prompt_id)

            turbo_summary, turbo_transition = execute_verified_host_repeat(
                lane="H2_TURBO_M2_05",
                submit=submit_turbo,
                verify=verify_turbo,
            )
            evidence["h2_turbo"] = turbo_summary
            attempts["h2_turbo"] = turbo_transition

            lora_results: list[dict[str, object]] = []
            for lora_variant, case_id in (
                ("LoRA Experimental (RAW mu)", "raw-mu"),
                ("LoRA Experimental (Turbo mu)", "turbo-mu"),
            ):

                def submit_lora(
                    ordinal: int,
                    *,
                    selected_variant: str = lora_variant,
                    selected_case: str = case_id,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-m4-11-{selected_case}-attempt-{ordinal}",
                        prompt=build_krea2_lora_experimental_h2_api_prompt(selected_variant),
                        execution_timeout=args.execution_timeout,
                    )

                def verify_lora(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_variant: str = lora_variant,
                ) -> dict[str, object]:
                    return verify_krea2_lora_experimental_h2_history(
                        history,
                        prompt_id=prompt_id,
                        variant=selected_variant,
                    )

                lora_summary, lora_transition = execute_verified_host_repeat(
                    lane="H2_KREA2_LORA_EXPERIMENTAL_M4_11",
                    submit=submit_lora,
                    verify=verify_lora,
                )
                lora_results.append(lora_summary)
                attempts[f"h2_krea2_lora.{case_id}"] = lora_transition
            if lora_results[0]["numerical_fingerprint"] == lora_results[1]["numerical_fingerprint"]:
                raise ScheduleContractError("experimental Krea 2 mu control is inert")
            evidence["h2_krea2_lora_experimental"] = lora_results

            fixtures = {item.identifier: item for item in load_canonical_workflow_fixtures()}
            raw_results: list[dict[str, object]] = []
            for case_id in _RAW_CASES:
                fixture = fixtures.get(case_id)
                if fixture is None:
                    raise ScheduleContractError("RAW host case has no canonical workflow")
                submitted_workflow = cast(dict[str, object], fixture.workflow)

                def submit_raw(
                    ordinal: int,
                    *,
                    selected_case: str = case_id,
                    workflow: dict[str, object] = submitted_workflow,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-m3-06-{selected_case}-attempt-{ordinal}",
                        prompt=build_raw_api_prompt(selected_case),
                        extra_data={"extra_pnginfo": {"workflow": workflow}},
                        execution_timeout=args.execution_timeout,
                    )

                def verify_raw(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_case: str = case_id,
                    workflow: dict[str, object] = submitted_workflow,
                ) -> dict[str, object]:
                    return verify_raw_history(
                        history,
                        prompt_id=prompt_id,
                        case_id=selected_case,
                        submitted_workflow=workflow,
                    )

                raw_summary, raw_transition = execute_verified_host_repeat(
                    lane="H2_RAW_M3_06",
                    submit=submit_raw,
                    verify=verify_raw,
                )
                raw_results.append(raw_summary)
                attempts[f"h2_raw.{case_id}"] = raw_transition
            evidence["h2_raw"] = raw_results

            z_image_results: list[dict[str, object]] = []
            for variant, case_id in (
                ("Base", "z-image-base-official-50"),
                ("Turbo", "z-image-turbo-official-8"),
            ):
                fixture = fixtures.get(case_id)
                if fixture is None:
                    raise ScheduleContractError("Z-Image host case has no canonical workflow")
                submitted_workflow = cast(dict[str, object], fixture.workflow)

                def submit_z_image(
                    ordinal: int,
                    *,
                    selected_variant: str = variant,
                    selected_case: str = case_id,
                    workflow: dict[str, object] = submitted_workflow,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-m6-04-{selected_case}-attempt-{ordinal}",
                        prompt=build_z_image_h2_api_prompt(selected_variant),
                        extra_data={"extra_pnginfo": {"workflow": workflow}},
                        execution_timeout=args.execution_timeout,
                    )

                def verify_z_image(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_variant: str = variant,
                ) -> dict[str, object]:
                    return verify_z_image_h2_history(
                        history, prompt_id=prompt_id, variant=selected_variant
                    )

                z_summary, z_transition = execute_verified_host_repeat(
                    lane="H2_Z_IMAGE_M6_04",
                    submit=submit_z_image,
                    verify=verify_z_image,
                )
                z_image_results.append(z_summary)
                attempts[f"h2_z_image.{case_id}"] = z_transition
            evidence["h2_z_image"] = z_image_results

            flux_fixture = fixtures.get("flux1-schnell-official-4")
            if flux_fixture is None:
                raise ScheduleContractError("FLUX.1-schnell host case has no canonical workflow")
            flux_workflow = cast(dict[str, object], flux_fixture.workflow)

            def submit_flux1_schnell(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_successful_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m6-05-flux1-schnell-attempt-{ordinal}",
                    prompt=build_flux1_schnell_h2_api_prompt(),
                    extra_data={"extra_pnginfo": {"workflow": flux_workflow}},
                    execution_timeout=args.execution_timeout,
                )

            def verify_flux1_schnell(history: object, prompt_id: str) -> dict[str, object]:
                return verify_flux1_schnell_h2_history(history, prompt_id=prompt_id)

            flux_summary, flux_transition = execute_verified_host_repeat(
                lane="H2_FLUX1_SCHNELL_M6_05",
                submit=submit_flux1_schnell,
                verify=verify_flux1_schnell,
            )
            evidence["h2_flux1_schnell"] = flux_summary
            attempts["h2_flux1_schnell.flux1-schnell-official-4"] = flux_transition

            qwen_results: list[dict[str, object]] = []
            for mode in ("Comfy Fixed", "Diffusers Dynamic"):

                def submit_qwen(
                    ordinal: int,
                    *,
                    selected_mode: str = mode,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=(
                            f"sigmax-m6-05-qwen-image-"
                            f"{selected_mode.casefold().replace(' ', '-')}-attempt-{ordinal}"
                        ),
                        prompt=build_qwen_image_h2_api_prompt(selected_mode),
                        execution_timeout=args.execution_timeout,
                    )

                def verify_qwen(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_mode: str = mode,
                ) -> dict[str, object]:
                    return verify_qwen_image_h2_history(
                        history,
                        prompt_id=prompt_id,
                        mode=selected_mode,
                    )

                qwen_summary, qwen_transition = execute_verified_host_repeat(
                    lane="H2_QWEN_IMAGE_M6_05",
                    submit=submit_qwen,
                    verify=verify_qwen,
                )
                qwen_results.append(qwen_summary)
                attempts[f"h2_qwen_image.{mode.casefold().replace(' ', '-')}"] = qwen_transition
            evidence["h2_qwen_image"] = qwen_results

            sd3_results: list[dict[str, object]] = []
            for mode, case_id in (
                ("Publisher Reference (1.0)", "sd3-publisher-reference-official-50"),
                ("Comfy/Diffusers Fixed (3.0)", "sd3-comfy-diffusers-fixed-framework-28"),
            ):
                fixture = fixtures.get(case_id)
                if fixture is None:
                    raise ScheduleContractError("SD3 host case has no canonical workflow")
                sd3_workflow = cast(dict[str, object], fixture.workflow)

                def submit_sd3(
                    ordinal: int,
                    *,
                    selected_mode: str = mode,
                    workflow: dict[str, object] = sd3_workflow,
                    selected_case: str = case_id,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=(f"sigmax-m6-05-sd3-{selected_case}-attempt-{ordinal}"),
                        prompt=build_sd3_h2_api_prompt(selected_mode),
                        extra_data={"extra_pnginfo": {"workflow": workflow}},
                        execution_timeout=args.execution_timeout,
                    )

                def verify_sd3(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_mode: str = mode,
                ) -> dict[str, object]:
                    return verify_sd3_h2_history(
                        history,
                        prompt_id=prompt_id,
                        mode=selected_mode,
                    )

                sd3_summary, sd3_transition = execute_verified_host_repeat(
                    lane="H2_SD3_M6_05",
                    submit=submit_sd3,
                    verify=verify_sd3,
                )
                sd3_results.append(sd3_summary)
                attempts[f"h2_sd3.{case_id}"] = sd3_transition
            evidence["h2_sd3"] = sd3_results

            aura_fixture = fixtures.get("auraflow-v0-2-official-50")
            if aura_fixture is None:
                raise ScheduleContractError("AuraFlow host case has no canonical workflow")
            aura_workflow = cast(dict[str, object], aura_fixture.workflow)

            def submit_aura_flow(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_successful_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m6-05-auraflow-attempt-{ordinal}",
                    prompt=build_aura_flow_h2_api_prompt(),
                    extra_data={"extra_pnginfo": {"workflow": aura_workflow}},
                    execution_timeout=args.execution_timeout,
                )

            def verify_aura_flow(history: object, prompt_id: str) -> dict[str, object]:
                return verify_aura_flow_h2_history(history, prompt_id=prompt_id)

            aura_summary, aura_transition = execute_verified_host_repeat(
                lane="H2_AURAFLOW_M6_05",
                submit=submit_aura_flow,
                verify=verify_aura_flow,
            )
            evidence["h2_auraflow"] = aura_summary
            attempts["h2_auraflow.auraflow-v0-2-official-50"] = aura_transition

            lumina2_fixture = fixtures.get("lumina2-v2-official-50")
            if lumina2_fixture is None:
                raise ScheduleContractError("Lumina2 host case has no canonical workflow")
            lumina2_workflow = cast(dict[str, object], lumina2_fixture.workflow)

            def submit_lumina2(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_successful_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m6-05-lumina2-attempt-{ordinal}",
                    prompt=build_lumina2_h2_api_prompt(),
                    extra_data={"extra_pnginfo": {"workflow": lumina2_workflow}},
                    execution_timeout=args.execution_timeout,
                )

            def verify_lumina2(history: object, prompt_id: str) -> dict[str, object]:
                return verify_lumina2_h2_history(history, prompt_id=prompt_id)

            lumina2_summary, lumina2_transition = execute_verified_host_repeat(
                lane="H2_LUMINA2_M6_05",
                submit=submit_lumina2,
                verify=verify_lumina2,
            )
            evidence["h2_lumina2"] = lumina2_summary
            attempts["h2_lumina2.lumina2-v2-official-50"] = lumina2_transition

            hunyuan_base_fixture = fixtures.get("hunyuan-image21-base-official-50")
            hunyuan_distilled_fixture = fixtures.get("hunyuan-image21-distilled-official-8")
            if hunyuan_base_fixture is None or hunyuan_distilled_fixture is None:
                raise ScheduleContractError(
                    "HunyuanImage 2.1 host cases have no canonical workflows"
                )
            hunyuan_results: list[dict[str, object]] = []
            for variant, fixture, case_id in (
                ("Base (5.0)", hunyuan_base_fixture, "hunyuan-image21-base-official-50"),
                (
                    "Distilled (4.0)",
                    hunyuan_distilled_fixture,
                    "hunyuan-image21-distilled-official-8",
                ),
            ):
                hunyuan_workflow = cast(dict[str, object], fixture.workflow)

                def submit_hunyuan(
                    ordinal: int,
                    *,
                    selected_variant: str = variant,
                    selected_case: str = case_id,
                    selected_workflow: dict[str, object] = hunyuan_workflow,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=(
                            f"sigmax-m6-05-hunyuan-image21-{selected_case}-attempt-{ordinal}"
                        ),
                        prompt=build_hunyuan_image21_h2_api_prompt(selected_variant),
                        extra_data={"extra_pnginfo": {"workflow": selected_workflow}},
                        execution_timeout=args.execution_timeout,
                    )

                def verify_hunyuan(
                    history: object, prompt_id: str, *, selected_variant: str = variant
                ) -> dict[str, object]:
                    return verify_hunyuan_image21_h2_history(
                        history, prompt_id=prompt_id, variant=selected_variant
                    )

                hunyuan_summary, hunyuan_transition = execute_verified_host_repeat(
                    lane="H2_HUNYUAN_IMAGE21_M6_05",
                    submit=submit_hunyuan,
                    verify=verify_hunyuan,
                )
                hunyuan_results.append(hunyuan_summary)
                attempts[f"h2_hunyuan_image21.{case_id}"] = hunyuan_transition
            evidence["h2_hunyuan_image21"] = hunyuan_results

            anima_results: list[dict[str, object]] = []
            for variant, case_id in (
                ("Base (3.0)", "anima-base-v1-framework-50"),
                ("Aesthetic (3.0)", "anima-aesthetic-v1-framework-50"),
                ("Turbo (3.0)", "anima-turbo-v1-framework-8"),
            ):
                anima_fixture = fixtures.get(case_id)
                if anima_fixture is None:
                    raise ScheduleContractError("Anima host case has no canonical workflow")

                def submit_anima(
                    ordinal: int,
                    *,
                    selected_variant: str = variant,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=(
                            f"sigmax-m6-05-anima-"
                            f"{selected_variant.casefold().replace(' ', '-')}-attempt-{ordinal}"
                        ),
                        prompt=build_anima_h2_api_prompt(selected_variant),
                        execution_timeout=args.execution_timeout,
                    )

                def verify_anima(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_variant: str = variant,
                ) -> dict[str, object]:
                    return verify_anima_h2_history(
                        history,
                        prompt_id=prompt_id,
                        variant=selected_variant,
                    )

                anima_summary, anima_transition = execute_verified_host_repeat(
                    lane="H2_ANIMA_M6_05",
                    submit=submit_anima,
                    verify=verify_anima,
                )
                anima_results.append(anima_summary)
                attempts[f"h2_anima.{case_id}"] = anima_transition
            evidence["h2_anima"] = anima_results

            wan_results: list[dict[str, object]] = []
            for case in (
                {
                    "id": "wan21-t2v-official-50",
                    "generation": "Wan 2.1",
                    "task": "T2V",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 50,
                },
                {
                    "id": "wan21-i2v-480p-official-40",
                    "generation": "Wan 2.1",
                    "task": "I2V",
                    "source": "Official native",
                    "resolution": "480P",
                    "steps": 40,
                },
                {
                    "id": "wan22-ti2v-5b-native-50",
                    "generation": "Wan 2.2",
                    "task": "TI2V",
                    "source": "ComfyUI native",
                    "resolution": "None",
                    "steps": 50,
                },
                {
                    "id": "wan22-t2v-a14b-native-40",
                    "generation": "Wan 2.2",
                    "task": "T2V A14B",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 40,
                },
                {
                    "id": "wan21-flf2v-720p-official-50",
                    "generation": "Wan 2.1",
                    "task": "FLF2V",
                    "source": "Official native",
                    "resolution": "720P",
                    "steps": 50,
                },
                {
                    "id": "wan21-vace-1-3b-official-50",
                    "generation": "Wan 2.1",
                    "task": "VACE 1.3B",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 50,
                },
                {
                    "id": "wan21-vace-14b-official-50",
                    "generation": "Wan 2.1",
                    "task": "VACE 14B",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 50,
                },
                {
                    "id": "wan22-s2v-14b-official-40",
                    "generation": "Wan 2.2",
                    "task": "S2V",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 40,
                },
                {
                    "id": "wan22-animate-14b-official-20",
                    "generation": "Wan 2.2",
                    "task": "Animate",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 20,
                },
                {
                    "id": "wan-animate2-base-14b-official-40",
                    "generation": "Wan Animate 2",
                    "task": "Animate Base",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 40,
                },
                {
                    "id": "wan-animate2-distilled-14b-official-10",
                    "generation": "Wan Animate 2",
                    "task": "Animate Distilled",
                    "source": "Official native",
                    "resolution": "None",
                    "steps": 10,
                },
            ):
                case_id = cast(str, case["id"])
                wan_fixture = fixtures.get(case_id)
                if wan_fixture is None:
                    raise ScheduleContractError("Wan host case has no canonical workflow")
                generation = cast(str, case["generation"])
                task = cast(str, case["task"])
                source = cast(str, case["source"])
                resolution = cast(str, case["resolution"])
                steps = cast(int, case["steps"])
                wan_workflow = cast(dict[str, object], wan_fixture.workflow)

                def submit_wan(
                    ordinal: int,
                    *,
                    selected_generation: str = generation,
                    selected_task: str = task,
                    selected_source: str = source,
                    selected_resolution: str = resolution,
                    selected_steps: int = steps,
                    selected_case: str = case_id,
                    selected_workflow: dict[str, object] = wan_workflow,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-m6-05-wan-{selected_case}-attempt-{ordinal}",
                        prompt=build_wan_h2_api_prompt(
                            generation=selected_generation,
                            task=selected_task,
                            source=selected_source,
                            resolution=selected_resolution,
                            steps=selected_steps,
                        ),
                        extra_data={"extra_pnginfo": {"workflow": selected_workflow}},
                        execution_timeout=args.execution_timeout,
                    )

                def verify_wan(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_generation: str = generation,
                    selected_task: str = task,
                    selected_source: str = source,
                    selected_resolution: str = resolution,
                    selected_steps: int = steps,
                ) -> dict[str, object]:
                    return verify_wan_h2_history(
                        history,
                        prompt_id=prompt_id,
                        generation=selected_generation,
                        task=selected_task,
                        source=selected_source,
                        resolution=selected_resolution,
                        steps=selected_steps,
                    )

                wan_summary, wan_transition = execute_verified_host_repeat(
                    lane="H2_WAN_M4_14",
                    submit=submit_wan,
                    verify=verify_wan,
                )
                wan_results.append(wan_summary)
                attempts[f"h2_wan.{case_id}"] = wan_transition
            evidence["h2_wan"] = wan_results

            ltx_results: list[dict[str, object]] = []
            for case in (
                {
                    "id": "ltxv-0-9-8-dev-20",
                    "generation": "LTXV 0.9.8",
                    "stage": "Dev",
                    "steps": 20,
                },
                {
                    "id": "ltx2-19b-distilled-stage1-8",
                    "generation": "LTX-2 19B",
                    "stage": "Distilled Stage 1",
                    "steps": 8,
                },
                {
                    "id": "ltx2-3-22b-dev-30",
                    "generation": "LTX-2.3 22B",
                    "stage": "Dev",
                    "steps": 30,
                },
                {
                    "id": "ltx2-3-22b-distilled-stage2-3",
                    "generation": "LTX-2.3 22B",
                    "stage": "Distilled Stage 2",
                    "steps": 3,
                },
            ):
                case_id = cast(str, case["id"])
                ltx_fixture = fixtures.get(case_id)
                if ltx_fixture is None:
                    raise ScheduleContractError("LTX host case has no canonical workflow")
                generation = cast(str, case["generation"])
                stage = cast(str, case["stage"])
                steps = cast(int, case["steps"])
                ltx_workflow = cast(dict[str, object], ltx_fixture.workflow)

                def submit_ltx(
                    ordinal: int,
                    *,
                    selected_generation: str = generation,
                    selected_stage: str = stage,
                    selected_steps: int = steps,
                    selected_case: str = case_id,
                    selected_workflow: dict[str, object] = ltx_workflow,
                ) -> tuple[str, dict[str, object]]:
                    return _submit_successful_prompt(
                        base_url=base_url,
                        client_id=f"sigmax-m6-05-ltx-{selected_case}-attempt-{ordinal}",
                        prompt=build_ltx_h2_api_prompt(
                            generation=selected_generation,
                            stage=selected_stage,
                            steps=selected_steps,
                        ),
                        extra_data={"extra_pnginfo": {"workflow": selected_workflow}},
                        execution_timeout=args.execution_timeout,
                    )

                def verify_ltx(
                    history: object,
                    prompt_id: str,
                    *,
                    selected_generation: str = generation,
                    selected_stage: str = stage,
                    selected_steps: int = steps,
                ) -> dict[str, object]:
                    return verify_ltx_h2_history(
                        history,
                        prompt_id=prompt_id,
                        generation=selected_generation,
                        stage=selected_stage,
                        steps=selected_steps,
                    )

                ltx_summary, ltx_transition = execute_verified_host_repeat(
                    lane="H2_LTX_M6_05",
                    submit=submit_ltx,
                    verify=verify_ltx,
                )
                ltx_results.append(ltx_summary)
                attempts[f"h2_ltx.{case_id}"] = ltx_transition
            evidence["h2_ltx"] = ltx_results

            def submit_runtime_rejection(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_rejected_runtime_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m3-06-raw-auto-variant-attempt-{ordinal}",
                    prompt=_rejected_raw_api_prompt("raw-auto-variant"),
                    execution_timeout=args.execution_timeout,
                )

            def verify_runtime_rejection(
                history: object,
                prompt_id: str,
            ) -> dict[str, object]:
                return verify_rejected_history(
                    history,
                    prompt_id=prompt_id,
                    case_id="raw-auto-variant",
                    expected_message=("variant must be a supported explicit Krea 2 variant"),
                )

            runtime_rejection, runtime_transition = execute_verified_host_repeat(
                lane="H2_RAW_M3_06",
                submit=submit_runtime_rejection,
                verify=verify_runtime_rejection,
            )
            rejected_results = [runtime_rejection]
            attempts["h2_raw.raw-auto-variant"] = runtime_transition

            def submit_prequeue_rejection(ordinal: int) -> tuple[str, dict[str, object]]:
                response = _submit_rejected_prequeue_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m3-06-raw-invalid-steps-attempt-{ordinal}",
                    prompt=_rejected_raw_api_prompt("raw-invalid-steps"),
                )
                return f"prequeue-attempt-{ordinal}", _object(
                    response,
                    label="prequeue rejection response",
                )

            def verify_prequeue(response: object, _attempt_id: str) -> dict[str, object]:
                return verify_prequeue_rejection(
                    response,
                    case_id="raw-invalid-steps",
                )

            prequeue_rejection, prequeue_transition = execute_verified_host_repeat(
                lane="H2_RAW_M3_06",
                submit=submit_prequeue_rejection,
                verify=verify_prequeue,
            )
            rejected_results.append(prequeue_rejection)
            attempts["h2_raw.raw-invalid-steps"] = prequeue_transition
            evidence["h2_raw_rejections"] = rejected_results

            def submit_algebra(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_successful_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m4-09-algebra-attempt-{ordinal}",
                    prompt=build_schedule_algebra_h2_api_prompt(),
                    execution_timeout=args.execution_timeout,
                )

            def verify_algebra(history: object, prompt_id: str) -> dict[str, object]:
                return verify_schedule_algebra_h2_history(history, prompt_id=prompt_id)

            algebra_summary, algebra_transition = execute_verified_host_repeat(
                lane="H2_ALGEBRA_M4_09",
                submit=submit_algebra,
                verify=verify_algebra,
            )
            evidence["h2_schedule_algebra"] = algebra_summary
            attempts["h2_schedule_algebra"] = algebra_transition

            def submit_algebra_rejection(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_rejected_runtime_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m4-09-algebra-noop-attempt-{ordinal}",
                    prompt=build_schedule_algebra_h2_noop_rejection_prompt(),
                    execution_timeout=args.execution_timeout,
                )

            def verify_algebra_rejection(
                history: object,
                prompt_id: str,
            ) -> dict[str, object]:
                return verify_schedule_algebra_h2_noop_rejection(
                    history,
                    prompt_id=prompt_id,
                )

            algebra_rejection, algebra_rejection_transition = execute_verified_host_repeat(
                lane="H2_ALGEBRA_M4_09",
                submit=submit_algebra_rejection,
                verify=verify_algebra_rejection,
            )
            evidence["h2_schedule_algebra_rejections"] = [algebra_rejection]
            attempts["h2_schedule_algebra.noop_resample"] = algebra_rejection_transition

            def submit_checkpoint_evidence(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_successful_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m6-08-checkpoint-evidence-attempt-{ordinal}",
                    prompt=build_checkpoint_evidence_h2_api_prompt(),
                    execution_timeout=args.execution_timeout,
                )

            def verify_checkpoint_evidence(
                history: object,
                prompt_id: str,
            ) -> dict[str, object]:
                return verify_checkpoint_evidence_h2_history(history, prompt_id=prompt_id)

            checkpoint_summary, checkpoint_transition = execute_verified_host_repeat(
                lane="H2_CHECKPOINT_EVIDENCE_M6_08",
                submit=submit_checkpoint_evidence,
                verify=verify_checkpoint_evidence,
            )
            evidence["h2_checkpoint_evidence"] = checkpoint_summary
            attempts["h2_checkpoint_evidence"] = checkpoint_transition

            def submit_h3(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_successful_prompt(
                    base_url=base_url,
                    client_id=f"sigmax-m5-01-native-euler-attempt-{ordinal}",
                    prompt=build_native_euler_h3_api_prompt(),
                    execution_timeout=args.execution_timeout,
                )

            def verify_h3(history: object, prompt_id: str) -> dict[str, object]:
                return verify_native_euler_h3_history(history, prompt_id=prompt_id)

            h3_summary, h3_transition = execute_verified_host_repeat(
                lane="H3_EULER_M5_01",
                submit=submit_h3,
                verify=verify_h3,
            )
            evidence["h3_native_euler"] = h3_summary
            attempts["h3_native_euler"] = h3_transition

            def submit_h3_rejection(ordinal: int) -> tuple[str, dict[str, object]]:
                return _submit_rejected_runtime_prompt(
                    base_url=base_url,
                    client_id=(f"sigmax-m5-01-native-euler-partial-rejection-attempt-{ordinal}"),
                    prompt=build_native_euler_h3_partial_rejection_prompt(),
                    execution_timeout=args.execution_timeout,
                )

            def verify_h3_rejection(history: object, prompt_id: str) -> dict[str, object]:
                return verify_native_euler_h3_partial_rejection(
                    history,
                    prompt_id=prompt_id,
                )

            h3_rejection, h3_rejection_transition = execute_verified_host_repeat(
                lane="H3_EULER_M5_01",
                submit=submit_h3_rejection,
                verify=verify_h3_rejection,
            )
            evidence["h3_native_euler_rejections"] = [h3_rejection]
            attempts["h3_native_euler.partial_denoise"] = h3_rejection_transition
            succeeded = True
    finally:
        if process is not None:
            shutdown = _terminate_owned_process(process, base_url=base_url)
        _wait_for_port_release(port)
        evidence["shutdown"] = shutdown
        evidence["duration_seconds"] = round(time.time() - started, 3)
        if log_path.exists():
            evidence["host_log_tail"] = redact_text(
                log_path.read_text(encoding="utf-8", errors="replace"),
                sensitive_paths=(REPOSITORY_ROOT, comfyui_root, run_path, host_python),
            )[-8_000:]
        evidence["cleanup"] = "removed" if succeeded else "retained_failure_artifacts"
        _write_evidence(Path(args.evidence_file) if args.evidence_file else None, evidence)
        if succeeded:
            shutil.rmtree(run_path)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditioning-only",
        action="store_true",
        help="run only the M4-12 conditioning H1/H2 lane",
    )
    parser.add_argument(
        "--minimax-h3-only",
        action="store_true",
        help="run only the model-free MiniMax H3 H1/H2 contract on the pinned 0.30.0 host",
    )
    parser.add_argument(
        "--minimax-h3-model-plan",
        action="store_true",
        help="build an authorization-gated H3 model workflow plan without executing weights",
    )
    parser.add_argument(
        "--minimax-h3-model-variant",
        choices=_MINIMAX_H3_PUBLIC_VARIANTS,
        default=None,
        help="explicit H3 variant for --minimax-h3-model-plan",
    )
    parser.add_argument(
        "--minimax-h3-model-prompt",
        default=None,
        help="prompt for --minimax-h3-model-plan",
    )
    parser.add_argument(
        "--minimax-h3-model-artifact",
        default=None,
        help="host-relative diffusion artifact for --minimax-h3-model-plan",
    )
    parser.add_argument(
        "--minimax-h3-first-frame",
        default=None,
        help="host-relative first-frame image for --minimax-h3-model-plan",
    )
    parser.add_argument(
        "--minimax-h3-last-frame",
        default=None,
        help="host-relative last-frame image for --minimax-h3-model-plan",
    )
    parser.add_argument(
        "--minimax-h3-reference-image",
        action="append",
        default=[],
        help="repeatable host-relative Ref2VA image for --minimax-h3-model-plan",
    )
    parser.add_argument(
        "--minimax-h3-allow-model-weights",
        action="store_true",
        help="explicitly authorize a future weight-backed H3 host lane",
    )
    parser.add_argument(
        "--minimax-h3-license-ack",
        action="store_true",
        help="acknowledge the local MiniMax H3 license boundary for a future model lane",
    )
    parser.add_argument(
        "--minimax-h3-expected-revision",
        default=MINIMAX_H3_COMFYUI_REVISION,
        help="exact ComfyUI revision required by --minimax-h3-only",
    )
    parser.add_argument(
        "--minimax-h3-host-version",
        default=_MINIMAX_H3_HOST_VERSION,
        help="host version required by --minimax-h3-only",
    )
    parser.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT"))
    parser.add_argument("--host-python", default=os.environ.get("SIGMAX_COMFYUI_PYTHON"))
    parser.add_argument(
        "--expected-revision",
        default=os.environ.get("SIGMAX_COMFYUI_REVISION", CANONICAL_HOST_REVISION),
    )
    parser.add_argument(
        "--host-version",
        default=CANONICAL_HOST_VERSION,
    )
    parser.add_argument(
        "--validation-lane",
        choices=[item.value for item in WorkflowValidationLane] + [_MINIMAX_H3_LATEST_LANE],
        default=WorkflowValidationLane.KNOWN_GOOD.value,
    )
    parser.add_argument(
        "--temp-root",
        default=os.environ.get("SIGMAX_E2E_TMP", str(REPOSITORY_ROOT / ".tmp" / "e2e")),
    )
    parser.add_argument("--evidence-file", default="")
    parser.add_argument("--readiness-timeout", type=float, default=180.0)
    parser.add_argument("--execution-timeout", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.minimax_h3_model_plan:
        if args.conditioning_only or args.minimax_h3_only:
            parser.error(
                "--minimax-h3-model-plan cannot be combined with --conditioning-only or --minimax-h3-only"
            )
        if not args.minimax_h3_model_variant:
            parser.error("--minimax-h3-model-variant is required with --minimax-h3-model-plan")
        if not args.minimax_h3_model_prompt:
            parser.error("--minimax-h3-model-prompt is required with --minimax-h3-model-plan")
        if not args.minimax_h3_model_artifact:
            parser.error("--minimax-h3-model-artifact is required with --minimax-h3-model-plan")
        try:
            evidence = build_minimax_h3_model_lane_plan(
                variant=args.minimax_h3_model_variant,
                prompt=args.minimax_h3_model_prompt,
                model_artifact=args.minimax_h3_model_artifact,
                allow_model_weights=args.minimax_h3_allow_model_weights,
                license_ack=args.minimax_h3_license_ack,
                first_frame=args.minimax_h3_first_frame,
                last_frame=args.minimax_h3_last_frame,
                reference_images=tuple(args.minimax_h3_reference_image),
            )
        except Exception as exc:
            print(
                redact_text(
                    f"MiniMax H3 model plan failed: {type(exc).__name__}: {exc}",
                    sensitive_paths=(REPOSITORY_ROOT,),
                ),
                file=sys.stderr,
            )
            return 1
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    if not args.comfyui_root:
        parser.error("COMFYUI_ROOT or --comfyui-root is required")
    if not args.host_python:
        parser.error("SIGMAX_COMFYUI_PYTHON or --host-python is required")
    if args.validation_lane == _MINIMAX_H3_LATEST_LANE and not args.minimax_h3_only:
        parser.error("--validation-lane latest is reserved for --minimax-h3-only")
    try:
        if args.conditioning_only and args.minimax_h3_only:
            parser.error("--conditioning-only and --minimax-h3-only are mutually exclusive")
        if args.minimax_h3_only:
            evidence = run_minimax_h3(args)
        elif args.conditioning_only:
            evidence = run_conditioning(args)
        else:
            evidence = run(args)
    except Exception as exc:
        print(
            redact_text(
                f"ComfyUI E2E failed: {type(exc).__name__}: {exc}",
                sensitive_paths=(REPOSITORY_ROOT,),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
