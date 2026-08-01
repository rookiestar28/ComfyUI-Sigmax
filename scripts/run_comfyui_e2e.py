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
from comfyui_sigmax.profiles import KREA2_TURBO_SCHEMA  # noqa: E402
from comfyui_sigmax.workflows import (  # noqa: E402
    WorkflowValidationLane,
    load_canonical_workflow_fixtures,
    validate_live_workflow_fixtures,
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
_ALGEBRA_OUTPUT_NODE_ID: Final = "7"
_ALGEBRA_TRACE_KEY: Final = "sigmax_schedule_algebra"
_CHECKPOINT_OUTPUT_NODE_ID: Final = "2"
_CHECKPOINT_TRACE_KEY: Final = "sigmax_checkpoint_evidence"
_Z_IMAGE_OUTPUT_NODE_ID: Final = "2"
_Z_IMAGE_TRACE_KEY: Final = "sigmax_z_image_schedule"
_FLUX1_SCHNELL_OUTPUT_NODE_ID: Final = "2"
_FLUX1_SCHNELL_TRACE_KEY: Final = "sigmax_flux1_schnell_schedule"
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
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in sigmas)
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
    if isinstance(value, (list, tuple)):
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


def _stage_extension(run_path: Path) -> Path:
    custom_node = run_path / "base" / "custom_nodes" / "ComfyUI-Sigmax"
    custom_node.mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / "__init__.py", custom_node / "__init__.py")
    shutil.copytree(
        REPOSITORY_ROOT / "comfyui_sigmax",
        custom_node / "comfyui_sigmax",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
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
            "H2_RAW_M3_06",
            "H2_ALGEBRA_M4_09",
            "H2_CHECKPOINT_EVIDENCE_M6_08",
            "H2_Z_IMAGE_M6_04",
            "H2_FLUX1_SCHNELL_M6_05",
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
                    expected_message="variant must be Turbo or RAW",
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
        choices=[item.value for item in WorkflowValidationLane],
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
    if not args.comfyui_root:
        parser.error("COMFYUI_ROOT or --comfyui-root is required")
    if not args.host_python:
        parser.error("SIGMAX_COMFYUI_PYTHON or --host-python is required")
    try:
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
