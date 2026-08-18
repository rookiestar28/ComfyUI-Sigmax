"""Run the authorization-gated private MiniMax H3 H4 validation lane.

This script is deliberately outside the public node/runtime surface.  It requires an explicit
protocol, exact Sigmax candidate, caller-supplied host/model files, and an explicit GPU switch
before it can start a host.  Turbo artifacts are never discovered or substituted: publisher-full
artifacts must be supplied with an exact allowlisted identity; the observed reduced/local files
remain blocked or rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Final, NoReturn, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from comfyui_sigmax.core import ScheduleContractError  # noqa: E402
from comfyui_sigmax.core.safetensors_header import (  # noqa: E402
    SafetensorsHeader,
    SafetensorsHeaderError,
    read_safetensors_header,
)
from comfyui_sigmax.profiles.minimax_h3_turbo import (  # noqa: E402
    MiniMaxH3TurboError,
    MiniMaxH3TurboProfile,
    build_minimax_h3_turbo_schedule,
    get_minimax_h3_turbo_profile,
)

_LOOPBACK: Final = "127.0.0.1"
_MAX_HTTP_BYTES: Final = 4_000_000
_MAX_LOG_BYTES: Final = 1_000_000
_MAX_DISPATCH_TRACE_BYTES: Final = 64 * 1024
_MAX_REFERENCE_IMAGE_BYTES: Final = 16 * 1024 * 1024
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_REVISION: Final = re.compile(r"^[0-9a-f]{40}$")
_SEMVER: Final = re.compile(r"^\d+\.\d+\.\d+$")
_PRIVATE_PATH: Final = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|users|mnt)/)")
_SECRET: Final = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|token|cookie|authorization)"
)
_DISPATCH_TRACE_SCHEMA: Final = "sigmax.h4-dispatch-observation/1"
_DISPATCH_ATTENTION_BACKENDS: Final = frozenset(
    {"ck_int8", "flash", "pytorch", "sage", "sage3", "split", "sub_quad", "xformers"}
)
_DISPATCH_OPERATION_BACKENDS: Final = frozenset({"cuda", "eager", "hip", "triton"})
_DISPATCH_OPERATION_NAMES: Final = frozenset(
    {
        "convrot_w4a4_linear",
        "gemv_awq_w4a16",
        "int8_linear",
        "scaled_mm_mxfp8",
        "scaled_mm_nvfp4",
        "scaled_mm_svdquant_w4a4",
        "w4a8_int8_linear",
    }
)
_DISPATCH_ADAPTER_VERSION: Final = "m7-13-h4-dispatch-observer/1"

H4_SCHEMA: Final = "sigmax.minimax-h3-h4-private-validation/2"
PROTOCOL_STATUS: Final = "ACTIVE_PENDING_REVIEW"
AUTHORIZATION_MARKER: Final = "M7-13-H4-AUTHORIZED-2026-08-18"
_DEFAULT_PROTOCOL: Final = (
    REPOSITORY_ROOT / ".planning" / "260818-M7-13_MINIMAX_H3_ACCELERATED_VALIDATION_PROTOCOL.md"
)
_DEFAULT_AUTHORIZATION: Final = REPOSITORY_ROOT / ".planning" / "260818-M7-13_H4_AUTHORIZATION.md"

_PUBLISHER_ARTIFACTS: Final = {
    "h3.fl2va.lightx2v-turbo-4-v1.0-768p": (
        1956192992,
        "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",  # pragma: allowlist secret
    ),
    "h3.fl2va.lightx2v-turbo-8-v1.0-544p": (
        1956193000,
        "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",  # pragma: allowlist secret
    ),
    "h3.ref2va.lightx2v-turbo-4-v0.1-544p": (
        1956193000,
        "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",  # pragma: allowlist secret
    ),
}


@dataclass(frozen=True, slots=True)
class TurboRowSpec:
    """One protocol row's immutable recipe/task binding for private H4 execution."""

    row: str
    recipe_id: str
    variant: str
    steps: int
    shift_video: float
    shift_audio: float
    width: int
    height: int
    requires_reference_image: bool


_TURBO_ROW_SPECS: Final = {
    "T4-768": TurboRowSpec(
        row="T4-768",
        recipe_id="h3.fl2va.lightx2v-turbo-4-v1.0-768p",
        variant="H3 Base FL2VA",
        steps=4,
        shift_video=6.0,
        shift_audio=3.0,
        width=1344,
        height=768,
        requires_reference_image=False,
    ),
    "T8-544": TurboRowSpec(
        row="T8-544",
        recipe_id="h3.fl2va.lightx2v-turbo-8-v1.0-544p",
        variant="H3 Base FL2VA",
        steps=8,
        shift_video=12.0,
        shift_audio=3.0,
        width=960,
        height=544,
        requires_reference_image=False,
    ),
    "T4-544": TurboRowSpec(
        row="T4-544",
        recipe_id="h3.fl2va.lightx2v-turbo-4-v0.1-544p",
        variant="H3 Base FL2VA",
        steps=4,
        shift_video=12.0,
        shift_audio=3.0,
        width=960,
        height=544,
        requires_reference_image=False,
    ),
    "R4-544": TurboRowSpec(
        row="R4-544",
        recipe_id="h3.ref2va.lightx2v-turbo-4-v0.1-544p",
        variant="H3 Base Ref2VA",
        steps=4,
        shift_video=12.0,
        shift_audio=3.0,
        width=960,
        height=544,
        requires_reference_image=True,
    ),
}

_BLOCKED_REDUCED: Final = {
    "9515eee9f642aa0e7fcc401f56d408ef2d6388f81881fe50bddded8220870a4d",  # pragma: allowlist secret
    "8e05b7b982c3aff7deb692a188c8a8d8acaeff8a12abfe1aeac822fb8ee3f0b7",  # pragma: allowlist secret
    "9ea3bd3a6aac22994153e294cf1ecab0a8766fc0f8d056ace645a01d1a6a4daf",  # pragma: allowlist secret
}
_REJECTED_LOCAL: Final = {
    "1b85da614014024a0c9507f12558917dcc69b6adb564e716324594f401723115",  # pragma: allowlist secret
    "a3208be61329c27a6754c53db9a21a3c86e2a285381700adf2d97e279c062840",  # pragma: allowlist secret
    "2c6abb194cff3e26c2295c87892913adf0c92d8f784f305238246759f9b333d0",  # pragma: allowlist secret
}


class RowDisposition(str, Enum):
    ACCEPTED = "accepted"
    BLOCKED = "blocked"
    REJECTED = "rejected"
    NO_PROMOTION = "no_promotion"
    NOT_EXECUTED = "not_executed"


@dataclass(frozen=True, slots=True)
class ArtifactObservation:
    """Private artifact evidence with a path-free public projection."""

    artifact_id: str
    disposition: RowDisposition
    reason_code: str | None
    file_bytes: int | None
    sha256: str | None
    header_bytes: int | None
    tensor_count: int | None
    dtype_counts: tuple[tuple[str, int], ...]

    def projection(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "disposition": self.disposition.value,
            "dtype_counts": {key: value for key, value in self.dtype_counts},
            "file_bytes": self.file_bytes,
            "header_bytes": self.header_bytes,
            "reason_code": self.reason_code,
            "sha256": None if self.sha256 is None else f"sha256:{self.sha256}",
            "tensor_count": self.tensor_count,
        }


def _fail(message: str) -> NoReturn:
    raise ScheduleContractError(message)


def _turbo_row_spec(row: str) -> TurboRowSpec:
    spec = _TURBO_ROW_SPECS.get(row)
    if spec is None:
        _fail("H4 Turbo row is not in the frozen protocol matrix")
    try:
        profile = get_minimax_h3_turbo_profile(spec.recipe_id)
    except MiniMaxH3TurboError as exc:
        raise ScheduleContractError("H4 Turbo row profile is not registered") from exc
    if (
        profile.task != ("ref2va" if spec.variant == "H3 Base Ref2VA" else "fl2va")
        or profile.video_shift != spec.shift_video
        or profile.audio_shift != spec.shift_audio
        or spec.steps not in profile.allowed_nfe
    ):
        _fail("H4 Turbo row binding disagrees with the pure recipe profile")
    return spec


def _validate_h4_schema(value: object) -> None:
    if value != H4_SCHEMA:
        _fail("H4 evidence schema is not the current private version")


def _safe_relative_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(f"{field} must be non-empty relative text")
    normalized = value.replace("\\", "/")
    if (
        normalized.startswith("/")
        or re.match(r"^[A-Za-z]:/", normalized) is not None
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        _fail(f"{field} must be a host-relative model name")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, PermissionError) as exc:
        raise ScheduleContractError("artifact hash could not be read") from exc
    return digest.hexdigest()


def _header_observation(path: Path) -> tuple[SafetensorsHeader | None, int | None]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            return read_safetensors_header(stream, file_size=file_size), file_size
    except FileNotFoundError:
        return None, None
    except (OSError, SafetensorsHeaderError) as exc:
        raise ScheduleContractError("artifact safetensors header is invalid") from exc


def _artifact_observation(
    *,
    path: Path,
    artifact_id: str,
    disposition: RowDisposition,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    reason_code: str | None = None,
) -> ArtifactObservation:
    """Hash and inspect one explicit file; never crawls a model directory."""

    if not path.is_file():
        return ArtifactObservation(
            artifact_id=artifact_id,
            disposition=RowDisposition.BLOCKED,
            reason_code="artifact.unavailable",
            file_bytes=None,
            sha256=None,
            header_bytes=None,
            tensor_count=None,
            dtype_counts=(),
        )
    header, file_bytes = _header_observation(path)
    if header is None or file_bytes is None:
        return ArtifactObservation(
            artifact_id=artifact_id,
            disposition=RowDisposition.REJECTED,
            reason_code="artifact.header_unavailable",
            file_bytes=file_bytes,
            sha256=None,
            header_bytes=None,
            tensor_count=None,
            dtype_counts=(),
        )
    digest = _sha256_file(path)
    mismatch = (expected_sha256 is not None and digest != expected_sha256) or (
        expected_bytes is not None and file_bytes != expected_bytes
    )
    final_disposition = RowDisposition.REJECTED if mismatch else disposition
    final_reason = "artifact.hash_or_size_mismatch" if mismatch else reason_code
    counts = tuple(sorted(Counter(item.dtype for item in header.tensors).items()))
    return ArtifactObservation(
        artifact_id=artifact_id,
        disposition=final_disposition,
        reason_code=final_reason,
        file_bytes=file_bytes,
        sha256=digest,
        header_bytes=header.header_bytes,
        tensor_count=len(header.tensors),
        dtype_counts=counts,
    )


def _component_observation(
    *,
    models_root: Path,
    folder: str,
    name: str,
    artifact_id: str,
    expected_sha256: str | None,
) -> ArtifactObservation:
    """Inspect one explicitly named loader component without directory discovery."""

    _safe_relative_name(name, field=artifact_id)
    return _artifact_observation(
        path=models_root / folder / Path(name),
        artifact_id=artifact_id,
        disposition=RowDisposition.ACCEPTED,
        expected_sha256=expected_sha256,
        reason_code=None,
    )


def classify_turbo_artifact(
    *,
    path: Path,
    artifact_id: str,
    source: str,
    license_ack: bool,
) -> ArtifactObservation:
    """Apply the exact publisher/reduced/local disposition policy."""

    _safe_relative_name(path.name, field="turbo artifact filename")
    if source not in {"publisher-full", "kijai-reduced", "local-modified"}:
        _fail("turbo artifact source is unsupported")
    base = _artifact_observation(
        path=path,
        artifact_id=artifact_id,
        disposition=RowDisposition.BLOCKED,
        reason_code="artifact.provenance_or_license_blocked",
    )
    if base.sha256 is None:
        return base
    if base.sha256 in _REJECTED_LOCAL or source == "local-modified":
        return replace(
            base, disposition=RowDisposition.REJECTED, reason_code="artifact.local_modified"
        )
    if base.sha256 in _BLOCKED_REDUCED or source == "kijai-reduced":
        return base
    expected = _PUBLISHER_ARTIFACTS.get(artifact_id)
    if source != "publisher-full" or expected is None:
        return replace(
            base,
            reason_code=(
                "artifact.publisher_full_not_available"
                if source == "publisher-full" and expected is None
                else base.reason_code
            ),
        )
    expected_bytes, expected_hash = expected
    if not license_ack:
        return replace(base, reason_code="artifact.license_ack_required")
    return _artifact_observation(
        path=path,
        artifact_id=artifact_id,
        disposition=RowDisposition.ACCEPTED,
        expected_sha256=expected_hash,
        expected_bytes=expected_bytes or None,
        reason_code=None,
    )


def _protocol_binding(path: Path, *, expected_commit: str, expected_tree: str) -> None:
    if not path.is_file() or REPOSITORY_ROOT.resolve() not in path.resolve().parents:
        _fail("protocol file must be an existing repository-local file")
    text = path.read_text(encoding="utf-8")
    status = re.search(r"\*\*Protocol status:\*\*\s*`([^`]+)`", text)
    commit = re.search(r"^commit:\s*([0-9a-f]{40})\s*$", text, re.MULTILINE)
    tree = re.search(r"^tree:\s*([0-9a-f]{40})\s*$", text, re.MULTILINE)
    if status is None or status.group(1) != PROTOCOL_STATUS:
        _fail("M7-13 protocol is not active for preflight")
    if commit is None or tree is None:
        _fail("M7-13 protocol has no exact candidate binding")
    if commit.group(1) != expected_commit or tree.group(1) != expected_tree:
        _fail("M7-13 protocol candidate binding does not match current exact candidate")


def _git_output(*args: str) -> str:
    git = shutil.which("git")
    if git is None:
        _fail("git executable is unavailable")
    result = subprocess.run(  # noqa: S603
        [git, *args], cwd=REPOSITORY_ROOT, check=False, capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        _fail("git exact-candidate query failed")
    return result.stdout.strip()


def current_candidate() -> tuple[str, str]:
    commit = _git_output("rev-parse", "HEAD")
    tree = _git_output("rev-parse", "HEAD^{tree}")
    if _REVISION.fullmatch(commit) is None or _REVISION.fullmatch(tree) is None:
        _fail("current candidate identity is malformed")
    return commit, tree


def build_h4_prompt(
    *,
    variant: str,
    model_name: str,
    clip_name: str,
    video_vae_name: str,
    audio_vae_name: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    steps: int,
    seed: int,
    shift_video: float,
    shift_audio: float,
    lora_name: str | None = None,
    trace_file: str | None = None,
    requested_attention_backend: str = "pytorch",
    requested_operation_backend: str = "auto",
    recipe_id: str | None = None,
    reference_image_name: str | None = None,
) -> dict[str, dict[str, object]]:
    """Build a native H3 graph with one external Sigmax schedule and no double shift."""

    if variant not in {"H3 Base FL2VA", "H3 Base Ref2VA"}:
        _fail("H4 graph variant must be explicit")
    turbo_profile: MiniMaxH3TurboProfile | None = None
    if recipe_id is not None:
        try:
            turbo_profile = get_minimax_h3_turbo_profile(recipe_id)
        except MiniMaxH3TurboError as exc:
            raise ScheduleContractError("H4 graph Turbo recipe is unknown") from exc
        expected_task = "ref2va" if variant == "H3 Base Ref2VA" else "fl2va"
        if turbo_profile.task != expected_task:
            _fail("H4 graph Turbo recipe task does not match the selected variant")
        if steps not in turbo_profile.allowed_nfe:
            _fail("H4 graph Turbo recipe does not allow the requested NFE")
        if shift_video != turbo_profile.video_shift or shift_audio != turbo_profile.audio_shift:
            _fail("H4 graph Turbo shifts must remain recipe-owned")
        if lora_name is None:
            _fail("H4 Turbo graph requires an explicit LoRA artifact")
    elif lora_name is not None:
        _fail("H4 LoRA artifact requires an explicit Turbo recipe")
    if variant == "H3 Base Ref2VA" and reference_image_name is None:
        _fail("H4 Ref2VA graph requires an explicit reference image")
    if variant == "H3 Base FL2VA" and reference_image_name is not None:
        _fail("H4 FL2VA graph cannot receive a reference image")
    for field, value in (
        ("model_name", model_name),
        ("clip_name", clip_name),
        ("video_vae_name", video_vae_name),
        ("audio_vae_name", audio_vae_name),
    ):
        _safe_relative_name(value, field=field)
    # ComfyUI's Windows loader choices are serialized with backslashes; preserve the
    # caller-relative boundary while matching that exact host schema spelling.
    host_model_name = model_name.replace("/", "\\")
    host_clip_name = clip_name.replace("/", "\\")
    host_video_vae_name = video_vae_name.replace("/", "\\")
    host_audio_vae_name = audio_vae_name.replace("/", "\\")
    host_lora_name = None if lora_name is None else lora_name.replace("/", "\\")
    host_reference_image_name = (
        None
        if reference_image_name is None
        else _safe_relative_name(reference_image_name, field="reference_image_name")
    )
    if (
        not isinstance(prompt, str)
        or not prompt
        or _PRIVATE_PATH.search(prompt)
        or _SECRET.search(prompt)
    ):
        _fail("H4 prompt is private and may not contain path or secret text")
    # The preregistered protocol intentionally sends 17 as a negative-shape probe; native H3
    # may snap it to 22.  All other accepted lengths must already be on the 17k+5 grid.
    if length < 5 or (length != 17 and (length - 5) % 17):
        _fail("H4 length must use the frozen 17k+5 grid or the explicit 17-frame probe")
    if width < 32 or height < 32 or width % 32 or height % 32:
        _fail("H4 dimensions must be positive multiples of 32")
    if requested_attention_backend not in _DISPATCH_ATTENTION_BACKENDS:
        _fail("H4 attention backend request is not allowlisted")
    if requested_operation_backend != "auto":
        _fail("H4 operation backend request must remain auto")
    if trace_file is not None:
        trace_path = Path(trace_file)
        if not trace_path.is_absolute() or "\x00" in trace_file or ".." in trace_path.parts:
            _fail("H4 dispatch trace file must be absolute and traversal-free")
    model_id = "1"
    clip_id = "2"
    video_vae_id = "3"
    audio_vae_id = "4"
    model_sampling_id = "5"
    condition_id = "6"
    schedule_id = "7"
    sampler_id = "8"
    guider_id = "9"
    noise_id = "10"
    sample_id = "11"
    video_decode_id = "12"
    audio_decode_id = "13"
    video_id = "14"
    save_id = "15"
    observer_id = "16"
    finalizer_id = "17"
    reference_image_id = "18"
    model_link = [model_id, 0]
    if lora_name is not None:
        _safe_relative_name(lora_name, field="lora_name")
        model_link = [model_sampling_id, 0]
    schedule_inputs: dict[str, object] = {
        "variant": variant,
        "steps": steps,
        "start_step": 0,
        "end_step": -1,
    }
    if recipe_id is not None:
        schedule_inputs["recipe_id"] = recipe_id
    conditioning_inputs: dict[str, object] = {
        "clip": [clip_id, 0],
        "vae": [video_vae_id, 0],
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
    }
    if variant == "H3 Base Ref2VA":
        conditioning_inputs.update(
            {
                "audio_vae": [audio_vae_id, 0],
                "ref_image_size": "match",
                "ref_images.ref_image_0": [reference_image_id, 0],
            }
        )
    conditioning_node: dict[str, object] = {
        "class_type": (
            "MiniMaxH3ReferenceToVideo" if variant == "H3 Base Ref2VA" else "MiniMaxH3ImageToVideo"
        ),
        "inputs": conditioning_inputs,
    }
    return {
        model_id: {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": host_model_name, "weight_dtype": "default"},
        },
        clip_id: {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": host_clip_name, "type": "minimax", "device": "default"},
        },
        video_vae_id: {"class_type": "VAELoader", "inputs": {"vae_name": host_video_vae_name}},
        audio_vae_id: {"class_type": "VAELoader", "inputs": {"vae_name": host_audio_vae_name}},
        **(
            {
                model_sampling_id: {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": [model_id, 0],
                        "lora_name": host_lora_name,
                        "strength_model": 1.0,
                    },
                }
            }
            if lora_name is not None
            else {}
        ),
        **(
            {
                observer_id: {
                    "class_type": "Sigmax.H4DispatchObserver",
                    "inputs": {
                        "model": model_link,
                        "trace_file": trace_file,
                        "requested_attention_backend": requested_attention_backend,
                        "requested_operation_backend": requested_operation_backend,
                    },
                }
            }
            if trace_file is not None
            else {}
        ),
        condition_id: conditioning_node,
        # ModelSamplingAV already carries 12/3.  Do not insert MiniMaxH3SigmaShift here:
        # CRITICAL: Sigmax's schedule is already video-shifted; a second shift changes parity.
        schedule_id: {
            "class_type": "Sigmax.MiniMaxH3SigmaScheduler",
            "inputs": schedule_inputs,
        },
        sampler_id: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        guider_id: {
            "class_type": "BasicGuider",
            "inputs": {
                "model": [observer_id, 0] if trace_file is not None else model_link,
                "conditioning": [condition_id, 0],
            },
        },
        noise_id: {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        sample_id: {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [noise_id, 0],
                "guider": [guider_id, 0],
                "sampler": [sampler_id, 0],
                "sigmas": [schedule_id, 0],
                "latent_image": [condition_id, 1],
            },
        },
        video_decode_id: {
            "class_type": "VAEDecode",
            "inputs": {"samples": [sample_id, 0], "vae": [video_vae_id, 0]},
        },
        audio_decode_id: {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": [sample_id, 0], "vae": [audio_vae_id, 0]},
        },
        video_id: {
            "class_type": "CreateVideo",
            "inputs": {"images": [video_decode_id, 0], "fps": 24.0, "audio": [audio_decode_id, 0]},
        },
        save_id: {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [video_id, 0],
                "filename_prefix": "m7_13_h3",
                "format": "mp4",
                "codec": "auto",
            },
        },
        **(
            {
                finalizer_id: {
                    "class_type": "Sigmax.H4DispatchFinalize",
                    "inputs": {"video": [save_id, 0], "trace_file": trace_file},
                }
            }
            if trace_file is not None
            else {}
        ),
        **(
            {
                reference_image_id: {
                    "class_type": "LoadImage",
                    "inputs": {"image": host_reference_image_name},
                }
            }
            if host_reference_image_name is not None
            else {}
        ),
    }


def _json_unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("host JSON contains duplicate object names")
        result[key] = value
    return result


def _decode_json(payload: bytes, *, label: str) -> object:
    if not payload or len(payload) > _MAX_HTTP_BYTES:
        _fail(f"{label} size is outside the allowed range")
    try:
        return json.loads(payload, object_pairs_hook=_json_unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScheduleContractError(f"{label} is not valid JSON") from exc


def _loopback_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ScheduleContractError("host URL is malformed") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != _LOOPBACK
        or port is None
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        _fail("host URL must be credential-free loopback HTTP")
    return url


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: Mapping[str, object] | None = None,
    timeout: float = 10.0,
) -> object:
    payload = None
    headers = {"Accept": "application/json"}
    if body is not None:
        payload = json.dumps(
            dict(body), ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(_loopback_url(url), data=payload, headers=headers, method=method)  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return _decode_json(response.read(_MAX_HTTP_BYTES + 1), label="host response")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ScheduleContractError("loopback host request failed") from exc


def _http_no_content(url: str, *, method: str, timeout: float) -> None:
    """Send a bounded loopback request whose endpoint may legitimately return 204."""

    request = Request(  # noqa: S310
        _loopback_url(url),
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            response.read(_MAX_HTTP_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ScheduleContractError("loopback host request failed") from exc


def _api_unreachable(base_url: str) -> bool:
    """Return only a boolean readback result; never retain a private response/error."""

    try:
        _http_json(f"{base_url}/system_stats", timeout=1)
    except ScheduleContractError:
        return True
    return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_LOOPBACK, 0))
        return cast(int, sock.getsockname()[1])


def _stage_extension(run_path: Path) -> Path:
    target = run_path / "base" / "custom_nodes" / "ComfyUI-Sigmax"
    target.mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / "__init__.py", target / "__init__.py")
    shutil.copytree(
        REPOSITORY_ROOT / "comfyui_sigmax",
        target / "comfyui_sigmax",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(REPOSITORY_ROOT / "web", target / "web")
    observer_target = run_path / "base" / "custom_nodes" / "SigmaxH4Observer"
    observer_target.mkdir(parents=True)
    shutil.copy2(
        REPOSITORY_ROOT / "scripts" / "h4_dispatch_observer.py", observer_target / "__init__.py"
    )
    return target


def _host_command(
    *,
    host_python: Path,
    comfyui_root: Path,
    models_root: Path,
    run_path: Path,
    port: int,
    use_ck_attention: bool,
    enable_triton: bool,
) -> list[str]:
    command = [
        str(host_python),
        str(comfyui_root / "main.py"),
        "--listen",
        _LOOPBACK,
        "--port",
        str(port),
        "--base-directory",
        str(run_path / "base"),
        "--models-directory",
        str(models_root),
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
        "--cache-none",
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        "ComfyUI-Sigmax",
        "SigmaxH4Observer",
    ]
    if use_ck_attention:
        command.append("--use-ck-attention")
    if enable_triton:
        command.append("--enable-triton-backend")
    return command


def _readiness(
    *, base_url: str, process: subprocess.Popen[bytes], deadline: float
) -> dict[str, object]:
    last = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _fail(f"ComfyUI exited before H4 readiness ({process.returncode})")
        try:
            value = _http_json(f"{base_url}/object_info")
            if not isinstance(value, dict):
                _fail("H4 object_info is not an object")
            return cast(dict[str, object], value)
        except ScheduleContractError as exc:
            last = str(exc)
        time.sleep(0.25)
    _fail(f"H4 host readiness deadline expired: {last}")


def _verify_live_host_version(*, base_url: str, expected: str) -> str:
    if _SEMVER.fullmatch(expected) is None:
        _fail("H4 host version must be semantic X.Y.Z text")
    value = _http_json(f"{base_url}/system_stats")
    if not isinstance(value, dict):
        _fail("H4 system stats are malformed")
    system = value.get("system")
    if not isinstance(system, dict) or system.get("comfyui_version") != expected:
        _fail("running H4 host version does not match the exact expected version")
    return expected


def _wait_history(*, base_url: str, prompt_id: str, deadline: float) -> dict[str, object]:
    while time.monotonic() < deadline:
        value = _http_json(f"{base_url}/history/{prompt_id}")
        if isinstance(value, dict):
            entry = value.get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status")
                if isinstance(status, dict) and status.get("completed") is True:
                    return cast(dict[str, object], value)
        time.sleep(0.25)
    _fail("H4 prompt history deadline expired")


def _history_summary(history: Mapping[str, object], prompt_id: str) -> dict[str, object]:
    entry = history.get(prompt_id)
    if not isinstance(entry, Mapping):
        _fail("H4 history entry is missing")
    status = entry.get("status")
    if not isinstance(status, Mapping):
        _fail("H4 history status is missing")
    return {"completed": status.get("completed"), "status_str": status.get("status_str")}


def _submit(
    *, base_url: str, prompt: Mapping[str, object], timeout: float
) -> tuple[str, dict[str, object]]:
    value = _http_json(
        f"{base_url}/prompt",
        method="POST",
        body={"client_id": f"sigmax-m7-13-{uuid.uuid4().hex}", "prompt": dict(prompt)},
        timeout=30,
    )
    if not isinstance(value, dict) or not isinstance(value.get("prompt_id"), str):
        _fail("H4 prompt did not return a prompt ID")
    if value.get("node_errors") not in ({}, None):
        _fail("H4 prompt validation returned node errors")
    prompt_id = cast(str, value["prompt_id"])
    return prompt_id, _wait_history(
        base_url=base_url, prompt_id=prompt_id, deadline=time.monotonic() + timeout
    )


def _terminate(process: subprocess.Popen[bytes], *, base_url: str) -> dict[str, object]:
    interrupt_requested = False
    termination = "requested"
    termination_method = "already_exited"
    with suppress(ScheduleContractError):
        # ComfyUI's interrupt route is a no-content endpoint; JSON decoding a 204 falsely
        # records the cooperative request as unavailable and forces the hard-kill path below.
        _http_no_content(f"{base_url}/interrupt", method="POST", timeout=2)
        interrupt_requested = True
    if process.poll() is None:
        try:
            if os.name == "nt":
                sigint = getattr(signal, "SIGINT", None)
                if not isinstance(sigint, int):
                    _fail("Windows cooperative process signaling is unavailable")
                os.kill(process.pid, sigint)
            else:
                killpg = getattr(os, "killpg", None)
                sigint = getattr(signal, "SIGINT", None)
                if not callable(killpg) or not isinstance(sigint, int):
                    _fail("POSIX process-group signaling is unavailable")
                cast(Callable[[int, int], None], killpg)(process.pid, sigint)
            termination_method = "cooperative_sigint"
            process.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            termination = "forced"
            termination_method = "forced_terminate"
            if os.name == "nt":
                taskkill = shutil.which("taskkill")
                if taskkill is not None:
                    subprocess.run(  # noqa: S603
                        [taskkill, "/PID", str(process.pid), "/T", "/F"],
                        check=False,
                        capture_output=True,
                        timeout=15,
                    )
            else:
                killpg = getattr(os, "killpg", None)
                sigkill = getattr(signal, "SIGKILL", None)
                if callable(killpg) and isinstance(sigkill, int):
                    with suppress(OSError):
                        cast(Callable[[int, int], None], killpg)(process.pid, sigkill)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
    process_exited = process.poll() is not None
    if process_exited and termination == "requested":
        termination = "graceful"
    if not process_exited:
        termination = "failed"
    return {
        "interrupt_requested": interrupt_requested,
        "process_exited": process_exited,
        "return_code": process.returncode,
        "termination": termination,
        "termination_method": termination_method,
    }


def _port_release_receipt(port: int, *, timeout: float = 10.0) -> dict[str, object]:
    if port <= 0 or port > 65535 or timeout <= 0:
        return {
            "attempts": 0,
            "elapsed_ms": 0,
            "reason_code": "port_release_input_invalid",
            "status": "unavailable",
            "verified_by": "bind_probe",
        }
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((_LOOPBACK, port))
            except OSError:
                time.sleep(0.1)
                continue
            return {
                "attempts": attempts,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
                "reason_code": "port_released",
                "status": "pass",
                "verified_by": "bind_probe",
            }
    return {
        "attempts": attempts,
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "reason_code": "port_release_timeout",
        "status": "fail",
        "verified_by": "bind_probe",
    }


def _wait_for_port_release(port: int, *, timeout: float = 10.0) -> dict[str, object]:
    receipt = _port_release_receipt(port, timeout=timeout)
    if receipt["status"] != "pass":
        _fail("owned H4 loopback port was not released")
    return receipt


def _output_fingerprints(run_path: Path) -> tuple[str, ...]:
    results: list[str] = []
    output_root = run_path / "output"
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            results.append("sha256:" + _sha256_file(path))
    return tuple(results)


def _new_output_fingerprints(before: Sequence[str], after: Sequence[str]) -> tuple[str, ...]:
    """Return the multiset of files created by one queue, retaining duplicate hashes."""

    return tuple(sorted((Counter(after) - Counter(before)).elements()))


def _media_summary(run_path: Path) -> dict[str, object]:
    """Capture bounded video/audio stream facts without retaining private filenames."""

    ffprobe = shutil.which("ffprobe")
    files = sorted(
        item
        for item in (run_path / "output").rglob("*")
        if item.is_file() and item.suffix.casefold() == ".mp4"
    )
    if ffprobe is None:
        return {"status": "unavailable", "reason_code": "ffprobe_unavailable"}
    if not files:
        return {"status": "unavailable", "reason_code": "video_output_missing"}
    result = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,nb_frames,sample_rate,channels,avg_frame_rate",
            "-of",
            "json",
            str(files[-1]),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return {"status": "unavailable", "reason_code": "ffprobe_failed"}
    try:
        decoded = json.loads(result.stdout)
    except (TypeError, ValueError):
        return {"status": "unavailable", "reason_code": "ffprobe_invalid_json"}
    streams = decoded.get("streams") if isinstance(decoded, dict) else None
    if not isinstance(streams, list):
        return {"status": "unavailable", "reason_code": "ffprobe_streams_missing"}
    video = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
        None,
    )
    audio = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"),
        None,
    )
    if not isinstance(video, dict) or not isinstance(audio, dict):
        return {"status": "failed", "reason_code": "native_audio_or_video_stream_missing"}
    return {
        "audio_channels": audio.get("channels"),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": audio.get("sample_rate"),
        "status": "pass",
        "video_codec": video.get("codec_name"),
        "video_frames": video.get("nb_frames"),
        "video_rate": video.get("avg_frame_rate"),
    }


_GPU_MEMORY_INTERVAL_SECONDS: Final = 0.25
_GPU_MEMORY_UNAVAILABLE_REASONS: Final = frozenset(
    {"gpu_memory_tool_unavailable", "gpu_memory_timeout", "gpu_memory_no_samples"}
)


def _parse_gpu_memory_output(output: str) -> dict[int, int]:
    """Parse strict ``index,memory.used`` MiB rows into device-indexed bytes."""

    if not isinstance(output, str):
        raise ValueError("GPU memory output is not text")
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if not rows:
        raise ValueError("GPU memory output is empty")
    parsed: dict[int, int] = {}
    integer = re.compile(r"(?:0|[1-9][0-9]*)")
    for row in rows:
        fields = [item.strip() for item in row.split(",")]
        if len(fields) != 2 or any(integer.fullmatch(item) is None for item in fields):
            raise ValueError("GPU memory output is malformed")
        device, memory_mib = (int(item) for item in fields)
        if device in parsed or device > 4096 or memory_mib > (1 << 40):
            raise ValueError("GPU memory output is out of bounds")
        parsed[device] = memory_mib * 1024 * 1024
    return parsed


def _gpu_memory_observation() -> tuple[dict[int, int] | None, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, "gpu_memory_tool_unavailable"
    try:
        result = subprocess.run(  # noqa: S603
            [
                executable,
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return None, "gpu_memory_timeout"
    except OSError:
        return None, "gpu_memory_command_failed"
    if result.returncode != 0:
        return None, "gpu_memory_command_failed"
    try:
        return _parse_gpu_memory_output(result.stdout), "gpu_memory_observed"
    except ValueError:
        return None, "gpu_memory_malformed"


def _gpu_memory_snapshot() -> int | None:
    """Retain the historical max-device before/after snapshot semantics."""

    observation, _reason = _gpu_memory_observation()
    return max(observation.values(), default=0) if observation is not None else None


def _gpu_memory_projection(
    samples: Sequence[Mapping[int, int]],
    *,
    status: str = "pass",
    reason_code: str | None = None,
    sample_interval_ms: int = int(_GPU_MEMORY_INTERVAL_SECONDS * 1000),
) -> dict[str, object]:
    """Project bounded per-device samples without converting missing data to zero."""

    if status not in {"pass", "unavailable", "failed"}:
        raise ValueError("GPU memory status is invalid")
    if sample_interval_ms <= 0:
        raise ValueError("GPU memory interval is invalid")
    normalized: list[dict[int, int]] = []
    for sample in samples:
        item: dict[int, int] = {}
        for device, value in sample.items():
            if not isinstance(device, int) or device < 0 or device > 4096:
                raise ValueError("GPU memory device index is invalid")
            if not isinstance(value, int) or value < 0:
                raise ValueError("GPU memory value is invalid")
            item[device] = value
        if item:
            normalized.append(item)
    if status == "pass" and not normalized:
        status = "unavailable"
        reason_code = reason_code or "gpu_memory_no_samples"
    peak_by_device = {
        str(device): max(sample.get(device, 0) for sample in normalized)
        for device in sorted({device for sample in normalized for device in sample})
    }
    peak_used = max((sum(sample.values()) for sample in normalized), default=0)
    if status == "pass" and peak_used == 0:
        status = "failed"
        reason_code = reason_code or "gpu_memory_zero_sample"
    return {
        "aggregation_policy": "max_sum_per_sample",
        "peak_used_bytes": peak_used if normalized and status == "pass" else None,
        "peak_used_bytes_by_device": peak_by_device if normalized and status == "pass" else {},
        "reason_code": reason_code or "gpu_memory_observed",
        "sample_count": len(normalized),
        "sample_interval_ms": sample_interval_ms,
        "source": "nvidia-smi.memory.used",
        "status": status,
    }


class _GpuMemorySampler:
    """Bounded queue sampler; failures remain explicit in the private projection."""

    def __init__(
        self,
        *,
        snapshot: Callable[[], tuple[dict[int, int] | None, str]] = _gpu_memory_observation,
        interval_seconds: float = _GPU_MEMORY_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds <= 0 or interval_seconds > 5:
            raise ValueError("GPU memory sampler interval is invalid")
        self._snapshot = snapshot
        self._interval_seconds = interval_seconds
        self._samples: list[dict[int, int]] = []
        self._reasons: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _record(self) -> None:
        try:
            sample, reason = self._snapshot()
        except (OSError, ScheduleContractError, ValueError, subprocess.SubprocessError):
            sample, reason = None, "gpu_memory_sampler_failed"
        with self._lock:
            if sample:
                self._samples.append(dict(sample))
            else:
                self._reasons.append(reason)

    def _poll(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._record()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU memory sampler already started")
        self._record()
        self._thread = threading.Thread(
            target=self._poll, name="sigmax-h4-gpu-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, object]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self._interval_seconds * 4))
            self._thread = None
        self._record()
        with self._lock:
            samples = tuple(dict(item) for item in self._samples)
            reasons = tuple(self._reasons)
        if samples:
            return _gpu_memory_projection(
                samples,
                sample_interval_ms=int(self._interval_seconds * 1000),
            )
        reason = reasons[0] if reasons else "gpu_memory_no_samples"
        status = "unavailable" if reason in _GPU_MEMORY_UNAVAILABLE_REASONS else "failed"
        return _gpu_memory_projection(
            (),
            status=status,
            reason_code=reason,
            sample_interval_ms=int(self._interval_seconds * 1000),
        )


def _submit_measured(
    *, base_url: str, prompt: Mapping[str, object], timeout: float
) -> tuple[str, dict[str, object], int, dict[str, object]]:
    sampler = _GpuMemorySampler()
    started = time.perf_counter()
    sampler.start()
    try:
        prompt_id, history = _submit(base_url=base_url, prompt=prompt, timeout=timeout)
    finally:
        memory = sampler.stop()
    return prompt_id, history, int((time.perf_counter() - started) * 1_000_000), memory


def _submit_if_eligible(
    observation: ArtifactObservation,
    submit: Callable[[], tuple[str, dict[str, object], int, dict[str, object]]],
) -> tuple[str, dict[str, object], int, dict[str, object]] | None:
    """Keep the queue seam fail-closed for every blocked/rejected artifact row."""

    if observation.disposition is not RowDisposition.ACCEPTED:
        return None
    return submit()


def _temp_root_receipt(run_path: Path, *, owned_root: Path, remove: bool) -> dict[str, object]:
    """Remove only the exact harness-owned run directory when finalization permits it."""

    resolved_run = run_path.resolve()
    resolved_root = owned_root.resolve()
    owned = (
        resolved_run.parent == resolved_root
        and resolved_run.name.startswith("h4-")
        and REPOSITORY_ROOT.resolve() in resolved_root.parents
    )
    if not owned:
        return {
            "cleanup_status": "failed",
            "owned": False,
            "reason_code": "temp_root_ownership_failed",
            "remaining_entries": None,
        }
    if not resolved_run.exists():
        return {
            "cleanup_status": "removed",
            "owned": True,
            "reason_code": "temp_root_already_removed",
            "remaining_entries": 0,
        }
    if not remove:
        try:
            remaining = sum(1 for _ in resolved_run.rglob("*"))
        except OSError:
            remaining = None
        return {
            "cleanup_status": "retained",
            "owned": True,
            "reason_code": "temp_root_retained_after_failure",
            "remaining_entries": remaining,
        }
    try:
        shutil.rmtree(resolved_run)
    except OSError:
        return {
            "cleanup_status": "failed",
            "owned": True,
            "reason_code": "temp_root_cleanup_failed",
            "remaining_entries": None,
        }
    return {
        "cleanup_status": "removed",
        "owned": True,
        "reason_code": "temp_root_removed",
        "remaining_entries": 0,
    }


def _cleanup_projection(
    shutdown: Mapping[str, object],
    port_release: Mapping[str, object],
    temp_root: Mapping[str, object],
    host_readback: Mapping[str, object],
) -> dict[str, object]:
    termination = shutdown.get("termination")
    termination_method = shutdown.get("termination_method")
    process_exited = shutdown.get("process_exited") is True
    return_code = shutdown.get("return_code")
    termination_ok = termination == "graceful" and process_exited and return_code in (0, None)
    port_ok = port_release.get("status") == "pass"
    temp_ok = temp_root.get("cleanup_status") == "removed"
    host_ok = host_readback.get("status") == "pass"
    if termination_ok and port_ok and temp_ok and host_ok:
        status = "pass"
    elif any(item.get("status") == "unavailable" for item in (port_release, host_readback)):
        status = "unavailable"
    else:
        status = "fail"
    if status == "pass":
        reason_code = "cleanup_complete"
    elif termination_method == "cooperative_sigint" and return_code not in (0, None):
        reason_code = "nonzero_cooperative_return"
    elif termination == "forced" or termination_method == "forced_terminate":
        reason_code = "forced_termination"
    else:
        reason_code = "cleanup_incomplete"
    return {
        "host_readback": dict(host_readback),
        "host_mutation": dict(cast(Mapping[str, object], host_readback.get("host_mutation", {}))),
        "port_release": dict(port_release),
        "process_exited": process_exited,
        "reason_code": reason_code,
        "return_code": return_code,
        "status": status,
        "temp_root": dict(temp_root),
        "termination": termination if isinstance(termination, str) else "failed",
        "termination_method": (
            termination_method if isinstance(termination_method, str) else "failed"
        ),
    }


def _failure_reason_code(error: BaseException) -> str:
    if isinstance(error, ScheduleContractError):
        return "execution_contract_failed"
    if isinstance(error, (OSError, PermissionError)):
        return "execution_io_failed"
    if isinstance(error, subprocess.SubprocessError):
        return "execution_subprocess_failed"
    if isinstance(error, ValueError):
        return "execution_value_failed"
    return "execution_failed"


def _dispatch_unavailable(
    reason_code: str, *, requested_attention_backend: str
) -> dict[str, object]:
    return {
        "actual_attention_backend": "not_observed",
        "actual_operation_backend": "not_observed",
        "attention_calls": 0,
        "dispatch_trace": {},
        "observation_source": "not_observed",
        "operation_calls": 0,
        "reason_code": reason_code,
        "requested_attention_backend": requested_attention_backend,
        "requested_operation_backend": "auto",
        "status": "unavailable",
    }


def _contains_private_dispatch_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(_PRIVATE_PATH.search(value) or _SECRET.search(value))
    if isinstance(value, Mapping):
        return any(
            _contains_private_dispatch_value(key) or _contains_private_dispatch_value(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_dispatch_value(item) for item in value)
    return False


def _dispatch_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 1_000_000_000:
        return None
    return value


def _dispatch_backend_counts(
    value: object, *, allowed: frozenset[str], calls: int
) -> dict[str, int] | None:
    if not isinstance(value, Mapping) or len(value) > len(allowed):
        return None
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in allowed:
            return None
        count = _dispatch_count(item)
        if count is None or count == 0:
            return None
        result[key] = count
    if sum(result.values()) != calls:
        return None
    return result


def _dispatch_events(value: object, *, kind: str) -> list[dict[str, object]] | None:
    if not isinstance(value, list) or len(value) > 32:
        return None
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        expected_keys = (
            {"backend", "operation", "ordinal", "outcome"}
            if kind == "operation"
            else {"backend", "ordinal", "outcome"}
        )
        if set(item) != expected_keys:
            return None
        backend = item.get("backend")
        ordinal = item.get("ordinal")
        outcome = item.get("outcome")
        if not isinstance(backend, str) or backend not in (
            _DISPATCH_OPERATION_BACKENDS if kind == "operation" else _DISPATCH_ATTENTION_BACKENDS
        ):
            return None
        if _dispatch_count(ordinal) is None or ordinal == 0:
            return None
        if outcome not in {"raised", "returned"}:
            return None
        event: dict[str, object] = {
            "backend": backend,
            "ordinal": ordinal,
            "outcome": outcome,
        }
        if kind == "operation":
            operation = item.get("operation")
            if not isinstance(operation, str) or operation not in _DISPATCH_OPERATION_NAMES:
                return None
            event["operation"] = operation
        result.append(event)
    return result


def _read_dispatch_trace(path: Path, *, expected_attention_backend: str) -> dict[str, object]:
    """Read and validate only the adapter's bounded, path-free private projection."""

    if expected_attention_backend not in _DISPATCH_ATTENTION_BACKENDS:
        return _dispatch_unavailable(
            "dispatch_request_not_allowlisted",
            requested_attention_backend=expected_attention_backend,
        )
    try:
        payload = path.read_bytes()
    except (OSError, PermissionError):
        return _dispatch_unavailable(
            "dispatch_trace_missing", requested_attention_backend=expected_attention_backend
        )
    if not payload or len(payload) > _MAX_DISPATCH_TRACE_BYTES:
        return _dispatch_unavailable(
            "dispatch_trace_size_invalid", requested_attention_backend=expected_attention_backend
        )
    try:
        decoded = json.loads(payload, object_pairs_hook=_json_unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _dispatch_unavailable(
            "dispatch_trace_invalid_json", requested_attention_backend=expected_attention_backend
        )
    if not isinstance(decoded, Mapping) or _contains_private_dispatch_value(decoded):
        return _dispatch_unavailable(
            "dispatch_trace_redaction_failed",
            requested_attention_backend=expected_attention_backend,
        )
    if decoded.get("schema") != _DISPATCH_TRACE_SCHEMA:
        return _dispatch_unavailable(
            "dispatch_trace_schema_mismatch", requested_attention_backend=expected_attention_backend
        )
    if decoded.get("adapter_version") != _DISPATCH_ADAPTER_VERSION:
        return _dispatch_unavailable(
            "dispatch_trace_adapter_mismatch",
            requested_attention_backend=expected_attention_backend,
        )
    if decoded.get("status") != "DISARMED":
        return _dispatch_unavailable(
            "dispatch_trace_not_disarmed", requested_attention_backend=expected_attention_backend
        )
    requested_attention = decoded.get("requested_attention_backend")
    if requested_attention != expected_attention_backend:
        return _dispatch_unavailable(
            "dispatch_trace_request_mismatch",
            requested_attention_backend=expected_attention_backend,
        )
    if decoded.get("requested_operation_backend") != "auto":
        return _dispatch_unavailable(
            "dispatch_trace_operation_request_mismatch",
            requested_attention_backend=expected_attention_backend,
        )

    operation = decoded.get("operation")
    attention = decoded.get("attention")
    if not isinstance(operation, Mapping) or not isinstance(attention, Mapping):
        return _dispatch_unavailable(
            "dispatch_trace_axis_missing", requested_attention_backend=expected_attention_backend
        )
    operation_calls = _dispatch_count(operation.get("calls"))
    attention_calls = _dispatch_count(attention.get("calls"))
    if operation_calls is None or attention_calls is None:
        return _dispatch_unavailable(
            "dispatch_trace_call_count_invalid",
            requested_attention_backend=expected_attention_backend,
        )
    operation_counts = _dispatch_backend_counts(
        operation.get("backend_counts"),
        allowed=_DISPATCH_OPERATION_BACKENDS,
        calls=operation_calls,
    )
    attention_counts = _dispatch_backend_counts(
        attention.get("backend_counts"),
        allowed=_DISPATCH_ATTENTION_BACKENDS,
        calls=attention_calls,
    )
    if operation_counts is None and operation_calls != 0:
        return _dispatch_unavailable(
            "dispatch_trace_operation_counts_invalid",
            requested_attention_backend=expected_attention_backend,
        )
    if attention_counts is None and attention_calls != 0:
        return _dispatch_unavailable(
            "dispatch_trace_attention_counts_invalid",
            requested_attention_backend=expected_attention_backend,
        )
    operation_events = _dispatch_events(operation.get("events"), kind="operation")
    attention_events = _dispatch_events(attention.get("events"), kind="attention")
    if operation_events is None or attention_events is None:
        return _dispatch_unavailable(
            "dispatch_trace_events_invalid", requested_attention_backend=expected_attention_backend
        )
    expected_operation = (
        next(iter(operation_counts))
        if operation_counts is not None and len(operation_counts) == 1
        else "not_observed"
    )
    expected_attention = (
        next(iter(attention_counts))
        if attention_counts is not None and len(attention_counts) == 1
        else "not_observed"
    )
    actual_operation = decoded.get("actual_operation_backend")
    actual_attention = decoded.get("actual_attention_backend")
    if actual_operation not in _DISPATCH_OPERATION_BACKENDS | {"not_observed"}:
        return _dispatch_unavailable(
            "dispatch_trace_operation_backend_invalid",
            requested_attention_backend=expected_attention_backend,
        )
    if actual_attention not in _DISPATCH_ATTENTION_BACKENDS | {"not_observed"}:
        return _dispatch_unavailable(
            "dispatch_trace_attention_backend_invalid",
            requested_attention_backend=expected_attention_backend,
        )
    if actual_operation != expected_operation or actual_attention != expected_attention:
        return _dispatch_unavailable(
            "dispatch_trace_backend_mismatch",
            requested_attention_backend=expected_attention_backend,
        )
    expected_source = (
        "authorized_host_dispatch"
        if actual_operation != "not_observed" or actual_attention != "not_observed"
        else "not_observed"
    )
    if decoded.get("observation_source") != expected_source:
        return _dispatch_unavailable(
            "dispatch_trace_source_invalid", requested_attention_backend=expected_attention_backend
        )
    reason_codes = decoded.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or len(reason_codes) > 16
        or any(not isinstance(item, str) for item in reason_codes)
    ):
        return _dispatch_unavailable(
            "dispatch_trace_reasons_invalid", requested_attention_backend=expected_attention_backend
        )
    return {
        "actual_attention_backend": actual_attention,
        "actual_operation_backend": actual_operation,
        "attention_calls": attention_calls,
        "dispatch_trace": {
            "adapter_version": _DISPATCH_ADAPTER_VERSION,
            "attention_backend_counts": dict(attention_counts or {}),
            "attention_events": attention_events,
            "operation_backend_counts": dict(operation_counts or {}),
            "operation_events": operation_events,
            "reason_codes": list(reason_codes),
            "schema": _DISPATCH_TRACE_SCHEMA,
            "status": "DISARMED",
        },
        "observation_source": decoded.get("observation_source"),
        "operation_calls": operation_calls,
        "reason_code": "dispatch_trace_valid",
        "requested_attention_backend": requested_attention,
        "requested_operation_backend": "auto",
        "status": "valid",
    }


def _host_revision(root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        _fail("git executable is unavailable")
    result = subprocess.run(  # noqa: S603
        [git, "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    revision = result.stdout.strip()
    if result.returncode != 0 or _REVISION.fullmatch(revision) is None:
        _fail("selected host is not an exact Git checkout")
    return revision


def _host_git_value(root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        _fail("git executable is unavailable")
    result = subprocess.run(  # noqa: S603
        [git, "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        _fail("selected host Git readback failed")
    return result.stdout.strip()


def _host_readback_snapshot(
    root: Path, selected_artifacts: Sequence[tuple[str, Path]]
) -> dict[str, object]:
    """Capture only exact checkout identities and explicit artifact digests."""

    revision = _host_git_value(root, "rev-parse", "HEAD")
    tree = _host_git_value(root, "rev-parse", "HEAD^{tree}")
    worktree = _host_git_value(root, "status", "--porcelain=v1", "--untracked-files=no")
    if _REVISION.fullmatch(revision) is None or _REVISION.fullmatch(tree) is None:
        _fail("selected host Git readback identity is malformed")
    worktree_digest = "sha256:" + hashlib.sha256(worktree.encode("utf-8")).hexdigest()
    artifacts: dict[str, str] = {}
    for artifact_id, path in selected_artifacts:
        if not path.is_file():
            artifacts[artifact_id] = "missing"
            continue
        artifacts[artifact_id] = "sha256:" + _sha256_file(path)
    return {
        "artifacts": artifacts,
        "revision": revision,
        "tree": tree,
        "worktree_state": worktree_digest,
    }


def _host_readback_projection(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    process_alive: bool,
    api_unreachable: bool,
) -> dict[str, object]:
    """Compare private readback snapshots while emitting no checkout paths or status text."""

    required = ("revision", "tree", "worktree_state", "artifacts")
    complete = all(key in before and key in after for key in required)
    revision_unchanged: bool | None
    tree_unchanged: bool | None
    worktree_unchanged: bool | None
    artifacts_unchanged: bool | None
    if complete:
        revision_unchanged = before["revision"] == after["revision"]
        tree_unchanged = before["tree"] == after["tree"]
        worktree_unchanged = before["worktree_state"] == after["worktree_state"]
        artifacts_unchanged = before["artifacts"] == after["artifacts"]
    else:
        revision_unchanged = tree_unchanged = worktree_unchanged = artifacts_unchanged = None
    checkout_unchanged = (
        bool(revision_unchanged and tree_unchanged and worktree_unchanged) if complete else None
    )
    if not complete:
        status = "unavailable"
        reason_code = "host_readback_unavailable"
    elif process_alive or not api_unreachable:
        status = "fail"
        reason_code = "host_process_or_api_still_alive"
    elif not checkout_unchanged or not artifacts_unchanged:
        status = "fail"
        reason_code = "host_mutation_detected"
    else:
        status = "pass"
        reason_code = "host_readback_unchanged"
    mutation_status = status
    return {
        "api_unreachable": api_unreachable,
        "host_mutation": {
            "checkout_unchanged": checkout_unchanged,
            "selected_artifacts_unchanged": artifacts_unchanged,
            "status": mutation_status,
        },
        "process_alive": process_alive,
        "reason_code": reason_code,
        "revision_after": after.get("revision"),
        "revision_before": before.get("revision"),
        "status": status,
        "tree_after": after.get("tree"),
        "tree_before": before.get("tree"),
        "worktree_state_unchanged": worktree_unchanged,
    }


def _private_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if (
        label in {"protocol", "authorization", "evidence"}
        and REPOSITORY_ROOT.resolve() not in resolved.parents
    ):
        _fail(f"{label} file must stay inside the repository")
    return resolved


def _authorization(path: Path) -> None:
    target = _private_path(path, label="authorization")
    if not target.is_file():
        _fail("H4 authorization record is missing")
    text = target.read_text(encoding="utf-8")
    required = (AUTHORIZATION_MARKER, "GPU", "LoRA", "non-redistribution")
    if any(item.casefold() not in text.casefold() for item in required):
        _fail("H4 authorization record does not cover the required private scope")


def _redacted_diagnostic(value: object, *, sensitive: Sequence[Path] = ()) -> str:
    rendered = str(value)
    for path in sorted((str(item.resolve()) for item in sensitive), key=len, reverse=True):
        rendered = rendered.replace(path, "<redacted-path>").replace(
            path.replace("\\", "/"), "<redacted-path>"
        )
    rendered = _SECRET.sub("<redacted-secret>", rendered)
    return rendered[-_MAX_LOG_BYTES:]


def _row_artifact(
    *,
    row: str,
    models_root: Path,
    turbo_artifact: str | None,
    turbo_artifact_id: str | None,
    turbo_source: str,
    license_ack: bool,
    turbo_rows: Sequence[str] = (),
) -> ArtifactObservation | None:
    if row.startswith("B-"):
        model_name = (
            "H3/minimax_h3_fl2va_bf16.safetensors"
            if row == "B-BF16"
            else "H3/minimax_h3_fl2va_int8_convrot.safetensors"
        )
        path = models_root / "diffusion_models" / Path(model_name)
        expected = (
            "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182"  # pragma: allowlist secret
            if row == "B-BF16"
            else "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5"  # pragma: allowlist secret
        )  # pragma: allowlist secret
        return _artifact_observation(
            path=path,
            artifact_id=row.casefold(),
            disposition=RowDisposition.ACCEPTED,
            expected_sha256=expected,
            reason_code=None,
        )
    if len(tuple(turbo_rows)) > 1:
        return ArtifactObservation(
            artifact_id=row.casefold(),
            disposition=RowDisposition.REJECTED,
            reason_code="artifact.multi_row_ambiguity",
            file_bytes=None,
            sha256=None,
            header_bytes=None,
            tensor_count=None,
            dtype_counts=(),
        )
    if turbo_artifact is None or turbo_artifact_id is None:
        return ArtifactObservation(
            artifact_id=row.casefold(),
            disposition=RowDisposition.BLOCKED,
            reason_code="artifact.publisher_full_not_supplied",
            file_bytes=None,
            sha256=None,
            header_bytes=None,
            tensor_count=None,
            dtype_counts=(),
        )
    spec = _turbo_row_spec(row)
    if turbo_artifact_id != spec.recipe_id:
        return ArtifactObservation(
            artifact_id=row.casefold(),
            disposition=RowDisposition.REJECTED,
            reason_code="artifact.recipe_row_mismatch",
            file_bytes=None,
            sha256=None,
            header_bytes=None,
            tensor_count=None,
            dtype_counts=(),
        )
    path = models_root / "loras" / Path(_safe_relative_name(turbo_artifact, field="turbo_artifact"))
    return classify_turbo_artifact(
        path=path, artifact_id=turbo_artifact_id, source=turbo_source, license_ack=license_ack
    )


_VALID_H4_ROWS: Final = frozenset({"B-BF16", "B-INT8", "T4-768", "T8-544", "T4-544", "R4-544"})


def _artifact_observations(
    args: argparse.Namespace, models_root: Path
) -> dict[str, ArtifactObservation]:
    rows = tuple(args.rows)
    turbo_rows = tuple(row for row in rows if row.startswith(("T", "R")))
    observations: dict[str, ArtifactObservation] = {}
    for row in rows:
        if row not in _VALID_H4_ROWS:
            _fail("H4 row is not in the frozen protocol matrix")
        observation = _row_artifact(
            row=row,
            models_root=models_root,
            turbo_artifact=args.turbo_artifact,
            turbo_artifact_id=args.turbo_artifact_id,
            turbo_source=args.turbo_source,
            license_ack=args.license_ack,
            turbo_rows=turbo_rows,
        )
        if observation is None:
            _fail("H4 row has no artifact observation")
        observations[row] = observation
    return observations


def _queueable_rows(
    args: argparse.Namespace,
    models_root: Path,
    *,
    observations: Mapping[str, ArtifactObservation] | None = None,
) -> tuple[str, ...]:
    """Return only rows that may reach the first queue submission seam."""

    observed = observations or _artifact_observations(args, models_root)
    return tuple(
        row
        for row in args.rows
        if observed[row].disposition is RowDisposition.ACCEPTED
        and not (row == "R4-544" and _reference_image_preflight_reason(args) is not None)
    )


def _preflight_rows(args: argparse.Namespace, models_root: Path) -> dict[str, object]:
    observations = _artifact_observations(args, models_root)
    projections: dict[str, object] = {}
    for row, observation in observations.items():
        projection = observation.projection()
        if row == "R4-544" and observation.disposition is RowDisposition.ACCEPTED:
            reference_reason = _reference_image_preflight_reason(args)
            if reference_reason is not None:
                projection["disposition"] = RowDisposition.BLOCKED.value
                projection["reason_code"] = reference_reason
        projections[row] = projection
    return projections


def _preflight_components(args: argparse.Namespace, models_root: Path) -> dict[str, object]:
    observations = (
        _component_observation(
            models_root=models_root,
            folder="clip",
            name=args.text_encoder,
            artifact_id="text_encoder",
            expected_sha256=args.text_encoder_sha256,
        ),
        _component_observation(
            models_root=models_root,
            folder="vae",
            name=args.video_vae,
            artifact_id="video_vae",
            expected_sha256=args.video_vae_sha256,
        ),
        _component_observation(
            models_root=models_root,
            folder="vae",
            name=args.audio_vae,
            artifact_id="audio_vae",
            expected_sha256=args.audio_vae_sha256,
        ),
    )
    return {item.artifact_id: item.projection() for item in observations}


def _selected_artifact_paths(
    args: argparse.Namespace, models_root: Path
) -> tuple[tuple[str, Path], ...]:
    """Return only explicit model/component paths for readback; never scan model directories."""

    rows: list[tuple[str, Path]] = []
    for row in args.rows:
        if row == "B-BF16":
            name = args.diffusion_model
        elif row == "B-INT8":
            name = args.int8_diffusion_model
        else:
            continue
        rows.append(
            (
                row.casefold(),
                models_root
                / "diffusion_models"
                / Path(_safe_relative_name(name, field="diffusion_model")),
            )
        )
    if args.turbo_artifact is not None:
        turbo_path = (
            models_root
            / "loras"
            / Path(_safe_relative_name(args.turbo_artifact, field="turbo_artifact"))
        )
        rows.append(("turbo_artifact", turbo_path))
    rows.extend(
        (
            (
                "text_encoder",
                models_root
                / "clip"
                / Path(_safe_relative_name(args.text_encoder, field="text_encoder")),
            ),
            (
                "video_vae",
                models_root / "vae" / Path(_safe_relative_name(args.video_vae, field="video_vae")),
            ),
            (
                "audio_vae",
                models_root / "vae" / Path(_safe_relative_name(args.audio_vae, field="audio_vae")),
            ),
        )
    )
    return tuple(rows)


def _path_is_reparse_point(path: Path) -> bool:
    """Reject links/junctions before resolving caller-owned input paths."""

    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if callable(is_junction) and bool(is_junction()):
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except (OSError, PermissionError):
        return True
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _reference_image_source(source: Path, owner_root: Path | None) -> Path:
    """Validate a caller-owned image without exposing or following reparse paths."""

    if owner_root is None:
        raise ScheduleContractError("H4 reference image owner root is required")
    if not os.path.lexists(source):
        raise ScheduleContractError("H4 reference image is unavailable")
    if not os.path.lexists(owner_root):
        raise ScheduleContractError("H4 reference image owner root is invalid")
    if _path_is_reparse_point(source) or _path_is_reparse_point(owner_root):
        raise ScheduleContractError("H4 reference image links are not accepted")
    try:
        owner_path = owner_root.resolve(strict=True)
        source_path = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ScheduleContractError("H4 reference image is unavailable") from exc
    if not owner_path.is_dir() or _path_is_reparse_point(owner_path):
        raise ScheduleContractError("H4 reference image owner root is invalid")
    try:
        source_path.relative_to(owner_path)
    except ValueError as exc:
        raise ScheduleContractError("H4 reference image is outside the owner root") from exc
    # CRITICAL: reject a symlink/junction anywhere in the caller-owned path before copying.
    cursor = source
    while True:
        if _path_is_reparse_point(cursor):
            raise ScheduleContractError("H4 reference image links are not accepted")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    suffix = source_path.suffix.casefold()
    if not source_path.is_file() or suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ScheduleContractError("H4 reference image must be a supported image file")
    try:
        size = source_path.stat().st_size
    except (OSError, PermissionError) as exc:
        raise ScheduleContractError("H4 reference image is unreadable") from exc
    if size <= 0 or size > _MAX_REFERENCE_IMAGE_BYTES:
        raise ScheduleContractError("H4 reference image exceeds the size bound")
    return source_path


def _reference_image_preflight_reason(args: argparse.Namespace) -> str | None:
    source = getattr(args, "reference_image", None)
    owner_root = getattr(args, "reference_image_root", None)
    if source is None:
        return "input.reference_image_not_supplied"
    try:
        _reference_image_source(Path(source), None if owner_root is None else Path(owner_root))
    except ScheduleContractError as exc:
        if "owner root is required" in str(exc):
            return "input.reference_image_root_not_supplied"
        if "outside" in str(exc):
            return "input.reference_image_outside_owner_root"
        if "links" in str(exc):
            return "input.reference_image_reparse_point"
        if "size bound" in str(exc):
            return "input.reference_image_size_bound"
        if "supported image" in str(exc):
            return "input.reference_image_format_unsupported"
        return "input.reference_image_unavailable"
    return None


def _stage_reference_image(source: Path, run_path: Path, *, owner_root: Path | None = None) -> str:
    """Copy one caller-owned reference image into the exact private host input root."""

    source_path = _reference_image_source(source, owner_root)
    target = run_path / "input" / f"m7_13_reference{source_path.suffix.casefold()}"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
    except (OSError, PermissionError) as exc:
        raise ScheduleContractError("H4 reference image could not be staged") from exc
    if target.stat().st_size != source_path.stat().st_size:
        raise ScheduleContractError("H4 reference image staging readback failed")
    return target.name


def _run_rows(
    args: argparse.Namespace,
    models_root: Path,
    run_path: Path,
    *,
    host_before: Mapping[str, object] | None = None,
    selected_artifacts: Sequence[tuple[str, Path]] = (),
) -> dict[str, object]:
    if not args.prompt:
        _fail("private H4 prompt text is required through --prompt")
    clip_name = _safe_relative_name(args.text_encoder, field="text_encoder")
    video_vae = _safe_relative_name(args.video_vae, field="video_vae")
    audio_vae = _safe_relative_name(args.audio_vae, field="audio_vae")
    host_root = Path(args.comfyui_root).resolve()
    if host_before is None:
        host_before = _host_readback_snapshot(
            host_root, selected_artifacts or _selected_artifact_paths(args, models_root)
        )
    selected = tuple(selected_artifacts or _selected_artifact_paths(args, models_root))
    row_observations = _artifact_observations(args, models_root)
    queueable_rows = frozenset(_queueable_rows(args, models_root, observations=row_observations))
    host_revision = host_before.get("revision")
    port = _free_port()
    base_url = f"http://{_LOOPBACK}:{port}"
    log_path = run_path / "comfyui.log"
    process: subprocess.Popen[bytes] | None = None
    results: dict[str, object] = {}
    run_failed = False
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(  # noqa: S603
                _host_command(
                    host_python=Path(args.host_python).resolve(),
                    comfyui_root=host_root,
                    models_root=models_root,
                    run_path=run_path,
                    port=port,
                    use_ck_attention=args.use_ck_attention,
                    enable_triton=args.enable_triton,
                ),
                cwd=run_path,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=cast(int, getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
                if os.name == "nt"
                else 0,
                start_new_session=os.name == "posix",
            )
            try:
                _readiness(
                    base_url=base_url,
                    process=process,
                    deadline=time.monotonic() + args.readiness_timeout,
                )
                live_version = _verify_live_host_version(
                    base_url=base_url, expected=args.host_version
                )
                for row in args.rows:
                    observation = row_observations[row]
                    if row not in queueable_rows:
                        disposition = observation.disposition.value
                        reason_code = observation.reason_code
                        if row == "R4-544" and observation.disposition is RowDisposition.ACCEPTED:
                            disposition = RowDisposition.BLOCKED.value
                            reason_code = _reference_image_preflight_reason(args)
                        results[row] = {
                            "artifact": observation.projection(),
                            "disposition": disposition,
                            "reason_code": reason_code,
                            "status": "not_executed",
                        }
                        continue
                    turbo_spec = None if row.startswith("B-") else _turbo_row_spec(row)
                    if turbo_spec is None:
                        variant = "H3 Base FL2VA"
                        steps = 20
                        width = args.width
                        height = args.height
                        shift_video = 12.0
                        shift_audio = 3.0
                        recipe_id = None
                        lora_name = None
                        reference_image_name = None
                        schedule_receipt: dict[str, object] | None = None
                    else:
                        variant = turbo_spec.variant
                        steps = turbo_spec.steps
                        width = turbo_spec.width
                        height = turbo_spec.height
                        shift_video = turbo_spec.shift_video
                        shift_audio = turbo_spec.shift_audio
                        recipe_id = turbo_spec.recipe_id
                        lora_name = _safe_relative_name(args.turbo_artifact, field="turbo_artifact")
                        reference_image_name = None
                        reference_receipt: dict[str, object] | None = None
                        if turbo_spec.requires_reference_image:
                            reference_reason = _reference_image_preflight_reason(args)
                            if reference_reason is not None:
                                results[row] = {
                                    "artifact": observation.projection(),
                                    "disposition": RowDisposition.BLOCKED.value,
                                    "status": "not_executed",
                                    "reason_code": reference_reason,
                                }
                                continue
                            reference_image_name = _stage_reference_image(
                                Path(args.reference_image),
                                run_path,
                                owner_root=Path(args.reference_image_root),
                            )
                            reference_receipt = {
                                "image_name": reference_image_name,
                                "sha256": (
                                    "sha256:"
                                    + _sha256_file(run_path / "input" / reference_image_name)
                                ),
                                "staged": True,
                            }
                        schedule = build_minimax_h3_turbo_schedule(
                            turbo_spec.recipe_id,
                            nfe=turbo_spec.steps,
                            precision="float64",
                            task=("ref2va" if turbo_spec.variant == "H3 Base Ref2VA" else "fl2va"),
                            video_shift=turbo_spec.shift_video,
                            audio_shift=turbo_spec.shift_audio,
                            loader_strength=1.0,
                        )
                        schedule_receipt = {
                            "audio_shift": turbo_spec.shift_audio,
                            "fingerprint": schedule.fingerprint,
                            "nfe": turbo_spec.steps,
                            "recipe_id": turbo_spec.recipe_id,
                            "video_shift": turbo_spec.shift_video,
                        }
                    model_name = _safe_relative_name(
                        args.int8_diffusion_model if row == "B-INT8" else args.diffusion_model,
                        field="diffusion_model",
                    )
                    if re.fullmatch(r"[A-Za-z0-9-]+", row) is None:
                        _fail("H4 row identifier is not safe for a private trace filename")
                    trace_path = run_path / f"{row.casefold()}_dispatch_trace.json"
                    requested_attention_backend = "ck_int8" if args.use_ck_attention else "pytorch"
                    before = _gpu_memory_snapshot()
                    prompt = build_h4_prompt(
                        variant=variant,
                        model_name=model_name,
                        clip_name=clip_name,
                        video_vae_name=video_vae,
                        audio_vae_name=audio_vae,
                        prompt=args.prompt,
                        width=width,
                        height=height,
                        length=args.length,
                        steps=steps,
                        seed=args.seed,
                        shift_video=shift_video,
                        shift_audio=shift_audio,
                        lora_name=lora_name,
                        trace_file=str(trace_path),
                        requested_attention_backend=requested_attention_backend,
                        requested_operation_backend="auto",
                        recipe_id=recipe_id,
                        reference_image_name=reference_image_name,
                    )

                    def submit_current_prompt(
                        current_prompt: Mapping[str, object] = prompt,
                    ) -> tuple[str, dict[str, object], int, dict[str, object]]:
                        return _submit_measured(
                            base_url=base_url,
                            prompt=current_prompt,
                            timeout=args.execution_timeout,
                        )

                    warmup_result = _submit_if_eligible(observation, submit_current_prompt)
                    if warmup_result is None:
                        _fail("H4 queue gate rejected an accepted row")
                    warmup_prompt_id, _warmup_history, warmup_latency, warmup_memory = warmup_result
                    warmup_outputs = _output_fingerprints(run_path)
                    first_result = _submit_if_eligible(observation, submit_current_prompt)
                    if first_result is None:
                        _fail("H4 queue gate rejected an accepted row")
                    first_prompt_id, first_history, first_latency, first_memory = first_result
                    first_all_outputs = _output_fingerprints(run_path)
                    first_outputs = _new_output_fingerprints(warmup_outputs, first_all_outputs)
                    first_media = _media_summary(run_path)
                    repeat_result = _submit_if_eligible(observation, submit_current_prompt)
                    if repeat_result is None:
                        _fail("H4 queue gate rejected an accepted row")
                    second_prompt_id, second_history, repeat_latency, repeat_memory = repeat_result
                    second_all_outputs = _output_fingerprints(run_path)
                    repeat_outputs = _new_output_fingerprints(first_all_outputs, second_all_outputs)
                    repeat_media = _media_summary(run_path)
                    after = _gpu_memory_snapshot()
                    dispatch = _read_dispatch_trace(
                        trace_path, expected_attention_backend=requested_attention_backend
                    )
                    results[row] = {
                        "artifact": observation.projection(),
                        "backend": {
                            "actual_attention_backend": dispatch["actual_attention_backend"],
                            "actual_operation_backend": dispatch["actual_operation_backend"],
                            "dispatch_trace": dispatch["dispatch_trace"],
                            "launch_flags_are_not_proof": True,
                            "observation_source": dispatch["observation_source"],
                            "requested_attention_backend": dispatch["requested_attention_backend"],
                            "requested_operation_backend": dispatch["requested_operation_backend"],
                        },
                        "disposition": RowDisposition.NO_PROMOTION.value,
                        **(
                            {
                                "recipe": {
                                    "audio_shift": turbo_spec.shift_audio,
                                    "id": turbo_spec.recipe_id,
                                    "nfe": turbo_spec.steps,
                                    "task": (
                                        "ref2va"
                                        if turbo_spec.variant == "H3 Base Ref2VA"
                                        else "fl2va"
                                    ),
                                    "variant": turbo_spec.variant,
                                    "video_shift": turbo_spec.shift_video,
                                    "width": turbo_spec.width,
                                    "height": turbo_spec.height,
                                },
                                "schedule": schedule_receipt,
                                "reference": reference_receipt,
                            }
                            if turbo_spec is not None
                            else {}
                        ),
                        "first_history_status": _history_summary(first_history, first_prompt_id),
                        "first_latency_us": first_latency,
                        "first_media": first_media,
                        "first_output_fingerprints": list(first_outputs),
                        "gpu_memory_after": after,
                        "gpu_memory_before": before,
                        "host_version": live_version,
                        "queues": {
                            "first": {
                                "history": _history_summary(first_history, first_prompt_id),
                                "latency_us": first_latency,
                                "memory": first_memory,
                                "outputs": list(first_outputs),
                                "prompt_id": first_prompt_id,
                            },
                            "repeat": {
                                "history": _history_summary(second_history, second_prompt_id),
                                "latency_us": repeat_latency,
                                "memory": repeat_memory,
                                "outputs": list(repeat_outputs),
                                "prompt_id": second_prompt_id,
                            },
                            "warmup": {
                                "latency_us": warmup_latency,
                                "memory": warmup_memory,
                                "prompt_id": warmup_prompt_id,
                            },
                        },
                        "reason_code": dispatch["reason_code"],
                        "repeat_history_status": _history_summary(second_history, second_prompt_id),
                        "repeat_latency_us": repeat_latency,
                        "repeat_media": repeat_media,
                        "repeat_output_fingerprints": list(repeat_outputs),
                        "repeat_stable": first_outputs == repeat_outputs,
                        "status": "succeeded",
                        "warmup_prompt_id": warmup_prompt_id,
                        "first_prompt_id": first_prompt_id,
                        "repeat_prompt_id": second_prompt_id,
                    }
            except (OSError, ScheduleContractError, ValueError, subprocess.SubprocessError) as exc:
                run_failed = True
                results["failure_reason_code"] = _failure_reason_code(exc)
        if process is None:
            run_failed = True
    except (OSError, ScheduleContractError, ValueError, subprocess.SubprocessError) as exc:
        run_failed = True
        results["failure_reason_code"] = _failure_reason_code(exc)
    finally:
        if process is None:
            shutdown: dict[str, object] = {
                "interrupt_requested": False,
                "process_exited": True,
                "return_code": None,
                "termination": "failed",
                "termination_method": "not_started",
            }
        else:
            try:
                shutdown = _terminate(process, base_url=base_url)
            except (OSError, ScheduleContractError, ValueError, subprocess.SubprocessError) as exc:
                shutdown = {
                    "interrupt_requested": False,
                    "process_exited": process.poll() is not None,
                    "return_code": process.returncode,
                    "termination": "failed",
                    "termination_method": "failed",
                }
                run_failed = True
                results["failure_reason_code"] = _failure_reason_code(exc)
        port_release = _port_release_receipt(port)
        try:
            host_after = _host_readback_snapshot(host_root, selected)
        except (OSError, ScheduleContractError, ValueError, subprocess.SubprocessError) as exc:
            host_after = {}
            results.setdefault("failure_reason_code", _failure_reason_code(exc))
        api_unreachable = _api_unreachable(base_url)
        host_readback = _host_readback_projection(
            host_before,
            host_after,
            process_alive=process is not None and process.poll() is None,
            api_unreachable=api_unreachable,
        )
        safe_remove = (
            not run_failed
            and shutdown.get("termination") == "graceful"
            and port_release.get("status") == "pass"
            and host_readback.get("status") == "pass"
        )
        temp_root = _temp_root_receipt(run_path, owned_root=run_path.parent, remove=safe_remove)
        results["cleanup"] = _cleanup_projection(shutdown, port_release, temp_root, host_readback)
        results["shutdown"] = shutdown
        results["port_release"] = port_release
    results["host_revision"] = host_revision
    results["status"] = "failed" if run_failed else "succeeded"
    return results


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.allow_gpu_execution:
        _fail("H4 execution requires explicit --allow-gpu-execution")
    if not args.license_ack:
        _fail("H4 execution requires --license-ack for caller-owned weights/artifacts")
    commit, tree = current_candidate()
    protocol = _private_path(Path(args.protocol_file), label="protocol")
    _protocol_binding(protocol, expected_commit=commit, expected_tree=tree)
    _authorization(Path(args.authorization_file))
    host_root = Path(args.comfyui_root).resolve()
    host_python = Path(args.host_python).resolve()
    models_root = Path(args.models_directory).resolve()
    if (
        not (host_root / "main.py").is_file()
        or not host_python.is_file()
        or not models_root.is_dir()
    ):
        _fail("H4 host/python/models inputs are not valid explicit paths")
    expected_host_revision = args.expected_host_revision
    actual_host_revision = _host_revision(host_root)
    if expected_host_revision != actual_host_revision:
        _fail("selected H4 host revision does not match the exact expected revision")
    preflight = _preflight_rows(args, models_root)
    components = _preflight_components(args, models_root)
    selected_artifacts = _selected_artifact_paths(args, models_root)
    host_before = _host_readback_snapshot(host_root, selected_artifacts)
    evidence: dict[str, object] = {
        "schema": H4_SCHEMA,
        "candidate": {"commit": commit, "tree": tree},
        "host": {
            "revision": actual_host_revision,
            "tree": host_before.get("tree"),
            "version": args.host_version,
            "worktree_state": host_before.get("worktree_state"),
        },
        "components": components,
        "rows": preflight,
        "authorization": "private_non_redistribution",
        "gpu_execution_requested": True,
    }
    if args.preflight_only:
        evidence["status"] = "preflight_complete"
        return evidence
    owned_root = _private_path(Path(args.temp_root), label="run")
    if REPOSITORY_ROOT.resolve() not in owned_root.parents:
        _fail("H4 temp root must be a repository-local private root")
    run_path = owned_root / f"h4-{uuid.uuid4().hex}"
    run_path.mkdir(parents=True)
    for name in ("base", "input", "output", "temp", "user"):
        (run_path / name).mkdir()
    _stage_extension(run_path)
    evidence["execution"] = _run_rows(
        args,
        models_root,
        run_path,
        host_before=host_before,
        selected_artifacts=selected_artifacts,
    )
    evidence["status"] = (
        "execution_complete"
        if cast(Mapping[str, object], evidence["execution"]).get("status") == "succeeded"
        else "execution_failed"
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", type=Path, default=_DEFAULT_PROTOCOL)
    parser.add_argument("--authorization-file", type=Path, default=_DEFAULT_AUTHORIZATION)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--host-python", type=Path, required=True)
    parser.add_argument("--models-directory", type=Path, required=True)
    parser.add_argument("--expected-host-revision", required=True)
    parser.add_argument("--host-version", default="0.32.0")
    parser.add_argument("--temp-root", type=Path, default=REPOSITORY_ROOT / ".tmp" / "h4")
    parser.add_argument("--evidence-file", type=Path)
    parser.add_argument(
        "--rows", nargs="+", default=["B-BF16", "B-INT8", "T4-768", "T8-544", "T4-544", "R4-544"]
    )
    parser.add_argument("--allow-gpu-execution", action="store_true")
    parser.add_argument("--license-ack", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--use-ck-attention", action="store_true")
    parser.add_argument("--enable-triton", action="store_true")
    parser.add_argument("--turbo-artifact")
    parser.add_argument("--turbo-artifact-id")
    parser.add_argument(
        "--turbo-source",
        choices=["publisher-full", "kijai-reduced", "local-modified"],
        default="publisher-full",
    )
    parser.add_argument("--diffusion-model", default="H3/minimax_h3_fl2va_bf16.safetensors")
    parser.add_argument(
        "--int8-diffusion-model", default="H3/minimax_h3_fl2va_int8_convrot.safetensors"
    )
    parser.add_argument("--text-encoder", default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    parser.add_argument("--text-encoder-sha256")
    parser.add_argument("--video-vae", default="minimax_h3_video_vae_fp16.safetensors")
    parser.add_argument("--video-vae-sha256")
    parser.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
    parser.add_argument("--audio-vae-sha256")
    parser.add_argument("--prompt")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--reference-image-root", type=Path)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--length", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1844674407370955161)
    parser.add_argument("--readiness-timeout", type=float, default=180.0)
    parser.add_argument("--execution-timeout", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        evidence = run(args)
        _validate_h4_schema(evidence.get("schema"))
        if args.evidence_file is not None:
            target = _private_path(args.evidence_file, label="evidence")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ScheduleContractError, OSError, ValueError) as exc:
        print(f"H4_VALIDATION_ERROR: {_redacted_diagnostic(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
