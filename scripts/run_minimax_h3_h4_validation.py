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
import subprocess
import sys
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

_LOOPBACK: Final = "127.0.0.1"
_MAX_HTTP_BYTES: Final = 4_000_000
_MAX_LOG_BYTES: Final = 1_000_000
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_REVISION: Final = re.compile(r"^[0-9a-f]{40}$")
_SEMVER: Final = re.compile(r"^\d+\.\d+\.\d+$")
_PRIVATE_PATH: Final = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|users|mnt)/)")
_SECRET: Final = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|token|cookie|authorization)"
)

H4_SCHEMA: Final = "sigmax.minimax-h3-h4-private-validation/1"
PROTOCOL_STATUS: Final = "ACTIVE_PENDING_PREFLIGHT"
AUTHORIZATION_MARKER: Final = "M7-13-H4-AUTHORIZED-2026-08-18"
_DEFAULT_PROTOCOL: Final = (
    REPOSITORY_ROOT
    / ".planning"
    / "260818-M7-13_MINIMAX_H3_ACCELERATED_VALIDATION_PROTOCOL.md"
)
_DEFAULT_AUTHORIZATION: Final = (
    REPOSITORY_ROOT / ".planning" / "260818-M7-13_H4_AUTHORIZATION.md"
)

_PUBLISHER_ARTIFACTS: Final = {
    "h3.fl2va.lightx2v-turbo-4-v1.0-768p": (
        0,
        "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",
    ),
    "h3.fl2va.lightx2v-turbo-8-v1.0-544p": (
        0,
        "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",
    ),
    "h3.fl2va.lightx2v-turbo-4-v0.1-544p": (
        0,
        "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",
    ),
    "h3.ref2va.lightx2v-turbo-4-v0.1-544p": (
        0,
        "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
    ),
}

_BLOCKED_REDUCED: Final = {
    "9515eee9f642aa0e7fcc401f56d408ef2d6388f81881fe50bddded8220870a4d",
    "8e05b7b982c3aff7deb692a188c8a8d8acaeff8a12abfe1aeac822fb8ee3f0b7",
    "9ea3bd3a6aac22994153e294cf1ecab0a8766fc0f8d056ace645a01d1a6a4daf",
}
_REJECTED_LOCAL: Final = {
    "1b85da614014024a0c9507f12558917dcc69b6adb564e716324594f401723115",
    "a3208be61329c27a6754c53db9a21a3c86e2a285381700adf2d97e279c062840",
    "2c6abb194cff3e26c2295c87892913adf0c92d8f784f305238246759f9b333d0",
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
        return replace(base, disposition=RowDisposition.REJECTED, reason_code="artifact.local_modified")
    if base.sha256 in _BLOCKED_REDUCED or source == "kijai-reduced":
        return base
    expected = _PUBLISHER_ARTIFACTS.get(artifact_id)
    if source != "publisher-full" or expected is None:
        return base
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
) -> dict[str, dict[str, object]]:
    """Build a native H3 graph with one external Sigmax schedule and no double shift."""

    if variant not in {"H3 Base FL2VA", "H3 Base Ref2VA"}:
        _fail("H4 graph variant must be explicit")
    if variant != "H3 Base FL2VA":
        _fail("H4 graph currently freezes FL2VA controls only")
    for field, value in (
        ("model_name", model_name),
        ("clip_name", clip_name),
        ("video_vae_name", video_vae_name),
        ("audio_vae_name", audio_vae_name),
    ):
        _safe_relative_name(value, field=field)
    if not isinstance(prompt, str) or not prompt or _PRIVATE_PATH.search(prompt) or _SECRET.search(prompt):
        _fail("H4 prompt is private and may not contain path or secret text")
    # The preregistered protocol intentionally sends 17 as a negative-shape probe; native H3
    # may snap it to 22.  All other accepted lengths must already be on the 17k+5 grid.
    if length < 5 or (length != 17 and (length - 5) % 17):
        _fail("H4 length must use the frozen 17k+5 grid or the explicit 17-frame probe")
    if width < 32 or height < 32 or width % 32 or height % 32:
        _fail("H4 dimensions must be positive multiples of 32")
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
    model_link = [model_id, 0]
    if lora_name is not None:
        _safe_relative_name(lora_name, field="lora_name")
        model_link = [model_sampling_id, 0]
    return {
        model_id: {"class_type": "UNETLoader", "inputs": {"unet_name": model_name, "weight_dtype": "default"}},
        clip_id: {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "minimax", "device": "default"}},
        video_vae_id: {"class_type": "VAELoader", "inputs": {"vae_name": video_vae_name}},
        audio_vae_id: {"class_type": "VAELoader", "inputs": {"vae_name": audio_vae_name}},
        **({model_sampling_id: {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [model_id, 0], "lora_name": lora_name, "strength_model": 1.0}}} if lora_name is not None else {}),
        condition_id: {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"clip": [clip_id, 0], "vae": [video_vae_id, 0], "prompt": prompt, "width": width, "height": height, "length": length},
        },
        # ModelSamplingAV already carries 12/3.  Do not insert MiniMaxH3SigmaShift here:
        # CRITICAL: Sigmax's schedule is already video-shifted; a second shift changes parity.
        schedule_id: {"class_type": "Sigmax.MiniMaxH3SigmaScheduler", "inputs": {"variant": variant, "steps": steps, "start_step": 0, "end_step": -1}},
        sampler_id: {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        guider_id: {"class_type": "BasicGuider", "inputs": {"model": model_link, "conditioning": [condition_id, 0]}},
        noise_id: {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        sample_id: {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [noise_id, 0], "guider": [guider_id, 0], "sampler": [sampler_id, 0], "sigmas": [schedule_id, 0], "latent_image": [condition_id, 1]}},
        video_decode_id: {"class_type": "VAEDecode", "inputs": {"samples": [sample_id, 0], "vae": [video_vae_id, 0]}},
        audio_decode_id: {"class_type": "VAEDecodeAudio", "inputs": {"samples": [sample_id, 0], "vae": [audio_vae_id, 0]}},
        video_id: {"class_type": "CreateVideo", "inputs": {"images": [video_decode_id, 0], "fps": 24.0, "audio": [audio_decode_id, 0]}},
        save_id: {"class_type": "SaveVideo", "inputs": {"video": [video_id, 0], "filename_prefix": "m7_13_h3", "format": "mp4", "codec": "auto"}},
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
    if parsed.scheme != "http" or parsed.hostname != _LOOPBACK or port is None or parsed.username or parsed.password or parsed.fragment:
        _fail("host URL must be credential-free loopback HTTP")
    return url


def _http_json(url: str, *, method: str = "GET", body: Mapping[str, object] | None = None, timeout: float = 10.0) -> object:
    payload = None
    headers = {"Accept": "application/json"}
    if body is not None:
        payload = json.dumps(dict(body), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(_loopback_url(url), data=payload, headers=headers, method=method)  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return _decode_json(response.read(_MAX_HTTP_BYTES + 1), label="host response")
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ScheduleContractError("loopback host request failed") from exc


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((_LOOPBACK, 0))
        return cast(int, sock.getsockname()[1])


def _stage_extension(run_path: Path) -> Path:
    target = run_path / "base" / "custom_nodes" / "ComfyUI-Sigmax"
    target.mkdir(parents=True)
    shutil.copy2(REPOSITORY_ROOT / "__init__.py", target / "__init__.py")
    shutil.copytree(REPOSITORY_ROOT / "comfyui_sigmax", target / "comfyui_sigmax", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(REPOSITORY_ROOT / "web", target / "web")
    return target


def _host_command(*, host_python: Path, comfyui_root: Path, models_root: Path, run_path: Path, port: int, use_ck_attention: bool, enable_triton: bool) -> list[str]:
    command = [str(host_python), str(comfyui_root / "main.py"), "--listen", _LOOPBACK, "--port", str(port), "--base-directory", str(run_path / "base"), "--models-directory", str(models_root), "--output-directory", str(run_path / "output"), "--input-directory", str(run_path / "input"), "--temp-directory", str(run_path / "temp"), "--user-directory", str(run_path / "user"), "--database-url", "sqlite:///:memory:", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-Sigmax"]
    if use_ck_attention:
        command.append("--use-ck-attention")
    if enable_triton:
        command.append("--enable-triton-backend")
    return command


def _readiness(*, base_url: str, process: subprocess.Popen[bytes], deadline: float) -> dict[str, object]:
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


def _submit(*, base_url: str, prompt: Mapping[str, object], timeout: float) -> tuple[str, dict[str, object]]:
    value = _http_json(f"{base_url}/prompt", method="POST", body={"client_id": f"sigmax-m7-13-{uuid.uuid4().hex}", "prompt": dict(prompt)}, timeout=30)
    if not isinstance(value, dict) or not isinstance(value.get("prompt_id"), str):
        _fail("H4 prompt did not return a prompt ID")
    if value.get("node_errors") not in ({}, None):
        _fail("H4 prompt validation returned node errors")
    prompt_id = cast(str, value["prompt_id"])
    return prompt_id, _wait_history(base_url=base_url, prompt_id=prompt_id, deadline=time.monotonic() + timeout)


def _terminate(process: subprocess.Popen[bytes], *, base_url: str) -> dict[str, object]:
    with suppress(ScheduleContractError):
        _http_json(f"{base_url}/interrupt", method="POST", body={}, timeout=2)
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                killpg = getattr(os, "killpg", None)
                sigint = getattr(signal, "SIGINT", None)
                if not callable(killpg) or not isinstance(sigint, int):
                    _fail("POSIX process-group signaling is unavailable")
                cast(Callable[[int, int], None], killpg)(process.pid, sigint)
            process.wait(timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            if os.name == "nt":
                taskkill = shutil.which("taskkill")
                if taskkill is not None:
                    subprocess.run([taskkill, "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True, timeout=15)  # noqa: S603
            else:
                killpg = getattr(os, "killpg", None)
                sigkill = getattr(signal, "SIGKILL", None)
                if callable(killpg) and isinstance(sigkill, int):
                    with suppress(OSError):
                        cast(Callable[[int, int], None], killpg)(process.pid, sigkill)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
    return {"return_code": process.returncode}


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
    _fail("owned H4 loopback port was not released")


def _output_fingerprints(run_path: Path) -> tuple[str, ...]:
    results: list[str] = []
    output_root = run_path / "output"
    for path in sorted(output_root.rglob("*")):
        if path.is_file():
            results.append("sha256:" + _sha256_file(path))
    return tuple(results)


def _gpu_memory_snapshot() -> int | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run([executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True, timeout=10)  # noqa: S603
    values: list[int] = []
    for line in result.stdout.splitlines():
        with suppress(ValueError):
            values.append(int(line.strip()))
    return max(values, default=0) * 1024 * 1024 if result.returncode == 0 else None


def _host_revision(root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        _fail("git executable is unavailable")
    result = subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=15)  # noqa: S603
    revision = result.stdout.strip()
    if result.returncode != 0 or _REVISION.fullmatch(revision) is None:
        _fail("selected host is not an exact Git checkout")
    return revision


def _private_path(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if label in {"protocol", "authorization", "evidence"} and REPOSITORY_ROOT.resolve() not in resolved.parents:
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
        rendered = rendered.replace(path, "<redacted-path>").replace(path.replace("\\", "/"), "<redacted-path>")
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
) -> ArtifactObservation | None:
    if row.startswith("B-"):
        model_name = "H3/minimax_h3_fl2va_bf16.safetensors" if row == "B-BF16" else "H3/minimax_h3_fl2va_int8_convrot.safetensors"
        path = models_root / "diffusion_models" / Path(model_name)
        expected = "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182" if row == "B-BF16" else "7ad4c73e6e378b822ffd1629f27f632d3787d95f5e468e3af958f98c58df96a5"
        return _artifact_observation(path=path, artifact_id=row.casefold(), disposition=RowDisposition.ACCEPTED, expected_sha256=expected, reason_code=None)
    if turbo_artifact is None or turbo_artifact_id is None:
        return ArtifactObservation(artifact_id=row.casefold(), disposition=RowDisposition.BLOCKED, reason_code="artifact.publisher_full_not_supplied", file_bytes=None, sha256=None, header_bytes=None, tensor_count=None, dtype_counts=())
    path = models_root / "loras" / Path(_safe_relative_name(turbo_artifact, field="turbo_artifact"))
    return classify_turbo_artifact(path=path, artifact_id=turbo_artifact_id, source=turbo_source, license_ack=license_ack)


def _preflight_rows(args: argparse.Namespace, models_root: Path) -> dict[str, object]:
    rows = tuple(args.rows)
    observations: dict[str, object] = {}
    for row in rows:
        if row not in {"B-BF16", "B-INT8", "T4-768", "T8-544", "T4-544", "R4-544"}:
            _fail("H4 row is not in the frozen protocol matrix")
        observation = _row_artifact(row=row, models_root=models_root, turbo_artifact=args.turbo_artifact, turbo_artifact_id=args.turbo_artifact_id, turbo_source=args.turbo_source, license_ack=args.license_ack)
        if observation is None:
            _fail("H4 row has no artifact observation")
        observations[row] = observation.projection()
    return observations


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


def _run_rows(args: argparse.Namespace, models_root: Path, run_path: Path) -> dict[str, object]:
    if not args.prompt:
        _fail("private H4 prompt text is required through --prompt")
    clip_name = _safe_relative_name(args.text_encoder, field="text_encoder")
    video_vae = _safe_relative_name(args.video_vae, field="video_vae")
    audio_vae = _safe_relative_name(args.audio_vae, field="audio_vae")
    host_revision = _host_revision(Path(args.comfyui_root).resolve())
    port = _free_port()
    base_url = f"http://{_LOOPBACK}:{port}"
    log_path = run_path / "comfyui.log"
    process: subprocess.Popen[bytes] | None = None
    results: dict[str, object] = {}
    with log_path.open("wb") as log:
        process = subprocess.Popen(_host_command(host_python=Path(args.host_python).resolve(), comfyui_root=Path(args.comfyui_root).resolve(), models_root=models_root, run_path=run_path, port=port, use_ck_attention=args.use_ck_attention, enable_triton=args.enable_triton), cwd=run_path, stdout=log, stderr=subprocess.STDOUT, creationflags=cast(int, getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)) if os.name == "nt" else 0, start_new_session=os.name == "posix")  # noqa: S603
        try:
            _readiness(base_url=base_url, process=process, deadline=time.monotonic() + args.readiness_timeout)
            live_version = _verify_live_host_version(base_url=base_url, expected=args.host_version)
            for row in args.rows:
                observation = _row_artifact(row=row, models_root=models_root, turbo_artifact=args.turbo_artifact, turbo_artifact_id=args.turbo_artifact_id, turbo_source=args.turbo_source, license_ack=args.license_ack)
                if observation is None:
                    _fail("H4 row artifact preflight returned no observation")
                if observation.disposition is not RowDisposition.ACCEPTED:
                    results[row] = {"artifact": observation.projection(), "disposition": observation.disposition.value, "status": "not_executed"}
                    continue
                if not row.startswith("B-"):
                    results[row] = {"artifact": observation.projection(), "disposition": RowDisposition.NO_PROMOTION.value, "status": "not_executed", "reason_code": "turbo.artifact_not_eligible"}
                    continue
                steps = 20
                model_name = _safe_relative_name(
                    args.int8_diffusion_model if row == "B-INT8" else args.diffusion_model,
                    field="diffusion_model",
                )
                before = _gpu_memory_snapshot()
                prompt = build_h4_prompt(variant="H3 Base FL2VA", model_name=model_name, clip_name=clip_name, video_vae_name=video_vae, audio_vae_name=audio_vae, prompt=args.prompt, width=args.width, height=args.height, length=args.length, steps=steps, seed=args.seed, shift_video=12.0, shift_audio=3.0)
                started = time.perf_counter()
                first_prompt_id, first_history = _submit(base_url=base_url, prompt=prompt, timeout=args.execution_timeout)
                first_latency = int((time.perf_counter() - started) * 1_000_000)
                first_outputs = _output_fingerprints(run_path)
                started = time.perf_counter()
                second_prompt_id, second_history = _submit(base_url=base_url, prompt=prompt, timeout=args.execution_timeout)
                repeat_latency = int((time.perf_counter() - started) * 1_000_000)
                repeat_outputs = _output_fingerprints(run_path)
                after = _gpu_memory_snapshot()
                results[row] = {"artifact": observation.projection(), "disposition": RowDisposition.NO_PROMOTION.value, "status": "succeeded", "first_prompt_id": first_prompt_id, "repeat_prompt_id": second_prompt_id, "first_history_status": _history_summary(first_history, first_prompt_id), "repeat_history_status": _history_summary(second_history, second_prompt_id), "first_latency_us": first_latency, "repeat_latency_us": repeat_latency, "first_output_fingerprints": list(first_outputs), "repeat_output_fingerprints": list(repeat_outputs), "repeat_stable": first_outputs == repeat_outputs, "gpu_memory_before": before, "gpu_memory_after": after, "backend": {"requested_operation_backend": "unavailable", "requested_attention_backend": "ck_int8" if args.use_ck_attention else "pytorch", "actual_operation_backend": "not_observed", "actual_attention_backend": "not_observed", "observation_source": "not_observed", "launch_flags_are_not_proof": True}, "host_version": live_version, "reason_code": "backend_observation_not_supplied"}
        finally:
            shutdown = _terminate(process, base_url=base_url)
            results["shutdown"] = shutdown
            _wait_for_port_release(port)
    results["host_revision"] = host_revision
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
    if not (host_root / "main.py").is_file() or not host_python.is_file() or not models_root.is_dir():
        _fail("H4 host/python/models inputs are not valid explicit paths")
    expected_host_revision = args.expected_host_revision
    actual_host_revision = _host_revision(host_root)
    if expected_host_revision != actual_host_revision:
        _fail("selected H4 host revision does not match the exact expected revision")
    preflight = _preflight_rows(args, models_root)
    components = _preflight_components(args, models_root)
    evidence: dict[str, object] = {"schema": H4_SCHEMA, "candidate": {"commit": commit, "tree": tree}, "host": {"version": args.host_version, "revision": actual_host_revision}, "components": components, "rows": preflight, "authorization": "private_non_redistribution", "gpu_execution_requested": True}
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
    evidence["execution"] = _run_rows(args, models_root, run_path)
    evidence["status"] = "execution_complete"
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
    parser.add_argument("--rows", nargs="+", default=["B-BF16", "B-INT8", "T4-768", "T8-544", "T4-544", "R4-544"])
    parser.add_argument("--allow-gpu-execution", action="store_true")
    parser.add_argument("--license-ack", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--use-ck-attention", action="store_true")
    parser.add_argument("--enable-triton", action="store_true")
    parser.add_argument("--turbo-artifact")
    parser.add_argument("--turbo-artifact-id")
    parser.add_argument("--turbo-source", choices=["publisher-full", "kijai-reduced", "local-modified"], default="publisher-full")
    parser.add_argument("--diffusion-model", default="H3/minimax_h3_fl2va_bf16.safetensors")
    parser.add_argument("--int8-diffusion-model", default="H3/minimax_h3_fl2va_int8_convrot.safetensors")
    parser.add_argument("--text-encoder", default="qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    parser.add_argument("--text-encoder-sha256")
    parser.add_argument("--video-vae", default="minimax_h3_video_vae_fp16.safetensors")
    parser.add_argument("--video-vae-sha256")
    parser.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
    parser.add_argument("--audio-vae-sha256")
    parser.add_argument("--prompt")
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
        if args.evidence_file is not None:
            target = _private_path(args.evidence_file, label="evidence")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ScheduleContractError, OSError, ValueError) as exc:
        print(f"H4_VALIDATION_ERROR: {_redacted_diagnostic(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
