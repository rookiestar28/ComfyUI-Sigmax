"""Execute pinned native ComfyUI MiniMax H3 schedule/adapter parity on CPU."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from comfyui_sigmax.profiles.minimax_h3 import MINIMAX_H3_AUDIO_SHIFT, MINIMAX_H3_VIDEO_SHIFT
from scripts.parity.minimax_h3_comfy_native_report import (
    COMFYUI_H3_REVISION,
    DEPENDENCY_VERSIONS,
    REQUIRED_AUDIO_PROBES,
    REQUIRED_TRANSITIONS,
    SOURCE_BLOBS,
    build_native_report,
    canonical_json,
)


def _git(root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required for native H3 parity")
    result = subprocess.run(  # noqa: S603 - fixed git executable and local source-root tokens
        [git, "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_source_root(root: Path) -> Path:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RuntimeError("ComfyUI H3 source root does not exist")
    if _git(resolved, "rev-parse", "HEAD") != COMFYUI_H3_REVISION:
        raise RuntimeError("ComfyUI source revision does not match the pinned H3 commit")
    if _git(resolved, "status", "--porcelain"):
        raise RuntimeError("ComfyUI H3 source worktree must be clean")
    for relative_path, expected_blob in SOURCE_BLOBS.items():
        actual_blob = _git(resolved, "rev-parse", f"HEAD:{relative_path}")
        if actual_blob != expected_blob:
            raise RuntimeError(f"ComfyUI H3 source blob mismatch: {relative_path}")
    return resolved


def _validate_environment() -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("native ComfyUI H3 parity requires CPython 3.13")
    for distribution, expected in DEPENDENCY_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"missing native H3 parity dependency {distribution}=={expected}; "
                "install requirements/parity-comfyui-h3-native.txt"
            ) from error
        cpu_build = distribution == "torch" and actual == f"{expected}+cpu"
        if actual != expected and not cpu_build:
            raise RuntimeError(
                f"native H3 parity dependency {distribution} must be {expected}, found {actual}"
            )


def _require_module_within(module: Any, source_root: Path) -> None:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise RuntimeError("native H3 module has no source path")
    module_path = Path(raw_path).resolve()
    if not module_path.is_relative_to(source_root):
        raise RuntimeError("native H3 module resolved outside the pinned source root")


def _load_native_values(
    source_root: Path,
) -> tuple[dict[int, tuple[float, ...]], dict[float, tuple[float, float]]]:
    # CPU/offline policy must be established before importing host modules.
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()

    cli_args = importlib.import_module("comfy.cli_args")
    cast(Any, cli_args).args.cpu = True
    torch = importlib.import_module("torch")
    model_sampling = importlib.import_module("comfy.model_sampling")
    samplers = importlib.import_module("comfy.samplers")
    minimax = importlib.import_module("comfy.ldm.minimax.model")
    _require_module_within(model_sampling, source_root)
    _require_module_within(samplers, source_root)
    _require_module_within(minimax, source_root)

    source_text = (source_root / "comfy/ldm/minimax/model.py").read_text(encoding="utf-8")
    if (
        "return [-video_out.to(video_x.dtype), (-slope_a) * audio_out.to(audio_x.dtype)]"
        not in source_text
    ):
        raise RuntimeError("native H3 model-output sign adapter source is not the pinned contract")

    config = SimpleNamespace(sampling_settings={"shift": MINIMAX_H3_VIDEO_SHIFT})
    native_sampling = cast(Any, model_sampling).ModelSamplingDiscreteFlow(config)
    if native_sampling.sigmas.device.type != "cpu":
        raise RuntimeError("native H3 sigma table must execute on CPU")
    if str(native_sampling.sigmas.dtype) != "torch.float32":
        raise RuntimeError("native H3 sigma table must use float32")
    if len(native_sampling.sigmas) != 1000:
        raise RuntimeError("native H3 sigma table length must be 1000")

    vectors: dict[int, tuple[float, ...]] = {}
    for transitions in REQUIRED_TRANSITIONS:
        raw = cast(Any, samplers).calculate_sigmas(native_sampling, "simple", transitions)
        vectors[transitions] = tuple(float(value) for value in raw.tolist())

    mappings: dict[float, tuple[float, float]] = {}
    for video_sigma in REQUIRED_AUDIO_PROBES:
        sigma = cast(Any, torch).tensor(video_sigma, dtype=cast(Any, torch).float32, device="cpu")
        audio = cast(Any, minimax).time_shift_sigma(
            sigma, MINIMAX_H3_VIDEO_SHIFT, MINIMAX_H3_AUDIO_SHIFT
        )
        derivative = cast(Any, minimax).time_shift_slope(
            sigma, MINIMAX_H3_VIDEO_SHIFT, MINIMAX_H3_AUDIO_SHIFT
        )
        mappings[video_sigma] = (float(audio.item()), float(derivative.item()))
    return vectors, mappings


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate one complete native report or fail without writing partial evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    try:
        source_root = _validate_source_root(parsed.comfyui_root)
        _validate_environment()
        vectors, mappings = _load_native_values(source_root)
        report = build_native_report(
            vectors,
            native_mappings=mappings,
            environment={
                "device": "cpu",
                "numpy": DEPENDENCY_VERSIONS["numpy"],
                "python": "3.13",
                "torch": DEPENDENCY_VERSIONS["torch"],
            },
        )
        encoded = canonical_json(report)
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"NATIVE_H3_PARITY=FAIL\n{error}", file=sys.stderr)
        return 2

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"NATIVE_H3_PARITY=PASS\nREPORT={parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
