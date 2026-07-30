"""Execute pinned native ComfyUI schedule parity in an isolated CPU process."""

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

from scripts.parity.krea2_comfy_native_report import (
    COMFYUI_REVISION,
    DEPENDENCY_VERSIONS,
    REQUIRED_STEPS,
    SOURCE_BLOBS,
    build_native_report,
    canonical_json,
)


def _git(root: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required for native parity")
    result = subprocess.run(
        [git, "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_source_root(root: Path) -> Path:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RuntimeError("ComfyUI source root does not exist")
    if _git(resolved, "rev-parse", "HEAD") != COMFYUI_REVISION:
        raise RuntimeError("ComfyUI source revision does not match the pinned contract")
    if _git(resolved, "status", "--porcelain"):
        raise RuntimeError("ComfyUI source worktree must be clean")
    for relative_path, expected_blob in SOURCE_BLOBS.items():
        actual_blob = _git(resolved, "rev-parse", f"HEAD:{relative_path}")
        if actual_blob != expected_blob:
            raise RuntimeError(f"ComfyUI source blob mismatch: {relative_path}")
    return resolved


def _validate_environment() -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("native ComfyUI parity requires CPython 3.13")
    for distribution, expected in DEPENDENCY_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"missing native parity dependency {distribution}=={expected}; "
                "install requirements/parity-comfyui-native.txt"
            ) from error
        if actual != expected:
            raise RuntimeError(
                f"native parity dependency {distribution} must be {expected}, found {actual}"
            )


def _require_module_within(module: Any, source_root: Path) -> None:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise RuntimeError("native ComfyUI module has no source path")
    module_path = Path(raw_path).resolve()
    if not module_path.is_relative_to(source_root):
        raise RuntimeError("native ComfyUI module resolved outside the pinned source root")


def _load_native_vectors(source_root: Path) -> dict[int, tuple[float, ...]]:
    # CRITICAL: CPU mode must be set before importing samplers/model_management.
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()

    cli_args = importlib.import_module("comfy.cli_args")
    cast(Any, cli_args).args.cpu = True
    model_sampling = importlib.import_module("comfy.model_sampling")
    samplers = importlib.import_module("comfy.samplers")
    _require_module_within(model_sampling, source_root)
    _require_module_within(samplers, source_root)

    scheduler_handlers = cast(Any, samplers).SCHEDULER_HANDLERS
    if scheduler_handlers["simple"].handler is not cast(Any, samplers).simple_scheduler:
        raise RuntimeError("registered simple scheduler is not the pinned native handler")

    config = SimpleNamespace(sampling_settings={"multiplier": 1.0, "shift": 1.15})
    native_sampling = cast(Any, model_sampling).ModelSamplingFlux(config)
    if native_sampling.__class__.__name__ != "ModelSamplingFlux":
        raise RuntimeError("native model sampling class is invalid")
    if native_sampling.sigmas.device.type != "cpu":
        raise RuntimeError("native model sampling must execute on CPU")
    if str(native_sampling.sigmas.dtype) != "torch.float32":
        raise RuntimeError("native sigma table must use float32")
    if len(native_sampling.sigmas) != 10000:
        raise RuntimeError("native sigma table length must be 10000")

    vectors: dict[int, tuple[float, ...]] = {}
    for steps in REQUIRED_STEPS:
        raw = cast(Any, samplers).calculate_sigmas(native_sampling, "simple", steps)
        vectors[steps] = tuple(float(value) for value in raw.tolist())
    return vectors


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate canonical native evidence or fail without writing output."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    try:
        source_root = _validate_source_root(parsed.comfyui_root)
        _validate_environment()
        vectors = _load_native_vectors(source_root)
        encoded = canonical_json(build_native_report(vectors))
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"NATIVE_PARITY=FAIL\n{error}", file=sys.stderr)
        return 2

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"NATIVE_PARITY=PASS\nREPORT={parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
