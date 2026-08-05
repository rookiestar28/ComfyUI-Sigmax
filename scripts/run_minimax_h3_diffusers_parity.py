"""Run isolated MiniMax H3 Diffusers parity against the pinned unreleased branch."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from scripts.parity.minimax_h3_diffusers_report import (
    NUMPY_VERSION,
    REQUIRED_GRID_POINTS,
    TORCH_VERSION,
    build_parity_report,
    canonical_json,
)
from scripts.parity.minimax_h3_official import DIFFUSERS_REVISION


def _require_distribution_version(distribution: str, expected: str) -> str:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"missing parity dependency {distribution}=={expected}; "
            "install requirements/parity-minimax-h3-diffusers.txt in an isolated environment"
        ) from error
    cpu_build = distribution == "torch" and actual == f"{expected}+cpu"
    if actual != expected and not cpu_build:
        raise RuntimeError(f"parity dependency {distribution} must be {expected}, found {actual}")
    return expected


def _require_diffusers_revision() -> str:
    """Require a VCS-installed Diffusers distribution at the frozen H3 commit."""

    try:
        distribution = importlib.metadata.distribution("diffusers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Diffusers is not installed in the isolated parity environment"
        ) from error
    raw_direct_url = distribution.read_text("direct_url.json")
    if not raw_direct_url:
        raise RuntimeError("Diffusers direct_url.json is missing; a VCS commit cannot be proven")
    try:
        direct_url = json.loads(raw_direct_url)
    except json.JSONDecodeError as error:
        raise RuntimeError("Diffusers direct_url.json is malformed") from error
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("commit_id") != DIFFUSERS_REVISION:
        raise RuntimeError("Diffusers is not installed from the pinned MiniMax H3 commit")
    return DIFFUSERS_REVISION


def _tensor_values(tensor: object) -> tuple[float, ...]:
    try:
        detached = tensor.detach()  # type: ignore[attr-defined]
        cpu = detached.cpu()
        values = cpu.tolist()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError("Diffusers parity tensor cannot be read as CPU values") from error
    if not isinstance(values, list):
        raise RuntimeError("Diffusers parity tensor must flatten to a list")
    return tuple(float(value) for value in values)


def _load_diffusers_cases() -> tuple[dict[int, dict[str, tuple[float, ...]]], dict[str, str]]:
    diffusers_revision = _require_diffusers_revision()
    versions = {
        "device": "cpu",
        "diffusers": importlib.metadata.version("diffusers"),
        "diffusers_revision": diffusers_revision,
        "numpy": _require_distribution_version("numpy", NUMPY_VERSION),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": _require_distribution_version("torch", TORCH_VERSION),
    }
    torch = importlib.import_module("torch")
    scheduler_module = importlib.import_module("diffusers.schedulers.scheduling_minimax_h3")
    scheduler_type = cast(Any, scheduler_module.MiniMaxH3Scheduler)
    cases: dict[int, dict[str, tuple[float, ...]]] = {}
    with torch.no_grad():
        for points in REQUIRED_GRID_POINTS:
            video_scheduler = scheduler_type(shift=12.0)
            video_scheduler.set_timesteps(num_inference_steps=points, device="cpu")
            audio_scheduler = scheduler_type(shift=3.0)
            audio_scheduler.set_timesteps(num_inference_steps=points, device="cpu")
            cases[points] = {
                "audio": _tensor_values(audio_scheduler.sigmas),
                "video": _tensor_values(video_scheduler.sigmas),
            }
    return cases, versions


def _load_velocity_probe() -> dict[str, object]:
    torch = importlib.import_module("torch")
    scheduler_module = importlib.import_module("diffusers.schedulers.scheduling_minimax_h3")
    scheduler_type = cast(Any, scheduler_module.MiniMaxH3Scheduler)
    scheduler = scheduler_type(shift=12.0)
    scheduler.set_timesteps(num_inference_steps=4, device="cpu")
    sample = torch.tensor((0.25, -0.5), dtype=torch.float32, device="cpu")
    velocity = torch.tensor((0.125, -0.25), dtype=torch.float32, device="cpu")
    timestep = scheduler.timesteps[0]
    with torch.no_grad():
        result = scheduler.step(velocity, timestep, sample)
    return {
        "reference": _tensor_values(result.prev_sample),
        "sample": _tensor_values(sample),
        "sigma": float(scheduler.sigmas[0].item()),
        "sigma_next": float(scheduler.sigmas[1].item()),
        "timestep": float(timestep.item()),
        "velocity": _tensor_values(velocity),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate a complete report or fail without creating partial evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    try:
        cases, environment = _load_diffusers_cases()
        report = build_parity_report(
            cases,
            velocity_reference=_load_velocity_probe(),
            environment=environment,
        )
        encoded = canonical_json(report)
    except (
        AttributeError,
        ImportError,
        KeyError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"PARITY=FAIL\n{error}", file=sys.stderr)
        return 2

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"PARITY=PASS\nREPORT={parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
