"""Run isolated Krea 2 Turbo parity against pinned Diffusers 0.39.0."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from scripts.parity.krea2_turbo_report import (
    DIFFUSERS_VERSION,
    NUMPY_VERSION,
    REQUIRED_STEPS,
    TORCH_VERSION,
    build_parity_report,
    canonical_json,
)


def _require_distribution_version(distribution: str, expected: str) -> str:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"missing parity dependency {distribution}=={expected}; "
            "install requirements/parity-krea2-turbo.txt in an isolated environment"
        ) from error
    # IMPORTANT: official CPU wheels use a PEP 440 local suffix; keep report metadata canonical.
    cpu_build = distribution == "torch" and actual == f"{expected}+cpu"
    if actual != expected and not cpu_build:
        raise RuntimeError(f"parity dependency {distribution} must be {expected}, found {actual}")
    return expected


def _load_diffusers_vectors() -> tuple[dict[int, tuple[float, ...]], dict[str, str]]:
    versions = {
        "device": "cpu",
        "diffusers": _require_distribution_version("diffusers", DIFFUSERS_VERSION),
        "numpy": _require_distribution_version("numpy", NUMPY_VERSION),
        "torch": _require_distribution_version("torch", TORCH_VERSION),
    }
    numpy_module = importlib.import_module("numpy")
    importlib.import_module("torch")
    scheduler_module = importlib.import_module(
        "diffusers.schedulers.scheduling_flow_match_euler_discrete"
    )
    scheduler_type = cast(
        Any,
        scheduler_module.FlowMatchEulerDiscreteScheduler,
    )
    linspace = cast(Any, numpy_module.linspace)

    vectors: dict[int, tuple[float, ...]] = {}
    for steps in REQUIRED_STEPS:
        scheduler = scheduler_type(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=6400,
            invert_sigmas=False,
            shift_terminal=None,
            use_karras_sigmas=False,
            use_exponential_sigmas=False,
            use_beta_sigmas=False,
            time_shift_type="exponential",
            stochastic_sampling=False,
        )
        base_sigmas = cast(
            list[float],
            linspace(1.0, 1.0 / steps, steps).tolist(),
        )
        scheduler.set_timesteps(
            num_inference_steps=steps,
            device="cpu",
            sigmas=base_sigmas,
            mu=1.15,
        )
        raw_values = cast(list[float], scheduler.sigmas.tolist())
        vectors[steps] = tuple(float(value) for value in raw_values)
    return vectors, versions


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate a canonical report or fail without creating misleading evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    try:
        vectors, environment = _load_diffusers_vectors()
        report = build_parity_report(vectors, environment=environment)
        encoded = canonical_json(report)
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as error:
        print(f"PARITY=FAIL\n{error}", file=sys.stderr)
        return 2

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"PARITY=PASS\nREPORT={parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
