"""Run isolated Krea 2 RAW parity against pinned Diffusers 0.39.0."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from scripts.parity.krea2_raw_report import (
    CASE_SPECS,
    build_parity_report,
    canonical_json,
    raw_case_id,
)
from scripts.parity.krea2_turbo_report import (
    DIFFUSERS_VERSION,
    NUMPY_VERSION,
    TORCH_VERSION,
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


def _load_pinned_calculate_shift() -> Callable[..., float]:
    """Execute only Diffusers' pinned pure helper without pipeline-only dependencies."""

    spec = importlib.util.find_spec("diffusers.pipelines.krea2.pipeline_krea2")
    if spec is None or spec.origin is None:
        raise RuntimeError("pinned Diffusers Krea 2 pipeline source is unavailable")
    source_path = Path(spec.origin)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "calculate_shift"
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise RuntimeError("pinned Diffusers calculate_shift definition is unavailable")
    module = ast.Module(body=[matches[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    function = namespace.get("calculate_shift")
    if not callable(function):
        raise RuntimeError("pinned Diffusers calculate_shift is not executable")
    return cast(Callable[..., float], function)


def _load_diffusers_cases() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
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
    scheduler_type = cast(Any, scheduler_module.FlowMatchEulerDiscreteScheduler)
    linspace = cast(Any, numpy_module.linspace)
    calculate_shift = _load_pinned_calculate_shift()

    cases: dict[str, dict[str, object]] = {}
    for recipe_id, steps, width, height in CASE_SPECS:
        effective_width = ((width + 15) // 16) * 16
        effective_height = ((height + 15) // 16) * 16
        image_seq_len = (effective_width // 16) * (effective_height // 16)
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=256,
            max_seq_len=6400,
            base_shift=0.5,
            max_shift=1.15,
        )
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
        base_sigmas = cast(list[float], linspace(1.0, 1.0 / steps, steps).tolist())
        scheduler.set_timesteps(
            num_inference_steps=steps,
            device="cpu",
            sigmas=base_sigmas,
            mu=mu,
        )
        raw_values = cast(list[float], scheduler.sigmas.tolist())
        cases[raw_case_id(recipe_id, width, height)] = {
            "mu": float(mu),
            "sigmas": tuple(float(value) for value in raw_values),
        }
    return cases, versions


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate canonical RAW evidence or fail without writing a partial report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)

    try:
        cases, environment = _load_diffusers_cases()
        report = build_parity_report(cases, environment=environment)
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
