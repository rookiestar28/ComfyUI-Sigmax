"""Execute pinned native ComfyUI Euler parity in an isolated CPU process."""

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
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from comfyui_sigmax.profiles import build_krea2_turbo_schedule  # noqa: E402
from scripts.parity.krea2_native_euler_report import (  # noqa: E402
    COMFYUI_REVISION,
    CONTROL_BIASES,
    CONTROL_INITIAL_STATE,
    DEPENDENCY_VERSIONS,
    SOURCE_BLOBS,
    build_native_euler_report,
    canonical_json,
)


def _git(root: Path, *arguments: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("git executable is required for native Euler parity")
    # SECURITY: Git is resolved from PATH; the root and argument tokens are locally controlled.
    result = subprocess.run(  # noqa: S603
        [executable, "-C", str(root), *arguments],
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
        if _git(resolved, "rev-parse", f"HEAD:{relative_path}") != expected_blob:
            raise RuntimeError(f"ComfyUI source blob mismatch: {relative_path}")
    return resolved


def _validate_environment() -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("native ComfyUI Euler parity requires CPython 3.13")
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
    if not isinstance(raw_path, str) or not Path(raw_path).resolve().is_relative_to(source_root):
        raise RuntimeError("native ComfyUI module resolved outside the pinned source root")


def _tensor_vector(value: Any) -> list[float]:
    flat = value.detach().cpu().reshape(-1).tolist()
    return [float(item) for item in flat]


def _execute_once(
    *,
    torch: Any,
    sample_euler: Any,
    converter: Any,
    sigmas: Any,
) -> tuple[list[dict[str, object]], list[float]]:
    calls: list[dict[str, object]] = []

    class ControlledFlowModel:
        def __call__(self, state: Any, sigma: Any, **_extra: object) -> Any:
            index = len(calls)
            bias = state.new_tensor(CONTROL_BIASES).reshape(1, -1)
            velocity = state * 0.125 + sigma.reshape(-1, 1) * 0.25 + bias + (index + 1) * 0.03125
            denoised = converter.calculate_denoised(sigma, velocity, state)
            calls.append(
                {
                    "denoised": _tensor_vector(denoised),
                    "index": index,
                    "input_state": _tensor_vector(state),
                    "sigma": float(sigma.detach().cpu().reshape(-1)[0]),
                    "velocity": _tensor_vector(velocity),
                }
            )
            return denoised

    initial = torch.tensor(
        [CONTROL_INITIAL_STATE],
        dtype=torch.float32,
        device="cpu",
    )
    final = sample_euler(
        ControlledFlowModel(),
        initial,
        sigmas,
        disable=True,
        s_churn=0.0,
    )
    final_vector = _tensor_vector(final)
    if len(calls) != len(sigmas) - 1:
        raise RuntimeError("native Euler model-evaluation count drifted")
    steps: list[dict[str, object]] = []
    for index, call in enumerate(calls):
        output = calls[index + 1]["input_state"] if index + 1 < len(calls) else final_vector
        steps.append(
            {
                **call,
                "output_state": output,
                "sigma_next": float(sigmas[index + 1].detach().cpu()),
            }
        )
    return steps, final_vector


def _load_native_case(source_root: Path) -> dict[str, object]:
    # CRITICAL: CPU/offline arguments must be fixed before importing ComfyUI modules.
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()

    cli_args = importlib.import_module("comfy.cli_args")
    cast(Any, cli_args).args.cpu = True
    sampling = importlib.import_module("comfy.k_diffusion.sampling")
    model_sampling = importlib.import_module("comfy.model_sampling")
    torch = importlib.import_module("torch")
    _require_module_within(sampling, source_root)
    _require_module_within(model_sampling, source_root)

    sigmas = torch.tensor(
        build_krea2_turbo_schedule(steps=8).sigmas,
        dtype=torch.float32,
        device="cpu",
    )
    converter = cast(Any, model_sampling).CONST()
    first_steps, first_final = _execute_once(
        torch=torch,
        sample_euler=cast(Any, sampling).sample_euler,
        converter=converter,
        sigmas=sigmas,
    )
    second_steps, second_final = _execute_once(
        torch=torch,
        sample_euler=cast(Any, sampling).sample_euler,
        converter=converter,
        sigmas=sigmas,
    )
    if first_steps != second_steps or first_final != second_final:
        raise RuntimeError("native deterministic Euler rerun drifted")
    return {
        "counts": {
            "effective_model_evaluations": len(first_steps),
            "effective_transitions": len(first_steps),
            "requested_model_evaluations": len(sigmas) - 1,
            "requested_transitions": len(sigmas) - 1,
        },
        "deterministic_rerun": True,
        "initial_state": list(CONTROL_INITIAL_STATE),
        "native_final": first_final,
        "native_steps": first_steps,
        "rerun_final": second_final,
        "sigmas": _tensor_vector(sigmas),
        "steps": len(sigmas) - 1,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    """Generate canonical native Euler evidence or fail without publishing output."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    try:
        source_root = _validate_source_root(parsed.comfyui_root)
        _validate_environment()
        encoded = canonical_json(build_native_euler_report(_load_native_case(source_root)))
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"NATIVE_EULER_PARITY=FAIL\n{error}", file=sys.stderr)
        return 2

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(f"NATIVE_EULER_PARITY=PASS\nREPORT={parsed.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
