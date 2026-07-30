"""Run the canonical foundation validation gate."""

from __future__ import annotations

import email
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

STAGES: Final = (
    "preflight",
    "detect-secrets",
    "pre-commit",
    "ruff-format",
    "ruff-lint",
    "mypy",
    "pytest",
    "coverage",
    "package",
)

UNAVAILABLE_LANES: Final = {
    "browser_e2e": "NOT_APPLICABLE",
    "comfyui_host_e2e": "NOT_IMPLEMENTED",
    "golden_parity": "NOT_IMPLEMENTED",
    "gpu_model_weights": "NOT_IMPLEMENTED",
    "mutation": "NOT_IMPLEMENTED",
    "property": "NOT_IMPLEMENTED",
}


def _run(stage: str, arguments: list[str], environment: dict[str, str]) -> None:
    print(f"\n== {stage} ==")
    print(" ".join(arguments))
    subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def _inspect_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))

    required = {
        "comfyui_sigmax/__init__.py",
        "comfyui_sigmax/py.typed",
        "comfyui_sigmax/core/__init__.py",
        "comfyui_sigmax/core/schedule_contracts.py",
    }
    if not required.issubset(names):
        raise RuntimeError(f"Wheel is missing required files: {sorted(required - set(names))}")

    forbidden = (".planning/", "reference/", "tests/", ".venv/", "models/", "cache/")
    leaked = [name for name in names if any(token in name.lower() for token in forbidden)]
    if leaked:
        raise RuntimeError(f"Wheel contains forbidden internal paths: {leaked}")

    requirements = metadata.get_all("Requires-Dist") or []
    runtime_requirements = [value for value in requirements if "extra ==" not in value]
    if runtime_requirements:
        raise RuntimeError(f"Unexpected mandatory runtime dependencies: {runtime_requirements}")

    print(f"Wheel: {wheel.name}")
    print(f"Wheel files: {len(names)}")
    print("Mandatory runtime dependencies: 0")


def main() -> int:
    python = sys.executable
    environment = os.environ.copy()
    cache_name = "pre-commit-win" if os.name == "nt" else "pre-commit-linux"
    environment["PRE_COMMIT_HOME"] = str(REPOSITORY_ROOT / ".tmp" / cache_name)

    commands = {
        "preflight": [python, "scripts/preflight_check.py"],
        "detect-secrets": [
            python,
            "-m",
            "pre_commit",
            "run",
            "detect-secrets",
            "--all-files",
        ],
        "pre-commit": [
            python,
            "-m",
            "pre_commit",
            "run",
            "--all-files",
            "--show-diff-on-failure",
        ],
        "ruff-format": [python, "-m", "ruff", "format", "--check", "."],
        "ruff-lint": [python, "-m", "ruff", "check", "."],
        "mypy": [python, "-m", "mypy", "comfyui_sigmax", "tests", "scripts"],
        "pytest": [python, "-m", "pytest"],
        "coverage": [
            python,
            "-m",
            "pytest",
            "--cov=comfyui_sigmax",
            "--cov-branch",
        ],
    }

    try:
        for stage in STAGES[:-1]:
            _run(stage, commands[stage], environment)

        output_root = REPOSITORY_ROOT / ".tmp"
        output_root.mkdir(exist_ok=True)
        wheel_directory = Path(tempfile.mkdtemp(prefix="full-gate-wheel-", dir=output_root))
        _run(
            "package",
            [python, "-m", "build", "--wheel", "--outdir", str(wheel_directory)],
            environment,
        )
        wheel = next(wheel_directory.glob("*.whl"))
        _inspect_wheel(wheel)
    except (OSError, RuntimeError, StopIteration, subprocess.CalledProcessError) as error:
        print(f"\nFULL_GATE=FAIL\n{error}", file=sys.stderr)
        return 1

    print("\nFULL_GATE=PASS")
    print(json.dumps(UNAVAILABLE_LANES, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
