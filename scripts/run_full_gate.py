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
    "core-independence",
    "frontend-policy",
    "parity-contract",
    "pytest",
    "coverage",
    "package",
)

LANE_STATUS: Final = {
    "browser_e2e": "IMPLEMENTED_SEPARATE_GATE",
    "coinstallation_mutation": "IMPLEMENTED_SYNTHETIC",
    "comfyui_host_e2e": "IMPLEMENTED_SEPARATE_GATE",
    "core_independence": "IMPLEMENTED",
    "environment_guardrails": "IMPLEMENTED",
    "framework_parity": "IMPLEMENTED",
    "frontend_policy": "IMPLEMENTED_NODE_TEST",
    "golden": "IMPLEMENTED",
    "gpu_model_weights": "NOT_IMPLEMENTED",
    "mutation": "NOT_IMPLEMENTED",
    "native_comfyui_parity": "IMPLEMENTED",
    "performance_budgets": "IMPLEMENTED",
    "property": "IMPLEMENTED",
}


def _isolated_tool_path(python: str, path_value: str) -> str:
    """Prefer the selected venv and hide conflicting pre-commit launchers."""

    selected = Path(python).resolve().parent
    names = (
        ("pre-commit.exe", "pre-commit.cmd", "pre-commit.bat", "pre-commit")
        if os.name == "nt"
        else ("pre-commit",)
    )
    retained: list[str] = [str(selected)]
    seen = {os.path.normcase(str(selected))}
    for value in path_value.split(os.pathsep):
        if not value:
            continue
        directory = Path(value)
        try:
            identity = os.path.normcase(str(directory.resolve()))
        except OSError:
            identity = os.path.normcase(str(directory.absolute()))
        if identity in seen:
            continue
        seen.add(identity)
        if any((directory / name).is_file() for name in names):
            continue
        retained.append(value)
    return os.pathsep.join(retained)


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
        "comfyui_sigmax/adapters/__init__.py",
        "comfyui_sigmax/adapters/comfyui.py",
        "comfyui_sigmax/adapters/registration.py",
        "comfyui_sigmax/coinstallation/__init__.py",
        "comfyui_sigmax/coinstallation/matrix_v1.json",
        "comfyui_sigmax/coinstallation_matrix.py",
        "comfyui_sigmax/compatibility/__init__.py",
        "comfyui_sigmax/compatibility/matrix_v1.json",
        "comfyui_sigmax/compatibility_matrix.py",
        "comfyui_sigmax/core/__init__.py",
        "comfyui_sigmax/core/artifacts.py",
        "comfyui_sigmax/core/base_grids.py",
        "comfyui_sigmax/core/capabilities.py",
        "comfyui_sigmax/core/fingerprints.py",
        "comfyui_sigmax/core/request_result.py",
        "comfyui_sigmax/core/schedule_contracts.py",
        "comfyui_sigmax/core/shifts.py",
        "comfyui_sigmax/core/terminal_slicing.py",
        "comfyui_sigmax/core/validation.py",
        "comfyui_sigmax/host_mutation.py",
        "comfyui_sigmax/nodes/__init__.py",
        "comfyui_sigmax/nodes/flux1_schnell_sigma_scheduler.py",
        "comfyui_sigmax/nodes/krea2_sigma_scheduler.py",
        "comfyui_sigmax/nodes/sd3_sigma_scheduler.py",
        "comfyui_sigmax/nodes/aura_flow_sigma_scheduler.py",
        "comfyui_sigmax/nodes/z_image_sigma_scheduler.py",
        "comfyui_sigmax/nodes/raw_workflow_output.py",
        "comfyui_sigmax/performance/__init__.py",
        "comfyui_sigmax/performance/matrix_v1.json",
        "comfyui_sigmax/performance_budgets.py",
        "comfyui_sigmax/performance_matrix.py",
        "comfyui_sigmax/profiles/__init__.py",
        "comfyui_sigmax/profiles/flux1_schnell.py",
        "comfyui_sigmax/profiles/krea2_common.py",
        "comfyui_sigmax/profiles/krea2_raw.py",
        "comfyui_sigmax/profiles/krea2_turbo.py",
        "comfyui_sigmax/profiles/sd3.py",
        "comfyui_sigmax/profiles/aura_flow.py",
        "comfyui_sigmax/profiles/z_image.py",
        "comfyui_sigmax/profiles/resolution.py",
        "comfyui_sigmax/workflows/__init__.py",
        "comfyui_sigmax/workflows/fixtures.json",
        "comfyui_sigmax/workflows/host_baseline.json",
        "comfyui_sigmax/workflows/validation.py",
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
    temp_name = "runtime-win" if os.name == "nt" else "runtime-linux"
    output_root = REPOSITORY_ROOT / ".tmp"
    cache_root = output_root / cache_name
    temp_root = output_root / temp_name
    cache_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    environment["PRE_COMMIT_HOME"] = str(cache_root)
    environment["SIGMAX_TEMP_ROOT"] = str(temp_root)
    environment["TMPDIR"] = str(temp_root)
    environment["TMP"] = str(temp_root)
    environment["TEMP"] = str(temp_root)
    if os.name != "nt":
        environment["SIGMAX_PYTEST_CAPTURE_MODE"] = "sys"
    # CRITICAL: keep gate tooling on the selected venv; mixed pre-commit launchers corrupt caches.
    environment["PATH"] = _isolated_tool_path(python, environment.get("PATH", ""))

    pytest_capture = ["--capture=sys"] if os.name != "nt" else []
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
        "core-independence": [python, "scripts/check_core_independence.py"],
        "frontend-policy": [python, "scripts/run_frontend_policy_tests.py"],
        "parity-contract": [
            python,
            "-m",
            "pytest",
            *pytest_capture,
            "tests/parity",
            "-m",
            "parity",
        ],
        "pytest": [python, "-m", "pytest", *pytest_capture],
        "coverage": [
            python,
            "-m",
            "pytest",
            *pytest_capture,
            "--cov=comfyui_sigmax",
            "--cov-branch",
        ],
    }

    try:
        for stage in STAGES[:-1]:
            _run(stage, commands[stage], environment)

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
    print(json.dumps(LANE_STATUS, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
