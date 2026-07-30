"""Validate the local interpreter and foundation-gate assumptions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

import tomli


class PreflightReport(TypedDict):
    status: str
    python: str
    executable: str
    environment_prefix: str
    expected_environment: str
    project_local_venv: bool
    node: str
    errors: list[str]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _expected_environment(override: Path | None = None) -> Path:
    if override is not None:
        return override.resolve()
    name = ".venv" if os.name == "nt" else ".venv-wsl"
    return (REPOSITORY_ROOT / name).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_report(expected_environment_override: Path | None = None) -> PreflightReport:
    expected_environment = _expected_environment(expected_environment_override)
    executable = Path(sys.executable).absolute()
    environment_prefix = Path(sys.prefix).resolve()
    local_environment = environment_prefix == expected_environment or _is_within(
        environment_prefix, expected_environment
    )

    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomli.load(stream)

    errors: list[str] = []
    if sys.version_info < (3, 10):  # noqa: UP036 - preflight must fail before package import
        errors.append("Python 3.10 or newer is required.")
    if not local_environment:
        errors.append(
            "Use the repository-local environment at "
            f"{expected_environment}. Create it and install '.[dev]' before retrying."
        )
    if metadata["project"]["requires-python"] != ">=3.10":
        errors.append("pyproject.toml Python floor differs from the preflight policy.")
    if metadata["project"]["dependencies"]:
        errors.append("Mandatory runtime dependencies must remain empty at the foundation gate.")

    node_status = (
        "NOT_IMPLEMENTED" if (REPOSITORY_ROOT / "package.json").exists() else "NOT_APPLICABLE"
    )
    return {
        "status": "FAIL" if errors else "PASS",
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "executable": str(executable),
        "environment_prefix": str(environment_prefix),
        "expected_environment": str(expected_environment),
        "project_local_venv": local_environment,
        "node": node_status,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument(
        "--expected-environment",
        type=Path,
        help="Override the expected environment path for contract testing.",
    )
    arguments = parser.parse_args()
    report = build_report(arguments.expected_environment)

    if arguments.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Preflight: {report['status']}")
        print(f"Python: {report['python']}")
        print(f"Interpreter: {report['executable']}")
        print(f"Environment: {report['environment_prefix']}")
        print(f"Node lane: {report['node']}")
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)

    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
