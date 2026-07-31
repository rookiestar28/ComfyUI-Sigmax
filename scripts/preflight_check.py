"""Validate the local interpreter and foundation-gate assumptions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import tomli

# IMPORTANT: keep this adjacent import; direct script entrypoints put scripts/ on sys.path.
from environment_diagnostics import (  # type: ignore[import-not-found]
    EnvironmentProbeConfig,
    collect_environment_observation,
    evaluate_environment,
)


class PreflightReport(TypedDict):
    status: str
    python: str
    executable: str
    environment_prefix: str
    expected_environment: str
    project_local_venv: bool
    node: str
    diagnostics: dict[str, object]
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


def build_report(
    expected_environment_override: Path | None = None,
    *,
    optional_lane: Literal["none", "plot", "reference"] = "none",
) -> PreflightReport:
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
    platform = "windows" if os.name == "nt" else "linux"
    suffix = "win" if os.name == "nt" else "linux"
    cache_root = Path(
        os.environ.get(
            "PRE_COMMIT_HOME",
            REPOSITORY_ROOT / ".tmp" / f"pre-commit-{suffix}",
        )
    )
    temp_root = Path(
        os.environ.get(
            "SIGMAX_TEMP_ROOT",
            REPOSITORY_ROOT / ".tmp" / f"runtime-{suffix}",
        )
    )
    diagnostics = evaluate_environment(
        collect_environment_observation(
            EnvironmentProbeConfig(
                repository_root=REPOSITORY_ROOT,
                environment_prefix=environment_prefix,
                expected_environment=expected_environment,
                cache_root=cache_root,
                temp_root=temp_root,
                platform=platform,
                optional_lane=optional_lane,
                path_value=os.environ.get("PATH", ""),
                temp_capture_mitigated=(os.environ.get("SIGMAX_PYTEST_CAPTURE_MODE") == "sys"),
            )
        )
    )
    errors.extend(
        f"[{issue.code}] {issue.summary} Remediation: {issue.remediation}"
        for issue in diagnostics.issues
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
        "diagnostics": diagnostics.projection(),
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
    parser.add_argument(
        "--optional-lane",
        choices=("none", "plot", "reference"),
        default="none",
        help="Require one fixed optional dependency lane.",
    )
    arguments = parser.parse_args()
    report = build_report(
        arguments.expected_environment,
        optional_lane=arguments.optional_lane,
    )

    if arguments.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Preflight: {report['status']}")
        print(f"Python: {report['python']}")
        print(f"Interpreter: {report['executable']}")
        print(f"Environment: {report['environment_prefix']}")
        print(f"Node lane: {report['node']}")
        mitigations = report["diagnostics"].get("mitigations")
        if isinstance(mitigations, list):
            for mitigation in mitigations:
                if isinstance(mitigation, str):
                    print(f"Environment mitigation: {mitigation}")
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)

    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
