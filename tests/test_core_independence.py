"""Executable contracts for the framework-independent pure-core boundary."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPOSITORY_ROOT / "comfyui_sigmax" / "core"
OPTIONAL_FRAMEWORKS = ("comfy", "diffusers")


def test_core_source_imports_only_stdlib_or_sigmax() -> None:
    allowed_roots = {*sys.stdlib_module_names, "__future__", "comfyui_sigmax"}
    violations: list[str] = []

    for path in sorted(CORE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for node in ast.walk(tree):
            roots: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                roots = tuple(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = (node.module.split(".", maxsplit=1)[0],)
            for root in roots:
                if root not in allowed_roots:
                    violations.append(f"{path.name}:{getattr(node, 'lineno', 0)}:{root}")

    assert violations == []


def test_clean_core_lane_has_no_optional_framework_installed() -> None:
    resolved = {name: importlib.util.find_spec(name) is not None for name in OPTIONAL_FRAMEWORKS}

    assert resolved == {"comfy": False, "diffusers": False}


def test_independence_probe_is_an_ordered_full_gate_stage() -> None:
    probe = REPOSITORY_ROOT / "scripts" / "check_core_independence.py"
    runner = (REPOSITORY_ROOT / "scripts" / "run_full_gate.py").read_text(encoding="utf-8")

    assert probe.is_file()
    assert '"core-independence"' in runner
    assert runner.index('"mypy"') < runner.index('"core-independence"') < runner.index('"pytest"')


def test_isolated_probe_imports_every_core_module_without_optional_frameworks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "check_core_independence.py"),
            "--json",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["optional_specs"] == {"comfy": False, "diffusers": False}
    assert report["attempted_optional_imports"] == []
    assert report["loaded_optional_modules"] == []
    assert report["modules"] == sorted(
        f"comfyui_sigmax.core.{path.stem}"
        for path in CORE_ROOT.glob("*.py")
        if path.name != "__init__.py"
    )
