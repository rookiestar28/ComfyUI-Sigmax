"""Prove that every pure-core module imports without optional frameworks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Final, cast

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
OPTIONAL_FRAMEWORKS: Final = ("comfy", "diffusers")

ISOLATED_PROBE: Final = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import json
    import pathlib
    import pkgutil
    import sys

    repository_root = pathlib.Path(sys.argv[1])
    blocked_roots = {"comfy", "diffusers"}

    class OptionalFrameworkBlocker(importlib.abc.MetaPathFinder):
        def __init__(self):
            self.attempted = []

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in blocked_roots:
                self.attempted.append(fullname)
                raise ModuleNotFoundError(
                    f"optional framework import is forbidden in pure core: {fullname}"
                )
            return None

    blocker = OptionalFrameworkBlocker()
    sys.meta_path.insert(0, blocker)
    sys.path.insert(0, str(repository_root))

    core = importlib.import_module("comfyui_sigmax.core")
    modules = sorted(
        module.name
        for module in pkgutil.walk_packages(
            core.__path__,
            prefix=f"{core.__name__}.",
        )
    )
    for module in modules:
        importlib.import_module(module)

    loaded_optional = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in blocked_roots
    )
    print(
        json.dumps(
            {
                "attempted_optional_imports": blocker.attempted,
                "loaded_optional_modules": loaded_optional,
                "modules": modules,
            },
            sort_keys=True,
        )
    )
    """
)


def _optional_specs() -> dict[str, bool]:
    return {module: importlib.util.find_spec(module) is not None for module in OPTIONAL_FRAMEWORKS}


def _run_isolated_probe() -> dict[str, object]:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            "-c",
            ISOLATED_PROBE,
            str(REPOSITORY_ROOT),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("isolated core import probe failed")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("isolated core import probe returned invalid JSON") from error
    if result.stderr or not isinstance(report, dict):
        raise RuntimeError("isolated core import probe returned an invalid result")
    return report


def _build_report() -> dict[str, object]:
    optional_specs = _optional_specs()
    if any(optional_specs.values()):
        raise RuntimeError(
            "core independence must run in a clean dev environment without optional frameworks"
        )

    probe = _run_isolated_probe()
    attempted = probe.get("attempted_optional_imports")
    loaded = probe.get("loaded_optional_modules")
    modules = probe.get("modules")
    if attempted != [] or loaded != []:
        raise RuntimeError("pure core attempted to load an optional framework")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("isolated core import probe did not enumerate modules")

    return {
        "attempted_optional_imports": attempted,
        "loaded_optional_modules": loaded,
        "modules": modules,
        "optional_specs": optional_specs,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    try:
        report = _build_report()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"CORE_INDEPENDENCE=FAIL: {error}", file=sys.stderr)
        return 1

    if arguments.json:
        print(json.dumps(report, sort_keys=True))
    else:
        modules = cast(list[object], report["modules"])
        print("CORE_INDEPENDENCE=PASS")
        print(f"Core modules: {len(modules)}")
        print("Optional frameworks installed: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
