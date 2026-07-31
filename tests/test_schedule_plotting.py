"""Optional, lazy schedule-report plotting boundary."""

from __future__ import annotations

import builtins
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from comfyui_sigmax.core import ScheduleContractError

ROOT = Path(__file__).resolve().parents[1]


def test_plotting_module_import_is_lazy_without_matplotlib() -> None:
    script = (
        "import builtins,sys; real=builtins.__import__; "
        "builtins.__import__=lambda n,*a,**k: "
        "(_ for _ in ()).throw(ImportError(n)) if n.split('.')[0]=='matplotlib' "
        "else real(n,*a,**k); "
        "import comfyui_sigmax.plotting; "
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", script], check=True, cwd=ROOT)


def test_missing_plot_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from comfyui_sigmax import plotting
    from comfyui_sigmax.core import build_schedule_report
    from tests.test_schedule_reports import _artifact

    real_import = builtins.__import__

    def blocked(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name.split(".", 1)[0] == "matplotlib":
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(plotting.SchedulePlotDependencyError, match=r"\[plot\]"):
        plotting.render_schedule_plot(build_schedule_report(_artifact()), image_format="png")


@pytest.mark.parametrize("image_format", ("jpg", "", 1, None))
def test_plot_renderer_rejects_unsupported_formats_before_import(
    image_format: object,
) -> None:
    from comfyui_sigmax import plotting
    from comfyui_sigmax.core import build_schedule_report
    from tests.test_schedule_reports import _artifact

    with pytest.raises(ScheduleContractError, match="format"):
        plotting.render_schedule_plot(
            build_schedule_report(_artifact()),
            image_format=image_format,  # type: ignore[arg-type]
        )


def test_noncomparable_plot_rejects_before_optional_import() -> None:
    from comfyui_sigmax import plotting
    from comfyui_sigmax.core import (
        build_schedule_comparison_report,
        build_schedule_report,
    )
    from tests.test_schedule_reports import _artifact

    comparison = build_schedule_comparison_report(
        build_schedule_report(_artifact()),
        build_schedule_report(_artifact((1.0, 0.5, 0.0))),
    )
    with pytest.raises(ScheduleContractError, match="comparable"):
        plotting.render_schedule_comparison_plot(comparison)


def test_plot_extra_is_declared_but_not_mandatory() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "dependencies = []" in pyproject
    assert 'plot = ["matplotlib>=3.10,<3.12"]' in pyproject


def test_plot_validation_script_uses_only_public_runtime_dependencies() -> None:
    script = (ROOT / "scripts" / "validate_schedule_plots.py").read_text(encoding="utf-8")

    assert "from tests" not in script
    assert "import pytest" not in script
    assert "build_validation_summary" in script
