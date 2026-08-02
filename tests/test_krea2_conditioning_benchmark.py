from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_krea2_conditioning_benchmark.py"


def test_conditioning_benchmark_script_is_present_and_torch_lazy() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    imported_modules = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported_modules
    assert "sigmax.krea2-conditioning-performance/1" in SCRIPT.read_text(encoding="utf-8")


def test_conditioning_benchmark_rejects_output_outside_repository() -> None:
    from scripts import run_krea2_conditioning_benchmark as benchmark

    with pytest.raises(ValueError, match="inside the repository"):
        benchmark._output_path("../outside-evidence.json")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Torch unavailable")
def test_conditioning_benchmark_evidence_passes_on_torch_host() -> None:
    from scripts import run_krea2_conditioning_benchmark as benchmark

    evidence = benchmark.build_evidence()
    assert evidence["schema"] == "sigmax.krea2-conditioning-performance/1"
    assert evidence["verdict"] == "pass"
    diagnostics = cast(dict[str, object], evidence["diagnostics"])
    assert cast(float, diagnostics["input_rms"]) > 0.0
