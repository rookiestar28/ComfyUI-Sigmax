"""Native ComfyUI Krea 2 Turbo schedule-parity evidence."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import tomli as tomllib
from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles import build_krea2_turbo_schedule

pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "krea2_turbo_comfy_native_parity_v1.json"
REPORT_MODULE = ROOT / "scripts" / "parity" / "krea2_comfy_native_report.py"
RUNNER = ROOT / "scripts" / "run_krea2_comfy_native_parity.py"
LOCK = ROOT / "requirements" / "parity-comfyui-native.txt"

EXPECTED_STEPS = (4, 8, 12, 16)
COMFYUI_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
SOURCE_BLOBS = {
    "comfy/model_base.py": (
        "ee6dc57a25b5d071623a81b5f82703a2e5d5c6b6"  # pragma: allowlist secret
    ),
    "comfy/model_sampling.py": (
        "5af336e76fd480a50425dd924f2ac9752083c09f"  # pragma: allowlist secret
    ),
    "comfy/samplers.py": (
        "9f571ece9ab7d79e35a8c3437a92fd93ccce0c09"  # pragma: allowlist secret
    ),
    "comfy/supported_models.py": (
        "ca89850a594fba87ff669dbd94b027efc1ad79dc"  # pragma: allowlist secret
    ),
}


def _git_chunks(value: str) -> list[str]:
    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _load_fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _report_module() -> ModuleType:
    return importlib.import_module("scripts.parity.krea2_comfy_native_report")


def test_native_parity_assets_exist() -> None:
    assert importlib.util.find_spec("scripts.parity.krea2_comfy_native_report") is not None
    assert FIXTURE.is_file()
    assert RUNNER.is_file()
    assert LOCK.is_file()


def test_native_dependency_lock_is_exact_and_isolated() -> None:
    lines = [
        line
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert lines == [
        "comfy-aimdo==0.4.10",
        "comfy-kitchen==0.2.23",
        "einops==0.8.2",
        "numpy==2.5.1",
        "packaging==26.2",
        "pillow==12.3.0",
        "psutil==7.2.2",
        "safetensors==0.8.0",
        "scipy==1.18.0",
        "sentencepiece==0.2.2",
        "torch==2.13.0",
        "torchsde==0.2.6",
        "torchvision==0.28.0",
        "tqdm==4.70.0",
        "transformers==5.14.1",
    ]
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    assert metadata["project"]["dependencies"] == []


def test_runner_defers_comfyui_and_tensor_imports_until_execution() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", maxsplit=1)[0])
    assert top_level_imports.isdisjoint({"comfy", "numpy", "torch", "torchvision", "transformers"})


def test_fixture_has_exact_native_source_and_execution_metadata() -> None:
    fixture = _load_fixture()

    assert fixture["schema"] == "sigmax.krea2-comfy-native-parity/1"
    assert fixture["status"] == "PASS"
    assert fixture["profile"] == {"id": "krea2.turbo.official", "version": "1"}
    assert fixture["configuration"] == {
        "model_sampling": "ModelSamplingFlux",
        "mu": "1.15",
        "scheduler": "simple",
        "table_length": 10000,
        "terminal": "zero",
    }
    assert fixture["environment"] == {
        "device": "cpu",
        "dtype": "float32",
        "numpy": "2.5.1",
        "python": "3.13",
        "torch": "2.13.0",
    }
    assert fixture["source"]["revision_chunks"] == _git_chunks(COMFYUI_REVISION)
    assert fixture["source"]["url"] == "https://github.com/Comfy-Org/ComfyUI"
    for path, blob in SOURCE_BLOBS.items():
        assert fixture["source"]["blobs"][path] == _git_chunks(blob)


def test_fixture_contains_complete_native_vectors_and_bounded_policies() -> None:
    fixture = _load_fixture()

    assert tuple(case["steps"] for case in fixture["cases"]) == EXPECTED_STEPS
    for case in fixture["cases"]:
        steps = case["steps"]
        exact = steps in (4, 8, 16)
        assert case["evidence"] == ("official" if steps == 8 else "modified")
        assert case["exact_table_positions"] is exact
        assert case["difference_reason"] == (
            "float32_evaluation" if exact else "simple_scheduler_integer_index_quantization"
        )
        assert case["tolerance"] == ("1e-06" if exact else "0.0002")
        assert len(case["native"]) == steps + 1
        assert len(case["sigmax"]) == steps + 1
        assert case["native"][0] == case["sigmax"][0] == 1.0
        assert case["native"][-1] == case["sigmax"][-1] == 0.0


def test_fixture_is_canonical_and_semantically_valid() -> None:
    report = _report_module()
    fixture = _load_fixture()

    assert report.validate_native_report(fixture) == fixture
    assert report.canonical_json(fixture) == FIXTURE.read_text(encoding="utf-8")


def test_fixture_matches_current_sigmax_profile() -> None:
    fixture = _load_fixture()

    for case in fixture["cases"]:
        sigmax = tuple(case["sigmax"])
        expected = build_krea2_turbo_schedule(steps=case["steps"]).sigmas
        assert sigmax == pytest.approx(expected, abs=6e-8)
        assert case["sigmax_fingerprint"] == numerical_fingerprint(
            sigmax,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float32",
        )
        assert case["native_fingerprint"] == numerical_fingerprint(
            tuple(case["native"]),
            domain=SigmaDomain.UNIT_FLOW,
            precision="float32",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.update(status="NOT_EXECUTED"), "status"),
        (lambda report: report["cases"].pop(), "case"),
        (lambda report: report["cases"][0]["native"].pop(), "length"),
        (lambda report: report["cases"][0].update(max_abs_error="0.5"), "error"),
        (lambda report: report["cases"][1].update(evidence="modified"), "evidence"),
        (lambda report: report["source"].update(revision_chunks=["main"]), "source"),
        (lambda report: report["configuration"].update(table_length=9999), "configuration"),
    ),
)
def test_native_report_validation_fails_closed(mutation: Any, message: str) -> None:
    report = _report_module()
    fixture = copy.deepcopy(_load_fixture())
    mutation(fixture)

    with pytest.raises(ValueError, match=message):
        report.validate_native_report(fixture)


def test_native_report_recomputes_errors_from_complete_vectors() -> None:
    fixture = _load_fixture()

    for case in fixture["cases"]:
        errors = [
            abs(actual - expected)
            for actual, expected in zip(case["native"], case["sigmax"], strict=True)
        ]
        assert float(case["max_abs_error"]) == max(errors)
        assert float(case["mean_abs_error"]) == math.fsum(errors) / len(errors)
        assert max(errors) <= float(case["tolerance"])


def test_runner_rejects_missing_source_without_writing_output(tmp_path: Path) -> None:
    runner = importlib.import_module("scripts.run_krea2_comfy_native_parity")
    output = tmp_path / "should-not-exist.json"

    result = runner.main(["--comfyui-root", str(tmp_path / "missing"), "--output", str(output)])

    assert result == 2
    assert not output.exists()


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        (RuntimeError("ComfyUI source revision does not match"), "revision"),
        (RuntimeError("ComfyUI source worktree must be clean"), "clean"),
        (RuntimeError("missing native parity dependency"), "dependency"),
    ),
)
def test_runner_failures_never_publish_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: RuntimeError,
    message: str,
) -> None:
    runner = importlib.import_module("scripts.run_krea2_comfy_native_parity")
    output = tmp_path / "should-not-exist.json"
    monkeypatch.setattr(
        runner, "_validate_source_root", lambda _root: (_ for _ in ()).throw(failure)
    )

    result = runner.main(["--comfyui-root", str(tmp_path), "--output", str(output)])

    captured = capsys.readouterr()
    assert result == 2
    assert message in captured.err
    assert not output.exists()
