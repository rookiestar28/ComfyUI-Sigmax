"""Pinned native ComfyUI deterministic-Euler execution evidence."""

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

pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "krea2_native_euler_parity_v1.json"
REPORT_MODULE = ROOT / "scripts" / "parity" / "krea2_native_euler_report.py"
RUNNER = ROOT / "scripts" / "run_krea2_native_euler_parity.py"


def _report_module() -> ModuleType:
    return importlib.import_module("scripts.parity.krea2_native_euler_report")


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_native_euler_assets_exist() -> None:
    assert importlib.util.find_spec("scripts.parity.krea2_native_euler_report") is not None
    assert REPORT_MODULE.is_file()
    assert RUNNER.is_file()
    assert FIXTURE.is_file()


def test_runner_keeps_comfyui_and_tensor_imports_inside_execution() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", maxsplit=1)[0])
    assert top_level_imports.isdisjoint({"comfy", "numpy", "torch"})


def test_fixture_freezes_native_source_and_execution_semantics() -> None:
    fixture = _fixture()

    assert fixture["schema"] == "sigmax.krea2-native-euler-parity/1"
    assert fixture["status"] == "PASS"
    assert fixture["profile"] == {"id": "krea2.turbo.official", "version": "1"}
    assert fixture["semantics"] == {
        "equation": "x_next=x+(sigma_next-sigma)*flow_velocity",
        "execution": "deterministic",
        "model_output_conversion": "denoised=x-flow_velocity*sigma",
        "noise_ownership": "none",
        "prediction_type": "flow_velocity",
        "sampler": "comfy.euler",
        "sampler_state": [],
        "schedule_ownership": "external_sigmas",
        "terminal": "zero_target_without_model_evaluation",
    }
    assert fixture["source"]["license"] == "GPL-3.0-only"
    assert fixture["source"]["locators"]["euler"] == ("comfy/k_diffusion/sampling.py:190-214")
    assert fixture["source"]["locators"]["flow_conversion"] == ("comfy/model_sampling.py:86-96")


def test_fixture_contains_complete_nontrivial_step_trace() -> None:
    case = cast(dict[str, Any], _fixture()["case"])

    assert case["steps"] == 8
    assert len(case["sigmas"]) == 9
    assert case["sigmas"][0] == 1.0
    assert case["sigmas"][-1] == 0.0
    assert len(case["initial_state"]) == 4
    assert len(case["native_steps"]) == 8
    assert len(case["oracle_states"]) == 9
    assert case["counts"] == {
        "effective_model_evaluations": 8,
        "effective_transitions": 8,
        "requested_model_evaluations": 8,
        "requested_transitions": 8,
    }
    assert case["deterministic_rerun"] is True
    assert case["native_final"] == case["rerun_final"]
    assert case["native_final"] != case["initial_state"]
    for index, step in enumerate(case["native_steps"]):
        assert step["index"] == index
        assert set(step) == {
            "denoised",
            "index",
            "input_state",
            "output_state",
            "sigma",
            "sigma_next",
            "velocity",
        }
        assert len(step["input_state"]) == len(step["velocity"]) == 4
        assert len(step["denoised"]) == len(step["output_state"]) == 4


def test_fixture_is_canonical_and_recomputes_every_step() -> None:
    report = _report_module()
    fixture = _fixture()

    assert report.validate_native_euler_report(fixture) == fixture
    assert report.canonical_json(fixture) == FIXTURE.read_text(encoding="utf-8")

    case = cast(dict[str, Any], fixture["case"])
    errors: list[float] = []
    for step, oracle_output in zip(
        case["native_steps"],
        case["oracle_states"][1:],
        strict=True,
    ):
        errors.extend(
            abs(float(actual) - float(expected))
            for actual, expected in zip(step["output_state"], oracle_output, strict=True)
        )
    assert float(case["max_abs_error"]) == max(errors)
    assert float(case["mean_abs_error"]) == math.fsum(errors) / len(errors)
    assert max(errors) <= float(case["tolerance"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.update(status="NOT_EXECUTED"), "status"),
        (lambda report: report["case"]["native_steps"].pop(), "step"),
        (
            lambda report: report["case"]["native_steps"][0].update(sigma_next=0.9),
            "sigma",
        ),
        (
            lambda report: report["case"]["native_steps"][2]["velocity"].__setitem__(0, 99.0),
            "velocity",
        ),
        (
            lambda report: report["case"]["native_steps"][4]["denoised"].__setitem__(1, 99.0),
            "denoised",
        ),
        (
            lambda report: report["case"]["native_steps"][6]["output_state"].__setitem__(2, 99.0),
            "output",
        ),
        (
            lambda report: report["case"]["counts"].update(effective_model_evaluations=7),
            "count",
        ),
        (lambda report: report["case"].update(deterministic_rerun=False), "deterministic"),
        (lambda report: report["semantics"].update(schedule_ownership="model_native"), "semantics"),
        (lambda report: report["source"].update(license="MIT"), "source"),
    ),
)
def test_report_rejects_tampered_or_incomplete_execution_evidence(
    mutation: Any,
    message: str,
) -> None:
    report = _report_module()
    fixture = copy.deepcopy(_fixture())
    mutation(fixture)

    with pytest.raises(ValueError, match=message):
        report.validate_native_euler_report(fixture)


def test_runner_failure_never_publishes_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = importlib.import_module("scripts.run_krea2_native_euler_parity")
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        runner,
        "_validate_source_root",
        lambda _root: (_ for _ in ()).throw(RuntimeError("source revision drift")),
    )

    result = runner.main(["--comfyui-root", str(tmp_path), "--output", str(output)])

    assert result == 2
    assert "revision" in capsys.readouterr().err
    assert not output.exists()
