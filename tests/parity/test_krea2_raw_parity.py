"""Pinned official and executable Diffusers parity for Krea 2 RAW."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles import (
    KREA2_RAW_DIFFUSERS_REFERENCE_28,
    KREA2_RAW_OFFICIAL_FULL_52,
    build_krea2_raw_schedule,
)
from scripts.parity.krea2_raw_report import (
    CASE_SPECS,
    build_parity_report,
    canonical_json,
    validate_parity_report,
)
from scripts.parity.krea2_turbo_report import (
    DIFFUSERS_VERSION,
    NUMPY_VERSION,
    TORCH_VERSION,
)

pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "krea2_raw_parity_v1.json"
OFFICIAL_ADAPTER = ROOT / "scripts" / "parity" / "krea2_official.py"
RUNNER = ROOT / "scripts" / "run_krea2_raw_parity.py"
LOCK = ROOT / "requirements" / "parity-krea2-turbo.txt"
KREA_REVISION = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret
DIFFUSERS_REVISION = "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def _maximum_error(actual: tuple[float, ...], expected: list[float]) -> float:
    return max(abs(left - right) for left, right in zip(actual, expected, strict=True))


def test_raw_parity_artifacts_exist() -> None:
    assert importlib.util.find_spec("scripts.parity.krea2_raw_report") is not None
    assert FIXTURE.is_file()
    assert RUNNER.is_file()
    assert LOCK.is_file()


def test_official_adapter_is_independent_and_exposes_raw_formula() -> None:
    tree = ast.parse(OFFICIAL_ADAPTER.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", maxsplit=1)[0])
    official = importlib.import_module("scripts.parity.krea2_official")

    assert roots.isdisjoint({"comfyui_sigmax", "comfy", "diffusers", "numpy", "torch"})
    assert official.KREA_REVISION == KREA_REVISION
    assert callable(official.official_krea2_raw_case)


@pytest.mark.parametrize("spec", CASE_SPECS)
def test_official_adapter_returns_complete_raw_cases(spec: tuple[str, int, int, int]) -> None:
    official = importlib.import_module("scripts.parity.krea2_official")
    _, steps, width, height = spec

    case = official.official_krea2_raw_case(width=width, height=height, steps=steps)

    assert case["effective_width"] % 16 == 0
    assert case["effective_height"] % 16 == 0
    assert len(case["sigmas"]) == steps + 1
    assert case["sigmas"][0] == 1.0 and case["sigmas"][-1] == 0.0
    assert all(left > right for left, right in pairwise(case["sigmas"]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": True, "height": 1024, "steps": 28},
        {"width": 0, "height": 1024, "steps": 28},
        {"width": 1024, "height": "1024", "steps": 28},
        {"width": 1024, "height": 1024, "steps": 0},
    ],
)
def test_official_adapter_rejects_invalid_raw_inputs(kwargs: dict[str, object]) -> None:
    official = importlib.import_module("scripts.parity.krea2_official")
    with pytest.raises(ValueError):
        official.official_krea2_raw_case(**kwargs)


def test_fixture_contract_and_complete_matrix() -> None:
    fixture = _fixture()

    assert fixture["schema"] == "sigmax.krea2-raw-parity/1"
    assert fixture["status"] == "PASS"
    assert fixture["profile"] == {"id": "krea2.raw.official", "version": "1"}
    assert fixture["environment"] == {
        "device": "cpu",
        "diffusers": DIFFUSERS_VERSION,
        "numpy": NUMPY_VERSION,
        "torch": TORCH_VERSION,
    }
    assert len(fixture["cases"]) == 14
    assert (
        tuple(
            (case["recipe_id"], case["steps"], case["requested_width"], case["requested_height"])
            for case in fixture["cases"]
        )
        == CASE_SPECS
    )
    for case in fixture["cases"]:
        for name in ("krea_float64", "diffusers_float32"):
            comparison = case["comparisons"][name]
            assert len(comparison["reference"]) == case["steps"] + 1
            assert len(comparison["sigmax"]) == case["steps"] + 1
            assert comparison["status"] == "PASS"


def test_fixture_is_canonical_and_matches_current_product_and_official_adapter() -> None:
    official = importlib.import_module("scripts.parity.krea2_official")
    fixture = _fixture()

    assert validate_parity_report(fixture) == fixture
    assert canonical_json(fixture) == FIXTURE.read_text(encoding="utf-8")
    recipes = {
        KREA2_RAW_DIFFUSERS_REFERENCE_28.recipe_id: KREA2_RAW_DIFFUSERS_REFERENCE_28,
        KREA2_RAW_OFFICIAL_FULL_52.recipe_id: KREA2_RAW_OFFICIAL_FULL_52,
    }
    for case in fixture["cases"]:
        product = build_krea2_raw_schedule(
            width=case["requested_width"],
            height=case["requested_height"],
            recipe=recipes[case["recipe_id"]],
        )
        official_case = official.official_krea2_raw_case(
            width=case["requested_width"],
            height=case["requested_height"],
            steps=case["steps"],
        )
        stored_sigmax = case["comparisons"]["krea_float64"]["sigmax"]
        stored_reference = case["comparisons"]["krea_float64"]["reference"]
        assert _maximum_error(product.sigmas, stored_sigmax) <= 1e-8
        assert _maximum_error(official_case["sigmas"], stored_reference) <= 1e-8
        assert case["comparisons"]["krea_float64"]["fingerprint"] == numerical_fingerprint(
            tuple(stored_sigmax),
            domain=SigmaDomain.UNIT_FLOW,
            precision="float64",
        )


def test_report_builder_reproduces_fixture_reference_vectors() -> None:
    fixture = _fixture()
    vectors = {
        case["case_id"]: {
            "mu": case["diffusers_mu"],
            "sigmas": case["comparisons"]["diffusers_float32"]["reference"],
        }
        for case in fixture["cases"]
    }

    report = build_parity_report(
        vectors,
        environment={
            "device": "cpu",
            "diffusers": DIFFUSERS_VERSION,
            "numpy": NUMPY_VERSION,
            "torch": TORCH_VERSION,
        },
    )

    assert report == fixture


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(status="FAIL"),
        lambda report: report["cases"].pop(),
        lambda report: report["cases"][0].update(mu="0"),
        lambda report: report["cases"][0]["comparisons"]["krea_float64"]["reference"].pop(),
        lambda report: report["cases"][0]["comparisons"]["diffusers_float32"].update(
            max_abs_error="0"
        ),
        lambda report: report["sources"]["krea"].update(revision_chunks=["main"]),
    ],
)
def test_report_validator_rejects_mutations(mutation: Any) -> None:
    report = copy.deepcopy(_fixture())
    mutation(report)
    with pytest.raises(ValueError):
        validate_parity_report(report)


def test_runner_defers_optional_imports_and_fails_without_pinned_environment(
    tmp_path: Path,
) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)
    roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    output = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_krea2_raw_parity", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert roots.isdisjoint({"diffusers", "numpy", "torch"})
    assert completed.returncode == 2
    assert "PARITY=FAIL" in completed.stderr
    assert not output.exists()


def test_ci_contains_raw_parity_regeneration_lane() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "raw-parity-pinned:" in workflow
    assert "scripts.run_krea2_raw_parity" in workflow
    assert "krea2_raw_parity_v1.json" in workflow
