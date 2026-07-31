"""Authoritative Krea 2 Turbo formula and Diffusers parity evidence."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import math
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import tomli as tomllib
from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles import KREA2_TURBO_PROFILE, build_krea2_turbo_schedule

pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "krea2_turbo_parity_v1.json"
OFFICIAL_ADAPTER = ROOT / "scripts" / "parity" / "krea2_official.py"
RUNNER = ROOT / "scripts" / "run_krea2_turbo_parity.py"
LOCK = ROOT / "requirements" / "parity-krea2-turbo.txt"

EXPECTED_STEPS = (4, 8, 12, 16)
KREA_REVISION = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret
DIFFUSERS_REVISION = "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret
DIFFUSERS_PIPELINE_BLOB = "51d33cb4861903ee1ac682f8da3b7256013656aa"  # pragma: allowlist secret
DIFFUSERS_SCHEDULER_BLOB = "7b207f7820797c53b093452ca2bc52938a8d84e7"  # pragma: allowlist secret


def _git_object_chunks(value: str) -> list[str]:
    return [value[index : index + 8] for index in range(0, len(value), 8)]


def _modules() -> tuple[ModuleType, ModuleType]:
    official = importlib.import_module("scripts.parity.krea2_official")
    report = importlib.import_module("scripts.parity.krea2_turbo_report")
    return official, report


def _load_fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_parity_modules_and_tracked_fixture_exist() -> None:
    official_spec = importlib.util.find_spec("scripts.parity.krea2_official")
    report_spec = importlib.util.find_spec("scripts.parity.krea2_turbo_report")

    assert official_spec is not None
    assert report_spec is not None
    assert FIXTURE.is_file()
    assert RUNNER.is_file()
    assert LOCK.is_file()


def test_official_adapter_is_independent_and_pinned() -> None:
    official, _ = _modules()
    tree = ast.parse(OFFICIAL_ADAPTER.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots.isdisjoint({"comfyui_sigmax", "comfy", "diffusers", "numpy", "torch"})
    assert official.KREA_REVISION == KREA_REVISION
    assert official.KREA_LOCATOR == "sampling.py:40-53"
    assert official.KREA_SOURCE_URL == "https://github.com/krea-ai/krea-2"


@pytest.mark.parametrize("steps", EXPECTED_STEPS)
def test_official_adapter_returns_complete_monotonic_vectors(steps: int) -> None:
    official, _ = _modules()

    vector = official.official_krea2_turbo_sigmas(steps)

    assert len(vector) == steps + 1
    assert vector[0] == 1.0
    assert vector[-1] == 0.0
    assert all(current > following for current, following in pairwise(vector))


@pytest.mark.parametrize("invalid_steps", (True, 0, -1, 1.5, "8"))
def test_official_adapter_rejects_invalid_steps(invalid_steps: object) -> None:
    official, _ = _modules()

    with pytest.raises(ValueError, match="positive integer"):
        official.official_krea2_turbo_sigmas(invalid_steps)


def test_parity_dependency_lock_and_profile_reference_are_exact() -> None:
    lines = [
        line
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert lines == [
        "diffusers==0.39.0",
        "numpy==2.3.4",
        "torch==2.9.0",
    ]
    framework = next(
        reference
        for reference in KREA2_TURBO_PROFILE.references
        if reference.source_id == "diffusers.krea2.framework"
    )
    assert framework.revision == DIFFUSERS_REVISION
    assert framework.locators == (
        "src/diffusers/pipelines/krea2/pipeline_krea2.py",
        "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
    )


def test_default_package_does_not_depend_on_parity_frameworks() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    assert metadata["project"]["dependencies"] == []
    assert metadata["project"]["optional-dependencies"]["reference"] == ["diffusers>=0.39,<0.40"]
    assert set(metadata["tool"]["setuptools"]["packages"]) == {
        "comfyui_sigmax",
        "comfyui_sigmax.adapters",
        "comfyui_sigmax.benchmarks",
        "comfyui_sigmax.coinstallation",
        "comfyui_sigmax.compatibility",
        "comfyui_sigmax.core",
        "comfyui_sigmax.nodes",
        "comfyui_sigmax.performance",
        "comfyui_sigmax.profiles",
        "comfyui_sigmax.workflows",
    }


def test_runner_defers_optional_imports_until_execution() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", maxsplit=1)[0])

    assert top_level_imports.isdisjoint({"diffusers", "numpy", "torch"})


def test_fixture_schema_sources_versions_and_configuration_are_exact() -> None:
    fixture = _load_fixture()

    assert fixture["schema"] == "sigmax.krea2-turbo-parity/1"
    assert fixture["status"] == "PASS"
    assert fixture["profile"] == {"id": "krea2.turbo.official", "version": "1"}
    assert fixture["configuration"] == {
        "base_grid": "krea.reciprocal_step",
        "mu": "1.15",
        "terminal": "zero",
    }
    assert fixture["environment"] == {
        "device": "cpu",
        "diffusers": "0.39.0",
        "numpy": "2.3.4",
        "torch": "2.9.0",
    }
    assert fixture["sources"]["krea"] == {
        "evidence": "official",
        "locator": "sampling.py:40-53",
        "revision_chunks": _git_object_chunks(KREA_REVISION),
        "url": "https://github.com/krea-ai/krea-2",
    }
    assert fixture["sources"]["diffusers"] == {
        "evidence": "framework_reference",
        "pipeline_blob_chunks": _git_object_chunks(DIFFUSERS_PIPELINE_BLOB),
        "pipeline_locator": "src/diffusers/pipelines/krea2/pipeline_krea2.py:613-630",
        "revision_chunks": _git_object_chunks(DIFFUSERS_REVISION),
        "scheduler_blob_chunks": _git_object_chunks(DIFFUSERS_SCHEDULER_BLOB),
        "scheduler_locator": (
            "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:283-378"
        ),
        "tag": "v0.39.0",
        "url": "https://github.com/huggingface/diffusers",
    }
    assert fixture["tolerances"] == {
        "diffusers_float32_max_abs": "1e-6",
        "krea_float64_max_abs": "1e-8",
    }


def test_fixture_contains_every_complete_parity_case() -> None:
    fixture = _load_fixture()

    assert tuple(case["steps"] for case in fixture["cases"]) == EXPECTED_STEPS
    for case in fixture["cases"]:
        steps = case["steps"]
        assert case["evidence"] == ("official" if steps == 8 else "modified")
        for comparison_name in ("krea_float64", "diffusers_float32"):
            comparison = case["comparisons"][comparison_name]
            assert len(comparison["reference"]) == steps + 1
            assert len(comparison["sigmax"]) == steps + 1
            assert comparison["reference"][0] == 1.0
            assert comparison["reference"][-1] == 0.0
            assert comparison["sigmax"][0] == 1.0
            assert comparison["sigmax"][-1] == 0.0
            assert comparison["device"] == "cpu"
            assert comparison["status"] == "PASS"


def test_report_fixture_is_canonical_and_semantically_valid() -> None:
    _, report = _modules()
    fixture = _load_fixture()

    validated = report.validate_parity_report(fixture)

    assert validated == fixture
    assert report.canonical_json(fixture) == FIXTURE.read_text(encoding="utf-8")


def test_report_builder_accepts_complete_reference_vectors() -> None:
    official, report = _modules()
    vectors = {steps: official.official_krea2_turbo_sigmas(steps) for steps in EXPECTED_STEPS}

    built = report.build_parity_report(
        vectors,
        environment={
            "device": "cpu",
            "diffusers": "0.39.0",
            "numpy": "2.3.4",
            "torch": "2.9.0",
        },
    )

    assert built["status"] == "PASS"


def test_report_fixture_matches_current_sigmax_and_official_adapter() -> None:
    official, _ = _modules()
    fixture = _load_fixture()

    for case in fixture["cases"]:
        steps = case["steps"]
        sigmax = build_krea2_turbo_schedule(steps=steps).sigmas
        expected_official = official.official_krea2_turbo_sigmas(steps)
        official_comparison = case["comparisons"]["krea_float64"]
        diffusers_comparison = case["comparisons"]["diffusers_float32"]

        assert tuple(official_comparison["reference"]) == expected_official
        assert tuple(official_comparison["sigmax"]) == sigmax
        assert official_comparison["fingerprint"] == numerical_fingerprint(
            sigmax,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float64",
        )
        assert diffusers_comparison["fingerprint"] == numerical_fingerprint(
            tuple(diffusers_comparison["sigmax"]),
            domain=SigmaDomain.UNIT_FLOW,
            precision="float32",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.update(status="NOT_EXECUTED"), "status"),
        (lambda report: report["cases"].pop(), "case"),
        (
            lambda report: report["cases"][0]["comparisons"]["krea_float64"]["reference"].pop(),
            "length",
        ),
        (
            lambda report: report["cases"][0]["comparisons"]["krea_float64"].update(
                max_abs_error="0"
            ),
            "error",
        ),
        (
            lambda report: report["cases"][0]["comparisons"]["diffusers_float32"].update(
                status="FAIL"
            ),
            "status",
        ),
        (
            lambda report: report["sources"]["krea"].update(revision_chunks=["main"]),
            "source",
        ),
    ),
)
def test_report_validation_fails_closed(
    mutation: Any,
    message: str,
) -> None:
    _, report = _modules()
    fixture = copy.deepcopy(_load_fixture())
    mutation(fixture)

    with pytest.raises(ValueError, match=message):
        report.validate_parity_report(fixture)


def test_reported_errors_are_recomputed_from_complete_vectors() -> None:
    fixture = _load_fixture()

    for case in fixture["cases"]:
        for comparison in case["comparisons"].values():
            errors = [
                abs(actual - expected)
                for actual, expected in zip(
                    comparison["sigmax"],
                    comparison["reference"],
                    strict=True,
                )
            ]
            expected_maximum = max(errors)
            expected_mean = math.fsum(errors) / len(errors)
            assert float(comparison["max_abs_error"]) == expected_maximum
            assert float(comparison["mean_abs_error"]) == expected_mean
            assert expected_maximum <= float(comparison["tolerance"])
