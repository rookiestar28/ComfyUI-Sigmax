"""Pinned native ComfyUI MiniMax H3 schedule and adapter report contracts."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any, cast

import pytest
import tomli as tomllib
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_BASE_FL2VA_PROFILE,
    MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT,
    MINIMAX_H3_VIDEO_SHIFT,
    build_minimax_h3_comfyui_simple_schedule,
    map_minimax_h3_audio_coordinate,
)

pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements" / "parity-comfyui-h3-native.txt"
RUNNER = ROOT / "scripts" / "run_minimax_h3_comfy_native_parity.py"
REPORT_MODULE = ROOT / "scripts" / "parity" / "minimax_h3_comfy_native_report.py"
FIXTURE = Path(__file__).with_name("fixtures") / "minimax_h3_comfy_native_parity_v1.json"


def _modules() -> tuple[Any, Any]:
    return (
        importlib.import_module("scripts.run_minimax_h3_comfy_native_parity"),
        importlib.import_module("scripts.parity.minimax_h3_comfy_native_report"),
    )


def _synthetic_inputs(
    report: Any,
) -> tuple[dict[int, tuple[float, ...]], dict[float, tuple[float, float]]]:
    vectors = {
        transitions: tuple(
            build_minimax_h3_comfyui_simple_schedule(
                variant=MINIMAX_H3_BASE_FL2VA_PROFILE.variant,
                transitions=transitions,
            ).sigmas
        )
        for transitions in report.REQUIRED_TRANSITIONS
    }
    mappings = {
        sigma: (
            map_minimax_h3_audio_coordinate(sigma, precision="float32").audio_sigma,
            map_minimax_h3_audio_coordinate(sigma, precision="float32").derivative,
        )
        for sigma in report.REQUIRED_AUDIO_PROBES
    }
    return vectors, mappings


def test_native_h3_assets_exist_and_package_has_no_runtime_dependencies() -> None:
    assert LOCK.is_file()
    assert RUNNER.is_file()
    assert REPORT_MODULE.is_file()
    assert FIXTURE.is_file()
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    assert metadata["project"]["dependencies"] == []


def test_native_h3_dependency_lock_is_exact_and_isolated() -> None:
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


def test_native_h3_runner_defers_optional_imports_until_execution() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", maxsplit=1)[0])
    assert top_level_imports.isdisjoint({"comfy", "numpy", "torch", "torchvision", "transformers"})


def test_native_h3_source_pin_and_adapter_contract() -> None:
    _, report = _modules()
    assert report.COMFYUI_H3_REVISION == MINIMAX_H3_COMFYUI_IMPLEMENTATION_COMMIT
    assert report.REQUIRED_TRANSITIONS == (4, 8, 12, 16, 20)
    assert report.SOURCE_BLOBS["comfy/ldm/minimax/model.py"] == (
        "494350d40b2678812af92c0cba75b5c564b02810"  # pragma: allowlist secret
    )
    assert report.REPORT_SCHEMA == "sigmax.minimax-h3-comfy-native-parity/1"


def test_native_h3_report_builder_proves_schedule_mapping_and_sign() -> None:
    _, report = _modules()
    vectors, mappings = _synthetic_inputs(report)
    built = report.build_native_report(
        vectors,
        native_mappings=mappings,
        environment={"device": "cpu", "numpy": "2.5.1", "python": "3.13", "torch": "2.13.0"},
    )
    assert built["status"] == "PASS"
    assert built["configuration"]["table_length"] == 1000
    assert built["configuration"]["video_shift"] == MINIMAX_H3_VIDEO_SHIFT
    assert built["configuration"]["audio_shift"] == MINIMAX_H3_AUDIO_SHIFT
    assert report.validate_native_report(built) == built
    adapter = cast(dict[str, Any], built["adapter"])
    assert adapter["model_output_sign"] == -1.0
    assert adapter["sign_adapter"] == "explicit_negate_to_data_ward"
    assert len(adapter["coordinate_mapping"]) == len(report.REQUIRED_AUDIO_PROBES)


def test_native_h3_fixture_is_canonical_and_semantically_valid() -> None:
    _, report = _modules()
    fixture = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert report.validate_native_report(fixture) == fixture
    assert report.canonical_json(fixture) == FIXTURE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.update(status="NOT_EXECUTED"), "schema or status"),
        (lambda report: report["cases"].pop(), "cases"),
        (lambda report: report["adapter"].update(model_output_sign=1.0), "sign"),
        (lambda report: report["adapter"]["coordinate_mapping"].pop(), "mapping"),
        (lambda report: report["cases"][0]["native"].pop(), "terminal"),
    ),
)
def test_native_h3_report_validation_fails_closed(mutation: Any, message: str) -> None:
    _, report = _modules()
    fixture = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    mutation(copy.deepcopy(fixture))
    mutated = copy.deepcopy(fixture)
    mutation(mutated)
    with pytest.raises(ValueError, match=message):
        report.validate_native_report(mutated)


def test_native_h3_runner_rejects_missing_source_without_writing_output(tmp_path: Path) -> None:
    runner, _ = _modules()
    output = tmp_path / "should-not-exist.json"
    result = runner.main(["--comfyui-root", str(tmp_path / "missing"), "--output", str(output)])
    assert result == 2
    assert not output.exists()


def test_native_h3_report_canonical_json_is_stable() -> None:
    _, report = _modules()
    vectors, mappings = _synthetic_inputs(report)
    first = report.build_native_report(
        vectors,
        native_mappings=mappings,
        environment={"device": "cpu", "numpy": "2.5.1", "python": "3.13", "torch": "2.13.0"},
    )
    assert json.loads(report.canonical_json(first)) == first
