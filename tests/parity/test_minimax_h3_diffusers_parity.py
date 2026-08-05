"""Isolated MiniMax H3 Diffusers parity runner and report contracts."""

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
    MINIMAX_H3_DIFFUSERS_REVISION,
    MINIMAX_H3_VIDEO_SHIFT,
)

pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements" / "parity-minimax-h3-diffusers.txt"
RUNNER = ROOT / "scripts" / "run_minimax_h3_diffusers_parity.py"
REPORT_MODULE = ROOT / "scripts" / "parity" / "minimax_h3_diffusers_report.py"
OFFICIAL_ADAPTER = ROOT / "scripts" / "parity" / "minimax_h3_official.py"


def _modules() -> tuple[Any, Any, Any]:
    runner = importlib.import_module("scripts.run_minimax_h3_diffusers_parity")
    report = importlib.import_module("scripts.parity.minimax_h3_diffusers_report")
    official = importlib.import_module("scripts.parity.minimax_h3_official")
    return runner, report, official


def test_h3_parity_assets_exist_and_package_has_no_runtime_dependencies() -> None:
    assert LOCK.is_file()
    assert RUNNER.is_file()
    assert REPORT_MODULE.is_file()
    assert OFFICIAL_ADAPTER.is_file()
    with (ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    assert metadata["project"]["dependencies"] == []


def test_h3_parity_lock_is_exact_and_pinned_to_the_unreleased_branch() -> None:
    lines = [
        line
        for line in LOCK.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert lines == [
        f"diffusers @ git+https://github.com/huggingface/diffusers.git@{MINIMAX_H3_DIFFUSERS_REVISION}",
        "numpy==2.3.4",
        "torch==2.9.0",
    ]


def test_h3_runner_defers_optional_imports_until_execution() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".", maxsplit=1)[0])
    assert top_level_imports.isdisjoint({"diffusers", "numpy", "torch"})


def test_h3_official_clean_room_adapter_is_source_pinned_and_independent() -> None:
    _, _, official = _modules()
    tree = ast.parse(OFFICIAL_ADAPTER.read_text(encoding="utf-8"), filename=OFFICIAL_ADAPTER.name)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    assert imported_roots.isdisjoint({"comfyui_sigmax", "comfy", "diffusers", "numpy", "torch"})
    assert official.DIFFUSERS_REVISION == MINIMAX_H3_DIFFUSERS_REVISION
    assert official.SOURCE_URL == "https://github.com/huggingface/diffusers"
    assert official.SCHEDULER_LOCATOR == "src/diffusers/schedulers/scheduling_minimax_h3.py"


def test_h3_clean_room_vectors_cover_video_audio_and_dataward_step() -> None:
    _, _, official = _modules()
    video = official.clean_room_sigma_grid(4, MINIMAX_H3_VIDEO_SHIFT)
    audio = official.clean_room_sigma_grid(4, MINIMAX_H3_AUDIO_SHIFT)
    assert video == pytest.approx((1.0, 0.96, 0.8571428060531616, 0.0), abs=1e-7)
    assert audio == pytest.approx((1.0, 0.8571428060531616, 0.5999999642372131, 0.0), abs=1e-7)
    expected = official.clean_room_dataward_step(
        sample=(0.25, -0.5),
        velocity=(0.125, -0.25),
        timestep=0.0,
        sigma=video[0],
        sigma_next=video[1],
    )
    assert expected == pytest.approx((0.2549999952316284, -0.5099999904632568), abs=1e-7)


def test_h3_report_builder_and_validator_are_fail_closed() -> None:
    _, report, official = _modules()
    assert report.REQUIRED_GRID_POINTS == (4, 8, 12, 16, 20)
    cases = {
        points: {
            "video": official.clean_room_sigma_grid(points, MINIMAX_H3_VIDEO_SHIFT),
            "audio": official.clean_room_sigma_grid(points, MINIMAX_H3_AUDIO_SHIFT),
        }
        for points in report.REQUIRED_GRID_POINTS
    }
    probe = {
        "sample": (0.25, -0.5),
        "velocity": (0.125, -0.25),
        "timestep": 0.0,
        "sigma": cases[4]["video"][0],
        "sigma_next": cases[4]["video"][1],
        "reference": official.clean_room_dataward_step(
            sample=(0.25, -0.5),
            velocity=(0.125, -0.25),
            timestep=0.0,
            sigma=cases[4]["video"][0],
            sigma_next=cases[4]["video"][1],
        ),
    }
    built = report.build_parity_report(
        cases,
        velocity_reference=probe,
        environment={
            "device": "cpu",
            "diffusers": "0.39.0.dev0",
            "diffusers_revision": MINIMAX_H3_DIFFUSERS_REVISION,
            "numpy": "2.3.4",
            "python": "3.13",
            "torch": "2.9.0",
        },
    )
    assert built["status"] == "PASS"
    assert report.validate_parity_report(built) == built

    mutated = copy.deepcopy(built)
    cast(dict[str, Any], mutated["cases"][0])["video"]["reference"][1] = 0.5
    with pytest.raises(ValueError, match=r"error|fingerprint|monotonic|strictly"):
        report.validate_parity_report(mutated)


def test_h3_runner_accepts_cpu_torch_build_and_rejects_wrong_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _modules()
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "2.9.0+cpu")
    assert runner._require_distribution_version("torch", "2.9.0") == "2.9.0"
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "2.8.0")
    with pytest.raises(RuntimeError, match="must be"):
        runner._require_distribution_version("torch", "2.9.0")


def test_h3_report_canonical_json_is_stable() -> None:
    _, report, official = _modules()
    cases = {
        points: {
            "video": official.clean_room_sigma_grid(points, MINIMAX_H3_VIDEO_SHIFT),
            "audio": official.clean_room_sigma_grid(points, MINIMAX_H3_AUDIO_SHIFT),
        }
        for points in report.REQUIRED_GRID_POINTS
    }
    probe = {
        "sample": (0.25, -0.5),
        "velocity": (0.125, -0.25),
        "timestep": 0.0,
        "sigma": cases[4]["video"][0],
        "sigma_next": cases[4]["video"][1],
        "reference": official.clean_room_dataward_step(
            sample=(0.25, -0.5),
            velocity=(0.125, -0.25),
            timestep=0.0,
            sigma=cases[4]["video"][0],
            sigma_next=cases[4]["video"][1],
        ),
    }
    first = report.build_parity_report(
        cases,
        velocity_reference=probe,
        environment={
            "device": "cpu",
            "diffusers": "0.39.0.dev0",
            "diffusers_revision": MINIMAX_H3_DIFFUSERS_REVISION,
            "numpy": "2.3.4",
            "python": "3.13",
            "torch": "2.9.0",
        },
    )
    assert report.canonical_json(first) == report.canonical_json(first)
    assert json.loads(report.canonical_json(first)) == first
