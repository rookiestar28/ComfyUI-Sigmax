"""M5-04 pinned Diffusers stochastic Flow Euler parity evidence."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.parity

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_stochastic_flow_euler_parity.py"
REPORT_MODULE = ROOT / "scripts" / "parity" / "stochastic_flow_euler_report.py"
FIXTURE = Path(__file__).with_name("fixtures") / "stochastic_flow_euler_diffusers_v0390.json"
LOCK = ROOT / "requirements" / "parity-stochastic-flow-euler.txt"


def _load_fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))


def test_parity_runner_report_fixture_and_lock_exist() -> None:
    assert RUNNER.is_file()
    assert REPORT_MODULE.is_file()
    assert FIXTURE.is_file()
    assert LOCK.read_text(encoding="utf-8").splitlines() == [
        "diffusers==0.39.0",
        "torch==2.9.0",
    ]


def test_runner_defers_optional_framework_imports() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=RUNNER.name)
    roots = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert roots.isdisjoint({"diffusers", "torch"})


def test_fixture_is_canonical_valid_and_pinned() -> None:
    report = importlib.import_module("scripts.parity.stochastic_flow_euler_report")
    fixture = _load_fixture()

    assert report.validate_parity_report(fixture) == fixture
    assert report.canonical_json(fixture) == FIXTURE.read_text(encoding="utf-8")
    assert fixture["schema"] == "sigmax.stochastic-flow-euler-diffusers-parity/1"
    assert fixture["status"] == "PASS"
    assert fixture["environment"] == {
        "device": "cpu",
        "diffusers": "0.39.0",
        "torch": "2.9.0",
    }
    assert fixture["source"] == {
        "blob_chunks": ["7b207f78", "20797c53", "b093452c", "a2bc5293", "8a8d84e7"],
        "evidence": "framework_reference",
        "locator": "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py:509-513",
        "revision_chunks": ["a3608b51", "2ed72484", "99a44c61", "d954965e", "d9bdae4d"],
        "tag": "v0.39.0",
        "url": "https://github.com/huggingface/diffusers",
    }


def test_fixture_proves_step_rng_and_seed_contracts() -> None:
    fixture = _load_fixture()
    case = fixture["case"]

    assert case["sigmas"] == [1.0, 0.75, 0.25, 0.0]
    assert case["transition_count"] == 3
    assert case["model_evaluation_count"] == 3
    assert case["noise_draw_count"] == 3
    assert case["terminal_noise_draw"] is True
    assert case["same_seed_repeat"] is True
    assert case["different_seed_diverges"] is True
    assert case["local_generator_state_matches"] is True
    assert case["global_rng_unchanged"] is True
    assert len(case["steps"]) == 3
    assert all(step["max_abs_error"] == "0.0" for step in case["steps"])
    assert all(step["mean_abs_error"] == "0.0" for step in case["steps"])
    assert len(case["noise_fingerprints"]) == 3
