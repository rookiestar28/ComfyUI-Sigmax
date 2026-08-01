"""Independent complete-vector goldens for FLUX.1-schnell."""

from __future__ import annotations

import ast
import json
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.profiles import build_flux1_schnell_schedule
from scripts.generate_flux1_schnell_goldens import build_fixture, canonical_json, main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).with_name("flux1_schnell_v1.json")
GENERATOR_PATH = ROOT / "scripts" / "generate_flux1_schnell_goldens.py"
EXPECTED_STEPS = (1, 2, 3, 4)


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _float32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def test_fixture_is_four_source_pinned_and_complete() -> None:
    fixture = _fixture()
    assert fixture["schema"] == "sigmax.flux1-schnell-golden/1"
    revisions = fixture["source_revisions"]
    assert (
        {
            key: "".join(value) if isinstance(value, list) else value
            for key, value in revisions.items()
        }
        == {
            "comfyui": "2881e6161081439b1c3fb3b6c1f51b3d272da710",  # pragma: allowlist secret
            "comfyui_examples": "f9431bb000ce792094ff345446e22cac1ea6cef3",  # pragma: allowlist secret
            "official_github": "802fb4713906133fcbd0d8dc5351620ca4773036",  # pragma: allowlist secret
            "official_huggingface": "741f7c3ce8b383c54771c7003378a50191e9efe9",  # pragma: allowlist secret
            "official_huggingface_readme": "adb67b7ac923e832bfb7284be9ae3d00bcdad000",  # pragma: allowlist secret
            "official_site": "2024-08-01",
        }
    )
    assert tuple(case["steps"] for case in fixture["cases"]) == EXPECTED_STEPS
    for case in fixture["cases"]:
        assert len(case["float64"]) == case["steps"] + 1
        assert case["float64"][0] == 1.0
        assert case["float64"][-1] == 0.0
        assert all(left > right for left, right in pairwise(case["float64"]))


def test_generator_is_independent_and_reproduces_fixture() -> None:
    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", maxsplit=1)[0])
    assert roots.isdisjoint({"comfyui_sigmax", "comfy", "diffusers", "numpy", "torch"})
    assert build_fixture() == _fixture()
    assert canonical_json(build_fixture()) == FIXTURE_PATH.read_text(encoding="utf-8")


def test_generator_cli_requires_explicit_output(tmp_path: Path) -> None:
    output = tmp_path / "golden.json"
    assert main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == canonical_json(build_fixture())


@pytest.mark.parametrize("steps", EXPECTED_STEPS)
def test_product_matches_independent_float64_and_float32(steps: int) -> None:
    case = next(item for item in _fixture()["cases"] if item["steps"] == steps)
    result = build_flux1_schnell_schedule(steps=steps)
    assert max(abs(a - b) for a, b in zip(result.sigmas, case["float64"], strict=True)) <= 1e-15
    projected = tuple(_float32(value) for value in result.sigmas)
    assert max(abs(a - b) for a, b in zip(projected, case["float32"], strict=True)) <= 1e-6
