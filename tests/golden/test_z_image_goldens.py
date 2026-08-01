"""Independent complete-vector goldens for Z-Image Base and Turbo."""

from __future__ import annotations

import ast
import json
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.profiles import ZImageVariant, build_z_image_schedule
from scripts.generate_z_image_goldens import build_fixture, canonical_json, main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).with_name("z_image_v1.json")
GENERATOR_PATH = ROOT / "scripts" / "generate_z_image_goldens.py"
EXPECTED_CASES = (("base", 28), ("base", 50), ("turbo", 8))


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _float32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def test_fixture_is_four_source_pinned_and_complete() -> None:
    fixture = _fixture()
    assert fixture["schema"] == "sigmax.z-image-golden/1"
    revisions = fixture["source_revisions"]
    assert (
        {key: "".join(chunks) for key, chunks in revisions.items()}
        == {
            "comfyui": "235b466a0cb26d47c24f2ab66d1a8c5e70b21070",  # pragma: allowlist secret
            "official_github": "26f23eda626ffadda020b04ff79488e1d72004cd",  # pragma: allowlist secret
            "official_huggingface_base": "04cc4abb7c5069926f75c9bfde9ef43d49423021",  # pragma: allowlist secret
            "official_huggingface_turbo": "f332072aa78be7aecdf3ee76d5c247082da564a6",  # pragma: allowlist secret
            "official_site": "e67bafb673fa19d301f903ac62de26c48b4cc1c4",  # pragma: allowlist secret
        }
    )
    assert tuple((case["variant"], case["steps"]) for case in fixture["cases"]) == EXPECTED_CASES
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


@pytest.mark.parametrize(("variant", "steps"), EXPECTED_CASES)
def test_product_matches_independent_float64_and_float32(variant: str, steps: int) -> None:
    case = next(
        item for item in _fixture()["cases"] if (item["variant"], item["steps"]) == (variant, steps)
    )
    result = build_z_image_schedule(variant=ZImageVariant(variant), steps=steps)
    assert max(abs(a - b) for a, b in zip(result.sigmas, case["float64"], strict=True)) <= 1e-12
    projected = tuple(_float32(value) for value in result.sigmas)
    assert max(abs(a - b) for a, b in zip(projected, case["float32"], strict=True)) <= 1e-6
