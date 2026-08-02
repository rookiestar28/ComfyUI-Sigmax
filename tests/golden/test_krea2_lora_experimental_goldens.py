"""Independent goldens for both experimental Krea 2 LoRA mu sources."""

from __future__ import annotations

import ast
import json
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from comfyui_sigmax.core import EvidenceLevel
from comfyui_sigmax.profiles import (
    Krea2ExperimentalMuSource,
    build_krea2_lora_experimental_schedule,
)
from scripts.generate_krea2_lora_experimental_goldens import (
    build_fixture,
    canonical_json,
    main,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).with_name("krea2_lora_experimental_v1.json")
GENERATOR_PATH = ROOT / "scripts" / "generate_krea2_lora_experimental_goldens.py"


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _float32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _max_error(actual: tuple[float, ...], expected: list[float]) -> float:
    return max(abs(left - right) for left, right in zip(actual, expected, strict=True))


def test_experimental_fixture_is_complete_and_truthfully_labeled() -> None:
    fixture = _fixture()

    assert fixture["schema"] == "sigmax.krea2-lora-experimental-golden/1"
    assert fixture["profile"] == {
        "id": "krea2.raw-turbo-lora.experimental",
        "version": "1",
    }
    assert fixture["evidence"]["level"] == "experimental"
    assert set(fixture["evidence"]["sources"]) == {
        "https://github.com/krea-ai/krea-2",
        "https://huggingface.co/krea/Krea-2-Raw",
        "https://huggingface.co/Comfy-Org/Krea-2",
        "https://www.krea.ai/blog/krea-2-technical-report",
        "https://github.com/Comfy-Org/ComfyUI",
    }
    assert fixture["parameters"] == {
        "height": 1024,
        "image_seq_len": 4096,
        "raw_mu": "0.90625",
        "terminal": "zero",
        "turbo_mu": "1.15",
        "width": 1024,
    }
    assert tuple((case["mu_source"], case["steps"]) for case in fixture["cases"]) == tuple(
        (source, steps) for source in ("raw", "turbo") for steps in (4, 8, 12, 16)
    )
    for case in fixture["cases"]:
        assert len(case["float64"]) == case["steps"] + 1
        assert case["float64"][-1] == 0.0
        assert all(left > right for left, right in pairwise(case["float64"]))


def test_experimental_generator_is_product_and_framework_independent() -> None:
    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"), filename=GENERATOR_PATH.name)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", maxsplit=1)[0])
    assert imported.isdisjoint({"comfyui_sigmax", "comfy", "diffusers", "numpy", "torch"})


def test_experimental_generator_reproduces_fixture_canonically(tmp_path: Path) -> None:
    fixture = _fixture()
    output = tmp_path / "experimental.json"

    assert build_fixture() == fixture
    assert canonical_json(fixture) == FIXTURE_PATH.read_text(encoding="utf-8")
    assert main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == canonical_json(fixture)


def test_production_matches_all_experimental_goldens() -> None:
    fixture = _fixture()
    tolerance64 = float(fixture["tolerances"]["float64_max_abs"])
    tolerance32 = float(fixture["tolerances"]["float32_max_abs"])

    for case in fixture["cases"]:
        result = build_krea2_lora_experimental_schedule(
            steps=case["steps"],
            width=1024,
            height=1024,
            mu_source=Krea2ExperimentalMuSource(case["mu_source"]),
        )
        projected = tuple(_float32(value) for value in result.sigmas)

        assert result.request.provenance.evidence is EvidenceLevel.EXPERIMENTAL
        assert _max_error(result.sigmas, case["float64"]) <= tolerance64
        assert _max_error(projected, case["float32"]) <= tolerance32
