"""Independent complete-vector goldens for Krea 2 RAW schedules."""

from __future__ import annotations

import ast
import json
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError
from comfyui_sigmax.profiles import (
    KREA2_RAW_DIFFUSERS_REFERENCE_28,
    KREA2_RAW_OFFICIAL_FULL_52,
    KREA2_RAW_PROFILE,
    Krea2RawProfile,
    Krea2RawRecipe,
    build_krea2_raw_schedule,
)
from scripts.generate_krea2_raw_goldens import build_fixture, canonical_json, main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).with_name("krea2_raw_v1.json")
GENERATOR_PATH = ROOT / "scripts" / "generate_krea2_raw_goldens.py"

EXPECTED_SCHEMA = "sigmax.krea2-raw-golden/1"
EXPECTED_RESOLUTIONS = (
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1280, 1280),
    (1360, 768),
    (768, 1360),
)
EXPECTED_RECIPES = (
    "krea2.raw.diffusers-reference-28",
    "krea2.raw.official-full-52",
)
KREA_REVISION = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret


def _load_fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _float32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _maximum_error(actual: tuple[float, ...], expected: list[float]) -> float:
    return max(abs(left - right) for left, right in zip(actual, expected, strict=True))


def _recipe(recipe_id: str) -> Krea2RawRecipe:
    if recipe_id == KREA2_RAW_DIFFUSERS_REFERENCE_28.recipe_id:
        return KREA2_RAW_DIFFUSERS_REFERENCE_28
    if recipe_id == KREA2_RAW_OFFICIAL_FULL_52.recipe_id:
        return KREA2_RAW_OFFICIAL_FULL_52
    raise AssertionError(f"unexpected recipe: {recipe_id}")


def test_raw_fixture_metadata_is_exact_and_evidence_pinned() -> None:
    fixture = _load_fixture()

    assert fixture["schema"] == EXPECTED_SCHEMA
    assert fixture["profile"] == {"id": "krea2.raw.official", "version": "1"}
    assert fixture["evidence"]["level"] == "official"
    assert fixture["evidence"]["source"]["url"] == "https://github.com/krea-ai/krea-2"
    assert "".join(fixture["evidence"]["source"]["revision"]["hex_chunks"]) == KREA_REVISION
    assert fixture["parameters"] == {
        "alignment": 16,
        "base_image_seq_len": 256,
        "base_mu": "0.5",
        "max_image_seq_len": 6400,
        "max_mu": "1.15",
        "shift": "resolution_linear_exponential_mu",
        "terminal": "zero",
    }
    assert fixture["tolerances"] == {
        "float32_max_abs": "1e-6",
        "float64_max_abs": "1e-8",
    }
    assert fixture["generator"] == {
        "decimal_precision": 80,
        "float32_quantization": "ieee754-binary32-round-to-nearest-even",
        "method": "independent-decimal-affine-rational-v1",
    }


def test_raw_fixture_has_exact_complete_matrix() -> None:
    cases = _load_fixture()["cases"]

    assert len(cases) == len(EXPECTED_RESOLUTIONS) * len(EXPECTED_RECIPES)
    assert tuple(
        (case["requested_width"], case["requested_height"], case["recipe_id"]) for case in cases
    ) == tuple(
        (width, height, recipe)
        for width, height in EXPECTED_RESOLUTIONS
        for recipe in EXPECTED_RECIPES
    )
    for case in cases:
        steps = case["steps"]
        assert case["effective_width"] % 16 == 0
        assert case["effective_height"] % 16 == 0
        assert case["image_seq_len"] == (case["effective_width"] // 16) * (
            case["effective_height"] // 16
        )
        for precision in ("float64", "float32"):
            vector = case[precision]
            assert len(vector) == steps + 1
            assert vector[0] == 1.0
            assert vector[-1] == 0.0
            assert all(current > following for current, following in pairwise(vector))


def test_raw_generator_is_independent_of_product_and_optional_frameworks() -> None:
    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"), filename=GENERATOR_PATH.name)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots.isdisjoint({"comfyui_sigmax", "comfy", "diffusers", "numpy", "torch"})


def test_raw_generator_reproduces_tracked_fixture_canonically() -> None:
    fixture = _load_fixture()

    assert build_fixture() == fixture
    assert canonical_json(fixture) == FIXTURE_PATH.read_text(encoding="utf-8")


def test_raw_generator_cli_requires_explicit_output_and_writes_fixture(tmp_path: Path) -> None:
    output = tmp_path / "raw.json"

    assert main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == canonical_json(build_fixture())


def test_production_matches_every_raw_golden_case() -> None:
    fixture = _load_fixture()
    float64_tolerance = float(fixture["tolerances"]["float64_max_abs"])
    float32_tolerance = float(fixture["tolerances"]["float32_max_abs"])

    for case in fixture["cases"]:
        recipe = _recipe(case["recipe_id"])
        result = build_krea2_raw_schedule(
            width=case["requested_width"],
            height=case["requested_height"],
            recipe=recipe,
        )
        projected = tuple(_float32(value) for value in result.sigmas)

        assert result.effective_inputs.steps == case["steps"]
        assert result.effective_inputs.width == case["effective_width"]
        assert result.effective_inputs.height == case["effective_height"]
        assert _maximum_error(result.sigmas, case["float64"]) <= float64_tolerance
        assert _maximum_error(projected, case["float32"]) <= float32_tolerance
        assert result.request.provenance.evidence.value == case["evidence"]


def test_raw_recipe_provenance_remains_distinct() -> None:
    framework = build_krea2_raw_schedule(recipe=KREA2_RAW_DIFFUSERS_REFERENCE_28)
    official = build_krea2_raw_schedule(recipe=KREA2_RAW_OFFICIAL_FULL_52)

    assert framework.effective_inputs.steps == 28
    assert framework.request.provenance.evidence is EvidenceLevel.FRAMEWORK_REFERENCE
    assert framework.request.provenance.source == "https://github.com/huggingface/diffusers"
    assert official.effective_inputs.steps == 52
    assert official.request.provenance.evidence is EvidenceLevel.OFFICIAL
    assert official.request.provenance.source == "https://github.com/krea-ai/krea-2"


def test_raw_builder_records_dimension_overrides() -> None:
    result = build_krea2_raw_schedule(width=1025, height=767)

    assert result.effective_inputs.width == 1040
    assert result.effective_inputs.height == 768
    assert tuple(record.field for record in result.request.overrides) == ("width", "height")


@pytest.mark.parametrize(
    ("profile", "recipe"),
    (
        (cast(Krea2RawProfile, object()), KREA2_RAW_OFFICIAL_FULL_52),
        (KREA2_RAW_PROFILE, cast(Krea2RawRecipe, object())),
    ),
)
def test_raw_builder_rejects_invalid_profile_or_recipe(
    profile: Krea2RawProfile,
    recipe: Krea2RawRecipe,
) -> None:
    with pytest.raises(ScheduleContractError):
        build_krea2_raw_schedule(profile=profile, recipe=recipe)
