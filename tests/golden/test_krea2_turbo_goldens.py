"""Independent complete-vector goldens for the Krea 2 Turbo schedule."""

from __future__ import annotations

import ast
import json
import math
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import EvidenceLevel
from comfyui_sigmax.profiles import build_krea2_turbo_schedule
from scripts.generate_krea2_turbo_goldens import (
    build_fixture,
    canonical_json,
    main,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).with_name("krea2_turbo_v1.json")
GENERATOR_PATH = ROOT / "scripts" / "generate_krea2_turbo_goldens.py"

EXPECTED_SCHEMA = "sigmax.krea2-turbo-golden/1"
EXPECTED_STEPS = (4, 8, 12, 16)
KREA_REVISION = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret


def _load_fixture() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    )


def _float32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _maximum_error(actual: tuple[float, ...], expected: list[float]) -> float:
    return max(abs(left - right) for left, right in zip(actual, expected, strict=True))


def test_fixture_metadata_is_exact_and_evidence_pinned() -> None:
    fixture = _load_fixture()

    assert fixture["schema"] == EXPECTED_SCHEMA
    assert fixture["profile"] == {"id": "krea2.turbo.official", "version": "1"}
    assert fixture["evidence"] == {
        "level": "official",
        "source": {
            "locator": "sampling.py",
            "revision": {
                "algorithm": "git-sha1",
                "hex_chunks": [
                    "db3984fb",
                    "c6e13b34",
                    "c0064990",
                    "fc2d95ac",
                    "64d00058",
                ],
            },
            "url": "https://github.com/krea-ai/krea-2",
        },
    }
    assert "".join(fixture["evidence"]["source"]["revision"]["hex_chunks"]) == KREA_REVISION
    assert fixture["parameters"] == {
        "mu": "1.15",
        "shift": "exponential_mu",
        "terminal": "zero",
    }
    assert fixture["tolerances"] == {
        "float32_max_abs": "1e-6",
        "float64_max_abs": "1e-8",
    }
    assert fixture["generator"] == {
        "decimal_precision": 80,
        "float32_quantization": "ieee754-binary32-round-to-nearest-even",
        "method": "decimal-rational-v1",
    }


def test_fixture_has_only_complete_required_vectors() -> None:
    cases = _load_fixture()["cases"]

    assert isinstance(cases, list)
    assert tuple(case["steps"] for case in cases) == EXPECTED_STEPS
    for case in cases:
        steps = case["steps"]
        for precision in ("float64", "float32"):
            vector = case[precision]
            assert len(vector) == steps + 1
            assert vector[0] == 1.0
            assert vector[-1] == 0.0
            assert all(current > following for current, following in pairwise(vector))


def test_generator_is_independent_of_product_and_optional_frameworks() -> None:
    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"), filename=GENERATOR_PATH.name)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    assert imported_roots.isdisjoint({"comfyui_sigmax", "comfy", "diffusers", "numpy", "torch"})


def test_generator_reproduces_tracked_fixture_canonically() -> None:
    fixture = _load_fixture()

    assert build_fixture() == fixture
    assert canonical_json(fixture) == FIXTURE_PATH.read_text(encoding="utf-8")


def test_generator_cli_requires_an_explicit_output_and_writes_canonical_fixture(
    tmp_path: Path,
) -> None:
    output = tmp_path / "golden.json"

    assert main(["--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == canonical_json(build_fixture())


@pytest.mark.parametrize("steps", EXPECTED_STEPS)
def test_production_float64_schedule_matches_complete_golden(steps: int) -> None:
    fixture = _load_fixture()
    case = next(case for case in fixture["cases"] if case["steps"] == steps)
    result = build_krea2_turbo_schedule(steps=steps)
    tolerance = float(fixture["tolerances"]["float64_max_abs"])

    assert _maximum_error(result.sigmas, case["float64"]) <= tolerance
    expected_evidence = EvidenceLevel.OFFICIAL if steps == 8 else EvidenceLevel.MODIFIED
    assert result.request.provenance.evidence is expected_evidence


@pytest.mark.parametrize("steps", EXPECTED_STEPS)
def test_production_float32_projection_matches_complete_golden(steps: int) -> None:
    fixture = _load_fixture()
    case = next(case for case in fixture["cases"] if case["steps"] == steps)
    result = build_krea2_turbo_schedule(steps=steps)
    projected = tuple(_float32(value) for value in result.sigmas)
    tolerance = float(fixture["tolerances"]["float32_max_abs"])

    assert _maximum_error(projected, case["float32"]) <= tolerance


def test_eight_step_vector_matches_independent_official_direct_expression() -> None:
    fixture = _load_fixture()
    expected = next(case for case in fixture["cases"] if case["steps"] == 8)["float64"]
    exponential = math.exp(1.15)
    direct = [1.0]
    for index in range(1, 8):
        timestep = (8 - index) / 8
        direct.append(exponential / (exponential + (1.0 / timestep - 1.0)))
    direct.append(0.0)

    assert _maximum_error(tuple(direct), expected) <= 1e-12
