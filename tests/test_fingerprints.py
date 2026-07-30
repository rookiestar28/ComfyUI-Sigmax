"""Canonical projection and schedule fingerprint tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    ScheduleContractError,
    SigmaDomain,
    build_numerical_projection,
    canonical_projection_bytes,
    construction_fingerprint,
    float_to_ieee_hex,
    numerical_fingerprint,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "artifacts"
NUMERICAL = FIXTURES / "numerical_projection_v1.json"
CONSTRUCTION_A = FIXTURES / "construction_projection_a_v1.json"
CONSTRUCTION_B = FIXTURES / "construction_projection_b_v1.json"
GOLDEN = json.loads((FIXTURES / "golden_hashes_v1.json").read_text(encoding="utf-8"))


def _preimage(path: Path) -> bytes:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    return raw[:-1]


def _construction(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [
        (0.0, "float32", "00000000"),
        (-0.0, "float32", "00000000"),
        (1.0, "float32", "3f800000"),
        (0.5, "float32", "3f000000"),
        (0.0, "float64", "0000000000000000"),
        (-0.0, "float64", "0000000000000000"),
        (1.0, "float64", "3ff0000000000000"),
        (0.5, "float64", "3fe0000000000000"),
    ],
)
def test_float_tokens_are_exact_and_normalize_negative_zero(
    value: float,
    precision: str,
    expected: str,
) -> None:
    assert float_to_ieee_hex(value, cast(Any, precision)) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_float_tokens_reject_nonfinite_and_boolean_values(value: float) -> None:
    with pytest.raises(ScheduleContractError):
        float_to_ieee_hex(value, "float64")


def test_float32_overflow_fails_closed() -> None:
    with pytest.raises(ScheduleContractError, match="float32"):
        float_to_ieee_hex(1e300, "float32")


def test_unknown_precision_fails_closed() -> None:
    with pytest.raises(ScheduleContractError, match="precision"):
        float_to_ieee_hex(1.0, cast(Any, "float16"))


def test_numerical_projection_and_fingerprint_match_m1_08_golden() -> None:
    sigmas = (1.0, 0.75, 0.5, 0.25, 0.0)
    projection = build_numerical_projection(
        sigmas,
        domain=SigmaDomain.UNIT_FLOW,
        precision="float64",
    )

    assert canonical_projection_bytes(projection) == _preimage(NUMERICAL)
    assert (
        numerical_fingerprint(
            sigmas,
            domain=SigmaDomain.UNIT_FLOW,
            precision="float64",
        )
        == GOLDEN["numerical"]
    )


def test_float32_precision_changes_numerical_identity() -> None:
    sigmas = (1.0, 0.75, 0.5, 0.25, 0.0)
    float32 = numerical_fingerprint(
        sigmas,
        domain=SigmaDomain.UNIT_FLOW,
        precision="float32",
    )
    float64 = numerical_fingerprint(
        sigmas,
        domain=SigmaDomain.UNIT_FLOW,
        precision="float64",
    )

    assert float32 != float64


def test_numerical_projection_requires_one_transition() -> None:
    with pytest.raises(ScheduleContractError, match="one transition"):
        build_numerical_projection(
            (0.0,),
            domain=SigmaDomain.UNIT_FLOW,
            precision="float64",
        )


def test_construction_fingerprints_match_paired_m1_08_goldens() -> None:
    construction_a = _construction(CONSTRUCTION_A)
    construction_b = _construction(CONSTRUCTION_B)

    assert canonical_projection_bytes(construction_a) == _preimage(CONSTRUCTION_A)
    assert canonical_projection_bytes(construction_b) == _preimage(CONSTRUCTION_B)
    assert construction_fingerprint(construction_a) == GOLDEN["construction_a"]
    assert construction_fingerprint(construction_b) == GOLDEN["construction_b"]
    assert construction_a["numerical_fingerprint"] == construction_b["numerical_fingerprint"]
    assert GOLDEN["construction_a"] != GOLDEN["construction_b"]


def test_canonicalizer_normalizes_nfc_values_and_sorts_nested_ascii_keys() -> None:
    composed = {"schema": "fixture/1", "value": {"a": "Kréa", "z": None}}
    decomposed = {"value": {"z": None, "a": "Kre\u0301a"}, "schema": "fixture/1"}

    assert canonical_projection_bytes(composed) == canonical_projection_bytes(decomposed)
    assert canonical_projection_bytes(composed) == (
        '{"schema":"fixture/1","value":{"a":"Kréa","z":null}}'.encode()
    )


@pytest.mark.parametrize(
    ("projection", "message"),
    [
        ({"é": "value"}, "ASCII"),
        ({"Upper": "value"}, "ASCII"),
        (cast(dict[str, object], {1: "value"}), "ASCII"),
        ({"value": 0.5}, "floating"),
        ({"value": 2**53}, "integer"),
        ({"value": object()}, "type"),
        ({"value": "x" * 4097}, "string"),
        ({"value": [None] * 1025}, "collection"),
    ],
)
def test_canonicalizer_rejects_unsupported_or_unbounded_values(
    projection: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ScheduleContractError, match=message):
        canonical_projection_bytes(projection)


def test_canonicalizer_rejects_excessive_nesting() -> None:
    nested: dict[str, object] = {}
    current = nested
    for _ in range(33):
        child: dict[str, object] = {}
        current["value"] = child
        current = child

    with pytest.raises(ScheduleContractError, match="depth"):
        canonical_projection_bytes(nested)


def test_canonicalizer_rejects_oversized_mapping() -> None:
    oversized = {f"k{index}": None for index in range(1025)}

    with pytest.raises(ScheduleContractError, match="collection"):
        canonical_projection_bytes(oversized)


def test_canonicalizer_rejects_non_mapping_root() -> None:
    with pytest.raises(ScheduleContractError, match="root must be a mapping"):
        canonical_projection_bytes(cast(Any, []))


def test_canonicalizer_rejects_oversized_encoded_projection() -> None:
    oversized = {"value": ["x" * 4096 for _ in range(1024)]}

    with pytest.raises(ScheduleContractError, match="byte limit"):
        canonical_projection_bytes(oversized)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(schema="unknown/1"),
        lambda value: value.update(unexpected=None),
        lambda value: value.update(construction_fingerprint="sha256:" + "0" * 64),
        lambda value: value.update(numerical_fingerprint="invalid"),
    ],
)
def test_construction_fingerprint_rejects_schema_boundary_violations(
    mutator: Any,
) -> None:
    construction = deepcopy(_construction(CONSTRUCTION_A))
    mutator(construction)

    with pytest.raises(ScheduleContractError):
        construction_fingerprint(construction)


def test_construction_fingerprint_requires_mapping() -> None:
    with pytest.raises(ScheduleContractError, match="must be a mapping"):
        construction_fingerprint(cast(Any, []))


def test_fingerprints_are_stable_across_subprocess_hash_seeds() -> None:
    script = """
import json
import pathlib
import sys
from comfyui_sigmax.core import construction_fingerprint

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(construction_fingerprint(value))
"""

    for seed in ("1", "777", "random"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script, str(CONSTRUCTION_A)],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )
        assert result.stdout.strip() == GOLDEN["construction_a"]
