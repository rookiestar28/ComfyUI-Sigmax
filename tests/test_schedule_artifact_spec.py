"""Contract tests for canonical schedule artifact specification v1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs" / "SCHEDULE_ARTIFACT_SPEC.md"
FIXTURES = ROOT / "tests" / "fixtures" / "artifacts"
NUMERICAL = FIXTURES / "numerical_projection_v1.json"
CONSTRUCTION_A = FIXTURES / "construction_projection_a_v1.json"
CONSTRUCTION_B = FIXTURES / "construction_projection_b_v1.json"
GOLDEN_HASHES = FIXTURES / "golden_hashes_v1.json"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_fixture_text(path: Path) -> str:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"{path.name} has a UTF-8 BOM"
    assert raw.endswith(b"\n"), f"{path.name} transport file must end in one LF"
    assert not raw.endswith(b"\n\n"), f"{path.name} has more than one trailing LF"
    return raw[:-1].decode("utf-8")


def _load_canonical_fixture(path: Path) -> tuple[str, dict[str, Any]]:
    text = _canonical_fixture_text(path)
    value = json.loads(text)
    assert isinstance(value, dict)
    return text, value


def _walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _walk_object_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk_object_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_object_keys(child)


def test_public_spec_defines_the_normative_v1_contract() -> None:
    content = SPEC.read_text(encoding="utf-8")

    required_terms = (
        "sigmax.schedule-artifact/1",
        "sigmax.numerical-schedule/1",
        "Sigmax Canonical Projection v1",
        "numerical fingerprint",
        "construction fingerprint",
        "requested inputs",
        "effective inputs",
        "ordered transform",
        "IEEE-754",
        "binary32",
        "binary64",
        "negative zero",
        "non-finite",
        "NFC",
        "ASCII",
        "UTF-8",
        "no trailing newline",
        "SHA-256",
        "identity and integrity",
        "not authenticity",
        "JCS-informed",
        "not JCS-compliant",
    )
    missing = [term for term in required_terms if term.casefold() not in content.casefold()]
    assert not missing, f"Artifact specification is missing: {missing}"


def test_projection_fixtures_are_canonical_schema_restricted_json() -> None:
    for path in (NUMERICAL, CONSTRUCTION_A, CONSTRUCTION_B):
        text, value = _load_canonical_fixture(path)
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        assert text == canonical
        assert all(not isinstance(item, float) for item in _walk(value))
        assert all(key.isascii() for key in _walk_object_keys(value))
        assert all(
            unicodedata.is_normalized("NFC", item) for item in _walk(value) if isinstance(item, str)
        )


def test_numerical_projection_pins_domain_precision_and_ordered_sigma_bits() -> None:
    _, numerical = _load_canonical_fixture(NUMERICAL)

    assert numerical == {
        "domain": "unit_flow",
        "precision": "float64",
        "schema": "sigmax.numerical-schedule/1",
        "sigmas": [
            "3ff0000000000000",
            "3fe8000000000000",
            "3fe0000000000000",
            "3fd0000000000000",
            "0000000000000000",
        ],
    }
    assert all(len(token) == 16 for token in numerical["sigmas"])
    assert numerical["sigmas"][-1] == "0000000000000000"


def test_construction_fixtures_separate_numerical_and_construction_identity() -> None:
    numerical_text, _ = _load_canonical_fixture(NUMERICAL)
    construction_a_text, construction_a = _load_canonical_fixture(CONSTRUCTION_A)
    construction_b_text, construction_b = _load_canonical_fixture(CONSTRUCTION_B)
    expected_numerical = f"sha256:{hashlib.sha256(numerical_text.encode()).hexdigest()}"

    assert construction_a["numerical_fingerprint"] == expected_numerical
    assert construction_b["numerical_fingerprint"] == expected_numerical
    assert construction_a["requested"] != construction_b["requested"]
    assert construction_a["effective"] == construction_b["effective"]
    assert construction_a["overrides"] != construction_b["overrides"]
    assert construction_a_text != construction_b_text
    assert (
        hashlib.sha256(construction_a_text.encode()).digest()
        != hashlib.sha256(construction_b_text.encode()).digest()
    )


def test_construction_fixture_records_required_provenance_and_order() -> None:
    _, construction = _load_canonical_fixture(CONSTRUCTION_A)

    assert construction["schema"] == "sigmax.schedule-artifact/1"
    assert construction["evidence"]["level"] == "experimental"
    assert construction["requested"]["profile"] == "fixture.power-of-two"
    assert construction["effective"]["precision"] == "float64"
    assert construction["base_grid"]["id"] == "fixture.power-of-two"
    assert construction["transforms"] == [
        {
            "from_domain": "unit_flow",
            "id": "no_shift",
            "parameters": {},
            "stage": 0,
            "to_domain": "unit_flow",
        }
    ]
    assert construction["terminal"]["policy"] == "append_zero"
    assert construction["slicing"]["policy"] == "full"
    assert "Kréa 臺灣" in construction["source"]["label"]


def test_typed_parameter_fixture_covers_float32_and_float64_widths() -> None:
    _, construction = _load_canonical_fixture(CONSTRUCTION_B)
    typed_parameters = construction["overrides"][0]["typed_parameters"]

    assert typed_parameters["float32_zero"] == {
        "bits": "00000000",
        "precision": "float32",
    }
    assert typed_parameters["float64_one"] == {
        "bits": "3ff0000000000000",
        "precision": "float64",
    }


def test_golden_hashes_match_exact_canonical_preimages() -> None:
    golden = json.loads(GOLDEN_HASHES.read_text(encoding="utf-8"))
    fixtures = {
        "construction_a": CONSTRUCTION_A,
        "construction_b": CONSTRUCTION_B,
        "numerical": NUMERICAL,
    }

    assert set(golden) == set(fixtures)
    for name, path in fixtures.items():
        text = _canonical_fixture_text(path)
        actual = f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"
        assert SHA256_PATTERN.fullmatch(golden[name])
        assert actual == golden[name]


def test_canonical_bytes_and_hashes_are_stable_across_process_hash_seeds() -> None:
    script = """
import hashlib
import json
import pathlib
import sys

raw = pathlib.Path(sys.argv[1]).read_bytes()
value = json.loads(raw)
canonical = json.dumps(
    value,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode()
print(hashlib.sha256(canonical).hexdigest())
"""
    expected = hashlib.sha256(_canonical_fixture_text(CONSTRUCTION_A).encode()).hexdigest()

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
        assert result.stdout.strip() == expected


def test_fixtures_exclude_nonfinite_values_secrets_and_private_paths() -> None:
    forbidden_markers = ("b:\\", "/mnt/", "cookie", "password", "secret", "access_token")
    forbidden_values = {"nan", "infinity", "+infinity", "-infinity"}

    for path in (NUMERICAL, CONSTRUCTION_A, CONSTRUCTION_B, GOLDEN_HASHES):
        text = path.read_text(encoding="utf-8")
        values = {item.casefold() for item in _walk(json.loads(text)) if isinstance(item, str)}
        leaked_markers = [marker for marker in forbidden_markers if marker in text.casefold()]
        leaked_values = forbidden_values.intersection(values)
        assert not leaked_markers, f"{path.name} contains forbidden data: {leaked_markers}"
        assert not leaked_values, f"{path.name} contains non-finite values: {leaked_values}"
