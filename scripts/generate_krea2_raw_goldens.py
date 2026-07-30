"""Generate independent Krea 2 RAW golden vectors without product imports."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Final, cast

_DECIMAL_PRECISION: Final = 80
_ALIGNMENT: Final = 16
_BASE_SEQ_LEN: Final = 256
_MAX_SEQ_LEN: Final = 6400
_BASE_MU: Final = Decimal("0.5")
_MAX_MU: Final = Decimal("1.15")
_RESOLUTIONS: Final = (
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1280, 1280),
    (1360, 768),
    (768, 1360),
)
_RECIPES: Final = (
    ("krea2.raw.diffusers-reference-28", 28, "framework_reference"),
    ("krea2.raw.official-full-52", 52, "official"),
)
_KREA_REVISION_CHUNKS: Final = (
    "db3984fb",
    "c6e13b34",
    "c0064990",
    "fc2d95ac",
    "64d00058",
)


def _binary32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _geometry(width: int, height: int) -> tuple[int, int, int]:
    effective_width = ((width + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT
    effective_height = ((height + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT
    image_seq_len = (effective_width // _ALIGNMENT) * (effective_height // _ALIGNMENT)
    return effective_width, effective_height, image_seq_len


def _decimal_mu(image_seq_len: int) -> Decimal:
    return _BASE_MU + (_MAX_MU - _BASE_MU) * (
        Decimal(image_seq_len - _BASE_SEQ_LEN) / Decimal(_MAX_SEQ_LEN - _BASE_SEQ_LEN)
    )


def _decimal_vector(steps: int, mu: Decimal) -> list[float]:
    exponential = mu.exp()
    values = [1.0]
    for index in range(1, steps):
        numerator = exponential * Decimal(steps - index)
        values.append(float(numerator / (numerator + Decimal(index))))
    values.append(0.0)
    return values


def build_fixture() -> dict[str, Any]:
    """Return the complete versioned RAW fixture as JSON-compatible values."""

    cases: list[dict[str, Any]] = []
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        for width, height in _RESOLUTIONS:
            effective_width, effective_height, image_seq_len = _geometry(width, height)
            mu = _decimal_mu(image_seq_len)
            for recipe_id, steps, evidence in _RECIPES:
                float64 = _decimal_vector(steps, mu)
                cases.append(
                    {
                        "effective_height": effective_height,
                        "effective_width": effective_width,
                        "evidence": evidence,
                        "float32": [_binary32(value) for value in float64],
                        "float64": float64,
                        "image_seq_len": image_seq_len,
                        "mu": float(mu),
                        "recipe_id": recipe_id,
                        "requested_height": height,
                        "requested_width": width,
                        "steps": steps,
                    }
                )

    return {
        "cases": cases,
        "evidence": {
            "level": "official",
            "source": {
                "locator": "sampling.py",
                "revision": {
                    "algorithm": "git-sha1",
                    "hex_chunks": list(_KREA_REVISION_CHUNKS),
                },
                "url": "https://github.com/krea-ai/krea-2",
            },
        },
        "generator": {
            "decimal_precision": _DECIMAL_PRECISION,
            "float32_quantization": "ieee754-binary32-round-to-nearest-even",
            "method": "independent-decimal-affine-rational-v1",
        },
        "parameters": {
            "alignment": _ALIGNMENT,
            "base_image_seq_len": _BASE_SEQ_LEN,
            "base_mu": str(_BASE_MU),
            "max_image_seq_len": _MAX_SEQ_LEN,
            "max_mu": str(_MAX_MU),
            "shift": "resolution_linear_exponential_mu",
            "terminal": "zero",
        },
        "profile": {
            "id": "krea2.raw.official",
            "version": "1",
        },
        "recipes": [
            {"evidence": evidence, "id": recipe_id, "steps": steps}
            for recipe_id, steps, evidence in _RECIPES
        ],
        "schema": "sigmax.krea2-raw-golden/1",
        "tolerances": {
            "float32_max_abs": "1e-6",
            "float64_max_abs": "1e-8",
        },
    }


def canonical_json(fixture: dict[str, Any]) -> str:
    """Serialize a fixture deterministically with one terminal newline."""

    return (
        json.dumps(
            fixture,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Write the canonical fixture to an explicit caller-selected path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    parsed.output.write_text(canonical_json(build_fixture()), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
