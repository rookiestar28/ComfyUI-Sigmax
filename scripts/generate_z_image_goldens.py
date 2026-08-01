"""Generate independent Z-Image golden vectors without product imports."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Final, cast

_CASES: Final = (("base", 28, Decimal("6")), ("base", 50, Decimal("6")), ("turbo", 8, Decimal("3")))
_DECIMAL_PRECISION: Final = 80


def _binary32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _vector(steps: int, ratio: Decimal) -> list[float]:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        values = [1.0]
        for index in range(1, steps):
            numerator = ratio * Decimal(steps - index)
            values.append(float(numerator / (numerator + Decimal(index))))
        values.append(0.0)
    return values


def build_fixture() -> dict[str, Any]:
    """Return complete vectors and all four required evidence-lane revisions."""

    cases: list[dict[str, Any]] = []
    for variant, steps, ratio in _CASES:
        float64 = _vector(steps, ratio)
        cases.append(
            {
                "float32": [_binary32(value) for value in float64],
                "float64": float64,
                "ratio": str(ratio),
                "steps": steps,
                "variant": variant,
            }
        )
    return {
        "cases": cases,
        "generator": {
            "decimal_precision": _DECIMAL_PRECISION,
            "float32_quantization": "ieee754-binary32-round-to-nearest-even",
            "method": "decimal-direct-ratio-v1",
        },
        "schema": "sigmax.z-image-golden/1",
        "source_revisions": {
            "comfyui": ["235b466a", "0cb26d47", "c24f2ab6", "6d1a8c5e", "70b21070"],
            "official_github": ["26f23eda", "626ffadd", "a020b04f", "f79488e1", "d72004cd"],
            "official_huggingface_base": [
                "04cc4abb",
                "7c506992",
                "6f75c9bf",
                "de9ef43d",
                "49423021",
            ],
            "official_huggingface_turbo": [
                "f332072a",
                "a78be7ae",
                "cdf3ee76",
                "d5c24708",
                "2da564a6",
            ],
            "official_site": ["e67bafb6", "73fa19d3", "01f903ac", "62de26c4", "8b4cc1c4"],
        },
        "tolerances": {"float32_max_abs": "1e-6", "float64_max_abs": "1e-12"},
    }


def canonical_json(fixture: dict[str, Any]) -> str:
    """Serialize deterministically with one terminal newline."""

    return (
        json.dumps(
            fixture, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Write the generated fixture to an explicit caller-selected path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    parsed.output.write_text(canonical_json(build_fixture()), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
