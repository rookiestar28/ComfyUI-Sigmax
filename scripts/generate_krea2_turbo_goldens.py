"""Generate independent Krea 2 Turbo golden vectors without product imports."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Final, cast

_STEPS: Final = (4, 8, 12, 16)
_DECIMAL_PRECISION: Final = 80
_MU: Final = Decimal("1.15")
_KREA_REVISION_CHUNKS: Final = (
    "db3984fb",
    "c6e13b34",
    "c0064990",
    "fc2d95ac",
    "64d00058",
)


def _binary32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _decimal_vector(steps: int) -> list[float]:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        exponential = _MU.exp()
        values = [1.0]
        for index in range(1, steps):
            numerator = exponential * Decimal(steps - index)
            values.append(float(numerator / (numerator + Decimal(index))))
        values.append(0.0)
    return values


def build_fixture() -> dict[str, Any]:
    """Return the complete versioned fixture as JSON-compatible values."""

    cases: list[dict[str, Any]] = []
    for steps in _STEPS:
        float64 = _decimal_vector(steps)
        cases.append(
            {
                "float32": [_binary32(value) for value in float64],
                "float64": float64,
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
            "method": "decimal-rational-v1",
        },
        "parameters": {
            "mu": str(_MU),
            "shift": "exponential_mu",
            "terminal": "zero",
        },
        "profile": {
            "id": "krea2.turbo.official",
            "version": "1",
        },
        "schema": "sigmax.krea2-turbo-golden/1",
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
