"""Generate independent FLUX.1-schnell goldens without product imports."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Final, cast

_STEPS: Final = (1, 2, 3, 4)
_DECIMAL_PRECISION: Final = 80


def _binary32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _vector(steps: int) -> list[float]:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        count = Decimal(steps)
        return [float(Decimal(steps - index) / count) for index in range(steps + 1)]


def build_fixture() -> dict[str, Any]:
    """Return complete vectors plus every mandatory source-lane revision."""

    cases: list[dict[str, Any]] = []
    for steps in _STEPS:
        float64 = _vector(steps)
        cases.append(
            {
                "float32": [_binary32(value) for value in float64],
                "float64": float64,
                "steps": steps,
            }
        )
    return {
        "cases": cases,
        "generator": {
            "decimal_precision": _DECIMAL_PRECISION,
            "float32_quantization": "ieee754-binary32-round-to-nearest-even",
            "method": "decimal-unshifted-endpoint-grid-v1",
        },
        "schema": "sigmax.flux1-schnell-golden/1",
        "source_revisions": {
            "comfyui": ["2881e616", "1081439b", "1c3fb3b6", "c1f51b3d", "272da710"],
            "comfyui_examples": ["f9431bb0", "00ce7920", "94ff3454", "46e22cac", "1ea6cef3"],
            "official_github": ["802fb471", "3906133f", "cbd0d8dc", "5351620c", "a4773036"],
            "official_huggingface": ["741f7c3c", "e8b383c5", "4771c700", "3378a501", "91e9efe9"],
            "official_huggingface_readme": [
                "adb67b7a",
                "c923e832",
                "bfb7284b",
                "e9ae3d00",
                "bcdad000",
            ],
            "official_site": "2024-08-01",
        },
        "tolerances": {"float32_max_abs": "1e-6", "float64_max_abs": "1e-15"},
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
