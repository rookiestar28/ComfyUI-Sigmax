"""Generate independent Wan 2.1/2.2 schedule golden vectors."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Final, cast

_DECIMAL_PRECISION: Final = 80
_TRAINING_TIMESTEPS: Final = 1000

_CASES: Final = (
    ("wan2.1.t2v.comfy-native", 50, "none", 8.0),
    ("wan2.1.t2v.official-native", 50, "none", 5.0),
    ("wan2.1.i2v.480p.official-native", 40, "480p", 3.0),
    ("wan2.1.i2v.720p.official-native", 40, "720p", 5.0),
    ("wan2.1.t2v.diffusers-reference", 50, "none", 3.0),
    ("wan2.1.i2v.480p.diffusers-reference", 40, "480p", 3.0),
    ("wan2.1.i2v.720p.diffusers-reference", 40, "720p", 5.0),
    ("wan2.2.ti2v.5b.comfy-native", 50, "none", 5.0),
    ("wan2.2.t2v-a14b.official-native", 40, "none", 12.0),
    ("wan2.2.i2v-a14b.official-native", 40, "none", 5.0),
    ("wan2.2.ti2v.5b.diffusers-reference", 50, "none", 5.0),
    ("wan2.2.t2v-a14b.diffusers-reference", 40, "none", 3.0),
    ("wan2.2.i2v-a14b.diffusers-reference", 40, "none", 3.0),
)


def _binary32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _vector(steps: int, ratio: float) -> list[float]:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        count = Decimal(steps)
        multiplier = Decimal(str(ratio))
        return [
            float(
                multiplier
                * (Decimal(steps - index) / count)
                / (Decimal(1) + (multiplier - Decimal(1)) * (Decimal(steps - index) / count))
            )
            for index in range(steps)
        ] + [0.0]


def build_fixture() -> dict[str, Any]:
    """Return every released first-slice profile at its pinned recipe steps."""

    cases: list[dict[str, Any]] = []
    for profile, steps, resolution, ratio in _CASES:
        float64 = _vector(steps, ratio)
        cases.append(
            {
                "float32": [_binary32(value) for value in float64],
                "float64": float64,
                "profile": profile,
                "ratio": ratio,
                "resolution": resolution,
                "steps": steps,
            }
        )
    return {
        "cases": cases,
        "generator": {
            "decimal_precision": _DECIMAL_PRECISION,
            "float32_quantization": "ieee754-binary32-round-to-nearest-even",
            "method": "decimal-reciprocal-step-direct-ratio-v1",
            "training_timesteps": _TRAINING_TIMESTEPS,
        },
        "schema": "sigmax.wan-golden/1",
        "source_revisions": {
            "official_wan21": "9737cba9c1c3c4d04b33fcad41c111989865d315",
            "official_wan22": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
            "comfyui": "5cc026f5b81b3f01fe7a1438a0fd4131d2ebda25",
            "diffusers": "3c468926ffd12b69baa4316e27b09306b8da19a6",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    parsed.output.write_text(canonical_json(build_fixture()), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
