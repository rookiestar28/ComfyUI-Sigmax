"""Generate independent experimental Krea 2 LoRA vectors without product imports."""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

_STEPS: Final = (4, 8, 12, 16)
_DECIMAL_PRECISION: Final = 80
_IMAGE_SEQ_LEN: Final = 4096
_TURBO_MU: Final = Decimal("1.15")


def _binary32(value: float) -> float:
    return cast(float, struct.unpack(">f", struct.pack(">f", value))[0])


def _raw_mu() -> Decimal:
    shift = Fraction(1, 2) + Fraction(13, 20) * Fraction(4096 - 256, 6400 - 256)
    return Decimal(shift.numerator) / Decimal(shift.denominator)


def _decimal_vector(*, steps: int, mu: Decimal) -> list[float]:
    with localcontext() as context:
        context.prec = _DECIMAL_PRECISION
        exponential = mu.exp()
        values = [1.0]
        for index in range(1, steps):
            numerator = exponential * Decimal(steps - index)
            values.append(float(numerator / (numerator + Decimal(index))))
        values.append(0.0)
    return values


def build_fixture() -> dict[str, Any]:
    """Return the complete versioned experimental fixture."""

    cases: list[dict[str, Any]] = []
    for mu_source, mu in (("raw", _raw_mu()), ("turbo", _TURBO_MU)):
        for steps in _STEPS:
            float64 = _decimal_vector(steps=steps, mu=mu)
            cases.append(
                {
                    "float32": [_binary32(value) for value in float64],
                    "float64": float64,
                    "mu": str(mu),
                    "mu_source": mu_source,
                    "steps": steps,
                }
            )
    return {
        "cases": cases,
        "evidence": {
            "level": "experimental",
            "sources": [
                "https://github.com/krea-ai/krea-2",
                "https://huggingface.co/krea/Krea-2-Raw",
                "https://huggingface.co/Comfy-Org/Krea-2",
                "https://www.krea.ai/blog/krea-2-technical-report",
                "https://github.com/Comfy-Org/ComfyUI",
            ],
        },
        "generator": {
            "decimal_precision": _DECIMAL_PRECISION,
            "float32_quantization": "ieee754-binary32-round-to-nearest-even",
            "method": "independent-decimal-affine-rational-v1",
        },
        "parameters": {
            "height": 1024,
            "image_seq_len": _IMAGE_SEQ_LEN,
            "raw_mu": str(_raw_mu()),
            "terminal": "zero",
            "turbo_mu": str(_TURBO_MU),
            "width": 1024,
        },
        "profile": {"id": "krea2.raw-turbo-lora.experimental", "version": "1"},
        "schema": "sigmax.krea2-lora-experimental-golden/1",
        "tolerances": {"float32_max_abs": "1e-6", "float64_max_abs": "1e-8"},
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
    """Write the fixture to an explicit caller-selected path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    parsed.output.write_text(canonical_json(build_fixture()), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
