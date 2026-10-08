"""Independent high-precision arithmetic for the REF-42 written schedule specification.

ComfyUI 87c32827: Flux table 10000, simple integer indices, exp(mu).
Diffusers 122b1e11 / Qwen model d26bb612: target-token affine mu, stretch then zero.
This module does not import Sigmax, Torch, ComfyUI, Diffusers or downloaded publisher code.
"""

from __future__ import annotations

from decimal import Decimal, localcontext


def qwen_source_vector(lane: str, steps: int, tokens: int | None) -> tuple[float, ...]:
    with localcontext() as context:
        context.prec = 70
        dynamic = lane == "qwen_image21.diffusers-dynamic.framework-reference"
        if dynamic:
            if tokens is None:
                raise ValueError("dynamic source arithmetic requires tokens")
            mu = Decimal("0.5") + Decimal(tokens - 256) * Decimal("0.4") / Decimal(7936)
            grid = [Decimal(steps - i) / Decimal(steps) for i in range(steps)]
        else:
            mu = Decimal("1.15" if lane.startswith("qwen_image.") else "0.69")
            grid = [Decimal(10000 - i * 10000 // steps) / Decimal(10000) for i in range(steps)]
        alpha = mu.exp()
        shifted = [alpha * t / (1 + (alpha - 1) * t) for t in grid]
        if dynamic:
            denominator = 1 - shifted[-1]
            shifted = [1 - (1 - t) * Decimal("0.98") / denominator for t in shifted]
        return (*tuple(float(t) for t in shifted), 0.0)
