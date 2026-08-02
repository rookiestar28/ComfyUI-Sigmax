from __future__ import annotations

import time
import tracemalloc

import pytest
from comfyui_sigmax.adapters.krea2_conditioning import transform_krea2_conditioning
from comfyui_sigmax.conditioning import (
    ConditioningModifierRequest,
    Krea2ConditioningProfileId,
    Krea2ConditioningVariant,
    get_krea2_profile,
)
from comfyui_sigmax.conditioning.diagnostics import (
    ConditioningDiagnostics,
    compute_conditioning_diagnostics,
)


def test_diagnostics_module_is_importable_without_torch() -> None:
    assert (
        "torch"
        not in __import__("comfyui_sigmax.conditioning.diagnostics", fromlist=["torch"]).__dict__
    )


torch = pytest.importorskip("torch")


def test_diagnostics_report_rms_cosine_relative_change_and_per_tap_rms() -> None:
    source = torch.ones((1, 2, 30720), dtype=torch.float32)
    transformed = source.clone()
    transformed.reshape(1, 2, 12, 2560)[:, :, 8] *= 2.0

    diagnostics = compute_conditioning_diagnostics(source, transformed)

    assert isinstance(diagnostics, ConditioningDiagnostics)
    assert diagnostics.input_rms == pytest.approx(1.0)
    assert diagnostics.output_rms > diagnostics.input_rms
    assert 0.0 < diagnostics.cosine < 1.0
    assert diagnostics.relative_change > 0.0
    assert len(diagnostics.input_tap_rms) == 12
    assert len(diagnostics.output_tap_rms) == 12
    assert diagnostics.output_tap_rms[8] == pytest.approx(2.0)


def test_diagnostics_rejects_shape_or_finite_drift() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_conditioning_diagnostics(torch.zeros((1, 2, 30720)), torch.zeros((1, 2, 2560)))
    with pytest.raises(ValueError, match="finite"):
        compute_conditioning_diagnostics(
            torch.full((1, 2, 30720), float("nan")),
            torch.zeros((1, 2, 30720)),
        )


@pytest.mark.slow
def test_long_cpu_transform_has_repeatable_bounded_latency_and_allocation() -> None:
    tensor = torch.arange(97 * 30720, dtype=torch.float32).reshape(1, 97, 30720)
    request = ConditioningModifierRequest(
        variant=Krea2ConditioningVariant.RAW,
        profile=get_krea2_profile(Krea2ConditioningProfileId.SUBTLE_EXPERIMENTAL),
        strength=0.5,
    )
    tracemalloc.start()
    first_started = time.perf_counter()
    first, _ = transform_krea2_conditioning([[tensor, {}]], request)
    first_seconds = time.perf_counter() - first_started
    _, first_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    repeat_started = time.perf_counter()
    repeat, _ = transform_krea2_conditioning([[tensor, {}]], request)
    repeat_seconds = time.perf_counter() - repeat_started
    _, repeat_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert torch.equal(first[0][0], repeat[0][0])
    assert first_seconds < 20.0
    assert repeat_seconds < 20.0
    assert max(first_peak, repeat_peak) < 512 * 1024 * 1024
