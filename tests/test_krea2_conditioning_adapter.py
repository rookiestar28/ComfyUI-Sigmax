from __future__ import annotations

from typing import Any, cast

import pytest
from comfyui_sigmax.adapters.krea2_conditioning import (
    ConditioningAdapterError,
    transform_krea2_conditioning,
)
from comfyui_sigmax.conditioning import (
    ConditioningModifierRequest,
    Krea2ConditioningProfileId,
    Krea2ConditioningVariant,
    get_krea2_profile,
)

torch = pytest.importorskip("torch")


def _request(
    *,
    variant: Krea2ConditioningVariant = Krea2ConditioningVariant.RAW,
    strength: float = 0.5,
) -> ConditioningModifierRequest:
    return ConditioningModifierRequest(
        variant=variant,
        profile=get_krea2_profile(Krea2ConditioningProfileId.CLASSIC_EXPERIMENTAL),
        strength=strength,
    )


def _conditioning() -> tuple[list[list[object]], Any, object]:
    tensor = torch.arange(2 * 30720, dtype=torch.float32).reshape(1, 2, 30720)
    reference = object()
    metadata = {
        "attention_mask": torch.ones((1, 2), dtype=torch.bool),
        "reference_latents": reference,
        "pooled_output": {"keep": reference},
    }
    return [[tensor, metadata]], tensor, reference


def test_rebalance_preserves_conditioning_metadata_dtype_device_and_input() -> None:
    conditioning, original, reference = _conditioning()
    original_copy = original.clone()

    output, stats = transform_krea2_conditioning(conditioning, _request())

    assert output is not conditioning
    assert output[0] is not conditioning[0]
    result = cast(Any, output[0][0])
    assert isinstance(result, torch.Tensor)
    assert result.shape == original.shape
    assert result.dtype is original.dtype
    assert result.device == original.device
    assert torch.equal(original, original_copy)
    assert not torch.equal(result, original)
    assert output[0][1] is not conditioning[0][1]
    metadata = cast(dict[str, Any], output[0][1])
    assert metadata["reference_latents"] is reference
    assert cast(dict[str, Any], metadata["pooled_output"])["keep"] is reference
    assert stats.input_shape == (1, 2, 30720)
    assert stats.transformed_entries == 1


def test_multiple_entries_support_long_sequences_without_mutating_metadata() -> None:
    first = torch.randn((1, 3, 30720), dtype=torch.float32)
    second = torch.randn((2, 97, 30720), dtype=torch.float32)
    first_metadata = {"branch": "positive", "mask": torch.ones((1, 3), dtype=torch.bool)}
    second_metadata = {"branch": "negative", "area": (0, 0, 64, 64)}
    conditioning = [[first, first_metadata], [second, second_metadata]]
    first_copy = first.clone()
    second_copy = second.clone()

    output, stats = transform_krea2_conditioning(conditioning, _request())

    assert stats.input_shapes == ((1, 3, 30720), (2, 97, 30720))
    assert stats.conditioning_entries == stats.transformed_entries == 2
    assert [entry[1] for entry in output] == [first_metadata, second_metadata]
    assert output[0][1] is not first_metadata
    assert output[1][1] is not second_metadata
    assert torch.equal(first, first_copy)
    assert torch.equal(second, second_copy)
    assert all(torch.isfinite(entry[0]).all() for entry in output)


def test_zero_input_sample_is_preserved_while_nonzero_sample_is_rebalanced() -> None:
    tensor = torch.zeros((2, 4, 30720), dtype=torch.float32)
    tensor[1] = torch.randn((4, 30720), dtype=torch.float32)

    output, _ = transform_krea2_conditioning([[tensor, {}]], _request())

    result = output[0][0]
    result = cast(Any, result)
    assert torch.equal(result[0], tensor[0])
    assert torch.isclose(
        torch.sqrt(torch.mean(result[1] ** 2)),
        torch.sqrt(torch.mean(tensor[1] ** 2)),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("variant", [Krea2ConditioningVariant.RAW, Krea2ConditioningVariant.TURBO])
def test_raw_and_turbo_use_the_same_validated_tensor_contract(
    variant: Krea2ConditioningVariant,
) -> None:
    conditioning, _, _ = _conditioning()
    output, stats = transform_krea2_conditioning(conditioning, _request(variant=variant))

    assert cast(Any, output[0][0]).shape == (1, 2, 30720)
    assert stats.variant is variant


def test_zero_strength_is_exact_identity_without_tensor_arithmetic() -> None:
    conditioning, original, _ = _conditioning()

    output, stats = transform_krea2_conditioning(conditioning, _request(strength=0.0))

    result = cast(Any, output[0][0])
    assert torch.equal(result, original)
    assert result is original
    assert stats.transformed_entries == 0


def test_non_contiguous_input_is_supported_and_rms_is_preserved() -> None:
    base = torch.randn((1, 30720, 2), dtype=torch.float32)
    tensor = base.transpose(1, 2)
    assert not tensor.is_contiguous()
    conditioning = [[tensor, {}]]
    input_rms = torch.sqrt(torch.mean(tensor.float() ** 2))

    output, _ = transform_krea2_conditioning(conditioning, _request())
    result = cast(Any, output[0][0])
    output_rms = torch.sqrt(torch.mean(result.float() ** 2))

    assert torch.isfinite(result).all()
    assert torch.isclose(input_rms, output_rms, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "conditioning",
    [
        [[torch.zeros((1, 2, 2560), dtype=torch.float32), {}]],
        [[torch.full((1, 2, 30720), float("nan")), {}]],
        [[torch.zeros((1, 2, 30720), dtype=torch.int64), {}]],
        [torch.zeros((1, 2, 30720), dtype=torch.float32)],
        "not-conditioning",
    ],
)
def test_adapter_rejects_malformed_or_unsafe_conditioning(conditioning: object) -> None:
    with pytest.raises(ConditioningAdapterError):
        transform_krea2_conditioning(conditioning, _request())
