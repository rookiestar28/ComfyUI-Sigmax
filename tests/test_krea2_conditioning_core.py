from __future__ import annotations

import json
import math

import pytest
from comfyui_sigmax.conditioning import (
    CONDITIONING_MODIFIER_ALGORITHM_ID,
    CONDITIONING_MODIFIER_REPORT_SCHEMA_ID,
    KREA2_CONDITIONING_PROFILE_SCHEMA_ID,
    KREA2_FEATURE_DIM,
    KREA2_TAP_COUNT,
    KREA2_TAP_DIM,
    ConditioningModifierReport,
    ConditioningModifierRequest,
    Krea2ConditioningProfile,
    Krea2ConditioningProfileId,
    Krea2ConditioningVariant,
    build_modifier_report,
    effective_gains,
    get_krea2_profile,
    validate_krea2_conditioning_shape,
)
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError


def test_krea2_shape_contract_requires_exact_three_dimensional_12_by_2560_features() -> None:
    assert KREA2_TAP_COUNT == 12
    assert KREA2_TAP_DIM == 2560
    assert KREA2_FEATURE_DIM == 30720
    assert validate_krea2_conditioning_shape((2, 64, 30720)) == (2, 64, 30720)

    for shape in ((30720,), (2, 64), (2, 64, 2560), (2, 64, 61440), (0, 64, 30720)):
        with pytest.raises(ScheduleContractError, match=r"rank-3|30720|positive"):
            validate_krea2_conditioning_shape(shape)


def test_builtin_profiles_are_immutable_exactly_twelve_positive_finite_gains() -> None:
    classic = get_krea2_profile(Krea2ConditioningProfileId.CLASSIC_EXPERIMENTAL)
    subtle = get_krea2_profile(Krea2ConditioningProfileId.SUBTLE_EXPERIMENTAL)
    disabled = get_krea2_profile(Krea2ConditioningProfileId.DISABLED)

    assert classic.gains == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
    assert subtle.gains == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.5, 1.0)
    assert disabled.gains == (1.0,) * 12
    for profile in (classic, subtle, disabled):
        assert profile.schema_id == KREA2_CONDITIONING_PROFILE_SCHEMA_ID
        assert profile.evidence is EvidenceLevel.COMMUNITY_RECOMMENDED
        assert len(profile.gains) == KREA2_TAP_COUNT
        assert all(math.isfinite(value) and 0.0 < value <= 8.0 for value in profile.gains)


@pytest.mark.parametrize(
    "gains",
    [
        (1.0,) * 11,
        (1.0,) * 13,
        (1.0,) * 11 + (math.nan,),
        (1.0,) * 11 + (0.0,),
        (1.0,) * 11 + (-1.0,),
        (1.0,) * 11 + (8.1,),
    ],
)
def test_profile_contract_rejects_invalid_gain_declarations(gains: tuple[float, ...]) -> None:
    with pytest.raises(ScheduleContractError):
        Krea2ConditioningProfile(
            profile_id="invalid",
            profile_version="1",
            evidence=EvidenceLevel.COMMUNITY_RECOMMENDED,
            source="test",
            gains=gains,
        )


def test_request_bounds_strength_and_requires_explicit_variant() -> None:
    profile = get_krea2_profile(Krea2ConditioningProfileId.CLASSIC_EXPERIMENTAL)
    request = ConditioningModifierRequest(
        variant=Krea2ConditioningVariant.RAW,
        profile=profile,
        strength=0.5,
    )

    assert request.variant_evidence == "user_selected"
    assert effective_gains(request) == tuple(1.0 + 0.5 * (gain - 1.0) for gain in profile.gains)
    assert (
        effective_gains(
            ConditioningModifierRequest(
                variant=Krea2ConditioningVariant.TURBO,
                profile=profile,
                strength=0.0,
            )
        )
        == (1.0,) * 12
    )

    for strength in (-0.01, 1.01, math.nan, math.inf):
        with pytest.raises(ScheduleContractError, match="strength"):
            ConditioningModifierRequest(
                variant=Krea2ConditioningVariant.RAW,
                profile=profile,
                strength=strength,
            )

    with pytest.raises(ScheduleContractError, match="variant"):
        ConditioningModifierRequest(
            variant="Auto",  # type: ignore[arg-type]
            profile=profile,
            strength=0.5,
        )


def test_modifier_report_is_canonical_bounded_and_payload_free() -> None:
    request = ConditioningModifierRequest(
        variant=Krea2ConditioningVariant.TURBO,
        profile=get_krea2_profile(Krea2ConditioningProfileId.SUBTLE_EXPERIMENTAL),
        strength=0.25,
    )
    report = build_modifier_report(
        request=request,
        input_shape=(1, 32, 30720),
        input_shapes=((1, 32, 30720), (1, 96, 30720)),
        dtype="torch.float16",
        device="cuda:0",
        conditioning_entries=2,
        transformed_entries=2,
        warnings=("experimental_profile",),
    )

    assert isinstance(report, ConditioningModifierReport)
    assert report.schema_id == CONDITIONING_MODIFIER_REPORT_SCHEMA_ID
    assert report.algorithm_id == CONDITIONING_MODIFIER_ALGORITHM_ID
    assert report.schedule_affected is False
    assert report.variant_evidence == "user_selected"
    parsed = json.loads(report.json_text)
    assert parsed["fingerprint"] == report.fingerprint
    assert parsed["input"]["shape"] == [1, 32, 30720]
    assert parsed["input"]["shapes"] == [[1, 32, 30720], [1, 96, 30720]]
    assert parsed["output"]["transformed_entries"] == 2
    assert "tensor" not in report.json_text
    assert "prompt" not in report.json_text
    assert report.json_text == json.dumps(
        parsed,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_modifier_report_rejects_shape_facts_that_disagree_with_primary_shape() -> None:
    request = ConditioningModifierRequest(
        variant=Krea2ConditioningVariant.RAW,
        profile=get_krea2_profile(Krea2ConditioningProfileId.DISABLED),
        strength=0.0,
    )

    with pytest.raises(ScheduleContractError, match="shape"):
        build_modifier_report(
            request=request,
            input_shape=(1, 32, 30720),
            input_shapes=((1, 33, 30720),),
            dtype="torch.float32",
            device="cpu",
            conditioning_entries=1,
            transformed_entries=0,
        )
