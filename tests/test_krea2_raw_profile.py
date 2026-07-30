"""Contracts for the immutable Krea 2 RAW structural profile."""

from __future__ import annotations

import importlib
import struct
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    CompatibilityLevel,
    EvidenceLevel,
    ExecutionFeatureRequest,
    PredictionType,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    TerminalSigma,
    evaluate_compatibility,
)
from comfyui_sigmax.profiles import (
    KREA2_RAW_DIFFUSERS_REFERENCE_28,
    KREA2_RAW_OFFICIAL_FULL_52,
    KREA2_RAW_PROFILE,
    DimensionAlignmentMode,
    DimensionPolicy,
    EvidenceReference,
    ExtrapolationPolicy,
    GuidanceConvention,
    Krea2RawProfile,
    Krea2RawRecipe,
    ResolutionShiftMode,
    ResolutionShiftPolicy,
    ShiftParameterization,
    build_krea2_raw_schedule,
)

KREA_REVISION = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret
DIFFUSERS_REVISION = "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret
COMFYUI_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret


def test_raw_builder_matches_frozen_bits_across_platforms() -> None:
    result = build_krea2_raw_schedule(
        width=1024,
        height=1024,
        recipe=KREA2_RAW_OFFICIAL_FULL_52,
    )

    assert tuple(struct.pack(">d", result.sigmas[index]).hex() for index in (12, 34)) == (
        "3fec8a62ce8eea56",  # pragma: allowlist secret
        "3fe226268349cb93",  # pragma: allowlist secret
    )


def test_builtin_raw_profile_declares_exact_structural_recipe() -> None:
    profile = KREA2_RAW_PROFILE

    assert isinstance(profile, Krea2RawProfile)
    assert profile.profile_id == "krea2.raw.official"
    assert profile.profile_version == "1"
    assert profile.evidence is EvidenceLevel.OFFICIAL
    assert profile.model_family == "krea2"
    assert profile.model_variant == "raw"
    assert profile.prediction_type is PredictionType.FLOW_VELOCITY
    assert profile.sigma_domain is SigmaDomain.UNIT_FLOW
    assert profile.ownership is ScheduleOwnership.EXTERNAL_SIGMAS
    assert profile.base_grid_identifier == "krea.reciprocal_step"
    assert profile.shift_parameterization is ShiftParameterization.EXPONENTIAL_MU
    assert profile.terminal_policy is TerminalPolicy.APPEND_ZERO
    assert profile.terminal_sigma is TerminalSigma.ZERO
    assert profile.reference_sampler_id == "comfy.euler"
    assert profile.dimensions == DimensionPolicy(
        mode=DimensionAlignmentMode.CEIL_MULTIPLE,
        multiple=16,
        evidence_source_id="krea.krea2.official",
    )


def test_raw_profile_declares_dynamic_shift_endpoints_without_a_fixed_mu() -> None:
    policy = KREA2_RAW_PROFILE.shift_policy

    assert policy == ResolutionShiftPolicy(
        mode=ResolutionShiftMode.RESOLUTION_LINEAR,
        base_image_seq_len=256,
        max_image_seq_len=6400,
        base_mu=0.5,
        max_mu=1.15,
        extrapolation=ExtrapolationPolicy.UPSTREAM_UNCLAMPED,
    )
    assert not hasattr(KREA2_RAW_PROFILE, "fixed_mu")


def test_raw_recipes_are_separate_named_evidence_contracts() -> None:
    official = KREA2_RAW_OFFICIAL_FULL_52
    framework = KREA2_RAW_DIFFUSERS_REFERENCE_28

    assert official == Krea2RawRecipe(
        recipe_id="krea2.raw.official-full-52",
        evidence=EvidenceLevel.OFFICIAL,
        steps=52,
        guidance=GuidanceConvention(krea_guidance=3.5, comfy_cfg=4.5),
        evidence_source_id="krea.krea2.official",
    )
    assert framework == Krea2RawRecipe(
        recipe_id="krea2.raw.diffusers-reference-28",
        evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
        steps=28,
        guidance=GuidanceConvention(krea_guidance=4.5, comfy_cfg=5.5),
        evidence_source_id="diffusers.krea2.framework",
    )
    assert KREA2_RAW_PROFILE.recipes == (framework, official)
    assert KREA2_RAW_PROFILE.official_full_recipe is official
    assert KREA2_RAW_PROFILE.framework_reference_recipe is framework


def test_raw_references_are_pinned_and_primary_source_is_official() -> None:
    references = KREA2_RAW_PROFILE.references

    assert references == (
        EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=("README.md", "inference.py", "sampling.py"),
        ),
        EvidenceReference(
            source_id="diffusers.krea2.framework",
            evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
            url="https://github.com/huggingface/diffusers",
            revision=DIFFUSERS_REVISION,
            locators=(
                "src/diffusers/pipelines/krea2/pipeline_krea2.py",
                "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
            ),
        ),
        EvidenceReference(
            source_id="comfyui.krea2.framework",
            evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
            url="https://github.com/Comfy-Org/ComfyUI",
            revision=COMFYUI_REVISION,
            locators=(
                "comfy/k_diffusion/sampling.py",
                "comfy/model_sampling.py",
                "comfy/supported_models.py",
            ),
        ),
    )
    assert KREA2_RAW_PROFILE.primary_reference is references[0]


def test_raw_capabilities_allow_declared_model_and_reference_euler() -> None:
    profile = KREA2_RAW_PROFILE
    decision = evaluate_compatibility(
        model=profile.model_capabilities,
        profile=profile.profile_capabilities,
        sampler=profile.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )

    assert decision.level is CompatibilityLevel.ALLOW
    assert profile.profile_capabilities.reference_sampler_ids == ("comfy.euler",)


def test_raw_profile_and_nested_declarations_are_deeply_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        KREA2_RAW_PROFILE.profile_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        KREA2_RAW_PROFILE.shift_policy.base_mu = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        KREA2_RAW_PROFILE.recipes[0].steps = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        KREA2_RAW_PROFILE.recipes[0].guidance.comfy_cfg = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ResolutionShiftPolicy(
            mode=cast(Any, "resolution_linear"),
            base_image_seq_len=256,
            max_image_seq_len=6400,
            base_mu=0.5,
            max_mu=1.15,
            extrapolation=ExtrapolationPolicy.UPSTREAM_UNCLAMPED,
        ),
        lambda: ResolutionShiftPolicy(
            mode=ResolutionShiftMode.RESOLUTION_LINEAR,
            base_image_seq_len=6400,
            max_image_seq_len=256,
            base_mu=0.5,
            max_mu=1.15,
            extrapolation=ExtrapolationPolicy.UPSTREAM_UNCLAMPED,
        ),
        lambda: ResolutionShiftPolicy(
            mode=ResolutionShiftMode.RESOLUTION_LINEAR,
            base_image_seq_len=256,
            max_image_seq_len=6400,
            base_mu=1.15,
            max_mu=0.5,
            extrapolation=ExtrapolationPolicy.UPSTREAM_UNCLAMPED,
        ),
        lambda: ResolutionShiftPolicy(
            mode=ResolutionShiftMode.RESOLUTION_LINEAR,
            base_image_seq_len=256,
            max_image_seq_len=6400,
            base_mu=float("nan"),
            max_mu=1.15,
            extrapolation=ExtrapolationPolicy.UPSTREAM_UNCLAMPED,
        ),
        lambda: ResolutionShiftPolicy(
            mode=ResolutionShiftMode.RESOLUTION_LINEAR,
            base_image_seq_len=256,
            max_image_seq_len=6400,
            base_mu=0.5,
            max_mu=1.15,
            extrapolation=cast(Any, "upstream_unclamped"),
        ),
        lambda: GuidanceConvention(krea_guidance=4.5, comfy_cfg=4.5),
        lambda: Krea2RawRecipe(
            recipe_id="Bad Recipe",
            evidence=EvidenceLevel.OFFICIAL,
            steps=52,
            guidance=GuidanceConvention(krea_guidance=3.5, comfy_cfg=4.5),
            evidence_source_id="krea.krea2.official",
        ),
        lambda: Krea2RawRecipe(
            recipe_id="krea2.raw.official-full-52",
            evidence=EvidenceLevel.FRAMEWORK_REFERENCE,
            steps=52,
            guidance=GuidanceConvention(krea_guidance=3.5, comfy_cfg=4.5),
            evidence_source_id="krea.krea2.official",
        ),
        lambda: Krea2RawRecipe(
            recipe_id="krea2.raw.official-full-52",
            evidence=cast(Any, "official"),
            steps=52,
            guidance=GuidanceConvention(krea_guidance=3.5, comfy_cfg=4.5),
            evidence_source_id="krea.krea2.official",
        ),
        lambda: Krea2RawRecipe(
            recipe_id="krea2.raw.official-full-52",
            evidence=EvidenceLevel.OFFICIAL,
            steps=52,
            guidance=cast(Any, "3.5/4.5"),
            evidence_source_id="krea.krea2.official",
        ),
        lambda: replace(KREA2_RAW_PROFILE, shift_parameterization=cast(Any, "direct_ratio")),
        lambda: replace(KREA2_RAW_PROFILE, shift_policy=cast(Any, object())),
        lambda: replace(KREA2_RAW_PROFILE, dimensions=cast(Any, object())),
        lambda: replace(KREA2_RAW_PROFILE, recipes=tuple(reversed(KREA2_RAW_PROFILE.recipes))),
        lambda: replace(
            KREA2_RAW_PROFILE,
            references=tuple(reversed(KREA2_RAW_PROFILE.references)),
        ),
        lambda: replace(KREA2_RAW_PROFILE, model_capabilities=cast(Any, object())),
    ),
)
def test_invalid_raw_declarations_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


def test_raw_profile_exports_geometry_derivation_and_exact_recipe_builder() -> None:
    profiles = importlib.import_module("comfyui_sigmax.profiles")

    assert hasattr(profiles, "build_krea2_raw_schedule")
    assert hasattr(profiles, "calculate_krea2_raw_mu")
    assert hasattr(profiles, "derive_krea2_raw_shift")
