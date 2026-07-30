"""Contracts for the first evidence-pinned Krea 2 Turbo profile."""

from __future__ import annotations

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
    apply_terminal_policy,
    evaluate_compatibility,
    exponential_mu_shift,
    krea_reciprocal_step_grid,
    validate_sigma_schedule,
)
from comfyui_sigmax.profiles import (
    KREA2_TURBO_PROFILE,
    DimensionAlignmentMode,
    DimensionPolicy,
    EvidenceReference,
    GuidanceConvention,
    Krea2TurboProfile,
    ShiftParameterization,
    build_krea2_turbo_schedule,
)

KREA_REVISION = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret
DIFFUSERS_REVISION = "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret
COMFYUI_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret


def test_builtin_turbo_profile_declares_exact_official_recipe() -> None:
    profile = KREA2_TURBO_PROFILE

    assert isinstance(profile, Krea2TurboProfile)
    assert profile.profile_id == "krea2.turbo.official"
    assert profile.profile_version == "1"
    assert profile.evidence is EvidenceLevel.OFFICIAL
    assert profile.model_family == "krea2"
    assert profile.model_variant == "turbo"
    assert profile.prediction_type is PredictionType.FLOW_VELOCITY
    assert profile.sigma_domain is SigmaDomain.UNIT_FLOW
    assert profile.ownership is ScheduleOwnership.EXTERNAL_SIGMAS
    assert profile.base_grid_identifier == "krea.reciprocal_step"
    assert profile.shift_parameterization is ShiftParameterization.EXPONENTIAL_MU
    assert profile.fixed_mu == 1.15
    assert profile.terminal_policy is TerminalPolicy.APPEND_ZERO
    assert profile.terminal_sigma is TerminalSigma.ZERO
    assert profile.default_steps == 8
    assert profile.reference_sampler_id == "comfy.euler"
    assert profile.guidance == GuidanceConvention(
        krea_guidance=0.0,
        comfy_cfg=1.0,
    )
    assert profile.dimensions == DimensionPolicy(
        mode=DimensionAlignmentMode.CEIL_MULTIPLE,
        multiple=16,
        evidence_source_id="krea.krea2.official",
    )


def test_profile_references_are_pinned_and_primary_source_is_official() -> None:
    references = KREA2_TURBO_PROFILE.references

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
    assert KREA2_TURBO_PROFILE.primary_reference is references[0]


def test_profile_capabilities_allow_declared_model_and_reference_euler() -> None:
    profile = KREA2_TURBO_PROFILE
    decision = evaluate_compatibility(
        model=profile.model_capabilities,
        profile=profile.profile_capabilities,
        sampler=profile.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )

    assert decision.level is CompatibilityLevel.ALLOW
    assert profile.profile_capabilities.reference_sampler_ids == ("comfy.euler",)


@pytest.mark.parametrize(
    ("requested", "effective"),
    [
        ((1024, 1024), (1024, 1024)),
        ((1025, 1024), (1040, 1024)),
        ((1360, 769), (1360, 784)),
        ((17, 31), (32, 32)),
    ],
)
def test_turbo_builder_aligns_dimensions_and_records_effective_provenance(
    requested: tuple[int, int],
    effective: tuple[int, int],
) -> None:
    width, height = requested
    result = build_krea2_turbo_schedule(width=width, height=height)

    assert result.request.requested_inputs.width == width
    assert result.request.requested_inputs.height == height
    assert (result.effective_inputs.width, result.effective_inputs.height) == effective
    changed = {
        override.field: (override.requested_value, override.effective_value)
        for override in result.request.overrides
    }
    expected_changes = {
        field: (str(before), str(after))
        for field, before, after in (
            ("width", width, effective[0]),
            ("height", height, effective[1]),
        )
        if before != after
    }
    assert changed == expected_changes
    assert result.request.provenance.evidence is EvidenceLevel.OFFICIAL


def test_default_builder_composes_existing_core_without_embedding_a_golden() -> None:
    result = build_krea2_turbo_schedule(width=1024, height=1024)
    expected = apply_terminal_policy(
        exponential_mu_shift(
            krea_reciprocal_step_grid(8),
            mu=1.15,
        ),
        policy=TerminalPolicy.APPEND_ZERO,
        domain=SigmaDomain.UNIT_FLOW,
    )

    assert result.sigmas == expected
    assert len(result.sigmas) == 9
    assert result.request.provenance.source == "https://github.com/krea-ai/krea-2"
    assert result.request.provenance.source_revision == KREA_REVISION
    assert result.request.provenance.profile_id == "krea2.turbo.official"
    assert result.request.provenance.profile_version == "1"
    assert result.request.base_grid is not None
    assert result.request.base_grid.identifier == "krea.reciprocal_step"
    assert [transform.name for transform in result.request.transforms] == [
        "krea.exponential_mu",
        "terminal.append_zero",
    ]
    validate_sigma_schedule(
        result.sigmas,
        domain=SigmaDomain.UNIT_FLOW,
        expected_steps=8,
        require_terminal_zero=True,
    )


def test_nondefault_steps_preserve_formula_but_become_modified() -> None:
    result = build_krea2_turbo_schedule(steps=4, width=1024, height=1024)
    expected = apply_terminal_policy(
        exponential_mu_shift(
            krea_reciprocal_step_grid(4),
            mu=1.15,
        ),
        policy=TerminalPolicy.APPEND_ZERO,
        domain=SigmaDomain.UNIT_FLOW,
    )

    assert result.sigmas == expected
    assert result.request.provenance.evidence is EvidenceLevel.MODIFIED
    assert result.warnings == (
        "requested steps differ from the official Turbo 8-step recipe; evidence is modified",
    )


def test_official_profile_is_deeply_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        KREA2_TURBO_PROFILE.fixed_mu = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        KREA2_TURBO_PROFILE.guidance.comfy_cfg = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: GuidanceConvention(krea_guidance=1.0, comfy_cfg=1.0),
        lambda: GuidanceConvention(krea_guidance=0.0, comfy_cfg=2.0),
        lambda: GuidanceConvention(krea_guidance=float("nan"), comfy_cfg=1.0),
        lambda: DimensionPolicy(
            mode=cast(Any, "ceil_multiple"),
            multiple=16,
            evidence_source_id="krea.krea2.official",
        ),
        lambda: DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=0,
            evidence_source_id="krea.krea2.official",
        ),
        lambda: DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=8,
            evidence_source_id="krea.krea2.official",
        ),
        lambda: DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=16,
            evidence_source_id="other.source",
        ),
        lambda: DimensionPolicy(
            mode=DimensionAlignmentMode.CEIL_MULTIPLE,
            multiple=16,
            evidence_source_id="",
        ),
        lambda: EvidenceReference(
            source_id="Bad ID",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=("README.md",),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=cast(Any, "official"),
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=("README.md",),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="http://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=("README.md",),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision="main",
            locators=("README.md",),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=cast(Any, ["README.md"]),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=(),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=("",),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=("README.md", "README.md"),
        ),
        lambda: EvidenceReference(
            source_id="krea.krea2.official",
            evidence=EvidenceLevel.OFFICIAL,
            url="https://github.com/krea-ai/krea-2",
            revision=KREA_REVISION,
            locators=("sampling.py", "README.md"),
        ),
        lambda: replace(KREA2_TURBO_PROFILE, fixed_mu=0.9),
        lambda: replace(KREA2_TURBO_PROFILE, default_steps=12),
        lambda: replace(KREA2_TURBO_PROFILE, model_variant="raw"),
        lambda: replace(KREA2_TURBO_PROFILE, guidance=cast(Any, object())),
        lambda: replace(KREA2_TURBO_PROFILE, dimensions=cast(Any, object())),
        lambda: replace(
            KREA2_TURBO_PROFILE,
            references=KREA2_TURBO_PROFILE.references[1:],
        ),
        lambda: replace(
            KREA2_TURBO_PROFILE,
            reference_sampler_id="other.euler",
        ),
        lambda: replace(
            KREA2_TURBO_PROFILE,
            model_capabilities=cast(Any, object()),
        ),
        lambda: replace(
            KREA2_TURBO_PROFILE,
            profile_capabilities=cast(Any, object()),
        ),
        lambda: replace(
            KREA2_TURBO_PROFILE,
            reference_sampler_capabilities=cast(Any, object()),
        ),
    ],
)
def test_invalid_profile_declarations_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    "arguments",
    [
        {"steps": 0, "width": 1024, "height": 1024},
        {"steps": True, "width": 1024, "height": 1024},
        {"steps": 8, "width": 0, "height": 1024},
        {"steps": 8, "width": 1024, "height": False},
        {"steps": 8, "width": 1024, "height": 1024, "profile": object()},
    ],
)
def test_builder_rejects_invalid_inputs(arguments: dict[str, object]) -> None:
    with pytest.raises(ScheduleContractError):
        build_krea2_turbo_schedule(**cast(Any, arguments))
