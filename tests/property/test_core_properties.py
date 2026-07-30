"""Deterministic property and metamorphic checks for pure-core invariants."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from comfyui_sigmax.core import (
    CapabilityDimension,
    CompatibilityDecision,
    CompatibilityLevel,
    CompatibilityReason,
    ExecutionBehavior,
    ExecutionFeatureRequest,
    ModelCapabilities,
    NoiseOwnership,
    PredictionType,
    ProfileCapabilities,
    SamplerCapabilities,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    TerminalRequirement,
    TerminalSigma,
    apply_terminal_policy,
    direct_ratio_shift,
    evaluate_compatibility,
    exponential_mu_shift,
    krea_reciprocal_step_grid,
    slice_step_range,
)


def test_krea_grid_and_terminal_properties_across_step_range() -> None:
    for steps in range(1, 129):
        grid = krea_reciprocal_step_grid(steps)
        terminal = apply_terminal_policy(
            grid,
            policy=TerminalPolicy.APPEND_ZERO,
            domain=SigmaDomain.UNIT_FLOW,
        )

        assert len(grid) == steps
        assert grid[0] == 1.0
        assert grid[-1] == 1.0 / steps
        assert all(left > right for left, right in pairwise(grid))
        assert len(terminal) == steps + 1
        assert terminal[-1] == 0.0
        assert slice_step_range(terminal) == terminal


@pytest.mark.parametrize("steps", [2, 3, 4, 8, 16, 31, 64])
@pytest.mark.parametrize("mu", [-20.0, -2.0, 0.0, 1.15, 2.0, 20.0])
def test_shift_parameterizations_are_metamorphically_equivalent(
    steps: int,
    mu: float,
) -> None:
    terminal = apply_terminal_policy(
        krea_reciprocal_step_grid(steps),
        policy=TerminalPolicy.APPEND_ZERO,
        domain=SigmaDomain.UNIT_FLOW,
    )
    exponential = exponential_mu_shift(terminal, mu=mu)
    direct = direct_ratio_shift(terminal, ratio=math.exp(mu))

    assert exponential == pytest.approx(direct, rel=1e-12, abs=1e-15)
    assert exponential[0] == 1.0
    assert exponential[-1] == 0.0
    assert all(left > right for left, right in pairwise(exponential))


def _capability_decision() -> CompatibilityDecision:
    model = ModelCapabilities(
        model_family="krea2",
        model_variant="turbo",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    profile = ProfileCapabilities(
        profile_id="krea2.turbo.official",
        profile_version="1",
        model_family="krea2",
        model_variant="turbo",
        prediction_type=PredictionType.FLOW_VELOCITY,
        sigma_domain=SigmaDomain.UNIT_FLOW,
        ownership=ScheduleOwnership.EXTERNAL_SIGMAS,
        terminal_sigma=TerminalSigma.ZERO,
        allowed_execution_behaviors=(ExecutionBehavior.DETERMINISTIC,),
        allowed_noise_ownerships=(NoiseOwnership.NONE,),
        allowed_sampler_state=(),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
        reference_sampler_ids=("comfy.euler",),
    )
    sampler = SamplerCapabilities(
        sampler_id="comfy.euler",
        sampler_version="native",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        terminal_requirement=TerminalRequirement.REQUIRES_ZERO,
        execution_behavior=ExecutionBehavior.DETERMINISTIC,
        noise_ownership=NoiseOwnership.NONE,
        required_state=(),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )
    return evaluate_compatibility(
        model=model,
        profile=profile,
        sampler=sampler,
        request=ExecutionFeatureRequest(),
    )


def test_capability_decision_is_repeatable_and_complete() -> None:
    decisions = tuple(_capability_decision() for _ in range(64))

    assert all(decision == decisions[0] for decision in decisions)
    assert decisions[0].level is CompatibilityLevel.ALLOW
    assert decisions[0].considered == tuple(CapabilityDimension)
    assert decisions[0].reasons == (CompatibilityReason.COMPATIBLE,)
