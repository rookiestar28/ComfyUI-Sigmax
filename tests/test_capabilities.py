"""Framework-independent capability declaration and compatibility tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    CapabilityCompatibilityError,
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
    SamplerState,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalRequirement,
    TerminalSigma,
    evaluate_compatibility,
    require_compatible,
)


def _model() -> ModelCapabilities:
    return ModelCapabilities(
        model_family="krea2",
        model_variant="turbo",
        accepted_prediction_types=(PredictionType.FLOW_VELOCITY,),
        accepted_sigma_domains=(SigmaDomain.UNIT_FLOW,),
        accepted_ownerships=(ScheduleOwnership.EXTERNAL_SIGMAS,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
    )


def _profile() -> ProfileCapabilities:
    return ProfileCapabilities(
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
        allowed_sampler_state=(SamplerState.STEP_INDEX,),
        supports_partial_denoise=True,
        supports_per_token_timesteps=False,
        reference_sampler_ids=("comfy.euler",),
    )


def _sampler() -> SamplerCapabilities:
    return SamplerCapabilities(
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


def _decision(
    *,
    model: ModelCapabilities | None = None,
    profile: ProfileCapabilities | None = None,
    sampler: SamplerCapabilities | None = None,
    request: ExecutionFeatureRequest | None = None,
) -> CompatibilityDecision:
    return evaluate_compatibility(
        model=model or _model(),
        profile=profile or _profile(),
        sampler=sampler or _sampler(),
        request=request or ExecutionFeatureRequest(),
    )


def test_krea_like_native_euler_capabilities_allow_execution() -> None:
    decision = _decision()

    assert decision.level is CompatibilityLevel.ALLOW
    assert decision.reasons == (CompatibilityReason.COMPATIBLE,)
    assert decision.considered == tuple(CapabilityDimension)
    assert require_compatible(decision) is decision


def test_capability_compatible_nonreference_sampler_warns() -> None:
    decision = _decision(sampler=replace(_sampler(), sampler_id="other.euler"))

    assert decision.level is CompatibilityLevel.WARN
    assert decision.reasons == (CompatibilityReason.SAMPLER_NOT_PROFILE_REFERENCE,)
    assert require_compatible(decision) is decision


@pytest.mark.parametrize(
    ("terminal", "requirement"),
    [
        (TerminalSigma.ZERO, TerminalRequirement.ACCEPTS_EITHER),
        (TerminalSigma.NONZERO, TerminalRequirement.ACCEPTS_EITHER),
        (TerminalSigma.NONZERO, TerminalRequirement.FORBIDS_ZERO),
    ],
)
def test_compatible_terminal_policies_allow(
    terminal: TerminalSigma,
    requirement: TerminalRequirement,
) -> None:
    decision = _decision(
        profile=replace(_profile(), terminal_sigma=terminal),
        sampler=replace(_sampler(), terminal_requirement=requirement),
    )

    assert decision.level is CompatibilityLevel.ALLOW


@pytest.mark.parametrize(
    ("model", "profile", "sampler", "reason"),
    [
        (
            replace(_model(), model_family="other"),
            _profile(),
            _sampler(),
            CompatibilityReason.MODEL_FAMILY_MISMATCH,
        ),
        (
            replace(_model(), model_variant="raw"),
            _profile(),
            _sampler(),
            CompatibilityReason.MODEL_VARIANT_MISMATCH,
        ),
        (
            replace(
                _model(),
                accepted_prediction_types=(PredictionType.EPSILON,),
            ),
            _profile(),
            _sampler(),
            CompatibilityReason.MODEL_PREDICTION_UNSUPPORTED,
        ),
        (
            replace(
                _model(),
                accepted_sigma_domains=(SigmaDomain.CONTINUOUS_EDM,),
            ),
            _profile(),
            _sampler(),
            CompatibilityReason.MODEL_SIGMA_DOMAIN_UNSUPPORTED,
        ),
        (
            replace(
                _model(),
                accepted_ownerships=(ScheduleOwnership.MODEL_NATIVE,),
            ),
            _profile(),
            _sampler(),
            CompatibilityReason.MODEL_OWNERSHIP_UNSUPPORTED,
        ),
        (
            _model(),
            _profile(),
            replace(
                _sampler(),
                accepted_prediction_types=(PredictionType.EPSILON,),
            ),
            CompatibilityReason.SAMPLER_PREDICTION_UNSUPPORTED,
        ),
        (
            _model(),
            _profile(),
            replace(
                _sampler(),
                accepted_sigma_domains=(SigmaDomain.CONTINUOUS_EDM,),
            ),
            CompatibilityReason.SAMPLER_SIGMA_DOMAIN_UNSUPPORTED,
        ),
        (
            _model(),
            _profile(),
            replace(
                _sampler(),
                accepted_ownerships=(ScheduleOwnership.MODEL_NATIVE,),
            ),
            CompatibilityReason.SAMPLER_OWNERSHIP_UNSUPPORTED,
        ),
        (
            _model(),
            _profile(),
            replace(
                _sampler(),
                terminal_requirement=TerminalRequirement.FORBIDS_ZERO,
            ),
            CompatibilityReason.TERMINAL_REQUIREMENT_MISMATCH,
        ),
        (
            _model(),
            _profile(),
            replace(
                _sampler(),
                execution_behavior=ExecutionBehavior.STOCHASTIC,
                noise_ownership=NoiseOwnership.SAMPLER,
            ),
            CompatibilityReason.EXECUTION_BEHAVIOR_MISMATCH,
        ),
        (
            _model(),
            replace(
                _profile(),
                allowed_execution_behaviors=(
                    ExecutionBehavior.DETERMINISTIC,
                    ExecutionBehavior.STOCHASTIC,
                ),
                allowed_noise_ownerships=(NoiseOwnership.NONE, NoiseOwnership.CALLER),
            ),
            replace(
                _sampler(),
                execution_behavior=ExecutionBehavior.STOCHASTIC,
                noise_ownership=NoiseOwnership.SAMPLER,
            ),
            CompatibilityReason.NOISE_OWNERSHIP_MISMATCH,
        ),
        (
            _model(),
            _profile(),
            replace(_sampler(), required_state=(SamplerState.MULTISTEP_HISTORY,)),
            CompatibilityReason.SAMPLER_STATE_UNSUPPORTED,
        ),
    ],
)
def test_each_model_profile_sampler_semantic_mismatch_rejects(
    model: ModelCapabilities,
    profile: ProfileCapabilities,
    sampler: SamplerCapabilities,
    reason: CompatibilityReason,
) -> None:
    decision = _decision(model=model, profile=profile, sampler=sampler)

    assert decision.level is CompatibilityLevel.REJECT
    assert reason in decision.reasons
    with pytest.raises(CapabilityCompatibilityError, match=reason.value):
        require_compatible(decision)


@pytest.mark.parametrize(
    ("feature", "component", "reason"),
    [
        (
            "partial",
            "model",
            CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_MODEL,
        ),
        (
            "partial",
            "profile",
            CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_PROFILE,
        ),
        (
            "partial",
            "sampler",
            CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_SAMPLER,
        ),
        (
            "per_token",
            "model",
            CompatibilityReason.PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_MODEL,
        ),
        (
            "per_token",
            "profile",
            CompatibilityReason.PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_PROFILE,
        ),
        (
            "per_token",
            "sampler",
            CompatibilityReason.PER_TOKEN_TIMESTEPS_UNSUPPORTED_BY_SAMPLER,
        ),
    ],
)
def test_requested_execution_feature_requires_three_party_support(
    feature: str,
    component: str,
    reason: CompatibilityReason,
) -> None:
    model = _model()
    profile = _profile()
    sampler = _sampler()
    if feature == "partial":
        request = ExecutionFeatureRequest(use_partial_denoise=True)
        if component == "model":
            model = replace(model, supports_partial_denoise=False)
        elif component == "profile":
            profile = replace(profile, supports_partial_denoise=False)
        else:
            sampler = replace(sampler, supports_partial_denoise=False)
    else:
        request = ExecutionFeatureRequest(use_per_token_timesteps=True)
        model = replace(model, supports_per_token_timesteps=True)
        profile = replace(profile, supports_per_token_timesteps=True)
        sampler = replace(sampler, supports_per_token_timesteps=True)
        if component == "model":
            model = replace(model, supports_per_token_timesteps=False)
        elif component == "profile":
            profile = replace(profile, supports_per_token_timesteps=False)
        else:
            sampler = replace(sampler, supports_per_token_timesteps=False)

    decision = _decision(model=model, profile=profile, sampler=sampler, request=request)

    assert decision.level is CompatibilityLevel.REJECT
    assert reason in decision.reasons


def test_supported_partial_and_per_token_features_allow() -> None:
    request = ExecutionFeatureRequest(
        use_partial_denoise=True,
        use_per_token_timesteps=True,
    )
    decision = _decision(
        model=replace(_model(), supports_per_token_timesteps=True),
        profile=replace(_profile(), supports_per_token_timesteps=True),
        sampler=replace(_sampler(), supports_per_token_timesteps=True),
        request=request,
    )

    assert decision.level is CompatibilityLevel.ALLOW


def test_multiple_rejections_have_stable_evaluator_order_and_no_warning() -> None:
    decision = _decision(
        model=replace(
            _model(),
            model_family="other",
            accepted_prediction_types=(PredictionType.EPSILON,),
            supports_partial_denoise=False,
        ),
        sampler=replace(_sampler(), sampler_id="other.euler", supports_partial_denoise=False),
        request=ExecutionFeatureRequest(use_partial_denoise=True),
    )

    assert decision.level is CompatibilityLevel.REJECT
    assert decision.reasons == (
        CompatibilityReason.MODEL_FAMILY_MISMATCH,
        CompatibilityReason.MODEL_PREDICTION_UNSUPPORTED,
        CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_MODEL,
        CompatibilityReason.PARTIAL_DENOISE_UNSUPPORTED_BY_SAMPLER,
    )


def test_rejected_gate_stops_before_execution_sentinel() -> None:
    executed: list[bool] = []
    decision = _decision(
        sampler=replace(
            _sampler(),
            accepted_sigma_domains=(SigmaDomain.CONTINUOUS_EDM,),
        )
    )

    def guarded_execution() -> None:
        require_compatible(decision)
        executed.append(True)

    with pytest.raises(CapabilityCompatibilityError):
        guarded_execution()
    assert executed == []


def test_capability_contracts_are_frozen() -> None:
    model = _model()
    decision = _decision()

    with pytest.raises(FrozenInstanceError):
        model.model_variant = "raw"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.level = CompatibilityLevel.REJECT  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_model(), model_family=""),
        lambda: replace(_model(), model_family="Krea2"),
        lambda: replace(_model(), model_family="x" * 257),
        lambda: replace(_model(), supports_partial_denoise=cast(Any, 1)),
        lambda: replace(_model(), accepted_prediction_types=cast(Any, [])),
        lambda: replace(_model(), accepted_prediction_types=()),
        lambda: replace(
            _model(),
            accepted_prediction_types=(
                PredictionType.FLOW_VELOCITY,
                PredictionType.FLOW_VELOCITY,
            ),
        ),
        lambda: replace(
            _model(),
            accepted_prediction_types=(
                PredictionType.EPSILON,
                PredictionType.FLOW_VELOCITY,
            ),
        ),
        lambda: replace(
            _model(),
            accepted_prediction_types=cast(Any, ("flow_velocity",)),
        ),
        lambda: replace(_profile(), reference_sampler_ids=("",)),
        lambda: replace(_profile(), reference_sampler_ids=cast(Any, [])),
        lambda: replace(_profile(), profile_version="x" * 257),
        lambda: replace(_profile(), prediction_type=cast(Any, "flow_velocity")),
        lambda: replace(_profile(), sigma_domain=cast(Any, "unit_flow")),
        lambda: replace(_profile(), ownership=cast(Any, "external_sigmas")),
        lambda: replace(_profile(), terminal_sigma=cast(Any, "zero")),
        lambda: replace(
            _profile(),
            reference_sampler_ids=("comfy.euler", "comfy.euler"),
        ),
        lambda: replace(
            _profile(),
            reference_sampler_ids=("z.euler", "a.euler"),
        ),
        lambda: replace(
            _profile(),
            allowed_execution_behaviors=(ExecutionBehavior.DETERMINISTIC,),
            allowed_noise_ownerships=(NoiseOwnership.SAMPLER,),
        ),
        lambda: replace(
            _profile(),
            allowed_execution_behaviors=(ExecutionBehavior.STOCHASTIC,),
            allowed_noise_ownerships=(NoiseOwnership.NONE,),
        ),
        lambda: replace(
            _profile(),
            allowed_execution_behaviors=(
                ExecutionBehavior.DETERMINISTIC,
                ExecutionBehavior.STOCHASTIC,
            ),
            allowed_noise_ownerships=(NoiseOwnership.NONE,),
        ),
        lambda: replace(
            _profile(),
            allowed_execution_behaviors=(ExecutionBehavior.DETERMINISTIC,),
            allowed_noise_ownerships=(NoiseOwnership.NONE, NoiseOwnership.CALLER),
        ),
        lambda: replace(
            _sampler(),
            terminal_requirement=cast(Any, "requires_zero"),
        ),
        lambda: replace(
            _sampler(),
            execution_behavior=cast(Any, "deterministic"),
        ),
        lambda: replace(
            _sampler(),
            noise_ownership=cast(Any, "none"),
        ),
        lambda: replace(
            _sampler(),
            execution_behavior=ExecutionBehavior.DETERMINISTIC,
            noise_ownership=NoiseOwnership.SAMPLER,
        ),
        lambda: replace(
            _sampler(),
            execution_behavior=ExecutionBehavior.STOCHASTIC,
            noise_ownership=NoiseOwnership.NONE,
        ),
        lambda: ExecutionFeatureRequest(use_partial_denoise=cast(Any, 1)),
    ],
)
def test_invalid_capability_declarations_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.ALLOW,
            considered=tuple(CapabilityDimension),
            reasons=(CompatibilityReason.SAMPLER_NOT_PROFILE_REFERENCE,),
        ),
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.WARN,
            considered=tuple(CapabilityDimension),
            reasons=(CompatibilityReason.COMPATIBLE,),
        ),
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.REJECT,
            considered=tuple(CapabilityDimension),
            reasons=(CompatibilityReason.SAMPLER_NOT_PROFILE_REFERENCE,),
        ),
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.REJECT,
            considered=tuple(CapabilityDimension),
            reasons=(
                CompatibilityReason.MODEL_FAMILY_MISMATCH,
                CompatibilityReason.MODEL_FAMILY_MISMATCH,
            ),
        ),
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.REJECT,
            considered=(),
            reasons=(CompatibilityReason.MODEL_FAMILY_MISMATCH,),
        ),
        lambda: CompatibilityDecision(
            level=cast(Any, "allow"),
            considered=tuple(CapabilityDimension),
            reasons=(CompatibilityReason.COMPATIBLE,),
        ),
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.REJECT,
            considered=cast(Any, list(CapabilityDimension)),
            reasons=(CompatibilityReason.MODEL_FAMILY_MISMATCH,),
        ),
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.REJECT,
            considered=tuple(reversed(tuple(CapabilityDimension))),
            reasons=(CompatibilityReason.MODEL_FAMILY_MISMATCH,),
        ),
        lambda: CompatibilityDecision(
            level=CompatibilityLevel.REJECT,
            considered=tuple(CapabilityDimension)[:-1],
            reasons=(CompatibilityReason.MODEL_FAMILY_MISMATCH,),
        ),
    ],
)
def test_invalid_compatibility_decisions_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


def test_evaluator_rejects_noncontract_arguments() -> None:
    valid_arguments = {
        "model": _model(),
        "profile": _profile(),
        "sampler": _sampler(),
        "request": ExecutionFeatureRequest(),
    }
    for name, expected in (
        ("model", "ModelCapabilities"),
        ("profile", "ProfileCapabilities"),
        ("sampler", "SamplerCapabilities"),
        ("request", "ExecutionFeatureRequest"),
    ):
        invalid_arguments = {**valid_arguments, name: object()}
        with pytest.raises(ScheduleContractError, match=expected):
            evaluate_compatibility(**cast(Any, invalid_arguments))

    with pytest.raises(ScheduleContractError, match="CompatibilityDecision"):
        require_compatible(cast(Any, object()))
