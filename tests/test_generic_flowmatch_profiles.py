"""M6-03 generic FlowMatch framework-profile contracts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    EvidenceLevel,
    ExecutionBehavior,
    NoiseOwnership,
    PredictionType,
    SamplerState,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    TerminalRequirement,
    TerminalSigma,
    TransformStage,
)
from comfyui_sigmax.profiles import (
    GENERIC_FLOWMATCH_DYNAMIC_PROFILE,
    GENERIC_FLOWMATCH_FIXED_PROFILE,
    GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID,
    GENERIC_FLOWMATCH_PROFILE_SCHEMA_VERSION,
    BaseGridDeclaration,
    GenericFlowMatchProfileV1,
    GenericFlowMatchShiftMode,
    ProfileField,
    SlicingDeclaration,
    TerminalDeclaration,
    TransformDeclaration,
    generic_flowmatch_profile_fingerprint,
    generic_flowmatch_profile_projection,
    generic_flowmatch_profiles,
    resolve_generic_flowmatch_profile,
)
from comfyui_sigmax.profiles.schema_v1 import ProfileSchemaV1

DYNAMIC_PROFILE_FINGERPRINT = "sha256:bcc7e0a083ebda868e01dc7377419706e8f61952db81268580d6634d847a87d4"  # pragma: allowlist secret
FIXED_PROFILE_FINGERPRINT = "sha256:dcecddc3777e8e8442c24d98527eda590d84eab184ca6cca109a6fab5e3c7cb5"  # pragma: allowlist secret


def test_generic_profiles_have_a_separate_frozen_identity_and_exact_catalog() -> None:
    profiles = generic_flowmatch_profiles()

    assert GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID == "sigmax.generic-flowmatch-profile/1"
    assert GENERIC_FLOWMATCH_PROFILE_SCHEMA_VERSION == "1"
    assert profiles == (
        GENERIC_FLOWMATCH_DYNAMIC_PROFILE,
        GENERIC_FLOWMATCH_FIXED_PROFILE,
    )
    assert tuple(profile.profile_id for profile in profiles) == (
        "flowmatch.generic.dynamic",
        "flowmatch.generic.fixed",
    )
    assert all(isinstance(profile, GenericFlowMatchProfileV1) for profile in profiles)
    assert not any(isinstance(profile, ProfileSchemaV1) for profile in profiles)
    assert not hasattr(GENERIC_FLOWMATCH_FIXED_PROFILE, "model_weights")

    for profile in profiles:
        assert (
            resolve_generic_flowmatch_profile(
                profile.profile_id,
                profile.profile_version,
            )
            is profile
        )

    with pytest.raises(FrozenInstanceError):
        GENERIC_FLOWMATCH_FIXED_PROFILE.profile_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("profile_id", "profile_version"),
    [
        ("flowmatch.generic", "1"),
        ("flowmatch.generic.fixed", "2"),
        ("FLOWMATCH.generic.fixed", "1"),
        ("flowmatch.generic.fixed.latest", "1"),
    ],
)
def test_generic_catalog_has_no_alias_latest_prefix_or_fallback_lookup(
    profile_id: str,
    profile_version: str,
) -> None:
    with pytest.raises(ScheduleContractError, match="not registered"):
        resolve_generic_flowmatch_profile(profile_id, profile_version)


def test_fixed_profile_pins_the_framework_static_direct_ratio_contract() -> None:
    profile = GENERIC_FLOWMATCH_FIXED_PROFILE

    assert profile.profile_id == "flowmatch.generic.fixed"
    assert profile.profile_version == "1"
    assert profile.evidence is EvidenceLevel.FRAMEWORK_REFERENCE
    assert profile.shift_mode is GenericFlowMatchShiftMode.FIXED
    assert profile.prediction_type is PredictionType.FLOW_VELOCITY
    assert profile.sigma_domain is SigmaDomain.UNIT_FLOW
    assert profile.ownership is ScheduleOwnership.EXTERNAL_SIGMAS
    assert profile.base_grid.identifier == "diffusers.flowmatch_linear"
    assert profile.base_grid.parameters == (
        ProfileField(name="num_train_timesteps", value=1000),
        ProfileField(name="sigma_end", value=0.001),
        ProfileField(name="sigma_start", value=1.0),
    )
    assert profile.transform.identifier == "diffusers.direct_ratio"
    assert profile.transform.stage is TransformStage.PRIMARY_TIME_SHIFT
    assert profile.transform.parameters == (ProfileField(name="ratio", value=1.0),)
    assert profile.required_inputs == ()
    assert profile.parameters == (ProfileField(name="use_dynamic_shifting", value=False),)


def test_dynamic_profile_requires_explicit_mu_and_stays_experimental() -> None:
    profile = GENERIC_FLOWMATCH_DYNAMIC_PROFILE

    assert profile.profile_id == "flowmatch.generic.dynamic"
    assert profile.evidence is EvidenceLevel.EXPERIMENTAL
    assert profile.shift_mode is GenericFlowMatchShiftMode.DYNAMIC
    assert profile.transform.identifier == "diffusers.exponential_mu"
    assert profile.transform.parameters == (
        ProfileField(name="exponent", value=1.0),
        ProfileField(name="mu", value=None),
        ProfileField(name="time_shift_type", value="exponential"),
    )
    assert profile.required_inputs == ("mu",)
    assert profile.parameters == (
        ProfileField(name="base_image_seq_len", value=256),
        ProfileField(name="base_shift", value=0.5),
        ProfileField(name="max_image_seq_len", value=4096),
        ProfileField(name="max_shift", value=1.15),
        ProfileField(name="use_dynamic_shifting", value=True),
    )
    assert any("mu" in limitation for limitation in profile.known_limitations)


def test_generic_evidence_and_selection_can_never_auto_promote_to_official() -> None:
    allowed = {EvidenceLevel.FRAMEWORK_REFERENCE, EvidenceLevel.EXPERIMENTAL}

    for profile in generic_flowmatch_profiles():
        assert profile.evidence in allowed
        assert profile.evidence is not EvidenceLevel.OFFICIAL
        assert profile.selection.strict_default is True
        assert profile.selection.ambiguity_requires_explicit is True
        assert profile.selection.resolving_sources == ("explicit_selection",)
        assert profile.selection.suggestion_sources == ()
        assert profile.selection.family_only_sources == ()
        assert profile.step_policy == "explicit_positive_integer"
        assert profile.guidance_policy == "model_specific_unset"

    with pytest.raises(ScheduleContractError, match="evidence"):
        replace(GENERIC_FLOWMATCH_FIXED_PROFILE, evidence=EvidenceLevel.OFFICIAL)
    with pytest.raises(ScheduleContractError, match="explicit"):
        replace(
            GENERIC_FLOWMATCH_FIXED_PROFILE,
            selection=replace(
                GENERIC_FLOWMATCH_FIXED_PROFILE.selection,
                resolving_sources=("trusted_profile_metadata",),
            ),
        )


def test_framework_provenance_and_structural_sampler_contract_are_bounded() -> None:
    for profile in generic_flowmatch_profiles():
        framework = profile.framework
        sampler = profile.reference_sampler_capabilities

        assert profile.primary_source_id == framework.framework_id
        assert framework.framework_id == "diffusers.flowmatch.framework"
        assert framework.resource_version == "0.39.0"
        assert (
            framework.revision
            == "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret
        )
        assert framework.locators == (
            "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
        )
        assert sampler.sampler_id == "flowmatch.euler"
        assert sampler.terminal_requirement is TerminalRequirement.REQUIRES_ZERO
        assert sampler.supports_partial_denoise is False
        assert sampler.supports_per_token_timesteps is False
        assert profile.terminal.policy is TerminalPolicy.APPEND_ZERO
        assert profile.terminal.sigma is TerminalSigma.ZERO
        assert profile.terminal.value == 0.0


def test_generic_projection_and_fingerprint_are_deterministic_and_typed() -> None:
    fingerprints: set[str] = set()

    for profile in generic_flowmatch_profiles():
        projection = generic_flowmatch_profile_projection(profile)
        payload = json.dumps(projection, allow_nan=False, separators=(",", ":"), sort_keys=True)
        fingerprint = generic_flowmatch_profile_fingerprint(profile)

        assert projection["schema"] == GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID
        assert projection["schema_version"] == GENERIC_FLOWMATCH_PROFILE_SCHEMA_VERSION
        assert projection["evidence"] in {"framework_reference", "experimental"}
        assert '"precision":"float64"' in payload
        assert fingerprint.startswith("sha256:")
        assert len(fingerprint) == 71
        assert fingerprint == generic_flowmatch_profile_fingerprint(profile)
        fingerprints.add(fingerprint)

    assert len(fingerprints) == 2
    assert generic_flowmatch_profile_fingerprint(GENERIC_FLOWMATCH_DYNAMIC_PROFILE) == (
        DYNAMIC_PROFILE_FINGERPRINT
    )
    assert generic_flowmatch_profile_fingerprint(GENERIC_FLOWMATCH_FIXED_PROFILE) == (
        FIXED_PROFILE_FINGERPRINT
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda profile: replace(profile, schema_id="sigmax.generic-flowmatch-profile/2"),
        lambda profile: replace(profile, profile_id="not_namespaced"),
        lambda profile: replace(profile, profile_version="latest"),
        lambda profile: replace(profile, primary_source_id="other.framework"),
        lambda profile: replace(profile, required_inputs=cast(Any, ["mu"])),
        lambda profile: replace(profile, required_inputs=("mu", "mu")),
        lambda profile: replace(profile, step_policy="implicit_default"),
        lambda profile: replace(profile, guidance_policy="generic_zero"),
    ],
)
def test_generic_profile_contract_rejects_ambiguous_or_noncanonical_content(
    mutation: Any,
) -> None:
    with pytest.raises(ScheduleContractError):
        mutation(GENERIC_FLOWMATCH_FIXED_PROFILE)


def test_mode_specific_contract_rejects_mismatched_shift_semantics() -> None:
    with pytest.raises(ScheduleContractError, match="fixed"):
        replace(
            GENERIC_FLOWMATCH_FIXED_PROFILE,
            required_inputs=("mu",),
        )
    with pytest.raises(ScheduleContractError, match="dynamic"):
        replace(
            GENERIC_FLOWMATCH_DYNAMIC_PROFILE,
            transform=GENERIC_FLOWMATCH_FIXED_PROFILE.transform,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda profile: replace(profile, display_name=""),
        lambda profile: replace(profile, primary_source_id="Bad ID"),
        lambda profile: replace(profile, shift_mode=cast(Any, object())),
        lambda profile: replace(profile, prediction_type=PredictionType.EPSILON),
        lambda profile: replace(profile, sigma_domain=SigmaDomain.CONTINUOUS_EDM),
        lambda profile: replace(profile, ownership=ScheduleOwnership.MODEL_NATIVE),
        lambda profile: replace(profile, framework=cast(Any, object())),
        lambda profile: replace(profile, required_inputs=("Bad ID",)),
        lambda profile: replace(profile, parameters=cast(Any, [])),
        lambda profile: replace(
            profile,
            parameters=(
                ProfileField(name="z_parameter", value=1),
                ProfileField(name="a_parameter", value=2),
            ),
        ),
        lambda profile: replace(profile, known_limitations=()),
        lambda profile: replace(profile, known_limitations=("",)),
        lambda profile: replace(profile, base_grid=cast(Any, object())),
        lambda profile: replace(
            profile,
            base_grid=replace(profile.base_grid, terminal_included=True),
        ),
        lambda profile: replace(profile, transform=cast(Any, object())),
        lambda profile: replace(
            profile,
            transform=TransformDeclaration(
                identifier="diffusers.direct_ratio",
                stage=TransformStage.OPTIONAL_SPACING,
                input_domain=SigmaDomain.UNIT_FLOW,
                output_domain=SigmaDomain.UNIT_FLOW,
                parameters=(ProfileField(name="ratio", value=1.0),),
            ),
        ),
        lambda profile: replace(profile, terminal=cast(Any, object())),
        lambda profile: replace(
            profile,
            terminal=TerminalDeclaration(
                policy=TerminalPolicy.PRESERVE,
                sigma=TerminalSigma.ZERO,
                value=0.0,
            ),
        ),
        lambda profile: replace(profile, slicing=cast(Any, object())),
        lambda profile: replace(profile, reference_sampler_capabilities=cast(Any, object())),
        lambda profile: replace(
            profile,
            reference_sampler_capabilities=replace(
                profile.reference_sampler_capabilities,
                sampler_id="other.euler",
            ),
        ),
        lambda profile: replace(
            profile,
            reference_sampler_capabilities=replace(
                profile.reference_sampler_capabilities,
                execution_behavior=ExecutionBehavior.STOCHASTIC,
                noise_ownership=NoiseOwnership.SAMPLER,
            ),
        ),
        lambda profile: replace(
            profile,
            reference_sampler_capabilities=replace(
                profile.reference_sampler_capabilities,
                required_state=(SamplerState.BEGIN_INDEX,),
            ),
        ),
    ],
)
def test_generic_profile_rejects_invalid_boundary_components(mutation: Any) -> None:
    with pytest.raises(ScheduleContractError):
        mutation(GENERIC_FLOWMATCH_FIXED_PROFILE)


def test_generic_projection_requires_the_generic_contract() -> None:
    with pytest.raises(ScheduleContractError, match="requires a profile"):
        generic_flowmatch_profile_projection(cast(Any, object()))


def test_generic_profile_accepts_equivalent_immutable_declarations() -> None:
    assert replace(GENERIC_FLOWMATCH_FIXED_PROFILE) == GENERIC_FLOWMATCH_FIXED_PROFILE
    assert replace(GENERIC_FLOWMATCH_DYNAMIC_PROFILE) == GENERIC_FLOWMATCH_DYNAMIC_PROFILE
    assert isinstance(GENERIC_FLOWMATCH_FIXED_PROFILE.base_grid, BaseGridDeclaration)
    assert isinstance(GENERIC_FLOWMATCH_FIXED_PROFILE.slicing, SlicingDeclaration)
