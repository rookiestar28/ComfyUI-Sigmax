"""M6-07 profile, model, host, and sampler capability resolution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    CompatibilityLevel,
    ExecutionFeatureRequest,
    ScheduleContractError,
    ScheduleOwnership,
)
from comfyui_sigmax.profiles import (
    CAPABILITY_RESOLUTION_SCHEMA_ID,
    CAPABILITY_RESOLUTION_SCHEMA_VERSION,
    HostCapabilities,
    HostCapabilityEvidence,
    HostCapabilityLifecycle,
    HostCapabilityRequirement,
    ModelCapabilityEvidence,
    ModelIdentityEvidence,
    ModelIdentityStatus,
    ProfileCapabilityDecision,
    builtin_profile_registry,
    model_identity_from_krea2_resolution,
    resolve_krea2_variant,
    resolve_profile_capabilities,
)
from comfyui_sigmax.profiles.krea2_variant import Krea2VariantResolutionStatus


def _profile(variant: str = "turbo") -> Any:
    registry = builtin_profile_registry()
    return next(entry for entry in registry.entries if entry.schema.model_variant == variant)


def _confirmed_identity(variant: str = "turbo") -> ModelIdentityEvidence:
    return ModelIdentityEvidence(
        evidence_version="1",
        model_family="krea2",
        status=ModelIdentityStatus.CONFIRMED,
        confirmed_variant=variant,
        suggested_variant=None,
        confidence="authoritative",
        decisive_source="explicit_selection",
        reason_codes=("explicit_variant_selected",),
    )


def _model_evidence(
    *,
    profile: Any | None = None,
    identity: ModelIdentityEvidence | None = None,
) -> ModelCapabilityEvidence:
    selected = profile or _profile()
    return ModelCapabilityEvidence(
        evidence_version="1",
        identity=identity or _confirmed_identity(selected.schema.model_variant),
        capabilities=selected.schema.model_capabilities,
    )


def _host(
    *lifecycles: tuple[str, HostCapabilityLifecycle],
) -> HostCapabilities:
    values = lifecycles or (
        ("sampler.comfy.euler", HostCapabilityLifecycle.LANDED),
        ("schedule.external_sigmas", HostCapabilityLifecycle.LANDED),
    )
    return HostCapabilities(
        evidence_version="1",
        host_id="comfyui",
        host_version="0.29.0",
        host_revision="e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
        capabilities=tuple(
            HostCapabilityEvidence(
                capability_id=capability_id,
                lifecycle=lifecycle,
            )
            for capability_id, lifecycle in sorted(values)
        ),
    )


def _resolve(
    *,
    profile: Any | None = None,
    model: ModelCapabilityEvidence | None = None,
    host: HostCapabilities | None = None,
    sampler: Any | None = None,
    request: ExecutionFeatureRequest | None = None,
) -> ProfileCapabilityDecision:
    selected = profile or _profile()
    return resolve_profile_capabilities(
        registered_profile=selected,
        model=model or _model_evidence(profile=selected),
        host=host or _host(),
        sampler=sampler or selected.schema.reference_sampler_capabilities,
        request=request or ExecutionFeatureRequest(),
    )


def test_landed_reference_capabilities_allow_without_sampling() -> None:
    profile = _profile()

    decision = _resolve(profile=profile)

    assert decision.schema_id == CAPABILITY_RESOLUTION_SCHEMA_ID
    assert decision.schema_version == CAPABILITY_RESOLUTION_SCHEMA_VERSION == "1"
    assert decision.level is CompatibilityLevel.ALLOW
    assert decision.reason_codes == ("core.compatible",)
    assert decision.profile_key == profile.key.canonical
    assert decision.profile_fingerprint == profile.fingerprint
    assert decision.model_identity.status is ModelIdentityStatus.CONFIRMED
    assert tuple(item.capability_id for item in decision.host_requirements) == (
        "sampler.comfy.euler",
        "schedule.external_sigmas",
    )
    assert all(item.satisfied for item in decision.host_requirements)
    assert decision.core_decision.level is CompatibilityLevel.ALLOW


def test_nonreference_sampler_propagates_core_warning() -> None:
    profile = _profile()
    sampler = replace(
        profile.schema.reference_sampler_capabilities,
        sampler_id="other.euler",
    )
    host = _host(
        ("sampler.other.euler", HostCapabilityLifecycle.LANDED),
        ("schedule.external_sigmas", HostCapabilityLifecycle.LANDED),
    )

    decision = _resolve(profile=profile, sampler=sampler, host=host)

    assert decision.level is CompatibilityLevel.WARN
    assert decision.reason_codes == ("core.sampler_not_profile_reference",)


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (ModelIdentityStatus.SUGGESTED, "model.identity_suggested"),
        (ModelIdentityStatus.AMBIGUOUS, "model.identity_ambiguous"),
        (ModelIdentityStatus.CONFLICT, "model.identity_conflict"),
        (ModelIdentityStatus.UNKNOWN, "model.identity_unknown"),
    ],
)
def test_unconfirmed_identity_rejects_even_when_capability_variant_matches(
    status: ModelIdentityStatus,
    reason: str,
) -> None:
    suggested = "turbo" if status is ModelIdentityStatus.SUGGESTED else None
    identity = ModelIdentityEvidence(
        evidence_version="1",
        model_family="krea2",
        status=status,
        confirmed_variant=None,
        suggested_variant=suggested,
        confidence="weak" if suggested else "none",
        decisive_source="filename_signal" if suggested else None,
        reason_codes=(f"identity_{status.value}",),
    )

    decision = _resolve(model=_model_evidence(identity=identity))

    assert decision.level is CompatibilityLevel.REJECT
    assert decision.reason_codes == (reason,)
    assert decision.core_decision.level is CompatibilityLevel.ALLOW


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        (
            replace(_confirmed_identity(), model_family="other"),
            "model.family_mismatch",
        ),
        (
            replace(_confirmed_identity(), confirmed_variant="raw"),
            "model.variant_mismatch",
        ),
    ],
)
def test_confirmed_identity_mismatch_rejects_with_stable_reason(
    identity: ModelIdentityEvidence,
    expected: str,
) -> None:
    decision = _resolve(model=_model_evidence(identity=identity))

    assert decision.level is CompatibilityLevel.REJECT
    assert expected in decision.reason_codes


@pytest.mark.parametrize(
    ("host", "reason", "lifecycle"),
    [
        (
            _host(("schedule.external_sigmas", HostCapabilityLifecycle.LANDED)),
            "host.capability_missing",
            None,
        ),
        (
            _host(
                ("sampler.comfy.euler", HostCapabilityLifecycle.EXPERIMENTAL),
                ("schedule.external_sigmas", HostCapabilityLifecycle.LANDED),
            ),
            "host.capability_experimental",
            HostCapabilityLifecycle.EXPERIMENTAL,
        ),
        (
            _host(
                ("sampler.comfy.euler", HostCapabilityLifecycle.UNSUPPORTED),
                ("schedule.external_sigmas", HostCapabilityLifecycle.LANDED),
            ),
            "host.capability_unsupported",
            HostCapabilityLifecycle.UNSUPPORTED,
        ),
    ],
)
def test_required_host_capability_must_be_landed(
    host: HostCapabilities,
    reason: str,
    lifecycle: HostCapabilityLifecycle | None,
) -> None:
    decision = _resolve(host=host)

    assert decision.level is CompatibilityLevel.REJECT
    assert decision.reason_codes == (reason,)
    sampler_requirement = decision.host_requirements[0]
    assert sampler_requirement.capability_id == "sampler.comfy.euler"
    assert sampler_requirement.lifecycle is lifecycle
    assert sampler_requirement.satisfied is False
    assert sampler_requirement.reason_code == reason


@pytest.mark.parametrize(
    ("feature_request", "capability_id"),
    [
        (ExecutionFeatureRequest(use_partial_denoise=True), "execution.partial_denoise"),
        (ExecutionFeatureRequest(use_per_token_timesteps=True), "execution.per_token_timesteps"),
    ],
)
def test_requested_execution_features_add_host_requirements(
    feature_request: ExecutionFeatureRequest,
    capability_id: str,
) -> None:
    profile = _profile("raw")
    host = _host(
        ("sampler.comfy.euler", HostCapabilityLifecycle.LANDED),
        ("schedule.external_sigmas", HostCapabilityLifecycle.LANDED),
    )

    decision = _resolve(profile=profile, host=host, request=feature_request)

    assert decision.level is CompatibilityLevel.REJECT
    assert "host.capability_missing" in decision.reason_codes
    assert capability_id in tuple(item.capability_id for item in decision.host_requirements)


def test_invalid_sampler_contracts_fail_before_resolution() -> None:
    profile = _profile()

    with pytest.raises(ScheduleContractError):
        replace(
            profile.schema.reference_sampler_capabilities,
            accepted_prediction_types=(),
        )

    with pytest.raises(ScheduleContractError):
        replace(
            profile.schema.reference_sampler_capabilities,
            terminal_requirement=cast(Any, object()),
        )


def test_valid_core_rejection_is_namespaced_and_preserved() -> None:
    from comfyui_sigmax.core import TerminalRequirement

    profile = _profile()
    sampler = replace(
        profile.schema.reference_sampler_capabilities,
        terminal_requirement=TerminalRequirement.FORBIDS_ZERO,
    )

    decision = _resolve(profile=profile, sampler=sampler)

    assert decision.level is CompatibilityLevel.REJECT
    assert decision.reason_codes == ("core.terminal_requirement_mismatch",)


def test_krea2_resolution_bridge_preserves_all_statuses() -> None:
    resolved = resolve_krea2_variant(explicit_variant="turbo")
    suggested = resolve_krea2_variant(
        strict_official=False,
        filename="krea_2_turbo.safetensors",
    )
    ambiguous = resolve_krea2_variant(
        strict_official=False,
        filename="krea_2.safetensors",
    )
    conflict = resolve_krea2_variant(
        strict_official=False,
        explicit_variant="turbo",
        trusted_profile_id="krea2.raw.official",
    )

    mapped = tuple(
        model_identity_from_krea2_resolution(item)
        for item in (resolved, suggested, ambiguous, conflict)
    )

    assert tuple(item.status for item in mapped) == (
        ModelIdentityStatus.CONFIRMED,
        ModelIdentityStatus.SUGGESTED,
        ModelIdentityStatus.AMBIGUOUS,
        ModelIdentityStatus.CONFLICT,
    )
    assert mapped[0].confirmed_variant == "turbo"
    assert mapped[1].suggested_variant == "turbo"
    assert mapped[2].confidence == "none"
    assert conflict.status is Krea2VariantResolutionStatus.CONFLICT
    assert mapped[3].confirmed_variant is None


def test_resolution_contracts_are_immutable() -> None:
    decision = _resolve()

    with pytest.raises((FrozenInstanceError, AttributeError)):
        decision.level = CompatibilityLevel.REJECT  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        decision.host_requirements[0].satisfied = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ModelIdentityEvidence(
            evidence_version="2",
            model_family="krea2",
            status=ModelIdentityStatus.UNKNOWN,
            confirmed_variant=None,
            suggested_variant=None,
            confidence="none",
            decisive_source=None,
            reason_codes=("unknown",),
        ),
        lambda: replace(_confirmed_identity(), confirmed_variant=None),
        lambda: replace(_confirmed_identity(), suggested_variant="raw"),
        lambda: replace(
            _confirmed_identity(),
            reason_codes=("z_reason", "a_reason"),
        ),
        lambda: HostCapabilities(
            evidence_version="1",
            host_id="comfyui",
            host_version="0.29.0",
            host_revision="not-a-revision",
            capabilities=(),
        ),
        lambda: HostCapabilities(
            evidence_version="1",
            host_id="comfyui",
            host_version="0.29.0",
            host_revision="e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
            capabilities=(
                HostCapabilityEvidence(
                    capability_id="x.capability",
                    lifecycle=HostCapabilityLifecycle.LANDED,
                ),
                HostCapabilityEvidence(
                    capability_id="x.capability",
                    lifecycle=HostCapabilityLifecycle.LANDED,
                ),
            ),
        ),
        lambda: HostCapabilityEvidence(
            capability_id="Bad Capability",
            lifecycle=HostCapabilityLifecycle.LANDED,
        ),
        lambda: HostCapabilityEvidence(
            capability_id="x.capability",
            lifecycle=cast(Any, "landed"),
        ),
    ],
)
def test_evidence_contracts_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registered_profile", object()),
        ("model", object()),
        ("host", object()),
        ("sampler", object()),
        ("request", object()),
    ],
)
def test_resolver_rejects_runtime_type_confusion(field: str, value: object) -> None:
    profile = _profile()
    arguments: dict[str, object] = {
        "registered_profile": profile,
        "model": _model_evidence(profile=profile),
        "host": _host(),
        "sampler": profile.schema.reference_sampler_capabilities,
        "request": ExecutionFeatureRequest(),
    }
    arguments[field] = value

    with pytest.raises(ScheduleContractError):
        resolve_profile_capabilities(**cast(Any, arguments))


def test_host_capabilities_require_canonical_order() -> None:
    with pytest.raises(ScheduleContractError):
        HostCapabilities(
            evidence_version="1",
            host_id="comfyui",
            host_version="0.29.0",
            host_revision="e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
            capabilities=(
                HostCapabilityEvidence(
                    capability_id="schedule.external_sigmas",
                    lifecycle=HostCapabilityLifecycle.LANDED,
                ),
                HostCapabilityEvidence(
                    capability_id="sampler.comfy.euler",
                    lifecycle=HostCapabilityLifecycle.LANDED,
                ),
            ),
        )


def test_model_capability_evidence_requires_exact_types() -> None:
    with pytest.raises(ScheduleContractError):
        ModelCapabilityEvidence(
            evidence_version="1",
            identity=cast(Any, object()),
            capabilities=_profile().schema.model_capabilities,
        )
    with pytest.raises(ScheduleContractError):
        ModelCapabilityEvidence(
            evidence_version="1",
            identity=_confirmed_identity(),
            capabilities=cast(Any, object()),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_confirmed_identity(), status=cast(Any, "confirmed")),
        lambda: replace(_confirmed_identity(), reason_codes=cast(Any, ["reason"])),
        lambda: replace(_confirmed_identity(), reason_codes=()),
        lambda: replace(_confirmed_identity(), reason_codes=("same", "same")),
        lambda: replace(_confirmed_identity(), confidence="Bad Value"),
        lambda: replace(_confirmed_identity(), suggested_variant="Bad Value"),
        lambda: replace(_confirmed_identity(), decisive_source="Bad Value"),
        lambda: replace(
            _confirmed_identity(),
            status=ModelIdentityStatus.SUGGESTED,
            confirmed_variant="turbo",
            suggested_variant="raw",
            confidence="weak",
            decisive_source="filename_signal",
        ),
        lambda: replace(
            _confirmed_identity(),
            status=ModelIdentityStatus.SUGGESTED,
            confirmed_variant=None,
            suggested_variant=None,
            confidence="weak",
            decisive_source="filename_signal",
        ),
        lambda: replace(
            _confirmed_identity(),
            status=ModelIdentityStatus.SUGGESTED,
            confirmed_variant=None,
            suggested_variant="turbo",
            confidence="none",
            decisive_source="filename_signal",
        ),
        lambda: replace(
            _confirmed_identity(),
            status=ModelIdentityStatus.SUGGESTED,
            confirmed_variant=None,
            suggested_variant="turbo",
            confidence="weak",
            decisive_source=None,
        ),
        lambda: replace(
            _confirmed_identity(),
            status=ModelIdentityStatus.UNKNOWN,
            confirmed_variant=None,
            confidence="weak",
            decisive_source=None,
        ),
    ],
)
def test_model_identity_contract_rejects_malformed_shapes(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_model_evidence(), evidence_version="2"),
        lambda: replace(_host(), evidence_version="2"),
        lambda: replace(_host(), host_id="Bad Host"),
        lambda: replace(_host(), host_version=""),
        lambda: replace(_host(), host_version=cast(Any, 29)),
        lambda: replace(_host(), capabilities=cast(Any, [])),
        lambda: replace(_host(), capabilities=cast(Any, (object(),))),
    ],
)
def test_model_and_host_container_contracts_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HostCapabilityRequirement(
            capability_id="x.capability",
            lifecycle=cast(Any, "landed"),
            satisfied=True,
            reason_code=None,
        ),
        lambda: HostCapabilityRequirement(
            capability_id="x.capability",
            lifecycle=HostCapabilityLifecycle.LANDED,
            satisfied=cast(Any, 1),
            reason_code=None,
        ),
        lambda: HostCapabilityRequirement(
            capability_id="x.capability",
            lifecycle=None,
            satisfied=False,
            reason_code="Bad Reason",
        ),
        lambda: HostCapabilityRequirement(
            capability_id="x.capability",
            lifecycle=None,
            satisfied=True,
            reason_code=None,
        ),
        lambda: HostCapabilityRequirement(
            capability_id="x.capability",
            lifecycle=HostCapabilityLifecycle.LANDED,
            satisfied=True,
            reason_code="unexpected_reason",
        ),
    ],
)
def test_host_requirement_contract_rejects_inconsistent_shapes(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_id", "other.schema/1"),
        ("schema_version", "2"),
        ("profile_key", "not-canonical"),
        ("profile_key", "bad@latest"),
        ("profile_key", cast(Any, object())),
        ("profile_fingerprint", "not-a-fingerprint"),
        ("profile_fingerprint", "sha256:abc"),
        ("profile_fingerprint", cast(Any, object())),
        ("level", "allow"),
        ("reason_codes", ()),
        ("reason_codes", ("unknown.reason",)),
        ("model_identity", object()),
        ("host_id", "Bad Host"),
        ("host_version", ""),
        ("host_version", 29),
        ("host_revision", "not-a-revision"),
        ("host_requirements", []),
        ("host_requirements", (object(),)),
        (
            "host_requirements",
            (
                HostCapabilityRequirement(
                    capability_id="z.capability",
                    lifecycle=None,
                    satisfied=False,
                    reason_code="host.capability_missing",
                ),
                HostCapabilityRequirement(
                    capability_id="a.capability",
                    lifecycle=None,
                    satisfied=False,
                    reason_code="host.capability_missing",
                ),
            ),
        ),
        (
            "host_requirements",
            (
                HostCapabilityRequirement(
                    capability_id="x.capability",
                    lifecycle=None,
                    satisfied=False,
                    reason_code="host.capability_missing",
                ),
                HostCapabilityRequirement(
                    capability_id="x.capability",
                    lifecycle=None,
                    satisfied=False,
                    reason_code="host.capability_missing",
                ),
            ),
        ),
        ("core_decision", object()),
    ],
)
def test_decision_contract_rejects_malformed_shapes(field: str, value: object) -> None:
    decision = _resolve()

    with pytest.raises(ScheduleContractError):
        replace(decision, **cast(Any, {field: value}))


@pytest.mark.parametrize(
    ("level", "reason_codes"),
    [
        (CompatibilityLevel.ALLOW, ("core.sampler_not_profile_reference",)),
        (CompatibilityLevel.WARN, ("core.compatible",)),
        (CompatibilityLevel.REJECT, ("core.compatible",)),
        (CompatibilityLevel.REJECT, ("core.sampler_not_profile_reference",)),
    ],
)
def test_decision_level_must_match_reason_semantics(
    level: CompatibilityLevel,
    reason_codes: tuple[str, ...],
) -> None:
    decision = _resolve()

    with pytest.raises(ScheduleContractError):
        replace(decision, level=level, reason_codes=reason_codes)


def test_krea_bridge_rejects_runtime_type_confusion() -> None:
    with pytest.raises(ScheduleContractError):
        model_identity_from_krea2_resolution(cast(Any, object()))


def test_nonexternal_profile_omits_external_sigma_host_requirement() -> None:
    from comfyui_sigmax.profiles.resolution import _required_host_capability_ids

    profile = _profile()
    fake_profile = SimpleNamespace(
        schema=SimpleNamespace(
            profile_capabilities=replace(
                profile.schema.profile_capabilities,
                ownership=ScheduleOwnership.MODEL_NATIVE,
            )
        )
    )

    required = _required_host_capability_ids(
        registered_profile=cast(Any, fake_profile),
        sampler=profile.schema.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )

    assert required == ("sampler.comfy.euler",)
