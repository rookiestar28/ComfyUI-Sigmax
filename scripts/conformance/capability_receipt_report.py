"""Strict M7-07 capability, receipt, and host-attempt conformance evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final, cast

from comfyui_sigmax.core import (
    ArtifactBuildMetadata,
    ArtifactField,
    CapabilityDimension,
    CompatibilityDecision,
    ExecutionBehavior,
    ExecutionComponent,
    ExecutionFeatureRequest,
    ExecutionHost,
    ExecutionReceiptMetadata,
    ExecutionRngOwnership,
    ExecutionStatus,
    ModelCapabilities,
    NoiseOwnership,
    PortableExecutionBundle,
    ProfileCapabilities,
    SamplerCapabilities,
    SamplerState,
    ScheduleOwnership,
    TerminalRequirement,
    TypedArtifactValue,
    build_execution_receipt,
    build_schedule_artifact,
    canonical_projection_bytes,
    deserialize_execution_receipt,
    deserialize_portable_execution_bundle,
    deserialize_schedule_artifact,
    evaluate_compatibility,
    serialize_execution_receipt,
    serialize_portable_execution_bundle,
    serialize_schedule_artifact,
)
from comfyui_sigmax.profiles import (
    KREA2_TURBO_PROFILE,
    KREA2_TURBO_SCHEMA,
    build_krea2_turbo_schedule,
    profile_schema_fingerprint,
)
from comfyui_sigmax.workflows.validation import (
    CANONICAL_HOST_REVISION,
    CANONICAL_HOST_VERSION,
)

REPORT_SCHEMA: Final = "sigmax.capability-receipt-conformance/1"
HOST_ATTEMPT_SCHEMA: Final = "sigmax.host-attempt-transition/1"
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_REASON_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_STATUS_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_LANE_PATTERN: Final = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_ROOT_FIELDS: Final = frozenset(
    {
        "cases",
        "coverage",
        "host_attempt_regressions",
        "portable_execution",
        "profile",
        "schema",
        "status",
    }
)
_TRANSITION_FIELDS: Final = frozenset(
    {"accepted", "first", "lane", "repeat", "schema", "transition"}
)
_ATTEMPT_FIELDS: Final = frozenset(
    {"observed_status", "ordinal", "reason_code", "result_fingerprint", "verdict"}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class HostAttempt:
    """One bounded non-pure host conformance attempt."""

    ordinal: int
    verdict: str
    observed_status: str
    reason_code: str | None
    result_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal not in (1, 2):
            raise ValueError("host attempt ordinal must be one or two")
        if not isinstance(self.verdict, str) or self.verdict not in {"pass", "fail"}:
            raise ValueError("host attempt verdict is unsupported")
        if not isinstance(self.observed_status, str) or not _STATUS_PATTERN.fullmatch(
            self.observed_status
        ):
            raise ValueError("host attempt observed status is invalid")
        if self.reason_code is not None and (
            not isinstance(self.reason_code, str) or not _REASON_PATTERN.fullmatch(self.reason_code)
        ):
            raise ValueError("host attempt reason code is invalid")
        if self.verdict == "fail" and self.reason_code is None:
            raise ValueError("failed host attempt requires a reason code")
        if not isinstance(self.result_fingerprint, str) or not _SHA256_PATTERN.fullmatch(
            self.result_fingerprint
        ):
            raise ValueError("host attempt result fingerprint is invalid")

    def projection(self) -> dict[str, object]:
        return {
            "observed_status": self.observed_status,
            "ordinal": self.ordinal,
            "reason_code": self.reason_code,
            "result_fingerprint": self.result_fingerprint,
            "verdict": self.verdict,
        }


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return dict(cast(Mapping[str, object], value))


def _exact(value: Mapping[str, object], fields: frozenset[str], *, name: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{name} fields are incomplete or unknown")


def _attempt_from_projection(value: object, *, ordinal: int) -> HostAttempt:
    projection = _mapping(value, name="host attempt")
    _exact(projection, _ATTEMPT_FIELDS, name="host attempt")
    attempt = HostAttempt(
        ordinal=cast(int, projection["ordinal"]),
        verdict=cast(str, projection["verdict"]),
        observed_status=cast(str, projection["observed_status"]),
        reason_code=cast(str | None, projection["reason_code"]),
        result_fingerprint=cast(str, projection["result_fingerprint"]),
    )
    if attempt.ordinal != ordinal:
        raise ValueError("host attempt ordinal does not match its position")
    return attempt


def build_host_attempt_transition(
    *,
    lane: str,
    first: HostAttempt,
    repeat: HostAttempt,
) -> dict[str, object]:
    """Build first-attempt/repeat evidence without retry masking."""

    if not isinstance(lane, str) or not _LANE_PATTERN.fullmatch(lane):
        raise ValueError("host attempt lane is invalid")
    if not isinstance(first, HostAttempt) or not isinstance(repeat, HostAttempt):
        raise TypeError("host attempts must use HostAttempt")
    if first.ordinal != 1 or repeat.ordinal != 2:
        raise ValueError("host attempts must retain first/repeat order")
    unchanged = (
        first.result_fingerprint == repeat.result_fingerprint
        and first.observed_status == repeat.observed_status
        and first.reason_code == repeat.reason_code
    )
    transition = f"{first.verdict}_to_{repeat.verdict}"
    if first.verdict == repeat.verdict == "pass" and not unchanged:
        transition = "pass_to_pass_changed"
    accepted = first.verdict == repeat.verdict == "pass" and unchanged
    return {
        "accepted": accepted,
        "first": first.projection(),
        "lane": lane,
        "repeat": repeat.projection(),
        "schema": HOST_ATTEMPT_SCHEMA,
        "transition": transition,
    }


def validate_host_attempt_transition(value: object) -> dict[str, object]:
    """Validate one host attempt transition."""

    projection = _mapping(value, name="host attempt transition")
    _exact(projection, _TRANSITION_FIELDS, name="host attempt transition")
    if projection.get("schema") != HOST_ATTEMPT_SCHEMA:
        raise ValueError("host attempt transition schema is unsupported")
    rebuilt = build_host_attempt_transition(
        lane=cast(str, projection["lane"]),
        first=_attempt_from_projection(projection["first"], ordinal=1),
        repeat=_attempt_from_projection(projection["repeat"], ordinal=2),
    )
    if projection != rebuilt:
        raise ValueError("host attempt transition evidence is inconsistent")
    return cast(dict[str, object], value)


def _decision_projection(decision: CompatibilityDecision) -> dict[str, object]:
    if decision.considered != tuple(CapabilityDimension):
        raise ValueError("compatibility dimensions are incomplete or reordered")
    return {
        "considered": [item.value for item in decision.considered],
        "level": decision.level.value,
        "reasons": [item.value for item in decision.reasons],
    }


def _capability_case(
    identifier: str,
    *,
    model: ModelCapabilities | None = None,
    profile: ProfileCapabilities | None = None,
    sampler: SamplerCapabilities | None = None,
    request: ExecutionFeatureRequest | None = None,
) -> dict[str, object]:
    effective_model = model or KREA2_TURBO_SCHEMA.model_capabilities
    effective_profile = profile or KREA2_TURBO_SCHEMA.profile_capabilities
    effective_sampler = sampler or KREA2_TURBO_SCHEMA.reference_sampler_capabilities
    effective_request = request or ExecutionFeatureRequest()
    decision = evaluate_compatibility(
        model=effective_model,
        profile=effective_profile,
        sampler=effective_sampler,
        request=effective_request,
    )
    return {
        "decision": _decision_projection(decision),
        "id": identifier,
        "semantics": {
            "execution_behavior": effective_sampler.execution_behavior.value,
            "noise_ownership": effective_sampler.noise_ownership.value,
            "partial_denoise_requested": effective_request.use_partial_denoise,
            "per_token_timesteps_requested": effective_request.use_per_token_timesteps,
            "required_state": [item.value for item in effective_sampler.required_state],
            "sampler_id": effective_sampler.sampler_id,
            "schedule_ownership": effective_profile.ownership.name.casefold(),
            "terminal_requirement": effective_sampler.terminal_requirement.value,
        },
    }


def _capability_cases() -> list[dict[str, object]]:
    model = KREA2_TURBO_SCHEMA.model_capabilities
    sampler = KREA2_TURBO_SCHEMA.reference_sampler_capabilities
    return sorted(
        [
            _capability_case(
                "model_owned_sigmas_reject",
                model=replace(
                    model,
                    accepted_ownerships=(ScheduleOwnership.MODEL_NATIVE,),
                ),
            ),
            _capability_case("native_euler_allow"),
            _capability_case(
                "nonreference_euler_warn",
                sampler=replace(sampler, sampler_id="fixture.nonreference-euler"),
            ),
            _capability_case(
                "partial_denoise_reject",
                sampler=replace(sampler, supports_partial_denoise=False),
                request=ExecutionFeatureRequest(use_partial_denoise=True),
            ),
            _capability_case(
                "per_token_timesteps_reject",
                request=ExecutionFeatureRequest(use_per_token_timesteps=True),
            ),
            _capability_case(
                "resume_state_reject",
                sampler=replace(sampler, required_state=(SamplerState.RESUME,)),
            ),
            _capability_case(
                "stochastic_noise_reject",
                sampler=replace(
                    sampler,
                    execution_behavior=ExecutionBehavior.STOCHASTIC,
                    noise_ownership=NoiseOwnership.SAMPLER,
                ),
            ),
            _capability_case(
                "terminal_zero_reject",
                sampler=replace(
                    sampler,
                    terminal_requirement=TerminalRequirement.FORBIDS_ZERO,
                ),
            ),
        ],
        key=lambda item: cast(str, item["id"]),
    )


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _component(
    *,
    identifier: str,
    version: str,
    projection: Mapping[str, object],
) -> ExecutionComponent:
    return ExecutionComponent(
        identifier=identifier,
        version=version,
        fingerprint=_sha256(canonical_projection_bytes(dict(projection))),
    )


def _portable_execution() -> dict[str, object]:
    decision = evaluate_compatibility(
        model=KREA2_TURBO_SCHEMA.model_capabilities,
        profile=KREA2_TURBO_SCHEMA.profile_capabilities,
        sampler=KREA2_TURBO_SCHEMA.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )
    schedule = build_krea2_turbo_schedule(steps=8, width=1024, height=1024)
    artifact = build_schedule_artifact(
        schedule,
        metadata=ArtifactBuildMetadata(
            source_id=KREA2_TURBO_SCHEMA.primary_source_id,
            source_label=KREA2_TURBO_SCHEMA.display_name,
            base_grid_parameters=(ArtifactField(name="steps", value=8),),
            transform_parameters=(
                (
                    ArtifactField(
                        name="mu",
                        value=TypedArtifactValue(
                            value=KREA2_TURBO_PROFILE.fixed_mu,
                            precision="float64",
                        ),
                    ),
                ),
                (),
            ),
            compatibility=(
                ArtifactField(name="decision", value=decision.level.value),
                ArtifactField(name="double_shift", value=False),
                ArtifactField(
                    name="reference_sampler",
                    value=KREA2_TURBO_SCHEMA.reference_sampler_capabilities.sampler_id,
                ),
            ),
        ),
        precision="float64",
    )
    receipt = build_execution_receipt(
        artifact,
        metadata=ExecutionReceiptMetadata(
            compatibility=decision,
            host=ExecutionHost(
                identifier="comfyui",
                version=CANONICAL_HOST_VERSION,
                revision=CANONICAL_HOST_REVISION,
                api_version="legacy_v1",
            ),
            model=_component(
                identifier="sigmax.controlled-flow-model",
                version="1",
                projection={"fixture": "m7-07", "profile": KREA2_TURBO_SCHEMA.profile_id},
            ),
            sampler=_component(
                identifier=KREA2_TURBO_SCHEMA.reference_sampler_capabilities.sampler_id,
                version=CANONICAL_HOST_VERSION,
                projection={
                    "fixture": "m7-07",
                    "sampler": KREA2_TURBO_SCHEMA.reference_sampler_capabilities.sampler_id,
                },
            ),
            rng_ownership=ExecutionRngOwnership(
                schedule=NoiseOwnership.NONE,
                model=NoiseOwnership.NONE,
                sampler=NoiseOwnership.NONE,
            ),
            requested_transitions=8,
            effective_transitions=8,
            requested_model_evaluations=8,
            effective_model_evaluations=8,
            status=ExecutionStatus.SUCCEEDED,
        ),
    )
    bundle = PortableExecutionBundle(artifact=artifact, receipt=receipt)
    artifact_payload = serialize_schedule_artifact(artifact)
    receipt_payload = serialize_execution_receipt(receipt)
    bundle_payload = serialize_portable_execution_bundle(bundle)
    artifact_round_trip = deserialize_schedule_artifact(artifact_payload)
    receipt_round_trip = deserialize_execution_receipt(receipt_payload)
    bundle_round_trip = deserialize_portable_execution_bundle(bundle_payload)
    construction = artifact.construction_projection()
    transforms = cast(list[dict[str, object]], construction["transforms"])
    shift_count = sum(item["id"] == "krea.exponential_mu" for item in transforms)
    receipt_projection = receipt.projection()
    effective = _mapping(receipt_projection["effective_inputs"], name="effective inputs")
    return {
        "artifact": {
            "construction_fingerprint": artifact.construction_fingerprint,
            "numerical_fingerprint": artifact.numerical_fingerprint,
        },
        "artifact_round_trip": artifact_round_trip == artifact,
        "bundle_fingerprint": _sha256(bundle_payload),
        "bundle_round_trip": (
            bundle_round_trip == bundle
            and serialize_portable_execution_bundle(bundle_round_trip) == bundle_payload
        ),
        "counts": receipt_projection["counts"],
        "double_shift": False,
        "effective_inputs": {
            field: effective[field]
            for field in (
                "height",
                "precision",
                "profile",
                "profile_version",
                "steps",
                "width",
            )
        },
        "receipt_fingerprint": receipt.receipt_fingerprint,
        "receipt_round_trip": (
            receipt_round_trip == receipt
            and serialize_execution_receipt(receipt_round_trip) == receipt_payload
        ),
        "rng_ownership": receipt_projection["rng_ownership"],
        "shift_count": shift_count,
        "status": _mapping(receipt_projection["execution"], name="execution")["status"],
    }


def _fingerprint_projection(value: Mapping[str, object]) -> str:
    return _sha256(canonical_projection_bytes(value))


def _host_attempt_regressions() -> list[dict[str, object]]:
    stable_rejection = _fingerprint_projection(
        {"reason_code": "execution.partial_denoise_unsupported", "status": "error"}
    )
    return [
        {
            "id": "first_failure_then_pass",
            **build_host_attempt_transition(
                lane="H3_EULER_M5_01",
                first=HostAttempt(
                    ordinal=1,
                    verdict="fail",
                    observed_status="error",
                    reason_code="host.first_attempt_failure",
                    result_fingerprint=_fingerprint_projection({"attempt": 1, "status": "error"}),
                ),
                repeat=HostAttempt(
                    ordinal=2,
                    verdict="pass",
                    observed_status="succeeded",
                    reason_code=None,
                    result_fingerprint=_fingerprint_projection(
                        {"attempt": 2, "status": "succeeded"}
                    ),
                ),
            ),
        },
        {
            "id": "repeat_result_drift",
            **build_host_attempt_transition(
                lane="H2_TURBO_M2_05",
                first=HostAttempt(
                    ordinal=1,
                    verdict="pass",
                    observed_status="not_executed",
                    reason_code=None,
                    result_fingerprint=_fingerprint_projection({"schedule": "first"}),
                ),
                repeat=HostAttempt(
                    ordinal=2,
                    verdict="pass",
                    observed_status="not_executed",
                    reason_code=None,
                    result_fingerprint=_fingerprint_projection({"schedule": "changed"}),
                ),
            ),
        },
        {
            "id": "stable_expected_rejection",
            **build_host_attempt_transition(
                lane="H3_EULER_M5_01",
                first=HostAttempt(
                    ordinal=1,
                    verdict="pass",
                    observed_status="error",
                    reason_code="execution.partial_denoise_unsupported",
                    result_fingerprint=stable_rejection,
                ),
                repeat=HostAttempt(
                    ordinal=2,
                    verdict="pass",
                    observed_status="error",
                    reason_code="execution.partial_denoise_unsupported",
                    result_fingerprint=stable_rejection,
                ),
            ),
        },
    ]


def _build_unchecked() -> dict[str, object]:
    return {
        "cases": _capability_cases(),
        "coverage": {
            "capabilities": [
                "deterministic_native_euler",
                "effective_model_evaluations",
                "noise_ownership",
                "partial_denoise",
                "per_token_timesteps",
                "resume",
                "rng_separation",
                "sampler_state",
                "schedule_ownership",
                "stochastic_execution",
                "terminal_sigma",
            ],
            "levels": ["allow", "warn", "reject"],
            "skipped": [],
            "unsupported": [
                "advanced_workflows",
                "partial_denoise_execution",
                "resume_execution",
                "stochastic_euler_execution",
            ],
        },
        "host_attempt_regressions": _host_attempt_regressions(),
        "portable_execution": _portable_execution(),
        "profile": {
            "fingerprint": profile_schema_fingerprint(KREA2_TURBO_SCHEMA),
            "id": KREA2_TURBO_SCHEMA.profile_id,
            "version": KREA2_TURBO_SCHEMA.profile_version,
        },
        "schema": REPORT_SCHEMA,
        "status": "PASS",
    }


def build_conformance_report() -> dict[str, object]:
    """Build canonical M7-07 conformance evidence from production contracts."""

    return _build_unchecked()


def validate_conformance_report(value: object) -> dict[str, object]:
    """Rebuild and validate the complete conformance report."""

    report = _mapping(value, name="conformance report")
    _exact(report, _ROOT_FIELDS, name="conformance report")
    if report != _build_unchecked():
        raise ValueError("conformance report differs from recomputed production contracts")
    return cast(dict[str, object], value)


def canonical_json(value: object) -> str:
    """Serialize validated evidence deterministically."""

    report = validate_conformance_report(value)
    return canonical_projection_bytes(report).decode("utf-8") + "\n"
