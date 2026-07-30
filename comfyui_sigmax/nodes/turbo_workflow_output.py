"""Executable output boundary for the strict official Krea 2 Turbo workflow."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

from comfyui_sigmax.core import (
    ArtifactBuildMetadata,
    ArtifactField,
    ExecutionComponent,
    ExecutionFeatureRequest,
    ExecutionHost,
    ExecutionReceiptMetadata,
    ExecutionRngOwnership,
    ExecutionStatus,
    NoiseOwnership,
    PortableExecutionBundle,
    ScheduleContractError,
    TypedArtifactValue,
    build_execution_receipt,
    build_schedule_artifact,
    canonical_projection_bytes,
    evaluate_compatibility,
    serialize_portable_execution_bundle,
)
from comfyui_sigmax.nodes.inspectors import build_schedule_inspection
from comfyui_sigmax.nodes.krea2_sigma_scheduler import (
    bind_krea2_sigma_output_info,
    build_krea2_sigma_schedule,
)
from comfyui_sigmax.profiles import (
    KREA2_TURBO_PROFILE,
    KREA2_TURBO_SCHEMA,
    build_krea2_turbo_schedule,
    profile_schema_fingerprint,
)

TURBO_WORKFLOW_OUTPUT_NODE_ID: Final = "Sigmax.TurboWorkflowOutput"
TURBO_WORKFLOW_OUTPUT_SCHEMA_ID: Final = "sigmax.turbo-workflow-output/1"
TURBO_WORKFLOW_BUNDLE_UI_KEY: Final = "sigmax_execution_bundle"

# IMPORTANT: this is the reviewed M2-04/M4-10 known-good host, not ambient host detection.
_HOST_VERSION: Final = "0.29.0"
_HOST_REVISION: Final = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
_MAX_HOST_SIGMAS: Final = 10_001
_MAX_TEXT_BYTES: Final = 1_048_576


@dataclass(frozen=True, slots=True, kw_only=True)
class TurboWorkflowOutputResult:
    """Canonical bounded payload returned by the pure publication boundary."""

    schema_id: str
    bundle_json: str
    bundle_fingerprint: str
    artifact_construction_fingerprint: str
    artifact_numerical_fingerprint: str
    receipt_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_id != TURBO_WORKFLOW_OUTPUT_SCHEMA_ID:
            raise ScheduleContractError("Turbo workflow output schema is unsupported")
        if not isinstance(self.bundle_json, str) or not self.bundle_json:
            raise ScheduleContractError("Turbo workflow bundle must be non-empty text")
        for field_name in (
            "bundle_fingerprint",
            "artifact_construction_fingerprint",
            "artifact_numerical_fingerprint",
            "receipt_fingerprint",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                raise ScheduleContractError(f"{field_name} must be a SHA-256 fingerprint")


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _bounded_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleContractError(f"{label} must be non-empty text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise ScheduleContractError(f"{label} must be valid Unicode") from exc
    if size > _MAX_TEXT_BYTES:
        raise ScheduleContractError(f"{label} exceeds the publication limit")
    return value


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ScheduleContractError(f"{label} must be a positive integer")
    return value


def _host_sigma_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise ScheduleContractError("pure Turbo workflow sigmas must be a tuple")
    if not 2 <= len(value) <= _MAX_HOST_SIGMAS:
        raise ScheduleContractError("Turbo workflow sigmas length is outside the limit")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleContractError("Turbo workflow sigmas must be numeric") from exc


def _artifact_metadata() -> ArtifactBuildMetadata:
    compatibility = evaluate_compatibility(
        model=KREA2_TURBO_SCHEMA.model_capabilities,
        profile=KREA2_TURBO_SCHEMA.profile_capabilities,
        sampler=KREA2_TURBO_SCHEMA.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )
    return ArtifactBuildMetadata(
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
            ArtifactField(name="decision", value=compatibility.level.value),
            ArtifactField(name="double_shift", value=False),
            ArtifactField(
                name="reference_sampler",
                value=KREA2_TURBO_SCHEMA.reference_sampler_capabilities.sampler_id,
            ),
        ),
    )


def _declared_component(
    *,
    identifier: str,
    version: str,
    projection: dict[str, object],
) -> ExecutionComponent:
    return ExecutionComponent(
        identifier=identifier,
        version=version,
        fingerprint=_sha256(canonical_projection_bytes(projection)),
    )


def build_turbo_workflow_output(
    *,
    sigmas: object,
    schedule_info: object,
    schedule_report: object,
) -> TurboWorkflowOutputResult:
    """Verify the strict Turbo graph and build a truthful model-free execution bundle."""

    values = _host_sigma_tuple(sigmas)
    info_text = _bounded_text(schedule_info, label="schedule_info")
    report_text = _bounded_text(schedule_report, label="schedule_report")

    inspection = build_schedule_inspection(sigmas=values, schedule_info=info_text)
    if report_text != inspection.report_json:
        raise ScheduleContractError("schedule_report does not match the connected schedule")

    try:
        info = json.loads(info_text)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ScheduleContractError("schedule_info is not valid JSON") from exc
    source = _object(info, label="schedule_info")
    profile = _object(source.get("profile"), label="schedule_info profile")
    slicing = _object(source.get("slicing"), label="schedule_info slicing")
    dimensions = _object(source.get("dimensions"), label="schedule_info dimensions")
    requested = _object(dimensions.get("requested"), label="requested dimensions")

    if profile != {
        "evidence": "official",
        "id": "krea2.turbo.official",
        "recipe": "krea2.turbo.official-8",
        "variant": "turbo",
        "version": "1",
    }:
        raise ScheduleContractError("workflow output requires the official Turbo profile")
    if source.get("strict_official") is not True:
        raise ScheduleContractError("workflow output requires strict official mode")
    if slicing != {
        "available_steps": 8,
        "end_step": 8,
        "output_steps": 8,
        "start_step": 0,
    }:
        raise ScheduleContractError("workflow output requires the complete eight-step schedule")

    width = _integer(requested.get("width"), label="requested width")
    height = _integer(requested.get("height"), label="requested height")
    rebuilt_node = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=width,
        height=height,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    float32_values = tuple(
        struct.unpack(">f", struct.pack(">f", value))[0] for value in rebuilt_node.sigmas
    )
    if values not in (rebuilt_node.sigmas, float32_values):
        raise ScheduleContractError("connected Turbo evidence differs from authoritative rebuild")
    expected_info = bind_krea2_sigma_output_info(
        rebuilt_node,
        output_sigmas=values,
    )
    if expected_info != info_text:
        raise ScheduleContractError("connected schedule information differs from its host values")

    schedule = build_krea2_turbo_schedule(steps=8, width=width, height=height)
    if schedule.sigmas != rebuilt_node.sigmas:
        raise ScheduleContractError("artifact schedule differs from authoritative Turbo sigmas")
    artifact = build_schedule_artifact(
        schedule,
        metadata=_artifact_metadata(),
        precision="float64",
    )
    compatibility = evaluate_compatibility(
        model=KREA2_TURBO_SCHEMA.model_capabilities,
        profile=KREA2_TURBO_SCHEMA.profile_capabilities,
        sampler=KREA2_TURBO_SCHEMA.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )
    receipt = build_execution_receipt(
        artifact,
        metadata=ExecutionReceiptMetadata(
            compatibility=compatibility,
            host=ExecutionHost(
                identifier="comfyui",
                version=_HOST_VERSION,
                revision=_HOST_REVISION,
                api_version="legacy_v1",
            ),
            model=_declared_component(
                identifier="krea2.turbo.declared-model",
                version=KREA2_TURBO_SCHEMA.profile_version,
                projection={
                    "profile_fingerprint": profile_schema_fingerprint(KREA2_TURBO_SCHEMA),
                    "role": "requested_model",
                },
            ),
            sampler=_declared_component(
                identifier=KREA2_TURBO_SCHEMA.reference_sampler_capabilities.sampler_id,
                version=_HOST_VERSION,
                projection={
                    "host_revision": _HOST_REVISION,
                    "role": "requested_sampler",
                    "sampler_id": (KREA2_TURBO_SCHEMA.reference_sampler_capabilities.sampler_id),
                },
            ),
            rng_ownership=ExecutionRngOwnership(
                schedule=NoiseOwnership.NONE,
                model=NoiseOwnership.NONE,
                sampler=NoiseOwnership.NONE,
            ),
            requested_transitions=8,
            effective_transitions=0,
            requested_model_evaluations=8,
            effective_model_evaluations=0,
            status=ExecutionStatus.NOT_EXECUTED,
        ),
    )
    bundle_bytes = serialize_portable_execution_bundle(
        PortableExecutionBundle(artifact=artifact, receipt=receipt)
    )
    return TurboWorkflowOutputResult(
        schema_id=TURBO_WORKFLOW_OUTPUT_SCHEMA_ID,
        bundle_json=bundle_bytes.decode("utf-8"),
        bundle_fingerprint=_sha256(bundle_bytes),
        artifact_construction_fingerprint=artifact.construction_fingerprint,
        artifact_numerical_fingerprint=artifact.numerical_fingerprint,
        receipt_fingerprint=receipt.receipt_fingerprint,
    )


class TurboWorkflowOutput:
    """Publish verified strict Turbo schedule evidence through ComfyUI prompt history."""

    DESCRIPTION = (
        "Verifies the strict eight-step Turbo schedule and publishes a model-free artifact "
        "bundle with a not-executed sampler receipt."
    )
    CATEGORY = "Sigmax/workflows"
    FUNCTION = "publish"
    RETURN_TYPES: Final = ()
    RETURN_NAMES: Final = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "schedule_info": ("STRING", {"default": "", "multiline": True}),
                "schedule_report": ("STRING", {"default": "", "multiline": True}),
            }
        }

    def publish(
        self,
        sigmas: object,
        schedule_info: object,
        schedule_report: object,
    ) -> dict[str, object]:
        try:
            length = len(cast(Sequence[object], sigmas))
        except (TypeError, AttributeError) as exc:
            raise ScheduleContractError("host SIGMAS must be a bounded sequence") from exc
        if not 2 <= length <= _MAX_HOST_SIGMAS:
            raise ScheduleContractError("host SIGMAS length is outside the publication limit")
        try:
            values = tuple(float(cast(Any, item)) for item in cast(Sequence[object], sigmas))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ScheduleContractError("host SIGMAS must contain numeric values") from exc
        result = build_turbo_workflow_output(
            sigmas=values,
            schedule_info=schedule_info,
            schedule_report=schedule_report,
        )
        return {
            "ui": {TURBO_WORKFLOW_BUNDLE_UI_KEY: [result.bundle_json]},
            "result": (),
        }
