"""Strict, source-bound loader for the frozen public contract manifest."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Final, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

PUBLIC_CONTRACT_MANIFEST_SCHEMA: Final = "sigmax.public-contract-manifest/1"
PUBLIC_CONTRACT_MANIFEST_ENVELOPE_SCHEMA: Final = "sigmax.public-contract-manifest-envelope/1"
PUBLIC_CONTRACT_VERSION: Final = "1"

_MAX_BYTES: Final = 65_536
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}")
_NODE_ID: Final = re.compile(r"Sigmax\.[A-Za-z][A-Za-z0-9]*")
_SCHEMA_ID: Final = re.compile(r"sigmax\.[a-z0-9-]+/[1-9][0-9]*")
_REASON_CODE: Final = re.compile(r"[a-z][a-z0-9_.-]*")
_PRIVATE_PATH: Final = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")
_SECRET_WORD: Final = re.compile(
    r"(?:^|[_.-])(?:authorization|cookie|credential|password|secret|token)(?:[_.-]|$)",
    re.IGNORECASE,
)

_MANIFEST_FIELDS: Final = frozenset(
    {"contract_version", "migration", "nodes", "reason_codes", "schema", "schemas"}
)
_ENVELOPE_FIELDS: Final = frozenset({"manifest", "manifest_fingerprint", "schema"})
_NODE_FIELDS: Final = frozenset({"id", "schema"})
_SCHEMA_GROUPS: Final = frozenset(
    {"conditioning", "construction", "execution", "profile_capability"}
)
_REASON_GROUPS: Final = frozenset(
    {"capability_resolution", "checkpoint_inspection", "compatibility"}
)
_MIGRATION_FIELDS: Final = frozenset(
    {
        "breaking_change",
        "deprecation",
        "node_id_change",
        "policy_version",
        "reader_support",
        "schema_addition",
        "unknown_schema",
    }
)
_MIGRATION_POLICY: Final = {
    "breaking_change": "new_schema_major_and_project_major",
    "deprecation": "document_before_release_and_retain_through_current_major",
    "node_id_change": "alias_and_migration_required",
    "policy_version": "1",
    "reader_support": "all_frozen_v1_identifiers",
    "schema_addition": "new_identifier_required",
    "unknown_schema": "reject",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ScheduleContractError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise ScheduleContractError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ScheduleContractError(f"non-finite JSON value is forbidden: {value}")


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScheduleContractError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ScheduleContractError(f"{name} must be an array")
    return value


def _exact(value: dict[str, object], fields: frozenset[str], *, name: str) -> None:
    if set(value) != fields:
        raise ScheduleContractError(f"{name} fields do not match schema")


def _text(
    value: object,
    *,
    name: str,
    pattern: re.Pattern[str] | None = None,
    reject_secret_like: bool = False,
) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ScheduleContractError(f"{name} must be bounded non-empty text")
    if _PRIVATE_PATH.search(value):
        raise ScheduleContractError("public contract manifest contains a private or absolute path")
    if reject_secret_like and _SECRET_WORD.search(value):
        raise ScheduleContractError("public contract manifest contains secret-like text")
    if pattern is not None and not pattern.fullmatch(value):
        raise ScheduleContractError(f"{name} has an invalid identifier")
    return value


def source_contract_projection() -> dict[str, object]:
    """Build the reviewed M8-01 boundary from its owning public declarations."""

    from comfyui_sigmax.adapters.registration import builtin_node_registry
    from comfyui_sigmax.conditioning import (
        CONDITIONING_MODIFIER_REPORT_SCHEMA_ID,
        KREA2_CONDITIONING_PROFILE_SCHEMA_ID,
    )
    from comfyui_sigmax.core import (
        EXECUTION_RECEIPT_ENVELOPE_SCHEMA,
        EXECUTION_RECEIPT_SCHEMA,
        NUMERICAL_SCHEDULE_SCHEMA,
        PORTABLE_EXECUTION_BUNDLE_SCHEMA,
        SCHEDULE_ARTIFACT_ENVELOPE_SCHEMA,
        SCHEDULE_ARTIFACT_SCHEMA,
        CompatibilityReason,
    )
    from comfyui_sigmax.nodes.advanced_flowmatch_scheduler import (
        ADVANCED_FLOWMATCH_NODE_ID,
        ADVANCED_FLOWMATCH_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.anima_sigma_scheduler import (
        ANIMA_SIGMA_NODE_ID,
        ANIMA_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.aura_flow_sigma_scheduler import (
        AURAFLOW_SIGMA_NODE_ID,
        AURAFLOW_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.checkpoint_evidence_inspector import (
        CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID,
        CHECKPOINT_EVIDENCE_INSPECTOR_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.flux1_schnell_sigma_scheduler import (
        FLUX1_SCHNELL_SIGMA_NODE_ID,
        FLUX1_SCHNELL_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.hunyuan_image21_sigma_scheduler import (
        HUNYUAN_IMAGE21_SIGMA_NODE_ID,
        HUNYUAN_IMAGE21_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.inspectors import (
        PROFILE_INSPECTOR_NODE_ID,
        PROFILE_INSPECTOR_SCHEMA_ID,
        SCHEDULE_COMPARISON_NODE_ID,
        SCHEDULE_COMPARISON_SCHEMA_ID,
        SCHEDULE_INSPECTOR_NODE_ID,
        SCHEDULE_INSPECTOR_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.krea2_conditioning_rebalance import (
        KREA2_CONDITIONING_NODE_ID,
        KREA2_CONDITIONING_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.krea2_sigma_scheduler import (
        KREA2_SIGMA_NODE_ID,
        KREA2_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.lumina2_sigma_scheduler import (
        LUMINA2_SIGMA_NODE_ID,
        LUMINA2_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.model_aware_sigma_scheduler import (
        MODEL_AWARE_SIGMA_NODE_ID,
        MODEL_AWARE_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.qwen_image_sigma_scheduler import (
        QWEN_IMAGE_SIGMA_NODE_ID,
        QWEN_IMAGE_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.raw_workflow_output import (
        RAW_WORKFLOW_OUTPUT_NODE_ID,
        RAW_WORKFLOW_OUTPUT_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.schedule_algebra import (
        SCHEDULE_CONCATENATE_NODE_ID,
        SCHEDULE_CONCATENATE_SCHEMA_ID,
        SCHEDULE_RESAMPLE_NODE_ID,
        SCHEDULE_RESAMPLE_SCHEMA_ID,
        SCHEDULE_SLICE_NODE_ID,
        SCHEDULE_SLICE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.sd3_sigma_scheduler import (
        SD3_SIGMA_NODE_ID,
        SD3_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.turbo_workflow_output import (
        TURBO_WORKFLOW_OUTPUT_NODE_ID,
        TURBO_WORKFLOW_OUTPUT_SCHEMA_ID,
    )
    from comfyui_sigmax.nodes.z_image_sigma_scheduler import (
        Z_IMAGE_SIGMA_NODE_ID,
        Z_IMAGE_SIGMA_NODE_SCHEMA_ID,
    )
    from comfyui_sigmax.profiles import (
        CAPABILITY_RESOLUTION_ADDITIONAL_REASON_CODES,
        CAPABILITY_RESOLUTION_CORE_REASON_CODES,
        CAPABILITY_RESOLUTION_SCHEMA_ID,
        CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID,
        CHECKPOINT_EVIDENCE_REASON_CODES,
        GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID,
        PROFILE_SCHEMA_ID,
    )

    nodes = sorted(
        (
            {"id": ADVANCED_FLOWMATCH_NODE_ID, "schema": ADVANCED_FLOWMATCH_NODE_SCHEMA_ID},
            {"id": ANIMA_SIGMA_NODE_ID, "schema": ANIMA_SIGMA_NODE_SCHEMA_ID},
            {"id": AURAFLOW_SIGMA_NODE_ID, "schema": AURAFLOW_SIGMA_NODE_SCHEMA_ID},
            {
                "id": CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID,
                "schema": CHECKPOINT_EVIDENCE_INSPECTOR_SCHEMA_ID,
            },
            {
                "id": FLUX1_SCHNELL_SIGMA_NODE_ID,
                "schema": FLUX1_SCHNELL_SIGMA_NODE_SCHEMA_ID,
            },
            {
                "id": KREA2_CONDITIONING_NODE_ID,
                "schema": KREA2_CONDITIONING_NODE_SCHEMA_ID,
            },
            {"id": KREA2_SIGMA_NODE_ID, "schema": KREA2_SIGMA_NODE_SCHEMA_ID},
            {"id": LUMINA2_SIGMA_NODE_ID, "schema": LUMINA2_SIGMA_NODE_SCHEMA_ID},
            {"id": HUNYUAN_IMAGE21_SIGMA_NODE_ID, "schema": HUNYUAN_IMAGE21_SIGMA_NODE_SCHEMA_ID},
            {"id": QWEN_IMAGE_SIGMA_NODE_ID, "schema": QWEN_IMAGE_SIGMA_NODE_SCHEMA_ID},
            {"id": SD3_SIGMA_NODE_ID, "schema": SD3_SIGMA_NODE_SCHEMA_ID},
            {"id": MODEL_AWARE_SIGMA_NODE_ID, "schema": MODEL_AWARE_SIGMA_NODE_SCHEMA_ID},
            {"id": PROFILE_INSPECTOR_NODE_ID, "schema": PROFILE_INSPECTOR_SCHEMA_ID},
            {"id": RAW_WORKFLOW_OUTPUT_NODE_ID, "schema": RAW_WORKFLOW_OUTPUT_SCHEMA_ID},
            {"id": SCHEDULE_COMPARISON_NODE_ID, "schema": SCHEDULE_COMPARISON_SCHEMA_ID},
            {"id": SCHEDULE_CONCATENATE_NODE_ID, "schema": SCHEDULE_CONCATENATE_SCHEMA_ID},
            {"id": SCHEDULE_INSPECTOR_NODE_ID, "schema": SCHEDULE_INSPECTOR_SCHEMA_ID},
            {"id": SCHEDULE_RESAMPLE_NODE_ID, "schema": SCHEDULE_RESAMPLE_SCHEMA_ID},
            {"id": SCHEDULE_SLICE_NODE_ID, "schema": SCHEDULE_SLICE_SCHEMA_ID},
            {"id": TURBO_WORKFLOW_OUTPUT_NODE_ID, "schema": TURBO_WORKFLOW_OUTPUT_SCHEMA_ID},
            {"id": Z_IMAGE_SIGMA_NODE_ID, "schema": Z_IMAGE_SIGMA_NODE_SCHEMA_ID},
        ),
        key=lambda item: item["id"],
    )
    registered_node_ids = sorted(builtin_node_registry().class_mappings())
    if registered_node_ids != [item["id"] for item in nodes]:
        raise ScheduleContractError("built-in node registry does not match public node contracts")
    return {
        "contract_version": PUBLIC_CONTRACT_VERSION,
        "migration": dict(_MIGRATION_POLICY),
        "nodes": nodes,
        "reason_codes": {
            "capability_resolution": sorted(
                CAPABILITY_RESOLUTION_CORE_REASON_CODES
                | CAPABILITY_RESOLUTION_ADDITIONAL_REASON_CODES
            ),
            "checkpoint_inspection": sorted(CHECKPOINT_EVIDENCE_REASON_CODES),
            "compatibility": [reason.value for reason in CompatibilityReason],
        },
        "schema": PUBLIC_CONTRACT_MANIFEST_SCHEMA,
        "schemas": {
            "conditioning": sorted(
                {
                    CONDITIONING_MODIFIER_REPORT_SCHEMA_ID,
                    KREA2_CONDITIONING_PROFILE_SCHEMA_ID,
                }
            ),
            "construction": sorted(
                {
                    NUMERICAL_SCHEDULE_SCHEMA,
                    SCHEDULE_ARTIFACT_ENVELOPE_SCHEMA,
                    SCHEDULE_ARTIFACT_SCHEMA,
                }
            ),
            "execution": sorted(
                {
                    EXECUTION_RECEIPT_ENVELOPE_SCHEMA,
                    EXECUTION_RECEIPT_SCHEMA,
                    PORTABLE_EXECUTION_BUNDLE_SCHEMA,
                }
            ),
            "profile_capability": sorted(
                {
                    CAPABILITY_RESOLUTION_SCHEMA_ID,
                    CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID,
                    GENERIC_FLOWMATCH_PROFILE_SCHEMA_ID,
                    PROFILE_SCHEMA_ID,
                }
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class PublicContractManifest:
    """Immutable validated projection of the frozen release boundary."""

    schema: str
    contract_version: str
    manifest_fingerprint: str
    _projection_bytes: bytes

    def projection(self) -> dict[str, object]:
        """Return a detached JSON-compatible projection."""

        return cast(dict[str, object], json.loads(self._projection_bytes))


def _validate_manifest(manifest: dict[str, object]) -> None:
    _exact(manifest, _MANIFEST_FIELDS, name="public contract manifest")
    if manifest["schema"] != PUBLIC_CONTRACT_MANIFEST_SCHEMA:
        raise ScheduleContractError("public contract manifest schema is unsupported")
    if manifest["contract_version"] != PUBLIC_CONTRACT_VERSION:
        raise ScheduleContractError("public contract version is unsupported")

    migration = _object(manifest["migration"], name="migration policy")
    _exact(migration, _MIGRATION_FIELDS, name="migration policy")
    for key, value in migration.items():
        pattern = None if key == "policy_version" else _REASON_CODE
        text = _text(
            value,
            name=f"migration.{key}",
            pattern=pattern,
            reject_secret_like=True,
        )
        if key == "policy_version" and not text.isdecimal():
            raise ScheduleContractError("migration.policy_version must be numeric")

    raw_nodes = _array(manifest["nodes"], name="nodes")
    if len(raw_nodes) > 64:
        raise ScheduleContractError("node contracts exceed maximum size")
    nodes: list[tuple[str, str]] = []
    for index, value in enumerate(raw_nodes):
        node = _object(value, name=f"nodes[{index}]")
        _exact(node, _NODE_FIELDS, name=f"nodes[{index}]")
        nodes.append(
            (
                _text(node["id"], name="node id", pattern=_NODE_ID),
                _text(node["schema"], name="node schema", pattern=_SCHEMA_ID),
            )
        )
    if nodes != sorted(set(nodes)):
        raise ScheduleContractError("node contracts must be canonical and unique")

    schemas = _object(manifest["schemas"], name="schema groups")
    _exact(schemas, _SCHEMA_GROUPS, name="schema groups")
    for group, value in schemas.items():
        identifiers = [
            _text(item, name=f"schemas.{group}", pattern=_SCHEMA_ID)
            for item in _array(value, name=f"schemas.{group}")
        ]
        if identifiers != sorted(set(identifiers)):
            raise ScheduleContractError(f"schemas.{group} must be canonical and unique")

    reasons = _object(manifest["reason_codes"], name="reason code groups")
    _exact(reasons, _REASON_GROUPS, name="reason code groups")
    for group, value in reasons.items():
        identifiers = [
            _text(item, name=f"reason_codes.{group}", pattern=_REASON_CODE)
            for item in _array(value, name=f"reason_codes.{group}")
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ScheduleContractError(f"reason_codes.{group} must be unique")
        if group == "capability_resolution" and identifiers != sorted(identifiers):
            raise ScheduleContractError("capability resolution reason codes must be canonical")

    if manifest != source_contract_projection():
        raise ScheduleContractError("public contract manifest does not match source contracts")


def load_public_contract_manifest(payload: bytes | None = None) -> PublicContractManifest:
    """Load and verify the canonical packaged public-contract manifest."""

    if payload is None:
        payload = (
            importlib.resources.files("comfyui_sigmax.contracts")
            .joinpath("manifest_v1.json")
            .read_bytes()
        )
    if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_BYTES:
        raise ScheduleContractError("public contract manifest bytes are invalid")
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ScheduleContractError("public contract manifest BOM is forbidden")
    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScheduleContractError("public contract manifest is not valid UTF-8 JSON") from exc
    envelope = _object(value, name="public contract envelope")
    _exact(envelope, _ENVELOPE_FIELDS, name="public contract envelope")
    if envelope["schema"] != PUBLIC_CONTRACT_MANIFEST_ENVELOPE_SCHEMA:
        raise ScheduleContractError("public contract envelope schema is unsupported")
    manifest = _object(envelope["manifest"], name="public contract manifest")
    fingerprint = envelope["manifest_fingerprint"]
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise ScheduleContractError("public contract fingerprint is invalid")
    if fingerprint != _identity(_canonical(manifest)):
        raise ScheduleContractError("public contract fingerprint does not match manifest")
    if payload != _canonical(envelope) + b"\n":
        raise ScheduleContractError("public contract manifest encoding is not canonical")
    _validate_manifest(manifest)
    return PublicContractManifest(
        schema=PUBLIC_CONTRACT_MANIFEST_SCHEMA,
        contract_version=PUBLIC_CONTRACT_VERSION,
        manifest_fingerprint=fingerprint,
        _projection_bytes=_canonical(manifest),
    )
