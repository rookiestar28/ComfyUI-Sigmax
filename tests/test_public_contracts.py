from __future__ import annotations

import copy
import hashlib
import importlib.resources
import json
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import CompatibilityReason, ScheduleContractError
from comfyui_sigmax.profiles import CHECKPOINT_EVIDENCE_REASON_CODES
from comfyui_sigmax.public_contracts import load_public_contract_manifest
from scripts import generate_public_contract_manifest as generator


def _envelope() -> dict[str, Any]:
    payload = (
        importlib.resources.files("comfyui_sigmax.contracts")
        .joinpath("manifest_v1.json")
        .read_bytes()
    )
    return cast(dict[str, Any], json.loads(payload))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _rehashed(envelope: dict[str, Any]) -> bytes:
    envelope["manifest_fingerprint"] = (
        "sha256:" + hashlib.sha256(_canonical(envelope["manifest"])).hexdigest()
    )
    return _canonical(envelope) + b"\n"


def test_packaged_manifest_freezes_the_complete_m8_01_boundary() -> None:
    manifest = load_public_contract_manifest()
    projection = manifest.projection()

    assert manifest.schema == "sigmax.public-contract-manifest/1"
    assert manifest.contract_version == "1"
    assert manifest.manifest_fingerprint.startswith("sha256:")
    assert projection["nodes"] == [
        {
            "id": "Sigmax.AdvancedFlowMatchScheduler",
            "schema": "sigmax.advanced-flowmatch-node/1",
        },
        {
            "id": "Sigmax.AuraFlowSigmaScheduler",
            "schema": "sigmax.aura-flow-sigma-node/1",
        },
        {
            "id": "Sigmax.CheckpointEvidenceInspector",
            "schema": "sigmax.checkpoint-evidence-inspector/1",
        },
        {
            "id": "Sigmax.Flux1SchnellSigmaScheduler",
            "schema": "sigmax.flux1-schnell-sigma-node/1",
        },
        {
            "id": "Sigmax.Krea2ConditioningRebalance",
            "schema": "sigmax.krea2-conditioning-node/1",
        },
        {"id": "Sigmax.Krea2SigmaScheduler", "schema": "sigmax.krea2-sigma-node/1"},
        {
            "id": "Sigmax.ModelAwareSigmaScheduler",
            "schema": "sigmax.model-aware-sigma-node/1",
        },
        {"id": "Sigmax.ProfileInspector", "schema": "sigmax.profile-inspector/1"},
        {
            "id": "Sigmax.QwenImageSigmaScheduler",
            "schema": "sigmax.qwen-image-sigma-node/1",
        },
        {"id": "Sigmax.RawWorkflowOutput", "schema": "sigmax.raw-workflow-output/1"},
        {"id": "Sigmax.SD3SigmaScheduler", "schema": "sigmax.sd3-sigma-node/1"},
        {
            "id": "Sigmax.ScheduleComparison",
            "schema": "sigmax.schedule-comparison/1",
        },
        {
            "id": "Sigmax.ScheduleConcatenate",
            "schema": "sigmax.schedule-concatenate-node/1",
        },
        {"id": "Sigmax.ScheduleInspector", "schema": "sigmax.schedule-inspector/1"},
        {
            "id": "Sigmax.ScheduleResample",
            "schema": "sigmax.schedule-resample-node/1",
        },
        {
            "id": "Sigmax.ScheduleSlice",
            "schema": "sigmax.schedule-slice-node/1",
        },
        {
            "id": "Sigmax.TurboWorkflowOutput",
            "schema": "sigmax.turbo-workflow-output/1",
        },
        {
            "id": "Sigmax.ZImageSigmaScheduler",
            "schema": "sigmax.z-image-sigma-node/1",
        },
    ]
    assert projection["schemas"] == {
        "conditioning": [
            "sigmax.conditioning-modifier/1",
            "sigmax.krea2-conditioning-profile/1",
        ],
        "construction": [
            "sigmax.numerical-schedule/1",
            "sigmax.schedule-artifact-envelope/1",
            "sigmax.schedule-artifact/1",
        ],
        "execution": [
            "sigmax.execution-receipt-envelope/1",
            "sigmax.execution-receipt/1",
            "sigmax.portable-execution-bundle/1",
        ],
        "profile_capability": [
            "sigmax.capability-resolution/1",
            "sigmax.checkpoint-evidence-inspection/1",
            "sigmax.generic-flowmatch-profile/1",
            "sigmax.model-profile/1",
        ],
    }


def test_reason_code_vocabularies_are_complete_and_canonical() -> None:
    projection = load_public_contract_manifest().projection()
    reasons = cast(dict[str, list[str]], projection["reason_codes"])
    compatibility = [
        "model_family_mismatch",
        "model_variant_mismatch",
        "model_prediction_unsupported",
        "model_sigma_domain_unsupported",
        "model_ownership_unsupported",
        "sampler_prediction_unsupported",
        "sampler_sigma_domain_unsupported",
        "sampler_ownership_unsupported",
        "terminal_requirement_mismatch",
        "execution_behavior_mismatch",
        "noise_ownership_mismatch",
        "sampler_state_unsupported",
        "partial_denoise_unsupported_by_model",
        "partial_denoise_unsupported_by_profile",
        "partial_denoise_unsupported_by_sampler",
        "per_token_timesteps_unsupported_by_model",
        "per_token_timesteps_unsupported_by_profile",
        "per_token_timesteps_unsupported_by_sampler",
        "sampler_not_profile_reference",
        "compatible",
    ]

    assert reasons["compatibility"] == compatibility
    assert [reason.value for reason in CompatibilityReason] == compatibility
    assert reasons["capability_resolution"] == sorted(
        {
            *(f"core.{reason}" for reason in compatibility),
            "host.capability_experimental",
            "host.capability_missing",
            "host.capability_unsupported",
            "model.family_mismatch",
            "model.identity_ambiguous",
            "model.identity_conflict",
            "model.identity_suggested",
            "model.identity_unknown",
            "model.variant_mismatch",
        }
    )
    assert reasons["checkpoint_inspection"] == sorted(CHECKPOINT_EVIDENCE_REASON_CODES)


def test_migration_policy_is_explicit_and_versioned() -> None:
    policy = cast(dict[str, object], load_public_contract_manifest().projection()["migration"])

    assert policy == {
        "breaking_change": "new_schema_major_and_project_major",
        "deprecation": "document_before_release_and_retain_through_current_major",
        "node_id_change": "alias_and_migration_required",
        "policy_version": "1",
        "reader_support": "all_frozen_v1_identifiers",
        "schema_addition": "new_identifier_required",
        "unknown_schema": "reject",
    }


def test_generator_matches_packaged_manifest_and_check_mode() -> None:
    expected = generator._canonical(generator.build_envelope()) + b"\n"
    actual = (
        importlib.resources.files("comfyui_sigmax.contracts")
        .joinpath("manifest_v1.json")
        .read_bytes()
    )

    assert actual == expected
    assert generator.main(["--check"]) == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["manifest"].update({"extra": True}), "fields"),
        (
            lambda value: value["manifest"]["nodes"].append(
                copy.deepcopy(value["manifest"]["nodes"][0])
            ),
            "canonical and unique",
        ),
        (
            lambda value: value["manifest"]["schemas"]["construction"].append("sigmax.unknown/1"),
            "source contracts",
        ),
        (
            lambda value: value["manifest"]["migration"].update(
                {"reader_support": "C:/private/path"}
            ),
            "private or absolute path",
        ),
        (
            lambda value: value["manifest"]["migration"].update({"reader_support": "api_token"}),
            "secret-like text",
        ),
    ],
)
def test_manifest_rejects_tampering(mutate: Any, message: str) -> None:
    envelope = _envelope()
    mutate(envelope)

    with pytest.raises(ScheduleContractError, match=message):
        load_public_contract_manifest(_rehashed(envelope))


def test_manifest_rejects_duplicate_json_names_and_noncanonical_encoding() -> None:
    payload = (
        importlib.resources.files("comfyui_sigmax.contracts")
        .joinpath("manifest_v1.json")
        .read_bytes()
    )
    duplicate = payload.replace(b'{"manifest":', b'{"manifest":null,"manifest":', 1)

    with pytest.raises(ScheduleContractError, match="duplicate JSON object name"):
        load_public_contract_manifest(duplicate)
    with pytest.raises(ScheduleContractError, match="canonical"):
        load_public_contract_manifest(
            payload.replace(b'"contract_version":"1"', b'"contract_version": "1"')
        )


def test_contract_manifest_is_listed_as_package_data() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"comfyui_sigmax.contracts" = ["*.json"]' in pyproject
