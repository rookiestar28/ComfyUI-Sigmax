"""M6-12 MiniMax H3 accelerated recipe and artifact qualification contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles.minimax_h3 import (
    MINIMAX_H3_BASE_FL2VA_SCHEMA,
    MINIMAX_H3_BASE_REF2VA_SCHEMA,
)
from comfyui_sigmax.profiles.registry import builtin_profile_registry


def _module() -> ModuleType:
    return importlib.import_module("comfyui_sigmax.profiles.minimax_h3_acceleration")


def test_m6_12_is_a_separate_non_runtime_seam_and_preserves_base_registry() -> None:
    module = _module()
    assert module.MINIMAX_H3_ACCELERATION_SCHEMA_ID == "sigmax.minimax-h3-acceleration/1"
    assert module.MINIMAX_H3_ACCELERATION_SCHEMA_VERSION == "1"
    registry_ids = {entry.schema.profile_id for entry in builtin_profile_registry().entries}
    assert "minimax-h3.base_fl2va" in registry_ids
    assert "minimax-h3.base_ref2va" in registry_ids
    assert not any("turbo" in profile_id for profile_id in registry_ids if "minimax" in profile_id)
    assert MINIMAX_H3_BASE_FL2VA_SCHEMA.profile_id == "minimax-h3.base_fl2va"
    assert MINIMAX_H3_BASE_REF2VA_SCHEMA.profile_id == "minimax-h3.base_ref2va"


def test_m6_12_source_pins_are_exact_and_role_labeled() -> None:
    module = _module()
    sources = {source.source_id: source for source in module.MINIMAX_H3_ACCELERATION_SOURCES}
    assert (
        {source_id: source.revision for source_id, source in sources.items()}
        == {
            "modeltc.minimax-h3-turbo": "a7e148b8dc7db8ad976966060dcc022adf11fc8d",  # pragma: allowlist secret
            "lightx2v.minimax-h3-turbo": "5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1",  # pragma: allowlist secret
            "minimaxai.minimax-h3": "42ed227ee7df40d41602854ae760620d6eb651fe",  # pragma: allowlist secret
            "kijai.minimax-h3-comfy": "2dc3cedb9b58b0e448d9e950f794f25bf28dbbb5",  # pragma: allowlist secret
            "comfyui.repository": "c1739380c6fab78e7e263cb665d04aafbfe24593",  # pragma: allowlist secret
            "comfy-kitchen.repository": "cfcc843b6e8ec1e119b8fe8f7f8f6a46dad8599e",  # pragma: allowlist secret
            "minimaxai.msa": "80434d7f67877c6570ca19cac444b84bc9855dac",  # pragma: allowlist secret
            "thu-ml.sageattention": "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5",  # pragma: allowlist secret
            "dao-ai-lab.flash-attention": "0251105a2fb19d2957484b7f023cd8c115286ced",  # pragma: allowlist secret
            "kijai.comfyui-kjnodes": "3f20054214fec9f9234fd3841ae6f1e4287948f6",  # pragma: allowlist secret
        }
    )
    assert all(source.url.startswith("https://") for source in sources.values())
    assert all(
        source.locators == tuple(sorted(set(source.locators))) for source in sources.values()
    )
    assert {source_id: source.role.value for source_id, source in sources.items()} == {
        "modeltc.minimax-h3-turbo": "recipe",
        "lightx2v.minimax-h3-turbo": "artifact",
        "minimaxai.minimax-h3": "model",
        "kijai.minimax-h3-comfy": "artifact",
        "comfyui.repository": "host",
        "comfy-kitchen.repository": "backend",
        "minimaxai.msa": "attention",
        "thu-ml.sageattention": "attention",
        "dao-ai-lab.flash-attention": "attention",
        "kijai.comfyui-kjnodes": "attention",
    }


@pytest.mark.parametrize(
    ("recipe_id", "task", "nfe", "video_shift", "audio_shift", "resolution"),
    [
        ("h3.fl2va.lightx2v-turbo-4-v0.1-544p", "fl2va", (4,), 12.0, 3.0, "544p_mixed_aspect"),
        ("h3.fl2va.lightx2v-turbo-8-v1.0-544p", "fl2va", (4, 8), 12.0, 3.0, "544p_mixed_aspect"),
        ("h3.fl2va.lightx2v-turbo-4-v1.0-768p", "fl2va", (4,), 6.0, 3.0, "1344x768"),
        ("h3.ref2va.lightx2v-turbo-4-v0.1-544p", "ref2va", (4,), 12.0, 3.0, "544p_mixed_aspect"),
    ],
)
def test_m6_12_published_recipe_contracts_are_frozen(
    recipe_id: str,
    task: str,
    nfe: tuple[int, ...],
    video_shift: float,
    audio_shift: float,
    resolution: str,
) -> None:
    module = _module()
    recipes = {recipe.recipe_id: recipe for recipe in module.MINIMAX_H3_ACCELERATION_RECIPES}
    recipe = recipes[recipe_id]
    assert recipe.task == task
    assert recipe.allowed_nfe == nfe
    assert recipe.default_nfe == max(nfe)
    assert recipe.video_shift == video_shift
    assert recipe.audio_shift == audio_shift
    assert recipe.resolution_policy == resolution
    assert recipe.sampler == "euler"
    assert recipe.schedule_owner == "sigmax.external_video_sigma"
    assert recipe.audio_owner == "minimax_h3.model_native"
    assert recipe.lora_owner == "comfyui.LoraLoaderModelOnly"
    assert recipe.attention_owner == "host.comfyui"
    assert recipe.disposition is module.MiniMaxH3AccelerationDisposition.QUALIFIED
    assert recipe.reason_codes == ()
    assert recipe.runtime_registered is False


def test_m6_12_artifact_matrix_keeps_exact_and_modified_identities_distinct() -> None:
    module = _module()
    artifacts = {
        artifact.artifact_id: artifact for artifact in module.MINIMAX_H3_ACCELERATION_ARTIFACTS
    }
    assert len(artifacts) == 9
    expected = {
        "lightx2v.fl2v-4-768.full": (
            "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.BLOCKED,
        ),
        "lightx2v.fl2v-8.full": (
            "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.BLOCKED,
        ),
        "lightx2v.ref2v-4.full": (
            "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.BLOCKED,
        ),
        "kijai.fl2v-4-768.reduced": (
            "9515eee9f642aa0e7fcc401f56d408ef2d6388f81881fe50bddded8220870a4d",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.BLOCKED,
        ),
        "kijai.fl2v-8.reduced": (
            "8e05b7b982c3aff7deb692a188c8a8d8acaeff8a12abfe1aeac822fb8ee3f0b7",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.BLOCKED,
        ),
        "kijai.ref2v-4.reduced": (
            "9ea3bd3a6aac22994153e294cf1ecab0a8766fc0f8d056ace645a01d1a6a4daf",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.BLOCKED,
        ),
        "local.fl2v-4-768.modified": (
            "1b85da614014024a0c9507f12558917dcc69b6adb564e716324594f401723115",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.REJECTED,
        ),
        "local.fl2v-8.modified": (
            "a3208be61329c27a6754c53db9a21a3c86e2a285381700adf2d97e279c062840",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.REJECTED,
        ),
        "local.ref2v-4.modified": (
            "2c6abb194cff3e26c2295c87892913adf0c92d8f784f305238246759f9b333d0",  # pragma: allowlist secret
            module.MiniMaxH3AccelerationDisposition.REJECTED,
        ),
    }
    assert {
        artifact_id: (artifact.sha256, artifact.disposition)
        for artifact_id, artifact in artifacts.items()
    } == expected
    assert artifacts["local.ref2v-4.modified"].reason_codes == (
        module.MiniMaxH3AccelerationReasonCode.LOCAL_MODIFICATION,
        module.MiniMaxH3AccelerationReasonCode.TASK_METADATA_CONFLICT,
    )
    assert all("\\" not in artifact.filename for artifact in artifacts.values())


def test_m6_12_artifacts_freeze_tensor_layout_and_baked_scale() -> None:
    module = _module()
    artifacts = {
        artifact.artifact_id: artifact for artifact in module.MINIMAX_H3_ACCELERATION_ARTIFACTS
    }
    for artifact in artifacts.values():
        assert artifact.tensor_count == 416
        assert artifact.tensor_layout == "208_lora_A+208_lora_B;50_main+2_refiner;qkv,out,fc1,fc2"
        assert artifact.rank_policy == "dynamic_per_projection"
        assert artifact.loader_strength == 1.0
        assert artifact.baked_scale in {1.0, 0.0625}
    assert artifacts["local.fl2v-4-768.modified"].baked_scale == 1.0
    assert artifacts["local.fl2v-8.modified"].baked_scale == 0.0625
    assert artifacts["local.ref2v-4.modified"].declared_task == "ref2va"
    assert artifacts["local.ref2v-4.modified"].metadata_task == "fl2va"


def test_m6_12_backend_axes_are_separate_and_msa_link_is_unproven() -> None:
    module = _module()
    backends = {backend.backend_id: backend for backend in module.MINIMAX_H3_ACCELERATION_BACKENDS}
    assert set(backends) == {
        "comfy-kitchen.quantized-operations",
        "comfy-kitchen.int8-attention",
        "comfyui.core-attention",
        "minimaxai.msa",
        "sageattention.upstream",
        "flashattention.upstream",
    }
    assert backends["comfy-kitchen.quantized-operations"].scope is (
        module.MiniMaxH3AccelerationBackendScope.QUANTIZED_OPERATIONS
    )
    assert backends["comfy-kitchen.int8-attention"].scope is (
        module.MiniMaxH3AccelerationBackendScope.INT8_ATTENTION
    )
    assert backends["minimaxai.msa"].reason_codes == (
        module.MiniMaxH3AccelerationReasonCode.UNPROVEN_MSA_H3_LINK,
    )
    assert all(backend.runtime_selected is False for backend in backends.values())


def test_m6_12_candidate_qualification_is_exact_and_fail_closed() -> None:
    module = _module()
    exact = module.qualify_minimax_h3_candidate(
        candidate_id="kijai.fl2v-8.reduced",
        task="fl2va",
        nfe=8,
        video_shift=12.0,
        audio_shift=3.0,
        artifact_sha256="8e05b7b982c3aff7deb692a188c8a8d8acaeff8a12abfe1aeac822fb8ee3f0b7",  # pragma: allowlist secret
        artifact_size_bytes=364638304,
        loader_strength=1.0,
    )
    assert exact.artifact_id == "kijai.fl2v-8.reduced"
    assert exact.disposition is module.MiniMaxH3AccelerationDisposition.BLOCKED

    with pytest.raises(module.MiniMaxH3AccelerationError) as task_error:
        module.qualify_minimax_h3_candidate(
            candidate_id="kijai.fl2v-8.reduced",
            task="ref2va",
        )
    assert task_error.value.reason_code is module.MiniMaxH3AccelerationReasonCode.WRONG_TASK

    negative_cases = [
        ("artifact_sha256", "0" * 64, module.MiniMaxH3AccelerationReasonCode.SIZE_HASH_MISMATCH),
        ("artifact_size_bytes", 1, module.MiniMaxH3AccelerationReasonCode.SIZE_HASH_MISMATCH),
        ("nfe", 2, module.MiniMaxH3AccelerationReasonCode.UNSUPPORTED_RECIPE_NFE),
        ("video_shift", 6.0, module.MiniMaxH3AccelerationReasonCode.DUPLICATE_SHIFT_RISK),
        ("loader_strength", 0.0625, module.MiniMaxH3AccelerationReasonCode.DUPLICATE_SCALE_RISK),
        (
            "resolution_policy",
            "1344x768",
            module.MiniMaxH3AccelerationReasonCode.RESOLUTION_POLICY_MISMATCH,
        ),
    ]
    for field, value, reason in negative_cases:
        with pytest.raises(module.MiniMaxH3AccelerationError) as error:
            module.qualify_minimax_h3_candidate(
                candidate_id="kijai.fl2v-8.reduced", **{field: value}
            )
        assert error.value.reason_code is reason

    with pytest.raises(module.MiniMaxH3AccelerationError) as filename_error:
        module.qualify_minimax_h3_candidate(
            candidate_id="kijai.fl2v-8.reduced",
            artifact_filename="looks_similar.safetensors",
        )
    assert (
        filename_error.value.reason_code
        is module.MiniMaxH3AccelerationReasonCode.FILENAME_ONLY_IDENTITY
    )

    with pytest.raises(module.MiniMaxH3AccelerationError) as backend_error:
        module.qualify_minimax_h3_candidate(
            candidate_id="kijai.fl2v-8.reduced",
            backend_scope=module.MiniMaxH3AccelerationBackendScope.QUANTIZED_OPERATIONS,
        )
    assert (
        backend_error.value.reason_code
        is module.MiniMaxH3AccelerationReasonCode.BACKEND_SCOPE_MISMATCH
    )

    with pytest.raises(module.MiniMaxH3AccelerationError) as unknown:
        module.qualify_minimax_h3_candidate(candidate_id="unknown.filename-only")
    assert unknown.value.reason_code is module.MiniMaxH3AccelerationReasonCode.UNKNOWN_ARTIFACT_HASH


def test_m6_12_blocked_candidates_require_explicit_promotion_and_no_runtime_claim() -> None:
    module = _module()
    with pytest.raises(module.MiniMaxH3AccelerationError) as error:
        module.qualify_minimax_h3_candidate(
            candidate_id="kijai.fl2v-8.reduced", require_eligible=True
        )
    assert error.value.reason_code is module.MiniMaxH3AccelerationReasonCode.UNVERIFIED_LICENSE
    assert all(
        candidate.runtime_registered is False
        for candidate in module.MINIMAX_H3_ACCELERATION_CANDIDATES
    )


def test_m6_12_canonical_serialization_is_deterministic_and_private_data_free() -> None:
    module = _module()
    first = module.serialize_minimax_h3_acceleration()
    second = module.serialize_minimax_h3_acceleration()
    assert first == second
    assert first["schema"] == "sigmax.minimax-h3-acceleration/1"
    encoded = json.dumps(
        first, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    assert module.minimax_h3_acceleration_fingerprint() == (
        "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    )
    lowered = encoded.lower()
    assert "a:\\" not in lowered
    assert "\\\\" not in lowered
    assert "credential" not in lowered
    assert "weights_payload" not in lowered


def test_m6_12_contract_constructors_reject_unpinned_or_private_evidence() -> None:
    module = _module()
    with pytest.raises(ScheduleContractError, match="revision"):
        module.MiniMaxH3AccelerationSource(
            source_id="bad.source",
            role=module.MiniMaxH3AccelerationSourceRole.RECIPE,
            url="https://example.invalid/source",
            revision="not-pinned",
            locators=("README.md",),
            license_id="Apache-2.0",
        )
    with pytest.raises(ScheduleContractError, match="private"):
        module.MiniMaxH3AccelerationSource(
            source_id="bad.source",
            role=module.MiniMaxH3AccelerationSourceRole.RECIPE,
            url="https://example.invalid/source",
            revision="a" * 40,  # pragma: allowlist secret
            locators=("C:\\private\\artifact.safetensors",),
            license_id="Apache-2.0",
        )
    immutable = module.MINIMAX_H3_ACCELERATION_RECIPES[0]
    with pytest.raises(FrozenInstanceError):
        immutable.recipe_id = "changed"
