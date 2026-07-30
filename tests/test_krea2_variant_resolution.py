"""Fail-closed Krea 2 RAW/Turbo evidence resolution."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles import (
    KREA2_RAW_OFFICIAL_SHA256,
    KREA2_TURBO_OFFICIAL_SHA256,
    Krea2Variant,
    Krea2VariantConfidence,
    Krea2VariantEvidence,
    Krea2VariantEvidenceSource,
    Krea2VariantResolution,
    Krea2VariantResolutionError,
    Krea2VariantResolutionStatus,
    collect_krea2_variant_evidence,
    resolve_krea2_variant,
)


def test_explicit_selection_resolves_with_authoritative_confidence() -> None:
    result = resolve_krea2_variant(explicit_variant="raw")

    assert result.status is Krea2VariantResolutionStatus.RESOLVED
    assert result.resolved_variant is Krea2Variant.RAW
    assert result.suggested_variant is None
    assert result.confidence is Krea2VariantConfidence.AUTHORITATIVE
    assert result.decisive_source is Krea2VariantEvidenceSource.EXPLICIT_SELECTION


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    [
        ("krea2.raw.official", Krea2Variant.RAW),
        ("krea2.turbo.official", Krea2Variant.TURBO),
    ],
)
def test_trusted_profile_metadata_resolves(profile_id: str, expected: Krea2Variant) -> None:
    result = resolve_krea2_variant(trusted_profile_id=profile_id)

    assert result.resolved_variant is expected
    assert result.decisive_source is Krea2VariantEvidenceSource.TRUSTED_PROFILE_METADATA


@pytest.mark.parametrize(
    ("is_distilled", "expected"),
    [(False, Krea2Variant.RAW), (True, Krea2Variant.TURBO)],
)
def test_trusted_diffusers_metadata_resolves(
    is_distilled: bool,
    expected: Krea2Variant,
) -> None:
    result = resolve_krea2_variant(
        trusted_framework_metadata={
            "_class_name": "Krea2Pipeline",
            "is_distilled": is_distilled,
        }
    )

    assert result.resolved_variant is expected
    assert result.decisive_source is Krea2VariantEvidenceSource.TRUSTED_FRAMEWORK_METADATA


@pytest.mark.parametrize(
    ("sha256", "expected"),
    [
        (KREA2_RAW_OFFICIAL_SHA256.upper(), Krea2Variant.RAW),
        (KREA2_TURBO_OFFICIAL_SHA256, Krea2Variant.TURBO),
    ],
)
def test_exact_official_hash_resolves(sha256: str, expected: Krea2Variant) -> None:
    result = resolve_krea2_variant(checkpoint_sha256=sha256)

    assert result.resolved_variant is expected
    assert result.confidence is Krea2VariantConfidence.VERIFIED
    assert result.decisive_source is Krea2VariantEvidenceSource.VERIFIED_SHA256


def test_agreeing_strong_evidence_resolves_by_highest_source() -> None:
    result = resolve_krea2_variant(
        trusted_framework_metadata={
            "_class_name": "Krea2Pipeline",
            "is_distilled": True,
        },
        checkpoint_sha256=KREA2_TURBO_OFFICIAL_SHA256,
    )

    assert result.resolved_variant is Krea2Variant.TURBO
    assert result.decisive_source is Krea2VariantEvidenceSource.TRUSTED_FRAMEWORK_METADATA
    assert len(result.evidence) == 2
    assert result.warnings == ()


def test_conflicting_strong_evidence_is_visible_and_strict_mode_fails_closed() -> None:
    flexible = resolve_krea2_variant(
        strict_official=False,
        explicit_variant="raw",
        checkpoint_sha256=KREA2_TURBO_OFFICIAL_SHA256,
    )

    assert flexible.status is Krea2VariantResolutionStatus.CONFLICT
    assert flexible.resolved_variant is None
    assert flexible.suggested_variant is None
    assert flexible.confidence is Krea2VariantConfidence.NONE
    assert "conflicting_resolving_evidence" in flexible.warnings
    with pytest.raises(Krea2VariantResolutionError, match="conflict"):
        resolve_krea2_variant(
            explicit_variant="raw",
            checkpoint_sha256=KREA2_TURBO_OFFICIAL_SHA256,
        )


@pytest.mark.parametrize(
    ("field_name", "payload", "expected", "source"),
    [
        (
            "safetensors_metadata",
            {"krea2_variant": "raw"},
            Krea2Variant.RAW,
            Krea2VariantEvidenceSource.LOCAL_HEADER_SIGNAL,
        ),
        (
            "safetensors_metadata",
            {"is_distilled": "true"},
            Krea2Variant.TURBO,
            Krea2VariantEvidenceSource.LOCAL_HEADER_SIGNAL,
        ),
        (
            "filename",
            r"C:\models\renamed.krea2_raw_fp8.safetensors",
            Krea2Variant.RAW,
            Krea2VariantEvidenceSource.FILENAME_SIGNAL,
        ),
        (
            "filename",
            "/models/krea-2-turbo-bf16.safetensors",
            Krea2Variant.TURBO,
            Krea2VariantEvidenceSource.FILENAME_SIGNAL,
        ),
    ],
)
def test_local_header_and_filename_are_suggestions_only(
    field_name: str,
    payload: object,
    expected: Krea2Variant,
    source: Krea2VariantEvidenceSource,
) -> None:
    if field_name == "safetensors_metadata":
        metadata = cast(dict[str, object], payload)
        flexible = resolve_krea2_variant(
            strict_official=False,
            safetensors_metadata=metadata,
        )
    else:
        filename = cast(str, payload)
        flexible = resolve_krea2_variant(strict_official=False, filename=filename)

    assert flexible.status is Krea2VariantResolutionStatus.SUGGESTED
    assert flexible.resolved_variant is None
    assert flexible.suggested_variant is expected
    assert flexible.decisive_source is source
    with pytest.raises(Krea2VariantResolutionError, match="suggested"):
        if field_name == "safetensors_metadata":
            resolve_krea2_variant(safetensors_metadata=cast(dict[str, object], payload))
        else:
            resolve_krea2_variant(filename=cast(str, payload))


def test_filename_path_is_not_retained_in_evidence() -> None:
    private_path = r"C:\Users\Ray\private\models\krea2_raw.safetensors"
    result = resolve_krea2_variant(filename=private_path, strict_official=False)

    rendered = repr(result)
    assert private_path not in rendered
    assert "Users" not in rendered
    assert result.evidence[0].reason_code == "filename.raw_token"


def test_conflicting_suggestions_remain_ambiguous() -> None:
    result = resolve_krea2_variant(
        safetensors_metadata={"krea2_variant": "raw"},
        filename="krea2_turbo.safetensors",
        strict_official=False,
    )

    assert result.status is Krea2VariantResolutionStatus.AMBIGUOUS
    assert result.resolved_variant is None
    assert result.suggested_variant is None
    assert "conflicting_suggestion_evidence" in result.warnings


def test_filename_with_both_variant_tokens_is_ambiguous() -> None:
    result = resolve_krea2_variant(
        filename="krea2_raw_turbo_merge.safetensors",
        strict_official=False,
    )

    assert result.status is Krea2VariantResolutionStatus.AMBIGUOUS
    assert {item.variant for item in result.evidence} == {
        Krea2Variant.RAW,
        Krea2Variant.TURBO,
    }


def test_model_class_and_tensor_keys_confirm_family_only() -> None:
    result = resolve_krea2_variant(
        model_class="comfy.model_base.Krea2",
        tensor_keys=(
            "diffusion_model.first.weight",
            "diffusion_model.blocks.0.attn.wq.weight",
            "diffusion_model.blocks.0.attn.wk.weight",
            "diffusion_model.txtfusion.projector.weight",
        ),
        strict_official=False,
    )

    assert result.status is Krea2VariantResolutionStatus.AMBIGUOUS
    assert result.resolved_variant is None
    assert result.suggested_variant is None
    assert {item.source for item in result.evidence} == {
        Krea2VariantEvidenceSource.LOCAL_TENSOR_SIGNAL,
        Krea2VariantEvidenceSource.MODEL_CLASS_SIGNAL,
    }
    assert all(item.variant is None for item in result.evidence)
    assert all(item.confidence is Krea2VariantConfidence.FAMILY_ONLY for item in result.evidence)
    with pytest.raises(Krea2VariantResolutionError, match="ambiguous"):
        resolve_krea2_variant(
            model_class="Krea2",
            tensor_keys=("txtfusion.projector.weight",),
        )


def test_unknown_hash_and_renamed_file_never_guess() -> None:
    unknown = "0" * 64
    result = resolve_krea2_variant(
        checkpoint_sha256=unknown,
        filename="model.safetensors",
        strict_official=False,
    )

    assert result.status is Krea2VariantResolutionStatus.AMBIGUOUS
    assert result.resolved_variant is None
    assert result.suggested_variant is None
    assert "checkpoint_hash_not_verified" in result.warnings


def test_strong_evidence_ignores_conflicting_weak_signal_but_warns() -> None:
    result = resolve_krea2_variant(
        explicit_variant="raw",
        filename="krea2_turbo.safetensors",
    )

    assert result.status is Krea2VariantResolutionStatus.RESOLVED
    assert result.resolved_variant is Krea2Variant.RAW
    assert "lower_confidence_evidence_disagrees" in result.warnings


def test_collection_is_deterministic_and_does_not_retain_arbitrary_metadata() -> None:
    arbitrary = {
        "_class_name": "Krea2Pipeline",
        "is_distilled": False,
        "private_path": r"C:\secret\model",
        "token": "do-not-retain",
    }
    first = collect_krea2_variant_evidence(trusted_framework_metadata=arbitrary)
    second = collect_krea2_variant_evidence(
        trusted_framework_metadata=dict(reversed(arbitrary.items()))
    )

    assert first == second
    rendered = repr(first)
    assert "private_path" not in rendered
    assert "do-not-retain" not in rendered


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (False, Krea2Variant.RAW),
        (True, Krea2Variant.TURBO),
        ("false", Krea2Variant.RAW),
        ("TRUE", Krea2Variant.TURBO),
    ],
)
def test_local_is_distilled_header_normalization(
    value: object,
    expected: Krea2Variant,
) -> None:
    evidence = collect_krea2_variant_evidence(safetensors_metadata={"is_distilled": value})

    assert evidence[0].variant is expected


def test_irrelevant_family_inputs_do_not_create_evidence() -> None:
    assert (
        collect_krea2_variant_evidence(
            safetensors_metadata={"unrelated": "value"},
            tensor_keys=("other.weight",),
            model_class="OtherModel",
            filename="renamed.safetensors",
        )
        == ()
    )


def test_unknown_hash_is_retained_as_unverified_beside_strong_evidence() -> None:
    result = resolve_krea2_variant(
        explicit_variant="raw",
        checkpoint_sha256="0" * 64,
    )

    assert result.resolved_variant is Krea2Variant.RAW
    assert "checkpoint_hash_not_verified" in result.warnings
    hash_evidence = next(
        item
        for item in result.evidence
        if item.source is Krea2VariantEvidenceSource.VERIFIED_SHA256
    )
    assert hash_evidence.variant is None
    assert hash_evidence.confidence is Krea2VariantConfidence.NONE


@pytest.mark.parametrize(
    "factory",
    [
        lambda: collect_krea2_variant_evidence(explicit_variant="base"),
        lambda: collect_krea2_variant_evidence(trusted_profile_id="krea2.unknown"),
        lambda: collect_krea2_variant_evidence(
            trusted_framework_metadata={"_class_name": "OtherPipeline", "is_distilled": True}
        ),
        lambda: collect_krea2_variant_evidence(
            trusted_framework_metadata={"_class_name": "Krea2Pipeline", "is_distilled": 1}
        ),
        lambda: collect_krea2_variant_evidence(trusted_framework_metadata=cast(Any, [])),
        lambda: collect_krea2_variant_evidence(checkpoint_sha256="not-a-hash"),
        lambda: collect_krea2_variant_evidence(safetensors_metadata={"krea2_variant": object()}),
        lambda: collect_krea2_variant_evidence(safetensors_metadata={"is_distilled": "unknown"}),
        lambda: collect_krea2_variant_evidence(safetensors_metadata=cast(Any, [])),
        lambda: collect_krea2_variant_evidence(tensor_keys=("valid", 1)),  # type: ignore[arg-type]
        lambda: collect_krea2_variant_evidence(tensor_keys=cast(Any, None)),
        lambda: collect_krea2_variant_evidence(model_class=""),
        lambda: collect_krea2_variant_evidence(filename=""),
        lambda: resolve_krea2_variant(strict_official=1),  # type: ignore[arg-type]
    ],
)
def test_invalid_inputs_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


def test_evidence_and_result_contracts_reject_invalid_combinations() -> None:
    with pytest.raises(ScheduleContractError):
        Krea2VariantEvidence(
            source=Krea2VariantEvidenceSource.FILENAME_SIGNAL,
            confidence=Krea2VariantConfidence.AUTHORITATIVE,
            variant=Krea2Variant.RAW,
            reason_code="filename.raw_token",
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantEvidence(
            source=cast(Any, "filename_signal"),
            confidence=Krea2VariantConfidence.WEAK,
            variant=Krea2Variant.RAW,
            reason_code="filename.raw_token",
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantEvidence(
            source=Krea2VariantEvidenceSource.FILENAME_SIGNAL,
            confidence=cast(Any, "weak"),
            variant=Krea2Variant.RAW,
            reason_code="filename.raw_token",
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantEvidence(
            source=Krea2VariantEvidenceSource.FILENAME_SIGNAL,
            confidence=Krea2VariantConfidence.WEAK,
            variant=cast(Any, "raw"),
            reason_code="filename.raw_token",
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantEvidence(
            source=Krea2VariantEvidenceSource.FILENAME_SIGNAL,
            confidence=Krea2VariantConfidence.WEAK,
            variant=None,
            reason_code="filename.raw_token",
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantEvidence(
            source=Krea2VariantEvidenceSource.LOCAL_TENSOR_SIGNAL,
            confidence=Krea2VariantConfidence.FAMILY_ONLY,
            variant=Krea2Variant.RAW,
            reason_code="tensor.krea2_family",
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantEvidence(
            source=Krea2VariantEvidenceSource.FILENAME_SIGNAL,
            confidence=Krea2VariantConfidence.WEAK,
            variant=Krea2Variant.RAW,
            reason_code="not valid",
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantResolution(
            status=Krea2VariantResolutionStatus.RESOLVED,
            resolved_variant=None,
            suggested_variant=None,
            confidence=Krea2VariantConfidence.AUTHORITATIVE,
            decisive_source=Krea2VariantEvidenceSource.EXPLICIT_SELECTION,
            evidence=(),
            warnings=(),
        )


def _valid_resolution_fields() -> dict[str, object]:
    result = resolve_krea2_variant(explicit_variant="raw")
    return {
        "status": result.status,
        "resolved_variant": result.resolved_variant,
        "suggested_variant": result.suggested_variant,
        "confidence": result.confidence,
        "decisive_source": result.decisive_source,
        "evidence": result.evidence,
        "warnings": result.warnings,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "resolved"),
        ("resolved_variant", "raw"),
        ("suggested_variant", "raw"),
        ("confidence", "authoritative"),
        ("decisive_source", "explicit_selection"),
        ("evidence", []),
        ("evidence", (object(),)),
        ("warnings", []),
        ("warnings", ("not valid",)),
        ("warnings", ("duplicate", "duplicate")),
    ],
)
def test_resolution_contract_rejects_invalid_field_types(
    field: str,
    value: object,
) -> None:
    fields = _valid_resolution_fields()
    fields[field] = value

    with pytest.raises(ScheduleContractError):
        Krea2VariantResolution(**cast(Any, fields))


def test_suggested_and_unresolved_contracts_reject_inconsistent_states() -> None:
    evidence = collect_krea2_variant_evidence(filename="krea2_raw.safetensors")
    with pytest.raises(ScheduleContractError):
        Krea2VariantResolution(
            status=Krea2VariantResolutionStatus.SUGGESTED,
            resolved_variant=Krea2Variant.RAW,
            suggested_variant=Krea2Variant.RAW,
            confidence=Krea2VariantConfidence.WEAK,
            decisive_source=Krea2VariantEvidenceSource.FILENAME_SIGNAL,
            evidence=evidence,
            warnings=(),
        )
    with pytest.raises(ScheduleContractError):
        Krea2VariantResolution(
            status=Krea2VariantResolutionStatus.AMBIGUOUS,
            resolved_variant=Krea2Variant.RAW,
            suggested_variant=None,
            confidence=Krea2VariantConfidence.NONE,
            decisive_source=None,
            evidence=evidence,
            warnings=(),
        )


def test_evidence_and_results_are_immutable() -> None:
    result = resolve_krea2_variant(explicit_variant=Krea2Variant.TURBO)

    with pytest.raises(FrozenInstanceError):
        result.resolved_variant = Krea2Variant.RAW  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.evidence[0].variant = Krea2Variant.RAW  # type: ignore[misc]


def test_variant_module_remains_framework_and_filesystem_independent() -> None:
    module_path = Path(__file__).parents[1] / "comfyui_sigmax" / "profiles" / "krea2_variant.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=module_path.name)
    import_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    import_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    for forbidden in ("torch", "diffusers", "comfy", "safetensors", "huggingface_hub"):
        assert forbidden not in import_roots
    assert "open(" not in source
    assert "Path(" not in source
