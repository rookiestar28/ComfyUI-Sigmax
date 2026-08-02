"""M4-11 contracts for the explicit experimental Krea 2 LoRA profile."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import pairwise

import pytest
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError, SigmaDomain
from comfyui_sigmax.profiles import (
    KREA2_LORA_EXPERIMENTAL_PROFILE,
    KREA2_LORA_EXPERIMENTAL_SCHEMA,
    Krea2ExperimentalMuSource,
    ProfileKey,
    build_krea2_lora_experimental_schedule,
    builtin_profile_registry,
    derive_krea2_lora_experimental_shift,
)


def test_experimental_schema_is_complete_explicit_only_and_registered() -> None:
    schema = KREA2_LORA_EXPERIMENTAL_SCHEMA

    assert schema.profile_id == "krea2.raw-turbo-lora.experimental"
    assert schema.profile_version == "1"
    assert schema.model_family == "krea2"
    assert schema.model_variant == "raw_turbo_lora"
    assert schema.evidence is EvidenceLevel.EXPERIMENTAL
    assert schema.detection.resolving_sources == ("explicit_selection",)
    assert schema.detection.suggestion_sources == ()
    assert schema.detection.family_only_sources == ()
    assert schema.recipes[0].recipe_id == "krea2.raw-turbo-lora.experimental"
    assert schema.recipes[0].steps.minimum == 1
    assert schema.recipes[0].steps.maximum == 10_000
    assert schema.recipes[0].steps.default == 12
    assert schema.recipes[0].steps.reference_steps == (12,)
    assert schema.recipes[0].guidance.host_value == 1.0
    assert {field.name: field.value for field in schema.parameters}[
        "official_technical_report_recipe_finding"
    ] == "no_raw_to_turbo_lora_recipe_published"
    assert tuple(weight.weight_id for weight in schema.model_weights) == (
        "comfy_org.krea2.raw_to_turbo_lora.rank64",
        "krea.krea2.raw.weights",
    )
    assert (
        schema.model_weights[0].sha256
        == (
            "db8c5bae0a415d448da9d842111d6e51f7d32e47143a3118eb267e5c4773de87"  # pragma: allowlist secret
        )
    )
    assert builtin_profile_registry().resolve(ProfileKey.from_schema(schema)).schema is schema
    assert KREA2_LORA_EXPERIMENTAL_PROFILE.schema is schema

    with pytest.raises(FrozenInstanceError):
        schema.profile_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("source", "expected_mu"),
    (
        (Krea2ExperimentalMuSource.RAW, 0.90625),
        (Krea2ExperimentalMuSource.TURBO, 1.15),
    ),
)
def test_experimental_shift_derivation_is_explicit(
    source: Krea2ExperimentalMuSource,
    expected_mu: float,
) -> None:
    shift = derive_krea2_lora_experimental_shift(
        width=1024,
        height=1024,
        mu_source=source,
    )

    assert shift.mu_source is source
    assert shift.mu == expected_mu
    assert shift.geometry.image_seq_len == 4096
    assert shift.extrapolated is False


@pytest.mark.parametrize("steps", (1, 4, 8, 12, 16, 37, 10_000))
@pytest.mark.parametrize("source", tuple(Krea2ExperimentalMuSource))
def test_experimental_builder_accepts_arbitrary_bounded_steps(
    steps: int,
    source: Krea2ExperimentalMuSource,
) -> None:
    result = build_krea2_lora_experimental_schedule(
        steps=steps,
        width=1024,
        height=768,
        mu_source=source,
    )

    assert result.request.provenance.evidence is EvidenceLevel.EXPERIMENTAL
    assert result.request.provenance.profile_id == "krea2.raw-turbo-lora.experimental"
    assert result.final_domain is SigmaDomain.UNIT_FLOW
    assert len(result.sigmas) == steps + 1
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert all(left > right for left, right in pairwise(result.sigmas))
    assert any("experimental" in warning.casefold() for warning in result.warnings)


@pytest.mark.parametrize(
    ("steps", "source"),
    (
        (0, Krea2ExperimentalMuSource.RAW),
        (10_001, Krea2ExperimentalMuSource.TURBO),
        (12, "raw"),
    ),
)
def test_experimental_builder_rejects_invalid_contracts(
    steps: int,
    source: object,
) -> None:
    with pytest.raises(ScheduleContractError):
        build_krea2_lora_experimental_schedule(
            steps=steps,
            width=1024,
            height=1024,
            mu_source=source,  # type: ignore[arg-type]
        )
