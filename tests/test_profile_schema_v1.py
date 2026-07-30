"""Contracts for the frozen, dependency-free model-profile schema v1."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    EvidenceLevel,
    ModelCapabilities,
    PredictionType,
    ProfileCapabilities,
    SamplerCapabilities,
    ScheduleContractError,
    ScheduleOwnership,
    SigmaDomain,
    TerminalPolicy,
    TerminalSigma,
    TransformStage,
)
from comfyui_sigmax.profiles import (
    KREA2_RAW_PROFILE,
    KREA2_RAW_SCHEMA,
    KREA2_TURBO_PROFILE,
    KREA2_TURBO_SCHEMA,
    PROFILE_SCHEMA_ID,
    PROFILE_SCHEMA_VERSION,
    ArtifactVersionDeclaration,
    BaseGridDeclaration,
    DetectionDeclaration,
    FrameworkProvenance,
    GuidanceDeclaration,
    InferenceRecipe,
    LicenseDeclaration,
    ModelWeightProvenance,
    ProfileField,
    ProfileSchemaV1,
    SlicingDeclaration,
    SoftwareSourceProvenance,
    StepRangeDeclaration,
    TerminalDeclaration,
    TransformDeclaration,
    profile_schema_fingerprint,
    profile_schema_projection,
)

KREA_REVISION = "db3984fbc6e13b34c0064990fc2d95ac64d00058"  # pragma: allowlist secret
DIFFUSERS_REVISION = "a3608b512ed7248499a44c61d954965ed9bdae4d"  # pragma: allowlist secret
COMFYUI_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret
RAW_WEIGHT_REVISION = "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"  # pragma: allowlist secret
TURBO_WEIGHT_REVISION = "98e0fe118d17c9e3547fbb2e25acdbae2cadf7c7"  # pragma: allowlist secret
RAW_WEIGHT_SHA256 = (
    "f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7"  # pragma: allowlist secret
)
TURBO_WEIGHT_SHA256 = (
    "78bbf8f4165eda19cea3cb06c78089221932a39e2eed8af9da741f942c47ffb3"  # pragma: allowlist secret
)
RAW_SCHEMA_FINGERPRINT = "sha256:4e222aa62ff443db37c67d46ae8995707756048afce92c052f0529f92b6b25db"
TURBO_SCHEMA_FINGERPRINT = "sha256:ba8e1694b48e20301ac76f666d9e163d6960a47922c5188d31eac6a98d8ca97e"


def test_schema_version_and_builtin_bindings_are_frozen() -> None:
    assert PROFILE_SCHEMA_ID == "sigmax.model-profile/1"
    assert PROFILE_SCHEMA_VERSION == "1"
    assert isinstance(KREA2_TURBO_SCHEMA, ProfileSchemaV1)
    assert isinstance(KREA2_RAW_SCHEMA, ProfileSchemaV1)
    assert KREA2_TURBO_PROFILE.schema is KREA2_TURBO_SCHEMA
    assert KREA2_RAW_PROFILE.schema is KREA2_RAW_SCHEMA

    with pytest.raises(FrozenInstanceError):
        KREA2_TURBO_SCHEMA.profile_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        KREA2_RAW_SCHEMA.model_weights[0].resource_version = "2"  # type: ignore[misc]


def test_turbo_schema_declares_complete_schedule_and_recipe_contract() -> None:
    schema = KREA2_TURBO_SCHEMA

    assert schema.schema_id == PROFILE_SCHEMA_ID
    assert schema.schema_version == PROFILE_SCHEMA_VERSION
    assert schema.profile_id == "krea2.turbo.official"
    assert schema.profile_version == "1"
    assert schema.display_name == "Krea 2 Turbo Official"
    assert schema.model_family == "krea2"
    assert schema.model_variant == "turbo"
    assert schema.evidence is EvidenceLevel.OFFICIAL
    assert schema.primary_source_id == "krea.krea2.official"
    assert schema.prediction_type is PredictionType.FLOW_VELOCITY
    assert schema.sigma_domain is SigmaDomain.UNIT_FLOW
    assert schema.ownership is ScheduleOwnership.EXTERNAL_SIGMAS
    assert schema.base_grid == BaseGridDeclaration(
        identifier="krea.reciprocal_step",
        output_domain=SigmaDomain.UNIT_FLOW,
        terminal_included=False,
    )
    assert schema.transforms == (
        TransformDeclaration(
            identifier="krea.exponential_mu",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
            parameters=(ProfileField(name="mu", value=1.15),),
        ),
        TransformDeclaration(
            identifier="terminal.append_zero",
            stage=TransformStage.TERMINAL,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
    )
    assert schema.terminal == TerminalDeclaration(
        policy=TerminalPolicy.APPEND_ZERO,
        sigma=TerminalSigma.ZERO,
        value=0.0,
    )
    assert schema.slicing == SlicingDeclaration(
        supports_step_range=True,
        supports_denoise_tail=True,
        zero_denoise_is_empty=True,
    )
    assert schema.parameters == (
        ProfileField(name="dimension_alignment_mode", value="ceil_multiple"),
        ProfileField(name="dimension_multiple", value=16),
    )

    assert schema.recipes == (
        InferenceRecipe(
            recipe_id="krea2.turbo.official-8",
            evidence=EvidenceLevel.OFFICIAL,
            source_id="krea.krea2.official",
            steps=StepRangeDeclaration(
                minimum=1,
                maximum=None,
                default=8,
                reference_steps=(8,),
                allow_modified=True,
            ),
            guidance=GuidanceDeclaration(
                model_convention="krea.guidance",
                host_convention="comfy.cfg",
                model_value=0.0,
                host_value=1.0,
            ),
        ),
    )


def test_raw_schema_declares_dynamic_shift_and_named_recipe_contracts() -> None:
    schema = KREA2_RAW_SCHEMA
    shift = schema.transforms[0]

    assert schema.profile_id == "krea2.raw.official"
    assert schema.model_variant == "raw"
    assert shift == TransformDeclaration(
        identifier="krea.exponential_mu",
        stage=TransformStage.PRIMARY_TIME_SHIFT,
        input_domain=SigmaDomain.UNIT_FLOW,
        output_domain=SigmaDomain.UNIT_FLOW,
        parameters=(
            ProfileField(name="base_image_seq_len", value=256),
            ProfileField(name="base_mu", value=0.5),
            ProfileField(name="extrapolation", value="upstream_unclamped"),
            ProfileField(name="max_image_seq_len", value=6400),
            ProfileField(name="max_mu", value=1.15),
            ProfileField(name="mode", value="resolution_linear"),
        ),
    )
    assert tuple(recipe.recipe_id for recipe in schema.recipes) == (
        "krea2.raw.diffusers-reference-28",
        "krea2.raw.official-full-52",
    )
    assert tuple(recipe.steps for recipe in schema.recipes) == (
        StepRangeDeclaration(
            minimum=28,
            maximum=28,
            default=28,
            reference_steps=(28,),
            allow_modified=False,
        ),
        StepRangeDeclaration(
            minimum=52,
            maximum=52,
            default=52,
            reference_steps=(52,),
            allow_modified=False,
        ),
    )
    assert tuple(recipe.guidance for recipe in schema.recipes) == (
        GuidanceDeclaration(
            model_convention="krea.guidance",
            host_convention="comfy.cfg",
            model_value=4.5,
            host_value=5.5,
        ),
        GuidanceDeclaration(
            model_convention="krea.guidance",
            host_convention="comfy.cfg",
            model_value=3.5,
            host_value=4.5,
        ),
    )


def test_detection_is_explicit_and_never_promotes_weak_evidence() -> None:
    expected = DetectionDeclaration(
        strategy_id="krea2.variant.evidence-v1",
        strict_default=True,
        ambiguity_requires_explicit=True,
        resolving_sources=(
            "explicit_selection",
            "trusted_profile_metadata",
            "trusted_framework_metadata",
            "verified_sha256",
        ),
        suggestion_sources=("local_header_signal", "filename_signal"),
        family_only_sources=("local_tensor_signal", "model_class_signal"),
    )

    assert KREA2_TURBO_SCHEMA.detection == expected
    assert KREA2_RAW_SCHEMA.detection == expected
    assert not (
        set(expected.suggestion_sources + expected.family_only_sources)
        & set(expected.resolving_sources)
    )


def test_capabilities_are_bound_to_the_same_profile_semantics() -> None:
    for schema, profile in (
        (KREA2_TURBO_SCHEMA, KREA2_TURBO_PROFILE),
        (KREA2_RAW_SCHEMA, KREA2_RAW_PROFILE),
    ):
        assert schema.model_capabilities is profile.model_capabilities
        assert schema.profile_capabilities is profile.profile_capabilities
        assert schema.reference_sampler_capabilities is profile.reference_sampler_capabilities
        assert schema.profile_capabilities.profile_id == schema.profile_id
        assert schema.profile_capabilities.profile_version == schema.profile_version
        assert schema.profile_capabilities.prediction_type is schema.prediction_type
        assert schema.profile_capabilities.sigma_domain is schema.sigma_domain
        assert schema.profile_capabilities.ownership is schema.ownership
        assert schema.profile_capabilities.terminal_sigma is schema.terminal.sigma


def test_artifact_versions_name_only_current_construction_contracts() -> None:
    expected = ArtifactVersionDeclaration(
        numerical_schema="sigmax.numerical-schedule/1",
        construction_schema="sigmax.schedule-artifact/1",
        envelope_schema="sigmax.schedule-artifact-envelope/1",
    )

    assert KREA2_TURBO_SCHEMA.artifact_versions == expected
    assert KREA2_RAW_SCHEMA.artifact_versions == expected
    projection = profile_schema_projection(KREA2_TURBO_SCHEMA)
    assert "execution_receipt" not in json.dumps(projection, sort_keys=True)


def test_provenance_and_licenses_are_separate_versioned_resource_contracts() -> None:
    turbo = KREA2_TURBO_SCHEMA
    raw = KREA2_RAW_SCHEMA

    assert turbo.software_sources == (
        SoftwareSourceProvenance(
            record_version="1",
            source_id="krea.krea2.official",
            resource_version=None,
            revision=KREA_REVISION,
            url="https://github.com/krea-ai/krea-2",
            license=LicenseDeclaration(
                declaration_version="1",
                identifier="Apache-2.0",
                name="Apache License 2.0",
                url="https://www.apache.org/licenses/LICENSE-2.0",
            ),
            locators=("README.md", "inference.py", "sampling.py"),
        ),
    )
    assert turbo.frameworks == (
        FrameworkProvenance(
            record_version="1",
            framework_id="comfyui.krea2.framework",
            resource_version=None,
            revision=COMFYUI_REVISION,
            url="https://github.com/Comfy-Org/ComfyUI",
            license=LicenseDeclaration(
                declaration_version="1",
                identifier="GPL-3.0-only",
                name="GNU General Public License v3.0 only",
                url="https://www.gnu.org/licenses/gpl-3.0.html",
            ),
            locators=(
                "comfy/k_diffusion/sampling.py",
                "comfy/model_sampling.py",
                "comfy/supported_models.py",
            ),
        ),
        FrameworkProvenance(
            record_version="1",
            framework_id="diffusers.krea2.framework",
            resource_version="0.39.0",
            revision=DIFFUSERS_REVISION,
            url="https://github.com/huggingface/diffusers",
            license=LicenseDeclaration(
                declaration_version="1",
                identifier="Apache-2.0",
                name="Apache License 2.0",
                url="https://www.apache.org/licenses/LICENSE-2.0",
            ),
            locators=(
                "src/diffusers/pipelines/krea2/pipeline_krea2.py",
                "src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
            ),
        ),
    )
    assert turbo.model_weights == (
        ModelWeightProvenance(
            record_version="1",
            weight_id="krea.krea2.turbo.weights",
            resource_version="1.0",
            revision=TURBO_WEIGHT_REVISION,
            sha256=TURBO_WEIGHT_SHA256,
            url="https://huggingface.co/krea/Krea-2-Turbo",
            license=LicenseDeclaration(
                declaration_version="1",
                identifier="LicenseRef-Krea-2-Community",
                name="Krea 2 Community License",
                url="https://huggingface.co/krea/Krea-2-Turbo/blob/main/LICENSE.pdf",
            ),
        ),
    )
    assert raw.model_weights == (
        ModelWeightProvenance(
            record_version="1",
            weight_id="krea.krea2.raw.weights",
            resource_version="1.0",
            revision=RAW_WEIGHT_REVISION,
            sha256=RAW_WEIGHT_SHA256,
            url="https://huggingface.co/krea/Krea-2-Raw",
            license=LicenseDeclaration(
                declaration_version="1",
                identifier="LicenseRef-Krea-2-Community",
                name="Krea 2 Community License",
                url="https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf",
            ),
        ),
    )
    assert isinstance(turbo.software_sources[0], SoftwareSourceProvenance)
    assert isinstance(turbo.frameworks[0], FrameworkProvenance)
    assert isinstance(turbo.model_weights[0], ModelWeightProvenance)
    assert turbo.software_sources[0].license != turbo.model_weights[0].license
    assert turbo.frameworks[0].license != turbo.model_weights[0].license


def test_projection_and_fingerprint_are_deterministic_typed_and_resource_separated() -> None:
    projection = profile_schema_projection(KREA2_RAW_SCHEMA)
    payload = json.dumps(projection, allow_nan=False, separators=(",", ":"), sort_keys=True)

    assert projection["schema"] == PROFILE_SCHEMA_ID
    assert projection["schema_version"] == PROFILE_SCHEMA_VERSION
    assert set(cast(dict[str, object], projection["provenance"])) == {
        "frameworks",
        "model_weights",
        "software_sources",
    }
    assert '"base_mu":{"bits":"3fe0000000000000","precision":"float64"}' in payload
    assert '"max_mu":{"bits":"3ff2666666666666","precision":"float64"}' in payload
    assert profile_schema_projection(KREA2_RAW_SCHEMA) == projection
    identity = profile_schema_fingerprint(KREA2_RAW_SCHEMA)
    assert identity == RAW_SCHEMA_FINGERPRINT
    assert profile_schema_fingerprint(KREA2_RAW_SCHEMA) == identity
    assert profile_schema_fingerprint(KREA2_TURBO_SCHEMA) == TURBO_SCHEMA_FINGERPRINT


def test_profile_fields_support_every_bounded_scalar_kind() -> None:
    schema = replace(
        KREA2_TURBO_SCHEMA,
        parameters=(
            ProfileField(name="boolean_value", value=True),
            ProfileField(name="integer_value", value=16),
            ProfileField(name="none_value", value=None),
            ProfileField(name="text_value", value="value"),
        ),
    )

    assert profile_schema_projection(schema)["parameters"] == {
        "boolean_value": True,
        "integer_value": 16,
        "none_value": None,
        "text_value": "value",
    }


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ProfileField(name="value", value="x" * 513),
        lambda: ProfileField(name="value", value=r"C:\private\value"),
        lambda: ProfileField(name="Bad Name", value=1),
        lambda: ProfileField(name="api_key", value="redacted"),
        lambda: ProfileField(name="value_path", value="redacted"),
        lambda: ProfileField(name="value", value=2**53),
        lambda: ProfileField(name="value", value=float("nan")),
        lambda: ProfileField(name="value", value=cast(Any, object())),
        lambda: LicenseDeclaration(
            declaration_version="2",
            identifier="Apache-2.0",
            name="Apache License 2.0",
            url="https://www.apache.org/licenses/LICENSE-2.0",
        ),
        lambda: LicenseDeclaration(
            declaration_version="1",
            identifier="bad license",
            name="Apache License 2.0",
            url="https://www.apache.org/licenses/LICENSE-2.0",
        ),
        lambda: LicenseDeclaration(
            declaration_version="1",
            identifier="Apache-2.0",
            name="",
            url="https://www.apache.org/licenses/LICENSE-2.0",
        ),
        lambda: LicenseDeclaration(
            declaration_version="1",
            identifier="Apache-2.0",
            name="Apache License 2.0",
            url="http://www.apache.org/licenses/LICENSE-2.0",
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA.software_sources[0],
            record_version="2",
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA.software_sources[0],
            revision="main",
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA.software_sources[0],
            license=cast(Any, object()),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA.software_sources[0],
            locators=("sampling.py", "README.md"),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA.frameworks[0],
            framework_id="Bad ID",
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA.model_weights[0],
            resource_version=cast(Any, None),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA.model_weights[0],
            sha256="not-a-sha",
        ),
        lambda: BaseGridDeclaration(
            identifier="grid",
            output_domain=SigmaDomain.MODEL_NATIVE,
            terminal_included=False,
        ),
        lambda: TransformDeclaration(
            identifier="shift",
            stage=cast(Any, "primary"),
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        lambda: BaseGridDeclaration(
            identifier="grid",
            output_domain=SigmaDomain.UNIT_FLOW,
            terminal_included=False,
            parameters=cast(Any, []),
        ),
        lambda: BaseGridDeclaration(
            identifier="grid",
            output_domain=SigmaDomain.UNIT_FLOW,
            terminal_included=False,
            parameters=(cast(Any, object()),),
        ),
        lambda: TransformDeclaration(
            identifier="shift",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=cast(Any, "unit_flow"),
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        lambda: TransformDeclaration(
            identifier="shift",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=SigmaDomain.MODEL_NATIVE,
            output_domain=SigmaDomain.UNIT_FLOW,
        ),
        lambda: TransformDeclaration(
            identifier="shift",
            stage=TransformStage.PRIMARY_TIME_SHIFT,
            input_domain=SigmaDomain.UNIT_FLOW,
            output_domain=SigmaDomain.UNIT_FLOW,
            parameters=(
                ProfileField(name="mu", value=1.0),
                ProfileField(name="mu", value=2.0),
            ),
        ),
        lambda: TerminalDeclaration(
            policy=cast(Any, "append_zero"),
            sigma=TerminalSigma.ZERO,
            value=0.0,
        ),
        lambda: TerminalDeclaration(
            policy=TerminalPolicy.APPEND_ZERO,
            sigma=cast(Any, "zero"),
            value=0.0,
        ),
        lambda: TerminalDeclaration(
            policy=TerminalPolicy.APPEND_ZERO,
            sigma=TerminalSigma.ZERO,
            value=1.0,
        ),
        lambda: TerminalDeclaration(
            policy=TerminalPolicy.PRESERVE,
            sigma=TerminalSigma.NONZERO,
            value=0.0,
        ),
        lambda: TerminalDeclaration(
            policy=TerminalPolicy.APPEND_ZERO,
            sigma=TerminalSigma.NONZERO,
            value=1.0,
        ),
        lambda: SlicingDeclaration(
            supports_step_range=True,
            supports_denoise_tail=True,
            zero_denoise_is_empty=cast(Any, 1),
        ),
        lambda: SlicingDeclaration(
            supports_step_range=True,
            supports_denoise_tail=False,
            zero_denoise_is_empty=True,
        ),
        lambda: GuidanceDeclaration(
            model_convention="krea.guidance",
            host_convention="comfy.cfg",
            model_value=0.0,
            host_value=float("inf"),
        ),
        lambda: StepRangeDeclaration(
            minimum=8,
            maximum=4,
            default=8,
            reference_steps=(8,),
            allow_modified=True,
        ),
        lambda: StepRangeDeclaration(
            minimum=0,
            maximum=8,
            default=8,
            reference_steps=(8,),
            allow_modified=True,
        ),
        lambda: StepRangeDeclaration(
            minimum=1,
            maximum=8,
            default=9,
            reference_steps=(8,),
            allow_modified=True,
        ),
        lambda: StepRangeDeclaration(
            minimum=1,
            maximum=8,
            default=8,
            reference_steps=(),
            allow_modified=True,
        ),
        lambda: StepRangeDeclaration(
            minimum=1,
            maximum=8,
            default=8,
            reference_steps=cast(Any, (False,)),
            allow_modified=True,
        ),
        lambda: StepRangeDeclaration(
            minimum=1,
            maximum=8,
            default=8,
            reference_steps=(9,),
            allow_modified=True,
        ),
        lambda: StepRangeDeclaration(
            minimum=1,
            maximum=None,
            default=8,
            reference_steps=(8, 8),
            allow_modified=True,
        ),
        lambda: StepRangeDeclaration(
            minimum=1,
            maximum=None,
            default=8,
            reference_steps=(8,),
            allow_modified=cast(Any, 1),
        ),
        lambda: StepRangeDeclaration(
            minimum=1,
            maximum=8,
            default=8,
            reference_steps=(8,),
            allow_modified=False,
        ),
        lambda: InferenceRecipe(
            recipe_id="Bad ID",
            evidence=EvidenceLevel.OFFICIAL,
            source_id="krea.krea2.official",
            steps=KREA2_TURBO_SCHEMA.recipes[0].steps,
            guidance=KREA2_TURBO_SCHEMA.recipes[0].guidance,
        ),
        lambda: InferenceRecipe(
            recipe_id="recipe",
            evidence=cast(Any, "official"),
            source_id="krea.krea2.official",
            steps=KREA2_TURBO_SCHEMA.recipes[0].steps,
            guidance=KREA2_TURBO_SCHEMA.recipes[0].guidance,
        ),
        lambda: InferenceRecipe(
            recipe_id="recipe",
            evidence=EvidenceLevel.OFFICIAL,
            source_id="krea.krea2.official",
            steps=cast(Any, object()),
            guidance=KREA2_TURBO_SCHEMA.recipes[0].guidance,
        ),
        lambda: InferenceRecipe(
            recipe_id="recipe",
            evidence=EvidenceLevel.OFFICIAL,
            source_id="krea.krea2.official",
            steps=KREA2_TURBO_SCHEMA.recipes[0].steps,
            guidance=cast(Any, object()),
        ),
        lambda: DetectionDeclaration(
            strategy_id="krea2.variant.evidence-v1",
            strict_default=True,
            ambiguity_requires_explicit=True,
            resolving_sources=("verified_sha256",),
            suggestion_sources=("verified_sha256",),
            family_only_sources=(),
        ),
        lambda: DetectionDeclaration(
            strategy_id="krea2.variant.evidence-v1",
            strict_default=cast(Any, 1),
            ambiguity_requires_explicit=True,
            resolving_sources=("verified_sha256",),
            suggestion_sources=(),
            family_only_sources=(),
        ),
        lambda: DetectionDeclaration(
            strategy_id="krea2.variant.evidence-v1",
            strict_default=True,
            ambiguity_requires_explicit=True,
            resolving_sources=(),
            suggestion_sources=(),
            family_only_sources=(),
        ),
        lambda: DetectionDeclaration(
            strategy_id="krea2.variant.evidence-v1",
            strict_default=True,
            ambiguity_requires_explicit=True,
            resolving_sources=("verified_sha256", "verified_sha256"),
            suggestion_sources=(),
            family_only_sources=(),
        ),
        lambda: DetectionDeclaration(
            strategy_id="krea2.variant.evidence-v1",
            strict_default=True,
            ambiguity_requires_explicit=False,
            resolving_sources=("verified_sha256",),
            suggestion_sources=(),
            family_only_sources=(),
        ),
        lambda: ArtifactVersionDeclaration(
            numerical_schema="sigmax.numerical-schedule/2",
            construction_schema="sigmax.schedule-artifact/1",
            envelope_schema="sigmax.schedule-artifact-envelope/1",
        ),
    ],
)
def test_malformed_nested_schema_declarations_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(KREA2_TURBO_SCHEMA, schema_id="sigmax.model-profile/2"),
        lambda: replace(KREA2_TURBO_SCHEMA, schema_version="2"),
        lambda: replace(KREA2_TURBO_SCHEMA, evidence=cast(Any, "official")),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            prediction_type=cast(Any, "flow_velocity"),
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, sigma_domain=cast(Any, "unit_flow")),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            ownership=ScheduleOwnership.MODEL_NATIVE,
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            sigma_domain=SigmaDomain.MODEL_NATIVE,
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, profile_id="krea2.raw.official"),
        lambda: replace(KREA2_TURBO_SCHEMA, primary_source_id="missing.source"),
        lambda: replace(KREA2_TURBO_SCHEMA, base_grid=None),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            base_grid=replace(
                cast(BaseGridDeclaration, KREA2_TURBO_SCHEMA.base_grid),
                output_domain=SigmaDomain.CONTINUOUS_EDM,
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            base_grid=replace(
                cast(BaseGridDeclaration, KREA2_TURBO_SCHEMA.base_grid),
                terminal_included=True,
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            transforms=tuple(reversed(KREA2_TURBO_SCHEMA.transforms)),
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, transforms=KREA2_TURBO_SCHEMA.transforms[:1]),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            transforms=(
                TransformDeclaration(
                    identifier="shift.domain_change",
                    stage=TransformStage.PRIMARY_TIME_SHIFT,
                    input_domain=SigmaDomain.UNIT_FLOW,
                    output_domain=SigmaDomain.CONTINUOUS_EDM,
                ),
                TransformDeclaration(
                    identifier="terminal.append_zero",
                    stage=TransformStage.TERMINAL,
                    input_domain=SigmaDomain.CONTINUOUS_EDM,
                    output_domain=SigmaDomain.CONTINUOUS_EDM,
                ),
            ),
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, terminal=cast(Any, object())),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            transforms=(
                KREA2_TURBO_SCHEMA.transforms[0],
                replace(
                    KREA2_TURBO_SCHEMA.transforms[1],
                    identifier="terminal.preserve",
                ),
            ),
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, slicing=cast(Any, object())),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            recipes=(
                replace(
                    KREA2_TURBO_SCHEMA.recipes[0],
                    source_id="missing.source",
                ),
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            recipes=(
                KREA2_TURBO_SCHEMA.recipes[0],
                KREA2_TURBO_SCHEMA.recipes[0],
            ),
        ),
        lambda: replace(
            KREA2_RAW_SCHEMA,
            recipes=tuple(reversed(KREA2_RAW_SCHEMA.recipes)),
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, detection=cast(Any, object())),
        lambda: replace(KREA2_TURBO_SCHEMA, artifact_versions=cast(Any, object())),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            transforms=(cast(Any, object()),),
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, software_sources=()),
        lambda: replace(KREA2_TURBO_SCHEMA, frameworks=()),
        lambda: replace(KREA2_TURBO_SCHEMA, model_weights=()),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            software_sources=(
                replace(
                    KREA2_TURBO_SCHEMA.software_sources[0],
                    source_id="comfyui.krea2.framework",
                ),
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            parameters=(
                ProfileField(name="dimension_multiple", value=16),
                ProfileField(name="dimension_alignment_mode", value="ceil_multiple"),
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            model_capabilities=replace(
                KREA2_TURBO_SCHEMA.model_capabilities,
                model_variant="raw",
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            model_capabilities=replace(
                KREA2_TURBO_SCHEMA.model_capabilities,
                accepted_prediction_types=(PredictionType.EPSILON,),
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            profile_capabilities=replace(
                KREA2_TURBO_SCHEMA.profile_capabilities,
                sigma_domain=SigmaDomain.CONTINUOUS_EDM,
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            reference_sampler_capabilities=replace(
                KREA2_TURBO_SCHEMA.reference_sampler_capabilities,
                sampler_id="other.euler",
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            reference_sampler_capabilities=replace(
                KREA2_TURBO_SCHEMA.reference_sampler_capabilities,
                accepted_prediction_types=(PredictionType.EPSILON,),
            ),
        ),
        lambda: replace(
            KREA2_TURBO_SCHEMA,
            known_limitations=("duplicate", "duplicate"),
        ),
        lambda: replace(KREA2_TURBO_SCHEMA, known_limitations=()),
    ],
)
def test_cross_field_schema_mismatches_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()


def test_schema_requires_exact_capability_contract_types() -> None:
    for field_name, invalid in (
        ("model_capabilities", cast(ModelCapabilities, object())),
        ("profile_capabilities", cast(ProfileCapabilities, object())),
        ("reference_sampler_capabilities", cast(SamplerCapabilities, object())),
    ):
        with pytest.raises(ScheduleContractError):
            cast(Any, replace)(KREA2_TURBO_SCHEMA, **{field_name: invalid})


def test_projection_rejects_non_schema_values() -> None:
    with pytest.raises(ScheduleContractError):
        profile_schema_projection(cast(Any, object()))
