"""M6-11 contracts for distinct Wan Animate and Wan-Animate-2 profiles."""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError, numerical_fingerprint
from comfyui_sigmax.profiles.registry import ProfileKey, builtin_profile_registry
from comfyui_sigmax.profiles.schema_v1 import profile_schema_fingerprint

_EXPECTED_NATIVE = {
    "wan2.2.animate.14b.official-native": {
        "generation": "wan2.2",
        "task": "animate",
        "variant": "14b",
        "resolution": "none",
        "shift": 5.0,
        "steps": 20,
        "guidance": 1.0,
        "guidance_mode": "no_cfg",
        "solver": "unipc.multistep",
        "solver_options": "dpm++,unipc",
        "source_id": "wan.video.wan2-2.animate.task-profiles.m6-11",
        "framework_id": "wan.video.wan2-2.animate.native-inference.m6-11",
        "software_revision": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",  # pragma: allowlist secret
        "card_revision": "cb93a225fbaf1ca100f54e79da8f994995b689b3",  # pragma: allowlist secret
        "artifact": "diffusion_pytorch_model-00001-of-00004.safetensors",
        "sha256": "575c2dba750c3b40240fb742a4224453aa97dfbd3c5f5a0086be431cdefdd69c",  # pragma: allowlist secret
        "weight_url": "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B",
    },
    "wan-animate2.14b.base.official-native": {
        "generation": "wan-animate2",
        "task": "animate",
        "variant": "base-14b",
        "resolution": "none",
        "shift": 5.0,
        "steps": 40,
        "guidance": 3.0,
        "guidance_mode": "cfg",
        "solver": "flow_dpm.multistep",
        "solver_options": "flow_dpm",
        "source_id": "wan.video.wan-animate-2.task-profiles.m6-11",
        "framework_id": "wan.video.wan-animate-2.native-inference.m6-11",
        "software_revision": "3ad2fef7d61d6200c9c653e0fe47be7616b323f3",  # pragma: allowlist secret
        "card_revision": "6e8f1973bf0abc2aafd517992e8b6d88c3c46e69",  # pragma: allowlist secret
        "artifact": "wan_animate_2/wan_animate_2_bf16.safetensors",
        "sha256": "48abc389b8d9bba17a7f54a1cd7f1286fd3e3e0e292ddf756721aee324aede09",  # pragma: allowlist secret
        "weight_url": "https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B",
    },
    "wan-animate2.14b.distilled.official-native": {
        "generation": "wan-animate2",
        "task": "animate",
        "variant": "distilled-14b",
        "resolution": "none",
        "shift": 5.0,
        "steps": 10,
        "guidance": 1.0,
        "guidance_mode": "no_cfg",
        "solver": "flow_dpm.multistep",
        "solver_options": "flow_dpm",
        "source_id": "wan.video.wan-animate-2.task-profiles.m6-11",
        "framework_id": "wan.video.wan-animate-2.native-inference.m6-11",
        "software_revision": "3ad2fef7d61d6200c9c653e0fe47be7616b323f3",  # pragma: allowlist secret
        "card_revision": "6e8f1973bf0abc2aafd517992e8b6d88c3c46e69",  # pragma: allowlist secret
        "artifact": "wan_animate_2/wan_animate_2_bf16_distillation.safetensors",
        "sha256": "66161359fc58a1d6c46e14fe2d81c881ccaca440757b1905e01a91902bff29d2",  # pragma: allowlist secret
        "weight_url": "https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B",
    },
}

_BLOCKED_DIFFUSERS = {
    "wan-animate2.14b.base.diffusers-reference": (
        "framework_release_unpinned",
        "framework_schedule_ownership_unresolved",
        "framework_support_pr_unmerged",
        "scheduler_metadata_conflict",
    ),
    "wan-animate2.14b.distilled.diffusers-reference": (
        "framework_release_unpinned",
        "framework_schedule_ownership_unresolved",
        "framework_support_pr_unmerged",
    ),
}

# Protected M6-09/M6-10 schema identities. These values must not be refreshed when adding M6-11.
_PREDECESSOR_SCHEMA_FINGERPRINTS = {
    "wan2.1.i2v.480p.diffusers-reference": "sha256:eb8dd5d7d146286900111a02df0d5b11e32c0336fcdf72c52593d13e4f61a878",
    "wan2.1.i2v.480p.official-native": "sha256:87b284217c39f3b4baa9fbc18b37de7c8d4078a286109f782627009bebc7dae5",
    "wan2.1.i2v.720p.diffusers-reference": "sha256:40db910e7fa2f4095f93793a30a41f68a2f4748126caa0869d340d48675eb5d6",
    "wan2.1.i2v.720p.official-native": "sha256:c858aa2f83dcc636f66c5e689feea0f70e1486e040c7e5499360017285afb5db",
    "wan2.1.t2v.comfy-native": "sha256:2ede1acc8be25440fc9879a62a2f8697db03fa2a8ee9008ee2670bb8d7077cfe",
    "wan2.1.t2v.diffusers-reference": "sha256:214354d2c403a88414db843e8c37d52d5f29f274da75dfbd0f9730f2777cb0c0",
    "wan2.1.t2v.official-native": "sha256:5008d297c4cb1ae3d30c50d2dec09651f24e577ffb23966a3329df31a07ad81c",
    "wan2.2.i2v-a14b.diffusers-reference": "sha256:a61e42d9a12480ec4b6315b65190a0a87951ca13f828716872b3645ea256a00f",
    "wan2.2.i2v-a14b.official-native": "sha256:44e8e810d7a8bd9e781cd72c4f74e502014ad4bab9b3878ce94edb5de095a234",
    "wan2.2.t2v-a14b.diffusers-reference": "sha256:abf5a26203ee7140f0c35049a44fbddc8105ac72c5cfcc97118b0c0a6ffe17d1",
    "wan2.2.t2v-a14b.official-native": "sha256:eb9a8614506e32ecad9368cb69658c605694c3dc0f4820b3a409f289744a6624",
    "wan2.2.ti2v.5b.comfy-native": "sha256:a5c0e1083d29238cc6ba88674af2d0b8c73e7ff591428d7d36b3e924e23670dd",
    "wan2.2.ti2v.5b.diffusers-reference": "sha256:2fe7c2905246c6d2e66f443e5c079d4053fbc3454fb31762ef1c109b3dbd20e3",
    "wan2.1.flf2v.14b.720p.official-native": "sha256:9fbd092e1dd45275ef2be0051fdb18cc91bdc7a7071d178726cbb6bcacfc6f3b",
    "wan2.1.vace.1.3b.official-native": "sha256:827bfbd38f51f8fbc89ce116d55d3624937c016d11d8879f329460b14dd26089",
    "wan2.1.vace.14b.official-native": "sha256:a5f583099d5f6022c90e17e4f15b412df2aa5e9a6815edaeea439491199531de",
    "wan2.2.s2v.14b.official-native": "sha256:7bc547ec5d495bf2c6c013ef83c247a9dfb45e5097e6ea5d99d502547567ab0c",
}

_PREDECESSOR_NUMERICAL_FINGERPRINTS = {
    "wan2.1.i2v.480p.diffusers-reference": "sha256:200b7ff3ca8c7a5fc5dd8149617cf831383967f11977aa3ead14b710bf891332",
    "wan2.1.i2v.480p.official-native": "sha256:200b7ff3ca8c7a5fc5dd8149617cf831383967f11977aa3ead14b710bf891332",
    "wan2.1.i2v.720p.diffusers-reference": "sha256:b63061f223fb88503755a788e198bb2ce2501880c4fbe3d92f5bda3aa0f757b3",
    "wan2.1.i2v.720p.official-native": "sha256:b63061f223fb88503755a788e198bb2ce2501880c4fbe3d92f5bda3aa0f757b3",
    "wan2.1.t2v.comfy-native": "sha256:df485381c329639b7a469ea398e5cc77d3cc9fa5033c1a097be099d56c8c2232",
    "wan2.1.t2v.diffusers-reference": "sha256:de0cff68e0b219fae873cd6beb4d7468e7fdc0d472381ddaf225edb97804e406",
    "wan2.1.t2v.official-native": "sha256:63640d1f280b8208ef399d45bd14a3ef6eecd651431cc12c3e25be45eeea7739",
    "wan2.2.i2v-a14b.diffusers-reference": "sha256:200b7ff3ca8c7a5fc5dd8149617cf831383967f11977aa3ead14b710bf891332",
    "wan2.2.i2v-a14b.official-native": "sha256:b63061f223fb88503755a788e198bb2ce2501880c4fbe3d92f5bda3aa0f757b3",
    "wan2.2.t2v-a14b.diffusers-reference": "sha256:200b7ff3ca8c7a5fc5dd8149617cf831383967f11977aa3ead14b710bf891332",
    "wan2.2.t2v-a14b.official-native": "sha256:42b6524db8e2f827a84480d5ec4aff8336a3e2615f6456ea4780c4eeae0b7e31",
    "wan2.2.ti2v.5b.comfy-native": "sha256:63640d1f280b8208ef399d45bd14a3ef6eecd651431cc12c3e25be45eeea7739",
    "wan2.2.ti2v.5b.diffusers-reference": "sha256:63640d1f280b8208ef399d45bd14a3ef6eecd651431cc12c3e25be45eeea7739",
    "wan2.1.flf2v.14b.720p.official-native": "sha256:67ed8a87707583b5c9076ee413c6f42954d7e7928291fc3430036720a277e1a5",
    "wan2.1.vace.1.3b.official-native": "sha256:67ed8a87707583b5c9076ee413c6f42954d7e7928291fc3430036720a277e1a5",
    "wan2.1.vace.14b.official-native": "sha256:67ed8a87707583b5c9076ee413c6f42954d7e7928291fc3430036720a277e1a5",
    "wan2.2.s2v.14b.official-native": "sha256:200b7ff3ca8c7a5fc5dd8149617cf831383967f11977aa3ead14b710bf891332",
}


def _wan() -> Any:
    return importlib.import_module("comfyui_sigmax.profiles.wan")


def _parameters(schema: Any) -> dict[str, object]:
    return {field.name: field.value for field in schema.parameters}


def test_m6_11_axes_and_exact_native_identities_are_present() -> None:
    module = _wan()
    assert tuple(item.value for item in module.WanGeneration) == (
        "wan2.1",
        "wan2.2",
        "wan-animate2",
    )
    assert tuple(item.value for item in module.WanTask) == (
        "t2v",
        "i2v",
        "ti2v",
        "flf2v",
        "vace",
        "s2v",
        "animate",
    )
    assert tuple(item.value for item in module.WanProfileId if "animate" in item.value) == (
        "wan2.2.animate.14b.official-native",
        "wan-animate2.14b.base.official-native",
        "wan-animate2.14b.distilled.official-native",
        "wan-animate2.14b.comfy-optimized-6.framework-reference",
    )
    assert len(module.WanProfileId) == 21


@pytest.mark.parametrize("profile_id,expected", _EXPECTED_NATIVE.items())
def test_m6_11_native_profiles_bind_effective_recipe_and_provenance(
    profile_id: str, expected: dict[str, object]
) -> None:
    module = _wan()
    enum_id = module.WanProfileId(profile_id)
    profile = module._PROFILES_BY_ID[enum_id]
    definition = module._definition_for(enum_id)
    schema = profile.schema
    parameters = _parameters(schema)
    weight = schema.model_weights[0]

    assert definition.generation.value == expected["generation"]
    assert definition.task.value == expected["task"]
    assert definition.model_variant == expected["variant"]
    assert definition.resolution.value == expected["resolution"]
    assert definition.ratio == expected["shift"]
    assert definition.steps == expected["steps"]
    assert definition.guidance == expected["guidance"]
    assert definition.guidance_mode == expected["guidance_mode"]
    assert parameters["generation"] == expected["generation"]
    assert parameters["guidance_mode"] == expected["guidance_mode"]
    assert parameters["shift"] == expected["shift"]
    assert parameters["solver"] == expected["solver"]
    assert parameters["source_mode"] == "official_native"
    assert parameters["task"] == expected["task"]
    assert parameters["solver_options"] == expected["solver_options"]
    assert schema.evidence is EvidenceLevel.OFFICIAL
    assert schema.primary_source_id == expected["source_id"]
    assert schema.recipes[0].steps.default == expected["steps"]
    assert schema.recipes[0].guidance.host_value == expected["guidance"]
    convention = "cfg_scale" if expected["guidance_mode"] == "cfg" else "none"
    assert schema.recipes[0].guidance.model_convention == convention
    assert schema.recipes[0].guidance.host_convention == convention
    assert schema.software_sources[0].revision == expected["software_revision"]
    assert tuple(source.source_id for source in schema.software_sources) == (expected["source_id"],)
    assert tuple(framework.framework_id for framework in schema.frameworks) == (
        expected["framework_id"],
    )
    assert weight.revision == expected["card_revision"]
    assert weight.resource_version == expected["artifact"]
    assert weight.sha256 == expected["sha256"]
    assert weight.url == expected["weight_url"]
    assert weight.license.identifier == "Apache-2.0"
    assert any("native" in item.lower() for item in schema.known_limitations)


def test_m6_11_native_profiles_are_distinct_even_when_shift_and_card_match() -> None:
    module = _wan()
    base = module._PROFILES_BY_ID[module.WanProfileId("wan-animate2.14b.base.official-native")]
    distilled = module._PROFILES_BY_ID[
        module.WanProfileId("wan-animate2.14b.distilled.official-native")
    ]
    assert base.schema.model_variant != distilled.schema.model_variant
    assert (
        base.schema.model_weights[0].resource_version
        != distilled.schema.model_weights[0].resource_version
    )
    assert base.schema.recipes[0].steps.default != distilled.schema.recipes[0].steps.default
    assert base.schema.model_weights[0].revision == distilled.schema.model_weights[0].revision
    assert base.schema.model_weights[0].sha256 != distilled.schema.model_weights[0].sha256


def test_m6_11_diffusers_candidates_remain_unregistered_with_stable_blockers() -> None:
    module = _wan()
    qualification = importlib.import_module("comfyui_sigmax.profiles.wan_qualification")
    registry_ids = {entry.schema.profile_id for entry in builtin_profile_registry().entries}
    runtime_ids = {item.value for item in module.WanProfileId}
    planned = {item.profile_id: item for item in qualification.WAN_PLANNED_PROFILES}
    for profile_id, blockers in _BLOCKED_DIFFUSERS.items():
        assert profile_id not in runtime_ids
        assert profile_id not in registry_ids
        assert planned[profile_id].recipe_blockers == blockers
        assert planned[profile_id].runtime_registered is False


def test_m6_11_native_profiles_are_exported_registered_in_the_successor_wan_node_v1() -> None:
    module = _wan()
    public = importlib.import_module("comfyui_sigmax.profiles")
    registry = builtin_profile_registry()
    keys = {entry.key for entry in registry.entries}
    for profile_id in _EXPECTED_NATIVE:
        member = module.WanProfileId(profile_id)
        member_name = next(item.name for item in module.WanProfileId if item is member)
        profile = module._PROFILES_BY_ID[member]
        assert getattr(public, f"{member_name}_PROFILE") is profile
        assert getattr(public, f"{member_name}_SCHEMA") is profile.schema
        assert ProfileKey.from_schema(profile.schema) in keys
    node = importlib.import_module("comfyui_sigmax.nodes.wan_sigma_scheduler")
    assert node._TASKS[:5] == ("T2V", "I2V", "TI2V", "T2V A14B", "I2V A14B")
    native_ids = {module.WanProfileId(profile_id) for profile_id in _EXPECTED_NATIVE}
    assert native_ids == {
        profile_id
        for profile_id, _resolution in node._PROFILES.values()
        if profile_id in native_ids
    }
    assert len(registry.entries) == 47


def test_m6_11_schedule_semantics_are_strict_and_single_shifted() -> None:
    module = _wan()
    for profile_id, expected in _EXPECTED_NATIVE.items():
        enum_id = module.WanProfileId(profile_id)
        steps = cast(int, expected["steps"])
        result = module.build_wan_schedule(
            profile=enum_id,
            steps=steps,
            resolution=module.WanResolution.NONE,
            strict_source=True,
        )
        assert result.request.provenance.evidence is EvidenceLevel.OFFICIAL
        assert result.sigmas[0] == 1.0 and result.sigmas[-1] == 0.0
        assert result.request.base_grid.identifier == "flowmatch.reciprocal_step"
        assert tuple(item.name for item in result.request.transforms) == (
            "direct_ratio.shift",
            "terminal.append_zero",
        )
        with pytest.raises(ScheduleContractError, match="already shifted"):
            module.build_wan_schedule(
                profile=enum_id,
                steps=steps,
                resolution=module.WanResolution.NONE,
                already_shifted=True,
            )
        with pytest.raises(ScheduleContractError, match="pinned"):
            module.build_wan_schedule(
                profile=enum_id,
                steps=steps - 1,
                resolution=module.WanResolution.NONE,
                strict_source=True,
            )
        modified = module.build_wan_schedule(
            profile=enum_id,
            steps=steps - 1,
            resolution=module.WanResolution.NONE,
        )
        assert modified.request.provenance.evidence is EvidenceLevel.MODIFIED
        assert modified.warnings


def test_all_seventeen_predecessor_schema_and_numerical_fingerprints_are_unchanged() -> None:
    module = _wan()
    assert len(_PREDECESSOR_SCHEMA_FINGERPRINTS) == 17
    assert len(_PREDECESSOR_NUMERICAL_FINGERPRINTS) == 17
    assert set(_PREDECESSOR_SCHEMA_FINGERPRINTS) == set(_PREDECESSOR_NUMERICAL_FINGERPRINTS)
    for profile_id, schema_fingerprint in _PREDECESSOR_SCHEMA_FINGERPRINTS.items():
        enum_id = module.WanProfileId(profile_id)
        profile = module._PROFILES_BY_ID[enum_id]
        definition = module._definition_for(enum_id)
        result = module.build_wan_schedule(
            profile=enum_id,
            steps=definition.steps,
            resolution=definition.resolution,
            strict_source=True,
        )
        assert profile_schema_fingerprint(profile.schema) == schema_fingerprint
        assert (
            numerical_fingerprint(result.sigmas, domain=result.final_domain, precision="float64")
            == _PREDECESSOR_NUMERICAL_FINGERPRINTS[profile_id]
        )
