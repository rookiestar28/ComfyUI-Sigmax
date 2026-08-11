"""M6-10 contracts for official Wan FLF2V, VACE, and S2V task profiles."""

from __future__ import annotations

import importlib
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    EvidenceLevel,
    ScheduleContractError,
    numerical_fingerprint,
)
from comfyui_sigmax.profiles.registry import ProfileKey, builtin_profile_registry
from comfyui_sigmax.profiles.schema_v1 import profile_schema_fingerprint

_BASELINE_SCHEMA_FINGERPRINTS = {
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
}

_BASELINE_NUMERICAL_FINGERPRINTS = {
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
}

_NEW_PROFILE_NAMES = (
    "WAN21_FLF2V_14B_720P_OFFICIAL",
    "WAN21_VACE_1_3B_OFFICIAL",
    "WAN21_VACE_14B_OFFICIAL",
    "WAN22_S2V_14B_OFFICIAL",
)

_NEW_FINGERPRINTS = {
    "wan2.1.flf2v.14b.720p.official-native": (
        "sha256:9fbd092e1dd45275ef2be0051fdb18cc91bdc7a7071d178726cbb6bcacfc6f3b",
        "sha256:67ed8a87707583b5c9076ee413c6f42954d7e7928291fc3430036720a277e1a5",
        "sha256:a91e2e017d32d7ad7fd6072136bfd92569f581eaecb56765bd67a8fbc5ffb3d5",
    ),
    "wan2.1.vace.1.3b.official-native": (
        "sha256:827bfbd38f51f8fbc89ce116d55d3624937c016d11d8879f329460b14dd26089",
        "sha256:67ed8a87707583b5c9076ee413c6f42954d7e7928291fc3430036720a277e1a5",
        "sha256:a91e2e017d32d7ad7fd6072136bfd92569f581eaecb56765bd67a8fbc5ffb3d5",
    ),
    "wan2.1.vace.14b.official-native": (
        "sha256:a5f583099d5f6022c90e17e4f15b412df2aa5e9a6815edaeea439491199531de",
        "sha256:67ed8a87707583b5c9076ee413c6f42954d7e7928291fc3430036720a277e1a5",
        "sha256:a91e2e017d32d7ad7fd6072136bfd92569f581eaecb56765bd67a8fbc5ffb3d5",
    ),
    "wan2.2.s2v.14b.official-native": (
        "sha256:7bc547ec5d495bf2c6c013ef83c247a9dfb45e5097e6ea5d99d502547567ab0c",
        "sha256:200b7ff3ca8c7a5fc5dd8149617cf831383967f11977aa3ead14b710bf891332",
        "sha256:b4408570757b81d652fbddd0d8fa4bc8951ffcfbb344da7d0edf7ed842f97967",
    ),
}

_EXPECTED_RUNTIME = {
    "wan2.1.flf2v.14b.720p.official-native": {
        "card_revision": "c8db168d95d3ebeb63430b3b6d264885cb8a0df3",  # pragma: allowlist secret
        "guidance": 5.0,
        "resolution": "720p",
        "sha256": "c8644162efd3f6f7407daeff84f2e54f285cd3b2553e4c7282c0c7299c896df6",  # pragma: allowlist secret
        "shift": 16.0,
        "steps": 50,
        "task": "flf2v",
        "weight_path": "diffusion_pytorch_model-00001-of-00007.safetensors",
    },
    "wan2.1.vace.1.3b.official-native": {
        "card_revision": "574e6a744642ce3bee319afc31496b88bde8aac4",  # pragma: allowlist secret
        "guidance": 5.0,
        "resolution": "none",
        "sha256": "c46a6f5f7d32c453c3983bbc59761ea41cd02ad584fb55d1a7ee2b76145847a2",  # pragma: allowlist secret
        "shift": 16.0,
        "steps": 50,
        "task": "vace",
        "weight_path": "diffusion_pytorch_model.safetensors",
    },
    "wan2.1.vace.14b.official-native": {
        "card_revision": "539c162b1387eac9dc4c20bd3f74671309e76a4c",  # pragma: allowlist secret
        "guidance": 5.0,
        "resolution": "none",
        "sha256": "569d54a07279b89f8281421fccf27ee2459ea853ce6845d3536b8664b0070078",  # pragma: allowlist secret
        "shift": 16.0,
        "steps": 50,
        "task": "vace",
        "weight_path": "diffusion_pytorch_model-00001-of-00007.safetensors",
    },
    "wan2.2.s2v.14b.official-native": {
        "card_revision": "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b",  # pragma: allowlist secret
        "guidance": 4.5,
        "resolution": "none",
        "sha256": "5fb54febf10b729a6da7da222625d6ecaedde78becda01efbc13f6bebaeb6d43",  # pragma: allowlist secret
        "shift": 3.0,
        "steps": 40,
        "task": "s2v",
        "weight_path": "diffusion_pytorch_model-00001-of-00004.safetensors",
    },
}


def _wan() -> Any:
    return importlib.import_module("comfyui_sigmax.profiles.wan")


def _profile_for(module: Any, member_name: str) -> Any:
    return getattr(module, f"{member_name}_PROFILE")


def _parameters(schema: Any) -> dict[str, object]:
    return {field.name: field.value for field in schema.parameters}


def test_four_exact_m6_10_profile_identities_exist() -> None:
    module = _wan()
    expected_ids = tuple(_EXPECTED_RUNTIME)
    actual = tuple(getattr(module.WanProfileId, name).value for name in _NEW_PROFILE_NAMES)
    assert actual == expected_ids
    assert module.WanTask.FLF2V.value == "flf2v"
    assert module.WanTask.VACE.value == "vace"
    assert module.WanTask.S2V.value == "s2v"
    assert len(module.WanProfileId) == 20


@pytest.mark.parametrize("member_name", _NEW_PROFILE_NAMES)
def test_new_profiles_match_m6_09_identity_recipe_and_exact_artifact(member_name: str) -> None:
    module = _wan()
    qualification = importlib.import_module("comfyui_sigmax.profiles.wan_qualification")
    profile = _profile_for(module, member_name)
    schema = profile.schema
    expected = _EXPECTED_RUNTIME[schema.profile_id]
    planned = next(
        item
        for item in qualification.WAN_PLANNED_PROFILES
        if item.profile_id == schema.profile_id and item.implementation_item == "M6-10"
    )
    parameters = _parameters(schema)
    weight = schema.model_weights[0]

    assert schema.evidence is EvidenceLevel.OFFICIAL
    assert schema.primary_source_id.endswith("m6-10")
    assert schema.model_variant == planned.variant
    assert parameters["task"] == expected["task"] == planned.task
    assert parameters["resolution_class"] == expected["resolution"] == planned.resolution
    assert parameters["shift"] == expected["shift"] == planned.shift
    assert parameters["solver"] == "unipc.multistep"
    assert parameters["solver_options"] == "dpm++,unipc"
    assert schema.recipes[0].steps.default == expected["steps"] == planned.steps
    assert schema.recipes[0].guidance.host_value == expected["guidance"] == planned.guidance
    assert weight.revision == expected["card_revision"] == planned.model_card_revision
    assert weight.resource_version == expected["weight_path"]
    assert weight.sha256 == expected["sha256"]
    assert weight.license.identifier == "Apache-2.0"
    assert any("DPM++" in item for item in schema.known_limitations)


def test_new_profiles_are_exported_and_registered_in_the_successor_wan_node_v1() -> None:
    module = _wan()
    public = importlib.import_module("comfyui_sigmax.profiles")
    registry = builtin_profile_registry()
    keys = {entry.key for entry in registry.entries}
    for member_name in _NEW_PROFILE_NAMES:
        profile = _profile_for(module, member_name)
        assert getattr(public, f"{member_name}_PROFILE") is profile
        assert getattr(public, f"{member_name}_SCHEMA") is profile.schema
        assert ProfileKey.from_schema(profile.schema) in keys

    node = importlib.import_module("comfyui_sigmax.nodes.wan_sigma_scheduler")
    assert node.WAN_SIGMA_NODE_SCHEMA_ID == "sigmax.wan-sigma-node/1"
    assert node._TASKS[:5] == ("T2V", "I2V", "TI2V", "T2V A14B", "I2V A14B")
    assert node._TASKS[5:] == (
        "FLF2V",
        "VACE 1.3B",
        "VACE 14B",
        "S2V",
        "Animate",
        "Animate Base",
        "Animate Distilled",
    )
    new_ids = {getattr(module.WanProfileId, name) for name in _NEW_PROFILE_NAMES}
    assert new_ids == {
        profile_id for profile_id, _resolution in node._PROFILES.values() if profile_id in new_ids
    }


@pytest.mark.parametrize("member_name", _NEW_PROFILE_NAMES)
def test_new_profile_schema_and_numerical_fingerprints_are_exact(member_name: str) -> None:
    module = _wan()
    profile = _profile_for(module, member_name)
    definition = module._definition_for(profile.profile)
    result = module.build_wan_schedule(
        profile=profile.profile,
        steps=definition.steps,
        resolution=definition.resolution,
        strict_source=True,
    )
    schema_fingerprint, numerical_float64, numerical_float32 = _NEW_FINGERPRINTS[profile.profile_id]
    assert profile_schema_fingerprint(profile.schema) == schema_fingerprint
    assert (
        numerical_fingerprint(result.sigmas, domain=result.final_domain, precision="float64")
        == numerical_float64
    )
    assert (
        numerical_fingerprint(result.sigmas, domain=result.final_domain, precision="float32")
        == numerical_float32
    )


@pytest.mark.parametrize("member_name", _NEW_PROFILE_NAMES)
def test_new_profiles_enforce_resolution_strict_source_and_no_second_shift(
    member_name: str,
) -> None:
    module = _wan()
    profile_id = getattr(module.WanProfileId, member_name)
    profile = _profile_for(module, member_name)
    expected = _EXPECTED_RUNTIME[profile.profile_id]
    resolution = module.WanResolution(expected["resolution"])
    steps = cast(int, expected["steps"])

    official = module.build_wan_schedule(
        profile=profile_id,
        steps=steps,
        resolution=resolution,
        strict_source=True,
    )
    assert official.request.provenance.evidence is EvidenceLevel.OFFICIAL
    assert official.effective_inputs.steps == steps
    assert len(official.sigmas) == steps + 1
    assert official.sigmas[0] == 1.0 and official.sigmas[-1] == 0.0

    modified = module.build_wan_schedule(
        profile=profile_id,
        steps=steps - 1,
        resolution=resolution,
    )
    assert modified.request.provenance.evidence is EvidenceLevel.MODIFIED
    assert modified.warnings
    with pytest.raises(ScheduleContractError, match="pinned"):
        module.build_wan_schedule(
            profile=profile_id,
            steps=steps - 1,
            resolution=resolution,
            strict_source=True,
        )
    with pytest.raises(ScheduleContractError, match="already shifted"):
        module.build_wan_schedule(
            profile=profile_id,
            steps=steps,
            resolution=resolution,
            already_shifted=True,
        )
    wrong = (
        module.WanResolution.NONE
        if resolution is module.WanResolution.P720
        else module.WanResolution.P720
    )
    with pytest.raises(ScheduleContractError, match="resolution"):
        module.build_wan_schedule(profile=profile_id, steps=steps, resolution=wrong)


def test_all_thirteen_accepted_wan_profiles_keep_exact_schema_and_numerical_fingerprints() -> None:
    module = _wan()
    existing_ids = tuple(
        profile_id
        for profile_id in module.WanProfileId
        if profile_id.value in _BASELINE_SCHEMA_FINGERPRINTS
    )
    assert len(existing_ids) == 13
    for profile_id in existing_ids:
        profile = module._PROFILES_BY_ID[profile_id]
        definition = module._definition_for(profile_id)
        result = module.build_wan_schedule(
            profile=profile_id,
            steps=definition.steps,
            resolution=definition.resolution,
            strict_source=True,
        )
        assert (
            profile_schema_fingerprint(profile.schema)
            == _BASELINE_SCHEMA_FINGERPRINTS[profile_id.value]
        )
        assert (
            numerical_fingerprint(
                result.sigmas,
                domain=result.final_domain,
                precision="float64",
            )
            == _BASELINE_NUMERICAL_FINGERPRINTS[profile_id.value]
        )
