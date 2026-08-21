"""M4-18 exact ComfyUI optimized Wan Animate 2 contracts."""

from __future__ import annotations

import importlib
import json
import math
from itertools import pairwise

import pytest
from comfyui_sigmax.core import (
    EvidenceLevel,
    ScheduleContractError,
    SigmaDomain,
    numerical_fingerprint,
)
from comfyui_sigmax.profiles.registry import ProfileKey, builtin_profile_registry
from comfyui_sigmax.profiles.schema_v1 import profile_schema_fingerprint

_PROFILE_ID = "wan-animate2.14b.comfy-optimized-6.framework-reference"
_EXPECTED_SIGMAS = (
    1.0,
    0.9617158671586715,
    0.9092148309705561,
    0.8333333333333334,
    0.7148972602739726,
    0.500599520383693,
    0.0,
)
_PREDECESSOR_FINGERPRINTS = {
    "wan-animate2.14b.base.official-native": (
        "sha256:4e56370bfba3fd42b68cb8892756a0958c1004a07b55127fe53010ed445d7757",
        "sha256:b63061f223fb88503755a788e198bb2ce2501880c4fbe3d92f5bda3aa0f757b3",
    ),
    "wan-animate2.14b.distilled.official-native": (
        "sha256:a368865fbd33afea577b3a878113fc3b766531dd64542b9f35ecc0266e049522",
        "sha256:049be0cb724103f3ea898d52b5dbef060635afc2d5250ced0f365190c23aa5ff",
    ),
}


def _parameters(schema: object) -> dict[str, object]:
    return {field.name: field.value for field in schema.parameters}  # type: ignore[attr-defined]


def test_comfyui_simple_discrete_flow_grid_uses_integer_table_indices() -> None:
    core = importlib.import_module("comfyui_sigmax.core")
    assert core.comfyui_simple_discrete_flow_grid(6) == (
        1.0,
        0.834,
        0.667,
        0.5,
        0.334,
        0.167,
    )
    assert core.comfyui_simple_discrete_flow_grid(1) == (1.0,)
    with pytest.raises(ScheduleContractError, match="UNIT_FLOW"):
        core.comfyui_simple_discrete_flow_grid(6, domain=SigmaDomain.CONTINUOUS_EDM)


def test_comfy_optimized_profile_is_exact_and_artifact_coupled() -> None:
    wan = importlib.import_module("comfyui_sigmax.profiles.wan")
    profile_id = wan.WanProfileId(_PROFILE_ID)
    profile = wan._PROFILES_BY_ID[profile_id]
    definition = wan._definition_for(profile_id)
    schema = profile.schema
    parameters = _parameters(schema)

    assert profile_id is wan.WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6
    assert definition.source is wan.WanSource.COMFY_NATIVE
    assert definition.resolution is wan.WanResolution.P480
    assert definition.steps == 6
    assert definition.ratio == 5.0
    assert definition.guidance == 1.0
    assert definition.guidance_mode == "no_cfg"
    assert definition.sampler_id == "lcm"
    assert schema.evidence is EvidenceLevel.FRAMEWORK_REFERENCE
    assert schema.base_grid.identifier == "comfyui.simple_discrete_flow"
    assert schema.recipes[0].steps.minimum == 6
    assert schema.recipes[0].steps.maximum == 6
    assert schema.recipes[0].steps.default == 6
    assert schema.recipes[0].steps.reference_steps == (6,)
    assert schema.recipes[0].steps.allow_modified is False
    assert parameters["scheduler"] == "simple"
    assert parameters["solver"] == "lcm"
    assert parameters["recommended_frames"] == 81
    assert parameters["frame_step"] == 4
    assert parameters["frame_overlap"] == 1
    assert parameters["fps_min"] == 16
    assert parameters["fps_max"] == 24
    assert tuple(weight.resource_version for weight in schema.model_weights) == (
        "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "diffusion_models/wan_animate_2_int8_convrot.safetensors",
    )
    assert (
        {weight.sha256 for weight in schema.model_weights}
        == {
            "0580ecdd65e47e97c30df9670d13a6c4a131d26de5a1faf2ccc78392d5167584",  # pragma: allowlist secret
            "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3",  # pragma: allowlist secret
        }
    )
    assert all(weight.license.identifier == "Apache-2.0" for weight in schema.model_weights)
    assert any(
        framework.revision == "76135e557da1ec7dcb270160f01e597565e3e003"  # pragma: allowlist secret
        for framework in schema.frameworks
    )
    assert any(
        source.revision == "e95e3b20567bea8df16510c8390b7f897b7e6d4b"  # pragma: allowlist secret
        for source in schema.software_sources
    )


def test_comfy_optimized_schedule_matches_exact_six_step_vector_and_fails_closed() -> None:
    wan = importlib.import_module("comfyui_sigmax.profiles.wan")
    result = wan.build_wan_schedule(
        profile=wan.WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6,
        steps=6,
        resolution=wan.WanResolution.P480,
        strict_source=True,
    )
    assert result.request.provenance.evidence is EvidenceLevel.FRAMEWORK_REFERENCE
    assert result.request.base_grid.identifier == "comfyui.simple_discrete_flow"
    assert result.sigmas == pytest.approx(_EXPECTED_SIGMAS, rel=0.0, abs=1e-15)
    assert len(result.sigmas) == 7
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result.sigmas)
    assert all(left > right for left, right in pairwise(result.sigmas))
    with pytest.raises(ScheduleContractError, match="exactly 6 steps"):
        wan.build_wan_schedule(
            profile=wan.WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6,
            steps=7,
            resolution=wan.WanResolution.P480,
        )
    with pytest.raises(ScheduleContractError, match="already shifted"):
        wan.build_wan_schedule(
            profile=wan.WanProfileId.WAN_ANIMATE2_COMFY_OPTIMIZED_6,
            steps=6,
            resolution=wan.WanResolution.P480,
            already_shifted=True,
        )


def test_comfy_optimized_profile_is_exported_registered_and_node_selectable() -> None:
    wan = importlib.import_module("comfyui_sigmax.profiles.wan")
    public = importlib.import_module("comfyui_sigmax.profiles")
    node = importlib.import_module("comfyui_sigmax.nodes.wan_sigma_scheduler")
    profile = wan.WAN_ANIMATE2_COMFY_OPTIMIZED_6_PROFILE

    assert public.WAN_ANIMATE2_COMFY_OPTIMIZED_6_PROFILE is profile
    assert public.WAN_ANIMATE2_COMFY_OPTIMIZED_6_SCHEMA is profile.schema
    assert ProfileKey.from_schema(profile.schema) in {
        entry.key for entry in builtin_profile_registry().entries
    }
    assert len(builtin_profile_registry().entries) == 47
    assert "Animate Optimized" in node._TASKS
    result = node.build_wan_sigma_schedule(
        generation="Wan Animate 2",
        task="Animate Optimized",
        source="ComfyUI native",
        resolution="480P",
        steps=6,
        strict_source=True,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert result.sigmas == pytest.approx(_EXPECTED_SIGMAS, rel=0.0, abs=1e-15)
    assert info["solver_ownership"] == "lcm"
    assert info["scheduler"] == "simple"
    assert info["frame_guidance"] == {
        "fps_max": 24,
        "fps_min": 16,
        "frame_overlap": 1,
        "frame_step": 4,
        "recommended_frames": 81,
    }


def test_base_and_distilled_predecessor_fingerprints_are_unchanged() -> None:
    wan = importlib.import_module("comfyui_sigmax.profiles.wan")
    for profile_value, (
        schema_fingerprint,
        schedule_fingerprint,
    ) in _PREDECESSOR_FINGERPRINTS.items():
        profile_id = wan.WanProfileId(profile_value)
        profile = wan._PROFILES_BY_ID[profile_id]
        definition = wan._definition_for(profile_id)
        result = wan.build_wan_schedule(
            profile=profile_id,
            steps=definition.steps,
            resolution=definition.resolution,
            strict_source=True,
        )
        assert profile_schema_fingerprint(profile.schema) == schema_fingerprint
        assert (
            numerical_fingerprint(
                result.sigmas,
                domain=SigmaDomain.UNIT_FLOW,
                precision="float64",
            )
            == schedule_fingerprint
        )
