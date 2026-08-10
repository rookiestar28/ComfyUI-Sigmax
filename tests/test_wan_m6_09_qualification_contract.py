"""M6-09 RED contracts for planned Wan identities and source ownership."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from enum import Enum
from typing import Any

import pytest

_EXPECTED_PROFILES = (
    (
        "wan2.1.flf2v.14b.720p.official-native",
        "wan2.1",
        "flf2v",
        "14b",
        "official_native",
        "720p",
        16.0,
        50,
        5.0,
        "cfg",
        "unipc",
        ("dpm++", "unipc"),
        "Wan-AI/Wan2.1-FLF2V-14B-720P",
        "M6-10",
        (),
    ),
    (
        "wan2.1.vace.1.3b.official-native",
        "wan2.1",
        "vace",
        "1.3b",
        "official_native",
        "none",
        16.0,
        50,
        5.0,
        "cfg",
        "unipc",
        ("dpm++", "unipc"),
        "Wan-AI/Wan2.1-VACE-1.3B",
        "M6-10",
        (),
    ),
    (
        "wan2.1.vace.14b.official-native",
        "wan2.1",
        "vace",
        "14b",
        "official_native",
        "none",
        16.0,
        50,
        5.0,
        "cfg",
        "unipc",
        ("dpm++", "unipc"),
        "Wan-AI/Wan2.1-VACE-14B",
        "M6-10",
        (),
    ),
    (
        "wan2.2.s2v.14b.official-native",
        "wan2.2",
        "s2v",
        "14b",
        "official_native",
        "none",
        3.0,
        40,
        4.5,
        "cfg",
        "unipc",
        ("dpm++", "unipc"),
        "Wan-AI/Wan2.2-S2V-14B",
        "M6-10",
        (),
    ),
    (
        "wan2.2.animate.14b.official-native",
        "wan2.2",
        "animate",
        "14b",
        "official_native",
        "none",
        5.0,
        20,
        1.0,
        "cfg",
        "unipc",
        ("dpm++", "unipc"),
        "Wan-AI/Wan2.2-Animate-14B",
        "M6-11",
        (),
    ),
    (
        "wan-animate2.14b.base.official-native",
        "wan-animate2",
        "animate",
        "base-14b",
        "official_native",
        "none",
        5.0,
        20,
        0.0,
        "no_cfg",
        "euler",
        ("euler",),
        "Wan-AI/Wan2.2-Animate-2-14B",
        "M6-11",
        (),
    ),
    (
        "wan-animate2.14b.distilled.official-native",
        "wan-animate2",
        "animate",
        "distilled-14b",
        "official_native",
        "none",
        5.0,
        10,
        1.0,
        "example_override",
        "euler",
        ("euler",),
        "Wan-AI/Wan2.2-Animate-2-14B",
        "M6-11",
        (),
    ),
    (
        "wan-animate2.14b.base.diffusers-reference",
        "wan-animate2",
        "animate",
        "base-14b",
        "framework_reference",
        "none",
        5.0,
        40,
        None,
        "framework_default",
        None,
        ("dpm_solver", "unipc"),
        "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
        "M6-11",
        (
            "framework_guidance_default_unpinned",
            "framework_revision_unpinned",
            "solver_metadata_conflict",
        ),
    ),
    (
        "wan-animate2.14b.distilled.diffusers-reference",
        "wan-animate2",
        "animate",
        "distilled-14b",
        "framework_reference",
        "none",
        5.0,
        10,
        1.0,
        "no_cfg",
        "euler",
        ("euler",),
        "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
        "M6-11",
        ("framework_revision_unpinned",),
    ),
)


def _module() -> Any:
    return __import__("comfyui_sigmax.profiles.wan_qualification", fromlist=["*"])


def test_wan_qualification_module_is_a_separate_non_runtime_seam() -> None:
    assert importlib.util.find_spec("comfyui_sigmax.profiles.wan_qualification") is not None
    module = _module()
    assert module.WAN_QUALIFICATION_SCHEMA_ID == "sigmax.wan-qualification/1"
    assert issubclass(module.WanSourceLane, Enum)
    assert tuple(item.value for item in module.WanSourceLane) == (
        "comfyui_model_native",
        "comfyui_workflow",
        "framework_reference",
        "official_native",
    )


def test_wan_source_pins_separate_software_and_weight_license_scopes() -> None:
    module = _module()
    sources = {source.source_id: source for source in module.WAN_QUALIFICATION_SOURCES}
    expected_revisions = {
        "comfyui.repository": "2a68ce33b4c9ea6ee4283e618a74560cefb32694",  # pragma: allowlist secret
        "wan2.1.repository": "9737cba9c1c3c4d04b33fcad41c111989865d315",  # pragma: allowlist secret
        "wan2.2.repository": "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",  # pragma: allowlist secret
        "wan-animate2.repository": "3ad2fef7d61d6200c9c653e0fe47be7616b323f3",  # pragma: allowlist secret
        "wan2.1.flf2v.14b.720p.card": "c8db168d95d3ebeb63430b3b6d264885cb8a0df3",  # pragma: allowlist secret
        "wan2.1.vace.1.3b.card": "574e6a744642ce3bee319afc31496b88bde8aac4",  # pragma: allowlist secret
        "wan2.1.vace.14b.card": "539c162b1387eac9dc4c20bd3f74671309e76a4c",  # pragma: allowlist secret
        "wan2.2.s2v.14b.card": "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b",  # pragma: allowlist secret
        "wan2.2.animate.14b.card": "cb93a225fbaf1ca100f54e79da8f994995b689b3",  # pragma: allowlist secret
        "wan-animate2.14b.native.card": "6e8f1973bf0abc2aafd517992e8b6d88c3c46e69",  # pragma: allowlist secret
        "wan-animate2.14b.diffusers.card": "a84c891208322be6ea1130b1db95df1baedb0459",  # pragma: allowlist secret
        "wan-animate2.14b.distilled.diffusers.card": (
            "36b185201c469c756601cb0779f6597dda1d6c01"  # pragma: allowlist secret
        ),
    }
    assert {source_id: source.revision for source_id, source in sources.items()} == (
        expected_revisions
    )
    assert all(source.license_id in {"Apache-2.0", "GPL-3.0-only"} for source in sources.values())
    assert {source.scope.value for source in sources.values()} == {"software", "model_weights"}
    assert all(
        source.locators == tuple(sorted(set(source.locators))) for source in sources.values()
    )
    assert {
        source_id: (source.url, source.scope.value, source.locators)
        for source_id, source in sources.items()
        if source.scope.value == "software"
    } == {
        "comfyui.repository": (
            "https://github.com/Comfy-Org/ComfyUI",
            "software",
            (
                "blueprints/Image to Video (Wan 2.2).json",
                "blueprints/Text to Video (Wan 2.2).json",
                "blueprints/Video Inpainting (Wan2.1 VACE).json",
                "comfy/supported_models.py",
                "comfy_extras/nodes_model_advanced.py",
            ),
        ),
        "wan-animate2.repository": (
            "https://github.com/Wan-Video/Wan-Animate-2",
            "software",
            (
                "LICENSE",
                "README.md",
                "infer/wan_animate_2.yaml",
                "infer/wan_animate_2_distillation.yaml",
            ),
        ),
        "wan2.1.repository": (
            "https://github.com/Wan-Video/Wan2.1",
            "software",
            ("LICENSE", "README.md", "generate.py"),
        ),
        "wan2.2.repository": (
            "https://github.com/Wan-Video/Wan2.2",
            "software",
            (
                "LICENSE",
                "README.md",
                "generate.py",
                "wan/animate.py",
                "wan/configs/wan_animate_14B.py",
                "wan/configs/wan_s2v_14B.py",
            ),
        ),
    }
    for profile in module.WAN_PLANNED_PROFILES:
        card = sources[profile.model_card_source_id]
        software = sources[profile.software_source_id]
        assert card.scope.value == "model_weights"
        assert card.url == f"https://huggingface.co/{profile.model_card_id}"
        assert profile.model_card_revision == card.revision
        assert software.scope.value == "software"


@pytest.mark.parametrize(
    (
        "profile_id",
        "family",
        "task",
        "variant",
        "source_lane",
        "resolution",
        "shift",
        "steps",
        "guidance",
        "guidance_mode",
        "solver",
        "allowed_solvers",
        "model_card_id",
        "implementation_item",
        "recipe_blockers",
    ),
    _EXPECTED_PROFILES,
)
def test_every_planned_wan_profile_has_explicit_identity_and_ownership(
    profile_id: str,
    family: str,
    task: str,
    variant: str,
    source_lane: str,
    resolution: str,
    shift: float,
    steps: int,
    guidance: float | None,
    guidance_mode: str,
    solver: str | None,
    allowed_solvers: tuple[str, ...],
    model_card_id: str,
    implementation_item: str,
    recipe_blockers: tuple[str, ...],
) -> None:
    module = _module()
    profiles = {profile.profile_id: profile for profile in module.WAN_PLANNED_PROFILES}
    profile = profiles[profile_id]
    assert profile.family == family
    assert profile.task == task
    assert profile.variant == variant
    assert profile.source_lane.value == source_lane
    assert profile.resolution == resolution
    assert profile.shift == shift
    assert profile.steps == steps
    assert profile.guidance == guidance
    assert profile.guidance_mode == guidance_mode
    assert profile.solver == solver
    assert profile.allowed_solvers == allowed_solvers
    assert profile.model_card_id == model_card_id
    assert profile.implementation_item == implementation_item
    assert profile.recipe_blockers == recipe_blockers
    assert profile.readiness.value == "planned"
    assert profile.runtime_registered is False


def test_successor_items_register_only_their_owned_planned_profiles() -> None:
    module = _module()
    from comfyui_sigmax.profiles.registry import builtin_profile_registry
    from comfyui_sigmax.profiles.wan import WanProfileId

    planned_m6_10 = {
        profile.profile_id
        for profile in module.WAN_PLANNED_PROFILES
        if profile.implementation_item == "M6-10"
    }
    planned_m6_11 = {
        profile.profile_id
        for profile in module.WAN_PLANNED_PROFILES
        if profile.implementation_item == "M6-11"
    }
    runtime_enum = {profile.value for profile in WanProfileId}
    runtime_registry = {entry.schema.profile_id for entry in builtin_profile_registry().entries}
    assert len(planned_m6_10 | planned_m6_11) == len(_EXPECTED_PROFILES)
    assert planned_m6_10 <= runtime_enum
    assert planned_m6_10 <= runtime_registry
    assert planned_m6_11.isdisjoint(runtime_enum)
    assert planned_m6_11.isdisjoint(runtime_registry)


def test_current_comfyui_inheritance_and_nested_workflow_observations_are_distinct() -> None:
    module = _module()
    observations = {
        observation.observation_id: (observation.source_lane.value, observation.shift)
        for observation in module.WAN_COMFYUI_OBSERVATIONS
    }
    assert observations == {
        "comfyui.wan21-t2v.model-default": ("comfyui_model_native", 8.0),
        "comfyui.wan21-i2v.inherited-default": ("comfyui_model_native", 8.0),
        "comfyui.wan21-vace.inherited-default": ("comfyui_model_native", 8.0),
        "comfyui.wan22-s2v.inherited-default": ("comfyui_model_native", 8.0),
        "comfyui.wan22-animate.inherited-default": ("comfyui_model_native", 8.0),
        "comfyui.wan-animate2.model-default": ("comfyui_model_native", 5.0),
        "comfyui.wan22-i2v.workflow-patch": ("comfyui_workflow", 5.0),
        "comfyui.wan22-t2v.workflow-patch": ("comfyui_workflow", 5.0),
        "comfyui.wan21-vace.workflow-patch": ("comfyui_workflow", 5.0),
    }
    assert all(
        observation.official_native is False for observation in module.WAN_COMFYUI_OBSERVATIONS
    )
    observations_by_id = {
        observation.observation_id: observation for observation in module.WAN_COMFYUI_OBSERVATIONS
    }
    assert observations_by_id["comfyui.wan21-i2v.inherited-default"].inheritance == "WAN21_T2V"
    assert observations_by_id["comfyui.wan22-s2v.inherited-default"].inheritance == "WAN21_T2V"
    assert observations_by_id["comfyui.wan22-animate.inherited-default"].inheritance == "WAN21_T2V"
    assert {
        observation.workflow
        for observation in module.WAN_COMFYUI_OBSERVATIONS
        if observation.source_lane.value == "comfyui_workflow"
    } == {
        "blueprints/Image to Video (Wan 2.2).json",
        "blueprints/Text to Video (Wan 2.2).json",
        "blueprints/Video Inpainting (Wan2.1 VACE).json",
    }
    assert all(
        observation.locator == "definitions.subgraphs[*].nodes[type=ModelSamplingSD3]"
        for observation in module.WAN_COMFYUI_OBSERVATIONS
        if observation.source_lane.value == "comfyui_workflow"
    )


def test_conflict_rules_and_detection_fail_closed_with_stable_reasons() -> None:
    module = _module()
    assert tuple(rule.reason_code for rule in module.WAN_CONFLICT_RULES) == (
        "wan.identity.card_revision_mismatch",
        "wan.identity.conflict",
        "wan.identity.incomplete_recipe",
        "wan.identity.runtime_not_implemented",
        "wan.identity.unsupported",
        "wan.identity.weak_signal_only",
    )
    profile = module.qualify_planned_wan_identity(profile_id="wan2.2.s2v.14b.official-native")
    assert profile.profile_id == "wan2.2.s2v.14b.official-native"
    with pytest.raises(module.WanQualificationError) as weak_error:
        module.qualify_planned_wan_identity(weak_name="Wan S2V 14B")
    assert weak_error.value.reason_code == "wan.identity.weak_signal_only"
    with pytest.raises(module.WanQualificationError) as revision_error:
        module.qualify_planned_wan_identity(
            model_card_id="Wan-AI/Wan2.2-S2V-14B", model_card_revision="0" * 40
        )
    assert revision_error.value.reason_code == "wan.identity.card_revision_mismatch"
    with pytest.raises(module.WanQualificationError) as conflict_error:
        module.qualify_planned_wan_identity(
            profile_id="wan2.2.s2v.14b.official-native",
            model_card_id="Wan-AI/Wan2.2-Animate-14B",
            model_card_revision="cb93a225fbaf1ca100f54e79da8f994995b689b3",  # pragma: allowlist secret
        )
    assert conflict_error.value.reason_code == "wan.identity.conflict"
    with pytest.raises(module.WanQualificationError) as incomplete_error:
        module.qualify_planned_wan_identity(
            profile_id="wan-animate2.14b.base.diffusers-reference",
            require_recipe_complete=True,
        )
    assert incomplete_error.value.reason_code == "wan.identity.incomplete_recipe"
    exact_card = module.qualify_planned_wan_identity(
        model_card_id="Wan-AI/Wan2.2-S2V-14B",
        model_card_revision="dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b",  # pragma: allowlist secret
    )
    assert exact_card.profile_id == "wan2.2.s2v.14b.official-native"
    with pytest.raises(module.WanQualificationError) as runtime_error:
        module.qualify_planned_wan_identity(
            profile_id="wan2.2.s2v.14b.official-native", require_runtime=True
        )
    assert runtime_error.value.reason_code == "wan.identity.runtime_not_implemented"


def test_qualification_serialization_is_canonical_and_fingerprinted() -> None:
    module = _module()
    payload = module.serialize_wan_qualification()
    assert payload["schema"] == "sigmax.wan-qualification/1"
    assert tuple(profile["profile_id"] for profile in payload["planned_profiles"]) == tuple(
        sorted(profile[0] for profile in _EXPECTED_PROFILES)
    )
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert module.wan_qualification_fingerprint() == "sha256:" + hashlib.sha256(encoded).hexdigest()
