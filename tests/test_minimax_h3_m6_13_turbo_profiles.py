"""M6-13 pure MiniMax H3 Turbo catalog, schedule, and boundary contracts."""

from __future__ import annotations

import importlib
import json
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError

MODULE_NAME = "comfyui_sigmax.profiles.minimax_h3_turbo"
FIXTURE_PATH = Path(__file__).parent / "golden" / "minimax_h3_turbo_v1.json"
EXPECTED_RECIPE_IDS = (
    "h3.fl2va.lightx2v-turbo-4-v0.1-544p",
    "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
    "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
    "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
)


def _module() -> ModuleType:
    return importlib.import_module(MODULE_NAME)


def _fixture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def test_private_catalog_has_exact_recipe_identity_and_no_public_registration() -> None:
    module = _module()
    assert tuple(profile.recipe_id for profile in module.MINIMAX_H3_TURBO_PROFILES) == tuple(
        sorted(EXPECTED_RECIPE_IDS)
    )
    assert module.MINIMAX_H3_TURBO_SCHEMA_ID == "sigmax.minimax-h3-turbo-pure/1"
    assert module.MINIMAX_H3_TURBO_MODELTECH_REVISION == (
        "a7e148b8dc7db8ad976966060dcc022adf11fc8d"
    )
    assert all(profile.runtime_registered is False for profile in module.MINIMAX_H3_TURBO_PROFILES)
    assert all(profile.runtime_supported is False for profile in module.MINIMAX_H3_TURBO_PROFILES)
    assert all(profile.eligible_artifact_ids == () for profile in module.MINIMAX_H3_TURBO_PROFILES)

    registry = importlib.import_module(
        "comfyui_sigmax.profiles.registry"
    ).builtin_profile_registry()
    assert len(registry.entries) == 46
    assert not any(
        entry.schema.profile_id.startswith("minimax-h3.turbo") for entry in registry.entries
    )

    package = importlib.import_module("comfyui_sigmax.profiles")
    assert not hasattr(package, "MINIMAX_H3_TURBO_PROFILES")


def test_recipe_contracts_freeze_task_resolution_shifts_and_conservative_steps() -> None:
    module = _module()
    profiles = {profile.recipe_id: profile for profile in module.MINIMAX_H3_TURBO_PROFILES}
    assert (
        profiles[EXPECTED_RECIPE_IDS[0]].task,
        profiles[EXPECTED_RECIPE_IDS[0]].video_shift,
    ) == (
        "fl2va",
        12.0,
    )
    assert profiles[EXPECTED_RECIPE_IDS[0]].allowed_nfe == (4,)
    assert profiles[EXPECTED_RECIPE_IDS[1]].allowed_nfe == (8,)
    assert profiles[EXPECTED_RECIPE_IDS[2]].resolution_policy == "1344x768"
    assert (
        profiles[EXPECTED_RECIPE_IDS[2]].video_shift,
        profiles[EXPECTED_RECIPE_IDS[2]].audio_shift,
    ) == (
        6.0,
        3.0,
    )
    assert profiles[EXPECTED_RECIPE_IDS[3]].task == "ref2va"
    assert profiles[EXPECTED_RECIPE_IDS[3]].reference_policy == "explicit_reference_images"


@pytest.mark.parametrize(
    ("recipe_id", "nfe"),
    (
        ("h3.fl2va.lightx2v-turbo-4-v0.1-544p", 4),
        ("h3.fl2va.lightx2v-turbo-8-v1.0-544p", 8),
        ("h3.fl2va.lightx2v-turbo-4-v1.0-768p", 4),
        ("h3.ref2va.lightx2v-turbo-4-v0.1-544p", 4),
    ),
)
def test_endpoint_inclusive_video_and_audio_vectors_match_independent_goldens(
    recipe_id: str, nfe: int
) -> None:
    module = _module()
    fixture_case = next(
        case
        for case in _fixture()["cases"]
        if case["recipe_id"] == recipe_id and case["nfe"] == nfe
    )
    result64 = module.build_minimax_h3_turbo_schedule(recipe_id, nfe=nfe, precision="float64")
    result32 = module.build_minimax_h3_turbo_schedule(recipe_id, nfe=nfe, precision="float32")
    assert len(result64.video_sigmas) == nfe + 1
    assert result64.video_sigmas[0] == 1.0
    assert result64.video_sigmas[-1] == 0.0
    assert all(left >= right for left, right in pairwise(result64.video_sigmas))
    assert result64.video_sigmas == pytest.approx(fixture_case["video_float64"], abs=1e-12)
    assert result64.audio_sigmas == pytest.approx(fixture_case["audio_float64"], abs=1e-12)
    assert result32.video_sigmas == pytest.approx(fixture_case["video_float32"], abs=1e-6)
    assert result32.audio_sigmas == pytest.approx(fixture_case["audio_float32"], abs=1e-6)
    assert result64.fingerprint == fixture_case["fingerprint_float64"]
    assert result32.fingerprint == fixture_case["fingerprint_float32"]
    assert result64.fingerprint == module.minimax_h3_turbo_schedule_fingerprint(result64)
    assert result32.fingerprint == module.minimax_h3_turbo_schedule_fingerprint(result32)
    assert result64.fingerprint != result32.fingerprint


def test_768p_uses_6_over_3_and_not_base_12_over_3() -> None:
    module = _module()
    result = module.build_minimax_h3_turbo_schedule(
        "h3.fl2va.lightx2v-turbo-4-v1.0-768p", nfe=4, precision="float64"
    )
    assert result.video_sigmas == pytest.approx(
        (1.0, 0.9473684210526315, 0.8571428571428571, 0.6666666666666666, 0.0),
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"task": "ref2va"}, "WRONG_TASK"),
        ({"resolution_policy": "1344x768"}, "WRONG_RESOLUTION"),
        ({"video_shift": 6.0}, "DUPLICATE_SHIFT_RISK"),
        ({"audio_shift": 12.0}, "DUPLICATE_SHIFT_RISK"),
        ({"input_already_shifted": True}, "DUPLICATE_SHIFT_RISK"),
        ({"loader_strength": 0.5}, "DUPLICATE_SCALE_RISK"),
    ),
)
def test_schedule_rejects_wrong_ownership_and_double_application(
    kwargs: dict[str, object], reason: str
) -> None:
    module = _module()
    with pytest.raises(module.MiniMaxH3TurboError, match=reason):
        module.build_minimax_h3_turbo_schedule(
            "h3.fl2va.lightx2v-turbo-4-v0.1-544p", nfe=4, **kwargs
        )


def test_unsupported_4_step_v1_profile_and_unknown_recipe_fail_closed() -> None:
    module = _module()
    with pytest.raises(module.MiniMaxH3TurboError, match="UNSUPPORTED_RECIPE_NFE"):
        module.build_minimax_h3_turbo_schedule("h3.fl2va.lightx2v-turbo-8-v1.0-544p", nfe=4)
    with pytest.raises(module.MiniMaxH3TurboError, match="UNKNOWN_RECIPE"):
        module.build_minimax_h3_turbo_schedule("h3.fl2va.unknown", nfe=4)


@pytest.mark.parametrize(
    "artifact_id",
    (
        "unknown-artifact",
        "lightx2v.fl2v-4-768.full",
        "kijai.fl2v-8.reduced",
        "local.ref2v-4.modified",
    ),
)
def test_artifact_identity_and_eligibility_remain_fail_closed(artifact_id: str) -> None:
    module = _module()
    profile = module.get_minimax_h3_turbo_profile("h3.fl2va.lightx2v-turbo-4-v0.1-544p")
    with pytest.raises((module.MiniMaxH3TurboError, ScheduleContractError)):
        module.validate_minimax_h3_turbo_artifact(profile, artifact_id)


@pytest.mark.parametrize(
    ("recipe_id", "artifact_id", "reason"),
    (
        (
            "h3.fl2va.lightx2v-turbo-8-v1.0-544p",
            "lightx2v.fl2v-8.full",
            "UNAVAILABLE_EXACT_ARTIFACT",
        ),
        (
            "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
            "kijai.fl2v-4-768.reduced",
            "UNVERIFIED_LICENSE",
        ),
        (
            "h3.ref2va.lightx2v-turbo-4-v0.1-544p",
            "local.ref2v-4.modified",
            "TASK_METADATA_CONFLICT",
        ),
    ),
)
def test_exact_artifact_records_cannot_be_promoted_to_eligibility(
    recipe_id: str, artifact_id: str, reason: str
) -> None:
    module = _module()
    profile = module.get_minimax_h3_turbo_profile(recipe_id)
    with pytest.raises(ScheduleContractError, match=reason):
        module.validate_minimax_h3_turbo_artifact(profile, artifact_id)


def test_base_module_and_public_surface_remain_unchanged() -> None:
    base = importlib.import_module("comfyui_sigmax.profiles.minimax_h3")
    before = (
        base.MINIMAX_H3_BASE_FL2VA_PROFILE.profile_id,
        base.MINIMAX_H3_BASE_REF2VA_PROFILE.profile_id,
        base.MINIMAX_H3_VIDEO_SHIFT,
        base.MINIMAX_H3_AUDIO_SHIFT,
    )
    module = _module()
    module.build_minimax_h3_turbo_schedule("h3.ref2va.lightx2v-turbo-4-v0.1-544p", nfe=4)
    after = (
        base.MINIMAX_H3_BASE_FL2VA_PROFILE.profile_id,
        base.MINIMAX_H3_BASE_REF2VA_PROFILE.profile_id,
        base.MINIMAX_H3_VIDEO_SHIFT,
        base.MINIMAX_H3_AUDIO_SHIFT,
    )
    assert after == before
