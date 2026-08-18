"""M4-16 readiness-only MiniMax H3 Turbo public contract tests."""

from __future__ import annotations

import json
import struct
from typing import cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.minimax_h3_sigma_scheduler import (
    MINIMAX_H3_TURBO_RECIPE_CHOICES,
    MiniMaxH3SigmaNodeResult,
    MiniMaxH3SigmaScheduler,
    bind_minimax_h3_sigma_output_info,
    build_minimax_h3_sigma_schedule,
)
from comfyui_sigmax.profiles.minimax_h3_acceleration import MINIMAX_H3_ACCELERATION_ARTIFACTS
from comfyui_sigmax.profiles.minimax_h3_turbo_public import (
    MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID,
    MiniMaxH3TurboPublicReceiptV1,
    build_minimax_h3_turbo_public_receipt,
    canonical_minimax_h3_turbo_task,
    deserialize_minimax_h3_turbo_public_receipt,
    require_minimax_h3_turbo_artifact,
    serialize_minimax_h3_turbo_public_receipt,
)
from comfyui_sigmax.workflows.minimax_h3 import MiniMaxH3WorkflowSpec

_FL2VA_4 = "h3.fl2va.lightx2v-turbo-4-v0.1-544p"
_FL2VA_8 = "h3.fl2va.lightx2v-turbo-8-v1.0-544p"
_FL2VA_768 = "h3.fl2va.lightx2v-turbo-4-v1.0-768p"
_REF2VA_4 = "h3.ref2va.lightx2v-turbo-4-v0.1-544p"
_FL2VA_8_HASH = next(
    item.sha256
    for item in MINIMAX_H3_ACCELERATION_ARTIFACTS
    if item.artifact_id == "lightx2v.fl2v-8.full"
)


def _fingerprint(character: str = "a") -> str:
    return "sha256:" + character * 64


def _info(result: object) -> dict[str, object]:
    schedule_info_json = cast(MiniMaxH3SigmaNodeResult, result).schedule_info_json
    return cast(dict[str, object], json.loads(schedule_info_json))


def _f32(value: float) -> float:
    return float(struct.unpack(">f", struct.pack(">f", value))[0])


def _sorted_recipe_ids() -> tuple[str, ...]:
    return tuple(sorted((_FL2VA_4, _FL2VA_8, _FL2VA_768, _REF2VA_4)))


def test_turbo_selector_is_explicit_and_base_remains_default() -> None:
    inputs = MiniMaxH3SigmaScheduler.INPUT_TYPES()
    assert tuple(inputs["required"]) == ("variant", "steps", "start_step", "end_step")
    optional = inputs["optional"]
    assert tuple(optional) == ("turbo", "recipe_id")
    assert optional["turbo"][0] == MINIMAX_H3_TURBO_RECIPE_CHOICES
    assert optional["recipe_id"][0] == MINIMAX_H3_TURBO_RECIPE_CHOICES
    turbo_options = cast(dict[str, object], optional["turbo"][1])
    legacy_options = cast(dict[str, object], optional["recipe_id"][1])
    assert turbo_options["default"] == "disabled"
    turbo_tooltip = cast(str, turbo_options["tooltip"])
    assert "Experimental" in turbo_tooltip
    assert "community" in turbo_tooltip.lower()
    assert legacy_options["advanced"] is True
    assert ("disabled", *_sorted_recipe_ids()) == MINIMAX_H3_TURBO_RECIPE_CHOICES

    base = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA", steps=4, start_step=0, end_step=-1
    )
    assert base.recipe_id is None
    assert _info(base)["schema"] == "sigmax.minimax-h3-sigma-node/1"
    assert "turbo_receipt" not in _info(base)


@pytest.mark.parametrize(
    ("variant", "recipe_id", "steps", "video_shift"),
    (
        ("H3 Base FL2VA", _FL2VA_4, 4, 12.0),
        ("H3 Base FL2VA", _FL2VA_8, 8, 12.0),
        ("H3 Base FL2VA", _FL2VA_768, 4, 6.0),
        ("H3 Base Ref2VA", _REF2VA_4, 4, 12.0),
    ),
)
def test_turbo_selector_builds_recipe_owned_readiness_schedule(
    variant: str, recipe_id: str, steps: int, video_shift: float
) -> None:
    result = build_minimax_h3_sigma_schedule(
        variant=variant,
        steps=steps,
        start_step=0,
        end_step=-1,
        recipe_id=recipe_id,
    )
    info = _info(result)
    assert result.recipe_id == recipe_id
    assert len(result.sigmas) == steps + 1
    assert result.sigmas[0] == 1.0
    assert result.sigmas[-1] == 0.0
    assert info["mode"] == "turbo_experimental_community"
    assert info["experimental"] == {
        "enabled": True,
        "evidence": "experimental",
        "promotion": "not_claimed",
        "scope": "community_unofficial_turbo_lora",
        "selector": "recipe_id",
    }
    assert info["profile"] == {
        "id": f"minimax-h3.turbo.{recipe_id.removeprefix('h3.')}",
        "recipe_id": recipe_id,
        "version": "1",
    }
    assert cast(dict[str, object], info["shift"])["video"] == video_shift
    receipt = cast(dict[str, object], info["turbo_receipt"])
    assert receipt["schema"] == MINIMAX_H3_TURBO_PUBLIC_SCHEMA_ID
    assert receipt["artifact_status"] == "not_provided"
    assert receipt["validation_status"] == "readiness_only"
    encoded = json.dumps(info, ensure_ascii=False, sort_keys=True)
    assert "C:\\" not in encoded
    assert "/home/" not in encoded


def test_new_turbo_selector_matches_legacy_recipe_path() -> None:
    turbo = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA",
        steps=4,
        start_step=0,
        end_step=-1,
        turbo=_FL2VA_4,
    )
    legacy = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA",
        steps=4,
        start_step=0,
        end_step=-1,
        recipe_id=_FL2VA_4,
    )
    assert turbo.sigmas == legacy.sigmas
    assert turbo.recipe_id == legacy.recipe_id == _FL2VA_4
    assert cast(dict[str, object], _info(turbo)["experimental"])["selector"] == "turbo"


def test_turbo_selector_precedence_is_deterministic_and_conflicts_fail_closed() -> None:
    equal = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA",
        steps=4,
        start_step=0,
        end_step=-1,
        turbo=_FL2VA_4,
        recipe_id=_FL2VA_4,
    )
    disabled_alias = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA",
        steps=4,
        start_step=0,
        end_step=-1,
        turbo=_FL2VA_4,
        recipe_id="disabled",
    )
    assert equal.sigmas == disabled_alias.sigmas
    assert cast(dict[str, object], _info(equal)["experimental"])["selector"] == "turbo+recipe_id"

    with pytest.raises(ScheduleContractError, match="conflict"):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            turbo=_FL2VA_4,
            recipe_id=_FL2VA_8,
        )


def test_turbo_selector_rejects_unknown_wrong_variant_and_unproven_steps() -> None:
    with pytest.raises(ScheduleContractError, match="UNKNOWN_RECIPE"):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base FL2VA", steps=4, start_step=0, end_step=-1, turbo="Turbo"
        )
    with pytest.raises(ScheduleContractError, match=r"task|variant"):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base Ref2VA", steps=4, start_step=0, end_step=-1, recipe_id=_FL2VA_4
        )
    with pytest.raises(ScheduleContractError, match="UNSUPPORTED_RECIPE_NFE"):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base FL2VA", steps=4, start_step=0, end_step=-1, recipe_id=_FL2VA_8
        )


def test_public_task_aliases_are_explicit_and_i2va_is_not_admitted() -> None:
    assert canonical_minimax_h3_turbo_task("FL2VA") == "fl2va"
    assert canonical_minimax_h3_turbo_task("T2VA") == "t2va"
    assert canonical_minimax_h3_turbo_task("Ref2VA") == "ref2va"
    with pytest.raises(ScheduleContractError, match="task"):
        canonical_minimax_h3_turbo_task("I2VA")


def test_public_receipt_is_versioned_allowlisted_redacted_and_deterministic() -> None:
    first = build_minimax_h3_turbo_public_receipt(
        recipe_id=_FL2VA_4,
        task="T2VA",
        nfe=4,
        schedule_fingerprint=_fingerprint("b"),
    )
    second = build_minimax_h3_turbo_public_receipt(
        recipe_id=_FL2VA_4,
        task="T2VA",
        nfe=4,
        schedule_fingerprint=_fingerprint("b"),
    )
    assert isinstance(first, MiniMaxH3TurboPublicReceiptV1)
    assert first.projection() == second.projection()
    assert set(first.projection()) == {
        "artifact_id",
        "artifact_sha256",
        "artifact_status",
        "allowed_nfe",
        "audio_shift",
        "evidence",
        "limitation",
        "nfe",
        "profile_id",
        "recipe_id",
        "receipt_fingerprint",
        "reference_policy",
        "resolution_policy",
        "sampler",
        "schema",
        "schedule_fingerprint",
        "task",
        "validation_status",
        "version",
        "video_shift",
    }
    payload = serialize_minimax_h3_turbo_public_receipt(first)
    assert deserialize_minimax_h3_turbo_public_receipt(payload) == first
    encoded = payload.decode("utf-8")
    assert "C:\\" not in encoded
    assert "prompt" not in encoded
    assert "weights" not in encoded


def test_public_receipt_tamper_and_unknown_hash_fail_closed() -> None:
    receipt = build_minimax_h3_turbo_public_receipt(
        recipe_id=_FL2VA_4,
        nfe=4,
        schedule_fingerprint=_fingerprint("c"),
    )
    tampered = receipt.projection()
    tampered["schedule_fingerprint"] = _fingerprint("d")
    with pytest.raises(ScheduleContractError, match="fingerprint"):
        deserialize_minimax_h3_turbo_public_receipt(
            json.dumps(tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    with pytest.raises(ScheduleContractError, match="SIZE_HASH_MISMATCH"):
        build_minimax_h3_turbo_public_receipt(
            recipe_id=_FL2VA_8,
            artifact_id="lightx2v.fl2v-8.full",
            artifact_sha256=_fingerprint("e"),
            nfe=8,
            schedule_fingerprint=_fingerprint("f"),
        )


def test_turbo_output_binding_rebinds_receipt_to_crossing_values() -> None:
    result = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA", steps=4, start_step=0, end_step=-1, recipe_id=_FL2VA_4
    )
    output = tuple(_f32(value) for value in result.sigmas)
    metadata = bind_minimax_h3_sigma_output_info(result, output_sigmas=output)
    info = cast(dict[str, object], json.loads(metadata))
    receipt = deserialize_minimax_h3_turbo_public_receipt(
        json.dumps(info["turbo_receipt"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    assert receipt.schedule_fingerprint.startswith("sha256:")
    assert receipt.schedule_fingerprint != _fingerprint("a")


def test_model_bound_turbo_workflow_fails_before_host_execution_without_eligible_artifact() -> None:
    with pytest.raises(
        ScheduleContractError,
        match=r"ARTIFACT_NOT_ELIGIBLE|UNAVAILABLE_EXACT_ARTIFACT|evidence",
    ):
        MiniMaxH3WorkflowSpec(
            variant="H3 Base FL2VA",
            prompt="readiness only",
            steps=8,
            recipe_id=_FL2VA_8,
            artifact_id="lightx2v.fl2v-8.full",
        )
    with pytest.raises(ScheduleContractError, match="artifact"):
        MiniMaxH3WorkflowSpec(
            variant="H3 Base FL2VA",
            prompt="readiness only",
            recipe_id=_FL2VA_4,
        )


def test_exact_artifact_requirement_preserves_m6_12_negative_evidence() -> None:
    with pytest.raises(ScheduleContractError, match="UNAVAILABLE_EXACT_ARTIFACT"):
        require_minimax_h3_turbo_artifact(
            recipe_id=_FL2VA_8,
            artifact_id="lightx2v.fl2v-8.full",
            artifact_sha256=_FL2VA_8_HASH,
            nfe=8,
        )
