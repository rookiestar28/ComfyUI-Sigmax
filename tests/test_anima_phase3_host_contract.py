"""Model-free H1/H2 contract coverage for the Anima host lane."""

from __future__ import annotations

import json
from typing import Any

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles.anima import AnimaVariant, build_anima_schedule
from scripts import run_comfyui_e2e as harness


@pytest.mark.parametrize(
    ("variant", "steps"),
    [("Base (3.0)", 50), ("Aesthetic (3.0)", 50), ("Turbo (3.0)", 8)],
)
def test_anima_h2_prompt_is_explicit_and_model_free(variant: str, steps: int) -> None:
    prompt = harness.build_anima_h2_api_prompt(variant)
    assert prompt["1"] == {
        "class_type": "Sigmax.AnimaSigmaScheduler",
        "inputs": {
            "already_shifted": False,
            "end_step": -1,
            "start_step": 0,
            "steps": steps,
            "strict_source": True,
            "variant": variant,
        },
    }
    assert prompt["2"] == {
        "class_type": "SigmaxTest.AnimaScheduleProbe",
        "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
    }


@pytest.mark.parametrize(
    ("variant_name", "variant", "steps"),
    [
        ("Base (3.0)", AnimaVariant.BASE, 50),
        ("Aesthetic (3.0)", AnimaVariant.AESTHETIC, 50),
        ("Turbo (3.0)", AnimaVariant.TURBO, 8),
    ],
)
def test_anima_h2_verifier_accepts_canonical_model_free_trace(
    variant_name: str, variant: AnimaVariant, steps: int
) -> None:
    result = build_anima_schedule(variant=variant, steps=steps, strict_source=True)
    profile_id = result.request.provenance.profile_id
    assert profile_id is not None
    info: dict[str, Any] = {
        "fingerprints": {
            "complete": "sha256:" + "0" * 64,
            "output": "sha256:" + "1" * 64,
        },
        "profile": {
            "evidence": "framework_reference",
            "id": profile_id,
            "variant": profile_id.split(".")[1]
            if variant is AnimaVariant.AESTHETIC
            else ("base-v1.0" if variant is AnimaVariant.BASE else "turbo-v1.0"),
        },
        "schema": "sigmax.anima-sigma-node/1",
        "shift": {"kind": "rational", "multiplier": 1.0, "shift": 3.0},
        "slicing": {"output_steps": steps},
    }
    # The production node uses the exact model-variant strings; retain them in the fixture.
    info["profile"]["variant"] = {
        AnimaVariant.BASE: "base-v1.0",
        AnimaVariant.AESTHETIC: "aesthetic-v1",
        AnimaVariant.TURBO: "turbo-v1.0",
    }[variant]
    history: dict[str, Any] = {
        "prompt": {},
        "prompt-id": {},
        "history-id": {},
        "trace-id": {},
        "h2": {},
        "prompt-1": {},
    }
    prompt_id = "prompt-1"
    history[prompt_id] = {
        "status": {"completed": True, "status_str": "success"},
        "outputs": {
            "2": {
                "sigmax_anima_schedule": [
                    json.dumps(
                        {"schedule_info": info, "sigmas": list(result.sigmas)},
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                ]
            }
        },
    }
    verified = harness.verify_anima_h2_history(history, prompt_id=prompt_id, variant=variant_name)
    assert verified["profile_id"] == result.request.provenance.profile_id
    assert verified["requested_transitions"] == steps
    assert verified["status"] == "succeeded"


def test_anima_h2_verifier_rejects_implicit_variant() -> None:
    with pytest.raises(ScheduleContractError, match="explicit"):
        harness.build_anima_h2_api_prompt("auto")
