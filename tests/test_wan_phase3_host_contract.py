"""Model-free H1/H2 host-contract coverage for the Wan family."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.wan_sigma_scheduler import build_wan_sigma_schedule
from scripts import run_comfyui_e2e as harness

_CASES = (
    ("Wan 2.1", "T2V", "Official native", "None", 50),
    ("Wan 2.1", "I2V", "Official native", "480P", 40),
    ("Wan 2.2", "TI2V", "ComfyUI native", "None", 50),
    ("Wan 2.2", "T2V A14B", "Official native", "None", 40),
)


@pytest.mark.parametrize("generation,task,source,resolution,steps", _CASES)
def test_wan_h2_prompt_is_explicit_and_model_free(
    generation: str,
    task: str,
    source: str,
    resolution: str,
    steps: int,
) -> None:
    prompt = harness.build_wan_h2_api_prompt(
        generation=generation,
        task=task,
        source=source,
        resolution=resolution,
        steps=steps,
    )
    assert prompt["1"] == {
        "class_type": "Sigmax.WanSigmaScheduler",
        "inputs": {
            "already_shifted": False,
            "end_step": -1,
            "generation": generation,
            "resolution": resolution,
            "source": source,
            "start_step": 0,
            "steps": steps,
            "strict_source": True,
            "task": task,
        },
    }
    assert prompt["2"] == {
        "class_type": "SigmaxTest.WanScheduleProbe",
        "inputs": {"schedule_info": ["1", 2], "sigmas": ["1", 0]},
    }


@pytest.mark.parametrize("generation,task,source,resolution,steps", _CASES)
def test_wan_h2_verifier_accepts_canonical_model_free_trace(
    generation: str,
    task: str,
    source: str,
    resolution: str,
    steps: int,
) -> None:
    prompt = harness.build_wan_h2_api_prompt(
        generation=generation,
        task=task,
        source=source,
        resolution=resolution,
        steps=steps,
    )
    result = build_wan_sigma_schedule(
        generation=generation,
        task=task,
        source=source,
        resolution=resolution,
        steps=steps,
        strict_source=True,
        start_step=0,
        end_step=-1,
    )
    history: dict[str, object] = {
        "prompt-1": {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "2": {
                    "sigmax_wan_schedule": [
                        json.dumps(
                            {
                                "schedule_info": json.loads(result.schedule_info_json),
                                "sigmas": list(result.sigmas),
                            },
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    ]
                }
            },
        }
    }
    assert prompt["1"] is not None
    verified = harness.verify_wan_h2_history(
        history,
        prompt_id="prompt-1",
        generation=generation,
        task=task,
        source=source,
        resolution=resolution,
        steps=steps,
    )
    assert verified["requested_transitions"] == steps
    assert verified["status"] == "succeeded"


def test_wan_h2_rejects_unqualified_selection() -> None:
    with pytest.raises(ScheduleContractError, match="pinned|unsupported"):
        harness.build_wan_h2_api_prompt(
            generation="Wan 2.2",
            task="Fun-Control",
            source="Official native",
            resolution="None",
            steps=40,
        )
