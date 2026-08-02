"""Model-free H1/H2 host-contract coverage for the LTX family."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.ltx_sigma_scheduler import build_ltx_sigma_schedule
from scripts import run_comfyui_e2e as harness

_CASES = (
    ("LTXV 0.9.8", "Dev", 20),
    ("LTX-2 19B", "Distilled Stage 1", 8),
    ("LTX-2.3 22B", "Dev", 30),
    ("LTX-2.3 22B", "Distilled Stage 2", 3),
)


@pytest.mark.parametrize("generation,stage,steps", _CASES)
def test_ltx_h2_prompt_is_explicit_and_model_free(generation: str, stage: str, steps: int) -> None:
    prompt = harness.build_ltx_h2_api_prompt(
        generation=generation,
        stage=stage,
        steps=steps,
    )
    assert prompt["1"] == {
        "class_type": "Sigmax.LTXSigmaScheduler",
        "inputs": {
            "end_step": -1,
            "generation": generation,
            "stage": stage,
            "steps": steps,
            "start_step": 0,
            "stretch": True,
            "strict_official": True,
            "terminal": 0.1,
            "token_count": 4096,
        },
    }
    assert prompt["2"] == {
        "class_type": "SigmaxTest.LTXScheduleProbe",
        "inputs": {"schedule_info": ["1", 1], "sigmas": ["1", 0]},
    }


@pytest.mark.parametrize("generation,stage,steps", _CASES)
def test_ltx_h2_verifier_accepts_canonical_model_free_trace(
    generation: str, stage: str, steps: int
) -> None:
    prompt = harness.build_ltx_h2_api_prompt(
        generation=generation,
        stage=stage,
        steps=steps,
    )
    result = build_ltx_sigma_schedule(
        generation=generation,
        stage=stage,
        steps=steps,
        token_count=4096,
        stretch=True,
        terminal=0.1,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    history: dict[str, object] = {
        "prompt-1": {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "2": {
                    "sigmax_ltx_schedule": [
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
    verified = harness.verify_ltx_h2_history(
        history,
        prompt_id="prompt-1",
        generation=generation,
        stage=stage,
        steps=steps,
    )
    assert verified["requested_transitions"] == steps
    assert verified["status"] == "succeeded"


def test_ltx_h2_rejects_unqualified_selection() -> None:
    with pytest.raises(ScheduleContractError, match=r"pinned|unsupported"):
        harness.build_ltx_h2_api_prompt(
            generation="LTXV 0.9.8",
            stage="Distilled Stage 1",
            steps=8,
        )
