"""Phase 0 contracts for the explicit LTX SIGMAS node."""

from __future__ import annotations

import json

import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.nodes.ltx_sigma_scheduler import (
    LTX_SIGMA_NODE_ID,
    LTX_SIGMA_NODE_SCHEMA_ID,
    LTXSigmaScheduler,
    build_ltx_sigma_schedule,
)


def test_node_is_thin_and_explicit() -> None:
    assert LTX_SIGMA_NODE_ID == "Sigmax.LTXSigmaScheduler"
    assert LTX_SIGMA_NODE_SCHEMA_ID == "sigmax.ltx-sigma-node/1"
    choices = LTXSigmaScheduler.INPUT_TYPES()["required"]
    assert choices["generation"][0] == ("LTXV 0.9.8", "LTX-2 19B", "LTX-2.3 22B")
    assert choices["stage"][0] == ("Dev", "Distilled Stage 1", "Distilled Stage 2")


def test_node_builds_adaptive_schedule_and_canonical_metadata() -> None:
    result = build_ltx_sigma_schedule(
        generation="LTX-2 19B",
        stage="Dev",
        steps=40,
        token_count=4096,
        stretch=True,
        terminal=0.1,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    assert result.domain is SigmaDomain.UNIT_FLOW
    info = json.loads(result.schedule_info_json)
    assert info["profile"] == "ltx2.19b.dev"
    assert info["shift"] == pytest.approx(2.05)
    assert info["token_source"] == "explicit"  # noqa: S105 - provenance label, not a secret
    assert info["fingerprints"]["complete"].startswith("sha256:")


def test_node_rejects_unsupported_ltxv_distilled_selection() -> None:
    with pytest.raises(ScheduleContractError):
        build_ltx_sigma_schedule(
            generation="LTXV 0.9.8",
            stage="Distilled Stage 1",
            steps=8,
            token_count=4096,
            stretch=True,
            terminal=0.1,
            strict_official=True,
            start_step=0,
            end_step=-1,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"stretch": False},
        {"terminal": 0.2},
        {"strict_official": False},
        {"token_count": 1024},
    ),
)
def test_node_rejects_inert_distilled_controls(overrides: dict[str, object]) -> None:
    inputs: dict[str, object] = {
        "generation": "LTX-2 19B",
        "stage": "Distilled Stage 1",
        "steps": 8,
        "token_count": 4096,
        "stretch": True,
        "terminal": 0.1,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
    }
    inputs.update(overrides)
    with pytest.raises(ScheduleContractError, match="distilled"):
        build_ltx_sigma_schedule(**inputs)
