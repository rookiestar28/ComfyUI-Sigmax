"""MiniMax H3 scheduler/profile numerical and node contracts."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from comfyui_sigmax import NODE_CLASS_MAPPINGS
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.minimax_h3_sigma_scheduler import (
    MINIMAX_H3_SIGMA_NODE_ID,
    MINIMAX_H3_SIGMA_NODE_SCHEMA_ID,
    MiniMaxH3SigmaScheduler,
    build_minimax_h3_sigma_schedule,
)
from comfyui_sigmax.profiles.minimax_h3 import MiniMaxH3Variant


def test_minimax_h3_node_will_be_registered() -> None:
    """The H3 node must enter the namespaced public mapping."""

    assert "Sigmax.MiniMaxH3SigmaScheduler" in NODE_CLASS_MAPPINGS


def test_minimax_h3_node_schema_requires_explicit_variants() -> None:
    assert MINIMAX_H3_SIGMA_NODE_ID == "Sigmax.MiniMaxH3SigmaScheduler"
    assert NODE_CLASS_MAPPINGS[MINIMAX_H3_SIGMA_NODE_ID] is MiniMaxH3SigmaScheduler
    inputs = MiniMaxH3SigmaScheduler.INPUT_TYPES()["required"]
    assert inputs["variant"][0] == ("H3 Base FL2VA", "H3 Base Ref2VA")
    assert MiniMaxH3SigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")
    assert MiniMaxH3SigmaScheduler.RETURN_NAMES == ("sigmas", "schedule_info")


@pytest.mark.parametrize("variant", tuple(MiniMaxH3Variant))
def test_minimax_h3_node_metadata_exposes_video_only_diffusers_lane(
    variant: MiniMaxH3Variant,
) -> None:
    public = "H3 Base FL2VA" if variant is MiniMaxH3Variant.BASE_FL2VA else "H3 Base Ref2VA"
    result = build_minimax_h3_sigma_schedule(
        variant=public,
        grid_points=20,
        start_step=0,
        end_step=-1,
    )
    info = json.loads(result.schedule_info_json)
    assert info["schema"] == MINIMAX_H3_SIGMA_NODE_SCHEMA_ID
    assert info["lane"] == "diffusers_endpoint_inclusive"
    assert info["shift"]["video"] == 12.0
    assert info["shift"]["audio"] == 3.0
    assert info["audio"]["ownership"] == "model_native"
    assert info["velocity"] == {"direction": "data_ward", "sign_adapter": "explicit_only"}
    assert info["timestep"] == {
        "clean": 1.0,
        "convention": "t_equals_one_minus_sigma",
        "terminal_sigma": 0.0,
    }
    assert info["counts"] == {
        "effective_grid_points": 20,
        "effective_model_evaluations": 19,
        "effective_transitions": 19,
        "requested_grid_points": 20,
        "requested_transitions": 19,
    }


def test_minimax_h3_node_rejects_implicit_variant_and_double_shift() -> None:
    with pytest.raises(ScheduleContractError, match="variant"):
        build_minimax_h3_sigma_schedule(
            variant="auto",
            grid_points=20,
            start_step=0,
            end_step=-1,
        )
    with pytest.raises(ScheduleContractError, match=r"second|already shifted"):
        build_minimax_h3_sigma_schedule(
            variant="H3 Base FL2VA",
            grid_points=20,
            start_step=0,
            end_step=-1,
            already_shifted=True,
        )


def test_minimax_h3_node_can_execute_with_a_torch_like_float_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor:
        def __init__(self, values: tuple[float, ...]) -> None:
            self._values = tuple(values)

        def tolist(self) -> list[float]:
            return list(self._values)

    float32 = object()

    def make_tensor(values: tuple[float, ...], *, dtype: object) -> FakeTensor:
        assert dtype is float32
        return FakeTensor(tuple(values))

    fake_torch = SimpleNamespace(float32=float32, tensor=make_tensor)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    output_tensor, metadata = MiniMaxH3SigmaScheduler().build("H3 Base FL2VA", 20, 0, -1, False)
    assert isinstance(output_tensor, FakeTensor)
    info = json.loads(metadata)
    assert info["fingerprints"]["output"].startswith("sha256:")
