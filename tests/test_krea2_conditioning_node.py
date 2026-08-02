from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import comfyui_sigmax.nodes.krea2_conditioning_rebalance as node_module
import pytest
from comfyui_sigmax.adapters.krea2_conditioning import ConditioningTransformStats
from comfyui_sigmax.conditioning import (
    ConditioningModifierRequest,
    Krea2ConditioningVariant,
)
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.krea2_conditioning_rebalance import (
    KREA2_CONDITIONING_NODE_ID,
    KREA2_CONDITIONING_NODE_SCHEMA_ID,
    Krea2ConditioningRebalance,
)


def test_node_schema_is_explicit_experimental_and_does_not_touch_scheduler_inputs() -> None:
    schema = Krea2ConditioningRebalance.INPUT_TYPES()

    assert KREA2_CONDITIONING_NODE_ID == "Sigmax.Krea2ConditioningRebalance"
    assert KREA2_CONDITIONING_NODE_SCHEMA_ID == "sigmax.krea2-conditioning-node/1"
    assert schema["required"]["conditioning"] == ("CONDITIONING",)
    required = schema["required"]
    assert cast(tuple[object, ...], required["conditioning"]) == ("CONDITIONING",)
    assert cast(list[object], required["variant"])[0] == ["RAW", "Turbo"]
    assert cast(list[object], required["profile"])[0] == [
        "Disabled",
        "Subtle Experimental",
        "Classic Experimental",
    ]
    strength = cast(tuple[object, dict[str, object]], required["strength"])
    assert strength[1]["min"] == 0.0
    assert strength[1]["max"] == 1.0
    assert Krea2ConditioningRebalance.RETURN_TYPES == ("CONDITIONING", "STRING")
    assert Krea2ConditioningRebalance.RETURN_NAMES == ("conditioning", "modifier_info")
    assert Krea2ConditioningRebalance.EXPERIMENTAL is True


def test_node_builds_canonical_experimental_modifier_report_without_tensor_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats = ConditioningTransformStats(
        input_shape=(1, 2, 30720),
        input_shapes=((1, 2, 30720),),
        dtype="torch.float32",
        device="cpu",
        variant=Krea2ConditioningVariant.RAW,
        conditioning_entries=1,
        transformed_entries=1,
    )
    sentinel = object()

    def fake_transform(
        conditioning: object, request: object
    ) -> tuple[list[list[object]], ConditioningTransformStats]:
        selected_request = cast(ConditioningModifierRequest, request)
        return [[sentinel, {"keep": True}]], replace(stats, variant=selected_request.variant)

    monkeypatch.setattr(node_module, "transform_krea2_conditioning", fake_transform)
    output, report_json = Krea2ConditioningRebalance().rebalance(
        conditioning=[[object(), {}]],
        variant="RAW",
        profile="Subtle Experimental",
        strength=0.25,
    )

    report = json.loads(report_json)
    assert output[0][0] is sentinel
    assert report["schema"] == "sigmax.conditioning-modifier/1"
    assert report["evidence"] == "experimental"
    assert report["variant"] == {"evidence": "user_selected", "value": "RAW"}
    assert report["schedule_affected"] is False


@pytest.mark.parametrize("variant", ["Auto", "RAW/Turbo", ""])
def test_node_rejects_non_explicit_variant_before_host_execution(
    variant: str,
) -> None:
    with pytest.raises(ScheduleContractError, match="variant"):
        Krea2ConditioningRebalance().rebalance(
            conditioning=[],
            variant=variant,
            profile="Disabled",
            strength=0.0,
        )
