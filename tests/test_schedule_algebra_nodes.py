"""M4-09 validated schedule algebra node contracts."""

from __future__ import annotations

import json
import struct
import sys
from types import SimpleNamespace
from typing import cast

import pytest
from comfyui_sigmax.core import EvidenceLevel, ScheduleContractError, SigmaDomain
from comfyui_sigmax.nodes import (
    SCHEDULE_CONCATENATE_NODE_ID,
    SCHEDULE_CONCATENATE_SCHEMA_ID,
    SCHEDULE_RESAMPLE_NODE_ID,
    SCHEDULE_RESAMPLE_SCHEMA_ID,
    SCHEDULE_SLICE_NODE_ID,
    SCHEDULE_SLICE_SCHEMA_ID,
    ScheduleConcatenate,
    ScheduleResample,
    ScheduleSlice,
    build_schedule_concatenation,
    build_schedule_inspection,
    build_schedule_resample,
    build_schedule_slice,
)
from comfyui_sigmax.nodes.advanced_flowmatch_scheduler import build_advanced_flowmatch_schedule
from comfyui_sigmax.nodes.krea2_sigma_scheduler import sigma_output_fingerprint


def _source(
    *,
    steps: int = 4,
    start: float = 1.0,
    end: float = 0.2,
    terminal_policy: str = "append_zero",
) -> tuple[tuple[float, ...], str]:
    result = build_advanced_flowmatch_schedule(
        domain="UNIT_FLOW",
        steps=steps,
        sigma_start=start,
        sigma_end=end,
        shift_mode="direct_ratio",
        shift_value=1.0,
        terminal_policy=terminal_policy,
        start_step=0,
        end_step=-1,
    )
    return result.sigmas, result.schedule_info_json


def _info(value: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(value))


def test_algebra_nodes_have_separate_stable_non_output_contracts() -> None:
    expected = (
        (ScheduleSlice, SCHEDULE_SLICE_NODE_ID, SCHEDULE_SLICE_SCHEMA_ID, "slice"),
        (
            ScheduleConcatenate,
            SCHEDULE_CONCATENATE_NODE_ID,
            SCHEDULE_CONCATENATE_SCHEMA_ID,
            "concatenate",
        ),
        (ScheduleResample, SCHEDULE_RESAMPLE_NODE_ID, SCHEDULE_RESAMPLE_SCHEMA_ID, "resample"),
    )
    for node, node_id, schema_id, function in expected:
        assert node_id.startswith("Sigmax.Schedule")
        assert schema_id.startswith("sigmax.schedule-") and schema_id.endswith("-node/1")
        assert function == node.FUNCTION
        assert node.RETURN_TYPES == ("SIGMAS", "STRING")
        assert node.RETURN_NAMES == ("sigmas", "schedule_info")
        assert node.OUTPUT_NODE is False
        assert node.CATEGORY == "Sigmax/scheduling"


def test_slice_is_terminal_inclusive_verified_and_modified() -> None:
    sigmas, source_info = _source()
    result = build_schedule_slice(
        sigmas=sigmas,
        schedule_info=source_info,
        start_step=1,
        end_step=3,
    )
    info = _info(result.schedule_info_json)

    assert result.sigmas == sigmas[1:4]
    assert result.domain is SigmaDomain.UNIT_FLOW
    assert info["schema"] == SCHEDULE_SLICE_SCHEMA_ID
    assert info["operation"] == "slice"
    assert info["evidence"] == EvidenceLevel.MODIFIED.value
    assert info["parameters"] == {"end_step": 3, "start_step": 1}
    assert info["terminal"] == {"is_zero": False, "value": result.sigmas[-1]}
    inspection = _info(
        build_schedule_inspection(
            sigmas=result.sigmas,
            schedule_info=result.schedule_info_json,
        ).report_json
    )
    assert cast(dict[str, object], inspection["fingerprints"])["verified"] is True


@pytest.mark.parametrize(
    ("start", "end"),
    [(0, 4), (-1, 2), (2, 2), (3, 2), (0, 5), (True, 2), (0, False)],
)
def test_slice_rejects_noop_and_invalid_bounds(start: object, end: object) -> None:
    sigmas, source_info = _source()
    with pytest.raises(ScheduleContractError):
        build_schedule_slice(
            sigmas=sigmas,
            schedule_info=source_info,
            start_step=start,
            end_step=end,
        )


def test_concatenation_requires_one_exact_shared_boundary() -> None:
    sigmas, source_info = _source()
    left = build_schedule_slice(
        sigmas=sigmas,
        schedule_info=source_info,
        start_step=0,
        end_step=2,
    )
    right = build_schedule_slice(
        sigmas=sigmas,
        schedule_info=source_info,
        start_step=2,
        end_step=4,
    )
    combined = build_schedule_concatenation(
        sigmas_left=left.sigmas,
        schedule_info_left=left.schedule_info_json,
        sigmas_right=right.sigmas,
        schedule_info_right=right.schedule_info_json,
    )
    info = _info(combined.schedule_info_json)

    assert combined.sigmas == sigmas
    assert info["schema"] == SCHEDULE_CONCATENATE_SCHEMA_ID
    assert info["evidence"] == "modified"
    assert info["parameters"] == {"boundary": left.sigmas[-1], "shared_boundary_count": 1}
    assert len(cast(list[object], info["sources"])) == 2
    build_schedule_inspection(sigmas=combined.sigmas, schedule_info=combined.schedule_info_json)


def test_concatenation_rejects_gap_even_when_both_sources_are_individually_valid() -> None:
    left_sigmas, left_info = _source(
        steps=2,
        start=1.0,
        end=0.6,
        terminal_policy="preserve",
    )
    right_sigmas, right_info = _source(steps=2, start=0.5, end=0.2)

    with pytest.raises(ScheduleContractError, match="boundary"):
        build_schedule_concatenation(
            sigmas_left=left_sigmas,
            schedule_info_left=left_info,
            sigmas_right=right_sigmas,
            schedule_info_right=right_info,
        )


@pytest.mark.parametrize("output_steps", [2, 8])
def test_resample_is_explicit_linear_preserves_endpoints_and_terminal(output_steps: int) -> None:
    sigmas, source_info = _source(steps=4)
    result = build_schedule_resample(
        sigmas=sigmas,
        schedule_info=source_info,
        output_steps=output_steps,
    )
    info = _info(result.schedule_info_json)

    assert len(result.sigmas) == output_steps + 1
    assert result.sigmas[0] == sigmas[0]
    assert result.sigmas[-1] == sigmas[-1] == 0.0
    assert info["schema"] == SCHEDULE_RESAMPLE_SCHEMA_ID
    assert info["operation"] == "resample"
    assert info["evidence"] == "modified"
    assert info["parameters"] == {
        "input_steps": 4,
        "method": "index_linear_v1",
        "output_steps": output_steps,
    }
    build_schedule_inspection(sigmas=result.sigmas, schedule_info=result.schedule_info_json)


@pytest.mark.parametrize("output_steps", [4, 0, 10001, True, 2.5])
def test_resample_rejects_noop_or_invalid_counts(output_steps: object) -> None:
    sigmas, source_info = _source()
    with pytest.raises(ScheduleContractError):
        build_schedule_resample(
            sigmas=sigmas,
            schedule_info=source_info,
            output_steps=output_steps,
        )


def test_every_operation_rejects_disconnected_or_tampered_source_identity() -> None:
    sigmas, source_info = _source()
    tampered = (sigmas[0], sigmas[1] - 0.01, *sigmas[2:])

    with pytest.raises(ScheduleContractError, match="fingerprint"):
        build_schedule_slice(
            sigmas=tampered,
            schedule_info=source_info,
            start_step=0,
            end_step=2,
        )
    with pytest.raises(ScheduleContractError, match="fingerprint"):
        build_schedule_resample(
            sigmas=tampered,
            schedule_info=source_info,
            output_steps=8,
        )


def test_algebra_rejects_domain_relabeling_instead_of_converting() -> None:
    sigmas, source_info = _source()
    changed = _info(source_info)
    domain = cast(dict[str, object], changed["domain"])
    domain["sigma"] = SigmaDomain.CONTINUOUS_EDM.value
    fingerprints = cast(dict[str, object], changed["fingerprints"])
    relabeled = sigma_output_fingerprint(sigmas, domain=SigmaDomain.CONTINUOUS_EDM)
    fingerprints.update(complete=relabeled, output=relabeled)

    with pytest.raises(ScheduleContractError, match="does not support"):
        build_schedule_slice(
            sigmas=sigmas,
            schedule_info=json.dumps(changed, separators=(",", ":"), sort_keys=True),
            start_step=0,
            end_step=2,
        )


def test_runtime_node_rebinds_information_to_float32_host_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sigmas, source_info = _source()

    def float_tensor(values: tuple[float, ...]) -> object:
        quantized = tuple(struct.unpack(">f", struct.pack(">f", value))[0] for value in values)
        return SimpleNamespace(tolist=lambda: list(quantized), values=quantized)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(FloatTensor=float_tensor))
    tensor, schedule_info = ScheduleResample().resample(
        sigmas,
        source_info,
        8,
    )
    host_values = cast(SimpleNamespace, tensor).values
    info = _info(schedule_info)

    assert cast(dict[str, object], info["fingerprints"])["output"] == sigma_output_fingerprint(
        host_values,
        domain=SigmaDomain.UNIT_FLOW,
    )
    build_schedule_inspection(sigmas=host_values, schedule_info=schedule_info)
