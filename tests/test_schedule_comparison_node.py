"""Step-aligned schedule comparison node contracts."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain
from comfyui_sigmax.nodes.advanced_flowmatch_scheduler import (
    build_advanced_flowmatch_schedule,
)
from comfyui_sigmax.nodes.inspectors import (
    SCHEDULE_COMPARISON_NODE_ID,
    SCHEDULE_COMPARISON_SCHEMA_ID,
    ScheduleComparison,
    ScheduleComparisonResult,
    build_schedule_comparison,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import (
    build_krea2_sigma_schedule,
    sigma_output_fingerprint,
)
from comfyui_sigmax.nodes.model_aware_sigma_scheduler import (
    build_model_aware_sigma_schedule,
)


def _schedule(*, steps: int = 4, shift: float = 2.0) -> tuple[tuple[float, ...], str]:
    result = build_advanced_flowmatch_schedule(
        domain="UNIT_FLOW",
        steps=steps,
        sigma_start=1.0,
        sigma_end=0.1,
        shift_mode="direct_ratio",
        shift_value=shift,
        terminal_policy="append_zero",
        start_step=0,
        end_step=-1,
    )
    return result.sigmas, result.schedule_info_json


def _decoded(value: str) -> dict[str, Any]:
    result = json.loads(value)
    assert isinstance(result, dict)
    return result


def _comparison(
    left: tuple[tuple[float, ...], str],
    right: tuple[tuple[float, ...], str],
) -> dict[str, Any]:
    result = build_schedule_comparison(
        sigmas_a=left[0],
        schedule_info_a=left[1],
        sigmas_b=right[0],
        schedule_info_b=right[1],
    )
    return _decoded(result.report_json)


def _supported_sources() -> tuple[tuple[tuple[float, ...], str, str], ...]:
    krea = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    inner = type("Krea2", (), {})()
    inner.model_sampling = type("ModelSamplingFlux", (), {})()
    model_aware = build_model_aware_sigma_schedule(
        model=SimpleNamespace(model=inner),
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    advanced_sigmas, advanced_information = _schedule(steps=8)
    return (
        (krea.sigmas, krea.schedule_info_json, "sigmax.krea2-sigma-node/1"),
        (
            model_aware.sigmas,
            model_aware.schedule_info_json,
            "sigmax.model-aware-sigma-node/1",
        ),
        (
            advanced_sigmas,
            advanced_information,
            "sigmax.advanced-flowmatch-node/1",
        ),
    )


def test_comparison_declares_stable_read_only_schema() -> None:
    assert SCHEDULE_COMPARISON_NODE_ID == "Sigmax.ScheduleComparison"
    assert SCHEDULE_COMPARISON_SCHEMA_ID == "sigmax.schedule-comparison/1"
    assert ScheduleComparison.CATEGORY == "Sigmax/inspection"
    assert ScheduleComparison.RETURN_TYPES == ("STRING",)
    assert ScheduleComparison.RETURN_NAMES == ("comparison_report",)
    assert ScheduleComparison.OUTPUT_NODE is False
    assert ScheduleComparison.INPUT_TYPES()["required"] == {
        "sigmas_a": ("SIGMAS",),
        "schedule_info_a": ("STRING", {"default": "", "multiline": True}),
        "sigmas_b": ("SIGMAS",),
        "schedule_info_b": ("STRING", {"default": "", "multiline": True}),
    }


def test_comparison_emits_step_aligned_absolute_and_symmetric_relative_differences() -> None:
    left = _schedule(shift=2.0)
    right = _schedule(shift=3.0)
    report = _comparison(left, right)

    assert report["schema"] == SCHEDULE_COMPARISON_SCHEMA_ID
    assert report["comparable"] is True
    assert report["reason"] is None
    assert report["alignment"] == {
        "kind": "sigma_index",
        "length": len(left[0]),
        "terminal_inclusive": True,
    }
    assert len(report["steps"]) == len(left[0])
    for index, row in enumerate(report["steps"]):
        expected_absolute = abs(left[0][index] - right[0][index])
        denominator = max(abs(left[0][index]), abs(right[0][index]))
        expected_relative = 0.0 if denominator == 0.0 else expected_absolute / denominator
        assert row == {
            "absolute_difference": expected_absolute,
            "index": index,
            "relative_difference": expected_relative,
            "sigma_a": left[0][index],
            "sigma_b": right[0][index],
        }
    assert report["steps"][-1]["relative_difference"] == 0.0
    assert report["summary"]["exact_match_count"] >= 1
    assert report["summary"]["maximum_absolute_difference"] == max(
        row["absolute_difference"] for row in report["steps"]
    )
    assert report["summary"]["maximum_relative_difference"] == max(
        row["relative_difference"] for row in report["steps"]
    )
    assert report["summary"]["mean_absolute_difference"] == pytest.approx(
        sum(row["absolute_difference"] for row in report["steps"]) / len(report["steps"])
    )
    assert report["summary"]["mean_relative_difference"] == pytest.approx(
        sum(row["relative_difference"] for row in report["steps"]) / len(report["steps"])
    )


def test_comparison_emits_verified_transform_metadata_for_both_sources() -> None:
    report = _comparison(_schedule(shift=2.0), _schedule(shift=3.0))

    for label, expected_shift in (("a", 2.0), ("b", 3.0)):
        source = report["sources"][label]
        assert source["domain"] == "UNIT_FLOW"
        assert source["fingerprints"]["verified"] is True
        assert source["length"] == 5
        assert source["source_schema"] == "sigmax.advanced-flowmatch-node/1"
        assert source["transforms"]["shift"]["value"] == expected_shift
        assert source["transforms"]["base_grid"]["identifier"] == "sigmax.linear_endpoint"
        assert source["transforms"]["terminal"]["policy"] == "append_zero"
        assert source["transforms"]["transform_order"] == [
            "PRIMARY_TIME_SHIFT",
            "TERMINAL",
            "SLICE",
        ]


@pytest.mark.parametrize(("left_index", "right_index"), ((0, 1), (1, 2), (2, 0)))
def test_comparison_accepts_every_supported_source_schema_in_both_positions(
    left_index: int,
    right_index: int,
) -> None:
    sources = _supported_sources()
    left = sources[left_index]
    right = sources[right_index]
    report = _comparison(left[:2], right[:2])

    assert report["comparable"] is True
    assert report["sources"]["a"]["source_schema"] == left[2]
    assert report["sources"]["b"]["source_schema"] == right[2]
    assert report["sources"]["a"]["fingerprints"]["verified"] is True
    assert report["sources"]["b"]["fingerprints"]["verified"] is True


def test_equal_schedules_are_deterministic_and_exact() -> None:
    schedule = _schedule()
    first = build_schedule_comparison(
        sigmas_a=schedule[0],
        schedule_info_a=schedule[1],
        sigmas_b=schedule[0],
        schedule_info_b=schedule[1],
    )
    second = build_schedule_comparison(
        sigmas_a=schedule[0],
        schedule_info_a=schedule[1],
        sigmas_b=schedule[0],
        schedule_info_b=schedule[1],
    )
    report = _decoded(first.report_json)

    assert first == second
    assert report["summary"] == {
        "exact_match_count": len(schedule[0]),
        "maximum_absolute_difference": 0.0,
        "maximum_absolute_index": 0,
        "maximum_relative_difference": 0.0,
        "maximum_relative_index": 0,
        "mean_absolute_difference": 0.0,
        "mean_relative_difference": 0.0,
    }
    with pytest.raises(FrozenInstanceError):
        first.report_json = ""  # type: ignore[misc]


def test_length_mismatch_is_explicit_and_never_truncated() -> None:
    left = _schedule(steps=4)
    right = _schedule(steps=5)
    report = _comparison(left, right)

    assert report["comparable"] is False
    assert report["reason"] == "comparison.length_mismatch"
    assert report["alignment"] == {
        "kind": "none",
        "length_a": len(left[0]),
        "length_b": len(right[0]),
        "terminal_inclusive": True,
    }
    assert report["steps"] == []
    assert report["summary"] is None


def test_domain_mismatch_is_explicit_and_never_converted() -> None:
    left = _schedule()
    right_sigmas, right_information = _schedule()
    changed = _decoded(right_information)
    changed["domain"]["sigma"] = SigmaDomain.CONTINUOUS_EDM.value
    changed["fingerprints"]["output"] = sigma_output_fingerprint(
        right_sigmas,
        domain=SigmaDomain.CONTINUOUS_EDM,
    )
    right = (right_sigmas, json.dumps(changed, separators=(",", ":"), sort_keys=True))

    report = _comparison(left, right)

    assert report["comparable"] is False
    assert report["reason"] == "comparison.domain_mismatch"
    assert report["alignment"] == {
        "domain_a": "UNIT_FLOW",
        "domain_b": "CONTINUOUS_EDM",
        "kind": "none",
        "terminal_inclusive": True,
    }
    assert report["steps"] == []
    assert report["summary"] is None


@pytest.mark.parametrize("domain", (1, "UNKNOWN", "MODEL_NATIVE"))
def test_comparison_rejects_malformed_unsupported_or_opaque_domains(domain: object) -> None:
    schedule = _schedule()
    changed = _decoded(schedule[1])
    changed["domain"]["sigma"] = domain

    with pytest.raises(ScheduleContractError, match=r"domain|MODEL_NATIVE"):
        build_schedule_comparison(
            sigmas_a=schedule[0],
            schedule_info_a=schedule[1],
            sigmas_b=schedule[0],
            schedule_info_b=json.dumps(changed, separators=(",", ":"), sort_keys=True),
        )


def test_comparison_rejects_unverified_or_invalid_inputs() -> None:
    left = _schedule()
    right = _schedule()
    changed_sigmas = (right[0][0], right[0][1] * 0.99, *right[0][2:])

    with pytest.raises(ScheduleContractError, match="fingerprint"):
        build_schedule_comparison(
            sigmas_a=left[0],
            schedule_info_a=left[1],
            sigmas_b=changed_sigmas,
            schedule_info_b=right[1],
        )
    with pytest.raises(ScheduleContractError):
        build_schedule_comparison(
            sigmas_a=left[0],
            schedule_info_a=left[1],
            sigmas_b=right[0],
            schedule_info_b='{"schema":"unknown"}',
        )
    with pytest.raises(ScheduleContractError):
        build_schedule_comparison(
            sigmas_a=left[0],
            schedule_info_a=left[1],
            sigmas_b=(1.0, math.nan, 0.0),
            schedule_info_b=right[1],
        )


def test_result_contract_rejects_invalid_values() -> None:
    assert (
        ScheduleComparisonResult(
            schema_id=SCHEDULE_COMPARISON_SCHEMA_ID,
            report_json="{}",
        ).report_json
        == "{}"
    )
    with pytest.raises(ScheduleContractError):
        ScheduleComparisonResult(schema_id="other", report_json="{}")
    with pytest.raises(ScheduleContractError):
        ScheduleComparisonResult(
            schema_id=SCHEDULE_COMPARISON_SCHEMA_ID,
            report_json="",
        )


def test_runtime_node_converts_both_bounded_host_sequences_and_is_registered() -> None:
    left = _schedule(shift=2.0)
    right = _schedule(shift=3.0)

    class HostSigmas:
        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self.values)

    output = ScheduleComparison().compare(
        HostSigmas(left[0]),
        left[1],
        HostSigmas(right[0]),
        right[1],
    )

    assert _decoded(output[0])["comparable"] is True
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS[SCHEDULE_COMPARISON_NODE_ID] is ScheduleComparison
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[SCHEDULE_COMPARISON_NODE_ID]
        == "Schedule Comparison"
    )


@pytest.mark.parametrize("value", (object(), (1.0,), (1.0, object())))
def test_runtime_node_rejects_invalid_host_sequences(value: object) -> None:
    schedule = _schedule()
    with pytest.raises(ScheduleContractError):
        ScheduleComparison().compare(value, schedule[1], schedule[0], schedule[1])
