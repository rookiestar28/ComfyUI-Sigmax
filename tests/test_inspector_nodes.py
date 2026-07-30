"""Read-only profile and schedule inspector node contracts."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes import inspectors as inspectors_module
from comfyui_sigmax.nodes.advanced_flowmatch_scheduler import (
    build_advanced_flowmatch_schedule,
)
from comfyui_sigmax.nodes.inspectors import (
    PROFILE_INSPECTOR_NODE_ID,
    PROFILE_INSPECTOR_SCHEMA_ID,
    SCHEDULE_INSPECTOR_NODE_ID,
    SCHEDULE_INSPECTOR_SCHEMA_ID,
    ProfileInspector,
    ProfileInspectorResult,
    ScheduleInspector,
    ScheduleInspectorResult,
    build_profile_inspection,
    build_schedule_inspection,
)
from comfyui_sigmax.nodes.krea2_sigma_scheduler import build_krea2_sigma_schedule
from comfyui_sigmax.nodes.model_aware_sigma_scheduler import (
    build_model_aware_sigma_schedule,
)


def _model(*, sampling: object | None = None) -> object:
    krea2 = type("Krea2", (), {})
    inner = krea2()
    inner.model_sampling = type("ModelSamplingFlux", (), {})() if sampling is None else sampling
    return SimpleNamespace(model=inner)


def _profile_arguments(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model": _model(),
        "variant": "Turbo",
        "steps": 8,
        "width": 1025,
        "height": 769,
        "strict_official": True,
    }
    values.update(changes)
    return values


def _decoded(value: str) -> dict[str, Any]:
    result = json.loads(value)
    assert isinstance(result, dict)
    return result


def test_inspectors_declare_stable_read_only_schemas() -> None:
    profile_inputs = ProfileInspector.INPUT_TYPES()
    schedule_inputs = ScheduleInspector.INPUT_TYPES()

    assert PROFILE_INSPECTOR_NODE_ID == "Sigmax.ProfileInspector"
    assert PROFILE_INSPECTOR_SCHEMA_ID == "sigmax.profile-inspector/1"
    assert SCHEDULE_INSPECTOR_NODE_ID == "Sigmax.ScheduleInspector"
    assert SCHEDULE_INSPECTOR_SCHEMA_ID == "sigmax.schedule-inspector/1"
    assert ProfileInspector.RETURN_TYPES == ("STRING",)
    assert ScheduleInspector.RETURN_TYPES == ("STRING",)
    assert ProfileInspector.CATEGORY == ScheduleInspector.CATEGORY == "Sigmax/inspection"
    assert ProfileInspector.OUTPUT_NODE is ScheduleInspector.OUTPUT_NODE is False
    assert set(profile_inputs["required"]) == {
        "model",
        "variant",
        "steps",
        "width",
        "height",
        "strict_official",
    }
    assert profile_inputs["required"]["variant"][0] == ("Turbo", "RAW")
    assert schedule_inputs["required"] == {
        "sigmas": ("SIGMAS",),
        "schedule_info": (
            "STRING",
            {"default": "", "multiline": True},
        ),
    }


def test_builtin_mapping_registers_both_inspectors() -> None:
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS[PROFILE_INSPECTOR_NODE_ID] is ProfileInspector
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS[SCHEDULE_INSPECTOR_NODE_ID] is ScheduleInspector
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[PROFILE_INSPECTOR_NODE_ID] == "Profile Inspector"
    )
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[SCHEDULE_INSPECTOR_NODE_ID]
        == "Schedule Inspector"
    )


@pytest.mark.parametrize(
    ("variant", "steps", "expected_shift"),
    (("Turbo", 8, "fixed_exponential_mu"), ("RAW", 52, "resolution_exponential_mu")),
)
def test_profile_inspector_shows_complete_required_evidence(
    variant: str,
    steps: int,
    expected_shift: str,
) -> None:
    result = build_profile_inspection(**_profile_arguments(variant=variant, steps=steps))
    report = _decoded(result.report_json)

    assert report["schema"] == PROFILE_INSPECTOR_SCHEMA_ID
    assert report["model_identity"] == {
        "confidence": "authoritative",
        "confirmed_variant": variant.casefold(),
        "decisive_source": "explicit_selection",
        "family": "krea2",
        "reason_codes": ["explicit.variant", "model_class.krea2_family"],
        "status": "confirmed",
        "suggested_variant": None,
    }
    assert report["native_sampling"] == {
        "class": "ModelSamplingFlux",
        "reference_sampler_id": "comfy.euler",
    }
    assert report["dimensions"] == {
        "effective": {"height": 784, "width": 1040},
        "requested": {"height": 769, "width": 1025},
    }
    assert report["shift"]["kind"] == expected_shift
    assert report["compatibility"]["level"] == "allow"
    assert report["profile"]["key"] == f"krea2.{variant.casefold()}.official@1"
    assert report["profile"]["fingerprint"].startswith("sha256:")
    assert report["provenance"]["host"]["kind"] == "static_contract"
    assert report["warnings"] == []
    assert report["fingerprints"]["complete"].startswith("sha256:")
    assert report["fingerprints"]["output"].startswith("sha256:")


def test_profile_inspector_uses_static_sampling_class_read() -> None:
    calls: list[str] = []

    class ExecutableSampling:
        @property
        def model_sampling(self) -> object:
            calls.append("property")
            return object()

    with pytest.raises(ScheduleContractError, match="model_sampling"):
        build_profile_inspection(
            **_profile_arguments(model=SimpleNamespace(model=ExecutableSampling()))
        )

    assert calls == []


@pytest.mark.parametrize(
    "sampling",
    (
        None,
        object(),
        type("bad-name", (), {})(),
    ),
)
def test_profile_inspector_rejects_missing_or_uncontrolled_sampling_class(
    sampling: object | None,
) -> None:
    model = (
        SimpleNamespace(model=type("Krea2", (), {})())
        if sampling is None
        else _model(sampling=sampling)
    )
    with pytest.raises(ScheduleContractError):
        build_profile_inspection(**_profile_arguments(model=model))


def _source_results() -> tuple[tuple[tuple[float, ...], str, str], ...]:
    krea = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    model_aware = build_model_aware_sigma_schedule(
        model=_model(),
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    advanced = build_advanced_flowmatch_schedule(
        domain="UNIT_FLOW",
        steps=4,
        sigma_start=1.0,
        sigma_end=0.1,
        shift_mode="direct_ratio",
        shift_value=2.0,
        terminal_policy="append_zero",
        start_step=0,
        end_step=-1,
    )
    return (
        (krea.sigmas, krea.schedule_info_json, "sigmax.krea2-sigma-node/1"),
        (
            model_aware.sigmas,
            model_aware.schedule_info_json,
            "sigmax.model-aware-sigma-node/1",
        ),
        (
            advanced.sigmas,
            advanced.schedule_info_json,
            "sigmax.advanced-flowmatch-node/1",
        ),
    )


@pytest.mark.parametrize("source_index", (0, 1, 2))
def test_schedule_inspector_normalizes_and_verifies_all_source_schemas(
    source_index: int,
) -> None:
    sigmas, information, source_schema = _source_results()[source_index]
    result = build_schedule_inspection(sigmas=sigmas, schedule_info=information)
    report = _decoded(result.report_json)

    assert report["schema"] == SCHEDULE_INSPECTOR_SCHEMA_ID
    assert report["source_schema"] == source_schema
    assert report["domain"] == "UNIT_FLOW"
    assert report["fingerprints"]["verified"] is True
    assert report["fingerprints"]["advertised_output"] == report["fingerprints"]["computed_output"]
    assert report["shift"]["kind"] in {
        "fixed_exponential_mu",
        "direct_ratio",
    }
    assert isinstance(report["slicing"], dict)
    assert isinstance(report["warnings"], list)
    if source_schema == "sigmax.model-aware-sigma-node/1":
        assert report["compatibility"]["level"] == "allow"
        assert report["model_identity"]["family"] == "krea2"
    else:
        assert report["compatibility"] is None


def test_schedule_inspector_rejects_fingerprint_mismatch() -> None:
    sigmas, information, _ = _source_results()[0]
    changed = (sigmas[0], sigmas[1] * 0.99, *sigmas[2:])

    with pytest.raises(ScheduleContractError, match="fingerprint"):
        build_schedule_inspection(sigmas=changed, schedule_info=information)


@pytest.mark.parametrize(
    "schedule_info",
    (
        "",
        "[]",
        '{"schema":"unknown"}',
        '{"schema":"sigmax.krea2-sigma-node/1","schema":"other"}',
        '{"schema":NaN}',
        '{"schema":1.0}',
        "{" + '"x":' * 40 + "0" + "}" * 40,
        "x" * 1_048_577,
    ),
    ids=(
        "empty",
        "array-root",
        "unknown-schema",
        "duplicate-key",
        "non-finite",
        "float-schema",
        "too-deep",
        "too-large",
    ),
)
def test_schedule_inspector_rejects_untrusted_json(schedule_info: str) -> None:
    with pytest.raises(ScheduleContractError):
        build_schedule_inspection(sigmas=(1.0, 0.0), schedule_info=schedule_info)


@pytest.mark.parametrize(
    "sigmas",
    (
        cast(Any, [1.0, 0.0]),
        (),
        (1.0,),
        (1.0, math.nan, 0.0),
        (1.0, 1.0, 0.0),
        (1.1, 0.0),
        tuple(1.0 - index / 10_001 for index in range(10_002)),
    ),
)
def test_schedule_inspector_rejects_invalid_pure_sigmas(
    sigmas: tuple[float, ...],
) -> None:
    _, information, _ = _source_results()[0]
    with pytest.raises(ScheduleContractError):
        build_schedule_inspection(sigmas=sigmas, schedule_info=information)


def test_inspector_results_are_immutable_and_deterministic() -> None:
    profile_first = build_profile_inspection(**_profile_arguments())
    profile_second = build_profile_inspection(**_profile_arguments())
    sigmas, information, _ = _source_results()[2]
    schedule_first = build_schedule_inspection(sigmas=sigmas, schedule_info=information)
    schedule_second = build_schedule_inspection(sigmas=sigmas, schedule_info=information)

    assert profile_first == profile_second
    assert schedule_first == schedule_second
    with pytest.raises(FrozenInstanceError):
        profile_first.report_json = ""  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        schedule_first.report_json = ""  # type: ignore[misc]


@pytest.mark.parametrize(
    ("result_type", "schema_id"),
    (
        (ProfileInspectorResult, PROFILE_INSPECTOR_SCHEMA_ID),
        (ScheduleInspectorResult, SCHEDULE_INSPECTOR_SCHEMA_ID),
    ),
)
def test_result_contracts_reject_invalid_values(
    result_type: type[ProfileInspectorResult] | type[ScheduleInspectorResult],
    schema_id: str,
) -> None:
    assert result_type(schema_id=schema_id, report_json="{}").report_json == "{}"
    with pytest.raises(ScheduleContractError):
        result_type(schema_id="other", report_json="{}")
    with pytest.raises(ScheduleContractError):
        result_type(schema_id=schema_id, report_json="")


def test_runtime_schedule_node_converts_bounded_host_sigmas() -> None:
    sigmas, information, _ = _source_results()[0]

    class HostSigmas:
        def __len__(self) -> int:
            return len(sigmas)

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(sigmas)

    output = ScheduleInspector().inspect(HostSigmas(), information)

    assert len(output) == 1
    assert _decoded(output[0])["fingerprints"]["verified"] is True


def test_runtime_profile_node_returns_report() -> None:
    output = ProfileInspector().inspect(**_profile_arguments())

    assert len(output) == 1
    assert _decoded(output[0])["schema"] == PROFILE_INSPECTOR_SCHEMA_ID


@pytest.mark.parametrize(
    "value",
    (
        math.inf,
        "x" * 4097,
        [None] * 1025,
        {str(index): None for index in range(1025)},
        cast(Any, {1: None}),
        object(),
    ),
    ids=(
        "non-finite",
        "long-string",
        "long-list",
        "large-object",
        "invalid-key",
        "unsupported-value",
    ),
)
def test_recursive_json_bounds_reject_invalid_values(value: object) -> None:
    with pytest.raises(ScheduleContractError):
        inspectors_module._bound_json(value)
    with pytest.raises(ScheduleContractError, match="depth"):
        inspectors_module._bound_json(None, depth=33)


def test_private_inspector_projection_guards_fail_closed() -> None:
    with pytest.raises(ScheduleContractError, match="canonical JSON"):
        inspectors_module._canonical_json({"value": object()})
    with pytest.raises(ScheduleContractError, match="fields"):
        inspectors_module._require_exact_fields({}, {"schema"}, label="test")
    with pytest.raises(ScheduleContractError, match="object"):
        inspectors_module._object([], label="test")
    with pytest.raises(ScheduleContractError, match="array"):
        inspectors_module._list({}, label="test")
    with pytest.raises(ScheduleContractError, match="fields"):
        inspectors_module._fingerprints({})
    with pytest.raises(ScheduleContractError, match="invalid"):
        inspectors_module._fingerprints({"complete": "bad", "output": "bad"})


def _mutated_information(
    source_index: int,
    mutate: Any,
) -> tuple[tuple[float, ...], str]:
    sigmas, information, _ = _source_results()[source_index]
    decoded = _decoded(information)
    mutate(decoded)
    return sigmas, json.dumps(decoded, separators=(",", ":"), sort_keys=True)


@pytest.mark.parametrize(
    ("source_index", "mutate"),
    (
        (0, lambda value: value.update(extra=True)),
        (1, lambda value: value["schedule"].update(extra=True)),
        (1, lambda value: value.update(schedule=[])),
        (2, lambda value: value["domain"].update(sigma="CONTINUOUS_EDM")),
        (2, lambda value: value.update(domain=[])),
        (2, lambda value: value.update(fingerprints=[])),
        (0, lambda value: value.update(warnings={})),
        (0, lambda value: value.update(warnings=[""])),
        (1, lambda value: value.update(capability_decision=[])),
    ),
)
def test_structural_schedule_information_mutations_reject(
    source_index: int,
    mutate: Any,
) -> None:
    sigmas, information = _mutated_information(source_index, mutate)
    with pytest.raises(ScheduleContractError):
        build_schedule_inspection(sigmas=sigmas, schedule_info=information)


@pytest.mark.parametrize(
    "schedule_info",
    (
        cast(Any, 1),
        "\ud800",
    ),
)
def test_schedule_information_type_and_unicode_reject(schedule_info: object) -> None:
    with pytest.raises(ScheduleContractError):
        build_schedule_inspection(sigmas=(1.0, 0.0), schedule_info=schedule_info)


@pytest.mark.parametrize(
    "sigmas",
    (
        object(),
        (1.0,),
        (1.0, object()),
    ),
)
def test_runtime_schedule_node_rejects_invalid_host_sequences(sigmas: object) -> None:
    _, information, _ = _source_results()[0]
    with pytest.raises(ScheduleContractError):
        ScheduleInspector().inspect(sigmas, information)
