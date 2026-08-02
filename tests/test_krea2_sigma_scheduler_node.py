"""Product contracts for the first Krea 2 SIGMAS scheduler node."""

from __future__ import annotations

import importlib
import json
import struct
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import ScheduleContractError, SigmaDomain, numerical_fingerprint
from comfyui_sigmax.nodes import krea2_sigma_scheduler as scheduler_module
from comfyui_sigmax.nodes.krea2_sigma_scheduler import (
    KREA2_SIGMA_NODE_ID,
    KREA2_SIGMA_NODE_SCHEMA_ID,
    Krea2SigmaNodeResult,
    Krea2SigmaScheduler,
    Krea2SigmaVariant,
    build_krea2_sigma_schedule,
    sigma_output_fingerprint,
)


def _info(result: Krea2SigmaNodeResult) -> dict[str, Any]:
    decoded = json.loads(result.schedule_info_json)
    assert isinstance(decoded, dict)
    return decoded


def test_node_declares_stable_legacy_current_schema() -> None:
    inputs = Krea2SigmaScheduler.INPUT_TYPES()

    assert KREA2_SIGMA_NODE_ID == "Sigmax.Krea2SigmaScheduler"
    assert KREA2_SIGMA_NODE_SCHEMA_ID == "sigmax.krea2-sigma-node/1"
    assert Krea2SigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")
    assert Krea2SigmaScheduler.RETURN_NAMES == ("sigmas", "schedule_info")
    assert Krea2SigmaScheduler.FUNCTION == "build"
    assert Krea2SigmaScheduler.CATEGORY == "Sigmax/scheduling"
    assert Krea2SigmaScheduler.OUTPUT_NODE is False
    assert inputs["required"]["variant"][0] == (
        "Turbo",
        "RAW",
        "LoRA Experimental (RAW mu)",
        "LoRA Experimental (Turbo mu)",
    )
    assert set(inputs["required"]) == {
        "variant",
        "steps",
        "width",
        "height",
        "strict_official",
        "start_step",
        "end_step",
    }
    assert inputs == Krea2SigmaScheduler.INPUT_TYPES()
    assert inputs is not Krea2SigmaScheduler.INPUT_TYPES()


def test_builtin_mapping_registers_the_explicit_product_node() -> None:
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS[KREA2_SIGMA_NODE_ID] is Krea2SigmaScheduler
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[KREA2_SIGMA_NODE_ID] == "Krea 2 Sigma Scheduler"
    )


def test_strict_turbo_builds_exact_official_eight_step_schedule() -> None:
    result = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    info = _info(result)

    assert result.variant is Krea2SigmaVariant.TURBO
    assert len(result.sigmas) == 9
    assert result.sigmas[-1] == 0.0
    assert info["schema"] == KREA2_SIGMA_NODE_SCHEMA_ID
    assert info["profile"] == {
        "evidence": "official",
        "id": "krea2.turbo.official",
        "recipe": "krea2.turbo.official-8",
        "variant": "turbo",
        "version": "1",
    }
    assert info["shift"]["kind"] == "fixed_exponential_mu"
    assert info["shift"]["mu"] == 1.15
    assert info["warnings"] == []


def test_non_strict_turbo_records_modified_evidence() -> None:
    result = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=12,
        width=1024,
        height=1024,
        strict_official=False,
        start_step=0,
        end_step=-1,
    )
    info = _info(result)

    assert len(result.sigmas) == 13
    assert info["profile"]["evidence"] == "modified"
    assert info["profile"]["recipe"] == "krea2.turbo.modified-12"
    assert info["warnings"] == [
        "requested steps differ from the official Turbo 8-step recipe; evidence is modified"
    ]


@pytest.mark.parametrize(
    ("variant", "mu_source", "expected_mu"),
    (
        ("LoRA Experimental (RAW mu)", "raw", 0.90625),
        ("LoRA Experimental (Turbo mu)", "turbo", 1.15),
    ),
)
def test_experimental_lora_variants_execute_selected_mu(
    variant: str,
    mu_source: str,
    expected_mu: float,
) -> None:
    result = build_krea2_sigma_schedule(
        variant=variant,
        steps=12,
        width=1024,
        height=1024,
        strict_official=False,
        start_step=0,
        end_step=-1,
    )
    info = _info(result)

    assert len(result.sigmas) == 13
    assert result.sigmas[-1] == 0.0
    assert info["profile"] == {
        "evidence": "experimental",
        "id": "krea2.raw-turbo-lora.experimental",
        "recipe": "krea2.raw-turbo-lora.experimental",
        "variant": "raw_turbo_lora",
        "version": "1",
    }
    assert info["shift"]["mu_source"] == mu_source
    assert info["shift"]["mu"] == expected_mu
    assert info["strict_official"] is False
    assert any("experimental" in warning.casefold() for warning in info["warnings"])
    assert any("RAW checkpoint" in warning for warning in info["warnings"])


def test_experimental_mu_selection_is_not_inert() -> None:
    common = {
        "steps": 12,
        "width": 1024,
        "height": 1024,
        "strict_official": False,
        "start_step": 0,
        "end_step": -1,
    }

    raw_mu = build_krea2_sigma_schedule(
        variant="LoRA Experimental (RAW mu)",
        **common,
    )
    turbo_mu = build_krea2_sigma_schedule(
        variant="LoRA Experimental (Turbo mu)",
        **common,
    )

    assert raw_mu.sigmas != turbo_mu.sigmas
    assert _info(raw_mu)["fingerprints"]["complete"] != _info(turbo_mu)["fingerprints"]["complete"]


@pytest.mark.parametrize(
    "variant",
    ("LoRA Experimental (RAW mu)", "LoRA Experimental (Turbo mu)"),
)
def test_experimental_lora_variants_force_strict_official_false(variant: str) -> None:
    result = build_krea2_sigma_schedule(
        variant=variant,
        steps=12,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )

    assert _info(result)["strict_official"] is False


@pytest.mark.parametrize("steps", (1, 7, 12, 37, 10_000))
def test_experimental_lora_accepts_any_bounded_positive_step_count(steps: int) -> None:
    result = build_krea2_sigma_schedule(
        variant="LoRA Experimental (Turbo mu)",
        steps=steps,
        width=1024,
        height=1024,
        strict_official=False,
        start_step=0,
        end_step=-1,
    )

    assert len(result.sigmas) == steps + 1


@pytest.mark.parametrize(
    ("steps", "strict_official", "recipe", "evidence"),
    (
        (52, True, "krea2.raw.official-full-52", "official"),
        (52, False, "krea2.raw.official-full-52", "official"),
        (28, False, "krea2.raw.diffusers-reference-28", "framework_reference"),
    ),
)
def test_raw_selects_only_named_recipes(
    steps: int,
    strict_official: bool,
    recipe: str,
    evidence: str,
) -> None:
    result = build_krea2_sigma_schedule(
        variant="RAW",
        steps=steps,
        width=1024,
        height=768,
        strict_official=strict_official,
        start_step=0,
        end_step=-1,
    )
    info = _info(result)

    assert result.variant is Krea2SigmaVariant.RAW
    assert len(result.sigmas) == steps + 1
    assert info["profile"]["recipe"] == recipe
    assert info["profile"]["evidence"] == evidence
    assert info["shift"]["kind"] == "resolution_exponential_mu"
    assert info["shift"]["image_seq_len"] == 3072


@pytest.mark.parametrize(
    "arguments",
    (
        {
            "variant": "Turbo",
            "steps": 12,
            "width": 1024,
            "height": 1024,
            "strict_official": True,
            "start_step": 0,
            "end_step": -1,
        },
        {
            "variant": "RAW",
            "steps": 28,
            "width": 1024,
            "height": 1024,
            "strict_official": True,
            "start_step": 0,
            "end_step": -1,
        },
        {
            "variant": "RAW",
            "steps": 8,
            "width": 1024,
            "height": 1024,
            "strict_official": False,
            "start_step": 0,
            "end_step": -1,
        },
    ),
)
def test_unsupported_or_non_official_requests_fail_closed(
    arguments: dict[str, Any],
) -> None:
    with pytest.raises(ScheduleContractError):
        build_krea2_sigma_schedule(**arguments)


def test_dimensions_are_required_aligned_and_affect_raw_shift() -> None:
    result = build_krea2_sigma_schedule(
        variant="RAW",
        steps=52,
        width=1025,
        height=769,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    info = _info(result)

    assert info["dimensions"] == {
        "effective": {"height": 784, "width": 1040},
        "requested": {"height": 769, "width": 1025},
    }
    assert info["shift"]["image_seq_len"] == 3185
    assert info["shift"]["mu"] != 1.15


def test_manual_slicing_is_terminal_inclusive_and_fingerprinted_separately() -> None:
    full = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    sliced = build_krea2_sigma_schedule(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=2,
        end_step=6,
    )
    info = _info(sliced)

    assert sliced.sigmas == full.sigmas[2:7]
    assert len(sliced.sigmas) == 5
    assert info["slicing"] == {
        "available_steps": 8,
        "end_step": 6,
        "output_steps": 4,
        "start_step": 2,
    }
    assert info["fingerprints"]["complete"] == numerical_fingerprint(
        full.sigmas,
        domain=full.domain,
        precision="float64",
    )
    assert info["fingerprints"]["output"] == sigma_output_fingerprint(
        sliced.sigmas,
        domain=sliced.domain,
    )
    assert info["fingerprints"]["complete"] != info["fingerprints"]["output"]


@pytest.mark.parametrize(
    "changes",
    (
        {"variant": "unknown"},
        {"variant": 1},
        {"steps": 0},
        {"steps": True},
        {"steps": 10_001},
        {"width": 0},
        {"width": 65_537},
        {"height": False},
        {"strict_official": 1},
        {"start_step": -1},
        {"start_step": True},
        {"start_step": 8},
        {"end_step": -2},
        {"end_step": False},
        {"start_step": 7, "end_step": 7},
        {"start_step": 7, "end_step": 9},
    ),
)
def test_invalid_inputs_fail_before_runtime_conversion(changes: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "variant": "Turbo",
        "steps": 8,
        "width": 1024,
        "height": 1024,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
    }
    arguments.update(changes)

    with pytest.raises((ScheduleContractError, TypeError, ValueError)):
        build_krea2_sigma_schedule(**arguments)


def test_node_result_is_immutable_and_information_is_deterministic() -> None:
    arguments = {
        "variant": "RAW",
        "steps": 52,
        "width": 1024,
        "height": 1024,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
    }
    first = build_krea2_sigma_schedule(**arguments)
    second = build_krea2_sigma_schedule(**arguments)

    assert first == second
    assert first.schedule_info_json == second.schedule_info_json
    assert first.schedule_info_json.encode("utf-8").decode("utf-8") == first.schedule_info_json
    with pytest.raises(FrozenInstanceError):
        first.sigmas = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    (
        {"variant": cast(Any, object())},
        {"domain": SigmaDomain.DISCRETE_TRAINING_INDEX},
        {"sigmas": cast(Any, [1.0, 0.0])},
        {"sigmas": (0.0,)},
        {"schedule_info_json": ""},
        {"schedule_info_json": cast(Any, 1)},
    ),
)
def test_invalid_pure_node_results_fail_closed(changes: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "variant": Krea2SigmaVariant.TURBO,
        "domain": SigmaDomain.UNIT_FLOW,
        "sigmas": (1.0, 0.0),
        "schedule_info_json": "{}",
    }
    arguments.update(changes)

    with pytest.raises(ScheduleContractError):
        Krea2SigmaNodeResult(**arguments)


def test_information_projection_rejects_non_json_values() -> None:
    with pytest.raises(ScheduleContractError, match="canonical JSON"):
        scheduler_module._canonical_info({"invalid": object()})


def test_runtime_node_converts_only_at_execution_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, ...]] = []

    def float_tensor(values: tuple[float, ...]) -> object:
        calls.append(tuple(values))
        quantized = tuple(struct.unpack(">f", struct.pack(">f", value))[0] for value in values)
        return SimpleNamespace(
            values=quantized,
            device="cpu",
            tolist=lambda: list(quantized),
        )

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(FloatTensor=float_tensor))
    output = Krea2SigmaScheduler().build(
        variant="Turbo",
        steps=8,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )

    tensor = cast(SimpleNamespace, output[0])
    assert len(output) == 2
    assert tensor.device == "cpu"
    assert calls != [tensor.values]
    info = json.loads(output[1])
    assert info["schema"] == KREA2_SIGMA_NODE_SCHEMA_ID
    assert info["fingerprints"]["output"] == sigma_output_fingerprint(
        tensor.values,
        domain=SigmaDomain.UNIT_FLOW,
    )


@pytest.mark.parametrize("failure", ("missing_module", "missing_float_tensor"))
def test_runtime_node_fails_actionably_without_torch_contract(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    if failure == "missing_module":

        def missing_module(name: str) -> object:
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", missing_module)
    else:
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: SimpleNamespace(),
        )

    with pytest.raises(RuntimeError, match="requires Torch FloatTensor"):
        Krea2SigmaScheduler().build(
            variant="Turbo",
            steps=8,
            width=1024,
            height=1024,
            strict_official=True,
            start_step=0,
            end_step=-1,
        )


def test_package_import_does_not_eagerly_resolve_torch() -> None:
    source = (
        __import__("pathlib")
        .Path("comfyui_sigmax/nodes/krea2_sigma_scheduler.py")
        .read_text(encoding="utf-8")
    )

    assert "\nimport torch" not in source
    assert "\nfrom torch" not in source
