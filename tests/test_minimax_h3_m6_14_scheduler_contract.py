"""M6-14 pure MiniMax H3 ten-scheduler qualification contracts."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from types import ModuleType
from typing import Any

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes.minimax_h3_sigma_scheduler import MiniMaxH3SigmaScheduler


def _module() -> ModuleType:
    return importlib.import_module("comfyui_sigmax.profiles.minimax_h3_scheduler_contract")


def _model(
    module: ModuleType,
    *,
    family_id: str = "minimax_h3",
    task: str = "fl2va",
    is_model_sampling_av: bool = True,
    sampling_api: str = "model_sampling_av_v032",
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    already_shifted: bool = True,
) -> object:
    return module.MiniMaxH3ModelSamplingEvidence(
        family_id=family_id,
        task=task,
        is_model_sampling_av=is_model_sampling_av,
        sampling_api=module.MiniMaxH3SamplingAPI(sampling_api),
        video_shift=video_shift,
        audio_shift=audio_shift,
        already_shifted=already_shifted,
    )


def _qualified(
    module: ModuleType,
    scheduler: str,
    steps: int,
    *,
    model_sampling: object | None = None,
    recipe_id: str | None = None,
) -> object:
    if scheduler == "h3_endpoint":
        return module.qualify_minimax_h3_scheduler_request(
            scheduler=scheduler,
            steps=steps,
            recipe_id=recipe_id,
        )
    return module.qualify_minimax_h3_scheduler_request(
        scheduler=scheduler,
        steps=steps,
        model_sampling=_model(module) if model_sampling is None else model_sampling,
        recipe_id=recipe_id,
        host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
        available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
    )


def test_m6_14_remains_the_pure_contract_behind_the_m4_17_public_seam() -> None:
    module = _module()
    assert module.MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_ID == (
        "sigmax.minimax-h3-ten-scheduler-contract/1"
    )
    assert module.MINIMAX_H3_SCHEDULER_CONTRACT_SCHEMA_VERSION == "1"
    assert module.MINIMAX_H3_SCHEDULER_RESULT_SCHEMA_ID == ("sigmax.minimax-h3-scheduler-result/1")
    schema = MiniMaxH3SigmaScheduler.INPUT_TYPES()
    assert "scheduler" not in schema["required"]
    assert schema["optional"]["scheduler"][0] == module.MINIMAX_H3_SCHEDULER_CHOICES
    assert "model" not in schema["required"]
    assert schema["optional"]["model"] == ("MODEL",)

    script = """
import sys
import comfyui_sigmax
assert not any(name == 'comfy' or name.startswith('comfy.') for name in sys.modules)
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_m6_14_exact_order_default_and_owner_matrix_are_frozen() -> None:
    module = _module()
    expected = (
        "h3_endpoint",
        "simple",
        "sgm_uniform",
        "karras",
        "exponential",
        "ddim_uniform",
        "beta",
        "normal",
        "linear_quadratic",
        "kl_optimal",
    )
    assert expected == module.MINIMAX_H3_SCHEDULER_CHOICES
    assert module.MINIMAX_H3_DEFAULT_SCHEDULER == "h3_endpoint"
    assert expected[1:] == module.MINIMAX_H3_NATIVE_SCHEDULERS

    contracts = module.MINIMAX_H3_SCHEDULER_CONTRACTS
    assert tuple(contract.name for contract in contracts) == expected
    assert tuple(contract.ordinal for contract in contracts) == tuple(range(1, 11))
    pure = contracts[0]
    assert pure.owner is module.MiniMaxH3SchedulerOwner.SIGMAX_PURE
    assert pure.model_policy is module.MiniMaxH3ModelPolicy.FORBIDDEN
    assert pure.handler_name is None
    assert pure.count_policy is module.MiniMaxH3CountPolicy.EXACT_ENDPOINT
    for contract in contracts[1:]:
        assert contract.owner is module.MiniMaxH3SchedulerOwner.COMFYUI_NATIVE
        assert contract.model_policy is module.MiniMaxH3ModelPolicy.REQUIRED
        assert contract.handler_name == contract.name
        assert contract.additional_shift_allowed is False
        assert contract.basic_scheduler_tail is True


def test_m6_14_contract_records_are_immutable_and_self_validating() -> None:
    module = _module()
    contract = module.MINIMAX_H3_SCHEDULER_CONTRACTS[0]
    with pytest.raises(FrozenInstanceError):
        contract.ordinal = 99
    with pytest.raises(ScheduleContractError, match="ordinal"):
        replace(contract, ordinal=0)
    with pytest.raises(ScheduleContractError, match="handler"):
        replace(contract, handler_name="simple")
    native = module.MINIMAX_H3_SCHEDULER_CONTRACTS[1]
    with pytest.raises(ScheduleContractError, match="MODEL"):
        replace(native, model_policy=module.MiniMaxH3ModelPolicy.FORBIDDEN)


def test_m6_14_supported_host_source_matrix_is_exact_and_gpl_scoped() -> None:
    module = _module()
    hosts = module.MINIMAX_H3_SCHEDULER_HOSTS
    assert [(host.version, host.revision, host.role.value) for host in hosts] == [
        (
            "0.30.0",
            "14b05228cef127ce529bc0c08660770d4af3e9a8",  # pragma: allowlist secret
            "accepted_known_good",
        ),
        (
            "0.32.0",
            "b323a345bbbfb2f3a95b5b73b68eb7919a26515e",  # pragma: allowlist secret
            "supplied_current",
        ),
    ]
    for host in hosts:
        assert host.scheduler_names == module.MINIMAX_H3_NATIVE_SCHEDULERS
        assert host.license_id == "GPL-3.0-only"
        assert host.delegation_only is True
        assert host.url.startswith("https://github.com/Comfy-Org/ComfyUI")
        assert host.source_locators == tuple(sorted(set(host.source_locators)))
        assert "comfy/samplers.py" in host.source_locators
        assert "comfy/model_sampling.py" in host.source_locators
    assert [host.sampling_api.value for host in hosts] == [
        "model_sampling_discrete_flow_h3_v030",
        "model_sampling_av_v032",
    ]


def test_m6_14_sampling_api_must_match_the_exact_host_revision() -> None:
    module = _module()
    legacy_host, current_host = module.MINIMAX_H3_SCHEDULER_HOSTS
    legacy = module.qualify_minimax_h3_scheduler_request(
        scheduler="simple",
        steps=4,
        model_sampling=_model(
            module,
            is_model_sampling_av=False,
            sampling_api="model_sampling_discrete_flow_h3_v030",
        ),
        host_revision=legacy_host.revision,
        available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
    )
    assert legacy.sampling_api.value == "model_sampling_discrete_flow_h3_v030"

    with pytest.raises(module.MiniMaxH3SchedulerContractError) as mismatch:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="simple",
            steps=4,
            model_sampling=_model(
                module,
                is_model_sampling_av=False,
                sampling_api="model_sampling_discrete_flow_h3_v030",
            ),
            host_revision=current_host.revision,
            available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
        )
    assert mismatch.value.reason_code is module.MiniMaxH3SchedulerReasonCode.MODEL_SAMPLING_NOT_AV


@pytest.mark.parametrize("scheduler", ("beta", "kl_optimal"))
def test_m6_14_beta_and_kl_optimal_reject_one_step_before_dispatch(scheduler: str) -> None:
    module = _module()
    with pytest.raises(module.MiniMaxH3SchedulerContractError) as error:
        module.qualify_minimax_h3_scheduler_request(
            scheduler=scheduler,
            steps=1,
            model_sampling=_model(module),
            host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
            available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
        )
    assert error.value.reason_code is module.MiniMaxH3SchedulerReasonCode.INVALID_STEPS


def test_m6_14_linear_quadratic_retains_the_host_one_step_case() -> None:
    module = _module()
    result = module.qualify_minimax_h3_scheduler_request(
        scheduler="linear_quadratic",
        steps=1,
        model_sampling=_model(module),
        host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
        available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
    )
    assert result.steps == 1
    assert result.handler_name == "linear_quadratic"
    assert result.additional_shift_allowed is False


def test_m6_14_pure_path_forbids_inert_model_and_native_requires_model() -> None:
    module = _module()
    pure = module.qualify_minimax_h3_scheduler_request(
        scheduler="h3_endpoint", steps=20, model_sampling=None
    )
    assert pure.owner is module.MiniMaxH3SchedulerOwner.SIGMAX_PURE
    assert pure.expected_video_shift == 12.0
    assert pure.expected_audio_shift == 3.0
    with pytest.raises(module.MiniMaxH3SchedulerContractError) as forbidden:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="h3_endpoint", steps=20, model_sampling=_model(module)
        )
    assert forbidden.value.reason_code is module.MiniMaxH3SchedulerReasonCode.MODEL_FORBIDDEN

    with pytest.raises(module.MiniMaxH3SchedulerContractError) as required:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="simple",
            steps=20,
            model_sampling=None,
            host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
            available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
        )
    assert required.value.reason_code is module.MiniMaxH3SchedulerReasonCode.MODEL_REQUIRED


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"family_id": "flux"}, "MODEL_FAMILY_MISMATCH"),
        ({"is_model_sampling_av": False}, "MODEL_SAMPLING_NOT_AV"),
        ({"already_shifted": False}, "MODEL_NOT_ALREADY_SHIFTED"),
        ({"video_shift": 6.0}, "SHIFT_MISMATCH"),
        ({"audio_shift": 4.0}, "SHIFT_MISMATCH"),
    ],
)
def test_m6_14_native_model_evidence_fails_closed(overrides: dict[str, Any], reason: str) -> None:
    module = _module()
    with pytest.raises(module.MiniMaxH3SchedulerContractError) as error:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="normal",
            steps=20,
            model_sampling=_model(module, **overrides),
            host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
            available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
        )
    assert error.value.reason_code.value == reason


def test_m6_14_turbo_recipe_shift_matrix_is_exact() -> None:
    module = _module()
    assert {
        item.recipe_id: (item.task, item.allowed_nfe, item.video_shift, item.audio_shift)
        for item in module.MINIMAX_H3_SCHEDULER_RECIPES
    } == {
        "h3.fl2va.lightx2v-turbo-4-v0.1-544p": ("fl2va", (4,), 12.0, 3.0),
        "h3.fl2va.lightx2v-turbo-8-v1.0-544p": ("fl2va", (4, 8), 12.0, 3.0),
        "h3.fl2va.lightx2v-turbo-4-v1.0-768p": ("fl2va", (4,), 6.0, 3.0),
        "h3.ref2va.lightx2v-turbo-4-v0.1-544p": ("ref2va", (4,), 12.0, 3.0),
    }

    selected = module.qualify_minimax_h3_scheduler_request(
        scheduler="simple",
        steps=4,
        recipe_id="h3.fl2va.lightx2v-turbo-4-v1.0-768p",
        model_sampling=_model(module, video_shift=6.0),
        host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
        available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
    )
    assert (selected.expected_video_shift, selected.expected_audio_shift) == (6.0, 3.0)
    with pytest.raises(module.MiniMaxH3SchedulerContractError) as mismatch:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="simple",
            steps=4,
            recipe_id="h3.fl2va.lightx2v-turbo-4-v1.0-768p",
            model_sampling=_model(module),
            host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
            available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
        )
    assert mismatch.value.reason_code is module.MiniMaxH3SchedulerReasonCode.SHIFT_MISMATCH

    with pytest.raises(module.MiniMaxH3SchedulerContractError) as task_mismatch:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="normal",
            steps=4,
            recipe_id="h3.ref2va.lightx2v-turbo-4-v0.1-544p",
            model_sampling=_model(module, task="fl2va"),
            host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
            available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
        )
    assert task_mismatch.value.reason_code is (
        module.MiniMaxH3SchedulerReasonCode.MODEL_TASK_MISMATCH
    )

    ref2va = module.qualify_minimax_h3_scheduler_request(
        scheduler="normal",
        steps=4,
        recipe_id="h3.ref2va.lightx2v-turbo-4-v0.1-544p",
        model_sampling=_model(module, task="ref2va"),
        host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
        available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
    )
    assert ref2va.model_task == "ref2va"
    assert ref2va.recipe_task == "ref2va"


def test_m6_14_unknown_scheduler_host_handler_and_recipe_fail_closed() -> None:
    module = _module()
    with pytest.raises(module.MiniMaxH3SchedulerContractError) as unknown:
        module.qualify_minimax_h3_scheduler_request(scheduler="future", steps=4)
    assert unknown.value.reason_code is module.MiniMaxH3SchedulerReasonCode.UNSUPPORTED_SCHEDULER

    with pytest.raises(module.MiniMaxH3SchedulerContractError) as host:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="simple",
            steps=4,
            model_sampling=_model(module),
            host_revision="0" * 40,
            available_handlers=module.MINIMAX_H3_NATIVE_SCHEDULERS,
        )
    assert host.value.reason_code is module.MiniMaxH3SchedulerReasonCode.UNSUPPORTED_HOST

    with pytest.raises(module.MiniMaxH3SchedulerContractError) as handler:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="simple",
            steps=4,
            model_sampling=_model(module),
            host_revision=module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision,
            available_handlers=("normal",),
        )
    assert handler.value.reason_code is module.MiniMaxH3SchedulerReasonCode.MISSING_HANDLER

    with pytest.raises(module.MiniMaxH3SchedulerContractError) as recipe:
        module.qualify_minimax_h3_scheduler_request(
            scheduler="h3_endpoint", steps=4, recipe_id="unknown.recipe"
        )
    assert recipe.value.reason_code is module.MiniMaxH3SchedulerReasonCode.UNSUPPORTED_RECIPE


def test_m6_14_native_result_tracks_raw_basic_tail_and_sigmax_slice_counts() -> None:
    module = _module()
    # Synthetic monotonic values model a raw DDIM handler result longer than steps+1.
    result = module.validate_minimax_h3_scheduler_result(
        qualification=_qualified(module, "ddim_uniform", 4),
        raw_sigmas=(1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0),
        dtype="float32",
        start_step=1,
        end_step=3,
    )
    assert result.raw_count == 7
    assert result.basic_scheduler_count == 5
    assert result.basic_scheduler_sigmas == (0.7, 0.5, 0.3, 0.1, 0.0)
    assert result.output_sigmas == (0.5, 0.3, 0.1)
    assert result.output_transitions == 2

    beta = module.validate_minimax_h3_scheduler_result(
        qualification=_qualified(module, "beta", 8),
        raw_sigmas=(1.0, 0.4, 0.0),
        dtype="float32",
    )
    assert beta.raw_count == 3
    assert beta.basic_scheduler_count == 3
    assert beta.output_sigmas == (1.0, 0.4, 0.0)


def test_m6_14_pure_result_preserves_exact_endpoint_count_and_dtypes() -> None:
    module = _module()
    qualification = _qualified(module, "h3_endpoint", 4)
    values = (1.0, 0.75, 0.5, 0.25, 0.0)
    for dtype in ("float64", "float32"):
        result = module.validate_minimax_h3_scheduler_result(
            qualification=qualification,
            raw_sigmas=values,
            dtype=dtype,
        )
        assert result.raw_count == 5
        assert result.basic_scheduler_count == 5
        assert result.output_transitions == 4
        assert result.output_fingerprint.startswith("sha256:")

    with pytest.raises(module.MiniMaxH3SchedulerContractError) as count:
        module.validate_minimax_h3_scheduler_result(
            qualification=qualification,
            raw_sigmas=(1.0, 0.5, 0.0),
            dtype="float64",
        )
    assert count.value.reason_code is module.MiniMaxH3SchedulerReasonCode.RESULT_COUNT_INVALID


@pytest.mark.parametrize(
    ("sigmas", "dtype", "reason"),
    [
        ((1.0, float("nan"), 0.0), "float32", "RESULT_NON_FINITE"),
        ((1.0, 0.2, 0.3, 0.0), "float32", "RESULT_NOT_MONOTONIC"),
        ((1.0, 0.2, 0.1), "float32", "RESULT_TERMINAL_INVALID"),
        ((1.0, 0.2, 1e-12), "float32", "RESULT_TERMINAL_INVALID"),
        ((1.0, 0.2, 0.0), "float64", "RESULT_DTYPE_INVALID"),
    ],
)
def test_m6_14_native_result_validation_fails_closed(
    sigmas: tuple[float, ...], dtype: str, reason: str
) -> None:
    module = _module()
    with pytest.raises(module.MiniMaxH3SchedulerContractError) as error:
        module.validate_minimax_h3_scheduler_result(
            qualification=_qualified(module, "normal", 4),
            raw_sigmas=sigmas,
            dtype=dtype,
        )
    assert error.value.reason_code.value == reason


def test_m6_14_result_fingerprint_binds_complete_scheduler_identity() -> None:
    module = _module()
    values = (1.0, 0.5, 0.0)
    simple = module.validate_minimax_h3_scheduler_result(
        qualification=_qualified(module, "simple", 4),
        raw_sigmas=values,
        dtype="float32",
    )
    normal = module.validate_minimax_h3_scheduler_result(
        qualification=_qualified(module, "normal", 4),
        raw_sigmas=values,
        dtype="float32",
    )
    assert simple.output_sigmas == normal.output_sigmas
    assert simple.output_fingerprint != normal.output_fingerprint
    assert simple.contract_fingerprint == module.minimax_h3_scheduler_contract_fingerprint()
    assert simple.host_revision == module.MINIMAX_H3_SCHEDULER_HOSTS[1].revision
    assert simple.start_step == 0
    assert simple.end_step == 2


def test_m6_14_manifest_serialization_and_fingerprint_are_deterministic_and_public_safe() -> None:
    module = _module()
    first = module.serialize_minimax_h3_scheduler_contract()
    second = module.serialize_minimax_h3_scheduler_contract()
    assert first == second
    assert module.minimax_h3_scheduler_contract_fingerprint() == (
        module.minimax_h3_scheduler_contract_fingerprint()
    )
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"))
    assert "A:\\" not in encoded
    assert "/home/" not in encoded
    assert "prompt" not in encoded.lower()
    assert "model_filename" not in encoded.lower()
    assert first["choices"] == list(module.MINIMAX_H3_SCHEDULER_CHOICES)
    assert first["default"] == "h3_endpoint"
    assert first["native_formula_copied"] is False
    assert first["runtime_registered"] is False
    assert first["qualification"] == {
        "family_id": "minimax_h3",
        "model_tasks": ["fl2va", "ref2va"],
        "native_model_required": True,
        "native_recipe_task_match": True,
        "native_requires_already_shifted": True,
        "native_sampling_apis": [
            "model_sampling_discrete_flow_h3_v030",
            "model_sampling_av_v032",
        ],
        "pure_model_forbidden": True,
    }
    assert first["result_identity"]["result_schema_id"] == (
        module.MINIMAX_H3_SCHEDULER_RESULT_SCHEMA_ID
    )
    assert first["result_identity"]["terminal_policy"] == "require_exact_zero_preserve"
    assert "scheduler" in first["result_identity"]["fields"]
    assert "model_task" in first["result_identity"]["fields"]
    assert "raw_values" in first["result_identity"]["fields"]
