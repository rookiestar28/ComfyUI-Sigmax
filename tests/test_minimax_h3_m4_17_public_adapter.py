"""M4-17 public ten-scheduler selector and lazy native-adapter contracts."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest
from comfyui_sigmax.adapters import minimax_h3_scheduler as adapter
from comfyui_sigmax.nodes.minimax_h3_sigma_scheduler import (
    MINIMAX_H3_TURBO_RECIPE_CHOICES,
    MiniMaxH3SigmaScheduler,
    build_minimax_h3_sigma_schedule,
)
from comfyui_sigmax.profiles.minimax_h3_scheduler_contract import (
    MINIMAX_H3_DEFAULT_SCHEDULER,
    MINIMAX_H3_NATIVE_SCHEDULERS,
    MINIMAX_H3_SCHEDULER_CHOICES,
    MiniMaxH3SchedulerContractError,
)
from comfyui_sigmax.workflows.minimax_h3 import (
    MiniMaxH3WorkflowSpec,
    build_minimax_h3_host_workflow_prompt,
)


class _MiniMaxH3Config:
    unet_config: ClassVar[dict[str, str]] = {"image_model": "minimax_h3"}


class _ModelSamplingDiscreteFlow:
    pass


class _ModelSamplingAV(_ModelSamplingDiscreteFlow):
    pass


class _Sampling(_ModelSamplingAV):
    shift = 12.0
    audio_shift = 3.0


class _LegacySampling(_ModelSamplingDiscreteFlow):
    shift = 12.0


class _FakeTensor:
    dtype = "torch.float32"

    def __init__(self, values: tuple[float, ...]) -> None:
        self._values = values

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[float]:
        return list(self._values)


class _Model:
    def __init__(
        self,
        *,
        sampling: object | None = None,
        config: object | None = None,
        transformer_options: dict[str, object] | None = None,
    ) -> None:
        self._sampling = _Sampling() if sampling is None else sampling
        self.model = SimpleNamespace(model_config=_MiniMaxH3Config() if config is None else config)
        self.model_options = {"transformer_options": transformer_options or {}}

    def get_model_object(self, name: str) -> object:
        assert name == "model_sampling"
        return self._sampling


def _install_host_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "0.32.0",
    raw: tuple[float, ...] = (0.95, 0.8, 0.6, 0.4, 0.2, 0.0),
    handlers: tuple[str, ...] = MINIMAX_H3_NATIVE_SCHEDULERS,
) -> list[tuple[object, str, int]]:
    calls: list[tuple[object, str, int]] = []

    def calculate_sigmas(model_sampling: object, scheduler: str, steps: int) -> _FakeTensor:
        calls.append((model_sampling, scheduler, steps))
        return _FakeTensor(raw)

    modules = {
        "comfyui_version": SimpleNamespace(__version__=version),
        "comfy.samplers": SimpleNamespace(
            SCHEDULER_NAMES=list(handlers),
            SCHEDULER_HANDLERS={name: object() for name in handlers},
            calculate_sigmas=calculate_sigmas,
        ),
        "comfy.model_sampling": SimpleNamespace(ModelSamplingAV=_ModelSamplingAV),
        "comfy.supported_models": SimpleNamespace(MiniMaxH3=_MiniMaxH3Config),
        "torch": SimpleNamespace(
            float32="torch.float32",
            tensor=lambda values, *, dtype: _FakeTensor(tuple(float(value) for value in values)),
        ),
    }

    def import_module(name: str) -> object:
        try:
            return modules[name]
        except KeyError as exc:  # pragma: no cover - catches an unexpected production import
            raise AssertionError(f"unexpected host import: {name}") from exc

    monkeypatch.setattr(importlib, "import_module", import_module)
    return calls


def _install_legacy_host_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, str, int]]:
    calls = _install_host_modules(monkeypatch, version="0.30.0")
    original_import = importlib.import_module

    def import_module(name: str) -> object:
        if name == "comfy.model_sampling":
            return SimpleNamespace(ModelSamplingDiscreteFlow=_ModelSamplingDiscreteFlow)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    return calls


def _reason(exc: pytest.ExceptionInfo[MiniMaxH3SchedulerContractError]) -> str:
    return exc.value.reason_code.value


def test_public_schema_appends_exact_scheduler_and_model_without_moving_legacy_widgets() -> None:
    inputs = MiniMaxH3SigmaScheduler.INPUT_TYPES()
    optional = inputs["optional"]

    assert tuple(optional) == ("turbo", "recipe_id", "scheduler", "model")
    assert optional["turbo"][0] == MINIMAX_H3_TURBO_RECIPE_CHOICES
    assert optional["recipe_id"][0] == MINIMAX_H3_TURBO_RECIPE_CHOICES
    assert optional["scheduler"][0] == MINIMAX_H3_SCHEDULER_CHOICES
    assert cast(dict[str, object], optional["scheduler"][1])["default"] == (
        MINIMAX_H3_DEFAULT_SCHEDULER
    )
    assert optional["model"] == ("MODEL",)


def test_omitted_scheduler_keeps_exact_accepted_pure_base_result() -> None:
    legacy = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA", steps=20, start_step=0, end_step=-1
    )
    node = MiniMaxH3SigmaScheduler()

    # The public build helper remains the dependency-free compatibility oracle. The node default
    # is asserted through its schema and dedicated routing tests below.
    repeated = build_minimax_h3_sigma_schedule(
        variant="H3 Base FL2VA", steps=20, start_step=0, end_step=-1
    )
    assert repeated == legacy
    assert MINIMAX_H3_DEFAULT_SCHEDULER == "h3_endpoint"
    scheduler_options = cast(dict[str, object], node.INPUT_TYPES()["optional"]["scheduler"][1])
    assert scheduler_options["default"] == "h3_endpoint"


def test_native_adapter_delegates_once_and_mirrors_basic_scheduler_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_host_modules(monkeypatch)
    model = _Model(
        transformer_options={
            "minimax_h3_sigma_shift_video": 12.0,
            "minimax_h3_sigma_shift_audio": 3.0,
        }
    )

    result = adapter.build_minimax_h3_native_schedule(
        model=model,
        scheduler="simple",
        variant="H3 Base FL2VA",
        steps=4,
        start_step=1,
        end_step=3,
        recipe_id=None,
    )

    assert calls == [(model._sampling, "simple", 4)]
    assert result.host_version == "0.32.0"
    assert result.qualified_host_revision == (
        "b323a345bbbfb2f3a95b5b73b68eb7919a26515e"  # pragma: allowlist secret
    )
    assert result.validation.scheduler == "simple"
    assert result.validation.raw_count == 6
    assert result.validation.basic_scheduler_sigmas == (0.8, 0.6, 0.4, 0.2, 0.0)
    assert result.validation.output_sigmas == (0.6, 0.4, 0.2)
    assert result.validation.start_step == 1
    assert result.validation.end_step == 3
    assert result.validation.output_fingerprint.startswith("sha256:")


def test_legacy_host_accepts_discrete_flow_only_with_complete_shift_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_legacy_host_modules(monkeypatch)
    model = _Model(
        sampling=_LegacySampling(),
        transformer_options={
            "minimax_h3_sigma_shift_video": 12.0,
            "minimax_h3_sigma_shift_audio": 3.0,
        },
    )

    result = adapter.build_minimax_h3_native_schedule(
        model=model,
        scheduler="simple",
        variant="H3 Base FL2VA",
        steps=4,
        start_step=0,
        end_step=-1,
        recipe_id=None,
    )

    assert calls == [(model._sampling, "simple", 4)]
    assert result.host_version == "0.30.0"
    assert result.sampling_api == "model_sampling_discrete_flow_h3_v030"
    assert result.projection()["sampling_api"] == result.sampling_api


def test_sampling_api_is_qualified_per_host_and_legacy_markers_are_mandatory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_legacy_host_modules(monkeypatch)
    with pytest.raises(MiniMaxH3SchedulerContractError) as missing_markers:
        adapter.build_minimax_h3_native_schedule(
            model=_Model(sampling=_LegacySampling()),
            scheduler="simple",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(missing_markers) == "SHIFT_MISMATCH"

    _install_host_modules(monkeypatch, version="0.32.0")
    with pytest.raises(MiniMaxH3SchedulerContractError) as wrong_api:
        adapter.build_minimax_h3_native_schedule(
            model=_Model(
                sampling=_LegacySampling(),
                transformer_options={
                    "minimax_h3_sigma_shift_video": 12.0,
                    "minimax_h3_sigma_shift_audio": 3.0,
                },
            ),
            scheduler="simple",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(wrong_api) == "MODEL_SAMPLING_NOT_AV"


@pytest.mark.parametrize(
    ("scheduler", "steps"),
    tuple((name, 4) for name in MINIMAX_H3_NATIVE_SCHEDULERS),
)
def test_every_native_choice_reaches_the_selected_host_handler(
    monkeypatch: pytest.MonkeyPatch, scheduler: str, steps: int
) -> None:
    calls = _install_host_modules(monkeypatch, raw=(0.8, 0.6, 0.4, 0.2, 0.0))
    result = adapter.build_minimax_h3_native_schedule(
        model=_Model(),
        scheduler=scheduler,
        variant="H3 Base Ref2VA",
        steps=steps,
        start_step=0,
        end_step=-1,
        recipe_id=None,
    )
    assert calls[0][1:] == (scheduler, steps)
    assert result.validation.model_task == "ref2va"
    assert result.validation.output_sigmas[-1] == 0.0


def test_adapter_fails_closed_for_missing_model_before_import_or_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(_name: str) -> object:
        raise AssertionError("missing MODEL must fail before host import")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    with pytest.raises(MiniMaxH3SchedulerContractError) as exc:
        adapter.build_minimax_h3_native_schedule(
            model=None,
            scheduler="simple",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(exc) == "MODEL_REQUIRED"


def test_adapter_fails_closed_for_unknown_host_missing_handler_and_non_h3_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_host_modules(monkeypatch, version="9.9.9")
    with pytest.raises(MiniMaxH3SchedulerContractError) as host:
        adapter.build_minimax_h3_native_schedule(
            model=_Model(),
            scheduler="simple",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(host) == "UNSUPPORTED_HOST"

    _install_host_modules(monkeypatch, handlers=("normal",))
    with pytest.raises(MiniMaxH3SchedulerContractError) as handler:
        adapter.build_minimax_h3_native_schedule(
            model=_Model(),
            scheduler="simple",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(handler) == "MISSING_HANDLER"

    _install_host_modules(monkeypatch)
    with pytest.raises(MiniMaxH3SchedulerContractError) as family:
        adapter.build_minimax_h3_native_schedule(
            model=_Model(config=object()),
            scheduler="simple",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(family) == "MODEL_FAMILY_MISMATCH"


def test_adapter_fails_closed_for_non_av_and_conflicting_shift_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_host_modules(monkeypatch)
    sampling = SimpleNamespace(shift=12.0, audio_shift=3.0)
    with pytest.raises(MiniMaxH3SchedulerContractError) as av:
        adapter.build_minimax_h3_native_schedule(
            model=_Model(sampling=sampling),
            scheduler="normal",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(av) == "MODEL_SAMPLING_NOT_AV"

    with pytest.raises(MiniMaxH3SchedulerContractError) as shift:
        adapter.build_minimax_h3_native_schedule(
            model=_Model(
                transformer_options={
                    "minimax_h3_sigma_shift_video": 6.0,
                    "minimax_h3_sigma_shift_audio": 3.0,
                }
            ),
            scheduler="normal",
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            recipe_id=None,
        )
    assert _reason(shift) == "SHIFT_MISMATCH"


def test_native_metadata_is_allowlisted_truthful_and_path_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_host_modules(monkeypatch, raw=(0.8, 0.6, 0.4, 0.2, 0.0))
    result = adapter.build_minimax_h3_native_schedule(
        model=_Model(),
        scheduler="karras",
        variant="H3 Base FL2VA",
        steps=4,
        start_step=0,
        end_step=-1,
        recipe_id=None,
    )
    projection = result.projection()
    encoded = json.dumps(projection, sort_keys=True)

    assert projection["owner"] == "comfyui_native"
    assert projection["scheduler"] == "karras"
    assert projection["host"] == {
        "observed_version": "0.32.0",
        "qualified_revision": "b323a345bbbfb2f3a95b5b73b68eb7919a26515e",  # pragma: allowlist secret
    }
    assert projection["counts"] == {
        "actual_sigmas": 5,
        "actual_transitions": 4,
        "raw_sigmas": 5,
        "requested_steps": 4,
    }
    assert "model_filename" not in encoded
    assert "checkpoint" not in encoded.lower()
    assert ":\\" not in encoded and "/home/" not in encoded


def test_public_node_routes_native_result_and_binds_turbo_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_host_modules(monkeypatch, raw=(0.8, 0.6, 0.4, 0.2, 0.0))
    tensor, encoded = MiniMaxH3SigmaScheduler().build(
        variant="H3 Base FL2VA",
        steps=4,
        start_step=0,
        end_step=-1,
        turbo="h3.fl2va.lightx2v-turbo-4-v0.1-544p",
        scheduler="simple",
        model=_Model(),
    )
    info = cast(dict[str, object], json.loads(encoded))
    scheduler_info = cast(dict[str, object], info["scheduler"])
    receipt = cast(dict[str, object], info["turbo_receipt"])

    assert cast(_FakeTensor, tensor).tolist() == [0.8, 0.6, 0.4, 0.2, 0.0]
    assert calls[0][1:] == ("simple", 4)
    assert info["mode"] == "experimental_comfyui_native_scheduler"
    assert scheduler_info["owner"] == "comfyui_native"
    assert scheduler_info["scheduler"] == "simple"
    assert (
        receipt["schedule_fingerprint"] == cast(dict[str, object], info["fingerprints"])["output"]
    )
    assert "experimental_comfyui_native_scheduler_not_official_minimax" in cast(
        list[str], info["warnings"]
    )


def test_public_node_rejects_inert_model_and_unknown_scheduler_before_host_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(_name: str) -> object:
        raise AssertionError("request must fail before host import")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    node = MiniMaxH3SigmaScheduler()
    with pytest.raises(MiniMaxH3SchedulerContractError) as inert:
        node.build(
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            scheduler="h3_endpoint",
            model=_Model(),
        )
    assert _reason(inert) == "MODEL_FORBIDDEN"

    with pytest.raises(MiniMaxH3SchedulerContractError) as unknown:
        node.build(
            variant="H3 Base FL2VA",
            steps=4,
            start_step=0,
            end_step=-1,
            scheduler="future",
            model=None,
        )
    assert _reason(unknown) == "UNSUPPORTED_SCHEDULER"


def test_workflow_default_is_unchanged_and_explicit_native_connects_shifted_model() -> None:
    legacy = build_minimax_h3_host_workflow_prompt(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="legacy")
    )
    assert legacy["7"]["inputs"] == {
        "variant": "H3 Base FL2VA",
        "steps": 20,
        "start_step": 0,
        "end_step": -1,
    }

    native = build_minimax_h3_host_workflow_prompt(
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="native", scheduler="simple")
    )
    native_schedule = native["7"]
    native_inputs = cast(dict[str, object], native_schedule["inputs"])
    native_shift = native["5"]
    assert native_inputs["scheduler"] == "simple"
    assert native_inputs["model"] == ["5", 0]
    assert native_shift["class_type"] == "MiniMaxH3SigmaShift"


def test_workflow_rejects_unknown_scheduler_and_pure_model_wiring_is_absent() -> None:
    with pytest.raises(Exception, match="scheduler"):
        MiniMaxH3WorkflowSpec(variant="H3 Base FL2VA", prompt="unknown", scheduler="future")
    default = build_minimax_h3_host_workflow_prompt(
        MiniMaxH3WorkflowSpec(
            variant="H3 Base Ref2VA",
            prompt="pure",
            reference_images=("ref.png",),
            scheduler="h3_endpoint",
        )
    )
    default_schedule = default["7"]
    default_inputs = cast(dict[str, object], default_schedule["inputs"])
    assert "scheduler" not in default_inputs
    assert "model" not in default_inputs
