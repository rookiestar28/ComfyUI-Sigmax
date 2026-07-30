"""Product contracts for the model-aware Krea 2 SIGMAS scheduler node."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import CompatibilityLevel, ScheduleContractError, SigmaDomain
from comfyui_sigmax.nodes import model_aware_sigma_scheduler as model_aware_module
from comfyui_sigmax.nodes.krea2_sigma_scheduler import build_krea2_sigma_schedule
from comfyui_sigmax.nodes.model_aware_sigma_scheduler import (
    MODEL_AWARE_SIGMA_NODE_ID,
    MODEL_AWARE_SIGMA_NODE_SCHEMA_ID,
    ModelAwareScheduleError,
    ModelAwareSigmaNodeResult,
    ModelAwareSigmaScheduler,
    ModelFamilyProbe,
    build_model_aware_sigma_schedule,
    probe_model_family,
)
from comfyui_sigmax.profiles import HostCapabilities


def _krea2_model() -> object:
    krea2 = type("Krea2", (), {})
    return SimpleNamespace(model=krea2())


def _info(result: ModelAwareSigmaNodeResult) -> dict[str, Any]:
    decoded = json.loads(result.schedule_info_json)
    assert isinstance(decoded, dict)
    return decoded


def _arguments(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model": _krea2_model(),
        "variant": "Turbo",
        "steps": 8,
        "width": 1024,
        "height": 1024,
        "strict_official": True,
        "start_step": 0,
        "end_step": -1,
    }
    values.update(changes)
    return values


def test_node_declares_model_aware_legacy_current_schema() -> None:
    inputs = ModelAwareSigmaScheduler.INPUT_TYPES()

    assert MODEL_AWARE_SIGMA_NODE_ID == "Sigmax.ModelAwareSigmaScheduler"
    assert MODEL_AWARE_SIGMA_NODE_SCHEMA_ID == "sigmax.model-aware-sigma-node/1"
    assert ModelAwareSigmaScheduler.RETURN_TYPES == ("SIGMAS", "STRING")
    assert ModelAwareSigmaScheduler.RETURN_NAMES == ("sigmas", "schedule_info")
    assert ModelAwareSigmaScheduler.FUNCTION == "build"
    assert ModelAwareSigmaScheduler.CATEGORY == "Sigmax/scheduling"
    assert ModelAwareSigmaScheduler.OUTPUT_NODE is False
    assert inputs["required"]["model"] == ("MODEL",)
    assert inputs["required"]["variant"][0] == ("Auto", "Turbo", "RAW")
    assert set(inputs["required"]) == {
        "model",
        "variant",
        "steps",
        "width",
        "height",
        "strict_official",
        "start_step",
        "end_step",
    }
    assert inputs == ModelAwareSigmaScheduler.INPUT_TYPES()
    assert inputs is not ModelAwareSigmaScheduler.INPUT_TYPES()


@pytest.mark.parametrize(
    ("model", "expected_signals"),
    (
        (_krea2_model(), ("model_class:krea2",)),
        (
            SimpleNamespace(
                model=SimpleNamespace(
                    model_config=type("Krea2", (), {"unet_config": {}})(),
                )
            ),
            ("model_config_class:krea2",),
        ),
        (
            SimpleNamespace(
                model=SimpleNamespace(
                    model_config=SimpleNamespace(unet_config={"image_model": "krea2"})
                )
            ),
            ("unet_config.image_model:krea2",),
        ),
    ),
)
def test_probe_accepts_only_bounded_public_krea2_family_signals(
    model: object,
    expected_signals: tuple[str, ...],
) -> None:
    probe = probe_model_family(model)

    assert probe == ModelFamilyProbe(
        schema_id="sigmax.model-family-probe/1",
        family="krea2",
        supported=True,
        signals=expected_signals,
        reason_code="model.family_krea2",
    )


def test_probe_does_not_invoke_foreign_methods_or_leak_object_representations() -> None:
    calls: list[str] = []

    class Foreign:
        def get_model_object(self, name: str) -> object:
            calls.append(name)
            return object()

        def __repr__(self) -> str:
            raise AssertionError("foreign object representation must not be evaluated")

    probe = probe_model_family(SimpleNamespace(model=Foreign()))

    assert calls == []
    assert probe.supported is False
    assert probe.signals == ()
    assert "Foreign" not in json.dumps(
        {
            "family": probe.family,
            "reason_code": probe.reason_code,
            "signals": list(probe.signals),
        }
    )


@pytest.mark.parametrize("descriptor_kind", ("property", "method"))
def test_probe_ignores_executable_model_descriptors(descriptor_kind: str) -> None:
    calls: list[str] = []

    if descriptor_kind == "property":

        class PropertyDescriptorModel:
            @property
            def model(self) -> object:
                calls.append("property")
                return _krea2_model()

        descriptor_model: object = PropertyDescriptorModel()
    else:

        class MethodDescriptorModel:
            def model(self) -> object:
                calls.append("method")
                return _krea2_model()

        descriptor_model = MethodDescriptorModel()

    probe = probe_model_family(descriptor_model)

    assert calls == []
    assert probe.reason_code == "model.family_unsupported"


@pytest.mark.parametrize(
    "model",
    (
        object(),
        SimpleNamespace(model=None),
        SimpleNamespace(model=SimpleNamespace(model_config=SimpleNamespace())),
        SimpleNamespace(model=SimpleNamespace(model_config=SimpleNamespace(unet_config=[]))),
    ),
)
def test_malformed_or_unsupported_models_fail_closed(model: object) -> None:
    with pytest.raises(ModelAwareScheduleError) as captured:
        build_model_aware_sigma_schedule(**_arguments(model=model))

    assert captured.value.reason_code in {
        "model.family_unsupported",
        "model.probe_invalid",
    }
    assert "C:\\" not in str(captured.value)


def test_auto_mode_exposes_family_only_ambiguity_without_returning_sigmas() -> None:
    with pytest.raises(ModelAwareScheduleError) as captured:
        build_model_aware_sigma_schedule(**_arguments(variant="Auto"))

    error = captured.value
    decision = json.loads(error.decision_json)
    assert error.reason_code == "model.identity_ambiguous"
    assert decision["schema"] == MODEL_AWARE_SIGMA_NODE_SCHEMA_ID
    assert decision["model_probe"]["family"] == "krea2"
    assert decision["variant_resolution"]["status"] == "ambiguous"
    assert decision["variant_resolution"]["reason_codes"] == [
        "insufficient_variant_evidence",
        "krea2_family_does_not_identify_variant",
        "model_class.krea2_family",
    ]
    assert "sigmas" not in decision
    assert "fallback" not in decision


@pytest.mark.parametrize(
    ("variant", "steps"),
    (("Turbo", 8), ("RAW", 52)),
)
def test_explicit_variant_uses_exact_profile_capability_gate_and_m4_01_schedule(
    variant: str,
    steps: int,
) -> None:
    result = build_model_aware_sigma_schedule(**_arguments(variant=variant, steps=steps))
    baseline = build_krea2_sigma_schedule(
        variant=variant,
        steps=steps,
        width=1024,
        height=1024,
        strict_official=True,
        start_step=0,
        end_step=-1,
    )
    info = _info(result)
    capability = info["capability_decision"]

    assert result.domain is SigmaDomain.UNIT_FLOW
    assert result.sigmas == baseline.sigmas
    assert result.selected_variant == variant.casefold()
    assert capability["level"] == CompatibilityLevel.ALLOW.value
    assert capability["reason_codes"] == ["core.compatible"]
    assert capability["model_identity"]["decisive_source"] == "explicit_selection"
    assert capability["model_identity"]["status"] == "confirmed"
    assert capability["profile_key"] == (
        "krea2.turbo.official@1" if variant == "Turbo" else "krea2.raw.official@1"
    )
    assert capability["profile_fingerprint"].startswith("sha256:")
    assert info["profile"]["evidence"] == "official"
    assert info["profile"]["key"] == capability["profile_key"]
    assert info["host_evidence"] == {
        "kind": "static_contract",
        "host_id": "comfyui",
        "host_version": "0.29.0",
        "host_revision": "e651b7bef55a5376343dcb1c0edb79f0142c985e",  # pragma: allowlist secret
    }
    assert info["schedule"] == json.loads(baseline.schedule_info_json)


def test_non_official_recipe_never_upgrades_profile_or_schedule_evidence() -> None:
    result = build_model_aware_sigma_schedule(
        **_arguments(variant="Turbo", steps=12, strict_official=False)
    )
    info = _info(result)

    assert info["profile"]["evidence"] == "official"
    assert info["schedule"]["profile"]["evidence"] == "modified"
    assert info["schedule"]["profile"]["recipe"] == "krea2.turbo.modified-12"


def test_model_aware_information_is_deterministic_and_result_is_immutable() -> None:
    first = build_model_aware_sigma_schedule(**_arguments())
    second = build_model_aware_sigma_schedule(**_arguments())

    assert first == second
    assert first.schedule_info_json == second.schedule_info_json
    with pytest.raises(FrozenInstanceError):
        first.sigmas = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    (
        {"schema_id": "other"},
        {"family": "other"},
        {"supported": cast(Any, 1)},
        {"family": None, "supported": True},
        {"signals": cast(Any, [])},
        {"signals": ("",)},
        {"signals": ("z", "a")},
        {"signals": ("a", "a")},
        {"reason_code": "invalid"},
        {"reason_code": cast(Any, 1)},
    ),
)
def test_model_family_probe_contract_rejects_invalid_values(
    changes: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "schema_id": "sigmax.model-family-probe/1",
        "family": "krea2",
        "supported": True,
        "signals": ("model_class:krea2",),
        "reason_code": "model.family_krea2",
    }
    arguments.update(changes)

    with pytest.raises(ScheduleContractError):
        ModelFamilyProbe(**cast(Any, arguments))


@pytest.mark.parametrize(
    "arguments",
    (
        {"reason_code": "invalid", "action": "fix", "decision_json": "{}"},
        {"reason_code": cast(Any, 1), "action": "fix", "decision_json": "{}"},
        {"reason_code": "model.test", "action": "", "decision_json": "{}"},
        {"reason_code": "model.test", "action": cast(Any, 1), "decision_json": "{}"},
        {"reason_code": "model.test", "action": "fix", "decision_json": ""},
        {"reason_code": "model.test", "action": "fix", "decision_json": cast(Any, 1)},
    ),
)
def test_model_aware_error_contract_rejects_invalid_values(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ScheduleContractError):
        ModelAwareScheduleError(**cast(Any, arguments))


def test_information_projection_rejects_non_json_values() -> None:
    with pytest.raises(ScheduleContractError, match="canonical JSON"):
        model_aware_module._canonical_json({"invalid": object()})


def test_rejected_capability_decision_stops_before_schedule_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_host = model_aware_module._static_host_capabilities()
    rejected_host = HostCapabilities(
        evidence_version=original_host.evidence_version,
        host_id=original_host.host_id,
        host_version=original_host.host_version,
        host_revision=original_host.host_revision,
        capabilities=(),
    )
    monkeypatch.setattr(
        model_aware_module,
        "_static_host_capabilities",
        lambda: rejected_host,
    )

    with pytest.raises(ModelAwareScheduleError) as captured:
        build_model_aware_sigma_schedule(**_arguments())

    decision = json.loads(captured.value.decision_json)
    assert captured.value.reason_code == "host.capability_missing"
    assert decision["capability_decision"]["level"] == "reject"
    assert decision["capability_decision"]["reason_codes"] == ["host.capability_missing"]


@pytest.mark.parametrize(
    "changes",
    (
        {"variant": "unknown"},
        {"variant": 1},
        {"model": None},
    ),
)
def test_invalid_public_inputs_fail_before_tensor_conversion(
    changes: dict[str, object],
) -> None:
    with pytest.raises((ModelAwareScheduleError, ScheduleContractError)):
        build_model_aware_sigma_schedule(**_arguments(**changes))


def test_runtime_node_converts_only_after_model_and_capability_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, ...]] = []

    def float_tensor(values: tuple[float, ...]) -> object:
        calls.append(tuple(values))
        return SimpleNamespace(values=tuple(values), device="cpu")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(FloatTensor=float_tensor))
    output = ModelAwareSigmaScheduler().build(**_arguments())

    tensor = cast(SimpleNamespace, output[0])
    assert tensor.device == "cpu"
    assert calls == [tensor.values]
    assert json.loads(output[1])["schema"] == MODEL_AWARE_SIGMA_NODE_SCHEMA_ID


def test_auto_rejection_occurs_before_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def forbidden_import(name: str) -> object:
        imports.append(name)
        raise AssertionError("Torch must not be imported for a rejected decision")

    monkeypatch.setattr(importlib, "import_module", forbidden_import)
    with pytest.raises(ModelAwareScheduleError):
        ModelAwareSigmaScheduler().build(**_arguments(variant="Auto"))

    assert imports == []


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
        monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace())

    with pytest.raises(RuntimeError, match="requires Torch FloatTensor"):
        ModelAwareSigmaScheduler().build(**_arguments())


@pytest.mark.parametrize(
    "changes",
    (
        {"selected_variant": "other"},
        {"domain": SigmaDomain.DISCRETE_TRAINING_INDEX},
        {"sigmas": cast(Any, [1.0, 0.0])},
        {"sigmas": (0.0,)},
        {"schedule_info_json": ""},
        {"schedule_info_json": cast(Any, 1)},
    ),
)
def test_model_aware_result_contract_rejects_invalid_values(
    changes: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "selected_variant": "turbo",
        "domain": SigmaDomain.UNIT_FLOW,
        "sigmas": (1.0, 0.0),
        "schedule_info_json": "{}",
    }
    arguments.update(changes)
    with pytest.raises(ScheduleContractError):
        ModelAwareSigmaNodeResult(**cast(Any, arguments))
