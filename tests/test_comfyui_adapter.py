"""M4-00 capability-aware ComfyUI adapter contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from comfyui_sigmax.adapters import (
    COMFYUI_ADAPTER_SCHEMA_ID,
    COMFYUI_ADAPTER_SCHEMA_VERSION,
    COMFYUI_HOST_WINDOW,
    ComfyAdapterCompatibilityError,
    ComfyAdapterReason,
    ComfyApiLifecycle,
    ComfyApiProbe,
    ComfyHostWindow,
    ComfyNodeDefinition,
    ComfyNodeInput,
    ComfyNodeOutput,
    ComfyNodeSchemaForm,
    adapt_comfyui_host,
    adapt_krea2_model_evidence,
    normalize_node_definition_v2,
    normalize_object_info,
    probe_comfy_api,
    require_stable_numbered_api,
)
from comfyui_sigmax.core import CompatibilityLevel, ExecutionFeatureRequest, ScheduleContractError
from comfyui_sigmax.profiles import (
    HostCapabilityLifecycle,
    ModelIdentityStatus,
    builtin_profile_registry,
    resolve_profile_capabilities,
)

PINNED_REVISION = "e651b7bef55a5376343dcb1c0edb79f0142c985e"  # pragma: allowlist secret


def _module(*, version: str = "0.0.2", stable: bool = False) -> Any:
    api = type("ComfyAPI", (), {"VERSION": version, "STABLE": stable})
    return SimpleNamespace(
        ComfyAPI=api,
        ComfyExtension=object(),
        io=object(),
        ui=object(),
    )


def _api_probe(*, version: str = "0.0.2", stable: bool = False) -> Any:
    module_name = (
        "comfy_api.latest" if version == "latest" else f"comfy_api.v{version.replace('.', '_')}"
    )
    return probe_comfy_api(_module(version=version, stable=stable), module_name=module_name)


def _system_stats(version: str = "0.29.0") -> dict[str, object]:
    return {"system": {"comfyui_version": version}}


def _object_info() -> dict[str, object]:
    return {
        "KSamplerSelect": {
            "input": {
                "required": {
                    "sampler_name": [["euler", "dpmpp_2m"], {"tooltip": "sampler"}],
                }
            },
            "output": ["SAMPLER"],
            "output_is_list": [False],
            "output_name": ["SAMPLER"],
            "name": "KSamplerSelect",
            "display_name": "KSampler Select",
            "description": "",
            "python_module": "comfy_extras.nodes_custom_sampler",
            "category": "model/sampling/samplers",
            "output_node": False,
        },
        "SamplerCustomAdvanced": {
            "input": {
                "required": {
                    "noise": ["NOISE"],
                    "guider": ["GUIDER"],
                    "sampler": ["SAMPLER"],
                    "sigmas": ["SIGMAS"],
                    "latent_image": ["LATENT"],
                }
            },
            "output": ["LATENT", "LATENT"],
            "output_is_list": [False, False],
            "output_name": ["output", "denoised_output"],
            "name": "SamplerCustomAdvanced",
            "display_name": "SamplerCustomAdvanced",
            "description": "",
            "python_module": "comfy_extras.nodes_custom_sampler",
            "category": "model/sampling/custom",
            "output_node": False,
        },
        "BasicScheduler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "scheduler": [["simple", "normal"]],
                    "steps": ["INT", {"default": 20}],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            },
            "output": ["SIGMAS"],
            "output_is_list": [False],
            "output_name": ["SIGMAS"],
            "name": "BasicScheduler",
            "display_name": "BasicScheduler",
            "description": "",
            "python_module": "comfy_extras.nodes_custom_sampler",
            "category": "model/sampling/schedulers",
            "output_node": False,
        },
        "ManualSigmas": {
            "input": {"required": {"sigmas": ["STRING"]}},
            "output": ["SIGMAS"],
            "output_is_list": [False],
            "output_name": ["SIGMAS"],
            "name": "ManualSigmas",
            "display_name": "ManualSigmas",
            "description": "",
            "python_module": "comfy_extras.nodes_custom_sampler",
            "category": "model/sampling/sigmas",
            "output_node": False,
            "experimental": True,
        },
    }


def _adapter(**overrides: Any) -> Any:
    arguments = {
        "api_probe": _api_probe(),
        "system_stats": _system_stats(),
        "features": {"supports_preview_metadata": True},
        "object_info": _object_info(),
        "host_revision": PINNED_REVISION,
    }
    arguments.update(overrides)
    return adapt_comfyui_host(**arguments)


def _profile(variant: str = "turbo") -> Any:
    return next(
        entry
        for entry in builtin_profile_registry().entries
        if entry.schema.model_variant == variant
    )


def test_probe_uses_only_numbered_api_public_symbols() -> None:
    probe = _api_probe()

    assert probe.module_name == "comfy_api.v0_0_2"
    assert probe.api_version == "0.0.2"
    assert probe.lifecycle is ComfyApiLifecycle.EXPERIMENTAL
    assert probe.public_symbols == ("ComfyExtension", "io", "ui")
    assert probe.is_numbered is True


def test_probe_preserves_latest_as_experimental_not_numbered() -> None:
    probe = _api_probe(version="latest")

    assert probe.is_numbered is False
    assert probe.lifecycle is ComfyApiLifecycle.EXPERIMENTAL


@pytest.mark.parametrize(
    ("module", "reason"),
    [
        (SimpleNamespace(), ComfyAdapterReason.API_PUBLIC_SURFACE_MISSING),
        (
            SimpleNamespace(
                ComfyAPI=type("ComfyAPI", (), {"VERSION": "0.0.2", "STABLE": "false"}),
                ComfyExtension=object(),
                io=object(),
                ui=object(),
            ),
            ComfyAdapterReason.API_MANIFEST_MALFORMED,
        ),
    ],
)
def test_probe_fails_actionably_for_missing_or_malformed_public_api(
    module: object,
    reason: ComfyAdapterReason,
) -> None:
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        probe_comfy_api(module, module_name="comfy_api.v0_0_2")

    assert captured.value.reason is reason
    assert captured.value.action


def test_stable_numbered_api_requirement_rejects_current_experimental_api() -> None:
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        require_stable_numbered_api(_api_probe())

    assert captured.value.reason is ComfyAdapterReason.API_EXPERIMENTAL
    assert "stable numbered Comfy API" in captured.value.action


def test_stable_numbered_api_requirement_accepts_future_explicit_stable_probe() -> None:
    probe = _api_probe(version="0.0.3", stable=True)

    assert require_stable_numbered_api(probe) is probe
    assert probe.lifecycle is ComfyApiLifecycle.LANDED


def test_stable_requirement_rejects_latest_even_if_manifest_claims_stable() -> None:
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        require_stable_numbered_api(_api_probe(version="latest", stable=True))

    assert captured.value.reason is ComfyAdapterReason.API_NOT_NUMBERED


def test_normalizes_legacy_and_v3_object_info_compatibility_projection() -> None:
    nodes = normalize_object_info(_object_info())

    assert tuple(node.node_id for node in nodes) == (
        "BasicScheduler",
        "KSamplerSelect",
        "ManualSigmas",
        "SamplerCustomAdvanced",
    )
    selector = next(node for node in nodes if node.node_id == "KSamplerSelect")
    assert selector.schema_form is ComfyNodeSchemaForm.OBJECT_INFO_V1
    assert selector.inputs[0].name == "sampler_name"
    assert selector.inputs[0].type_name == "COMBO"
    assert selector.inputs[0].options == ("euler", "dpmpp_2m")
    assert selector.experimental is False
    manual = next(node for node in nodes if node.node_id == "ManualSigmas")
    assert manual.experimental is True


def test_normalizes_documented_node_definition_v2() -> None:
    node = normalize_node_definition_v2(
        {
            "inputs": {
                "sampler_name": {
                    "name": "sampler_name",
                    "type": "COMBO",
                    "options": ["euler", "dpmpp_2m"],
                    "isOptional": False,
                }
            },
            "outputs": [
                {
                    "index": 0,
                    "name": "SAMPLER",
                    "type": "SAMPLER",
                    "is_list": False,
                }
            ],
            "name": "KSamplerSelect",
            "display_name": "KSampler Select",
            "description": "",
            "category": "model/sampling/samplers",
            "output_node": False,
            "python_module": "comfy_extras.nodes_custom_sampler",
            "deprecated": False,
            "experimental": False,
        }
    )

    assert node.schema_form is ComfyNodeSchemaForm.NODE_DEFINITION_V2
    assert node.inputs[0].options == ("euler", "dpmpp_2m")
    assert node.outputs[0].index == 0


def test_normalizes_explicit_v1_combo_metadata_and_autogrow_input() -> None:
    nodes = normalize_object_info(
        {
            "MiniMaxH3ReferenceToVideo": {
                "input": {
                    "required": {
                        "ref_image_size": [
                            "COMBO",
                            {"options": ["match", "max"]},
                        ],
                    },
                    "optional": {
                        "ref_images": [
                            "COMFY_AUTOGROW_V3",
                            {
                                "template": {"input": {"required": {"ref_image": ["IMAGE", {}]}}},
                                "prefix": "ref_image_",
                                "min": 0,
                                "max": 9,
                            },
                        ]
                    },
                },
                "output": ["CONDITIONING"],
                "output_is_list": [False],
                "output_name": ["positive"],
                "name": "MiniMaxH3ReferenceToVideo",
            }
        }
    )

    assert len(nodes) == 1
    node = nodes[0]
    assert node.inputs[0].name == "ref_image_size"
    assert node.inputs[0].type_name == "COMBO"
    assert node.inputs[0].options == ("match", "max")
    assert node.inputs[1].name == "ref_images"
    assert node.inputs[1].type_name == "COMFY_AUTOGROW_V3"
    assert node.inputs[1].options == ()


def test_normalizes_bare_v1_hidden_input_type() -> None:
    node = normalize_object_info(
        {
            "PreviewImage": {
                "input": {
                    "required": {"images": ["IMAGE"]},
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                },
                "output": ["IMAGE"],
                "name": "PreviewImage",
            }
        }
    )[0]

    assert tuple((item.name, item.type_name, item.section) for item in node.inputs) == (
        ("images", "IMAGE", "required"),
        ("extra_pnginfo", "EXTRA_PNGINFO", "hidden"),
        ("prompt", "PROMPT", "hidden"),
    )


def test_explicit_v1_combo_requires_metadata_options() -> None:
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        normalize_object_info(
            {
                "Node": {
                    "input": {"required": {"choice": ["COMBO", {}]}},
                    "output": [],
                    "name": "Node",
                }
            }
        )

    assert captured.value.reason is ComfyAdapterReason.NODE_SCHEMA_MALFORMED


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"inputs": {}, "input": {}, "outputs": []},
        {
            "inputs": {},
            "outputs": [{"index": 1, "name": "x", "type": "SIGMAS", "is_list": False}],
            "name": "Broken",
            "display_name": "Broken",
            "description": "",
            "category": "test",
            "output_node": False,
            "python_module": "nodes",
        },
    ],
)
def test_node_schema_rejects_missing_hybrid_or_noncanonical_forms(payload: object) -> None:
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        normalize_node_definition_v2(payload)

    assert captured.value.reason is ComfyAdapterReason.NODE_SCHEMA_MALFORMED


def test_adapter_derives_host_and_sampler_evidence_from_landed_nodes() -> None:
    evidence = _adapter()

    assert evidence.schema_id == COMFYUI_ADAPTER_SCHEMA_ID
    assert evidence.schema_version == COMFYUI_ADAPTER_SCHEMA_VERSION == "1"
    assert evidence.host_window is COMFYUI_HOST_WINDOW
    assert evidence.sampler_ids == ("comfy.euler",)
    assert tuple(
        (item.capability_id, item.lifecycle) for item in evidence.host_capabilities.capabilities
    ) == (
        ("api.comfy.v3", HostCapabilityLifecycle.EXPERIMENTAL),
        ("execution.partial_denoise", HostCapabilityLifecycle.LANDED),
        ("execution.per_token_timesteps", HostCapabilityLifecycle.UNSUPPORTED),
        ("sampler.comfy.euler", HostCapabilityLifecycle.LANDED),
        ("schedule.external_sigmas", HostCapabilityLifecycle.LANDED),
    )


def test_adapter_preserves_experimental_node_lifecycle() -> None:
    payload = _object_info()
    custom = dict(cast(dict[str, object], payload["SamplerCustomAdvanced"]))
    custom["experimental"] = True
    payload["SamplerCustomAdvanced"] = custom

    evidence = _adapter(object_info=payload)
    available = {
        item.capability_id: item.lifecycle for item in evidence.host_capabilities.capabilities
    }

    assert available["schedule.external_sigmas"] is HostCapabilityLifecycle.EXPERIMENTAL


def test_adapter_does_not_infer_euler_from_selector_node_id() -> None:
    payload = _object_info()
    selector = dict(cast(dict[str, object], payload["KSamplerSelect"]))
    selector["input"] = {"required": {"sampler_name": [["dpmpp_2m"]]}}
    payload["KSamplerSelect"] = selector

    evidence = _adapter(object_info=payload)
    available = {
        item.capability_id: item.lifecycle for item in evidence.host_capabilities.capabilities
    }

    assert evidence.sampler_ids == ()
    assert available["sampler.comfy.euler"] is HostCapabilityLifecycle.UNSUPPORTED


def test_adapter_host_evidence_drives_existing_profile_rejection() -> None:
    payload = _object_info()
    custom = dict(cast(dict[str, object], payload["SamplerCustomAdvanced"]))
    custom["experimental"] = True
    payload["SamplerCustomAdvanced"] = custom
    profile = _profile()
    model = adapt_krea2_model_evidence(
        registered_profile=profile,
        explicit_variant="turbo",
    )
    evidence = _adapter(object_info=payload)

    decision = resolve_profile_capabilities(
        registered_profile=profile,
        model=model,
        host=evidence.host_capabilities,
        sampler=profile.schema.reference_sampler_capabilities,
        request=ExecutionFeatureRequest(),
    )

    assert decision.level is CompatibilityLevel.REJECT
    assert decision.reason_codes == ("host.capability_experimental",)


def test_adapter_reuses_variant_trust_boundary_for_model_evidence() -> None:
    profile = _profile()

    confirmed = adapt_krea2_model_evidence(
        registered_profile=profile,
        explicit_variant="turbo",
    )
    suggested = adapt_krea2_model_evidence(
        registered_profile=profile,
        filename="local-krea2-turbo.safetensors",
    )

    assert confirmed.identity.status is ModelIdentityStatus.CONFIRMED
    assert suggested.identity.status is ModelIdentityStatus.SUGGESTED
    assert confirmed.capabilities is profile.schema.model_capabilities


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"system_stats": {"system": {}}}, ComfyAdapterReason.SYSTEM_STATS_MALFORMED),
        ({"features": []}, ComfyAdapterReason.FEATURES_MALFORMED),
        (
            {"system_stats": _system_stats("0.30.0")},
            ComfyAdapterReason.HOST_OUTSIDE_TESTED_WINDOW,
        ),
        (
            {"host_revision": "0" * 40},
            ComfyAdapterReason.HOST_OUTSIDE_TESTED_WINDOW,
        ),
    ],
)
def test_adapter_rejects_malformed_or_untested_host_evidence(
    overrides: dict[str, object],
    reason: ComfyAdapterReason,
) -> None:
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        _adapter(**overrides)

    assert captured.value.reason is reason
    assert captured.value.action


def test_adapter_errors_do_not_echo_untrusted_private_text() -> None:
    payload = _object_info()
    payload["Private"] = {
        "input": {"required": {"C:\\private\\model.ckpt": [object()]}},
        "output": [],
        "name": "Private",
    }

    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        _adapter(object_info=payload)

    message = str(captured.value)
    assert "C:\\private" not in message
    assert "model.ckpt" not in message


def test_adapter_contracts_are_immutable() -> None:
    evidence = _adapter()

    with pytest.raises(FrozenInstanceError):
        replace(evidence.api_probe, api_version="0.0.3").api_version = "0.0.4"
    with pytest.raises(FrozenInstanceError):
        evidence.host_window.minimum_version = "0.0.0"


def test_host_window_is_exact_and_public() -> None:
    assert COMFYUI_HOST_WINDOW.minimum_version == "0.29.0"
    assert COMFYUI_HOST_WINDOW.maximum_version == "0.29.0"
    assert COMFYUI_HOST_WINDOW.tested_revisions == (PINNED_REVISION,)
    assert COMFYUI_HOST_WINDOW.validation_level == "static_contract"


def test_wrong_adapter_argument_types_fail_closed() -> None:
    with pytest.raises(ScheduleContractError):
        adapt_comfyui_host(
            api_probe=cast(Any, object()),
            system_stats=_system_stats(),
            features={},
            object_info={},
            host_revision=PINNED_REVISION,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"reason": cast(Any, "bad"), "action": "fix it"},
        {"reason": ComfyAdapterReason.API_UNSUPPORTED, "action": ""},
        {"reason": ComfyAdapterReason.API_UNSUPPORTED, "action": cast(Any, None)},
    ],
)
def test_adapter_error_contract_rejects_invalid_construction(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ScheduleContractError):
        ComfyAdapterCompatibilityError(**cast(Any, arguments))


@pytest.mark.parametrize("value", [None, "", "x" * 257, "line\nbreak"])
def test_public_text_fields_are_bounded(value: object) -> None:
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        ComfyNodeInput(
            name=cast(Any, value),
            type_name="MODEL",
            section="required",
            optional=False,
        )

    assert captured.value.reason is ComfyAdapterReason.NODE_SCHEMA_MALFORMED


@pytest.mark.parametrize(
    "changes",
    [
        {"module_name": cast(Any, None)},
        {"module_name": ""},
        {"api_version": cast(Any, None)},
        {"api_version": ""},
        {"is_numbered": cast(Any, 1)},
        {"lifecycle": cast(Any, "landed")},
        {"public_symbols": ("io",)},
        {"is_numbered": False},
    ],
)
def test_api_probe_contract_rejects_inconsistent_values(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "module_name": "comfy_api.v0_0_2",
        "api_version": "0.0.2",
        "is_numbered": True,
        "lifecycle": ComfyApiLifecycle.EXPERIMENTAL,
        "public_symbols": ("ComfyExtension", "io", "ui"),
    }
    arguments.update(changes)

    with pytest.raises(ScheduleContractError):
        ComfyApiProbe(**cast(Any, arguments))


@pytest.mark.parametrize(
    "changes",
    [
        {"minimum_version": "bad"},
        {"maximum_version": "bad"},
        {"tested_revisions": ()},
        {"tested_revisions": ("not-a-revision",)},
        {"tested_revisions": (PINNED_REVISION, PINNED_REVISION)},
        {"validation_level": "live"},
    ],
)
def test_host_window_contract_rejects_invalid_values(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "minimum_version": "0.29.0",
        "maximum_version": "0.29.0",
        "tested_revisions": (PINNED_REVISION,),
        "validation_level": "static_contract",
    }
    arguments.update(changes)

    with pytest.raises(ScheduleContractError):
        ComfyHostWindow(**cast(Any, arguments))


@pytest.mark.parametrize(
    "changes",
    [
        {"section": "dynamic"},
        {"optional": cast(Any, 1)},
        {"options": cast(Any, ["euler"])},
        {"options": tuple(f"s{index}" for index in range(513)), "type_name": "COMBO"},
        {"options": ("euler", "euler"), "type_name": "COMBO"},
        {"options": ("euler",), "type_name": "SAMPLER"},
        {"options": (), "type_name": "COMBO"},
        {"options": ("bad\noption",), "type_name": "COMBO"},
    ],
)
def test_node_input_contract_rejects_invalid_values(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "name": "sampler_name",
        "type_name": "SAMPLER",
        "section": "required",
        "optional": False,
        "options": (),
    }
    arguments.update(changes)

    with pytest.raises((ScheduleContractError, ComfyAdapterCompatibilityError)):
        ComfyNodeInput(**cast(Any, arguments))


@pytest.mark.parametrize(
    "changes",
    [
        {"index": -1},
        {"index": cast(Any, True)},
        {"index": cast(Any, "0")},
        {"name": ""},
        {"type_name": ""},
        {"is_list": cast(Any, 0)},
    ],
)
def test_node_output_contract_rejects_invalid_values(changes: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "index": 0,
        "name": "SIGMAS",
        "type_name": "SIGMAS",
        "is_list": False,
    }
    arguments.update(changes)

    with pytest.raises((ScheduleContractError, ComfyAdapterCompatibilityError)):
        ComfyNodeOutput(**cast(Any, arguments))


def _canonical_node() -> ComfyNodeDefinition:
    return ComfyNodeDefinition(
        node_id="Node",
        schema_form=ComfyNodeSchemaForm.OBJECT_INFO_V1,
        inputs=(
            ComfyNodeInput(
                name="model",
                type_name="MODEL",
                section="required",
                optional=False,
            ),
        ),
        outputs=(ComfyNodeOutput(index=0, name="SIGMAS", type_name="SIGMAS", is_list=False),),
        deprecated=False,
        experimental=False,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_form": cast(Any, "object_info_v1")},
        {"inputs": cast(Any, [])},
        {"inputs": cast(Any, (object(),))},
        {"outputs": cast(Any, [])},
        {"outputs": cast(Any, (object(),))},
        {
            "inputs": tuple(
                ComfyNodeInput(
                    name=f"input_{index}",
                    type_name="MODEL",
                    section="required",
                    optional=False,
                )
                for index in range(257)
            )
        },
        {
            "outputs": tuple(
                ComfyNodeOutput(
                    index=index,
                    name=f"output_{index}",
                    type_name="SIGMAS",
                    is_list=False,
                )
                for index in range(65)
            )
        },
        {
            "inputs": (
                ComfyNodeInput(
                    name="model",
                    type_name="MODEL",
                    section="required",
                    optional=False,
                ),
                ComfyNodeInput(
                    name="model",
                    type_name="MODEL",
                    section="optional",
                    optional=True,
                ),
            )
        },
        {
            "inputs": (
                ComfyNodeInput(
                    name="z",
                    type_name="MODEL",
                    section="required",
                    optional=False,
                ),
                ComfyNodeInput(
                    name="a",
                    type_name="MODEL",
                    section="required",
                    optional=False,
                ),
            )
        },
        {"outputs": (ComfyNodeOutput(index=1, name="SIGMAS", type_name="SIGMAS", is_list=False),)},
        {"deprecated": cast(Any, 0)},
        {"experimental": cast(Any, 0)},
    ],
)
def test_node_definition_contract_rejects_invalid_values(
    changes: dict[str, object],
) -> None:
    node = _canonical_node()

    with pytest.raises(ScheduleContractError):
        replace(node, **cast(Any, changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_id": "other"},
        {"schema_version": "2"},
        {"api_probe": cast(Any, object())},
        {"host_window": cast(Any, object())},
        {"feature_ids": ("z", "a")},
        {"feature_ids": ("a", "a")},
        {"nodes": (_canonical_node(), replace(_canonical_node(), node_id="A"))},
        {"sampler_ids": ("z", "a")},
        {"sampler_ids": ("a", "a")},
        {"host_capabilities": cast(Any, object())},
    ],
)
def test_adapter_evidence_contract_rejects_invalid_values(
    changes: dict[str, object],
) -> None:
    evidence = _adapter()

    with pytest.raises(ScheduleContractError):
        replace(evidence, **cast(Any, changes))


@pytest.mark.parametrize(
    ("module_name", "module"),
    [
        (cast(Any, None), _module()),
        ("comfy_api.private", _module()),
        ("comfy_api.latest", _module(version="0.0.2")),
        ("comfy_api.v0_0_2", _module(version="latest")),
        (
            "comfy_api.v0_0_2",
            SimpleNamespace(
                ComfyAPI=type("ComfyAPI", (), {"VERSION": 2, "STABLE": False}),
                ComfyExtension=object(),
                io=object(),
                ui=object(),
            ),
        ),
        (
            "comfy_api.v0_0_2",
            SimpleNamespace(
                ComfyAPI=type("ComfyAPI", (), {"VERSION": "0.0.2", "STABLE": False}),
                ComfyExtension=object(),
                io=object(),
            ),
        ),
    ],
)
def test_api_probe_rejects_all_malformed_manifest_paths(
    module_name: object,
    module: object,
) -> None:
    with pytest.raises(ComfyAdapterCompatibilityError):
        probe_comfy_api(module, module_name=cast(Any, module_name))


def test_stable_requirement_rejects_wrong_type_and_unsupported_lifecycle() -> None:
    with pytest.raises(ScheduleContractError):
        require_stable_numbered_api(cast(Any, object()))

    unsupported = replace(_api_probe(), lifecycle=ComfyApiLifecycle.UNSUPPORTED)
    with pytest.raises(ComfyAdapterCompatibilityError) as captured:
        require_stable_numbered_api(unsupported)
    assert captured.value.reason is ComfyAdapterReason.API_UNSUPPORTED


def test_v1_parser_supports_optional_hidden_defaults_and_lifecycle() -> None:
    payload = {
        "Node": {
            "input": {
                "required": {"model": ["MODEL"]},
                "optional": {"mask": ["MASK"]},
                "hidden": {"prompt": ["PROMPT"]},
            },
            "output": ["SIGMAS"],
            "name": "Node",
            "deprecated": True,
        }
    }

    node = normalize_object_info(payload)[0]

    assert tuple((item.name, item.section, item.optional) for item in node.inputs) == (
        ("model", "required", False),
        ("mask", "optional", True),
        ("prompt", "hidden", False),
    )
    assert node.outputs[0].name == "SIGMAS"
    assert node.deprecated is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda node: {"Wrong": node},
        lambda node: {"Node": {**node, "input": []}},
        lambda node: {"Node": {**node, "input": {"required": []}}},
        lambda node: {"Node": {**node, "input": {"required": {"x": []}}}},
        lambda node: {"Node": {**node, "input": {"required": {"x": [[]]}}}},
        lambda node: {"Node": {**node, "output": "SIGMAS"}},
        lambda node: {"Node": {**node, "output_name": []}},
        lambda node: {"Node": {**node, "output_is_list": []}},
        lambda node: {"Node": {**node, "output_is_list": [0]}},
        lambda node: {"Node": {**node, "experimental": "true"}},
    ],
)
def test_v1_parser_rejects_malformed_variants(mutation: Any) -> None:
    node = {
        "input": {"required": {"x": ["MODEL"]}},
        "output": ["SIGMAS"],
        "name": "Node",
    }
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info(mutation(node))


def test_v1_parser_enforces_section_total_output_and_option_limits() -> None:
    base = {"output": [], "name": "Node"}
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info(
            {
                "Node": {
                    **base,
                    "input": {"required": {f"x{index}": ["MODEL"] for index in range(257)}},
                }
            }
        )
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info(
            {
                "Node": {
                    **base,
                    "input": {"required": {"choice": [["duplicate", "duplicate"]]}},
                }
            }
        )
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info(
            {
                "Node": {
                    **base,
                    "input": {
                        "required": {f"r{index}": ["MODEL"] for index in range(129)},
                        "optional": {f"o{index}": ["MODEL"] for index in range(129)},
                    },
                }
            }
        )
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info(
            {
                "Node": {
                    "input": {},
                    "output": ["SIGMAS"] * 65,
                    "name": "Node",
                }
            }
        )
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info(
            {
                "Node": {
                    **base,
                    "input": {"required": {"choice": [[f"x{index}" for index in range(513)]]}},
                }
            }
        )


def _v2_node() -> dict[str, object]:
    return {
        "inputs": {},
        "outputs": [],
        "name": "Node",
        "display_name": "Node",
        "description": "",
        "category": "test",
        "output_node": False,
        "python_module": "nodes",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda node: {**node, "inputs": []},
        lambda node: {
            **node,
            "inputs": {"wrong": {"name": "right", "type": "MODEL"}},
        },
        lambda node: {**node, "inputs": {"x": []}},
        lambda node: {
            **node,
            "inputs": {"x": {"name": "x", "type": "MODEL", "isOptional": 1}},
        },
        lambda node: {
            **node,
            "inputs": {"x": {"name": "x", "type": "COMBO"}},
        },
        lambda node: {**node, "outputs": "SIGMAS"},
        lambda node: {**node, "outputs": [[]]},
        lambda node: {
            **node,
            "outputs": [{"index": 0, "name": "x", "type": "SIGMAS", "is_list": 0}],
        },
        lambda node: {**node, "experimental": "true"},
    ],
)
def test_v2_parser_rejects_malformed_variants(mutation: Any) -> None:
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_node_definition_v2(mutation(_v2_node()))


def test_v2_parser_supports_optional_input_and_object_info_wrapper() -> None:
    node = _v2_node()
    node["inputs"] = {
        "mask": {"name": "mask", "type": "MASK", "isOptional": True},
    }

    direct = normalize_node_definition_v2(node)
    wrapped = normalize_object_info({"Node": node})[0]

    assert direct.inputs[0].section == "optional"
    assert direct == wrapped


def test_v2_parser_enforces_input_output_and_object_info_limits() -> None:
    node = _v2_node()
    node["inputs"] = {f"x{index}": {"name": f"x{index}", "type": "MODEL"} for index in range(257)}
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_node_definition_v2(node)

    node = _v2_node()
    node["outputs"] = [
        {"index": index, "name": f"x{index}", "type": "SIGMAS", "is_list": False}
        for index in range(65)
    ]
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_node_definition_v2(node)

    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info({f"Node{index}": {} for index in range(4097)})


@pytest.mark.parametrize("payload", [[], {"Node": []}, {"Node": {}}, {1: {}}])
def test_object_info_rejects_nonmapping_or_unsupported_entries(payload: object) -> None:
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info(payload)


def test_object_info_rejects_v2_key_name_mismatch() -> None:
    with pytest.raises(ComfyAdapterCompatibilityError):
        normalize_object_info({"Wrong": _v2_node()})


def test_adapter_feature_and_system_stats_limits_fail_closed() -> None:
    with pytest.raises(ComfyAdapterCompatibilityError):
        _adapter(features={f"feature_{index}": True for index in range(257)})
    with pytest.raises(ComfyAdapterCompatibilityError):
        _adapter(features={1: True})
    with pytest.raises(ComfyAdapterCompatibilityError):
        _adapter(system_stats=[])
    with pytest.raises(ComfyAdapterCompatibilityError):
        _adapter(system_stats={"system": []})
    with pytest.raises(ComfyAdapterCompatibilityError):
        _adapter(system_stats={"system": {"comfyui_version": 29}})
    with pytest.raises(ComfyAdapterCompatibilityError):
        _adapter(host_revision=cast(Any, None))
    with pytest.raises(ComfyAdapterCompatibilityError):
        _adapter(host_revision="bad")


def test_adapter_derives_split_lifecycle_and_deprecated_capabilities() -> None:
    split = {
        "input": {"required": {"sigmas": ["SIGMAS"], "step": ["INT"]}},
        "output": ["SIGMAS", "SIGMAS"],
        "name": "SplitSigmas",
        "deprecated": True,
    }
    custom = {
        "input": {"required": {"sigmas": ["SIGMAS"]}},
        "output": [],
        "name": "SamplerCustom",
        "deprecated": True,
    }
    selector = {
        "input": {"required": {"sampler_name": [["euler"]]}},
        "output": ["SAMPLER"],
        "name": "KSamplerSelect",
        "deprecated": True,
    }

    evidence = _adapter(
        object_info={
            "SplitSigmas": split,
            "SamplerCustom": custom,
            "KSamplerSelect": selector,
        }
    )
    available = {
        item.capability_id: item.lifecycle for item in evidence.host_capabilities.capabilities
    }

    assert available["execution.partial_denoise"] is HostCapabilityLifecycle.UNSUPPORTED
    assert available["schedule.external_sigmas"] is HostCapabilityLifecycle.UNSUPPORTED
    assert available["sampler.comfy.euler"] is HostCapabilityLifecycle.UNSUPPORTED


def test_adapter_uses_landed_candidate_over_experimental_candidate() -> None:
    payload = _object_info()
    advanced = dict(cast(dict[str, object], payload["SamplerCustomAdvanced"]))
    advanced["experimental"] = True
    payload["SamplerCustomAdvanced"] = advanced
    payload["SamplerCustom"] = {
        "input": {"required": {"sigmas": ["SIGMAS"]}},
        "output": [],
        "name": "SamplerCustom",
    }

    evidence = _adapter(object_info=payload)
    available = {
        item.capability_id: item.lifecycle for item in evidence.host_capabilities.capabilities
    }

    assert available["schedule.external_sigmas"] is HostCapabilityLifecycle.LANDED


def test_adapter_detects_split_denoise_and_missing_selector_input() -> None:
    split = {
        "input": {"required": {"sigmas": ["SIGMAS"], "denoise": ["FLOAT"]}},
        "output": ["SIGMAS", "SIGMAS"],
        "name": "SplitSigmasDenoise",
        "experimental": True,
    }
    selector = {
        "input": {"required": {}},
        "output": ["SAMPLER"],
        "name": "KSamplerSelect",
    }

    evidence = _adapter(object_info={"SplitSigmasDenoise": split, "KSamplerSelect": selector})
    available = {
        item.capability_id: item.lifecycle for item in evidence.host_capabilities.capabilities
    }

    assert available["execution.partial_denoise"] is HostCapabilityLifecycle.EXPERIMENTAL
    assert available["sampler.comfy.euler"] is HostCapabilityLifecycle.UNSUPPORTED


def test_adapter_ignores_custom_sampler_without_exact_sigmas_input() -> None:
    missing = {
        "input": {"required": {"model": ["MODEL"]}},
        "output": [],
        "name": "SamplerCustom",
    }
    wrong_type = {
        "input": {"required": {"sigmas": ["FLOAT"]}},
        "output": [],
        "name": "SamplerCustomAdvanced",
    }

    evidence = _adapter(
        object_info={
            "SamplerCustom": missing,
            "SamplerCustomAdvanced": wrong_type,
        }
    )
    available = {
        item.capability_id: item.lifecycle for item in evidence.host_capabilities.capabilities
    }

    assert available["schedule.external_sigmas"] is HostCapabilityLifecycle.UNSUPPORTED


def test_adapter_projects_landed_and_unsupported_api_lifecycle() -> None:
    landed = _adapter(api_probe=_api_probe(version="0.0.3", stable=True))
    unsupported_probe = replace(_api_probe(), lifecycle=ComfyApiLifecycle.UNSUPPORTED)
    unsupported = _adapter(api_probe=unsupported_probe)

    assert landed.host_capabilities.capabilities[0].lifecycle is HostCapabilityLifecycle.LANDED
    assert (
        unsupported.host_capabilities.capabilities[0].lifecycle
        is HostCapabilityLifecycle.UNSUPPORTED
    )


def test_model_adapter_rejects_wrong_profile_type() -> None:
    with pytest.raises(ScheduleContractError):
        adapt_krea2_model_evidence(registered_profile=cast(Any, object()))
