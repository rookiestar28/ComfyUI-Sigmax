"""M4-06 namespaced registration and schema-discovery contracts."""

from __future__ import annotations

import ast
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from comfyui_sigmax.adapters import (
    ComfyApiLifecycle,
    ComfyApiProbe,
    ComfyNodeSchemaForm,
)
from comfyui_sigmax.adapters.registration import (
    NODE_REGISTRATION_SCHEMA_ID,
    NODE_REGISTRATION_SCHEMA_VERSION,
    SIGMAX_NODE_MODULE,
    NodeDefinitionKind,
    NodeRegistration,
    NodeRegistrationError,
    NodeRegistrationReason,
    NodeRegistry,
    builtin_node_registry,
    discover_legacy_registration,
    discover_v3_registration,
    registration_from_node_definition_v2,
    require_registration_compatible,
)
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles import HostCapabilityLifecycle

ROOT = Path(__file__).resolve().parents[1]


class LegacyNode:
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    OUTPUT_IS_LIST = (False,)
    CATEGORY = "Sigmax/testing"
    DESCRIPTION = "Legacy test node"
    OUTPUT_NODE = False
    DEPRECATED = False
    EXPERIMENTAL = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["simple", "normal"],),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0}),
            },
        }


def _v3_info(
    *,
    node_id: str = "Sigmax.V3Fixture",
    experimental: bool = True,
) -> dict[str, object]:
    return {
        "input": {
            "required": {
                "model": ["MODEL"],
                "sigmas": ["SIGMAS"],
            }
        },
        "output": ["SIGMAS"],
        "output_is_list": [False],
        "output_name": ["sigmas"],
        "name": node_id,
        "display_name": "Sigmax V3 Fixture",
        "description": "V3 test node",
        "python_module": "custom_nodes.any_directory",
        "category": "Sigmax/testing",
        "output_node": False,
        "deprecated": False,
        "experimental": experimental,
    }


class V3Node:
    @classmethod
    def GET_SCHEMA(cls) -> object:
        return SimpleNamespace(
            node_id="Sigmax.V3Fixture",
            display_name="Sigmax V3 Fixture",
            description="V3 test node",
            category="Sigmax/testing",
            is_output_node=False,
            is_deprecated=False,
            is_experimental=True,
        )

    @classmethod
    def GET_NODE_INFO_V1(cls) -> dict[str, object]:
        return _v3_info()


def _v2_payload(
    *,
    node_id: str = "Sigmax.V2Fixture",
    experimental: bool = False,
) -> dict[str, object]:
    return {
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
                "name": "sampler",
                "type": "SAMPLER",
                "is_list": False,
            }
        ],
        "name": node_id,
        "display_name": "Sigmax V2 Fixture",
        "description": "V2 test node",
        "category": "Sigmax/testing",
        "output_node": False,
        "python_module": "comfyui_sigmax.nodes",
        "deprecated": False,
        "experimental": experimental,
    }


def _probe(*, stable: bool = False) -> ComfyApiProbe:
    return ComfyApiProbe(
        module_name="comfy_api.v0_0_2" if not stable else "comfy_api.v0_0_3",
        api_version="0.0.2" if not stable else "0.0.3",
        is_numbered=True,
        lifecycle=ComfyApiLifecycle.EXPERIMENTAL if not stable else ComfyApiLifecycle.LANDED,
        public_symbols=("ComfyExtension", "io", "ui"),
    )


def _legacy() -> NodeRegistration:
    return discover_legacy_registration(
        node_id="Sigmax.LegacyFixture",
        display_name="Sigmax Legacy Fixture",
        node_class=LegacyNode,
    )


def _v3() -> NodeRegistration:
    return discover_v3_registration(V3Node)


def _v2() -> NodeRegistration:
    return registration_from_node_definition_v2(
        node_class=type("V2Node", (), {}),
        payload=_v2_payload(),
    )


def test_legacy_discovery_uses_explicit_namespaced_identity() -> None:
    registration = _legacy()

    assert registration.schema_id == NODE_REGISTRATION_SCHEMA_ID
    assert registration.schema_version == NODE_REGISTRATION_SCHEMA_VERSION == "1"
    assert registration.node_id == "Sigmax.LegacyFixture"
    assert registration.display_name == "Sigmax Legacy Fixture"
    assert registration.definition_kind is NodeDefinitionKind.LEGACY_V1
    assert registration.definition.schema_form is ComfyNodeSchemaForm.OBJECT_INFO_V1
    assert registration.definition.node_id == registration.node_id
    assert registration.python_module == SIGMAX_NODE_MODULE
    assert registration.lifecycle is HostCapabilityLifecycle.LANDED


def test_v3_discovery_requires_public_schema_and_compatibility_projection() -> None:
    registration = _v3()

    assert registration.node_id == "Sigmax.V3Fixture"
    assert registration.definition_kind is NodeDefinitionKind.COMFY_V3
    assert registration.definition.schema_form is ComfyNodeSchemaForm.OBJECT_INFO_V1
    assert registration.lifecycle is HostCapabilityLifecycle.EXPERIMENTAL
    assert tuple(item.name for item in registration.definition.inputs) == ("model", "sigmas")


def test_node_definition_v2_discovery_preserves_wire_form_and_lifecycle() -> None:
    registration = _v2()

    assert registration.node_id == "Sigmax.V2Fixture"
    assert registration.definition_kind is NodeDefinitionKind.NODE_DEFINITION_V2
    assert registration.definition.schema_form is ComfyNodeSchemaForm.NODE_DEFINITION_V2
    assert registration.lifecycle is HostCapabilityLifecycle.LANDED


def test_registry_is_copy_on_write_canonical_and_idempotent() -> None:
    empty = NodeRegistry.empty()
    first = empty.register(_v3()).register(_legacy())
    repeated = first.register(_legacy())

    assert empty.entries == ()
    assert tuple(item.node_id for item in first.entries) == (
        "Sigmax.LegacyFixture",
        "Sigmax.V3Fixture",
    )
    assert repeated is first


def test_register_many_accepts_mixed_public_definition_forms() -> None:
    registry = NodeRegistry.empty().register_many((_v3(), _v2(), _legacy()))

    assert tuple(item.definition_kind for item in registry.entries) == (
        NodeDefinitionKind.LEGACY_V1,
        NodeDefinitionKind.NODE_DEFINITION_V2,
        NodeDefinitionKind.COMFY_V3,
    )


def test_collision_never_overwrites_unrelated_class_or_schema() -> None:
    registry = NodeRegistry.empty().register(_legacy())
    different_class = discover_legacy_registration(
        node_id="Sigmax.LegacyFixture",
        display_name="Sigmax Legacy Fixture",
        node_class=type(
            "OtherLegacy",
            (),
            {
                "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
                "RETURN_TYPES": (),
            },
        ),
    )

    with pytest.raises(NodeRegistrationError) as captured:
        registry.register(different_class)

    assert captured.value.reason is NodeRegistrationReason.NODE_ID_COLLISION
    assert registry.class_mappings()["Sigmax.LegacyFixture"] is LegacyNode


def test_mapping_and_display_projections_are_fresh_and_exact() -> None:
    registry = NodeRegistry.empty().register_many((_legacy(), _v3()))

    classes = registry.class_mappings()
    displays = registry.display_name_mappings()
    classes.clear()
    displays.clear()

    assert registry.class_mappings() == {
        "Sigmax.LegacyFixture": LegacyNode,
        "Sigmax.V3Fixture": V3Node,
    }
    assert registry.display_name_mappings() == {
        "Sigmax.LegacyFixture": "Sigmax Legacy Fixture",
        "Sigmax.V3Fixture": "Sigmax V3 Fixture",
    }


def test_object_info_projection_round_trips_through_m4_adapter_shape() -> None:
    registry = NodeRegistry.empty().register_many((_legacy(), _v3(), _v2()))
    wire = registry.object_info_projection()

    assert tuple(wire) == (
        "Sigmax.LegacyFixture",
        "Sigmax.V2Fixture",
        "Sigmax.V3Fixture",
    )
    assert wire["Sigmax.LegacyFixture"]["input"]["required"]["mode"][0] == [
        "simple",
        "normal",
    ]
    assert wire["Sigmax.V3Fixture"]["experimental"] is True
    assert wire["Sigmax.V2Fixture"]["experimental"] is False


def test_node_definition_v2_projection_is_documented_and_deterministic() -> None:
    registry = NodeRegistry.empty().register_many((_legacy(), _v3(), _v2()))
    wire = registry.node_definition_v2_projection()

    assert tuple(wire) == (
        "Sigmax.LegacyFixture",
        "Sigmax.V2Fixture",
        "Sigmax.V3Fixture",
    )
    assert wire["Sigmax.LegacyFixture"]["inputs"]["strength"]["isOptional"] is True
    assert wire["Sigmax.V2Fixture"] == _v2_payload()
    assert wire["Sigmax.V3Fixture"]["experimental"] is True


def test_projection_mutation_does_not_change_registry() -> None:
    registry = NodeRegistry.empty().register(_legacy())
    first = registry.object_info_projection()
    first["Sigmax.LegacyFixture"]["input"]["required"].clear()

    second = registry.object_info_projection()
    assert tuple(second["Sigmax.LegacyFixture"]["input"]["required"]) == ("mode", "model")


def test_node_id_never_depends_on_installation_directory_or_module_name() -> None:
    original_module = LegacyNode.__module__
    try:
        LegacyNode.__module__ = "custom_nodes.renamed_directory.nodes"
        registration = _legacy()
    finally:
        LegacyNode.__module__ = original_module

    assert registration.node_id == "Sigmax.LegacyFixture"
    assert registration.python_module == "comfyui_sigmax.nodes"


@pytest.mark.parametrize(
    "node_id",
    [
        "LegacyFixture",
        "sigmax.LegacyFixture",
        "Sigmax/LegacyFixture",
        "Sigmax.legacy_fixture",
        "Sigmax.",
        "Sigmax." + ("A" * 65),
    ],
)
def test_non_namespaced_or_unstable_node_ids_reject(node_id: str) -> None:
    with pytest.raises(NodeRegistrationError) as captured:
        discover_legacy_registration(
            node_id=node_id,
            display_name="Fixture",
            node_class=LegacyNode,
        )

    assert captured.value.reason is NodeRegistrationReason.INVALID_NODE_ID


def test_current_experimental_api_allows_discovery_but_rejects_v3_activation() -> None:
    registration = _v3()

    with pytest.raises(NodeRegistrationError) as captured:
        require_registration_compatible(registration, _probe())

    assert captured.value.reason is NodeRegistrationReason.V3_API_NOT_STABLE
    assert "stable numbered Comfy API" in captured.value.action


def test_stable_api_allows_v3_and_legacy_does_not_require_v3() -> None:
    assert require_registration_compatible(_v3(), _probe(stable=True)) == _v3()
    assert require_registration_compatible(_legacy(), _probe()) == _legacy()


def test_v3_schema_and_object_info_identity_disagreement_rejects() -> None:
    class Mismatch(V3Node):
        @classmethod
        def GET_NODE_INFO_V1(cls) -> dict[str, object]:
            return _v3_info(node_id="Sigmax.Other")

    with pytest.raises(NodeRegistrationError) as captured:
        discover_v3_registration(Mismatch)

    assert captured.value.reason is NodeRegistrationReason.SCHEMA_ID_MISMATCH


def test_v3_lifecycle_disagreement_rejects() -> None:
    class Mismatch(V3Node):
        @classmethod
        def GET_NODE_INFO_V1(cls) -> dict[str, object]:
            return _v3_info(experimental=False)

    with pytest.raises(NodeRegistrationError) as captured:
        discover_v3_registration(Mismatch)

    assert captured.value.reason is NodeRegistrationReason.SCHEMA_LIFECYCLE_MISMATCH


@pytest.mark.parametrize(
    "node_class",
    [
        object(),
        type("MissingInputTypes", (), {"RETURN_TYPES": ()}),
        type(
            "MissingReturns",
            (),
            {"INPUT_TYPES": classmethod(lambda cls: {"required": {}})},
        ),
        type(
            "BrokenInputs",
            (),
            {
                "INPUT_TYPES": classmethod(lambda cls: []),
                "RETURN_TYPES": (),
            },
        ),
    ],
)
def test_legacy_discovery_fails_actionably_for_invalid_public_class(
    node_class: object,
) -> None:
    with pytest.raises(NodeRegistrationError) as captured:
        discover_legacy_registration(
            node_id="Sigmax.Broken",
            display_name="Broken",
            node_class=cast(Any, node_class),
        )

    assert captured.value.reason in {
        NodeRegistrationReason.INVALID_NODE_CLASS,
        NodeRegistrationReason.SCHEMA_MALFORMED,
    }
    assert captured.value.action


@pytest.mark.parametrize(
    "node_class",
    [
        object(),
        type("MissingInfo", (), {"GET_SCHEMA": classmethod(lambda cls: object())}),
        type(
            "BrokenInfo",
            (),
            {
                "GET_SCHEMA": classmethod(
                    lambda cls: SimpleNamespace(
                        node_id="Sigmax.Broken",
                        display_name="Broken",
                        description="",
                        category="Sigmax/testing",
                        is_output_node=False,
                        is_deprecated=False,
                        is_experimental=False,
                    )
                ),
                "GET_NODE_INFO_V1": classmethod(lambda cls: []),
            },
        ),
    ],
)
def test_v3_discovery_fails_actionably_for_invalid_public_class(
    node_class: object,
) -> None:
    with pytest.raises(NodeRegistrationError) as captured:
        discover_v3_registration(cast(Any, node_class))

    assert captured.value.reason in {
        NodeRegistrationReason.INVALID_NODE_CLASS,
        NodeRegistrationReason.SCHEMA_MALFORMED,
    }
    assert captured.value.action


def test_registration_contracts_are_immutable() -> None:
    registration = _legacy()
    registry = NodeRegistry.empty().register(registration)

    with pytest.raises(FrozenInstanceError):
        registration.node_id = "Sigmax.Changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        registry.entries = ()  # type: ignore[misc]


def test_builtin_registry_and_package_mappings_expose_only_validated_product_nodes() -> None:
    import comfyui_sigmax
    from comfyui_sigmax.nodes import (
        AdvancedFlowMatchScheduler,
        AnimaSigmaScheduler,
        AuraFlowSigmaScheduler,
        CheckpointEvidenceInspector,
        Flux1SchnellSigmaScheduler,
        HunyuanImage21SigmaScheduler,
        Krea2ConditioningRebalance,
        Krea2SigmaScheduler,
        LTXSigmaScheduler,
        Lumina2SigmaScheduler,
        MiniMaxH3SigmaScheduler,
        ModelAwareSigmaScheduler,
        ProfileInspector,
        QwenImageSigmaScheduler,
        RawWorkflowOutput,
        ScheduleComparison,
        ScheduleConcatenate,
        ScheduleInspector,
        ScheduleResample,
        ScheduleSlice,
        SD3SigmaScheduler,
        TurboWorkflowOutput,
        WanSigmaScheduler,
        ZImageSigmaScheduler,
    )

    registry = builtin_node_registry()
    assert tuple(item.node_id for item in registry.entries) == (
        "Sigmax.AdvancedFlowMatchScheduler",
        "Sigmax.AnimaSigmaScheduler",
        "Sigmax.AuraFlowSigmaScheduler",
        "Sigmax.CheckpointEvidenceInspector",
        "Sigmax.Flux1SchnellSigmaScheduler",
        "Sigmax.HunyuanImage21SigmaScheduler",
        "Sigmax.Krea2ConditioningRebalance",
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.LTXSigmaScheduler",
        "Sigmax.Lumina2SigmaScheduler",
        "Sigmax.MiniMaxH3SigmaScheduler",
        "Sigmax.ModelAwareSigmaScheduler",
        "Sigmax.ProfileInspector",
        "Sigmax.QwenImageSigmaScheduler",
        "Sigmax.RawWorkflowOutput",
        "Sigmax.SD3SigmaScheduler",
        "Sigmax.ScheduleComparison",
        "Sigmax.ScheduleConcatenate",
        "Sigmax.ScheduleInspector",
        "Sigmax.ScheduleResample",
        "Sigmax.ScheduleSlice",
        "Sigmax.TurboWorkflowOutput",
        "Sigmax.WanSigmaScheduler",
        "Sigmax.ZImageSigmaScheduler",
    )
    assert {
        "Sigmax.AdvancedFlowMatchScheduler": AdvancedFlowMatchScheduler,
        "Sigmax.AnimaSigmaScheduler": AnimaSigmaScheduler,
        "Sigmax.AuraFlowSigmaScheduler": AuraFlowSigmaScheduler,
        "Sigmax.CheckpointEvidenceInspector": CheckpointEvidenceInspector,
        "Sigmax.Flux1SchnellSigmaScheduler": Flux1SchnellSigmaScheduler,
        "Sigmax.HunyuanImage21SigmaScheduler": HunyuanImage21SigmaScheduler,
        "Sigmax.Krea2ConditioningRebalance": Krea2ConditioningRebalance,
        "Sigmax.Krea2SigmaScheduler": Krea2SigmaScheduler,
        "Sigmax.LTXSigmaScheduler": LTXSigmaScheduler,
        "Sigmax.Lumina2SigmaScheduler": Lumina2SigmaScheduler,
        "Sigmax.MiniMaxH3SigmaScheduler": MiniMaxH3SigmaScheduler,
        "Sigmax.QwenImageSigmaScheduler": QwenImageSigmaScheduler,
        "Sigmax.ModelAwareSigmaScheduler": ModelAwareSigmaScheduler,
        "Sigmax.SD3SigmaScheduler": SD3SigmaScheduler,
        "Sigmax.ProfileInspector": ProfileInspector,
        "Sigmax.RawWorkflowOutput": RawWorkflowOutput,
        "Sigmax.ScheduleComparison": ScheduleComparison,
        "Sigmax.ScheduleConcatenate": ScheduleConcatenate,
        "Sigmax.ScheduleInspector": ScheduleInspector,
        "Sigmax.ScheduleResample": ScheduleResample,
        "Sigmax.ScheduleSlice": ScheduleSlice,
        "Sigmax.TurboWorkflowOutput": TurboWorkflowOutput,
        "Sigmax.WanSigmaScheduler": WanSigmaScheduler,
        "Sigmax.ZImageSigmaScheduler": ZImageSigmaScheduler,
    } == comfyui_sigmax.NODE_CLASS_MAPPINGS
    assert comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS == {
        "Sigmax.AdvancedFlowMatchScheduler": "Advanced FlowMatch Scheduler",
        "Sigmax.AnimaSigmaScheduler": "Anima Sigma Scheduler",
        "Sigmax.AuraFlowSigmaScheduler": "AuraFlow Sigma Scheduler",
        "Sigmax.CheckpointEvidenceInspector": "Checkpoint Evidence Inspector",
        "Sigmax.Flux1SchnellSigmaScheduler": "FLUX.1-schnell Sigma Scheduler",
        "Sigmax.HunyuanImage21SigmaScheduler": "HunyuanImage 2.1 Sigma Scheduler",
        "Sigmax.Krea2ConditioningRebalance": "Krea 2 Conditioning Rebalance",
        "Sigmax.Krea2SigmaScheduler": "Krea 2 Sigma Scheduler",
        "Sigmax.LTXSigmaScheduler": "LTX Sigma Scheduler",
        "Sigmax.Lumina2SigmaScheduler": "Lumina-Image 2.0 Sigma Scheduler",
        "Sigmax.MiniMaxH3SigmaScheduler": "MiniMax H3 Sigma Scheduler",
        "Sigmax.QwenImageSigmaScheduler": "Qwen Image Sigma Scheduler",
        "Sigmax.ModelAwareSigmaScheduler": "Model-Aware Sigma Scheduler",
        "Sigmax.SD3SigmaScheduler": "Stable Diffusion 3 Sigma Scheduler",
        "Sigmax.ProfileInspector": "Profile Inspector",
        "Sigmax.RawWorkflowOutput": "RAW Workflow Output",
        "Sigmax.ScheduleComparison": "Schedule Comparison",
        "Sigmax.ScheduleConcatenate": "Schedule Concatenate",
        "Sigmax.ScheduleInspector": "Schedule Inspector",
        "Sigmax.ScheduleResample": "Schedule Resample",
        "Sigmax.ScheduleSlice": "Schedule Slice",
        "Sigmax.TurboWorkflowOutput": "Turbo Workflow Output",
        "Sigmax.WanSigmaScheduler": "Wan Sigma Scheduler",
        "Sigmax.ZImageSigmaScheduler": "Z-Image Sigma Scheduler",
    }
    assert comfyui_sigmax.NODE_CLASS_MAPPINGS is not builtin_node_registry().class_mappings()


def test_bootstrap_uses_catalog_without_inert_v3_entrypoint() -> None:
    package_source = (ROOT / "comfyui_sigmax" / "__init__.py").read_text(encoding="utf-8")

    assert "builtin_node_registry" in package_source
    assert "comfy_entrypoint" not in package_source


def test_registration_source_never_imports_or_mutates_comfy_sampler_registry() -> None:
    path = ROOT / "comfyui_sigmax" / "adapters" / "registration.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path.name)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(name == "comfy" or name.startswith("comfy.") for name in imports)
    assert "comfy.samplers" not in source
    assert "add_scheduler" not in source


def test_wrong_argument_types_fail_closed() -> None:
    with pytest.raises(ScheduleContractError):
        NodeRegistry(entries=cast(Any, []))
    with pytest.raises(ScheduleContractError):
        NodeRegistry.empty().register(cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        NodeRegistry.empty().register_many(cast(Any, object()))
    with pytest.raises(ScheduleContractError):
        require_registration_compatible(cast(Any, object()), _probe())
    with pytest.raises(ScheduleContractError):
        require_registration_compatible(_legacy(), cast(Any, object()))


def test_registration_reason_and_action_do_not_echo_private_payload() -> None:
    class PrivateNode:
        RETURN_TYPES = ()

        @classmethod
        def INPUT_TYPES(cls) -> object:
            return {"required": {"C:\\private\\model.ckpt": [object()]}}

    with pytest.raises(NodeRegistrationError) as captured:
        discover_legacy_registration(
            node_id="Sigmax.Private",
            display_name="Private",
            node_class=PrivateNode,
        )

    assert "C:\\private" not in str(captured.value)
    assert "model.ckpt" not in str(captured.value)


@pytest.mark.parametrize(
    "arguments",
    [
        {"reason": cast(Any, "bad"), "action": "fix"},
        {"reason": NodeRegistrationReason.SCHEMA_MALFORMED, "action": ""},
        {
            "reason": NodeRegistrationReason.SCHEMA_MALFORMED,
            "action": cast(Any, None),
        },
    ],
)
def test_registration_error_contract_rejects_invalid_construction(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ScheduleContractError):
        NodeRegistrationError(**cast(Any, arguments))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_id": "other"},
        {"schema_version": "2"},
        {"node_id": "Other"},
        {"display_name": cast(Any, None)},
        {"display_name": ""},
        {"display_name": "x" * 513},
        {"display_name": "line\nbreak"},
        {"description": cast(Any, None)},
        {"category": "testing"},
        {"python_module": "custom_nodes.folder"},
        {"output_node": cast(Any, 0)},
        {"node_class": cast(Any, object())},
        {"definition_kind": cast(Any, "legacy_v1")},
        {"definition": cast(Any, object())},
        {"definition": replace(_legacy().definition, node_id="Sigmax.Other")},
        {"lifecycle": cast(Any, "landed")},
        {"lifecycle": HostCapabilityLifecycle.EXPERIMENTAL},
        {"source_payload_json": cast(Any, b"{}")},
        {"source_payload_json": "[]"},
        {"source_payload_json": '{ "name": "not-canonical" }'},
    ],
)
def test_registration_contract_rejects_invalid_values(
    changes: dict[str, object],
) -> None:
    with pytest.raises((ScheduleContractError, NodeRegistrationError)):
        replace(_legacy(), **cast(Any, changes))


def test_deprecated_lifecycle_is_explicitly_unsupported() -> None:
    class DeprecatedLegacy(LegacyNode):
        DEPRECATED = True

    registration = discover_legacy_registration(
        node_id="Sigmax.Deprecated",
        display_name="Deprecated",
        node_class=DeprecatedLegacy,
    )

    assert registration.lifecycle is HostCapabilityLifecycle.UNSUPPORTED


def test_registry_contract_rejects_duplicates_noncanonical_and_invalid_items() -> None:
    legacy = _legacy()
    v3 = _v3()

    with pytest.raises(ScheduleContractError):
        NodeRegistry(entries=(legacy, legacy))
    with pytest.raises(ScheduleContractError):
        NodeRegistry(entries=(v3, legacy))
    with pytest.raises(ScheduleContractError):
        NodeRegistry(entries=cast(Any, (object(),)))
    with pytest.raises(ScheduleContractError):
        NodeRegistry.empty().register_many((cast(Any, object()),))


def test_legacy_discovery_catches_input_execution_and_wire_encoding_failures() -> None:
    class Raises:
        RETURN_TYPES = ()

        @classmethod
        def INPUT_TYPES(cls) -> object:
            raise RuntimeError("private payload")

    with pytest.raises(NodeRegistrationError) as captured:
        discover_legacy_registration(
            node_id="Sigmax.Raises",
            display_name="Raises",
            node_class=Raises,
        )
    assert captured.value.reason is NodeRegistrationReason.SCHEMA_MALFORMED
    assert "private payload" not in str(captured.value)

    class NonFinite(LegacyNode):
        @classmethod
        def INPUT_TYPES(cls) -> dict[str, object]:
            return {"required": {"value": ("FLOAT", {"default": float("nan")})}}

    with pytest.raises(NodeRegistrationError):
        discover_legacy_registration(
            node_id="Sigmax.NonFinite",
            display_name="Non Finite",
            node_class=NonFinite,
        )

    class Oversized(LegacyNode):
        @classmethod
        def INPUT_TYPES(cls) -> dict[str, object]:
            return {"required": {"value": ("STRING", {"default": "x" * 1_048_577})}}

    with pytest.raises(NodeRegistrationError):
        discover_legacy_registration(
            node_id="Sigmax.Oversized",
            display_name="Oversized",
            node_class=Oversized,
        )


def test_legacy_discovery_rejects_invalid_output_fields_and_output_node() -> None:
    for node_class in (
        type(
            "StringReturns",
            (),
            {
                "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
                "RETURN_TYPES": "SIGMAS",
            },
        ),
        type(
            "BrokenNames",
            (),
            {
                "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
                "RETURN_TYPES": ("SIGMAS",),
                "RETURN_NAMES": object(),
            },
        ),
        type(
            "BrokenLists",
            (),
            {
                "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
                "RETURN_TYPES": ("SIGMAS",),
                "OUTPUT_IS_LIST": object(),
            },
        ),
        type(
            "BrokenOutputNode",
            (),
            {
                "INPUT_TYPES": classmethod(lambda cls: {"required": {}}),
                "RETURN_TYPES": (),
                "OUTPUT_NODE": "false",
            },
        ),
    ):
        with pytest.raises((NodeRegistrationError, TypeError)):
            discover_legacy_registration(
                node_id="Sigmax.BrokenOutput",
                display_name="Broken Output",
                node_class=node_class,
            )


@pytest.mark.parametrize(
    "schema_changes",
    [
        {"is_experimental": "false"},
        {"is_deprecated": "false"},
        {"is_deprecated": True},
    ],
)
def test_v3_schema_lifecycle_validation_covers_all_paths(
    schema_changes: dict[str, object],
) -> None:
    class Lifecycle(V3Node):
        @classmethod
        def GET_SCHEMA(cls) -> object:
            values: dict[str, object] = {
                "node_id": "Sigmax.V3Fixture",
                "display_name": "Sigmax V3 Fixture",
                "description": "V3 test node",
                "category": "Sigmax/testing",
                "is_output_node": False,
                "is_deprecated": False,
                "is_experimental": True,
            }
            values.update(schema_changes)
            return SimpleNamespace(**values)

    with pytest.raises(NodeRegistrationError) as captured:
        discover_v3_registration(Lifecycle)

    assert captured.value.reason is NodeRegistrationReason.SCHEMA_LIFECYCLE_MISMATCH


def test_v3_schema_defaults_display_and_metadata_when_omitted() -> None:
    class MinimalV3:
        @classmethod
        def GET_SCHEMA(cls) -> object:
            return SimpleNamespace(node_id="Sigmax.MinimalV3")

        @classmethod
        def GET_NODE_INFO_V1(cls) -> dict[str, object]:
            return {
                "input": {},
                "output": [],
                "name": "Sigmax.MinimalV3",
                "display_name": "Sigmax.MinimalV3",
                "description": "",
                "category": "Sigmax",
                "output_node": False,
            }

    registration = discover_v3_registration(MinimalV3)

    assert registration.display_name == "Sigmax.MinimalV3"
    assert registration.description == ""
    assert registration.category == "Sigmax"
    assert registration.output_node is False


def test_v3_public_method_execution_failure_is_actionable() -> None:
    class Raises:
        @classmethod
        def GET_SCHEMA(cls) -> object:
            raise ValueError("private schema failure")

        @classmethod
        def GET_NODE_INFO_V1(cls) -> dict[str, object]:
            return {}

    with pytest.raises(NodeRegistrationError) as captured:
        discover_v3_registration(Raises)

    assert captured.value.reason is NodeRegistrationReason.SCHEMA_MALFORMED
    assert "private schema failure" not in str(captured.value)


def test_v2_registration_rejects_invalid_class_module_and_payload() -> None:
    with pytest.raises(NodeRegistrationError):
        registration_from_node_definition_v2(
            node_class=cast(Any, object()),
            payload=_v2_payload(),
        )
    wrong_module = _v2_payload()
    wrong_module["python_module"] = "custom_nodes.folder"
    with pytest.raises(NodeRegistrationError):
        registration_from_node_definition_v2(
            node_class=type("Node", (), {}),
            payload=wrong_module,
        )
    with pytest.raises(NodeRegistrationError):
        registration_from_node_definition_v2(
            node_class=type("Node", (), {}),
            payload=[],
        )


def test_v2_object_info_projection_handles_noncombo_and_hidden_exclusion() -> None:
    payload = _v2_payload(node_id="Sigmax.V2Values")
    payload["inputs"] = {
        "model": {"name": "model", "type": "MODEL", "isOptional": False},
        "mask": {"name": "mask", "type": "MASK", "isOptional": True},
    }
    registration = registration_from_node_definition_v2(
        node_class=type("V2Values", (), {}),
        payload=payload,
    )
    registry = NodeRegistry.empty().register(registration)

    object_info = registry.object_info_projection()["Sigmax.V2Values"]
    assert object_info["input"] == {
        "required": {"model": ["MODEL"]},
        "optional": {"mask": ["MASK"]},
    }

    hidden_definition = replace(
        _legacy().definition,
        inputs=(
            *_legacy().definition.inputs,
            cast(
                Any,
                type(_legacy().definition.inputs[0])(
                    name="prompt",
                    type_name="PROMPT",
                    section="hidden",
                    optional=False,
                ),
            ),
        ),
    )
    hidden_registration = replace(
        _legacy(),
        definition=hidden_definition,
        source_payload_json=json.dumps(
            {
                **json.loads(_legacy().source_payload_json),
                "input": {
                    **json.loads(_legacy().source_payload_json)["input"],
                    "hidden": {"prompt": ["PROMPT"]},
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    v2_wire = (
        NodeRegistry.empty()
        .register(hidden_registration)
        .node_definition_v2_projection()["Sigmax.LegacyFixture"]
    )
    assert "prompt" not in v2_wire["inputs"]
