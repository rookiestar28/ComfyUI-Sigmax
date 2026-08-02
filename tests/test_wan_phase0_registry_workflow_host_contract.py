"""Phase 0 RED contracts for Wan registry, workflow, and pinned-host seams."""

from __future__ import annotations

import importlib
from typing import Any

from comfyui_sigmax.profiles.registry import ProfileKey, builtin_profile_registry
from comfyui_sigmax.workflows import load_canonical_workflow_fixtures, load_pinned_host_baseline


def test_wan_profiles_are_builtin_registry_entries() -> None:
    module = importlib.import_module("comfyui_sigmax.profiles.wan")
    schemas: tuple[Any, ...] = tuple(
        getattr(module, name, None)
        for name in (
            "WAN21_COMFY_NATIVE_SCHEMA",
            "WAN21_T2V_OFFICIAL_SCHEMA",
            "WAN22_T2V_A14B_NATIVE_SCHEMA",
        )
    )
    assert all(schema is not None for schema in schemas)
    registry = builtin_profile_registry()
    keys = tuple(entry.key for entry in registry.entries)
    assert all(ProfileKey.from_schema(schema) in keys for schema in schemas)


def test_wan_workflow_fixtures_and_pinned_host_object_info_are_present() -> None:
    fixtures = load_canonical_workflow_fixtures()
    identifiers = tuple(item.identifier for item in fixtures)
    assert "wan21-t2v-official-50" in identifiers
    assert "wan22-t2v-a14b-native-40" in identifiers
    baseline = load_pinned_host_baseline()
    assert "Sigmax.WanSigmaScheduler" in baseline.object_info
    node = baseline.object_info["Sigmax.WanSigmaScheduler"]
    assert isinstance(node, dict)
    assert node["output"] == ["SIGMAS", "INT", "STRING"]
