"""Phase 0 RED contracts for Anima registry, workflow, and host seams."""

from __future__ import annotations

from typing import Any

from comfyui_sigmax.profiles.registry import ProfileKey, builtin_profile_registry
from comfyui_sigmax.workflows import load_canonical_workflow_fixtures, load_pinned_host_baseline


def test_anima_profiles_are_builtin_registry_entries() -> None:
    import importlib

    module = importlib.import_module("comfyui_sigmax.profiles.anima")
    schemas: tuple[Any, ...] = tuple(
        getattr(module, name, None)
        for name in ("ANIMA_AESTHETIC_SCHEMA", "ANIMA_BASE_SCHEMA", "ANIMA_TURBO_SCHEMA")
    )
    assert all(schema is not None for schema in schemas)

    registry = builtin_profile_registry()
    keys = tuple(entry.key for entry in registry.entries)
    assert tuple(ProfileKey.from_schema(schema) for schema in schemas) <= keys


def test_anima_workflow_fixtures_and_pinned_host_object_info_are_present() -> None:
    fixtures = load_canonical_workflow_fixtures()
    identifiers = tuple(item.identifier for item in fixtures)
    assert "anima-base-v1-framework-50" in identifiers
    assert "anima-turbo-v1-framework-8" in identifiers

    baseline = load_pinned_host_baseline()
    assert "Sigmax.AnimaSigmaScheduler" in baseline.object_info
    node = baseline.object_info["Sigmax.AnimaSigmaScheduler"]
    assert isinstance(node, dict)
    assert node["output"] == ["SIGMAS", "STRING"]
    assert node["input"]["required"]["variant"][0] == [
        "Base (3.0)",
        "Aesthetic (3.0)",
        "Turbo (3.0)",
    ]
