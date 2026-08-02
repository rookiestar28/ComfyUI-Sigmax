"""Phase 0 RED contracts for LTX registry, workflow, and pinned-host seams."""

from __future__ import annotations

from typing import Any, cast

from comfyui_sigmax.profiles.ltx import (
    LTX2_19B_DISTILLED_STAGE1_PROFILE,
    LTX2_19B_PROFILE,
    LTX23_22B_DISTILLED_STAGE2_PROFILE,
    LTX23_22B_PROFILE,
    LTXV_098_PROFILE,
)
from comfyui_sigmax.profiles.registry import ProfileKey, builtin_profile_registry
from comfyui_sigmax.workflows import load_canonical_workflow_fixtures, load_pinned_host_baseline


def test_ltx_profiles_are_builtin_registry_entries() -> None:
    keys = tuple(entry.key for entry in builtin_profile_registry().entries)
    for profile in (
        LTXV_098_PROFILE,
        LTX2_19B_PROFILE,
        LTX2_19B_DISTILLED_STAGE1_PROFILE,
        LTX23_22B_PROFILE,
        LTX23_22B_DISTILLED_STAGE2_PROFILE,
    ):
        assert ProfileKey.from_schema(profile.schema) in keys


def test_ltx_workflows_and_pinned_host_object_info_are_present() -> None:
    identifiers = {item.identifier for item in load_canonical_workflow_fixtures()}
    assert {
        "ltxv-0-9-8-dev-20",
        "ltx2-3-22b-dev-30",
        "ltx2-19b-distilled-stage1-8",
        "ltx2-3-22b-distilled-stage2-3",
    } <= identifiers
    baseline = load_pinned_host_baseline()
    assert "Sigmax.LTXSigmaScheduler" in baseline.object_info
    node = cast(dict[str, Any], baseline.object_info["Sigmax.LTXSigmaScheduler"])
    assert node["output"] == ["SIGMAS", "STRING"]
    assert node["input"]["required"]["generation"][0] == [
        "LTXV 0.9.8",
        "LTX-2 19B",
        "LTX-2.3 22B",
    ]
