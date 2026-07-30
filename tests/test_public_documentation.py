"""Contract tests for the repository's public documentation surface."""

from __future__ import annotations

import re
from pathlib import Path

from comfyui_sigmax.profiles import (
    KREA2_RAW_OFFICIAL_SHA256,
    KREA2_TURBO_OFFICIAL_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "PROFILE_SPEC.md",
    ROOT / "docs" / "SCHEDULE_ARTIFACT_SPEC.md",
    ROOT / "docs" / "COMPATIBILITY.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_public_documents_exist() -> None:
    missing = [path.relative_to(ROOT).as_posix() for path in PUBLIC_DOCUMENTS if not path.is_file()]
    assert not missing, f"Missing public documentation: {missing}"


def test_public_documentation_describes_current_maturity_honestly() -> None:
    readme = _read(ROOT / "README.md").lower()
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md").lower()
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md").lower()
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md").lower()

    assert "pre-alpha" in readme
    assert "no user-facing comfyui nodes" in readme
    assert "planned" in architecture and "implemented" in architecture
    assert "frozen" in profile_spec and "sigmax.model-profile/1" in profile_spec
    assert "not yet validated" in compatibility


def test_public_documentation_exposes_current_artifact_transport_contract() -> None:
    readme = _read(ROOT / "README.md")
    artifact_spec = _read(ROOT / "docs" / "SCHEDULE_ARTIFACT_SPEC.md")

    assert "serialize_schedule_artifact" in readme
    assert "deserialize_schedule_artifact" in readme
    assert "sigmax.schedule-artifact-envelope/1" in artifact_spec
    assert "1,048,576 bytes" in artifact_spec


def test_public_documentation_exposes_capability_preflight_contract() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")

    assert "ModelCapabilities" in readme
    assert "ProfileCapabilities" in readme
    assert "SamplerCapabilities" in readme
    assert "CompatibilityDecision" in architecture
    assert all(level in profile_spec.lower() for level in ("allow", "warn", "reject"))


def test_public_documentation_exposes_frozen_profile_schema_v1() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, changelog):
        assert "ProfileSchemaV1" in content
    assert "sigmax.model-profile/1" in profile_spec
    assert "SoftwareSourceProvenance" in profile_spec
    assert "FrameworkProvenance" in profile_spec
    assert "ModelWeightProvenance" in profile_spec
    assert "profile_schema_fingerprint" in readme
    assert "profiles/schema_v1.py" in architecture


def test_public_documentation_exposes_namespaced_profile_registry() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, changelog):
        assert "ProfileRegistry" in content
    assert "builtin_profile_registry" in readme
    assert "REPLACE_EXTERNAL" in profile_spec
    assert "profiles/registry.py" in architecture
    assert "cannot replace a built-in" in profile_spec


def test_public_documentation_exposes_core_independence_boundary() -> None:
    readme = _read(ROOT / "README.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")

    assert "check_core_independence.py" in readme
    assert "ComfyUI" in architecture and "Diffusers" in architecture
    assert "python -I" in architecture
    assert "deterministic property" in compatibility
    assert "framework" in compatibility and "native ComfyUI" in compatibility


def test_public_documentation_exposes_structural_krea2_turbo_profile() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")

    assert "KREA2_TURBO_PROFILE" in readme
    assert "build_krea2_turbo_schedule" in readme
    assert "krea2.turbo.official" in profile_spec
    assert "profiles/krea2_turbo.py" in architecture
    assert "structural profile" in compatibility.lower()
    assert "4, 8," in compatibility and "12, and 16" in compatibility
    assert "authoritative framework parity" in compatibility.lower()
    assert "diffusers 0.39.0" in compatibility.lower()
    assert "native comfyui parity" in compatibility.lower()
    assert "real-host" in compatibility.lower()


def test_public_documentation_exposes_turbo_golden_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    assert "krea2_turbo_v1.json" in readme
    assert "krea2_turbo_parity_v1.json" in readme
    assert "krea2_turbo_comfy_native_parity_v1.json" in readme
    assert "ModelSamplingFlux" in readme
    assert "integer-index quantization" in readme
    assert "Decimal" in architecture
    assert "float64" in compatibility and "float32" in compatibility
    assert "5.960464477539063e-08" in compatibility
    assert "framework" in changelog.lower() and "native-comfyui parity" in changelog.lower()


def test_public_documentation_exposes_raw_structural_profile_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    for content in (readme, architecture, profile_spec, compatibility, changelog):
        assert "krea2.raw.official" in content
    assert "krea2.raw.official-full-52" in readme
    assert "krea2.raw.diffusers-reference-28" in readme
    assert "256" in compatibility and "6400" in compatibility
    assert "0.5" in compatibility and "1.15" in compatibility
    assert "unclamped" in compatibility.lower()
    assert "derive_krea2_raw_shift" in readme
    assert "build_krea2_raw_schedule" in readme
    assert "krea2_raw_v1.json" in readme
    assert "requested" in profile_spec.lower() and "effective" in profile_spec.lower()
    assert "packed image sequence length" in compatibility.lower()
    assert "resolve_krea2_image_geometry" in architecture
    assert "M3-02" not in readme


def test_public_documentation_exposes_raw_parity_boundary() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    assert "krea2_raw_parity_v1.json" in readme
    assert "FlowMatchEulerDiscreteScheduler" in readme
    assert "1.1920928955078125e-07" in readme
    assert "run_krea2_raw_parity.py" in architecture
    assert "all 14" in compatibility
    assert "Krea 2 RAW parity" in changelog


def test_public_documentation_exposes_fail_closed_variant_resolution() -> None:
    readme = _read(ROOT / "README.md")
    architecture = _read(ROOT / "docs" / "ARCHITECTURE.md")
    profile_spec = _read(ROOT / "docs" / "PROFILE_SPEC.md")
    compatibility = _read(ROOT / "docs" / "COMPATIBILITY.md")

    assert "resolve_krea2_variant" in readme
    assert "suggestions" in readme.lower()
    assert "strict official mode" in readme.lower()
    assert "krea2_variant.py" in architecture
    assert "conflicting resolving evidence" in architecture.lower()
    assert KREA2_RAW_OFFICIAL_SHA256 in profile_spec
    assert KREA2_TURBO_OFFICIAL_SHA256 in profile_spec
    assert "family-only" in compatibility.lower()


def test_public_contribution_and_changelog_contract() -> None:
    contributing = _read(ROOT / "CONTRIBUTING.md")
    changelog = _read(ROOT / "CHANGELOG.md")

    assert "scripts/run_full_tests_windows.ps1" in contributing
    assert "scripts/run_full_tests_linux.sh" in contributing
    assert ".venv" in contributing
    assert ".venv-wsl" in contributing
    assert "## [Unreleased]" in changelog


def test_public_documents_do_not_expose_internal_workspace_material() -> None:
    forbidden = (
        ".planning/",
        ".planning\\",
        "ROADMAP.md",
        "AGENTS.md",
        "reference/docs",
        "reference\\docs",
        "B:\\",
        "/mnt/b/",
    )

    for path in PUBLIC_DOCUMENTS:
        content = _read(path)
        leaked = [marker for marker in forbidden if marker.lower() in content.lower()]
        assert not leaked, f"{path.relative_to(ROOT)} exposes internal markers: {leaked}"


def test_public_relative_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    broken: list[str] = []

    for path in PUBLIC_DOCUMENTS:
        for target in link_pattern.findall(_read(path)):
            target_without_fragment = target.split("#", maxsplit=1)[0].strip()
            if not target_without_fragment or "://" in target_without_fragment:
                continue
            resolved = (path.parent / target_without_fragment).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")

    assert not broken, f"Broken local documentation links: {broken}"
