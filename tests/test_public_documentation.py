"""Contracts for the intentionally small public documentation surface."""

from __future__ import annotations

import re
from pathlib import Path

from comfyui_sigmax import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
COMPATIBILITY = ROOT / "docs" / "COMPATIBILITY.md"
PUBLIC_DOCUMENTS = (
    README,
    COMPATIBILITY,
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "tests" / "TEST_SOP.md",
    ROOT / "tests" / "E2E_TESTING_NOTICE.md",
    ROOT / "tests" / "E2E_TESTING_SOP.md",
    ROOT / "tests" / "CI_TEST_MATRIX.md",
)
REMOVED_DOCS = (
    "ARCHITECTURE.md",
    "COINSTALLATION_MUTATION_MATRIX_SPEC.md",
    "COMFY_REGISTRY_RELEASE_ARTIFACT.md",
    "DEPENDENCY_COMPATIBILITY_MATRIX_SPEC.md",
    "EXECUTION_RECEIPT_SPEC.md",
    "IMAGE_BENCHMARK_PROTOCOL_SPEC.md",
    "NUMERICAL_BENCHMARK_MATRIX_SPEC.md",
    "PERFORMANCE_BUDGET_SPEC.md",
    "PROFILE_CONTRIBUTION_GUIDE.md",
    "PROFILE_SPEC.md",
    "SCHEDULE_ARTIFACT_SPEC.md",
    "SCHEDULE_REPORT_SPEC.md",
    "SECURITY_RELEASE_AUDIT.md",
    "STABLE_PUBLIC_CONTRACTS.md",
    "USER_GUIDE.md",
    "WORKFLOW_METADATA_SPEC.md",
    "WORKFLOW_VALIDATION_SPEC.md",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_public_user_documentation_surface_is_intentionally_small() -> None:
    assert README.is_file()
    assert COMPATIBILITY.is_file()
    assert sorted(path.name for path in (ROOT / "docs").iterdir() if path.is_file()) == [
        "COMPATIBILITY.md"
    ]
    assert len(_read(README).splitlines()) <= 140
    assert len(_read(COMPATIBILITY).splitlines()) <= 100


def test_readme_is_limited_to_product_installation_and_use() -> None:
    readme = _read(README)
    headings = re.findall(r"^## .+$", readme, flags=re.MULTILINE)
    assert headings == [
        "## Features",
        "## Installation",
        "## Use in ComfyUI",
        "## Update or remove",
        "## Troubleshooting",
    ]
    for required in (
        "Krea 2 Turbo",
        "Krea 2 RAW",
        "Z-Image Base",
        "Z-Image Turbo",
        "FLUX.1-schnell",
        "Original Stable Diffusion 3",
        "git clone https://github.com/rookiestar28/ComfyUI-Sigmax comfyui-sigmax",
        "Python 3.10 or newer",
        "ComfyUI 0.29.0 or newer",
        "SIGMAS",
        "schedule_info",
    ):
        assert required in readme
    for secondary in (
        "Planned Product Shape",
        "Repository Surface",
        "Development Setup",
        "benchmark matrix",
        "schema specification",
        "implementation record",
    ):
        assert secondary.casefold() not in readme.casefold()


def test_readme_matches_the_registered_node_surface() -> None:
    readme = _read(README)
    assert len(NODE_CLASS_MAPPINGS) == 17
    assert NODE_CLASS_MAPPINGS.keys() == NODE_DISPLAY_NAME_MAPPINGS.keys()
    assert "registers 17 namespaced nodes" in readme
    for node_id in (
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.ZImageSigmaScheduler",
        "Sigmax.Flux1SchnellSigmaScheduler",
        "Sigmax.ModelAwareSigmaScheduler",
        "Sigmax.AdvancedFlowMatchScheduler",
        "Sigmax.CheckpointEvidenceInspector",
        "Sigmax.Krea2ConditioningRebalance",
        "Sigmax.QwenImageSigmaScheduler",
        "Sigmax.SD3SigmaScheduler",
        "Sigmax.ProfileInspector",
        "Sigmax.ScheduleInspector",
        "Sigmax.ScheduleComparison",
        "Sigmax.ScheduleSlice",
        "Sigmax.ScheduleConcatenate",
        "Sigmax.ScheduleResample",
    ):
        assert node_id in readme


def test_readme_records_the_reviewed_user_recipes_and_safety_boundaries() -> None:
    readme = _read(README)
    for recipe in (
        "`Turbo`, 8 steps, Euler, CFG 1.0",
        "`RAW`, 52 steps with CFG 4.5, or 28 steps with CFG 5.5",
        "`Base`, 28-50 steps, default 50, CFG 4.0",
        "`Turbo`, 8 steps, CFG 1.0",
        "1-4 steps, default 4, CFG 1.0",
        "`Comfy Fixed`, 50 steps, or `Diffusers Dynamic` with explicit `image_seq_len`",
        "`Publisher Reference (1.0)` at 50 steps or `Comfy/Diffusers Fixed (3.0)` at 28 steps",
    ):
        assert recipe in readme
    assert "Do not pass the result through another scheduler" in readme
    assert "suggested variant is advisory, not confirmation" in readme
    assert "does not replace the sampler" in readme
    assert "download models" in readme.casefold()


def test_compatibility_is_user_facing_and_fail_closed() -> None:
    compatibility = _read(COMPATIBILITY)
    for required in (
        "## Environment",
        "## Supported model profiles",
        "## Usage boundary",
        "## Not currently claimed",
        "Windows and Linux/WSL",
        "Real-model GPU execution or image-quality parity",
        "Automatic compatibility with unlisted model families",
        "Original Qwen Image",
        "unvalidated",
    ):
        assert required in compatibility


def test_removed_secondary_documents_are_absent_and_unreferenced() -> None:
    public_text = "\n".join(_read(path) for path in PUBLIC_DOCUMENTS)
    for name in REMOVED_DOCS:
        assert not (ROOT / "docs" / name).exists()
        assert name not in public_text


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
        content = _read(path).casefold()
        leaked = [marker for marker in forbidden if marker.casefold() in content]
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


def test_release_and_contributor_documents_retain_their_core_contracts() -> None:
    changelog = _read(ROOT / "CHANGELOG.md")
    contributing = _read(ROOT / "CONTRIBUTING.md")
    assert "## [Unreleased]" in changelog
    assert "## [1.0.0] - 2026-08-01" in changelog
    for command in (
        "scripts/run_full_tests_windows.ps1",
        "scripts/run_full_tests_linux.sh",
        "scripts/run_release_audit.py",
        "scripts/validate_registry_artifact.py",
    ):
        assert command in contributing
