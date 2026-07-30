"""Contract tests for the repository's public documentation surface."""

from __future__ import annotations

import re
from pathlib import Path

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
    assert "provisional" in profile_spec and "not frozen" in profile_spec
    assert "not yet validated" in compatibility


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
