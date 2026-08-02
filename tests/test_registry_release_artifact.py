from __future__ import annotations

import copy
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest
from scripts import validate_registry_artifact as registry

ROOT = Path(__file__).resolve().parents[1]


def _repo_temp(prefix: str) -> Path:
    root = ROOT / ".tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def test_source_manifest_binds_frozen_registry_package_node_and_workflow_identity() -> None:
    envelope = registry.build_release_manifest(ROOT)
    manifest = cast(dict[str, Any], envelope["manifest"])

    assert envelope["schema"] == "sigmax.registry-release-manifest-envelope/1"
    assert registry.fingerprint(manifest) == envelope["manifest_fingerprint"]
    assert manifest["schema"] == "sigmax.registry-release-manifest/1"
    assert manifest["registry"] == {
        "display_name": "ComfyUI-Sigmax",
        "node_id": "comfyui-sigmax",
        "publisher_id": "rookiestar28",
        "publication_performed": False,
        "repository": "https://github.com/rookiestar28/ComfyUI-Sigmax",
    }
    assert manifest["package"] == {
        "license": "MIT",
        "requires_comfyui": ">=0.29.0",
        "requires_python": ">=3.10",
        "version": "1.0.0",
    }
    assert len(manifest["nodes"]) == 16
    assert len(manifest["workflows"]) == 8
    assert all(
        row["package"] == {"id": "comfyui-sigmax", "version": "1.0.0"}
        for row in manifest["workflows"]
    )
    assert all(row["host"]["version"] == "0.29.0" for row in manifest["workflows"])


def test_manifest_validation_rejects_version_node_and_host_drift() -> None:
    envelope = registry.build_release_manifest(ROOT)

    for mutation, finding in (
        (("package", "version", "1.0.1"), "manifest.package_version_mismatch"),
        (
            ("workflows", 0, "package", "version", "0.1.0.dev0"),
            "manifest.workflow_package_mismatch",
        ),
        (("workflows", 0, "nodes", 0, "version", "2"), "manifest.workflow_node_mismatch"),
        (("workflows", 0, "host", "version", "0.30.0"), "manifest.workflow_host_mismatch"),
    ):
        changed = copy.deepcopy(envelope)
        cursor: Any = changed["manifest"]
        for key in mutation[:-2]:
            cursor = cursor[key]
        cursor[mutation[-2]] = mutation[-1]
        changed["manifest_fingerprint"] = registry.fingerprint(changed["manifest"])
        assert finding in registry.validate_release_manifest(changed, ROOT)


def test_comfyignore_selection_matches_reviewed_runtime_boundary() -> None:
    selected = registry.select_registry_paths(ROOT)

    assert selected == sorted(selected)
    assert ".comfyignore" in selected
    assert "__init__.py" in selected
    assert "README.md" in selected
    assert "LICENSE.TXT" in selected
    assert "NOTICE" in selected
    assert "pyproject.toml" in selected
    assert "web/krea2_strict_official_extension.js" in selected
    assert "web/krea2_strict_official_policy.js" in selected
    assert "comfyui_sigmax/registry/release_manifest_v1.json" in selected
    assert any(path.startswith("comfyui_sigmax/") for path in selected)
    assert not any(
        path == denied or path.startswith(denied + "/")
        for path in selected
        for denied in registry.FORBIDDEN_REGISTRY_ROOTS
    )
    assert not any(path in registry.FORBIDDEN_REGISTRY_FILES for path in selected)


def test_registry_member_audit_rejects_paths_links_size_and_missing_contracts() -> None:
    clean = [
        ("__init__.py", b"pass\n", False),
        ("README.md", b"public\n", False),
        ("LICENSE.TXT", b"MIT\n", False),
        ("NOTICE", b"notice\n", False),
        ("pyproject.toml", b"[project]\n", False),
        ("comfyui_sigmax/__init__.py", b"pass\n", False),
        ("comfyui_sigmax/contracts/manifest_v1.json", b"{}\n", False),
        ("comfyui_sigmax/registry/release_manifest_v1.json", b"{}\n", False),
        ("comfyui_sigmax/workflows/fixtures.json", b"{}\n", False),
        ("web/krea2_strict_official_extension.js", b"export {};\n", False),
        ("web/krea2_strict_official_policy.js", b"export {};\n", False),
    ]
    assert registry.audit_registry_members(clean)["status"] == "PASS"

    bad = [
        *clean,
        ("../escape.py", b"pass\n", False),
        ("tests/private.py", b"pass\n", False),
        ("models/example.safetensors", b"weight", False),
        ("linked.py", b"", True),
        ("large.bin", b"x" * (registry.MAX_MEMBER_BYTES + 1), False),
    ]
    assert registry.audit_registry_members(bad)["findings"] == [
        "archive.forbidden_path",
        "archive.link_forbidden",
        "archive.model_weight",
        "archive.oversized",
        "archive.path_unsafe",
        "archive.top_level_invalid",
    ]


def test_registry_archive_is_byte_reproducible_and_matches_index_content() -> None:
    output = _repo_temp("registry-repeat-")
    first = output / "first.zip"
    second = output / "second.zip"

    first_report = registry.build_and_audit_registry_archive(ROOT, first)
    second_report = registry.build_and_audit_registry_archive(ROOT, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_report == second_report
    assert first_report["status"] == "PASS"
    assert first_report["archive_sha256"] == registry.file_sha256(first)
    assert first_report["selected_paths"] == registry.select_registry_paths(ROOT)


def test_registry_archive_normalized_directory_install_imports_exact_contract() -> None:
    output = _repo_temp("registry-install-")
    archive = output / "candidate.zip"
    registry.build_and_audit_registry_archive(ROOT, archive)

    result = registry.probe_normalized_install(archive, output / "custom_nodes" / "renamed-pack")

    assert result["status"] == "PASS"
    assert result["package_version"] == "1.0.0"
    assert result["node_ids"] == registry.expected_node_ids(ROOT)
    assert result["external_modules_loaded"] == []


def test_registry_observation_is_read_only_and_does_not_infer_ownership() -> None:
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, object]:
        calls.append(url)
        if "/nodes?" in url:
            return 200, {"nodes": [], "total": 0}
        if "/publishers/validate?" in url:
            return 200, {"isAvailable": True}
        return 404, {"error": "not found"}

    result = registry.observe_registry_identity(
        node_id="comfyui-sigmax",
        publisher_id="rookiestar28",
        fetch=fetch,
    )

    assert result["status"] == "PASS"
    assert result["node_id_available"] is True
    assert result["publisher_id_available"] is True
    assert result["publisher_owned"] is False
    assert result["publication_performed"] is False
    assert len(calls) == 3
    assert all(url.startswith("https://api.comfy.org/") for url in calls)
    assert all("token" not in url.lower() for url in calls)


def test_registry_archive_contains_no_duplicate_or_nondeterministic_zip_metadata() -> None:
    output = _repo_temp("registry-metadata-")
    archive = output / "candidate.zip"
    registry.build_and_audit_registry_archive(ROOT, archive)

    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        assert len(infos) == len({item.filename for item in infos})
        assert all(item.date_time == registry.ZIP_TIMESTAMP for item in infos)
        assert all(item.create_system == 3 for item in infos)
        assert all(item.external_attr >> 16 == 0o100644 for item in infos)


def test_cli_surface_has_no_publish_or_credential_option() -> None:
    parser = registry.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}

    assert "--publish" not in options
    assert "--token" not in options
    assert "--api-key" not in options
    assert {"--archive", "--check-manifest", "--observe-registry", "--output"} <= options


@pytest.mark.parametrize(
    "name",
    ["C:/escape.py", "/absolute.py", "folder\\alias.py", "a/../escape.py", "a//b.py"],
)
def test_registry_archive_rejects_noncanonical_member_names(name: str) -> None:
    result = registry.audit_registry_members([(name, b"pass\n", False)])
    assert "archive.path_unsafe" in cast(list[str], result["findings"])
