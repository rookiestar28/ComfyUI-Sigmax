from __future__ import annotations

import copy
import io
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest
from scripts import run_release_audit as audit

ROOT = Path(__file__).resolve().parents[1]


def _repo_temp(prefix: str) -> Path:
    root = ROOT / ".tmp"
    root.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def _profile_projections() -> list[dict[str, Any]]:
    return audit.builtin_profile_provenance()


def test_tracked_file_audit_rejects_private_and_binary_material() -> None:
    clean = audit.audit_tracked_paths(["README.md", "comfyui_sigmax/core/artifacts.py"])
    assert clean == {"count": 2, "findings": [], "status": "PASS"}

    result = audit.audit_tracked_paths(
        [
            ".planning/private.md",
            "reference/source/README.md",
            "models/example.safetensors",
            ".env",
        ]
    )
    assert result["status"] == "FAIL"
    assert result["findings"] == [
        "tracked.internal_path",
        "tracked.model_weight",
        "tracked.sensitive_filename",
    ]


def test_dependency_audit_separates_all_four_classes() -> None:
    pyproject = audit.read_pyproject(ROOT / "pyproject.toml")
    result = audit.audit_dependencies(pyproject)

    assert result["status"] == "PASS"
    assert result["mandatory"] == []
    assert set(cast(dict[str, list[str]], result["optional"])) == {"plot", "reference"}
    assert result["development"] == pyproject["project"]["optional-dependencies"]["dev"]
    assert result["build"] == ["setuptools>=80,<84"]


def test_dependency_audit_rejects_runtime_and_unbounded_sources() -> None:
    pyproject = copy.deepcopy(audit.read_pyproject(ROOT / "pyproject.toml"))
    pyproject["project"]["dependencies"] = ["requests>=2"]
    pyproject["project"]["optional-dependencies"]["plot"] = [
        "unsafe @ https://private.invalid/package.whl"
    ]

    result = audit.audit_dependencies(pyproject)

    assert result["findings"] == [
        "dependency.mandatory_present",
        "dependency.source_unsafe",
        "dependency.unbounded",
    ]


def test_builtin_provenance_layers_and_licenses_are_distinct() -> None:
    profiles = _profile_projections()
    result = audit.audit_provenance(profiles)

    assert result["status"] == "PASS"
    rows = cast(list[dict[str, Any]], result["profiles"])
    assert [row["profile_key"] for row in rows] == [
        "auraflow.v0-2.official@1",
        "flux1.schnell.official@1",
        "krea2.raw-turbo-lora.experimental@1",
        "krea2.raw.official@1",
        "krea2.turbo.official@1",
        "lumina2.v2.official@1",
        "qwen_image.comfy-fixed.official@1",
        "qwen_image.diffusers-dynamic.framework-reference@1",
        "sd3.comfy-diffusers-fixed.framework-reference@1",
        "sd3.publisher-reference.official@1",
        "z_image.base.official@1",
        "z_image.turbo.official@1",
    ]
    for row in rows:
        if row["profile_key"].startswith("auraflow."):
            assert row["resource_counts"] == {
                "frameworks": 2,
                "model_weights": 1,
                "software_sources": 1,
            }
            assert set(row["license_identifiers"]) == {"Apache-2.0", "GPL-3.0-only"}
        elif row["profile_key"].startswith("krea2."):
            assert row["resource_counts"] == {
                "frameworks": 2,
                "model_weights": (2 if ".raw-turbo-lora." in row["profile_key"] else 1),
                "software_sources": 1,
            }
            assert set(row["license_identifiers"]) == {
                "Apache-2.0",
                "GPL-3.0-only",
                "LicenseRef-Krea-2-Community",
            }
        elif row["profile_key"].startswith("z_image."):
            expected_weights = 3 if ".base." in row["profile_key"] else 4
            assert row["resource_counts"] == {
                "frameworks": 1,
                "model_weights": expected_weights,
                "software_sources": 1,
            }
            assert set(row["license_identifiers"]) == {"Apache-2.0", "GPL-3.0-only"}
        elif row["profile_key"].startswith("qwen_image."):
            assert row["resource_counts"] == {
                "frameworks": 2,
                "model_weights": 1,
                "software_sources": 1,
            }
            assert set(row["license_identifiers"]) == {"Apache-2.0", "GPL-3.0-only"}
        elif row["profile_key"].startswith("sd3."):
            assert row["resource_counts"] == {
                "frameworks": 2,
                "model_weights": 1,
                "software_sources": 1,
            }
            assert set(row["license_identifiers"]) == {
                "Apache-2.0",
                "GPL-3.0-only",
                "LicenseRef-Stability-AI-Community",
                "MIT",
            }
        elif row["profile_key"].startswith("lumina2."):
            assert row["resource_counts"] == {
                "frameworks": 2,
                "model_weights": 1,
                "software_sources": 1,
            }
            assert set(row["license_identifiers"]) == {"Apache-2.0", "GPL-3.0-only"}
        else:
            assert row["profile_key"] == "flux1.schnell.official@1"
            assert row["resource_counts"] == {
                "frameworks": 1,
                "model_weights": 1,
                "software_sources": 1,
            }
            assert set(row["license_identifiers"]) == {"Apache-2.0", "GPL-3.0-only"}


def test_provenance_audit_rejects_missing_or_conflated_layers() -> None:
    profiles = _profile_projections()
    profiles[0]["provenance"]["frameworks"] = []
    profiles[1]["provenance"]["model_weights"][0]["id"] = profiles[1]["provenance"][
        "software_sources"
    ][0]["id"]

    result = audit.audit_provenance(profiles)

    assert result["findings"] == ["provenance.layer_alias", "provenance.layer_missing"]


def test_registry_metadata_is_a_separate_nonpublishing_section() -> None:
    result = audit.audit_registry(audit.read_pyproject(ROOT / "pyproject.toml"))

    assert result == {
        "comfy_requirement": ">=0.29.0",
        "display_name": "ComfyUI-Sigmax",
        "findings": [],
        "package_version": "1.0.0",
        "publisher_id": "rookiestar28",
        "publish_performed": False,
        "python_requirement": ">=3.10",
        "status": "PASS",
    }


def _zip(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"fixture")


def _tar(path: Path, names: list[str], *, symlink: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            info = tarfile.TarInfo(name)
            payload = b"fixture"
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if symlink is not None:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "../outside"
            archive.addfile(info)


@pytest.mark.parametrize(
    ("kind", "names", "symlink", "code"),
    [
        ("wheel", ["../escape.py"], None, "archive.path_unsafe"),
        ("wheel", ["C:/escape.py"], None, "archive.path_unsafe"),
        ("wheel", ["tests/test_leak.py"], None, "archive.forbidden_path"),
        ("sdist", ["pkg/.planning/private.md"], None, "archive.forbidden_path"),
        ("sdist", ["pkg/README.md"], "pkg/link", "archive.link_forbidden"),
    ],
)
def test_archive_audit_rejects_unsafe_members(
    kind: str,
    names: list[str],
    symlink: str | None,
    code: str,
) -> None:
    temp_root = _repo_temp("release-audit-malicious-")
    path = temp_root / ("fixture.whl" if kind == "wheel" else "fixture.tar.gz")
    if kind == "wheel":
        _zip(path, names)
    else:
        _tar(path, names, symlink=symlink)

    result = audit.audit_archive(path, kind=kind)

    assert code in cast(list[str], result["findings"])
    assert result["status"] == "FAIL"


def test_archive_audit_rejects_oversized_payload_before_reading() -> None:
    path = _repo_temp("release-audit-oversized-") / "oversized.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("comfyui_sigmax/large.bin", b"x" * (4 * 1024 * 1024 + 1))

    result = audit.audit_archive(path, kind="wheel")

    assert result["findings"] == ["archive.oversized"]


@pytest.fixture(scope="module")
def built_archives() -> tuple[Path, Path]:
    return audit.build_archives(ROOT, _repo_temp("release-audit-archive-"))


def test_fresh_wheel_and_sdist_are_public_safe(
    built_archives: tuple[Path, Path],
) -> None:
    wheel, sdist = built_archives

    wheel_result = audit.audit_archive(wheel, kind="wheel")
    sdist_result = audit.audit_archive(sdist, kind="sdist")

    assert wheel_result["status"] == sdist_result["status"] == "PASS"
    assert cast(int, wheel_result["file_count"]) >= 60
    assert cast(int, sdist_result["file_count"]) >= 60


def test_canonical_report_is_fingerprinted_and_repeat_stable(
    built_archives: tuple[Path, Path],
) -> None:
    wheel, sdist = built_archives
    first = audit.build_release_audit(ROOT, wheel, sdist, secret_scan_passed=True)
    second = audit.build_release_audit(ROOT, wheel, sdist, secret_scan_passed=True)

    assert first == second
    assert first["schema"] == "sigmax.release-audit-envelope/1"
    first_audit = cast(dict[str, object], first["audit"])
    assert first_audit["schema"] == "sigmax.release-audit/1"
    assert first_audit["status"] == "PASS"
    encoded = audit.canonical_report_bytes(first)
    assert encoded.endswith(b"\n")
    assert json.loads(encoded)["audit_fingerprint"].startswith("sha256:")
