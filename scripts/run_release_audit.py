"""Build and audit public release archives without publishing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import tomli as tomllib

ROOT: Final = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA: Final = "sigmax.release-audit/1"
AUDIT_ENVELOPE_SCHEMA: Final = "sigmax.release-audit-envelope/1"
_FINGERPRINT: Final = re.compile(r"^[0-9a-f]{64}$")
_REVISION: Final = re.compile(r"^[0-9a-f]{40}$")
_VERSION: Final = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:\.[a-z]+[0-9]+)?$")
_MAX_MEMBER_BYTES: Final = 4 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 32 * 1024 * 1024
_MAX_MEMBERS: Final = 1024
_FORBIDDEN_PARTS: Final = frozenset(
    {
        ".git",
        ".github",
        ".planning",
        ".tmp",
        ".venv",
        ".venv-wsl",
        "__pycache__",
        "build",
        "dist",
        "docs",
        "reference",
        "scripts",
        "tests",
    }
)
_TRACKED_FORBIDDEN_PARTS: Final = frozenset(
    {
        ".git",
        ".planning",
        ".tmp",
        ".venv",
        ".venv-wsl",
        "__pycache__",
        "build",
        "dist",
        "reference",
    }
)
_MODEL_SUFFIXES: Final = frozenset(
    {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
)
_SENSITIVE_NAMES: Final = frozenset(
    {".env", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
)
_SDIST_ROOT_FILES: Final = frozenset(
    {
        "LICENSE.TXT",
        "MANIFEST.in",
        "NOTICE",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "setup.cfg",
    }
)


class ReleaseAuditError(RuntimeError):
    """Raised when the audit cannot safely construct or inspect its evidence."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _section(findings: set[str], **values: object) -> dict[str, object]:
    return {
        **values,
        "findings": sorted(findings),
        "status": "PASS" if not findings else "FAIL",
    }


def read_pyproject(path: Path) -> dict[str, Any]:
    """Read one local pyproject mapping."""

    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if not isinstance(value, dict):
        raise ReleaseAuditError("pyproject must contain a table")
    return value


def audit_tracked_paths(paths: list[str]) -> dict[str, object]:
    """Audit repository-tracked names without reading ignored private material."""

    findings: set[str] = set()
    canonical = sorted(set(path.replace("\\", "/") for path in paths))
    for name in canonical:
        pure = PurePosixPath(name)
        lowered = tuple(part.lower() for part in pure.parts)
        if any(part in _TRACKED_FORBIDDEN_PARTS for part in lowered):
            findings.add("tracked.internal_path")
        if pure.suffix.lower() in _MODEL_SUFFIXES:
            findings.add("tracked.model_weight")
        filename = pure.name.lower()
        if filename in _SENSITIVE_NAMES or filename.startswith(".env."):
            findings.add("tracked.sensitive_filename")
    return _section(findings, count=len(canonical))


def _requirement_findings(requirement: object) -> set[str]:
    findings: set[str] = set()
    if not isinstance(requirement, str) or not requirement.strip():
        return {"dependency.invalid"}
    lowered = requirement.lower()
    if "@" in requirement or "://" in lowered or "file:" in lowered:
        findings.add("dependency.source_unsafe")
    if "<" not in requirement:
        findings.add("dependency.unbounded")
    return findings


def audit_dependencies(pyproject: dict[str, Any]) -> dict[str, object]:
    """Inventory and validate independent dependency classes."""

    findings: set[str] = set()
    try:
        project = cast(dict[str, Any], pyproject["project"])
        mandatory = list(cast(list[str], project["dependencies"]))
        extras = cast(dict[str, list[str]], project["optional-dependencies"])
        development = list(extras["dev"])
        optional = {key: list(value) for key, value in extras.items() if key != "dev"}
        build = list(cast(list[str], pyproject["build-system"]["requires"]))
    except (KeyError, TypeError) as exc:
        raise ReleaseAuditError("dependency metadata is incomplete") from exc
    if mandatory:
        findings.add("dependency.mandatory_present")
    for requirement in [*mandatory, *development, *build]:
        findings.update(_requirement_findings(requirement))
    for requirements in optional.values():
        for requirement in requirements:
            findings.update(_requirement_findings(requirement))
    return _section(
        findings,
        build=build,
        development=development,
        mandatory=mandatory,
        optional=dict(sorted(optional.items())),
    )


def builtin_profile_provenance() -> list[dict[str, Any]]:
    """Return detached provenance projections for every built-in profile."""

    from comfyui_sigmax.profiles import builtin_profile_registry, profile_schema_projection

    rows: list[dict[str, Any]] = []
    for entry in builtin_profile_registry().entries:
        projection = profile_schema_projection(entry.schema)
        rows.append(
            {
                "profile_key": entry.key.canonical,
                "provenance": projection["provenance"],
            }
        )
    return cast(list[dict[str, Any]], json.loads(_canonical(rows)))


def _valid_resource(resource: dict[str, Any], *, weight: bool) -> bool:
    license_value = resource.get("license")
    if not isinstance(license_value, dict):
        return False
    required_license = {"declaration_version", "identifier", "name", "url"}
    if set(license_value) != required_license:
        return False
    texts = [
        resource.get("id"),
        resource.get("record_version"),
        resource.get("revision"),
        resource.get("url"),
        license_value.get("declaration_version"),
        license_value.get("identifier"),
        license_value.get("name"),
        license_value.get("url"),
    ]
    if not all(isinstance(value, str) and value for value in texts):
        return False
    if not cast(str, resource["url"]).startswith("https://"):
        return False
    if not cast(str, license_value["url"]).startswith("https://"):
        return False
    if not _REVISION.fullmatch(cast(str, resource["revision"])):
        return False
    if weight:
        fingerprint = resource.get("sha256")
        return isinstance(fingerprint, str) and _FINGERPRINT.fullmatch(fingerprint) is not None
    locators = resource.get("locators")
    return (
        isinstance(locators, list)
        and bool(locators)
        and all(
            isinstance(locator, str) and locator and not PurePosixPath(locator).is_absolute()
            for locator in locators
        )
    )


def audit_provenance(profiles: list[dict[str, Any]]) -> dict[str, object]:
    """Audit source, framework, and model-weight identities as separate layers."""

    findings: set[str] = set()
    rows: list[dict[str, object]] = []
    for profile in sorted(profiles, key=lambda value: cast(str, value.get("profile_key", ""))):
        key = profile.get("profile_key")
        provenance = profile.get("provenance")
        if not isinstance(key, str) or not isinstance(provenance, dict):
            findings.add("provenance.invalid")
            continue
        layer_names = ("frameworks", "model_weights", "software_sources")
        layers: dict[str, list[dict[str, Any]]] = {}
        for layer in layer_names:
            value = provenance.get(layer)
            if not isinstance(value, list) or not value:
                findings.add("provenance.layer_missing")
                layers[layer] = []
                continue
            if not all(isinstance(item, dict) for item in value):
                findings.add("provenance.invalid")
                layers[layer] = []
                continue
            layers[layer] = cast(list[dict[str, Any]], value)
        identifiers = [
            cast(str, resource.get("id", "")) for layer in layer_names for resource in layers[layer]
        ]
        if len(identifiers) != len(set(identifiers)):
            findings.add("provenance.layer_alias")
        license_ids: set[str] = set()
        for layer in layer_names:
            for resource in layers[layer]:
                if not _valid_resource(resource, weight=layer == "model_weights"):
                    findings.add("provenance.resource_invalid")
                    continue
                license_ids.add(cast(str, cast(dict[str, Any], resource["license"])["identifier"]))
        rows.append(
            {
                "license_identifiers": sorted(license_ids),
                "profile_key": key,
                "resource_counts": {layer: len(layers[layer]) for layer in layer_names},
                "resource_ids": sorted(identifier for identifier in identifiers if identifier),
            }
        )
    return _section(findings, profiles=rows)


def audit_registry(pyproject: dict[str, Any]) -> dict[str, object]:
    """Audit Registry metadata without invoking or authenticating to the Registry."""

    findings: set[str] = set()
    try:
        project = cast(dict[str, Any], pyproject["project"])
        registry = cast(dict[str, Any], pyproject["tool"]["comfy"])
        publisher = registry["PublisherId"]
        display = registry["DisplayName"]
        comfy = registry["requires-comfyui"]
        version = project["version"]
        python = project["requires-python"]
    except (KeyError, TypeError) as exc:
        raise ReleaseAuditError("Registry metadata is incomplete") from exc
    if publisher != "rookiestar28" or display != "ComfyUI-Sigmax":
        findings.add("registry.identity_invalid")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        findings.add("registry.version_invalid")
    if python != ">=3.10" or comfy != ">=0.29.0":
        findings.add("registry.requirement_invalid")
    return _section(
        findings,
        comfy_requirement=comfy,
        display_name=display,
        package_version=version,
        publisher_id=publisher,
        publish_performed=False,
        python_requirement=python,
    )


def _path_findings(name: str) -> set[str]:
    findings: set[str] = set()
    if "\\" in name or re.match(r"^[A-Za-z]:", name):
        findings.add("archive.path_unsafe")
        return findings
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        findings.add("archive.path_unsafe")
    lowered = tuple(part.lower() for part in path.parts)
    if any(part in _FORBIDDEN_PARTS for part in lowered):
        findings.add("archive.forbidden_path")
    if path.suffix.lower() in _MODEL_SUFFIXES:
        findings.add("archive.model_weight")
    if path.name.lower() in _SENSITIVE_NAMES or path.name.lower().startswith(".env."):
        findings.add("archive.sensitive_filename")
    return findings


def _archive_projection(
    *,
    kind: str,
    members: list[tuple[str, bytes, bool]],
) -> dict[str, object]:
    findings: set[str] = set()
    names = [name for name, _payload, _is_link in members]
    if len(names) != len(set(names)):
        findings.add("archive.duplicate_name")
    if sum(len(payload) for _name, payload, _is_link in members) > _MAX_ARCHIVE_BYTES:
        findings.add("archive.oversized")
    for name, payload, is_link in members:
        findings.update(_path_findings(name))
        if is_link:
            findings.add("archive.link_forbidden")
        if len(payload) > _MAX_MEMBER_BYTES:
            findings.add("archive.oversized")

    file_members = [
        (name, payload) for name, payload, is_link in members if payload and not is_link
    ]
    if kind == "wheel":
        relative = file_members
        allowed = all(
            name.startswith("comfyui_sigmax/")
            or (".dist-info/" in name and name.startswith("comfyui_sigmax-"))
            for name, _payload in relative
        )
        required = {
            "comfyui_sigmax/__init__.py": any(
                name == "comfyui_sigmax/__init__.py" for name, _ in relative
            ),
            "comfyui_sigmax/contracts/manifest_v1.json": any(
                name == "comfyui_sigmax/contracts/manifest_v1.json" for name, _ in relative
            ),
            "LICENSE.TXT": any(
                name.endswith(".dist-info/licenses/LICENSE.TXT") for name, _ in relative
            ),
            "NOTICE": any(name.endswith(".dist-info/licenses/NOTICE") for name, _ in relative),
        }
    elif kind == "sdist":
        roots = {PurePosixPath(name).parts[0] for name, _payload in file_members if name}
        if len(roots) != 1:
            findings.add("archive.top_level_invalid")
            root = ""
        else:
            root = next(iter(roots))
        relative = [
            (PurePosixPath(name).relative_to(root).as_posix(), payload)
            for name, payload in file_members
            if root and PurePosixPath(name).parts[0] == root
        ]
        allowed = all(
            name in _SDIST_ROOT_FILES
            or name.startswith("comfyui_sigmax/")
            or name.startswith("comfyui_sigmax.egg-info/")
            for name, _payload in relative
        )
        required = {
            "LICENSE.TXT": any(name == "LICENSE.TXT" for name, _ in relative),
            "NOTICE": any(name == "NOTICE" for name, _ in relative),
            "README.md": any(name == "README.md" for name, _ in relative),
            "comfyui_sigmax/__init__.py": any(
                name == "comfyui_sigmax/__init__.py" for name, _ in relative
            ),
            "comfyui_sigmax/contracts/manifest_v1.json": any(
                name == "comfyui_sigmax/contracts/manifest_v1.json" for name, _ in relative
            ),
            "pyproject.toml": any(name == "pyproject.toml" for name, _ in relative),
        }
    else:
        raise ReleaseAuditError("archive kind is unsupported")
    if not allowed:
        findings.add("archive.top_level_invalid")
    if not all(required.values()):
        findings.add("archive.required_missing")
    content_rows = [
        {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(relative)
    ]
    return _section(
        findings,
        content_fingerprint=_identity(content_rows),
        file_count=len(relative),
        kind=kind,
        required=required,
    )


def audit_archive(path: Path, *, kind: str) -> dict[str, object]:
    """Inspect a wheel or sdist without extracting it."""

    if not path.is_file():
        raise ReleaseAuditError("archive does not exist")
    members: list[tuple[str, bytes, bool]] = []
    if kind == "wheel":
        with zipfile.ZipFile(path) as archive:
            zip_items = archive.infolist()
            if (
                len(zip_items) > _MAX_MEMBERS
                or sum(zip_item.file_size for zip_item in zip_items) > _MAX_ARCHIVE_BYTES
                or any(zip_item.file_size > _MAX_MEMBER_BYTES for zip_item in zip_items)
            ):
                return _section(
                    {"archive.oversized"},
                    content_fingerprint=_identity([]),
                    file_count=len(zip_items),
                    kind=kind,
                    required={},
                )
            for zip_item in zip_items:
                mode = zip_item.external_attr >> 16
                is_link = stat.S_ISLNK(mode)
                payload = b"" if zip_item.is_dir() or is_link else archive.read(zip_item)
                members.append((zip_item.filename, payload, is_link))
    elif kind == "sdist":
        with tarfile.open(path, "r:gz") as archive:
            tar_items = archive.getmembers()
            if (
                len(tar_items) > _MAX_MEMBERS
                or sum(tar_item.size for tar_item in tar_items) > _MAX_ARCHIVE_BYTES
                or any(tar_item.size > _MAX_MEMBER_BYTES for tar_item in tar_items)
            ):
                return _section(
                    {"archive.oversized"},
                    content_fingerprint=_identity([]),
                    file_count=len(tar_items),
                    kind=kind,
                    required={},
                )
            for tar_item in tar_items:
                is_link = tar_item.issym() or tar_item.islnk()
                stream = archive.extractfile(tar_item) if tar_item.isfile() else None
                payload = stream.read() if stream is not None else b""
                members.append((tar_item.name, payload, is_link))
    else:
        raise ReleaseAuditError("archive kind is unsupported")
    return _archive_projection(kind=kind, members=members)


def build_archives(root: Path, dist_dir: Path) -> tuple[Path, Path]:
    """Build fresh local archives with the selected interpreter and reviewed build isolation."""

    resolved_root = root.resolve()
    resolved_dist = dist_dir.resolve()
    if resolved_root not in resolved_dist.parents:
        raise ReleaseAuditError("distribution directory must stay inside the repository")
    resolved_dist.mkdir(parents=True, exist_ok=True)
    if any(resolved_dist.iterdir()):
        raise ReleaseAuditError("distribution directory must be empty")
    # SECURITY: fixed module and arguments; the selected project interpreter owns the build.
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--outdir",
            str(resolved_dist),
        ],
        cwd=resolved_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ReleaseAuditError("archive build failed")
    wheels = sorted(resolved_dist.glob("*.whl"))
    sdists = sorted(resolved_dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ReleaseAuditError("build must produce exactly one wheel and one sdist")
    return wheels[0], sdists[0]


def _tracked_paths(root: Path) -> list[str]:
    # SECURITY: fixed read-only Git query in the reviewed repository root.
    git = shutil.which("git")
    if git is None:
        raise ReleaseAuditError("Git executable is unavailable")
    completed = subprocess.run(  # noqa: S603
        [git, "ls-files", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ReleaseAuditError("tracked-file inventory failed")
    return [item.decode("utf-8") for item in completed.stdout.split(b"\0") if item]


def build_release_audit(
    root: Path,
    wheel: Path,
    sdist: Path,
    *,
    secret_scan_passed: bool,
) -> dict[str, object]:
    """Build a canonical semantic release-audit envelope."""

    pyproject = read_pyproject(root / "pyproject.toml")
    sections = {
        "archives": {
            "sdist": audit_archive(sdist, kind="sdist"),
            "wheel": audit_archive(wheel, kind="wheel"),
        },
        "dependencies": audit_dependencies(pyproject),
        "provenance": audit_provenance(builtin_profile_provenance()),
        "registry": audit_registry(pyproject),
        "secret_scan": _section(
            set() if secret_scan_passed else {"secret_scan.failed"},
            method="detect-secrets --all-files",
        ),
        "tracked_files": audit_tracked_paths(_tracked_paths(root)),
    }
    findings: set[str] = set()
    for name, section in sections.items():
        if name == "archives":
            for archive in cast(dict[str, dict[str, object]], section).values():
                findings.update(cast(list[str], archive["findings"]))
        else:
            findings.update(cast(list[str], cast(dict[str, object], section)["findings"]))
    audit = {
        **sections,
        "findings": sorted(findings),
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
    }
    return {
        "audit": audit,
        "audit_fingerprint": _identity(audit),
        "schema": AUDIT_ENVELOPE_SCHEMA,
    }


def canonical_report_bytes(report: dict[str, object]) -> bytes:
    """Return canonical newline-terminated report bytes after fingerprint verification."""

    if set(report) != {"audit", "audit_fingerprint", "schema"}:
        raise ReleaseAuditError("release audit envelope fields do not match schema")
    if report["schema"] != AUDIT_ENVELOPE_SCHEMA:
        raise ReleaseAuditError("release audit envelope schema is unsupported")
    audit = report["audit"]
    if not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA:
        raise ReleaseAuditError("release audit schema is unsupported")
    if report["audit_fingerprint"] != _identity(audit):
        raise ReleaseAuditError("release audit fingerprint drifted")
    return _canonical(report) + b"\n"


def _secret_scan(root: Path) -> bool:
    # SECURITY: fixed selected-interpreter hook invocation; it cannot publish or repair files.
    completed = subprocess.run(
        [sys.executable, "-m", "pre_commit", "run", "detect-secrets", "--all-files"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _inside_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ReleaseAuditError("audit output must stay inside the repository")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    dist_dir = _inside_root(args.dist_dir)
    output = _inside_root(args.output)
    wheel, sdist = build_archives(ROOT, dist_dir)
    report = build_release_audit(
        ROOT,
        wheel,
        sdist,
        secret_scan_passed=_secret_scan(ROOT),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_report_bytes(report))
    print(f"RELEASE_AUDIT={cast(dict[str, object], report['audit'])['status']}")
    print(f"AUDIT_FINGERPRINT={report['audit_fingerprint']}")
    return 0 if cast(dict[str, object], report["audit"])["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
