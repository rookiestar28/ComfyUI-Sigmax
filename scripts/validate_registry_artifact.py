"""Build and dry-validate a deterministic Comfy Registry archive without publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Final, cast

import tomli as tomllib

ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST_PATH: Final = ROOT / "comfyui_sigmax" / "registry" / "release_manifest_v1.json"
MANIFEST_SCHEMA: Final = "sigmax.registry-release-manifest/1"
MANIFEST_ENVELOPE_SCHEMA: Final = "sigmax.registry-release-manifest-envelope/1"
REPORT_SCHEMA: Final = "sigmax.registry-artifact-report/1"
REPORT_ENVELOPE_SCHEMA: Final = "sigmax.registry-artifact-report-envelope/1"
REGISTRY_API: Final = "https://api.comfy.org"
COMFY_CLI_RELEASE: Final = "v1.13.0"
COMFY_CLI_SOURCE_REVISION: Final = (
    "d3220c94598416c347404c89aafa3f0231be75f4"  # pragma: allowlist secret -- public commit SHA
)
ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
MAX_MEMBER_BYTES: Final = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES: Final = 32 * 1024 * 1024
MAX_MEMBERS: Final = 1024
FORBIDDEN_REGISTRY_ROOTS: Final = (
    ".github",
    "docs",
    "requirements",
    "scripts",
    "tests",
)
FORBIDDEN_REGISTRY_FILES: Final = frozenset(
    {
        ".gitignore",
        ".pre-commit-config.yaml",
        ".secrets.baseline",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "MANIFEST.in",
    }
)
_ALLOWED_ROOT_FILES: Final = frozenset(
    {".comfyignore", "LICENSE.TXT", "NOTICE", "README.md", "__init__.py", "pyproject.toml"}
)
_ALLOWED_ROOT_PREFIXES: Final = ("comfyui_sigmax/", "web/")
_REQUIRED_FILES: Final = frozenset(
    {
        "LICENSE.TXT",
        "NOTICE",
        "README.md",
        "__init__.py",
        "pyproject.toml",
        "comfyui_sigmax/__init__.py",
        "comfyui_sigmax/contracts/manifest_v1.json",
        "comfyui_sigmax/registry/release_manifest_v1.json",
        "comfyui_sigmax/workflows/fixtures.json",
        "web/krea2_strict_official_extension.js",
        "web/krea2_strict_official_policy.js",
    }
)
_MODEL_SUFFIXES: Final = frozenset(
    {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
)
_SENSITIVE_NAMES: Final = frozenset(
    {".env", "credentials", "credentials.json", "id_ed25519", "id_rsa"}
)
_SEMVER: Final = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA256: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_NODE_ID: Final = re.compile(r"^Sigmax\.[A-Za-z][A-Za-z0-9]*$")
_SCHEMA_ID: Final = re.compile(r"^sigmax\.[a-z0-9-]+/[1-9][0-9]*$")
_Fetch = Callable[[str], tuple[int, object]]


class RegistryArtifactError(RuntimeError):
    """A fail-closed Registry dry-validation error."""


def canonical(value: object) -> bytes:
    """Encode one canonical JSON value."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    """Return a typed SHA-256 identity for canonical JSON."""

    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    """Return a typed SHA-256 identity for file bytes."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RegistryArtifactError(f"{path.name} must contain a JSON object")
    return value


def _read_pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as stream:
        value = tomllib.load(stream)
    return value


def _source_row(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": file_sha256(root / relative)}


def expected_node_ids(root: Path) -> list[str]:
    """Return the sorted frozen public node IDs from the M8-01 manifest."""

    envelope = _read_json(root / "comfyui_sigmax" / "contracts" / "manifest_v1.json")
    nodes = cast(list[dict[str, str]], envelope["manifest"]["nodes"])
    return sorted(row["id"] for row in nodes)


def _workflow_projection(root: Path) -> list[dict[str, object]]:
    bundle = _read_json(root / "comfyui_sigmax" / "workflows" / "fixtures.json")
    rows: list[dict[str, object]] = []
    for raw in cast(list[dict[str, Any]], bundle["fixtures"]):
        rows.append(
            {
                "host": raw["host"],
                "id": raw["id"],
                "nodes": raw["nodes"],
                "package": raw["package"],
                "profile": raw["profile"],
                "workflow_fingerprint": fingerprint(raw["workflow"]),
            }
        )
    return sorted(rows, key=lambda row: cast(str, row["id"]))


def build_release_manifest(root: Path = ROOT) -> dict[str, object]:
    """Build the canonical release manifest from current public sources."""

    pyproject = _read_pyproject(root)
    project = cast(dict[str, Any], pyproject["project"])
    comfy = cast(dict[str, Any], pyproject["tool"]["comfy"])
    urls = cast(dict[str, Any], project["urls"])
    public = _read_json(root / "comfyui_sigmax" / "contracts" / "manifest_v1.json")
    public_manifest = cast(dict[str, Any], public["manifest"])
    nodes = [
        {"id": row["id"], "schema": row["schema"]}
        for row in cast(list[dict[str, str]], public_manifest["nodes"])
    ]
    sources = [
        _source_row(root, relative)
        for relative in (
            ".comfyignore",
            "__init__.py",
            "comfyui_sigmax/__init__.py",
            "comfyui_sigmax/contracts/manifest_v1.json",
            "comfyui_sigmax/version.py",
            "comfyui_sigmax/workflows/fixtures.json",
            "comfyui_sigmax/workflows/host_baseline.json",
            "pyproject.toml",
            "web/krea2_strict_official_extension.js",
            "web/krea2_strict_official_policy.js",
        )
    ]
    manifest: dict[str, object] = {
        "nodes": sorted(nodes, key=lambda row: row["id"]),
        "package": {
            "license": project["license"],
            "requires_comfyui": comfy["requires-comfyui"],
            "requires_python": project["requires-python"],
            "version": project["version"],
        },
        "public_contract": {
            "fingerprint": public["manifest_fingerprint"],
            "schema": public_manifest["schema"],
        },
        "registry": {
            "display_name": comfy["DisplayName"],
            "node_id": project["name"],
            "publication_performed": False,
            "publisher_id": comfy["PublisherId"],
            "repository": urls["Repository"],
        },
        "schema": MANIFEST_SCHEMA,
        "selection_contract": {
            "comfy_cli_release": COMFY_CLI_RELEASE,
            "comfy_cli_source_revision": COMFY_CLI_SOURCE_REVISION,
            "method": "git_tracked_minus_comfyignore",
        },
        "sources": sources,
        "workflows": _workflow_projection(root),
    }
    return {
        "manifest": manifest,
        "manifest_fingerprint": fingerprint(manifest),
        "schema": MANIFEST_ENVELOPE_SCHEMA,
    }


def _manifest_semantic_findings(manifest: dict[str, Any], root: Path) -> set[str]:
    findings: set[str] = set()
    expected = cast(dict[str, Any], build_release_manifest(root)["manifest"])
    package = cast(dict[str, Any], manifest.get("package", {}))
    expected_package = cast(dict[str, Any], expected["package"])
    if package.get("version") != expected_package["version"] or not _SEMVER.fullmatch(
        str(package.get("version", ""))
    ):
        findings.add("manifest.package_version_mismatch")
    if manifest.get("registry") != expected["registry"]:
        findings.add("manifest.registry_identity_mismatch")
    nodes = manifest.get("nodes")
    if nodes != expected["nodes"]:
        findings.add("manifest.node_contract_mismatch")
    else:
        for row in cast(list[dict[str, Any]], nodes):
            if not _NODE_ID.fullmatch(str(row.get("id", ""))) or not _SCHEMA_ID.fullmatch(
                str(row.get("schema", ""))
            ):
                findings.add("manifest.node_contract_mismatch")

    workflows = manifest.get("workflows")
    expected_workflows = cast(list[dict[str, Any]], expected["workflows"])
    if not isinstance(workflows, list) or len(workflows) != len(expected_workflows):
        findings.add("manifest.workflow_inventory_mismatch")
        return findings
    expected_by_id = {row["id"]: row for row in expected_workflows}
    for row in cast(list[dict[str, Any]], workflows):
        expected_row = expected_by_id.get(row.get("id"))
        if expected_row is None:
            findings.add("manifest.workflow_inventory_mismatch")
            continue
        if row.get("package") != expected_row["package"]:
            findings.add("manifest.workflow_package_mismatch")
        if row.get("nodes") != expected_row["nodes"]:
            findings.add("manifest.workflow_node_mismatch")
        if row.get("host") != expected_row["host"]:
            findings.add("manifest.workflow_host_mismatch")
        if row.get("profile") != expected_row["profile"]:
            findings.add("manifest.workflow_profile_mismatch")
        if row.get("workflow_fingerprint") != expected_row["workflow_fingerprint"]:
            findings.add("manifest.workflow_content_mismatch")
    if manifest.get("public_contract") != expected["public_contract"]:
        findings.add("manifest.public_contract_mismatch")
    if manifest.get("selection_contract") != expected["selection_contract"]:
        findings.add("manifest.selection_contract_mismatch")
    if manifest.get("sources") != expected["sources"]:
        findings.add("manifest.source_mismatch")
    return findings


def validate_release_manifest(envelope: object, root: Path = ROOT) -> list[str]:
    """Return stable findings for one release-manifest envelope."""

    if not isinstance(envelope, dict):
        return ["manifest.envelope_invalid"]
    value = cast(dict[str, Any], envelope)
    manifest = value.get("manifest")
    if value.get("schema") != MANIFEST_ENVELOPE_SCHEMA or not isinstance(manifest, dict):
        return ["manifest.envelope_invalid"]
    findings: set[str] = set()
    supplied = value.get("manifest_fingerprint")
    if (
        not isinstance(supplied, str)
        or not _SHA256.fullmatch(supplied)
        or supplied != fingerprint(manifest)
    ):
        findings.add("manifest.fingerprint_mismatch")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        findings.add("manifest.schema_invalid")
    findings.update(_manifest_semantic_findings(cast(dict[str, Any], manifest), root))
    return sorted(findings)


def _git(root: Path, *arguments: str) -> bytes:
    git = shutil.which("git")
    if git is None:
        raise RegistryArtifactError("Git executable is unavailable")
    # SECURITY: fixed executable and caller-controlled arguments are never passed through a shell.
    completed = subprocess.run(  # noqa: S603
        [git, *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RegistryArtifactError("Git query failed")
    return completed.stdout


def _tracked_paths(root: Path) -> list[str]:
    paths = [part.decode("utf-8") for part in _git(root, "ls-files", "-z").split(b"\0") if part]
    if not paths or len(paths) != len(set(paths)):
        raise RegistryArtifactError("tracked-file inventory is empty or contains duplicates")
    return sorted(paths)


def _index_bytes(root: Path, relative: str) -> bytes:
    return _git(root, "cat-file", "blob", f":{relative}")


def _ignore_patterns(root: Path) -> tuple[str, ...]:
    text = _index_bytes(root, ".comfyignore").decode("utf-8")
    patterns: list[str] = []
    for raw in text.splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith(("!", "/")) or any(character in value for character in "*?["):
            raise RegistryArtifactError("release .comfyignore must use exact top-level paths")
        normalized = value.replace("\\", "/")
        if "//" in normalized or ".." in PurePosixPath(normalized).parts:
            raise RegistryArtifactError("release .comfyignore contains an unsafe path")
        patterns.append(normalized)
    return tuple(patterns)


def _ignored(path: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            prefix = pattern.rstrip("/")
            if path == prefix or path.startswith(prefix + "/"):
                return True
        elif path == pattern:
            return True
    return False


def select_registry_paths(root: Path = ROOT) -> list[str]:
    """Apply the committed reviewed `.comfyignore` to Git's tracked inventory."""

    patterns = _ignore_patterns(root)
    selected = [path for path in _tracked_paths(root) if not _ignored(path, patterns)]
    if not selected or any(path.startswith(".") and path != ".comfyignore" for path in selected):
        raise RegistryArtifactError("Registry selection contains an unexpected hidden path")
    return selected


def _path_findings(name: str) -> set[str]:
    findings: set[str] = set()
    if "\\" in name or re.match(r"^[A-Za-z]:", name) or "//" in name:
        findings.add("archive.path_unsafe")
        return findings
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        findings.add("archive.path_unsafe")
    lowered = tuple(part.lower() for part in path.parts)
    if any(part in FORBIDDEN_REGISTRY_ROOTS for part in lowered):
        findings.add("archive.forbidden_path")
    if path.suffix.lower() in _MODEL_SUFFIXES:
        findings.add("archive.model_weight")
    if path.name.lower() in _SENSITIVE_NAMES or path.name.lower().startswith(".env."):
        findings.add("archive.sensitive_filename")
    return findings


def audit_registry_members(members: list[tuple[str, bytes, bool]]) -> dict[str, object]:
    """Audit in-memory Registry ZIP members without extraction."""

    findings: set[str] = set()
    names = [name for name, _payload, _link in members]
    if len(names) != len(set(names)):
        findings.add("archive.duplicate_name")
    if (
        len(members) > MAX_MEMBERS
        or sum(len(payload) for _, payload, _ in members) > MAX_ARCHIVE_BYTES
    ):
        findings.add("archive.oversized")
    for name, payload, is_link in members:
        findings.update(_path_findings(name))
        if is_link:
            findings.add("archive.link_forbidden")
        if len(payload) > MAX_MEMBER_BYTES:
            findings.add("archive.oversized")
        if not (name in _ALLOWED_ROOT_FILES or name.startswith(_ALLOWED_ROOT_PREFIXES)):
            findings.add("archive.top_level_invalid")
    if not _REQUIRED_FILES.issubset(names):
        findings.add("archive.required_missing")
    rows = [
        {"path": name, "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}
        for name, payload, is_link in sorted(members)
        if not is_link
    ]
    return {
        "content_fingerprint": fingerprint(rows),
        "file_count": len(members),
        "findings": sorted(findings),
        "status": "PASS" if not findings else "FAIL",
    }


def audit_registry_archive(path: Path) -> dict[str, object]:
    """Inspect one Registry ZIP with bounded reads."""

    if not path.is_file():
        raise RegistryArtifactError("Registry archive does not exist")
    members: list[tuple[str, bytes, bool]] = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if (
            len(infos) > MAX_MEMBERS
            or sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES
            or any(info.file_size > MAX_MEMBER_BYTES for info in infos)
        ):
            return {
                "content_fingerprint": fingerprint([]),
                "file_count": len(infos),
                "findings": ["archive.oversized"],
                "status": "FAIL",
            }
        for info in infos:
            mode = info.external_attr >> 16
            is_link = stat.S_ISLNK(mode)
            payload = b"" if info.is_dir() or is_link else archive.read(info)
            members.append((info.filename, payload, is_link))
    return audit_registry_members(members)


def _write_deterministic_zip(root: Path, output: Path, selected: list[str]) -> None:
    resolved_root = root.resolve()
    resolved_output = output.resolve()
    if resolved_root not in resolved_output.parents:
        raise RegistryArtifactError("Registry archive output must stay inside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RegistryArtifactError("Registry archive output must be fresh")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in selected:
            payload = _index_bytes(root, relative)
            if len(payload) > MAX_MEMBER_BYTES:
                raise RegistryArtifactError(f"Registry source exceeds member limit: {relative}")
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits |= 0x800
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_and_audit_registry_archive(root: Path, output: Path) -> dict[str, object]:
    """Build one deterministic index-bound ZIP and return its strict audit."""

    selected = select_registry_paths(root)
    _write_deterministic_zip(root, output, selected)
    report = audit_registry_archive(output)
    with zipfile.ZipFile(output) as archive:
        embedded_manifest = archive.read("comfyui_sigmax/registry/release_manifest_v1.json")
        expected_manifest = canonical(build_release_manifest(root)) + b"\n"
        findings = set(cast(list[str], report["findings"]))
        if embedded_manifest != expected_manifest:
            findings.add("archive.manifest_stale")
        else:
            envelope = cast(dict[str, Any], json.loads(embedded_manifest))
            manifest = cast(dict[str, Any], envelope["manifest"])
            for source in cast(list[dict[str, str]], manifest["sources"]):
                if source["path"] not in selected:
                    findings.add("archive.manifest_source_missing")
                    continue
                payload = archive.read(source["path"])
                identity = "sha256:" + hashlib.sha256(payload).hexdigest()
                if identity != source["sha256"]:
                    findings.add("archive.manifest_source_mismatch")
        report["findings"] = sorted(findings)
        report["status"] = "PASS" if not findings else "FAIL"
    report["archive_sha256"] = file_sha256(output)
    report["selected_paths"] = selected
    report["selection_fingerprint"] = fingerprint(selected)
    return report


def _safe_extract(archive_path: Path, target: Path) -> None:
    if target.exists():
        raise RegistryArtifactError("normalized install target must be fresh")
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if _path_findings(info.filename):
                raise RegistryArtifactError("unsafe archive cannot be installed")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or info.is_dir():
                raise RegistryArtifactError("Registry archive links/directories are forbidden")
            destination = target.joinpath(*PurePosixPath(info.filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))


def probe_normalized_install(archive_path: Path, target: Path) -> dict[str, object]:
    """Install beneath an arbitrary safe directory name and import the root bootstrap."""

    audit = audit_registry_archive(archive_path)
    if audit["status"] != "PASS":
        raise RegistryArtifactError("normalized install requires a passing archive")
    _safe_extract(archive_path, target)
    probe = """
import importlib.util, json, pathlib, sys
target = pathlib.Path(sys.argv[1]).resolve()
before = set(sys.modules)
spec = importlib.util.spec_from_file_location(
    "sigmax_registry_probe", target / "__init__.py", submodule_search_locations=[str(target)]
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load normalized package")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
forbidden = {"aiohttp", "diffusers", "numpy", "torch"}
print(json.dumps({
    "external_modules_loaded": sorted(forbidden & (set(sys.modules) - before)),
    "node_ids": sorted(module.NODE_CLASS_MAPPINGS),
    "package_version": module.__version__,
}, separators=(",", ":"), sort_keys=True))
"""
    # SECURITY: the selected interpreter executes only the fixed local probe against our audited ZIP.
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "-c", probe, str(target)],
        cwd=target.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RegistryArtifactError("normalized Registry installation failed to import")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise RegistryArtifactError("normalized install probe returned invalid output")
    projection = cast(dict[str, object], result)
    expected_version = _read_pyproject(ROOT)["project"]["version"]
    expected_nodes = expected_node_ids(ROOT)
    findings: list[str] = []
    if projection.get("package_version") != expected_version:
        findings.append("install.package_version_mismatch")
    if projection.get("node_ids") != expected_nodes:
        findings.append("install.node_ids_mismatch")
    if projection.get("external_modules_loaded") != []:
        findings.append("install.external_dependency_loaded")
    projection["findings"] = findings
    projection["status"] = "PASS" if not findings else "FAIL"
    return projection


def _http_get_json(url: str) -> tuple[int, object]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "api.comfy.org":
        raise RegistryArtifactError("Registry observation requires the official HTTPS API origin")
    request = urllib.request.Request(  # noqa: S310 -- origin is validated immediately above
        url,
        headers={"User-Agent": "ComfyUI-Sigmax-dry-audit"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            payload: object = json.loads(error.read())
        except json.JSONDecodeError:
            payload = {"error": "non-json response"}
        return error.code, payload


def observe_registry_identity(
    *,
    node_id: str,
    publisher_id: str,
    fetch: _Fetch = _http_get_json,
) -> dict[str, object]:
    """Perform three unauthenticated GET observations; never infer publisher ownership."""

    encoded_node = urllib.parse.quote(node_id, safe="")
    encoded_publisher = urllib.parse.quote(publisher_id, safe="")
    node_status, node_payload = fetch(
        f"{REGISTRY_API}/nodes?node_id={encoded_node}&latest=true&limit=10"
    )
    publisher_status, publisher_payload = fetch(f"{REGISTRY_API}/publishers/{encoded_publisher}")
    availability_status, availability_payload = fetch(
        f"{REGISTRY_API}/publishers/validate?username={encoded_publisher}"
    )
    node_available = (
        node_status == 200
        and isinstance(node_payload, dict)
        and node_payload.get("total") == 0
        and node_payload.get("nodes") == []
    )
    publisher_available = (
        publisher_status == 404
        and availability_status == 200
        and isinstance(availability_payload, dict)
        and availability_payload.get("isAvailable") is True
    )
    findings: list[str] = []
    if not node_available:
        findings.append("registry.node_id_unavailable")
    if not publisher_available:
        findings.append("registry.publisher_state_unverified")
    return {
        "findings": findings,
        "node_id_available": node_available,
        "publication_performed": False,
        "publisher_id_available": publisher_available,
        "publisher_lookup_status": publisher_status,
        "publisher_owned": False,
        "publisher_payload_present": publisher_status == 200
        and isinstance(publisher_payload, dict),
        "status": "PASS" if not findings else "FAIL",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--observe-registry", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    return parser


def _write_manifest() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_bytes(canonical(build_release_manifest(ROOT)) + b"\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_manifest:
        _write_manifest()
    if args.check_manifest:
        expected = canonical(build_release_manifest(ROOT)) + b"\n"
        if not MANIFEST_PATH.is_file() or MANIFEST_PATH.read_bytes() != expected:
            raise SystemExit("Registry release manifest is stale; regenerate it")
    if args.archive is None and args.output is None:
        return 0
    if args.archive is None or args.output is None:
        raise SystemExit("--archive and --output are required together")
    archive_path = args.archive.resolve()
    output_path = args.output.resolve()
    for path in (archive_path, output_path):
        if ROOT.resolve() not in path.parents:
            raise SystemExit("outputs must stay inside the repository")
    archive_report = build_and_audit_registry_archive(ROOT, archive_path)
    temp_root = ROOT / ".tmp"
    temp_root.mkdir(exist_ok=True)
    install_parent = Path(tempfile.mkdtemp(prefix="registry-normalized-", dir=temp_root))
    install_report = probe_normalized_install(
        archive_path, install_parent / "custom_nodes" / "renamed-pack"
    )
    manifest_envelope = _read_json(MANIFEST_PATH)
    manifest_findings = validate_release_manifest(manifest_envelope, ROOT)
    registry_report: dict[str, object] = {
        "findings": [],
        "publication_performed": False,
        "status": "NOT_OBSERVED",
    }
    if args.observe_registry:
        registry_identity = cast(dict[str, str], manifest_envelope["manifest"]["registry"])
        registry_report = observe_registry_identity(
            node_id=registry_identity["node_id"],
            publisher_id=registry_identity["publisher_id"],
        )
    findings = sorted(
        set(cast(list[str], archive_report["findings"]))
        | set(cast(list[str], install_report["findings"]))
        | set(manifest_findings)
        | set(cast(list[str], registry_report["findings"]))
    )
    report: dict[str, object] = {
        "archive": archive_report,
        "findings": findings,
        "manifest_fingerprint": manifest_envelope["manifest_fingerprint"],
        "normalized_install": install_report,
        "publication_performed": False,
        "registry_observation": registry_report,
        "schema": REPORT_SCHEMA,
        "status": "PASS" if not findings else "FAIL",
    }
    envelope = {
        "report": report,
        "report_fingerprint": fingerprint(report),
        "schema": REPORT_ENVELOPE_SCHEMA,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical(envelope) + b"\n")
    print(f"REGISTRY_ARTIFACT={report['status']}")
    print(f"REGISTRY_ARCHIVE_SHA256={archive_report['archive_sha256']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
