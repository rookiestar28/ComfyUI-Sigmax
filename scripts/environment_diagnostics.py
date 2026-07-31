"""Dependency-free Windows/WSL environment diagnostics for repository gates."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import quote

DIAGNOSTIC_SCHEMA: Final = "sigmax.environment-diagnostics/1"
PlatformId = Literal["windows", "linux"]
CacheIntegrity = Literal["ok", "corrupt", "locked"]
FilesystemStatus = Literal["ok", "locked", "cleanup_failed"]
OptionalLane = Literal["none", "plot", "reference"]

_OPTIONAL_MODULES: Final[dict[OptionalLane, tuple[str, ...]]] = {
    "none": (),
    "plot": ("matplotlib",),
    "reference": ("diffusers",),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentIssue:
    """One stable environment failure with a direct recovery action."""

    code: str
    summary: str
    remediation: str

    def projection(self) -> dict[str, str]:
        return {
            "code": self.code,
            "remediation": self.remediation,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentObservation:
    """Pure observations consumed by the policy evaluator."""

    platform: PlatformId
    local_venv: bool
    cache_contained: bool
    cache_writable: bool
    cache_integrity: CacheIntegrity
    precommit_candidate_count: int
    precommit_foreign: bool
    filesystem_status: FilesystemStatus
    unicode_roundtrip: bool
    temp_contained: bool
    temp_writable: bool
    anonymous_temp_compatible: bool
    temp_capture_mitigated: bool
    optional_lane: OptionalLane
    missing_optional_modules: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentDiagnosticReport:
    """Deterministic diagnostic result without private absolute paths."""

    platform: PlatformId
    optional_lane: OptionalLane
    issues: tuple[EnvironmentIssue, ...]
    mitigations: tuple[str, ...]

    @property
    def status(self) -> str:
        return "FAIL" if self.issues else "PASS"

    def projection(self) -> dict[str, object]:
        return {
            "issues": [issue.projection() for issue in self.issues],
            "mitigations": list(self.mitigations),
            "optional_lane": self.optional_lane,
            "platform": self.platform,
            "schema": DIAGNOSTIC_SCHEMA,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentProbeConfig:
    """Resolved runtime inputs for a bounded live probe."""

    repository_root: Path
    environment_prefix: Path
    expected_environment: Path
    cache_root: Path
    temp_root: Path
    platform: PlatformId
    optional_lane: OptionalLane
    path_value: str
    temp_capture_mitigated: bool = False


def _venv_label(platform: PlatformId) -> str:
    return ".venv" if platform == "windows" else ".venv-wsl"


def _cache_label(platform: PlatformId) -> str:
    return ".tmp/pre-commit-win" if platform == "windows" else ".tmp/pre-commit-linux"


def _temp_label(platform: PlatformId) -> str:
    return ".tmp/runtime-win" if platform == "windows" else ".tmp/runtime-linux"


def _venv_install_command(platform: PlatformId) -> str:
    if platform == "windows":
        return 'python -m venv .venv; .venv\\Scripts\\python.exe -m pip install -e ".[dev]"'
    return "python3 -m venv .venv-wsl && .venv-wsl/bin/python -m pip install -e '.[dev]'"


def evaluate_environment(observation: EnvironmentObservation) -> EnvironmentDiagnosticReport:
    """Classify collected observations into stable actionable failures."""

    issues: list[EnvironmentIssue] = []
    cache_label = _cache_label(observation.platform)
    temp_label = _temp_label(observation.platform)

    if not observation.local_venv:
        issues.append(
            EnvironmentIssue(
                code="venv.not_local",
                summary=f"The active interpreter is not owned by {_venv_label(observation.platform)}.",
                remediation=(
                    "Create it and install '.[dev]' before retrying. Run: "
                    f"{_venv_install_command(observation.platform)}."
                ),
            )
        )
    if not observation.cache_contained:
        issues.append(
            EnvironmentIssue(
                code="cache.outside_workspace",
                summary="The configured pre-commit cache is outside repository .tmp.",
                remediation=(
                    f"Set PRE_COMMIT_HOME to {cache_label} and rerun with the local venv."
                ),
            )
        )
    elif not observation.cache_writable:
        issues.append(
            EnvironmentIssue(
                code="cache.not_writable",
                summary="The repository-local pre-commit cache is not writable.",
                remediation=(
                    f"Close hook processes, restore write permission on {cache_label}, and retry."
                ),
            )
        )
    if observation.cache_integrity == "corrupt":
        issues.append(
            EnvironmentIssue(
                code="cache.corrupt",
                summary="The repository-local pre-commit SQLite manifest is corrupt.",
                remediation=(
                    f"Close pre-commit, remove only {cache_label}, then rerun to rebuild the cache."
                ),
            )
        )
    elif observation.cache_integrity == "locked":
        issues.append(
            EnvironmentIssue(
                code="cache.locked",
                summary="The repository-local pre-commit SQLite manifest is locked.",
                remediation=(
                    "Close other pre-commit/Python processes, verify antivirus exclusions for "
                    f"{cache_label}, and retry serially."
                ),
            )
        )
    if observation.precommit_candidate_count > 1 or observation.precommit_foreign:
        issues.append(
            EnvironmentIssue(
                code="tooling.precommit_conflict",
                summary="PATH exposes multiple or non-venv pre-commit executables.",
                remediation=(
                    "Remove conflicting PATH entries and run the selected local interpreter with "
                    "python -m pre_commit."
                ),
            )
        )
    if observation.filesystem_status == "locked":
        issues.append(
            EnvironmentIssue(
                code="filesystem.locked",
                summary="The gate cannot complete an owned file write/replace/delete cycle.",
                remediation=(
                    "Close processes holding the file, inspect antivirus or sync-tool locks, restore "
                    "permissions, and rerun the preflight."
                ),
            )
        )
    elif observation.filesystem_status == "cleanup_failed":
        issues.append(
            EnvironmentIssue(
                code="filesystem.cleanup_failed",
                summary="The owned environment probe directory could not be removed.",
                remediation=(
                    f"Close file handles, remove only the probe directory below {temp_label}, and retry."
                ),
            )
        )
    if not observation.unicode_roundtrip:
        issues.append(
            EnvironmentIssue(
                code="unicode.roundtrip_failed",
                summary="The filesystem failed a non-ASCII create/read/rename round-trip.",
                remediation=(
                    "Use a Unicode-capable filesystem and UTF-8 locale, then recreate the local venv "
                    "and rerun."
                ),
            )
        )
    if not observation.temp_contained:
        issues.append(
            EnvironmentIssue(
                code="temp.outside_workspace",
                summary="The configured gate temp directory is outside repository .tmp.",
                remediation=f"Set SIGMAX_TEMP_ROOT, TMPDIR, TMP, and TEMP to {temp_label}.",
            )
        )
    elif not observation.temp_writable:
        issues.append(
            EnvironmentIssue(
                code="temp.not_writable",
                summary="The repository-local gate temp directory is not writable.",
                remediation=(
                    f"Create {temp_label}, restore write permission, and rerun the local wrapper."
                ),
            )
        )
    if not observation.anonymous_temp_compatible and not observation.temp_capture_mitigated:
        issues.append(
            EnvironmentIssue(
                code="temp.incompatible",
                summary="The configured temp path cannot sustain anonymous file operations.",
                remediation=(
                    "On WSL mounted workspaces, run the canonical Linux wrapper so pytest uses "
                    "SIGMAX_PYTEST_CAPTURE_MODE=sys; otherwise choose a compatible writable temp path."
                ),
            )
        )
    if observation.missing_optional_modules:
        modules = ", ".join(observation.missing_optional_modules)
        issues.append(
            EnvironmentIssue(
                code="optional.missing",
                summary=f"Optional lane {observation.optional_lane} is missing: {modules}.",
                remediation=(
                    "Install the reviewed extra with the selected local interpreter: "
                    f'python -m pip install -e ".[{observation.optional_lane}]".'
                ),
            )
        )
    mitigations = (
        ("pytest.capture_sys",)
        if not observation.anonymous_temp_compatible and observation.temp_capture_mitigated
        else ()
    )
    return EnvironmentDiagnosticReport(
        platform=observation.platform,
        optional_lane=observation.optional_lane,
        issues=tuple(issues),
        mitigations=mitigations,
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _probe_writable_directory(path: Path) -> bool:
    probe: Path | None = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".sigmax-write-", dir=path)
        os.close(descriptor)
        probe = Path(name)
        probe.write_bytes(b"sigmax")
        return probe.read_bytes() == b"sigmax"
    except OSError:
        return False
    finally:
        if probe is not None:
            with suppress(OSError):
                probe.unlink(missing_ok=True)


def _cache_integrity(cache_root: Path) -> CacheIntegrity:
    database = cache_root / "db.db"
    if not database.is_file():
        return "ok"
    # IMPORTANT: use SQLite read-only mode; diagnostics must never repair or recreate a cache DB.
    uri = f"file:{quote(str(database.resolve()))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=0)
        try:
            result = connection.execute("PRAGMA quick_check").fetchone()
        finally:
            connection.close()
    except sqlite3.Error as error:
        lowered = str(error).lower()
        return "locked" if "locked" in lowered or "busy" in lowered else "corrupt"
    return "ok" if result == ("ok",) else "corrupt"


def _find_precommit_candidates(path_value: str, platform: PlatformId) -> tuple[Path, ...]:
    names = (
        ("pre-commit.exe", "pre-commit.cmd", "pre-commit.bat", "pre-commit")
        if platform == "windows"
        else ("pre-commit",)
    )
    found: dict[str, Path] = {}
    for raw_directory in path_value.split(os.pathsep):
        if not raw_directory:
            continue
        directory = Path(raw_directory)
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                try:
                    resolved = candidate.resolve()
                except OSError:
                    resolved = candidate.absolute()
                found[os.path.normcase(str(resolved))] = resolved
    return tuple(found[key] for key in sorted(found))


def _probe_file_operations(temp_root: Path) -> tuple[FilesystemStatus, bool, bool, bool]:
    probe_directory: Path | None = None
    source: Path | None = None
    target: Path | None = None
    filesystem_status: FilesystemStatus = "ok"
    unicode_roundtrip = True
    temp_writable = True
    anonymous_temp_compatible = True
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        probe_directory = Path(tempfile.mkdtemp(prefix=".sigmax-environment-", dir=temp_root))
        source = probe_directory / "probe.txt"
        target = probe_directory / "環境-探測.txt"
        source.write_text("Sigmax 路徑", encoding="utf-8")
        if source.read_text(encoding="utf-8") != "Sigmax 路徑":
            unicode_roundtrip = False
        source.replace(target)
        if target.read_text(encoding="utf-8") != "Sigmax 路徑":
            unicode_roundtrip = False
        target.unlink()
        target = None
        try:
            with tempfile.TemporaryFile(dir=temp_root) as anonymous:
                anonymous.write(b"sigmax")
                anonymous.seek(0)
                if anonymous.read() != b"sigmax":
                    anonymous_temp_compatible = False
                anonymous.truncate()
        except OSError:
            anonymous_temp_compatible = False
    except UnicodeError:
        unicode_roundtrip = False
    except OSError:
        filesystem_status = "locked"
        temp_writable = False
    finally:
        cleanup_failed = False
        for candidate in (target, source):
            if candidate is not None:
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    cleanup_failed = True
        if probe_directory is not None:
            try:
                probe_directory.rmdir()
            except OSError:
                cleanup_failed = True
        if cleanup_failed:
            filesystem_status = "cleanup_failed"
    return filesystem_status, unicode_roundtrip, temp_writable, anonymous_temp_compatible


def collect_environment_observation(
    config: EnvironmentProbeConfig,
    *,
    module_available: Callable[[str], bool] | None = None,
    file_operation_probe: Callable[[Path], tuple[FilesystemStatus, bool, bool, bool]] | None = None,
) -> EnvironmentObservation:
    """Collect bounded observations without importing optional modules."""

    owned_root = config.repository_root / ".tmp"
    cache_contained = _is_within(config.cache_root, owned_root)
    temp_contained = _is_within(config.temp_root, owned_root)
    # CRITICAL: never touch cache/temp paths until containment below repository .tmp is proven.
    cache_writable = cache_contained and _probe_writable_directory(config.cache_root)
    cache_integrity: CacheIntegrity = (
        _cache_integrity(config.cache_root) if cache_contained else "ok"
    )
    if temp_contained:
        probe = file_operation_probe or _probe_file_operations
        (
            filesystem_status,
            unicode_roundtrip,
            temp_writable,
            anonymous_temp_compatible,
        ) = probe(config.temp_root)
    else:
        filesystem_status = "ok"
        unicode_roundtrip = True
        temp_writable = False
        anonymous_temp_compatible = True

    candidates = _find_precommit_candidates(config.path_value, config.platform)
    precommit_foreign = any(
        not _is_within(candidate, config.expected_environment) for candidate in candidates
    )
    available = module_available or (lambda name: importlib.util.find_spec(name) is not None)
    missing_optional = tuple(
        name for name in _OPTIONAL_MODULES[config.optional_lane] if not available(name)
    )
    local_venv = _is_within(config.environment_prefix, config.expected_environment)

    return EnvironmentObservation(
        platform=config.platform,
        local_venv=local_venv,
        cache_contained=cache_contained,
        cache_writable=cache_writable,
        cache_integrity=cache_integrity,
        precommit_candidate_count=len(candidates),
        precommit_foreign=precommit_foreign,
        filesystem_status=filesystem_status,
        unicode_roundtrip=unicode_roundtrip,
        temp_contained=temp_contained,
        temp_writable=temp_writable,
        anonymous_temp_compatible=anonymous_temp_compatible,
        temp_capture_mitigated=config.temp_capture_mitigated,
        optional_lane=config.optional_lane,
        missing_optional_modules=missing_optional,
    )
