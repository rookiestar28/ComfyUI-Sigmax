from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from scripts.environment_diagnostics import (
    EnvironmentObservation,
    EnvironmentProbeConfig,
    collect_environment_observation,
    evaluate_environment,
)
from scripts.run_full_gate import _isolated_tool_path

ROOT = Path(__file__).resolve().parents[1]


def _passing_observation() -> EnvironmentObservation:
    return EnvironmentObservation(
        platform="windows",
        local_venv=True,
        cache_contained=True,
        cache_writable=True,
        cache_integrity="ok",
        precommit_candidate_count=0,
        precommit_foreign=False,
        filesystem_status="ok",
        unicode_roundtrip=True,
        temp_contained=True,
        temp_writable=True,
        anonymous_temp_compatible=True,
        temp_capture_mitigated=False,
        optional_lane="none",
        missing_optional_modules=(),
    )


@pytest.mark.parametrize(
    ("observation", "code", "remediation"),
    [
        (
            replace(_passing_observation(), local_venv=False),
            "venv.not_local",
            'install -e ".[dev]"',
        ),
        (
            replace(_passing_observation(), cache_contained=False),
            "cache.outside_workspace",
            "PRE_COMMIT_HOME",
        ),
        (
            replace(_passing_observation(), cache_writable=False),
            "cache.not_writable",
            ".tmp/pre-commit-win",
        ),
        (
            replace(_passing_observation(), cache_integrity="corrupt"),
            "cache.corrupt",
            "rebuild",
        ),
        (
            replace(_passing_observation(), cache_integrity="locked"),
            "cache.locked",
            "Close",
        ),
        (
            replace(
                _passing_observation(),
                precommit_candidate_count=2,
                precommit_foreign=True,
            ),
            "tooling.precommit_conflict",
            "python -m pre_commit",
        ),
        (
            replace(_passing_observation(), filesystem_status="locked"),
            "filesystem.locked",
            "antivirus",
        ),
        (
            replace(_passing_observation(), filesystem_status="cleanup_failed"),
            "filesystem.cleanup_failed",
            "probe directory",
        ),
        (
            replace(_passing_observation(), unicode_roundtrip=False),
            "unicode.roundtrip_failed",
            "Unicode-capable",
        ),
        (
            replace(_passing_observation(), temp_contained=False),
            "temp.outside_workspace",
            "SIGMAX_TEMP_ROOT",
        ),
        (
            replace(_passing_observation(), temp_writable=False),
            "temp.not_writable",
            ".tmp/runtime-win",
        ),
        (
            replace(_passing_observation(), anonymous_temp_compatible=False),
            "temp.incompatible",
            "SIGMAX_PYTEST_CAPTURE_MODE=sys",
        ),
        (
            replace(
                _passing_observation(),
                optional_lane="reference",
                missing_optional_modules=("diffusers",),
            ),
            "optional.missing",
            ".[reference]",
        ),
    ],
)
def test_each_failure_has_a_stable_code_and_actionable_remediation(
    observation: EnvironmentObservation,
    code: str,
    remediation: str,
) -> None:
    report = evaluate_environment(observation)

    assert report.status == "FAIL"
    issue = next(item for item in report.issues if item.code == code)
    assert remediation in issue.remediation


def test_passing_report_is_deterministic_and_machine_readable() -> None:
    report = evaluate_environment(_passing_observation())

    assert report.status == "PASS"
    assert report.issues == ()
    assert report.projection() == report.projection()
    assert report.projection()["schema"] == "sigmax.environment-diagnostics/1"
    json.dumps(report.projection(), allow_nan=False, sort_keys=True)


def test_live_unicode_temp_cache_and_file_operations_pass_and_clean_up(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "專案-路徑"
    expected_venv = repository / ".venv"
    cache = repository / ".tmp" / "pre-commit-win"
    runtime = repository / ".tmp" / "runtime-win"
    expected_venv.mkdir(parents=True)

    observation = collect_environment_observation(
        EnvironmentProbeConfig(
            repository_root=repository,
            environment_prefix=expected_venv,
            expected_environment=expected_venv,
            cache_root=cache,
            temp_root=runtime,
            platform="windows",
            optional_lane="none",
            path_value="",
            temp_capture_mitigated=os.name != "nt",
        )
    )

    assert evaluate_environment(observation).status == "PASS"
    assert list(runtime.iterdir()) == []


def test_invalid_precommit_sqlite_database_is_detected_read_only(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    expected_venv = repository / ".venv-wsl"
    cache = repository / ".tmp" / "pre-commit-linux"
    runtime = repository / ".tmp" / "runtime-linux"
    expected_venv.mkdir(parents=True)
    cache.mkdir(parents=True)
    database = cache / "db.db"
    database.write_bytes(b"not a sqlite database")
    before = database.read_bytes()

    observation = collect_environment_observation(
        EnvironmentProbeConfig(
            repository_root=repository,
            environment_prefix=expected_venv,
            expected_environment=expected_venv,
            cache_root=cache,
            temp_root=runtime,
            platform="linux",
            optional_lane="none",
            path_value="",
        )
    )

    assert "cache.corrupt" in {item.code for item in evaluate_environment(observation).issues}
    assert database.read_bytes() == before


def test_locked_precommit_sqlite_database_is_detected(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    expected_venv = repository / ".venv"
    cache = repository / ".tmp" / "pre-commit-win"
    expected_venv.mkdir(parents=True)
    cache.mkdir(parents=True)
    database = cache / "db.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE probe (value INTEGER)")
    connection.commit()
    connection.execute("BEGIN EXCLUSIVE")
    try:
        observation = collect_environment_observation(
            EnvironmentProbeConfig(
                repository_root=repository,
                environment_prefix=expected_venv,
                expected_environment=expected_venv,
                cache_root=cache,
                temp_root=repository / ".tmp" / "runtime-win",
                platform="windows",
                optional_lane="none",
                path_value="",
            )
        )
    finally:
        connection.rollback()
        connection.close()

    assert "cache.locked" in {item.code for item in evaluate_environment(observation).issues}


def test_lock_like_file_failure_is_injected_through_the_live_collector(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    expected_venv = repository / ".venv"
    expected_venv.mkdir(parents=True)

    observation = collect_environment_observation(
        EnvironmentProbeConfig(
            repository_root=repository,
            environment_prefix=expected_venv,
            expected_environment=expected_venv,
            cache_root=repository / ".tmp" / "pre-commit-win",
            temp_root=repository / ".tmp" / "runtime-win",
            platform="windows",
            optional_lane="none",
            path_value="",
        ),
        file_operation_probe=lambda _path: ("locked", True, False, True),
    )

    assert "filesystem.locked" in {item.code for item in evaluate_environment(observation).issues}


def test_outside_temp_path_is_rejected_without_creation(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    expected_venv = repository / ".venv-wsl"
    outside = tmp_path / "outside-temp"
    expected_venv.mkdir(parents=True)

    observation = collect_environment_observation(
        EnvironmentProbeConfig(
            repository_root=repository,
            environment_prefix=expected_venv,
            expected_environment=expected_venv,
            cache_root=repository / ".tmp" / "pre-commit-linux",
            temp_root=outside,
            platform="linux",
            optional_lane="none",
            path_value="",
        )
    )

    assert outside.exists() is False
    assert "temp.outside_workspace" in {
        item.code for item in evaluate_environment(observation).issues
    }


def test_optional_lane_missing_module_is_injectable_without_importing_it(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    expected_venv = repository / ".venv-wsl"
    expected_venv.mkdir(parents=True)
    observation = collect_environment_observation(
        EnvironmentProbeConfig(
            repository_root=repository,
            environment_prefix=expected_venv,
            expected_environment=expected_venv,
            cache_root=repository / ".tmp" / "pre-commit-linux",
            temp_root=repository / ".tmp" / "runtime-linux",
            platform="linux",
            optional_lane="reference",
            path_value="",
        ),
        module_available=lambda _name: False,
    )

    assert observation.missing_optional_modules == ("diffusers",)
    assert "optional.missing" in {item.code for item in evaluate_environment(observation).issues}


def test_live_probe_detects_foreign_and_duplicate_precommit_launchers(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    expected_venv = repository / ".venv"
    local_bin = expected_venv / ("Scripts" if os.name == "nt" else "bin")
    foreign_bin = tmp_path / "foreign-bin"
    local_bin.mkdir(parents=True)
    foreign_bin.mkdir()
    executable = "pre-commit.exe" if os.name == "nt" else "pre-commit"
    (local_bin / executable).write_text("local", encoding="utf-8")
    (foreign_bin / executable).write_text("foreign", encoding="utf-8")

    observation = collect_environment_observation(
        EnvironmentProbeConfig(
            repository_root=repository,
            environment_prefix=expected_venv,
            expected_environment=expected_venv,
            cache_root=repository / ".tmp" / "pre-commit-win",
            temp_root=repository / ".tmp" / "runtime-win",
            platform="windows" if os.name == "nt" else "linux",
            optional_lane="none",
            path_value=os.pathsep.join((str(local_bin), str(foreign_bin))),
        )
    )

    assert observation.precommit_candidate_count == 2
    assert observation.precommit_foreign is True
    assert "tooling.precommit_conflict" in {
        item.code for item in evaluate_environment(observation).issues
    }


def test_preflight_optional_lane_failure_is_actionable_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/preflight_check.py",
            "--json",
            "--optional-lane",
            "reference",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    issues = report["diagnostics"]["issues"]
    assert any(item["code"] == "optional.missing" for item in issues)
    assert any(".[reference]" in item["remediation"] for item in issues)


def test_full_gate_exports_repo_local_cache_and_temp_roots() -> None:
    runner = (ROOT / "scripts" / "run_full_gate.py").read_text(encoding="utf-8")

    assert 'environment["PRE_COMMIT_HOME"]' in runner
    assert 'environment["SIGMAX_TEMP_ROOT"]' in runner
    for variable in ("TMPDIR", "TMP", "TEMP"):
        assert f'environment["{variable}"]' in runner
    assert 'environment["SIGMAX_PYTEST_CAPTURE_MODE"] = "sys"' in runner


def test_mounted_temp_incompatibility_can_be_explicitly_mitigated() -> None:
    observation = replace(
        _passing_observation(),
        platform="linux",
        anonymous_temp_compatible=False,
        temp_capture_mitigated=True,
    )

    report = evaluate_environment(observation)
    assert report.status == "PASS"
    assert report.mitigations == ("pytest.capture_sys",)


def test_full_gate_path_keeps_selected_venv_and_removes_foreign_precommit(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected" / ("Scripts" if os.name == "nt" else "bin")
    foreign = tmp_path / "foreign"
    system = tmp_path / "system"
    selected.mkdir(parents=True)
    foreign.mkdir()
    system.mkdir()
    executable = "pre-commit.exe" if os.name == "nt" else "pre-commit"
    (selected / executable).write_text("selected", encoding="utf-8")
    (foreign / executable).write_text("foreign", encoding="utf-8")

    isolated = _isolated_tool_path(
        str(selected / ("python.exe" if os.name == "nt" else "python")),
        os.pathsep.join((str(foreign), str(system))),
    ).split(os.pathsep)

    assert isolated == [str(selected.resolve()), str(system)]
