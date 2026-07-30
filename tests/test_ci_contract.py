from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CiContractTests(unittest.TestCase):
    def test_orchestration_files_exist(self) -> None:
        for relative_path in (
            "scripts/preflight_check.py",
            "scripts/run_full_gate.py",
            "scripts/run_full_tests_windows.ps1",
            "scripts/run_full_tests_linux.sh",
            ".github/workflows/ci.yml",
        ):
            with self.subTest(path=relative_path):
                self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())

    def test_preflight_passes_from_project_venv(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/preflight_check.py"),
                "--json",
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            result.returncode,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        report = json.loads(result.stdout)
        self.assertEqual("PASS", report["status"])
        self.assertEqual("NOT_APPLICABLE", report["node"])
        self.assertTrue(report["project_local_venv"])

    def test_preflight_fails_with_actionable_wrong_environment_message(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts/preflight_check.py"),
                "--expected-environment",
                str(REPOSITORY_ROOT / ".tmp" / "missing-venv"),
            ],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, result.returncode)
        self.assertIn("Create it and install '.[dev]' before retrying.", result.stderr)

    def test_wrappers_converge_on_common_runner(self) -> None:
        windows = (REPOSITORY_ROOT / "scripts/run_full_tests_windows.ps1").read_text(
            encoding="utf-8"
        )
        linux = (REPOSITORY_ROOT / "scripts/run_full_tests_linux.sh").read_text(encoding="utf-8")
        self.assertIn(".venv\\Scripts\\python.exe", windows)
        self.assertIn("scripts/run_full_gate.py", windows)
        self.assertIn(".venv-wsl/bin/python", linux)
        self.assertIn("scripts/run_full_gate.py", linux)

    def test_common_runner_declares_required_ordered_stages(self) -> None:
        runner = (REPOSITORY_ROOT / "scripts/run_full_gate.py").read_text(encoding="utf-8")
        positions = [
            runner.index(f'"{stage}"')
            for stage in (
                "preflight",
                "detect-secrets",
                "pre-commit",
                "ruff-format",
                "ruff-lint",
                "mypy",
                "core-independence",
                "pytest",
                "coverage",
                "package",
            )
        ]
        self.assertEqual(sorted(positions), positions)
        self.assertIn('"browser_e2e": "NOT_APPLICABLE"', runner)
        self.assertIn('"comfyui_host_e2e": "NOT_IMPLEMENTED"', runner)
        self.assertIn('"comfyui_sigmax/core/__init__.py"', runner)
        self.assertIn('"comfyui_sigmax/core/base_grids.py"', runner)
        self.assertIn('"comfyui_sigmax/core/schedule_contracts.py"', runner)
        self.assertIn('"comfyui_sigmax/core/request_result.py"', runner)
        self.assertIn('"comfyui_sigmax/core/shifts.py"', runner)
        self.assertIn('"comfyui_sigmax/core/terminal_slicing.py"', runner)
        self.assertIn('"comfyui_sigmax/core/artifacts.py"', runner)
        self.assertIn('"comfyui_sigmax/core/capabilities.py"', runner)
        self.assertIn('"comfyui_sigmax/core/fingerprints.py"', runner)
        self.assertIn('"comfyui_sigmax/core/validation.py"', runner)
        self.assertIn('"core_independence": "IMPLEMENTED"', runner)
        self.assertIn('"property": "IMPLEMENTED"', runner)

    def test_workflow_contract(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertIn(
            "de0fac2e4500dabe0009e67214ff5f5447ce83dd",  # pragma: allowlist secret
            workflow,
        )
        self.assertIn(
            "a309ff8b426b58ec0e2a45f0f869d46889d02405",  # pragma: allowlist secret
            workflow,
        )
        self.assertIn("ubuntu-latest", workflow)
        self.assertIn("windows-latest", workflow)
        self.assertIn('"3.10"', workflow)
        self.assertIn('"3.13"', workflow)
        self.assertIn("scripts/run_full_tests_windows.ps1", workflow)
        self.assertIn("scripts/run_full_tests_linux.sh", workflow)
        lowered = workflow.lower()
        for forbidden in ("npm ", "node ", "playwright", "permissions: write-all"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
