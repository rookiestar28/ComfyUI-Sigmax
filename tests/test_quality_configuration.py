from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DEV_DEPENDENCIES = [
    "build>=1.5,<1.6",
    "detect-secrets>=1.5,<1.6",
    "mypy>=2.3,<2.4",
    "pre-commit>=4.6,<4.7",
    "pytest>=9.1,<9.2",
    "pytest-cov>=7.1,<7.2",
    "ruff>=0.16,<0.17",
    "tomli>=2.4,<2.5",
]

EXPECTED_HOOK_REVISIONS = {
    "https://github.com/pre-commit/pre-commit-hooks": (
        "3e8a8703264a2f4a69428a0aa4dcb512790b2c8c"  # pragma: allowlist secret
    ),
    "https://github.com/Yelp/detect-secrets": (
        "68e8b45440415753fff70a312ece8da92ba85b4a"  # pragma: allowlist secret
    ),
    "https://github.com/astral-sh/ruff-pre-commit": (
        "cb8c523fd4835aba42af70f4cad5568db4df0b6c"  # pragma: allowlist secret
    ),
    "https://github.com/pre-commit/mirrors-mypy": (
        "41e691678310dfd3833f7ab4e180ddb014310356"  # pragma: allowlist secret
    ),
}


class QualityConfigurationTests(unittest.TestCase):
    def test_pyproject_quality_contract(self) -> None:
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as stream:
            metadata = tomllib.load(stream)

        project = metadata["project"]
        self.assertEqual([], project["dependencies"])
        self.assertEqual(
            EXPECTED_DEV_DEPENDENCIES,
            project["optional-dependencies"]["dev"],
        )

        tools = metadata["tool"]
        self.assertEqual("py310", tools["ruff"]["target-version"])
        self.assertEqual(
            ["E4", "E7", "E9", "F", "I", "UP", "B", "S", "SIM", "RUF"],
            tools["ruff"]["lint"]["select"],
        )
        self.assertTrue(tools["mypy"]["strict"])
        self.assertEqual("3.10", tools["mypy"]["python_version"])
        self.assertTrue(tools["pytest"]["ini_options"]["strict_config"])
        self.assertTrue(tools["pytest"]["ini_options"]["strict_markers"])
        self.assertTrue(tools["coverage"]["run"]["branch"])
        self.assertEqual(["comfyui_sigmax"], tools["coverage"]["run"]["source"])

    def test_pre_commit_contract_uses_immutable_revisions(self) -> None:
        config_path = REPOSITORY_ROOT / ".pre-commit-config.yaml"
        self.assertTrue(config_path.is_file())
        config = config_path.read_text(encoding="utf-8")

        for repository, revision in EXPECTED_HOOK_REVISIONS.items():
            with self.subTest(repository=repository):
                block_pattern = (
                    rf"repo:\s*{re.escape(repository)}"
                    rf"[\s\S]*?rev:\s*{revision}"
                )
                self.assertRegex(config, block_pattern)

        for hook_id in (
            "check-ast",
            "check-toml",
            "check-yaml",
            "detect-secrets",
            "ruff-check",
            "ruff-format",
            "mypy",
        ):
            with self.subTest(hook_id=hook_id):
                self.assertRegex(config, rf"id:\s*{re.escape(hook_id)}(?:\s|$)")

        self.assertRegex(
            config,
            r"id:\s*mypy[\s\S]*?additional_dependencies:"
            r"[\s\S]*?pytest==9\.1\.1",
        )

    def test_secret_baseline_is_valid_and_reviewable(self) -> None:
        baseline_path = REPOSITORY_ROOT / ".secrets.baseline"
        self.assertTrue(baseline_path.is_file())
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        self.assertEqual("1.5.0", baseline["version"])
        self.assertIsInstance(baseline["plugins_used"], list)
        self.assertIsInstance(baseline["filters_used"], list)
        self.assertEqual({}, baseline["results"])

    def test_governance_documents_exist_and_are_current(self) -> None:
        expected_paths = [
            "tests/TEST_SOP.md",
            "tests/E2E_TESTING_NOTICE.md",
            "tests/E2E_TESTING_SOP.md",
            "tests/CI_TEST_MATRIX.md",
        ]
        for relative_path in expected_paths:
            with self.subTest(path=relative_path):
                document = REPOSITORY_ROOT / relative_path
                self.assertTrue(document.is_file())
                self.assertIn("ComfyUI-Sigmax", document.read_text(encoding="utf-8"))

        test_sop = (REPOSITORY_ROOT / "tests/TEST_SOP.md").read_text(encoding="utf-8")
        self.assertIn(
            "cross-platform full-gate wrappers, and CI workflow",
            test_sop,
        )
        self.assertIn(
            "The ComfyUI host fixture, numerical core, golden/parity suites, and product nodes "
            "do not yet",
            test_sop,
        )


if __name__ == "__main__":
    unittest.main()
