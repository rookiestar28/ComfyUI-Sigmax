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
        normalized_results = {
            path.replace("\\", "/"): [
                {
                    **finding,
                    "filename": finding["filename"].replace("\\", "/"),
                }
                for finding in findings
            ]
            for path, findings in baseline["results"].items()
        }
        expected_hash = "024469e0164c7a1285a3177a3ab35c7b110d39b9"  # pragma: allowlist secret
        head_hash = "ba2d9a512ac48100b11ca25836a795bc97546b8a"  # pragma: allowlist secret
        release_hash = "985aa069eea4d28101857c9f25efd3f7574c971c"  # pragma: allowlist secret
        anima_repository_hash = (
            "4e88ff4cb19447d3a5423e07847e8f8dab9f0b6b"  # pragma: allowlist secret
        )
        anima_diffusers_hash = (
            "f58458136cb7e75d5ca36d42759b1a3bad4dd74f"  # pragma: allowlist secret
        )
        wan21_repository_hash = (
            "5c2a1ae717b4a560a897024cd35315259a766130"  # pragma: allowlist secret
        )
        wan22_repository_hash = (
            "82646227edf451d1da04d45e198a1071c461a6c1"  # pragma: allowlist secret
        )
        wan_comfyui_hash = (
            "ba2d9a512ac48100b11ca25836a795bc97546b8a"  # pragma: allowlist secret
        )
        wan_diffusers_hash = (
            "ef2388a309fa35d5401f5952a447ca13a96ee801"  # pragma: allowlist secret
        )
        self.assertEqual(
            {
                "comfyui_sigmax/workflows/host_baseline.json": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "comfyui_sigmax/workflows/host_baseline.json",
                        "hashed_secret": expected_hash,
                        "is_verified": False,
                        "line_number": 4,
                    }
                ],
                "comfyui_sigmax/workflows/validation.py": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "comfyui_sigmax/workflows/validation.py",
                        "hashed_secret": expected_hash,
                        "is_verified": False,
                        "line_number": 37,
                    }
                ],
                "tests/compatibility/fixtures/comfyui_head_v1.json": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/compatibility/fixtures/comfyui_head_v1.json",
                        "hashed_secret": head_hash,
                        "is_verified": False,
                        "line_number": 1,
                    }
                ],
                "tests/compatibility/fixtures/comfyui_release_v0292_v1.json": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": ("tests/compatibility/fixtures/comfyui_release_v0292_v1.json"),
                        "hashed_secret": release_hash,
                        "is_verified": False,
                        "line_number": 1,
                    }
                ],
                "tests/compatibility/fixtures/dependency_compatibility_evidence_v1.json": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": (
                            "tests/compatibility/fixtures/dependency_compatibility_evidence_v1.json"
                        ),
                        "hashed_secret": head_hash,
                        "is_verified": False,
                        "line_number": 52,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": (
                            "tests/compatibility/fixtures/dependency_compatibility_evidence_v1.json"
                        ),
                        "hashed_secret": release_hash,
                        "is_verified": False,
                        "line_number": 72,
                    },
                ],
                "tests/golden/anima_v1.json": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/anima_v1.json",
                        "hashed_secret": anima_repository_hash,
                        "is_verified": False,
                        "line_number": 4,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/anima_v1.json",
                        "hashed_secret": anima_diffusers_hash,
                        "is_verified": False,
                        "line_number": 5,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/anima_v1.json",
                        "hashed_secret": head_hash,
                        "is_verified": False,
                        "line_number": 6,
                    },
                ],
                "tests/golden/test_anima_phase0_goldens.py": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/test_anima_phase0_goldens.py",
                        "hashed_secret": anima_repository_hash,
                        "is_verified": False,
                        "line_number": 30,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/test_anima_phase0_goldens.py",
                        "hashed_secret": anima_diffusers_hash,
                        "is_verified": False,
                        "line_number": 31,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/test_anima_phase0_goldens.py",
                        "hashed_secret": head_hash,
                        "is_verified": False,
                        "line_number": 32,
                    },
                ],
                "tests/golden/wan_v1.json": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/wan_v1.json",
                        "hashed_secret": wan21_repository_hash,
                        "is_verified": False,
                        "line_number": 1,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/wan_v1.json",
                        "hashed_secret": wan22_repository_hash,
                        "is_verified": False,
                        "line_number": 1,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/wan_v1.json",
                        "hashed_secret": wan_comfyui_hash,
                        "is_verified": False,
                        "line_number": 1,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/golden/wan_v1.json",
                        "hashed_secret": wan_diffusers_hash,
                        "is_verified": False,
                        "line_number": 1,
                    },
                ],
                "tests/parity/test_anima_phase0_parity.py": [
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/parity/test_anima_phase0_parity.py",
                        "hashed_secret": anima_repository_hash,
                        "is_verified": False,
                        "line_number": 42,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/parity/test_anima_phase0_parity.py",
                        "hashed_secret": anima_diffusers_hash,
                        "is_verified": False,
                        "line_number": 43,
                    },
                    {
                        "type": "Hex High Entropy String",
                        "filename": "tests/parity/test_anima_phase0_parity.py",
                        "hashed_secret": head_hash,
                        "is_verified": False,
                        "line_number": 44,
                    },
                ],
            },
            normalized_results,
        )

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
            "real-ComfyUI H1 harness",
            test_sop,
        )
        self.assertIn(
            "M2-05 strict Turbo and M3-06 RAW H2 workflow lanes now exist",
            test_sop,
        )
        self.assertIn("Krea 2 RAW golden-vector", test_sop)
        self.assertIn("M5-01 deterministic native-Euler H3 proof lane now exists", test_sop)
        self.assertIn("Partial-denoise execution is rejected", test_sop)
        self.assertIn(
            "Remaining sampler H3 capabilities and GPU H4 remain owned",
            test_sop,
        )
        self.assertIn("Validated product nodes, adapter/integration tests", test_sop)
        self.assertIn("core-independence", test_sop)
        self.assertIn("Turbo golden-vector", test_sop)

        ci_matrix = (REPOSITORY_ROOT / "tests" / "CI_TEST_MATRIX.md").read_text(encoding="utf-8")
        self.assertIn("| Pure-core tests | Implemented", ci_matrix)
        self.assertIn("| Deterministic property tests | Implemented", ci_matrix)
        self.assertIn("| Krea 2 Turbo golden vectors | Implemented", ci_matrix)
        self.assertIn("| Real ComfyUI H1 | Implemented", ci_matrix)
        self.assertIn(
            "| Real ComfyUI H2 | Turbo and RAW/auto implemented",
            ci_matrix,
        )
        self.assertIn(
            "| Real ComfyUI H3 | M5-01 deterministic native-Euler H3 implemented", ci_matrix
        )

        e2e_sop = (REPOSITORY_ROOT / "tests/E2E_TESTING_SOP.md").read_text(encoding="utf-8")
        self.assertIn("M3-06 RAW square/non-square", e2e_sop)
        self.assertIn("M5-01 deterministic native-Euler H3 controlled execution", e2e_sop)
        self.assertIn("Remaining H3 capabilities and H4 retain", e2e_sop)
        self.assertIn("runtime rejection", e2e_sop)
        self.assertIn("prequeue HTTP 400 rejection", e2e_sop)


if __name__ == "__main__":
    unittest.main()
