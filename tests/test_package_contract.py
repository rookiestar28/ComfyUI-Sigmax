from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import tomli as tomllib

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PackageContractTests(unittest.TestCase):
    def test_pyproject_metadata_contract(self) -> None:
        pyproject_path = REPOSITORY_ROOT / "pyproject.toml"
        self.assertTrue(pyproject_path.is_file(), msg="pyproject.toml is missing")

        with pyproject_path.open("rb") as stream:
            metadata = tomllib.load(stream)

        project = metadata["project"]
        self.assertEqual("comfyui-sigmax", project["name"])
        self.assertEqual("0.1.0.dev0", project["version"])
        self.assertEqual(">=3.10", project["requires-python"])
        self.assertEqual([], project["dependencies"])
        self.assertEqual("MIT", project["license"])
        self.assertEqual(["LICENSE.TXT", "NOTICE"], project["license-files"])
        self.assertEqual(
            ["diffusers>=0.39,<0.40"],
            project["optional-dependencies"]["reference"],
        )
        self.assertEqual(
            "https://github.com/rookiestar28/ComfyUI-Sigmax",
            project["urls"]["Repository"],
        )

        self.assertEqual(
            ["setuptools>=80,<84"],
            metadata["build-system"]["requires"],
        )
        self.assertEqual(
            "setuptools.build_meta",
            metadata["build-system"]["build-backend"],
        )
        self.assertEqual(
            [
                "comfyui_sigmax",
                "comfyui_sigmax.adapters",
                "comfyui_sigmax.core",
                "comfyui_sigmax.nodes",
                "comfyui_sigmax.profiles",
            ],
            metadata["tool"]["setuptools"]["packages"],
        )

        comfy = metadata["tool"]["comfy"]
        self.assertEqual("rookiestar28", comfy["PublisherId"])
        self.assertEqual("ComfyUI-Sigmax", comfy["DisplayName"])
        self.assertEqual(">=0.29.0", comfy["requires-comfyui"])

    def test_namespace_contract(self) -> None:
        package_path = REPOSITORY_ROOT / "comfyui_sigmax"
        self.assertTrue((package_path / "__init__.py").is_file())
        self.assertTrue((package_path / "py.typed").is_file())
        self.assertTrue((package_path / "core" / "__init__.py").is_file())
        self.assertTrue((package_path / "core" / "base_grids.py").is_file())
        self.assertTrue((package_path / "core" / "artifacts.py").is_file())
        self.assertTrue((package_path / "core" / "capabilities.py").is_file())
        self.assertTrue((package_path / "core" / "fingerprints.py").is_file())
        self.assertTrue((package_path / "core" / "request_result.py").is_file())
        self.assertTrue((package_path / "core" / "schedule_contracts.py").is_file())
        self.assertTrue((package_path / "core" / "shifts.py").is_file())
        self.assertTrue((package_path / "core" / "terminal_slicing.py").is_file())
        self.assertTrue((package_path / "core" / "validation.py").is_file())
        self.assertTrue((package_path / "profiles" / "__init__.py").is_file())
        self.assertTrue((package_path / "profiles" / "krea2_common.py").is_file())
        self.assertTrue((package_path / "profiles" / "krea2_raw.py").is_file())
        self.assertTrue((package_path / "profiles" / "registry.py").is_file())
        self.assertTrue((package_path / "profiles" / "resolution.py").is_file())
        self.assertTrue((package_path / "profiles" / "schema_v1.py").is_file())
        self.assertTrue((package_path / "profiles" / "krea2_turbo.py").is_file())
        self.assertTrue((package_path / "adapters" / "__init__.py").is_file())
        self.assertTrue((package_path / "adapters" / "comfyui.py").is_file())
        self.assertTrue((package_path / "adapters" / "registration.py").is_file())
        self.assertTrue((package_path / "nodes" / "__init__.py").is_file())
        self.assertTrue((package_path / "nodes" / "advanced_flowmatch_scheduler.py").is_file())
        self.assertTrue((package_path / "nodes" / "krea2_sigma_scheduler.py").is_file())
        self.assertTrue((package_path / "nodes" / "model_aware_sigma_scheduler.py").is_file())

        sys.path.insert(0, str(REPOSITORY_ROOT))
        self.addCleanup(sys.path.remove, str(REPOSITORY_ROOT))
        sys.modules.pop("comfyui_sigmax", None)
        package = importlib.import_module("comfyui_sigmax")
        self.addCleanup(sys.modules.pop, "comfyui_sigmax", None)

        self.assertEqual("0.1.0.dev0", package.__version__)
        self.assertEqual(
            [
                "Sigmax.AdvancedFlowMatchScheduler",
                "Sigmax.Krea2SigmaScheduler",
                "Sigmax.ModelAwareSigmaScheduler",
            ],
            sorted(package.NODE_CLASS_MAPPINGS),
        )
        self.assertEqual(
            {
                "Sigmax.AdvancedFlowMatchScheduler": "Advanced FlowMatch Scheduler",
                "Sigmax.Krea2SigmaScheduler": "Krea 2 Sigma Scheduler",
                "Sigmax.ModelAwareSigmaScheduler": "Model-Aware Sigma Scheduler",
            },
            package.NODE_DISPLAY_NAME_MAPPINGS,
        )


if __name__ == "__main__":
    unittest.main()
