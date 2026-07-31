from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP_PROBE = textwrap.dedent(
    """
    import importlib.util
    import json
    import pathlib
    import sys
    import types

    repository_root = pathlib.Path(sys.argv[1])

    fake_torch = types.ModuleType("torch")
    fake_torch_nn = types.ModuleType("torch.nn")

    class FakeModule:
        def __call__(self, *args, **kwargs):
            return (args, kwargs)

    original_module_call = FakeModule.__call__
    fake_torch_nn.Module = FakeModule
    fake_torch.nn = fake_torch_nn

    sys.modules["torch"] = fake_torch
    sys.modules["torch.nn"] = fake_torch_nn
    sys.modules["numpy"] = types.ModuleType("numpy")

    spec = importlib.util.spec_from_file_location(
        "comfyui_sigmax_bootstrap_probe",
        repository_root / "__init__.py",
        submodule_search_locations=[str(repository_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    payload = {
        "class_mappings": {
            key: value.__name__
            for key, value in module.NODE_CLASS_MAPPINGS.items()
        },
        "display_mappings": module.NODE_DISPLAY_NAME_MAPPINGS,
        "version": module.__version__,
        "uses_package_mappings": (
            module.NODE_CLASS_MAPPINGS
            is sys.modules[
                "comfyui_sigmax_bootstrap_probe.comfyui_sigmax"
            ].NODE_CLASS_MAPPINGS
            and module.NODE_DISPLAY_NAME_MAPPINGS
            is sys.modules[
                "comfyui_sigmax_bootstrap_probe.comfyui_sigmax"
            ].NODE_DISPLAY_NAME_MAPPINGS
        ),
        "torch_call_unchanged": fake_torch_nn.Module.__call__ is original_module_call,
        "diffusers_loaded": any(
            name == "diffusers" or name.startswith("diffusers.")
            for name in sys.modules
        ),
        "comfy_loaded": any(
            name == "comfy" or name.startswith("comfy.")
            for name in sys.modules
        ),
    }
    print(json.dumps(payload, sort_keys=True))
    """
)


class ImportSafetyTests(unittest.TestCase):
    def test_bootstrap_contract_is_dependency_free(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                BOOTSTRAP_PROBE,
                str(REPOSITORY_ROOT),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            0,
            result.returncode,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertEqual("", result.stderr)
        self.assertEqual(
            {
                "class_mappings": {
                    "Sigmax.AdvancedFlowMatchScheduler": "AdvancedFlowMatchScheduler",
                    "Sigmax.CheckpointEvidenceInspector": "CheckpointEvidenceInspector",
                    "Sigmax.Krea2SigmaScheduler": "Krea2SigmaScheduler",
                    "Sigmax.ModelAwareSigmaScheduler": "ModelAwareSigmaScheduler",
                    "Sigmax.ProfileInspector": "ProfileInspector",
                    "Sigmax.RawWorkflowOutput": "RawWorkflowOutput",
                    "Sigmax.ScheduleComparison": "ScheduleComparison",
                    "Sigmax.ScheduleConcatenate": "ScheduleConcatenate",
                    "Sigmax.ScheduleInspector": "ScheduleInspector",
                    "Sigmax.ScheduleResample": "ScheduleResample",
                    "Sigmax.ScheduleSlice": "ScheduleSlice",
                    "Sigmax.TurboWorkflowOutput": "TurboWorkflowOutput",
                },
                "comfy_loaded": False,
                "diffusers_loaded": False,
                "display_mappings": {
                    "Sigmax.AdvancedFlowMatchScheduler": "Advanced FlowMatch Scheduler",
                    "Sigmax.CheckpointEvidenceInspector": "Checkpoint Evidence Inspector",
                    "Sigmax.Krea2SigmaScheduler": "Krea 2 Sigma Scheduler",
                    "Sigmax.ModelAwareSigmaScheduler": "Model-Aware Sigma Scheduler",
                    "Sigmax.ProfileInspector": "Profile Inspector",
                    "Sigmax.RawWorkflowOutput": "RAW Workflow Output",
                    "Sigmax.ScheduleComparison": "Schedule Comparison",
                    "Sigmax.ScheduleConcatenate": "Schedule Concatenate",
                    "Sigmax.ScheduleInspector": "Schedule Inspector",
                    "Sigmax.ScheduleResample": "Schedule Resample",
                    "Sigmax.ScheduleSlice": "Schedule Slice",
                    "Sigmax.TurboWorkflowOutput": "Turbo Workflow Output",
                },
                "torch_call_unchanged": True,
                "uses_package_mappings": True,
                "version": "1.0.0",
            },
            json.loads(result.stdout),
        )

    def test_unrelated_legacy_modules_are_removed(self) -> None:
        for relative_path in (
            "nunchaku_compat.py",
            "extract_metadata_node.py",
        ):
            with self.subTest(path=relative_path):
                self.assertFalse(
                    (REPOSITORY_ROOT / relative_path).exists(),
                    msg=f"unrelated legacy module remains: {relative_path}",
                )

    def test_bootstrap_supports_pytest_style_top_level_import(self) -> None:
        probe = textwrap.dedent(
            """
            import sys

            sys.path.insert(0, sys.argv[1])
            import __init__ as bootstrap

            assert bootstrap.__version__ == "1.0.0"
            assert sorted(bootstrap.NODE_CLASS_MAPPINGS) == [
                "Sigmax.AdvancedFlowMatchScheduler",
                "Sigmax.CheckpointEvidenceInspector",
                "Sigmax.Krea2SigmaScheduler",
                "Sigmax.ModelAwareSigmaScheduler",
                "Sigmax.ProfileInspector",
                "Sigmax.RawWorkflowOutput",
                "Sigmax.ScheduleComparison",
                "Sigmax.ScheduleConcatenate",
                "Sigmax.ScheduleInspector",
                "Sigmax.ScheduleResample",
                "Sigmax.ScheduleSlice",
                "Sigmax.TurboWorkflowOutput",
            ]
            assert bootstrap.NODE_DISPLAY_NAME_MAPPINGS == {
                "Sigmax.AdvancedFlowMatchScheduler": "Advanced FlowMatch Scheduler",
                "Sigmax.CheckpointEvidenceInspector": "Checkpoint Evidence Inspector",
                "Sigmax.Krea2SigmaScheduler": "Krea 2 Sigma Scheduler",
                "Sigmax.ModelAwareSigmaScheduler": "Model-Aware Sigma Scheduler",
                "Sigmax.ProfileInspector": "Profile Inspector",
                "Sigmax.RawWorkflowOutput": "RAW Workflow Output",
                "Sigmax.ScheduleComparison": "Schedule Comparison",
                "Sigmax.ScheduleConcatenate": "Schedule Concatenate",
                "Sigmax.ScheduleInspector": "Schedule Inspector",
                "Sigmax.ScheduleResample": "Schedule Resample",
                "Sigmax.ScheduleSlice": "Schedule Slice",
                "Sigmax.TurboWorkflowOutput": "Turbo Workflow Output",
            }
            """
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe, str(REPOSITORY_ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            0,
            result.returncode,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


def test_bootstrap_supports_dynamic_comfyui_custom_node_loader() -> None:
    probe = textwrap.dedent(
        """
        import importlib.util
        import json
        import sys
        from pathlib import Path

        root = Path(sys.argv[1]).resolve()
        spec = importlib.util.spec_from_file_location(
            "custom_nodes.ComfyUI_Sigmax",
            root / "__init__.py",
            submodule_search_locations=[str(root)],
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print(json.dumps(sorted(module.NODE_CLASS_MAPPINGS)))
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(REPOSITORY_ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "Sigmax.AdvancedFlowMatchScheduler",
        "Sigmax.CheckpointEvidenceInspector",
        "Sigmax.Krea2SigmaScheduler",
        "Sigmax.ModelAwareSigmaScheduler",
        "Sigmax.ProfileInspector",
        "Sigmax.RawWorkflowOutput",
        "Sigmax.ScheduleComparison",
        "Sigmax.ScheduleConcatenate",
        "Sigmax.ScheduleInspector",
        "Sigmax.ScheduleResample",
        "Sigmax.ScheduleSlice",
        "Sigmax.TurboWorkflowOutput",
    ]


if __name__ == "__main__":
    unittest.main()
