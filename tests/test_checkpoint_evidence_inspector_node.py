"""Thin ComfyUI boundary for allowlisted local checkpoint inspection."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import comfyui_sigmax
import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.nodes import checkpoint_evidence_inspector as node_module
from comfyui_sigmax.nodes.checkpoint_evidence_inspector import (
    CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID,
    CHECKPOINT_EVIDENCE_INSPECTOR_SCHEMA_ID,
    CheckpointEvidenceInspector,
)


def _host(
    *,
    names: dict[str, list[str]],
    paths: dict[tuple[str, str], str | None],
) -> object:
    return SimpleNamespace(
        get_filename_list=lambda category: list(names.get(category, [])),
        get_full_path=lambda category, name: paths.get((category, name)),
    )


def _minimal_checkpoint(path: Path) -> None:
    header = json.dumps(
        {"x": {"data_offsets": [0, 1], "dtype": "U8", "shape": [1]}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0")


def test_node_declares_stable_schema_and_dependency_free_empty_selector() -> None:
    inputs = CheckpointEvidenceInspector.INPUT_TYPES()

    assert CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID == "Sigmax.CheckpointEvidenceInspector"
    assert CHECKPOINT_EVIDENCE_INSPECTOR_SCHEMA_ID == "sigmax.checkpoint-evidence-inspector/1"
    assert CheckpointEvidenceInspector.CATEGORY == "Sigmax/inspection"
    assert CheckpointEvidenceInspector.FUNCTION == "inspect"
    assert CheckpointEvidenceInspector.RETURN_TYPES == ("STRING",)
    assert CheckpointEvidenceInspector.RETURN_NAMES == ("checkpoint_evidence",)
    assert CheckpointEvidenceInspector.OUTPUT_NODE is False
    assert set(inputs["required"]) == {"checkpoint"}
    assert inputs["required"]["checkpoint"][0] == (node_module.NO_LOCAL_SAFETENSORS_CHOICE,)


def test_node_selector_lists_only_safetensors_from_allowlisted_model_folders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _host(
        names={
            "checkpoints": ["z.ckpt", "nested/raw.safetensors"],
            "diffusion_models": ["Turbo.SAFETENSORS", "bad.bin"],
            "loras": ["must-not-appear.safetensors"],
        },
        paths={},
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: fake)

    choices = CheckpointEvidenceInspector.INPUT_TYPES()["required"]["checkpoint"][0]

    assert choices == (
        "checkpoints::nested/raw.safetensors",
        "diffusion_models::Turbo.SAFETENSORS",
    )


def test_folder_paths_import_and_selector_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _host(names={}, paths={})
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)
    assert node_module._import_folder_paths() is fake

    broken = SimpleNamespace(
        get_filename_list=lambda category: (_ for _ in ()).throw(OSError("unavailable")),
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: broken)
    assert CheckpointEvidenceInspector.INPUT_TYPES()["required"]["checkpoint"][0] == (
        node_module.NO_LOCAL_SAFETENSORS_CHOICE,
    )


def test_node_resolves_choice_through_comfyui_and_inspects_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_path = tmp_path / "private" / "model.safetensors"
    private_path.parent.mkdir()
    _minimal_checkpoint(private_path)
    fake = _host(
        names={"checkpoints": ["model.safetensors"], "diffusion_models": []},
        paths={("checkpoints", "model.safetensors"): str(private_path)},
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: fake)

    (report_json,) = CheckpointEvidenceInspector().inspect("checkpoints::model.safetensors")
    report = json.loads(report_json)

    assert report["schema"] == "sigmax.checkpoint-evidence-inspection/1"
    assert report["status"] == "inspected"
    assert report["source"]["display_name"] == "checkpoints::model.safetensors"
    assert str(tmp_path) not in report_json


@pytest.mark.parametrize(
    "choice",
    (
        node_module.NO_LOCAL_SAFETENSORS_CHOICE,
        r"C:\private\model.safetensors",
        "loras::model.safetensors",
        "checkpoints::../model.safetensors",
        "checkpoints::model.ckpt",
        "checkpoints::",
    ),
)
def test_node_rejects_sentinel_raw_paths_unlisted_folders_and_non_safetensors(
    choice: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        node_module,
        "_import_folder_paths",
        lambda: _host(names={}, paths={}),
    )

    with pytest.raises(ScheduleContractError):
        CheckpointEvidenceInspector().inspect(choice)


def test_node_rejects_host_resolution_failure_without_opening_user_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def resolve(category: str, name: str) -> None:
        calls.append((category, name))

    fake = SimpleNamespace(
        get_filename_list=lambda category: ["model.safetensors"],
        get_full_path=resolve,
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: fake)

    with pytest.raises(ScheduleContractError, match="not found"):
        CheckpointEvidenceInspector().inspect("checkpoints::model.safetensors")

    assert calls == [("checkpoints", "model.safetensors")]


def test_node_rejects_unavailable_host_and_unlisted_or_broken_model_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: None)
    with pytest.raises(ScheduleContractError, match="folder_paths"):
        CheckpointEvidenceInspector().inspect("checkpoints::model.safetensors")

    broken_list = SimpleNamespace(
        get_filename_list=lambda category: (_ for _ in ()).throw(OSError("unavailable")),
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: broken_list)
    with pytest.raises(ScheduleContractError, match="model folder"):
        CheckpointEvidenceInspector().inspect("checkpoints::model.safetensors")

    unlisted = _host(names={"checkpoints": []}, paths={})
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: unlisted)
    with pytest.raises(ScheduleContractError, match="not listed"):
        CheckpointEvidenceInspector().inspect("checkpoints::model.safetensors")


def test_node_maps_host_resolution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = SimpleNamespace(
        get_filename_list=lambda category: ["model.safetensors"],
        get_full_path=lambda category, name: (_ for _ in ()).throw(OSError("unavailable")),
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: fake)

    with pytest.raises(ScheduleContractError, match="could not be resolved"):
        CheckpointEvidenceInspector().inspect("checkpoints::model.safetensors")


@pytest.mark.parametrize(
    "filename",
    (
        r"C:\private\model.safetensors",
        r"\\server\share\model.safetensors",
    ),
)
def test_node_rejects_prefixed_windows_or_unc_paths_before_host_calls(
    filename: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def filenames(category: str) -> list[str]:
        calls.append(category)
        return [filename]

    fake = SimpleNamespace(
        get_filename_list=filenames,
        get_full_path=lambda category, name: calls.append(f"{category}:{name}"),
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: fake)

    with pytest.raises(ScheduleContractError, match="outside allowed"):
        CheckpointEvidenceInspector().inspect(f"checkpoints::{filename}")

    assert calls == []


def test_node_registers_once_with_stable_display_name() -> None:
    assert (
        comfyui_sigmax.NODE_CLASS_MAPPINGS[CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID]
        is CheckpointEvidenceInspector
    )
    assert (
        comfyui_sigmax.NODE_DISPLAY_NAME_MAPPINGS[CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID]
        == "Checkpoint Evidence Inspector"
    )
    assert (
        tuple(comfyui_sigmax.NODE_CLASS_MAPPINGS).count(CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID) == 1
    )


def test_node_never_calls_network_or_accelerator_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _minimal_checkpoint(checkpoint)
    fake = _host(
        names={"checkpoints": [checkpoint.name]},
        paths={("checkpoints", checkpoint.name): str(checkpoint)},
    )
    monkeypatch.setattr(node_module, "_import_folder_paths", lambda: fake)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden external operation")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)

    (report_json,) = CheckpointEvidenceInspector().inspect("checkpoints::model.safetensors")

    assert json.loads(report_json)["status"] == "inspected"
