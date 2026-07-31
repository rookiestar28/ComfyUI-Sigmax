"""Thin ComfyUI selector for local safetensors evidence inspection."""

from __future__ import annotations

import importlib
import ntpath
import os
from pathlib import PurePosixPath
from typing import Final, Protocol, cast

from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles import inspect_local_checkpoint_evidence

CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID: Final = "Sigmax.CheckpointEvidenceInspector"
CHECKPOINT_EVIDENCE_INSPECTOR_SCHEMA_ID: Final = "sigmax.checkpoint-evidence-inspector/1"
NO_LOCAL_SAFETENSORS_CHOICE: Final = "<no local safetensors available>"

_FOLDER_KINDS: Final = ("checkpoints", "diffusion_models")
_SEPARATOR: Final = "::"


class _FolderPaths(Protocol):
    def get_filename_list(self, folder_name: str) -> list[str]: ...

    def get_full_path(self, folder_name: str, filename: str) -> str | None: ...


def _import_folder_paths() -> _FolderPaths | None:
    # CRITICAL: keep package import dependency-free outside ComfyUI; host lookup is optional.
    try:
        return cast(_FolderPaths, importlib.import_module("folder_paths"))
    except ImportError:
        return None


def _choices() -> tuple[str, ...]:
    folder_paths = _import_folder_paths()
    if folder_paths is None:
        return (NO_LOCAL_SAFETENSORS_CHOICE,)
    choices: list[str] = []
    try:
        for folder_kind in _FOLDER_KINDS:
            for filename in folder_paths.get_filename_list(folder_kind):
                if isinstance(filename, str) and filename.casefold().endswith(".safetensors"):
                    choices.append(f"{folder_kind}{_SEPARATOR}{filename}")
    except (AttributeError, KeyError, OSError, TypeError):
        return (NO_LOCAL_SAFETENSORS_CHOICE,)
    return tuple(sorted(set(choices))) or (NO_LOCAL_SAFETENSORS_CHOICE,)


def _selection(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or value == NO_LOCAL_SAFETENSORS_CHOICE:
        raise ScheduleContractError("select one local safetensors checkpoint")
    parts = value.split(_SEPARATOR, maxsplit=1)
    if len(parts) != 2:
        raise ScheduleContractError("checkpoint selection is malformed")
    folder_kind, filename = parts
    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        folder_kind not in _FOLDER_KINDS
        or not filename
        or ntpath.isabs(filename)
        or os.path.isabs(filename)
        or path.is_absolute()
        or ".." in path.parts
        or not filename.casefold().endswith(".safetensors")
    ):
        raise ScheduleContractError("checkpoint selection is outside allowed model folders")
    return folder_kind, filename


class CheckpointEvidenceInspector:
    """Inspect local safetensors metadata/structure without loading model payloads."""

    DESCRIPTION = (
        "Inspects a local safetensors header and emits non-authoritative model evidence without "
        "loading tensor payloads."
    )
    CATEGORY = "Sigmax/inspection"
    FUNCTION = "inspect"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("checkpoint_evidence",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, ...]]]:
        return {"required": {"checkpoint": (_choices(),)}}

    def inspect(self, checkpoint: object) -> tuple[str]:
        folder_kind, filename = _selection(checkpoint)
        folder_paths = _import_folder_paths()
        if folder_paths is None:
            raise ScheduleContractError("ComfyUI folder_paths is unavailable")
        try:
            listed = folder_paths.get_filename_list(folder_kind)
        except (AttributeError, KeyError, OSError, TypeError) as exc:
            raise ScheduleContractError("ComfyUI model folder is unavailable") from exc
        if filename not in listed:
            raise ScheduleContractError("selected checkpoint is not listed by ComfyUI")
        try:
            resolved = folder_paths.get_full_path(folder_kind, filename)
        except (AttributeError, KeyError, OSError, TypeError) as exc:
            raise ScheduleContractError("selected checkpoint could not be resolved") from exc
        if resolved is None:
            raise ScheduleContractError("selected checkpoint was not found")
        result = inspect_local_checkpoint_evidence(
            resolved,
            display_name=f"{folder_kind}{_SEPARATOR}{filename}",
        )
        return (result.report_json,)


__all__ = [
    "CHECKPOINT_EVIDENCE_INSPECTOR_NODE_ID",
    "CHECKPOINT_EVIDENCE_INSPECTOR_SCHEMA_ID",
    "NO_LOCAL_SAFETENSORS_CHOICE",
    "CheckpointEvidenceInspector",
]
