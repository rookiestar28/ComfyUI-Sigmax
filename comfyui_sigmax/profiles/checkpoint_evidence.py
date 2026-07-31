"""Privacy-safe evidence derived from one local safetensors header."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from comfyui_sigmax.core import (
    SafetensorsHeader,
    SafetensorsHeaderError,
    SafetensorsHeaderReason,
    ScheduleContractError,
    read_safetensors_header,
)
from comfyui_sigmax.profiles.krea2_variant import (
    Krea2VariantConfidence,
    Krea2VariantResolution,
    Krea2VariantResolutionStatus,
    resolve_krea2_variant,
)

CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID: Final = "sigmax.checkpoint-evidence-inspection/1"
CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_VERSION: Final = "1"
CHECKPOINT_EVIDENCE_REASON_CODES: Final = frozenset(
    {
        *(reason.value for reason in SafetensorsHeaderReason),
        "checkpoint.evidence_invalid",
        "checkpoint.file_not_found",
        "checkpoint.io_error",
        "checkpoint.not_regular_file",
        "checkpoint.permission_denied",
        "checkpoint.unsupported_format",
        "conflicting_suggestion_evidence",
        "filename.raw_token",
        "filename.turbo_token",
        "header.is_distilled.raw",
        "header.is_distilled.turbo",
        "header.krea2_variant.raw",
        "header.krea2_variant.turbo",
        "insufficient_variant_evidence",
        "krea2_family_does_not_identify_variant",
        "non_authoritative_variant_suggestion",
        "tensor.krea2_family",
    }
)

_MAX_DISPLAY_NAME: Final = 1_024
_PRIVATE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|^/|^\\\\)")
_CONFIDENCE_RANK: Final = {
    Krea2VariantConfidence.NONE: 0,
    Krea2VariantConfidence.FAMILY_ONLY: 1,
    Krea2VariantConfidence.WEAK: 2,
    Krea2VariantConfidence.CORROBORATING: 3,
    Krea2VariantConfidence.VERIFIED: 4,
    Krea2VariantConfidence.AUTHORITATIVE: 5,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointEvidenceInspection:
    """Canonical report text for one accepted or rejected local inspection."""

    schema_id: str
    report_json: str

    def __post_init__(self) -> None:
        if self.schema_id != CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID:
            raise ScheduleContractError("checkpoint inspection schema_id is unsupported")
        if not isinstance(self.report_json, str) or not self.report_json:
            raise ScheduleContractError("checkpoint inspection report must be non-empty JSON")
        try:
            decoded = json.loads(self.report_json)
        except json.JSONDecodeError as exc:
            raise ScheduleContractError("checkpoint inspection report must be valid JSON") from exc
        if not isinstance(decoded, dict) or decoded.get("schema") != self.schema_id:
            raise ScheduleContractError("checkpoint inspection report schema is invalid")
        if _canonical_json(decoded) != self.report_json:
            raise ScheduleContractError("checkpoint inspection report must be canonical JSON")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _display_name(path: Path, display_name: str | None) -> str:
    value = path.name if display_name is None else display_name
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_DISPLAY_NAME
        or any(ord(character) < 32 for character in value)
        or _PRIVATE_PATH.match(value)
    ):
        raise ScheduleContractError("checkpoint display_name must be bounded relative text")
    return value


def _empty_identity(*, reason_codes: list[str]) -> dict[str, object]:
    return {
        "confidence": "none",
        "confirmed_variant": None,
        "decisive_source": None,
        "family": None,
        "reason_codes": reason_codes,
        "resolution_status": "ambiguous",
        "suggested_variant": None,
    }


def _source(
    *,
    display_name: str,
    file_bytes: int | None,
    header_bytes: int | None,
) -> dict[str, object]:
    return {
        "display_name": display_name,
        "file_bytes": file_bytes,
        "format": "safetensors",
        "header_bytes": header_bytes,
        "payload_bytes_read": 0,
    }


def _inspection(
    *,
    display_name: str,
    status: str,
    file_bytes: int | None,
    header_bytes: int | None,
    structure: dict[str, object] | None,
    model_identity: dict[str, object],
    reason_codes: list[str],
) -> CheckpointEvidenceInspection:
    report = {
        "model_identity": model_identity,
        "reason_codes": reason_codes,
        "schema": CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID,
        "source": _source(
            display_name=display_name,
            file_bytes=file_bytes,
            header_bytes=header_bytes,
        ),
        "status": status,
        "structure": structure,
    }
    return CheckpointEvidenceInspection(
        schema_id=CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID,
        report_json=_canonical_json(report),
    )


def _rejected(
    *,
    display_name: str,
    reason_code: str,
    file_bytes: int | None = None,
    header_bytes: int | None = None,
) -> CheckpointEvidenceInspection:
    return _inspection(
        display_name=display_name,
        status="rejected",
        file_bytes=file_bytes,
        header_bytes=header_bytes,
        structure=None,
        model_identity=_empty_identity(reason_codes=[reason_code]),
        reason_codes=[reason_code],
    )


def _structure_fingerprint(header: SafetensorsHeader) -> str:
    projection = [
        {
            "data_offsets": list(item.data_offsets),
            "dtype": item.dtype,
            "name": item.name,
            "shape": list(item.shape),
        }
        for item in header.tensors
    ]
    return "sha256:" + hashlib.sha256(_canonical_json(projection).encode("utf-8")).hexdigest()


def _structure(header: SafetensorsHeader) -> dict[str, object]:
    dtype_counts = Counter(item.dtype for item in header.tensors)
    rank_counts = Counter(str(len(item.shape)) for item in header.tensors)
    return {
        "data_bytes": header.data_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "rank_counts": dict(sorted(rank_counts.items(), key=lambda item: int(item[0]))),
        "structure_fingerprint": _structure_fingerprint(header),
        "tensor_count": len(header.tensors),
    }


def _identity(resolution: Krea2VariantResolution) -> dict[str, object]:
    evidence_confidence = max(
        (item.confidence for item in resolution.evidence),
        key=_CONFIDENCE_RANK.__getitem__,
        default=Krea2VariantConfidence.NONE,
    )
    reason_codes = [item.reason_code for item in resolution.evidence]
    reason_codes.extend(resolution.warnings)
    return {
        "confidence": evidence_confidence.value,
        "confirmed_variant": (
            resolution.resolved_variant.value if resolution.resolved_variant is not None else None
        ),
        "decisive_source": (
            resolution.decisive_source.value if resolution.decisive_source is not None else None
        ),
        "family": "krea2" if resolution.evidence else None,
        "reason_codes": reason_codes,
        "resolution_status": resolution.status.value,
        "suggested_variant": (
            resolution.suggested_variant.value if resolution.suggested_variant is not None else None
        ),
    }


def _classify(header: SafetensorsHeader, *, display_name: str) -> dict[str, object]:
    try:
        resolution = resolve_krea2_variant(
            strict_official=False,
            safetensors_metadata=dict(header.metadata),
            tensor_keys=tuple(item.name for item in header.tensors),
            filename=display_name,
        )
    except ScheduleContractError:
        return _empty_identity(reason_codes=["checkpoint.evidence_invalid"])
    if resolution.status is Krea2VariantResolutionStatus.RESOLVED:
        raise ScheduleContractError("local header inspection cannot confirm a Krea 2 variant")
    return _identity(resolution)


def inspect_local_checkpoint_evidence(
    path: str | os.PathLike[str],
    *,
    display_name: str | None = None,
) -> CheckpointEvidenceInspection:
    """Inspect one local safetensors file without reading tensor payload bytes."""

    if not isinstance(path, (str, os.PathLike)):
        raise ScheduleContractError("checkpoint path must be local path text")
    checkpoint = Path(path)
    public_name = _display_name(checkpoint, display_name)
    if checkpoint.suffix.casefold() != ".safetensors":
        return _rejected(
            display_name=public_name,
            reason_code="checkpoint.unsupported_format",
        )
    try:
        with checkpoint.open("rb") as stream:
            file_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                return _rejected(
                    display_name=public_name,
                    reason_code="checkpoint.not_regular_file",
                    file_bytes=file_stat.st_size,
                )
            try:
                header = read_safetensors_header(stream, file_size=file_stat.st_size)
            except SafetensorsHeaderError as exc:
                declared_header_bytes: int | None = None
                if file_stat.st_size >= 8:
                    stream.seek(0)
                    prefix = stream.read(8)
                    declared_header_bytes = int.from_bytes(prefix, "little")
                return _rejected(
                    display_name=public_name,
                    reason_code=exc.reason.value,
                    file_bytes=file_stat.st_size,
                    header_bytes=declared_header_bytes,
                )
    except FileNotFoundError:
        return _rejected(
            display_name=public_name,
            reason_code="checkpoint.file_not_found",
        )
    except PermissionError:
        return _rejected(
            display_name=public_name,
            reason_code="checkpoint.permission_denied",
        )
    except IsADirectoryError:
        return _rejected(
            display_name=public_name,
            reason_code="checkpoint.not_regular_file",
        )
    except OSError:
        return _rejected(
            display_name=public_name,
            reason_code="checkpoint.io_error",
        )

    identity = _classify(header, display_name=public_name)
    return _inspection(
        display_name=public_name,
        status="inspected",
        file_bytes=header.file_bytes,
        header_bytes=header.header_bytes,
        structure=_structure(header),
        model_identity=identity,
        reason_codes=cast(list[str], identity["reason_codes"]),
    )


__all__ = [
    "CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID",
    "CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_VERSION",
    "CHECKPOINT_EVIDENCE_REASON_CODES",
    "CheckpointEvidenceInspection",
    "inspect_local_checkpoint_evidence",
]
