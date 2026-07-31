"""Local safetensors header inspection and fail-closed model evidence."""

from __future__ import annotations

import ast
import io
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import (
    SAFETENSORS_HEADER_MAX_BYTES,
    SafetensorsHeader,
    SafetensorsHeaderError,
    SafetensorsHeaderReason,
    SafetensorsTensorDescriptor,
    ScheduleContractError,
    read_safetensors_header,
)
from comfyui_sigmax.profiles import (
    CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID,
    CheckpointEvidenceInspection,
    inspect_local_checkpoint_evidence,
)
from comfyui_sigmax.profiles import checkpoint_evidence as evidence_module
from comfyui_sigmax.profiles.krea2_variant import Krea2VariantResolutionStatus


class _HeaderOnlyStream(io.BytesIO):
    def __init__(self, prefix_and_header: bytes) -> None:
        super().__init__(prefix_and_header)
        self.read_requests: list[int | None] = []

    def read(self, size: int | None = -1, /) -> bytes:
        self.read_requests.append(size)
        return super().read(size)


def _encoded_header(header: dict[str, object]) -> bytes:
    return json.dumps(
        header,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _safetensors_bytes(header: dict[str, object], *, payload: bytes | None = None) -> bytes:
    encoded = _encoded_header(header)
    if payload is None:
        offsets = [
            value["data_offsets"]
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict)
        ]
        payload_size = max((int(pair[1]) for pair in offsets), default=0)
        payload = bytes(payload_size)
    return len(encoded).to_bytes(8, "little") + encoded + payload


def _krea_header(*, metadata: dict[str, str] | None = None) -> dict[str, object]:
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    keys = (
        "diffusion_model.first.weight",
        "diffusion_model.blocks.0.attn.wq.weight",
        "diffusion_model.blocks.0.attn.wk.weight",
        "diffusion_model.txtfusion.projector.weight",
    )
    for index, key in enumerate(keys):
        header[key] = {
            "data_offsets": [index * 2, (index + 1) * 2],
            "dtype": "F16",
            "shape": [1],
        }
    return header


def test_header_reader_reads_exact_prefix_and_header_but_never_payload() -> None:
    header = {
        "__metadata__": {"format": "pt", "krea2_variant": "raw"},
        "scalar": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]},
        "empty": {"dtype": "BF16", "shape": [0, 4096], "data_offsets": [4, 4]},
        "packed4": {"dtype": "F4", "shape": [2], "data_offsets": [4, 5]},
        "packed6": {"dtype": "F6_E2M3", "shape": [4], "data_offsets": [5, 8]},
    }
    encoded = _encoded_header(header)
    stream = _HeaderOnlyStream(len(encoded).to_bytes(8, "little") + encoded)

    result = read_safetensors_header(stream, file_size=8 + len(encoded) + 8)

    assert stream.read_requests == [8, len(encoded)]
    assert result.header_bytes == len(encoded)
    assert result.file_bytes == 8 + len(encoded) + 8
    assert result.data_bytes == 8
    assert result.metadata == (("format", "pt"), ("krea2_variant", "raw"))
    assert tuple(item.name for item in result.tensors) == (
        "scalar",
        "empty",
        "packed4",
        "packed6",
    )
    assert tuple(item.dtype for item in result.tensors) == (
        "F32",
        "BF16",
        "F4",
        "F6_E2M3",
    )
    assert result.tensors[0].shape == ()
    assert result.tensors[1].shape == (0, 4096)


@pytest.mark.parametrize(
    ("blob", "file_size", "reason"),
    (
        (b"", 0, SafetensorsHeaderReason.HEADER_TOO_SMALL),
        (
            (SAFETENSORS_HEADER_MAX_BYTES + 1).to_bytes(8, "little"),
            SAFETENSORS_HEADER_MAX_BYTES + 9,
            SafetensorsHeaderReason.HEADER_TOO_LARGE,
        ),
        (
            (20).to_bytes(8, "little") + b"{}",
            28,
            SafetensorsHeaderReason.HEADER_TRUNCATED,
        ),
        (
            (2).to_bytes(8, "little") + b"[]",
            10,
            SafetensorsHeaderReason.HEADER_INVALID_JSON,
        ),
        (
            (2).to_bytes(8, "little") + b"{\xff",
            10,
            SafetensorsHeaderReason.HEADER_INVALID_UTF8,
        ),
    ),
)
def test_header_reader_rejects_bounded_prefix_and_json_failures(
    blob: bytes,
    file_size: int,
    reason: SafetensorsHeaderReason,
) -> None:
    with pytest.raises(SafetensorsHeaderError) as caught:
        read_safetensors_header(io.BytesIO(blob), file_size=file_size)

    assert caught.value.reason is reason


def test_header_reader_rejects_duplicate_json_keys() -> None:
    encoded = b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    blob = len(encoded).to_bytes(8, "little") + encoded

    with pytest.raises(SafetensorsHeaderError) as caught:
        read_safetensors_header(io.BytesIO(blob), file_size=len(blob) + 1)

    assert caught.value.reason is SafetensorsHeaderReason.HEADER_DUPLICATE_KEY


def test_header_reader_rejects_excessive_json_nesting_with_stable_reason() -> None:
    encoded = b'{"x":' + (b"[" * 2_000) + (b"]" * 2_000) + b"}"
    blob = len(encoded).to_bytes(8, "little") + encoded

    with pytest.raises(SafetensorsHeaderError) as caught:
        read_safetensors_header(io.BytesIO(blob), file_size=len(blob))

    assert caught.value.reason is SafetensorsHeaderReason.HEADER_INVALID_JSON


def test_header_reader_does_not_count_json_delimiters_inside_metadata_strings() -> None:
    header = {
        "__metadata__": {"text": "[" * 200 + r"\"}]"},
        "x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
    }
    blob = _safetensors_bytes(header)

    result = read_safetensors_header(io.BytesIO(blob), file_size=len(blob))

    assert dict(result.metadata)["text"] == "[" * 200 + r"\"}]"


@pytest.mark.parametrize(
    ("header", "payload_size", "reason"),
    (
        (
            {"__metadata__": {"private": 1}},
            0,
            SafetensorsHeaderReason.METADATA_INVALID,
        ),
        (
            {"x": {"dtype": "F16", "shape": [1]}},
            2,
            SafetensorsHeaderReason.TENSOR_DESCRIPTOR_INVALID,
        ),
        (
            {"x": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 1]}},
            1,
            SafetensorsHeaderReason.DTYPE_UNSUPPORTED,
        ),
        (
            {"x": {"dtype": "F4", "shape": [1], "data_offsets": [0, 1]}},
            1,
            SafetensorsHeaderReason.TENSOR_MISALIGNED,
        ),
        (
            {
                "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]},
            },
            3,
            SafetensorsHeaderReason.OFFSETS_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}},
            3,
            SafetensorsHeaderReason.BUFFER_INCOMPLETE,
        ),
    ),
)
def test_header_reader_rejects_invalid_metadata_tensor_and_buffer_structure(
    header: dict[str, object],
    payload_size: int,
    reason: SafetensorsHeaderReason,
) -> None:
    encoded = _encoded_header(header)
    stream = io.BytesIO(len(encoded).to_bytes(8, "little") + encoded)

    with pytest.raises(SafetensorsHeaderError) as caught:
        read_safetensors_header(stream, file_size=8 + len(encoded) + payload_size)

    assert caught.value.reason is reason


@pytest.mark.parametrize(
    ("header", "payload_size", "reason"),
    (
        ({"__metadata__": []}, 0, SafetensorsHeaderReason.METADATA_INVALID),
        (
            {"x": {"dtype": "U8", "shape": "bad", "data_offsets": [0, 1]}},
            1,
            SafetensorsHeaderReason.SHAPE_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}},
            1,
            SafetensorsHeaderReason.SHAPE_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}},
            1,
            SafetensorsHeaderReason.SHAPE_INVALID,
        ),
        (
            {
                "x": {
                    "dtype": "U8",
                    "shape": [sys.maxsize + 1],
                    "data_offsets": [0, 1],
                }
            },
            1,
            SafetensorsHeaderReason.SHAPE_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": "bad"}},
            1,
            SafetensorsHeaderReason.OFFSETS_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0]}},
            1,
            SafetensorsHeaderReason.OFFSETS_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": [True, 1]}},
            1,
            SafetensorsHeaderReason.OFFSETS_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": [-1, 0]}},
            1,
            SafetensorsHeaderReason.OFFSETS_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 0]}},
            1,
            SafetensorsHeaderReason.OFFSETS_INVALID,
        ),
        (
            {"x": {"dtype": "U8", "shape": [sys.maxsize, 2], "data_offsets": [0, 0]}},
            0,
            SafetensorsHeaderReason.TENSOR_SIZE_OVERFLOW,
        ),
        (
            {"x": {"dtype": "I16", "shape": [sys.maxsize], "data_offsets": [0, 0]}},
            0,
            SafetensorsHeaderReason.TENSOR_SIZE_OVERFLOW,
        ),
        (
            {"x": {"dtype": "U8", "shape": [2], "data_offsets": [0, 1]}},
            1,
            SafetensorsHeaderReason.TENSOR_DESCRIPTOR_INVALID,
        ),
    ),
)
def test_header_reader_rejects_each_descriptor_contract_branch(
    header: dict[str, object],
    payload_size: int,
    reason: SafetensorsHeaderReason,
) -> None:
    encoded = _encoded_header(header)

    with pytest.raises(SafetensorsHeaderError) as caught:
        read_safetensors_header(
            io.BytesIO(len(encoded).to_bytes(8, "little") + encoded),
            file_size=8 + len(encoded) + payload_size,
        )

    assert caught.value.reason is reason


@pytest.mark.parametrize(
    ("encoded", "reason"),
    (
        (b"", SafetensorsHeaderReason.HEADER_INVALID_JSON),
        (b"{", SafetensorsHeaderReason.HEADER_INVALID_JSON),
        (b'{"x":NaN}', SafetensorsHeaderReason.HEADER_INVALID_JSON),
        (b"{}\n", SafetensorsHeaderReason.HEADER_INVALID_JSON),
    ),
)
def test_header_reader_rejects_noncanonical_json_forms(
    encoded: bytes,
    reason: SafetensorsHeaderReason,
) -> None:
    blob = len(encoded).to_bytes(8, "little") + encoded

    with pytest.raises(SafetensorsHeaderError) as caught:
        read_safetensors_header(io.BytesIO(blob), file_size=len(blob))

    assert caught.value.reason is reason


@pytest.mark.parametrize("file_size", (-1, True))
def test_header_reader_rejects_invalid_file_size_contract(file_size: Any) -> None:
    with pytest.raises(ScheduleContractError, match="file_size"):
        read_safetensors_header(io.BytesIO(b""), file_size=file_size)


def test_header_reader_rejects_declared_header_larger_than_file_before_second_read() -> None:
    stream = _HeaderOnlyStream((3).to_bytes(8, "little"))

    with pytest.raises(SafetensorsHeaderError) as caught:
        read_safetensors_header(stream, file_size=10)

    assert caught.value.reason is SafetensorsHeaderReason.HEADER_TRUNCATED
    assert stream.read_requests == [8]


def test_safetensors_contract_objects_reject_invalid_direct_construction() -> None:
    with pytest.raises(ScheduleContractError, match="reason"):
        SafetensorsHeaderError(cast(Any, "unknown"))

    descriptor = SafetensorsTensorDescriptor(
        name="x",
        dtype="U8",
        shape=(1,),
        data_offsets=(0, 1),
    )
    assert descriptor.data_bytes == 1

    invalid_descriptors = (
        {"name": 1, "dtype": "U8", "shape": (1,), "data_offsets": (0, 1)},
        {"name": "x", "dtype": "unknown", "shape": (1,), "data_offsets": (0, 1)},
        {"name": "x", "dtype": "U8", "shape": [1], "data_offsets": (0, 1)},
        {"name": "x", "dtype": "U8", "shape": (1,), "data_offsets": (1, 0)},
    )
    for values in invalid_descriptors:
        with pytest.raises(ScheduleContractError):
            SafetensorsTensorDescriptor(**cast(Any, values))

    valid_header = {
        "header_bytes": 2,
        "file_bytes": 11,
        "data_bytes": 1,
        "metadata": (("format", "pt"),),
        "tensors": (descriptor,),
    }
    assert SafetensorsHeader(**cast(Any, valid_header)).data_bytes == 1
    invalid_headers = (
        {**valid_header, "file_bytes": 10},
        {**valid_header, "metadata": (("z", "1"), ("a", "2"))},
        {**valid_header, "metadata": (("format", "pt"), ("format", "pt"))},
        {**valid_header, "tensors": ("bad",)},
        {
            **valid_header,
            "data_bytes": 2,
            "file_bytes": 12,
            "tensors": (
                SafetensorsTensorDescriptor(
                    name="gap",
                    dtype="U8",
                    shape=(1,),
                    data_offsets=(1, 2),
                ),
            ),
        },
        {**valid_header, "tensors": (descriptor, descriptor)},
        {
            **valid_header,
            "data_bytes": 2,
            "file_bytes": 12,
            "tensors": (
                SafetensorsTensorDescriptor(
                    name="second", dtype="U8", shape=(1,), data_offsets=(1, 2)
                ),
                descriptor,
            ),
        },
        {**valid_header, "data_bytes": 2, "file_bytes": 12},
    )
    for values in invalid_headers:
        with pytest.raises(ScheduleContractError):
            SafetensorsHeader(**cast(Any, values))


def test_local_inspection_emits_corroborating_suggestion_without_private_metadata(
    tmp_path: Path,
) -> None:
    private_path = tmp_path / "private" / "renamed.safetensors"
    private_path.parent.mkdir()
    private_path.write_bytes(
        _safetensors_bytes(
            _krea_header(
                metadata={
                    "is_distilled": "true",
                    "private_path": r"C:\secret\model.safetensors",
                    "token": "do-not-retain",
                }
            )
        )
    )

    result = inspect_local_checkpoint_evidence(
        private_path,
        display_name="checkpoints::renamed.safetensors",
    )
    report = json.loads(result.report_json)

    assert isinstance(result, CheckpointEvidenceInspection)
    assert result.schema_id == CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID
    assert report["status"] == "inspected"
    assert report["source"] == {
        "display_name": "checkpoints::renamed.safetensors",
        "file_bytes": private_path.stat().st_size,
        "format": "safetensors",
        "header_bytes": report["source"]["header_bytes"],
        "payload_bytes_read": 0,
    }
    assert report["structure"]["tensor_count"] == 4
    assert report["structure"]["data_bytes"] == 8
    assert report["structure"]["dtype_counts"] == {"F16": 4}
    assert report["model_identity"]["resolution_status"] == "suggested"
    assert report["model_identity"]["confidence"] == "corroborating"
    assert report["model_identity"]["confirmed_variant"] is None
    assert report["model_identity"]["suggested_variant"] == "turbo"
    assert "header.is_distilled.turbo" in report["model_identity"]["reason_codes"]
    rendered = result.report_json
    assert str(tmp_path) not in rendered
    assert "private_path" not in rendered
    assert "do-not-retain" not in rendered


def test_tensor_structure_is_family_only_and_never_confirms_variant(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(_safetensors_bytes(_krea_header()))

    report = json.loads(
        inspect_local_checkpoint_evidence(checkpoint, display_name="model.safetensors").report_json
    )

    assert report["status"] == "inspected"
    assert report["model_identity"] == {
        "confidence": "family_only",
        "confirmed_variant": None,
        "decisive_source": None,
        "family": "krea2",
        "reason_codes": [
            "tensor.krea2_family",
            "krea2_family_does_not_identify_variant",
            "insufficient_variant_evidence",
        ],
        "resolution_status": "ambiguous",
        "suggested_variant": None,
    }


def test_conflicting_local_metadata_and_filename_remain_ambiguous(tmp_path: Path) -> None:
    checkpoint = tmp_path / "krea2_raw.safetensors"
    checkpoint.write_bytes(_safetensors_bytes(_krea_header(metadata={"is_distilled": "true"})))

    report = json.loads(inspect_local_checkpoint_evidence(checkpoint).report_json)

    assert report["model_identity"]["resolution_status"] == "ambiguous"
    assert report["model_identity"]["confirmed_variant"] is None
    assert report["model_identity"]["suggested_variant"] is None
    assert report["model_identity"]["confidence"] == "corroborating"
    assert "conflicting_suggestion_evidence" in report["model_identity"]["reason_codes"]


@pytest.mark.parametrize(
    ("filename", "expected_reason"),
    (
        ("missing.safetensors", "checkpoint.file_not_found"),
        ("wrong.ckpt", "checkpoint.unsupported_format"),
    ),
)
def test_local_inspection_returns_stable_rejection_for_file_failures(
    tmp_path: Path,
    filename: str,
    expected_reason: str,
) -> None:
    path = tmp_path / filename
    if path.suffix == ".ckpt":
        path.write_bytes(b"not pickle execution")

    first = inspect_local_checkpoint_evidence(path, display_name=filename)
    second = inspect_local_checkpoint_evidence(path, display_name=filename)
    report = json.loads(first.report_json)

    assert first == second
    assert report["status"] == "rejected"
    assert report["reason_codes"] == [expected_reason]
    assert report["structure"] is None
    assert report["model_identity"]["confirmed_variant"] is None
    assert report["model_identity"]["confidence"] == "none"
    assert str(tmp_path) not in first.report_json


def test_malformed_checkpoint_returns_reason_instead_of_loading_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "malformed.safetensors"
    checkpoint.write_bytes((SAFETENSORS_HEADER_MAX_BYTES + 1).to_bytes(8, "little"))

    report = json.loads(inspect_local_checkpoint_evidence(checkpoint).report_json)

    assert report["status"] == "rejected"
    assert report["reason_codes"] == ["safetensors.header_too_large"]
    assert report["source"]["payload_bytes_read"] == 0
    assert report["model_identity"]["confirmed_variant"] is None


def test_inspection_result_is_immutable_and_canonical(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(_safetensors_bytes(_krea_header()))
    result = inspect_local_checkpoint_evidence(checkpoint)

    assert result.report_json == json.dumps(
        json.loads(result.report_json),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises((AttributeError, TypeError)):
        result.report_json = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("schema_id", "report_json"),
    (
        ("wrong", '{"schema":"wrong"}'),
        (CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID, ""),
        (CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID, "{"),
        (CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID, "[]"),
        (CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID, '{"schema":"wrong"}'),
        (
            CHECKPOINT_EVIDENCE_INSPECTION_SCHEMA_ID,
            '{"schema": "sigmax.checkpoint-evidence-inspection/1"}',
        ),
    ),
)
def test_inspection_result_rejects_invalid_direct_contracts(
    schema_id: str,
    report_json: str,
) -> None:
    with pytest.raises(ScheduleContractError):
        CheckpointEvidenceInspection(schema_id=schema_id, report_json=report_json)


@pytest.mark.parametrize(
    "display_name",
    (
        "",
        "a" * 1_025,
        "bad\nname.safetensors",
        r"C:\private\model.safetensors",
        "/private/model.safetensors",
        r"\\server\share\model.safetensors",
    ),
)
def test_local_inspection_rejects_private_or_unbounded_display_names(
    tmp_path: Path,
    display_name: str,
) -> None:
    with pytest.raises(ScheduleContractError, match="display_name"):
        inspect_local_checkpoint_evidence(tmp_path / "model.safetensors", display_name=display_name)


def test_local_inspection_rejects_non_path_input() -> None:
    with pytest.raises(ScheduleContractError, match="path"):
        inspect_local_checkpoint_evidence(cast(Any, 3))


def test_local_inspection_maps_small_malformed_file_without_reopening_payload(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "small.safetensors"
    checkpoint.write_bytes(b"x")

    report = json.loads(inspect_local_checkpoint_evidence(checkpoint).report_json)

    assert report["reason_codes"] == ["safetensors.header_too_small"]
    assert report["source"]["header_bytes"] is None
    assert report["source"]["payload_bytes_read"] == 0


def test_local_inspection_maps_non_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"")
    monkeypatch.setattr(
        os,
        "fstat",
        lambda _: SimpleNamespace(st_mode=stat.S_IFDIR, st_size=0),
    )

    report = json.loads(inspect_local_checkpoint_evidence(checkpoint).report_json)

    assert report["reason_codes"] == ["checkpoint.not_regular_file"]


@pytest.mark.parametrize(
    ("error", "reason"),
    (
        (PermissionError(), "checkpoint.permission_denied"),
        (IsADirectoryError(), "checkpoint.not_regular_file"),
        (OSError(), "checkpoint.io_error"),
    ),
)
def test_local_inspection_maps_controlled_open_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: OSError,
    reason: str,
) -> None:
    def fail_open(*args: object, **kwargs: object) -> Any:
        raise error

    monkeypatch.setattr(Path, "open", fail_open)

    report = json.loads(
        inspect_local_checkpoint_evidence(tmp_path / "model.safetensors").report_json
    )

    assert report["reason_codes"] == [reason]


def test_local_inspection_converts_invalid_evidence_to_stable_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(_safetensors_bytes(_krea_header()))

    def reject_evidence(**kwargs: object) -> Any:
        raise ScheduleContractError("invalid evidence")

    monkeypatch.setattr(evidence_module, "resolve_krea2_variant", reject_evidence)

    report = json.loads(inspect_local_checkpoint_evidence(checkpoint).report_json)

    assert report["model_identity"]["reason_codes"] == ["checkpoint.evidence_invalid"]
    assert report["model_identity"]["confirmed_variant"] is None


def test_local_inspection_refuses_an_accidental_resolved_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(_safetensors_bytes(_krea_header()))
    resolution = SimpleNamespace(
        status=Krea2VariantResolutionStatus.RESOLVED,
    )
    monkeypatch.setattr(
        evidence_module,
        "resolve_krea2_variant",
        lambda **kwargs: resolution,
    )

    with pytest.raises(ScheduleContractError, match="cannot confirm"):
        inspect_local_checkpoint_evidence(checkpoint)


def test_checkpoint_inspection_modules_remain_accelerator_and_network_independent() -> None:
    root = Path(__file__).parents[1] / "comfyui_sigmax"
    source = "\n".join(
        (
            (root / "core" / "safetensors_header.py").read_text(encoding="utf-8"),
            (root / "profiles" / "checkpoint_evidence.py").read_text(encoding="utf-8"),
        )
    )

    tree = ast.parse(source)
    import_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    import_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    for forbidden in (
        "torch",
        "safetensors",
        "diffusers",
        "requests",
        "urllib",
        "socket",
        "mmap",
    ):
        assert forbidden not in import_roots
    for accelerator_call in ("torch.", ".cuda", ".mps", ".xpu", "device="):
        assert accelerator_call not in source
