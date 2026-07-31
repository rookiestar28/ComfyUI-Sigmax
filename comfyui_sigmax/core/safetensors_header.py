"""Bounded safetensors header parsing without tensor payload access."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import BinaryIO, Final, NoReturn, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

SAFETENSORS_HEADER_MAX_BYTES: Final = 100_000_000
_MAX_JSON_NESTING: Final = 128

_DTYPE_BITS: Final = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


class SafetensorsHeaderReason(str, Enum):
    """Stable fail-closed reasons for an invalid safetensors header."""

    HEADER_TOO_SMALL = "safetensors.header_too_small"
    HEADER_TOO_LARGE = "safetensors.header_too_large"
    HEADER_TRUNCATED = "safetensors.header_truncated"
    HEADER_INVALID_UTF8 = "safetensors.header_invalid_utf8"
    HEADER_INVALID_JSON = "safetensors.header_invalid_json"
    HEADER_DUPLICATE_KEY = "safetensors.header_duplicate_key"
    METADATA_INVALID = "safetensors.metadata_invalid"
    TENSOR_DESCRIPTOR_INVALID = "safetensors.tensor_descriptor_invalid"
    DTYPE_UNSUPPORTED = "safetensors.dtype_unsupported"
    SHAPE_INVALID = "safetensors.shape_invalid"
    TENSOR_SIZE_OVERFLOW = "safetensors.tensor_size_overflow"
    TENSOR_MISALIGNED = "safetensors.tensor_misaligned"
    OFFSETS_INVALID = "safetensors.offsets_invalid"
    BUFFER_INCOMPLETE = "safetensors.buffer_incomplete"


class SafetensorsHeaderError(ScheduleContractError):
    """One controlled, path-free safetensors structural rejection."""

    def __init__(self, reason: SafetensorsHeaderReason) -> None:
        if not isinstance(reason, SafetensorsHeaderReason):
            raise ScheduleContractError("safetensors header reason is unsupported")
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetensorsTensorDescriptor:
    """One tensor declaration from the header, never its payload."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ScheduleContractError("safetensors tensor name must be text")
        if self.dtype not in _DTYPE_BITS:
            raise ScheduleContractError("safetensors tensor dtype is unsupported")
        if not isinstance(self.shape, tuple) or any(
            type(value) is not int or value < 0 for value in self.shape
        ):
            raise ScheduleContractError("safetensors tensor shape is invalid")
        if (
            not isinstance(self.data_offsets, tuple)
            or len(self.data_offsets) != 2
            or any(type(value) is not int or value < 0 for value in self.data_offsets)
            or self.data_offsets[1] < self.data_offsets[0]
        ):
            raise ScheduleContractError("safetensors tensor offsets are invalid")

    @property
    def data_bytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetensorsHeader:
    """Validated header-only structure bound to the local file size."""

    header_bytes: int
    file_bytes: int
    data_bytes: int
    metadata: tuple[tuple[str, str], ...]
    tensors: tuple[SafetensorsTensorDescriptor, ...]

    def __post_init__(self) -> None:
        if (
            type(self.header_bytes) is not int
            or not 0 <= self.header_bytes <= SAFETENSORS_HEADER_MAX_BYTES
            or type(self.file_bytes) is not int
            or self.file_bytes < 8 + self.header_bytes
            or type(self.data_bytes) is not int
            or self.data_bytes != self.file_bytes - 8 - self.header_bytes
        ):
            raise ScheduleContractError("safetensors header size contract is invalid")
        if (
            not isinstance(self.metadata, tuple)
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(value, str) for value in item)
                for item in self.metadata
            )
            or tuple(sorted(self.metadata)) != self.metadata
            or len({item[0] for item in self.metadata}) != len(self.metadata)
        ):
            raise ScheduleContractError("safetensors metadata contract is invalid")
        if not isinstance(self.tensors, tuple) or any(
            not isinstance(item, SafetensorsTensorDescriptor) for item in self.tensors
        ):
            raise ScheduleContractError("safetensors tensors contract is invalid")
        canonical_tensors = tuple(
            sorted(self.tensors, key=lambda item: (*item.data_offsets, item.name))
        )
        if canonical_tensors != self.tensors or len({item.name for item in self.tensors}) != len(
            self.tensors
        ):
            raise ScheduleContractError("safetensors tensors contract is not canonical")
        expected_offset = 0
        for tensor in self.tensors:
            if tensor.data_offsets[0] != expected_offset:
                raise ScheduleContractError("safetensors tensor offsets are not contiguous")
            expected_offset = tensor.data_offsets[1]
        if expected_offset != self.data_bytes:
            raise ScheduleContractError("safetensors tensor coverage is incomplete")


class _DuplicateKeyError(ValueError):
    pass


def _fail(reason: SafetensorsHeaderReason) -> NoReturn:
    raise SafetensorsHeaderError(reason)


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def _reject_constant(_: str) -> object:
    raise ValueError("non-finite JSON constants are forbidden")


def _validate_json_nesting(text: str) -> None:
    """Apply one deterministic nesting ceiling without counting delimiters in strings."""

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > _MAX_JSON_NESTING:
                _fail(SafetensorsHeaderReason.HEADER_INVALID_JSON)
        elif character in "]}":
            depth -= 1


def _metadata(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        _fail(SafetensorsHeaderReason.METADATA_INVALID)
    normalized: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            _fail(SafetensorsHeaderReason.METADATA_INVALID)
        normalized.append((key, item))
    return tuple(sorted(normalized))


def _integer(value: object, *, reason: SafetensorsHeaderReason) -> int:
    if type(value) is not int or value < 0 or value > sys.maxsize:
        _fail(reason)
    return value


def _tensor(name: str, value: object) -> SafetensorsTensorDescriptor:
    if not isinstance(value, Mapping) or set(value) != {"dtype", "shape", "data_offsets"}:
        _fail(SafetensorsHeaderReason.TENSOR_DESCRIPTOR_INVALID)
    dtype = value["dtype"]
    if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
        _fail(SafetensorsHeaderReason.DTYPE_UNSUPPORTED)
    raw_shape = value["shape"]
    if not isinstance(raw_shape, list):
        _fail(SafetensorsHeaderReason.SHAPE_INVALID)
    shape = tuple(
        _integer(item, reason=SafetensorsHeaderReason.SHAPE_INVALID) for item in raw_shape
    )
    raw_offsets = value["data_offsets"]
    if not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
        _fail(SafetensorsHeaderReason.OFFSETS_INVALID)
    offsets = tuple(
        _integer(item, reason=SafetensorsHeaderReason.OFFSETS_INVALID) for item in raw_offsets
    )
    start, end = cast(tuple[int, int], offsets)
    if end < start:
        _fail(SafetensorsHeaderReason.OFFSETS_INVALID)

    elements = 1
    for dimension in shape:
        if dimension and elements > sys.maxsize // dimension:
            _fail(SafetensorsHeaderReason.TENSOR_SIZE_OVERFLOW)
        elements *= dimension
    bits = _DTYPE_BITS[dtype]
    if elements and elements > sys.maxsize // bits:
        _fail(SafetensorsHeaderReason.TENSOR_SIZE_OVERFLOW)
    total_bits = elements * bits
    if total_bits % 8:
        _fail(SafetensorsHeaderReason.TENSOR_MISALIGNED)
    if end - start != total_bits // 8:
        _fail(SafetensorsHeaderReason.TENSOR_DESCRIPTOR_INVALID)
    return SafetensorsTensorDescriptor(
        name=name,
        dtype=dtype,
        shape=shape,
        data_offsets=(start, end),
    )


def read_safetensors_header(stream: BinaryIO, *, file_size: int) -> SafetensorsHeader:
    """Read and validate exactly the safetensors prefix and JSON header."""

    if type(file_size) is not int or file_size < 0:
        raise ScheduleContractError("safetensors file_size must be a non-negative integer")
    prefix = stream.read(8)
    if len(prefix) != 8:
        _fail(SafetensorsHeaderReason.HEADER_TOO_SMALL)
    header_length = int.from_bytes(prefix, "little", signed=False)
    if header_length > SAFETENSORS_HEADER_MAX_BYTES:
        _fail(SafetensorsHeaderReason.HEADER_TOO_LARGE)
    if header_length > file_size - 8:
        _fail(SafetensorsHeaderReason.HEADER_TRUNCATED)
    raw_header = stream.read(header_length)
    if len(raw_header) != header_length:
        _fail(SafetensorsHeaderReason.HEADER_TRUNCATED)
    if not raw_header or raw_header[0] != ord("{"):
        _fail(SafetensorsHeaderReason.HEADER_INVALID_JSON)
    try:
        text = raw_header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SafetensorsHeaderError(SafetensorsHeaderReason.HEADER_INVALID_UTF8) from exc
    try:
        # CRITICAL: decoder recursion limits differ by Python version; enforce one stable bound.
        _validate_json_nesting(text)
        decoder = json.JSONDecoder(
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
        decoded, end = decoder.raw_decode(text)
        if any(character != " " for character in text[end:]):
            _fail(SafetensorsHeaderReason.HEADER_INVALID_JSON)
    except _DuplicateKeyError as exc:
        raise SafetensorsHeaderError(SafetensorsHeaderReason.HEADER_DUPLICATE_KEY) from exc
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SafetensorsHeaderError(SafetensorsHeaderReason.HEADER_INVALID_JSON) from exc
    metadata: tuple[tuple[str, str], ...] = ()
    tensors: list[SafetensorsTensorDescriptor] = []
    for name, value in decoded.items():
        if name == "__metadata__":
            metadata = _metadata(value)
        else:
            tensors.append(_tensor(name, value))
    tensors.sort(key=lambda item: (*item.data_offsets, item.name))

    expected_offset = 0
    for tensor in tensors:
        if tensor.data_offsets[0] != expected_offset:
            _fail(SafetensorsHeaderReason.OFFSETS_INVALID)
        expected_offset = tensor.data_offsets[1]
    data_bytes = file_size - 8 - header_length
    if expected_offset != data_bytes:
        _fail(SafetensorsHeaderReason.BUFFER_INCOMPLETE)
    return SafetensorsHeader(
        header_bytes=header_length,
        file_bytes=file_size,
        data_bytes=data_bytes,
        metadata=metadata,
        tensors=tuple(tensors),
    )


__all__ = [
    "SAFETENSORS_HEADER_MAX_BYTES",
    "SafetensorsHeader",
    "SafetensorsHeaderError",
    "SafetensorsHeaderReason",
    "SafetensorsTensorDescriptor",
    "read_safetensors_header",
]
