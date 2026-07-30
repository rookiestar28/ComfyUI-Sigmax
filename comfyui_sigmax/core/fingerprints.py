"""Canonical projections and deterministic schedule fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from collections.abc import Mapping
from typing import Literal

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError, SigmaDomain
from comfyui_sigmax.core.validation import validate_sigma_schedule

FloatPrecision = Literal["float32", "float64"]

_NUMERICAL_SCHEMA = "sigmax.numerical-schedule/1"
_CONSTRUCTION_SCHEMA = "sigmax.schedule-artifact/1"
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_DEPTH = 32
_MAX_COLLECTION_LENGTH = 1024
_MAX_STRING_LENGTH = 4096
_MAX_CANONICAL_BYTES = 1_048_576
_MAX_INTEROPERABLE_INTEGER = (2**53) - 1
_CONSTRUCTION_FIELDS = frozenset(
    {
        "base_grid",
        "effective",
        "engine",
        "evidence",
        "numerical_fingerprint",
        "overrides",
        "ownership",
        "requested",
        "schema",
        "slicing",
        "source",
        "terminal",
        "transforms",
        "warnings",
    }
)


def float_to_ieee_hex(value: float, precision: FloatPrecision) -> str:
    """Encode one finite float as a normalized big-endian IEEE-754 bit token."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScheduleContractError("floating-point token input must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ScheduleContractError("floating-point token input must be finite")
    if normalized == 0.0:
        normalized = 0.0

    if precision == "float32":
        format_code = ">f"
    elif precision == "float64":
        format_code = ">d"
    else:
        raise ScheduleContractError("precision must be float32 or float64")

    try:
        return struct.pack(format_code, normalized).hex()
    except (OverflowError, struct.error) as error:
        raise ScheduleContractError(f"value cannot be represented as {precision}") from error


def build_numerical_projection(
    sigmas: tuple[float, ...],
    *,
    domain: SigmaDomain,
    precision: FloatPrecision,
) -> dict[str, object]:
    """Build the exact M1-08 numerical projection."""

    values = tuple(sigmas)
    if len(values) < 2:
        raise ScheduleContractError("numerical schedule requires at least one transition")
    normalized = validate_sigma_schedule(
        values,
        domain=domain,
        expected_steps=len(values) - 1,
        require_terminal_zero=True,
    )
    return {
        "domain": domain.value.casefold(),
        "precision": precision,
        "schema": _NUMERICAL_SCHEMA,
        "sigmas": [float_to_ieee_hex(value, precision) for value in normalized],
    }


def _normalize_projection(value: object, *, depth: int) -> object:
    if depth > _MAX_DEPTH:
        raise ScheduleContractError("canonical projection exceeds maximum depth")

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > _MAX_INTEROPERABLE_INTEGER:
            raise ScheduleContractError("integer exceeds interoperable JSON range")
        return value
    if isinstance(value, float):
        raise ScheduleContractError(
            "JSON floating-point values are forbidden; use typed IEEE-754 tokens"
        )
    if isinstance(value, str):
        if len(value) > _MAX_STRING_LENGTH:
            raise ScheduleContractError("string exceeds canonical projection limit")
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        if len(value) > _MAX_COLLECTION_LENGTH:
            raise ScheduleContractError("collection exceeds canonical projection limit")
        return [_normalize_projection(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION_LENGTH:
            raise ScheduleContractError("collection exceeds canonical projection limit")
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key.isascii() or not _KEY_PATTERN.fullmatch(key):
                raise ScheduleContractError(
                    "canonical projection keys must be controlled ASCII identifiers"
                )
            normalized[key] = _normalize_projection(child, depth=depth + 1)
        return normalized
    raise ScheduleContractError(f"unsupported canonical projection type: {type(value).__name__}")


def canonical_projection_bytes(projection: Mapping[str, object]) -> bytes:
    """Return canonical UTF-8 bytes for one bounded Sigmax projection."""

    if not isinstance(projection, Mapping):
        raise ScheduleContractError("canonical projection root must be a mapping")
    normalized = _normalize_projection(projection, depth=0)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_CANONICAL_BYTES:
        raise ScheduleContractError("canonical projection exceeds byte limit")
    return encoded


def _sha256_identity(preimage: bytes) -> str:
    return f"sha256:{hashlib.sha256(preimage).hexdigest()}"


def numerical_fingerprint(
    sigmas: tuple[float, ...],
    *,
    domain: SigmaDomain,
    precision: FloatPrecision,
) -> str:
    """Fingerprint an exact normalized numerical schedule."""

    projection = build_numerical_projection(sigmas, domain=domain, precision=precision)
    return _sha256_identity(canonical_projection_bytes(projection))


def construction_fingerprint(projection: Mapping[str, object]) -> str:
    """Validate and fingerprint one M1-08 construction projection."""

    if not isinstance(projection, Mapping):
        raise ScheduleContractError("construction projection must be a mapping")
    fields = set(projection)
    if fields != _CONSTRUCTION_FIELDS:
        missing = sorted(_CONSTRUCTION_FIELDS - fields)
        unknown = sorted(fields - _CONSTRUCTION_FIELDS)
        raise ScheduleContractError(
            f"construction projection fields do not match schema; "
            f"missing={missing}, unknown={unknown}"
        )
    if projection.get("schema") != _CONSTRUCTION_SCHEMA:
        raise ScheduleContractError(f"schema must be {_CONSTRUCTION_SCHEMA}")
    numerical_identity = projection.get("numerical_fingerprint")
    if not isinstance(numerical_identity, str) or not _SHA256_PATTERN.fullmatch(numerical_identity):
        raise ScheduleContractError("numerical_fingerprint must be a lowercase SHA-256 identity")

    return _sha256_identity(canonical_projection_bytes(projection))
