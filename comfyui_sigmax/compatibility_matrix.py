"""Strict loader for the packaged dependency compatibility matrix."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Final, cast

from comfyui_sigmax.core.schedule_contracts import ScheduleContractError

DEPENDENCY_COMPATIBILITY_MATRIX_SCHEMA: Final = "sigmax.dependency-compatibility-matrix/1"
DEPENDENCY_COMPATIBILITY_MATRIX_ENVELOPE_SCHEMA: Final = (
    "sigmax.dependency-compatibility-matrix-envelope/1"
)
COMPATIBILITY_INVARIANT_CONTRACT_SCHEMA: Final = "sigmax.compatibility-invariant-contract/1"
_MAX_BYTES: Final = 262_144
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}")
_PRIVATE_PATH: Final = re.compile(r"(?:[A-Za-z]:[\\/]|/|\\\\)")
_SECRET_WORDS: Final = (
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
_ROLES: Final = frozenset({"known_good", "latest_informational", "supported"})
_STATUSES: Final = frozenset({"not_evaluated", "passed", "unavailable"})
_REASONS: Final = frozenset({"approval_required", "compatible", "registry_access_denied"})
_PLATFORMS: Final = frozenset({"linux_container", "linux_host", "windows", "wsl"})
_MATRIX_FIELDS: Final = frozenset({"contract", "lanes", "policy", "schema", "sources"})
_LANE_FIELDS: Final = frozenset(
    {
        "blocking",
        "components",
        "evidence",
        "id",
        "platform",
        "reason",
        "role",
        "status",
    }
)
_COMPONENT_FIELDS: Final = frozenset(
    {"comfy_api", "comfyui", "container", "diffusers", "python", "torch"}
)
_EVIDENCE_FIELDS: Final = frozenset(
    {"first_attempt", "kind", "repeat", "result_fingerprint", "source"}
)
_POLICY_FIELDS: Final = frozenset(
    {
        "api_stability",
        "official_container_requires_resolvable_digest",
        "official_container_unavailable_is_blocking",
        "third_party_container_substitution",
        "known_good_is_blocking",
        "latest_can_expand_support",
        "reference_diffusers",
        "supported_comfyui",
        "supported_python",
        "unavailable_is_pass",
    }
)
_CONTRACT_FIELDS: Final = frozenset({"expected", "id", "schema", "source_fingerprints"})
_EXPECTED_FIELDS: Final = frozenset(
    {
        "benchmark_matrix_fingerprint",
        "lane_contract_fingerprint",
        "mandatory_dependencies",
        "test_selection_fingerprint",
    }
)
_SOURCE_FIELDS: Final = frozenset({"path", "sha256"})


class CompatibilityMatrixError(ScheduleContractError):
    """Raised when dependency compatibility evidence is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CompatibilityMatrixError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise CompatibilityMatrixError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise CompatibilityMatrixError(f"non-finite JSON value is forbidden: {value}")


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CompatibilityMatrixError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise CompatibilityMatrixError(f"{name} must be an array")
    return value


def _exact(value: dict[str, object], fields: frozenset[str], *, name: str) -> None:
    if set(value) != fields:
        raise CompatibilityMatrixError(f"{name} fields do not match schema")


def _text(value: object, *, name: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CompatibilityMatrixError(f"{name} must be bounded non-empty text")
    return value


def _optional_text(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name=name)


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise CompatibilityMatrixError(f"{name} must be a boolean")
    return value


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CompatibilityMatrixError(f"{name} must be a non-negative integer")
    return value


def _fingerprint(value: object, *, name: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CompatibilityMatrixError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _scan_safe(value: object, *, depth: int = 0) -> None:
    if depth > 16:
        raise CompatibilityMatrixError("compatibility matrix exceeds maximum depth")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if len(value) > 1024:
            raise CompatibilityMatrixError("compatibility matrix string exceeds limit")
        if _PRIVATE_PATH.match(value):
            raise CompatibilityMatrixError(
                "compatibility matrix contains a private or absolute path"
            )
        return
    if isinstance(value, list):
        if len(value) > 128:
            raise CompatibilityMatrixError("compatibility matrix collection exceeds limit")
        for child in value:
            _scan_safe(child, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 128:
            raise CompatibilityMatrixError("compatibility matrix collection exceeds limit")
        for key, child in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in _SECRET_WORDS):
                raise CompatibilityMatrixError(
                    "compatibility matrix contains a forbidden field name"
                )
            _scan_safe(child, depth=depth + 1)
        return
    raise CompatibilityMatrixError("compatibility matrix contains an unsupported JSON value")


def _decode(payload: bytes | str) -> dict[str, object]:
    if isinstance(payload, str):
        if payload.startswith("\ufeff"):
            raise CompatibilityMatrixError("compatibility matrix must not contain a BOM")
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
        if raw.startswith(b"\xef\xbb\xbf"):
            raise CompatibilityMatrixError("compatibility matrix must not contain a BOM")
    else:
        raise CompatibilityMatrixError("compatibility matrix transport must be bytes or text")
    if not raw or len(raw) > _MAX_BYTES:
        raise CompatibilityMatrixError(
            "compatibility matrix transport size is outside the allowed range"
        )
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CompatibilityMatrixError("compatibility matrix transport is not valid JSON") from exc
    root = _object(decoded, name="compatibility matrix envelope")
    _scan_safe(root)
    if _canonical(root) + b"\n" != raw:
        raise CompatibilityMatrixError("compatibility matrix transport must use canonical JSON")
    return root


def _validate_contract(value: object) -> None:
    contract = _object(value, name="compatibility invariant contract")
    _exact(contract, _CONTRACT_FIELDS, name="compatibility invariant contract")
    if contract["schema"] != COMPATIBILITY_INVARIANT_CONTRACT_SCHEMA:
        raise CompatibilityMatrixError("compatibility invariant contract schema is unsupported")
    _text(contract["id"], name="compatibility invariant contract id")
    expected = _object(contract["expected"], name="compatibility expected identities")
    _exact(expected, _EXPECTED_FIELDS, name="compatibility expected identities")
    _fingerprint(
        expected["benchmark_matrix_fingerprint"],
        name="benchmark matrix fingerprint",
    )
    _fingerprint(
        expected["lane_contract_fingerprint"],
        name="compatibility lane contract fingerprint",
    )
    _fingerprint(
        expected["test_selection_fingerprint"],
        name="compatibility test-selection fingerprint",
    )
    if _integer(expected["mandatory_dependencies"], name="mandatory dependency count") != 0:
        raise CompatibilityMatrixError("mandatory runtime dependencies must remain zero")
    sources = _array(contract["source_fingerprints"], name="compatibility source fingerprints")
    if not sources:
        raise CompatibilityMatrixError("compatibility invariant contract must identify sources")
    paths: list[str] = []
    for index, item in enumerate(sources):
        source = _object(item, name=f"compatibility source {index}")
        _exact(source, _SOURCE_FIELDS, name=f"compatibility source {index}")
        paths.append(_text(source["path"], name=f"compatibility source {index} path"))
        _fingerprint(source["sha256"], name=f"compatibility source {index} fingerprint")
    if paths != sorted(set(paths)):
        raise CompatibilityMatrixError("compatibility invariant sources must be unique and sorted")


def _validate_policy(value: object) -> None:
    policy = _object(value, name="compatibility policy")
    _exact(policy, _POLICY_FIELDS, name="compatibility policy")
    if not _boolean(policy["known_good_is_blocking"], name="known-good blocking policy"):
        raise CompatibilityMatrixError("known-good lanes must remain blocking")
    if _boolean(policy["latest_can_expand_support"], name="latest support-expansion policy"):
        raise CompatibilityMatrixError("latest lanes cannot silently expand support")
    if _boolean(policy["unavailable_is_pass"], name="unavailable PASS policy"):
        raise CompatibilityMatrixError("unavailable lanes cannot count as PASS")
    if not _boolean(
        policy["official_container_requires_resolvable_digest"],
        name="official container digest policy",
    ):
        raise CompatibilityMatrixError("official container execution requires a resolvable digest")
    if _boolean(
        policy["official_container_unavailable_is_blocking"],
        name="official container unavailable blocking policy",
    ):
        raise CompatibilityMatrixError("unavailable official container cannot block acceptance")
    if _boolean(
        policy["third_party_container_substitution"],
        name="third-party container substitution policy",
    ):
        raise CompatibilityMatrixError("third-party container substitution is forbidden")
    supported_python = _array(policy["supported_python"], name="supported Python")
    if supported_python != ["3.10", "3.13"]:
        raise CompatibilityMatrixError("supported Python policy drifted")
    if policy["supported_comfyui"] != "0.29.0":
        raise CompatibilityMatrixError("supported ComfyUI policy drifted")
    if policy["reference_diffusers"] != "0.39.0":
        raise CompatibilityMatrixError("reference Diffusers policy drifted")
    api = _object(policy["api_stability"], name="Comfy API stability policy")
    if api != {"v0_0_2": "experimental"}:
        raise CompatibilityMatrixError("Comfy API stability policy drifted")


def _validate_lane(value: object, *, index: int) -> str:
    lane = _object(value, name=f"compatibility lane {index}")
    _exact(lane, _LANE_FIELDS, name=f"compatibility lane {index}")
    lane_id = _text(lane["id"], name=f"compatibility lane {index} id")
    role = _text(lane["role"], name=f"compatibility lane {lane_id} role")
    status = _text(lane["status"], name=f"compatibility lane {lane_id} status")
    reason = _text(lane["reason"], name=f"compatibility lane {lane_id} reason")
    platform = _text(lane["platform"], name=f"compatibility lane {lane_id} platform")
    blocking = _boolean(lane["blocking"], name=f"compatibility lane {lane_id} blocking")
    if role not in _ROLES or status not in _STATUSES or reason not in _REASONS:
        raise CompatibilityMatrixError(f"compatibility lane {lane_id} has an unsupported state")
    if platform not in _PLATFORMS:
        raise CompatibilityMatrixError(f"compatibility lane {lane_id} has an unsupported platform")
    if role == "latest_informational" and blocking:
        raise CompatibilityMatrixError("latest compatibility lanes cannot be blocking")
    if role == "known_good" and not blocking:
        raise CompatibilityMatrixError("known-good compatibility lanes must be blocking")
    if status == "passed" and reason != "compatible":
        raise CompatibilityMatrixError("passed compatibility lanes must be compatible")
    if status != "passed" and reason == "compatible":
        raise CompatibilityMatrixError("non-passed lanes cannot be compatible")

    components = _object(lane["components"], name=f"compatibility lane {lane_id} components")
    _exact(
        components,
        _COMPONENT_FIELDS,
        name=f"compatibility lane {lane_id} components",
    )
    for key, component in components.items():
        _optional_text(component, name=f"compatibility lane {lane_id} {key}")

    evidence = _object(lane["evidence"], name=f"compatibility lane {lane_id} evidence")
    _exact(
        evidence,
        _EVIDENCE_FIELDS,
        name=f"compatibility lane {lane_id} evidence",
    )
    _text(evidence["kind"], name=f"compatibility lane {lane_id} evidence kind")
    _text(evidence["source"], name=f"compatibility lane {lane_id} evidence source")
    result = _fingerprint(
        evidence["result_fingerprint"],
        name=f"compatibility lane {lane_id} result fingerprint",
        optional=True,
    )
    first = _text(
        evidence["first_attempt"],
        name=f"compatibility lane {lane_id} first attempt",
    )
    repeat = _text(evidence["repeat"], name=f"compatibility lane {lane_id} repeat")
    if status == "passed":
        if result is None or first != "passed" or repeat != "passed":
            raise CompatibilityMatrixError(
                "passed compatibility lanes require first/repeat evidence"
            )
    elif result is not None or first != "not_evaluated" or repeat != "not_evaluated":
        raise CompatibilityMatrixError(
            "non-passed compatibility lanes cannot carry passing evidence"
        )
    container = components["container"]
    if (
        platform == "linux_container"
        and status == "passed"
        and (not isinstance(container, str) or "@sha256:" not in container)
    ):
        raise CompatibilityMatrixError(
            "executed container compatibility lanes require an immutable digest"
        )
    return lane_id


def _validate_matrix(value: object) -> dict[str, object]:
    matrix = _object(value, name="dependency compatibility matrix")
    _exact(matrix, _MATRIX_FIELDS, name="dependency compatibility matrix")
    if matrix["schema"] != DEPENDENCY_COMPATIBILITY_MATRIX_SCHEMA:
        raise CompatibilityMatrixError("dependency compatibility matrix schema is unsupported")
    _validate_contract(matrix["contract"])
    _validate_policy(matrix["policy"])
    lanes = _array(matrix["lanes"], name="compatibility lanes")
    if not lanes:
        raise CompatibilityMatrixError("compatibility matrix must contain lanes")
    lane_ids = [_validate_lane(lane, index=index) for index, lane in enumerate(lanes)]
    if lane_ids != sorted(set(lane_ids)):
        raise CompatibilityMatrixError("compatibility lane IDs must be unique and sorted")
    sources = _array(matrix["sources"], name="compatibility matrix sources")
    source_paths: list[str] = []
    for index, item in enumerate(sources):
        source = _object(item, name=f"compatibility matrix source {index}")
        _exact(source, _SOURCE_FIELDS, name=f"compatibility matrix source {index}")
        source_paths.append(_text(source["path"], name=f"compatibility matrix source {index} path"))
        _fingerprint(source["sha256"], name=f"compatibility matrix source {index} fingerprint")
    if source_paths != sorted(set(source_paths)):
        raise CompatibilityMatrixError("compatibility matrix sources must be unique and sorted")
    return matrix


@dataclass(frozen=True, slots=True)
class DependencyCompatibilityMatrix:
    """Validated immutable dependency compatibility evidence."""

    _matrix: dict[str, object]
    matrix_fingerprint: str

    @property
    def schema(self) -> str:
        return cast(str, self._matrix["schema"])

    def projection(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(_canonical(self._matrix)))

    def require_lane(self, lane_id: str) -> dict[str, object]:
        lanes = cast(list[dict[str, object]], self._matrix["lanes"])
        for lane in lanes:
            if lane["id"] == lane_id:
                return cast(dict[str, object], json.loads(_canonical(lane)))
        raise CompatibilityMatrixError(f"unknown compatibility lane: {lane_id}")


def load_dependency_compatibility_matrix(
    payload: bytes | str | None = None,
) -> DependencyCompatibilityMatrix:
    """Load and validate canonical dependency compatibility evidence."""

    if payload is None:
        payload = (
            importlib.resources.files("comfyui_sigmax.compatibility")
            .joinpath("matrix_v1.json")
            .read_bytes()
        )
    envelope = _decode(payload)
    if set(envelope) != {"matrix", "matrix_fingerprint", "schema"}:
        raise CompatibilityMatrixError(
            "dependency compatibility envelope fields do not match schema"
        )
    if envelope["schema"] != DEPENDENCY_COMPATIBILITY_MATRIX_ENVELOPE_SCHEMA:
        raise CompatibilityMatrixError("dependency compatibility envelope schema is unsupported")
    matrix = _validate_matrix(envelope["matrix"])
    advertised = _fingerprint(
        envelope["matrix_fingerprint"], name="dependency compatibility matrix fingerprint"
    )
    computed = _identity(_canonical(matrix))
    if advertised != computed:
        raise CompatibilityMatrixError("dependency compatibility matrix fingerprint does not match")
    return DependencyCompatibilityMatrix(matrix, computed)
