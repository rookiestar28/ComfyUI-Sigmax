"""Strict loader for the packaged co-installation and host-mutation matrix."""

from __future__ import annotations

import hashlib
import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import Final, cast

from comfyui_sigmax.compatibility_matrix import (
    load_dependency_compatibility_matrix,
)
from comfyui_sigmax.core.schedule_contracts import ScheduleContractError
from comfyui_sigmax.host_mutation import HostMutationFinding, MutationVerdict

COINSTALLATION_MATRIX_SCHEMA: Final = "sigmax.co-installation-mutation-matrix/1"
COINSTALLATION_MATRIX_ENVELOPE_SCHEMA: Final = "sigmax.co-installation-mutation-matrix-envelope/1"
_MAX_BYTES: Final = 262_144
_FINGERPRINT: Final = re.compile(r"sha256:[0-9a-f]{64}")
_PRIVATE_PATH: Final = re.compile(r"(?:[A-Za-z]:[\\/]|/|\\\\)")
_SECRET_WORDS: Final = (
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)
_OPERATIONS: Final = frozenset(
    {
        "add_sigmax_node",
        "add_unrelated_node",
        "add_unrelated_scheduler",
        "clean_install",
        "double_shift",
        "idempotent_reload",
        "replace_model_patch",
        "replace_node",
        "replace_scheduler",
        "replace_torch_call",
    }
)
_FINDINGS: Final = frozenset(item.value for item in HostMutationFinding)
_VERDICTS: Final = frozenset(item.value for item in MutationVerdict)
_MATRIX_FIELDS: Final = frozenset({"context", "policy", "rows", "schema", "sources"})
_CONTEXT_FIELDS: Final = frozenset(
    {
        "baseline_snapshot_fingerprint",
        "built_in_node_ids",
        "dependency_compatibility_matrix_fingerprint",
        "external_reference_code_executed",
    }
)
_POLICY_FIELDS: Final = frozenset(
    {
        "external_reference_code_executed",
        "protected_existing_identities",
        "third_party_claims",
    }
)
_ROW_FIELDS: Final = frozenset(
    {
        "evidence_source",
        "expected_findings",
        "expected_verdict",
        "first_attempt",
        "first_report_fingerprint",
        "id",
        "observed_findings",
        "observed_verdict",
        "operation",
        "pack_id",
        "repeat",
        "repeat_report_fingerprint",
        "result_fingerprint",
        "status",
    }
)
_SOURCE_FIELDS: Final = frozenset({"path", "sha256"})


class CoInstallationMatrixError(ScheduleContractError):
    """Raised when co-installation mutation evidence is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CoInstallationMatrixError(f"duplicate JSON object name: {key}")
        result[key] = value
    return result


def _reject_float(value: str) -> object:
    raise CoInstallationMatrixError(f"untyped JSON float is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise CoInstallationMatrixError(f"non-finite JSON value is forbidden: {value}")


def _object(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CoInstallationMatrixError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise CoInstallationMatrixError(f"{label} must be an array")
    return value


def _exact(value: dict[str, object], fields: frozenset[str], *, label: str) -> None:
    if set(value) != fields:
        raise CoInstallationMatrixError(f"{label} fields do not match schema")


def _text(value: object, *, label: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CoInstallationMatrixError(f"{label} must be bounded non-empty text")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise CoInstallationMatrixError(f"{label} must be boolean")
    return value


def _fingerprint(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT.fullmatch(value):
        raise CoInstallationMatrixError(f"{label} must be a lowercase SHA-256 identity")
    return value


def _scan_safe(value: object, *, depth: int = 0) -> None:
    if depth > 16:
        raise CoInstallationMatrixError("co-installation matrix exceeds maximum depth")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        if len(value) > 1024:
            raise CoInstallationMatrixError("co-installation matrix string exceeds limit")
        if _PRIVATE_PATH.match(value):
            raise CoInstallationMatrixError(
                "co-installation matrix contains a private or absolute path"
            )
        return
    if isinstance(value, list):
        if len(value) > 128:
            raise CoInstallationMatrixError("co-installation collection exceeds limit")
        for child in value:
            _scan_safe(child, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 128:
            raise CoInstallationMatrixError("co-installation collection exceeds limit")
        for key, child in value.items():
            if not isinstance(key, str) or any(word in key.lower() for word in _SECRET_WORDS):
                raise CoInstallationMatrixError(
                    "co-installation matrix contains a forbidden field name"
                )
            _scan_safe(child, depth=depth + 1)
        return
    raise CoInstallationMatrixError("co-installation matrix contains unsupported JSON")


def _decode(payload: bytes | str) -> dict[str, object]:
    if isinstance(payload, str):
        if payload.startswith("\ufeff"):
            raise CoInstallationMatrixError("co-installation matrix must not contain a BOM")
        raw = payload.encode("utf-8")
    elif isinstance(payload, bytes):
        raw = payload
        if raw.startswith(b"\xef\xbb\xbf"):
            raise CoInstallationMatrixError("co-installation matrix must not contain a BOM")
    else:
        raise CoInstallationMatrixError("co-installation transport must be bytes or text")
    if not raw or len(raw) > _MAX_BYTES:
        raise CoInstallationMatrixError("co-installation transport size is invalid")
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise CoInstallationMatrixError("co-installation transport is not valid JSON") from exc
    root = _object(decoded, label="co-installation matrix envelope")
    _scan_safe(root)
    if _canonical(root) + b"\n" != raw:
        raise CoInstallationMatrixError("co-installation matrix must use canonical JSON")
    return root


def _validate_context(value: object) -> None:
    context = _object(value, label="co-installation context")
    _exact(context, _CONTEXT_FIELDS, label="co-installation context")
    _fingerprint(
        context["baseline_snapshot_fingerprint"],
        label="baseline snapshot fingerprint",
    )
    dependency_fingerprint = _fingerprint(
        context["dependency_compatibility_matrix_fingerprint"],
        label="dependency compatibility matrix fingerprint",
    )
    if dependency_fingerprint != load_dependency_compatibility_matrix().matrix_fingerprint:
        raise CoInstallationMatrixError("dependency compatibility matrix fingerprint drifted")
    if _boolean(
        context["external_reference_code_executed"],
        label="external reference execution context",
    ):
        raise CoInstallationMatrixError("external reference code execution is forbidden")
    node_ids = _array(context["built_in_node_ids"], label="built-in node IDs")
    string_node_ids = cast(list[str], node_ids)
    if (
        not node_ids
        or any(not isinstance(item, str) or not item.startswith("Sigmax.") for item in node_ids)
        or string_node_ids != sorted(set(string_node_ids))
    ):
        raise CoInstallationMatrixError("built-in node IDs must be unique and sorted")


def _validate_policy(value: object) -> None:
    policy = _object(value, label="co-installation policy")
    _exact(policy, _POLICY_FIELDS, label="co-installation policy")
    if _boolean(
        policy["external_reference_code_executed"],
        label="external reference code policy",
    ):
        raise CoInstallationMatrixError("external reference code cannot be executed")
    if not _boolean(
        policy["protected_existing_identities"],
        label="protected identity policy",
    ):
        raise CoInstallationMatrixError("existing identities must remain protected")
    if _boolean(policy["third_party_claims"], label="third-party claim policy"):
        raise CoInstallationMatrixError("synthetic evidence cannot make third-party claims")


def _findings(value: object, *, label: str) -> list[str]:
    items = _array(value, label=label)
    string_items = cast(list[str], items)
    if string_items != sorted(set(string_items)) or any(
        not isinstance(item, str) or item not in _FINDINGS for item in items
    ):
        raise CoInstallationMatrixError(f"{label} must be unique supported findings")
    return string_items


def _validate_row(value: object, *, index: int) -> tuple[str, str]:
    row = _object(value, label=f"co-installation row {index}")
    _exact(row, _ROW_FIELDS, label=f"co-installation row {index}")
    row_id = _text(row["id"], label=f"co-installation row {index} id")
    operation = _text(row["operation"], label=f"co-installation row {row_id} operation")
    if operation not in _OPERATIONS:
        raise CoInstallationMatrixError("co-installation operation is unsupported")
    _text(row["pack_id"], label=f"co-installation row {row_id} pack ID")
    expected_verdict = _text(
        row["expected_verdict"],
        label=f"co-installation row {row_id} expected verdict",
    )
    observed_verdict = _text(
        row["observed_verdict"],
        label=f"co-installation row {row_id} observed verdict",
    )
    if expected_verdict not in _VERDICTS or observed_verdict not in _VERDICTS:
        raise CoInstallationMatrixError("co-installation verdict is unsupported")
    expected_findings = _findings(
        row["expected_findings"],
        label=f"co-installation row {row_id} expected findings",
    )
    observed_findings = _findings(
        row["observed_findings"],
        label=f"co-installation row {row_id} observed findings",
    )
    if expected_verdict != observed_verdict or expected_findings != observed_findings:
        raise CoInstallationMatrixError("co-installation expectation disagrees with observation")
    if (observed_verdict == "allow") != (not observed_findings):
        raise CoInstallationMatrixError("co-installation verdict disagrees with findings")
    if row["status"] != "passed":
        raise CoInstallationMatrixError("only passed rows may be published")
    if row["first_attempt"] != "passed" or row["repeat"] != "passed":
        raise CoInstallationMatrixError("passed rows require first/repeat PASS")
    first = _fingerprint(
        row["first_report_fingerprint"],
        label=f"co-installation row {row_id} first report",
    )
    repeat = _fingerprint(
        row["repeat_report_fingerprint"],
        label=f"co-installation row {row_id} repeat report",
    )
    if first != repeat:
        raise CoInstallationMatrixError("co-installation first/repeat fingerprints differ")
    result_fingerprint = _fingerprint(
        row["result_fingerprint"],
        label=f"co-installation row {row_id} result",
    )
    evidence_source = _text(
        row["evidence_source"],
        label=f"co-installation row {row_id} evidence source",
    )
    evidence_projection = {
        field: row[field]
        for field in sorted(_ROW_FIELDS - {"evidence_source", "result_fingerprint"})
    }
    if result_fingerprint != _identity(evidence_projection):
        raise CoInstallationMatrixError(f"co-installation row {row_id} result fingerprint drifted")
    return row_id, evidence_source


def _validate_matrix(value: object) -> dict[str, object]:
    matrix = _object(value, label="co-installation mutation matrix")
    _exact(matrix, _MATRIX_FIELDS, label="co-installation mutation matrix")
    if matrix["schema"] != COINSTALLATION_MATRIX_SCHEMA:
        raise CoInstallationMatrixError("co-installation matrix schema is unsupported")
    _validate_context(matrix["context"])
    _validate_policy(matrix["policy"])
    rows = _array(matrix["rows"], label="co-installation rows")
    if len(rows) != 10:
        raise CoInstallationMatrixError("co-installation row inventory is incomplete")
    row_results = [_validate_row(row, index=index) for index, row in enumerate(rows)]
    row_ids = [row_id for row_id, _ in row_results]
    if row_ids != sorted(set(row_ids)):
        raise CoInstallationMatrixError("co-installation row IDs must be unique and sorted")
    sources = _array(matrix["sources"], label="co-installation sources")
    paths: list[str] = []
    for index, value in enumerate(sources):
        source = _object(value, label=f"co-installation source {index}")
        _exact(source, _SOURCE_FIELDS, label=f"co-installation source {index}")
        paths.append(_text(source["path"], label=f"co-installation source {index} path"))
        _fingerprint(
            source["sha256"],
            label=f"co-installation source {index} fingerprint",
        )
    if paths != sorted(set(paths)):
        raise CoInstallationMatrixError("co-installation sources must be unique and sorted")
    missing_sources = sorted(
        {evidence_source for _, evidence_source in row_results if evidence_source not in paths}
    )
    if missing_sources:
        raise CoInstallationMatrixError(
            f"co-installation evidence source is undeclared: {missing_sources}"
        )
    return matrix


@dataclass(frozen=True, slots=True)
class CoInstallationMutationMatrix:
    """Validated immutable co-installation matrix."""

    _matrix: dict[str, object]
    matrix_fingerprint: str

    @property
    def schema(self) -> str:
        return cast(str, self._matrix["schema"])

    def projection(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(_canonical(self._matrix)))

    def require_row(self, row_id: str) -> dict[str, object]:
        rows = cast(list[dict[str, object]], self._matrix["rows"])
        for row in rows:
            if row["id"] == row_id:
                return cast(dict[str, object], json.loads(_canonical(row)))
        raise CoInstallationMatrixError("unknown co-installation row")


def load_coinstallation_mutation_matrix(
    payload: bytes | str | None = None,
) -> CoInstallationMutationMatrix:
    """Load and validate the packaged matrix or caller-provided canonical transport."""

    if payload is None:
        payload = (
            importlib.resources.files("comfyui_sigmax.coinstallation")
            .joinpath("matrix_v1.json")
            .read_bytes()
        )
    envelope = _decode(payload)
    if set(envelope) != {"matrix", "matrix_fingerprint", "schema"}:
        raise CoInstallationMatrixError("co-installation envelope fields do not match schema")
    if envelope["schema"] != COINSTALLATION_MATRIX_ENVELOPE_SCHEMA:
        raise CoInstallationMatrixError("co-installation envelope schema is unsupported")
    matrix = _validate_matrix(envelope["matrix"])
    observed = _fingerprint(
        envelope["matrix_fingerprint"],
        label="co-installation matrix fingerprint",
    )
    expected = _identity(matrix)
    if observed != expected:
        raise CoInstallationMatrixError("co-installation matrix fingerprint drifted")
    return CoInstallationMutationMatrix(matrix, observed)


__all__ = [
    "COINSTALLATION_MATRIX_ENVELOPE_SCHEMA",
    "COINSTALLATION_MATRIX_SCHEMA",
    "CoInstallationMatrixError",
    "CoInstallationMutationMatrix",
    "load_coinstallation_mutation_matrix",
]
