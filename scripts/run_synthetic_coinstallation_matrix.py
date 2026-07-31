"""Execute the fixed repository-owned M7-08 synthetic mutation fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Final, cast

from comfyui_sigmax.adapters.registration import builtin_node_registry
from comfyui_sigmax.compatibility_matrix import load_dependency_compatibility_matrix
from comfyui_sigmax.host_mutation import (
    HostMutationFinding,
    HostMutationSnapshot,
    MutationVerdict,
    NodeRegistryIdentity,
    SchedulerRegistryIdentity,
    evaluate_host_mutation,
)

ROOT: Final = Path(__file__).resolve().parents[1]
FIXTURES: Final = ROOT / "tests" / "coinstallation" / "fixtures" / "synthetic_mutations_v1.json"
EVIDENCE_SCHEMA: Final = "sigmax.synthetic-host-mutation-evidence/1"
FIXTURE_SCHEMA: Final = "sigmax.synthetic-host-mutation-fixtures/1"
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


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _synthetic_identity(label: str) -> str:
    return _identity({"synthetic_identity": label})


def build_baseline_snapshot() -> HostMutationSnapshot:
    """Bind the synthetic host baseline to the actual built-in node catalog."""

    registry = builtin_node_registry()
    nodes = tuple(
        NodeRegistryIdentity(
            node_id=entry.node_id,
            provider="comfyui_sigmax",
            definition_fingerprint=(
                "sha256:" + hashlib.sha256(entry.source_payload_json.encode("utf-8")).hexdigest()
            ),
        )
        for entry in registry.entries
    )
    return HostMutationSnapshot(
        nodes=nodes,
        schedulers=(
            SchedulerRegistryIdentity(
                name="simple",
                provider="comfyui",
                handler_fingerprint=_synthetic_identity("comfy.samplers.simple@known-good"),
            ),
        ),
        torch_call_fingerprint=_synthetic_identity("torch.nn.Module.__call__@unmodified"),
        model_patch_fingerprint=_synthetic_identity("model.patch.state@pristine"),
        schedule_ownership="external_sigmas",
        construction_shift_count=1,
        model_native_shifted=False,
    )


def _with_node(
    snapshot: HostMutationSnapshot,
    identity: NodeRegistryIdentity,
) -> HostMutationSnapshot:
    by_id = {item.node_id: item for item in snapshot.nodes}
    by_id[identity.node_id] = identity
    return replace(snapshot, nodes=tuple(by_id[key] for key in sorted(by_id)))


def _with_scheduler(
    snapshot: HostMutationSnapshot,
    identity: SchedulerRegistryIdentity,
) -> HostMutationSnapshot:
    by_name = {item.name: item for item in snapshot.schedulers}
    by_name[identity.name] = identity
    return replace(
        snapshot,
        schedulers=tuple(by_name[key] for key in sorted(by_name)),
    )


def apply_operation(
    snapshot: HostMutationSnapshot,
    operation: str,
) -> HostMutationSnapshot:
    """Apply one fixed declarative operation; arbitrary code is not accepted."""

    if operation not in _OPERATIONS:
        raise RuntimeError("synthetic mutation operation is unsupported")
    if operation == "clean_install":
        return snapshot
    if operation == "idempotent_reload":
        return _with_node(
            snapshot,
            NodeRegistryIdentity(
                node_id="Other.ReloadNode",
                provider="synthetic.reload",
                definition_fingerprint=_synthetic_identity("other.reload.node@v1"),
            ),
        )
    if operation == "add_unrelated_node":
        return _with_node(
            snapshot,
            NodeRegistryIdentity(
                node_id="Other.ExampleNode",
                provider="synthetic.node_addition",
                definition_fingerprint=_synthetic_identity("other.example.node@v1"),
            ),
        )
    if operation == "replace_node":
        target = snapshot.nodes[0]
        return _with_node(
            snapshot,
            NodeRegistryIdentity(
                node_id=target.node_id,
                provider="synthetic.node_collision",
                definition_fingerprint=_synthetic_identity("collision.replacement@v1"),
            ),
        )
    if operation == "add_sigmax_node":
        return _with_node(
            snapshot,
            NodeRegistryIdentity(
                node_id="Sigmax.ForeignPackNode",
                provider="synthetic.namespace_hijack",
                definition_fingerprint=_synthetic_identity("foreign.sigmax.node@v1"),
            ),
        )
    if operation == "add_unrelated_scheduler":
        return _with_scheduler(
            snapshot,
            SchedulerRegistryIdentity(
                name="other_scheduler",
                provider="synthetic.scheduler_addition",
                handler_fingerprint=_synthetic_identity("other.scheduler@v1"),
            ),
        )
    if operation == "replace_scheduler":
        return _with_scheduler(
            snapshot,
            SchedulerRegistryIdentity(
                name="simple",
                provider="synthetic.scheduler_overwrite",
                handler_fingerprint=_synthetic_identity("replacement.scheduler@v1"),
            ),
        )
    if operation == "replace_torch_call":
        return replace(
            snapshot,
            torch_call_fingerprint=_synthetic_identity("wrapped.module.call@v1"),
        )
    if operation == "replace_model_patch":
        return replace(
            snapshot,
            model_patch_fingerprint=_synthetic_identity("global.model.patch@v1"),
        )
    if operation == "double_shift":
        return replace(
            snapshot,
            schedule_ownership="model_native",
            construction_shift_count=2,
            model_native_shifted=True,
        )
    raise AssertionError("fixed operation dispatch is incomplete")


def _load_fixtures() -> list[dict[str, Any]]:
    root = _object(json.loads(FIXTURES.read_text(encoding="utf-8")), label="fixture root")
    if set(root) != {"rows", "schema"} or root["schema"] != FIXTURE_SCHEMA:
        raise RuntimeError("synthetic mutation fixture schema is unsupported")
    rows = root["rows"]
    if not isinstance(rows, list) or len(rows) != 10:
        raise RuntimeError("synthetic mutation fixture inventory is incomplete")
    result: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, value in enumerate(rows):
        row = _object(value, label=f"synthetic mutation fixture {index}")
        if set(row) != {
            "expected_findings",
            "expected_verdict",
            "id",
            "operation",
            "pack_id",
        }:
            raise RuntimeError("synthetic mutation fixture fields do not match schema")
        if row["operation"] not in _OPERATIONS:
            raise RuntimeError("synthetic mutation fixture operation is unsupported")
        if row["expected_verdict"] not in {item.value for item in MutationVerdict}:
            raise RuntimeError("synthetic mutation expected verdict is unsupported")
        findings = row["expected_findings"]
        if (
            not isinstance(findings, list)
            or findings != sorted(set(findings))
            or any(
                item not in {finding.value for finding in HostMutationFinding} for item in findings
            )
        ):
            raise RuntimeError("synthetic mutation expected findings are invalid")
        ids.append(cast(str, row["id"]))
        result.append(row)
    if ids != sorted(set(ids)):
        raise RuntimeError("synthetic mutation fixture IDs must be unique and sorted")
    return result


def build_evidence() -> dict[str, object]:
    """Execute every fixture twice from a fresh immutable baseline."""

    baseline = build_baseline_snapshot()
    evidence_rows: list[dict[str, object]] = []
    for fixture in _load_fixtures():
        first_after = apply_operation(baseline, cast(str, fixture["operation"]))
        if fixture["operation"] == "idempotent_reload":
            reloaded = apply_operation(first_after, cast(str, fixture["operation"]))
            if reloaded != first_after:
                raise RuntimeError("synthetic reload is not idempotent")
        first = evaluate_host_mutation(
            before=baseline,
            after=first_after,
            pack_id=cast(str, fixture["pack_id"]),
        )
        repeat_after = apply_operation(
            build_baseline_snapshot(),
            cast(str, fixture["operation"]),
        )
        repeat = evaluate_host_mutation(
            before=build_baseline_snapshot(),
            after=repeat_after,
            pack_id=cast(str, fixture["pack_id"]),
        )
        observed_findings = [item.value for item in first.findings]
        expected_findings = cast(list[str], fixture["expected_findings"])
        matched = (
            first.verdict.value == fixture["expected_verdict"]
            and repeat.verdict == first.verdict
            and observed_findings == expected_findings
            and [item.value for item in repeat.findings] == expected_findings
            and repeat.report_fingerprint == first.report_fingerprint
        )
        if not matched:
            raise RuntimeError("synthetic mutation result disagrees with expectation")
        evidence_rows.append(
            {
                "expected_findings": expected_findings,
                "expected_verdict": fixture["expected_verdict"],
                "first_attempt": "passed",
                "first_report_fingerprint": first.report_fingerprint,
                "id": fixture["id"],
                "observed_findings": observed_findings,
                "observed_verdict": first.verdict.value,
                "operation": fixture["operation"],
                "pack_id": fixture["pack_id"],
                "repeat": "passed",
                "repeat_report_fingerprint": repeat.report_fingerprint,
                "status": "passed",
            }
        )
    compatibility = load_dependency_compatibility_matrix()
    context = {
        "baseline_snapshot_fingerprint": baseline.snapshot_fingerprint,
        "built_in_node_ids": [item.node_id for item in baseline.nodes],
        "dependency_compatibility_matrix_fingerprint": compatibility.matrix_fingerprint,
        "external_reference_code_executed": False,
    }
    return {
        "context": context,
        "evidence_fingerprint": _identity(
            {
                "context": context,
                "rows": evidence_rows,
            }
        ),
        "rows": evidence_rows,
        "schema": EVIDENCE_SCHEMA,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    payload = _canonical(build_evidence()) + b"\n"
    target = arguments.output.resolve()
    if arguments.check:
        if not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("synthetic mutation evidence drifted")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print("SYNTHETIC_COINSTALLATION_MATRIX=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
