from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.host_mutation import (
    CoInstallationEvaluation,
    HostMutationFinding,
    HostMutationSnapshot,
    MutationVerdict,
    NodeRegistryIdentity,
    SchedulerRegistryIdentity,
    evaluate_host_mutation,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64


def _snapshot() -> HostMutationSnapshot:
    return HostMutationSnapshot(
        nodes=(
            NodeRegistryIdentity(
                node_id="Sigmax.Krea2SigmaScheduler",
                provider="comfyui_sigmax",
                definition_fingerprint=HASH_A,
            ),
        ),
        schedulers=(
            SchedulerRegistryIdentity(
                name="simple",
                provider="comfyui",
                handler_fingerprint=HASH_B,
            ),
        ),
        torch_call_fingerprint=HASH_C,
        model_patch_fingerprint=HASH_D,
        schedule_ownership="external_sigmas",
        construction_shift_count=1,
        model_native_shifted=False,
    )


def _evaluate(after: HostMutationSnapshot) -> CoInstallationEvaluation:
    return evaluate_host_mutation(
        before=_snapshot(),
        after=after,
        pack_id="synthetic.fixture",
    )


def test_unrelated_node_and_scheduler_additions_are_compatible() -> None:
    before = _snapshot()
    after = replace(
        before,
        nodes=(
            NodeRegistryIdentity(
                node_id="Other.ExampleNode",
                provider="synthetic.other",
                definition_fingerprint=HASH_B,
            ),
            *before.nodes,
        ),
        schedulers=(
            SchedulerRegistryIdentity(
                name="other_scheduler",
                provider="synthetic.other",
                handler_fingerprint=HASH_A,
            ),
            *before.schedulers,
        ),
    )

    report = _evaluate(after)

    assert report.verdict is MutationVerdict.ALLOW
    assert report.findings == ()
    assert report.report_fingerprint.startswith("sha256:")


@pytest.mark.parametrize(
    ("after", "finding"),
    [
        (
            replace(
                _snapshot(),
                nodes=(
                    NodeRegistryIdentity(
                        node_id="Sigmax.Krea2SigmaScheduler",
                        provider="synthetic.other",
                        definition_fingerprint=HASH_B,
                    ),
                ),
            ),
            HostMutationFinding.NODE_REGISTRY_COLLISION,
        ),
        (
            replace(
                _snapshot(),
                nodes=(
                    NodeRegistryIdentity(
                        node_id="Sigmax.ForeignNode",
                        provider="synthetic.other",
                        definition_fingerprint=HASH_B,
                    ),
                    *_snapshot().nodes,
                ),
            ),
            HostMutationFinding.SIGMAX_NAMESPACE_HIJACK,
        ),
        (
            replace(
                _snapshot(),
                schedulers=(
                    SchedulerRegistryIdentity(
                        name="simple",
                        provider="synthetic.other",
                        handler_fingerprint=HASH_A,
                    ),
                ),
            ),
            HostMutationFinding.SCHEDULER_REGISTRY_OVERWRITE,
        ),
        (
            replace(_snapshot(), torch_call_fingerprint=HASH_A),
            HostMutationFinding.TORCH_CALL_PATH_CHANGED,
        ),
        (
            replace(_snapshot(), model_patch_fingerprint=HASH_A),
            HostMutationFinding.MODEL_PATCH_STATE_CHANGED,
        ),
        (
            replace(_snapshot(), construction_shift_count=2),
            HostMutationFinding.CONSTRUCTION_SHIFT_REPEATED,
        ),
        (
            replace(
                _snapshot(),
                schedule_ownership="model_native",
                model_native_shifted=True,
            ),
            HostMutationFinding.MODEL_NATIVE_EXTERNAL_DOUBLE_SHIFT,
        ),
    ],
)
def test_protected_host_mutations_are_detected_and_rejected(
    after: HostMutationSnapshot,
    finding: HostMutationFinding,
) -> None:
    report = _evaluate(after)

    assert report.verdict is MutationVerdict.REJECT
    assert finding in report.findings


def test_evaluation_is_deterministic_and_snapshot_is_immutable() -> None:
    before = _snapshot()
    first = evaluate_host_mutation(before=before, after=before, pack_id="synthetic.reload")
    repeat = evaluate_host_mutation(before=before, after=before, pack_id="synthetic.reload")

    assert first == repeat
    assert first.report_fingerprint == repeat.report_fingerprint
    with pytest.raises(FrozenInstanceError):
        before.construction_shift_count = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda snapshot: replace(
            snapshot,
            nodes=(
                *snapshot.nodes,
                NodeRegistryIdentity(
                    node_id="Other.Unsorted",
                    provider="synthetic.other",
                    definition_fingerprint=HASH_A,
                ),
            ),
        ),
        lambda snapshot: replace(snapshot, torch_call_fingerprint="not-a-fingerprint"),
        lambda snapshot: replace(snapshot, construction_shift_count=-1),
        lambda snapshot: replace(snapshot, schedule_ownership="implicit"),
    ],
)
def test_snapshot_rejects_noncanonical_or_invalid_state(
    mutation: Callable[[HostMutationSnapshot], HostMutationSnapshot],
) -> None:
    with pytest.raises(ScheduleContractError):
        mutation(_snapshot())
