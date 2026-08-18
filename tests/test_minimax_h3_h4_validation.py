"""Pure guards for the private MiniMax H3 H4 validation lane."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import scripts.run_minimax_h3_h4_validation as h4
from comfyui_sigmax.core import ScheduleContractError
from scripts.h4_dispatch_observer import _TraceState
from scripts.run_minimax_h3_h4_validation import (
    H4_SCHEMA,
    PROTOCOL_STATUS,
    RowDisposition,
    _artifact_observation,
    _cleanup_projection,
    _free_port,
    _gpu_memory_projection,
    _GpuMemorySampler,
    _host_readback_projection,
    _parse_gpu_memory_output,
    _port_release_receipt,
    _protocol_binding,
    _read_dispatch_trace,
    _temp_root_receipt,
    _validate_h4_schema,
    build_h4_prompt,
    classify_turbo_artifact,
    current_candidate,
)


def _tiny_safetensors(path: Path) -> None:
    header = {
        "__metadata__": {"format": "fixture"},
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0" * 4)


def test_h4_prompt_has_one_sigmax_schedule_and_no_second_shift_node() -> None:
    prompt = build_h4_prompt(
        variant="H3 Base FL2VA",
        model_name="H3/model.safetensors",
        clip_name="clip.safetensors",
        video_vae_name="video.safetensors",
        audio_vae_name="audio.safetensors",
        prompt="A quiet studio.",
        width=608,
        height=352,
        length=17,
        steps=20,
        seed=1,
        shift_video=12.0,
        shift_audio=3.0,
    )
    assert [node["class_type"] for node in prompt.values()].count(
        "Sigmax.MiniMaxH3SigmaScheduler"
    ) == 1
    assert "MiniMaxH3SigmaShift" not in {node["class_type"] for node in prompt.values()}
    node = prompt["11"]
    inputs = cast(dict[str, object], node["inputs"])
    assert inputs["sigmas"] == ["7", 0]


def test_h4_trace_prompt_observes_model_clone_and_finalizes_after_save(tmp_path: Path) -> None:
    trace_file = str(tmp_path / "dispatch.json")
    prompt = build_h4_prompt(
        variant="H3 Base FL2VA",
        model_name="H3/model.safetensors",
        clip_name="clip.safetensors",
        video_vae_name="video.safetensors",
        audio_vae_name="audio.safetensors",
        prompt="A quiet studio.",
        width=608,
        height=352,
        length=17,
        steps=20,
        seed=1,
        shift_video=12.0,
        shift_audio=3.0,
        trace_file=trace_file,
        requested_attention_backend="ck_int8",
        requested_operation_backend="auto",
    )
    assert prompt["16"]["class_type"] == "Sigmax.H4DispatchObserver"
    assert prompt["17"]["class_type"] == "Sigmax.H4DispatchFinalize"
    guider_inputs = cast(dict[str, object], prompt["9"]["inputs"])
    assert guider_inputs["model"] == ["16", 0]
    finalizer_inputs = cast(dict[str, object], prompt["17"]["inputs"])
    assert finalizer_inputs["video"] == ["15", 0]
    assert finalizer_inputs["trace_file"] == trace_file
    assert "MiniMaxH3SigmaShift" not in {node["class_type"] for node in prompt.values()}


def test_h4_prompt_rejects_private_prompt_and_bad_frame_grid() -> None:
    with pytest.raises(ScheduleContractError):
        build_h4_prompt(
            variant="H3 Base FL2VA",
            model_name="H3/model.safetensors",
            clip_name="clip.safetensors",
            video_vae_name="video.safetensors",
            audio_vae_name="audio.safetensors",
            prompt=r"C:\private\prompt.txt",
            width=608,
            height=352,
            length=17,
            steps=20,
            seed=1,
            shift_video=12.0,
            shift_audio=3.0,
        )
    with pytest.raises(ScheduleContractError):
        build_h4_prompt(
            variant="H3 Base FL2VA",
            model_name="H3/model.safetensors",
            clip_name="clip.safetensors",
            video_vae_name="video.safetensors",
            audio_vae_name="audio.safetensors",
            prompt="A quiet studio.",
            width=608,
            height=352,
            length=18,
            steps=20,
            seed=1,
            shift_video=12.0,
            shift_audio=3.0,
        )


def test_blocked_reduced_artifact_never_becomes_eligible(tmp_path: Path) -> None:
    path = tmp_path / "reduced.safetensors"
    _tiny_safetensors(path)
    # The fixture's digest is not a known reduced hash, but the explicit source disposition still
    # keeps it blocked; no filename or successful header can promote it.
    result = classify_turbo_artifact(
        path=path,
        artifact_id="h3.fl2va.lightx2v-turbo-8-v1.0-544p",
        source="kijai-reduced",
        license_ack=True,
    )
    assert result.disposition is RowDisposition.BLOCKED
    assert result.reason_code == "artifact.provenance_or_license_blocked"


def test_publisher_hash_or_size_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "publisher.safetensors"
    _tiny_safetensors(path)
    result = _artifact_observation(
        path=path,
        artifact_id="publisher",
        disposition=RowDisposition.ACCEPTED,
        expected_sha256="0" * 64,
    )
    assert result.disposition is RowDisposition.REJECTED
    assert result.reason_code == "artifact.hash_or_size_mismatch"


def test_dispatch_trace_projection_requires_disarmed_redacted_observation(tmp_path: Path) -> None:
    trace_file = tmp_path / "dispatch.json"
    state = _TraceState(
        trace_file=trace_file,
        requested_attention_backend="pytorch",
        requested_operation_backend="auto",
        status="DISARMED",
    )
    state.record_attention("pytorch", "returned")
    observed = _read_dispatch_trace(trace_file, expected_attention_backend="pytorch")
    assert observed["status"] == "valid"
    assert observed["actual_attention_backend"] == "pytorch"
    trace_file.write_text(
        json.dumps({"schema": "sigmax.h4-dispatch-observation/1", "path": r"C:\private"}),
        encoding="utf-8",
    )
    rejected = _read_dispatch_trace(trace_file, expected_attention_backend="pytorch")
    assert rejected["status"] == "unavailable"
    assert rejected["reason_code"] == "dispatch_trace_redaction_failed"


def test_h4_protocol_binding_accepts_review_state_and_rejects_stale_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit, tree = current_candidate()
    monkeypatch.setattr(h4, "REPOSITORY_ROOT", tmp_path)

    def write_protocol(status: str) -> Path:
        target = tmp_path / f"protocol-{status}.md"
        target.write_text(
            f"**Protocol status:** `{status}`\ncommit: {commit}\ntree: {tree}\n",
            encoding="utf-8",
        )
        return target

    _protocol_binding(write_protocol(PROTOCOL_STATUS), expected_commit=commit, expected_tree=tree)
    with pytest.raises(ScheduleContractError):
        _protocol_binding(
            write_protocol("ACTIVE_PENDING_PREFLIGHT"), expected_commit=commit, expected_tree=tree
        )
    assert H4_SCHEMA.endswith("/2")
    with pytest.raises(ScheduleContractError):
        _validate_h4_schema("sigmax.minimax-h3-h4-private-validation/1")


def test_gpu_memory_projection_is_strict_per_device_and_never_promotes_zero() -> None:
    parsed = _parse_gpu_memory_output("0, 128\n1, 256\n")
    assert parsed == {0: 128 * 1024 * 1024, 1: 256 * 1024 * 1024}
    projected = _gpu_memory_projection(
        [parsed, {0: 192 * 1024 * 1024, 1: 224 * 1024 * 1024}],
        sample_interval_ms=250,
    )
    assert projected["status"] == "pass"
    assert projected["peak_used_bytes"] == 416 * 1024 * 1024
    assert projected["peak_used_bytes_by_device"] == {
        "0": 192 * 1024 * 1024,
        "1": 256 * 1024 * 1024,
    }
    with pytest.raises(ValueError):
        _parse_gpu_memory_output("0, 128 MiB\n")
    assert _gpu_memory_projection([{0: 0}], sample_interval_ms=250)["status"] == "failed"


def test_host_readback_projection_detects_revision_tree_and_artifact_drift() -> None:
    before = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "worktree_state": "sha256:clean",
        "artifacts": {"model": "sha256:one"},
    }
    after = {
        "revision": "a" * 40,
        "tree": "c" * 40,
        "worktree_state": "sha256:dirty",
        "artifacts": {"model": "sha256:two"},
    }
    result = _host_readback_projection(
        before,
        after,
        process_alive=False,
        api_unreachable=True,
    )
    assert result["status"] == "fail"
    assert result["worktree_state_unchanged"] is False
    mutation = cast(Mapping[str, object], result["host_mutation"])
    assert mutation["checkout_unchanged"] is False
    assert mutation["selected_artifacts_unchanged"] is False


def test_gpu_sampler_retains_unavailable_reason_without_fabricating_peak() -> None:
    sampler = _GpuMemorySampler(snapshot=lambda: (None, "gpu_memory_timeout"), interval_seconds=5.0)
    sampler.start()
    result = sampler.stop()
    assert result["status"] == "unavailable"
    assert result["peak_used_bytes"] is None
    assert result["reason_code"] == "gpu_memory_timeout"


def test_cleanup_projection_requires_graceful_exit_port_readback_and_owned_root() -> None:
    shutdown = {
        "process_exited": True,
        "return_code": 0,
        "termination": "graceful",
    }
    port = {
        "status": "pass",
        "verified_by": "bind_probe",
    }
    temp = {"cleanup_status": "removed", "owned": True}
    readback = {"status": "pass"}
    assert _cleanup_projection(shutdown, port, temp, readback)["status"] == "pass"
    forced = dict(shutdown, return_code=1, termination="forced")
    assert _cleanup_projection(forced, port, temp, readback)["status"] == "fail"


def test_port_receipt_and_temp_root_are_bounded_and_owned(tmp_path: Path) -> None:
    port = _port_release_receipt(0, timeout=1)
    assert port["status"] == "unavailable"
    assert _port_release_receipt(_free_port(), timeout=1)["status"] == "pass"
    owned = tmp_path / "h4-owned"
    owned.mkdir()
    (owned / "marker").write_text("private", encoding="utf-8")
    receipt = _temp_root_receipt(owned, owned_root=tmp_path, remove=False)
    assert receipt["owned"] is False
