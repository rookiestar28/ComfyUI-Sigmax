"""Pure guards for the private MiniMax H3 H4 validation lane."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from argparse import Namespace
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
    _stage_reference_image,
    _temp_root_receipt,
    _turbo_row_spec,
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


def test_turbo_prompt_binds_exact_recipe_lora_and_scheduler() -> None:
    recipe_id = "h3.fl2va.lightx2v-turbo-8-v1.0-544p"
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
        steps=8,
        seed=1,
        shift_video=12.0,
        shift_audio=3.0,
        recipe_id=recipe_id,
        lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    )
    sample_node = prompt["11"]
    schedule_node = prompt["7"]
    lora_node = prompt["5"]
    sample_inputs = cast(dict[str, object], sample_node["inputs"])
    schedule_inputs = cast(dict[str, object], schedule_node["inputs"])
    lora_inputs = cast(dict[str, object], lora_node["inputs"])
    assert sample_inputs["sigmas"] == ["7", 0]
    assert schedule_inputs["recipe_id"] == recipe_id
    assert schedule_inputs["steps"] == 8
    assert schedule_inputs["start_step"] == 0
    assert schedule_inputs["end_step"] == -1
    assert prompt["5"]["class_type"] == "LoraLoaderModelOnly"
    assert lora_inputs["strength_model"] == 1.0


def test_ref2v_turbo_prompt_binds_reference_image_and_task_node() -> None:
    recipe_id = "h3.ref2va.lightx2v-turbo-4-v0.1-544p"
    prompt = build_h4_prompt(
        variant="H3 Base Ref2VA",
        model_name="H3/model.safetensors",
        clip_name="clip.safetensors",
        video_vae_name="video.safetensors",
        audio_vae_name="audio.safetensors",
        prompt="<Picture 1> in a quiet studio.",
        width=608,
        height=352,
        length=17,
        steps=4,
        seed=1,
        shift_video=12.0,
        shift_audio=3.0,
        recipe_id=recipe_id,
        lora_name="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        reference_image_name="m7_13_reference.png",
    )
    reference_node = prompt["6"]
    ref_schedule_node = prompt["7"]
    reference_inputs = cast(dict[str, object], reference_node["inputs"])
    ref_schedule_inputs = cast(dict[str, object], ref_schedule_node["inputs"])
    assert prompt["6"]["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert reference_inputs["audio_vae"] == ["4", 0]
    assert reference_inputs["ref_images.ref_image_0"] == ["18", 0]
    assert prompt["18"] == {"class_type": "LoadImage", "inputs": {"image": "m7_13_reference.png"}}
    assert ref_schedule_inputs["recipe_id"] == recipe_id


def test_turbo_publisher_allowlist_does_not_alias_unsupported_544p_v01() -> None:
    assert "h3.fl2va.lightx2v-turbo-4-v0.1-544p" not in h4._PUBLISHER_ARTIFACTS
    assert (
        h4._PUBLISHER_ARTIFACTS["h3.fl2va.lightx2v-turbo-8-v1.0-544p"][1]
        == (
            "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e"  # pragma: allowlist secret
        )
    )


def test_turbo_row_artifact_identity_cannot_be_reused_across_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "publisher.safetensors"
    _tiny_safetensors(artifact)
    result = h4._row_artifact(
        row="T8-544",
        models_root=tmp_path,
        turbo_artifact=artifact.name,
        turbo_artifact_id="h3.fl2va.lightx2v-turbo-4-v1.0-768p",
        turbo_source="publisher-full",
        license_ack=True,
    )
    assert result is not None
    assert result.disposition is RowDisposition.REJECTED
    assert result.reason_code == "artifact.recipe_row_mismatch"


def test_multiple_turbo_rows_are_rejected_before_queueing(tmp_path: Path) -> None:
    artifact = tmp_path / "publisher.safetensors"
    _tiny_safetensors(artifact)
    result = h4._row_artifact(
        row="T8-544",
        models_root=tmp_path,
        turbo_artifact=artifact.name,
        turbo_artifact_id="h3.fl2va.lightx2v-turbo-8-v1.0-544p",
        turbo_source="publisher-full",
        license_ack=True,
        turbo_rows=("T8-544", "T4-768"),
    )
    assert result is not None
    assert result.disposition is RowDisposition.REJECTED
    assert result.reason_code == "artifact.multi_row_ambiguity"


def test_preflight_retains_multi_row_negative_receipts(tmp_path: Path) -> None:
    args = Namespace(
        rows=["T8-544", "T4-768"],
        turbo_artifact=None,
        turbo_artifact_id=None,
        turbo_source="publisher-full",
        license_ack=True,
        reference_image=None,
        reference_image_root=None,
    )
    rows = h4._preflight_rows(args, tmp_path)
    t8 = cast(dict[str, object], rows["T8-544"])
    t4 = cast(dict[str, object], rows["T4-768"])
    assert t8["disposition"] == RowDisposition.REJECTED.value
    assert t8["reason_code"] == "artifact.multi_row_ambiguity"
    assert t4["reason_code"] == "artifact.multi_row_ambiguity"


def test_queue_gate_seam_counts_only_accepted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted = h4.ArtifactObservation(
        artifact_id="accepted",
        disposition=RowDisposition.ACCEPTED,
        reason_code=None,
        file_bytes=1,
        sha256="a" * 64,
        header_bytes=1,
        tensor_count=1,
        dtype_counts=(("F32", 1),),
    )
    blocked = h4.ArtifactObservation(
        artifact_id="blocked",
        disposition=RowDisposition.BLOCKED,
        reason_code="artifact.publisher_full_not_available",
        file_bytes=None,
        sha256=None,
        header_bytes=None,
        tensor_count=None,
        dtype_counts=(),
    )

    def fake_row_artifact(*, row: str, **_kwargs: object) -> h4.ArtifactObservation:
        return accepted if row == "T8-544" else blocked

    monkeypatch.setattr(h4, "_row_artifact", fake_row_artifact)
    accepted_args = Namespace(
        rows=["T8-544", "T4-544"],
        turbo_artifact="publisher.safetensors",
        turbo_artifact_id="h3.fl2va.lightx2v-turbo-8-v1.0-544p",
        turbo_source="publisher-full",
        license_ack=True,
    )
    assert h4._queueable_rows(accepted_args, tmp_path) == ("T8-544",)
    blocked_args = Namespace(
        rows=["T4-544"],
        turbo_artifact=None,
        turbo_artifact_id=None,
        turbo_source="publisher-full",
        license_ack=True,
    )
    assert h4._queueable_rows(blocked_args, tmp_path) == ()
    ref_args = Namespace(
        rows=["R4-544"],
        turbo_artifact="publisher.safetensors",
        turbo_artifact_id="h3.ref2va.lightx2v-turbo-4-v0.1-544p",
        turbo_source="publisher-full",
        license_ack=True,
        reference_image=None,
        reference_image_root=None,
    )
    monkeypatch.setattr(h4, "_row_artifact", lambda **_kwargs: accepted)
    assert h4._queueable_rows(ref_args, tmp_path) == ()
    submits: list[str] = []

    def fake_submit() -> tuple[str, dict[str, object], int, dict[str, object]]:
        submits.append("queued")
        return ("prompt-1", {}, 1, {})

    assert h4._submit_if_eligible(accepted, fake_submit) == ("prompt-1", {}, 1, {})
    assert h4._submit_if_eligible(blocked, fake_submit) is None
    assert submits == ["queued"]


def test_turbo_row_specs_bind_recipe_task_and_resolution() -> None:
    t4 = _turbo_row_spec("T4-768")
    assert (t4.recipe_id, t4.variant, t4.steps, t4.shift_video) == (
        "h3.fl2va.lightx2v-turbo-4-v1.0-768p",
        "H3 Base FL2VA",
        4,
        6.0,
    )
    assert (t4.width, t4.height, t4.requires_reference_image) == (1344, 768, False)
    ref = _turbo_row_spec("R4-544")
    assert (ref.variant, ref.width, ref.height, ref.requires_reference_image) == (
        "H3 Base Ref2VA",
        960,
        544,
        True,
    )


def test_reference_image_staging_is_private_and_bounded(tmp_path: Path) -> None:
    source = tmp_path / "caller-reference.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"PNG fixture")
    run_path = tmp_path / "run"
    (run_path / "input").mkdir(parents=True)
    name = _stage_reference_image(source, run_path, owner_root=tmp_path)
    assert name == "m7_13_reference.png"
    assert (run_path / "input" / name).read_bytes() == source.read_bytes()
    with pytest.raises(ScheduleContractError, match="unavailable"):
        _stage_reference_image(tmp_path / "missing.png", run_path, owner_root=tmp_path)
    outside = tmp_path.parent / "outside-reference.png"
    outside.write_bytes(source.read_bytes())
    with pytest.raises(ScheduleContractError, match="outside"):
        _stage_reference_image(outside, run_path, owner_root=tmp_path)
    outside.unlink()


def test_reference_image_staging_rejects_missing_owner_root_and_oversize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "caller-reference.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"PNG fixture")
    run_path = tmp_path / "run"
    (run_path / "input").mkdir(parents=True)
    with pytest.raises(ScheduleContractError, match="owner root"):
        _stage_reference_image(source, run_path)
    monkeypatch.setattr(h4, "_MAX_REFERENCE_IMAGE_BYTES", 8)
    with pytest.raises(ScheduleContractError, match="size bound"):
        _stage_reference_image(source, run_path, owner_root=tmp_path)


def test_reference_image_preflight_is_path_free_and_blocks_missing_root(tmp_path: Path) -> None:
    source = tmp_path / "caller-reference.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n" + b"PNG fixture")
    missing_root = Namespace(reference_image=source, reference_image_root=None)
    assert h4._reference_image_preflight_reason(missing_root) == (
        "input.reference_image_root_not_supplied"
    )
    outside = tmp_path.parent / "outside-reference.png"
    outside.write_bytes(source.read_bytes())
    bounded = Namespace(reference_image=outside, reference_image_root=tmp_path)
    assert h4._reference_image_preflight_reason(bounded) == (
        "input.reference_image_outside_owner_root"
    )
    outside.unlink()
    linked = tmp_path / "linked-reference.png"
    try:
        linked.symlink_to(source)
    except (OSError, NotImplementedError):
        pass
    else:
        assert (
            h4._reference_image_preflight_reason(
                Namespace(reference_image=linked, reference_image_root=tmp_path)
            )
            == "input.reference_image_reparse_point"
        )
        linked.unlink()


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
    cooperative_nonzero = dict(shutdown, return_code=2, termination_method="cooperative_ctrl_break")
    projection = _cleanup_projection(cooperative_nonzero, port, temp, readback)
    assert projection["status"] == "fail"
    assert projection["reason_code"] == "nonzero_cooperative_return"
    assert projection["termination_method"] == "cooperative_ctrl_break"


def test_windows_host_command_uses_fixed_sigbreak_bootstrap_and_preserves_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    host_root = tmp_path / "reviewed-host"
    host_python = tmp_path / "host-python.exe"
    models_root = tmp_path / "models"
    run_path = tmp_path / "run"
    monkeypatch.setattr(os, "name", "nt")

    command = h4._host_command(
        host_python=host_python,
        comfyui_root=host_root,
        models_root=models_root,
        run_path=run_path,
        port=12345,
        use_ck_attention=True,
        enable_triton=True,
    )

    assert command[:3] == [str(host_python), "-c", h4._WINDOWS_HOST_BOOTSTRAP]
    assert command[3] == str(host_root / "main.py")
    assert str(host_root) not in h4._WINDOWS_HOST_BOOTSTRAP
    assert "signal.signal(signal.SIGBREAK, signal.default_int_handler)" in command[2]
    assert command[4:8] == ["--listen", "127.0.0.1", "--port", "12345"]
    assert command[-2:] == ["--use-ck-attention", "--enable-triton-backend"]


def test_host_process_group_options_are_platform_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_windows_flag = 512
    monkeypatch.setattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", expected_windows_flag, raising=False
    )
    monkeypatch.setattr(os, "name", "nt")
    windows_flags, windows_session = h4._host_process_group_options()
    assert windows_flags == expected_windows_flag
    assert windows_session is False

    monkeypatch.setattr(os, "name", "posix")
    posix_flags, posix_session = h4._host_process_group_options()
    assert posix_flags == 0
    assert posix_session is True


def test_posix_host_command_keeps_direct_main_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    host_root = tmp_path / "reviewed-host"
    host_python = tmp_path / "python"
    monkeypatch.setattr(os, "name", "posix")

    command = h4._host_command(
        host_python=host_python,
        comfyui_root=host_root,
        models_root=tmp_path / "models",
        run_path=tmp_path / "run",
        port=12345,
        use_ck_attention=False,
        enable_triton=False,
    )

    assert command[:2] == [str(host_python), str(host_root / "main.py")]
    assert "-c" not in command[:3]


def test_terminate_uses_cooperative_windows_ctrl_break_after_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminate_called = False
            self.sent_signals: list[int] = []

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def terminate(self) -> None:
            self.terminate_called = True
            self.returncode = 1

        def send_signal(self, signum: int) -> None:
            self.sent_signals.append(signum)
            self.returncode = 0

    process = FakeProcess()
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(h4, "_http_no_content", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        os,
        "kill",
        lambda *_args, **_kwargs: pytest.fail("Windows termination must not call os.kill"),
    )
    result = h4._terminate(
        cast(subprocess.Popen[bytes], process), base_url="http://127.0.0.1:12345"
    )

    assert result["interrupt_requested"] is True
    assert result["termination"] == "graceful"
    assert result["termination_method"] == "cooperative_ctrl_break"
    assert result["return_code"] == 0
    assert process.sent_signals == [signal.CTRL_BREAK_EVENT]
    assert process.terminate_called is False


def test_terminate_fails_closed_when_windows_ctrl_break_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        pid = 1234
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.delattr(signal, "CTRL_BREAK_EVENT")
    monkeypatch.setattr(h4, "_http_no_content", lambda *args, **kwargs: None)

    with pytest.raises(ScheduleContractError, match=r"CTRL\+BREAK"):
        h4._terminate(
            cast(subprocess.Popen[bytes], FakeProcess()),
            base_url="http://127.0.0.1:12345",
        )


def test_interrupt_request_accepts_no_content_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoContentResponse:
        def __enter__(self) -> NoContentResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, limit: int) -> bytes:
            assert limit > 0
            return b""

    monkeypatch.setattr(h4, "urlopen", lambda *args, **kwargs: NoContentResponse())
    h4._http_no_content("http://127.0.0.1:12345/interrupt", method="POST", timeout=1)


def test_port_receipt_and_temp_root_are_bounded_and_owned(tmp_path: Path) -> None:
    port = _port_release_receipt(0, timeout=1)
    assert port["status"] == "unavailable"
    assert _port_release_receipt(_free_port(), timeout=1)["status"] == "pass"
    owned = tmp_path / "h4-owned"
    owned.mkdir()
    (owned / "marker").write_text("private", encoding="utf-8")
    receipt = _temp_root_receipt(owned, owned_root=tmp_path, remove=False)
    expected_owned = h4.REPOSITORY_ROOT.resolve() in tmp_path.resolve().parents
    assert receipt["owned"] is expected_owned
    assert receipt["cleanup_status"] == ("retained" if expected_owned else "failed")
