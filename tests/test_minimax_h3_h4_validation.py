"""Pure guards for the private MiniMax H3 H4 validation lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from scripts.h4_dispatch_observer import _TraceState
from scripts.run_minimax_h3_h4_validation import (
    RowDisposition,
    _artifact_observation,
    _read_dispatch_trace,
    build_h4_prompt,
    classify_turbo_artifact,
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
