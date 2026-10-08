"""Source-pinned arithmetic parity for Qwen native/2.1, independent of host imports."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from comfyui_sigmax.profiles.qwen_native import (
    QwenNativeLane,
    QwenNativeRequest,
    build_qwen_native_schedule,
)
from tests.parity.qwen_native_oracle import qwen_source_vector

pytestmark = pytest.mark.parity
FIXTURE = Path(__file__).parents[1] / "golden" / "qwen_native_v1.json"


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"])
def test_source_vector_float64_and_float32(case: dict[str, Any]) -> None:
    lane = QwenNativeLane(case["lane"])
    result = build_qwen_native_schedule(
        QwenNativeRequest(lane=lane, steps=case["steps"], image_seq_len=case["image_seq_len"])
    )
    oracle = qwen_source_vector(case["lane"], case["steps"], case["image_seq_len"])
    assert tuple(case["float64"]) == oracle
    assert result.schedule.sigmas == pytest.approx(oracle, rel=0.0, abs=1e-15)
    actual32 = [
        struct.unpack(">f", struct.pack(">f", value))[0] for value in result.schedule.sigmas
    ]
    assert actual32 == pytest.approx(case["float32"], rel=0.0, abs=1.2e-7)


@pytest.mark.parametrize("steps", [1, 2, 25, 27, 257, 1000, 5000, 10000])
@pytest.mark.parametrize("lane", [QwenNativeLane.ORIGINAL_COMFY, QwenNativeLane.QWEN21_COMFY])
def test_native_table_integer_selection_at_boundaries(steps: int, lane: QwenNativeLane) -> None:
    result = build_qwen_native_schedule(QwenNativeRequest(lane=lane, steps=steps))
    assert result.schedule.sigmas == pytest.approx(
        qwen_source_vector(lane.value, steps, None), rel=0.0, abs=1e-15
    )
