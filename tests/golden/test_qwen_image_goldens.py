"""Independent golden vectors for the two Qwen Image shift modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from comfyui_sigmax.profiles.qwen_image import QwenImageShiftMode, build_qwen_image_schedule


@pytest.mark.parametrize(
    "case", json.loads((Path(__file__).parent / "qwen_image_v1.json").read_text()).get("cases", [])
)
def test_qwen_image_float64_golden(case: dict[str, object]) -> None:
    mode = QwenImageShiftMode(str(case["mode"]))
    seq = case["image_seq_len"]
    result = build_qwen_image_schedule(
        mode=mode,
        steps=cast(int, case["steps"]),
        image_seq_len=None if seq is None else cast(int, seq),
        strict_official=False,
    )
    assert result.sigmas == pytest.approx(
        tuple(cast(float, value) for value in cast(list[object], case["float64"])), abs=1e-15
    )
