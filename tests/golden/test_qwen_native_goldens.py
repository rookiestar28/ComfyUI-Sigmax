"""Immutable independent vectors and legacy-profile fingerprint preservation."""

from __future__ import annotations

import hashlib
import json
import struct
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from comfyui_sigmax.core import SigmaDomain, numerical_fingerprint
from comfyui_sigmax.profiles.qwen_image import (
    QWEN_IMAGE_COMFY_FIXED_SCHEMA,
    QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA,
)
from comfyui_sigmax.profiles.qwen_native import (
    QWEN_NATIVE_PROFILES,
    QwenNativeLane,
    QwenNativeRequest,
    build_qwen_native_schedule,
)
from comfyui_sigmax.profiles.schema_v1 import profile_schema_fingerprint

FIXTURE = json.loads((Path(__file__).parent / "qwen_native_v1.json").read_text())


def test_legacy_profiles_remain_frozen_and_new_lanes_have_distinct_identities() -> None:
    assert FIXTURE["legacy_profile_fingerprints"] == [
        profile_schema_fingerprint(QWEN_IMAGE_COMFY_FIXED_SCHEMA),
        profile_schema_fingerprint(QWEN_IMAGE_DIFFUSERS_DYNAMIC_SCHEMA),
    ]
    fingerprints = [profile_schema_fingerprint(profile.schema) for profile in QWEN_NATIVE_PROFILES]
    assert len(set(fingerprints)) == 3
    assert not set(fingerprints).intersection(FIXTURE["legacy_profile_fingerprints"])


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_complete_golden_integrity_and_deterministic_fingerprints(case: dict[str, Any]) -> None:
    for dtype, packing in (("float64", ">d"), ("float32", ">f")):
        payload = b"".join(struct.pack(packing, value) for value in case[dtype])
        assert hashlib.sha256(payload).hexdigest() == case[f"raw_{dtype}_sha256"]
    request = QwenNativeRequest(
        lane=QwenNativeLane(case["lane"]), steps=case["steps"], image_seq_len=case["image_seq_len"]
    )
    first = build_qwen_native_schedule(request)
    repeat = build_qwen_native_schedule(request)
    values = first.schedule.sigmas
    assert values == repeat.schedule.sigmas
    assert len(values) == case["steps"] + 1
    assert values[-1] == 0.0
    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(left > right for left, right in pairwise(values))
    assert numerical_fingerprint(
        values, domain=SigmaDomain.UNIT_FLOW, precision="float64"
    ) == numerical_fingerprint(
        repeat.schedule.sigmas, domain=SigmaDomain.UNIT_FLOW, precision="float64"
    )
