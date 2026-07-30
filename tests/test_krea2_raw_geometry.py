"""Contracts for Krea 2 RAW effective geometry and dynamic-mu derivation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest
from comfyui_sigmax.core import ScheduleContractError
from comfyui_sigmax.profiles import (
    KREA2_RAW_PROFILE,
    DimensionPolicy,
    Krea2ImageGeometry,
    Krea2RawProfile,
    Krea2RawShiftDerivation,
    ResolutionShiftPolicy,
    calculate_krea2_raw_mu,
    derive_krea2_raw_shift,
    resolve_krea2_image_geometry,
)


@pytest.mark.parametrize(
    ("width", "height", "effective_width", "effective_height", "grid_width", "grid_height"),
    (
        (256, 256, 256, 256, 16, 16),
        (1024, 1024, 1024, 1024, 64, 64),
        (1360, 768, 1360, 768, 85, 48),
        (768, 1360, 768, 1360, 48, 85),
        (1025, 767, 1040, 768, 65, 48),
        (1, 17, 16, 32, 1, 2),
    ),
)
def test_geometry_retains_requested_and_effective_ceil_to_16_dimensions(
    width: int,
    height: int,
    effective_width: int,
    effective_height: int,
    grid_width: int,
    grid_height: int,
) -> None:
    geometry = resolve_krea2_image_geometry(
        width,
        height,
        policy=KREA2_RAW_PROFILE.dimensions,
    )

    assert geometry == Krea2ImageGeometry(
        requested_width=width,
        requested_height=height,
        effective_width=effective_width,
        effective_height=effective_height,
        alignment_multiple=16,
        grid_width=grid_width,
        grid_height=grid_height,
        image_seq_len=grid_width * grid_height,
    )


@pytest.mark.parametrize(
    ("width", "height", "image_seq_len", "expected_mu", "extrapolated"),
    (
        (256, 256, 256, 0.5, False),
        (512, 512, 1024, 0.58125, False),
        (768, 768, 2304, 0.7166666666666667, False),
        (1024, 1024, 4096, 0.90625, False),
        (1280, 1280, 6400, 1.15, False),
        (1360, 768, 4080, 0.9045572916666667, False),
        (768, 1360, 4080, 0.9045572916666667, False),
        (1025, 767, 3120, 0.8029947916666667, False),
        (16, 16, 1, 0.4730224609375, True),
        (2048, 2048, 16384, 2.20625, True),
    ),
)
def test_raw_shift_derivation_matches_official_unclamped_geometry(
    width: int,
    height: int,
    image_seq_len: int,
    expected_mu: float,
    extrapolated: bool,
) -> None:
    result = derive_krea2_raw_shift(width, height)

    assert result.profile_id == "krea2.raw.official"
    assert result.profile_version == "1"
    assert result.geometry.image_seq_len == image_seq_len
    assert result.mu == pytest.approx(expected_mu, abs=1e-15)
    assert result.extrapolated is extrapolated
    assert calculate_krea2_raw_mu(image_seq_len) == pytest.approx(expected_mu, abs=1e-15)


def test_orientation_changes_grid_axes_but_not_sequence_length_or_mu() -> None:
    landscape = derive_krea2_raw_shift(1360, 768)
    portrait = derive_krea2_raw_shift(768, 1360)

    assert (landscape.geometry.grid_width, landscape.geometry.grid_height) == (85, 48)
    assert (portrait.geometry.grid_width, portrait.geometry.grid_height) == (48, 85)
    assert landscape.geometry.image_seq_len == portrait.geometry.image_seq_len
    assert landscape.mu == portrait.mu


def test_raw_shift_result_is_deeply_immutable_and_self_consistent() -> None:
    result = derive_krea2_raw_shift(1025, 767)

    with pytest.raises(FrozenInstanceError):
        result.mu = 1.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.geometry.effective_width = 1024  # type: ignore[misc]
    with pytest.raises(ScheduleContractError):
        replace(result.geometry, effective_width=1024)
    with pytest.raises(ScheduleContractError):
        replace(result.geometry, grid_width=64)
    with pytest.raises(ScheduleContractError):
        replace(result.geometry, image_seq_len=1)
    with pytest.raises(ScheduleContractError):
        replace(result, mu=result.mu + 0.1)
    with pytest.raises(ScheduleContractError):
        replace(result, extrapolated=True)
    with pytest.raises(ScheduleContractError):
        replace(result, profile_version="2")


@pytest.mark.parametrize(
    ("width", "height"),
    (
        (0, 1024),
        (-1, 1024),
        (1024, 0),
        (1024, -1),
        (True, 1024),
        (1024, False),
        (1024.0, 1024),
        (1024, 1024.0),
        ("1024", 1024),
        (1024, None),
    ),
)
def test_invalid_requested_dimensions_fail_closed(width: object, height: object) -> None:
    with pytest.raises(ScheduleContractError):
        resolve_krea2_image_geometry(
            cast(Any, width),
            cast(Any, height),
            policy=KREA2_RAW_PROFILE.dimensions,
        )


@pytest.mark.parametrize(
    "image_seq_len",
    (0, -1, True, False, 256.0, "256", None),
)
def test_invalid_sequence_lengths_fail_closed(image_seq_len: object) -> None:
    with pytest.raises(ScheduleContractError):
        calculate_krea2_raw_mu(cast(Any, image_seq_len))


@pytest.mark.parametrize(
    "factory",
    (
        lambda: resolve_krea2_image_geometry(
            1024,
            1024,
            policy=cast(DimensionPolicy, object()),
        ),
        lambda: calculate_krea2_raw_mu(
            4096,
            policy=cast(ResolutionShiftPolicy, object()),
        ),
        lambda: derive_krea2_raw_shift(
            1024,
            1024,
            profile=cast(Krea2RawProfile, object()),
        ),
        lambda: Krea2ImageGeometry(
            requested_width=1024,
            requested_height=1024,
            effective_width=1024,
            effective_height=1024,
            alignment_multiple=8,
            grid_width=128,
            grid_height=128,
            image_seq_len=16384,
        ),
        lambda: Krea2ImageGeometry(
            requested_width=1024,
            requested_height=1024,
            effective_width=1024,
            effective_height=1024,
            alignment_multiple=16,
            grid_width=64,
            grid_height=0,
            image_seq_len=0,
        ),
        lambda: Krea2RawShiftDerivation(
            profile_id="krea2.raw.official",
            profile_version="1",
            geometry=cast(Krea2ImageGeometry, object()),
            mu=0.90625,
            extrapolated=False,
        ),
        lambda: Krea2RawShiftDerivation(
            profile_id="krea2.raw.official",
            profile_version="1",
            geometry=resolve_krea2_image_geometry(
                1024,
                1024,
                policy=KREA2_RAW_PROFILE.dimensions,
            ),
            mu=float("nan"),
            extrapolated=False,
        ),
        lambda: Krea2RawShiftDerivation(
            profile_id="krea2.raw.official",
            profile_version="1",
            geometry=resolve_krea2_image_geometry(
                1024,
                1024,
                policy=KREA2_RAW_PROFILE.dimensions,
            ),
            mu=0.90625,
            extrapolated=cast(bool, 1),
        ),
    ),
)
def test_invalid_geometry_and_derivation_contracts_fail_closed(factory: Any) -> None:
    with pytest.raises(ScheduleContractError):
        factory()
