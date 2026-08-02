"""Versioned community-derived Krea 2 conditioning profiles."""

from __future__ import annotations

from typing import Final

from comfyui_sigmax.conditioning.contracts import (
    Krea2ConditioningProfile,
    Krea2ConditioningProfileId,
)
from comfyui_sigmax.core import EvidenceLevel

_SOURCE = "github.com/huwhitememes/comfyui-krea2-conditioning"
_SOURCE_REVISION = (
    "729cda4fade982988a375b01928f515458407a5c"  # pragma: allowlist secret -- public commit SHA
)

DISABLED_PROFILE: Final = Krea2ConditioningProfile(
    profile_id=Krea2ConditioningProfileId.DISABLED.value,
    profile_version="1",
    evidence=EvidenceLevel.COMMUNITY_RECOMMENDED,
    source=_SOURCE,
    source_revision=_SOURCE_REVISION,
    gains=(1.0,) * 12,
)
SUBTLE_EXPERIMENTAL_PROFILE: Final = Krea2ConditioningProfile(
    profile_id=Krea2ConditioningProfileId.SUBTLE_EXPERIMENTAL.value,
    profile_version="1",
    evidence=EvidenceLevel.COMMUNITY_RECOMMENDED,
    source=_SOURCE,
    source_revision=_SOURCE_REVISION,
    gains=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.5, 1.0),
)
CLASSIC_EXPERIMENTAL_PROFILE: Final = Krea2ConditioningProfile(
    profile_id=Krea2ConditioningProfileId.CLASSIC_EXPERIMENTAL.value,
    profile_version="1",
    evidence=EvidenceLevel.COMMUNITY_RECOMMENDED,
    source=_SOURCE,
    source_revision=_SOURCE_REVISION,
    gains=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0),
)

KREA2_CONDITIONING_PROFILES: Final = {
    Krea2ConditioningProfileId.DISABLED: DISABLED_PROFILE,
    Krea2ConditioningProfileId.SUBTLE_EXPERIMENTAL: SUBTLE_EXPERIMENTAL_PROFILE,
    Krea2ConditioningProfileId.CLASSIC_EXPERIMENTAL: CLASSIC_EXPERIMENTAL_PROFILE,
}
