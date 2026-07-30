"""Dependency-free bootstrap for the ComfyUI-Sigmax custom-node package."""

# CRITICAL: keep bootstrap dependency-free; optional ComfyUI/runtime imports belong in adapters.
from .comfyui_sigmax import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    __version__,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
