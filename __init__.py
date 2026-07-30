"""Dependency-free bootstrap for the ComfyUI-Sigmax custom-node package."""

# CRITICAL: keep bootstrap dependency-free; optional ComfyUI/runtime imports belong in adapters.
NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
