"""Dependency-free bootstrap for the ComfyUI-Sigmax custom-node package."""

# CRITICAL: keep bootstrap dependency-free; optional ComfyUI/runtime imports belong in adapters.
# IMPORTANT: pytest may import this checkout as top-level ``__init__`` without a package.
if __package__:
    from .comfyui_sigmax import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        __version__,
    )
else:
    from comfyui_sigmax import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        __version__,
    )

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
