"""Public package contract for ComfyUI-Sigmax."""

__version__ = "0.1.0.dev0"

# Node milestones populate these mappings only after their behavior is validated.
NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
