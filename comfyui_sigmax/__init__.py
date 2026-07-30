"""Public package contract for ComfyUI-Sigmax."""

from comfyui_sigmax.adapters.registration import builtin_node_registry

__version__ = "0.1.0.dev0"

# Node milestones populate these mappings only after their behavior is validated.
_BUILTIN_NODE_REGISTRY = builtin_node_registry()
NODE_CLASS_MAPPINGS: dict[str, type] = _BUILTIN_NODE_REGISTRY.class_mappings()
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = _BUILTIN_NODE_REGISTRY.display_name_mappings()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
]
