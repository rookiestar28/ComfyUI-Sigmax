"""Public package contract for ComfyUI-Sigmax."""

from comfyui_sigmax.adapters.registration import builtin_node_registry
from comfyui_sigmax.public_contracts import (
    PUBLIC_CONTRACT_MANIFEST_SCHEMA,
    PUBLIC_CONTRACT_VERSION,
    PublicContractManifest,
    load_public_contract_manifest,
)

__version__ = "0.1.0.dev0"

# Node milestones populate these mappings only after their behavior is validated.
_BUILTIN_NODE_REGISTRY = builtin_node_registry()
NODE_CLASS_MAPPINGS: dict[str, type] = _BUILTIN_NODE_REGISTRY.class_mappings()
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = _BUILTIN_NODE_REGISTRY.display_name_mappings()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "PUBLIC_CONTRACT_MANIFEST_SCHEMA",
    "PUBLIC_CONTRACT_VERSION",
    "PublicContractManifest",
    "__version__",
    "load_public_contract_manifest",
]
