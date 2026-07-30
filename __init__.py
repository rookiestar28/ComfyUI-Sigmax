"""Dependency-free bootstrap for the ComfyUI-Sigmax custom-node package."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_checkout_package() -> ModuleType:
    """Load this checkout's nested package under its canonical absolute name."""

    package_root = Path(__file__).resolve().parent / "comfyui_sigmax"
    package_init = package_root / "__init__.py"
    existing = sys.modules.get("comfyui_sigmax")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(existing_file).resolve() != package_init:
            raise ImportError("refusing to overwrite a different comfyui_sigmax package")
        return existing
    spec = importlib.util.spec_from_file_location(
        "comfyui_sigmax",
        package_init,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot create the comfyui_sigmax checkout package spec")
    package = importlib.util.module_from_spec(spec)
    # CRITICAL: ComfyUI loads custom-node roots under dynamic names; absolute internal imports
    # require this exact checkout alias before executing the nested package.
    sys.modules["comfyui_sigmax"] = package
    try:
        spec.loader.exec_module(package)
    except Exception:
        sys.modules.pop("comfyui_sigmax", None)
        raise
    return package


# CRITICAL: keep bootstrap dependency-free; optional ComfyUI/runtime imports belong in adapters.
# IMPORTANT: pytest may import this checkout as top-level ``__init__`` without a package.
if __package__:
    _package = _load_checkout_package()
    sys.modules[f"{__package__}.comfyui_sigmax"] = _package
    NODE_CLASS_MAPPINGS = _package.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = _package.NODE_DISPLAY_NAME_MAPPINGS
    __version__ = _package.__version__
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
