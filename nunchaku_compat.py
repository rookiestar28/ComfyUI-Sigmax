# Nunchaku Qwen Direct Model Patcher
# 
# This module monkey-patches Nunch aku models at runtime to fix dimension mismatches
# It installs a wrapper around model.apply_model that detects and fixes tensor shapes

import torch
import logging

logger = logging.getLogger(__name__)

_original_module_call = None
_original_tiled_call = None
_patch_applied = False

def is_nunchaku_qwen_model(model):
    """Detect if the model is a Nunchaku Qwen model"""
    try:
        if hasattr(model, 'diffusion_model'):
            dm = model.diffusion_model
            if hasattr(dm, 'txt_norm') and hasattr(dm.txt_norm, 'normalized_shape'):
                return True
        return False
    except Exception:
        return False


def get_expected_txt_dim(model):
    """Get the expected text encoder dimension from txt_norm"""
    try:
        if hasattr(model, 'diffusion_model'):
            dm = model.diffusion_model
            if hasattr(dm, 'txt_norm') and hasattr(dm.txt_norm, 'normalized_shape'):
                return dm.txt_norm.normalized_shape[0]
        return None
    except Exception:
        return None


def patched_apply_model(original_func):
    """Wrapper for model.apply_model that fixes dimension mismatches"""
    projection_cache = {}
    
    def wrapper(self, *args, **kwargs):
        if not is_nunchaku_qwen_model(self):
            return original_func(self, *args, **kwargs)
        
        expected_dim = get_expected_txt_dim(self)
        if expected_dim is None:
            return original_func(self, *args, **kwargs)
        
        context = kwargs.get('context', None)
        
        if context is not None and context.shape[-1] != expected_dim:
            actual_dim = context.shape[-1]
            print(f"[Nunchaku Compat] Fixing dimension: {actual_dim} -> {expected_dim}")
            
            cache_key = (actual_dim, expected_dim, context.device, context.dtype)
            
            if cache_key not in projection_cache:
                projection = torch.nn.Linear(actual_dim, expected_dim, bias=False, 
                                            device=context.device, dtype=context.dtype)
                with torch.no_grad():
                    if actual_dim > expected_dim:
                        projection.weight.data = torch.eye(expected_dim, actual_dim, 
                                                          device=context.device, dtype=context.dtype)
                    else:
                        projection.weight.data = torch.zeros(expected_dim, actual_dim, 
                                                            device=context.device, dtype=context.dtype)
                        projection.weight.data[:actual_dim, :] = torch.eye(actual_dim, 
                                                                          device=context.device, dtype=context.dtype)
                projection_cache[cache_key] = projection
            
            context = projection_cache[cache_key](context)
            kwargs['context'] = context
        
        return original_func(self, *args, **kwargs)
    
    return wrapper


def patch_diffusion_model_forward(original_forward):
    """Wrap diffusion_model's forward/__call__ to fix encoder_hidden_states"""
    projection_cache = {}
    
    def wrapper(self, *args, **kwargs):
        # Check if this is a Nunchaku model by looking for txt_norm
        if not (hasattr(self, 'txt_norm') and hasattr(self.txt_norm, 'normalized_shape')):
            return original_forward(self, *args, **kwargs)
        
        expected_dim = self.txt_norm.normalized_shape[0]
        
        # Extract encoder_hidden_states from kwargs
        # The diffusion model is called with: diffusion_model(xc, t, context=context, ...)
        # which maps to: forward(hidden_states, encoder_hidden_states, ...)
        # So 'context' kwarg becomes encoder_hidden_states parameter
        
        encoder_hidden_states = None
        param_name = None
        
        # Try common parameter names
        for name in ['context', 'encoder_hidden_states', 'text_embeds']:
            if name in kwargs:
                encoder_hidden_states = kwargs[name]
                param_name = name
                break
        
        # If not in kwargs, it might be in args (positional)
        # Typical signature: forward(self, hidden_states, encoder_hidden_states, timestep, ...)
        if encoder_hidden_states is None and len(args) > 1:
            # args[0] = hidden_states, args[1] = encoder_hidden_states
            if isinstance(args[1], torch.Tensor) and len(args[1].shape) >= 2:
                encoder_hidden_states = args[1]
                param_name = 'args[1]'
        
        # Check and fix dimension mismatch
        if encoder_hidden_states is not None and isinstance(encoder_hidden_states, torch.Tensor):
            # encoder_hidden_states should be shape [batch, seq_len, dim]
            if len(encoder_hidden_states.shape) >= 2:
                actual_dim = encoder_hidden_states.shape[-1]
                
                if actual_dim != expected_dim:
                    print(f"[Nunchaku Compat] Fixing {param_name}: shape={list(encoder_hidden_states.shape)}, {actual_dim} -> {expected_dim}")
                    
                    cache_key = (actual_dim, expected_dim, encoder_hidden_states.device, encoder_hidden_states.dtype)
                    
                    if cache_key not in projection_cache:
                        projection = torch.nn.Linear(actual_dim, expected_dim, bias=False,
                                                    device=encoder_hidden_states.device, 
                                                    dtype=encoder_hidden_states.dtype)
                        with torch.no_grad():
                            if actual_dim > expected_dim:
                                projection.weight.data = torch.eye(expected_dim, actual_dim,
                                                                  device=encoder_hidden_states.device, 
                                                                  dtype=encoder_hidden_states.dtype)
                        projection_cache[cache_key] = projection
                        print(f"[Nunchaku Compat] Created projection layer {actual_dim}->{expected_dim}")
                    
                    encoder_hidden_states = projection_cache[cache_key](encoder_hidden_states)
                    
                    # Update the parameter
                    if param_name in kwargs:
                        kwargs[param_name] = encoder_hidden_states
                    elif param_name == 'args[1]':
                        args = list(args)
                        args[1] = encoder_hidden_states
                        args = tuple(args)
        
        return original_forward(self, *args, **kwargs)
    
    return wrapper


# Global variables to store original functions
_original_module_call = None
_original_tiled_call = None

def apply_nunchaku_patches():
    """Apply monkey patches to fix Nunchaku compatibility issues"""
    global _patch_applied, _original_module_call, _original_tiled_call
    
    if _patch_applied:
        print("[Nunchaku Compat] Patches already applied")
        return
    
    try:
        # We need to patch at the diffusion_model level, not model.apply_model
        # The patch will be applied when models are loaded
        import torch.nn as nn
        
        # Store original if not already stored
        if _original_module_call is None:
            _original_module_call = nn.Module.__call__
        
        # Patch torch.nn.Module's __call__ for modules that have txt_norm
        # This is tricky - we'll patch specific Nunchaku model classes when we detect them
        
        def patched_module_call(self, *args, **kwargs):
            # ONLY patch Nunchaku diffusion models - very specific detection
            # Must have all three: txt_norm, txt_in, img_in (unique to Nunchaku Qwen)
            is_nunchaku_diffusion = (
                hasattr(self, 'txt_norm') and 
                hasattr(self, 'txt_in') and 
                hasattr(self, 'img_in') and
                hasattr(self, 'transformer_blocks')  # Extra check to ensure it's the diffusion model
            )
            
            if is_nunchaku_diffusion:
                # This is a NunchakuQwenImageTransformer2DModel
                if not hasattr(self, '_nunchaku_patched'):
                    print(f"[Nunchaku Compat] Detected and patching Nunchaku diffusion model: {type(self).__name__}")
                    self._nunchaku_patched = True
                
                return patch_diffusion_model_forward(_original_module_call)(self, *args, **kwargs)
            
            # For all other modules (VAE, etc.), use original __call__ without modification
            return _original_module_call(self, *args, **kwargs)
        
        nn.Module.__call__ = patched_module_call
        
        # Also patch TiledDiffusion if present
        try:
            import sys
            if 'ComfyUI-TiledDiffusion.tiled_diffusion' in sys.modules:
                tiled_diff = sys.modules['ComfyUI-TiledDiffusion.tiled_diffusion']
                if hasattr(tiled_diff, 'TiledDiffusion'):
                    if _original_tiled_call is None:
                        _original_tiled_call = tiled_diff.TiledDiffusion.__call__
                    
                    def patched_tiled_call(self, model_function, kwargs):
                        """Wrap TiledDiffusion to handle 5D tensors from Qwen Image models"""
                        x_in = kwargs.get('input', None)
                        
                        # Check if we have a 5D tensor
                        if x_in is not None and len(x_in.shape) == 5:
                            # Shape is [N, C, F, H, W], squeeze F dimension if it's 1
                            N, C, F, H, W = x_in.shape
                            
                            if F == 1:
                                print(f"[Nunchaku Compat] TiledDiffusion: Squeezing 5D tensor {list(x_in.shape)} -> 4D")
                                kwargs['input'] = x_in.squeeze(2)  # Remove F dimension
                                
                                # Call original with 4D tensor
                                result = _original_tiled_call(self, model_function, kwargs)
                                
                                # Restore 5D shape if result is 4D
                                if isinstance(result, torch.Tensor) and len(result.shape) == 4:
                                    result = result.unsqueeze(2)  # Add F dimension back
                                    print(f"[Nunchaku Compat] TiledDiffusion: Restored to 5D shape {list(result.shape)}")
                                
                                return result
                            else:
                                print(f"[Nunchaku Compat] TiledDiffusion: Warning - 5D tensor with F={F} (not 1), cannot safely squeeze")
                        
                        return _original_tiled_call(self, model_function, kwargs)
                    
                    tiled_diff.TiledDiffusion.__call__ = patched_tiled_call
                    print("[Nunchaku Compat] Successfully patched TiledDiffusion for 5D tensor support")
        except Exception as e:
            print(f"[Nunchaku Compat] Could not patch TiledDiffusion (not installed or incompatible): {e}")
        
        print("[Nunchaku Compat] Successfully installed Nunchaku compatibility patches")
        _patch_applied = True
            
    except Exception as e:
        print(f"[Nunchaku Compat] Error applying patches: {e}")
        import traceback
        traceback.print_exc()


def remove_nunchaku_patches():
    """Remove Nunchaku compatibility patches"""
    global _patch_applied, _original_module_call, _original_tiled_call
    
    if not _patch_applied:
        print("[Nunchaku Compat] Patches not applied, nothing to remove")
        return

    try:
        import torch.nn as nn
        
        # Restore nn.Module.__call__
        if _original_module_call is not None:
            nn.Module.__call__ = _original_module_call
            print("[Nunchaku Compat] Restored original nn.Module.__call__")
        
        # Restore TiledDiffusion.__call__
        try:
            import sys
            if 'ComfyUI-TiledDiffusion.tiled_diffusion' in sys.modules:
                tiled_diff = sys.modules['ComfyUI-TiledDiffusion.tiled_diffusion']
                if hasattr(tiled_diff, 'TiledDiffusion') and _original_tiled_call is not None:
                    tiled_diff.TiledDiffusion.__call__ = _original_tiled_call
                    print("[Nunchaku Compat] Restored original TiledDiffusion.__call__")
        except Exception as e:
            print(f"[Nunchaku Compat] Error restoring TiledDiffusion: {e}")
            
        _patch_applied = False
        print("[Nunchaku Compat] Successfully removed Nunchaku compatibility patches")
        
    except Exception as e:
        print(f"[Nunchaku Compat] Error removing patches: {e}")
        import traceback
        traceback.print_exc()


class NunchakuQwenPatches:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["enable", "disable"], {"default": "enable"}),
            },
            "optional": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("MODEL", "IMAGE",)
    RETURN_NAMES = ("model", "image",)
    FUNCTION = "execute"
    CATEGORY = "utils"
    
    def execute(self, mode, model=None, image=None):
        if mode == "enable":
            apply_nunchaku_patches()
        else:
            remove_nunchaku_patches()
        return (model, image)

NODE_CLASS_MAPPINGS = {
    "NunchakuQwenPatches": NunchakuQwenPatches
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NunchakuQwenPatches": "Nunchaku Qwen Patches"
}
