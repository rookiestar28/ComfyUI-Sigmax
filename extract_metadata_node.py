import torch
import os
import json
from PIL import Image, ImageOps
import folder_paths
import numpy as np

class ImageMetadataExtractor:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "positive_prompt", "width", "height", "filename")
    FUNCTION = "extract_metadata"
    CATEGORY = "utils"

    def extract_metadata(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        
        output_image = ImageOps.exif_transpose(img)
        output_image = output_image.convert("RGB")
        output_image = np.array(output_image).astype(np.float32) / 255.0
        output_image = torch.from_numpy(output_image)[None,]

        positive_prompt = ""
        width = 0
        height = 0

        # Extract from 'prompt' (API format) which is what ComfyUI uses for execution
        if 'prompt' in img.info:
            try:
                prompt = json.loads(img.info['prompt'])
                
                # 1. Find Positive Prompt
                # Strategy: Find KSampler -> positive input -> CLIPTextEncode -> text
                ksampler_nodes = []
                for node_id, node in prompt.items():
                    class_type = node.get('class_type', '')
                    if 'KSampler' in class_type or 'SamplerCustom' in class_type:
                        ksampler_nodes.append(node)
                
                for ksampler in ksampler_nodes:
                    inputs = ksampler.get('inputs', {})
                    if 'positive' in inputs:
                        positive_link = inputs['positive']
                        if isinstance(positive_link, list): # It's a link [node_id, slot_index]
                            positive_node_id = str(positive_link[0])
                            if positive_node_id in prompt:
                                positive_node = prompt[positive_node_id]
                                if positive_node.get('class_type') == 'CLIPTextEncode':
                                    positive_prompt = positive_node.get('inputs', {}).get('text', "")
                                    break # Found it
                
                # Fallback: Look for any CLIPTextEncode with "positive" in title/meta if not found via KSampler
                if not positive_prompt:
                     candidates = []
                     for node_id, node in prompt.items():
                        if node.get('class_type') == 'CLIPTextEncode':
                            title = node.get('_meta', {}).get('title', '').lower()
                            text = node.get('inputs', {}).get('text', "")
                            if 'positive' in title and 'negative' not in title:
                                candidates.append(text)
                            # Also consider just long text if no clear title match
                            elif len(text) > 50:
                                candidates.append(text)
                     
                     # Pick the longest candidate if any found
                     if candidates:
                         positive_prompt = max(candidates, key=len)

                # 2. Find Width/Height
                # Strategy: Find EmptyLatentImage
                for node_id, node in prompt.items():
                    if node.get('class_type') == 'EmptyLatentImage':
                        width = node.get('inputs', {}).get('width', 0)
                        height = node.get('inputs', {}).get('height', 0)
                        break
                
                # Fallback: Look for width/height in any node if still 0
                if width == 0 or height == 0:
                     for node_id, node in prompt.items():
                        inputs = node.get('inputs', {})
                        if 'width' in inputs and 'height' in inputs:
                            width = inputs['width']
                            height = inputs['height']
                            break

            except Exception as e:
                print(f"Error parsing metadata: {e}")

        return (output_image, positive_prompt, width, height, image)

# Node registration
NODE_CLASS_MAPPINGS = {
    "ImageMetadataExtractor": ImageMetadataExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMetadataExtractor": "Load Image ErosDiffusion"
}
