"""
ComfyUI Custom Nodes for LLaDA2.0_Uni
Unified multimodal: Text-to-Image, Image Understanding (VQA), Image Editing

This node package lives inside LLaDA2.0_Uni/apps/comfyui/ and imports
encoder/decoder from the parent project directly.
"""

from .nodes import (
    LLaDA2UniLoader,
    LLaDA2UniTextToImage,
    LLaDA2UniImageUnderstanding,
    LLaDA2UniImageEditing,
    LLaDA2UniImageDecode,
    LLaDA2UniUnloadModel,
)

NODE_CLASS_MAPPINGS = {
    "LLaDA2UniLoader": LLaDA2UniLoader,
    "LLaDA2UniTextToImage": LLaDA2UniTextToImage,
    "LLaDA2UniImageUnderstanding": LLaDA2UniImageUnderstanding,
    "LLaDA2UniImageEditing": LLaDA2UniImageEditing,
    "LLaDA2UniImageDecode": LLaDA2UniImageDecode,
    "LLaDA2UniUnloadModel": LLaDA2UniUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLaDA2UniLoader": "LLaDA2.0_Uni Loader",
    "LLaDA2UniTextToImage": "LLaDA2.0_Uni Text-to-Image",
    "LLaDA2UniImageUnderstanding": "LLaDA2.0_Uni Image Understanding",
    "LLaDA2UniImageEditing": "LLaDA2.0_Uni Image Editing",
    "LLaDA2UniImageDecode": "LLaDA2.0_Uni Token Decoder",
    "LLaDA2UniUnloadModel": "LLaDA2.0_Uni Unload Model",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
