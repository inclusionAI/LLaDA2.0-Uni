"""
ComfyUI Custom Nodes for LLaDA2.0_Uni
Supports: Text-to-Image, Image Understanding, Image Editing, Decode, Unload

Features:
- Attention backend selection: Flash Attention, SDPA
- CPU offload mode for limited VRAM
- Auto VRAM cleanup after inference
- Decoder model caching (60-67% speedup on repeated decodes)
- Manual unload node
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
import os
import shutil
import subprocess


# ═══════════════════════════════════════════════════════════════
#  LLaDA2.0_Uni Loader
# ═══════════════════════════════════════════════════════════════

class LLaDA2UniLoader:
    """Load LLaDA2.0_Uni model with attention & offload options."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "inclusionAI/LLaDA2.0-Uni"}),
                "attention": (["flash_attn", "sdpa"],),
                "dtype": (["bf16", "fp8"],),
                "offload": ("BOOLEAN", {"default": False}),
                "device": (["cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("LLADA2UNI_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LLaDA2.0_Uni"
    DESCRIPTION = "Load LLaDA2.0_Uni model from specified path."

    def load(self, model_path, attention, dtype, offload, device):
        if not isinstance(device, str) or not device:
            device = "cuda"
        
        # Validate model path
        if not os.path.isdir(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        from .model_manager import load_llm
        model, tokenizer = load_llm(model_path, device=device,
                                     attention=attention, offload=offload, dtype=dtype)
        return ({
            "model_path": model_path,
            "device": device,
            "attention": attention,
            "dtype": dtype,
            "offload": offload,
        },)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ═══════════════════════════════════════════════════════════════
#  Text-to-Image
# ═══════════════════════════════════════════════════════════════

class LLaDA2UniTextToImage:
    """Generate image tokens from text prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLADA2UNI_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 32}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "mode": (["standard", "thinking"],),
                "thinking_steps": ("INT", {"default": 32, "min": 1, "max": 64}),
                "thinking_length": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 512}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
                "block_length": ("INT", {"default": 32, "min": 1, "max": 128}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LLADA2UNI_TOKENS", "STRING",)
    RETURN_NAMES = ("tokens", "thinking_output",)
    FUNCTION = "generate"
    CATEGORY = "LLaDA2.0_Uni"
    DESCRIPTION = "Generate VQ image tokens from text prompt."

    def generate(self, model, prompt, width, height, steps, cfg_scale,
                 mode="standard", thinking_steps=32, thinking_length=4096,
                 seed=-1, block_length=32, unload_after=True):

        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        from .model_manager import load_llm, unload_llm

        llm, _ = load_llm(model["model_path"], model["device"],
                          model.get("attention", "flash_attn"),
                          model.get("offload", False),
                          model.get("dtype", "bf16"))

        # Generate image tokens
        if mode == "thinking":
            result = llm.generate_image(
                prompt, image_h=height, image_w=width,
                mode="thinking",
                steps=steps, cfg_scale=cfg_scale,
                thinking_steps=thinking_steps,
                thinking_gen_length=thinking_length,
                block_length=block_length,
            )
            thinking_text = result.get("thinking", "")
        else:
            result = llm.generate_image(
                prompt, image_h=height, image_w=width,
                steps=steps, cfg_scale=cfg_scale,
                block_length=block_length,
            )
            thinking_text = ""

        token_data = {
            "token_ids": result["token_ids"],
            "h": result["h"],
            "w": result["w"],
            "model_path": model["model_path"],
        }

        # Auto VRAM cleanup
        if unload_after:
            unload_llm()

        return (token_data, thinking_text,)


# ═══════════════════════════════════════════════════════════════
#  Image Understanding (VQA)
# ═══════════════════════════════════════════════════════════════

class LLaDA2UniImageUnderstanding:
    """Understand/describe an image using VQA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLADA2UNI_MODEL",),
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
            },
            "optional": {
                "gen_steps": ("INT", {"default": 32, "min": 1, "max": 64}),
                "gen_length": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "block_length": ("INT", {"default": 32, "min": 1, "max": 128}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "understand"
    CATEGORY = "LLaDA2.0_Uni"
    DESCRIPTION = "Ask questions about an image (VQA)."

    def understand(self, model, image, question, gen_steps=32, gen_length=2048,
                   block_length=32, unload_after=True):

        from .model_manager import load_llm, get_image_tokenizer, unload_llm
        from decoder.smart_img_process import smart_resize_images

        llm, _ = load_llm(model["model_path"], model["device"],
                          model.get("attention", "flash_attn"),
                          model.get("offload", False),
                          model.get("dtype", "bf16"))

        # ComfyUI tensor (B,H,W,C) → PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Encode image
        image_tokenizer = get_image_tokenizer(model["model_path"], model["device"])
        pil_image = smart_resize_images([pil_image])[0]
        info = image_tokenizer.encode_with_info(pil_image)
        image_tokens = [x + llm.config.image_token_offset for x in info["token_ids"]]
        _, h, w = info["grid_thw"]

        # Understand the image
        response = llm.understand_image(
            image_tokens, h, w,
            question=question,
            steps=gen_steps,
            gen_length=gen_length,
            block_length=block_length,
        )

        # Auto VRAM cleanup
        if unload_after:
            unload_llm()

        return (response,)


# ═══════════════════════════════════════════════════════════════
#  Image Editing
# ═══════════════════════════════════════════════════════════════

class LLaDA2UniImageEditing:
    """Edit an image based on a text instruction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLADA2UNI_MODEL",),
                "image": ("IMAGE",),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "steps": ("INT", {"default": 8, "min": 1, "max": 32}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cfg_image_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "block_length": ("INT", {"default": 32, "min": 1, "max": 128}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LLADA2UNI_TOKENS",)
    RETURN_NAMES = ("tokens",)
    FUNCTION = "edit"
    CATEGORY = "LLaDA2.0_Uni"
    DESCRIPTION = "Edit an image with a text instruction. Connect output to Decode node."

    def edit(self, model, image, instruction, steps=8, cfg_text_scale=4.0,
             cfg_image_scale=0.0, block_length=32, seed=-1, unload_after=True):

        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        from .model_manager import load_llm, get_image_tokenizer, unload_llm
        from decoder.utils import generate_crop_size_list, var_center_crop

        llm, _ = load_llm(model["model_path"], model["device"],
                          model.get("attention", "flash_attn"),
                          model.get("offload", False),
                          model.get("dtype", "bf16"))

        # ComfyUI tensor → PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Encode source image (preserve aspect ratio via var_center_crop)
        image_tokenizer = get_image_tokenizer(model["model_path"], model["device"])
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        pil_image = var_center_crop(pil_image, crop_size_list=crop_size_list)
        info = image_tokenizer.encode_with_info(pil_image)
        image_tokens = [x + llm.config.image_token_offset for x in info["token_ids"]]
        _, h, w = info["grid_thw"]

        # Edit the image
        result = llm.edit_image(
            image_tokens, h, w, instruction,
            steps=steps,
            block_length=block_length,
            cfg_text_scale=cfg_text_scale,
            cfg_image_scale=cfg_image_scale,
        )

        token_data = {
            "token_ids": result["token_ids"],
            "h": result["h"],
            "w": result["w"],
            "model_path": model["model_path"],
        }

        # Auto VRAM cleanup
        if unload_after:
            unload_llm()

        return (token_data,)


# ═══════════════════════════════════════════════════════════════
#  Token Decoder
# ═══════════════════════════════════════════════════════════════

class LLaDA2UniImageDecode:
    """Decode VQ tokens from LLaDA2.0_Uni into a pixel image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokens": ("LLADA2UNI_TOKENS",),
            },
            "optional": {
                "decode_mode": (["decoder-turbo", "normal"],),
                "decoder_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "resolution_multiplier": ("INT", {"default": 2, "min": 1, "max": 4}),
                "unload_after": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "LLaDA2.0_Uni"
    DESCRIPTION = "Decode VQ tokens to pixel image. decoder-turbo is ~10× faster (8 steps vs 50)."

    def decode(self, tokens, decode_mode="decoder-turbo", decoder_steps=50,
               resolution_multiplier=2, unload_after=False):

        from .model_manager import decode_tokens, unload_all

        num_steps = 8 if decode_mode == "decoder-turbo" else decoder_steps

        # ComfyUI progress bar integration
        progress_callback = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_steps)
            def progress_callback(step, total):
                pbar.update(1)
        except ImportError:
            pass

        pil_image = decode_tokens(
            tokens["token_ids"],
            tokens["h"],
            tokens["w"],
            tokens["model_path"],
            device="cuda",
            num_steps=num_steps,
            decode_mode=decode_mode,
            resolution_multiplier=resolution_multiplier,
            progress_callback=progress_callback,
        )

        # PIL → ComfyUI tensor (1, H, W, 3) float32 [0,1]
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0)

        # Auto VRAM cleanup after full pipeline
        if unload_after:
            unload_all()

        return (tensor,)


# ═══════════════════════════════════════════════════════════════
#  Manual Unload
# ═══════════════════════════════════════════════════════════════

class LLaDA2UniUnloadModel:
    """Manually unload all LLaDA2.0_Uni components and free VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "LLaDA2.0_Uni"
    DESCRIPTION = "Unload all LLaDA2.0_Uni models from VRAM."

    def unload(self):
        from .model_manager import unload_all
        unload_all()
        print("[LLaDA2.0_Uni] ✅ All models unloaded, VRAM freed")
        return ()
