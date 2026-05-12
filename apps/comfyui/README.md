# LLaDA2.0-Uni ComfyUI Nodes

Custom ComfyUI nodes for [LLaDA 2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) — a unified multimodal diffusion language model supporting **text-to-image generation**, **image understanding (VQA)**, and **image editing**.

## Installation

> ⚠️ These nodes depend on the `encoder/` and `decoder/` modules in the project root. Do **not** copy `apps/comfyui` in isolation — the full repository must be present and the relative path `apps/comfyui` must be preserved.

### Option 1: Clone + symlink (recommended)

```bash
# 1. Clone the full project
git clone https://github.com/inclusionAI/LLaDA2.0-Uni.git

# 2. Symlink into ComfyUI's custom_nodes
cd /path/to/ComfyUI/custom_nodes
ln -s /path/to/LLaDA2.0-Uni/apps/comfyui ./LLaDA2Uni
```

### Option 2: One-line installer

```bash
bash /path/to/LLaDA2.0-Uni/apps/comfyui/install.sh /path/to/ComfyUI
```

### Dependencies

```bash
pip install -r apps/comfyui/requirements.txt
pip install flash-attn --no-build-isolation  # optional, recommended
```

## Model Weights

In the Loader node, set the model path to either a HuggingFace repo ID or a local directory:

**HuggingFace (auto-download):**
```
inclusionAI/LLaDA2.0-Uni
```

**Local path:**
```
/path/to/LLaDA2.0-Uni
```

Expected directory layout:

```
LLaDA2.0-Uni/
├── config.json                       # LLM config
├── model-*.safetensors               # LLM weights
├── tokenizer.json
├── decoder/
│   ├── config.json
│   └── model.safetensors             # diffusion decoder
├── decoder-turbo/
│   ├── config.json
│   └── model.safetensors             # turbo decoder (8-step)
├── vae/
│   └── diffusion_pytorch_model.safetensors
└── image_tokenizer/
    ├── config.json
    ├── preprocessor_config.json
    ├── model.safetensors             # SigLIP-VQ weights
    └── sigvq_embedding.pt
```

## Nodes

| Node | Description |
|------|-------------|
| **LLaDA2.0_Uni Loader** | Load the model (Flash Attention / SDPA, optional CPU offload) |
| **LLaDA2.0_Uni Text-to-Image** | Generate VQ image tokens from a text prompt (supports thinking mode) |
| **LLaDA2.0_Uni Image Understanding** | Visual question answering |
| **LLaDA2.0_Uni Image Editing** | Edit an image with a text instruction |
| **LLaDA2.0_Uni Token Decoder** | Decode VQ tokens to pixels (turbo or normal mode) |
| **LLaDA2.0_Uni Unload Model** | Manually free VRAM |

## Example Workflows

### Text-to-Image
```
Loader → Text-to-Image → Token Decoder → Preview Image
```

### Image Understanding
```
Load Image + Loader → Image Understanding → Show Text
```

### Image Editing
```
Load Image + Loader → Image Editing → Token Decoder → Preview Image
```

## Parameters

### Loader
- `model_path` — HuggingFace repo ID or local directory
- `attention` — `flash_attn` (recommended) or `sdpa`
- `dtype` — `bf16` (recommended) or `fp8`
- `offload` — enable CPU offload for limited VRAM
- `device` — `cuda` or `cpu`

### Text-to-Image
- `prompt` — text description
- `width` / `height` — output resolution
- `steps` — LLM denoising steps (8–32)
- `cfg_scale` — classifier-free guidance scale
- `mode` — `standard` or `thinking`
- `seed` — random seed (`-1` = random)
- `block_length` — block size for block-wise denoising

### Token Decoder
- `decode_mode` — `decoder-turbo` (fast, 8 steps) or `normal` (50 steps)
- `decoder_steps` — number of steps when using `normal` mode
- `resolution_multiplier` — upscale factor (typically `2`)
- `unload_after` — release decoder VRAM after decoding (set `False` to keep cached for faster repeated decodes)

## License

Same as the parent project. See the repository root for details.
