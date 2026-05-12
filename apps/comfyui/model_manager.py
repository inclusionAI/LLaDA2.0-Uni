"""
Model manager for LLaDA2.0_Uni ComfyUI nodes.
Handles loading/unloading, attention backends, CPU offload, VRAM management,
and decoder model caching.
"""

import torch
import gc
import sys
import os
from typing import Dict, Any

# ── Add project root to sys.path so encoder/ and decoder/ are importable ──
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Global state ──
_LLM_MODEL = None
_LLM_TOKENIZER = None
_IMAGE_TOKENIZER = None
_MODEL_PATH = None
_ATTENTION = None
_DEVICE = "cuda"
_OFFLOAD = False
_DTYPE = "bf16"

# Decoder cache (module-level)
_SIGVQ_MODEL = None
_DIFF_MODEL = None
_DIFF_MODE = None
_DIFF_CONFIG = None
_VAE_MODEL = None
_DECODER_MODEL_PATH = None


def _resolve_torch_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp8":
        print("[LLaDA2.0_Uni] FP8 mode: using bf16 compute dtype for compatibility.")
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


# ═══════════════════════════════════════════════════════════════
#  LLM Loading
# ═══════════════════════════════════════════════════════════════

def load_llm(model_path: str, device: str = "cuda", attention: str = "flash_attn",
             offload: bool = False, dtype: str = "bf16"):
    """Load the dLLM-MoE backbone. Returns (model, tokenizer)."""
    global _LLM_MODEL, _LLM_TOKENIZER, _MODEL_PATH, _ATTENTION, _DEVICE, _OFFLOAD, _DTYPE

    _ATTENTION = attention
    _DEVICE = device
    _OFFLOAD = offload
    _DTYPE = dtype

    if _LLM_MODEL is not None and _MODEL_PATH == model_path and _DTYPE == dtype:
        return _LLM_MODEL, _LLM_TOKENIZER

    unload_llm()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    attn_kwargs = {"trust_remote_code": True}
    if attention == "sdpa":
        attn_kwargs["attn_implementation"] = "sdpa"

    if offload:
        attn_kwargs["device_map"] = "auto"
        attn_kwargs["max_memory"] = {0: "20GiB", "cpu": "80GiB"}
        attn_kwargs["offload_folder"] = "offload_cache"
        attn_kwargs["torch_dtype"] = _resolve_torch_dtype(dtype)
    else:
        attn_kwargs["device_map"] = device
        attn_kwargs["torch_dtype"] = _resolve_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **attn_kwargs).eval()
    model.tokenizer = tokenizer

    _LLM_MODEL = model
    _LLM_TOKENIZER = tokenizer
    _MODEL_PATH = model_path
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════
#  Image Tokenizer
# ═══════════════════════════════════════════════════════════════

def get_image_tokenizer(model_path: str, device: str = "cuda"):
    """Load the SigLIP-VQ image tokenizer."""
    global _IMAGE_TOKENIZER
    if _IMAGE_TOKENIZER is None:
        from encoder.image_tokenizer import ImageTokenizer
        _IMAGE_TOKENIZER = ImageTokenizer(model_path=model_path, device=device)
    return _IMAGE_TOKENIZER


# ═══════════════════════════════════════════════════════════════
#  Decoder (with caching)
# ═══════════════════════════════════════════════════════════════

def decode_tokens(token_ids, h, w, model_path: str, device: str = "cuda",
                  num_steps: int = 50, decode_mode: str = "normal",
                  resolution_multiplier: int = 2, progress_callback=None):
    """Decode VQ tokens to PIL image, with model caching."""
    global _SIGVQ_MODEL, _DIFF_MODEL, _DIFF_MODE, _DIFF_CONFIG, _VAE_MODEL, _DECODER_MODEL_PATH

    import json
    import torch.nn.functional as F
    from tqdm import tqdm
    from torchvision.transforms.functional import to_pil_image
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file
    from decoder.sigvq import SigVQ
    from decoder.decoder_model import ZImageTransformer2DModel
    from decoder.transport import create_transport, Sampler

    dtype = torch.bfloat16

    # ── Stage 1: SigVQ → semantic features (cached) ──
    sigvq_path = os.path.join(model_path, "image_tokenizer", "sigvq_embedding.pt")
    if _SIGVQ_MODEL is None or _DECODER_MODEL_PATH != model_path:
        extractor = SigVQ(vocab_size=16384, inner_dim=4096).to(device, dtype=dtype)
        extractor.load_state_dict(torch.load(sigvq_path, map_location=device, weights_only=True))
        extractor.eval()
        _SIGVQ_MODEL = extractor
        _DECODER_MODEL_PATH = model_path
        print("[LLaDA2.0_Uni Decoder] SigVQ loaded and cached.")

    th = h * 16 * resolution_multiplier
    tw = w * 16 * resolution_multiplier
    tok = torch.tensor(token_ids).view(1, 1, h, w).float().to(device)
    up = F.interpolate(tok, scale_factor=2, mode="nearest").long().view(1, -1)
    cap_pos = [_SIGVQ_MODEL(up).squeeze(0)]
    cap_neg = [torch.zeros_like(cap_pos[0])]

    # ── Stage 2: Diffusion ODE sampling (cached) ──
    if decode_mode == "decoder-turbo":
        decoder_dir = os.path.join(model_path, "decoder-turbo")
    else:
        decoder_dir = os.path.join(model_path, "decoder")

    if _DIFF_MODEL is None or _DIFF_MODE != decode_mode or _DECODER_MODEL_PATH != model_path:
        # Free old model if mode changed
        if _DIFF_MODEL is not None:
            del _DIFF_MODEL
            gc.collect()
            torch.cuda.empty_cache()

        config_path = os.path.join(decoder_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["axes_lens"] = [32768, 1024, 1024]
        cfg["cap_feat_dim"] = 4096

        with torch.device("meta"):
            diff_model = ZImageTransformer2DModel(**cfg)
        ckpt = os.path.join(decoder_dir, "model.safetensors")
        diff_model.load_state_dict(load_file(ckpt, device=str(device)), assign=True)
        diff_model = diff_model.to(dtype=dtype).eval()

        _DIFF_MODEL = diff_model
        _DIFF_MODE = decode_mode
        _DIFF_CONFIG = cfg
        print(f"[LLaDA2.0_Uni Decoder] Diffusion model ({decode_mode}) loaded and cached.")

    cfg = _DIFF_CONFIG

    # Create model function for sampling
    n = len(cap_pos)
    doubled = cap_pos + cap_neg
    cfg_scale = 0.0 if decode_mode == "decoder-turbo" else 1.0
    patch_size = cfg.get("all_patch_size", (2,))[0]
    f_patch_size = cfg.get("all_f_patch_size", (1,))[0]

    def model_fn(x, t, **kw):
        t_t = torch.tensor([t], device=x.device, dtype=torch.float32) if not isinstance(t, torch.Tensor) else t.float()
        if t_t.dim() == 0: t_t = t_t.unsqueeze(0)
        if t_t.shape[0] == 1 and x.shape[0] > 1: t_t = t_t.expand(x.shape[0])
        if cfg_scale > 0:
            out = _DIFF_MODEL(x=list(x.to(dtype).repeat(2, 1, 1, 1, 1).unbind(0)), t=t_t.repeat(2),
                              cap_feats=doubled, patch_size=patch_size, f_patch_size=f_patch_size, return_dict=False)
            pos, neg = out[0][:n], out[0][n:]
            res = []
            for p, ng in zip(pos, neg):
                p, ng = p.float(), ng.float()
                pred = p + cfg_scale * (p - ng)
                on, nn_ = torch.linalg.vector_norm(p), torch.linalg.vector_norm(pred)
                if nn_ > on:
                    pred *= on / nn_
                res.append(pred)
            return torch.stack(res)
        out = _DIFF_MODEL(x=list(x.to(dtype).unbind(0)), t=t_t, cap_feats=cap_pos,
                          patch_size=patch_size, f_patch_size=f_patch_size, return_dict=False)
        return torch.stack([o.float() for o in out[0]])

    z = torch.randn([1, 16, 1, 2 * (th // 16), 2 * (tw // 16)], device=device)
    sampler = Sampler(create_transport("Linear", "velocity", None))
    sample_fn = sampler.sample_ode(
        sampling_method="euler", num_steps=num_steps,
        atol=1e-6, rtol=1e-3, reverse=False, time_shifting_factor=6,
        stochast_ratio=1.0 if decode_mode == "decoder-turbo" else 0.0)

    step_counter = [0]
    if progress_callback is not None:
        def wrapped(x, t, **kw):
            step_counter[0] += 1
            progress_callback(step_counter[0], num_steps)
            return model_fn(x, t, **kw)
    else:
        pbar = tqdm(total=num_steps, desc="Decoding", leave=False)
        def wrapped(x, t, **kw):
            pbar.update(1)
            return model_fn(x, t, **kw)

    with torch.inference_mode():
        samples = sample_fn(z, wrapped)[-1].squeeze(2)

    if progress_callback is None:
        pbar.close()

    # ── Stage 3: VAE decode (cached) ──
    vae_dir = os.path.join(model_path, "vae")
    if _VAE_MODEL is None or _DECODER_MODEL_PATH != model_path:
        _VAE_MODEL = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=dtype).to(device).eval()
        print("[LLaDA2.0_Uni Decoder] VAE loaded and cached.")

    with torch.inference_mode():
        s = samples.to(dtype)
        s = (s / _VAE_MODEL.config.scaling_factor) + _VAE_MODEL.config.shift_factor
        px = ((_VAE_MODEL.decode(s, return_dict=False)[0] + 1) / 2).clamp_(0, 1)

    return to_pil_image(px[0].float())


# ═══════════════════════════════════════════════════════════════
#  Unload functions
# ═══════════════════════════════════════════════════════════════

def unload_llm():
    """Unload LLM backbone to free VRAM."""
    global _LLM_MODEL, _LLM_TOKENIZER
    if _LLM_MODEL is not None:
        del _LLM_MODEL, _LLM_TOKENIZER
        _LLM_MODEL = None
        _LLM_TOKENIZER = None
        gc.collect()
        torch.cuda.empty_cache()


def unload_decoder():
    """Unload all decoder components from VRAM."""
    global _SIGVQ_MODEL, _DIFF_MODEL, _DIFF_MODE, _DIFF_CONFIG, _VAE_MODEL, _DECODER_MODEL_PATH
    for obj in (_SIGVQ_MODEL, _DIFF_MODEL, _VAE_MODEL):
        if obj is not None:
            del obj
    _SIGVQ_MODEL = None
    _DIFF_MODEL = None
    _DIFF_MODE = None
    _DIFF_CONFIG = None
    _VAE_MODEL = None
    _DECODER_MODEL_PATH = None
    gc.collect()
    torch.cuda.empty_cache()


def unload_image_tokenizer():
    """Unload image tokenizer."""
    global _IMAGE_TOKENIZER
    if _IMAGE_TOKENIZER is not None:
        del _IMAGE_TOKENIZER
        _IMAGE_TOKENIZER = None
        gc.collect()
        torch.cuda.empty_cache()


def unload_all():
    """Unload everything. Call this to free all VRAM."""
    unload_llm()
    unload_decoder()
    unload_image_tokenizer()
