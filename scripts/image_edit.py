import os
import sys
import gc
import json
import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

_CURRENT_FILE = globals().get("__file__")
if not _CURRENT_FILE:
    if sys.argv and sys.argv[0]:
        _CURRENT_FILE = os.path.abspath(sys.argv[0])
    else:
        _CURRENT_FILE = os.path.abspath(os.path.join(os.getcwd(), "image_edit.py"))

_CURRENT_FILE = os.path.abspath(_CURRENT_FILE)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_FILE))
if _PROJECT_ROOT and _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from decoder import decode_vq_tokens


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be an integer")
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be an integer")
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to zero")
    return parsed


def _finite_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a finite number")
    if parsed != parsed or parsed == float("inf") or parsed == float("-inf"):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


def _non_negative_finite_float(value):
    parsed = _finite_float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to zero")
    return parsed


def _normalize_path(path):
    return os.path.abspath(os.path.expanduser(path))


def _prepare_output_path(path):
    path = _normalize_path(path)
    root, ext = os.path.splitext(path)
    if not ext or ext == ".":
        path = (root if root else path) + ".png"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def parse_args():
    p = argparse.ArgumentParser(description="LLaDA-2.0-Uni Image Editing")
    p.add_argument("--model_path", type=str, required=True, help="Root model dir containing LLM weights, image_tokenizer/, decoder/, vae/")
    image_group = p.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", type=str, default=None)
    image_group.add_argument("--image_token", type=str, default=None)
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument("--steps", type=_positive_int, default=8)
    p.add_argument("--block_length", type=_positive_int, default=32)
    p.add_argument("--cfg_text_scale", type=_non_negative_finite_float, default=4.0)
    p.add_argument("--cfg_image_scale", type=_non_negative_finite_float, default=0.0)
    p.add_argument("--decoder_steps", type=_positive_int, default=50)
    p.add_argument("--resolution_multiplier", type=_positive_int, default=2)
    p.add_argument("--output", type=str, default="edited.png")
    p.add_argument("--seed", type=_non_negative_int, default=42)
    args = p.parse_args()

    args.model_path = _normalize_path(args.model_path)
    if not os.path.isdir(args.model_path):
        p.error("--model_path must be an existing directory")

    if args.image is not None:
        args.image = _normalize_path(args.image)
        if not os.path.isfile(args.image):
            p.error("--image must be an existing file")

    if args.image_token is not None:
        args.image_token = _normalize_path(args.image_token)
        if not os.path.isfile(args.image_token):
            p.error("--image_token must be an existing file")

    if not args.instruction.strip():
        p.error("--instruction must not be empty")

    args.output = _prepare_output_path(args.output)
    return args


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _select_dtype(device):
    if isinstance(device, str) and device.startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported"):
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
        return torch.float16
    return torch.float32


def _flatten_values(value):
    if torch.is_tensor(value):
        detached = value.detach()
        if detached.is_floating_point() or detached.is_complex():
            raise ValueError("token tensors must contain integer values")
        yield from _flatten_values(detached.cpu().tolist())
    elif isinstance(value, dict):
        raise ValueError("nested dictionaries are not supported in token data")
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_values(item)
    else:
        yield value


def _coerce_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} must be an integer")
        try:
            return int(stripped)
        except ValueError:
            raise ValueError(f"{name} must be an integer")
    if isinstance(value, float):
        if value != value or value == float("inf") or value == float("-inf"):
            raise ValueError(f"{name} must be a finite integer")
        if not value.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a single integer value")
        return _coerce_int(value.detach().cpu().item(), name)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")


def _normalize_token_ids(value, name):
    tokens = []
    for index, item in enumerate(_flatten_values(value)):
        token = _coerce_int(item, f"{name}[{index}]")
        if token < 0:
            raise ValueError(f"{name}[{index}] must be greater than or equal to zero")
        tokens.append(token)
    if not tokens:
        raise ValueError(f"{name} must not be empty")
    return tokens


def _parse_grid_thw(value, name):
    values = list(_flatten_values(value))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    t = _coerce_int(values[0], f"{name}[0]")
    h = _coerce_int(values[1], f"{name}[1]")
    w = _coerce_int(values[2], f"{name}[2]")
    if t != 1:
        raise ValueError(f"{name}[0] must be 1")
    if h <= 0 or w <= 0:
        raise ValueError(f"{name} height and width must be greater than zero")
    return t, h, w


def _hw_from_processed_size(processed_size):
    values = list(_flatten_values(processed_size))
    if len(values) != 2:
        raise ValueError("metadata.processed_size must contain exactly two values")
    width = _coerce_int(values[0], "metadata.processed_size[0]")
    height = _coerce_int(values[1], "metadata.processed_size[1]")
    if width <= 0 or height <= 0:
        raise ValueError("metadata.processed_size values must be greater than zero")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("metadata.processed_size values must be divisible by 16")
    return height // 16, width // 16


def _validate_grid_token_count(tokens, h, w, name):
    expected = h * w
    actual = len(tokens)
    if actual != expected:
        raise ValueError(f"{name} token count is {actual}, but image grid {h}x{w} requires {expected}")


def _ensure_image_token_offset(tokens, offset):
    if offset <= 0:
        return list(tokens)
    has_offset_tokens = any(token >= offset for token in tokens)
    has_unoffset_tokens = any(token < offset for token in tokens)
    if has_offset_tokens and has_unoffset_tokens:
        raise ValueError("token ids contain a mixture of offset and unoffset image tokens")
    if has_offset_tokens:
        return list(tokens)
    return [token + offset for token in tokens]


def _strip_image_token_offset(tokens, offset):
    if offset <= 0:
        return list(tokens)
    has_offset_tokens = any(token >= offset for token in tokens)
    has_unoffset_tokens = any(token < offset for token in tokens)
    if has_offset_tokens and has_unoffset_tokens:
        raise ValueError("decoded token ids contain a mixture of offset and unoffset image tokens")
    if has_offset_tokens:
        return [token - offset for token in tokens]
    return list(tokens)


def _get_image_token_offset(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return 157184
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    offset = _coerce_int(config.get("image_token_offset", 157184), "image_token_offset")
    if offset < 0:
        raise ValueError("image_token_offset must be greater than or equal to zero")
    return offset


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_grid_from_pt_data(data):
    metadata = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(metadata, dict):
        if "processed_size" in metadata:
            return _hw_from_processed_size(metadata["processed_size"])
        if "grid_thw" in metadata:
            _, h, w = _parse_grid_thw(metadata["grid_thw"], "metadata.grid_thw")
            return h, w
        if "h" in metadata and "w" in metadata:
            h = _coerce_int(metadata["h"], "metadata.h")
            w = _coerce_int(metadata["w"], "metadata.w")
            if h <= 0 or w <= 0:
                raise ValueError("metadata.h and metadata.w must be greater than zero")
            return h, w

    if isinstance(data, dict):
        if "grid_thw" in data:
            _, h, w = _parse_grid_thw(data["grid_thw"], "grid_thw")
            return h, w
        if "h" in data and "w" in data:
            h = _coerce_int(data["h"], "h")
            w = _coerce_int(data["w"], "w")
            if h <= 0 or w <= 0:
                raise ValueError("h and w must be greater than zero")
            return h, w

    raise KeyError("token file must contain metadata.processed_size, metadata.grid_thw, grid_thw, or h and w")


def encode_image_from_pt(pt_path, offset):
    data = _torch_load(pt_path)
    if not isinstance(data, dict):
        raise TypeError("image token file must contain a dictionary")

    if "semantic_token_ids" in data:
        raw_token_ids = data["semantic_token_ids"]
        token_name = "semantic_token_ids"
    elif "token_ids" in data:
        raw_token_ids = data["token_ids"]
        token_name = "token_ids"
    else:
        raise KeyError("image token file must contain semantic_token_ids or token_ids")

    base_token_ids = _normalize_token_ids(raw_token_ids, token_name)
    token_ids = _ensure_image_token_offset(base_token_ids, offset)
    h, w = _extract_grid_from_pt_data(data)
    _validate_grid_token_count(token_ids, h, w, "source image")
    return token_ids, h, w


def encode_image_from_pil(image_path, model_path, device, offset):
    from encoder.image_tokenizer import ImageTokenizer
    from decoder.utils import generate_crop_size_list, var_center_crop

    image_tokenizer = None
    try:
        dtype = _select_dtype(device)
        image_tokenizer = ImageTokenizer(model_path=model_path, device=device, dtype=dtype)
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)

        with Image.open(image_path) as source_image:
            rgb_image = source_image.convert("RGB")

        pil_image = var_center_crop(rgb_image, crop_size_list=crop_size_list)
        info = image_tokenizer.encode_with_info(pil_image)

        if not isinstance(info, dict):
            raise TypeError("image tokenizer must return a dictionary")

        if "grid_thw" not in info:
            raise KeyError("image tokenizer output must contain grid_thw")

        _, h, w = _parse_grid_thw(info["grid_thw"], "grid_thw")

        if "token_ids" in info:
            raw_token_ids = info["token_ids"]
            token_name = "token_ids"
        elif "semantic_token_ids" in info:
            raw_token_ids = info["semantic_token_ids"]
            token_name = "semantic_token_ids"
        else:
            raise KeyError("image tokenizer output must contain token_ids or semantic_token_ids")

        base_token_ids = _normalize_token_ids(raw_token_ids, token_name)
        token_ids = _ensure_image_token_offset(base_token_ids, offset)
        _validate_grid_token_count(token_ids, h, w, "source image")
        return token_ids, h, w
    finally:
        if image_tokenizer is not None:
            del image_tokenizer
        _clear_memory()


def _load_model(model_path, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}

    model = None
    if isinstance(device, str) and device.startswith("cuda"):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": device}, **kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if "accelerate" in message or "device_map" in message or "dispatch" in message or "bitsandbytes" in message:
                model = None
            else:
                raise

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model = model.to(device=device, dtype=dtype)

    model.eval()
    model.tokenizer = tokenizer

    if not hasattr(model, "edit_image") or not callable(getattr(model, "edit_image", None)):
        raise AttributeError("loaded model does not provide a callable edit_image method")

    return tokenizer, model


def _normalize_edit_result(result, offset):
    if not isinstance(result, dict):
        raise TypeError("edit_image must return a dictionary")

    if "token_ids" in result:
        raw_token_ids = result["token_ids"]
        token_name = "result.token_ids"
    elif "semantic_token_ids" in result:
        raw_token_ids = result["semantic_token_ids"]
        token_name = "result.semantic_token_ids"
    else:
        raise KeyError("edit_image result must contain token_ids or semantic_token_ids")

    token_ids = _normalize_token_ids(raw_token_ids, token_name)
    token_ids = _strip_image_token_offset(token_ids, offset)

    if "h" in result and "w" in result:
        h = _coerce_int(result["h"], "result.h")
        w = _coerce_int(result["w"], "result.w")
        if h <= 0 or w <= 0:
            raise ValueError("result.h and result.w must be greater than zero")
    elif "grid_thw" in result:
        _, h, w = _parse_grid_thw(result["grid_thw"], "result.grid_thw")
    else:
        raise KeyError("edit_image result must contain h and w or grid_thw")

    _validate_grid_token_count(token_ids, h, w, "edited image")
    return token_ids, h, w


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _select_dtype(device)

    offset = _get_image_token_offset(args.model_path)

    if args.image_token is not None:
        print(f"Loading pre-tokenized image: {args.image_token}", flush=True)
        image_tokens, image_h, image_w = encode_image_from_pt(args.image_token, offset)
    else:
        print(f"Encoding image: {args.image}", flush=True)
        image_tokens, image_h, image_w = encode_image_from_pil(args.image, args.model_path, device, offset)

    print(f"Image grid: {image_h}x{image_w}, instruction: {args.instruction}", flush=True)

    print("Loading model...", flush=True)
    tokenizer, model = _load_model(args.model_path, device, dtype)

    result = None
    try:
        with torch.inference_mode():
            result = model.edit_image(
                image_tokens,
                image_h,
                image_w,
                args.instruction,
                steps=args.steps,
                block_length=args.block_length,
                cfg_text_scale=args.cfg_text_scale,
                cfg_image_scale=args.cfg_image_scale,
            )

        result_token_ids, result_h, result_w = _normalize_edit_result(result, offset)
    finally:
        del result
        del model
        del tokenizer
        _clear_memory()

    print("Model unloaded.", flush=True)

    print("Decoding edited image...", flush=True)
    with torch.inference_mode():
        img = decode_vq_tokens(
            result_token_ids,
            result_h,
            result_w,
            args.model_path,
            device,
            resolution_multiplier=args.resolution_multiplier,
            num_steps=args.decoder_steps,
        )

    if not hasattr(img, "save") or not callable(getattr(img, "save", None)):
        raise TypeError("decode_vq_tokens must return an image object with a save method")

    img.save(args.output)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
