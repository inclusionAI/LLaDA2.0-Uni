"""
Deterministic CPU text-to-image smoke test using a compact Diffusers model.

This validates that the local Python environment can run an image generation
pipeline on CPU. It is intentionally separate from the full LLaDA2.0-Uni
checkpoint, which is much larger and may not fit in system memory.

Usage:
    python3 scripts/cpu_image_smoke.py
    python3 scripts/cpu_image_smoke.py --seed 7 --steps 4 --output artifacts/cpu_smoke/seed7.png
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run a deterministic CPU image-generation smoke test.")
    p.add_argument("--model-id", default="segmind/tiny-sd",
                   help="Diffusers model id or local model directory.")
    p.add_argument("--prompt", default=(
        "a centered red apple on a plain wooden table, simple product photo, "
        "clear object, soft daylight"
    ))
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 4))
    p.add_argument("--output", default="artifacts/cpu_smoke/meaningful_cpu_seed1234.png")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--use-safetensors", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _dependency_error(package, exc):
    print(f"Missing or broken dependency: {package} ({exc})", file=sys.stderr)
    print("Install the CPU smoke-test dependencies, then rerun this script:", file=sys.stderr)
    print("  python3 -m venv /tmp/llada_cpu_smoke_venv", file=sys.stderr)
    print("  /tmp/llada_cpu_smoke_venv/bin/pip install --upgrade pip", file=sys.stderr)
    print("  /tmp/llada_cpu_smoke_venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision", file=sys.stderr)
    print("  /tmp/llada_cpu_smoke_venv/bin/pip install diffusers transformers accelerate safetensors Pillow", file=sys.stderr)
    raise SystemExit(2)


def _load_dependencies():
    try:
        import torch
    except Exception as exc:
        _dependency_error("torch", exc)

    try:
        from diffusers import DiffusionPipeline
    except Exception as exc:
        _dependency_error("diffusers", exc)

    return torch, DiffusionPipeline


def _configure_determinism(torch, seed, threads, deterministic):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    torch, DiffusionPipeline = _load_dependencies()
    _configure_determinism(torch, args.seed, args.threads, args.deterministic)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
        use_safetensors=args.use_safetensors,
    )
    pipe = pipe.to("cpu")
    pipe.set_progress_bar_config(disable=True)

    if getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None
    if hasattr(pipe, "register_to_config"):
        pipe.register_to_config(requires_safety_checker=False)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    call_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "output_type": "pil",
    }
    if args.negative_prompt is not None:
        call_kwargs["negative_prompt"] = args.negative_prompt

    image = pipe(**call_kwargs).images[0]
    image.save(output)

    manifest = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "threads": max(1, args.threads),
        "local_files_only": args.local_files_only,
        "use_safetensors": args.use_safetensors,
        "deterministic": args.deterministic,
        "device": "cpu",
        "torch_version": torch.__version__,
        "elapsed_seconds": round(time.time() - started, 3),
        "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "png_sha256": _sha256_file(output),
        "output": str(output),
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Saved image: {output}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Pixel SHA256: {manifest['pixel_sha256']}")
    print(f"PNG SHA256: {manifest['png_sha256']}")


if __name__ == "__main__":
    main()
