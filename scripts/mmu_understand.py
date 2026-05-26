"""
LLaDA-2.0-Uni — Image Understanding (Multimodal Understanding)

Usage:
    python mmu_understand.py --model_path /path/to/LLaDA-2.0-Uni --image photo.jpg
    python mmu_understand.py --model_path /path/to/LLaDA-2.0-Uni --image_token photo.pt
    python mmu_understand.py --model_path /path/to/LLaDA-2.0-Uni --image photo.jpg --question "Describe this image."
"""

import os, sys, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="LLaDA-2.0-Uni Image Understanding")
    p.add_argument("--model_path", type=str, required=True,
                   help="Root model dir containing LLM weights and image_tokenizer/")
    p.add_argument("--image", type=str, default=None, help="Path to input image (jpg/png)")
    p.add_argument("--image_token", type=str, default=None, help="Path to pre-tokenized .pt file")
    p.add_argument("--question", type=str, default="", help="Optional question/prefix")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--gen_length", type=int, default=2048)
    return p.parse_args()


def _get_image_token_offset(model_path):
    """Read image_token_offset from model config."""
    import json
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f).get("image_token_offset", 157184)


def encode_image_from_pt(pt_path, offset):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    token_ids = (data["semantic_token_ids"] + offset).tolist()
    w, h = data["metadata"]["processed_size"]
    return token_ids, h // 16, w // 16


def encode_image_from_pil(image_path, model_path, device, offset):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from encoder.image_tokenizer import ImageTokenizer
    from decoder.smart_img_process import smart_resize_images

    image_tokenizer = ImageTokenizer(
        model_path=model_path, device=device, dtype=torch.bfloat16,
    )
    pil_image = smart_resize_images([image_path])[0]
    info = image_tokenizer.encode_with_info(pil_image)
    _, h, w = info["grid_thw"]
    token_ids = [x + offset for x in info["token_ids"]]
    del image_tokenizer; torch.cuda.empty_cache()
    return token_ids, h, w


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode image
    offset = _get_image_token_offset(args.model_path)
    if args.image_token:
        print(f"Loading pre-tokenized image: {args.image_token}")
        image_tokens, image_h, image_w = encode_image_from_pt(args.image_token, offset)
    elif args.image:
        print(f"Encoding image: {args.image}")
        image_tokens, image_h, image_w = encode_image_from_pil(args.image, args.model_path, device, offset)
    else:
        raise ValueError("Provide --image or --image_token")

    print(f"Image grid: {image_h}x{image_w}, tokens: {len(image_tokens)}")

    # Load model and use high-level API
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device, trust_remote_code=True
    ).to(torch.bfloat16).eval()
    model.tokenizer = tokenizer

    print("Generating...")
    response = model.understand_image(
        image_tokens, image_h, image_w,
        question=args.question, steps=args.steps,
        block_length=args.block_length, gen_length=args.gen_length,
    )

    print(f"\n{'='*60}")
    print(f"Question: {args.question or '(none)'}")
    print(f"{'='*60}")
    print(f"Response:\n{response}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
