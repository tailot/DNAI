"""Generate a sample completion from a checkpoint produced by a ZeroAI run.

    python scripts/test_merged_model.py runs/demo_search/candidates/cand_0000
    python scripts/test_merged_model.py runs/demo_merge --prompt "Spiega le reti neurali" --max-new-tokens 300
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick generation test on a merged checkpoint.")
    parser.add_argument("model_path", help="candidate directory (model.safetensors + config.json)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=200)
    parser.add_argument(
        "--prompt",
        default="Scrivi una funzione Python che restituisce la somma dei quadrati da 1 a n.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise SystemExit(f"not a directory: {args.model_path}")

    import torch
    from transformers import pipeline

    dtype = getattr(torch, args.dtype, torch.bfloat16)
    device_map = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    generator = pipeline(
        "text-generation",
        model=args.model_path,
        device_map=device_map,
        torch_dtype=dtype,
    )
    output = generator(args.prompt, max_new_tokens=args.max_new_tokens)
    print(output[0]["generated_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())