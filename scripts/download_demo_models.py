"""Download the three demo checkpoints used by scripts/real_model_demo.ps1.

    python scripts/download_demo_models.py [--dir models] [--force]

The three repos are all Qwen2ForCausalLM with identical tensor names and shapes
(494,032,768 parameters, ~988 MB each, not gated), so they can be merged with
each other: base = Qwen2.5-0.5B, experts = its chat fine-tune and its code
fine-tune.
"""

from __future__ import annotations

import argparse
import os

REPOS = (
    ("base", "Qwen/Qwen2.5-0.5B"),
    ("instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("coder", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the ZeroAI demo models.")
    parser.add_argument("--dir", default="models", help="local directory for the checkpoints")
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    for label, repo in REPOS:
        target = os.path.join(args.dir, label)
        marker = os.path.join(target, "model.safetensors")
        if os.path.isfile(marker) and not args.force:
            print(f"[skip] {repo} -> {target} (already present, {os.path.getsize(marker) / 1e9:.2f} GB)")
            continue
        path = snapshot_download(repo, local_dir=target)
        print(f"[ok]   {repo} -> {path}")

    print(f"\nDemo models ready under {os.path.abspath(args.dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())