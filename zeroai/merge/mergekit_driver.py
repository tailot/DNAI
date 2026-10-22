"""Driver executed by :class:`MergekitBackend` inside the mergekit interpreter.

Usage::

    python -m zeroai.merge.mergekit_driver --config merge.yaml --out ./merged --device cuda

mergekit is imported lazily so this module can be shipped (and imported) in an
environment where mergekit is not installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ZeroAI -> mergekit driver")
    parser.add_argument("--config", required=True, help="path to the mergekit YAML")
    parser.add_argument("--out", required=True, help="output directory for the merged model")
    parser.add_argument("--device", default="cpu", help="math device for mergekit")
    parser.add_argument("--low-cpu-memory", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--write-model-card", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    try:
        from mergekit.merge import run_merge
        from mergekit.options import MergeOptions
    except Exception as exc:  # mergekit missing or incompatible
        print(
            "zeroai.mergekit_driver: mergekit is not usable in this interpreter: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3

    with open(args.config, "r", encoding="utf-8") as handle:
        raw = handle.read()

    try:
        config = _load_config(raw)
    except Exception as exc:
        print(f"zeroai.mergekit_driver: invalid merge config: {exc}", file=sys.stderr)
        return 4

    options = MergeOptions(
        device=args.device,
        low_cpu_memory=args.low_cpu_memory,
        trust_remote_code=args.trust_remote_code,
        write_model_card=args.write_model_card,
        quiet=args.quiet,
        random_seed=args.seed,
        copy_tokenizer=True,
    )

    started = time.time()
    run_merge(config, out_path=args.out, options=options, config_source=raw)
    elapsed = time.time() - started

    print(
        json.dumps(
            {
                "zeroai_stats": True,
                "merge_seconds": elapsed,
                "mergekit": 1.0,
            }
        )
    )
    return 0


def _load_config(raw: str):
    """Load a mergekit YAML through whichever mergekit layout is installed."""
    try:
        from mergekit.config import MergeConfiguration  # mergekit >= 0.1
    except ImportError:
        from mergekit._options import MergeConfiguration  # mergekit 0.0.x
    try:
        return MergeConfiguration.model_validate(yaml_load(raw))
    except AttributeError:
        return MergeConfiguration.parse_obj(yaml_load(raw))


def yaml_load(raw: str):
    import yaml

    return yaml.safe_load(raw)


if __name__ == "__main__":
    raise SystemExit(main())