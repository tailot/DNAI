"""Command line interface.

    python -m zeroai.cli doctor
    python -m zeroai.cli run  --mode mock --blocks 4 --optimizer optuna --trials 40
    python -m zeroai.cli run  --mode model --base ./base --models ./e1 ./e2 --write-checkpoints
    python -m zeroai.cli yaml --mode mock --models a b --method ties --weights 0.3 0.6 0.8 0.4
    python -m zeroai.cli merge --mode mock --models a b --method dare_ties --density 0.6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Sequence, Tuple

from zeroai.candidate import CandidateGenerator
from zeroai.genome import Genome, genome_from_dict
from zeroai.pipelines import EvolutionaryMergePipeline, PipelineConfig
from zeroai.report import render_summary
from zeroai.schemas import DType, MergeMethod, ModelSpec, SearchSpace


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["mock", "model"], default="mock")
    parser.add_argument("--models", nargs="*", default=[], help="input expert model paths")
    parser.add_argument("--base", default=None, help="base model path (model mode)")
    parser.add_argument("--blocks", type=int, default=4, help="number of per-block coefficient slots")
    parser.add_argument("--dtype", choices=[d.value for d in DType], default="float16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--log-level", default="INFO")


def _add_search(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--optimizer", choices=["optuna", "genetic", "none"], default="optuna")
    parser.add_argument("--trials", type=int, default=40, help="Optuna budget")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--methods", nargs="*", default=None, help="restrict the merge-method gene")
    parser.add_argument("--weight-low", type=float, default=0.0)
    parser.add_argument("--weight-high", type=float, default=1.5)
    parser.add_argument("--density-low", type=float, default=0.05)
    parser.add_argument("--density-high", type=float, default=1.0)
    parser.add_argument("--dare-low", type=float, default=0.0)
    parser.add_argument("--dare-high", type=float, default=0.9)
    parser.add_argument("--genome", default=None, help="JSON genome file to seed or evaluate")


def _add_genome(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", default=None, help="explicit merge method")
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="per-block coefficients: len(blocks) for all models, or blocks*models",
    )
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--dare-density", dest="dare_density", type=float, default=0.0)
    parser.add_argument("--dare-scale", dest="dare_scale", type=float, default=1.0)
    parser.add_argument("--normalize", default=None, action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)


def _add_merge(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--merge-backend", choices=["auto", "native", "mergekit"], default="native")
    parser.add_argument("--mergekit-python", default=None, help="python of the mergekit venv")
    parser.add_argument("--mergekit-device", default="cpu")
    parser.add_argument("--write-checkpoints", action="store_true")


def _search_space(args) -> SearchSpace:
    methods = tuple(MergeMethod(m) for m in args.methods) if args.methods else None
    if args.method and methods is None:
        methods = (MergeMethod(args.method),)
    return SearchSpace(
        methods=methods or SearchSpace().methods,
        weight_low=args.weight_low,
        weight_high=args.weight_high,
        density_low=args.density_low,
        density_high=args.density_high,
        dare_density_low=args.dare_low,
        dare_density_high=args.dare_high,
        dtype=DType(args.dtype),
    )


def _model_specs(args) -> Tuple[ModelSpec, ...]:
    specs: List[ModelSpec] = []
    for index, raw in enumerate(args.models):
        label = os.path.basename(str(raw).rstrip("/\\")) or f"model_{index}"
        specs.append(ModelSpec(path=str(raw), label=label))
    return tuple(specs)


def _explicit_genome(args, n_models: int, n_blocks: int, space: SearchSpace) -> Genome:
    """Build a genome from --method/--weights flags (used by the yaml and merge commands)."""
    flat = list(args.weights or [])
    if not flat:
        return Genome.uniform(n_models, n_blocks).clipped(space)
    if len(flat) == n_blocks:
        rows = [list(flat) for _ in range(n_models)]
    elif len(flat) == n_blocks * n_models:
        rows = [flat[i * n_blocks : (i + 1) * n_blocks] for i in range(n_models)]
    else:
        raise SystemExit(
            f"--weights needs {n_blocks} values (shared by every model) or "
            f"{n_blocks * n_models} values (model-major); got {len(flat)}"
        )
    method = MergeMethod(args.method) if args.method else MergeMethod.LINEAR
    genome = Genome(
        method=method,
        weights=rows,
        density=args.density,
        dare_density=args.dare_density,
        dare_scale=args.dare_scale,
        normalize=True if args.normalize is None else bool(args.normalize),
    )
    return genome.clipped(space)


def _initial_genomes(args, n_models: int = 0, n_blocks: int = 0) -> Tuple[Genome, ...]:
    if getattr(args, "genome", None):
        with open(args.genome, "r", encoding="utf-8") as handle:
            return (genome_from_dict(json.load(handle)),)
    if getattr(args, "weights", None) or getattr(args, "method", None):
        if n_models and n_blocks:
            return (_explicit_genome(args, n_models, n_blocks, _search_space(args)),)
    return ()


def build_config(args, initial: Tuple[Genome, ...] = ()) -> PipelineConfig:
    models = _model_specs(args)
    base = ModelSpec(path=args.base, label="base") if args.base else None
    return PipelineConfig(
        mode=args.mode,
        base_model=base,
        models=models,
        num_blocks=args.blocks,
        dtype=DType(args.dtype),
        seed=args.seed,
        run_dir=args.run_dir,
        run_name=args.run_name,
        search_space=_search_space(args),
        write_checkpoints=getattr(args, "write_checkpoints", False),
        merge_backend=getattr(args, "merge_backend", "auto"),
        mergekit_python=getattr(args, "mergekit_python", None),
        mergekit_device=getattr(args, "mergekit_device", "cpu"),
        device=args.device,
        optimizer=args.optimizer,
        n_trials=args.trials,
        population_size=args.population,
        generations=args.generations,
        initial_genomes=initial,
    )


def cmd_run(args) -> int:
    _configure_logging(args.log_level)
    probe = EvolutionaryMergePipeline(build_config(args))
    _, models, _base, _stores, names = probe.prepare()
    initial = _initial_genomes(args, len(models), args.blocks)
    pipeline = EvolutionaryMergePipeline(build_config(args, initial))
    result = pipeline.run()
    print(render_summary(result))
    print("\nartifacts:")
    for key, path in result.report_paths.items():
        print(f"  {key}: {path}")
    return 0


def cmd_yaml(args) -> int:
    _configure_logging(args.log_level)
    args.optimizer = "none"
    config = build_config(args)
    pipeline = EvolutionaryMergePipeline(config)
    base_model, models, _stores_base, _model_stores, names = pipeline.prepare()
    blocks = pipeline._blocks(names)
    generator = CandidateGenerator(
        base_model=base_model,
        models=models,
        blocks=blocks,
        dtype=config.dtype,
        seed=config.seed,
    )
    initial = _initial_genomes(args, len(models), len(blocks))
    genome = initial[0] if initial else Genome.uniform(len(models), len(blocks))
    genome = genome.clipped(config.search_space)
    print(generator.yaml(genome, candidate_id="manual"))
    return 0


def cmd_merge(args) -> int:
    args.optimizer = "none"
    args.write_checkpoints = True
    args.merge_backend = "native"
    return cmd_run(args)


def cmd_doctor(args) -> int:
    _configure_logging("WARNING")
    from zeroai.merge.mergekit_backend import MergekitBackend

    report = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "accelerate", "safetensors", "optuna", "yaml"):
        try:
            module = __import__(package)
            report[package] = getattr(module, "__version__", "ok")
        except Exception as exc:
            report[package] = f"missing ({type(exc).__name__})"
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except Exception as exc:
        report["cuda_available"] = False
        report["cuda_error"] = str(exc)
    report["mergekit_in_process"] = MergekitBackend.is_available()
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zeroai",
        description="ZeroAI - evolutionary model merging with automatic weight optimisation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the closed evolutionary merge loop")
    _add_common(run)
    _add_search(run)
    _add_genome(run)
    _add_merge(run)
    run.set_defaults(func=cmd_run)

    yaml_cmd = sub.add_parser("yaml", help="emit the mergekit YAML for a genome")
    _add_common(yaml_cmd)
    _add_search(yaml_cmd)
    _add_genome(yaml_cmd)
    yaml_cmd.set_defaults(func=cmd_yaml)

    merge_cmd = sub.add_parser("merge", help="execute one merge, no optimisation")
    _add_common(merge_cmd)
    _add_search(merge_cmd)
    _add_genome(merge_cmd)
    _add_merge(merge_cmd)
    merge_cmd.set_defaults(func=cmd_merge)

    doctor = sub.add_parser("doctor", help="report environment / GPU / mergekit readiness")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())