"""Module 5: the orchestrator.

    CandidateGenerator -> MergeBackend -> Evaluator -> Optimizer -> Report

The pipeline owns no optimisation logic of its own: it wires the four modules
together, caches repeated evaluations, keeps every candidate on disk, and
survives a failing candidate (a merge that produces NaN, an OOM during
inference) without aborting the whole search.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field, replace

from zeroai.blocks import partition_tensors
from zeroai.candidate import CandidateGenerator
from zeroai.eval.base import Evaluator
from zeroai.eval.benchmarks import BenchmarkSuite, default_suite
from zeroai.eval.llm import LLMEvaluator
from zeroai.eval.synthetic import (
    DEFAULT_DOMAINS,
    SyntheticBenchmark,
    SyntheticEvaluator,
    build_synthetic_world,
)
from zeroai.genome import Genome
from zeroai.merge.backend import MergeBackend, NativeMergeBackend
from zeroai.merge.mergekit_backend import MergekitBackend
from zeroai.merge.tensor_store import SafetensorsTensorStore, TensorStore, common_tensor_names
from zeroai.optimizers.base import OptimizationResult, Optimizer
from zeroai.optimizers.genetic import GeneticOptimizer
from zeroai.optimizers.optuna_opt import OptunaOptimizer
from zeroai.report import write_report
from zeroai.schemas import (
    BlockSpec,
    DType,
    DomainSpec,
    EvaluationReport,
    MergeMethod,
    MergePlan,
    MergedArtifact,
    ModelSpec,
    SearchSpace,
)

LOG = logging.getLogger(__name__)


@dataclass
class CandidateRecord:
    candidate_id: str
    genome: Genome
    fitness: float
    report: Optional[EvaluationReport] = None
    error: Optional[str] = None
    merge_seconds: float = 0.0
    eval_seconds: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "method": self.genome.method.value,
            "fitness": self.fitness,
            "error": self.error,
            "merge_seconds": self.merge_seconds,
            "eval_seconds": self.eval_seconds,
            "genome": self.genome.to_dict(),
            "metrics": (
                {score.name: score.value for score in self.report.metrics} if self.report else {}
            ),
            "stats": dict(self.stats),
        }


@dataclass
class PipelineResult:
    best_genome: Genome
    best_fitness: float
    candidates: Tuple[CandidateRecord, ...]
    optimization: Optional[OptimizationResult]
    run_dir: str
    mode: str
    blocks: Tuple[BlockSpec, ...]
    domains: Tuple[DomainSpec, ...]
    elapsed_seconds: float
    report_paths: Dict[str, str] = field(default_factory=dict)

    def top(self, n: int = 5) -> List[CandidateRecord]:
        return sorted(
            (c for c in self.candidates if c.error is None),
            key=lambda c: c.fitness,
            reverse=True,
        )[:n]

    def to_dict(self) -> Dict[str, object]:
        """Full machine-readable dump of the run (consumed by ``report.summary.json``)."""
        payload: Dict[str, object] = {
            "mode": self.mode,
            "run_dir": self.run_dir,
            "best_fitness": self.best_fitness,
            "best_genome": self.best_genome.to_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "candidates": [record.to_dict() for record in self.candidates],
            "failures": sum(1 for record in self.candidates if record.error),
            "blocks": [
                {"index": block.index, "name": block.name, "tensors": len(block.tensor_names)}
                for block in self.blocks
            ],
            "domains": [
                {"name": d.name, "weight": d.weight, "description": d.description} for d in self.domains
            ],
            "report_paths": dict(self.report_paths),
        }
        if self.optimization is not None:
            payload["optimization"] = {
                "optimizer": self.optimization.optimizer,
                "evaluations": self.optimization.evaluations,
                "best_fitness": self.optimization.best_fitness,
                "history": list(self.optimization.history),
                "extra": dict(self.optimization.extra),
            }
        return payload

@dataclass
class PipelineConfig:
    # ------------------------------------------------------------ general
    mode: str = "mock"  # "mock" (model-free) | "model" (real checkpoints)
    base_model: Optional[ModelSpec] = None
    models: Tuple[ModelSpec, ...] = ()
    num_blocks: int = 4
    dtype: DType = DType.FLOAT16
    seed: int = 1337
    run_dir: str = "runs"
    run_name: str = "run"
    search_space: SearchSpace = field(default_factory=SearchSpace)
    write_checkpoints: bool = False
    merge_backend: str = "auto"  # "native" | "mergekit" | "auto"
    mergekit_python: Optional[str] = None
    mergekit_device: str = "cpu"
    device: str = "cpu"
    # ------------------------------------------------------------ optimizer
    optimizer: str = "optuna"  # "optuna" | "genetic" | "none"
    n_trials: int = 40
    population_size: int = 20
    generations: int = 8
    initial_genomes: Tuple[Genome, ...] = ()
    # ------------------------------------------------------------- evaluator
    domains: Tuple[DomainSpec, ...] = DEFAULT_DOMAINS
    benchmark_suite: Optional[BenchmarkSuite] = None
    max_new_tokens: int = 256
    # ---------------------------------------------------------------- mock
    synthetic_tensors_per_block: int = 2
    synthetic_tensor_shape: Tuple[int, int] = (32, 32)
    synthetic_noise: float = 0.35

    def __post_init__(self):
        if self.mode not in ("mock", "model"):
            raise ValueError("mode must be 'mock' or 'model'")
        if self.optimizer not in ("optuna", "genetic", "none"):
            raise ValueError("optimizer must be 'optuna', 'genetic' or 'none'")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if self.mode == "model":
            if self.base_model is None or not self.models:
                raise ValueError("model mode needs base_model and models")
        # ``dtype`` is the run-level knob: keep the search space aligned with it
        # so every sampled / clipped genome carries it into its MergePlan and the
        # written checkpoints match --dtype.
        if self.search_space.dtype is not self.dtype:
            self.search_space = replace(self.search_space, dtype=self.dtype)


class EvolutionaryMergePipeline:
    """Wire candidate generation, merging, evaluation and optimisation together."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_dir = os.path.abspath(os.path.join(config.run_dir, config.run_name))
        os.makedirs(self.run_dir, exist_ok=True)
        self._score_cache: Dict[str, float] = {}
        self._records: List[CandidateRecord] = []
        self._counter = 0
        self._benchmark: Optional[SyntheticBenchmark] = None
        self._world = None

    # ------------------------------------------------------------------ setup
    def prepare(self):
        """Resolve the working world: models, tensor stores and block partition."""
        config = self.config
        if config.mode == "mock":
            n_models = len(config.models) or 3
            benchmark, base_store, model_stores = build_synthetic_world(
                num_models=n_models,
                num_blocks=config.num_blocks,
                tensors_per_block=config.synthetic_tensors_per_block,
                tensor_shape=config.synthetic_tensor_shape,
                domains=config.domains,
                seed=config.seed,
                noise_sigma=config.synthetic_noise,
            )
            self._benchmark = benchmark
            labels = (
                [spec.label for spec in config.models]
                if config.models
                else list(benchmark.expert_labels)
            )
            models = tuple(
                ModelSpec(path=f"synthetic://{label}", label=label) for label in labels
            )
            base = ModelSpec(path="synthetic://base", label="base")
            names = benchmark.tensor_names
            return base, models, base_store, model_stores, names
    
        base_store = SafetensorsTensorStore(config.base_model.path, device=config.device)
        model_stores = [SafetensorsTensorStore(m.path, device=config.device) for m in config.models]
        ordered = tuple(common_tensor_names([base_store] + model_stores))
        if not ordered:
            raise ValueError("no tensors shared between base model and input models")
        return config.base_model, tuple(config.models), base_store, model_stores, ordered

    def _blocks(self, tensor_names: Sequence[str]) -> Tuple[BlockSpec, ...]:
        return partition_tensors(tensor_names, self.config.num_blocks)

    def _make_backend(
        self,
        base_store: TensorStore,
        model_stores: Sequence[TensorStore],
    ) -> MergeBackend:
        config = self.config
        if config.merge_backend == "mergekit":
            return MergekitBackend(
                python_executable=config.mergekit_python,
                device=config.mergekit_device,
            )
        if config.merge_backend == "auto" and config.mode == "model":
            if MergekitBackend.is_available(config.mergekit_python):
                LOG.info("mergekit detected - using MergeKitBackend")
                return MergekitBackend(
                    python_executable=config.mergekit_python,
                    device=config.mergekit_device,
                )
            LOG.info("mergekit unavailable - falling back to NativeMergeBackend")
        return NativeMergeBackend(
            base_store=base_store,
            model_stores=model_stores,
            device=config.device,
            keep_deltas=config.mode == "mock" or not config.write_checkpoints,
            save_output=config.write_checkpoints,
        )

    def _make_evaluator(self) -> Evaluator:
        config = self.config
        if config.mode == "mock":
            return SyntheticEvaluator(self._benchmark)
        return LLMEvaluator(
            suite=config.benchmark_suite or default_suite(),
            device=config.device,
            dtype=config.dtype.value,
            max_new_tokens=config.max_new_tokens,
        )

    def _make_optimizer(self) -> Optional[Optimizer]:
        config = self.config
        if config.optimizer == "none":
            return None
        if config.optimizer == "optuna":
            return OptunaOptimizer(
                search_space=config.search_space,
                seed=config.seed,
                n_trials=config.n_trials,
            )
        return GeneticOptimizer(
            search_space=config.search_space,
            seed=config.seed,
            population_size=config.population_size,
            generations=config.generations,
        )

    # -------------------------------------------------------------------- run
    def run(self) -> PipelineResult:
        started = time.time()
        config = self.config
        base_model, models, base_store, model_stores, tensor_names = self.prepare()
        blocks = self._blocks(tensor_names)
        generator = CandidateGenerator(
            base_model=base_model,
            models=models,
            blocks=blocks,
            dtype=config.dtype,
            seed=config.seed,
            output_dir=os.path.join(self.run_dir, "candidates"),
        )
        backend = self._make_backend(base_store, model_stores)
        evaluator = self._make_evaluator()
        optimizer = self._make_optimizer()

        LOG.info(
            "run %s | mode=%s | %d models | %d blocks | %d tensors | backend=%s | evaluator=%s",
            config.run_name,
            config.mode,
            len(models),
            len(blocks),
            len(tensor_names),
            backend.name,
            evaluator.name,
        )

        def score_fn(genome: Genome) -> float:
            return self._score(genome, generator, backend, evaluator)

        optimization: Optional[OptimizationResult] = None
        if optimizer is None:
            genome = config.initial_genomes[0] if config.initial_genomes else Genome.random(
                len(models),
                len(blocks),
                space=config.search_space,
                seed=config.seed,
            )
            best_fitness = score_fn(genome)
            best_genome = genome.clipped(config.search_space)
        else:
            optimization = optimizer.optimize(
                score_fn,
                n_models=len(models),
                n_blocks=len(blocks),
                initial=config.initial_genomes,
            )
            best_genome = optimization.best_genome
            best_fitness = optimization.best_fitness

        evaluator.close()
        elapsed = time.time() - started

        result = PipelineResult(
            best_genome=best_genome,
            best_fitness=best_fitness,
            candidates=tuple(self._records),
            optimization=optimization,
            run_dir=self.run_dir,
            mode=config.mode,
            blocks=blocks,
            domains=tuple(self._benchmark.domains) if self._benchmark else tuple(config.domains),
            elapsed_seconds=elapsed,
        )
        result.report_paths = write_report(result, self.run_dir, generator=generator)
        LOG.info(
            "done | %d candidates | %d failures | best fitness %.6f | %.1fs",
            len(self._records),
            sum(1 for r in self._records if r.error),
            best_fitness,
            elapsed,
        )
        return result

    # ------------------------------------------------------------- evaluation
    def _score(self, genome: Genome, generator, backend, evaluator) -> float:
        key = genome.key()
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached

        self._counter += 1
        candidate_id = f"cand_{self._counter:04d}"
        merge_started = time.time()
        try:
            plan = generator.plan(genome, candidate_id)
            artifact = backend.merge(plan, candidate_id)
        except Exception as exc:
            LOG.warning("merge failed for %s: %s", candidate_id, exc)
            self._records.append(
                CandidateRecord(candidate_id, genome, fitness=0.0, error=f"merge: {exc}")
            )
            self._score_cache[key] = 0.0
            return 0.0
        merge_seconds = time.time() - merge_started

        eval_started = time.time()
        try:
            report = evaluator.evaluate(artifact)
        except Exception as exc:
            LOG.warning("evaluation failed for %s: %s", candidate_id, exc)
            self._records.append(
                CandidateRecord(
                    candidate_id,
                    genome,
                    fitness=0.0,
                    error=f"eval: {exc}",
                    merge_seconds=merge_seconds,
                )
            )
            self._score_cache[key] = 0.0
            return 0.0
        eval_seconds = time.time() - eval_started

        if not report.finite:
            self._records.append(
                CandidateRecord(
                    candidate_id,
                    genome,
                    fitness=0.0,
                    error="non-finite fitness",
                    merge_seconds=merge_seconds,
                    eval_seconds=eval_seconds,
                )
            )
            self._score_cache[key] = 0.0
            return 0.0

        record = CandidateRecord(
            candidate_id=candidate_id,
            genome=genome,
            fitness=report.fitness,
            report=report,
            merge_seconds=merge_seconds,
            eval_seconds=eval_seconds,
            stats=dict(artifact.stats),
        )
        self._records.append(record)
        self._score_cache[key] = report.fitness
        LOG.debug(
            "%s | %s | fitness %.6f | merge %.2fs eval %.2fs",
            candidate_id,
            genome.method.value,
            report.fitness,
            merge_seconds,
            eval_seconds,
        )
        return report.fitness