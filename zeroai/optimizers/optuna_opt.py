"""Optuna engine: Bayesian exploration of the spatial parameter space.

Optuna is the primary engine (spec section 3.4).  It handles the mixed
categorical/continuous space natively - merge method, per-block coefficients,
density, DARE parameters - and its pruner lets the pipeline abort a candidate
mid-merge when its proxy signal is already hopeless.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from zeroai.genome import Genome
from zeroai.optimizers.base import OptimizationResult, Optimizer, ScoreFunction
from zeroai.schemas import MergeMethod, SearchSpace

LOG = logging.getLogger(__name__)


def suggest_genome(
    trial,
    search_space: SearchSpace,
    n_models: int,
    n_blocks: int,
    method_names: Optional[Sequence[str]] = None,
) -> Genome:
    """Build one genome from an Optuna trial.  Also usable as a random sampler."""
    methods = list(method_names or [m.value for m in search_space.methods])
    if not methods:
        raise ValueError("search space contains no merge methods")

    method = MergeMethod(trial.suggest_categorical("method", methods))
    weights: List[List[float]] = []
    for model_index in range(n_models):
        row = [
            trial.suggest_float(
                f"weight_m{model_index}_b{block_index}",
                search_space.weight_low,
                search_space.weight_high,
            )
            for block_index in range(n_blocks)
        ]
        weights.append(row)

    genome = Genome(
        method=method,
        weights=weights,
        density=trial.suggest_float("density", search_space.density_low, search_space.density_high),
        dare_density=trial.suggest_float(
            "dare_density", search_space.dare_density_low, search_space.dare_density_high
        ),
        dare_scale=trial.suggest_float(
            "dare_scale", search_space.dare_scale_low, search_space.dare_scale_high
        ),
        normalize=trial.suggest_categorical("normalize", [True, False]),
    )
    return genome.clipped(search_space)


class OptunaOptimizer(Optimizer):
    name = "optuna"

    def __init__(
        self,
        search_space: SearchSpace,
        seed: int = 1337,
        n_trials: int = 60,
        timeout_seconds: Optional[int] = None,
        prune: bool = True,
        storage: Optional[str] = None,
    ):
        super().__init__(search_space, seed)
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.prune = prune
        self.storage = storage

    def optimize(
        self,
        score_fn: ScoreFunction,
        n_models: int,
        n_blocks: int,
        initial: Sequence[Genome] = (),
    ) -> OptimizationResult:
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "optuna is required for OptunaOptimizer - pip install optuna "
                "or use GeneticOptimizer"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=min(10, self.n_trials)
        )
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5) if self.prune else None
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=bool(self.storage),
        )

        method_names = [m.value for m in self.search_space.methods]
        history: List[float] = []
        best: Tuple[Optional[Genome], float] = (None, float("-inf"))

        def objective(trial):
            nonlocal best
            genome = suggest_genome(trial, self.search_space, n_models, n_blocks, method_names)
            fitness = float(score_fn(genome))
            history.append(fitness)
            if fitness > best[1]:
                best = (genome, fitness)
            return fitness

        # Seed the search with the user's / GA's starting candidates so the TPE
        # model starts from a measured region instead of from scratch.
        for genome in initial:
            study.enqueue_trial(
                _flatten_genome(genome.clipped(self.search_space), n_models, n_blocks),
                skip_if_exists=False,
            )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        if not history or best[0] is None:
            raise RuntimeError("optuna finished without evaluating a single trial")

        best_genome, best_fitness = best
        LOG.info(
            "optuna finished | %d trials | best fitness %.6f | %s",
            len(history),
            best_fitness,
            best_genome.method.value,
        )
        return OptimizationResult(
            best_genome=best_genome.clipped(self.search_space),
            best_fitness=float(best_fitness),
            evaluations=len(history),
            optimizer=self.name,
            history=tuple(history),
            all_fitness=tuple(history),
            extra={"n_trials": float(len(history))},
        )


def _flatten_genome(genome: Genome, n_models: int, n_blocks: int) -> dict:
    """Serialise a genome into the Optuna parameter names used by :func:`suggest_genome`."""
    params = {"method": genome.method.value}
    for model_index in range(n_models):
        for block_index in range(n_blocks):
            params[f"weight_m{model_index}_b{block_index}"] = float(
                genome.weights[model_index][block_index]
            )
    params["density"] = genome.density
    params["dare_density"] = genome.dare_density
    params["dare_scale"] = genome.dare_scale
    params["normalize"] = bool(genome.normalize)
    return params


def genome_from_trial_values(
    params: dict,
    search_space: SearchSpace,
    n_models: int,
    n_blocks: int,
    method_names: Optional[Sequence[str]] = None,
) -> Genome:
    """Rebuild a genome from stored Optuna trial parameters."""
    methods = list(method_names or [m.value for m in search_space.methods])
    mid_weight = 0.5 * (search_space.weight_low + search_space.weight_high)
    weights = [
        [float(params.get(f"weight_m{m}_b{b}", mid_weight)) for b in range(n_blocks)]
        for m in range(n_models)
    ]
    return Genome(
        method=MergeMethod(str(params.get("method", methods[0]))),
        weights=weights,
        density=float(params.get("density", search_space.density_high)),
        dare_density=float(params.get("dare_density", search_space.dare_density_low)),
        dare_scale=float(params.get("dare_scale", search_space.dare_scale_low)),
        normalize=bool(params.get("normalize", True)),
    ).clipped(search_space)