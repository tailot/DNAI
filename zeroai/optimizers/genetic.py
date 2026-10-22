"""Genetic engine: population search over the full genome.

Unlike Optuna, the GA mutates the *merge method gene* itself, so it can escape
local optima where a whole strategy family (e.g. all TIES variants) is trapped.
It is also the fallback engine when optuna is unavailable.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Sequence
from dataclasses import replace

from zeroai.genome import Genome
from zeroai.optimizers.base import OptimizationResult, Optimizer, ScoreFunction
from zeroai.schemas import SearchSpace

LOG = logging.getLogger(__name__)


class GeneticOptimizer(Optimizer):
    name = "genetic"

    def __init__(
        self,
        search_space: SearchSpace,
        seed: int = 1337,
        population_size: int = 24,
        generations: int = 12,
        mutation_rate: float = 0.25,
        mutation_sigma: float = 0.15,
        crossover_rate: float = 0.7,
        method_mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elite_keep: int = 2,
    ):
        super().__init__(search_space, seed)
        if population_size < 4:
            raise ValueError("population_size must be >= 4")
        if generations < 1:
            raise ValueError("generations must be >= 1")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.method_mutation_rate = method_mutation_rate
        self.tournament_size = tournament_size
        self.elite_keep = elite_keep
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ search
    def optimize(
        self,
        score_fn: ScoreFunction,
        n_models: int,
        n_blocks: int,
        initial: Sequence[Genome] = (),
    ) -> OptimizationResult:
        space = self.search_space
        methods = list(space.methods)
        evaluated: Dict[str, float] = {}
        all_fitness: List[float] = []

        def score(genome: Genome) -> float:
            key = genome.key()
            cached = evaluated.get(key)
            if cached is not None:
                return cached
            value = float(score_fn(genome))
            evaluated[key] = value
            all_fitness.append(value)
            return value

        population: List[Genome] = []
        seen = set()
        for genome in initial:
            clipped = genome.clipped(space)
            if clipped.key() not in seen:
                seen.add(clipped.key())
                population.append(clipped)
        while len(population) < self.population_size:
            genome = Genome.sample(n_models, n_blocks, space, self.rng)
            if genome.key() not in seen:
                seen.add(genome.key())
                population.append(genome)

        fitness = [score(g) for g in population]
        best_index = max(range(len(population)), key=lambda i: fitness[i])
        best_genome = population[best_index]
        best_fitness = fitness[best_index]
        history = [best_fitness]
        LOG.info("GA gen 0 | best fitness = %.6f", best_fitness)

        for generation in range(1, self.generations + 1):
            ranked = sorted(zip(population, fitness), key=lambda pair: pair[1], reverse=True)
            next_population: List[Genome] = [g for g, _ in ranked[: self.elite_keep]]

            while len(next_population) < self.population_size:
                parent_a = self._tournament(ranked)
                parent_b = self._tournament(ranked)
                if self.rng.random() < self.crossover_rate:
                    child = parent_a.crossover(parent_b, self.rng)
                else:
                    child = parent_a.clone()
                child = child.mutate(
                    self.rng,
                    rate=self.mutation_rate,
                    sigma=self.mutation_sigma,
                    methods=methods,
                )
                if self.method_mutation_rate > 0 and self.rng.random() < self.method_mutation_rate:
                    # Independent re-draw of the strategy gene: this is what lets
                    # the GA jump out of a basin where one whole method family
                    # (e.g. every TIES variant) is stuck.
                    child = replace(child, method=self.rng.choice(methods))
                child = child.clipped(space)
                if child.key() in {g.key() for g in next_population}:
                    continue
                next_population.append(child)

            population = next_population
            fitness = [score(g) for g in population]
            index = max(range(len(population)), key=lambda i: fitness[i])
            if fitness[index] > best_fitness:
                best_fitness = fitness[index]
                best_genome = population[index]
            history.append(best_fitness)
            LOG.info(
                "GA gen %d | best fitness = %.6f | mean = %.6f | evaluated = %d",
                generation,
                best_fitness,
                sum(fitness) / len(fitness),
                len(evaluated),
            )

        return OptimizationResult(
            best_genome=best_genome.clipped(space),
            best_fitness=float(best_fitness),
            evaluations=len(evaluated),
            optimizer=self.name,
            history=tuple(history),
            all_fitness=tuple(all_fitness),
            extra={"population_size": float(self.population_size), "generations": float(self.generations)},
        )

    def _tournament(self, ranked):
        size = min(self.tournament_size, len(ranked))
        contenders = self.rng.sample(ranked, size)
        return max(contenders, key=lambda pair: pair[1])[0]