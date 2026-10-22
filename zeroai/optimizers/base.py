"""Optimizer contract shared by the Optuna and genetic engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from zeroai.genome import Genome
from zeroai.schemas import SearchSpace

ScoreFunction = Callable[[Genome], float]


@dataclass
class OptimizationResult:
    best_genome: Genome
    best_fitness: float
    evaluations: int
    optimizer: str
    history: Tuple[float, ...] = ()
    all_fitness: Tuple[float, ...] = ()
    extra: Dict[str, float] = field(default_factory=dict)


class Optimizer(ABC):
    """Search engine over the spatial parameter space.

    ``score_fn`` must be deterministic and must return a *maximised* scalar.
    """

    name = "base"

    def __init__(self, search_space: SearchSpace, seed: int = 1337):
        self.search_space = search_space
        self.seed = seed

    @abstractmethod
    def optimize(
        self,
        score_fn: ScoreFunction,
        n_models: int,
        n_blocks: int,
        initial: Sequence[Genome] = (),
    ) -> OptimizationResult:
        raise NotImplementedError