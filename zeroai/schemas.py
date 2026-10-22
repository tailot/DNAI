"""Core data contracts shared by every ZeroAI module.

Everything that crosses a module boundary (genomes, merge plans, evaluation
reports) is defined here so that the merge backend, the evaluator and the
optimiser stay completely decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import math


class MergeMethod(str, Enum):
    """Merge strategies supported by both the native and the mergekit backend."""

    LINEAR = "linear"
    TASKSLIM = "taskslim"
    TIES = "ties"
    DARE_LINEAR = "dare_linear"
    DARE_TIES = "dare_ties"

    @property
    def uses_trim(self) -> bool:
        """True when the method magnitude-trims task vectors (``density`` gene)."""
        return self in (MergeMethod.TASKSLIM, MergeMethod.TIES, MergeMethod.DARE_TIES)

    @property
    def uses_sign_elect(self) -> bool:
        """True when the method resolves sign conflicts by consensus."""
        return self in (MergeMethod.TIES, MergeMethod.DARE_TIES)

    @property
    def uses_dare(self) -> bool:
        """True when the method applies Drop-And-REscale stochastic pruning."""
        return self in (MergeMethod.DARE_LINEAR, MergeMethod.DARE_TIES)

    @property
    def mergekit_method(self) -> str:
        """Name understood by the mergekit YAML ``merge_method`` key."""
        return _MERGEKIT_METHODS[self]


_MERGEKIT_METHODS: Dict[MergeMethod, str] = {
    MergeMethod.LINEAR: "linear",
    MergeMethod.TASKSLIM: "taskslim",
    MergeMethod.TIES: "ties",
    MergeMethod.DARE_LINEAR: "dare_linear",
    MergeMethod.DARE_TIES: "dare_ties",
}

ALL_METHODS: Tuple[MergeMethod, ...] = tuple(MergeMethod)


class DType(str, Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"

    @property
    def torch_dtype(self):  # -> torch.dtype
        import torch

        return {
            DType.FLOAT16: torch.float16,
            DType.BFLOAT16: torch.bfloat16,
            DType.FLOAT32: torch.float32,
        }[self]


@dataclass(frozen=True)
class DomainSpec:
    """A benchmark domain (reasoning, code, general/safety) and its weight."""

    name: str
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class MetricScore:
    """A single scalar metric produced by the evaluator."""

    name: str
    value: float
    weight: float = 1.0
    details: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    """Outcome of evaluating one merged candidate."""

    fitness: float
    metrics: Tuple[MetricScore, ...]
    candidate_id: str = ""
    notes: Mapping[str, str] = field(default_factory=dict)

    def metric(self, name: str) -> Optional[MetricScore]:
        for score in self.metrics:
            if score.name == name:
                return score
        return None

    @property
    def finite(self) -> bool:
        """False when the evaluation produced NaN/inf - such candidates are discarded."""
        if not math.isfinite(self.fitness):
            return False
        return all(math.isfinite(score.value) for score in self.metrics)

    def as_dict(self) -> Dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "fitness": self.fitness,
            "metrics": {s.name: s.value for s in self.metrics},
            "metric_weights": {s.name: s.weight for s in self.metrics},
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class ModelSpec:
    """One input model participating in the merge."""

    path: str
    label: str = ""
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("ModelSpec.path must be a non-empty string")
        if not self.label:
            object.__setattr__(self, "label", self.path.rstrip("/\\").split("/")[-1])


@dataclass(frozen=True)
class BlockSpec:
    """A contiguous group of layers treated as one weight-coefficient slot."""

    index: int
    name: str
    tensor_names: Tuple[str, ...]


@dataclass(frozen=True)
class MergePlan:
    """Backend-agnostic description of a merge to execute."""

    base_model: ModelSpec
    models: Tuple[ModelSpec, ...]
    method: MergeMethod
    weights: Tuple[Tuple[float, ...], ...]  # [model][block]
    blocks: Tuple[BlockSpec, ...]
    density: float
    dare_density: float
    dare_scale: float
    normalize: bool
    dtype: DType
    seed: int = 1337
    output_path: str = ""

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def validate(self) -> None:
        if not self.models:
            raise ValueError("MergePlan requires at least one model")
        if len(self.weights) != len(self.models):
            raise ValueError("weights must have one row per model")
        for row in self.weights:
            if len(row) != len(self.blocks):
                raise ValueError("every weight row must have one value per block")
        if not 0.0 < self.density <= 1.0:
            raise ValueError("density must be in (0, 1]")
        if not 0.0 <= self.dare_density < 1.0:
            raise ValueError("dare_density must be in [0, 1)")
        if self.dare_scale <= 0.0:
            raise ValueError("dare_scale must be > 0")


@dataclass
class MergedArtifact:
    """Handle on a merged model produced by a backend."""

    candidate_id: str
    plan: MergePlan
    output_path: Optional[str] = None
    #: Only populated by in-memory backends (native/synthetic mode).
    deltas: Optional[Dict[str, "object"]] = None
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchSpace:
    """Bounds of the evolutionary chromosome."""

    weight_low: float = 0.0
    weight_high: float = 1.5
    density_low: float = 0.05
    density_high: float = 1.0
    dare_density_low: float = 0.0
    dare_density_high: float = 0.9
    dare_scale_low: float = 1.0
    dare_scale_high: float = 3.0
    methods: Tuple[MergeMethod, ...] = ALL_METHODS

    def __post_init__(self) -> None:
        if self.weight_low > self.weight_high:
            raise ValueError("weight_low must be <= weight_high")
        if not 0.0 < self.density_low <= self.density_high <= 1.0:
            raise ValueError("density bounds must satisfy 0 < low <= high <= 1")
        if not 0.0 <= self.dare_density_low < self.dare_density_high < 1.0:
            raise ValueError("dare_density bounds must satisfy 0 <= low < high < 1")
        if self.dare_scale_low > self.dare_scale_high:
            raise ValueError("dare_scale_low must be <= dare_scale_high")
        if not self.methods:
            raise ValueError("SearchSpace.methods must not be empty")