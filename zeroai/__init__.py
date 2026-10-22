"""ZeroAI - Evolutionary Model Merging with automatic weight optimisation.

Pipeline (closed loop):

    N input models
        -> CandidateGenerator   (Genome -> MergePlan / mergekit YAML)
        -> MergeBackend         (native torch engine or mergekit driver)
        -> Evaluator            (multi-domain benchmark)
        -> Optimizer            (Optuna TPE / genetic algorithm)
        -> ... repeat until budget exhausted
"""

from zeroai.genome import Genome, normalize_weights
from zeroai.pipelines import EvolutionaryMergePipeline, PipelineConfig, PipelineResult
from zeroai.schemas import (
    ALL_METHODS,
    BlockSpec,
    DType,
    DomainSpec,
    EvaluationReport,
    MergeMethod,
    MergePlan,
    MergedArtifact,
    MetricScore,
    ModelSpec,
    SearchSpace,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Genome",
    "normalize_weights",
    "EvolutionaryMergePipeline",
    "PipelineConfig",
    "PipelineResult",
    "ALL_METHODS",
    "BlockSpec",
    "DType",
    "DomainSpec",
    "EvaluationReport",
    "MergeMethod",
    "MergePlan",
    "MergedArtifact",
    "MetricScore",
    "ModelSpec",
    "SearchSpace",
]