"""Module 1 of the pipeline: turn a genome into an executable merge configuration."""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

from zeroai.blocks import partition_tensors
from zeroai.genome import Genome, normalize_weights
from zeroai.merge.mergekit_backend import build_mergekit_yaml
from zeroai.schemas import BlockSpec, DType, MergeMethod, MergePlan, ModelSpec


class CandidateGenerator:
    """Translate explored spatial parameters into backend configurations.

    The generator is the single place that decides how a genome becomes a
    :class:`MergePlan`, so the optimiser, the merge engine and the mergekit YAML
    emitter can never disagree about what a genome means.
    """

    def __init__(
        self,
        base_model: ModelSpec,
        models: Sequence[ModelSpec],
        blocks: Sequence[BlockSpec],
        dtype: DType = DType.FLOAT16,
        seed: int = 1337,
        output_dir: Optional[str] = None,
        force_normalize: Optional[bool] = None,
    ):
        if not models:
            raise ValueError("CandidateGenerator requires at least one input model")
        if not blocks:
            raise ValueError("CandidateGenerator requires at least one block")
        self.base_model = base_model
        self.models = tuple(models)
        self.blocks = tuple(blocks)
        self.dtype = dtype
        self.seed = seed
        self.output_dir = output_dir
        self.force_normalize = force_normalize

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def blocks_from_names(tensor_names: Sequence[str], num_blocks: int) -> Tuple[BlockSpec, ...]:
        return partition_tensors(tensor_names, num_blocks)

    def output_path_for(self, candidate_id: str) -> str:
        if self.output_dir is None:
            return ""
        return os.path.join(self.output_dir, candidate_id)

    # ------------------------------------------------------------------- plan
    def plan(self, genome: Genome, candidate_id: str = "candidate") -> MergePlan:
        if genome.n_models != len(self.models):
            raise ValueError(
                f"genome has {genome.n_models} model rows, generator has {len(self.models)} models"
            )
        if genome.n_blocks != len(self.blocks):
            raise ValueError(
                f"genome has {genome.n_blocks} block columns, generator has {len(self.blocks)} blocks"
            )

        normalize = genome.normalize if self.force_normalize is None else self.force_normalize
        weights = normalize_weights(genome.weights) if normalize else genome.weights

        plan = MergePlan(
            base_model=self.base_model,
            models=self.models,
            method=genome.method,
            weights=tuple(tuple(float(w) for w in row) for row in weights),
            blocks=self.blocks,
            density=genome.density,
            dare_density=genome.dare_density,
            dare_scale=genome.dare_scale,
            normalize=normalize,
            dtype=genome.dtype if genome.dtype is not None else self.dtype,
            seed=self.seed,
            output_path=self.output_path_for(candidate_id),
        )
        plan.validate()
        return plan

    # ------------------------------------------------------------------- yaml
    def yaml(self, genome: Genome, candidate_id: str = "candidate") -> str:
        return build_mergekit_yaml(self.plan(genome, candidate_id))

    def write_yaml(self, genome: Genome, path: str, candidate_id: str = "candidate") -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.yaml(genome, candidate_id))
        return path