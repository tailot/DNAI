"""Merge backends.

:class:`NativeMergeBackend` is the always-available engine: it implements
trimming / elect-sign / DARE directly on torch tensors, which means it works on
real safetensors checkpoints *and* on the synthetic tensors used by the mock
mode, with no third-party dependency.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import torch

from zeroai.blocks import build_block_index
from zeroai.merge.algorithms import WORK_DTYPE, combine, interference_energy
from zeroai.merge.tensor_store import TensorStore, common_tensor_names
from zeroai.schemas import MergePlan, MergedArtifact

LOG = logging.getLogger(__name__)


class MergeBackend(ABC):
    """Interface every merge engine must honour."""

    name: str = "base"

    @abstractmethod
    def merge(self, plan: MergePlan, candidate_id: str) -> MergedArtifact:
        """Execute ``plan`` and return a handle on the merged model."""


class NativeMergeBackend(MergeBackend):
    """In-process torch merge engine.

    Args:
        base_store: tensor store of the common base model.
        model_stores: one store per input model, ordered like ``plan.models``.
        device: where the arithmetic happens.
        keep_deltas: keep the merged *task vector* in the artifact so that
            proxy evaluators (synthetic mode) can score it without loading the
            merged checkpoint.
        save_output: write the merged checkpoint to ``plan.output_path``.
    """

    name = "native"

    def __init__(
        self,
        base_store: TensorStore,
        model_stores: Sequence[TensorStore],
        device: str = "cpu",
        keep_deltas: bool = True,
        save_output: bool = False,
    ):
        if not model_stores:
            raise ValueError("NativeMergeBackend requires at least one model store")
        self.base_store = base_store
        self.model_stores = list(model_stores)
        self.device = torch.device(device)
        self.keep_deltas = keep_deltas
        self.save_output = save_output

    # ------------------------------------------------------------------ merge
    def merge(self, plan: MergePlan, candidate_id: str) -> MergedArtifact:
        plan.validate()
        if len(plan.models) != len(self.model_stores):
            raise ValueError(
                f"plan declares {len(plan.models)} models but backend has "
                f"{len(self.model_stores)} stores"
            )

        names = common_tensor_names([self.base_store] + self.model_stores)
        if not names:
            raise ValueError("no tensor name is shared between base model and input models")

        block_of = build_block_index(plan.blocks)
        missing = [name for name in names if name not in block_of]
        if missing:
            raise ValueError(
                f"plan.blocks does not cover {len(missing)} tensor(s), e.g. {missing[:3]}"
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(plan.seed)

        out_dtype = plan.dtype.torch_dtype
        merged_deltas: Dict[str, torch.Tensor] = {}
        out_tensors: Dict[str, torch.Tensor] = {}

        cross_energy = 0.0
        self_energy = 0.0
        delta_norm_sq = 0.0
        base_norm_sq = 0.0
        touched_blocks = set()

        for name in names:
            block = block_of[name]
            touched_blocks.add(block)
            weights = [plan.weights[i][block] for i in range(len(self.model_stores))]

            base_tensor = self.base_store.get(name).to(self.device)
            deltas: List[torch.Tensor] = []
            for store in self.model_stores:
                expert = store.get(name).to(self.device)
                deltas.append((expert - base_tensor).to(WORK_DTYPE))

            contributions = [w * delta for w, delta in zip(weights, deltas)]
            cross, own = interference_energy(contributions)
            cross_energy += cross
            self_energy += own

            merged_delta = combine(
                plan.method,
                deltas,
                weights,
                density=plan.density,
                dare_keep=1.0 - plan.dare_density,
                dare_scale=plan.dare_scale,
                generator=generator,
            )

            delta_norm_sq += float((merged_delta * merged_delta).sum())
            base_norm_sq += float((base_tensor.to(WORK_DTYPE) ** 2).sum())

            merged_deltas[name] = merged_delta
            out_tensors[name] = (base_tensor.to(WORK_DTYPE) + merged_delta).to(out_dtype)

        artifact = MergedArtifact(
            candidate_id=candidate_id,
            plan=plan,
            output_path=None,
            deltas=merged_deltas if self.keep_deltas else None,
            stats={
                "tensors": float(len(names)),
                "blocks_used": float(len(touched_blocks)),
                "interference_cross_energy": cross_energy,
                "interference_self_energy": self_energy,
                "interference_ratio": cross_energy / max(self_energy, 1e-12),
                "merged_delta_norm": delta_norm_sq ** 0.5,
                "base_norm": base_norm_sq ** 0.5,
            },
        )

        if self.save_output and plan.output_path:
            self._write_checkpoint(plan, out_tensors)
            artifact.output_path = plan.output_path

        return artifact

    # ----------------------------------------------------------------- output
    def _write_checkpoint(self, plan: MergePlan, tensors: Dict[str, torch.Tensor]) -> None:
        from safetensors.torch import save_file

        out_dir = plan.output_path
        os.makedirs(out_dir, exist_ok=True)
        save_file(
            {name: tensor.contiguous() for name, tensor in tensors.items()},
            os.path.join(out_dir, "model.safetensors"),
            metadata={"format": "pt"},
        )
        self._copy_aux_files(plan, out_dir)
        LOG.info("merged checkpoint written to %s", out_dir)

    @staticmethod
    def _copy_aux_files(plan: MergePlan, out_dir: str) -> None:
        """Copy config/tokenizer assets from the base model so the output loads."""
        import shutil

        base_dir = plan.base_model.path
        if not os.path.isdir(base_dir):
            return
        for filename in os.listdir(base_dir):
            if filename.endswith(".safetensors"):
                continue
            source = os.path.join(base_dir, filename)
            if os.path.isfile(source):
                shutil.copy2(source, os.path.join(out_dir, filename))


def merge_plan_from_parts(
    plan: MergePlan,
    base_store: TensorStore,
    model_stores: Sequence[TensorStore],
    device: str = "cpu",
    candidate_id: str = "candidate",
    save_output: bool = False,
) -> MergedArtifact:
    """Convenience wrapper used by scripts and tests."""
    backend = NativeMergeBackend(
        base_store, model_stores, device=device, keep_deltas=True, save_output=save_output
    )
    return backend.merge(plan, candidate_id)