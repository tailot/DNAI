"""Native (mergekit-free) merge backend: task vectors, stats and checkpoints."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file

from zeroai.blocks import partition_tensors
from zeroai.candidate import CandidateGenerator
from zeroai.eval.synthetic import build_synthetic_world
from zeroai.genome import Genome
from zeroai.merge.backend import MergeBackend, NativeMergeBackend, merge_plan_from_parts
from zeroai.merge.tensor_store import InMemoryTensorStore, SafetensorsTensorStore
from zeroai.schemas import DType, MergeMethod, ModelSpec


def make_world(num_models=2, num_blocks=2, tensors_per_block=2, seed=17):
    benchmark, base_store, model_stores = build_synthetic_world(
        num_models=num_models,
        num_blocks=num_blocks,
        tensors_per_block=tensors_per_block,
        tensor_shape=(6, 6),
        seed=seed,
    )
    models = tuple(
        ModelSpec(path=f"synthetic://{label}", label=label) for label in benchmark.expert_labels
    )
    blocks = partition_tensors(benchmark.tensor_names, num_blocks)
    base = ModelSpec(path="synthetic://base", label="base")
    return benchmark, base, base_store, model_stores, models, blocks


def make_genome(n_models, n_blocks, method=MergeMethod.LINEAR, weights=None, **kwargs):
    kwargs.setdefault("density", 1.0)
    kwargs.setdefault("dare_density", 0.0)
    kwargs.setdefault("dare_scale", 1.0)
    return Genome(
        method=method,
        weights=weights if weights is not None else [[1.0] * n_blocks for _ in range(n_models)],
        **kwargs,
    )


def make_generator(base, models, blocks, output_dir=None):
    return CandidateGenerator(base_model=base, models=models, blocks=blocks, output_dir=output_dir)


@pytest.fixture()
def world():
    return make_world()


@pytest.fixture()
def plan(world):
    _benchmark, base, _base_store, _stores, models, blocks = world
    # normalize=False keeps the plan weights literal, so expectations are exact.
    return make_generator(base, models, blocks).plan(
        make_genome(len(models), len(blocks), normalize=False), candidate_id="t0"
    )


def backend_for(world, **kwargs):
    _benchmark, _base, base_store, model_stores, _models, _blocks = world
    return NativeMergeBackend(base_store, model_stores, **kwargs)


# ------------------------------------------------------------------ contract
def test_merge_backend_is_abstract():
    with pytest.raises(TypeError):
        MergeBackend()


def test_native_backend_is_a_merge_backend(world):
    backend = backend_for(world)
    assert isinstance(backend, MergeBackend)
    assert backend.name == "native"


# ------------------------------------------------------------------ guards
def test_backend_requires_at_least_one_model_store(world):
    _benchmark, _base, base_store, _stores, _models, _blocks = world
    with pytest.raises(ValueError, match="at least one model store"):
        NativeMergeBackend(base_store, [])


def test_backend_rejects_a_plan_with_the_wrong_model_count(world, plan):
    _benchmark, _base, base_store, model_stores, _models, _blocks = world
    backend = NativeMergeBackend(base_store, model_stores[:1])
    with pytest.raises(ValueError, match="declares 2 models but backend has 1 stores"):
        backend.merge(plan, "cand")


def test_backend_requires_blocks_to_cover_every_tensor(world):
    benchmark, base, base_store, model_stores, models, _blocks = world
    partial = partition_tensors(list(benchmark.tensor_names[:1]), 1)
    broken = make_generator(base, models, partial).plan(
        make_genome(len(models), 1, normalize=False), candidate_id="x"
    )
    backend = NativeMergeBackend(base_store, model_stores)
    with pytest.raises(ValueError, match="does not cover"):
        backend.merge(broken, "x")


def test_backend_rejects_stores_without_shared_tensors(world, plan):
    _benchmark, _base, _base_store, model_stores, _models, _blocks = world
    unrelated = InMemoryTensorStore({"unrelated.weight": torch.zeros(6, 6)})
    with pytest.raises(ValueError, match="no tensor name is shared"):
        NativeMergeBackend(unrelated, model_stores).merge(plan, "cand")


def test_backend_rejects_an_invalid_plan(world, plan):
    broken = replace(plan, density=0.0)
    with pytest.raises(ValueError, match="density must be in"):
        backend_for(world).merge(broken, "x")


# ------------------------------------------------------------------- deltas
def test_merge_returns_a_delta_for_every_common_tensor(world, plan):
    benchmark, _base, _base_store, _stores, _models, _blocks = world
    artifact = backend_for(world).merge(plan, "cand")
    assert set(artifact.deltas) == set(benchmark.tensor_names)
    assert artifact.candidate_id == "cand"
    assert artifact.plan is plan
    assert artifact.output_path is None


def test_linear_merge_is_the_weighted_sum_of_task_vectors(world, plan):
    benchmark, _base, base_store, model_stores, _models, _blocks = world
    artifact = backend_for(world).merge(plan, "cand")
    for name in benchmark.tensor_names:
        expected = sum((store.get(name) - base_store.get(name)) for store in model_stores)
        assert torch.allclose(artifact.deltas[name], expected, atol=1e-5)


def test_weights_are_normalised_per_block_when_requested(world):
    benchmark, base, base_store, model_stores, models, blocks = world
    plan = make_generator(base, models, blocks).plan(
        make_genome(len(models), len(blocks), weights=[[2.0] * len(blocks)] * len(models)),
        candidate_id="norm",
    )
    assert all(abs(sum(column) - 1.0) < 1e-9 for column in zip(*plan.weights))
    artifact = backend_for(world).merge(plan, "cand")
    for name in benchmark.tensor_names:
        expected = 0.5 * sum((store.get(name) - base_store.get(name)) for store in model_stores)
        assert torch.allclose(artifact.deltas[name], expected, atol=1e-5)


def test_per_block_coefficients_silence_a_block(world):
    benchmark, base, base_store, model_stores, models, blocks = world
    flat = make_generator(base, models, blocks).plan(
        make_genome(len(models), len(blocks), weights=[[1.0] * len(blocks)] * len(models)),
        candidate_id="a",
    )
    shaped = make_generator(base, models, blocks).plan(
        make_genome(
            len(models), len(blocks), weights=[[1.0, 0.0] for _ in models], normalize=False
        ),
        candidate_id="b",
    )
    backend = backend_for(world)
    a = backend.merge(flat, "a")
    b = backend.merge(shaped, "b")
    silenced = set(blocks[1].tensor_names)
    assert silenced
    for name in silenced:
        assert torch.allclose(b.deltas[name], torch.zeros_like(b.deltas[name]), atol=1e-6)
    assert any(float(a.deltas[name].abs().sum()) > 0.0 for name in silenced)


def test_keep_deltas_false_skips_the_delta_tensors(world, plan):
    artifact = backend_for(world, keep_deltas=False).merge(plan, "cand")
    assert artifact.deltas is None
    assert artifact.stats


def test_merge_is_reproducible_for_the_same_plan(world, plan):
    benchmark, _base, _base_store, _stores, _models, _blocks = world
    backend = backend_for(world)
    a = backend.merge(plan, "a")
    b = backend.merge(plan, "b")
    for name in benchmark.tensor_names:
        assert torch.equal(a.deltas[name], b.deltas[name])


def test_deltas_are_float32_task_vectors(world, plan):
    artifact = backend_for(world).merge(plan, "cand")
    assert all(tensor.dtype is torch.float32 for tensor in artifact.deltas.values())


# -------------------------------------------------------------------- stats
def test_stats_cover_every_metric_family(world, plan):
    benchmark, _base, _base_store, _stores, models, blocks = world
    stats = backend_for(world).merge(plan, "cand").stats
    for key in (
        "tensors",
        "blocks_used",
        "interference_cross_energy",
        "interference_self_energy",
        "interference_ratio",
        "merged_delta_norm",
        "base_norm",
    ):
        assert key in stats
    assert stats["tensors"] == float(len(benchmark.tensor_names))
    assert stats["blocks_used"] == float(len(blocks))
    assert stats["merged_delta_norm"] > 0.0
    assert stats["base_norm"] > 0.0


def test_stats_are_all_finite(world, plan):
    stats = backend_for(world).merge(plan, "cand").stats
    scalars = [value for value in stats.values() if isinstance(value, (int, float))]
    assert scalars
    assert all(math.isfinite(float(value)) for value in scalars)


def test_interference_ratio_matches_its_components(world, plan):
    stats = backend_for(world).merge(plan, "cand").stats
    expected = stats["interference_cross_energy"] / stats["interference_self_energy"]
    assert stats["interference_ratio"] == pytest.approx(expected)


def test_aligned_experts_show_positive_interference():
    benchmark, base_store, model_stores = build_synthetic_world(
        num_models=2, num_blocks=1, tensors_per_block=1, tensor_shape=(4, 4), seed=3
    )
    names = list(benchmark.tensor_names)
    blocks = partition_tensors(names, 1)
    models = tuple(ModelSpec(path=f"synthetic://{l}", label=l) for l in benchmark.expert_labels)
    base = ModelSpec(path="synthetic://base", label="base")
    plan = make_generator(base, models, blocks).plan(make_genome(2, 1), candidate_id="aligned")
    backend = NativeMergeBackend(base_store, [model_stores[0], model_stores[0]])
    stats = backend.merge(plan, "aligned").stats
    assert stats["interference_cross_energy"] > 0.0
    assert stats["interference_ratio"] > 0.0


def test_conflicting_experts_show_negative_interference():
    benchmark, base_store, model_stores = build_synthetic_world(
        num_models=2, num_blocks=1, tensors_per_block=1, tensor_shape=(4, 4), seed=3
    )
    name = benchmark.tensor_names[0]
    flipped = {n: model_stores[1].get(n).clone() for n in benchmark.tensor_names}
    flipped[name] = -flipped[name]
    stores = [model_stores[0], InMemoryTensorStore(flipped)]
    blocks = partition_tensors(list(benchmark.tensor_names), 1)
    models = tuple(ModelSpec(path=f"synthetic://{l}", label=l) for l in benchmark.expert_labels)
    base = ModelSpec(path="synthetic://base", label="base")
    plan = make_generator(base, models, blocks).plan(make_genome(2, 1), candidate_id="conflict")
    stats = NativeMergeBackend(base_store, stores).merge(plan, "conflict").stats
    assert stats["interference_cross_energy"] < 0.0
    assert stats["interference_ratio"] < 0.0


def test_zero_weights_produce_zero_deltas_and_no_interference(world):
    benchmark, base, base_store, model_stores, models, blocks = world
    plan = make_generator(base, models, blocks).plan(
        make_genome(
            len(models),
            len(blocks),
            weights=[[0.0] * len(blocks) for _ in models],
            normalize=False,
        ),
        candidate_id="zero",
    )
    artifact = NativeMergeBackend(base_store, model_stores).merge(plan, "zero")
    for name in benchmark.tensor_names:
        assert torch.allclose(artifact.deltas[name], torch.zeros_like(artifact.deltas[name]))
    assert artifact.stats["interference_cross_energy"] == pytest.approx(0.0)
    assert artifact.stats["merged_delta_norm"] == pytest.approx(0.0)


# --------------------------------------------------------------- checkpoints
@pytest.fixture()
def base_checkpoint(tmp_path, world):
    _benchmark, _base, base_store, _stores, _models, _blocks = world
    path = tmp_path / "base"
    path.mkdir()
    save_file(
        {name: base_store.get(name) for name in base_store.names()},
        str(path / "model.safetensors"),
        metadata={"format": "pt"},
    )
    (path / "config.json").write_text('{"hidden_size": 6}', encoding="utf-8")
    return str(path)


def checkpoint_plan(world, base_dir, **genome_kwargs):
    """Plan whose base model is the on-disk checkpoint and whose output dir is writable."""
    benchmark, _base, _base_store, _stores, models, blocks = world
    store = SafetensorsTensorStore(base_dir)
    generator = make_generator(
        ModelSpec(path=base_dir, label="base"),
        models,
        blocks,
        output_dir=os.path.join(os.path.dirname(base_dir), "out"),
    )
    plan = generator.plan(
        make_genome(len(models), len(blocks), **genome_kwargs), candidate_id="ckpt"
    )
    return benchmark, store, plan


def test_save_output_writes_a_checkpoint_and_copies_aux_files(world, base_checkpoint):
    _benchmark, store, plan = checkpoint_plan(world, base_checkpoint)
    _b, _base, _base_store, model_stores, _models, _blocks = world
    backend = NativeMergeBackend(store, model_stores, save_output=True)
    artifact = backend.merge(plan, "ckpt")
    assert artifact.output_path == plan.output_path
    assert os.path.isfile(os.path.join(artifact.output_path, "model.safetensors"))
    assert os.path.isfile(os.path.join(artifact.output_path, "config.json"))


def test_checkpoint_is_base_plus_delta(world, base_checkpoint):
    benchmark, store, plan = checkpoint_plan(world, base_checkpoint)
    _b, _base, _base_store, model_stores, _models, _blocks = world
    artifact = NativeMergeBackend(store, model_stores, save_output=True).merge(plan, "ckpt")
    written = load_file(os.path.join(artifact.output_path, "model.safetensors"))
    assert set(written) == set(benchmark.tensor_names)
    for name in benchmark.tensor_names:
        expected = (store.get(name) + artifact.deltas[name]).to(written[name].dtype)
        assert torch.allclose(written[name], expected, atol=1e-3, rtol=1e-3)


def test_checkpoint_dtype_follows_the_plan(world, base_checkpoint):
    _benchmark, store, plan = checkpoint_plan(world, base_checkpoint, dtype=DType.FLOAT16)
    _b, _base, _base_store, model_stores, _models, _blocks = world
    artifact = NativeMergeBackend(store, model_stores, save_output=True).merge(plan, "ckpt")
    written = load_file(os.path.join(artifact.output_path, "model.safetensors"))
    assert all(tensor.dtype is torch.float16 for tensor in written.values())


def test_no_checkpoint_without_save_output(world, base_checkpoint):
    _benchmark, store, plan = checkpoint_plan(world, base_checkpoint)
    _b, _base, _base_store, model_stores, _models, _blocks = world
    artifact = NativeMergeBackend(store, model_stores, save_output=False).merge(plan, "ckpt")
    assert artifact.output_path is None
    assert not os.path.exists(plan.output_path)


def test_no_checkpoint_without_an_output_path(world, plan):
    _benchmark, _base, base_store, model_stores, _models, _blocks = world
    backend = NativeMergeBackend(base_store, model_stores, save_output=True)
    artifact = backend.merge(plan, "cand")
    assert artifact.output_path is None


# ------------------------------------------------------------------- misc
def test_merge_plan_from_parts_helper(world, plan):
    benchmark, _base, base_store, model_stores, _models, _blocks = world
    artifact = merge_plan_from_parts(plan, base_store, model_stores, candidate_id="helper")
    assert artifact.candidate_id == "helper"
    assert set(artifact.deltas) == set(benchmark.tensor_names)


def test_dare_ties_merge_produces_a_delta_for_every_tensor(world):
    benchmark, base, base_store, model_stores, models, blocks = world
    plan = make_generator(base, models, blocks).plan(
        make_genome(
            len(models),
            len(blocks),
            method=MergeMethod.DARE_TIES,
            density=0.8,
            dare_density=0.5,
            dare_scale=1.2,
        ),
        candidate_id="dare",
    )
    artifact = backend_for(world).merge(plan, "dare")
    assert set(artifact.deltas) == set(benchmark.tensor_names)
    assert any(float(tensor.abs().sum()) > 0.0 for tensor in artifact.deltas.values())


def test_every_merge_method_runs_on_the_synthetic_world(world):
    _benchmark, base, _base_store, _stores, models, blocks = world
    for method in MergeMethod:
        plan = make_generator(base, models, blocks).plan(
            make_genome(len(models), len(blocks), method=method), candidate_id=method.value
        )
        artifact = backend_for(world).merge(plan, method.value)
        assert artifact.deltas is not None
        assert math.isfinite(artifact.stats["merged_delta_norm"])