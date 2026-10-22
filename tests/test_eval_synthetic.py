"""Synthetic (mock-mode) benchmark world and its proxy evaluator."""

from __future__ import annotations

import pytest
import torch

from zeroai.blocks import partition_tensors
from zeroai.candidate import CandidateGenerator
from zeroai.eval.synthetic import (
    DEFAULT_DOMAINS,
    SyntheticEvaluator,
    build_synthetic_world,
)
from zeroai.genome import Genome
from zeroai.merge.backend import NativeMergeBackend
from zeroai.schemas import DType, MergeMethod, MergedArtifact, ModelSpec


def make_world(num_models=3, num_blocks=3, tensors_per_block=2, seed=23, **kwargs):
    return build_synthetic_world(
        num_models=num_models,
        num_blocks=num_blocks,
        tensors_per_block=tensors_per_block,
        tensor_shape=(8, 8),
        seed=seed,
        **kwargs,
    )


@pytest.fixture()
def world():
    return make_world()


@pytest.fixture()
def benchmark(world):
    return world[0]


def make_generator(benchmark, dtype=DType.FLOAT32):
    models = tuple(
        ModelSpec(path=f"synthetic://{label}", label=label)
        for label in benchmark.expert_labels
    )
    return CandidateGenerator(
        base_model=ModelSpec(path="synthetic://base", label="base"),
        models=models,
        blocks=partition_tensors(benchmark.tensor_names, benchmark.num_blocks),
        dtype=dtype,
        seed=3,
    )


def make_genome(benchmark, weights=None, **kwargs):
    kwargs.setdefault("method", MergeMethod.LINEAR)
    kwargs.setdefault("density", 1.0)
    kwargs.setdefault("dare_density", 0.0)
    kwargs.setdefault("dare_scale", 1.0)
    kwargs.setdefault("normalize", False)
    return Genome(
        weights=weights
        if weights is not None
        else [[1.0] * benchmark.num_blocks for _ in benchmark.expert_labels],
        **kwargs,
    )


def merged(benchmark, genome, **kwargs):
    generator = make_generator(benchmark)
    plan = generator.plan(genome, candidate_id="eval")
    base_store, model_stores = world_stores(benchmark)
    backend = NativeMergeBackend(base_store, model_stores, **kwargs)
    return backend.merge(plan, "eval")


def world_stores(benchmark):
    """Rebuild the stores for a benchmark produced by :func:`make_world`."""
    rebuilt, base_store, model_stores = build_synthetic_world(
        num_models=len(benchmark.expert_labels),
        num_blocks=benchmark.num_blocks,
        tensors_per_block=benchmark.tensors_per_block,
        tensor_shape=benchmark.tensor_shape,
        seed=benchmark.seed,
        noise_sigma=benchmark.noise_sigma,
        expert_labels=list(benchmark.expert_labels),
    )
    assert rebuilt.expert_labels == benchmark.expert_labels
    return base_store, model_stores


# ---------------------------------------------------------------- world builder
def test_world_is_reproducible_for_a_given_seed():
    first, _first_base, first_models = make_world(seed=99)
    second, _second_base, second_models = make_world(seed=99)
    other, _other_base, _other_models = make_world(seed=100)

    for name in first.tensor_names:
        assert torch.equal(first.base[name], second.base[name])
        assert torch.equal(first.expert_deltas["expert_0"][name], second.expert_deltas["expert_0"][name])
        assert not torch.equal(first.expert_deltas["expert_0"][name], other.expert_deltas["expert_0"][name])


def test_world_rejects_degenerate_geometry():
    with pytest.raises(ValueError, match="must all be >= 1"):
        build_synthetic_world(num_models=0, num_blocks=2)
    with pytest.raises(ValueError, match="must all be >= 1"):
        build_synthetic_world(num_models=2, num_blocks=0)


def test_world_rejects_a_label_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="exactly num_models entries"):
        build_synthetic_world(num_models=3, num_blocks=2, expert_labels=["a", "b"])


def test_world_accepts_custom_expert_labels():
    benchmark, _base, stores = build_synthetic_world(
        num_models=2, num_blocks=2, expert_labels=["alpha", "beta"]
    )
    assert benchmark.expert_labels == ("alpha", "beta")
    assert len(stores) == 2


def test_model_stores_hold_base_plus_expert_delta(world):
    benchmark, base_store, model_stores = world
    for label, store in zip(benchmark.expert_labels, model_stores):
        for name in benchmark.tensor_names:
            expected = base_store.get(name) + benchmark.expert_deltas[label][name]
            assert torch.allclose(store.get(name), expected)


def test_block_names_group_tensors_by_depth(benchmark):
    grouped = benchmark.block_names()
    assert sorted(grouped) == list(range(benchmark.num_blocks))
    assert grouped[0] == ("block_0.t0", "block_0.t1")
    assert sum(len(v) for v in grouped.values()) == len(benchmark.tensor_names)


def test_domain_depth_profiles_are_normalised(benchmark):
    for domain in benchmark.domains:
        profile = benchmark.domain_depth[domain.name]
        assert len(profile) == benchmark.num_blocks
        assert sum(profile) == pytest.approx(1.0)
        assert all(value > 0.0 for value in profile)


def test_every_expert_declares_coverage_for_every_domain(benchmark):
    names = {domain.name for domain in benchmark.domains}
    for label in benchmark.expert_labels:
        assert set(benchmark.expert_coverage[label]) == names
        assert all(0.0 < v <= 1.0 for v in benchmark.expert_coverage[label].values())


def test_default_domains_are_weighted_to_one():
    assert sum(d.weight for d in DEFAULT_DOMAINS) == pytest.approx(1.0)


# ---------------------------------------------------------------- evaluator
def test_evaluate_returns_every_declared_metric(benchmark):
    artifact = merged(benchmark, make_genome(benchmark))
    report = SyntheticEvaluator(benchmark).evaluate(artifact)

    assert report.candidate_id == "eval"
    assert report.notes["evaluator"] == "synthetic"
    assert report.finite

    names = {score.name for score in report.metrics}
    expected = {"capability", "purity", "stability", "cooperation"}
    expected |= {f"domain_{d.name}" for d in benchmark.domains}
    assert names == expected

    for score in report.metrics:
        assert 0.0 <= score.value <= 1.0

    domain_weights = {
        f"domain_{d.name}": d.weight for d in benchmark.domains
    }
    total = sum(d.weight for d in benchmark.domains)
    for score in report.metrics:
        if score.name in domain_weights:
            assert score.weight == pytest.approx(domain_weights[score.name] / total)


def test_fitness_is_the_weighted_blend_of_the_four_headline_metrics(benchmark):
    report = SyntheticEvaluator(benchmark).evaluate(merged(benchmark, make_genome(benchmark)))
    weights = benchmark.fitness_weights
    expected = sum(weights[name] * report.metric(name).value for name in weights)
    assert report.fitness == pytest.approx(expected)


def test_evaluate_needs_the_deltas_from_an_in_memory_backend(benchmark):
    generator = make_generator(benchmark)
    plan = generator.plan(make_genome(benchmark), candidate_id="no-deltas")
    artifact = MergedArtifact(candidate_id="no-deltas", plan=plan, deltas=None)
    with pytest.raises(ValueError, match="needs artifact.deltas"):
        SyntheticEvaluator(benchmark).evaluate(artifact)


def test_a_candidate_that_moves_nothing_scores_nothing(benchmark):
    generator = make_generator(benchmark)
    plan = generator.plan(make_genome(benchmark), candidate_id="frozen")
    frozen = {
        name: torch.zeros_like(benchmark.base[name]) for name in benchmark.tensor_names
    }
    artifact = MergedArtifact(candidate_id="frozen", plan=plan, deltas=frozen)
    report = SyntheticEvaluator(benchmark).evaluate(artifact)

    for name in ("capability", "purity", "stability"):
        assert report.metric(name).value == pytest.approx(0.0)


def test_a_pure_domain_delta_scores_that_domain_perfectly(benchmark):
    generator = make_generator(benchmark)
    plan = generator.plan(make_genome(benchmark), candidate_id="pure")
    artifact = MergedArtifact(
        candidate_id="pure",
        plan=plan,
        deltas={name: benchmark.truth["reasoning"][name] for name in benchmark.tensor_names},
    )
    report = SyntheticEvaluator(benchmark).evaluate(artifact)

    # The merged vector *is* the reasoning ground truth, so every block cosine is 1.
    assert report.metric("domain_reasoning").value == pytest.approx(1.0)
    assert report.metric("purity").value == pytest.approx(1.0, abs=1e-6)
    # ... but it says nothing about the other capabilities.
    assert report.metric("domain_code").value < 0.6


def test_a_truth_aligned_blend_scores_high_purity_and_perfect_stability(benchmark):
    total_weight = sum(d.weight for d in benchmark.domains)
    aligned = {
        name: sum(
            domain.weight * benchmark.truth[domain.name][name] for domain in benchmark.domains
        )
        / total_weight
        for name in benchmark.tensor_names
    }
    generator = make_generator(benchmark)
    plan = generator.plan(make_genome(benchmark), candidate_id="oracle")
    artifact = MergedArtifact(candidate_id="oracle", plan=plan, deltas=aligned)
    report = SyntheticEvaluator(benchmark).evaluate(artifact)

    # Capabilities overlap in weight space, so a blend can never align perfectly
    # with every ground truth at once; purity and stability can.
    assert report.metric("capability").value > 0.6
    assert report.metric("purity").value > 0.9
    assert report.metric("stability").value == pytest.approx(1.0, abs=1e-6)


def test_cooperative_weights_beat_self_cancelling_weights(benchmark):
    evaluator = SyntheticEvaluator(benchmark)
    n_models, n_blocks = len(benchmark.expert_labels), benchmark.num_blocks
    assert n_models == 3

    aligned = merged(
        benchmark,
        make_genome(benchmark, weights=[[1.0] * n_blocks for _ in range(n_models)]),
    )
    cancelling = merged(
        benchmark,
        make_genome(
            benchmark,
            weights=[[1.0] * n_blocks, [-1.0] * n_blocks, [-1.0] * n_blocks],
        ),
    )

    aligned_score = evaluator.evaluate(aligned)
    cancelling_score = evaluator.evaluate(cancelling)
    assert aligned_score.fitness > cancelling_score.fitness
    assert aligned_score.metric("capability").value > cancelling_score.metric("capability").value


def test_cooperation_metric_penalises_conflicting_contributions(benchmark):
    n_blocks = benchmark.num_blocks
    conflicting = merged(
        benchmark,
        make_genome(benchmark, weights=[[1.0] * n_blocks, [-1.0] * n_blocks, [1.0] * n_blocks]),
    )
    strict = SyntheticEvaluator(benchmark).evaluate(conflicting).metric("cooperation").value
    lenient = SyntheticEvaluator(benchmark, penalize_conflict=False).evaluate(conflicting)

    assert lenient.metric("cooperation").value == pytest.approx(1.0)
    assert strict <= 1.0


def test_trim_and_dare_genes_change_the_scored_delta(benchmark):
    dense = merged(benchmark, make_genome(benchmark, density=1.0))
    trimmed = merged(benchmark, make_genome(benchmark, method=MergeMethod.TASKSLIM, density=0.4))
    assert dense.deltas is not None and trimmed.deltas is not None
    assert any(
        not torch.allclose(dense.deltas[name], trimmed.deltas[name])
        for name in benchmark.tensor_names
    )


def test_evaluator_close_is_safe_to_call(benchmark):
    evaluator = SyntheticEvaluator(benchmark)
    assert evaluator.close() is None
    assert evaluator.name == "synthetic"


def test_genome_defaults_helper_matches_genome_contract(benchmark):
    genome = make_genome(benchmark)
    assert genome.n_models == len(benchmark.expert_labels)
    assert genome.n_blocks == benchmark.num_blocks
    assert genome.normalize is False