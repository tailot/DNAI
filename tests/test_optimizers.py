"""Search engines: genetic algorithm, Optuna TPE, and the shared contract."""

from __future__ import annotations

import math
import sys

import pytest

from zeroai.genome import Genome
from zeroai.optimizers.base import OptimizationResult, Optimizer
from zeroai.optimizers.genetic import GeneticOptimizer
from zeroai.optimizers.optuna_opt import (
    OptunaOptimizer,
    _flatten_genome,
    genome_from_trial_values,
)
from zeroai.schemas import ALL_METHODS, MergeMethod, SearchSpace

N_MODELS = 3
N_BLOCKS = 2


def target_weights(n_models=N_MODELS, n_blocks=N_BLOCKS):
    return [[0.9 - 0.2 * block for block in range(n_blocks)] for _ in range(n_models)]


def make_scorer(target=None, preferred=MergeMethod.DARE_TIES):
    """Deterministic fitness landscape: closeness to ``target`` plus a method bonus."""
    target = target or target_weights()
    scored: list = []

    def score(genome: Genome) -> float:
        assert len(genome.weights) == len(target)
        error = sum(
            abs(genome.weights[model][block] - target[model][block])
            for model in range(len(target))
            for block in range(len(target[0]))
        )
        total = len(target) * len(target[0])
        bonus = 0.25 if genome.method is preferred else 0.0
        value = max(0.0, 1.0 - error / total) + bonus
        scored.append(genome)
        return value

    return score, scored


def make_genome(**kwargs):
    kwargs.setdefault("method", MergeMethod.TIES)
    kwargs.setdefault("weights", target_weights())
    kwargs.setdefault("density", 0.6)
    kwargs.setdefault("dare_density", 0.2)
    kwargs.setdefault("dare_scale", 1.5)
    return Genome(**kwargs)


# ------------------------------------------------------------------- contract
def test_optimizer_is_abstract():
    with pytest.raises(TypeError):
        Optimizer(SearchSpace(), seed=0)


def test_optimization_result_defaults():
    result = OptimizationResult(
        best_genome=make_genome(), best_fitness=0.5, evaluations=1, optimizer="stub"
    )
    assert result.history == ()
    assert result.all_fitness == ()
    assert result.extra == {}


# --------------------------------------------------------------------- genetic
def test_genetic_rejects_degenerate_settings():
    with pytest.raises(ValueError, match="population_size"):
        GeneticOptimizer(SearchSpace(), population_size=3)
    with pytest.raises(ValueError, match="generations"):
        GeneticOptimizer(SearchSpace(), generations=0)


def test_genetic_is_reproducible_for_a_given_seed():
    def run():
        score, _ = make_scorer()
        return GeneticOptimizer(SearchSpace(), seed=7, population_size=8, generations=4).optimize(
            score, N_MODELS, N_BLOCKS
        )

    first, second = run(), run()
    assert first.best_genome.key() == second.best_genome.key()
    assert first.history == second.history
    assert first.evaluations == second.evaluations


def test_genetic_history_is_the_best_so_far_curve():
    score, _ = make_scorer()
    result = GeneticOptimizer(SearchSpace(), seed=3, population_size=10, generations=6).optimize(
        score, N_MODELS, N_BLOCKS
    )
    assert len(result.history) == 7  # generation 0 + six updates
    assert list(result.history) == sorted(result.history)
    assert result.history[-1] == pytest.approx(result.best_fitness)
    assert result.best_fitness >= result.history[0]


def test_genetic_evaluations_count_unique_genomes_only():
    score, scored = make_scorer()
    result = GeneticOptimizer(SearchSpace(), seed=5, population_size=6, generations=3).optimize(
        score, N_MODELS, N_BLOCKS
    )
    keys = [genome.key() for genome in scored]
    assert result.evaluations == len(keys) == len(set(keys))
    assert result.all_fitness and len(result.all_fitness) == len(keys)
    assert result.evaluations >= 6


def test_genetic_keeps_a_seeded_seedling_that_is_already_optimal():
    score, _ = make_scorer()
    champion = make_genome(method=MergeMethod.DARE_TIES)
    result = GeneticOptimizer(SearchSpace(), seed=2, population_size=6, generations=3).optimize(
        score, N_MODELS, N_BLOCKS, initial=(champion,)
    )
    assert result.best_fitness == pytest.approx(1.25)
    assert result.best_genome.key() == champion.key()


def test_genetic_initial_genomes_are_clipped_into_the_search_space():
    space = SearchSpace(weight_low=0.4, weight_high=0.6, methods=(MergeMethod.LINEAR,))
    score, scored = make_scorer()
    wild = make_genome(weights=[[9.0, -3.0] for _ in range(N_MODELS)], method=MergeMethod.TIES)
    GeneticOptimizer(space, seed=1, population_size=5, generations=1).optimize(
        score, N_MODELS, N_BLOCKS, initial=(wild,)
    )
    seeded = scored[0]
    assert seeded.method is MergeMethod.LINEAR
    for row in seeded.weights:
        assert all(space.weight_low <= value <= space.weight_high for value in row)


def test_genetic_search_space_pins_the_merge_method_gene():
    space = SearchSpace(methods=(MergeMethod.LINEAR,))
    score, scored = make_scorer()
    GeneticOptimizer(space, seed=4, population_size=8, generations=4, method_mutation_rate=1.0).optimize(
        score, N_MODELS, N_BLOCKS
    )
    assert {genome.method for genome in scored} == {MergeMethod.LINEAR}


def test_method_mutation_explores_several_merge_strategies():
    score, scored = make_scorer()
    GeneticOptimizer(
        SearchSpace(methods=ALL_METHODS),
        seed=9,
        population_size=12,
        generations=6,
        method_mutation_rate=0.5,
    ).optimize(score, N_MODELS, N_BLOCKS)
    assert len({genome.method for genome in scored}) > 1


def test_genetic_reports_its_configuration_in_extra():
    score, _ = make_scorer()
    result = GeneticOptimizer(SearchSpace(), seed=8, population_size=7, generations=2).optimize(
        score, N_MODELS, N_BLOCKS
    )
    assert result.optimizer == "genetic"
    assert result.extra["population_size"] == 7.0
    assert result.extra["generations"] == 2.0


def test_genetic_survives_a_flat_fitness_landscape():
    result = GeneticOptimizer(SearchSpace(), seed=11, population_size=5, generations=2).optimize(
        lambda genome: 0.42, N_MODELS, N_BLOCKS
    )
    assert result.best_fitness == pytest.approx(0.42)
    assert set(result.history) == {0.42}


# ---------------------------------------------------------------------- optuna
def test_optuna_rejects_a_zero_trial_budget():
    with pytest.raises(ValueError, match="n_trials"):
        OptunaOptimizer(SearchSpace(), n_trials=0)


def test_optuna_runs_exactly_the_requested_number_of_trials():
    score, scored = make_scorer()
    result = OptunaOptimizer(SearchSpace(), seed=13, n_trials=12).optimize(score, N_MODELS, N_BLOCKS)
    assert result.optimizer == "optuna"
    assert result.evaluations == 12
    assert len(result.history) == 12
    assert len(result.all_fitness) == 12
    assert result.best_fitness == pytest.approx(max(result.history))
    assert result.extra["n_trials"] == 12.0


def test_optuna_is_reproducible_for_a_given_seed():
    def run():
        score, _ = make_scorer()
        return OptunaOptimizer(SearchSpace(), seed=21, n_trials=9).optimize(score, N_MODELS, N_BLOCKS)

    first, second = run(), run()
    assert first.history == second.history
    assert first.best_genome.key() == second.best_genome.key()


def test_optuna_stays_inside_the_declared_search_space():
    space = SearchSpace(
        weight_low=0.1,
        weight_high=0.7,
        density_low=0.2,
        density_high=0.8,
        dare_density_low=0.1,
        dare_density_high=0.5,
        dare_scale_low=1.2,
        dare_scale_high=2.0,
        methods=(MergeMethod.TIES, MergeMethod.DARE_TIES),
    )
    score, scored = make_scorer()
    OptunaOptimizer(space, seed=17, n_trials=10).optimize(score, N_MODELS, N_BLOCKS)

    assert scored
    for genome in scored:
        assert genome.method in space.methods
        for row in genome.weights:
            assert all(space.weight_low <= v <= space.weight_high for v in row)
        assert space.density_low <= genome.density <= space.density_high
        assert space.dare_density_low <= genome.dare_density <= space.dare_density_high
        assert space.dare_scale_low <= genome.dare_scale <= space.dare_scale_high


def test_optuna_enqueues_initial_genomes_as_its_first_trials():
    score, scored = make_scorer()
    champion = make_genome(method=MergeMethod.DARE_TIES)
    result = OptunaOptimizer(SearchSpace(), seed=23, n_trials=6).optimize(
        score, N_MODELS, N_BLOCKS, initial=(champion,)
    )
    assert scored[0].key() == champion.key()
    assert result.history[0] == pytest.approx(1.25)
    assert result.best_fitness >= 1.25


def test_optuna_timeout_caps_the_trial_budget():
    score, _ = make_scorer()
    result = OptunaOptimizer(SearchSpace(), seed=31, n_trials=20000, timeout_seconds=1).optimize(
        score, N_MODELS, N_BLOCKS
    )
    assert 1 <= result.evaluations < 20000


def test_optuna_without_a_pruner_still_completes():
    score, _ = make_scorer()
    result = OptunaOptimizer(SearchSpace(), seed=41, n_trials=5, prune=False).optimize(
        score, N_MODELS, N_BLOCKS
    )
    assert result.evaluations == 5


def test_optuna_persists_a_study_to_storage(tmp_path):
    storage = f"sqlite:///{(tmp_path / 'study.db').as_posix()}"
    score_a, _ = make_scorer()
    first = OptunaOptimizer(SearchSpace(), seed=51, n_trials=4, storage=storage).optimize(
        score_a, N_MODELS, N_BLOCKS
    )
    assert (tmp_path / "study.db").exists()

    score_b, scored_b = make_scorer()
    second = OptunaOptimizer(SearchSpace(), seed=52, n_trials=3, storage=storage).optimize(
        score_b, N_MODELS, N_BLOCKS
    )
    # the resumed study only runs its own new trials
    assert second.evaluations == 3
    assert first.evaluations == 4


def test_optuna_reports_a_helpful_error_when_the_package_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(RuntimeError, match="optuna is required"):
        OptunaOptimizer(SearchSpace(), n_trials=1).optimize(lambda genome: 1.0, N_MODELS, N_BLOCKS)


# ------------------------------------------------------- optuna helper mapping
def test_flatten_and_rebuild_round_trip_a_genome():
    space = SearchSpace()
    genome = make_genome(method=MergeMethod.DARE_LINEAR, weights=[[0.1, 1.4], [0.7, 0.2], [1.1, 0.9]])
    params = _flatten_genome(genome, N_MODELS, N_BLOCKS)

    assert params["method"] == "dare_linear"
    assert params["weight_m2_b1"] == pytest.approx(0.9)
    assert params["normalize"] is True

    restored = genome_from_trial_values(params, space, N_MODELS, N_BLOCKS)
    assert restored.key() == genome.key()


def test_genome_from_trial_values_fills_missing_parameters_with_neutrals():
    space = SearchSpace(weight_low=0.0, weight_high=1.0)
    genome = genome_from_trial_values({}, space, 2, 2)
    assert genome.method is MergeMethod.LINEAR
    assert [list(row) for row in genome.weights] == [[0.5, 0.5], [0.5, 0.5]]
    assert genome.density == pytest.approx(space.density_high)
    assert genome.dare_density == pytest.approx(space.dare_density_low)
    assert genome.dare_scale == pytest.approx(space.dare_scale_low)
    assert genome.normalize is True


def test_genome_from_trial_values_clips_out_of_range_parameters():
    space = SearchSpace(weight_low=0.0, weight_high=1.0, dare_scale_low=1.0, dare_scale_high=2.0)
    genome = genome_from_trial_values(
        {"method": "ties", "weight_m0_b0": 9.0, "weight_m0_b1": -4.0, "dare_scale": 99.0},
        space,
        1,
        2,
    )
    assert genome.weights[0][0] <= 1.0
    assert genome.weights[0][1] >= 0.0
    assert genome.dare_scale <= space.dare_scale_high
    assert all(math.isfinite(v) for row in genome.weights for v in row)