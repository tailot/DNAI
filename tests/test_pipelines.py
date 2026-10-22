"""Orchestrator: world wiring, score cache, failure isolation, run artifacts."""

from __future__ import annotations

import json
import os

import pytest

from zeroai.eval.base import Evaluator
from zeroai.eval.synthetic import SyntheticEvaluator
from zeroai.genome import Genome
from zeroai.merge.backend import MergeBackend, NativeMergeBackend
from zeroai.merge.mergekit_backend import MergekitBackend
from zeroai.optimizers.base import OptimizationResult, Optimizer
from zeroai.optimizers.genetic import GeneticOptimizer
from zeroai.optimizers.optuna_opt import OptunaOptimizer
from zeroai.pipelines import (
    EvolutionaryMergePipeline,
    PipelineConfig,
    PipelineResult,
)
from zeroai.schemas import (
    DType,
    EvaluationReport,
    MergeMethod,
    MetricScore,
    SearchSpace,
)

ARTIFACT_KEYS = {"best_genome", "mergekit_yaml", "candidates_csv", "summary", "summary_json", "run_dir"}


def make_config(tmp_path, **overrides):
    kwargs = dict(
        mode="mock",
        num_blocks=2,
        run_dir=str(tmp_path / "runs"),
        run_name="run",
        optimizer="none",
        seed=11,
        dtype=DType.BFLOAT16,
        synthetic_tensors_per_block=2,
        synthetic_tensor_shape=(6, 6),
    )
    kwargs.update(overrides)
    return PipelineConfig(**kwargs)


def make_pipeline(tmp_path, **overrides) -> EvolutionaryMergePipeline:
    return EvolutionaryMergePipeline(make_config(tmp_path, **overrides))


def fixed_genome(n_models: int, n_blocks: int, **kwargs) -> Genome:
    kwargs.setdefault("method", MergeMethod.LINEAR)
    kwargs.setdefault("density", 1.0)
    kwargs.setdefault("dare_density", 0.0)
    kwargs.setdefault("dare_scale", 1.0)
    return Genome(
        weights=[[1.0 / n_models] * n_blocks for _ in range(n_models)], **kwargs
    )


class RepeatTwiceOptimizer(Optimizer):
    """Scores the very same genome twice - the pipeline must collapse it to one merge."""

    name = "repeat"

    def optimize(self, score_fn, n_models, n_blocks, initial=()):
        genome = fixed_genome(n_models, n_blocks)
        first = score_fn(genome)
        score_fn(genome.clone())
        return OptimizationResult(
            best_genome=genome,
            best_fitness=first,
            evaluations=2,
            optimizer=self.name,
            history=(first, first),
        )


class ExplodingBackend(MergeBackend):
    name = "exploding"

    def merge(self, plan, candidate_id):
        raise RuntimeError("cuda out of memory")


class FlakyEvaluator(Evaluator):
    name = "flaky"

    def __init__(self, inner: Evaluator, fail_calls=1):
        self.inner = inner
        self.fail_calls = fail_calls
        self.calls = 0

    def evaluate(self, artifact):
        self.calls += 1
        if self.calls <= self.fail_calls:
            raise RuntimeError("tokenizer missing")
        return self.inner.evaluate(artifact)

    def close(self):
        self.inner.close()


class NonFiniteEvaluator(Evaluator):
    name = "non-finite"

    def evaluate(self, artifact):
        return EvaluationReport(
            fitness=float("nan"),
            metrics=(MetricScore("capability", float("nan")),),
            candidate_id=artifact.candidate_id,
        )


def run_with(tmp_path, evaluator=None, backend=None, optimizer_instance=None, **config_kwargs):
    """Run a mock pipeline, optionally swapping one of its collaborators.

    ``optimizer`` stays a plain config string; ``optimizer_instance`` replaces the
    factory outright so a test can inject its own search engine.
    """
    pipeline = make_pipeline(tmp_path, **config_kwargs)
    if evaluator is not None:
        pipeline._make_evaluator = lambda: evaluator
    if backend is not None:
        pipeline._make_backend = lambda base_store, model_stores: backend
    if optimizer_instance is not None:
        pipeline._make_optimizer = lambda: optimizer_instance
    return pipeline.run()


# ------------------------------------------------------------------ config
def test_config_rejects_unknown_mode_and_optimizer():
    with pytest.raises(ValueError, match="mode must be"):
        PipelineConfig(mode="dream")
    with pytest.raises(ValueError, match="optimizer must be"):
        PipelineConfig(mode="mock", optimizer="simulated_annealing")
    with pytest.raises(ValueError, match="num_blocks must be"):
        PipelineConfig(mode="mock", num_blocks=0)


def test_model_mode_requires_a_base_and_at_least_one_model():
    with pytest.raises(ValueError, match="needs base_model and models"):
        PipelineConfig(mode="model")


def test_run_directory_is_created_eagerly(tmp_path):
    pipeline = make_pipeline(tmp_path, run_name="nested")
    assert os.path.isdir(pipeline.run_dir)
    assert pipeline.run_dir == os.path.abspath(str(tmp_path / "runs" / "nested"))


# ------------------------------------------------------------------ prepare
def test_mock_prepare_builds_a_synthetic_world(tmp_path):
    base, models, base_store, model_stores, names = make_pipeline(
        tmp_path, num_blocks=3
    ).prepare()

    assert base.label == "base"
    assert base.path == "synthetic://base"
    assert len(models) == 3
    assert all(model.path.startswith("synthetic://") for model in models)
    assert len(model_stores) == len(models)
    assert len(names) == 3 * 2  # num_blocks * tensors_per_block
    assert set(base_store.names()) == set(names)


def test_mock_prepare_keeps_user_supplied_model_labels(tmp_path):
    from zeroai.schemas import ModelSpec

    config = make_config(
        tmp_path,
        models=(ModelSpec(path="experts/alpha", label="alpha"), ModelSpec(path="experts/beta", label="beta")),
    )
    _base, models, _base_store, model_stores, names = EvolutionaryMergePipeline(config).prepare()
    assert [model.label for model in models] == ["alpha", "beta"]
    assert len(model_stores) == 2


def test_blocks_partition_every_tensor_exactly_once(tmp_path):
    pipeline = make_pipeline(tmp_path, num_blocks=3)
    _base, _models, _stores_base, _stores_models, names = pipeline.prepare()
    blocks = pipeline._blocks(names)

    assert len(blocks) == 3
    covered = [name for block in blocks for name in block.tensor_names]
    assert sorted(covered) == sorted(names)
    assert len(covered) == len(set(covered))


def test_model_prepare_requires_shared_tensors(tmp_path):
    from zeroai.schemas import ModelSpec

    empty_dir = tmp_path / "empty_checkpoint"
    empty_dir.mkdir()
    missing = ModelSpec(path=str(empty_dir), label="empty")
    config = PipelineConfig(
        mode="model",
        base_model=missing,
        models=(missing,),
        num_blocks=2,
        run_dir=str(tmp_path / "runs"),
    )
    with pytest.raises(Exception, match="no .safetensors files"):
        EvolutionaryMergePipeline(config).prepare()


# ------------------------------------------------------------- module wiring
def test_backend_and_evaluator_selection(tmp_path):
    pipeline = make_pipeline(tmp_path)
    _base, _models, base_store, model_stores, _names = pipeline.prepare()

    assert isinstance(pipeline._make_backend(base_store, model_stores), NativeMergeBackend)
    assert isinstance(pipeline._make_evaluator(), SyntheticEvaluator)

    forced = make_pipeline(tmp_path, merge_backend="mergekit")
    assert isinstance(forced._make_backend(base_store, model_stores), MergekitBackend)


def test_optimizer_selection(tmp_path):
    assert make_pipeline(tmp_path, optimizer="none")._make_optimizer() is None
    assert isinstance(
        make_pipeline(tmp_path, optimizer="genetic")._make_optimizer(), GeneticOptimizer
    )
    assert isinstance(
        make_pipeline(tmp_path, optimizer="optuna")._make_optimizer(), OptunaOptimizer
    )


# ------------------------------------------------------------------- single run
def test_single_genome_run_produces_a_scored_candidate_and_artifacts(tmp_path):
    result = run_with(tmp_path, initial_genomes=(fixed_genome(3, 2),))

    assert isinstance(result, PipelineResult)
    assert result.mode == "mock"
    assert result.optimization is None
    assert len(result.candidates) == 1
    record = result.candidates[0]
    assert record.error is None
    assert record.report is not None
    assert 0.0 < record.fitness <= 1.0
    assert record.stats["tensors"] > 0
    assert result.best_fitness == pytest.approx(record.fitness)

    assert set(result.report_paths) == ARTIFACT_KEYS
    for key, path in result.report_paths.items():
        if key != "run_dir":
            assert os.path.isfile(path), f"missing artifact {key}: {path}"


def test_write_checkpoints_flag_materialises_a_merged_checkpoint(tmp_path):
    result = run_with(tmp_path, write_checkpoints=True, initial_genomes=(fixed_genome(3, 2),))
    candidate_dir = os.path.join(result.run_dir, "candidates", result.candidates[0].candidate_id)
    assert os.path.isfile(os.path.join(candidate_dir, "model.safetensors"))
    assert result.candidates[0].report is not None  # deltas are still kept in mock mode


def test_backend_stats_report_every_block_touched(tmp_path):
    result = run_with(tmp_path, initial_genomes=(fixed_genome(3, 2),))
    record = result.candidates[0]
    assert record.stats["blocks_used"] == 2.0
    assert record.stats["tensors"] == 4.0
    assert record.stats["merged_delta_norm"] > 0.0


# ---------------------------------------------------------------- optimisers
def test_genetic_run_reports_a_monotonic_convergence_history(tmp_path):
    result = run_with(tmp_path, optimizer="genetic", population_size=5, generations=3)
    optimization = result.optimization

    assert optimization.optimizer == "genetic"
    assert len(optimization.history) == 4  # generation 0 plus three updates
    assert list(optimization.history) == sorted(optimization.history)
    assert optimization.best_fitness == pytest.approx(result.best_fitness)
    assert optimization.evaluations == len(result.candidates)
    assert all(record.error is None for record in result.candidates)


def test_optuna_run_evaluates_exactly_the_requested_trial_budget(tmp_path):
    result = run_with(tmp_path, optimizer="optuna", n_trials=6)
    optimization = result.optimization

    assert optimization.optimizer == "optuna"
    assert optimization.evaluations == 6
    assert len(optimization.history) == 6
    assert result.best_fitness == pytest.approx(
        max(record.fitness for record in result.candidates)
    )


def test_search_space_restriction_is_respected_by_every_candidate(tmp_path):
    space = SearchSpace(methods=(MergeMethod.LINEAR,))
    result = run_with(
        tmp_path,
        optimizer="genetic",
        population_size=4,
        generations=2,
        search_space=space,
    )
    assert {record.genome.method for record in result.candidates} == {MergeMethod.LINEAR}
    assert result.best_genome.method is MergeMethod.LINEAR


def test_identical_genomes_are_merged_and_evaluated_only_once(tmp_path):
    result = run_with(tmp_path, optimizer_instance=RepeatTwiceOptimizer(SearchSpace(), seed=1))
    assert len(result.candidates) == 1
    assert result.candidates[0].error is None


# ------------------------------------------------------------- failure handling
def test_a_merge_failure_is_recorded_without_aborting_the_search(tmp_path):
    result = run_with(
        tmp_path,
        backend=ExplodingBackend(),
        optimizer="genetic",
        population_size=4,
        generations=1,
    )
    assert result.candidates
    assert all(record.error and record.error.startswith("merge:") for record in result.candidates)
    assert all(record.fitness == 0.0 for record in result.candidates)
    assert result.best_fitness == 0.0
    assert result.top(5) == []


def test_an_evaluation_failure_only_kills_that_candidate(tmp_path):
    pipeline = make_pipeline(tmp_path, optimizer="genetic", population_size=4, generations=1)
    inner = _prepare_benchmark_evaluator(pipeline)
    pipeline._make_evaluator = lambda: FlakyEvaluator(inner, fail_calls=1)
    result = pipeline.run()

    failures = [record for record in result.candidates if record.error]
    successes = [record for record in result.candidates if record.error is None]
    assert len(failures) == 1
    assert failures[0].error.startswith("eval: ")
    assert "tokenizer missing" in failures[0].error
    assert failures[0].merge_seconds > 0.0
    assert successes and all(record.fitness > 0.0 for record in successes)
    assert result.best_fitness > 0.0


def test_a_non_finite_report_is_discarded_as_a_failure(tmp_path):
    result = run_with(tmp_path, evaluator=NonFiniteEvaluator(), initial_genomes=(fixed_genome(3, 2),))
    record = result.candidates[0]
    assert record.error == "non-finite fitness"
    assert record.fitness == 0.0
    assert record.report is None


def test_failures_are_visible_in_the_report_but_never_in_top(tmp_path):
    pipeline = make_pipeline(tmp_path, optimizer="genetic", population_size=4, generations=1)
    inner = _prepare_benchmark_evaluator(pipeline)
    pipeline._make_evaluator = lambda: FlakyEvaluator(inner, fail_calls=2)
    result = pipeline.run()

    payload = result.to_dict()
    assert payload["failures"] == 2
    assert len(payload["candidates"]) == len(result.candidates)
    assert len(result.top(10)) == len(result.candidates) - 2


# ----------------------------------------------------------------- result api
def test_top_returns_the_best_error_free_candidates(tmp_path):
    result = run_with(tmp_path, optimizer="optuna", n_trials=8)
    top = result.top(3)
    assert len(top) == 3
    assert [record.fitness for record in top] == sorted((r.fitness for r in top), reverse=True)
    assert all(record.error is None for record in top)
    assert top[0].fitness == pytest.approx(result.best_fitness)


def test_result_to_dict_is_json_serialisable_and_complete(tmp_path):
    result = run_with(tmp_path, optimizer="genetic", population_size=4, generations=1)
    payload = json.loads(json.dumps(result.to_dict()))

    for key in ("mode", "run_dir", "best_fitness", "best_genome", "candidates", "blocks", "domains"):
        assert key in payload
    assert payload["optimization"]["optimizer"] == "genetic"
    assert payload["candidates"][0]["genome"]["method"] in {m.value for m in MergeMethod}
    assert payload["failures"] == 0
    assert len(payload["blocks"]) == 2
    assert payload["domains"]


def test_candidate_record_to_dict_carries_metrics_and_stats(tmp_path):
    result = run_with(tmp_path, initial_genomes=(fixed_genome(3, 2),))
    payload = result.candidates[0].to_dict()
    assert payload["error"] is None
    assert payload["metrics"]["capability"] == pytest.approx(
        result.candidates[0].report.metric("capability").value
    )
    assert payload["genome"]["weights"]
    assert payload["stats"]["merged_delta_norm"] > 0.0


def _prepare_benchmark(pipeline):
    pipeline.prepare()
    return pipeline._benchmark


def _prepare_benchmark_evaluator(pipeline):
    return SyntheticEvaluator(_prepare_benchmark(pipeline))