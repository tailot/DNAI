"""CandidateGenerator: genome -> MergePlan -> mergekit YAML."""

from __future__ import annotations

import os

import pytest
import yaml

from zeroai.blocks import partition_tensors
from zeroai.candidate import CandidateGenerator
from zeroai.eval.synthetic import build_synthetic_world
from zeroai.genome import Genome, normalize_weights
from zeroai.schemas import DType, MergeMethod, ModelSpec

BASE = ModelSpec(path="models/base", label="base")


def make_world(num_models=3, num_blocks=2, tensors_per_block=2, seed=5):
    benchmark, _base_store, _model_stores = build_synthetic_world(
        num_models=num_models,
        num_blocks=num_blocks,
        tensors_per_block=tensors_per_block,
        tensor_shape=(4, 4),
        seed=seed,
    )
    models = tuple(
        ModelSpec(path=f"models/{label}", label=label) for label in benchmark.expert_labels
    )
    blocks = partition_tensors(benchmark.tensor_names, num_blocks)
    return models, blocks


@pytest.fixture()
def world():
    return make_world()


@pytest.fixture()
def generator(world):
    models, blocks = world
    return CandidateGenerator(
        base_model=BASE, models=models, blocks=blocks, dtype=DType.BFLOAT16, seed=7
    )


def make_genome(n_models, n_blocks, **kwargs):
    kwargs.setdefault("weights", [[0.5] * n_blocks for _ in range(n_models)])
    kwargs.setdefault("method", MergeMethod.LINEAR)
    kwargs.setdefault("density", 1.0)
    kwargs.setdefault("dare_density", 0.0)
    kwargs.setdefault("dare_scale", 1.0)
    return Genome(**kwargs)


def generator_for(world, models=None, blocks=None, **kwargs):
    world_models, world_blocks = world
    return CandidateGenerator(
        base_model=BASE,
        models=world_models if models is None else models,
        blocks=world_blocks if blocks is None else blocks,
        **kwargs,
    )


# ------------------------------------------------------------ genome guards
def test_genome_requires_at_least_one_weight_row():
    with pytest.raises(ValueError, match="one row per model"):
        Genome(
            method=MergeMethod.LINEAR,
            weights=[],
            density=1.0,
            dare_density=0.0,
            dare_scale=1.0,
        )


def test_genome_requires_ragged_free_weights():
    with pytest.raises(ValueError, match="same block count"):
        Genome(
            method=MergeMethod.LINEAR,
            weights=[[0.5, 0.5], [0.5]],
            density=1.0,
            dare_density=0.0,
            dare_scale=1.0,
        )


# ----------------------------------------------------------------- plan shape
def test_plan_rejects_the_wrong_number_of_models(world):
    models, blocks = world
    generator = generator_for(world, models=models[:2])
    with pytest.raises(ValueError, match="genome has 3 model rows, generator has 2 models"):
        generator.plan(make_genome(3, len(blocks)), candidate_id="x")


def test_plan_rejects_the_wrong_number_of_blocks(world):
    models, blocks = world
    generator = generator_for(world, blocks=blocks[:1])
    with pytest.raises(ValueError, match="block columns, generator has 1 blocks"):
        generator.plan(make_genome(len(models), len(blocks)), candidate_id="x")


def test_plan_rejects_an_out_of_range_density(world):
    with pytest.raises(ValueError, match="density must be in"):
        generator_for(world).plan(
            make_genome(3, 2, density=0.0), candidate_id="x"
        )


def test_plan_rejects_an_out_of_range_dare_density(world):
    with pytest.raises(ValueError, match="dare_density must be in"):
        generator_for(world).plan(
            make_genome(3, 2, dare_density=1.0), candidate_id="x"
        )


def test_plan_rejects_a_non_positive_dare_scale(world):
    with pytest.raises(ValueError, match="dare_scale must be > 0"):
        generator_for(world).plan(
            make_genome(3, 2, dare_scale=0.0), candidate_id="x"
        )


def test_generator_requires_models_and_blocks():
    models, blocks = make_world(num_models=1, num_blocks=1)
    with pytest.raises(ValueError, match="at least one input model"):
        CandidateGenerator(base_model=BASE, models=[], blocks=blocks)
    with pytest.raises(ValueError, match="at least one block"):
        CandidateGenerator(base_model=BASE, models=models, blocks=[])


# ----------------------------------------------------------------- weighting
def test_plan_normalises_weights_per_block_by_default(world):
    plan = generator_for(world).plan(
        make_genome(3, 2, weights=[[2.0, 2.0]] * 3), candidate_id="norm"
    )
    for column in zip(*plan.weights):
        assert sum(column) == pytest.approx(1.0)


def test_plan_keeps_raw_weights_when_normalisation_is_off(world):
    plan = generator_for(world).plan(
        make_genome(3, 2, weights=[[2.0, 2.0]] * 3, normalize=False), candidate_id="raw"
    )
    assert all(value == 2.0 for row in plan.weights for value in row)
    assert plan.normalize is False


def test_force_normalize_true_overrides_the_genome(world):
    generator = generator_for(world, force_normalize=True)
    plan = generator.plan(
        make_genome(3, 2, weights=[[2.0, 2.0]] * 3, normalize=False), candidate_id="forced"
    )
    assert plan.normalize is True
    assert all(sum(column) == pytest.approx(1.0) for column in zip(*plan.weights))


def test_force_normalize_false_overrides_the_genome(world):
    generator = generator_for(world, force_normalize=False)
    plan = generator.plan(make_genome(3, 2, weights=[[2.0, 2.0]] * 3), candidate_id="raw")
    assert plan.normalize is False
    assert all(value == 2.0 for row in plan.weights for value in row)


def test_normalize_weights_scales_each_block_column_to_unit_sum():
    assert normalize_weights([[2.0, 1.0], [2.0, 3.0]]) == ((0.5, 0.25), (0.5, 0.75))


def test_normalize_weights_splits_a_dead_block_evenly():
    assert normalize_weights([[0.0, 1.0], [0.0, 1.0]]) == ((0.5, 0.5), (0.5, 0.5))


def test_normalize_weights_of_no_rows():
    assert normalize_weights([]) == ()


def test_normalisation_falls_back_to_an_even_split_for_a_dead_block(world):
    # A block every model silenced still has to receive *some* coefficient,
    # otherwise normalisation would leave it mathematically undefined.
    plan = generator_for(world).plan(
        make_genome(3, 2, weights=[[1.0, 0.0]] * 3), candidate_id="dead"
    )
    dead_column = [row[1] for row in plan.weights]
    assert dead_column == [pytest.approx(1 / 3)] * 3


# ------------------------------------------------------------------- plumbing
def test_output_path_for_is_candidate_scoped(world, tmp_path):
    generator = generator_for(world, output_dir=str(tmp_path / "candidates"))
    path = generator.output_path_for("cand_0007")
    assert path == os.path.join(str(tmp_path / "candidates"), "cand_0007")


def test_output_path_is_empty_without_an_output_dir(generator):
    assert generator.output_path_for("cand_0001") == ""
    plan = generator.plan(make_genome(3, 2), candidate_id="cand_0001")
    assert plan.output_path == ""


def test_plan_carries_generator_dtype_and_seed(world):
    generator = generator_for(world, dtype=DType.FLOAT32, seed=99)
    plan = generator.plan(make_genome(3, 2, dtype=None), candidate_id="p")
    assert plan.dtype is DType.FLOAT32
    assert plan.seed == 99


def test_genome_dtype_overrides_the_generator_dtype(generator):
    plan = generator.plan(make_genome(3, 2, dtype=DType.FLOAT16), candidate_id="p")
    assert plan.dtype is DType.FLOAT16


def test_blocks_from_names_rebuilds_a_full_partition(world):
    models, blocks = world
    names = [name for block in blocks for name in block.tensor_names]
    rebuilt = CandidateGenerator.blocks_from_names(names, len(blocks))
    assert {n for b in rebuilt for n in b.tensor_names} == set(names)
    assert [b.index for b in rebuilt] == list(range(len(blocks)))


def test_plan_is_internally_valid(generator):
    plan = generator.plan(make_genome(3, 2), candidate_id="p")
    plan.validate()
    assert plan.n_models == 3
    assert plan.n_blocks == 2
    assert len(plan.weights[0]) == 2


# --------------------------------------------------------------------- yaml
def test_yaml_is_a_mergekit_shaped_document(generator):
    document = yaml.safe_load(
        generator.yaml(make_genome(3, 2, dtype=DType.BFLOAT16), candidate_id="manual")
    )
    assert document["merge_method"] == "linear"
    assert document["dtype"] == "bfloat16"
    assert document["base_model"] == BASE.path
    assert len(document["models"]) == 3
    assert document["models"][0]["model"] == "models/expert_0"
    assert len(document["models"][0]["parameters"]["weight"]) == 2
    assert document["parameters"]["normalize"] is True


def test_yaml_header_documents_the_genome(generator):
    text = generator.yaml(
        make_genome(3, 2, method=MergeMethod.DARE_TIES, density=0.7, dare_density=0.3),
        candidate_id="manual",
    )
    first, second = text.splitlines()[:2]
    assert first.startswith("# Generated by ZeroAI CandidateGenerator")
    assert "method=dare_ties" in second
    assert "blocks=2" in second
    assert "dare_drop=0.3000" in second


def test_yaml_emits_per_model_density_only_for_trim_methods(generator):
    linear = yaml.safe_load(generator.yaml(make_genome(3, 2, method=MergeMethod.LINEAR), "m"))
    assert "density" not in linear["models"][0]["parameters"]
    plain = yaml.safe_load(
        generator.yaml(make_genome(3, 2, method=MergeMethod.LINEAR, density=0.6), "m")
    )
    assert "density" not in plain["models"][0]["parameters"]
    for method in (MergeMethod.TIES, MergeMethod.TASKSLIM):
        document = yaml.safe_load(
            generator.yaml(make_genome(3, 2, method=method, density=0.6), "m")
        )
        assert document["merge_method"] == method.value
        assert document["models"][0]["parameters"]["density"] == 0.6


def test_yaml_gives_dare_methods_a_per_model_keep_fraction(generator):
    for method in (MergeMethod.DARE_LINEAR, MergeMethod.DARE_TIES):
        document = yaml.safe_load(
            generator.yaml(
                make_genome(3, 2, method=method, density=0.6, dare_density=0.25), "m"
            )
        )
        parameters = document["models"][0]["parameters"]
        assert parameters["density"] == pytest.approx(0.75)
        assert parameters["rescale_norm"] == "l2"


def test_yaml_always_reports_the_density_gene_at_the_top_level(generator):
    for method in MergeMethod:
        document = yaml.safe_load(
            generator.yaml(make_genome(3, 2, method=method, density=0.6), "m")
        )
        assert document["merge_method"] == method.value
        assert document["parameters"]["density"] == 0.6


def test_yaml_inverts_dare_density_into_a_keep_fraction(generator):
    document = yaml.safe_load(
        generator.yaml(
            make_genome(
                3, 2, method=MergeMethod.DARE_TIES, density=0.7, dare_density=0.4, dare_scale=1.3
            ),
            "m",
        )
    )
    parameters = document["models"][0]["parameters"]
    assert parameters["density"] == pytest.approx(0.6)
    assert parameters["rescale_norm"] == "l2"


def test_yaml_weight_list_is_per_block_and_model_specific(generator):
    document = yaml.safe_load(
        generator.yaml(
            make_genome(3, 2, weights=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], normalize=False),
            "m",
        )
    )
    weights = [model["parameters"]["weight"] for model in document["models"]]
    assert weights == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]


def test_yaml_normalize_flag_follows_the_plan(generator):
    document = yaml.safe_load(
        generator.yaml(make_genome(3, 2, normalize=False), candidate_id="m")
    )
    assert document["parameters"]["normalize"] is False
    assert "normalize" not in document["models"][0]["parameters"]


def test_yaml_dtype_follows_the_plan(generator):
    document = yaml.safe_load(
        generator.yaml(make_genome(3, 2, dtype=DType.FLOAT32), candidate_id="m")
    )
    assert document["dtype"] == "float32"


def test_yaml_rejects_a_genome_with_the_wrong_shape(generator):
    with pytest.raises(ValueError, match="model rows"):
        generator.yaml(make_genome(4, 2), candidate_id="m")


def test_write_yaml_writes_the_document_to_disk(generator, tmp_path):
    path = generator.write_yaml(make_genome(3, 2), str(tmp_path / "sub" / "candidate.yaml"), "disk")
    assert os.path.isfile(path)
    document = yaml.safe_load(open(path, "r", encoding="utf-8"))
    assert document["merge_method"] == "linear"
    assert len(document["models"]) == 3