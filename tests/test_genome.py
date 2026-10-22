"""Chromosome semantics: identity, factories, operators and (de)serialisation."""

import random

import pytest

from zeroai.genome import Genome, genome_from_dict, genome_to_dict, normalize_weights
from zeroai.schemas import ALL_METHODS, DType, MergeMethod, SearchSpace


def make(weights=((0.5, 0.5), (0.5, 0.5)), method=MergeMethod.LINEAR, **overrides) -> Genome:
    kwargs = {"density": 1.0, "dare_density": 0.0, "dare_scale": 1.0, "normalize": True}
    kwargs.update(overrides)
    return Genome(method=method, weights=weights, **kwargs)


# --------------------------------------------------------------------- shape
def test_genome_requires_at_least_one_model_row():
    with pytest.raises(ValueError, match="one row per model"):
        make(weights=())


def test_genome_rejects_ragged_weight_rows():
    with pytest.raises(ValueError, match="same block count"):
        make(weights=((0.1, 0.2), (0.3,)))


def test_weights_are_coerced_to_float_tuples():
    genome = make(weights=((1, 2), (3, 4)))
    assert genome.weights == ((1.0, 2.0), (3.0, 4.0))
    assert all(isinstance(w, float) for row in genome.weights for w in row)
    assert genome.n_models == 2
    assert genome.n_blocks == 2


# ----------------------------------------------------------------- identity
def test_key_is_deterministic_and_discriminating():
    a = make()
    assert a.key() == make().key()
    assert a.key() != make(method=MergeMethod.TIES).key()
    assert a.key() != make(weights=((0.5, 0.6), (0.5, 0.5))).key()
    assert a.key() != make(density=0.5).key()
    assert a.key() != make(normalize=False).key()


def test_key_rounds_weights_so_near_duplicates_collapse():
    """The cache key is the dedup contract of the whole pipeline."""
    a = make()
    b = make(weights=((0.5 + 1e-9, 0.5), (0.5, 0.5)))
    assert a.key() == b.key()


def test_genome_is_immutable():
    genome = make()
    with pytest.raises(Exception):
        genome.density = 0.5  # type: ignore[misc]


def test_clone_is_an_equal_but_independent_object():
    genome = make()
    clone = genome.clone()
    assert clone is not genome
    assert clone == genome
    assert clone.key() == genome.key()


# --------------------------------------------------------------- factories
def test_sample_stays_inside_the_search_space():
    space = SearchSpace(
        weight_low=0.1, weight_high=0.9, methods=(MergeMethod.TIES, MergeMethod.LINEAR)
    )
    rng = random.Random(7)
    for _ in range(50):
        genome = Genome.sample(3, 4, space, rng)
        assert genome.n_models == 3 and genome.n_blocks == 4
        assert genome.method in space.methods
        for row in genome.weights:
            assert all(space.weight_low <= w <= space.weight_high for w in row)
        assert space.density_low <= genome.density <= space.density_high
        assert space.dare_density_low <= genome.dare_density <= space.dare_density_high
        assert space.dare_scale_low <= genome.dare_scale <= space.dare_scale_high


def test_random_is_seed_deterministic():
    space = SearchSpace()
    assert Genome.random(2, 3, space=space, seed=42) == Genome.random(2, 3, space=space, seed=42)
    assert Genome.random(2, 3, space=space, seed=1) != Genome.random(2, 3, space=space, seed=2)


def test_random_accepts_an_external_rng():
    rng = random.Random("seed")
    first = Genome.random(2, 2, rng=rng)
    second = Genome.random(2, 2, rng=rng)
    assert first != second


def test_uniform_defaults_to_one_over_models():
    genome = Genome.uniform(4, 3)
    assert genome.method is MergeMethod.LINEAR
    assert all(w == pytest.approx(0.25) for row in genome.weights for w in row)
    assert genome.density == 1.0
    assert genome.dare_density == 0.0
    assert genome.dare_scale == 1.0

    explicit = Genome.uniform(2, 2, weight=0.7)
    assert explicit.weights == ((0.7, 0.7), (0.7, 0.7))


# -------------------------------------------------------------- operators
def test_clipped_pulls_every_gene_back_inside():
    space = SearchSpace(
        weight_low=0.0,
        weight_high=1.0,
        density_low=0.2,
        density_high=0.8,
        dare_density_low=0.1,
        dare_density_high=0.5,
        dare_scale_low=1.0,
        dare_scale_high=2.0,
        methods=(MergeMethod.TIES,),
    )
    genome = make(
        weights=((-1.0, 9.0), (0.5, 0.5)),
        method=MergeMethod.DARE_TIES,
        density=5.0,
        dare_density=0.95,
        dare_scale=0.1,
    )
    clipped = genome.clipped(space)
    assert clipped.method is MergeMethod.TIES
    assert clipped.weights == ((0.0, 1.0), (0.5, 0.5))
    assert clipped.density == 0.8
    assert clipped.dare_density == 0.5
    assert clipped.dare_scale == 1.0
    # clipping never mutates the original genome
    assert genome.density == 5.0


def test_mutate_respects_its_internal_bounds():
    rng = random.Random(11)
    genome = make(weights=((0.5, 0.5), (0.5, 0.5)), density=0.5, dare_density=0.5, dare_scale=2.0)
    for _ in range(300):
        genome = genome.mutate(rng, rate=1.0, sigma=0.4)
        for row in genome.weights:
            assert all(w >= 0.0 for w in row)
        assert 0.05 <= genome.density <= 1.0
        assert 0.0 <= genome.dare_density <= 0.95
        assert 0.5 <= genome.dare_scale <= 4.0


def test_mutate_can_change_the_merge_method_gene():
    rng = random.Random(3)
    genome = make(method=MergeMethod.LINEAR)
    seen = {genome.method}
    for _ in range(200):
        genome = genome.mutate(rng, rate=1.0)
        seen.add(genome.method)
    assert len(seen) > 1
    assert seen.issubset(set(ALL_METHODS))


def test_mutate_with_zero_rate_is_a_no_op():
    genome = make(weights=((0.4, 0.6), (0.2, 0.8)))
    assert genome.mutate(random.Random(0), rate=0.0) == genome


def test_crossover_requires_matching_shapes():
    a = make(weights=((0.5, 0.5), (0.5, 0.5)))
    b = make(weights=((0.5, 0.5),))
    with pytest.raises(ValueError, match="identical shape"):
        a.crossover(b, random.Random(0))


def test_crossover_blends_inside_the_parents_span():
    rng = random.Random(5)
    a = make(weights=((0.0, 1.0), (0.0, 1.0)), method=MergeMethod.LINEAR, normalize=True)
    b = make(weights=((1.0, 0.0), (1.0, 0.0)), method=MergeMethod.TIES, normalize=False)
    for _ in range(40):
        child = a.crossover(b, rng, alpha=0.0)
        assert child.n_models == 2 and child.n_blocks == 2
        for row in child.weights:
            assert all(0.0 <= w <= 1.0 for w in row)
        assert child.method in (MergeMethod.LINEAR, MergeMethod.TIES)
        assert child.normalize in (True, False)


def test_crossover_with_alpha_can_explore_beyond_the_parents():
    """BLX-alpha deliberately overshoots - clipping is what contains it."""
    rng = random.Random(17)
    a = make(weights=((0.4, 0.4), (0.4, 0.4)))
    b = make(weights=((0.6, 0.6), (0.6, 0.6)))
    children = [a.crossover(b, rng, alpha=0.5) for _ in range(60)]
    assert any(w < 0.4 or w > 0.6 for child in children for w in child.weights[0])
    space = SearchSpace(weight_low=0.0, weight_high=1.0)
    clipped = [child.clipped(space) for child in children]
    assert all(0.0 <= w <= 1.0 for child in clipped for row in child.weights for w in row)


# ---------------------------------------------------------------------- dtype
def test_sample_stamps_the_search_space_dtype():
    space = SearchSpace(dtype=DType.BFLOAT16)
    genome = Genome.sample(2, 2, space, random.Random(4))
    assert genome.dtype is DType.BFLOAT16


def test_clipped_adopts_the_search_space_dtype():
    """The run dtype must survive clipping, otherwise --dtype is silently ignored."""
    space = SearchSpace(dtype=DType.BFLOAT16)
    assert make(weights=((0.5, 0.5),), dtype=DType.FLOAT16).clipped(space).dtype is DType.BFLOAT16
    assert make(weights=((0.5, 0.5),)).clipped(SearchSpace()).dtype is DType.FLOAT16


def test_mutate_and_crossover_keep_the_dtype():
    space = SearchSpace(dtype=DType.BFLOAT16)
    parent = Genome.sample(2, 2, space, random.Random(4))
    assert parent.crossover(parent.clone(), random.Random(6)).dtype is DType.BFLOAT16
    assert parent.mutate(random.Random(7)).dtype is DType.BFLOAT16


# ----------------------------------------------------------- (de)serialisation
def test_dict_roundtrip_preserves_every_gene():
    genome = make(
        weights=((0.11, 0.22), (0.33, 0.44)),
        method=MergeMethod.DARE_TIES,
        density=0.61,
        dare_density=0.24,
        dare_scale=1.7,
        normalize=False,
    )
    payload = genome.to_dict()
    assert payload["method"] == "dare_ties"
    assert payload["dtype"] == DType.FLOAT16.value
    assert payload["weights"] == [[0.11, 0.22], [0.33, 0.44]]
    assert Genome.from_dict(payload) == genome
    assert genome_from_dict(genome_to_dict(genome)) == genome


def test_from_dict_defaults_for_optional_genes():
    genome = Genome.from_dict(
        {
            "method": "linear",
            "weights": [[0.5, 0.5]],
            "density": 1.0,
            "dare_density": 0.0,
            "dare_scale": 1.0,
        }
    )
    assert genome.normalize is True
    assert genome.dtype is DType.FLOAT16


def test_short_repr_is_human_readable():
    text = make(method=MergeMethod.DARE_LINEAR).short_repr()
    assert "dare_linear" in text
    assert "density=" in text and "dare_density=" in text and "weights=" in text


# ------------------------------------------------------------- weight maths
def test_normalize_weights_normalises_per_block_column():
    rows = normalize_weights([[1.0, 2.0], [3.0, 4.0]])
    assert rows[0][0] == pytest.approx(0.25)
    assert rows[1][0] == pytest.approx(0.75)
    assert rows[0][1] == pytest.approx(2.0 / 6.0)
    assert rows[1][1] == pytest.approx(4.0 / 6.0)
    for block in range(2):
        assert sum(row[block] for row in rows) == pytest.approx(1.0)


def test_normalize_weights_splits_a_dead_block_evenly():
    rows = normalize_weights([[0.0, 1.0], [0.0, 1.0]])
    assert rows[0][0] == pytest.approx(0.5)
    assert rows[1][0] == pytest.approx(0.5)


def test_normalize_weights_keeps_sign_and_scales_by_magnitude():
    rows = normalize_weights([[1.0], [-3.0]])
    assert rows[0][0] == pytest.approx(0.25)
    assert rows[1][0] == pytest.approx(-0.75)


def test_normalize_weights_on_empty_input():
    assert normalize_weights([]) == ()