"""Tensor-naming heuristics and the block partition used by every plan."""

import pytest

from zeroai.blocks import (
    build_block_index,
    block_sizes,
    is_embedding_tensor,
    is_head_tensor,
    layer_index,
    partition_tensors,
)


LLAMA_NAMES = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.2.self_attn.q_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.layers.4.self_attn.q_proj.weight",
    "lm_head.weight",
]


# ------------------------------------------------------------- name probing
@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.layers.7.self_attn.q_proj.weight", 7),
        ("transformer.h.3.attn.c_attn.weight", 3),
        ("model.decoder_layer.12.mlp.up_proj.weight", 12),
        ("block_2.t0", 2),
        ("model.embed_tokens.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_layer_index(name, expected):
    assert layer_index(name) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("lm_head.weight", True),
        ("output.weight", True),
        ("model.final_dense.weight", True),
        ("cls.classifier.weight", True),
        ("model.embed_tokens.weight", False),
        ("model.layers.0.mlp.down_proj.weight", False),
    ],
)
def test_is_head_tensor(name, expected):
    assert is_head_tensor(name) is expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.embed_tokens.weight", True),
        ("transformer.word_embeddings.weight", True),
        ("tok_embeddings.weight", True),
        ("model.layers.0.input_embeddings.weight", True),
        ("lm_head.weight", False),
    ],
)
def test_is_embedding_tensor(name, expected):
    assert is_embedding_tensor(name) is expected


def test_a_head_tensor_is_never_also_an_embedding():
    for name in ("lm_head.weight", "output.weight"):
        assert not (is_head_tensor(name) and is_embedding_tensor(name))


# ---------------------------------------------------------------- partition
def test_partition_rejects_a_useless_block_count():
    for num_blocks in (0, -3):
        with pytest.raises(ValueError, match="num_blocks must be"):
            partition_tensors(LLAMA_NAMES, num_blocks)


def test_partition_rejects_an_empty_tensor_list():
    with pytest.raises(ValueError, match="at least one tensor name"):
        partition_tensors([], 4)


def test_partition_covers_every_tensor_exactly_once():
    blocks = partition_tensors(LLAMA_NAMES, 4)
    assert len(blocks) == 4
    flattened = [name for block in blocks for name in block.tensor_names]
    assert sorted(flattened) == sorted(LLAMA_NAMES)
    assert len(flattened) == len(set(flattened))
    assert [block.index for block in blocks] == [0, 1, 2, 3]
    assert [block.name for block in blocks] == ["block_0", "block_1", "block_2", "block_3"]


def test_partition_pins_embeddings_first_and_head_last():
    blocks = partition_tensors(LLAMA_NAMES, 4)
    assert "model.embed_tokens.weight" in blocks[0].tensor_names
    assert "lm_head.weight" in blocks[-1].tensor_names


def test_partition_keeps_layers_in_depth_order():
    blocks = partition_tensors(LLAMA_NAMES, 4)
    block_of_layer = {}
    for block in blocks:
        for name in block.tensor_names:
            index = layer_index(name)
            if index is not None:
                block_of_layer[index] = block.index
    layers = sorted(block_of_layer)
    assert [block_of_layer[i] for i in layers] == sorted(block_of_layer[i] for i in layers)
    assert block_of_layer[0] == 0
    assert block_of_layer[4] == 3


def test_partition_deduplicates_repeated_names():
    names = ["model.layers.0.mlp.gate_proj.weight"] * 5
    blocks = partition_tensors(names, 2)
    flattened = [name for block in blocks for name in block.tensor_names]
    assert flattened == [names[0]]


def test_partition_of_a_single_block_collects_everything():
    blocks = partition_tensors(LLAMA_NAMES, 1)
    assert len(blocks) == 1
    assert sorted(blocks[0].tensor_names) == sorted(LLAMA_NAMES)


def test_partition_maps_zeroai_synthetic_names_to_their_own_block():
    """The mock world names tensors ``block_<n>.t<j>``; blocks must stay aligned
    with the layout the synthetic evaluator scores, otherwise per-block
    coefficients would be inert in mock mode."""
    names = [f"block_{b}.t{j}" for b in range(4) for j in range(2)]
    blocks = partition_tensors(names, 4)
    assert [set(block.tensor_names) for block in blocks] == [
        {f"block_{b}.t{j}" for j in range(2)} for b in range(4)
    ]


def test_partition_spreads_unknown_names_across_blocks():
    names = [f"custom_tensor_{i}" for i in range(8)]
    blocks = partition_tensors(names, 4)
    sizes = block_sizes(blocks)
    assert sum(sizes) == 8
    assert all(size > 0 for size in sizes)


# ------------------------------------------------------------------ indexing
def test_build_block_index_and_sizes():
    blocks = partition_tensors(LLAMA_NAMES, 4)
    index = build_block_index(blocks)
    assert set(index) == set(LLAMA_NAMES)
    assert index["lm_head.weight"] == 3
    assert index["model.embed_tokens.weight"] == 0
    assert block_sizes(blocks) == tuple(len(b.tensor_names) for b in blocks)


def test_block_index_is_consistent_with_partition_membership():
    blocks = partition_tensors(LLAMA_NAMES, 3)
    index = build_block_index(blocks)
    for block in blocks:
        for name in block.tensor_names:
            assert index[name] == block.index