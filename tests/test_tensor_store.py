"""Tensor stores: the access layer shared by the mock and the real merge mode."""

from __future__ import annotations

import json
import os

import pytest
import torch

from zeroai.merge.tensor_store import (
    InMemoryTensorStore,
    SafetensorsTensorStore,
    TensorStore,
    common_tensor_names,
)

safetensors = pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402


# ------------------------------------------------------------- base contract
def test_base_store_is_not_usable_directly():
    store = TensorStore()
    with pytest.raises(NotImplementedError):
        store.names()
    with pytest.raises(NotImplementedError):
        store.get("anything")
    assert store.close() is None


# ------------------------------------------------------------- in-memory
def test_in_memory_store_exposes_names_and_tensors():
    store = InMemoryTensorStore({"a": torch.ones(2), "b": torch.zeros(3)})
    assert set(store.names()) == {"a", "b"}
    assert store.has("a") and not store.has("c")
    assert torch.equal(store.get("a"), torch.ones(2))


def test_in_memory_store_raises_a_named_key_error():
    store = InMemoryTensorStore({"a": torch.ones(2)})
    with pytest.raises(KeyError, match="not found in in-memory store"):
        store.get("missing")


def test_in_memory_store_refreshes_its_name_cache_after_set():
    store = InMemoryTensorStore({"a": torch.ones(2)})
    assert store.has("a") and not store.has("b")
    store.set("b", torch.zeros(2))
    assert store.has("b")
    assert set(store.names()) == {"a", "b"}


def test_in_memory_store_copies_the_mapping_it_was_given():
    tensors = {"a": torch.ones(2)}
    store = InMemoryTensorStore(tensors)
    store.set("b", torch.zeros(2))
    assert "b" not in tensors
    assert set(store.to_dict()) == {"a", "b"}


# ------------------------------------------------------------- safetensors
@pytest.fixture()
def checkpoint_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    save_file(
        {"model.layers.0.q_proj.weight": torch.ones(4, 4), "model.embed_tokens.weight": torch.zeros(4, 4)},
        str(path / "model-00001-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    save_file(
        {"model.layers.1.q_proj.weight": torch.full((4, 4), 2.0)},
        str(path / "model-00002-of-00002.safetensors"),
        metadata={"format": "pt"},
    )
    index = {
        "metadata": {"total_size": 192},
        "weight_map": {
            "model.layers.0.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.q_proj.weight": "model-00002-of-00002.safetensors",
        },
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"hidden_size": 4}), encoding="utf-8")
    return str(path)


def test_safetensors_store_lists_every_tensor_across_shards(checkpoint_dir):
    store = SafetensorsTensorStore(checkpoint_dir)
    assert store.names() == (
        "model.embed_tokens.weight",
        "model.layers.0.q_proj.weight",
        "model.layers.1.q_proj.weight",
    )
    store.close()


def test_safetensors_store_loads_tensors_from_the_right_shard(checkpoint_dir):
    store = SafetensorsTensorStore(checkpoint_dir)
    assert torch.equal(store.get("model.layers.1.q_proj.weight"), torch.full((4, 4), 2.0))
    assert torch.equal(store.get("model.embed_tokens.weight"), torch.zeros(4, 4))
    store.close()


def test_safetensors_store_caches_repeated_reads(checkpoint_dir):
    store = SafetensorsTensorStore(checkpoint_dir)
    first = store.get("model.layers.0.q_proj.weight")
    second = store.get("model.layers.0.q_proj.weight")
    assert first is second
    store.close()


def test_safetensors_store_can_disable_the_cache(checkpoint_dir):
    store = SafetensorsTensorStore(checkpoint_dir, cache=False)
    assert store.get("model.layers.0.q_proj.weight") is not store.get("model.layers.0.q_proj.weight")
    store.close()


def test_safetensors_store_raises_for_an_unknown_tensor(checkpoint_dir):
    store = SafetensorsTensorStore(checkpoint_dir)
    with pytest.raises(KeyError, match="not found"):
        store.get("model.layers.9.q_proj.weight")
    store.close()


def test_safetensors_store_accepts_a_single_file(tmp_path):
    file = tmp_path / "weights.safetensors"
    save_file({"a.weight": torch.ones(2, 2)}, str(file))
    store = SafetensorsTensorStore(str(file))
    assert store.names() == ("a.weight",)
    store.close()


def test_safetensors_store_rejects_a_directory_without_shards(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no .safetensors files"):
        SafetensorsTensorStore(str(empty))


def test_safetensors_store_rejects_a_non_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory or .safetensors file"):
        SafetensorsTensorStore(str(tmp_path / "does-not-exist"))


def test_safetensors_store_rejects_an_index_without_shards(tmp_path):
    path = tmp_path / "blank"
    path.mkdir()
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": {}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="no .safetensors files"):
        SafetensorsTensorStore(str(path))


# ----------------------------------------------------------- common names
def test_common_tensor_names_intersects_and_sorts():
    a = InMemoryTensorStore({"b.weight": torch.ones(1), "a.weight": torch.ones(1)})
    b = InMemoryTensorStore({"a.weight": torch.ones(1), "c.weight": torch.ones(1)})
    assert common_tensor_names([a, b]) == ["a.weight"]


def test_common_tensor_names_of_no_stores():
    assert common_tensor_names([]) == []


def test_common_tensor_names_of_a_single_store():
    store = InMemoryTensorStore({"z": torch.ones(1), "a": torch.ones(1)})
    assert common_tensor_names([store]) == ["a", "z"]


def test_common_tensor_names_drops_tensors_missing_from_one_expert():
    base = InMemoryTensorStore({f"t{i}": torch.ones(1) for i in range(5)})
    expert = InMemoryTensorStore({f"t{i}": torch.ones(1) for i in range(3)})
    assert common_tensor_names([base, expert]) == ["t0", "t1", "t2"]