"""Tensor access layer.

The merge engine never cares *where* a tensor comes from: an in-memory store
backs the synthetic/mock mode, a safetensors store backs real checkpoints.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import torch


class TensorStore:
    """Read-only view over a model's tensors."""

    def names(self) -> Sequence[str]:
        raise NotImplementedError

    def get(self, name: str) -> torch.Tensor:
        raise NotImplementedError

    def has(self, name: str) -> bool:
        return name in self.name_set()

    def name_set(self) -> frozenset:
        if not hasattr(self, "_name_set") or self._name_set is None:
            self._name_set = frozenset(self.names())
        return self._name_set

    def close(self) -> None:
        return None


class InMemoryTensorStore(TensorStore):
    """Store backed by a plain ``{name: torch.Tensor}`` mapping."""

    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self._tensors = dict(tensors)
        self._name_set: Optional[frozenset] = None

    def names(self) -> Sequence[str]:
        return tuple(self._tensors.keys())

    def get(self, name: str) -> torch.Tensor:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise KeyError(f"tensor {name!r} not found in in-memory store") from exc

    def set(self, name: str, tensor: torch.Tensor) -> None:
        self._tensors[name] = tensor
        self._name_set = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return dict(self._tensors)


class SafetensorsTensorStore(TensorStore):
    """Lazily loads tensors from a Hugging Face checkpoint directory.

    Only the tensors that are actually touched are materialised, and each is
    cached so a full evolutionary loop over many candidates does not re-read
    the same shards over and over.
    """

    def __init__(self, path: str, device: str = "cpu", cache: bool = True):
        from safetensors import safe_open

        self.path = os.path.abspath(path)
        self.device = device
        self.cache_enabled = cache
        self._cache: Dict[str, torch.Tensor] = {}
        self._files: List[str] = self._discover_files(self.path)
        self._index: Dict[str, str] = self._build_index(self.path, self._files)
        self._open_files: Dict[str, object] = {}
        self._name_set: Optional[frozenset] = None
        for file in self._files:
            with safe_open(file, framework="pt", device=device) as handle:
                for key in handle.keys():
                    self._index.setdefault(key, file)
        if not self._index:
            raise ValueError(f"no safetensors tensors found under {self.path!r}")

    @staticmethod
    def _discover_files(path: str) -> List[str]:
        if not os.path.isdir(path):
            if path.endswith(".safetensors"):
                return [path]
            raise FileNotFoundError(f"model path {path!r} is not a directory or .safetensors file")
        files = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".safetensors")
        )
        if not files:
            raise FileNotFoundError(f"no .safetensors files found under {path!r}")
        return files

    @staticmethod
    def _build_index(path: str, files: Sequence[str]) -> Dict[str, str]:
        index_path = os.path.join(path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return {}
        with open(index_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        mapping = payload.get("weight_map", {})
        return {key: os.path.join(path, value) for key, value in mapping.items()}

    def names(self) -> Sequence[str]:
        return tuple(sorted(self._index.keys()))

    def _handle(self, file: str):
        handle = self._open_files.get(file)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(file, framework="pt", device=self.device)
            self._open_files[file] = handle
        return handle

    def get(self, name: str) -> torch.Tensor:
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        file = self._index.get(name)
        if file is None:
            raise KeyError(f"tensor {name!r} not found in {self.path!r}")
        tensor = self._handle(file).get_tensor(name)
        if self.cache_enabled:
            self._cache[name] = tensor
        return tensor

    def close(self) -> None:
        self._open_files.clear()
        self._cache.clear()


def common_tensor_names(stores: Iterable[TensorStore]) -> List[str]:
    """Intersection of tensor names across stores, sorted for determinism."""
    stores = list(stores)
    if not stores:
        return []
    shared = set(stores[0].names())
    for store in stores[1:]:
        shared &= set(store.names())
    return sorted(shared)