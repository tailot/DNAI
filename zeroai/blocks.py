"""Mapping of tensor names onto fusion "blocks".

A *block* is a contiguous group of layers that receives its own fusion
coefficient.  This is what turns a single global weight into a per-layer-group
chromosome segment, exactly as described in section 3.1 of the technical spec.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from zeroai.schemas import BlockSpec

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_TRANSFORMER_BLOCK_RE = re.compile(r"(?:^|\.)(?:h|block|decoder_layer)\.(\d+)\.")
# ZeroAI's own synthetic/mock tensors are named ``block_<n>.t<j>``; recognising
# that naming keeps the mock world's block layout identical to the one the
# synthetic evaluator scores, so per-block coefficients really do matter.
_INDEXED_BLOCK_RE = re.compile(r"(?:^|\.)block[_-]?(\d+)\.")
_HEAD_RE = re.compile(r"(?:^|\.)lm_head\.|^output\.|(?:^|\.)final_dense\.|classifier")
_EMBED_RE = re.compile(r"embed_tokens|word_embeddings|\bwte\b|embeddings\.|tok_embeddings")


def layer_index(name: str) -> Optional[int]:
    """Return the transformer layer index of ``name`` or ``None`` for non-layer tensors."""
    match = (
        _LAYER_RE.search(name)
        or _TRANSFORMER_BLOCK_RE.search(name)
        or _INDEXED_BLOCK_RE.search(name)
    )
    if match:
        return int(match.group(1))
    return None


def is_head_tensor(name: str) -> bool:
    return bool(_HEAD_RE.search(name))


def is_embedding_tensor(name: str) -> bool:
    return bool(_EMBED_RE.search(name)) and not is_head_tensor(name)


def partition_tensors(names: Iterable[str], num_blocks: int) -> Tuple[BlockSpec, ...]:
    """Split ``names`` into ``num_blocks`` contiguous blocks ordered by depth.

    Embeddings are pinned to the first block, output heads to the last one and
    transformer layers are spread proportionally across the whole depth range.
    """
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")

    names = sorted(set(names))
    if not names:
        raise ValueError("partition_tensors requires at least one tensor name")

    indices = {name: layer_index(name) for name in names}
    layer_ids = sorted({i for i in indices.values() if i is not None})
    max_id = layer_ids[-1] if layer_ids else 0

    buckets: List[List[str]] = [[] for _ in range(num_blocks)]
    unknown: List[str] = []
    for name in names:
        idx = indices[name]
        if idx is None:
            if is_head_tensor(name):
                buckets[num_blocks - 1].append(name)
            elif is_embedding_tensor(name):
                buckets[0].append(name)
            else:
                unknown.append(name)
        else:
            block = min(num_blocks - 1, int((idx + 1) / (max_id + 1) * num_blocks) - 1)
            block = max(0, block)
            buckets[block].append(name)

    # Names whose depth we cannot infer (exotic or custom checkpoints) are spread
    # evenly over the blocks instead of piling up in block 0, so that the number
    # of blocks keeps meaning something for every model layout.
    if unknown:
        total = len(unknown)
        for position, name in enumerate(sorted(unknown)):
            buckets[min(num_blocks - 1, (position * num_blocks) // total)].append(name)

    return tuple(
        BlockSpec(index=i, name=f"block_{i}", tensor_names=tuple(bucket))
        for i, bucket in enumerate(buckets)
    )


def build_block_index(blocks: Sequence[BlockSpec]) -> Dict[str, int]:
    """Reverse lookup: tensor name -> block index."""
    index: Dict[str, int] = {}
    for block in blocks:
        for name in block.tensor_names:
            index[name] = block.index
    return index


def block_sizes(blocks: Sequence[BlockSpec]) -> Tuple[int, ...]:
    return tuple(len(b.tensor_names) for b in blocks)