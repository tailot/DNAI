"""Vector-space merge primitives: Trimming, Elect Sign, DARE.

The implementations mirror mergekit's ``mergekit.sparsify`` semantics so that a
merge executed by :class:`zeroai.merge.backend.NativeMergeBackend` and the same
merge executed by mergekit produce the same tensors:

* ``density`` is always the **fraction of coefficients retained** (never the
  fraction dropped).  mergekit 0.1.x uses the same convention for both
  ``magnitude`` and ``bernoulli`` sparsification.
* masked tensors are rescaled so that the chosen norm matches the original
  task vector (``l1`` / ``l2`` / ``linf``), which is mergekit's
  ``rescale_norm`` behaviour and the DARE paper's "REscale" step.

All arithmetic is accumulated in float32 regardless of the storage dtype, which
is what mergekit does internally and what keeps float16 merges stable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

WORK_DTYPE = torch.float32

VALID_NORMS = (None, "l1", "l2", "linf")


def _rescaled_masked_tensor(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    norm: Optional[str],
    eps: float = 1e-7,
) -> torch.Tensor:
    """Apply ``mask`` then restore the requested norm to its pre-mask value."""
    masked = tensor * mask
    if norm is None:
        return masked
    if norm == "l1":
        before, after = tensor.abs().sum(), masked.abs().sum()
    elif norm == "l2":
        before, after = tensor.norm(), masked.norm()
    elif norm == "linf":
        before, after = tensor.abs().max(), masked.abs().max()
    else:
        raise NotImplementedError(f"unsupported rescale_norm: {norm!r}")
    if float(before) < eps or float(after) < eps:
        return masked
    return masked * (before / after)


def magnitude_trim(
    tensor: torch.Tensor, density: float, rescale_norm: Optional[str] = None
) -> torch.Tensor:
    """Keep the ``density`` largest-magnitude coefficients (TIES "trimming")."""
    if density >= 1:
        return tensor.clone()
    if density <= 0 or tensor.numel() == 0:
        return torch.zeros_like(tensor)

    k = max(1, int(density * tensor.numel()))
    magnitudes = tensor.abs().reshape(-1)
    if magnitudes.device.type == "cpu":
        magnitudes = magnitudes.float()
    topk = torch.argsort(magnitudes, descending=True)[:k]

    mask = torch.zeros(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    mask[topk] = 1
    return _rescaled_masked_tensor(tensor, mask.reshape_as(tensor), rescale_norm)


def _bernoulli_mask(
    work: torch.Tensor,
    density: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Bernoulli mask shaped like ``work``, drawn on the generator's own device.

    torch refuses a CPU generator for CUDA tensors ("Expected a 'cuda' device
    type for generator but found 'cpu'"), which used to make every DARE method
    fail under ``--device cuda``.  Drawing the mask where the generator lives and
    moving it afterwards keeps DARE usable on GPU *and* keeps the drop pattern
    identical, for a given seed, on cpu and cuda.
    """
    if generator is not None and generator.device.type != work.device.type:
        mask = torch.bernoulli(
            torch.full(
                (work.numel(),), float(density), device=generator.device, dtype=WORK_DTYPE
            ),
            generator=generator,
        )
        return mask.to(device=work.device, dtype=work.dtype).reshape_as(work)
    return torch.bernoulli(torch.full_like(work, float(density)), generator=generator)


def bernoulli_drop(
    tensor: torch.Tensor,
    density: float,
    rescale_norm: Optional[str] = "l2",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Stochastically drop coefficients, keeping ``density`` of them (DARE)."""
    if density >= 1:
        return tensor.clone()
    if tensor.numel() == 0:
        return torch.zeros_like(tensor)

    # torch.bernoulli is not implemented for float16 on CPU -> upcast.
    work_dtype = tensor.dtype if (tensor.device.type != "cpu" or tensor.dtype == torch.bfloat16) else WORK_DTYPE
    work = tensor.to(work_dtype)
    mask = _bernoulli_mask(work, density, generator)
    return _rescaled_masked_tensor(work, mask, rescale_norm).to(tensor.dtype)


def elect_sign(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """Weighted-majority consensus sign for every coefficient (TIES "elect sign")."""
    if not tensors:
        raise ValueError("elect_sign requires at least one tensor")
    accumulator = torch.zeros_like(tensors[0], dtype=WORK_DTYPE)
    for tensor, weight in zip(tensors, weights):
        accumulator += float(weight) * tensor.to(WORK_DTYPE)
    # Zero-sum ties are resolved towards +1 so that a coefficient is never
    # silently discarded by an ambiguous vote.
    return torch.where(accumulator >= 0, 1.0, -1.0)


def linear_combine(tensors: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    """Plain weighted sum of task vectors."""
    if len(tensors) != len(weights):
        raise ValueError("tensors and weights must have the same length")
    if not tensors:
        raise ValueError("linear_combine requires at least one tensor")
    accumulator = torch.zeros_like(tensors[0], dtype=WORK_DTYPE)
    for tensor, weight in zip(tensors, weights):
        accumulator = accumulator + float(weight) * tensor.to(WORK_DTYPE)
    return accumulator


def ties_combine(
    tensors: Sequence[torch.Tensor],
    weights: Sequence[float],
    density: float,
    rescale_norm: Optional[str] = None,
) -> torch.Tensor:
    """Trim -> elect sign -> discard sign outliers -> weighted sum."""
    trimmed = [magnitude_trim(tensor, density, rescale_norm) for tensor in tensors]
    consensus = elect_sign(trimmed, weights)
    aligned = [tensor * (torch.sign(tensor) == consensus).to(WORK_DTYPE) for tensor in trimmed]
    return linear_combine(aligned, weights)


def dare_rescale(
    tensor: torch.Tensor,
    keep_density: float,
    scale: float = 1.0,
    rescale_norm: Optional[str] = "l2",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Drop-And-REscale a single task vector, then apply the extra ``scale`` gene."""
    dropped = bernoulli_drop(tensor, keep_density, rescale_norm=rescale_norm, generator=generator)
    if scale != 1.0:
        dropped = dropped * float(scale)
    return dropped


def combine(
    method: "MergeMethodLike",
    deltas: Sequence[torch.Tensor],
    weights: Sequence[float],
    *,
    density: float = 1.0,
    dare_keep: float = 1.0,
    dare_scale: float = 1.0,
    trim_norm: Optional[str] = None,
    dare_norm: Optional[str] = "l2",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Dispatch a per-tensor merge for one of the supported strategies."""
    from zeroai.schemas import MergeMethod

    if not deltas:
        raise ValueError("combine requires at least one task vector")
    if len(deltas) != len(weights):
        raise ValueError("deltas and weights must have the same length")

    method = MergeMethod(method)

    prepared: List[torch.Tensor] = list(deltas)
    if method.uses_dare:
        prepared = [
            dare_rescale(
                tensor, dare_keep, dare_scale, rescale_norm=dare_norm, generator=generator
            )
            for tensor in prepared
        ]

    if method.uses_sign_elect:
        return ties_combine(prepared, weights, density, rescale_norm=trim_norm)
    if method.uses_trim:  # taskslim
        trimmed = [magnitude_trim(tensor, density, rescale_norm=trim_norm) for tensor in prepared]
        return linear_combine(trimmed, weights)
    return linear_combine(prepared, weights)


def interference_energy(
    contributions: Sequence[torch.Tensor],
) -> tuple:
    """Return ``(cross_energy, self_energy)`` for a set of weighted contributions.

    ``cross_energy`` is the sum of the pairwise dot products between the
    contributions of different models.  Negative values are exactly the
    "destructive interference" the TIES/DARE machinery is meant to remove.
    """
    if not contributions:
        return 0.0, 0.0
    total = torch.zeros_like(contributions[0], dtype=WORK_DTYPE)
    self_energy = 0.0
    for contribution in contributions:
        contribution = contribution.to(WORK_DTYPE)
        total = total + contribution
        self_energy += float((contribution * contribution).sum())
    total_energy = float((total * total).sum())
    cross_energy = 0.5 * (total_energy - self_energy)
    return cross_energy, self_energy