"""The evolutionary chromosome.

A :class:`Genome` encodes every tunable knob of a merge:

* ``method``            -- merge strategy (linear / ties / dare_* / taskslim)
* ``weights``           -- per-model, per-block fusion coefficients
* ``density``           -- magnitude-trimming keep fraction (TIES / TaskSlim)
* ``dare_density``      -- Drop-And-REscale drop fraction
* ``dare_scale``        -- rescale factor applied to the surviving deltas
* ``normalize``         -- rescale each block's coefficients to sum to 1

Genomes are immutable so they can be hashed, stored in Optuna trials and used
as GA population members without defensive copying.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from zeroai.schemas import ALL_METHODS, DType, MergeMethod, SearchSpace


@dataclass(frozen=True)
class Genome:
    method: MergeMethod
    weights: Tuple[Tuple[float, ...], ...]
    density: float
    dare_density: float
    dare_scale: float
    normalize: bool = True
    dtype: DType = DType.FLOAT16

    # ------------------------------------------------------------------ shape
    @property
    def n_models(self) -> int:
        return len(self.weights)

    @property
    def n_blocks(self) -> int:
        return len(self.weights[0]) if self.weights else 0

    # --------------------------------------------------------------- identity
    def key(self) -> str:
        """Stable identity string, used to deduplicate candidate evaluations."""
        rows = ";".join(",".join(f"{w:.6f}" for w in row) for row in self.weights)
        return (
            f"{self.method.value}|{self.density:.6f}|{self.dare_density:.6f}|"
            f"{self.dare_scale:.6f}|{int(self.normalize)}|{rows}"
        )

    def clone(self) -> "Genome":
        return replace(self)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("Genome.weights must contain one row per model")
        widths = {len(row) for row in self.weights}
        if len(widths) != 1:
            raise ValueError("Genome.weights rows must all have the same block count")
        object.__setattr__(self, "weights", tuple(tuple(float(w) for w in row) for row in self.weights))

    # -------------------------------------------------------------- factories
    @classmethod
    def sample(cls, n_models: int, n_blocks: int, space: SearchSpace, rng: random.Random) -> "Genome":
        """Draw a uniformly random genome inside ``space``."""
        method = rng.choice(list(space.methods))
        weights = tuple(
            tuple(rng.uniform(space.weight_low, space.weight_high) for _ in range(n_blocks))
            for _ in range(n_models)
        )
        return cls(
            method=method,
            weights=weights,
            density=rng.uniform(space.density_low, space.density_high),
            dare_density=rng.uniform(space.dare_density_low, space.dare_density_high),
            dare_scale=rng.uniform(space.dare_scale_low, space.dare_scale_high),
            dtype=space.dtype,
        )

    @classmethod
    def uniform(cls, n_models: int, n_blocks: int, weight: Optional[float] = None) -> "Genome":
        """Every model contributes ``weight`` (default 1/n_models) on every block."""
        coefficient = 1.0 / max(1, n_models) if weight is None else float(weight)
        return cls(
            method=MergeMethod.LINEAR,
            weights=tuple(tuple(coefficient for _ in range(n_blocks)) for _ in range(n_models)),
            density=1.0,
            dare_density=0.0,
            dare_scale=1.0,
        )

    @classmethod
    def random(
        cls,
        n_models: int,
        n_blocks: int,
        space: Optional[SearchSpace] = None,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ) -> "Genome":
        """Convenience wrapper around :meth:`sample` with an optional RNG."""
        space = space or SearchSpace()
        if rng is None:
            rng = random.Random(seed)
        return cls.sample(n_models, n_blocks, space, rng)

    # ------------------------------------------------------------- operations
    def clipped(self, space: SearchSpace) -> "Genome":
        """Return a copy with every gene forced back inside the search space."""
        weights = tuple(
            tuple(min(max(w, space.weight_low), space.weight_high) for w in row)
            for row in self.weights
        )
        return replace(
            self,
            method=self.method if self.method in space.methods else space.methods[0],
            weights=weights,
            density=min(max(self.density, space.density_low), space.density_high),
            dare_density=min(max(self.dare_density, space.dare_density_low), space.dare_density_high),
            dare_scale=min(max(self.dare_scale, space.dare_scale_low), space.dare_scale_high),
            dtype=space.dtype,
        )

    def crossover(self, other: "Genome", rng: random.Random, alpha: float = 0.5) -> "Genome":
        """Blend crossover (per-gene BLX-alpha) + uniform crossover for discrete genes."""
        if other.n_models != self.n_models or other.n_blocks != self.n_blocks:
            raise ValueError("crossover requires genomes with identical shape")

        def blend(a: float, b: float) -> float:
            low, high = min(a, b), max(a, b)
            span = high - low
            return rng.uniform(low - alpha * span, high + alpha * span)

        weights = tuple(
            tuple(blend(a, b) for a, b in zip(row_self, row_other))
            for row_self, row_other in zip(self.weights, other.weights)
        )
        pick = lambda a, b: a if rng.random() < 0.5 else b  # noqa: E731
        return Genome(
            method=pick(self.method, other.method),
            weights=weights,
            density=blend(self.density, other.density),
            dare_density=blend(self.dare_density, other.dare_density),
            dare_scale=blend(self.dare_scale, other.dare_scale),
            normalize=pick(self.normalize, other.normalize),
            dtype=self.dtype,
        )

    def mutate(
        self,
        rng: random.Random,
        rate: float = 0.25,
        sigma: float = 0.15,
        methods: Optional[Sequence[MergeMethod]] = None,
    ) -> "Genome":
        """Gaussian mutation on continuous genes, random re-draw on the method gene."""
        weights = []
        for row in self.weights:
            new_row = []
            for w in row:
                if rng.random() < rate:
                    new_row.append(max(0.0, w + rng.gauss(0.0, sigma)))
                else:
                    new_row.append(w)
            weights.append(tuple(new_row))

        method = self.method
        if rng.random() < rate:
            method = rng.choice(list(methods or ALL_METHODS))

        def jitter(value: float, lo: float, hi: float) -> float:
            if rng.random() < rate:
                return min(max(value + rng.gauss(0.0, sigma), lo), hi)
            return value

        return Genome(
            method=method,
            weights=tuple(weights),
            density=jitter(self.density, 0.05, 1.0),
            dare_density=jitter(self.dare_density, 0.0, 0.95),
            dare_scale=jitter(self.dare_scale, 0.5, 4.0),
            normalize=self.normalize if rng.random() >= rate else not self.normalize,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------- (de)serial
    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method.value,
            "weights": [list(row) for row in self.weights],
            "density": self.density,
            "dare_density": self.dare_density,
            "dare_scale": self.dare_scale,
            "normalize": self.normalize,
            "dtype": self.dtype.value,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Genome":
        weights = tuple(tuple(float(w) for w in row) for row in payload["weights"])  # type: ignore[index]
        return cls(
            method=MergeMethod(str(payload["method"])),  # type: ignore[arg-type]
            weights=weights,
            density=float(payload["density"]),  # type: ignore[arg-type]
            dare_density=float(payload["dare_density"]),  # type: ignore[arg-type]
            dare_scale=float(payload["dare_scale"]),  # type: ignore[arg-type]
            normalize=bool(payload.get("normalize", True)),
            dtype=DType(str(payload.get("dtype", DType.FLOAT16.value))),  # type: ignore[arg-type]
        )

    def short_repr(self) -> str:
        rows = ", ".join("[{0}]".format(", ".join(f"{w:.2f}" for w in row)) for row in self.weights)
        return (
            f"{self.method.value} weights={rows} density={self.density:.2f} "
            f"dare_density={self.dare_density:.2f} dare_scale={self.dare_scale:.2f} "
            f"normalize={int(self.normalize)}"
        )


def genome_to_dict(genome: Genome) -> Dict[str, object]:
    """Module-level alias kept for scripts and tests."""
    return genome.to_dict()


def genome_from_dict(payload: Dict[str, object]) -> Genome:
    """Module-level alias kept for scripts and tests."""
    return Genome.from_dict(payload)


def normalize_weights(weights: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
    """Rescale each block column so the model coefficients sum to one.

    mergekit's ``normalize`` option works per-model; ZeroAI normalises *per block*
    because the whole point of block-wise coefficients is to keep the total
    contribution of each layer group bounded, which is what prevents the
    norm blow-up that destroys general conversational ability.
    """
    rows = [list(row) for row in weights]
    if not rows:
        return ()
    n_blocks = len(rows[0])
    out: List[List[float]] = [[] for _ in rows]
    for b in range(n_blocks):
        column = [rows[i][b] for i in range(len(rows))]
        total = sum(abs(c) for c in column)
        if total <= 1e-12:
            even = 1.0 / len(rows)
            for i in range(len(rows)):
                out[i].append(even)
            continue
        for i in range(len(rows)):
            out[i].append(column[i] / total)
    return tuple(tuple(row) for row in out)