"""Model-mode evaluator: a fully specified, dependency-free benchmark world.

The mock world is not a random-number toy.  It instantiates the exact failure
modes the real pipeline has to fight, so the evolutionary loop can be validated
end to end before a single checkpoint is downloaded:

* every benchmark **domain** (reasoning / code / math / general) has a ground
  truth task vector with its own *depth profile* - some capabilities live in
  early blocks, others in late ones, so per-block coefficients genuinely matter;
* input experts are noisy, partially overlapping, sometimes **sign-conflicting**
  approximations of those ground truths - which is what elect-sign must fix;
* experts carry independent noise, so naive weight summation amplifies junk
  directions - which is what trimming and DARE must suppress.

Metrics:

``capability``   cosine alignment of the merged task vector with each domain
``purity``       fraction of the merged delta that lies inside the capability
                 subspace (1 - off-manifold noise energy)
``stability``    how close the merged delta norm is to the ideal one
                 (over-amplification is what destroys general ability)
``cooperation``  1 - destructive interference between the weighted experts
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from zeroai.eval.base import Evaluator
from zeroai.merge.algorithms import WORK_DTYPE, interference_energy
from zeroai.merge.tensor_store import InMemoryTensorStore, TensorStore
from zeroai.schemas import DomainSpec, EvaluationReport, MetricScore, MergedArtifact

DEFAULT_DOMAINS: Tuple[DomainSpec, ...] = (
    DomainSpec("reasoning", 0.30, "logical deduction and problem solving"),
    DomainSpec("code", 0.25, "syntax and script completion"),
    DomainSpec("math", 0.20, "arithmetic and symbolic manipulation"),
    DomainSpec("general", 0.25, "conversational stability / generalisation"),
)


@dataclass
class SyntheticBenchmark:
    """A generated merge world: base model, ground-truth capabilities, experts."""

    num_blocks: int
    tensors_per_block: int
    tensor_shape: Tuple[int, int]
    domains: Tuple[DomainSpec, ...]
    base: Dict[str, torch.Tensor]
    truth: Dict[str, Dict[str, torch.Tensor]]
    expert_deltas: Dict[str, Dict[str, torch.Tensor]]
    expert_coverage: Dict[str, Dict[str, float]]
    domain_depth: Dict[str, Tuple[float, ...]]
    seed: int = 1337
    noise_sigma: float = 0.35
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "capability": 0.55,
            "purity": 0.15,
            "stability": 0.15,
            "cooperation": 0.15,
        }
    )

    @property
    def tensor_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self.base.keys()))

    @property
    def expert_labels(self) -> Tuple[str, ...]:
        return tuple(self.expert_deltas.keys())

    def block_names(self) -> Dict[int, Tuple[str, ...]]:
        out: Dict[int, List[str]] = {}
        for name in self.tensor_names:
            block = int(name.split(".")[0].split("_")[1])
            out.setdefault(block, []).append(name)
        return {b: tuple(sorted(v)) for b, v in out.items()}


def _depth_profile(domain: str, num_blocks: int, rng: torch.Generator) -> Tuple[float, ...]:
    """Where a capability lives along the depth of the network."""
    centres = {
        "general": 0.5,
        "reasoning": 0.65,
        "code": 0.35,
        "math": 0.5,
    }
    centre = centres.get(domain, 0.5)
    width = 0.35 if domain != "general" else 0.6
    profile: List[float] = []
    for b in range(num_blocks):
        t = (b + 0.5) / num_blocks
        profile.append(math.exp(-((t - centre) ** 2) / (2 * width * width)) + 0.15)
    total = sum(profile)
    return tuple(p / total for p in profile)


def build_synthetic_world(
    num_models: int = 3,
    num_blocks: int = 4,
    tensors_per_block: int = 2,
    tensor_shape: Tuple[int, int] = (32, 32),
    domains: Sequence[DomainSpec] = DEFAULT_DOMAINS,
    seed: int = 1337,
    noise_sigma: float = 0.35,
    conflict_rate: float = 0.25,
    expert_labels: Sequence[str] = (),
) -> Tuple[SyntheticBenchmark, InMemoryTensorStore, List[TensorStore]]:
    """Create a deterministic merge world plus the stores the merge engine reads."""
    if num_models < 1 or num_blocks < 1 or tensors_per_block < 1:
        raise ValueError("num_models, num_blocks and tensors_per_block must all be >= 1")

    domains = tuple(domains)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    names = [f"block_{b}.t{j}" for b in range(num_blocks) for j in range(tensors_per_block)]

    base = {n: torch.randn(tensor_shape, generator=rng) * 0.10 for n in names}

    # A shared core makes the domains partially overlapping, exactly like real
    # capabilities are not orthogonal in weight space.
    core = {n: torch.randn(tensor_shape, generator=rng) for n in names}

    truth: Dict[str, Dict[str, torch.Tensor]] = {}
    domain_depth: Dict[str, Tuple[float, ...]] = {}
    for domain in domains:
        depth = _depth_profile(domain.name, num_blocks, rng)
        domain_depth[domain.name] = depth
        strength = {"general": 0.8, "reasoning": 1.0, "code": 0.95, "math": 0.9}.get(domain.name, 1.0)
        truth[domain.name] = {}
        for name in names:
            block = int(name.split(".")[0].split("_")[1])
            specific = torch.randn(tensor_shape, generator=rng)
            truth[domain.name][name] = strength * depth[block] * (
                0.45 * core[name] + 0.55 * specific
            )

    labels = list(expert_labels) or [f"expert_{i}" for i in range(num_models)]
    if len(labels) != num_models:
        raise ValueError("expert_labels must contain exactly num_models entries")

    expert_deltas: Dict[str, Dict[str, torch.Tensor]] = {}
    expert_coverage: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(labels):
        coverage: Dict[str, float] = {}
        for d, domain in enumerate(domains):
            # Each expert specialises on a rotating subset of the domains.
            affinity = 1.0 if (i + d) % len(domains) == 0 else 0.25 + 0.15 * ((i * d) % 3)
            coverage[domain.name] = float(affinity)

        expert_coverage[label] = coverage
        expert_deltas[label] = {}
        for name in names:
            block = int(name.split(".")[0].split("_")[1])
            delta = torch.zeros(tensor_shape, dtype=WORK_DTYPE)
            for domain in domains:
                coefficient = coverage[domain.name]
                # Deliberate sign conflicts on a fraction of (expert, block) pairs:
                # this is the vector conflict elect-sign has to resolve.
                if (i + block) % 4 == 0 and block % 2 == 1:
                    coefficient = -0.6 * coefficient
                delta = delta + coefficient * truth[domain.name][name]
            delta = delta + noise_sigma * torch.randn(tensor_shape, generator=rng)
            expert_deltas[label][name] = delta

    benchmark = SyntheticBenchmark(
        num_blocks=num_blocks,
        tensors_per_block=tensors_per_block,
        tensor_shape=tuple(tensor_shape),
        domains=domains,
        base=base,
        truth=truth,
        expert_deltas=expert_deltas,
        expert_coverage=expert_coverage,
        domain_depth=domain_depth,
        seed=seed,
        noise_sigma=noise_sigma,
    )

    base_store = InMemoryTensorStore(dict(base))
    model_stores: List[TensorStore] = []
    for label in benchmark.expert_labels:
        tensors = {n: base[n] + expert_deltas[label][n] for n in names}
        model_stores.append(InMemoryTensorStore(tensors))
    return benchmark, base_store, model_stores


class SyntheticEvaluator(Evaluator):
    """Scores a merged task vector against the ground-truth capability vectors."""

    name = "synthetic"

    def __init__(self, benchmark: SyntheticBenchmark, penalize_conflict: bool = True):
        self.benchmark = benchmark
        self.penalize_conflict = penalize_conflict

    # ------------------------------------------------------------------ scoring
    def _block_vector(self, tensors: Dict[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
        # Scoring the synthetic world is tiny and must stay comparable across
        # devices: every tensor (merged deltas may live on cuda) is brought back
        # to cpu / float32 before it is mixed with the ground-truth vectors.
        return torch.cat(
            [tensors[n].to(device="cpu", dtype=WORK_DTYPE).reshape(-1) for n in names]
        )

    def evaluate(self, artifact: MergedArtifact) -> EvaluationReport:
        if artifact.deltas is None:
            raise ValueError("SyntheticEvaluator needs artifact.deltas (native backend)")

        benchmark = self.benchmark
        plan = artifact.plan
        block_names = benchmark.block_names()
        metrics: List[MetricScore] = []

        capability_total = 0.0
        weight_sum = sum(d.weight for d in benchmark.domains) or 1.0

        for domain in benchmark.domains:
            depth = benchmark.domain_depth[domain.name]
            numerator = 0.0
            denominator = 0.0
            for block in range(benchmark.num_blocks):
                names = block_names.get(block, ())
                if not names:
                    continue
                merged = self._block_vector(artifact.deltas, names)
                truth = self._block_vector(benchmark.truth[domain.name], names)
                merged_norm = float(merged.norm())
                truth_norm = float(truth.norm())
                if merged_norm < 1e-12 or truth_norm < 1e-12:
                    cosine = 0.0
                else:
                    cosine = float(torch.dot(merged, truth) / (merged_norm * truth_norm))
                score = max(0.0, min(1.0, cosine))
                numerator += depth[block] * score
                denominator += depth[block]
            domain_score = numerator / denominator if denominator else 0.0
            capability_total += domain.weight * domain_score
            metrics.append(
                MetricScore(
                    name=f"domain_{domain.name}",
                    value=domain_score,
                    weight=domain.weight / weight_sum,
                )
            )

        capability = capability_total / weight_sum
        metrics.append(MetricScore("capability", capability, weight=1.0))

        purity = self._purity(artifact.deltas, block_names)
        metrics.append(MetricScore("purity", purity, weight=1.0))

        stability = self._stability(artifact.deltas, block_names)
        metrics.append(MetricScore("stability", stability, weight=1.0))

        cooperation = self._cooperation(plan, block_names)
        metrics.append(MetricScore("cooperation", cooperation, weight=1.0))

        weights = benchmark.fitness_weights
        fitness = (
            weights.get("capability", 0.0) * capability
            + weights.get("purity", 0.0) * purity
            + weights.get("stability", 0.0) * stability
            + weights.get("cooperation", 0.0) * cooperation
        )

        return EvaluationReport(
            fitness=float(fitness),
            metrics=tuple(metrics),
            candidate_id=artifact.candidate_id,
            notes={"evaluator": self.name},
        )

    # ---------------------------------------------------------------- internals
    def _purity(self, deltas: Dict[str, torch.Tensor], block_names: Dict[int, Tuple[str, ...]]) -> float:
        """1 - (energy outside the capability subspace) / (total energy)."""
        off_energy = 0.0
        total_energy = 0.0
        for block in range(self.benchmark.num_blocks):
            names = block_names.get(block, ())
            if not names:
                continue
            merged = self._block_vector(deltas, names)
            basis = torch.stack(
                [self._block_vector(self.benchmark.truth[d.name], names) for d in self.benchmark.domains]
            )
            total_energy += float((merged * merged).sum())
            try:
                projection = torch.linalg.lstsq(basis.T, merged).solution
                fitted = basis.T @ projection
            except Exception:
                fitted = torch.zeros_like(merged)
            residual = merged - fitted
            off_energy += float((residual * residual).sum())

        if total_energy <= 1e-12:
            return 0.0
        return max(0.0, 1.0 - off_energy / total_energy)

    def _stability(self, deltas: Dict[str, torch.Tensor], block_names: Dict[int, Tuple[str, ...]]) -> float:
        """Penalise merged deltas whose norm drifts from the ideal capability norm."""
        merged_sq = 0.0
        ideal_sq = 0.0
        for block in range(self.benchmark.num_blocks):
            names = block_names.get(block, ())
            if not names:
                continue
            merged = self._block_vector(deltas, names)
            ideal = torch.zeros_like(merged)
            for domain in self.benchmark.domains:
                ideal = ideal + domain.weight * self._block_vector(self.benchmark.truth[domain.name], names)
            scale = sum(d.weight for d in self.benchmark.domains) or 1.0
            ideal = ideal / scale
            merged_sq += float((merged * merged).sum())
            ideal_sq += float((ideal * ideal).sum())

        ideal_norm = math.sqrt(ideal_sq)
        if ideal_norm <= 1e-12:
            return 0.0
        deviation = abs(math.sqrt(merged_sq) - ideal_norm) / ideal_norm
        return max(0.0, 1.0 - min(1.0, deviation))

    def _cooperation(self, plan, block_names: Dict[int, Tuple[str, ...]]) -> float:
        """1 - destructive interference between the weighted expert contributions."""
        if not self.penalize_conflict:
            return 1.0
        labels = list(self.benchmark.expert_labels)
        cross_total = 0.0
        self_total = 0.0
        for block in range(self.benchmark.num_blocks):
            names = block_names.get(block, ())
            if not names:
                continue
            contributions = []
            for i, label in enumerate(labels):
                weight = plan.weights[i][block] if i < len(plan.weights) else 0.0
                vector = self._block_vector(self.benchmark.expert_deltas[label], names)
                contributions.append(weight * vector)
            cross, own = interference_energy(contributions)
            cross_total += cross
            self_total += own
        if self_total <= 1e-12:
            return 0.0
        destructive = max(0.0, -cross_total / self_total)
        return max(0.0, 1.0 - destructive)