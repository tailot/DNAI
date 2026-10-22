"""Evaluator contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from zeroai.schemas import EvaluationReport, MergedArtifact


class Evaluator(ABC):
    """Scores one merged candidate and returns the fitness signal for the optimiser.

    Implementations must be deterministic for a given artifact: the optimiser
    compares scores across candidates, and any run-to-run noise in the fitness
    function directly degrades convergence.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(self, artifact: MergedArtifact) -> EvaluationReport:
        """Return the aggregated fitness plus the per-domain breakdown."""

    def close(self) -> None:
        """Release heavy resources (loaded models, subprocesses)."""
        return None