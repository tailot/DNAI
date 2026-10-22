"""Real-model evaluator: runs the benchmark battery against a merged checkpoint.

Two inference backends are supported:

* ``transformers`` - always available, greedy batched generation;
* ``vllm``         - used automatically when installed, much faster for the
  many-candidate loop described in section 4 of the spec.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from zeroai.eval.base import Evaluator
from zeroai.eval.benchmarks import BenchmarkItem, BenchmarkSuite, default_suite
from zeroai.schemas import EvaluationReport, MetricScore, MergedArtifact

LOG = logging.getLogger(__name__)


class LLMEvaluator(Evaluator):
    """Compile-and-test loop for real merged checkpoints (spec section 3.3)."""

    name = "llm"

    def __init__(
        self,
        suite: Optional[BenchmarkSuite] = None,
        device: str = "cuda",
        dtype: str = "float16",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        backend: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
    ):
        self.suite = suite or default_suite()
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.trust_remote_code = trust_remote_code
        self.use_chat_template = use_chat_template
        self.backend = self._resolve_backend(backend)
        self._model = None
        self._tokenizer = None
        self._loaded_from = None

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        if requested in ("transformers", "vllm"):
            return requested
        try:
            import vllm  # noqa: F401

            return "vllm"
        except Exception:
            return "transformers"

    # ------------------------------------------------------------------ loading
    def _load(self, path: str) -> None:
        if self._loaded_from == path and self._model is not None:
            return
        self.close()
        if self.backend == "vllm":
            from vllm import LLM, SamplingParams  # noqa: F401

            self._model = LLM(
                model=path,
                trust_remote_code=self.trust_remote_code,
                dtype=self.dtype,
            )
            self._tokenizer = self._model.get_tokenizer()
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = getattr(torch, self.dtype, torch.float16)
            self._tokenizer = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=self.trust_remote_code,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
            )
            self._model.eval()
        self._loaded_from = path

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded_from = None
        try:
            import gc

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    # --------------------------------------------------------------- generation
    def _generate(self, items: List[BenchmarkItem]) -> List[str]:
        prompts = [self._render(item) for item in items]
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
            )
            outputs = self._model.generate(prompts, params)
            return [output.outputs[0].text for output in outputs]

        import torch

        completions: List[str] = []
        for item, prompt in zip(items, prompts):
            encoded = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.inference_mode():
                output = self._model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p if self.temperature > 0 else None,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            generated = output[0][ encoded["input_ids"].shape[1] :]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            completions.append(text.strip())
        return completions

    def _render(self, item: BenchmarkItem) -> str:
        messages = []
        if item.system:
            messages.append({"role": "system", "content": item.system})
        messages.append({"role": "user", "content": item.prompt})
        if self.use_chat_template and getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return item.prompt

    # ------------------------------------------------------------------ scoring
    def evaluate(self, artifact: MergedArtifact) -> EvaluationReport:
        if not artifact.output_path:
            raise ValueError("LLMEvaluator needs a merged checkpoint on disk")

        self._load(artifact.output_path)
        grouped = self.suite.by_domain()
        metrics: List[MetricScore] = []
        weighted_total = 0.0
        weight_sum = 0.0
        total_items = 0
        total_score = 0.0

        for domain in self.suite.domains:
            items = grouped.get(domain.name, [])
            if not items:
                continue
            completions = self._generate(items)
            scores = [max(0.0, min(1.0, item.checker.score(text))) for item, text in zip(items, completions)]
            domain_score = sum(scores) / len(scores)
            weighted_total += domain.weight * domain_score
            weight_sum += domain.weight
            total_items += len(scores)
            total_score += sum(scores)
            metrics.append(
                MetricScore(
                    name=f"domain_{domain.name}",
                    value=domain_score,
                    weight=domain.weight,
                )
            )
            LOG.info(
                "candidate %s | %s = %.3f (%d items)",
                artifact.candidate_id,
                domain.name,
                domain_score,
                len(items),
            )

        if weight_sum == 0.0:
            raise ValueError("benchmark suite contains no scored domains")

        metrics.append(MetricScore("benchmark_pass_rate", total_score / max(1, total_items), weight=0.0))
        return EvaluationReport(
            fitness=weighted_total / weight_sum,
            metrics=tuple(metrics),
            candidate_id=artifact.candidate_id,
            notes={"evaluator": f"llm/{self.backend}", "items": float(total_items)},
        )