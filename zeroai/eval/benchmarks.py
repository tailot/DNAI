"""Multi-domain benchmark tasks used by the real-model evaluator.

Every item is (prompt, checker).  Checkers are deterministic and offline:
numeric answers are parsed and compared, code is executed in a sandboxed
subprocess, and general-instruction items are validated by structural checks.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from zeroai.schemas import DomainSpec

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_python_block(text: str) -> str:
    """Pull the python code out of a model completion."""
    match = _CODE_FENCE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def extract_last_number(text: str) -> float | None:
    numbers = _NUMBER_RE.findall(text or "")
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


class Checker(ABC):
    """Turns a raw completion into a score in [0, 1]."""

    @abstractmethod
    def score(self, completion: str) -> float:
        raise NotImplementedError


class NumericAnswerChecker(Checker):
    def __init__(self, expected: float, tolerance: float = 1e-4):
        self.expected = float(expected)
        self.tolerance = float(tolerance)

    def score(self, completion: str) -> float:
        value = extract_last_number(completion)
        if value is None:
            return 0.0
        return 1.0 if abs(value - self.expected) <= self.tolerance else 0.0


class PythonExecChecker(Checker):
    """Executes the produced snippet together with its unit assertions."""

    def __init__(self, test_code: str, timeout: int = 10):
        self.test_code = test_code
        self.timeout = timeout

    def score(self, completion: str) -> float:
        code = extract_python_block(completion)
        if not code:
            return 0.0
        script = code + "\n" + self.test_code
        handle = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as handle:
                handle.write(script)
                path = handle.name
            completed = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return 1.0 if completed.returncode == 0 else 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0
        finally:
            if handle is not None:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass


class InstructionChecker(Checker):
    """Structural stability check: length band, required markers, no refusal."""

    REFUSALS = (
        "i cannot",
        "i can't",
        "as an ai",
        "i am unable",
        "non posso",
        "mi dispiace",
    )

    def __init__(
        self,
        min_chars: int = 20,
        max_chars: int = 4000,
        must_contain: Sequence[str] = (),
        forbid_refusal: bool = True,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.must_contain = tuple(must_contain)
        self.forbid_refusal = forbid_refusal

    def score(self, completion: str) -> float:
        text = (completion or "").strip()
        if len(text) < self.min_chars or len(text) > self.max_chars:
            return 0.0
        lowered = text.lower()
        if self.forbid_refusal and any(marker in lowered for marker in self.REFUSALS):
            return 0.0
        if self.must_contain:
            hits = sum(1 for token in self.must_contain if token.lower() in lowered)
            return hits / len(self.must_contain)
        return 1.0


@dataclass(frozen=True)
class BenchmarkItem:
    name: str
    domain: str
    prompt: str
    checker: Checker
    system: str = ""


@dataclass
class BenchmarkSuite:
    items: Tuple[BenchmarkItem, ...]
    domains: Tuple[DomainSpec, ...] = (
        DomainSpec("reasoning", 0.35),
        DomainSpec("code", 0.35),
        DomainSpec("general", 0.30),
    )

    def by_domain(self) -> Dict[str, List[BenchmarkItem]]:
        grouped: Dict[str, List[BenchmarkItem]] = {}
        for item in self.items:
            grouped.setdefault(item.domain, []).append(item)
        return grouped

    def __len__(self) -> int:
        return len(self.items)


def default_suite() -> BenchmarkSuite:
    """A compact, self-contained benchmark battery covering the three metric families."""
    items: List[BenchmarkItem] = [
        BenchmarkItem(
            name="logic_arith_1",
            domain="reasoning",
            prompt=(
                "A train travels 180 km in 3 hours, then 240 km in 4 hours. "
                "What is the average speed in km/h over the whole trip? "
                "Answer with a single number."
            ),
            checker=NumericAnswerChecker(60.0),
        ),
        BenchmarkItem(
            name="logic_ratio_2",
            domain="reasoning",
            prompt=(
                "A recipe needs 250 g of flour for 4 pancakes. How many grams "
                "are needed for 10 pancakes? Answer with a single number."
            ),
            checker=NumericAnswerChecker(625.0),
        ),
        BenchmarkItem(
            name="math_sequence_3",
            domain="reasoning",
            prompt=(
                "The sequence is 2, 5, 11, 23, 47. Each term is 2*previous + 1. "
                "What is the next term? Answer with a single number."
            ),
            checker=NumericAnswerChecker(95.0),
        ),
        BenchmarkItem(
            name="math_percentage_4",
            domain="reasoning",
            prompt="A price rises from 80 to 94. What is the percentage increase? Answer with a single number.",
            checker=NumericAnswerChecker(17.5, tolerance=0.01),
        ),
        BenchmarkItem(
            name="code_sum_squares",
            domain="code",
            prompt=(
                "Write a Python function `sum_squares(n)` returning the sum of the "
                "squares of 1..n. Return only a Python code block."
            ),
            checker=PythonExecChecker(
                "assert sum_squares(3) == 14\nassert sum_squares(1) == 1\nassert sum_squares(0) == 0\n"
            ),
        ),
        BenchmarkItem(
            name="code_fizzbuzz",
            domain="code",
            prompt=(
                "Write a Python function `fizzbuzz(n)` returning a list of strings for 1..n "
                "using the usual FizzBuzz rules. Return only a Python code block."
            ),
            checker=PythonExecChecker(
                "r = fizzbuzz(15)\nassert r[0] == '1'\nassert r[2] == 'Fizz'\n"
                "assert r[4] == 'Buzz'\nassert r[14] == 'FizzBuzz'\nassert len(r) == 15\n"
            ),
        ),
        BenchmarkItem(
            name="code_reverse_words",
            domain="code",
            prompt=(
                "Write a Python function `reverse_words(s)` that reverses the order of the "
                "words in a string. Return only a Python code block."
            ),
            checker=PythonExecChecker(
                "assert reverse_words('a b c') == 'c b a'\nassert reverse_words('') == ''\n"
            ),
        ),
        BenchmarkItem(
            name="general_explain",
            domain="general",
            prompt="Explain in three short sentences what a hash table is and when to use one.",
            checker=InstructionChecker(min_chars=80, max_chars=2500),
        ),
        BenchmarkItem(
            name="general_rewrite",
            domain="general",
            prompt="Rewrite politely: 'Send me the file now, I need it.'",
            checker=InstructionChecker(min_chars=25, max_chars=1200),
        ),
        BenchmarkItem(
            name="general_format",
            domain="general",
            prompt="List three advantages of local inference. Format as a bullet list using '-' markers.",
            checker=InstructionChecker(min_chars=40, max_chars=1500, must_contain=["-"]),
        ),
    ]
    return BenchmarkSuite(items=tuple(items))