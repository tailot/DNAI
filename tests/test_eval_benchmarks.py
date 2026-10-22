"""Offline benchmark battery: prompt parsing, checkers and the default suite."""

from __future__ import annotations

import pytest

from zeroai.eval.benchmarks import (
    BenchmarkItem,
    BenchmarkSuite,
    Checker,
    InstructionChecker,
    NumericAnswerChecker,
    PythonExecChecker,
    default_suite,
    extract_last_number,
    extract_python_block,
)


# ------------------------------------------------------------------ extraction
def test_extract_python_block_prefers_a_fenced_block():
    completion = "Here you go:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_python_block(completion) == "def f():\n    return 1"


def test_extract_python_block_accepts_the_short_fence_forms():
    assert extract_python_block("```py\nx = 2\n```") == "x = 2"
    assert extract_python_block("```\ny = 3\n```") == "y = 3"


def test_extract_python_block_falls_back_to_the_raw_completion():
    assert extract_python_block("def g(): return 3") == "def g(): return 3"
    assert extract_python_block("") == ""
    assert extract_python_block(None) == ""


def test_extract_last_number_uses_the_final_number_in_the_text():
    assert extract_last_number("180 km in 3 hours, so 60") == 60.0
    assert extract_last_number("the answer is -5.5") == -5.5
    assert extract_last_number("no digits here") is None
    assert extract_last_number("") is None


# -------------------------------------------------------------------- checkers
def test_numeric_checker_is_exact_within_tolerance():
    checker = NumericAnswerChecker(60.0)
    assert checker.score("The average speed is 60 km/h.") == 1.0
    assert checker.score("60.00005") == 1.0
    assert checker.score("59.9 km/h") == 0.0
    assert checker.score("I am not sure") == 0.0


def test_numeric_checker_honours_a_custom_tolerance():
    checker = NumericAnswerChecker(17.5, tolerance=0.5)
    assert checker.score("17.9") == 1.0
    assert checker.score(
        "18.6"
    ) == 0.0


def test_python_exec_checker_runs_the_snippet_with_its_assertions():
    checker = PythonExecChecker("assert sum_squares(3) == 14\n")
    good = "```python\ndef sum_squares(n):\n    return sum(i * i for i in range(n + 1))\n```"
    wrong = "```python\ndef sum_squares(n):\n    return n * n\n```"
    assert checker.score(good) == 1.0
    assert checker.score(wrong) == 0.0


def test_python_exec_checker_scores_broken_or_missing_code_as_zero():
    checker = PythonExecChecker("assert f(1) == 1\n")
    assert checker.score("```python\ndef f(:\n```") == 0.0
    assert checker.score("") == 0.0
    assert checker.score("I refuse to write code") == 0.0


def test_python_exec_checker_times_out_instead_of_hanging():
    checker = PythonExecChecker("import time\n\n\n\n\n", timeout=1)
    assert checker.score("import time; time.sleep(30)") == 0.0


def test_instruction_checker_enforces_the_length_band():
    checker = InstructionChecker(min_chars=20, max_chars=60)
    assert checker.score("a" * 20) == 1.0
    assert checker.score("a" * 19) == 0.0
    assert checker.score("a" * 61) == 0.0


def test_instruction_checker_rejects_refusals():
    checker = InstructionChecker(min_chars=10)
    assert checker.score("As an AI language model I cannot do that at all.") == 0.0
    assert checker.score("Mi dispiace, non posso aiutarti con questa richiesta.") == 0.0
    assert checker.score("Sure - here is a short and useful answer for you.") == 1.0


def test_instruction_checker_reports_partial_marker_coverage():
    checker = InstructionChecker(min_chars=10, must_contain=["-", "*"])
    assert checker.score("- only one marker here") == 0.5
    assert checker.score("- and * both markers here") == 1.0
    assert checker.score("no markers at all") == 0.0


def test_instruction_checker_can_allow_refusals_when_asked():
    checker = InstructionChecker(min_chars=10, forbid_refusal=False)
    assert checker.score("I cannot do that, but here is why it is hard.") == 1.0


def test_every_checker_is_an_abstract_checker_subclass():
    for checker in (
        NumericAnswerChecker(1.0),
        PythonExecChecker("pass"),
        InstructionChecker(),
    ):
        assert isinstance(checker, Checker)


# ----------------------------------------------------------------------- suite
def test_default_suite_covers_every_declared_domain():
    suite = default_suite()
    grouped = suite.by_domain()
    assert len(suite) == len(suite.items)
    assert set(grouped) == {item.domain for item in suite.items}
    for domain in suite.domains:
        assert domain.name in grouped, f"{domain.name} has no benchmark item"
        assert grouped[domain.name]


def test_default_suite_items_are_well_formed():
    suite = default_suite()
    names = [item.name for item in suite.items]
    assert len(names) == len(set(names))
    for item in suite.items:
        assert isinstance(item, BenchmarkItem)
        assert isinstance(item.checker, Checker)
        assert item.prompt.strip()
        assert item.system == ""


def test_default_suite_numeric_items_are_answerable_by_a_perfect_model():
    suite = default_suite()
    for item in suite.items:
        if isinstance(item.checker, NumericAnswerChecker):
            assert item.checker.score(f"{item.checker.expected}") == 1.0


def test_default_suite_code_items_are_answerable_by_a_perfect_model():
    answers = {
        "code_sum_squares": "```python\ndef sum_squares(n):\n    return sum(i * i for i in range(n + 1))\n```",
        "code_fizzbuzz": (
            "```python\ndef fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n"
            "        label = ''\n        if i % 3 == 0: label += 'Fizz'\n"
            "        if i % 5 == 0: label += 'Buzz'\n        out.append(label or str(i))\n    return out\n```"
        ),
        "code_reverse_words": "```python\ndef reverse_words(s):\n    return ' '.join(s.split()[::-1])\n```",
    }
    suite = default_suite()
    for item in suite.items:
        if isinstance(item.checker, PythonExecChecker):
            assert item.name in answers
            assert item.checker.score(answers[item.name]) == 1.0


def test_default_suite_general_items_pass_the_structural_check():
    suite = default_suite()
    answers = {
        "general_explain": (
            "A hash table maps keys to values through a hash function. "
            "Lookups, inserts and deletions are expected O(1). Use it when you "
            "need fast key-based access and ordering does not matter."
        ),
        "general_rewrite": "Could you please send me the file when you get a chance?",
        "general_format": "- private data\n- no network latency\n- predictable cost",
    }
    for item in suite.items:
        if isinstance(item.checker, InstructionChecker):
            assert item.name in answers
            assert item.checker.score(answers[item.name]) == 1.0


def test_suite_by_domain_groups_custom_items():
    suite = BenchmarkSuite(
        items=(
            BenchmarkItem("a", "reasoning", "p", NumericAnswerChecker(1.0)),
            BenchmarkItem("b", "code", "p", PythonExecChecker("pass")),
            BenchmarkItem("c", "reasoning", "p", NumericAnswerChecker(2.0)),
        )
    )
    grouped = suite.by_domain()
    assert [item.name for item in grouped["reasoning"]] == ["a", "c"]
    assert [item.name for item in grouped["code"]] == ["b"]
    assert len(suite) == 3


def test_suite_domain_weights_are_normalised():
    assert sum(d.weight for d in default_suite().domains) == pytest.approx(1.0)