"""Benchmark a deterministic eval harness, then a judge-augmented one, offline.

Run from the repo root::

    python examples/benchmark.py

The deterministic harness catches blocklist, schema, and readability failures
but misses semantic regressions (factually wrong but well-formed text). Adding
an LLM-as-judge dimension raises the regression catch rate. The judge here is a
fake offline ``complete`` so the example needs no API key and no network.
"""

from __future__ import annotations

from llm_evalgate.bench import BenchmarkRunner, load_golden
from llm_evalgate.eval import EvalHarness
from llm_evalgate.eval.dimensions import (
    BlocklistDimension,
    ReadabilityDimension,
    SchemaComplianceDimension,
)
from llm_evalgate.judge import JudgeDimension

# Distinctive substrings from the four semantic-fail summaries in the golden
# dataset. A real judge reads the text and reasons; this fake one keys on these
# so the example is deterministic.
_SEMANTIC_MARKERS = (
    "capital of France is Berlin",
    "three parts gold",
    "ten trillion dollars",
    "freeze the flour",
)


def fake_complete(prompt: str) -> str:
    """Stand in for an LLM judge: flag known semantic regressions, else pass.

    Replace this with a real Anthropic-backed callable::

        from llm_evalgate.judge import anthropic_judge
        complete = anthropic_judge(model="claude-3-5-haiku-latest")
    """
    if any(marker in prompt for marker in _SEMANTIC_MARKERS):
        return "SCORE: 1\nREASON: semantic issue"
    return "SCORE: 5\nREASON: ok"


def build_deterministic_dimensions() -> list:
    return [
        BlocklistDimension(terms=["confidential", "internal use only", "[REDACTED]"]),
        SchemaComplianceDimension(required_fields=["title:", "summary:"]),
        ReadabilityDimension(threshold=0.2),
    ]


def main() -> None:
    samples = load_golden()

    deterministic = EvalHarness(build_deterministic_dimensions())
    deterministic_result = BenchmarkRunner(deterministic).run(samples)
    print("Deterministic harness")
    print(deterministic_result.table())
    print()

    judge = JudgeDimension(
        complete=fake_complete,
        rubric="Is the text factually correct, on topic, and coherent?",
    )
    augmented = EvalHarness(build_deterministic_dimensions() + [judge])
    augmented_result = BenchmarkRunner(augmented).run(samples)
    print("Judge-augmented harness")
    print(augmented_result.table())


if __name__ == "__main__":
    main()
