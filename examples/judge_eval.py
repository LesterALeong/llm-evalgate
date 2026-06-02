"""Minimal LLM-as-judge demo with a fake offline ``complete``.

Run from the repo root::

    python examples/judge_eval.py
"""

from __future__ import annotations

from llm_evalgate.judge import JudgeDimension


def fake_complete(prompt: str) -> str:
    """Stand in for an LLM judge based on a marker in the text.

    Replace this with a real Anthropic-backed callable::

        from llm_evalgate.judge import anthropic_judge
        complete = anthropic_judge(model="claude-3-5-haiku-latest")
    """
    if "capital of France is Berlin" in prompt:
        return "SCORE: 1\nREASON: the capital of France is Paris, not Berlin"
    return "SCORE: 5\nREASON: factually correct and on topic"


def main() -> None:
    judge = JudgeDimension(
        complete=fake_complete,
        rubric="Is the text factually correct, on topic, and coherent?",
    )

    good = "title: Capital of France\nsummary: The capital of France is Paris."
    bad = "title: Capital of France\nsummary: The capital of France is Berlin."

    for label, text in (("good", good), ("bad", bad)):
        result = judge.run(text)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{label}] {status} score={result.score:.3f} - {result.detail}")


if __name__ == "__main__":
    main()
