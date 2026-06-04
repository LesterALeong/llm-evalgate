"""Measure and de-bias an LLM-as-judge, offline.

Run from the repo root::

    python examples/judge_calibration.py

Shows the three judge-reliability tools added in v0.3.0:
- self-consistency: sample a stochastic judge N times for an uncertainty band
- pairwise debiasing: catch a position-biased judge by swapping answer order
- calibration: measure judge-vs-human agreement, plus a verbosity-bias check

Every judge here is a fake offline callable, so no API key or network is
needed. Swap in ``anthropic_judge(...)`` for live grading.
"""

from __future__ import annotations

import itertools

from llm_evalgate import (
    CalibrationSample,
    JudgeDimension,
    PairwiseJudge,
    SelfConsistencyJudge,
    calibrate_judge,
    position_bias_rate,
    verbosity_bias,
)


def main() -> None:
    # 1. Self-consistency: a stochastic judge whose score varies call to call.
    score_cycle = itertools.cycle(["4", "3", "5", "4", "3"])

    def varying(_prompt: str) -> str:
        return f"SCORE: {next(score_cycle)}"

    base = JudgeDimension(complete=varying, rubric="Is the answer correct and on topic?", scale=5)
    sc = SelfConsistencyJudge(base, samples=5)
    score, detail = sc.evaluate("Some answer.")
    print("Self-consistency")
    print(f"  aggregate score={score:.3f}")
    print(f"  {detail}")
    print()

    # 2. Pairwise debiasing: a judge that ALWAYS prefers whatever is shown first.
    def position_biased(_prompt: str) -> str:
        return "WINNER: First\nREASON: it just prefers the first one"

    pj = PairwiseJudge(complete=position_biased, criteria="Which answer is more helpful?")
    result = pj.compare("Answer A", "Answer B")
    print("Pairwise (position-biased judge)")
    print(f"  winner={result.winner}  consistent={result.consistent}")
    print(f"  order AB picked {result.order_ab_winner}, order BA picked {result.order_ba_winner}")
    rate = position_bias_rate(pj, [("a", "b"), ("c", "d"), ("e", "f")])
    print(f"  measured position-bias rate: {rate:.3f}")
    print()

    # 3. Calibration: a judge whose score tracks a marker, compared to known human scores.
    human = {"great answer": 1.0, "ok answer": 0.6, "bad answer": 0.2}

    def aligned(prompt: str) -> str:
        for marker, human_score in human.items():
            if marker in prompt:
                return f"SCORE: {human_score * 5:.1f}"
        return "SCORE: 3"

    judge = JudgeDimension(complete=aligned, rubric="Grade the answer.", scale=5, threshold=0.5)
    samples = [CalibrationSample(text=t, human_score=hs) for t, hs in human.items()]
    report = calibrate_judge(judge, samples)
    print("Calibration (judge vs human)")
    print(report.table())
    print()

    # 4. Verbosity bias: a judge that just rewards length, regardless of content.
    def length_judge(prompt: str) -> str:
        body = prompt.split("Text:")[-1]
        biased_score = min(5.0, 1.0 + len(body) / 40.0)
        return f"SCORE: {biased_score:.1f}"

    vjudge = JudgeDimension(complete=length_judge, rubric="Grade the answer.", scale=5)
    vbias = verbosity_bias(
        vjudge, ["short", "a bit longer answer", "a substantially longer answer here"]
    )
    print("Verbosity bias")
    print(f"  length-vs-score correlation: {vbias:.3f}  (near 1.0 means the judge rewards length)")


if __name__ == "__main__":
    main()
