import pytest

from llm_evalgate.eval import EvalHarness
from llm_evalgate.eval.dimensions import BlocklistDimension
from llm_evalgate.judge import (
    JudgeDimension,
    JudgeVerdict,
    JuryDimension,
    parse_verdict,
)


def _fixed(response: str):
    def complete(prompt: str) -> str:
        return response
    return complete


# --- JudgeDimension ---

def test_judge_scores_and_passes():
    dim = JudgeDimension(
        complete=_fixed("SCORE: 4\nREASON: clear and correct"),
        rubric="Is the text clear and correct?",
        scale=5,
        threshold=0.6,
    )
    result = dim.run("Some text under review.")
    assert result.score == pytest.approx(0.8)
    assert result.passed
    assert "clear and correct" in result.detail


def test_judge_unparseable_scores_zero():
    dim = JudgeDimension(
        complete=_fixed("I cannot grade this."),
        rubric="Is the text clear?",
        threshold=0.6,
    )
    result = dim.run("Some text under review.")
    assert result.score == 0.0
    assert not result.passed
    assert "parse failure" in result.detail


def test_judge_callable_raising_is_graceful():
    def boom(prompt: str) -> str:
        raise RuntimeError("model down")

    dim = JudgeDimension(complete=boom, rubric="Is the text clear?", threshold=0.6)
    result = dim.run("Some text under review.")
    assert result.score == 0.0
    assert not result.passed
    assert "judge error" in result.detail


def test_judge_cache_dedupes_by_prompt():
    calls = {"n": 0}

    def counting(prompt: str) -> str:
        calls["n"] += 1
        return "SCORE: 5\nREASON: excellent"

    cache: dict[str, JudgeVerdict] = {}
    dim = JudgeDimension(
        complete=counting,
        rubric="Is the text excellent?",
        cache=cache,
    )
    dim.run("identical text")
    dim.run("identical text")
    assert calls["n"] == 1


# --- parse_verdict ---

def test_parse_verdict_scale_one_keeps_float():
    verdict = parse_verdict("SCORE: 0.9\nREASON: strong", scale=1)
    assert verdict.score == pytest.approx(0.9)
    assert verdict.reason == "strong"
    assert verdict.raw == "SCORE: 0.9\nREASON: strong"


# --- JuryDimension ---

def test_jury_mean_passes():
    judges = [
        JudgeDimension(complete=_fixed("SCORE: 5\nREASON: a"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 4\nREASON: b"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 3\nREASON: c"), rubric="r"),
    ]
    jury = JuryDimension(judges, aggregate="mean", threshold=0.6)
    result = jury.run("text")
    assert result.score == pytest.approx(0.8)
    assert result.passed
    assert "stdev=" in result.detail


def test_jury_median():
    judges = [
        JudgeDimension(complete=_fixed("SCORE: 5\nREASON: a"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 4\nREASON: b"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 3\nREASON: c"), rubric="r"),
    ]
    jury = JuryDimension(judges, aggregate="median", threshold=0.6)
    result = jury.run("text")
    assert result.score == pytest.approx(0.8)
    assert result.passed


def test_jury_majority():
    judges = [
        JudgeDimension(complete=_fixed("SCORE: 5\nREASON: a"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 4\nREASON: b"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 3\nREASON: c"), rubric="r"),
    ]
    jury = JuryDimension(judges, aggregate="majority", threshold=0.6)
    result = jury.run("text")
    # All three judge scores (1.0, 0.8, 0.6) clear the 0.6 threshold.
    assert result.score == pytest.approx(1.0)
    assert result.passed
    assert "votes=3 pass / 0 fail" in result.detail


def test_jury_majority_split():
    judges = [
        JudgeDimension(complete=_fixed("SCORE: 5\nREASON: a"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 4\nREASON: b"), rubric="r"),
        JudgeDimension(complete=_fixed("SCORE: 1\nREASON: c"), rubric="r"),
    ]
    jury = JuryDimension(judges, aggregate="majority", threshold=0.6)
    result = jury.run("text")
    # Scores 1.0 and 0.8 pass, 0.2 fails -> 2 of 3.
    assert result.score == pytest.approx(2 / 3)
    assert result.passed
    assert "votes=2 pass / 1 fail" in result.detail


def test_jury_empty_raises():
    with pytest.raises(ValueError):
        JuryDimension([])


def test_jury_unknown_aggregate_raises():
    judges = [JudgeDimension(complete=_fixed("SCORE: 5\nREASON: a"), rubric="r")]
    with pytest.raises(ValueError):
        JuryDimension(judges, aggregate="harmonic")


# --- composition with EvalHarness ---

def test_judge_composes_in_harness():
    judge = JudgeDimension(
        complete=_fixed("SCORE: 4\nREASON: clear"),
        rubric="Is the text clear?",
        threshold=0.6,
    )
    blocklist = BlocklistDimension(terms=["secret"])
    harness = EvalHarness([judge, blocklist])
    report = harness.run("This is a public document.")
    assert report.passed
    assert "judge" in report.results
    assert report.results["judge"].score == pytest.approx(0.8)
