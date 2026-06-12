from __future__ import annotations

from llm_evalgate.eval.dimension import Dimension, DimensionResult
from llm_evalgate.eval.harness import EvalHarness
from llm_evalgate.judge.consistency import SelfConsistencyJudge
from llm_evalgate.judge.dimension import JudgeDimension, JuryDimension
from llm_evalgate.judge.pairwise import PairwiseResult


def _cycling_judge(scores, threshold=0.6):
    """A JudgeDimension whose model returns a repeating sequence of scores."""
    queue = list(scores)
    idx = {"i": 0}

    def complete(prompt: str) -> str:
        s = queue[idx["i"] % len(queue)]
        idx["i"] += 1
        return f"SCORE: {s}\nREASON: r"

    return JudgeDimension(complete, rubric="grade", scale=1, threshold=threshold)


def test_dimension_result_default_no_review():
    r = DimensionResult(score=1.0, passed=True, detail="x")
    assert r.needs_review is False


def test_self_consistency_stdev_trigger():
    # Wildly varying scores -> high stdev -> flagged.
    judge = _cycling_judge([0.1, 0.9, 0.2, 0.8, 0.5])
    sc = SelfConsistencyJudge(judge, samples=5, max_stdev=0.15)
    result = sc.run("text")
    assert result.needs_review
    assert "stdev" in result.detail


def test_self_consistency_stable_not_flagged():
    judge = _cycling_judge([0.8, 0.8, 0.8, 0.8, 0.8])
    sc = SelfConsistencyJudge(judge, samples=5, max_stdev=0.15)
    result = sc.run("text")
    assert not result.needs_review


def test_self_consistency_margin_trigger_near_threshold():
    # Scores hover right at the 0.6 threshold with spread, CI straddles it.
    judge = _cycling_judge([0.5, 0.7, 0.55, 0.65, 0.6], threshold=0.6)
    sc = SelfConsistencyJudge(judge, samples=5, review_margin=0.1)
    result = sc.run("text")
    assert result.needs_review
    assert "straddles threshold" in result.detail


def test_self_consistency_margin_not_flagged_when_clear():
    # Consistently high, far from threshold -> margin trigger does not fire.
    judge = _cycling_judge([0.95, 0.95, 0.95, 0.95, 0.95], threshold=0.6)
    sc = SelfConsistencyJudge(judge, samples=5, review_margin=0.1)
    result = sc.run("text")
    assert not result.needs_review


def test_jury_disagreement_trigger():
    j1 = _cycling_judge([0.1])
    j2 = _cycling_judge([0.9])
    jury = JuryDimension([j1, j2], max_disagreement=0.2)
    result = jury.run("text")
    assert result.needs_review
    assert "spread" in result.detail


def test_jury_agreement_not_flagged():
    j1 = _cycling_judge([0.8])
    j2 = _cycling_judge([0.82])
    jury = JuryDimension([j1, j2], max_disagreement=0.2)
    result = jury.run("text")
    assert not result.needs_review


def test_pairwise_result_needs_review_on_flip():
    flipped = PairwiseResult(
        winner="tie", consistent=False, order_ab_winner="A",
        order_ba_winner="B", reason="flip", raw=("", ""),
    )
    assert flipped.needs_review
    stable = PairwiseResult(
        winner="A", consistent=True, order_ab_winner="A",
        order_ba_winner="A", reason="ok", raw=("", ""),
    )
    assert not stable.needs_review


def test_report_surfaces_review_without_changing_pass():
    class _Flagger(Dimension):
        def evaluate(self, text):
            return 1.0, "fine"

        def run(self, text):
            return DimensionResult(1.0, True, "fine", needs_review=True)

    harness = EvalHarness([_Flagger(name="flagger")])
    report = harness.run("text")
    assert report.passed  # review is orthogonal to pass/fail
    assert "flagger" in report.needs_review
    assert "REVIEW" in str(report)
    assert "flagged for review" in str(report)
