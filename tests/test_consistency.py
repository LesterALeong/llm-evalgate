import pytest

from llm_evalgate.judge import JudgeDimension, JudgeVerdict
from llm_evalgate.judge.consistency import ScoreDistribution, SelfConsistencyJudge


def _round_robin(*responses: str):
    state = {"i": 0}

    def complete(prompt: str) -> str:
        response = responses[state["i"] % len(responses)]
        state["i"] += 1
        return response

    return complete


def _fixed(response: str):
    def complete(prompt: str) -> str:
        return response

    return complete


def test_distribution_collects_independent_samples():
    judge = JudgeDimension(
        complete=_round_robin("SCORE: 5", "SCORE: 3", "SCORE: 4"),
        rubric="r",
        scale=5,
    )
    consistency = SelfConsistencyJudge(judge, samples=3)
    dist = consistency.distribution("text")
    assert isinstance(dist, ScoreDistribution)
    assert dist.n == 3
    assert set(dist.scores) == {1.0, 0.6, 0.8}
    assert dist.mean == pytest.approx((1.0 + 0.6 + 0.8) / 3)
    assert dist.stdev > 0


def test_constant_judge_has_zero_spread():
    judge = JudgeDimension(complete=_fixed("SCORE: 4"), rubric="r", scale=5)
    consistency = SelfConsistencyJudge(judge, samples=4)
    dist = consistency.distribution("text")
    assert dist.stdev == 0.0
    assert dist.ci_low == dist.ci_high == dist.mean


def test_single_sample():
    judge = JudgeDimension(complete=_fixed("SCORE: 4"), rubric="r", scale=5)
    consistency = SelfConsistencyJudge(judge, samples=1)
    dist = consistency.distribution("text")
    assert dist.n == 1
    assert dist.stdev == 0.0


def test_threshold_defaults_to_base_judge():
    judge = JudgeDimension(
        complete=_fixed("SCORE: 4"), rubric="r", scale=5, threshold=0.5
    )
    consistency = SelfConsistencyJudge(judge)
    assert consistency.threshold == 0.5


def test_samples_zero_raises():
    judge = JudgeDimension(complete=_fixed("SCORE: 4"), rubric="r", scale=5)
    with pytest.raises(ValueError):
        SelfConsistencyJudge(judge, samples=0)


def test_unknown_aggregate_raises():
    judge = JudgeDimension(complete=_fixed("SCORE: 4"), rubric="r", scale=5)
    with pytest.raises(ValueError):
        SelfConsistencyJudge(judge, aggregate="mode")


def test_evaluate_detail_reports_distribution():
    judge = JudgeDimension(
        complete=_round_robin("SCORE: 5", "SCORE: 3", "SCORE: 4"),
        rubric="r",
        scale=5,
    )
    consistency = SelfConsistencyJudge(judge, samples=3)
    score, detail = consistency.evaluate("text")
    assert score == pytest.approx((1.0 + 0.6 + 0.8) / 3)
    assert "95% CI=" in detail
    assert "stdev=" in detail


def test_existing_cache_behavior_preserved():
    calls = {"n": 0}

    def counting(prompt: str) -> str:
        calls["n"] += 1
        return "SCORE: 5\nREASON: excellent"

    cache: dict[str, JudgeVerdict] = {}
    dim = JudgeDimension(complete=counting, rubric="r", cache=cache)
    dim.run("identical text")
    dim.run("identical text")
    assert calls["n"] == 1
