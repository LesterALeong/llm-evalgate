import pytest

from llm_evalgate.judge.pairwise import (
    DEFAULT_PAIRWISE_TEMPLATE,
    PairwiseJudge,
    parse_pairwise,
    position_bias_rate,
)

GOOD = "This answer is thorough, correct, and well structured."
BAD = "wrong"
GOOD2 = "Another genuinely excellent and complete response."
BAD2 = "nope"


def _unbiased(good: str):
    """A judge that always prefers ``good`` by locating it in the prompt.

    The default template puts the First response before the Second. We detect
    which slot holds ``good`` by comparing the index of each section.
    """

    def complete(prompt: str) -> str:
        first_start = prompt.index("First response:\n") + len("First response:\n")
        second_start = prompt.index("Second response:\n")
        first_block = prompt[first_start:second_start]
        if good in first_block:
            return "WINNER: First\nREASON: better"
        return "WINNER: Second\nREASON: better"

    return complete


def _always_first(prompt: str) -> str:
    return "WINNER: First\nREASON: I prefer whatever comes first."


# --- unbiased judge ---

def test_unbiased_good_as_a_wins_a():
    judge = PairwiseJudge(_unbiased(GOOD), criteria="quality")
    result = judge.compare(GOOD, BAD)
    assert result.winner == "A"
    assert result.consistent is True
    assert result.order_ab_winner == "A"
    assert result.order_ba_winner == "A"


def test_unbiased_good_as_b_wins_b():
    judge = PairwiseJudge(_unbiased(GOOD), criteria="quality")
    result = judge.compare(BAD, GOOD)
    assert result.winner == "B"
    assert result.consistent is True


def test_unbiased_position_bias_rate_zero():
    judge = PairwiseJudge(_unbiased(GOOD), criteria="quality")
    rate = position_bias_rate(judge, [(GOOD, BAD), (BAD, GOOD)])
    assert rate == 0.0


# --- position-biased judge ---

def test_biased_judge_caught_as_tie():
    judge = PairwiseJudge(_always_first, criteria="quality")
    result = judge.compare("alpha", "beta")
    assert result.consistent is False
    assert result.winner == "tie"
    assert result.order_ab_winner == "A"
    assert result.order_ba_winner == "B"


def test_biased_judge_position_bias_rate_one():
    judge = PairwiseJudge(_always_first, criteria="quality")
    rate = position_bias_rate(judge, [("alpha", "beta"), ("gamma", "delta")])
    assert rate == 1.0


# --- no debiasing ---

def test_no_debias_single_order():
    judge = PairwiseJudge(_unbiased(GOOD), criteria="quality", debias_position=False)
    result = judge.compare(GOOD, BAD)
    assert result.consistent is True
    assert result.winner == result.order_ab_winner == "A"
    assert result.order_ba_winner == "A"
    assert result.raw[1] == ""


# --- parse_pairwise ---

def test_parse_pairwise_tie():
    slot, reason = parse_pairwise("WINNER: Tie\nREASON: equal")
    assert slot == "tie"
    assert reason == "equal"


def test_parse_pairwise_garbage_defaults_to_tie():
    slot, reason = parse_pairwise("I really cannot tell.")
    assert slot == "tie"
    assert reason  # non-empty fallback


def test_parse_pairwise_case_insensitive():
    slot, _ = parse_pairwise("winner: second\nreason: clearer")
    assert slot == "second"


# --- error handling ---

def test_complete_raising_is_graceful():
    def boom(prompt: str) -> str:
        raise RuntimeError("model down")

    judge = PairwiseJudge(boom, criteria="quality")
    slot, reason = judge._judge_once("a", "b")
    assert slot == "tie"
    assert "judge error" in reason

    result = judge.compare("a", "b")
    assert result.winner == "tie"
    assert "judge error" in result.reason


def test_position_bias_rate_empty_raises():
    judge = PairwiseJudge(_unbiased(GOOD), criteria="quality")
    with pytest.raises(ValueError):
        position_bias_rate(judge, [])


def test_default_template_has_placeholders():
    assert "{criteria}" in DEFAULT_PAIRWISE_TEMPLATE
    assert "{first}" in DEFAULT_PAIRWISE_TEMPLATE
    assert "{second}" in DEFAULT_PAIRWISE_TEMPLATE
