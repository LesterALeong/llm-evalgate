from __future__ import annotations

from llm_evalgate.agentic.dimensions.step_progress import (
    StepProgressDimension,
    _render_prefix,
)
from llm_evalgate.agentic.harness import AgentEvalHarness
from llm_evalgate.agentic.trace import AgentStep, AgentTrace, ToolCall
from llm_evalgate.judge.dimension import JudgeDimension


def _scripted_judge(scores, threshold=0.7):
    """JudgeDimension whose model returns a queued score per call."""
    queue = list(scores)
    idx = {"i": 0}

    def complete(prompt: str) -> str:
        s = queue[idx["i"]]
        idx["i"] += 1
        return f"SCORE: {s}\nREASON: step reason {s}"

    return JudgeDimension(complete, rubric="progress?", scale=1, threshold=threshold)


def _trace(n_steps, with_repeat=False):
    steps = []
    for i in range(n_steps):
        args = {"q": "same"} if with_repeat else {"q": f"q{i}"}
        steps.append(AgentStep(thought=f"t{i}", tool_calls=[ToolCall(name="search", args=args)]))
    return AgentTrace(goal="solve it", steps=steps, final_answer="done")


def test_mean_aggregation():
    judge = _scripted_judge([0.9, 0.9, 0.2, 0.8, 0.8])
    dim = StepProgressDimension(judge, aggregate="mean")
    score, detail = dim.evaluate(_trace(5))
    assert abs(score - (0.9 + 0.9 + 0.2 + 0.8 + 0.8) / 5) < 1e-9
    assert "weakest step 3" in detail


def test_min_aggregation():
    judge = _scripted_judge([0.9, 0.2, 0.8])
    dim = StepProgressDimension(judge, aggregate="min")
    score, _ = dim.evaluate(_trace(3))
    assert score == 0.2


def test_inefficiency_penalty_matches_repeat_detection():
    judge = _scripted_judge([0.9, 0.9, 0.9])
    dim = StepProgressDimension(judge, aggregate="mean", inefficiency_penalty=0.1)
    # three identical (search, {"q": "same"}) calls -> 2 repeats -> -0.2
    score, detail = dim.evaluate(_trace(3, with_repeat=True))
    assert abs(score - (0.9 - 0.2)) < 1e-9
    assert "2 repeated call" in detail


def test_penalty_floors_at_zero():
    judge = _scripted_judge([0.1, 0.1, 0.1])
    dim = StepProgressDimension(judge, aggregate="mean", inefficiency_penalty=1.0)
    score, _ = dim.evaluate(_trace(3, with_repeat=True))
    assert score == 0.0


def test_prefix_has_no_lookahead():
    trace = _trace(3)
    prefix_for_step_2 = _render_prefix(trace, 1)  # k=1 is step index 2
    assert "CURRENT STEP 2" in prefix_for_step_2
    assert "t0" in prefix_for_step_2  # step 1 is in the history
    assert "t2" not in prefix_for_step_2  # step 3 must not leak in


def test_empty_steps_scores_zero():
    judge = _scripted_judge([])
    dim = StepProgressDimension(judge)
    score, detail = dim.evaluate(AgentTrace(goal="g", steps=[], final_answer=None))
    assert score == 0.0
    assert "no steps" in detail


def test_user_message_round_trips():
    data = {
        "goal": "g",
        "steps": [{"thought": "t", "user_message": "hello", "tool_calls": []}],
        "final_answer": "a",
    }
    trace = AgentTrace.from_dict(data)
    assert trace.steps[0].user_message == "hello"
    assert "User: hello" in trace.to_text()


def test_to_text_unchanged_without_user_message():
    trace = AgentTrace(goal="g", steps=[AgentStep(thought="t")], final_answer="a")
    assert "User:" not in trace.to_text()


def test_composes_in_agent_harness():
    judge = _scripted_judge([0.9, 0.9])
    dim = StepProgressDimension(judge, threshold=0.7)
    harness = AgentEvalHarness([dim])
    report = harness.run(_trace(2))
    assert report.passed
